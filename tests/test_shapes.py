"""
Unit tests for SPA-VAEGAN architecture, degradation, and gradient isolation.
Run with: python -m unittest tests/test_shapes.py
"""

import os
import sys
import unittest
import numpy as np

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from models.cvae import CVAE
from models.discriminator import PatchDiscriminator
from src.degradation import degrade
from src.losses import SobelEdge, edge_loss, kl_loss
from src.adaptive_weight import raw_adaptive_weight, AdaptiveWeightTracker


class TestSPAVAEGAN(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = torch.device("cpu")
        self.G = CVAE(zc=4, c=64, n_blocks=16).to(self.device)
        self.D = PatchDiscriminator(in_channels=6).to(self.device)

    def test_generator_shapes(self):
        """Test CVAE forward pass shapes in both training mode and prior inference mode."""
        lr = torch.randn(2, 3, 64, 64, device=self.device)
        hr = torch.randn(2, 3, 256, 256, device=self.device)

        # 1. Training mode
        sr_train, mu, logvar = self.G(lr, hr)
        self.assertEqual(sr_train.shape, (2, 3, 256, 256), "Training SR output shape mismatch")
        self.assertEqual(mu.shape, (2, 4, 64, 64), "Posterior mean shape mismatch")
        self.assertEqual(logvar.shape, (2, 4, 64, 64), "Posterior logvar shape mismatch")

        # 2. Prior mode (z = 0)
        sr_infer, mu_inf, logvar_inf = self.G(lr)
        self.assertEqual(sr_infer.shape, (2, 3, 256, 256), "Inference SR output shape mismatch")
        self.assertIsNone(mu_inf)
        self.assertIsNone(logvar_inf)

    def test_discriminator_shape(self):
        """Test PatchDiscriminator output shape: 32x32 patch logits."""
        img = torch.randn(2, 3, 256, 256, device=self.device)
        lr_up = torch.randn(2, 3, 256, 256, device=self.device)
        logits = self.D(img, lr_up)
        self.assertEqual(logits.shape, (2, 1, 32, 32), "Discriminator output grid shape mismatch")

    def test_degradation_pipeline(self):
        """Test degrade(): (B, 3, 256, 256) -> (B, 3, 64, 64) in range [0, 1]."""
        hr = torch.rand(2, 3, 256, 256, device=self.device)
        lr = degrade(hr, scale=4, sigma=1.0, k=7)
        self.assertEqual(lr.shape, (2, 3, 64, 64), "Degraded LR shape mismatch")
        self.assertTrue((lr >= 0.0).all() and (lr <= 1.0).all(), "Degraded LR values out of [0, 1] range")

    def test_sobel_edge_loss(self):
        """Test Sobel edge filter and loss function."""
        sobel = SobelEdge().to(self.device)
        img1 = torch.rand(2, 3, 256, 256, device=self.device)
        img2 = torch.rand(2, 3, 256, 256, device=self.device)
        e1 = sobel(img1)
        self.assertEqual(e1.shape, (2, 3, 256, 256), "Sobel edge map shape mismatch")
        loss = edge_loss(sobel, img1, img2)
        self.assertTrue(torch.isfinite(loss).item(), "Edge loss is non-finite")
        self.assertGreaterEqual(loss.item(), 0.0, "Edge loss must be non-negative")

    def test_gradient_isolation(self):
        """
        Verify strict gradient isolation:
        1. D update does NOT alter G parameters.
        2. G update does NOT alter D parameters.
        """
        lr = torch.rand(2, 3, 64, 64, device=self.device)
        hr = torch.rand(2, 3, 256, 256, device=self.device)
        lr_up = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)

        optG = torch.optim.Adam(self.G.parameters(), lr=1e-4)
        optD = torch.optim.Adam(self.D.parameters(), lr=1e-4)

        # Baseline snapshots
        g_weights_before = [p.clone().detach() for p in self.G.parameters()]
        d_weights_before = [p.clone().detach() for p in self.D.parameters()]

        # Step 1: Discriminator Update
        sr, _, _ = self.G(lr, hr)
        d_real = self.D(hr, lr_up)
        d_fake = self.D(sr.detach(), lr_up)  # Detached!
        loss_D = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()

        optD.zero_grad()
        loss_D.backward()
        optD.step()

        # Assert G parameters are unchanged bitwise
        for p_before, p_after in zip(g_weights_before, self.G.parameters()):
            self.assertTrue(torch.equal(p_before, p_after), "Discriminator update corrupted Generator parameters!")

        # Step 2: Generator Update
        for p in self.D.parameters():
            p.requires_grad_(False)

        d_weights_before_g = [p.clone().detach() for p in self.D.parameters()]

        l_adv = -self.D(sr, lr_up).mean()
        l_rec = F.l1_loss(sr, hr)
        loss_G = l_rec + 0.1 * l_adv

        optG.zero_grad()
        loss_G.backward()
        optG.step()

        for p in self.D.parameters():
            p.requires_grad_(True)

        # Assert D parameters are unchanged bitwise
        for p_before, p_after in zip(d_weights_before_g, self.D.parameters()):
            self.assertTrue(torch.equal(p_before, p_after), "Generator update corrupted Discriminator parameters!")

    def test_adaptive_weight_tracker(self):
        """Test adaptive weight tracker calculation and safeguards."""
        lr = torch.rand(2, 3, 64, 64, device=self.device)
        hr = torch.rand(2, 3, 256, 256, device=self.device)
        lr_up = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)

        sr, _, _ = self.G(lr, hr)
        l_rec = F.l1_loss(sr, hr)
        l_adv = -self.D(sr, lr_up).mean()

        tracker = AdaptiveWeightTracker(delta=0.1, lambda_max=1000.0)
        lam, raw, n_rec, n_adv = tracker.step(l_rec, l_adv, self.G.dec.last.weight)

        self.assertGreater(lam, 0.0, "Lambda must be strictly positive")
        self.assertLessEqual(lam, 1000.0, "Lambda exceeded maximum threshold")
        self.assertTrue(np.isfinite(lam), "Lambda is non-finite")


if __name__ == "__main__":
    unittest.main()
