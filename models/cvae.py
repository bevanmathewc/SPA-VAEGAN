"""
Conditional Variational Autoencoder (CVAE) Generator for 4x Super-Resolution.
Shared architecture for both Baseline CVAE and SPA-VAEGAN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import ResBlock


class Encoder(nn.Module):
    """
    Posterior encoder q(z | HR, LR_up). Used during training only.
    Takes concatenated ground-truth HR and bicubically upsampled LR (6 channels, 256x256).
    Produces Gaussian posterior parameters mu and logvar (4 channels, 64x64).
    """
    def __init__(self, zc: int = 4, c: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, c, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, c, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, c, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(c, scale=0.1),
            ResBlock(c, scale=0.1)
        )
        self.mu = nn.Conv2d(c, zc, kernel_size=3, padding=1)
        self.logvar = nn.Conv2d(c, zc, kernel_size=3, padding=1)

    def forward(self, hr: torch.Tensor, lr_up: torch.Tensor):
        h = self.net(torch.cat([hr, lr_up], dim=1))
        # Clamp logvar to [-10, 10] for numerical stability in fp16/fp32
        return self.mu(h), self.logvar(h).clamp(-10.0, 10.0)


class Decoder(nn.Module):
    """
    Conditional decoder p(HR | z, LR).
    Takes concatenated LR input (3 channels, HxW) and latent map z (zc=4 channels, HxW).
    Features pass through a deep trunk of 16 ResBlocks with a global residual connection,
    two PixelShuffle upsampling stages (2x twice = 4x), and a final convolution.
    Learns a residual over bicubic interpolation of the LR input.
    """
    def __init__(self, zc: int = 4, c: int = 64, n_blocks: int = 16):
        super().__init__()
        self.head = nn.Conv2d(3 + zc, c, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResBlock(c, scale=0.1) for _ in range(n_blocks)])
        self.up = nn.Sequential(
            nn.Conv2d(c, 4 * c, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, 4 * c, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Reference layer for adaptive adversarial weight calculation:
        self.last = nn.Conv2d(c, 3, kernel_size=3, padding=1)

    def forward(self, lr: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        f = self.head(torch.cat([lr, z], dim=1))
        f = f + self.body(f)
        out = self.last(self.up(f))
        bicubic_base = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
        return out + bicubic_base


class CVAE(nn.Module):
    """
    Complete Conditional VAE Generator for 4x Image Super-Resolution.
    Parameter count: ~1.8M parameters.
    Fully convolutional: supports arbitrary input resolutions at inference time.
    """
    def __init__(self, zc: int = 4, c: int = 64, n_blocks: int = 16):
        super().__init__()
        self.zc = zc
        self.enc = Encoder(zc=zc, c=c)
        self.dec = Decoder(zc=zc, c=c, n_blocks=n_blocks)

    def forward(self, lr: torch.Tensor, hr: torch.Tensor = None):
        """
        Forward pass:
        - Training mode (hr is provided):
            Encodes (hr, bicubic(lr)) -> mu, logvar.
            Samples z via reparameterization trick: z = mu + eps * exp(0.5 * logvar).
            Decodes (lr, z) -> reconstructed SR image.
            Returns sr, mu, logvar.
        - Inference mode (hr is None):
            Sets latent z to prior mean (zeros).
            Decodes (lr, 0) -> deterministic SR output.
            Returns sr, None, None.
        """
        if hr is not None:
            lr_up = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
            mu, logvar = self.enc(hr, lr_up)
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
            sr = self.dec(lr, z)
            return sr, mu, logvar

        # Prior mean inference: z = 0
        z = torch.zeros(lr.shape[0], self.zc, lr.shape[2], lr.shape[3], device=lr.device, dtype=lr.dtype)
        sr = self.dec(lr, z)
        return sr, None, None
