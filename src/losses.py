"""
Loss functions for Baseline CVAE and SPA-VAEGAN:
- KL Divergence with numerical safeguards
- Sobel Edge Filter and Edge-Aware Loss (Structure Preservation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Computes analytical KL divergence between N(mu, diag(exp(logvar))) and N(0, I):
    KL = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
    Evaluated in fp32 for numerical stability.
    """
    mu_f = mu.float()
    logvar_f = logvar.float()
    return -0.5 * torch.mean(1.0 + logvar_f - mu_f.pow(2) - logvar_f.exp())


class SobelEdge(nn.Module):
    """
    Normalized Sobel Edge Filter for structure and gradient preservation.
    Extracts high-frequency edge magnitudes across RGB channels.
    Kernels are normalized by 1/8 to preserve gradient scale.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0],
                           [-2.0, 0.0, 2.0],
                           [-1.0, 0.0, 1.0]], dtype=torch.float32) / 8.0
        ky = kx.t().contiguous()
        
        # Buffer shapes: (3, 1, 3, 3) for grouped convolution over RGB
        self.register_buffer("kx", kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer("ky", ky.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (B, 3, H, W)
        Returns:
            Gradient magnitude map (B, 3, H, W) in fp32
        """
        x_f = x.float()
        gx = F.conv2d(x_f, self.kx, padding=1, groups=3)
        gy = F.conv2d(x_f, self.ky, padding=1, groups=3)
        # eps inside sqrt prevents division-by-zero gradient at flat regions (where gx=gy=0)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + self.eps)


def edge_loss(sobel: SobelEdge, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    """Computes mean absolute error between Sobel edge maps of SR and HR."""
    return F.l1_loss(sobel(sr), sobel(hr))
