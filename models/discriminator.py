"""
Conditional PatchGAN Discriminator with Spectral Normalization.
Used strictly during SPA-VAEGAN adversarial training.
Judges local texture and high-frequency realism on a 32x32 patch grid.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    """
    Conditional PatchGAN Discriminator with Spectral Normalization.
    Input: Concatenation of candidate image (HR or SR) and bicubic-upsampled LR (6 channels, 256x256).
    Output: 1x32x32 patch logits grid.
    
    Spectral normalization ensures 1-Lipschitz continuity, stabilizing GAN dynamics and
    preventing discriminator gradient explosion when paired with Hinge Loss.
    """
    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            # 6 -> 64, Stride 1: 256x256
            spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 -> 64, Stride 2: 128x128
            spectral_norm(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 -> 128, Stride 1: 128x128
            spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 -> 128, Stride 2: 64x64
            spectral_norm(nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 -> 256, Stride 1: 64x64
            spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256 -> 256, Stride 2: 32x32
            spectral_norm(nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256 -> 1, Stride 1: 32x32
            spectral_norm(nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1))
        )

    def forward(self, img: torch.Tensor, lr_up: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Ground truth HR or reconstructed SR (B, 3, 256, 256)
            lr_up: Bicubically upsampled LR conditioning (B, 3, 256, 256)
        Returns:
            Logits tensor of shape (B, 1, 32, 32)
        """
        x = torch.cat([img, lr_up], dim=1)
        return self.net(x)
