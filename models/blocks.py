"""
Building blocks for SPA-VAEGAN architecture.
Includes ResBlock with residual scaling for super-resolution stability.
No Batch Normalization is used, preventing color shifts and spatial artifacts.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Residual Block with two 3x3 convolutions and LeakyReLU activations.
    Employs residual scaling (scale=0.1) as standard in super-resolution (e.g. EDSR/RCAN)
    to prevent gradient explosion and stabilize deep residual feature learning.
    """
    def __init__(self, c: int, scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True),
        )
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.body(x)
