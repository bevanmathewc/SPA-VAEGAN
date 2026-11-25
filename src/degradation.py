"""
Deterministic degradation pipeline for 4x Super-Resolution.
Matches real-world pipeline:
1. Gaussian blur (sigma=1.0, 7x7 kernel, reflect padding).
2. Bicubic downsampling by 4 with anti-aliasing.
3. Clamp to [0, 1] and quantize to 8-bit precision (round(x * 255) / 255).
"""

import torch
import torch.nn.functional as F


def gaussian_kernel(sigma: float = 1.0, k: int = 7, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Generate a normalized 2D Gaussian kernel."""
    ax = torch.arange(k, device=device, dtype=torch.float32) - k // 2
    g = torch.exp(-(ax ** 2) / (2 * (sigma ** 2)))
    g = g / g.sum()
    return g[:, None] * g[None, :]


@torch.no_grad()
def degrade(hr: torch.Tensor, scale: int = 4, sigma: float = 1.0, k: int = 7) -> torch.Tensor:
    """
    Applies fixed synthetic degradation to ground-truth HR images:
    hr: (B, 3, H, W) float in [0, 1] -> lr: (B, 3, H/scale, W/scale) float in [0, 1].
    Runs directly on GPU.
    """
    c = hr.shape[1]
    kernel_2d = gaussian_kernel(sigma=sigma, k=k, device=hr.device)
    w = kernel_2d.expand(c, 1, k, k).contiguous()
    
    # Reflect padding preserves border statistics
    pad_amt = k // 2
    padded = F.pad(hr, (pad_amt, pad_amt, pad_amt, pad_amt), mode="reflect")
    blurred = F.conv2d(padded, w, groups=c)
    
    # Bicubic downsampling with anti-aliasing filter
    h_lr, w_lr = hr.shape[2] // scale, hr.shape[3] // scale
    lr = F.interpolate(blurred, size=(h_lr, w_lr), mode="bicubic", antialias=True, align_corners=False)
    
    # 8-bit quantization simulating standard PNG/JPEG image input
    lr_quantized = (lr.clamp(0.0, 1.0) * 255.0).round() / 255.0
    return lr_quantized
