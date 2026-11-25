"""
Inference engine and tiled upscaling algorithm for Streamlit app.
Applies:
- Model caching
- Fully convolutional tiled inference handling arbitrary image resolutions
- Border reflection padding and halo overlap cropping to eliminate tile border seams
"""

import os
import time
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from models.cvae import CVAE


def load_model(
    checkpoint_path: str,
    device: str = "cpu",
    zc: int = 4,
    c: int = 64,
    n_blocks: int = 16
) -> CVAE:
    """Instantiates CVAE and loads exported generator weights."""
    model = CVAE(zc=zc, c=c, n_blocks=n_blocks).to(device).eval()
    if os.path.exists(checkpoint_path):
        chk = torch.load(checkpoint_path, map_location=device)
        state_dict = chk.get("state_dict", chk.get("G_state_dict", chk))
        model.load_state_dict(state_dict)
    else:
        print(f"[Warning] Generator checkpoint not found at: {checkpoint_path}. Using random weights.")
    return model


@torch.no_grad()
def upscale_tiled(
    model: torch.nn.Module,
    img_u8: np.ndarray,
    device: torch.device,
    tile: int = 64,
    halo: int = 24,
    bs: int = 4
) -> Tuple[np.ndarray, float]:
    """
    Upscales an arbitrary-sized uint8 RGB image (H, W, 3) by 4x using overlap-and-crop tiling.
    Returns:
        (sr_uint8_image, elapsed_ms)
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # (H, W, 3) -> (1, 3, H, W) float in [0, 1]
    x = torch.from_numpy(img_u8).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    _, _, h, w = x.shape

    # For tiny images where halo exceeds side, use replicate pad or direct pass
    if h <= tile and w <= tile:
        sr, _, _ = model(x)
        out = sr.clamp(0.0, 1.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        out_np = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        return out_np, elapsed_ms

    ph = (-h) % tile
    pw = (-w) % tile
    
    pad_mode = "reflect" if (min(h, w) > (halo + max(ph, pw))) else "replicate"
    x_pad = F.pad(x, (halo, halo + pw, halo, halo + ph), mode=pad_mode)

    out = torch.zeros(1, 3, (h + ph) * 4, (w + pw) * 4, device=device)
    tiles = [(i, j) for i in range(0, h + ph, tile) for j in range(0, w + pw, tile)]

    for k in range(0, len(tiles), bs):
        batch_coords = tiles[k:k + bs]
        inp_list = [x_pad[:, :, i:i + tile + 2 * halo, j:j + tile + 2 * halo] for i, j in batch_coords]
        inp_batch = torch.cat(inp_list, dim=0)

        # Forward pass on tile batch
        sr_batch, _, _ = model(inp_batch)

        # Crop out the central core (strip the 4x halo)
        core = sr_batch[:, :, 4 * halo:4 * (halo + tile), 4 * halo:4 * (halo + tile)]
        for n, (i, j) in enumerate(batch_coords):
            out[:, :, 4 * i:4 * (i + tile), 4 * j:4 * (j + tile)] = core[n:n + 1]

    # Crop away padding and clamp
    out = out[:, :, :4 * h, :4 * w].clamp(0.0, 1.0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    out_np = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return out_np, elapsed_ms
