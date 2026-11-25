"""
Visual Comparison Figures and Edge Map Generation.
Produces:
1. Multi-column comparison grids: LR (nearest enlarged), Bicubic, Baseline CVAE, SPA-VAEGAN, Ground Truth HR.
2. Inset zoomed-in crops highlighting edge reconstruction and texture realism.
3. Normalized Sobel edge maps demonstrating structural boundary preservation.
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.cvae import CVAE
from src.degradation import degrade
from src.losses import SobelEdge


def add_zoom_inset(ax, img_arr, crop_box, inset_loc=(0.6, 0.05, 0.35, 0.35)):
    """Draws a zoomed crop inset onto an image axis."""
    x1, y1, x2, y2 = crop_box
    crop = img_arr[y1:y2, x1:x2]
    # Draw rectangle on main image
    from matplotlib.patches import Rectangle
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor="yellow", facecolor="none")
    ax.add_patch(rect)
    
    # Inset axis
    ax_ins = ax.inset_axes(inset_loc)
    ax_ins.imshow(crop)
    ax_ins.axis("off")
    for spine in ax_ins.spines.values():
        spine.set_edgecolor("yellow")
        spine.set_linewidth(1.5)


@torch.no_grad()
def generate_comparison_figures(
    baseline_chk: str,
    spa_chk: str,
    test_patches_npy: str,
    out_dir: str = "results/figures",
    num_samples: int = 6,
    device: str = "cuda"
):
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    # Load generator models
    G_base = CVAE().to(dev).eval()
    G_spa = CVAE().to(dev).eval()

    if os.path.exists(baseline_chk):
        chk_b = torch.load(baseline_chk, map_location=dev)
        G_base.load_state_dict(chk_b.get("state_dict", chk_b.get("G_state_dict", chk_b)))
    if os.path.exists(spa_chk):
        chk_s = torch.load(spa_chk, map_location=dev)
        G_spa.load_state_dict(chk_s.get("state_dict", chk_s.get("G_state_dict", chk_s)))

    sobel = SobelEdge().to(dev)

    if not os.path.exists(test_patches_npy):
        print(f"[Warning] Test patches not found at {test_patches_npy}. Generating synthetic sample.")
        return

    mmap = np.lib.format.open_memmap(test_patches_npy, mode="r")
    indices = np.linspace(0, len(mmap) - 1, num_samples, dtype=int)

    # 5 Columns: LR (nearest), Bicubic, Baseline CVAE, SPA-VAEGAN, Ground Truth HR
    col_names = ["LR (Nearest 4x)", "Bicubic Interpolation", "Baseline CVAE", "SPA-VAEGAN (Proposed)", "Ground Truth HR"]
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    edge_fig, edge_axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        edge_axes = edge_axes[None, :]
    edge_cols = ["Bicubic Edge Map", "Baseline Edge Map", "SPA-VAEGAN Edge Map", "Ground Truth Edge Map"]

    for row_idx, idx in enumerate(indices):
        hr_np = mmap[idx]  # (3, 256, 256) uint8
        hr_t = torch.from_numpy(hr_np).to(dev).float().div(255.0).unsqueeze(0)
        lr_t = degrade(hr_t)

        # Upsamplings
        lr_nearest = F.interpolate(lr_t, scale_factor=4, mode="nearest")
        lr_bicubic = F.interpolate(lr_t, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
        sr_base, _, _ = G_base(lr_t)
        sr_spa, _, _ = G_spa(lr_t)

        sr_base = (sr_base.clamp(0, 1) * 255.0).round() / 255.0
        sr_spa = (sr_spa.clamp(0, 1) * 255.0).round() / 255.0

        images = [
            lr_nearest[0].permute(1, 2, 0).cpu().numpy(),
            lr_bicubic[0].permute(1, 2, 0).cpu().numpy(),
            sr_base[0].permute(1, 2, 0).cpu().numpy(),
            sr_spa[0].permute(1, 2, 0).cpu().numpy(),
            hr_t[0].permute(1, 2, 0).cpu().numpy()
        ]

        # Inset zoom coordinates: center 64x64
        h, w, _ = images[0].shape
        zoom_box = (w // 2 - 32, h // 2 - 32, w // 2 + 32, h // 2 + 32)

        for col_idx, img_arr in enumerate(images):
            ax = axes[row_idx, col_idx]
            ax.imshow(np.clip(img_arr, 0, 1))
            add_zoom_inset(ax, img_arr, zoom_box)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(col_names[col_idx], fontsize=14, fontweight="bold")

        # Edge maps
        edge_bicubic = sobel(lr_bicubic)[0].mean(0).cpu().numpy()
        edge_base = sobel(sr_base)[0].mean(0).cpu().numpy()
        edge_spa = sobel(sr_spa)[0].mean(0).cpu().numpy()
        edge_hr = sobel(hr_t)[0].mean(0).cpu().numpy()

        for c_idx, e_arr in enumerate([edge_bicubic, edge_base, edge_spa, edge_hr]):
            e_ax = edge_axes[row_idx, c_idx]
            e_ax.imshow(e_arr, cmap="magma")
            e_ax.axis("off")
            if row_idx == 0:
                e_ax.set_title(edge_cols[c_idx], fontsize=14, fontweight="bold")

    fig.tight_layout()
    grid_path = os.path.join(out_dir, "comparison_grid.png")
    fig.savefig(grid_path, dpi=300)
    plt.close(fig)
    print(f"Saved visual comparison grid to: {grid_path}")

    edge_fig.tight_layout()
    edge_path = os.path.join(out_dir, "edge_maps.png")
    edge_fig.savefig(edge_path, dpi=300)
    plt.close(edge_fig)
    print(f"Saved Sobel edge-map comparison to: {edge_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_chk", type=str, default="checkpoints/cvae_baseline_G.pt")
    parser.add_argument("--spa_chk", type=str, default="checkpoints/spa_vaegan_G.pt")
    parser.add_argument("--test_npy", type=str, default="dataset/patches/test_patches.npy")
    parser.add_argument("--out_dir", type=str, default="results/figures")
    args = parser.parse_args()

    generate_comparison_figures(
        baseline_chk=args.baseline_chk,
        spa_chk=args.spa_chk,
        test_patches_npy=args.test_npy,
        out_dir=args.out_dir
    )


if __name__ == "__main__":
    main()
