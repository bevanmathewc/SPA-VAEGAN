"""
Script to create memory-mapped patch banks from DIV2K images.
Extracts:
- Train: 21,600 patches of size 288x288 (30 per image across 720 images, filtering flat sky)
- Val: 400 patches of size 256x256 (5 fixed per image across 80 images)
- Test: 2,000 patches of size 256x256 (20 fixed per image across 100 test images)
Saved as memory-mapped .npy files (uint8, shape (N, 3, H, W)) for ultra-fast GPU training.
"""

import argparse
import os
import random
from typing import List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm
from src.config import load_config


def extract_fixed_crops(img: Image.Image, patch_size: int, num_crops: int, seed: int = 42) -> List[np.ndarray]:
    """Extract deterministic fixed patches distributed across an image."""
    w, h = img.size
    if w < patch_size or h < patch_size:
        img = img.resize((max(w, patch_size), max(h, patch_size)), Image.BICUBIC)
        w, h = img.size

    rng = random.Random(seed)
    crops = []
    
    # Grid coordinate spacing
    x_steps = np.linspace(0, max(0, w - patch_size), int(np.ceil(np.sqrt(num_crops))), dtype=int)
    y_steps = np.linspace(0, max(0, h - patch_size), int(np.ceil(np.sqrt(num_crops))), dtype=int)
    grid_coords = [(x, y) for x in x_steps for y in y_steps]
    rng.shuffle(grid_coords)

    for i in range(num_crops):
        if i < len(grid_coords):
            x, y = grid_coords[i]
        else:
            x = rng.randint(0, w - patch_size)
            y = rng.randint(0, h - patch_size)
        crop = img.crop((x, y, x + patch_size, y + patch_size))
        # Convert HWC to CHW uint8
        arr = np.array(crop).transpose(2, 0, 1)
        crops.append(arr)
    return crops


def extract_train_patches(img: Image.Image, patch_size: int = 288, num_crops: int = 30, rng: random.Random = None) -> List[np.ndarray]:
    """
    Extract random crops of patch_size from image.
    Filters out overly flat regions (pixel std < 10) to prioritize structural content,
    while allowing up to 10% smooth crops.
    """
    w, h = img.size
    if w < patch_size or h < patch_size:
        img = img.resize((max(w, patch_size), max(h, patch_size)), Image.BICUBIC)
        w, h = img.size

    crops = []
    smooth_count = 0
    max_smooth = int(num_crops * 0.1)

    attempts = 0
    while len(crops) < num_crops and attempts < num_crops * 10:
        attempts += 1
        x = rng.randint(0, w - patch_size)
        y = rng.randint(0, h - patch_size)
        crop = img.crop((x, y, x + patch_size, y + patch_size))
        arr = np.array(crop)  # HWC
        
        # Check standard deviation across grayscale luminance
        gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
        std = np.std(gray)
        
        if std < 10.0:  # Flat patch (e.g. uniform sky or wall)
            if smooth_count < max_smooth:
                smooth_count += 1
                crops.append(arr.transpose(2, 0, 1))
            continue
        crops.append(arr.transpose(2, 0, 1))

    # Fill remainder if strict filter exceeded attempts
    while len(crops) < num_crops:
        x = rng.randint(0, w - patch_size)
        y = rng.randint(0, h - patch_size)
        crop = img.crop((x, y, x + patch_size, y + patch_size))
        crops.append(np.array(crop).transpose(2, 0, 1))

    return crops


def build_patch_bank(
    image_names: List[str],
    img_dir: str,
    out_path: str,
    patch_size: int,
    patches_per_img: int,
    is_train: bool = False,
    seed: int = 42
) -> None:
    """Builds and writes a memory-mapped .npy patch bank."""
    total_patches = len(image_names) * patches_per_img
    shape = (total_patches, 3, patch_size, patch_size)
    print(f"Allocating memory-mapped array at {out_path} with shape {shape} (uint8)...")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mmap = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=shape)

    rng = random.Random(seed)
    ptr = 0
    for idx, name in enumerate(tqdm(image_names, desc=f"Processing {os.path.basename(out_path)}")):
        path = os.path.join(img_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Source image not found: {path}")
        with Image.open(path) as img:
            img = img.convert("RGB")
            if is_train:
                crops = extract_train_patches(img, patch_size, patches_per_img, rng)
            else:
                img_seed = seed + idx * 1000
                crops = extract_fixed_crops(img, patch_size, patches_per_img, img_seed)

        for c in crops:
            mmap[ptr] = c
            ptr += 1

    mmap.flush()
    print(f"Successfully created patch bank: {out_path} ({total_patches} patches).")


def main():
    parser = argparse.ArgumentParser(description="Create patch banks from DIV2K")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--patches_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    d_cfg = cfg["dataset"]
    patches_dir = args.patches_dir or d_cfg.get("patches_dir", "dataset/patches")

    # Load split text files
    with open(os.path.join(d_cfg["splits_dir"], "train.txt"), "r") as f:
        train_names = [line.strip() for line in f if line.strip()]
    with open(os.path.join(d_cfg["splits_dir"], "val.txt"), "r") as f:
        val_names = [line.strip() for line in f if line.strip()]
    with open(os.path.join(d_cfg["splits_dir"], "test.txt"), "r") as f:
        test_names = [line.strip() for line in f if line.strip()]

    print(f"Loaded splits: {len(train_names)} train, {len(val_names)} val, {len(test_names)} test.")

    # Build Train Patches
    train_out = os.path.join(patches_dir, "train_patches.npy")
    build_patch_bank(
        image_names=train_names,
        img_dir=d_cfg["div2k_train_dir"],
        out_path=train_out,
        patch_size=d_cfg["stored_patch_size"],  # 288
        patches_per_img=d_cfg["train_patches_per_image"],  # 30
        is_train=True,
        seed=cfg["seed"]
    )

    # Build Validation Patches
    val_out = os.path.join(patches_dir, "val_patches.npy")
    build_patch_bank(
        image_names=val_names,
        img_dir=d_cfg["div2k_train_dir"],
        out_path=val_out,
        patch_size=d_cfg["hr_patch_size"],  # 256
        patches_per_img=d_cfg["val_patches_per_image"],  # 5
        is_train=False,
        seed=cfg["seed"] + 100
    )

    # Build Test Patches
    test_out = os.path.join(patches_dir, "test_patches.npy")
    build_patch_bank(
        image_names=test_names,
        img_dir=d_cfg["div2k_valid_dir"],
        out_path=test_out,
        patch_size=d_cfg["hr_patch_size"],  # 256
        patches_per_img=d_cfg["test_patches_per_image"],  # 20
        is_train=False,
        seed=cfg["seed"] + 200
    )


if __name__ == "__main__":
    main()
