"""
PyTorch Dataset for memory-mapped DIV2K patches.
Applies:
- Random 256x256 sub-crop from stored 288x288 patch (training only)
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random 90-degree rotations (k in {0, 1, 2, 3})
Returns uint8 tensor of shape (3, 256, 256).
"""

import os
import random
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """
    Memory-mapped patch dataset.
    Reads uint8 arrays of shape (N, 3, H, W) directly from disk via np.lib.format.open_memmap.
    """
    def __init__(
        self,
        npy_path: str,
        is_train: bool = True,
        crop_size: int = 256,
        target_size: Optional[int] = None
    ):
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Patch bank not found at: {npy_path}")
            
        self.npy_path = npy_path
        self.is_train = is_train
        self.crop_size = crop_size
        self.data = np.lib.format.open_memmap(npy_path, mode="r")
        self.num_samples = self.data.shape[0]
        self.stored_size = self.data.shape[-1]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Load (3, H, W) uint8 patch from memmap
        patch = np.array(self.data[idx], copy=True)  # (3, H, W)

        if self.is_train:
            # 1. Random 256x256 subcrop from 288x288
            diff = self.stored_size - self.crop_size
            if diff > 0:
                top = random.randint(0, diff)
                left = random.randint(0, diff)
                patch = patch[:, top:top + self.crop_size, left:left + self.crop_size]

            # 2. Random horizontal flip
            if random.random() > 0.5:
                patch = patch[:, :, ::-1]

            # 3. Random vertical flip
            if random.random() > 0.5:
                patch = patch[:, ::-1, :]

            # 4. Random 90-degree rotations
            k = random.randint(0, 3)
            if k > 0:
                patch = np.rot90(patch, k, (1, 2))

        # Return as contiguous uint8 torch Tensor (3, 256, 256)
        return torch.from_numpy(np.ascontiguousarray(patch))
