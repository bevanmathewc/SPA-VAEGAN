"""
Atomic Checkpointing and Resumption for Baseline CVAE and SPA-VAEGAN.
Ensures zero corrupted checkpoints during Colab runtime disconnects.
Implements the objective checkpoint selection rule:
Lowest validation LPIPS within 1.5 dB of best validation PSNR.
"""

import os
import random
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch


def atomic_save(payload: Dict[str, Any], filepath: str) -> None:
    """Saves payload to a temporary file first, then atomically renames it."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp_path = filepath + ".tmp"
    torch.save(payload, tmp_path)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass
    os.replace(tmp_path, filepath)


def save_checkpoint(
    filepath: str,
    iteration: int,
    G: torch.nn.Module,
    optG: torch.optim.Optimizer,
    scalerG: Optional[torch.cuda.amp.GradScaler] = None,
    D: Optional[torch.nn.Module] = None,
    optD: Optional[torch.optim.Optimizer] = None,
    scalerD: Optional[torch.cuda.amp.GradScaler] = None,
    lam: Optional[float] = None,
    best_psnr: float = 0.0,
    best_lpips: float = 1.0,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Pack and save training state atomically."""
    payload = {
        "iteration": iteration,
        "G_state_dict": G.state_dict(),
        "optG_state_dict": optG.state_dict(),
        "scalerG_state_dict": scalerG.state_dict() if scalerG is not None else None,
        "D_state_dict": D.state_dict() if D is not None else None,
        "optD_state_dict": optD.state_dict() if optD is not None else None,
        "scalerD_state_dict": scalerD.state_dict() if scalerD is not None else None,
        "lam": lam,
        "best_psnr": best_psnr,
        "best_lpips": best_lpips,
        "config": config,
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
    }
    atomic_save(payload, filepath)



def load_checkpoint(
    filepath: str,
    G: torch.nn.Module,
    optG: Optional[torch.optim.Optimizer] = None,
    scalerG: Optional[torch.cuda.amp.GradScaler] = None,
    D: Optional[torch.nn.Module] = None,
    optD: Optional[torch.optim.Optimizer] = None,
    scalerD: Optional[torch.cuda.amp.GradScaler] = None,
    device: str = "cuda"
) -> Dict[str, Any]:

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Checkpoint not found at: {filepath}"
        )

    # These are trusted project checkpoints.
    checkpoint = torch.load(
        filepath,
        map_location=device,
        weights_only=False
    )

    G.load_state_dict(checkpoint["G_state_dict"])

    if optG is not None and checkpoint.get("optG_state_dict") is not None:
        optG.load_state_dict(checkpoint["optG_state_dict"])

    if scalerG is not None and checkpoint.get("scalerG_state_dict") is not None:
        scalerG.load_state_dict(checkpoint["scalerG_state_dict"])

    if D is not None and checkpoint.get("D_state_dict") is not None:
        D.load_state_dict(checkpoint["D_state_dict"])

    if optD is not None and checkpoint.get("optD_state_dict") is not None:
        optD.load_state_dict(checkpoint["optD_state_dict"])

    if scalerD is not None and checkpoint.get("scalerD_state_dict") is not None:
        scalerD.load_state_dict(checkpoint["scalerD_state_dict"])

    rng_states = checkpoint.get("rng_states")

    if rng_states is not None:

        if rng_states.get("torch") is not None:
            torch.set_rng_state(
                torch.as_tensor(
                    rng_states["torch"],
                    dtype=torch.uint8,
                    device="cpu"
                )
            )

        if (
            rng_states.get("cuda") is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state(
                torch.as_tensor(
                    rng_states["cuda"],
                    dtype=torch.uint8,
                    device="cpu"
                )
            )

        if rng_states.get("numpy") is not None:
            np.random.set_state(rng_states["numpy"])

        if rng_states.get("random") is not None:
            random.setstate(rng_states["random"])

    return checkpoint



class CheckpointSelector:
    """
    Evaluates candidates against the selection rule:
    Lowest validation LPIPS whose validation PSNR is within 1.5 dB of peak PSNR.
    Evaluated only on validation steps after iteration 10,000.
    """
    def __init__(self, psnr_margin: float = 1.5):
        self.psnr_margin = psnr_margin
        self.peak_psnr = -1.0
        self.best_lpips = 999.0
        self.selected_iteration = -1

    def update(self, iteration: int, val_psnr: float, val_lpips: float) -> bool:
        """
        Returns True if the current checkpoint is the new selected best model.
        """
        if val_psnr > self.peak_psnr:
            self.peak_psnr = val_psnr

        # Rule activates after 10k iteration milestone
        if iteration >= 10000:
            if val_psnr >= (self.peak_psnr - self.psnr_margin):
                if val_lpips < self.best_lpips:
                    self.best_lpips = val_lpips
                    self.selected_iteration = iteration
                    return True
        else:
            # Prior to 10k, track standard peak PSNR
            if val_psnr == self.peak_psnr:
                return True
        return False
