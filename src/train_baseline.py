"""
Training script for Baseline Conditional VAE (CVAE).
Trains for 30,000 iterations with L1 pixel loss and KL divergence with linear warmup.
Saves milestone checkpoint at iteration 10,000 (baseline_it10000.pt) as the warm-start for SPA-VAEGAN.
"""

import argparse
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cvae import CVAE
from src.config import load_config
from src.dataset import PatchDataset
from src.degradation import degrade
from src.losses import kl_loss
from src.checkpointing import save_checkpoint, load_checkpoint, CheckpointSelector
from src.logging_utils import TrainingMonitor


def validate(G: torch.nn.Module, val_loader: DataLoader, device: torch.device, max_samples: int = 400):
    """Computes validation PSNR, SSIM, and L1 loss under prior mode (z = 0)."""
    G.eval()
    total_l1 = 0.0
    total_mse = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            hr = batch.to(device, non_blocking=True).float() / 255.0
            lr = degrade(hr)
            # Prior mode inference: z = 0
            sr, _, _ = G(lr)
            sr_quant = (sr.clamp(0.0, 1.0) * 255.0).round() / 255.0

            mse = F.mse_loss(sr_quant, hr, reduction="none").flatten(1).mean(1)
            l1 = F.l1_loss(sr_quant, hr, reduction="none").flatten(1).mean(1)

            total_mse += mse.sum().item()
            total_l1 += l1.sum().item()
            count += hr.shape[0]
            if count >= max_samples:
                break

    mean_mse = total_mse / count
    mean_l1 = total_l1 / count
    psnr = min(100.0, 10.0 * torch.log10(torch.tensor(1.0 / max(mean_mse, 1e-10))).item())
    G.train()
    return {"val_l1": mean_l1, "val_psnr": psnr}


def train_baseline(config_path: str = "configs/default.yaml", patches_dir: str = None):
    cfg = load_config(config_path)
    b_cfg = cfg["baseline"]
    m_cfg = cfg["model"]
    d_cfg = cfg["dataset"]

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Patch paths
    p_dir = patches_dir or d_cfg.get("patches_dir", "dataset/patches")
    train_npy = os.path.join(p_dir, "train_patches.npy")
    val_npy = os.path.join(p_dir, "val_patches.npy")

    if not os.path.exists(train_npy):
        raise FileNotFoundError(f"Train patch bank not found at {train_npy}. Run src/make_patches.py first.")

    train_dataset = PatchDataset(train_npy, is_train=True, crop_size=d_cfg["hr_patch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=b_cfg["batch_size"],
        shuffle=True,
        num_workers=d_cfg.get("num_workers", 2),
        pin_memory=d_cfg.get("pin_memory", True),
        drop_last=True
    )

    val_loader = None
    if os.path.exists(val_npy):
        val_dataset = PatchDataset(val_npy, is_train=False, crop_size=d_cfg["hr_patch_size"])
        val_loader = DataLoader(val_dataset, batch_size=b_cfg["batch_size"], shuffle=False, num_workers=1)

    # Initialize model, optimizer, GradScaler
    G = CVAE(zc=m_cfg["zc"], c=m_cfg["c"], n_blocks=m_cfg["n_blocks"]).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=b_cfg["lr"], betas=tuple(b_cfg["betas"]))
    scalerG = torch.cuda.amp.GradScaler(enabled=(b_cfg["amp"] and device.type == "cuda"))

    # Checkpoint directories
    chk_dir = b_cfg.get("checkpoint_dir", "checkpoints/baseline")
    local_chk_dir = b_cfg.get("local_checkpoint_dir", "checkpoints/baseline")
    os.makedirs(chk_dir, exist_ok=True)
    os.makedirs(local_chk_dir, exist_ok=True)

    last_pt = os.path.join(chk_dir, "last.pt")
    start_iteration = 0
    best_psnr = 0.0

    # Resume if last.pt exists
    if os.path.exists(last_pt):
        print(f"Resuming training from checkpoint: {last_pt}")
        chk = load_checkpoint(last_pt, G, optG, scalerG, device=str(device))
        start_iteration = chk["iteration"]
        best_psnr = chk.get("best_psnr", 0.0)
        print(f"Resumed at iteration {start_iteration} with best PSNR: {best_psnr:.2f} dB")

    # Logging monitor
    log_file = "results/logs/baseline.csv"
    fieldnames = ["iteration", "step_time", "L_pix", "L_KL", "beta_t", "val_l1", "val_psnr"]
    monitor = TrainingMonitor(log_file, fieldnames)
    selector = CheckpointSelector(psnr_margin=1.5)

    data_iter = iter(train_loader)
    total_iters = b_cfg["total_iterations"]
    milestone_it = b_cfg["milestone_iteration"]
    warmup_steps = b_cfg["kl_warmup_steps"]
    beta_kl = b_cfg["beta_kl"]

    print(f"Beginning Baseline CVAE training: {start_iteration} -> {total_iters} iterations...")
    G.train()

    for it in range(start_iteration + 1, total_iters + 1):
        monitor.tic()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        hr = batch.to(device, non_blocking=True).float() / 255.0
        lr = degrade(hr)

        # Dynamic KL warm-up
        beta_t = min(beta_kl, beta_kl * (it / float(warmup_steps)))

        optG.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(b_cfg["amp"] and device.type == "cuda")):
            sr, mu, logvar = G(lr, hr)
            l_pix = F.l1_loss(sr.float(), hr)
            l_kl = kl_loss(mu, logvar)
            loss_G = l_pix + beta_t * l_kl

        scalerG.scale(loss_G).backward()
        scalerG.step(optG)
        scalerG.update()

        metrics = {
            "L_pix": round(l_pix.item(), 5),
            "L_KL": round(l_kl.item(), 5),
            "beta_t": round(beta_t, 6)
        }

        # Regular logging
        if it % b_cfg["log_interval"] == 0 or it == 1:
            monitor.log_step(it, metrics, print_console=(it % 500 == 0 or it == 1))

        # Milestone saving at 10,000 steps
        if it == milestone_it:
            milestone_path = os.path.join(chk_dir, "baseline_it10000.pt")
            local_milestone = os.path.join(local_chk_dir, "baseline_it10000.pt")
            print(f"Reached milestone {milestone_it}! Saving {milestone_path}...")
            save_checkpoint(milestone_path, it, G, optG, scalerG, best_psnr=best_psnr, config=cfg)
            save_checkpoint(local_milestone, it, G, optG, scalerG, best_psnr=best_psnr, config=cfg)

        # Validation
        if val_loader is not None and (it % b_cfg["val_interval"] == 0 or it == total_iters):
            val_results = validate(G, val_loader, device)
            metrics.update(val_results)
            monitor.log_step(it, metrics, print_console=True)

            val_psnr = val_results["val_psnr"]
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_path = os.path.join(chk_dir, "best.pt")
                local_best = os.path.join(local_chk_dir, "best.pt")
                save_checkpoint(best_path, it, G, optG, scalerG, best_psnr=best_psnr, config=cfg)
                save_checkpoint(local_best, it, G, optG, scalerG, best_psnr=best_psnr, config=cfg)
                print(f"New best validation PSNR: {best_psnr:.2f} dB (Saved best.pt)")

        # Periodic checkpointing
        if it % b_cfg["save_interval"] == 0 or it == total_iters:
            save_checkpoint(last_pt, it, G, optG, scalerG, best_psnr=best_psnr, config=cfg)
            save_checkpoint(os.path.join(local_chk_dir, "last.pt"), it, G, optG, scalerG, best_psnr=best_psnr, config=cfg)

    monitor.close()
    print("Baseline CVAE training completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--patches_dir", type=str, default=None)
    args = parser.parse_args()
    train_baseline(args.config, args.patches_dir)
