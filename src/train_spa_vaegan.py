"""
Training script for SPA-VAEGAN (Structure-Preserving Adaptive VAE-GAN).
Warm-starts from the Baseline CVAE 10,000-step checkpoint.
Trains for 20,000 iterations with:
- Conditional PatchGAN Discriminator with Spectral Normalization & Hinge Loss
- Sobel Edge Structure Loss (lambda_e = 0.5)
- Dynamic Gradient-Ratio Adaptive Adversarial Weighting (delta = 0.1)
- Strict D/G gradient isolation
"""

import argparse
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips

from models.cvae import CVAE
from models.discriminator import PatchDiscriminator
from src.config import load_config
from src.dataset import PatchDataset
from src.degradation import degrade
from src.losses import kl_loss, SobelEdge, edge_loss
from src.adaptive_weight import AdaptiveWeightTracker
from src.checkpointing import save_checkpoint, load_checkpoint, CheckpointSelector
from src.logging_utils import TrainingMonitor



def validate_spa(
    G: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    lpips_model: torch.nn.Module,
    max_samples: int = 400
):
    """Computes validation PSNR, L1 and LPIPS under prior mode (z = 0)."""

    G.eval()
    lpips_model.eval()

    total_l1 = 0.0
    total_mse = 0.0
    total_lpips = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:

            hr = batch.to(device, non_blocking=True).float() / 255.0
            lr = degrade(hr)

            sr, _, _ = G(lr)

            sr_clamped = sr.float().clamp(0.0, 1.0)
            sr_quant = (sr_clamped * 255.0).round() / 255.0

            mse = F.mse_loss(
                sr_quant, hr, reduction="none"
            ).flatten(1).mean(1)

            l1 = F.l1_loss(
                sr_quant, hr, reduction="none"
            ).flatten(1).mean(1)

            # LPIPS expects RGB images in [-1, 1].
            lpips_value = lpips_model(
                sr_clamped * 2.0 - 1.0,
                hr * 2.0 - 1.0
            ).flatten(1).mean(1)

            total_mse += mse.sum().item()
            total_l1 += l1.sum().item()
            total_lpips += lpips_value.sum().item()

            count += hr.shape[0]

            if count >= max_samples:
                break

    mean_mse = total_mse / count
    mean_l1 = total_l1 / count
    mean_lpips = total_lpips / count

    psnr = min(
        100.0,
        10.0 * torch.log10(
            torch.tensor(1.0 / max(mean_mse, 1e-10))
        ).item()
    )

    G.train()

    return {
        "val_l1": mean_l1,
        "val_psnr": psnr,
        "val_lpips": mean_lpips
    }



def train_spa_vaegan(config_path: str = "configs/default.yaml", patches_dir: str = None):
    cfg = load_config(config_path)
    s_cfg = cfg["spa_vaegan"]
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
        batch_size=s_cfg["batch_size"],
        shuffle=True,
        num_workers=d_cfg.get("num_workers", 2),
        pin_memory=d_cfg.get("pin_memory", True),
        drop_last=True
    )

    val_loader = None
    if os.path.exists(val_npy):
        val_dataset = PatchDataset(val_npy, is_train=False, crop_size=d_cfg["hr_patch_size"])
        val_loader = DataLoader(val_dataset, batch_size=s_cfg["batch_size"], shuffle=False, num_workers=1)

    # Initialize G and D
    G = CVAE(zc=m_cfg["zc"], c=m_cfg["c"], n_blocks=m_cfg["n_blocks"]).to(device)
    D = PatchDiscriminator(in_channels=6).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=s_cfg["lr_G"], betas=tuple(s_cfg["betas_G"]))
    optD = torch.optim.Adam(D.parameters(), lr=s_cfg["lr_D"], betas=tuple(s_cfg["betas_D"]))

    scalerG = torch.cuda.amp.GradScaler(enabled=(s_cfg["amp"] and device.type == "cuda"))
    scalerD = torch.cuda.amp.GradScaler(enabled=(s_cfg["amp"] and device.type == "cuda"))

    sobel = SobelEdge().to(device)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

for p in lpips_model.parameters():
    p.requires_grad_(False)

    weight_tracker = AdaptiveWeightTracker(
        delta=s_cfg["delta"],
        lambda_max=s_cfg["lambda_max"],
        eps=s_cfg["eps_adv"]
    )

    chk_dir = s_cfg.get("checkpoint_dir", "checkpoints/spa_vaegan")
    local_chk_dir = s_cfg.get("local_checkpoint_dir", "checkpoints/spa_vaegan")
    os.makedirs(chk_dir, exist_ok=True)
    os.makedirs(local_chk_dir, exist_ok=True)

    last_pt = os.path.join(chk_dir, "last.pt")
    start_iteration = 0
    best_psnr = 0.0

    # Warm-start or Resume
    if os.path.exists(last_pt):
        print(f"Resuming SPA-VAEGAN from checkpoint: {last_pt}")
        chk = load_checkpoint(last_pt, G, optG, scalerG, D, optD, scalerD, device=str(device))
        start_iteration = chk["iteration"]
        best_psnr = chk.get("best_psnr", 0.0)
        weight_tracker.lam = chk.get("lam")
        print(f"Resumed at iteration {start_iteration} with lambda: {weight_tracker.lam}")
    else:
        # Load 10k baseline milestone checkpoint for shared warm-start
        warmstart_path = s_cfg["warmstart_checkpoint"]
        alt_warmstart = os.path.join(cfg["baseline"].get("local_checkpoint_dir", "checkpoints/baseline"), "baseline_it10000.pt")
        target_path = warmstart_path if os.path.exists(warmstart_path) else alt_warmstart
        
        if os.path.exists(target_path):
            print(f"Initializing SPA-VAEGAN generator from 10k milestone: {target_path}")
            load_checkpoint(target_path, G, device=str(device))
        else:
            print(f"[Warning] Warmstart milestone not found at {target_path}. Initializing from scratch.")

    # Logging monitor
    log_file = "results/logs/spa_vaegan.csv"
    fieldnames = [
        "iteration", "step_time", "L_pix", "L_edge", "L_adv", "L_D",
        "D_real", "D_fake", "lambda_t", "raw_lambda", "norm_rec", "norm_adv",
        "val_l1", "val_psnr","val_lpips"
    ]
    monitor = TrainingMonitor(log_file, fieldnames)
    selector = CheckpointSelector(psnr_margin=1.5)

    data_iter = iter(train_loader)
    total_iters = s_cfg["total_iterations"]
    lam_e = s_cfg["lambda_edge"]
    beta_kl = s_cfg["beta_kl"]
    amp = s_cfg["amp"] and device.type == "cuda"

    print(f"Beginning SPA-VAEGAN adversarial training: {start_iteration} -> {total_iters} iterations...")
    G.train()
    D.train()

    for it in range(start_iteration + 1, total_iters + 1):
        monitor.tic()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        hr = batch.to(device, non_blocking=True).float() / 255.0
        lr = degrade(hr)
        lr_up = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)

        # -------------------------------------------------------------
        # Generator Forward Pass (graph attached to G only)
        # -------------------------------------------------------------
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            sr, mu, logvar = G(lr, hr)

        # -------------------------------------------------------------
        # 1) Discriminator Step: SR is DETACHED so G gets zero gradient
        # -------------------------------------------------------------
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            d_real = D(hr, lr_up)
            d_fake = D(sr.detach(), lr_up)
            # Hinge loss
            loss_D = F.relu(1.0 - d_real.float()).mean() + F.relu(1.0 + d_fake.float()).mean()

        optD.zero_grad(set_to_none=True)
        scalerD.scale(loss_D).backward()
        scalerD.step(optD)
        scalerD.update()

        # -------------------------------------------------------------
        # 2) Generator Step: D is FROZEN so optD receives no gradients
        # -------------------------------------------------------------
        for p in D.parameters():
            p.requires_grad_(False)

        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            # Adversarial generator loss evaluated on updated D
            l_adv = -D(sr, lr_up).float().mean()

        l_pix = F.l1_loss(sr.float(), hr)
        l_edge = edge_loss(sobel, sr, hr)
        l_kl = kl_loss(mu, logvar)
        l_rec = l_pix + lam_e * l_edge

        # Adaptive adversarial weight calculation
        lam_use, raw_val, norm_rec, norm_adv = weight_tracker.step(
            l_rec=l_rec,
            l_adv=l_adv,
            ref_weight=G.dec.last.weight,
            scaler=scalerG
        )

        loss_G = l_rec + beta_kl * l_kl + lam_use * l_adv

        optG.zero_grad(set_to_none=True)
        scalerG.scale(loss_G).backward()
        scalerG.step(optG)
        scalerG.update()

        # Restore D gradients for subsequent iteration
        for p in D.parameters():
            p.requires_grad_(True)

        metrics = {
            "L_pix": round(l_pix.item(), 5),
            "L_edge": round(l_edge.item(), 5),
            "L_adv": round(l_adv.item(), 5),
            "L_D": round(loss_D.item(), 5),
            "D_real": round(d_real.mean().item(), 4),
            "D_fake": round(d_fake.mean().item(), 4),
            "lambda_t": round(lam_use, 5),
            "raw_lambda": round(raw_val, 5) if raw_val is not None else "",
            "norm_rec": round(norm_rec, 5),
            "norm_adv": round(norm_adv, 5)
        }

        # Regular logging
        if it % s_cfg["log_interval"] == 0 or it == 1:
            monitor.log_step(it, metrics, print_console=(it % 500 == 0 or it == 1))

        # Validation
        if val_loader is not None and (it % s_cfg["val_interval"] == 0 or it == total_iters):
            val_results = validate_spa(G, val_loader, device, lpips_model)
            metrics.update(val_results)
            monitor.log_step(it, metrics, print_console=True)

            val_psnr = val_results["val_psnr"]
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_path = os.path.join(chk_dir, "best.pt")
                local_best = os.path.join(local_chk_dir, "best.pt")
                save_checkpoint(best_path, it, G, optG, scalerG, D, optD, scalerD, lam=lam_use, best_psnr=best_psnr, config=cfg)
                save_checkpoint(local_best, it, G, optG, scalerG, D, optD, scalerD, lam=lam_use, best_psnr=best_psnr, config=cfg)
                print(f"New best validation PSNR: {best_psnr:.2f} dB (Saved best.pt)")

        # Periodic checkpointing
        if it % s_cfg["save_interval"] == 0 or it == total_iters:
            save_checkpoint(last_pt, it, G, optG, scalerG, D, optD, scalerD, lam=lam_use, best_psnr=best_psnr, config=cfg)
            save_checkpoint(os.path.join(local_chk_dir, "last.pt"), it, G, optG, scalerG, D, optD, scalerD, lam=lam_use, best_psnr=best_psnr, config=cfg)

    monitor.close()
    print("SPA-VAEGAN adversarial training completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--patches_dir", type=str, default=None)
    args = parser.parse_args()
    train_spa_vaegan(args.config, args.patches_dir)
