"""
Plotting script for training dynamics and stability analysis.
Generates:
1. Training curves (loss progression, validation PSNR comparison)
2. SPA-VAEGAN stability analysis (Discriminator margin, adaptive weight lambda_t, gradient norms)
Saves figures to results/figures/.
"""

import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(baseline_csv: str, spa_csv: str, out_dir: str = "results/figures"):
    os.makedirs(out_dir, exist_ok=True)
    has_base = os.path.exists(baseline_csv) and os.path.getsize(baseline_csv) > 0
    has_spa = os.path.exists(spa_csv) and os.path.getsize(spa_csv) > 0

    if not has_base and not has_spa:
        print("[Warning] No log files found to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SPA-VAEGAN vs Baseline CVAE: Training Dynamics", fontsize=16, fontweight="bold")

    # 1. Pixel Loss (L1)
    ax1 = axes[0, 0]
    if has_base:
        df_b = pd.read_csv(baseline_csv)
        if "L_pix" in df_b.columns and "iteration" in df_b.columns:
            ax1.plot(df_b["iteration"], df_b["L_pix"], label="Baseline CVAE (L1)", color="#1f77b4", alpha=0.8)
    if has_spa:
        df_s = pd.read_csv(spa_csv)
        if "L_pix" in df_s.columns and "iteration" in df_s.columns:
            # SPA-VAEGAN starts from iteration 10,000
            ax1.plot(df_s["iteration"] + 10000, df_s["L_pix"], label="SPA-VAEGAN (L1)", color="#ff7f0e", alpha=0.8)
    ax1.set_title("Reconstruction Pixel Loss (L1)")
    ax1.set_xlabel("Effective Generator Iterations")
    ax1.set_ylabel("L1 Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # 2. Validation PSNR
    ax2 = axes[0, 1]
    if has_base:
        val_b = df_b[df_b["val_psnr"].notnull() & (df_b["val_psnr"] != "")]
        if not val_b.empty:
            ax2.plot(val_b["iteration"], pd.to_numeric(val_b["val_psnr"]), "o-", label="Baseline CVAE", color="#1f77b4")
    if has_spa:
        val_s = df_s[df_s["val_psnr"].notnull() & (df_s["val_psnr"] != "")]
        if not val_s.empty:
            ax2.plot(val_s["iteration"] + 10000, pd.to_numeric(val_s["val_psnr"]), "s-", label="SPA-VAEGAN", color="#ff7f0e")
    ax2.set_title("Validation PSNR (dB)")
    ax2.set_xlabel("Effective Generator Iterations")
    ax2.set_ylabel("PSNR (dB)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # 3. SPA-VAEGAN Discriminator Dynamics
    ax3 = axes[1, 0]
    if has_spa and "L_D" in df_s.columns:
        ax3.plot(df_s["iteration"], df_s["L_D"], label="Discriminator Loss (L_D)", color="#2ca02c", alpha=0.8)
        if "D_real" in df_s.columns and "D_fake" in df_s.columns:
            margin = pd.to_numeric(df_s["D_real"]) - pd.to_numeric(df_s["D_fake"])
            ax3.plot(df_s["iteration"], margin, label="D(real) - D(fake) Margin", color="#d62728", linestyle="--")
    ax3.set_title("Discriminator Stability (Hinge Loss & Margin)")
    ax3.set_xlabel("SPA-VAEGAN Iterations")
    ax3.set_ylabel("Loss / Margin")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    # 4. Adaptive Weight and Gradient Norms
    ax4 = axes[1, 1]
    if has_spa and "lambda_t" in df_s.columns:
        ax4.plot(df_s["iteration"], df_s["lambda_t"], label="Adaptive Weight (lambda_t)", color="#9467bd", linewidth=2)
        ax4_twin = ax4.twinx()
        if "norm_rec" in df_s.columns and "norm_adv" in df_s.columns:
            ax4_twin.plot(df_s["iteration"], df_s["norm_rec"], label="||grad L_rec||", color="#8c564b", alpha=0.5, linestyle=":")
            ax4_twin.plot(df_s["iteration"], df_s["norm_adv"], label="||grad L_adv||", color="#e377c2", alpha=0.5, linestyle=":")
            ax4_twin.set_ylabel("Gradient Norms", color="#8c564b")
    ax4.set_title("Gradient Balancing & Adaptive Weight Dynamics")
    ax4.set_xlabel("SPA-VAEGAN Iterations")
    ax4.set_ylabel("Lambda_t", color="#9467bd")
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend(loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved training curves to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", type=str, default="results/logs/baseline.csv")
    parser.add_argument("--spa_csv", type=str, default="results/logs/spa_vaegan.csv")
    parser.add_argument("--out_dir", type=str, default="results/figures")
    args = parser.parse_args()
    plot_training_curves(args.baseline_csv, args.spa_csv, args.out_dir)


if __name__ == "__main__":
    main()
