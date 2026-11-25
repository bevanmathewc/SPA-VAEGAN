"""
Export lightweight generator weights for deployment.
Strips optimizer states, GradScalers, and Discriminator weights.
Reduces checkpoint file size from >250 MB down to ~7 MB.
"""

import argparse
import os
import torch


def export_generator(input_checkpoint: str, output_path: str):
    """Loads a full training checkpoint and saves only G.state_dict(), config, and iteration."""
    if not os.path.exists(input_checkpoint):
        raise FileNotFoundError(f"Source checkpoint not found at: {input_checkpoint}")

    print(f"Loading checkpoint from: {input_checkpoint}...")
    chk = torch.load(input_checkpoint, map_location="cpu")

    # Extract state dict for generator
    g_state = chk.get("G_state_dict") or chk.get("state_dict") or chk
    clean_dict = {
        "state_dict": g_state,
        "iteration": chk.get("iteration", -1),
        "best_psnr": chk.get("best_psnr", None),
        "config": chk.get("config", None)
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(clean_dict, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Successfully exported generator to: {output_path} ({file_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export lightweight generator weights")
    parser.add_argument("--input", type=str, required=True, help="Path to input full checkpoint (.pt)")
    parser.add_argument("--output", type=str, required=True, help="Path to export stripped generator (.pt)")
    args = parser.parse_args()
    export_generator(args.input, args.output)


if __name__ == "__main__":
    main()
