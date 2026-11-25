"""
Inference Latency and Parameter Count Profiling.
Measures:
- Total and trainable parameters for Generator and Discriminator
- Runtime latency on 64x64 input (batch size 1, 20 warmup, 100 timed trials)
"""

import argparse
import time
from typing import Dict, Tuple
import numpy as np
import torch
from models.cvae import CVAE
from models.discriminator import PatchDiscriminator
from src.config import load_config


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 64, 64),
    device: str = "cuda",
    warmup: int = 20,
    trials: int = 100
) -> Dict[str, float]:
    """Measures median inference latency and standard deviation in milliseconds."""
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = model.to(dev).eval()

    dummy_input = torch.randn(input_shape, device=dev, dtype=torch.float32)

    # Warmup runs
    for _ in range(warmup):
        _ = model(dummy_input)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    timings = []
    for _ in range(trials):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(dummy_input)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)  # ms

    timings = np.array(timings)
    return {
        "median_ms": float(np.median(timings)),
        "mean_ms": float(np.mean(timings)),
        "std_ms": float(np.std(timings)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
        "device": str(dev)
    }


def main():
    parser = argparse.ArgumentParser(description="Latency and Parameter Count Profiling")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    m_cfg = cfg["model"]

    # Models
    G = CVAE(zc=m_cfg["zc"], c=m_cfg["c"], n_blocks=m_cfg["n_blocks"])
    D = PatchDiscriminator(in_channels=6)

    g_total, g_trainable = count_parameters(G)
    d_total, d_trainable = count_parameters(D)

    print("=" * 60)
    print("MODEL PARAMETER PROFILING")
    print("=" * 60)
    print(f"Generator (CVAE) Parameters:     {g_total:,} ({g_total / 1e6:.2f}M)")
    print(f"Discriminator (PatchGAN) Params: {d_total:,} ({d_total / 1e6:.2f}M)")
    print("=" * 60)

    # Benchmark GPU latency
    if torch.cuda.is_available():
        gpu_stats = measure_latency(G, device="cuda")
        print(f"GPU Inference Latency (Batch 1, 64x64 -> 256x256):")
        print(f"  Median: {gpu_stats['median_ms']:.2f} ms +/- {gpu_stats['std_ms']:.2f} ms")
        print(f"  Range:  [{gpu_stats['min_ms']:.2f} ms, {gpu_stats['max_ms']:.2f} ms]")

    # Benchmark CPU latency
    cpu_stats = measure_latency(G, device="cpu", warmup=5, trials=20)
    print(f"CPU Inference Latency (Batch 1, 64x64 -> 256x256):")
    print(f"  Median: {cpu_stats['median_ms']:.2f} ms +/- {cpu_stats['std_ms']:.2f} ms")
    print(f"  Range:  [{cpu_stats['min_ms']:.2f} ms, {cpu_stats['max_ms']:.2f} ms]")
    print("=" * 60)


if __name__ == "__main__":
    main()
