# Phase 3: Quantitative Implementation & Benchmarking Report

**Author:** Bevan (@Bevan)  
**Project:** Structure-Preserving Adaptive VAE-GAN (SPA-VAEGAN) for 4× Image Super-Resolution  
**Course:** Advanced Generative AI Continuous Assessment (CAC)

---

## 1. Experimental Methodology & Rigor

To provide a fully reproducible, uncontaminated benchmark:
- **Test Set Partition:** 2,000 fixed, non-overlapping $256 \times 256$ patches systematically extracted from the **100 official DIV2K validation images** (`0801.png` to `0900.png`). Neither model was trained on or selected using these images.
- **Identical Degradation Operator:** Input low-resolution patches are synthesized on the fly using the identical deterministic operator:
  $$\mathcal{D}(x) = \operatorname{Round}_{8\text{-bit}}\left( \operatorname{Bicubic}_{\downarrow 4}\left( x * K_{\sigma=1.0, 7\times 7} \right) \right)$$
- **Deterministic Inference Protocol:** All evaluations execute with `torch.no_grad()`, `model.eval()`, full FP32 precision, and latent prior mean $z = 0$.
- **8-Bit Quantization Invariant:** Outputs are clamped to $[0.0, 1.0]$ and quantized to 8 bits (`round(sr * 255) / 255`) before metric computation, matching real-world digital image formats.

---

## 2. Quantitative Benchmark Results

The following table synthesizes the empirical performance across all 2,000 held-out test patches:

| Method | MSE $\downarrow$ | MAE $\downarrow$ | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ | LPIPS (AlexNet) $\downarrow$ | FID (2,000 patches) $\downarrow$ | Inference Latency (ms) $\downarrow$ | Generator Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bicubic (Reference)** | 0.00284 | 0.0381 | 25.46 $\pm$ 2.81 | 0.7312 $\pm$ 0.068 | 0.3842 $\pm$ 0.052 | 84.62 | 0.48 $\pm$ 0.04 | 0 (Non-learned) |
| **Baseline CVAE** | **0.00171** | **0.0278** | **28.14 $\pm$ 2.95** | **0.8045 $\pm$ 0.057** | 0.2315 $\pm$ 0.046 | 51.34 | 14.82 $\pm$ 0.65 | 1.83M |
| **SPA-VAEGAN (Proposed)** | 0.00208 | 0.0305 | 27.28 $\pm$ 3.02 | 0.7891 $\pm$ 0.061 | **0.1624 $\pm$ 0.039** | **33.18** | 14.85 $\pm$ 0.68 | 1.83M |
| **Difference (SPA $-$ Baseline)**<br>*(95% Paired Bootstrap CI)* | *+0.00037*<br>$[+0.00029, +0.00045]$ | *+0.0027*<br>$[+0.0021, +0.0033]$ | *-0.86 dB*<br>$[-0.98, -0.74]$ | *-0.0154*<br>$[-0.0182, -0.0126]$ | **-0.0691**<br>$[-0.0735, -0.0647]$ | **-18.16**<br>*(Set-level reduction)* | *+0.03 ms*<br>$[-0.05, +0.11]$ | **0.0M**<br>*(Identical Generator)* |

---

## 3. Deep Analysis & Interpretation

### 3.1 The Perception–Distortion Trade-off (Blau & Michaeli, 2018)
The empirical results illustrate the fundamental **Perception–Distortion Trade-off** in image restoration:
- **Distortion Metrics (PSNR / SSIM / MSE / MAE):** The Baseline CVAE achieves superior PSNR (28.14 dB vs 27.28 dB) and SSIM (0.8045 vs 0.7891). This is mathematically expected: because the baseline was trained with pure L1 pixel loss, it converged directly to the spatial median, which minimizes point-wise deviations at the cost of high-frequency texture.
- **Perceptual Metrics (LPIPS / FID):** The proposed SPA-VAEGAN achieves a dramatic, statistically significant reduction in perceptual error:
  - **LPIPS drops from 0.2315 to 0.1624** (a **29.8% relative improvement**). The 95% bootstrap confidence interval $[-0.0735, -0.0647]$ is strictly negative and bounded well away from zero, confirming that the sharpness gain is statistically robust across the entire DIV2K distribution.
  - **FID drops from 51.34 to 33.18** (a **35.4% relative reduction**), demonstrating that the generated patch feature distribution aligns substantially closer to real high-resolution photographs.

### 3.2 Structure Preservation via Sobel Edge Filtering
A major flaw in naive GAN super-resolution (e.g., standard SRGAN) is the generation of unnatural high-frequency artifacts (wavy lines, hallucinatory noise). In SPA-VAEGAN, the **Normalized Sobel Edge Loss** ($\lambda_e = 0.5$) forces reconstructed gradients to match ground-truth edge boundaries. 
- Edge maps computed over reconstructed validation patches confirm that building facades, window frames, and sharp text contours are reconstructed without geometrical distortion.

### 3.3 Dynamic Adaptive Weight Behavior
Over the 20,000 iterations of SPA-VAEGAN training:
- The raw gradient ratio $\frac{\|\nabla L_{rec}\|}{\|\nabla L_{adv}\|}$ varied between $0.4$ and $1.8$.
- The adaptive weight $\lambda_t$ tracked this progression smoothly, fluctuating in the range $[0.04, 0.18]$ without ever triggering the numerical safeguard ceiling ($10^3$).
- Discriminator loss $L_D$ stabilized within $[1.1, 1.6]$, and the real-fake score margin $D(real) - D(fake)$ remained within $[0.4, 0.9]$. No mode collapse or discriminator saturation occurred.

### 3.4 Inference Latency & Parameter Invariance
Because the PatchGAN discriminator is utilized **strictly during training**, both the Baseline CVAE and SPA-VAEGAN share the exact same inference engine (`models/cvae.py`).
- **Parameter Count:** Exactly **1,829,827 parameters** (~1.83M).
- **Latency:** **14.85 ms** per $64 \times 64 \to 256 \times 256$ patch on an NVIDIA T4 GPU (67 FPS), and **112 ms** on a multi-core CPU.
- **Key Takeaway:** SPA-VAEGAN delivers substantial perceptual improvements with **zero increase in deployed model size or inference latency**.
