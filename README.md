# SPA-VAEGAN: Structure-Preserving Adaptive VAE-GAN for 4× Single Image Super-Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep generative framework designed to overcome the classical **blurriness limitation of Variational Autoencoders (VAEs)** in $4\times$ Single Image Super-Resolution ($64\times 64 \to 256\times 256$). 

Built for the **Advanced Generative AI Continuous Assessment (CAC)**, SPA-VAEGAN combines a 16-block residual Conditional VAE generator with a Spectral-Normalized PatchGAN discriminator, normalized Sobel edge preservation, and dynamic gradient-ratio balancing.

---

## Key Highlights & Findings
- **Resolves the VAE Blurriness Bottleneck:** Replaces the conditional mean prediction of standard $L_1$/MSE likelihoods with adversarial distribution matching and structural gradient preservation.
- **Unbiased Shared Warm-Start Protocol:** Both the Baseline CVAE and SPA-VAEGAN utilize the identical 1.83M generator trunk and end with exactly **30,000 generator updates** on identical mini-batches from the **DIV2K** dataset.
- **29.8% Perceptual Gain (LPIPS):** LPIPS distance decreases from $0.2315 \to 0.1624$ on 2,000 held-out test patches (95% bootstrap confidence interval $[-0.0735, -0.0647]$ strictly excludes zero).
- **35.4% Distribution Fidelity Gain (FID):** Set-level FID drops from $51.34 \to 33.18$.
- **Zero Latency / Parameter Penalty:** Because the discriminator is strictly active during training, the deployed generator retains an identical **14.8 ms GPU latency** and **1.83M parameter footprint**.
- **Interactive Streamlit Deployment:** Features overlap-and-crop tiled inference (Core: 64px, Halo: 24px) for seamless, border-free super-resolution on arbitrary image dimensions.

---

## Mathematical Formulation

### 1. Generator Objective ($\mathcal{L}_G$)
$$\mathcal{L}_G = \mathcal{L}_{pix} + \lambda_e \mathcal{L}_{edge} + \beta \mathcal{L}_{KL} + \lambda_t \mathcal{L}_{adv}$$

- **Pixel Reconstruction Loss:** $\mathcal{L}_{pix} = \frac{1}{N} \|\hat{x} - x\|_1$
- **Normalized Sobel Edge Loss ($\lambda_e = 0.5$):**
  $$S(u) = \sqrt{(K_x * u)^2 + (K_y * u)^2 + 10^{-6}}, \quad \mathcal{L}_{edge} = \frac{1}{N} \|S(\hat{x}) - S(x)\|_1$$
- **KL Divergence:** Analytical KL to prior $\mathcal{N}(0, I)$ with linear warm-up over the first 2,000 steps.
- **Gradient-Ratio Adaptive Adversarial Weight ($\lambda_t$):**
  $$\tilde{\lambda}_t = \delta \cdot \frac{\|\nabla_{\theta_L} \mathcal{L}_{rec}\|_2}{\|\nabla_{\theta_L} \mathcal{L}_{adv}\|_2 + 10^{-8}}, \quad \lambda_t = \operatorname{clip}\left(0.9\lambda_{t-1} + 0.1\tilde{\lambda}_t, \, 0.0, \, 1000.0\right)$$
  Evaluated at the generator's terminal convolution layer $\theta_L$ (`G.dec.last.weight`) with target ratio $\delta = 0.1$.

### 2. Discriminator Objective ($\mathcal{L}_D$)
7-layer Conditional PatchGAN ($6 \times 256 \times 256 \to 1 \times 32 \times 32$) with Spectral Normalization under Hinge loss:
$$\mathcal{L}_D = \mathbb{E}\left[\max(0, 1 - D(x, y^\uparrow))\right] + \mathbb{E}\left[\max(0, 1 + D(\hat{x}, y^\uparrow))\right]$$

---

## Benchmark Results (2,000 Held-Out DIV2K Test Patches)

| Method | MSE $\downarrow$ | MAE $\downarrow$ | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ | LPIPS $\downarrow$ | FID (2,000 patches) $\downarrow$ | Latency (ms) $\downarrow$ | Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bicubic (Reference)** | 0.00284 | 0.0381 | 25.46 $\pm$ 2.81 | 0.7312 $\pm$ 0.068 | 0.3842 $\pm$ 0.052 | 84.62 | 0.48 $\pm$ 0.04 | 0 (Non-learned) |
| **Baseline CVAE** | **0.00171** | **0.0278** | **28.14 $\pm$ 2.95** | **0.8045 $\pm$ 0.057** | 0.2315 $\pm$ 0.046 | 51.34 | 14.82 $\pm$ 0.65 | 1.83M |
| **SPA-VAEGAN (Proposed)** | 0.00208 | 0.0305 | 27.28 $\pm$ 3.02 | 0.7891 $\pm$ 0.061 | **0.1624 $\pm$ 0.039** | **33.18** | 14.85 $\pm$ 0.68 | 1.83M |
| **Difference (SPA $-$ Baseline)**<br>*(95% Paired Bootstrap CI)* | *+0.00037*<br>$[+0.00029, +0.00045]$ | *+0.0027*<br>$[+0.0021, +0.0033]$ | *-0.86 dB*<br>$[-0.98, -0.74]$ | *-0.0154*<br>$[-0.0182, -0.0126]$ | **-0.0691**<br>$[-0.0735, -0.0647]$ | **-18.16**<br>*(Set-level reduction)* | *+0.03 ms*<br>$[-0.05, +0.11]$ | **0.0M**<br>*(Identical Generator)* |

> **Perception–Distortion Trade-off:** The baseline model scores slightly higher on PSNR/SSIM because it predicts the conditional spatial median. SPA-VAEGAN achieves a dramatic reduction in perceptual distance (LPIPS) and distribution divergence (FID), restoring realistic textures without structural hallucinations.

---

## Repository Directory Structure

```
SPA-VAEGAN/
├── README.md                      # Primary project documentation & benchmark summary
├── requirements.txt              # Complete Python dependencies for training & evaluation
├── requirements-app.txt          # Lightweight dependencies for CPU-only Streamlit deployment
├── .gitignore                    # Git exclusions (large checkpoints, patch banks, datasets)
├── configs/
│   └── default.yaml              # Single source of truth for all paths & hyperparameters
├── notebooks/                    # Interactive Colab Jupyter Notebooks with rich markdown interpretations
│   ├── 01_colab_setup_and_data.ipynb     # Day 1: Hardware check, DIV2K download, patch banks, overfit test
│   ├── 02_train_baseline_cvae.ipynb      # Day 1: 30k baseline CVAE training & 10k milestone checkpoint
│   ├── 03_train_spa_vaegan.ipynb         # Day 2: 20k SPA-VAEGAN adversarial & edge training
│   └── 04_evaluate_and_figures.ipynb     # Day 3 & 4: 7-metric benchmarking, bootstrap CIs, figures, export
├── models/
│   ├── __init__.py
│   ├── blocks.py                 # ResBlock with 0.1 residual scaling
│   ├── cvae.py                   # Encoder, Decoder, and unified CVAE generator (~1.83M params)
│   └── discriminator.py          # Conditional PatchGAN with Spectral Normalization (32x32 logits)
├── src/
│   ├── __init__.py
│   ├── config.py                 # YAML configuration parser
│   ├── degradation.py            # Fixed synthetic degradation (Gaussian blur + 4x bicubic + 8-bit quant)
│   ├── make_patches.py           # Patch extraction to uint8 memory-mapped .npy files
│   ├── dataset.py                # PatchDataset with random sub-crop, flips, and rotations
│   ├── losses.py                 # L1, analytical KL, and normalized Sobel edge filter/loss
│   ├── adaptive_weight.py        # Gradient-ratio adaptive adversarial weighting tracker
│   ├── checkpointing.py          # Atomic saving, resumption logic, and checkpoint selection rule
│   ├── logging_utils.py          # CSV metric logger and TensorBoard summary writer
│   ├── train_baseline.py         # Baseline CVAE training pipeline
│   ├── train_spa_vaegan.py       # SPA-VAEGAN training pipeline (warm-started from 10k milestone)
│   └── export_model.py           # Strips checkpoints to compact ~7 MB generator weights
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                # Per-image metrics (MSE, MAE, PSNR, SSIM, LPIPS), FID, bootstrap CI
│   ├── latency.py                # CUDA-synchronized latency benchmarking & parameter counter
│   ├── plot_training.py          # Loss progression, discriminator margin, and adaptive weight curves
│   ├── make_figures.py           # Visual comparison grids with zoomed crop insets and edge maps
│   └── evaluate.py               # Comprehensive 2,000-patch test runner
├── app/
│   ├── app.py                    # Streamlit web UI ("ImageEnhance AI")
│   ├── inference.py              # Overlap-and-crop tiled inference engine (Core: 64px, Halo: 24px)
│   ├── image_utils.py            # Sanitization (EXIF orientation, alpha channel flattening, size bounds)
│   └── assets/
│       └── sample.png            # Demo test pattern
├── dataset/
│   ├── README.md                 # Dataset acquisition guide and partition details
│   └── splits/
│       ├── train.txt             # 720 training image names
│       ├── val.txt               # 80 validation image names
│       └── test.txt              # 100 held-out test image names
├── report/                       # Complete academic documentation for all CAC phases
│   ├── PHASE1_COMPARATIVE_STUDY.md   # Architectural comparison (Transformers, GANs, VAEs, Diffusion)
│   ├── PHASE2_BOTTLENECK_PROPOSAL.md # VAE blurriness analysis, mathematical derivations
│   ├── PHASE3_BENCHMARKING_REPORT.md # 7 quantitative metrics, bootstrap analysis, ablation notes
│   ├── PHASE4_DEPLOYMENT_GUIDE.md    # Streamlit architecture, tiled inference, error boundaries
│   ├── PROJECT_REPORT.md             # Master 10-chapter consolidated academic paper
│   └── PRESENTATION_SLIDES_OUTLINE.md# 12-slide presentation structure & 3-minute oral script
└── tests/
    ├── __init__.py
    └── test_shapes.py            # Automated unit tests (shapes, degradation, D/G gradient isolation)
```

---

## Execution Guide

### 1. Google Colab Workflow (GPU Recommended)
Upload the notebooks located in `notebooks/` to Google Colab and run them sequentially:
1. `notebooks/01_colab_setup_and_data.ipynb`: Mounts Drive, installs packages, downloads DIV2K, generates patch banks, and runs the overfit test.
2. `notebooks/02_train_baseline_cvae.ipynb`: Trains Baseline CVAE for 30,000 iterations and saves `baseline_it10000.pt`.
3. `notebooks/03_train_spa_vaegan.ipynb`: Warm-starts from `baseline_it10000.pt` and trains SPA-VAEGAN for 20,000 adversarial steps.
4. `notebooks/04_evaluate_and_figures.ipynb`: Executes the 2,000-patch benchmark, computes bootstrap CIs, plots figures, and exports stripped weights.

### 2. Local Training & CLI Commands
```bash
# Clone the repository
git clone https://github.com/your-username/SPA-VAEGAN.git
cd SPA-VAEGAN

# Create virtual environment and install requirements
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run automated unit tests
python tests/test_shapes.py

# Extract DIV2K patch banks
python -m src.make_patches --config configs/default.yaml

# Train Baseline CVAE
python -m src.train_baseline --config configs/default.yaml

# Train SPA-VAEGAN (warm-started from 10k baseline milestone)
python -m src.train_spa_vaegan --config configs/default.yaml

# Benchmark on held-out test set
python -m evaluation.evaluate --config configs/default.yaml

# Profile latency and parameter counts
python -m evaluation.latency --config configs/default.yaml
```

### 3. Launching the Interactive Web Application
```bash
# Run locally using Streamlit
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser. Upload any image (JPG/PNG), select the model, and enjoy real-time 4× super-resolution.

---

## Academic Citation
```bibtex
@misc{bevan2026spavaegan,
  title={SPA-VAEGAN: Structure-Preserving Adaptive VAE-GAN for 4× Single Image Super-Resolution},
  author={Bevan},
  year={2026},
  howpublished={\url{https://github.com/bevan/SPA-VAEGAN}}
}
```
