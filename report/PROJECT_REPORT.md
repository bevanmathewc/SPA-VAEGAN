# SPA-VAEGAN: Structure-Preserving Adaptive VAE-GAN for 4× Single Image Super-Resolution

**Course:** Advanced Generative AI Continuous Assessment (CAC)  
**Author:** Bevan (@Bevan)  
**Date:** November 2026  
**Repository:** [SPA-VAEGAN GitHub Repository](https://github.com/bevan/SPA-VAEGAN)

---

## Abstract
Single-image super-resolution (SISR) poses an ill-posed inverse problem characterized by significant high-frequency uncertainty. While Conditional Variational Autoencoders (CVAEs) offer rapid single-pass inference and mathematically principled optimization, models trained with element-wise likelihoods (L1/MSE) collapse to predicting the conditional spatial mean, producing characteristically blurred textures. To resolve this limitation, we present **SPA-VAEGAN (Structure-Preserving Adaptive VAE-GAN)**, a unified generative architecture for 4× super-resolution ($64\times 64 \to 256\times 256$). SPA-VAEGAN augments a 16-block residual CVAE generator with three stabilizing mechanisms: (1) a Spectral-Normalized Conditional PatchGAN discriminator evaluating $32\times 32$ patch logits under Hinge loss, (2) a normalized Sobel edge loss ($\lambda_e = 0.5$) enforcing structural gradient consistency, and (3) a gradient-ratio adaptive adversarial weighting scheme ($\lambda_t$) that dynamically rescales the adversarial gradient to a target ratio ($\delta = 0.1$) at the generator's terminal layer. Evaluated across 2,000 held-out patches from the DIV2K dataset under an identical shared warm-start protocol (30k total generator updates), SPA-VAEGAN achieves a **29.8% reduction in LPIPS** ($0.2315 \to 0.1624$) and a **35.4% reduction in FID** ($51.34 \to 33.18$) compared to the baseline CVAE, while preserving an identical parameter footprint (1.83M) and real-time inference latency (14.8 ms on GPU, 112 ms on CPU). The architecture is deployed as an interactive Streamlit application featuring overlap-and-crop tiling for arbitrary image resolutions.

---

## 1. Introduction

### 1.1 Problem & Motivation
High-resolution visual detail is fundamental across domains ranging from medical diagnostic imaging and satellite remote sensing to consumer photograph enhancement. Single-image super-resolution (SISR) aims to reconstruct a high-resolution (HR) estimate $x \in \mathbb{R}^{3 \times 4H \times 4W}$ from a degraded low-resolution (LR) observation $y \in \mathbb{R}^{3 \times H \times W}$. Because the forward degradation operator involves low-pass spatial filtering and downsampling, high-frequency spatial components above the Nyquist limit are irreversibly destroyed. Consequently, SISR is fundamentally ill-posed: an infinite family of high-resolution configurations could have produced the identical low-resolution image.

### 1.2 Objectives & Contribution Statement
The objectives of this project are:
1. **Systematic Comparative Study:** Critically analyze four generative paradigms (Transformers, GANs, VAEs, and Diffusion models) across training stability, output quality, generation speed, and latent interpretability.
2. **Bottleneck Resolution:** Isolate the fundamental flaw of Variational Autoencoders—spatial blurriness caused by conditional mean prediction under pixel-wise likelihoods—and develop a mathematically sound structural remedy.
3. **Equitable Benchmarking:** Implement both a Baseline CVAE and the proposed SPA-VAEGAN under a strictly controlled shared warm-start protocol on the benchmark DIV2K dataset, evaluating seven quantitative metrics alongside paired bootstrap confidence intervals.
4. **Real-World Deployment:** Package the trained generator into a robust, interactive Streamlit web application capable of processing arbitrary image dimensions without tile border artifacts.

---

## 2. Comparative Literature Study

The generative AI landscape is primarily structured around four mathematical frameworks:

| Architecture | Core Principle | Primary Strengths | Fundamental Weaknesses |
| :--- | :--- | :--- | :--- |
| **Transformer** | Autoregressive / masked self-attention over tokenized spatial patches. | Captures long-range spatial context; scales gracefully with massive compute and dataset scale. | $\mathcal{O}(L^2)$ computational complexity; slow autoregressive inference; heavy memory footprint. |
| **GAN** | Two-player minimax game between generator and adversarial discriminator. | Produces exceptionally sharp, photo-realistic textures; single-step forward pass $\mathcal{O}(1)$. | Optimization instability; mode collapse; sensitive hyperparameters; prone to hallucinated artifacts. |
| **VAE (Selected)** | Variational inference maximizing Evidence Lower Bound (ELBO) with KL regularization. | Highly stable optimization; principled probabilistic formulation; smooth, interpretable latent space. | Severe reconstruction blurriness from per-pixel likelihoods; susceptibility to posterior collapse. |
| **Diffusion** | Iterative reversal of a forward Gaussian noising Markov chain. | Outstanding sample fidelity and structural diversity; robust optimization without adversarial collapse. | Prohibitive sampling latency (requires 20–1000 iterative forward passes); computationally intensive. |

While diffusion models currently lead in unconstrained image synthesis benchmarks, their multi-step sampling latency (typically 1.5 to 10 seconds per image) renders them impractical for low-latency interactive consumer tools without high-end dedicated GPU clusters. Pure GANs, conversely, exhibit notorious training volatility and are prone to inventing false structures when trained on constrained academic budgets. Variational Autoencoders provide the ideal engineering sweet spot: they feature stable scalar optimization and sub-20 ms inference latency, provided their primary limitation—blurriness—can be overcome.

---

## 3. Limitation Identified: The VAE Blurriness Bottleneck

### 3.1 Mathematical Analysis of the Conditional Mean
In a Conditional VAE, the model optimizes the Evidence Lower Bound:
$$\log p(x \mid y) \ge \mathbb{E}_{q_\phi(z \mid x, y)}\left[\log p_\theta(x \mid y, z)\right] - D_{KL}\left(q_\phi(z \mid x, y) \,\|\, p(z)\right)$$

When modeling the observation likelihood as a factorized Laplacian or Gaussian distribution:
$$p_\theta(x \mid y, z) \propto \exp\left(-\frac{\|\hat{x} - x\|_1}{\sigma}\right)$$
maximizing log-likelihood is equivalent to minimizing the expected $L_1$ pixel loss $\frac{1}{N}\|\hat{x} - x\|_1$.

In ill-posed super-resolution, multiple plausible high-frequency textures $\{x_1, x_2, \dots, x_k\}$ correspond to the identical low-resolution patch $y$. The global minimizer of expected $L_1$ distortion is the spatial median of the plausible distribution:
$$\hat{x}^* = \arg\min_{\hat{x}} \int p(x \mid y) \|\hat{x} - x\|_1 \, dx$$

Because the spatial median of high-frequency patterns (e.g., foliage, brick mortar, cloth grain) is a flat, uniform average, the generator systematically eliminates high-frequency gradients. The output achieves high PSNR by avoiding large errors, but appears unnaturally smooth to human observers.

---

## 4. Proposed Method: SPA-VAEGAN

SPA-VAEGAN resolves this bottleneck by integrating three synergistic components:

### 4.1 Conditional Residual CVAE Architecture
The generator $G$ consists of an encoder $q(z \mid x, y)$ (used during training only) and a fully convolutional decoder $p(x \mid z, y)$.
- **Encoder:** Convolves the concatenated HR target and bicubic-upsampled LR conditioning ($6 \times 256 \times 256$), downsampling by factor 4 to produce latent posterior parameters $\mu, \log \sigma^2 \in \mathbb{R}^{4 \times 64 \times 64}$.
- **Decoder:** Takes concatenated LR ($3 \times 64 \times 64$) and latent $z$ ($4 \times 64 \times 64$). Features pass through 16 residual blocks with residual scaling ($\text{scale} = 0.1$) and a global skip connection, followed by two PixelShuffle upsampling stages ($2\times$ twice $= 4\times$).
- **Bicubic Residual Formulation:** The final convolution predicts a residual added directly to the bicubic upsampled LR image:
  $$\hat{x} = G_{trunk}(y, z) + \operatorname{Bicubic}_{\uparrow 4}(y)$$
  This protects low-frequency color fidelity and accelerates convergence. At inference time, $z$ is set deterministically to the prior mean ($z = 0$).

### 4.2 Normalized Sobel Edge Loss ($\mathcal{L}_{edge}$)
To explicitly penalize blurry edges without inducing geometric distortion, we compute normalized Sobel gradients across RGB channels:
$$K_x = \frac{1}{8}\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad K_y = K_x^T$$
$$S(u) = \sqrt{(K_x * u)^2 + (K_y * u)^2 + 10^{-6}}, \quad \mathcal{L}_{edge} = \frac{1}{N}\|S(\hat{x}) - S(x)\|_1$$
The edge loss is weighted by $\lambda_e = 0.5$.

### 4.3 Spectral-Normalized Conditional PatchGAN Discriminator
A 7-layer convolutional discriminator $D$ evaluates the concatenation of candidate image and bicubically upsampled LR conditioning ($6 \times 256 \times 256$), outputting a $1 \times 32 \times 32$ grid of patch logits. Every layer incorporates **Spectral Normalization**, bounding the Lipschitz constant and stabilizing the Hinge adversarial loss:
$$\mathcal{L}_D = \mathbb{E}\left[\max(0, 1 - D(x, y^\uparrow))\right] + \mathbb{E}\left[\max(0, 1 + D(\hat{x}, y^\uparrow))\right]$$
$$\mathcal{L}_{adv} = -\mathbb{E}\left[D(\hat{x}, y^\uparrow)\right]$$

### 4.4 Gradient-Ratio Adaptive Adversarial Weighting ($\lambda_t$)
To prevent the discriminator from overpowering the reconstruction loss or causing mode collapse, we evaluate gradient norms with respect to the generator's terminal layer weights $\theta_L$ (`G.dec.last.weight`):
$$\tilde{\lambda}_t = \delta \cdot \frac{\|\nabla_{\theta_L} \mathcal{L}_{rec}\|_2}{\|\nabla_{\theta_L} \mathcal{L}_{adv}\|_2 + 10^{-8}}, \quad \mathcal{L}_{rec} = \mathcal{L}_{pix} + \lambda_e \mathcal{L}_{edge}$$
$$\lambda_t = \operatorname{clip}\left(0.9 \lambda_{t-1} + 0.1 \tilde{\lambda}_t, \, 0.0, \, 1000.0\right)$$
with target ratio $\delta = 0.1$. This ensures the adversarial loss contributes a steady ~10% fraction of the total parameter update magnitude regardless of raw loss drift.

---

## 5. Experimental Setup

### 5.1 Dataset & Partition
Experiments utilize all 900 official high-resolution images from the **DIV2K** dataset:
- **Training Set (720 images):** 21,600 patches of $288 \times 288$ stored in an memory-mapped uint8 array. Data augmentation comprises random $256 \times 256$ sub-crops, random horizontal/vertical flips, and 90° rotations. Overly flat sky patches (std < 10) are filtered to prioritize structural content.
- **Validation Set (80 images):** 400 fixed $256 \times 256$ patches used for model monitoring.
- **Held-Out Test Set (100 images):** 2,000 fixed $256 \times 256$ patches extracted from the 100 official validation images, reserved strictly for the final benchmark.

### 5.2 Synthetic Degradation Pipeline
$$\mathcal{D}(x) = \operatorname{Round}_{8\text{-bit}}\left( \operatorname{Bicubic}_{\downarrow 4}\left( x * K_{\sigma=1.0, 7\times 7} \right) \right)$$
Deterministic and identical across all splits and models.

### 5.3 Shared Warm-Start Optimization Protocol
- **Baseline CVAE:** Trained for 30,000 iterations ($B=16$, Adam optimizer, $\text{lr}=10^{-4}$, $\beta_1=0.9, \beta_2=0.999$, linear KL warmup from 0 to $10^{-3}$ over first 2,000 steps). Milestone `baseline_it10000.pt` saved at step 10,000.
- **SPA-VAEGAN:** Initialized from `baseline_it10000.pt` and trained for 20,000 iterations with the adversarial and edge objectives ($B=16$, Adam for $G$ with $\text{lr}=10^{-4}$, Adam for $D$ with $\text{lr}=10^{-4}, \beta_1=0.5$).
- **Parity:** Both models finish with exactly 30,000 generator updates on identical mini-batches.

---

## 6. Results

### 6.1 Benchmark Table (2,000 Held-Out DIV2K Test Patches)

| Method | MSE $\downarrow$ | MAE $\downarrow$ | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ | LPIPS $\downarrow$ | FID (2,000 patches) $\downarrow$ | Latency (ms) $\downarrow$ | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bicubic (Reference)** | 0.00284 | 0.0381 | 25.46 $\pm$ 2.81 | 0.7312 $\pm$ 0.068 | 0.3842 $\pm$ 0.052 | 84.62 | 0.48 $\pm$ 0.04 | 0 |
| **Baseline CVAE** | **0.00171** | **0.0278** | **28.14 $\pm$ 2.95** | **0.8045 $\pm$ 0.057** | 0.2315 $\pm$ 0.046 | 51.34 | 14.82 $\pm$ 0.65 | 1.83M |
| **SPA-VAEGAN (Proposed)** | 0.00208 | 0.0305 | 27.28 $\pm$ 3.02 | 0.7891 $\pm$ 0.061 | **0.1624 $\pm$ 0.039** | **33.18** | 14.85 $\pm$ 0.68 | 1.83M |
| **Difference (SPA $-$ Baseline)**<br>*(95% Paired Bootstrap CI)* | *+0.00037*<br>$[+0.00029, +0.00045]$ | *+0.0027*<br>$[+0.0021, +0.0033]$ | *-0.86 dB*<br>$[-0.98, -0.74]$ | *-0.0154*<br>$[-0.0182, -0.0126]$ | **-0.0691**<br>$[-0.0735, -0.0647]$ | **-18.16**<br>*(Set-level reduction)* | *+0.03 ms*<br>$[-0.05, +0.11]$ | **0.0M** |

### 6.2 Statistical Significance Analysis
The 95% paired bootstrap confidence interval for LPIPS difference is $[-0.0735, -0.0647]$, which strictly excludes zero. This establishes with high statistical confidence that the perceptual sharpness gain of SPA-VAEGAN is genuine and reproducible across the image distribution.

---

## 7. Discussion

### 7.1 Perception–Distortion Trade-off
As formalized by Blau & Michaeli (2018), perceptual quality (measured by LPIPS and FID) and distortion (measured by MSE and PSNR) are inherently at trade-off. The Baseline CVAE achieves superior PSNR because it minimizes pixel distortion by predicting conditional averages. However, human observers perceive smooth textures as degraded. SPA-VAEGAN sacrifices a modest ~0.86 dB in PSNR to achieve dramatic gains in perceptual photorealism, generating convincing high-frequency textures that align with natural scene statistics.

### 7.2 Attribution Caveat
In accordance with our experimental plan, SPA-VAEGAN was evaluated as a cohesive package (PatchGAN + Sobel edge loss + adaptive weighting) against the baseline. While individual component ablations were not conducted due to compute constraints, tracking logs demonstrate that $\lambda_t$ remained stable throughout training and the Sobel loss consistently prevented geometric edge warping.

---

## 8. Deployment: Streamlit Application

The proposed model is deployed as **ImageEnhance AI**, an interactive Streamlit application:
- **Tiled Inference:** Arbitrary input resolutions are decomposed into overlapping $64 \times 64$ core tiles with a 24-pixel halo (112 px input tiles). Tile border seams are completely eliminated.
- **Robustness:** Handles RGBA alpha flattening, EXIF smartphone orientations, and dimension limits (512 px on CPU, 1024 px on GPU).
- **Inference Speed:** Runs in ~15 ms on GPU and ~112 ms on CPU, enabling instant user feedback and PNG export.

---

## 9. Limitations & Future Work

1. **Synthetic Degradation Assumption:** The model was trained on fixed bicubic downsampling with Gaussian blur. Real-world smartphone photographs exhibit non-uniform noise and motion blur ("blind super-resolution").
2. **FID Sample Size:** FID was evaluated on 2,000 patches rather than 50,000 images, introducing an upward bias; values are meaningful solely for internal relative comparison.
3. **Future Directions:** Incorporating Real-ESRGAN style high-order degradation modeling, exploring multi-seed ensemble evaluations, and investigating hybrid diffusion refinement on top of the CVAE latent space.

---

## 10. Conclusion

SPA-VAEGAN successfully resolves the persistent blurriness bottleneck of Variational Autoencoders in single-image super-resolution. By combining conditional residual learning with a Spectral-Normalized PatchGAN discriminator, normalized Sobel edge preservation, and gradient-ratio adaptive adversarial weighting, SPA-VAEGAN achieves a **29.8% reduction in LPIPS** and a **35.4% reduction in FID** over a well-tuned baseline CVAE on held-out DIV2K test data, without increasing generator model complexity or inference latency.

---

## References
1. Blau, Y., & Michaeli, T. (2018). The perception-distortion tradeoff. *CVPR*, 6228–6237.
2. Esser, P., Rombach, R., & Ommer, B. (2021). Taming transformers for high-resolution image synthesis. *CVPR*, 12873–12883.
3. Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. *CVPR*, 1125–1134.
4. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. *ICLR*.
5. Ledig, C., et al. (2017). Photo-realistic single image super-resolution using a generative adversarial network. *CVPR*, 4681–4690.
6. Miyato, T., et al. (2018). Spectral normalization for generative adversarial networks. *ICLR*.
7. Zhang, R., et al. (2018). The unreasonable effectiveness of deep features as a perceptual metric. *CVPR*, 586–595.
