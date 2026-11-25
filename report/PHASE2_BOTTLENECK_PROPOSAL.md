# Phase 2: Bottleneck Selection & Mathematical Proposal

**Author:** Bevan (@Bevan)  
**Project:** Structure-Preserving Adaptive VAE-GAN (SPA-VAEGAN) for 4× Image Super-Resolution  
**Course:** Advanced Generative AI Continuous Assessment (CAC)

---

## 1. Identified Architecture & Bottleneck

### 1.1 Selected Architecture: Conditional Variational Autoencoder (CVAE)
The **Conditional Variational Autoencoder (CVAE)** provides a principled probabilistic framework for single-image super-resolution (SISR). Given a low-resolution (LR) image $y \in \mathbb{R}^{3 \times 64 \times 64}$ and a ground-truth high-resolution (HR) target $x \in \mathbb{R}^{3 \times 256 \times 256}$, the CVAE models the conditional distribution $p(x \mid y)$ by introducing continuous latent variables $z \in \mathbb{R}^{4 \times 64 \times 64}$:
$$p(x \mid y) = \int p_\theta(x \mid y, z) \, p(z) \, dz$$
where $p(z) = \mathcal{N}(0, I)$ is an isotropic Gaussian prior, and $p_\theta(x \mid y, z)$ is parameterized by a deep convolutional decoder. During training, a posterior encoder $q_\phi(z \mid x, y) = \mathcal{N}\left(\mu_\phi(x, y^\uparrow), \text{diag}(\sigma_\phi^2(x, y^\uparrow))\right)$ infers the latent distribution given both HR and upsampled LR inputs.

### 1.2 The Bottleneck: Reconstruction Blurriness (The Conditional Mean Problem)
Standard VAE training optimizes the Evidence Lower Bound (ELBO):
$$\log p(x \mid y) \ge \mathbb{E}_{q_\phi(z \mid x, y)}\left[\log p_\theta(x \mid y, z)\right] - D_{KL}\left(q_\phi(z \mid x, y) \,\|\, p(z)\right)$$

Under the canonical assumption of an isotropic Gaussian observation model $p_\theta(x \mid y, z) = \mathcal{N}(\hat{x}, \sigma_{obs}^2 I)$, maximizing the expected log-likelihood reduces mathematically to minimizing Mean Squared Error (MSE / $L_2$ loss) or Mean Absolute Error (MAE / $L_1$ loss).

**The Mathematical Dilemma:**
Single-image super-resolution is an ill-posed inverse problem: an infinite ensemble of plausible high-resolution textures could correspond to the same low-resolution observation $y$. Let $\{x^{(1)}, x^{(2)}, \dots, x^{(K)}\}$ denote plausible high-resolution configurations that downsample to $y$. 

When trained with an element-wise pixel loss $\mathcal{L}_{pix} = \frac{1}{N}\|\hat{x} - x\|_1$, the optimal point estimate $\hat{x}^*$ predicted by the generator is the **conditional expectation (or median)** of the posterior distribution:
$$\hat{x}^* = \mathbb{E}_{x \sim p(x \mid y)}[x]$$

Averaging across multiple sharp, distinct spatial patterns (e.g., foliage, fabric weaves, hair strands) yields a blurred, washed-out composite. The resulting image exhibits artificially high PSNR by minimizing per-pixel discrepancy, but scores poorly on human perceptual visual metrics (high LPIPS and FID).

---

## 2. Proposed Solution: The SPA-VAEGAN Formulation

To resolve this limitation without sacrificing the training stability and fast single-pass inference of VAEs, we propose **SPA-VAEGAN (Structure-Preserving Adaptive VAE-GAN)**. SPA-VAEGAN introduces three unified mathematical and structural modifications:
1. **Adversarial Distribution Matching** via a Spectral-Normalized Conditional PatchGAN with Hinge Loss.
2. **Explicit Edge-Aware Structure Preservation** via a Normalized Sobel Gradient Loss.
3. **Dynamic Gradient-Ratio Adaptive Balancing** to ensure optimization stability.

```
                              ┌────────────────────────────────────────┐
                              │           Ground Truth HR (x)          │
                              └───────────────────┬────────────────────┘
                                                  │
                                                  ▼
┌──────────────────┐  degrade()   ┌────────────────────────────────────┐
│      LR (y)      ├─────────────►│        Sobel Edge Filter           │
└────────┬─────────┘              └───────────────┬────────────────────┘
         │                                        │  Sobel Edge Loss
         ▼                                        ▼  (lambda_e = 0.5)
┌──────────────────┐  PixelShuffle ┌───────────────────────────────────┐
│ CVAE Generator G ├─────────────►│ Reconstructed Super-Res (x_hat)   │
└────────┬─────────┘              └───────────────┬───────────────────┘
         │                                        │
         │                                        ▼
         │                        ┌───────────────────────────────────┐
         │                        │  Conditional PatchGAN D           │
         │                        │  (Spectral Norm + Hinge Loss)     │
         │                        └───────────────┬───────────────────┘
         │                                        │
         │      Adaptive Weight Tracker           ▼
         └─────────────────────────────────► lambda_t (Dynamic GAN Scale)
```

---

## 3. Mathematical Derivations

### 3.1 Baseline CVAE Loss with KL Warmup
The baseline generator minimizes:
$$\mathcal{L}_{base} = \mathcal{L}_{pix} + \beta_t \mathcal{L}_{KL}$$
where:
$$\mathcal{L}_{pix} = \frac{1}{3 \times 256 \times 256} \sum_{c=1}^3 \sum_{i=1}^{256} \sum_{j=1}^{256} \left| \hat{x}_{c,i,j} - x_{c,i,j} \right|$$
$$\mathcal{L}_{KL} = -\frac{1}{2 \times 4 \times 64 \times 64} \sum_{k=1}^4 \sum_{i=1}^{64} \sum_{j=1}^{64} \left( 1 + \log \sigma_{k,i,j}^2 - \mu_{k,i,j}^2 - \sigma_{k,i,j}^2 \right)$$

To prevent early posterior collapse (where the encoder maps all latents to the prior mean before the decoder learns high-level features), we implement a linear warmup schedule:
$$\beta_t = \min\left( \beta, \, \beta \cdot \frac{t}{T_{warm}} \right), \quad \beta = 10^{-3}, \, T_{warm} = 2,000$$

### 3.2 Structure-Preserving Sobel Edge Loss ($\mathcal{L}_{edge}$)
To explicitly preserve sharp object contours and suppress geometric warping, we apply horizontal and vertical Sobel convolution kernels normalized by $\frac{1}{8}$ across each RGB channel:
$$K_x = \frac{1}{8} \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad K_y = K_x^T$$

The edge magnitude operator $S(u)$ is defined as:
$$S(u) = \sqrt{ (K_x * u)^2 + (K_y * u)^2 + \epsilon_s }, \quad \epsilon_s = 10^{-6}$$
where $\epsilon_s$ sits inside the radical to guarantee strictly bounded gradients in uniform regions ($\frac{\partial \sqrt{u}}{\partial u} \to \infty$ as $u \to 0$).

The structural edge loss is then computed as:
$$\mathcal{L}_{edge} = \frac{1}{N} \left\| S(\hat{x}) - S(x) \right\|_1$$
We set $\lambda_e = 0.5$ in all experiments.

### 3.3 Conditional PatchGAN Discriminator with Hinge Loss
Instead of scoring the entire image with a scalar real/fake probability, we construct a 7-layer fully convolutional **PatchGAN** discriminator $D(u, y^\uparrow)$ that evaluates local $32 \times 32$ patch distributions. Every convolutional layer is constrained by **Spectral Normalization** ($\sigma(W) = 1$), enforcing a 1-Lipschitz continuity bound that stabilizes training.

The discriminator and generator adversarial objectives follow the geometric Hinge loss:
$$\mathcal{L}_D = \mathbb{E}_{(x, y)}\left[ \max(0, \, 1 - D(x, y^\uparrow)) \right] + \mathbb{E}_{(\hat{x}, y)}\left[ \max(0, \, 1 + D(\hat{x}, y^\uparrow)) \right]$$
$$\mathcal{L}_{adv} = -\,\mathbb{E}_{(\hat{x}, y)}\left[ D(\hat{x}, y^\uparrow) \right]$$

### 3.4 Gradient-Ratio Adaptive Adversarial Weighting ($\lambda_t$)
A chronic pathology in GAN super-resolution (e.g., SRGAN, ESRGAN) is hyperparameter sensitivity: if the adversarial weight is too small, blur persists; if too large, the generator hallucinates unrealistic geometric artifacts and destabilizes.

Inspired by VQGAN balancing principles, we introduce a **Gradient-Ratio Adaptive Adversarial Weight** $\lambda_t$. Let $\mathcal{L}_{rec} = \mathcal{L}_{pix} + \lambda_e \mathcal{L}_{edge}$, and let $\theta_L$ denote the parameters of the generator's final convolutional layer (`G.dec.last.weight`). Both reconstruction and adversarial loss signals must pass through this reference bottleneck.

We define the instantaneous raw balance ratio $\tilde{\lambda}_t$:
$$\tilde{\lambda}_t = \delta \cdot \frac{\left\| \nabla_{\theta_L} \mathcal{L}_{rec} \right\|_2}{\left\| \nabla_{\theta_L} \mathcal{L}_{adv} \right\|_2 + \epsilon_{adv}}$$
where $\delta = 0.1$ is the target gradient ratio, and $\epsilon_{adv} = 10^{-8}$.

To suppress high-frequency mini-batch variance while imposing strict safety envelopes:
$$\lambda_t = \operatorname{clip}\left( 0.9 \, \lambda_{t-1} + 0.1 \, \tilde{\lambda}_t, \, 0.0, \, \lambda_{max} \right), \quad \lambda_{max} = 1000.0$$

**Theoretical Safeguards:**
1. **Scale Invariance under Mixed Precision:** In Automatic Mixed Precision (FP16), loss scaling by factor $s$ multiplies both gradient norms equally: $\frac{\|s \cdot \nabla \mathcal{L}_{rec}\|}{\|s \cdot \nabla \mathcal{L}_{adv}\|} = \frac{\|\nabla \mathcal{L}_{rec}\|}{\|\nabla \mathcal{L}_{adv}\|}$, rendering $\lambda_t$ mathematically invariant to loss scaler adjustments.
2. **Asymmetric Protection:** The upper clip at $\lambda_{max}$ ensures the adversarial pull can never overpower the reconstruction loss, while the denominator $\epsilon_{adv}$ prevents numerical divergence if the discriminator saturates.

### 3.5 Full SPA-VAEGAN Objective
$$\min_G \mathcal{L}_G = \mathcal{L}_{pix} + \lambda_e \mathcal{L}_{edge} + \beta \mathcal{L}_{KL} + \lambda_t \mathcal{L}_{adv}$$
$$\min_D \mathcal{L}_D = \mathbb{E}\left[\max(0, 1 - D(x, y^\uparrow))\right] + \mathbb{E}\left[\max(0, 1 + D(\hat{x}, y^\uparrow))\right]$$

---

## 4. Shared Warm-Start Protocol

To guarantee an unbiased, academically rigorous comparison between the Baseline CVAE and SPA-VAEGAN:
- The Baseline CVAE trains for 30,000 iterations.
- At iteration 10,000, we serialize `baseline_it10000.pt`.
- SPA-VAEGAN initializes its generator from `baseline_it10000.pt` and trains for 20,000 iterations with the adversarial and edge objectives.
- **Result:** Both models receive exactly 30,000 generator updates on identical mini-batches, ruling out "longer training" as a confounding variable.
