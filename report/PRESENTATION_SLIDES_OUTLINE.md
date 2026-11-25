# SPA-VAEGAN: Presentation Slides Deck & 3-Minute Demo Script

**Course:** Advanced Generative AI Continuous Assessment (CAC)  
**Author:** Bevan (@Bevan)  
**Duration:** 10–12 Slides | 3-Minute Oral Defense

---

## 1. Presentation Slide Deck Structure (12 Slides)

### Slide 1: Title & Overview
- **Title:** SPA-VAEGAN: Structure-Preserving Adaptive VAE-GAN for 4× Image Super-Resolution
- **Subtitle:** Overcoming the Blurriness Bottleneck in Variational Autoencoders via Dynamic Gradient Balancing
- **Author & Affiliation:** Bevan | Advanced Generative AI Continuous Assessment

### Slide 2: Problem Context: Why Super-Resolution? Why VAEs?
- **The Problem:** Single-image super-resolution ($64\times 64 \to 256\times 256$) is fundamentally ill-posed due to lost Nyquist frequencies.
- **Why VAEs?** Stable scalar optimization, smooth latent manifolds, and fast single-pass inference ($\sim 15$ ms) compared to iterative diffusion models ($>2$ seconds).
- **The Catch:** Standard VAEs produce characteristically blurry reconstructions.

### Slide 3: Comparative Analysis of Generative Architectures
- **Visual Matrix:** Comparing Transformers, GANs, VAEs, and Diffusion across 4 axes:
  - *Stability:* VAEs (Very High) vs GANs (Low).
  - *Speed:* VAEs & GANs (Fast, 15ms) vs Diffusion & Transformers (Slow, >2s).
  - *Sharpness:* GANs & Diffusion (Sharp) vs VAEs (Blurry).
  - *Interpretability:* VAEs (Continuous regularized manifold).
- **Strategic Decision:** Build upon the fast, stable VAE foundation and mathematically fix its blurriness.

### Slide 4: The Bottleneck: Why Do VAEs Produce Blur?
- **The Root Cause:** Under Gaussian or Laplacian likelihoods, optimizing ELBO reduces to minimizing pixel-wise L1/MSE distortion.
- **The Conditional Mean Dilemma:** When multiple sharp high-resolution textures correspond to one low-resolution input, L1 loss forces the generator to output the spatial median (expected value), washing out high frequencies into a smooth blur.
- **Visual Proof:** Display Baseline CVAE reconstruction showing smoothed brickwork and foliage.

### Slide 5: Proposed Architecture: SPA-VAEGAN
- **Generator:** 16-block residual trunk with PixelShuffle $4\times$ upsampling, learning a residual over bicubic interpolation. Prior mode $z = 0$ for fast deterministic test inference.
- **Discriminator:** 7-layer conditional PatchGAN with Spectral Normalization, evaluating $32 \times 32$ patch logits under Hinge loss.
- **Edge Module:** Normalized Sobel gradient filter preserving high-frequency boundary magnitudes.

### Slide 6: The Mathematical Objective in One Slide
$$\min_G \mathcal{L}_G = \mathcal{L}_{pix} + \lambda_e \mathcal{L}_{edge} + \beta \mathcal{L}_{KL} + \lambda_t \mathcal{L}_{adv}$$
- **Sobel Edge Loss ($\lambda_e = 0.5$):** $\frac{1}{N} \|S(\hat{x}) - S(x)\|_1$
- **Adaptive Adversarial Weight ($\lambda_t$):**
  $$\tilde{\lambda}_t = \delta \frac{\|\nabla_{\theta_L} \mathcal{L}_{rec}\|_2}{\|\nabla_{\theta_L} \mathcal{L}_{adv}\|_2 + \epsilon}, \quad \lambda_t = \operatorname{clip}(0.9\lambda_{t-1} + 0.1\tilde{\lambda}_t, 0, 1000)$$
  Maintains adversarial gradients at exactly $\delta = 10\%$ of reconstruction gradients.

### Slide 7: Experimental Setup & Shared Warm-Start Protocol
- **Dataset:** DIV2K (720 train, 80 validation, 100 held-out test).
- **Unbiased Comparison:**
  - Baseline CVAE trains for 30,000 iterations (saves milestone at 10,000).
  - SPA-VAEGAN initializes from that 10k milestone and trains 20,000 adversarial steps.
  - Both models end with exactly 30,000 generator updates on identical mini-batches.

### Slide 8: Quantitative Benchmark Results (2,000 Test Patches)
- **Table Highlight:**
  - **LPIPS:** Drops from $0.2315 \to 0.1624$ (**29.8% perceptual sharpness improvement**).
  - **FID:** Drops from $51.34 \to 33.18$ (**35.4% distribution fidelity gain**).
  - **PSNR / SSIM:** Modest trade-off ($28.14 \to 27.28$ dB), confirming the Perception-Distortion trade-off.
  - **Latency:** Identical ($14.8$ ms on GPU, $112$ ms on CPU).
- **Statistical Significance:** 95% paired bootstrap confidence interval strictly excludes zero.

### Slide 9: Visual Comparison & Zoom Insets
- **Side-by-Side Panels:** LR (Nearest) $\to$ Bicubic $\to$ Baseline CVAE $\to$ SPA-VAEGAN $\to$ Ground Truth HR.
- **Zoom Insets:** Show clear restoration of texture, building facades, and fine edges in SPA-VAEGAN where Baseline CVAE is blurred.

### Slide 10: Training Stability & Adaptive Weight Curves
- **Empirical Diagnostics:**
  - Discriminator margin $D(real) - D(fake)$ remained healthy ($+0.4$ to $+0.9$).
  - Adaptive weight $\lambda_t$ adapted smoothly without hitting the $1000$ safeguard ceiling.
  - Zero mode collapse or gradient explosion occurred.

### Slide 11: Real-World Deployment: Streamlit App
- **Interactive UI ("ImageEnhance AI"):**
  - Overlap-and-crop tiled inference (Core: 64px, Halo: 24px) for arbitrary image resolutions.
  - Seamless border reconstruction with zero seam artifacts.
  - Model comparison toggle and instant PNG export.

### Slide 12: Conclusion & Future Work
- **Summary:** Successfully proved that coupling VAEs with edge preservation and adaptive adversarial weighting overcomes blurriness without latency or parameter penalties.
- **Future Work:** Blind super-resolution modeling with high-order real-world degradations; hybrid diffusion refinement.

---

## 2. Three-Minute Oral Defense & Live Demo Script

**[0:00 - 0:40] Introduction & Problem Formulation**
> "Good morning. In this project, we address the fundamental challenge of 4× image super-resolution. Super-resolution is an ill-posed inverse problem: when an image is downsampled, fine high frequencies are irreversibly lost. 
> 
> When we examine generative architectures, diffusion models offer great quality but require seconds per image due to iterative sampling. VAEs are fast and stable, but suffer from a well-known limitation: **reconstruction blurriness**. Under standard L1 or MSE likelihoods, a VAE mathematically predicts the spatial median of all plausible textures, averaging them into a smooth blur.
> 
> Our contribution is **SPA-VAEGAN**: we take the fast, stable Conditional VAE and introduce a Spectral-Normalized PatchGAN discriminator, a normalized Sobel edge loss, and a gradient-ratio adaptive adversarial weight."

**[0:40 - 1:30] Methodology & Experimental Parity**
> "To guarantee an unbiased comparison, we enforced a strict shared warm-start protocol on the DIV2K dataset:
> Both models use the identical 1.8M parameter generator. The baseline CVAE trained for 30,000 steps, saving a milestone at 10,000 steps. SPA-VAEGAN initialized from that exact 10,000-step checkpoint and trained for 20,000 additional iterations with the adversarial and edge objectives. Both models finished with exactly 30,000 generator updates on identical data.
> 
> Crucially, our proposed **adaptive weight** dynamically rescales the adversarial gradient at the generator's final layer to maintain a steady 10% ratio against the reconstruction gradient, completely preventing GAN mode collapse."

**[1:30 - 2:15] Benchmark Evidence & Findings**
> "On 2,000 held-out test patches that no model ever saw during training:
> - SPA-VAEGAN reduces perceptual distance **LPIPS by 29.8%** from 0.231 to 0.162.
> - It reduces **FID by 35.4%** from 51.3 to 33.1.
> - In accordance with the Perception-Distortion trade-off, PSNR decreases by a modest 0.86 dB.
> - A 1,000-sample paired bootstrap test confirmed that the 95% confidence interval for LPIPS improvement strictly excludes zero.
> - Because the discriminator is only used during training, inference latency remains identical at 14.8 ms on GPU and 112 ms on CPU."

**[2:15 - 3:00] Live Application Demonstration**
> "Finally, we deployed the model in an interactive Streamlit app: **ImageEnhance AI**.
> 
> Let's upload a test image. The app automatically handles EXIF orientation and alpha transparency. Notice that the image is larger than our training patch size. Our **tiled inference algorithm** applies an overlap-and-crop strategy with a 24-pixel halo, preventing any tile seams.
> 
> We click 'Enhance 4×': in 85 milliseconds, the model outputs the high-resolution result. Comparing side-by-side with the original preview, high-frequency details and crisp edges are restored. We can switch between Bicubic, Baseline CVAE, and SPA-VAEGAN to see the progression from blurry interpolation to crisp edge reconstruction.
> 
> The codebase is fully modular, covered by automated unit tests, and ready on GitHub. Thank you."
