# Phase 4: Real-World Deployment Guide (Streamlit Application)

**Author:** Bevan (@Bevan)  
**Application:** ImageEnhance AI – Powered by SPA-VAEGAN  
**Course:** Advanced Generative AI Continuous Assessment (CAC)

---

## 1. Deployment Architecture Overview

The deployment system packages the trained **SPA-VAEGAN** generator into an interactive, real-time web application built using **Streamlit**. The application is engineered for production deployment across both cloud servers and local consumer hardware (CPU or GPU).

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Streamlit Web UI (app/app.py)                        │
│   - Model Selector (SPA-VAEGAN, Baseline CVAE, Bicubic)                │
│   - Image Uploader (JPG, JPEG, PNG, max 10MB)                          │
│   - Demo Mode Toggle (Simulate synthetic 4x degradation)              │
│   - Interactive Side-by-Side Preview & Download PNG                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Input Validation & Sanitization (app/image_utils.py)      │
│   - EXIF Orientation Transpose                                         │
│   - Alpha Channel Flattening onto White Background                     │
│   - Dimension Validation (16x16 min; 1024x1024 GPU / 512x512 CPU)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              Tiled Inference Engine (app/inference.py)                 │
│   - Padding & Overlap-and-Crop (Core: 64x64, Halo: 24px)              │
│   - Batched Tile Evaluation via CVAE Generator (Prior Mode z = 0)     │
│   - Seamless Canvas Reconstruction (4x Resolution)                     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Overlap-and-Crop Tiled Inference Algorithm

### 2.1 The Problem: Arbitrary Dimensions & Memory Overflow
While the generator is trained on fixed $64 \times 64 \to 256 \times 256$ patches, real-world user uploads feature arbitrary aspect ratios and resolutions (e.g., $320 \times 480$, $800 \times 600$). Passing an entire high-resolution image through a deep residual network can easily trigger GPU Out-Of-Memory (OOM) errors.

### 2.2 Receptive Field & Halo Mathematics
A naive approach—dividing the image into non-overlapping $64 \times 64$ tiles—produces noticeable grid artifacts and discontinuous edge seams along tile boundaries. This occurs because convolutional receptive fields near tile edges lack spatial context from adjacent pixels.

The theoretical receptive field of our 16-block residual trunk is approximately **35 LR pixels**. To completely eliminate boundary seams:
1. **Core Tile ($T = 64$ px):** The central region preserved in the final output.
2. **Halo Overlap ($H = 24$ px):** Symmetrical context padding added to all four borders of each tile, resulting in an input tile size of:
   $$W_{tile} = T + 2H = 64 + 2(24) = 112\text{ px}$$
3. **Core Output Extraction:** The generator outputs a $448 \times 448$ patch ($4 \times 112$). We extract only the central $256 \times 256$ core ($4 \times 64$):
   $$\text{Core} = \text{SR}[:, \, 4H : 4(H+T), \, 4H : 4(H+T)]$$
   and place it into the reconstructed high-resolution canvas.
4. **Equivalence Invariance:** Tests confirm that the Mean Absolute Difference (MAD) between tiled inference and a full-image forward pass is $< 0.4 / 255$, well below human visual perception.

---

## 3. Production Hardening & Input Sanitization

The `app/image_utils.py` module enforces strict boundary checks:

| Invariant | Enforcement Rule | Fallback / User Message |
| :--- | :--- | :--- |
| **File Format** | Extension and MIME type verification. | Only `.jpg`, `.jpeg`, and `.png` allowed. Unsupported types are rejected with a clear warning. |
| **File Size** | Max 10 MB payload limit. | Files $>10$ MB are rejected prior to memory allocation. |
| **File Integrity** | `Image.verify()` execution. | Corrupted files trigger `"Could not read this image. The file may be corrupted."` |
| **EXIF Orientation** | `ImageOps.exif_transpose` applied. | Smartphone photographs taken in portrait/landscape modes are rotated upright automatically. |
| **Color Space** | Strict conversion to 3-channel RGB. | Grayscale and CMYK images are mapped to RGB. |
| **Alpha Transparency** | RGBA / LA images flattened. | Alpha channels are composited over solid white: `Image.alpha_composite(white_bg, img)`. |
| **Resolution Bounds** | Min: $16 \times 16$; Max: $1024 \times 1024$ (GPU) / $512 \times 512$ (CPU). | Oversized images are rejected with dimension indicators, preventing memory crashes. |

---

## 4. User Interface Features

1. **Model Selector:** Allows users to compare **SPA-VAEGAN (Proposed)**, **Baseline CVAE**, and **Bicubic Interpolation** on the same uploaded image.
2. **Demo Mode ("Simulate Low-Resolution Input"):** When enabled, high-resolution photographs are synthetically degraded using the exact DIV2K degradation operator ($\sigma=1.0$ Gaussian blur $+ 4\times$ downsampling). This lets users evaluate real 4× super-resolution reconstruction even when uploading sharp images.
3. **Hardware-Synchronized Latency Display:** Times inference with microsecond precision using `time.perf_counter()` and `torch.cuda.synchronize()`, reporting exact execution time in milliseconds.
4. **Lossless PNG Export:** Enhanced images can be downloaded immediately via `st.download_button`.

---

## 5. Execution Instructions

### Local Execution (CPU / GPU)
```bash
# 1. Activate virtual environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install deployment dependencies
pip install -r requirements-app.txt

# 3. Launch the Streamlit application
streamlit run app/app.py
```
The app initializes at `http://localhost:8501`.
