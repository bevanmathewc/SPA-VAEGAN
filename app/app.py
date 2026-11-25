"""
Streamlit Web Application: ImageEnhance AI – Powered by SPA-VAEGAN
4x Single Image Super-Resolution with Structural Edge Preservation.
"""

import io
import os
import time
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F

from app.image_utils import validate_and_preprocess_image
from app.inference import load_model, upscale_tiled
from src.degradation import degrade


# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ImageEnhance AI – SPA-VAEGAN",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Device & Model Cache
# -----------------------------------------------------------------------------
@st.cache_resource
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def get_model(model_name: str, device_str: str):
    dev = torch.device(device_str)
    if model_name == "spa_vaegan":
        path = "checkpoints/spa_vaegan_G.pt"
        if not os.path.exists(path):
            path = "checkpoints/spa_vaegan/best.pt"
    elif model_name == "baseline":
        path = "checkpoints/cvae_baseline_G.pt"
        if not os.path.exists(path):
            path = "checkpoints/baseline/best.pt"
    else:
        path = ""
    return load_model(path, device=device_str)


# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Model & Settings")
device = get_device()
st.sidebar.info(f"**Execution Device:** `{device.type.upper()}`")

model_choice = st.sidebar.selectbox(
    "Select Model:",
    options=["spa_vaegan", "baseline", "bicubic"],
    format_func=lambda x: {
        "spa_vaegan": "🌟 SPA-VAEGAN (Proposed)",
        "baseline": "📉 Baseline CVAE",
        "bicubic": "📐 Bicubic Interpolation"
    }[x]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧩 Tiled Inference")
tile_size = st.sidebar.slider("Tile Size (Core):", 32, 128, 64, step=32)
halo_size = st.sidebar.slider("Halo Padding (Overlapping):", 12, 48, 24, step=6)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Operational Limits")
max_dim = 1024 if device.type == "cuda" else 512
st.sidebar.markdown(f"- **Max Input Resolution:** `{max_dim}x{max_dim}`")
st.sidebar.markdown("- **Scale Factor:** `4x`")
st.sidebar.markdown("- **Supported Formats:** JPG, JPEG, PNG (≤10 MB)")

# -----------------------------------------------------------------------------
# Main Interface
# -----------------------------------------------------------------------------
st.title("ImageEnhance AI – Powered by SPA-VAEGAN")
st.markdown(
    "**Structure-Preserving Adaptive VAE-GAN for 4× Single Image Super-Resolution.**\n\n"
    "Overcomes the inherent blurriness of Variational Autoencoders by coupling conditional residual learning "
    "with a Spectral-Normalized PatchGAN discriminator, normalized Sobel edge preservation, and dynamic gradient balancing."
)

st.caption("Note: Models are trained on synthetic 4× degradation (Gaussian blur $\\sigma=1.0$ + bicubic downsampling).")

# Demo Mode Checkbox
demo_mode = st.checkbox("🧪 Demo Mode: Simulate low-resolution degradation on high-res upload", value=False)
if demo_mode:
    st.info("Demo Mode Active: Uploaded image will be synthetically blurred and downscaled by 4x to show the exact degradation pipeline.")

uploaded_file = st.file_uploader("Upload an Image to Enhance:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img, warning_msg, error_msg = validate_and_preprocess_image(
        uploaded_file,
        max_dim_gpu=1024,
        max_dim_cpu=512,
        device=device.type
    )

    if error_msg:
        st.error(error_msg)
    else:
        if warning_msg:
            st.warning(warning_msg)

        orig_w, orig_h = img.size
        
        # If demo mode is active, simulate synthetic degradation
        if demo_mode:
            img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
            lr_sim = degrade(img_t)
            lr_np = (lr_sim[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
            img = Image.fromarray(lr_np)
            orig_w, orig_h = img.size

        st.markdown(f"**Input Image Dimensions:** `{orig_w} × {orig_h}` pixels")

        col_orig, col_enhance = st.columns(2)
        with col_orig:
            st.markdown("#### Input Image (Nearest-Neighbor 4× Preview)")
            # Show original enlarged with nearest-neighbor to avoid interpolation smoothing
            st.image(img.resize((orig_w * 4, orig_h * 4), Image.NEAREST), use_column_width=True)

        with col_enhance:
            st.markdown("#### Enhanced Output (4× Super-Resolution)")
            if st.button("🚀 Enhance (4×)", type="primary"):
                with st.spinner("Processing super-resolution via deep neural network..."):
                    img_np = np.array(img)

                    if model_choice == "bicubic":
                        t0 = time.perf_counter()
                        sr_img = img.resize((orig_w * 4, orig_h * 4), Image.BICUBIC)
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        sr_np = np.array(sr_img)
                    else:
                        model = get_model(model_choice, device.type)
                        sr_np, elapsed_ms = upscale_tiled(
                            model,
                            img_np,
                            device=device,
                            tile=tile_size,
                            halo=halo_size,
                            bs=4 if device.type == "cpu" else 16
                        )
                        sr_img = Image.fromarray(sr_np)

                    out_h, out_w, _ = sr_np.shape
                    st.image(sr_img, use_column_width=True)
                    st.success(f"Enhanced in **{elapsed_ms:.1f} ms** | Output Resolution: **{out_w} × {out_h}** px")

                    # Download button
                    buf = io.BytesIO()
                    sr_img.save(buf, format="PNG")
                    st.download_button(
                        label="💾 Download Enhanced PNG",
                        data=buf.getvalue(),
                        file_name=f"enhanced_4x_{model_choice}.png",
                        mime="image/png"
                    )
