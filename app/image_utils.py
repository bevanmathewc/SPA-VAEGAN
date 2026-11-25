"""
Image Preprocessing and Validation Utilities for Streamlit Application.
Handles EXIF orientation, alpha channel flattening, format verification, and size bounds.
"""

import io
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageOps


def validate_and_preprocess_image(
    uploaded_file,
    max_dim_gpu: int = 1024,
    max_dim_cpu: int = 512,
    device: str = "cpu"
) -> Tuple[Optional[Image.Image], Optional[str], Optional[str]]:
    """
    Validates uploaded image and converts it into a standardized RGB PIL Image.
    Returns: (pil_image, warning_msg, error_msg)
    """
    # 1. File size check (max 10 MB)
    max_bytes = 10 * 1024 * 1024
    if hasattr(uploaded_file, "size") and uploaded_file.size > max_bytes:
        return None, None, "File too large (max 10 MB)."

    # 2. File type check
    filename = getattr(uploaded_file, "name", "").lower()
    if not (filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")):
        return None, None, "Unsupported file type. Only JPG, JPEG, and PNG are supported."

    # 3. Readability & Integrity
    try:
        bytes_data = uploaded_file.read()
        img = Image.open(io.BytesIO(bytes_data))
        img.verify()
        # Re-open for actual processing after verification
        img = Image.open(io.BytesIO(bytes_data))
    except Exception:
        return None, None, "Could not read this image. The file may be corrupted."

    # 4. EXIF Orientation correction
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # 5. Dimension checks
    w, h = img.size
    if w < 16 or h < 16:
        return None, None, f"Image too small ({w}x{h} px). Minimum dimension is 16x16."

    max_dim = max_dim_gpu if device == "cuda" else max_dim_cpu
    if w > max_dim or h > max_dim:
        return None, None, f"Image is {w}x{h} px; the maximum allowed input dimension on {device.upper()} is {max_dim}x{max_dim}."

    warning_msg = None
    # 6. Mode conversions
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img_rgba = img.convert("RGBA")
        background = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img_rgba).convert("RGB")
        warning_msg = "Note: Alpha transparency channel was detected and flattened onto white."
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img, warning_msg, None
