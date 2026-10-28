# Anima Reference for Forge Neo
# The reference image fitted into the output size, centered on a grey/white canvas (training used white, 255).
# Ported from ComfyUI-Anima-Ref-Adapter (ElectronGirl, Apache-2.0), __init__.py, AnimaRefLatent._letterbox.

import numpy as np
import torch
from PIL import Image


def letterbox(src: Image.Image, width: int, height: int, fill: int = 255) -> Image.Image:
    src = src.convert("RGB")
    scale = min(width / src.width, height / src.height)
    size = (max(1, round(src.width * scale)), max(1, round(src.height * scale)))
    canvas = Image.new("RGB", (width, height), (fill, fill, fill))
    canvas.paste(src.resize(size, Image.LANCZOS), ((width - size[0]) // 2, (height - size[1]) // 2))
    return canvas


def to_pixels(image: Image.Image) -> torch.Tensor:
    """(1, H, W, 3) in [0, 1], what Forge's VAE.encode takes."""
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
