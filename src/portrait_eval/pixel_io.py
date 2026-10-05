"""One fixed-range, precision-preserving decoder for Front measurements and previews."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from qwen_tmqa.asset_io import load_comparison_asset
from qwen_tmqa.comparison_models import ComparisonAsset


def load_normalized_rgb(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    asset = load_comparison_asset(
        ComparisonAsset(
            id="front-image", path=path, encoding="srgb", color_policy="embedded_to_srgb"
        )
    )
    return asset.pixels, asset.trace


def normalized_rgb(image: np.ndarray) -> np.ndarray:
    """Unsigned pixels use their dtype range; floats are already unit-range sRGB."""
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError("Front measurements require nonempty RGB pixels")
    if image.dtype.kind == "u" and image.dtype.itemsize in (1, 2):
        pixels = image.astype(np.float64) / float(np.iinfo(image.dtype).max)
    elif image.dtype.kind == "f":
        pixels = image.astype(np.float64)
    else:
        raise ValueError("Unsupported Front pixel precision")
    if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
        raise ValueError("Front pixels must be finite fixed-range sRGB")
    return pixels


def display_uint8(image: np.ndarray) -> np.ndarray:
    """Fixed unit-range quantization for detectors/JPEG, without per-image stretching."""
    return np.rint(normalized_rgb(image) * 255).astype(np.uint8)
