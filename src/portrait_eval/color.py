"""One explicit SDR rendering shared by metrics, model inputs and report previews."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageCms, ImageOps

RENDER_VERSION = "sdr-srgb-0.2"
SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def load_analysis_rgb(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with path.open("rb") as handle:
        input_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    with path.open("rb") as handle:
        header = handle.read(26)
    with Image.open(path) as source:
        source.seek(0)
        icc = source.info.get("icc_profile")
        record: dict[str, Any] = {
            "version": RENDER_VERSION,
            "input_sha256": input_sha256,
            "container": source.format,
            "source_mode": source.mode,
            "source_bit_depth": header[24]
            if header.startswith(b"\x89PNG\r\n\x1a\n")
            else source.info.get("bit_depth"),
            "decoder_mode_precision": 16 if "16" in source.mode else 8,
            "exif_orientation": int(source.getexif().get(274, 1)),
            "selected_frame": 0,
            "frame_count": getattr(source, "n_frames", 1),
            "hdr_analysis": "sdr_base_only",
            "source_transfer": "icc_defined" if icc else "unknown",
            "output_space": "sRGB",
            "output_bit_depth": 8,
            "output_transfer": "IEC_61966-2-1",
            "color_claims_allowed": False,
            "warnings": [],
        }
        oriented = ImageOps.exif_transpose(source)
        if icc:
            record["icc_sha256"] = hashlib.sha256(icc).hexdigest()
            try:
                profile = ImageCms.ImageCmsProfile(BytesIO(icc))
                record["icc_description"] = ImageCms.getProfileDescription(profile).strip()
                rgb = ImageCms.profileToProfile(
                    oriented,
                    profile,
                    ImageCms.ImageCmsProfile(BytesIO(SRGB_PROFILE)),
                    outputMode="RGB",
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                )
                record.update(color_status="converted_icc", color_claims_allowed=True)
            except (ImageCms.PyCMSError, OSError, ValueError, TypeError):
                rgb = oriented.convert("RGB")
                record["color_status"] = "invalid_icc"
                record["source_transfer"] = "unknown"
                record["warnings"].append(
                    "ICC conversion failed; preview only, color claims blocked"
                )
        else:
            rgb = oriented.convert("RGB")
            record["color_status"] = (
                "assumed_srgb" if source.mode in {"RGB", "RGBA", "L"} else "unsupported_color"
            )
            record["warnings"].append(
                "No usable ICC; assumed SDR sRGB for preview, color claims blocked"
            )
        if source.mode not in {"RGB", "L", "CMYK"}:
            record["color_claims_allowed"] = False
            record["warnings"].append("Source mode conversion is not color-calibrated")
        return np.asarray(rgb).copy(), record


def save_analysis_image(pixels: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, icc_profile=SRGB_PROFILE)
    return path
