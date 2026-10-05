from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class EncodedImagePayload:
    data: bytes
    mime_type: str
    width: int
    height: int
    encoding: dict[str, object]
    sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        converted = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(converted, dtype=np.float32) / 255.0


def image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = ImageOps.exif_transpose(image).size
    return width, height


def encode_image_payload(
    path: str | Path,
    sent_width: int | None,
    sent_height: int | None,
    *,
    jpeg_quality: int = 92,
) -> EncodedImagePayload:
    with Image.open(Path(path)) as image:
        converted = ImageOps.exif_transpose(image).convert("RGB")
        if sent_width and sent_height and converted.size != (sent_width, sent_height):
            converted = converted.resize((sent_width, sent_height), Image.Resampling.LANCZOS)
        width, height = converted.size
        buffer = io.BytesIO()
        converted.save(buffer, format="JPEG", quality=jpeg_quality)
    data = buffer.getvalue()
    return EncodedImagePayload(
        data=data,
        mime_type="image/jpeg",
        width=width,
        height=height,
        encoding={"format": "JPEG", "quality": jpeg_quality},
        sha256=hashlib.sha256(data).hexdigest(),
    )
