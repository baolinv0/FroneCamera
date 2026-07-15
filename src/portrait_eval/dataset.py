from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

from PIL import ExifTags, Image

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

from portrait_eval.models import DeviceScan, PairingDraft, PairingGroup

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


class ImageInspection(TypedDict):
    path: str
    filename: str
    checksum: str
    width: int
    height: int
    exif: dict[str, str]


def natural_key(value: str | Path) -> list[object]:
    text = Path(value).name if isinstance(value, Path) else value
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", text)]


def scan_folder(device_id: str, folder: Path) -> DeviceScan:
    if not folder.is_dir():
        raise ValueError(f"Device folder does not exist: {folder}")
    files = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=natural_key,
    )
    return DeviceScan(device_id=device_id, files=files)


def propose_pairing(scans: Iterable[DeviceScan]) -> PairingDraft:
    scan_list = list(scans)
    if len(scan_list) < 2:
        raise ValueError("At least two devices are required")
    count = max((len(scan.files) for scan in scan_list), default=0)
    groups = []
    for index in range(count):
        cells = {
            scan.device_id: scan.files[index] if index < len(scan.files) else None
            for scan in scan_list
        }
        groups.append(PairingGroup(group_id=f"G{index + 1:03d}", cells=cells))
    return PairingDraft(groups=groups)


def inspect_image(path: Path) -> ImageInspection:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    with Image.open(path) as image:
        exif_raw = image.getexif()
        exif = {ExifTags.TAGS.get(key, str(key)): str(value) for key, value in exif_raw.items()}
        return {
            "path": str(path.resolve()),
            "filename": path.name,
            "checksum": digest.hexdigest(),
            "width": image.width,
            "height": image.height,
            "exif": exif,
        }
