from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

import numpy as np
from PIL import ExifTags, Image, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

from portrait_eval.models import DeviceScan, PairingDraft, PairingGroup

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
SEQUENCE_PLACEHOLDER_EXTENSIONS = {".dng"}


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
    ordered_images = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in (SUPPORTED_EXTENSIONS | SEQUENCE_PLACEHOLDER_EXTENSIONS)
        ),
        key=natural_key,
    )
    files = [path for path in ordered_images if path.suffix.lower() in SUPPORTED_EXTENSIONS]
    sequence_slots = [
        path if path.suffix.lower() in SUPPORTED_EXTENSIONS else None for path in ordered_images
    ]
    return DeviceScan(device_id=device_id, files=files, sequence_slots=sequence_slots)


def _image_feature(path: Path, size: int = 32) -> np.ndarray | None:
    """Return a compact tone-normalized scene feature, or None for unreadable inputs."""
    try:
        with Image.open(path) as source:
            source.draft("RGB", (size * 2, size * 2))
            image = ImageOps.exif_transpose(source).convert("RGB").resize((size, size))
            rgb = np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, ValueError):
        return None

    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    luma = (luma - float(luma.mean())) / (float(luma.std()) + 1e-6)
    gradient_y, gradient_x = np.gradient(luma)
    feature = np.concatenate(
        (luma.reshape(-1), 0.45 * gradient_x.reshape(-1), 0.45 * gradient_y.reshape(-1))
    )
    norm = float(np.linalg.norm(feature))
    return feature / max(norm, 1e-6)


def _feature_distance(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    return float(np.clip(1.0 - np.dot(left, right), 0.0, 2.0))


def _align_to_reference(
    reference: DeviceScan,
    candidate: DeviceScan,
    feature_cache: dict[Path, np.ndarray | None],
) -> tuple[dict[int, int], dict[int, float]]:
    """Monotonically align every candidate image to a unique reference position."""
    reference_count = len(reference.files)
    candidate_count = len(candidate.files)
    if candidate_count > reference_count:
        raise ValueError("Reference scan must be at least as long as candidate scan")
    if candidate_count == 0:
        return {}, {}

    def feature(path: Path) -> np.ndarray | None:
        if path not in feature_cache:
            feature_cache[path] = _image_feature(path)
        return feature_cache[path]

    reference_features = [feature(path) for path in reference.files]
    candidate_features = [feature(path) for path in candidate.files]
    distances = np.empty((reference_count, candidate_count), dtype=np.float32)
    for ref_index, ref_feature in enumerate(reference_features):
        for candidate_index, candidate_feature in enumerate(candidate_features):
            feature_distance = _feature_distance(ref_feature, candidate_feature)
            relative_ref = ref_index / max(reference_count - 1, 1)
            relative_candidate = candidate_index / max(candidate_count - 1, 1)
            position_cost = 0.12 * abs(relative_ref - relative_candidate)
            distances[ref_index, candidate_index] = (
                feature_distance if feature_distance is not None else 0.65
            ) + position_cost

    infinity = float("inf")
    costs = np.full((reference_count + 1, candidate_count + 1), infinity, dtype=np.float64)
    choices = np.zeros((reference_count + 1, candidate_count + 1), dtype=np.int8)
    costs[:, 0] = 0.0
    for ref_count in range(1, reference_count + 1):
        max_candidates = min(ref_count, candidate_count)
        for candidate_used in range(1, max_candidates + 1):
            skip_cost = costs[ref_count - 1, candidate_used]
            match_cost = (
                costs[ref_count - 1, candidate_used - 1]
                + distances[ref_count - 1, candidate_used - 1]
            )
            if match_cost <= skip_cost:
                costs[ref_count, candidate_used] = match_cost
                choices[ref_count, candidate_used] = 1
            else:
                costs[ref_count, candidate_used] = skip_cost

    mapping: dict[int, int] = {}
    confidences: dict[int, float] = {}
    ref_count = reference_count
    candidate_used = candidate_count
    while candidate_used > 0 and ref_count > 0:
        if choices[ref_count, candidate_used] == 1:
            ref_index = ref_count - 1
            candidate_index = candidate_used - 1
            mapping[ref_index] = candidate_index
            raw_distance = _feature_distance(
                reference_features[ref_index], candidate_features[candidate_index]
            )
            if raw_distance is None:
                confidence = 0.25
            else:
                alternatives = sorted(
                    value
                    for index, feature in enumerate(reference_features)
                    if index != ref_index
                    and (value := _feature_distance(feature, candidate_features[candidate_index]))
                    is not None
                )
                margin = (alternatives[0] - raw_distance) if alternatives else 0.2
                quality_score = float(np.clip(1.0 - raw_distance / 0.75, 0.0, 1.0))
                margin_score = float(np.clip(margin / 0.18, 0.0, 1.0))
                confidence = 0.7 * quality_score + 0.3 * margin_score
            confidences[ref_index] = round(confidence, 3)
            ref_count -= 1
            candidate_used -= 1
        else:
            ref_count -= 1
    return mapping, confidences


def _strong_order_mapping(
    candidate: DeviceScan,
    reference_count: int,
) -> tuple[dict[int, int], dict[int, float], str] | None:
    slots = candidate.sequence_slots
    if slots and len(slots) == reference_count:
        file_indices = {path: index for index, path in enumerate(candidate.files)}
        mapping = {
            ref_index: file_indices[path]
            for ref_index, path in enumerate(slots)
            if path is not None
        }
        strategy = "ordered_slots_with_raw_placeholders" if None in slots else "equal_count_order"
        score = 0.99 if None in slots else 0.9
        confidence = {ref_index: score for ref_index in mapping}
        return mapping, confidence, strategy

    ordinal_values: list[int] = []
    for path in candidate.files:
        if not re.fullmatch(r"\d+", path.stem):
            break
        ordinal_values.append(int(path.stem))
    if (
        len(ordinal_values) == len(candidate.files)
        and len(set(ordinal_values)) == len(ordinal_values)
        and ordinal_values
        and min(ordinal_values) >= 1
        and max(ordinal_values) <= reference_count
    ):
        mapping = {
            ordinal - 1: candidate_index for candidate_index, ordinal in enumerate(ordinal_values)
        }
        return mapping, {ref_index: 0.99 for ref_index in mapping}, "explicit_numeric_ordinal"
    return None


def propose_pairing(scans: Iterable[DeviceScan]) -> PairingDraft:
    scan_list = list(scans)
    if len(scan_list) < 2:
        raise ValueError("At least two devices are required")
    reference = max(scan_list, key=lambda scan: len(scan.files))
    cells_by_group: list[dict[str, Path | None]] = [
        {scan.device_id: None for scan in scan_list} for _ in range(len(reference.files))
    ]
    confidence_by_group: list[list[float]] = [[] for _ in reference.files]
    strategies_by_group: list[list[str]] = [[] for _ in reference.files]
    feature_cache: dict[Path, np.ndarray | None] = {}
    cells_by_group_ids = range(len(reference.files))
    for ref_index, path in enumerate(reference.files):
        cells_by_group[ref_index][reference.device_id] = path

    for scan in scan_list:
        if scan is reference:
            continue
        strong_mapping = _strong_order_mapping(scan, len(reference.files))
        if strong_mapping is not None:
            mapping, confidences, strategy = strong_mapping
        else:
            mapping, confidences = _align_to_reference(reference, scan, feature_cache)
            strategy = "content_sequence_alignment"
        for ref_index, candidate_index in mapping.items():
            cells_by_group[ref_index][scan.device_id] = scan.files[candidate_index]
            confidence_by_group[ref_index].append(confidences[ref_index])
            strategies_by_group[ref_index].append(f"{scan.device_id}:{strategy}")

    groups = []
    for index in cells_by_group_ids:
        cells = cells_by_group[index]
        missing = [device_id for device_id, path in cells.items() if path is None]
        confidence_values = confidence_by_group[index]
        confidence = round(float(np.median(confidence_values)), 3) if confidence_values else None
        low_confidence = confidence is None or confidence < 0.55
        notes = [f"reference_device={reference.device_id}", *strategies_by_group[index]]
        if missing:
            notes.append(f"missing_devices={','.join(missing)}")
        if low_confidence:
            notes.append("content_match_requires_review")
        groups.append(
            PairingGroup(
                group_id=f"G{index + 1:03d}",
                cells=cells,
                matching_confidence=confidence,
                review_required=low_confidence
                or sum(path is not None for path in cells.values()) < 2,
                match_notes=notes,
            )
        )
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
