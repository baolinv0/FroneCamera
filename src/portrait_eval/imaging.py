from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(rgb)


def _luminance(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32) / 255.0
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def detect_primary_face(image: np.ndarray) -> list[int] | None:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda value: value[2] * value[3])
    return [int(x), int(y), int(w), int(h)]


def compute_image_metrics(image: np.ndarray) -> dict[str, float]:
    luma = _luminance(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    residual = gray.astype(np.float32) - cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    return {
        "luma_mean": round(float(luma.mean()), 6),
        "luma_p10": round(float(np.quantile(luma, 0.10)), 6),
        "luma_p50": round(float(np.quantile(luma, 0.50)), 6),
        "luma_p90": round(float(np.quantile(luma, 0.90)), 6),
        "highlight_clip_ratio": round(float((luma >= 0.98).mean()), 6),
        "deep_shadow_ratio": round(float((luma <= 0.02).mean()), 6),
        "saturation_mean": round(float((hsv[..., 1] / 255.0).mean()), 6),
        "apparent_detail": round(float(laplacian.var()), 6),
        "noise_proxy": round(float(residual.std()), 6),
    }


def _masked_metrics(image: np.ndarray, mask: np.ndarray) -> dict[str, float] | None:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if int(mask.sum()) < 16:
        return None
    pixels = image[mask].astype(np.float32) / 255.0
    luma = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    maximum = pixels.max(axis=1)
    minimum = pixels.min(axis=1)
    saturation = np.divide(
        maximum - minimum, maximum, out=np.zeros_like(maximum), where=maximum > 0
    )
    return {
        "pixel_count": float(mask.sum()),
        "luma_mean": round(float(luma.mean()), 6),
        "luma_p10": round(float(np.quantile(luma, 0.10)), 6),
        "luma_p50": round(float(np.quantile(luma, 0.50)), 6),
        "luma_p90": round(float(np.quantile(luma, 0.90)), 6),
        "highlight_clip_ratio": round(float((luma >= 0.98).mean()), 6),
        "deep_shadow_ratio": round(float((luma <= 0.02).mean()), 6),
        "saturation_mean": round(float(saturation.mean()), 6),
    }


def _region_masks(
    shape: tuple[int, int], face: list[int] | None, luma: np.ndarray
) -> dict[str, np.ndarray]:
    height, width = shape
    masks: dict[str, np.ndarray] = {
        "highlight": luma >= 0.98,
        "shadow": luma <= 0.05,
    }
    background = np.ones((height, width), dtype=bool)
    if face:
        x, y, w, h = face
        face_mask = np.zeros((height, width), dtype=np.uint8)
        center = (x + w // 2, y + h // 2)
        axes = (max(w // 2, 1), max(int(h * 0.55), 1))
        cv2.ellipse(face_mask, center, axes, 0, 0, 360, 1, -1)
        face_bool = face_mask.astype(bool)
        masks["face"] = face_bool

        expanded = np.zeros((height, width), dtype=np.uint8)
        x0 = max(0, x - int(0.35 * w))
        y0 = max(0, y - int(0.35 * h))
        x1 = min(width, x + w + int(0.35 * w))
        y1 = min(height, y + h + int(0.35 * h))
        expanded[y0:y1, x0:x1] = 1
        ring = expanded.astype(bool) & ~face_bool
        masks["face_ring"] = ring
        background[expanded.astype(bool)] = False
    masks["background"] = background
    return masks


def _write_diagnostic(
    image: np.ndarray,
    face: list[int] | None,
    masks: dict[str, np.ndarray],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = image.copy()
    if face:
        x, y, w, h = face
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), max(1, image.shape[1] // 500))
    overlay = canvas.astype(np.float32)
    highlight = masks.get("highlight")
    shadow = masks.get("shadow")
    if highlight is not None:
        overlay[highlight] = 0.65 * overlay[highlight] + 0.35 * np.array(
            [255, 80, 80], dtype=np.float32
        )
    if shadow is not None:
        overlay[shadow] = 0.65 * overlay[shadow] + 0.35 * np.array([80, 120, 255], dtype=np.float32)
    bgr = cv2.cvtColor(np.clip(overlay, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr)
    return output_path


def analyze_image(path: Path, artifact_dir: Path | None = None) -> dict[str, object]:
    image = load_rgb(path)
    luma = _luminance(image)
    face = detect_primary_face(image)
    masks = _region_masks(image.shape[:2], face, luma)
    regions: dict[str, dict[str, float]] = {}
    for name, mask in masks.items():
        value = _masked_metrics(image, mask)
        if value is not None:
            regions[name] = value

    payload: dict[str, object] = {
        "whole": compute_image_metrics(image),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "face_bbox": face,
        "regions": regions,
        "warnings": [],
    }
    if face is None:
        payload["warnings"] = ["face_not_detected"]
    else:
        face_metrics = regions.get("face")
        if face_metrics:
            payload["face"] = face_metrics

    if artifact_dir is not None:
        diagnostic_path = artifact_dir / f"{path.stem}-diagnostic.jpg"
        payload["diagnostic_path"] = str(_write_diagnostic(image, face, masks, diagnostic_path))
    return payload


def audit_scene(metrics_by_device: dict[str, dict[str, object]]) -> dict[str, object]:
    valid = [payload for payload in metrics_by_device.values() if payload]
    warnings: list[str] = []
    observations: list[str] = []
    tags: list[str] = []
    if len(valid) < 2:
        return {
            "status": "NOT_COMPARABLE",
            "warnings": ["fewer_than_two_valid_images"],
            "observations": observations,
            "suggested_tags": tags,
        }

    lumas = [float(payload["whole"]["luma_mean"]) for payload in valid]  # type: ignore[index]
    p90s = [float(payload["whole"]["luma_p90"]) for payload in valid]  # type: ignore[index]
    p10s = [float(payload["whole"]["luma_p10"]) for payload in valid]  # type: ignore[index]
    if max(lumas) - min(lumas) > 0.35:
        observations.append("large_rendered_brightness_difference")

    face_count = sum(payload.get("face_bbox") is not None for payload in valid)
    if face_count != len(valid):
        warnings.append("inconsistent_face_detection")
    else:
        face_areas: list[float] = []
        face_centers_x: list[float] = []
        face_centers_y: list[float] = []
        for payload in valid:
            face = payload.get("face_bbox")
            size = payload.get("image_size")
            if (
                not isinstance(face, list)
                or len(face) != 4
                or not isinstance(size, list)
                or len(size) != 2
            ):
                continue
            x, y, width, height = (float(value) for value in face)
            image_width, image_height = (float(value) for value in size)
            if image_width <= 0 or image_height <= 0:
                continue
            face_areas.append((width * height) / (image_width * image_height))
            face_centers_x.append((x + width / 2) / image_width)
            face_centers_y.append((y + height / 2) / image_height)
        if len(face_areas) == len(valid) and min(face_areas) > 0:
            if max(face_areas) / min(face_areas) > 1.35:
                warnings.append("face_scale_difference")
            if (
                max(face_centers_x) - min(face_centers_x) > 0.12
                or max(face_centers_y) - min(face_centers_y) > 0.12
            ):
                warnings.append("face_position_difference")

    if sum(lumas) / len(lumas) < 0.22:
        tags.append("low_light")
    if sum((p90 - p10) for p90, p10 in zip(p90s, p10s, strict=True)) / len(valid) > 0.65:
        tags.append("high_dynamic_range")
    if any(float(payload["whole"]["highlight_clip_ratio"]) > 0.02 for payload in valid):  # type: ignore[index]
        tags.append("strong_highlight")
    return {
        "status": "FULLY_COMPARABLE" if not warnings else "COMPARABLE_WITH_CONFOUNDERS",
        "warnings": warnings,
        "observations": observations,
        "suggested_tags": tags,
    }
