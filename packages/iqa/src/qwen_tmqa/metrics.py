from __future__ import annotations

import math

import cv2
import numpy as np

from .domain import ObjectiveMetrics, SequenceMetrics


def _luminance(image: np.ndarray) -> np.ndarray:
    return 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]


def _edge_map(image: np.ndarray) -> np.ndarray:
    gray = _luminance(image).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = left.reshape(-1).astype(np.float64)
    b = right.reshape(-1).astype(np.float64)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1, 1))


def compute_objective_metrics(
    image: np.ndarray,
    *,
    alpha: float,
    level: str,
    baseline: np.ndarray,
) -> ObjectiveMetrics:
    if image.shape != baseline.shape:
        raise ValueError("image and baseline shapes must match")
    y = _luminance(np.clip(image, 0, 1))
    epsilon = 1e-6
    mean_luminance = float(np.mean(y))
    ev_mean = float(math.log2(mean_luminance + epsilon))
    p20, p50, p90 = [float(value) for value in np.percentile(y, [20, 50, 90])]
    clipping_ratio = float(np.mean(np.max(image, axis=2) >= 0.99))
    shadow_ratio = float(np.mean(y <= 0.02))
    contrast = float(np.std(y))
    image_mean = np.mean(image, axis=(0, 1))
    baseline_mean = np.mean(baseline, axis=(0, 1))
    image_chroma = image_mean / max(float(np.sum(image_mean)), epsilon)
    baseline_chroma = baseline_mean / max(float(np.sum(baseline_mean)), epsilon)
    color_drift = float(np.linalg.norm(image_chroma - baseline_chroma))
    edge_similarity = (_correlation(_edge_map(image), _edge_map(baseline)) + 1) / 2
    return ObjectiveMetrics(
        alpha=alpha,
        level=level,
        mean_luminance=mean_luminance,
        ev_mean=ev_mean,
        p20=p20,
        p50=p50,
        p90=p90,
        clipping_ratio=clipping_ratio,
        shadow_ratio=shadow_ratio,
        contrast=contrast,
        color_drift=color_drift,
        edge_similarity=float(np.clip(edge_similarity, 0, 1)),
    )


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    for index, count in enumerate(counts):
        if count > 1:
            positions = np.where(inverse == index)[0]
            ranks[positions] = np.mean(ranks[positions])
    return ranks


def compute_sequence_metrics(metrics: list[ObjectiveMetrics]) -> SequenceMetrics:
    ordered = sorted(metrics, key=lambda item: item.alpha)
    if len(ordered) < 2:
        raise ValueError("at least two alpha levels are required")
    alphas = np.array([item.alpha for item in ordered], dtype=np.float64)
    brightness = np.array([item.ev_mean for item in ordered], dtype=np.float64)
    alpha_rank = _rank(alphas)
    brightness_rank = _rank(brightness)
    spearman = _correlation(alpha_rank, brightness_rank)
    deltas = np.diff(brightness)
    violation_rate = float(np.mean(deltas < -0.02))
    dead_zone_ratio = float(np.mean(np.abs(deltas) < 0.02))
    if len(brightness) > 2:
        second = np.diff(brightness, n=2)
        smoothness_score = float(np.exp(-3 * np.mean(np.abs(second))))
    else:
        smoothness_score = 1.0
    endpoint_range = float(brightness[-1] - brightness[0])
    clipping = np.array([item.clipping_ratio for item in ordered], dtype=np.float64)
    clipping_growth = float(max(0.0, clipping[-1] - clipping[len(clipping) // 2]))
    control_score = float(
        np.clip(
            0.35 * max(0, spearman)
            + 0.25 * (1 - violation_rate)
            + 0.20 * smoothness_score
            + 0.10 * (1 - dead_zone_ratio)
            + 0.10 * np.exp(-3 * clipping_growth),
            0,
            1,
        )
    )
    return SequenceMetrics(
        spearman_rho=spearman,
        violation_rate=violation_rate,
        smoothness_score=smoothness_score,
        dead_zone_ratio=dead_zone_ratio,
        endpoint_range_ev=endpoint_range,
        clipping_growth=clipping_growth,
        control_score=control_score,
    )
