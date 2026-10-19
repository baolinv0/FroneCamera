import numpy as np

from portrait_eval.imaging import compute_image_metrics


def test_metrics_detect_highlight_and_shadow() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:5] = 255
    metrics = compute_image_metrics(image)
    assert metrics["highlight_clip_ratio"] == 0.5
    assert metrics["deep_shadow_ratio"] == 0.5
    assert 0.49 <= metrics["luma_mean"] <= 0.51
