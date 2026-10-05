import cv2
import numpy as np

from qwen_tmqa.cli import _base_scene


def test_example_scene_uses_uint8_for_opencv_drawing(monkeypatch) -> None:
    original = cv2.putText

    def checking_put_text(image, *args, **kwargs):
        assert image.dtype == np.uint8
        return original(image, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", checking_put_text)
    scene = _base_scene(0)
    assert scene.dtype == np.float32
    assert 0.0 <= float(scene.min()) <= float(scene.max()) <= 1.0
