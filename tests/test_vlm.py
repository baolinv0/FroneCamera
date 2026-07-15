from base64 import b64decode
from io import BytesIO
from pathlib import Path

from PIL import Image

from portrait_eval.vlm import encode_image_data_url, parse_json_object


def test_parse_json_object_accepts_markdown_fenced_model_output() -> None:
    parsed = parse_json_object('```json\n{"observations": [], "confidence": 0.8}\n```')
    assert parsed["confidence"] == 0.8


def test_parse_json_object_extracts_single_object_from_explanation() -> None:
    parsed = parse_json_object('Result follows: {"observations": [], "confidence": 0.7} end.')
    assert parsed["confidence"] == 0.7


def test_encode_image_data_url_limits_long_edge_and_normalizes_mime(tmp_path: Path) -> None:
    path = tmp_path / "large.png"
    Image.new("RGB", (2400, 1200), "gray").save(path)
    url = encode_image_data_url(path, max_edge=600)
    assert url.startswith("data:image/jpeg;base64,")
    encoded = url.split(",", 1)[1]
    with Image.open(BytesIO(b64decode(encoded))) as image:
        assert image.size == (600, 300)
