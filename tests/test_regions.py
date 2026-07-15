from pathlib import Path

from PIL import Image

from portrait_eval.imaging import analyze_image


def test_analysis_contains_adaptive_regions_and_diagnostic(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (64, 64), (128, 128, 128)).save(image_path)
    result = analyze_image(image_path, tmp_path / "artifacts")
    assert "regions" in result
    assert "background" in result["regions"]
    assert Path(result["diagnostic_path"]).exists()
