from portrait_eval.attribution import attribute_output_claim


def test_global_luminance_claim_remains_indeterminate_without_exif() -> None:
    result = attribute_output_claim(
        statement="This device tends to render higher global luminance.",
        hardware_context={},
        exif_context={},
    )
    assert result.primary_layer == "indeterminate"
    assert "capture exposure" in result.candidate_causes


def test_attribution_preserves_hardware_and_exif_evidence_summary() -> None:
    result = attribute_output_claim(
        statement="This device tends to preserve more apparent detail.",
        hardware_context={"sources": [{"url": "https://example.test/spec"}]},
        exif_context={"ISO": "800", "ExposureTime": "1/30"},
    )
    assert result.hardware_evidence_count == 1
    assert result.exif_evidence_keys == ["ExposureTime", "ISO"]
    assert result.primary_layer == "capture_or_rendering"
