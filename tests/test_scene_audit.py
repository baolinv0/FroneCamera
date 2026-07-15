from portrait_eval.imaging import audit_scene


def payload(luma: float, face_bbox: list[int], size: tuple[int, int] = (100, 100)) -> dict[str, object]:
    return {"whole": {"luma_mean": luma, "luma_p90": min(1.0, luma + 0.2), "luma_p10": max(0.0, luma - 0.2), "highlight_clip_ratio": 0.0, "apparent_detail": 100.0}, "face_bbox": face_bbox, "image_size": list(size)}


def test_rendered_brightness_difference_is_observation_not_capture_confounder() -> None:
    audit = audit_scene({"a": payload(0.15, [25, 20, 40, 50]), "b": payload(0.75, [25, 20, 40, 50])})
    assert audit["status"] == "FULLY_COMPARABLE"
    assert "large_rendered_brightness_difference" in audit["observations"]
    assert "large_global_brightness_difference" not in audit["warnings"]


def test_face_scale_and_position_mismatch_are_capture_confounders() -> None:
    audit = audit_scene({"a": payload(0.4, [10, 10, 20, 30]), "b": payload(0.4, [55, 35, 40, 55])})
    assert audit["status"] == "COMPARABLE_WITH_CONFOUNDERS"
    assert "face_scale_difference" in audit["warnings"]
    assert "face_position_difference" in audit["warnings"]
