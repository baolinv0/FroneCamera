"""Boundary tests: removing validation/guarding must change these outcomes."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError


def api():
    from qwen_tmqa.comparison import evaluate_comparison
    from qwen_tmqa.comparison_models import ComparisonRequest

    return evaluate_comparison, ComparisonRequest


def write(tmp_path, name, array):
    path = tmp_path / name
    if path.suffix == ".npy":
        np.save(path, array)
    else:
        Image.fromarray(array).save(path)
    return str(path)


def manifest(tmp_path, base=None, cand=None, mode="algorithm", rois=None):
    base = (
        np.tile(np.arange(32, dtype=np.uint8)[None, :, None], (32, 1, 3)) * 7
        if base is None
        else base
    )
    cand = base if cand is None else cand
    return {
        "scene_id": "s",
        "mode": mode,
        "group_id": "g",
        "baseline": {"id": "B", "path": write(tmp_path, "B.png", base), "encoding": "srgb"},
        "candidates": [{"id": "C", "path": write(tmp_path, "C.png", cand), "encoding": "srgb"}],
        "rois": rois or [],
    }


def test_public_contract_is_present():
    import importlib.util

    assert importlib.util.find_spec("qwen_tmqa.comparison_models") is not None


def test_changed_pixels_multiple_candidates_ten_dimensions_and_finite(tmp_path):
    evaluate, Request = api()
    data = manifest(tmp_path)
    data["source"] = {"id": "S", "path": data["baseline"]["path"], "encoding": "srgb"}
    data["candidates"].append(
        {
            "id": "bright",
            "path": write(
                tmp_path,
                "bright.png",
                np.clip(
                    np.asarray(Image.open(data["baseline"]["path"])).astype(np.int16) + 180, 0, 255
                ).astype(np.uint8),
            ),
            "encoding": "srgb",
        }
    )
    result = evaluate(Request(**data))
    assert len(result.comparisons) == 2
    assert all(c.status == "REVIEW" for c in result.comparisons)
    assert result.comparisons[1].relative_metrics["highlight_fraction_delta"] > 0.5
    assert result.comparisons[0].relative_metrics["pixel_mae"] == 0
    assert result.comparisons[0].same_source_status == "declared_unverified"
    expected = {
        "face_exposure_readability",
        "highlight_integrity",
        "shadow_black_rendering",
        "skin_awb",
        "lighting_causality",
        "face_background_relation",
        "local_face_lift_naturalness",
        "multi_face_consistency",
        "scene_adaptability",
        "artifact_texture_control",
    }
    assert set(result.assets[0].dimensions) == expected
    assert result.assets[0].dimensions["scene_adaptability"].state == "not_applicable"
    assert result.assets[0].dimensions["skin_awb"].state == "unobservable"
    json.dumps(result.model_dump(mode="json"), allow_nan=False)
    assert evaluate(Request(**data)).model_dump() == result.model_dump()


def test_multiple_person_fixed_roi_and_geometry_failure(tmp_path):
    evaluate, Request = api()
    rois = [
        {"id": "f1", "kind": "face", "bbox": (0, 0, 8, 8), "person_id": "one"},
        {"id": "f2", "kind": "face", "bbox": (16, 16, 8, 8), "person_id": "two"},
    ]
    data = manifest(tmp_path, rois=rois)
    out = evaluate(Request(**data))
    assert set(out.assets[0].persons) == {"one", "two"}
    assert out.assets[0].dimensions["multi_face_consistency"].state == "measured_proxy"
    assert out.assets[1].rois["f1"].bbox == (0, 0, 8, 8)
    data["candidates"][0]["path"] = write(tmp_path, "small.png", np.ones((12, 12, 3), np.uint8))
    out = evaluate(Request(**data))
    assert out.comparisons[0].status == "REJECT"
    assert "geometry_mismatch" in out.comparisons[0].fatal_reasons
    assert out.assets[1].rois["f2"].state == "invalid"


def test_device_sizes_never_pixel_compare_without_correspondence(tmp_path):
    evaluate, Request = api()
    data = manifest(
        tmp_path,
        cand=np.ones((12, 12, 3), np.uint8),
        mode="device",
        rois=[{"id": "f", "kind": "face", "bbox": (0, 0, 8, 8), "person_id": "p"}],
    )
    out = evaluate(Request(**data))
    assert out.comparisons[0].status == "REVIEW"
    assert out.comparisons[0].relative_metrics["pixel_mae"] is None
    assert out.assets[0].rois["f"].state == "unobservable"
    assert out.comparisons[0].same_source_status == "not_comparable"


@pytest.mark.parametrize("array", [np.zeros((32, 32, 3), np.uint8), np.ones((2, 2, 3), np.uint8)])
def test_constant_or_sparse_never_perfect_fidelity(tmp_path, array):
    evaluate, Request = api()
    out = evaluate(Request(**manifest(tmp_path, base=array)))
    assert out.comparisons[0].alignment.state == "unobservable"
    assert out.comparisons[0].relative_metrics["edge_correlation"] is None


def test_invalid_nan_missing_and_range_return_explicit_finite_evidence(tmp_path):
    evaluate, Request = api()
    data = manifest(tmp_path)
    for arr in [
        np.full((32, 32, 3), np.nan),
        np.full((32, 32, 3), np.inf),
        np.full((32, 32, 3), -1.0),
    ]:
        data["candidates"][0]["path"] = write(tmp_path, "bad.npy", arr)
        out = evaluate(Request(**data))
        assert out.comparisons[0].status == "REJECT"
        assert out.assets[1].state == "invalid"
        json.dumps(out.model_dump(mode="json"), allow_nan=False)
    data["candidates"][0]["path"] = str(tmp_path / "missing.png")
    assert evaluate(Request(**data)).assets[1].state == "invalid"


def test_duplicate_unknown_extra_roi_and_empty_ids_rejected(tmp_path):
    _, Request = api()
    data = manifest(tmp_path)
    data["candidates"][0]["id"] = "B"
    with pytest.raises(ValidationError):
        Request(**data)
    data = manifest(tmp_path)
    data["unexpected"] = 4
    with pytest.raises(ValidationError):
        Request(**data)
    data.pop("unexpected")
    data["rois"] = [{"id": "bad", "kind": "face", "bbox": (0, 0, 0, 3)}]
    with pytest.raises(ValidationError):
        Request(**data)
    data["rois"] = []
    data["group_id"] = " "
    with pytest.raises(ValidationError):
        Request(**data)


def test_source_byte_hash_claim_is_checked(tmp_path):
    evaluate, Request = api()
    data = manifest(tmp_path)
    data["source"] = {"id": "S", "path": data["baseline"]["path"], "encoding": "srgb"}
    digest = hashlib.sha256(Path(data["baseline"]["path"]).read_bytes()).hexdigest()
    data["baseline"]["source_sha256"] = digest
    data["candidates"][0]["source_sha256"] = "0" * 64
    out = evaluate(Request(**data))
    assert out.comparisons[0].status == "REJECT"
    assert "source_lineage_hash_mismatch" in out.comparisons[0].fatal_reasons


def test_guarded_translation_records_overlap_and_preserves_fixed_roi(tmp_path):
    evaluate, Request = api()
    rng = np.random.default_rng(22)
    base = rng.integers(20, 220, (64, 64, 3), dtype=np.uint8)
    candidate = np.roll(base, 2, axis=1)
    out = evaluate(
        Request(
            **manifest(
                tmp_path,
                base=base,
                cand=candidate,
                rois=[{"id": "face", "kind": "face", "bbox": (0, 0, 16, 16), "person_id": "p"}],
            )
        )
    )
    alignment = out.comparisons[0].alignment
    assert alignment.state == "measured_proxy"
    assert alignment.shift_x == pytest.approx(2, abs=0.5)
    assert alignment.overlap_fraction > 0.9
    assert (
        out.comparisons[0].relative_metrics["aligned_pixel_mae"]
        < out.comparisons[0].relative_metrics["pixel_mae"]
    )
    assert out.assets[1].rois["face"].bbox == (0, 0, 16, 16)


def test_complete_output_content_collapse_is_explicit_fatal(tmp_path):
    evaluate, Request = api()
    data = manifest(tmp_path, cand=np.zeros((32, 32, 3), np.uint8))
    out = evaluate(Request(**data))
    assert out.comparisons[0].status == "REJECT"
    assert "complete_content_collapse" in out.comparisons[0].fatal_reasons


def test_sparse_pixels_cannot_look_like_perfect_structural_fidelity(tmp_path):
    evaluate, Request = api()
    image = np.zeros((64, 64, 3), np.uint8)
    image[20, 20] = 255
    out = evaluate(Request(**manifest(tmp_path, base=image)))
    assert out.comparisons[0].alignment.state == "unobservable"
    assert out.comparisons[0].relative_metrics["edge_correlation"] is None


def test_device_explicit_person_roi_correspondence(tmp_path):
    evaluate, Request = api()
    data = manifest(
        tmp_path,
        mode="device",
        rois=[
            {"id": "face", "asset_id": "B", "kind": "face", "person_id": "p", "bbox": (0, 0, 8, 8)},
            {"id": "face", "asset_id": "C", "kind": "face", "person_id": "p", "bbox": (1, 1, 8, 8)},
        ],
    )
    out = evaluate(Request(**data))
    assert "face" in out.comparisons[0].relative_rois
    assert out.assets[1].rois["face"].bbox == (1, 1, 8, 8)


def test_invalid_existing_pixels_keep_byte_trace_and_input_binding(tmp_path):
    evaluate, Request = api()
    data = manifest(tmp_path)
    data["candidates"][0]["path"] = write(tmp_path, "invalid.npy", np.full((32, 32, 3), np.nan))
    out = evaluate(Request(**data))
    assert out.assets[1].trace["source_bytes_sha256"]
    assert out.input_trace["input_sha256"]
    first = out.input_trace["input_sha256"]
    write(tmp_path, "invalid.npy", np.full((32, 32, 3), np.inf))
    assert evaluate(Request(**data)).input_trace["input_sha256"] != first


@pytest.mark.parametrize("fill", [0, 255])
def test_small_destroyed_person_face_cannot_hide_in_scene_average(tmp_path, fill):
    evaluate, Request = api()
    rng = np.random.default_rng(10)
    base = rng.integers(70, 190, (128, 128, 3), dtype=np.uint8)
    candidate = base.copy()
    candidate[24:32, 24:32] = fill
    data = manifest(
        tmp_path,
        base=base,
        cand=candidate,
        rois=[
            {"id": "person-one", "kind": "face", "person_id": "one", "bbox": (24, 24, 8, 8)},
            {"id": "person-two", "kind": "face", "person_id": "two", "bbox": (64, 64, 8, 8)},
        ],
    )
    data["source"] = {"id": "S", "path": data["baseline"]["path"], "encoding": "srgb"}
    out = evaluate(Request(**data))
    comparison = out.comparisons[0]
    assert comparison.relative_metrics["pixel_mae"] < 0.004
    assert comparison.status == "REJECT"
    assert "local_content_collapse:person-one" in comparison.fatal_reasons
    assert comparison.relative_rois["person-two"]["mean_luminance_delta"] == 0
    data.pop("source")
    uncertain = evaluate(Request(**data)).comparisons[0]
    assert uncertain.status == "REVIEW"
    assert "possible_local_content_collapse:person-one" in uncertain.review_reasons


def test_full_white_output_is_complete_content_collapse(tmp_path):
    evaluate, Request = api()
    out = evaluate(Request(**manifest(tmp_path, cand=np.full((32, 32, 3), 255, np.uint8))))
    assert "complete_content_collapse" in out.comparisons[0].fatal_reasons
