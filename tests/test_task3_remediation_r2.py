"""Fresh Critic3 R2 precision and structural-context regressions."""

from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image

from portrait_eval.imaging import analyze_image
from portrait_eval.model_validation import anonymous_context
from portrait_eval.vlm import encode_image_data_url


@pytest.mark.parametrize("channels", [1, 3])
def test_r2_01_uint16_png_measurement_and_model_jpeg_preserve_fixed_order(tmp_path, channels):
    means, rendered_means = [], []
    for level in (10000, 50000):
        shape = (96, 96) if channels == 1 else (96, 96, 3)
        path = tmp_path / f"{channels}-{level}.png"
        assert cv2.imwrite(str(path), np.full(shape, level, dtype=np.uint16))
        means.append(analyze_image(path)["whole"]["luma_mean"])
        url = encode_image_data_url(path)
        with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
            rendered_means.append(float(np.asarray(image).mean()) / 255)
    assert means[0] < means[1] < 0.9
    assert means == pytest.approx([10000 / 65535, 50000 / 65535], abs=1e-6)
    assert rendered_means[0] < rendered_means[1] < 0.9
    assert rendered_means == pytest.approx([10000 / 65535, 50000 / 65535], abs=1 / 255)


def test_r2_02_all_canonical_dimensions_and_scientific_keys_survive_anonymization():
    from portrait_eval.core.models_v2 import DimensionId

    dimensions = {item.value: {"state": "unobservable"} for item in DimensionId}
    context = {
        "iqa_evidence": {"assets": [{"id": "high", "dimensions": dimensions}]},
        "metrics": {"high": {"whole": {"highlight_clip_ratio": 0.25, "low_light": 0.5}}},
        "nested": {
            "diagnostic_path": "/private/high/image.png",
            "note": "device high uses low capture",
            "source_ref": "metric:G001:high:highlight_clip_ratio",
        },
    }
    result = anonymous_context(context, {"high": "A", "low": "B"})
    assert set(result["iqa_evidence"]["assets"][0]["dimensions"]) == {
        item.value for item in DimensionId
    }
    assert result["metrics"]["A"]["whole"] == {"highlight_clip_ratio": 0.25, "low_light": 0.5}
    assert result["nested"]["source_ref"] == "metric:G001:A:highlight_clip_ratio"
    assert "diagnostic_path" not in result["nested"]


def test_r2_01_actual_pipeline_does_not_support_dark_uint16_as_highest(tmp_path, monkeypatch):
    from portrait_eval.database import Database
    from portrait_eval.models import ModelEvaluationResult, ModelObservation
    from portrait_eval.pipeline import EvaluationPipeline
    from portrait_eval.repository import Repository
    from portrait_eval.vlm import VisionModelAdapter

    monkeypatch.setattr("portrait_eval.imaging.detect_faces", lambda image: [[12, 12, 45, 55]])

    class WrongConfiguredAdapter(VisionModelAdapter):
        def __init__(self, model):
            self.model = model
            self.contexts = []

        def analyze(self, scene_id, image_paths, context):
            self.contexts.append(context)

            def original_mean(code):
                with Image.open(image_paths[code]) as image:
                    return np.asarray(image).mean()

            code = min(image_paths, key=original_mean)
            return ModelEvaluationResult(
                scene_id=scene_id,
                role="probe",
                confidence=1,
                observations=[
                    ModelObservation(
                        device_id=code,
                        dimension="global_exposure",
                        statement="This output has the highest display-referred mean luminance in the matched group.",
                        evidence_refs=[f"asset:{scene_id}:{code}"],
                        certainty=1,
                    )
                ],
            )

    database = Database(f"sqlite:///{tmp_path / 'precision.sqlite'}")
    database.create_all()
    with database.session_factory() as session:
        repo = Repository(session)
        project = repo.create_project("uint16 supported input")
        ids = []
        for name, level in (("low", 10000), ("high", 50000)):
            folder = tmp_path / name
            folder.mkdir()
            assert cv2.imwrite(str(folder / "1.png"), np.full((96, 96), level, dtype=np.uint16))
            ids.append(repo.add_device(project.id, name, str(folder)).id)
        pairing = repo.scan_and_pair(project.id)
        repo.confirm_pairing(project.id, pairing["version"])
        primary, reviewer = WrongConfiguredAdapter("primary"), WrongConfiguredAdapter("reviewer")
        EvaluationPipeline(session, tmp_path / "workspace", primary=primary, reviewer=reviewer).run(
            project.id
        )
        metrics = {
            item["image_id"]: item["payload"]
            for item in repo.list_analysis(project.id, "image_metrics")
        }
        by_device = {
            device_id: metrics[pairing["groups"][0]["cells"][device_id]["image_id"]]
            for device_id in ids
        }
        assert by_device[ids[0]]["whole"]["luma_mean"] < by_device[ids[1]]["whole"]["luma_mean"]
        assert all(payload["input_trace"]["bit_depth"] == 16 for payload in metrics.values())
        evidence = repo.list_analysis(project.id, "iqa_evaluation")[0]["payload"]
        core_means = {
            asset["id"]: asset["objective"]["mean_luminance"] for asset in evidence["assets"]
        }
        for device_id in ids:
            display_mean = by_device[device_id]["whole"]["luma_mean"]
            assert core_means[device_id] == pytest.approx(
                ((display_mean + 0.055) / 1.055) ** 2.4, abs=1e-6
            )
        claim = repo.list_analysis(project.id, "claim")[0]["payload"]
        assert claim["device_id"] == ids[0]
        assert claim["grade"] == "C" and claim["objective_support"] == 0
        from portrait_eval.core.models_v2 import DimensionId

        for context in primary.contexts + reviewer.contexts:
            assert all(
                set(asset["dimensions"]) == {dimension.value for dimension in DimensionId}
                for asset in context["iqa_evidence"]["assets"]
            )
        assert all(
            "highlight_clip_ratio" in data["whole"]
            for data in primary.contexts[1]["metrics"].values()
        )


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("transport", ["compatible", "responses"])
def test_r2_actual_http_payload_keeps_uint16_order_all_dimensions_and_encoded_hash(
    tmp_path, monkeypatch, channels, transport
):
    import hashlib
    import json

    import httpx

    from portrait_eval.core.models_v2 import DimensionId
    from portrait_eval.vlm import OpenAICompatibleVisionAdapter, OpenAIResponsesVisionAdapter

    paths = {}
    for code, level in (("A", 10000), ("B", 50000)):
        path = tmp_path / f"{code}.png"
        shape = (96, 96) if channels == 1 else (96, 96, 3)
        assert cv2.imwrite(str(path), np.full(shape, level, dtype=np.uint16))
        paths[code] = path
    context = anonymous_context(
        {
            "iqa_evidence": {
                "assets": [
                    {
                        "id": "high",
                        "state": "valid",
                        "dimensions": {
                            dimension.value: {"state": "unobservable"} for dimension in DimensionId
                        },
                    }
                ]
            },
            "metrics": {"high": {"whole": {"highlight_clip_ratio": 0.25, "luma_mean": 0.5}}},
            "note": r"high image at \\private-server\captures\image.png",
        },
        {"high": "A", "low": "B"},
    )
    captured = {}
    response_text = json.dumps(
        {"observations": [], "scores": {}, "hypotheses": [], "confidence": 0.5}
    )

    def handle(request):
        captured.update(json.loads(request.content))
        raw = (
            {"choices": [{"message": {"content": response_text}}]}
            if transport == "compatible"
            else {
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": response_text}]}
                ]
            }
        )
        return httpx.Response(200, json=raw)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr("portrait_eval.vlm.httpx.post", client.post)
        adapter = (
            OpenAICompatibleVisionAdapter("https://model.test", "probe", "primary")
            if transport == "compatible"
            else OpenAIResponsesVisionAdapter(
                "test-key", "probe", "primary", base_url="https://model.test"
            )
        )
        result = adapter.analyze("G001", paths, context)
    content = (captured["messages"] if transport == "compatible" else captured["input"])[0][
        "content"
    ]
    prompt = content[0]["text"]
    request_context = json.loads(
        prompt.split(
            "Objective context=" if transport == "compatible" else "Evaluation context=", 1
        )[1]
    )
    assert set(request_context["iqa_evidence"]["assets"][0]["dimensions"]) == {
        dimension.value for dimension in DimensionId
    }
    assert request_context["metrics"]["A"]["whole"]["highlight_clip_ratio"] == 0.25
    assert "private-server" not in prompt
    image_items = [item for item in content if item["type"] in {"image_url", "input_image"}]
    rendered_means = []
    for item, trace in zip(image_items, result.input_trace["images"], strict=True):
        url = item["image_url"]["url"] if transport == "compatible" else item["image_url"]
        encoded = base64.b64decode(url.split(",", 1)[1])
        assert trace["encoded_sha256"] == hashlib.sha256(encoded).hexdigest()
        with Image.open(BytesIO(encoded)) as image:
            rendered_means.append(float(np.asarray(image).mean()) / 255)
    assert rendered_means[0] < rendered_means[1] < 0.9
    assert result.input_trace["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()


@pytest.mark.parametrize("channels", [1, 3])
def test_r2_uint16_exif_orientation_matches_core_measurement_and_model_render(tmp_path, channels):
    import struct
    import zlib

    from qwen_tmqa.asset_io import load_comparison_asset
    from qwen_tmqa.comparison_models import ComparisonAsset

    from portrait_eval.imaging import load_rgb
    from portrait_eval.pixel_io import display_uint8

    pixels = np.tile(np.linspace(10000, 50000, 35, dtype=np.uint16), (21, 1))
    if channels == 3:
        pixels = np.repeat(pixels[..., None], 3, axis=2)
    path = tmp_path / f"oriented-{channels}.png"
    assert cv2.imwrite(str(path), pixels)
    exif = Image.Exif()
    exif[274] = 6
    exif_bytes = exif.tobytes()[6:]
    chunk = b"eXIf" + exif_bytes
    png = path.read_bytes()
    ihdr_end = 8 + 12 + struct.unpack(">I", png[8:12])[0]
    path.write_bytes(
        png[:ihdr_end]
        + struct.pack(">I", len(exif_bytes))
        + chunk
        + struct.pack(">I", zlib.crc32(chunk))
        + png[ihdr_end:]
    )
    core = load_comparison_asset(ComparisonAsset(id="fixture", path=path, encoding="srgb"))
    actual = load_rgb(path)
    assert actual.shape == core.pixels.shape == (35, 21, 3)
    assert np.array_equal(actual, core.pixels)
    assert core.trace["bit_depth"] == 16
    url = encode_image_data_url(path)
    with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
        rendered = np.asarray(image)
    assert rendered.shape == actual.shape
    assert np.mean(np.abs(rendered.astype(float) - display_uint8(actual).astype(float))) < 2


def test_r2_structural_key_collisions_never_change_scientific_vocabulary():
    context = {
        "metrics": {
            "real-device": {
                "whole": {"luma_mean": 0.5},
                "regions": {"face": {"mean_luminance": 0.25}},
            }
        },
        "dimensions": {"artifact_texture_control": {"state": "unobservable"}},
    }
    result = anonymous_context(
        context,
        {
            "real-device": "A",
            "face": "B",
            "whole": "C",
            "luma_mean": "D",
            "artifact_texture_control": "E",
        },
    )
    assert result["metrics"]["A"]["whole"]["luma_mean"] == 0.5
    assert result["metrics"]["A"]["regions"]["face"]["mean_luminance"] == 0.25
    assert "artifact_texture_control" in result["dimensions"]


def test_r2_exact_canonical_alias_does_not_change_reference_vocabulary():
    context = {
        "color": {"color_space": "srgb"},
        "dimension": "artifact_texture_control",
        "evidence_refs": [
            "metric:G001:real-device:luma_mean",
            "iqa:G001:real-device:artifact_texture_control:mean_luminance",
        ],
        "facts": {"real-device:face:1.mean_luminance": 0.3},
        "note": "camera color generated this capture",
    }
    result = anonymous_context(
        context,
        {
            "real-device": "A",
            "color": "B",
            "face": "C",
            "luma_mean": "D",
            "artifact_texture_control": "E",
            "mean_luminance": "F",
        },
    )
    assert result["color"]["color_space"] == "srgb"
    assert result["dimension"] == "artifact_texture_control"
    assert result["evidence_refs"] == [
        "metric:G001:A:luma_mean",
        "iqa:G001:A:artifact_texture_control:mean_luminance",
    ]
    assert result["facts"] == {"A:face:1.mean_luminance": 0.3}
    assert result["note"] == "camera B generated this capture"


def test_r2_serialized_schema_preserves_canonical_alias_fields():
    import json

    source = {
        "dimension": "artifact_texture_control",
        "evidence_refs": ["metric:G001:real-device:whole.luma_mean"],
        "metrics": {"real-device": {"whole": {"luma_mean": 0.5}}},
        "diagnostic_path": "/private/image.png",
    }
    result = json.loads(
        anonymous_context(
            json.dumps(source),
            {"real-device": "A", "luma_mean": "D", "artifact_texture_control": "E"},
        )
    )
    assert result["dimension"] == "artifact_texture_control"
    assert result["evidence_refs"] == ["metric:G001:A:whole.luma_mean"]
    assert result["metrics"]["A"]["whole"]["luma_mean"] == 0.5
    assert "diagnostic_path" not in result


def test_r2_device_display_names_a_b_do_not_swap_already_anonymous_metrics(tmp_path):
    from portrait_eval.database import Database
    from portrait_eval.models import ModelEvaluationResult
    from portrait_eval.pipeline import EvaluationPipeline
    from portrait_eval.vlm import VisionModelAdapter

    class Capture(VisionModelAdapter):
        def analyze(self, scene_id, image_paths, context):
            self.context = context
            return ModelEvaluationResult(
                scene_id=scene_id, role="capture", confidence=0.8, observations=[]
            )

    database = Database(f"sqlite:///{tmp_path / 'names.sqlite'}")
    database.create_all()
    adapter = Capture()
    with database.session_factory() as session:
        pipeline = EvaluationPipeline(session, tmp_path, primary=adapter, reviewer=adapter)
        pipeline.identities = {"real-device-x": "A", "real-device-y": "B", "A": "B", "B": "A"}
        pipeline._analyze(
            adapter,
            "G001",
            {"A": tmp_path / "first.png", "B": tmp_path / "second.png"},
            {"metrics": {"A": {"whole": {"luma_mean": 0.2}}, "B": {"whole": {"luma_mean": 0.8}}}},
        )
    assert adapter.context["metrics"] == {
        "A": {"whole": {"luma_mean": 0.2}},
        "B": {"whole": {"luma_mean": 0.8}},
    }


def test_r2_schema_enum_values_remain_protocol_values_for_colliding_aliases():
    result = anonymous_context(
        {
            "mode": "device",
            "role": "primary",
            "encoding": "srgb",
            "state": "measured_proxy",
            "person_correspondence": "unverified",
            "note": "camera primary produced this capture",
        },
        {"device": "A", "primary": "B", "srgb": "C", "measured_proxy": "D", "unverified": "E"},
    )
    assert result == {
        "mode": "device",
        "role": "primary",
        "encoding": "srgb",
        "state": "measured_proxy",
        "person_correspondence": "unverified",
        "note": "camera B produced this capture",
    }
