import json
from base64 import b64decode
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from portrait_eval.vlm import (
    OpenAIResponsesVisionAdapter,
    build_vision_adapters,
    encode_image_data_url,
    extract_responses_output_text,
    parse_json_object,
    resolve_openai_api_key,
)


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


def test_extract_responses_output_text_reads_assistant_message() -> None:
    payload = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": '{"confidence": 0.9}'}],
            }
        ]
    }
    assert extract_responses_output_text(payload) == '{"confidence": 0.9}'


def test_openai_responses_adapter_sends_images_and_parses_result(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (80, 60), "gray").save(first)
    Image.new("RGB", (80, 60), "white").save(second)
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            result = {
                "observations": [
                    {
                        "device_id": "A",
                        "dimension": "face_exposure_readability",
                        "statement": "Device A renders the face brighter.",
                        "evidence_refs": ["asset:scene-1:A"],
                        "certainty": 0.8,
                    }
                ],
                "scores": {},
                "hypotheses": [],
                "confidence": 0.8,
            }
            return {
                "id": "resp_test",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": json.dumps(result)}],
                    }
                ],
            }

    def fake_post(url, *, headers, json, timeout):  # type: ignore[no-untyped-def]
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr("portrait_eval.vlm.httpx.post", fake_post)
    adapter = OpenAIResponsesVisionAdapter("test-key", "gpt-test", "primary")
    result = adapter.analyze("scene-1", {"A": first, "B": second}, {"pass": "visual"})

    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer test-key"  # type: ignore[index]
    request = captured["payload"]
    assert request["model"] == "gpt-test"  # type: ignore[index]
    content = request["input"][0]["content"]  # type: ignore[index]
    assert [item["type"] for item in content].count("input_image") == 2
    assert all(
        item["image_url"].startswith("data:image/jpeg;base64,")
        for item in content
        if item["type"] == "input_image"
    )
    assert result.scene_id == "scene-1"
    assert result.role == "primary"
    assert result.observations[0].device_id == "A"


def test_build_vision_adapters_selects_openai_branch() -> None:
    settings = SimpleNamespace(
        vlm_provider="openai",
        openai_api_key="test-key",
        openai_base_url="https://api.openai.com",
        openai_primary_model="gpt-primary",
        openai_reviewer_model="gpt-reviewer",
        openai_max_output_tokens=2000,
        vlm_max_image_edge=768,
        primary_vlm_url=None,
        primary_vlm_model="unused-primary",
        reviewer_vlm_url=None,
        reviewer_vlm_model="unused-reviewer",
    )
    primary, reviewer = build_vision_adapters(settings)
    assert isinstance(primary, OpenAIResponsesVisionAdapter)
    assert isinstance(reviewer, OpenAIResponsesVisionAdapter)
    assert primary.model == "gpt-primary"
    assert reviewer.model == "gpt-reviewer"


def test_resolve_openai_api_key_reads_only_api_key_field(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "OPENAI_API_KEY": "sk-test-from-auth-file",
                "tokens": {"access_token": "must-not-be-used"},
            }
        ),
        encoding="utf-8",
    )

    assert resolve_openai_api_key(auth_file=auth_file) == "sk-test-from-auth-file"


def test_build_vision_adapters_requires_openai_api_key(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = SimpleNamespace(
        vlm_provider="openai",
        openai_api_key=None,
        openai_auth_file=tmp_path / "missing-auth.json",
    )

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_vision_adapters(settings)
