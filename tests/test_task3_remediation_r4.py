"""R2-02 residual: model scores are dimension-indexed scientific schema."""

from __future__ import annotations

import hashlib
import json

import httpx
import numpy as np
import pytest
from PIL import Image

from portrait_eval.database import Database
from portrait_eval.model_validation import anonymous_context
from portrait_eval.models import KNOWN_DIMENSIONS, ModelEvaluationResult
from portrait_eval.pipeline import EvaluationPipeline
from portrait_eval.repository import Repository
from portrait_eval.vlm import OpenAICompatibleVisionAdapter, OpenAIResponsesVisionAdapter


@pytest.mark.parametrize("transport", ["compatible", "responses"])
def test_r4_actual_sqlite_reviewer_http_preserves_all_dimension_scores(
    tmp_path, monkeypatch, transport
):
    monkeypatch.setattr("portrait_eval.imaging.detect_faces", lambda image: [[12, 12, 45, 55]])
    scores = dict.fromkeys(sorted(KNOWN_DIMENSIONS), 68)
    response_text = json.dumps(
        {"observations": [], "scores": scores, "hypotheses": [], "confidence": 0.8}
    )
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        response = (
            {"choices": [{"message": {"content": response_text}}]}
            if transport == "compatible"
            else {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": response_text}],
                    }
                ]
            }
        )
        return httpx.Response(200, json=response)

    def adapter(model, role):
        return (
            OpenAICompatibleVisionAdapter("https://model.test", model, role)
            if transport == "compatible"
            else OpenAIResponsesVisionAdapter(
                "private-api-token", model, role, base_url="https://model.test"
            )
        )

    database = Database(f"sqlite:///{tmp_path / 'scores.sqlite'}")
    database.create_all()
    with database.session_factory() as session:
        repo = Repository(session)
        project = repo.create_project("all score-dimension display aliases")
        for index, dimension in enumerate(sorted(KNOWN_DIMENSIONS)):
            folder = tmp_path / dimension
            folder.mkdir()
            Image.fromarray(np.full((96, 96, 3), 100 + index, dtype=np.uint8)).save(
                folder / "1.png"
            )
            repo.add_device(project.id, dimension, str(folder))
        pairing = repo.scan_and_pair(project.id)
        repo.confirm_pairing(project.id, pairing["version"])
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            monkeypatch.setattr("portrait_eval.vlm.httpx.post", client.post)
            EvaluationPipeline(
                session,
                tmp_path / "workspace",
                primary=adapter("primary-model", "primary"),
                reviewer=adapter("reviewer-model", "reviewer"),
            ).run(project.id)
        assert len(requests) == 4
        contexts, prompts = [], []
        for payload in requests:
            content = (payload["messages"] if transport == "compatible" else payload["input"])[0][
                "content"
            ]
            prompt = content[0]["text"]
            prompts.append(prompt)
            contexts.append(
                json.loads(
                    prompt.split(
                        "Objective context="
                        if transport == "compatible"
                        else "Evaluation context=",
                        1,
                    )[1]
                )
            )
            assert "private-api-token" not in prompt
            assert str(tmp_path) not in prompt
        assert contexts[1]["visual_observation"]["scores"] == scores
        assert contexts[3]["primary"]["scores"] == scores
        assert contexts[3]["reviewer_independent"]["scores"] == scores
        evaluations = [
            result
            for kind in (
                "primary_vlm_visual",
                "primary_vlm",
                "reviewer_independent",
                "reviewer_challenge",
            )
            for result in repo.list_analysis(project.id, kind)
        ]
        assert len(evaluations) == 4
        for evaluation in evaluations:
            result = evaluation["payload"]
            assert result["scores"] == scores
            assert result["input_trace"]["prompt_sha256"] in {
                hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts
            }


def test_r4_non_provider_scores_and_untyped_backend_maps_are_scientific_not_device_maps():
    scores = dict.fromkeys(sorted(KNOWN_DIMENSIONS), 68)
    model = ModelEvaluationResult(
        scene_id="G001",
        role="primary",
        observations=[],
        scores=scores,
        confidence=0.8,
        raw={
            "mapping": scores,
            "metrics": scores,
            "devices": scores,
            "diagnostic_path": "/private/capture/image.png",
        },
    ).model_dump(mode="json")
    aliases = {dimension: f"CAM{index}" for index, dimension in enumerate(sorted(KNOWN_DIMENSIONS))}
    for source in (model, {"primary": model}, {"scores": scores}):
        result = anonymous_context(source, aliases)
        payload = result.get("primary", result)
        assert payload["scores"] == scores
        if "raw" in payload:
            assert payload["raw"] == {"mapping": scores, "metrics": scores, "devices": scores}
        assert json.loads(anonymous_context(json.dumps(source), aliases)) == result
        assert "/private" not in json.dumps(result)
    for field in ("metrics", "mapping", "devices"):
        # Backend/standalone numeric schemas do not satisfy declared device-map shapes.
        source = {field: scores}
        assert anonymous_context(source, aliases) == source
        assert json.loads(anonymous_context(json.dumps(source), aliases)) == source

    # Actual domain/request device maps still replace their declared key component.
    assert anonymous_context(
        {"mapping": {"skin_awb": "A"}, "metrics": {"skin_awb": {"whole": {"luma_mean": 0.5}}}},
        {"skin_awb": "A", "luma_mean": "B"},
    ) == {"mapping": {"A": "A"}, "metrics": {"A": {"whole": {"luma_mean": 0.5}}}}
