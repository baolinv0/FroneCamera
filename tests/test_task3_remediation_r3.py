"""Residual R2-02: scientific schemas survive arbitrary device display aliases."""

from __future__ import annotations

import copy
import hashlib
import json

import httpx
import numpy as np
from PIL import Image

from portrait_eval.database import Database
from portrait_eval.iqa_bridge import model_evidence_context
from portrait_eval.model_validation import anonymous_context
from portrait_eval.models import ModelEvaluationResult
from portrait_eval.pipeline import EvaluationPipeline
from portrait_eval.repository import Repository
from portrait_eval.vlm import (
    OpenAICompatibleVisionAdapter,
    OpenAIResponsesVisionAdapter,
    VisionModelAdapter,
)


def test_r3_actual_two_face_sqlite_context_retains_all_core_fact_key_aliases(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "portrait_eval.imaging.detect_faces", lambda image: [[8, 12, 32, 42], [52, 12, 32, 42]]
    )

    class Capture(VisionModelAdapter):
        def __init__(self, model):
            self.model = model
            self.contexts = []

        def analyze(self, scene_id, image_paths, context):
            self.contexts.append(context)
            return ModelEvaluationResult(
                scene_id=scene_id, role="capture", confidence=0.8, observations=[]
            )

    database = Database(f"sqlite:///{tmp_path / 'fact-alias.sqlite'}")
    database.create_all()
    with database.session_factory() as session:
        repo = Repository(session)
        project = repo.create_project("two-face schema alias probe")
        for name, level in (("person_count", 100), ("face_luminance_spread", 180)):
            folder = tmp_path / name
            folder.mkdir()
            Image.fromarray(np.full((96, 96, 3), level, dtype=np.uint8)).save(folder / "1.png")
            repo.add_device(project.id, name, str(folder))
        pairing = repo.scan_and_pair(project.id)
        repo.confirm_pairing(project.id, pairing["version"])
        primary, reviewer = Capture("primary"), Capture("reviewer")
        EvaluationPipeline(session, tmp_path / "workspace", primary=primary, reviewer=reviewer).run(
            project.id
        )
        core = repo.list_analysis(project.id, "iqa_evaluation")[0]["payload"]
        for asset in core["assets"]:
            assert asset["dimensions"]["multi_face_consistency"]["facts"] == {
                "person_count": 2,
                "face_luminance_spread": 0,
            }
        validation_contexts = [
            context
            for adapter in (primary, reviewer)
            for context in adapter.contexts
            if context["pass"] in {"metric_validation", "challenge"}
        ]
        for context in validation_contexts:
            for asset in context["iqa_evidence"]["assets"]:
                assert asset["dimensions"]["multi_face_consistency"]["facts"] == {
                    "person_count": 2,
                    "face_luminance_spread": 0,
                }

        # Discover current fact vocabulary from the actual core, not a second
        # hand-maintained list. Every fact leaf can also be a display alias.
        aliases = {
            key.split(":")[-1].split(".")[-1]
            for asset in core["assets"]
            for dimension in asset["dimensions"].values()
            for key in dimension["facts"]
        } | {key for asset in core["assets"] for key in asset["objective"]}
        assert {"person_count", "face_luminance_spread"} <= aliases
        source = model_evidence_context(core, measurements=True)
        base_identities = {
            asset["id"]: chr(ord("A") + index) for index, asset in enumerate(core["assets"])
        }
        expected = anonymous_context(source, base_identities)
        for alias in sorted(aliases):
            identities = {**base_identities, alias: "ALIAS"}
            assert anonymous_context(source, identities) == expected, alias
            assert json.loads(anonymous_context(json.dumps(source), identities)) == expected, alias

        # Inspect both actual provider encoders, including their final privacy
        # pass and stored hash of the exact prompt sent across HTTP.
        identities = {**base_identities, **{alias: "ALIAS" for alias in aliases}}
        request_context = anonymous_context(source, identities)
        request_context["note"] = r"capture at \\private-server\captures\image.png"
        expected_context = anonymous_context(request_context)
        image_paths = {
            "A": tmp_path / "person_count" / "1.png",
            "B": tmp_path / "face_luminance_spread" / "1.png",
        }
        for transport in ("compatible", "responses"):
            captured = {}
            response_text = json.dumps(
                {"observations": [], "scores": {}, "hypotheses": [], "confidence": 0.5}
            )

            def handle(
                request, captured=captured, response_text=response_text, transport=transport
            ):
                captured.update(json.loads(request.content))
                raw = (
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
                result = adapter.analyze("G001", image_paths, request_context)
            content = (captured["messages"] if transport == "compatible" else captured["input"])[0][
                "content"
            ]
            prompt = content[0]["text"]
            actual_context = json.loads(
                prompt.split(
                    "Objective context=" if transport == "compatible" else "Evaluation context=", 1
                )[1]
            )
            allowed_refs = actual_context.pop("allowed_evidence_refs")
            assert {"asset:G001:A", "asset:G001:B"} <= set(allowed_refs)
            assert actual_context == expected_context
            assert "private-server" not in prompt
            assert (
                result.input_trace["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
            )


def test_r3_future_scientific_fields_are_preserved_by_context_not_vocabulary():
    context = {
        "metrics": {
            "whole": {
                "whole": {"future_metric": 0.5},
                "regions": {"future_region": {"future_metric": 0.25}},
            }
        },
        "iqa_evidence": {
            "assets": [
                {
                    "id": "whole",
                    "objective": {"future_objective": 0.5},
                    "dimensions": {
                        "future_dimension": {
                            "state": "measured_proxy",
                            "facts": {
                                "future_fact": {"future_nested_fact": 2},
                                "whole:face:0.future_fact": 0.2,
                                "metrics": {"future_metric": 0.3},
                            },
                        }
                    },
                }
            ],
        },
        "devices": {"whole": {"device_id": "whole", "name": "whole"}},
        "note": r"whole capture at \\private-server\captures\image.png",
        "diagnostic_path": "/private/image.png",
    }
    expected = copy.deepcopy(context)
    expected["metrics"]["A"] = expected["metrics"].pop("whole")
    asset = expected["iqa_evidence"]["assets"][0]
    asset["id"] = "A"
    facts = asset["dimensions"]["future_dimension"]["facts"]
    facts["A:face:0.future_fact"] = facts.pop("whole:face:0.future_fact")
    expected["devices"] = {"A": {"device_id": "A", "name": "A"}}
    expected["note"] = "A capture at [location omitted]"
    del expected["diagnostic_path"]
    aliases = {
        "whole": "A",
        "future_metric": "B",
        "future_region": "C",
        "future_objective": "D",
        "future_dimension": "E",
        "future_fact": "F",
        "future_nested_fact": "G",
    }
    assert anonymous_context(context, aliases) == expected
    assert json.loads(anonymous_context(json.dumps(context), aliases)) == expected
