import json
import threading
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_review_calibration import _scene

from qwen_tmqa.domain import HumanReview
from qwen_tmqa.review import calibrate_models, load_reviews
from qwen_tmqa.server import create_server


@contextmanager
def serving(tmp_path, mode="reviewer"):
    dashboard = tmp_path / "dashboard"
    (dashboard / "data").mkdir(parents=True, exist_ok=True)
    for name in ["index.html", "review.html"]:
        (dashboard / name).write_text("review")
    for name, payload in [
        ("review_queue.json", ["s1"]),
        ("reviewer_payload.json", {}),
        ("reveal_payload.json", {"s1": {"secret": "model evidence"}}),
    ]:
        (dashboard / "data" / name).write_text(json.dumps(payload))
    server = create_server(dashboard, tmp_path / "reviews.jsonl", port=0, mode=mode)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def post(base, review_id, score=0.8, reviewer="human"):
    review = HumanReview(
        review_id=review_id,
        scene_id="s1",
        reviewer_id=reviewer,
        blind_review=True,
        decision="KEEP",
        scores={"overall": score},
        confidence=0.9,
    )
    request = Request(
        base + "/api/reviews",
        data=review.model_dump_json().encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request) as response:
        return json.load(response)


def test_pre_reveal_corrections_remain_gold_but_exposed_edits_are_rejected_after_restart(tmp_path):
    with serving(tmp_path) as base:
        post(base, "old", 0.0)
        correction = post(base, "corrected", 0.8)
        with urlopen(base + correction["reveal_url"]) as response:
            assert json.load(response)["secret"] == "model evidence"
        with pytest.raises(HTTPError) as error:
            post(base, "after-exposure", 0.1)
        assert error.value.code == 409
    with serving(tmp_path) as base:
        with pytest.raises(HTTPError) as error:
            post(base, "after-restart", 0.2)
        assert error.value.code == 409
        post(base, "independent-human", 0.6, reviewer="another-human")
    reviews = load_reviews(tmp_path / "reviews.jsonl")
    assert [review.review_id for review in reviews] == ["old", "corrected", "independent-human"]
    models = {
        m.model_id: m for m in calibrate_models([_scene("s1", 0.8, 1, [], (0.8, 0.6))], reviews)
    }
    assert models["m1"].sample_count == 2
    assert models["m1"].overall_mae == pytest.approx(0.1)


def test_failed_exposure_persistence_never_reveals_evidence(tmp_path):
    with serving(tmp_path) as base:
        created = post(base, "first")
        (tmp_path / "reviews.jsonl.exposure.jsonl").mkdir()
        with pytest.raises(HTTPError) as error:
            urlopen(base + created["reveal_url"])
        assert error.value.code == 500
        assert "secret" not in error.value.read().decode()


def test_duplicate_review_ids_cannot_impersonate_a_persisted_submission(tmp_path):
    with serving(tmp_path) as base:
        post(base, "same-id")
        with pytest.raises(HTTPError) as error:
            post(base, "same-id", reviewer="different-human")
        assert error.value.code == 409


def test_review_storage_failure_returns_http_500(tmp_path):
    with serving(tmp_path) as base:
        (tmp_path / "reviews.jsonl").mkdir()
        with pytest.raises(HTTPError) as error:
            post(base, "first")
        assert error.value.code == 500


def test_append_only_history_loaded_after_exposure_cannot_replace_blind_gold(tmp_path):
    with serving(tmp_path) as base:
        post(base, "old", 0.0)
        post(base, "corrected", 0.8)
        # Revealing an older submission still freezes the latest blind correction.
        with urlopen(base + "/api/reveal?scene_id=s1&review_id=old"):
            pass
    exposed = HumanReview(
        review_id="imported-exposed-edit",
        scene_id="s1",
        reviewer_id="human",
        blind_review=True,
        decision="REJECT",
        scores={"overall": 0.0},
        confidence=1,
        received_at="2099-01-01T00:00:00Z",
    )
    with (tmp_path / "reviews.jsonl").open("a") as handle:
        handle.write(exposed.model_dump_json() + "\n")
    reviews = load_reviews(tmp_path / "reviews.jsonl")
    assert len(reviews) == 3  # History is preserved, never rewritten.
    models = {
        m.model_id: m for m in calibrate_models([_scene("s1", 0.8, 1, [], (0.8, 0.6))], reviews)
    }
    assert models["m1"].overall_mae == pytest.approx(0)
    assert models["m1"].sample_count == 1


def test_corrupt_exposure_store_fails_closed_without_new_gold(tmp_path):
    with serving(tmp_path) as base:
        post(base, "first")
        (tmp_path / "reviews.jsonl.exposure.jsonl").write_text("{broken json\n")
        with pytest.raises(HTTPError) as error:
            post(base, "second")
        assert error.value.code == 500
    assert len((tmp_path / "reviews.jsonl").read_text().splitlines()) == 1


def test_engineering_surface_cannot_create_blind_real_gold(tmp_path):
    with serving(tmp_path, mode="engineering") as base:
        with pytest.raises(HTTPError) as error:
            post(base, "engineering-viewer")
        assert error.value.code == 403
    assert not (tmp_path / "reviews.jsonl").exists()


@pytest.mark.parametrize("field", ["review_id", "scene_id"])
def test_empty_review_identity_cannot_poison_durable_exposure(field):
    payload = {
        "review_id": "review",
        "scene_id": "scene",
        "reviewer_id": "human",
        "blind_review": True,
        "decision": "KEEP",
        "scores": {"overall": 0.8},
        "confidence": 1,
    }
    payload[field] = " "
    with pytest.raises(ValueError):
        HumanReview.model_validate(payload)
