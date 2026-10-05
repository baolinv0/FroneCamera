from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import quote

import pytest
from test_review_server_isolation import _dashboard

from qwen_tmqa.domain import HumanReview
from qwen_tmqa.review import load_exposures, load_reviews
from qwen_tmqa.server import create_server


@contextmanager
def _serving(dashboard: Path, reviews: Path, mode: str = "reviewer"):
    server = create_server(dashboard, reviews, port=0, mode=mode)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _request(port: int, path: str, method: str = "GET", body: bytes | None = None):
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_reviewer_cannot_read_protected_static_files(method: str, tmp_path: Path) -> None:
    dashboard, _ = _dashboard(tmp_path)
    with _serving(dashboard, tmp_path / "reviews.jsonl") as port:
        for path in [
            "/index.html",
            "/data/reveal_payload.json",
            "/data/engineering_evaluations.json",
            "/data/%72eveal_payload.json?cache=1",
            "/assets/../data/reveal_payload.json",
            "/assets/%2e%2e/data/engineering_evaluations.json",
        ]:
            status, _, body = _request(port, path, method)
            assert status == 403, (method, path, status)
            if method == "HEAD":
                assert body == b""


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_double_encoded_asset_traversal_cannot_reveal_evidence(
    method: str, tmp_path: Path
) -> None:
    dashboard, _ = _dashboard(tmp_path)
    with _serving(dashboard, tmp_path / "reviews.jsonl") as port:
        for path in [
            "/assets/%252e%252e/data/reveal_payload.json",
            "/assets/%252e%252e/data/engineering_evaluations.json",
            "/assets/%252e%252e/index.html",
            "/assets/%252e%252e%252fdata%252freveal_payload.json?cache=1",
        ]:
            status, _, body = _request(port, path, method)
            assert status in {403, 404}, (method, path, status)
            assert b"model_evaluations" not in body
            if method == "HEAD":
                assert body == b""


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_engineering_static_reveal_remains_api_only(method: str, tmp_path: Path) -> None:
    dashboard, _ = _dashboard(tmp_path)
    with _serving(dashboard, tmp_path / "reviews.jsonl", "engineering") as port:
        for path in [
            "/data/reveal_payload.json",
            "/data/%72eveal_payload.json?cache=1",
            "/assets/%252e%252e/data/reveal_payload.json",
            "/assets/%252e%252e%252fdata%252freveal_payload.json",
        ]:
            status, _, body = _request(port, path, method)
            assert status in {403, 404}, (method, path, status)
            assert b"model_evaluations" not in body


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_reviewer_symlinks_cannot_expand_the_blind_static_surface(
    method: str, tmp_path: Path
) -> None:
    dashboard, _ = _dashboard(tmp_path)
    assets = dashboard / "assets"
    outside = tmp_path / "outside.txt"
    outside.write_text("outside dashboard", encoding="utf-8")
    (assets / "reveal.json").symlink_to(dashboard / "data" / "reveal_payload.json")
    (assets / "engineering.json").symlink_to(
        dashboard / "data" / "engineering_evaluations.json"
    )
    (assets / "data").symlink_to(dashboard / "data", target_is_directory=True)
    (assets / "outside.txt").symlink_to(outside)
    (assets / "navigation").mkdir()
    (assets / "navigation" / "index.html").symlink_to(
        dashboard / "data" / "reveal_payload.json"
    )
    (dashboard / "review.html").unlink()
    (dashboard / "review.html").symlink_to(dashboard / "data" / "reveal_payload.json")

    with _serving(dashboard, tmp_path / "reviews.jsonl") as port:
        for path in [
            "/assets/reveal.json",
            "/assets/engineering.json",
            "/assets/data/engineering_evaluations.json",
            "/assets/outside.txt",
            "/assets/navigation/",
            "/review.html",
            "/",
        ]:
            status, _, body = _request(port, path, method)
            assert status == 403, (method, path, status)
            assert b"model_evaluations" not in body
            assert b"outside dashboard" not in body


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_reviewer_assets_directory_cannot_alias_protected_data(
    method: str, tmp_path: Path
) -> None:
    dashboard, _ = _dashboard(tmp_path)
    (dashboard / "assets").rename(dashboard / "original-assets")
    (dashboard / "assets").symlink_to(dashboard / "data", target_is_directory=True)
    with _serving(dashboard, tmp_path / "reviews.jsonl") as port:
        for path in ["/assets/reveal_payload.json", "/assets/engineering_evaluations.json"]:
            status, _, _ = _request(port, path, method)
            assert status == 403, (method, path, status)


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_engineering_symlinks_cannot_escape_dashboard_or_serve_reveal(
    method: str, tmp_path: Path
) -> None:
    dashboard, _ = _dashboard(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside dashboard", encoding="utf-8")
    (dashboard / "assets" / "outside.txt").symlink_to(outside)
    (dashboard / "assets" / "reveal.json").symlink_to(
        dashboard / "data" / "reveal_payload.json"
    )
    (dashboard / "public").mkdir()
    (dashboard / "public" / "index.html").symlink_to(
        dashboard / "data" / "reveal_payload.json"
    )
    (dashboard / "index.html").unlink()
    (dashboard / "index.html").symlink_to(dashboard / "data" / "reveal_payload.json")
    with _serving(dashboard, tmp_path / "reviews.jsonl", "engineering") as port:
        for path in ["/assets/outside.txt", "/assets/reveal.json", "/public/", "/"]:
            status, _, body = _request(port, path, method)
            assert status == 403, (method, path, status)
            assert b"model_evaluations" not in body
            assert b"outside dashboard" not in body


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("mode", ["reviewer", "engineering"])
def test_static_pages_and_asset_filenames_keep_normal_url_semantics(
    method: str, mode: str, tmp_path: Path
) -> None:
    dashboard, _ = _dashboard(tmp_path)
    assets = dashboard / "assets"
    image = next(assets.rglob("*.jpg")).read_bytes()
    paths = {}
    for name in ["50% scene.jpg", "literal%2e%2e.jpg", "hash#query?.jpg", "示例.jpg"]:
        asset = assets / name
        asset.write_bytes(image)
        paths["/assets/" + quote(name) + "?cache=%25fresh"] = image
    literal_directory = assets / "%2e%2e"
    literal_directory.mkdir()
    (literal_directory / "image.jpg").write_bytes(image)
    paths["/assets/%252e%252e/image.jpg"] = image
    (assets / "safe-link.jpg").symlink_to(next((assets / "images").rglob("*.jpg")))
    paths["/assets/safe-link.jpg"] = image
    paths["/review.html"] = (dashboard / "review.html").read_bytes()
    if mode == "reviewer":
        paths["/"] = paths["/review.html"]
        paths["/review"] = paths["/review.html"]
    else:
        paths["/"] = (dashboard / "index.html").read_bytes()
        paths["/data/engineering_evaluations.json"] = (
            dashboard / "data" / "engineering_evaluations.json"
        ).read_bytes()

    with _serving(dashboard, tmp_path / "reviews.jsonl", mode) as port:
        for path, expected in paths.items():
            status, headers, body = _request(port, path, method)
            assert status == 200, (mode, method, path, status)
            assert int(headers["Content-Length"]) == len(expected)
            assert body == (b"" if method == "HEAD" else expected)


def test_blocked_static_attempt_preserves_blind_review_and_persisted_api_reveal(
    tmp_path: Path,
) -> None:
    dashboard, scene_id = _dashboard(tmp_path)
    reviews = tmp_path / "reviews.jsonl"
    review = HumanReview(
        review_id="blind-first",
        scene_id=scene_id,
        reviewer_id="human",
        blind_review=True,
        decision="KEEP",
        scores={"overall": 0.8},
        confidence=0.9,
    )
    with _serving(dashboard, reviews) as port:
        status, _, _ = _request(port, "/assets/%252e%252e/data/reveal_payload.json")
        assert status in {403, 404}
        assert load_exposures(reviews) == set()
        status, _, _ = _request(port, f"/api/reveal?scene_id={quote(scene_id)}&review_id=missing")
        assert status == 403
        status, _, body = _request(port, "/api/reviews", "POST", review.model_dump_json().encode())
        assert status == 201
        assert load_reviews(reviews)[0].gold_eligible
        reveal_url = json.loads(body)["reveal_url"]
        status, headers, body = _request(port, reveal_url, "HEAD")
        assert status == 405
        assert headers["Allow"] == "GET"
        assert body == b""
        assert load_exposures(reviews) == set()
        status, _, body = _request(port, reveal_url)
        assert status == 200
        assert json.loads(body)["scene_id"] == scene_id
        assert load_exposures(reviews) == {(scene_id, "human")}
        updated = review.model_copy(update={"review_id": "after-reveal"})
        status, _, _ = _request(port, "/api/reviews", "POST", updated.model_dump_json().encode())
        assert status == 409
