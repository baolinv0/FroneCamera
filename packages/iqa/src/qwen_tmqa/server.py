from __future__ import annotations

import json
import posixpath
import threading
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, unquote, urlencode, urlparse

from pydantic import ValidationError

from .domain import HumanReview
from .review import append_exposure, append_review, load_exposures, load_reviews

ServerMode = Literal["reviewer", "engineering"]


def _canonical_path(raw_path: str) -> str:
    decoded = unquote(urlparse(raw_path).path).replace("\\", "/")
    normalized = posixpath.normpath("/" + decoded.lstrip("/"))
    return normalized if normalized.startswith("/") else "/" + normalized


class ReviewRequestHandler(SimpleHTTPRequestHandler):
    reviews_path: Path
    dashboard_dir: Path
    allowed_scene_ids: frozenset[str] | None = None
    reveal_payload: dict[str, object]
    reviewer_payload: dict[str, object]
    mode: ServerMode = "reviewer"
    storage_lock: threading.RLock

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, path: str) -> None:
        self.path = path
        super().do_GET()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        canonical = _canonical_path(self.path)

        if canonical == "/api/reviewer-payload":
            self._send_json(200, self.reviewer_payload)
            return
        if canonical == "/api/reveal":
            query = parse_qs(parsed.query)
            scene_id = query.get("scene_id", [""])[0]
            review_id = query.get("review_id", [""])[0]
            try:
                with self.storage_lock:
                    matches = [
                        review
                        for review in load_reviews(self.reviews_path)
                        if review.scene_id == scene_id and review.review_id == review_id
                    ]
                    if len(matches) != 1:
                        self._send_json(403, {"error": "review has not been persisted uniquely"})
                        return
                    reveal = self.reveal_payload.get(scene_id)
                    if reveal is None:
                        self._send_json(404, {"error": "reveal evidence not found"})
                        return
                    review = matches[0]
                    if (scene_id, review.reviewer_id) not in load_exposures(self.reviews_path):
                        append_exposure(self.reviews_path, review)
            except (OSError, ValueError) as exc:
                self._send_json(500, {"error": f"exposure persistence failed: {exc}"})
                return
            self._send_json(200, reveal)
            return

        if self.mode == "reviewer":
            if canonical in {"/", "/review", "/review.html"}:
                self._serve_static("/review.html")
                return
            if canonical.startswith("/assets/"):
                self._serve_static(canonical)
                return
            self._send_json(403, {"error": "route is unavailable in reviewer mode"})
            return

        if canonical == "/":
            self._serve_static("/index.html")
            return
        if canonical == "/data/reveal_payload.json":
            self._send_json(403, {"error": "reveal evidence is API-only"})
            return
        self._serve_static(canonical)

    def do_POST(self) -> None:
        if _canonical_path(self.path) != "/api/reviews":
            self.send_error(404)
            return
        try:
            if "application/json" not in self.headers.get("Content-Type", ""):
                raise ValueError("Content-Type must be application/json")
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("invalid content length")
            raw = self.rfile.read(length)
            review = HumanReview.model_validate_json(raw)
            if self.mode == "engineering" and not review.synthetic:
                self._send_json(
                    403, {"error": "engineering evidence cannot create real blind gold"}
                )
                return
            if self.allowed_scene_ids is not None and review.scene_id not in self.allowed_scene_ids:
                self._send_json(403, {"error": "scene is not in the focused review queue"})
                return
            review = review.model_copy(
                update={
                    "gold_eligible": True,
                    "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            )
            try:
                with self.storage_lock:
                    if any(
                        previous.review_id == review.review_id
                        for previous in load_reviews(self.reviews_path)
                    ):
                        self._send_json(409, {"error": "review_id already exists"})
                        return
                    if (review.scene_id, review.reviewer_id) in load_exposures(self.reviews_path):
                        self._send_json(
                            409, {"error": "reviewer has revealed this scene; blind gold is frozen"}
                        )
                        return
                    append_review(self.reviews_path, review)
            except (OSError, ValueError) as exc:
                self._send_json(500, {"error": f"review persistence failed: {exc}"})
                return
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(422, {"error": str(exc)})
            return
        except OSError as exc:
            self._send_json(500, {"error": f"review persistence failed: {exc}"})
            return

        reveal_query = urlencode({"scene_id": review.scene_id, "review_id": review.review_id})
        self._send_json(
            201,
            {
                "status": "created",
                "review_id": review.review_id,
                "reveal_url": f"/api/reveal?{reveal_query}",
            },
        )

    def log_message(self, format: str, *args: object) -> None:
        return


def create_server(
    dashboard_dir: Path,
    reviews_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    mode: ServerMode = "reviewer",
) -> ThreadingHTTPServer:
    if mode not in {"reviewer", "engineering"}:
        raise ValueError(f"unsupported server mode: {mode}")
    required_files = [
        dashboard_dir / "index.html",
        dashboard_dir / "review.html",
        dashboard_dir / "data" / "review_queue.json",
        dashboard_dir / "data" / "reviewer_payload.json",
        dashboard_dir / "data" / "reveal_payload.json",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        if any(path.endswith("review_queue.json") for path in missing):
            raise FileNotFoundError(f"focused review queue not found: {missing}")
        raise FileNotFoundError(f"dashboard review artifacts not found: {missing}")

    queue_path = dashboard_dir / "data" / "review_queue.json"
    queue_data = json.loads(queue_path.read_text(encoding="utf-8"))
    if not isinstance(queue_data, list) or not all(
        isinstance(scene_id, str) for scene_id in queue_data
    ):
        raise ValueError(f"invalid review queue: {queue_path}")
    allowed_scene_ids = frozenset(queue_data)
    reviewer_payload = json.loads(
        (dashboard_dir / "data" / "reviewer_payload.json").read_text(encoding="utf-8")
    )
    reveal_payload = json.loads(
        (dashboard_dir / "data" / "reveal_payload.json").read_text(encoding="utf-8")
    )

    storage_lock = threading.RLock()

    class BoundReviewRequestHandler(ReviewRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.storage_lock = storage_lock
            self.reviews_path = reviews_path
            self.dashboard_dir = dashboard_dir
            self.allowed_scene_ids = allowed_scene_ids
            self.reviewer_payload = reviewer_payload
            self.reveal_payload = reveal_payload
            self.mode = mode
            super().__init__(*args, directory=str(dashboard_dir), **kwargs)

    return ThreadingHTTPServer((host, port), BoundReviewRequestHandler)
