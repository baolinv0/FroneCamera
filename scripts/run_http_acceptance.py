"""Exercise the documented queued HTTP workflow with actual API and worker processes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Fresh directory for fixtures and evidence")
    args = parser.parse_args()
    if args.output is None:
        root = Path(tempfile.mkdtemp(prefix="fronecamera-http-acceptance-"))
    else:
        root = args.output.resolve()
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            parser.error("--output must be an empty directory")
    artifacts = root / "evidence"
    artifacts.mkdir()
    device_root = root / "devices"
    hashes = {}
    for device in range(4):
        folder = device_root / f"device-{device}"
        folder.mkdir(parents=True)
        for scene in range(1, 7):
            if device == 3 and scene == 3:
                continue
            y, x = np.mgrid[:64, :96]
            base = np.clip(25 + x + y + scene * 8 + device * 5, 0, 255).astype(np.uint8)
            rgb = np.stack([base, (base * 0.9).astype(np.uint8), (base * 0.8).astype(np.uint8)], -1)
            path = folder / f"{scene:02}.png"
            Image.fromarray(rgb).save(path)
            hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    env = os.environ.copy()
    env.update(
        {
            "PORTRAIT_EVAL_DATABASE_URL": f"sqlite:///{root / 'state.sqlite'}",
            "PORTRAIT_EVAL_WORKSPACE": str(root / "workspace"),
            "PORTRAIT_EVAL_ALLOWED_ROOTS": json.dumps([str(device_root)]),
            "PORTRAIT_EVAL_API_TOKEN": "software-acceptance-fixture-token",
            "PORTRAIT_EVAL_REPORT_SHARE_SECRET": "software-acceptance-fixture-share",
            "PORTRAIT_EVAL_VLM_PROVIDER": "local",
            "PORTRAIT_EVAL_PRIMARY_VLM_URL": "",
            "PORTRAIT_EVAL_REVIEWER_VLM_URL": "",
            "PORTRAIT_EVAL_SEARCH_PROVIDER": "disabled",
        }
    )
    with socket.socket() as available_port:
        available_port.bind(("127.0.0.1", 0))
        port = available_port.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"

    def request(path, body=None, *, method=None, admin=True, expect=200, binary=False):
        headers = {}
        if admin:
            headers["Authorization"] = f"Bearer {env['PORTRAIT_EVAL_API_TOKEN']}"
        data = None if body is None else json.dumps(body).encode()
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(base_url + path, data=data, headers=headers, method=method)
        try:
            response = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read()
            assert response.status == expect, (path, response.status, raw.decode(errors="replace"))
            return raw if binary else json.loads(raw)

    summary = {
        "evidence_kind": "synthetic_images_and_heuristic_adapters",
        "real_model_accuracy": "unverified",
        "modes": {},
    }

    def check_public_bundle(report):
        share = request(f"/api/reports/{report['id']}/share", {})
        html = request(share["url"], admin=False, binary=True).decode()
        assert str(root) not in html
        docx = request(share["docx_url"], admin=False, binary=True)
        assert docx.startswith(b"PK") and len(docx) > 1000
        with zipfile.ZipFile(io.BytesIO(docx)) as archive:
            for name in archive.namelist():
                if name.endswith((".xml", ".rels")):
                    assert str(root).encode() not in archive.read(name), name
        path = Path(report["html_path"])
        payload = json.loads(path.with_suffix(".json").read_text())
        assert str(root) not in json.dumps(payload)
        assert all(path.with_suffix(ext).is_file() for ext in (".html", ".json", ".docx"))
        csv_path = path.with_name(path.stem + "-findings.csv")
        assert csv_path.is_file() and str(root) not in csv_path.read_text()
        return payload

    with (artifacts / "http-api.log").open("w") as server_log:
        api = subprocess.Popen(
            [sys.executable, "-m", "portrait_eval.cli", "--host", "127.0.0.1", "--port", str(port)],
            cwd=REPO,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 15
            while True:
                try:
                    assert request("/health", admin=False)["status"] == "ok"
                    break
                except urllib.error.URLError as exc:
                    if api.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("API failed to become ready") from exc
                    time.sleep(0.1)
            for mode in ("quick", "professional"):
                project = request("/api/projects", {"name": f"Acceptance {mode}"}, expect=201)
                pid = project["id"]
                for device in range(4):
                    request(
                        f"/api/projects/{pid}/devices",
                        {
                            "name": f"Device {device}",
                            "folder_path": str(device_root / f"device-{device}"),
                        },
                        expect=201,
                    )
                pairing = request(f"/api/projects/{pid}/scan", {})
                assert len(pairing["groups"]) == 6
                assert (
                    sum(
                        cell is None
                        for group in pairing["groups"]
                        for cell in group["cells"].values()
                    )
                    == 1
                )
                assert any(cell is None for cell in pairing["groups"][2]["cells"].values())
                for ordinal, group in enumerate(pairing["groups"], start=1):
                    assert group["group_id"] == f"G{ordinal:03}"
                    assert all(
                        cell["filename"] == f"{ordinal:02}.png"
                        for cell in group["cells"].values()
                        if cell is not None
                    )
                confirmed = request(
                    f"/api/projects/{pid}/pairing/confirm", {"expected_version": pairing["version"]}
                )
                assert confirmed["confirmed"]
                task = request(f"/api/projects/{pid}/run-full-evaluation", {"mode": mode})
                assert task["status"] == "PENDING" and task["attempts"] == 0, task
                assert request(f"/api/projects/{pid}/reports") == []
                assert request(f"/api/projects/{pid}/analysis?kind=iqa_evaluation") == []
                worker = subprocess.run(
                    [sys.executable, "-m", "portrait_eval.worker", "--once"],
                    cwd=REPO,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                (artifacts / f"http-worker-{mode}.log").write_text(worker.stdout + worker.stderr)
                assert worker.returncode == 0
                finished = request(f"/api/tasks/{task['id']}")
                assert finished["status"] == "SUCCEEDED", finished
                assert finished["attempts"] == 1
                assert finished["result"]["mode"] == mode
                assert finished["result"]["scene_count"] == 6
                core = request(f"/api/projects/{pid}/analysis?kind=iqa_evaluation")
                assert len(core) == 6
                expected_scenes = {
                    group["group_id"]: {
                        device_id: cell["checksum"]
                        for device_id, cell in group["cells"].items()
                        if cell is not None
                    }
                    for group in confirmed["groups"]
                }
                assert {row["payload"]["scene_id"] for row in core} == set(expected_scenes)
                for row in core:
                    evidence = row["payload"]
                    assert evidence["schema_version"] == "iqa-comparison/1"
                    assert evidence["mode"] == "device"
                    assert evidence["bridge_provenance"]["entrypoint"] == (
                        "qwen_tmqa.comparison.evaluate_comparison"
                    )
                    assert all(asset["state"] == "valid" for asset in evidence["assets"])
                    actual = {
                        asset["id"]: asset["trace"]["source_bytes_sha256"]
                        for asset in evidence["assets"]
                    }
                    assert actual == expected_scenes[evidence["scene_id"]]
                reports = request(f"/api/projects/{pid}/reports")
                reviews = request(f"/api/projects/{pid}/review-items")
                if mode == "professional":
                    assert reports[0]["status"] == "draft"
                    check_public_bundle(reports[0])
                    assert any(item["status"] == "open" for item in reviews)
                    request(f"/api/projects/{pid}/reports/finalize", {}, expect=409)
                    for review in reviews:
                        if review["status"] == "open":
                            request(
                                f"/api/review-items/{review['id']}",
                                {
                                    "status": "accepted",
                                    "note": "Synthetic software acceptance only",
                                },
                                method="PATCH",
                            )
                    report = request(f"/api/projects/{pid}/reports/finalize", {})
                else:
                    report = reports[0]
                assert report["status"] == "final"
                payload = check_public_bundle(report)
                assert payload["evaluation_mode"] == mode
                assert payload["review_summary"]["skipped_mandatory_gates"] == (mode == "quick")
                if mode == "professional":
                    assert payload["review_summary"]["open"] == 0
                summary["modes"][mode] = {
                    "task": finished["status"],
                    "worker_attempts": finished["attempts"],
                    "groups": 6,
                    "missing_cells": 1,
                    "iqa_scene_count": len(core),
                    "iqa_source_checksums_verified": True,
                    "report_status": report["status"],
                    "public_html_without_admin": True,
                    "public_docx_without_admin": True,
                    "host_paths_in_publication": False,
                    "open_reviews": payload["review_summary"]["open"],
                }
            assert all(
                hashlib.sha256(path.read_bytes()).hexdigest() == digest
                for path, digest in hashes.items()
            )
            summary["source_files_unchanged"] = True
            summary["fixture_root"] = str(root)
        finally:
            api.terminate()
            try:
                api.wait(timeout=10)
            except subprocess.TimeoutExpired:
                api.kill()
                api.wait()
    (artifacts / "http-acceptance-result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Evidence: {artifacts}")


if __name__ == "__main__":
    main()
