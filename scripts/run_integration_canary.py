"""Exercise the standalone IQA and actual Front workflow with declared fixtures.

Images and adapters are synthetic/heuristic. This verifies software integration;
it does not establish human/model accuracy or tone-mapping training improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from qwen_tmqa.comparison import evaluate_comparison
from qwen_tmqa.comparison_models import ComparisonAsset, ComparisonRequest, ComparisonROI

from portrait_eval.database import Database
from portrait_eval.exporting import export_project
from portrait_eval.repository import Repository
from portrait_eval.workflow import run_evaluation_workflow


def _image(path: Path, gain: float) -> None:
    y, x = np.mgrid[0:96, 0:128]
    base = 0.12 + 0.45 * (x / 127) + 0.1 * (y / 95)
    texture = 0.04 * np.sin(x * 0.7) * np.cos(y * 0.5)
    rgb = np.stack((base + texture, base * 0.9, base * 0.8 - texture), axis=-1)
    image = np.rint(np.clip(rgb * gain, 0, 1) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def run(output: Path) -> dict[str, object]:
    output = output.resolve()
    if (output / "canary.sqlite").exists():
        raise ValueError("Use a fresh output directory so previous runs cannot hide failures")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_path = output / "algorithm" / "source.png"
    baseline_path = output / "algorithm" / "baseline.png"
    candidate_path = output / "algorithm" / "candidate.png"
    _image(source_path, 1)
    _image(baseline_path, 0.85)
    _image(candidate_path, 1.05)
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    request = ComparisonRequest(
        scene_id="canary-algorithm",
        mode="algorithm",
        baseline=ComparisonAsset(id="baseline", path=baseline_path, source_sha256=source_sha),
        candidates=[ComparisonAsset(id="candidate", path=candidate_path, source_sha256=source_sha)],
        source=ComparisonAsset(id="source", path=source_path),
        group_id="canary-source-group",
        split="audit",
        rois=[
            ComparisonROI(id="face-1", kind="face", bbox=(16, 16, 24, 32), person_id="p1"),
            ComparisonROI(id="face-2", kind="face", bbox=(72, 16, 24, 32), person_id="p2"),
        ],
    )
    first = evaluate_comparison(request)
    second = evaluate_comparison(request)
    deterministic = first.model_dump(mode="json") == second.model_dump(mode="json")
    if not deterministic:
        raise AssertionError("Comparison rerun changed stable outputs")
    (output / "comparison.json").write_text(first.model_dump_json(indent=2), encoding="utf-8")
    (output / "comparison-manifest.json").write_text(
        request.model_dump_json(indent=2), encoding="utf-8"
    )
    database = Database(f"sqlite:///{output / 'canary.sqlite'}")
    database.create_all()
    with database.session_factory() as session:
        repo = Repository(session)
        project = repo.create_project("Shared IQA software canary")
        for device_index, gain in enumerate((0.8, 1.1)):
            folder = output / "devices" / f"device-{device_index}"
            for scene_index in range(2):
                _image(folder / f"{scene_index + 1}.png", gain + scene_index * 0.03)
            repo.add_device(project.id, f"Device {device_index}", str(folder))
        pairing = repo.scan_and_pair(project.id)
        repo.confirm_pairing(project.id, pairing["version"])
        workflow = run_evaluation_workflow(
            session, output / "front-workspace", project.id, mode="quick"
        )
        core_records = repo.list_analysis(project.id, "iqa_evaluation")
        bindings = repo.list_analysis(project.id, "evaluation_run_binding")
        if len(core_records) != workflow["scene_count"] or not bindings:
            raise AssertionError("Actual Front run omitted shared IQA or source binding")
        export_path = export_project(repo, project.id, output / "front-workspace")
        with zipfile.ZipFile(export_path) as archive:
            exported_text = "\n".join(
                archive.read(name).decode("utf-8")
                for name in archive.namelist()
                if name.endswith(".json")
            )
        host_paths = str(output) in exported_text
        if host_paths:
            raise AssertionError("Privacy export contains local canary paths")
        summary: dict[str, object] = {
            "schema_version": "frone-iqa.software-canary/1",
            "evidence_kind": "synthetic_images_and_heuristic_adapters",
            "comparison_deterministic": deterministic,
            "front_scene_count": workflow["scene_count"],
            "front_iqa_scene_count": len(core_records),
            "source_byte_binding": bool(bindings),
            "privacy_export_has_host_paths": host_paths,
            "front_status": workflow["status"],
            "paid_endpoint_calls": 0,
            "training_runs": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    summary = run(parser.parse_args().output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
