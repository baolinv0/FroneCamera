from pathlib import Path

import pytest
from PIL import Image

from portrait_eval.attribution import attribute_output_claim
from portrait_eval.database import Database
from portrait_eval.repository import Repository


def project_fixture(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'data.sqlite'}")
    database.create_all()
    session = database.session_factory()
    repo = Repository(session)
    project = repo.create_project("annotations")
    for index in range(2):
        folder = tmp_path / f"d{index}"
        folder.mkdir()
        Image.new("RGB", (96, 96), (80 + index * 50,) * 3).save(folder / "1.png")
        repo.add_device(project.id, f"Device {index}", str(folder))
    draft = repo.scan_and_pair(project.id)
    pairing = repo.confirm_pairing(project.id, draft["version"])
    return session, repo, project, pairing


def test_annotations_are_append_only_and_consumed_by_pipeline(tmp_path):
    from portrait_eval.annotations import SceneAnnotation, save_scene_annotation
    from portrait_eval.pipeline import EvaluationPipeline

    session, repo, project, pairing = project_fixture(tmp_path)
    group = pairing["groups"][0]
    device = pairing["devices"][0]["id"]
    body = SceneAnnotation(
        author="engineer",
        reason="verified capture sheet",
        expected_pairing_version=pairing["version"],
        scene_family="room-1",
        challenge="mixed_light",
        faces={device: [{"bbox_xywh": [10, 10, 30, 30], "role": "SECONDARY"}]},
    )
    first = save_scene_annotation(repo, project.id, group["group_id"], body)
    assert first["revision"] == 1
    with pytest.raises(RuntimeError, match="revision_conflict"):
        save_scene_annotation(repo, project.id, group["group_id"], body)
    body.expected_revision = 1
    second = save_scene_annotation(repo, project.id, group["group_id"], body)
    assert second["revision"] == 2
    assert len(repo.list_analysis(project.id, "scene_annotation")) == 2
    EvaluationPipeline(session, tmp_path / "workspace").run(project.id)
    audit = repo.list_analysis(project.id, "scene_audit")[0]["payload"]
    assert audit["challenge"]["primary"] == "mixed_light"
    metrics = repo.list_analysis(project.id, "image_metrics")
    assert any(
        item["payload"]["faces"][0]["role"] == "SECONDARY"
        for item in metrics
        if item["payload"]["faces"]
    )
    session.close()


def test_annotations_reject_foreign_devices_and_out_of_bounds_roi(tmp_path):
    from portrait_eval.annotations import SceneAnnotation, save_scene_annotation

    session, repo, project, pairing = project_fixture(tmp_path)
    values = dict(author="engineer", reason="test", expected_pairing_version=pairing["version"])
    with pytest.raises(ValueError, match="device"):
        save_scene_annotation(
            repo,
            project.id,
            "G001",
            SceneAnnotation(**values, faces={"foreign": [{"bbox_xywh": [1, 1, 10, 10]}]}),
        )
    device = pairing["devices"][0]["id"]
    with pytest.raises(ValueError, match="outside"):
        save_scene_annotation(
            repo,
            project.id,
            "G001",
            SceneAnnotation(**values, faces={device: [{"bbox_xywh": [90, 90, 30, 30]}]}),
        )
    session.close()


def test_empty_exif_values_cannot_increase_mechanism_confidence():
    result = attribute_output_claim("More apparent detail", {}, {"ISO": "None", "ExposureTime": ""})
    assert result.primary_layer == "indeterminate"
    assert result.exif_evidence_keys == []
    assert result.minimal_experiment
    assert "focus error" in result.alternative_explanations


def test_reruns_preserve_previous_report_payload_and_assets(tmp_path):
    from portrait_eval.pipeline import EvaluationPipeline

    session, repo, project, pairing = project_fixture(tmp_path)
    first = EvaluationPipeline(session, tmp_path / "workspace").run(project.id)
    first_json = Path(first["json_path"])
    saved = first_json.read_bytes()
    pairing = repo.confirm_pairing(project.id, pairing["version"])
    second = EvaluationPipeline(session, tmp_path / "workspace").run(project.id)
    assert first["report_path"] != second["report_path"]
    assert first_json.read_bytes() == saved
    session.close()


def test_pipeline_reads_frozen_pairing_snapshot_not_mutable_grid(tmp_path, monkeypatch):
    from portrait_eval.pipeline import EvaluationPipeline

    session, _repo, project, _pairing = project_fixture(tmp_path)
    pipeline = EvaluationPipeline(session, tmp_path / "workspace")

    def unavailable(_project):
        raise RuntimeError("Live pairing must not be used after confirmation")

    monkeypatch.setattr(pipeline.repo, "get_pairing", unavailable)
    result = pipeline.run(project.id)
    assert result["scene_count"] == 1
    session.close()
