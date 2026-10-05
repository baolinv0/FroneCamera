"""Actual workflow acceptance checks for immutable run input binding."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json

import pytest
from PIL import Image

from portrait_eval.models import ProjectStatus
from portrait_eval.research import SearchProvider
from portrait_eval.workflow import run_evaluation_workflow
from tests.test_task3_runtime_pipeline import db, setup_project


def _mutate_pairing(database, project_id, pairing):
    with database.session_factory() as session:
        from portrait_eval.repository import Repository

        repo = Repository(session)
        group = pairing["groups"][0]
        repo.update_pairing_cell(
            project_id, group["id"], next(iter(group["cells"])), None, pairing["version"]
        )


@pytest.mark.parametrize("stage", ["search", "renderer"])
@pytest.mark.parametrize("mutation", ["pairing", "bytes"])
def test_workflow_refuses_changed_inputs_after_slow_operation(
    tmp_path, monkeypatch, stage, mutation
):
    """Removing late binding verification would publish a stale final report."""
    import portrait_eval.pipeline as pipeline_module

    database = db(tmp_path)
    with database.session_factory() as session:
        repo, project, pairing, paths = setup_project(tmp_path, session)
        changed = False

        def mutate():
            nonlocal changed
            if changed:
                return
            changed = True
            if mutation == "pairing":
                _mutate_pairing(database, project.id, pairing)
            else:
                Image.new("RGB", (96, 96), "red").save(paths[0])

        class MutatingSearch(SearchProvider):
            def search(self, query, limit=5):
                if stage == "search":
                    mutate()
                return []

        real_render = pipeline_module.render_report_bundle

        def mutating_render(*args, **kwargs):
            bundle = real_render(*args, **kwargs)
            if stage == "renderer":
                mutate()
            return bundle

        monkeypatch.setattr(pipeline_module, "render_report_bundle", mutating_render)
        with pytest.raises(ValueError, match=r"Pairing changed|Source bytes changed"):
            run_evaluation_workflow(
                session, tmp_path / "workspace", project.id, search=MutatingSearch()
            )
        assert changed
        assert repo.list_reports(project.id) == []
        session.expire_all()
        if mutation == "pairing":
            assert repo.get_project(project.id).status == ProjectStatus.PAIRING_REQUIRED.value
            assert repo.get_pairing(project.id)["version"] == pairing["version"] + 1
            assert not repo.get_pairing(project.id)["confirmed"]
        else:
            assert repo.get_project(project.id).status == ProjectStatus.FAILED.value
        failure = repo.list_analysis(project.id, "evaluation_failure")[-1]["payload"]
        assert failure["pairing_version"] == pairing["version"]
        assert failure["error_type"] == "ValueError"
        assert not repo.list_analysis(project.id, "image_metrics")


def test_unchanged_workflow_publishes_bound_final(tmp_path):
    with db(tmp_path).session_factory() as session:
        repo, project, pairing, _ = setup_project(tmp_path, session)
        result = run_evaluation_workflow(session, tmp_path / "workspace", project.id)
        assert result["status"] == "REPORT_FINALIZED"
        reports = repo.list_reports(project.id)
        final = next(report for report in reports if report["status"] == "final")
        from pathlib import Path

        payload = json.loads(Path(final["html_path"]).with_suffix(".json").read_text())
        assert payload["input_trace"]["run_binding"]["pairing_version"] == pairing["version"]
        assert payload["pairing_snapshot_id"] == pairing["pairing_snapshot_id"]


def _binding(repo, project_id):
    snapshot = repo.list_pairing_snapshots(project_id)[0]
    pairing = snapshot["payload"]
    return {
        "pairing_snapshot_id": snapshot["id"],
        "pairing_version": snapshot["version"],
        "pairing_sha256": hashlib.sha256(
            json.dumps(pairing, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_checksums": {
            cell["image_id"]: cell["checksum"]
            for group in pairing["groups"]
            for cell in group["cells"].values()
            if cell
        },
    }


def _validator():
    from portrait_eval import pipeline

    assert hasattr(pipeline, "validate_run_binding"), "Reusable run binding validation is missing"
    return importlib.import_module("portrait_eval.run_binding").validate_run_binding


@pytest.mark.parametrize(
    "field,value",
    [
        ("pairing_snapshot_id", "missing"),
        ("pairing_version", -1),
        ("pairing_version", True),
        ("pairing_sha256", "bad"),
        ("source_checksums", {}),
        ("source_checksums", {"unknown": "bad"}),
    ],
)
def test_binding_validator_rejects_invalid_snapshot_and_sources(tmp_path, field, value):
    validate = _validator()
    with db(tmp_path).session_factory() as session:
        repo, project, _, _ = setup_project(tmp_path, session)
        binding = _binding(repo, project.id)
        binding[field] = value
        with pytest.raises(ValueError):
            validate(repo, project.id, binding)


def test_binding_validator_is_read_only_and_refreshes_changed_project(tmp_path):
    validate = _validator()
    database = db(tmp_path)
    with database.session_factory() as session:
        repo, project, pairing, _ = setup_project(tmp_path, session)
        binding = _binding(repo, project.id)
        previous = copy.deepcopy(repo.get_pairing(project.id))
        assert validate(repo, project.id, binding) is None
        assert repo.get_pairing(project.id) == previous
        assert not repo.list_reports(project.id)
        _mutate_pairing(database, project.id, pairing)
        with pytest.raises(ValueError, match="Pairing changed"):
            validate(repo, project.id, binding)
        assert repo.get_project(project.id).status == ProjectStatus.PAIRING_REQUIRED.value


def test_workflow_conditional_publish_preserves_pairing_changed_after_validation(
    tmp_path, monkeypatch
):
    """The conditional UPDATE must close the validation-to-publication race."""
    from sqlalchemy.sql.dml import Update

    database = db(tmp_path)
    with database.session_factory() as session:
        repo, project, pairing, _ = setup_project(tmp_path, session)
        execute = session.execute
        changed = False

        def mutate_at_publication(statement, *args, **kwargs):
            nonlocal changed
            if (
                isinstance(statement, Update)
                and statement.table.name == "projects"
                and statement.compile().params.get("status")
                in {"HUMAN_REVIEW_REQUIRED", "REPORT_DRAFT_READY"}
                and not changed
            ):
                changed = True
                _mutate_pairing(database, project.id, pairing)
            return execute(statement, *args, **kwargs)

        monkeypatch.setattr(session, "execute", mutate_at_publication)
        with pytest.raises(ValueError, match="Pairing changed"):
            run_evaluation_workflow(session, tmp_path / "workspace", project.id)
        assert changed
        session.expire_all()
        assert repo.get_project(project.id).status == ProjectStatus.PAIRING_REQUIRED.value
        assert repo.get_project(project.id).version == pairing["version"] + 1
        assert repo.list_reports(project.id) == []


@pytest.mark.parametrize("mutation", ["replace", "remove"])
def test_binding_validator_reads_original_snapshot_file_bytes(tmp_path, mutation):
    validate = _validator()
    with db(tmp_path).session_factory() as session:
        repo, project, _, paths = setup_project(tmp_path, session)
        binding = _binding(repo, project.id)
        if mutation == "replace":
            Image.new("RGB", (96, 96), "red").save(paths[0])
        else:
            paths[0].unlink()
        with pytest.raises(ValueError, match="Source bytes changed"):
            validate(repo, project.id, binding)


def test_binding_validator_does_not_flush_or_commit_pending_rows(tmp_path):
    from portrait_eval.database import ReviewItemRow

    validate = _validator()
    database = db(tmp_path)
    with database.session_factory() as session:
        repo, project, _, _ = setup_project(tmp_path, session)
        binding = _binding(repo, project.id)
        pending = ReviewItemRow(
            id="pending-review", project_id=project.id, category="pending", payload_json="{}"
        )
        session.add(pending)
        validate(repo, project.id, binding)
        assert pending in session.new
        with database.session_factory() as other:
            assert other.get(ReviewItemRow, pending.id) is None


def test_stale_run_cleanup_preserves_other_sessions_report_and_review(tmp_path):
    """Cleanup must delete this attempt's outputs rather than newer user work."""
    from portrait_eval.repository import Repository

    database = db(tmp_path)
    with database.session_factory() as session:
        repo, project, pairing, _ = setup_project(tmp_path, session)
        external = {}

        class UserEditSearch(SearchProvider):
            def search(self, query, limit=5):
                if not external:
                    with database.session_factory() as other:
                        user_repo = Repository(other)
                        report = user_repo.save_report(
                            project.id, "user-v1", "final", str(tmp_path / "user-final.html")
                        )
                        review = user_repo.create_review_item(
                            project.id, "user-review", {"message": "preserve my review"}
                        )
                        external.update(report_id=report.id, review_id=review.id)
                        group = pairing["groups"][0]
                        user_repo.update_pairing_cell(
                            project.id,
                            group["id"],
                            next(iter(group["cells"])),
                            None,
                            pairing["version"],
                        )
                return []

        with pytest.raises(ValueError, match="Pairing changed"):
            run_evaluation_workflow(
                session, tmp_path / "workspace", project.id, search=UserEditSearch()
            )
        assert [item["id"] for item in repo.list_reports(project.id)] == [external["report_id"]]
        assert [item["id"] for item in repo.list_review_items(project.id)] == [
            external["review_id"]
        ]
        assert not repo.list_analysis(project.id, "image_metrics")
        assert repo.list_analysis(project.id, "evaluation_failure")
        assert repo.get_project(project.id).status == "PAIRING_REQUIRED"
