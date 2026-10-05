import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock

import pytest

from portrait_eval.repository import Repository
from portrait_eval.workflow import finalize_quick_report
from tests.test_report_projection import fixture_payload, setup_report


@pytest.mark.parametrize("first_mode", ["professional", "quick"])
def test_concurrent_publications_keep_distinct_projections_and_immutable_bundles(
    tmp_path: Path, monkeypatch, first_mode: str
):
    from portrait_eval.reporting import render_report_bundle

    client, database, project_id, _ = setup_report(tmp_path, fixture_payload())
    paused, release = Event(), Event()
    guard = Lock()
    calls = 0

    def pause_first(payload, directory, version, status):
        nonlocal calls
        with guard:
            calls += 1
            first = calls == 1
        if first:
            paused.set()
            assert release.wait(10)
        return render_report_bundle(payload, directory, version, status)

    monkeypatch.setattr("portrait_eval.api.render_report_bundle", pause_first)
    monkeypatch.setattr("portrait_eval.workflow.render_report_bundle", pause_first)

    def publish(mode):
        if mode == "quick":
            with database.session_factory() as session:
                return finalize_quick_report(Repository(session), project_id)
        response = client.post(f"/api/projects/{project_id}/reports/finalize")
        assert response.status_code == 200, response.text
        return response.json()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(publish, first_mode)
        try:
            assert paused.wait(10)
            with database.session_factory() as session:
                repo = Repository(session)
                review = repo.list_review_items(project_id)[0]
                repo.resolve_review_item(review["id"], "approved")
            second = executor.submit(publish, "professional").result(timeout=10)
            second_path = Path(second["html_path"])
            second_bytes = {
                file.relative_to(second_path.parent).as_posix(): file.read_bytes()
                for file in second_path.parent.rglob("*")
                if file.is_file()
            }
        finally:
            release.set()
        first_result = first.result(timeout=10)
    first_path = Path(first_result["html_path"])
    assert first_result["version"] != second["version"]
    assert first_path != second_path
    assert [
        item["claim_id"]
        for item in json.loads(first_path.with_suffix(".json").read_text())["findings"]
    ] == ["b"]
    assert [
        item["claim_id"]
        for item in json.loads(second_path.with_suffix(".json").read_text())["findings"]
    ] == ["a", "b"]
    assert second_bytes == {
        file.relative_to(second_path.parent).as_posix(): file.read_bytes()
        for file in second_path.parent.rglob("*")
        if file.is_file()
    }
    with database.session_factory() as session:
        rows = [
            row for row in Repository(session).list_reports(project_id) if row["status"] == "final"
        ]
        assert len(rows) == 2 and len({row["html_path"] for row in rows}) == 2
        assert rows[0]["id"] == second["id"]
    assert (
        client.get(f"/api/projects/{project_id}/reports/latest").content == second_path.read_bytes()
    )


@pytest.mark.parametrize("mode", ["professional", "quick"])
@pytest.mark.parametrize("failure", ["raises", "incomplete"])
def test_renderer_failure_registers_no_final_and_retry_is_safe(
    tmp_path: Path, monkeypatch, mode: str, failure: str
):
    from portrait_eval.reporting import render_report_bundle

    client, database, project_id, _ = setup_report(tmp_path, fixture_payload())
    attempted = []

    def fail_after_partial(payload, directory, version, status):
        attempted.append((Path(directory), version))
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / f"final-v{version}.html").write_text("partial")
        if failure == "raises":
            raise RuntimeError("DOCX renderer failed")
        return {"html": Path(directory) / f"final-v{version}.html"}

    target = f"portrait_eval.{'api' if mode == 'professional' else 'workflow'}.render_report_bundle"
    monkeypatch.setattr(target, fail_after_partial)
    with pytest.raises(RuntimeError, match=r"DOCX renderer failed|complete bundle"):
        if mode == "professional":
            client.post(f"/api/projects/{project_id}/reports/finalize")
        else:
            with database.session_factory() as session:
                finalize_quick_report(Repository(session), project_id)
    with database.session_factory() as session:
        assert not [
            row for row in Repository(session).list_reports(project_id) if row["status"] == "final"
        ]
    monkeypatch.setattr(target, render_report_bundle)
    if mode == "professional":
        response = client.post(f"/api/projects/{project_id}/reports/finalize")
        assert response.status_code == 200
        result = response.json()
    else:
        with database.session_factory() as session:
            result = finalize_quick_report(Repository(session), project_id)
    path = Path(result["html_path"])
    assert all(path.with_suffix(suffix).is_file() for suffix in (".html", ".json", ".docx"))
    assert result["version"] != attempted[0][1]
    assert not attempted[0][0].exists()


def _reserve_in_process(database_url: str, project_id: str) -> int:
    from portrait_eval.database import Database
    from portrait_eval.report_publication import reserve_publication

    database = Database(database_url)
    with database.session_factory() as session:
        return reserve_publication(Repository(session), project_id).ordinal


def test_reservation_is_shared_by_independent_processes(tmp_path: Path):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from portrait_eval.database import Database

    database_url = f"sqlite:///{tmp_path / 'process.db'}"
    database = Database(database_url)
    database.create_all()
    with database.session_factory() as session:
        project_id = Repository(session).create_project("processes").id
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as executor:
        futures = [executor.submit(_reserve_in_process, database_url, project_id) for _ in range(3)]
        assert sorted(future.result(timeout=20) for future in futures) == [0, 1, 2]


def test_published_relative_assets_are_copied_into_immutable_bundle(tmp_path: Path):
    from PIL import Image

    payload = fixture_payload()
    source_image = tmp_path / "photo.jpg"
    Image.new("RGB", (32, 32), "gray").save(source_image)
    payload.visual_assets = [
        {"path": str(source_image), "scene_id": "G001", "device_id": "device-b"}
    ]
    client, _, project_id, draft = setup_report(tmp_path, payload)
    original_data = json.loads(draft["json"].read_text())
    assert original_data["visual_assets"][0]["path"].startswith("assets/")
    response = client.post(f"/api/projects/{project_id}/reports/finalize")
    assert response.status_code == 200, response.text
    path = Path(response.json()["html_path"])
    data = json.loads(path.with_suffix(".json").read_text())
    asset = data["visual_assets"][0]
    assert asset["path"].startswith("assets/") and not asset.get("missing")
    assert (path.parent / asset["path"]).read_bytes() == source_image.read_bytes()


def test_publication_migration_upgrades_existing_schema_and_keeps_reports(tmp_path: Path):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect

    from portrait_eval.database import Database, ReportPublicationRow

    database_url = f"sqlite:///{tmp_path / 'migration.db'}"
    database = Database(database_url)
    database.create_all()
    with database.session_factory() as session:
        repo = Repository(session)
        project_id = repo.create_project("existing").id
        repo.save_report(project_id, "1.7", "final", "/existing/final-v1.7.html")
    # Simulate a deployment predating the additive reservation table.
    ReportPublicationRow.__table__.drop(database.engine)
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.stamp(config, "20260720_0002")
    command.upgrade(config, "head")
    assert "report_publications" in inspect(database.engine).get_table_names()
    assert _reserve_in_process(database_url, project_id) == 8
    with database.session_factory() as session:
        assert Repository(session).list_reports(project_id)[0]["version"] == "1.7"
    command.downgrade(config, "20260720_0002")
    assert "report_publications" not in inspect(database.engine).get_table_names()
    command.upgrade(config, "head")
    assert "report_publications" in inspect(database.engine).get_table_names()


def test_expiring_session_returns_committed_report_without_refresh_or_deleting_artifacts(
    tmp_path: Path,
):
    from sqlalchemy import event, select
    from sqlalchemy.orm import Session

    from portrait_eval.database import ReportPublicationRow, ReportRow
    from portrait_eval.report_projection import resolve_report_payload
    from portrait_eval.report_publication import publish_final_report
    from portrait_eval.reporting import ReportPayload, render_report_bundle
    from tests.test_report_projection import docx_text

    _, database, project_id, draft = setup_report(tmp_path, fixture_payload())
    commits = 0
    failed_selects = 0

    def lose_connection_after_commit(
        _connection, _cursor, statement, _parameters, _context, _executemany
    ):
        nonlocal failed_selects
        if commits >= 2 and statement.lstrip().lower().startswith("select reports."):
            failed_selects += 1
            raise RuntimeError("connection lost after durable final commit")

    with Session(database.engine) as session:
        assert session.expire_on_commit is True
        repo = Repository(session)
        payload = ReportPayload.model_validate_json(draft["json"].read_text())
        payload = resolve_report_payload(payload, repo.list_review_items(project_id))

        def after_commit(_session):
            nonlocal commits
            commits += 1

        event.listen(session, "after_commit", after_commit)
        event.listen(database.engine, "before_cursor_execute", lose_connection_after_commit)
        try:
            result = publish_final_report(
                repo, project_id, payload, draft["html"], render_report_bundle
            )
        finally:
            event.remove(database.engine, "before_cursor_execute", lose_connection_after_commit)
    assert commits == 2 and failed_selects == 0
    path = Path(result["html_path"])
    assert path.is_file() and "FINAL REPORT" in path.read_text()
    assert path.with_suffix(".json").is_file()
    assert path.with_suffix(".docx").is_file() and "Final Verdict" in docx_text(
        path.with_suffix(".docx")
    )
    assert path.with_name(path.stem + "-findings.csv").is_file()
    with database.session_factory() as session:
        row = session.get(ReportRow, result["id"])
        assert row is not None and row.status == "final" and row.html_path == str(path)
        allocation = session.scalar(
            select(ReportPublicationRow).where(ReportPublicationRow.project_id == project_id)
        )
        assert (
            allocation is not None and allocation.status == "published" and allocation.error is None
        )
