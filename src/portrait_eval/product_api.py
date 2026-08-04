from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from portrait_eval.api import _adapter_set
from portrait_eval.api import create_app as create_base_app
from portrait_eval.config import Settings
from portrait_eval.database import Database, ReportRow
from portrait_eval.reporting import render_pdf_report
from portrait_eval.repository import Repository
from portrait_eval.tasking import TaskService
from portrait_eval.workflow import EvaluationMode, run_evaluation_workflow


class FullEvaluationRequest(BaseModel):
    mode: Literal["quick", "professional"] = "quick"


def _share_token(secret: str, report_id: str) -> str:
    return hmac.new(secret.encode("utf-8"), report_id.encode("utf-8"), hashlib.sha256).hexdigest()


def _report_path(settings: Settings, row: ReportRow) -> Path:
    path = Path(row.html_path).resolve()
    workspace = settings.workspace.resolve()
    if workspace not in path.parents or not path.is_file():
        raise HTTPException(status_code=403, detail="Invalid report path")
    return path


def register_product_routes(app: FastAPI) -> None:
    settings: Settings = app.state.settings
    database: Database = app.state.database

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        if settings.api_token and authorization != f"Bearer {settings.api_token}":
            raise HTTPException(status_code=401, detail="Invalid token")

    def repository() -> Iterator[Repository]:
        session = database.session_factory()
        try:
            yield Repository(session)
        finally:
            session.close()

    def validate_share_token(report_id: str, token: str) -> None:
        expected = _share_token(settings.report_share_secret, report_id)
        if not hmac.compare_digest(expected, token):
            raise HTTPException(status_code=403, detail="Invalid report share token")

    @app.post(
        "/api/projects/{project_id}/run-full-evaluation",
        dependencies=[Depends(authorize)],
    )
    def run_full_evaluation(
        project_id: str,
        body: FullEvaluationRequest,
        sync: bool = Query(default=False),
    ) -> dict[str, Any]:
        mode: EvaluationMode = body.mode
        if not sync:
            with database.session_factory() as session:
                task = TaskService(session).enqueue(
                    project_id,
                    "evaluate_project",
                    {"project_id": project_id, "mode": mode},
                )
                return task.__dict__

        with database.session_factory() as session:
            primary, reviewer, search, corroborator = _adapter_set(settings)
            try:
                return run_evaluation_workflow(
                    session,
                    settings.workspace,
                    project_id,
                    mode,
                    primary=primary,
                    reviewer=reviewer,
                    search=search,
                    corroborator=corroborator,
                )
            except (KeyError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/reports/{report_id}/share", dependencies=[Depends(authorize)])
    def create_report_share(
        report_id: str,
        repo: Repository = Depends(repository),
    ) -> dict[str, str]:
        row = repo.session.get(ReportRow, report_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Report not found")
        _report_path(settings, row)
        token = _share_token(settings.report_share_secret, report_id)
        return {
            "report_id": report_id,
            "token": token,
            "url": f"/reports/{report_id}?token={token}",
            "pdf_url": f"/reports/{report_id}/pdf?token={token}",
            "docx_url": f"/reports/{report_id}/docx?token={token}",
        }

    @app.get("/reports/{report_id}", response_class=HTMLResponse)
    def public_report(
        report_id: str,
        token: str,
        repo: Repository = Depends(repository),
    ) -> HTMLResponse:
        validate_share_token(report_id, token)
        row = repo.session.get(ReportRow, report_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Report not found")
        path = _report_path(settings, row)
        return HTMLResponse(
            path.read_text(encoding="utf-8"),
            headers={"Cache-Control": "private, no-store"},
        )

    @app.get("/reports/{report_id}/pdf")
    def public_report_pdf(
        report_id: str,
        token: str,
        repo: Repository = Depends(repository),
    ) -> FileResponse:
        validate_share_token(report_id, token)
        row = repo.session.get(ReportRow, report_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Report not found")
        html_path = _report_path(settings, row)
        pdf_path = html_path.with_suffix(".pdf")
        if not pdf_path.exists():
            try:
                render_pdf_report(html_path, pdf_path)
            except RuntimeError as exc:
                raise HTTPException(status_code=501, detail=str(exc)) from exc
        return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path.name)

    @app.get("/reports/{report_id}/docx")
    def public_report_docx(
        report_id: str,
        token: str,
        repo: Repository = Depends(repository),
    ) -> FileResponse:
        validate_share_token(report_id, token)
        row = repo.session.get(ReportRow, report_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Report not found")
        html_path = _report_path(settings, row)
        docx_path = html_path.with_suffix(".docx")
        if not docx_path.is_file():
            raise HTTPException(status_code=404, detail="DOCX report not found")
        return FileResponse(
            docx_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=docx_path.name,
            headers={"Cache-Control": "private, no-store"},
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    app = create_base_app(settings)
    register_product_routes(app)
    return app


app = create_app()
