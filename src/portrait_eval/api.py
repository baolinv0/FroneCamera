from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from portrait_eval.config import Settings
from portrait_eval.corroboration import (
    CorroborationAdapter,
    HeuristicCorroborationAdapter,
    OpenAICompatibleCorroborationAdapter,
)
from portrait_eval.database import (
    AnalysisRow,
    Database,
    ImageRow,
    ReportRow,
    ReviewItemRow,
    json_load,
)
from portrait_eval.exporting import export_project
from portrait_eval.pipeline import EvaluationPipeline
from portrait_eval.reporting import ReportPayload, render_pdf_report, render_report_bundle
from portrait_eval.repository import Repository
from portrait_eval.research import DisabledSearchProvider, SearchProvider, SearxNGSearchProvider
from portrait_eval.tasking import TaskService
from portrait_eval.vlm import (
    HeuristicVisionAdapter,
    OpenAICompatibleVisionAdapter,
    VisionModelAdapter,
)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class DeviceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    folder_path: str
    canonical_model: str | None = None


class PairingUpdate(BaseModel):
    expected_version: int
    group_id: str
    device_id: str
    image_id: str | None


class VersionRequest(BaseModel):
    expected_version: int


class ReviewResolution(BaseModel):
    status: str = Field(pattern="^(accepted|rejected|edited|insufficient_evidence)$")
    note: str | None = None


def _adapter_set(
    settings: Settings,
) -> tuple[VisionModelAdapter, VisionModelAdapter, SearchProvider, CorroborationAdapter]:
    primary = (
        OpenAICompatibleVisionAdapter(
            settings.primary_vlm_url,
            settings.primary_vlm_model,
            "primary",
            max_image_edge=settings.vlm_max_image_edge,
        )
        if settings.primary_vlm_url
        else HeuristicVisionAdapter("primary")
    )
    reviewer = (
        OpenAICompatibleVisionAdapter(
            settings.reviewer_vlm_url,
            settings.reviewer_vlm_model,
            "reviewer",
            max_image_edge=settings.vlm_max_image_edge,
        )
        if settings.reviewer_vlm_url
        else HeuristicVisionAdapter("reviewer")
    )
    search = (
        SearxNGSearchProvider(settings.searxng_url)
        if settings.search_provider == "searxng" and settings.searxng_url
        else DisabledSearchProvider()
    )
    corroborator: CorroborationAdapter = (
        OpenAICompatibleCorroborationAdapter(settings.reviewer_vlm_url, settings.reviewer_vlm_model)
        if settings.reviewer_vlm_url
        else HeuristicCorroborationAdapter()
    )
    return primary, reviewer, search, corroborator


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.prepare()
    database = Database(settings.database_url)
    database.create_all()
    app = FastAPI(title="FroneCamera", version="0.1.0")
    app.state.settings = settings
    app.state.database = database
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        if settings.api_token and authorization != f"Bearer {settings.api_token}":
            raise HTTPException(status_code=401, detail="Invalid token")

    def repository() -> Iterator[Repository]:
        session = database.session_factory()
        try:
            yield Repository(session)
        finally:
            session.close()

    @app.get("/", response_class=HTMLResponse)
    def console() -> str:
        path = Path(__file__).parent / "static" / "index.html"
        return path.read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/projects", status_code=201, dependencies=[Depends(authorize)])
    def create_project(
        body: ProjectCreate, repo: Repository = Depends(repository)
    ) -> dict[str, Any]:
        row = repo.create_project(body.name)
        return {"id": row.id, "name": row.name, "status": row.status, "version": row.version}

    @app.get("/api/projects", dependencies=[Depends(authorize)])
    def list_projects(repo: Repository = Depends(repository)) -> list[dict[str, Any]]:
        return [
            {
                "id": row.id,
                "name": row.name,
                "status": row.status,
                "version": row.version,
                "created_at": row.created_at,
            }
            for row in repo.list_projects()
        ]

    @app.get("/api/projects/{project_id}", dependencies=[Depends(authorize)])
    def get_project(project_id: str, repo: Repository = Depends(repository)) -> dict[str, Any]:
        try:
            row = repo.get_project(project_id)
            pairing = repo.get_pairing(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Project not found") from exc
        return {
            "id": row.id,
            "name": row.name,
            "status": row.status,
            "version": row.version,
            "devices": pairing["devices"],
            "group_count": len(pairing["groups"]),
            "review_item_count": len(repo.list_review_items(project_id)),
            "reports": repo.list_reports(project_id),
        }

    @app.post(
        "/api/projects/{project_id}/devices", status_code=201, dependencies=[Depends(authorize)]
    )
    def add_device(
        project_id: str, body: DeviceCreate, repo: Repository = Depends(repository)
    ) -> dict[str, Any]:
        folder = Path(body.folder_path).expanduser().resolve()
        if settings.allowed_roots:
            roots = [root.expanduser().resolve() for root in settings.allowed_roots]
            if not any(folder == root or root in folder.parents for root in roots):
                raise HTTPException(
                    status_code=403, detail="Folder is outside configured allowed roots"
                )
        try:
            row = repo.add_device(project_id, body.name, str(folder), body.canonical_model)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "id": row.id,
            "name": row.name,
            "folder_path": row.folder_path,
            "canonical_model": row.canonical_model,
        }

    @app.post("/api/projects/{project_id}/scan", dependencies=[Depends(authorize)])
    def scan(project_id: str, repo: Repository = Depends(repository)) -> dict[str, Any]:
        try:
            return repo.scan_and_pair(project_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}/pairing", dependencies=[Depends(authorize)])
    def pairing(project_id: str, repo: Repository = Depends(repository)) -> dict[str, Any]:
        try:
            return repo.get_pairing(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Project not found") from exc

    @app.get(
        "/api/projects/{project_id}/pairing/snapshots",
        dependencies=[Depends(authorize)],
    )
    def pairing_snapshots(
        project_id: str, repo: Repository = Depends(repository)
    ) -> list[dict[str, Any]]:
        try:
            return repo.list_pairing_snapshots(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Project not found") from exc

    @app.put("/api/projects/{project_id}/pairing", dependencies=[Depends(authorize)])
    def update_pairing(
        project_id: str,
        body: PairingUpdate,
        repo: Repository = Depends(repository),
    ) -> dict[str, Any]:
        try:
            return repo.update_pairing_cell(
                project_id,
                body.group_id,
                body.device_id,
                body.image_id,
                body.expected_version,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/projects/{project_id}/pairing/confirm", dependencies=[Depends(authorize)])
    def confirm_pairing(
        project_id: str,
        body: VersionRequest,
        repo: Repository = Depends(repository),
    ) -> dict[str, Any]:
        try:
            return repo.confirm_pairing(project_id, body.expected_version)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/projects/{project_id}/run", dependencies=[Depends(authorize)])
    def run_project(project_id: str, sync: bool = Query(default=False)) -> dict[str, Any]:
        if not sync:
            with database.session_factory() as session:
                task = TaskService(session).enqueue(
                    project_id,
                    "evaluate_project",
                    {"project_id": project_id},
                )
                return task.__dict__
        with database.session_factory() as session:
            primary, reviewer, search, corroborator = _adapter_set(settings)
            try:
                return EvaluationPipeline(
                    session,
                    settings.workspace,
                    primary=primary,
                    reviewer=reviewer,
                    search=search,
                    corroborator=corroborator,
                ).run(project_id)
            except (KeyError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/tasks/{task_id}", dependencies=[Depends(authorize)])
    def task_status(task_id: str) -> dict[str, Any]:
        with database.session_factory() as session:
            try:
                return TaskService(session).get(task_id).__dict__
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Task not found") from exc

    @app.get("/api/projects/{project_id}/analysis", dependencies=[Depends(authorize)])
    def analysis(
        project_id: str,
        repo: Repository = Depends(repository),
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        return repo.list_analysis(project_id, kind)

    @app.get("/api/projects/{project_id}/review-items", dependencies=[Depends(authorize)])
    def review_items(
        project_id: str, repo: Repository = Depends(repository)
    ) -> list[dict[str, Any]]:
        return repo.list_review_items(project_id)

    @app.patch("/api/review-items/{review_id}", dependencies=[Depends(authorize)])
    def resolve_review(
        review_id: str,
        body: ReviewResolution,
        repo: Repository = Depends(repository),
    ) -> dict[str, Any]:
        try:
            return repo.resolve_review_item(review_id, body.status, body.note)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Review item not found") from exc

    @app.get("/api/analysis/{analysis_id}/diagnostic", dependencies=[Depends(authorize)])
    def diagnostic_asset(analysis_id: str, repo: Repository = Depends(repository)) -> FileResponse:
        row = repo.session.get(AnalysisRow, analysis_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        payload = json_load(row.payload_json)
        if not isinstance(payload, dict) or not isinstance(payload.get("diagnostic_path"), str):
            raise HTTPException(status_code=404, detail="Diagnostic not found")
        path = Path(payload["diagnostic_path"]).resolve()
        workspace = settings.workspace.resolve()
        if workspace not in path.parents or not path.is_file():
            raise HTTPException(status_code=403, detail="Invalid diagnostic path")
        return FileResponse(path)

    @app.get("/api/assets/{image_id}", dependencies=[Depends(authorize)])
    def image_asset(image_id: str, repo: Repository = Depends(repository)) -> FileResponse:
        row = repo.session.get(ImageRow, image_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Image not found")
        path = Path(row.path).resolve()
        return FileResponse(path, filename=row.filename)

    @app.get("/api/projects/{project_id}/export", dependencies=[Depends(authorize)])
    def project_export(project_id: str, repo: Repository = Depends(repository)) -> FileResponse:
        try:
            path = export_project(repo, project_id, settings.workspace)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Project not found") from exc
        return FileResponse(path, media_type="application/zip", filename=path.name)

    @app.get("/api/projects/{project_id}/reports", dependencies=[Depends(authorize)])
    def reports(project_id: str, repo: Repository = Depends(repository)) -> list[dict[str, Any]]:
        return repo.list_reports(project_id)

    @app.get("/api/projects/{project_id}/reports/latest", dependencies=[Depends(authorize)])
    def latest_report(project_id: str, repo: Repository = Depends(repository)) -> FileResponse:
        row = repo.session.scalar(
            select(ReportRow)
            .where(ReportRow.project_id == project_id)
            .order_by(ReportRow.created_at.desc())
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Report not found")
        path = Path(row.html_path).resolve()
        workspace = settings.workspace.resolve()
        if workspace not in path.parents:
            raise HTTPException(status_code=403, detail="Invalid report path")
        return FileResponse(path, media_type="text/html", filename=path.name)

    @app.get("/api/reports/{report_id}/html", dependencies=[Depends(authorize)])
    def report_html(report_id: str, repo: Repository = Depends(repository)) -> FileResponse:
        row = repo.session.get(ReportRow, report_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Report not found")
        html_path = Path(row.html_path).resolve()
        workspace = settings.workspace.resolve()
        if workspace not in html_path.parents or not html_path.is_file():
            raise HTTPException(status_code=403, detail="Invalid report path")
        return FileResponse(html_path, media_type="text/html", filename=html_path.name)

    @app.get("/api/reports/{report_id}/pdf", dependencies=[Depends(authorize)])
    def report_pdf(report_id: str, repo: Repository = Depends(repository)) -> FileResponse:
        row = repo.session.get(ReportRow, report_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Report not found")
        html_path = Path(row.html_path).resolve()
        workspace = settings.workspace.resolve()
        if workspace not in html_path.parents or not html_path.is_file():
            raise HTTPException(status_code=403, detail="Invalid report path")
        pdf_path = html_path.with_suffix(".pdf")
        if not pdf_path.exists():
            try:
                render_pdf_report(html_path, pdf_path)
            except RuntimeError as exc:
                raise HTTPException(status_code=501, detail=str(exc)) from exc
        return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path.name)

    @app.post("/api/projects/{project_id}/reports/finalize", dependencies=[Depends(authorize)])
    def finalize_report(project_id: str, repo: Repository = Depends(repository)) -> dict[str, Any]:
        open_items = list(
            repo.session.scalars(
                select(ReviewItemRow).where(
                    ReviewItemRow.project_id == project_id,
                    ReviewItemRow.status == "open",
                )
            )
        )
        if open_items:
            raise HTTPException(
                status_code=409, detail="Resolve all review items before finalizing"
            )
        draft = repo.session.scalar(
            select(ReportRow)
            .where(ReportRow.project_id == project_id, ReportRow.status == "draft")
            .order_by(ReportRow.created_at.desc())
        )
        if draft is None:
            raise HTTPException(status_code=404, detail="Draft report not found")
        source = Path(draft.html_path)
        existing_final = [
            item for item in repo.list_reports(project_id) if item["status"] == "final"
        ]
        version = f"1.{len(existing_final)}"
        source_json = source.with_suffix(".json")
        if source_json.is_file():
            payload = ReportPayload.model_validate_json(source_json.read_text(encoding="utf-8"))
            resolved_reviews = repo.list_review_items(project_id)
            rejected_statements = {
                str(item["payload"].get("statement"))
                for item in resolved_reviews
                if item["status"] in {"rejected", "insufficient_evidence"}
                and isinstance(item["payload"], dict)
                and item["payload"].get("statement")
            }
            rejected_external = {
                (
                    str(item["payload"].get("url")),
                    str(item["payload"].get("claim_statement")),
                )
                for item in resolved_reviews
                if item["category"] == "external_corroboration_review"
                and item["status"] in {"rejected", "insufficient_evidence"}
                and isinstance(item["payload"], dict)
            }
            payload.findings = [
                item
                for item in payload.findings
                if str(item.get("statement")) not in rejected_statements
            ]
            payload.attributions = [
                item
                for item in payload.attributions
                if str(item.get("statement")) not in rejected_statements
            ]
            payload.external_validation = [
                item
                for item in payload.external_validation
                if (str(item.get("url")), str(item.get("claim_statement"))) not in rejected_external
            ]
            bundle = render_report_bundle(payload, source.parent, version, "final")
            destination = bundle["html"]
        else:
            destination = source.with_name(f"final-v{version}.html")
            document = source.read_text(encoding="utf-8")
            document = document.replace(
                "DRAFT — verify review gates before external distribution.",
                "FINAL REPORT — review gates resolved.",
            )
            destination.write_text(document, encoding="utf-8")
        row = repo.save_report(project_id, version, "final", str(destination))
        project = repo.get_project(project_id)
        project.status = "REPORT_FINALIZED"
        repo.session.commit()
        return {
            "id": row.id,
            "version": row.version,
            "status": row.status,
            "html_path": row.html_path,
        }

    return app


app = create_app()
