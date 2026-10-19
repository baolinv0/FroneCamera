from __future__ import annotations

import argparse
import time

from portrait_eval.config import Settings
from portrait_eval.corroboration import (
    HeuristicCorroborationAdapter,
    OpenAICompatibleCorroborationAdapter,
)
from portrait_eval.database import Database
from portrait_eval.research import DisabledSearchProvider, SearxNGSearchProvider
from portrait_eval.tasking import TaskService
from portrait_eval.vlm import build_vision_adapters
from portrait_eval.workflow import run_evaluation_workflow


def _adapters(settings: Settings):
    primary, reviewer = build_vision_adapters(settings)
    search = (
        SearxNGSearchProvider(settings.searxng_url)
        if settings.search_provider == "searxng" and settings.searxng_url
        else DisabledSearchProvider()
    )
    corroborator = (
        OpenAICompatibleCorroborationAdapter(settings.reviewer_vlm_url, settings.reviewer_vlm_model)
        if settings.reviewer_vlm_url
        else HeuristicCorroborationAdapter()
    )
    return primary, reviewer, search, corroborator


def execute_one(settings: Settings) -> bool:
    database = Database(settings.database_url)
    database.create_all()
    with database.session_factory() as session:
        service = TaskService(session)
        task = service.claim_next_pending()
        if task is None:
            return False
        try:
            if task.kind != "evaluate_project":
                raise ValueError(f"Unsupported task kind: {task.kind}")
            primary, reviewer, search, corroborator = _adapters(settings)
            mode = str(task.payload.get("mode", "professional"))
            result = run_evaluation_workflow(
                session,
                settings.workspace,
                task.project_id,
                mode,  # type: ignore[arg-type]
                primary=primary,
                reviewer=reviewer,
                search=search,
                corroborator=corroborator,
            )
            service.mark_succeeded(task.id, result)
        except Exception as exc:  # worker boundary must persist failures
            error = f"{type(exc).__name__}: {exc}"
            if task.attempts < settings.task_max_attempts:
                service.mark_retryable(task.id, error)
            else:
                service.mark_failed(task.id, error)
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    settings = Settings()
    settings.prepare()
    while True:
        worked = execute_one(settings)
        if args.once:
            return
        if not worked:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
