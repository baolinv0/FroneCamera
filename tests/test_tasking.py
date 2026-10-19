from pathlib import Path

from portrait_eval.database import Database
from portrait_eval.tasking import TaskService


def test_task_state_is_persistent(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'db.sqlite'}")
    database.create_all()
    with database.session_factory() as session:
        service = TaskService(session)
        task = service.enqueue("project-1", "evaluate_project", {"project_id": "project-1"})
        assert task.status == "PENDING"
        service.mark_running(task.id)
        service.mark_succeeded(task.id, {"ok": True})
        restored = service.get(task.id)
        assert restored.status == "SUCCEEDED"
        assert restored.result == {"ok": True}


def test_claim_next_pending_is_atomic_and_counts_attempts(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'claim.db'}")
    database.create_all()
    with database.session_factory() as session:
        service = TaskService(session)
        queued = service.enqueue("project-1", "evaluate_project", {"project_id": "project-1"})
        claimed = service.claim_next_pending()
        assert claimed is not None
        assert claimed.id == queued.id
        assert claimed.status == "RUNNING"
        assert claimed.attempts == 1
    with database.session_factory() as session:
        assert TaskService(session).claim_next_pending() is None


def test_retryable_task_returns_to_pending_without_losing_attempt_count(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'retry.db'}")
    database.create_all()
    with database.session_factory() as session:
        service = TaskService(session)
        service.enqueue("project-1", "evaluate_project", {"project_id": "project-1"})
        claimed = service.claim_next_pending()
        assert claimed is not None
        retried = service.mark_retryable(claimed.id, "temporary model timeout")
        assert retried.status == "PENDING"
        assert retried.attempts == 1
        assert retried.error == "temporary model timeout"
        claimed_again = service.claim_next_pending()
        assert claimed_again is not None
        assert claimed_again.attempts == 2
