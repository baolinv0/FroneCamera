from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from portrait_eval.database import TaskRow, json_dump, json_load


@dataclass(frozen=True)
class TaskRecord:
    id: str
    project_id: str
    kind: str
    status: str
    payload: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    attempts: int


class TaskService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def enqueue(self, project_id: str, kind: str, payload: dict[str, Any]) -> TaskRecord:
        row = TaskRow(project_id=project_id, kind=kind, payload_json=json_dump(payload))
        self.session.add(row)
        self.session.commit()
        return self._record(row)

    def get(self, task_id: str) -> TaskRecord:
        row = self.session.get(TaskRow, task_id)
        if row is None:
            raise KeyError(task_id)
        return self._record(row)

    def next_pending(self) -> TaskRecord | None:
        row = self.session.scalar(
            select(TaskRow).where(TaskRow.status == "PENDING").order_by(TaskRow.created_at)
        )
        return self._record(row) if row else None

    def claim_next_pending(self) -> TaskRecord | None:
        while True:
            row = self.session.scalar(
                select(TaskRow).where(TaskRow.status == "PENDING").order_by(TaskRow.created_at)
            )
            if row is None:
                return None
            result = self.session.execute(
                update(TaskRow)
                .where(TaskRow.id == row.id, TaskRow.status == "PENDING")
                .values(status="RUNNING", attempt_count=TaskRow.attempt_count + 1)
            )
            self.session.commit()
            if result.rowcount == 1:
                claimed = self.session.get(TaskRow, row.id)
                if claimed is None:
                    raise RuntimeError("claimed task disappeared")
                return self._record(claimed)

    def mark_retryable(self, task_id: str, error: str) -> TaskRecord:
        return self._set(task_id, "PENDING", error=error)

    def mark_running(self, task_id: str) -> TaskRecord:
        return self._set(task_id, "RUNNING")

    def mark_succeeded(self, task_id: str, result: dict[str, Any]) -> TaskRecord:
        return self._set(task_id, "SUCCEEDED", result=result)

    def mark_failed(self, task_id: str, error: str) -> TaskRecord:
        return self._set(task_id, "FAILED", error=error)

    def _set(
        self,
        task_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> TaskRecord:
        row = self.session.get(TaskRow, task_id)
        if row is None:
            raise KeyError(task_id)
        row.status = status
        row.result_json = json_dump(result) if result is not None else row.result_json
        row.error = error
        self.session.commit()
        return self._record(row)

    @staticmethod
    def _record(row: TaskRow) -> TaskRecord:
        return TaskRecord(
            id=row.id,
            project_id=row.project_id,
            kind=row.kind,
            status=row.status,
            payload=cast(dict[str, Any], json_load(row.payload_json)),
            result=cast(dict[str, Any], json_load(row.result_json)) if row.result_json else None,
            error=row.error,
            attempts=row.attempt_count,
        )
