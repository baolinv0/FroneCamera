from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime

import pytest

from portrait_eval.core.models_v2 import EvaluationBatchV2
from portrait_eval.database import Database
from portrait_eval.repository import Repository
from portrait_eval.repository_v2 import V2Repository


def _evaluation_batch(project_id: str) -> EvaluationBatchV2:
    return EvaluationBatchV2(
        batch_id="batch-v2-001",
        project_id=project_id,
        device_ids=["device-a", "device-b"],
        dataset_version="dataset-2026-07",
        dimension_policy_version="dimension-policy-v1",
        scoring_policy_version="scoring-policy-v1",
        primary_model_version="primary-v1",
        judge_model_version="judge-v1",
        report_model_version=None,
        created_at=datetime(2026, 7, 16, 9, 0, tzinfo=UTC),
    )


def test_database_create_all_registers_v2_tables_without_repository_import() -> None:
    code = """
from sqlalchemy import inspect
from portrait_eval.database import Database

database = Database("sqlite+pysqlite:///:memory:")
database.create_all()
assert "evaluation_batches_v2" in inspect(database.engine).get_table_names()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_evaluation_batch_round_trips_without_changing_legacy_project() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_all()

    with database.session_factory() as session:
        legacy_repository = Repository(session)
        project = legacy_repository.create_project("Persistence v2 canary")
        repository = V2Repository(session)
        batch = _evaluation_batch(project.id)

        repository.save_evaluation_batch(batch)

        assert repository.get_evaluation_batch(batch.batch_id) == batch
        assert legacy_repository.get_project(project.id).name == "Persistence v2 canary"


def test_evaluation_batch_rejects_duplicate_batch_id() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_all()

    with database.session_factory() as session:
        project = Repository(session).create_project("Duplicate batch canary")
        repository = V2Repository(session)
        batch = _evaluation_batch(project.id)
        repository.save_evaluation_batch(batch)

        with pytest.raises(ValueError, match="batch already exists"):
            repository.save_evaluation_batch(batch)
