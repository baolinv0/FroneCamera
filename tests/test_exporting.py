import zipfile
from pathlib import Path

from portrait_eval.database import Database
from portrait_eval.exporting import export_project
from portrait_eval.repository import Repository


def test_export_omits_original_images_and_absolute_paths(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'db.sqlite'}")
    database.create_all()
    with database.session_factory() as session:
        repo = Repository(session)
        project = repo.create_project("demo")
        output = export_project(repo, project.id, tmp_path / "workspace")
    with zipfile.ZipFile(output) as archive:
        assert "manifest.json" in archive.namelist()
        assert not any(
            name.lower().endswith((".jpg", ".png", ".heic")) for name in archive.namelist()
        )
