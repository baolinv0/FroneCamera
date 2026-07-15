from pathlib import Path

from PIL import Image

from portrait_eval.database import Database
from portrait_eval.repository import Repository


def test_confirm_pairing_creates_immutable_versioned_snapshot(tmp_path: Path) -> None:
    folders = []
    for device_name in ("a", "b"):
        folder = tmp_path / device_name
        folder.mkdir()
        Image.new("RGB", (16, 16), "gray").save(folder / "1.jpg")
        Image.new("RGB", (16, 16), "white").save(folder / "2.jpg")
        folders.append(folder)
    database = Database(f"sqlite:///{tmp_path / 'db.sqlite'}")
    database.create_all()
    with database.session_factory() as session:
        repo = Repository(session)
        project = repo.create_project("snapshot")
        devices = [repo.add_device(project.id, name, str(folder)) for name, folder in zip(("A", "B"), folders, strict=True)]
        pairing = repo.scan_and_pair(project.id)
        confirmed = repo.confirm_pairing(project.id, pairing["version"])
        snapshots = repo.list_pairing_snapshots(project.id)
        assert len(snapshots) == 1
        first = snapshots[0]
        assert first["version"] == confirmed["version"]
        original_image = first["payload"]["groups"][0]["cells"][devices[0].id]["image_id"]
        available = confirmed["available_images"][devices[0].id]
        changed = repo.update_pairing_cell(project.id, confirmed["groups"][0]["id"], devices[0].id, available[1]["image_id"], confirmed["version"])
        repo.confirm_pairing(project.id, changed["version"])
        snapshots = repo.list_pairing_snapshots(project.id)
        assert len(snapshots) == 2
        assert snapshots[1]["payload"]["groups"][0]["cells"][devices[0].id]["image_id"] == original_image
        assert snapshots[0]["version"] > snapshots[1]["version"]
