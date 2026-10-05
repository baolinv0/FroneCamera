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


def test_recursive_privacy_retains_pairing_review_snapshot_and_trace(tmp_path: Path) -> None:
    import json

    from portrait_eval.exporting import sanitize_export

    payload = {
        "matching_confidence": 0.6,
        "review_required": True,
        "match_notes": ["Check grouping"],
        "snapshot_id": "s1",
        "input_trace": {"sha256": "abc", "encoded_path": "/private/encoded.jpg"},
        "nested": [
            {
                "diagnostic_path": "/secret/diagnostic.png",
                "error": "failed reading /secret/input.jpg",
                "windows": r"C:\secret\photo.jpg",
            }
        ],
        "review_history": [{"status": "rejected", "note": "Verified"}],
        "asset": "assets/001.jpg",
        "url": "https://example.com/evidence",
    }
    text = json.dumps(sanitize_export(payload))
    assert "/secret/" not in text and "/private/" not in text and "C:" not in text
    restored = json.loads(text)
    assert restored["snapshot_id"] == "s1"
    assert restored["matching_confidence"] == 0.6
    assert restored["review_required"] is True
    assert restored["input_trace"]["sha256"] == "abc"
    assert restored["review_history"][0]["note"] == "Verified"
    assert restored["asset"] == "assets/001.jpg"
    assert restored["url"] == "https://example.com/evidence"


def test_actual_zip_preserves_snapshot_and_redacts_nested_analysis(tmp_path: Path) -> None:
    import json

    from PIL import Image

    database = Database(f"sqlite:///{tmp_path / 'privacy.sqlite'}")
    database.create_all()
    with database.session_factory() as session:
        repo = Repository(session)
        project = repo.create_project("privacy")
        for name in ("a", "b"):
            folder = tmp_path / name
            folder.mkdir()
            Image.new("RGB", (16, 16), "gray").save(folder / "1.jpg")
            repo.add_device(project.id, name, str(folder))
        pairing = repo.scan_and_pair(project.id)
        repo.confirm_pairing(project.id, pairing["version"])
        repo.save_analysis(
            project.id,
            "evaluation_run_binding",
            {
                "snapshot_id": "snapshot-trace",
                "trace": {"hash": "abc", "diagnostic_path": str(tmp_path / "private.jpg")},
            },
        )
        review = repo.create_review_item(
            project.id,
            "conflict",
            {
                "claim_id": "c1",
                "review_history": [{"status": "open", "note": "before"}],
                "source_path": str(tmp_path / "raw.jpg"),
            },
        )
        repo.resolve_review_item(review.id, "rejected", "checked")
        output = export_project(repo, project.id, tmp_path / "workspace")
    with zipfile.ZipFile(output) as archive:
        documents = {name: archive.read(name).decode() for name in archive.namelist()}
        assert "pairing-snapshots.json" in documents
        assert str(tmp_path) not in "\n".join(documents.values())
        manifest = json.loads(documents["manifest.json"])
        group = manifest["pairing"]["groups"][0]
        assert (
            "matching_confidence" in group and "review_required" in group and "match_notes" in group
        )
        assert json.loads(documents["pairing-snapshots.json"])[0]["payload"]["confirmed"] is True
        analysis = json.loads(documents["analysis.json"])[0]["payload"]
        assert analysis["snapshot_id"] == "snapshot-trace" and analysis["trace"]["hash"] == "abc"
        review = json.loads(documents["reviews.json"])[0]
        assert review["payload"]["review_history"][0]["note"] == "before"
        assert review["payload"]["review_note"] == "checked"


def test_actual_zip_redacts_embedded_unc_and_keeps_error_provenance(tmp_path: Path) -> None:
    import json

    database = Database(f"sqlite:///{tmp_path / 'unc.sqlite'}")
    database.create_all()
    unc = r"\\secret-server\private-share\capture.jpg"
    with database.session_factory() as session:
        repo = Repository(session)
        project = repo.create_project("unc-privacy")
        repo.save_analysis(
            project.id,
            "evaluation_error",
            {
                "error": f"Cannot open {unc}",
                "nested": [
                    {"error": r"Cannot open C:\private\photo.jpg"},
                    {"error": "Cannot open /private/photo.jpg"},
                ],
                "attempt_id": "attempt-1",
                "input_sha256": "abc",
            },
        )
        output = export_project(repo, project.id, tmp_path / "workspace")
    with zipfile.ZipFile(output) as archive:
        analysis = json.loads(archive.read("analysis.json"))[0]["payload"]
        serialized = json.dumps(analysis)
        assert "secret-server" not in serialized and "private-share" not in serialized
        assert "C:" not in serialized and "/private/" not in serialized
        assert analysis["error"].startswith("Cannot open ")
        assert analysis["attempt_id"] == "attempt-1" and analysis["input_sha256"] == "abc"
