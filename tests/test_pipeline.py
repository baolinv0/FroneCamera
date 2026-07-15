from pathlib import Path

from PIL import Image

from portrait_eval.database import Database
from portrait_eval.pipeline import EvaluationPipeline
from portrait_eval.repository import Repository
from portrait_eval.workflow import run_evaluation_workflow


def test_pipeline_generates_metrics_claims_and_report(tmp_path: Path) -> None:
    folders = []
    for index, level in enumerate((40, 180)):
        folder = tmp_path / f"device-{index}"
        folder.mkdir()
        Image.new("RGB", (96, 96), (level, level, level)).save(folder / "1.jpg")
        folders.append(folder)
    database = Database(f"sqlite:///{tmp_path / 'db.sqlite'}")
    database.create_all()
    with database.session_factory() as session:
        repo = Repository(session)
        project = repo.create_project("demo")
        for index, folder in enumerate(folders):
            repo.add_device(project.id, f"Device {index}", str(folder))
        pairing = repo.scan_and_pair(project.id)
        repo.confirm_pairing(project.id, pairing["version"])
        result = EvaluationPipeline(session, tmp_path / "workspace").run(project.id)
        assert result["scene_count"] == 1
        assert Path(result["report_path"]).exists()
        assert repo.list_analysis(project.id, "image_metrics")
        assert repo.list_analysis(project.id, "claim")


def test_pipeline_quick_mode_creates_final_report_without_blocking_on_reviews(tmp_path: Path) -> None:
    folders = []
    for index, level in enumerate((45, 175)):
        folder = tmp_path / f"quick-{index}"
        folder.mkdir()
        Image.new("RGB", (96, 96), (level, level, level)).save(folder / "1.jpg")
        folders.append(folder)
    database = Database(f"sqlite:///{tmp_path / 'quick.sqlite'}")
    database.create_all()
    with database.session_factory() as session:
        repo = Repository(session)
        project = repo.create_project("quick")
        for index, folder in enumerate(folders):
            repo.add_device(project.id, f"Device {index}", str(folder))
        pairing = repo.scan_and_pair(project.id)
        repo.confirm_pairing(project.id, pairing["version"])
        result = run_evaluation_workflow(
            session,
            tmp_path / "workspace",
            project.id,
            mode="quick",
        )
        reports = repo.list_reports(project.id)
        assert result["mode"] == "quick"
        assert result["status"] == "REPORT_FINALIZED"
        assert result["report_id"] == reports[0]["id"]
        assert reports[0]["status"] == "final"
        assert Path(reports[0]["html_path"]).exists()


def test_pipeline_classifies_external_professional_corroboration(tmp_path: Path) -> None:
    from portrait_eval.corroboration import CorroborationAdapter, CorroborationDecision
    from portrait_eval.models import ExternalEvidence
    from portrait_eval.research import SearchProvider

    class SearchStub(SearchProvider):
        def search(self, query: str, limit: int = 5) -> list[ExternalEvidence]:
            return [ExternalEvidence(title="Device 1 selfie camera review", url=f"https://example.test/{abs(hash(query))}", source_domain="example.test", snippet="The front camera keeps the face bright in matched scenes.")]

    class CorroborationStub(CorroborationAdapter):
        def classify(self, claim: str, device_name: str, evidence: ExternalEvidence) -> CorroborationDecision:
            return CorroborationDecision(verdict="supports", confidence=0.9, reason="Direct front-camera evidence matches the output tendency.", front_camera_specific=True)

    folders = []
    for index, level in enumerate((40, 180)):
        folder = tmp_path / f"device-{index}"
        folder.mkdir()
        for scene in ("1.jpg", "2.jpg"):
            Image.new("RGB", (96, 96), (level, level, level)).save(folder / scene)
        folders.append(folder)
    database = Database(f"sqlite:///{tmp_path / 'corroboration.sqlite'}")
    database.create_all()
    with database.session_factory() as session:
        repo = Repository(session)
        project = repo.create_project("corroboration")
        for index, folder in enumerate(folders):
            repo.add_device(project.id, f"Device {index}", str(folder), f"Phone {index}")
        pairing = repo.scan_and_pair(project.id)
        repo.confirm_pairing(project.id, pairing["version"])
        EvaluationPipeline(session, tmp_path / "workspace", search=SearchStub(), corroborator=CorroborationStub()).run(project.id)
        external = repo.list_analysis(project.id, "external_corroboration")[0]["payload"]
        assert any(item["supports_claim"] is True for item in external)
        assert any(item["verdict"] == "supports" for item in external)


def test_pipeline_runs_blind_visual_then_metric_validation_protocol(tmp_path: Path) -> None:
    from portrait_eval.models import ModelEvaluationResult
    from portrait_eval.vlm import VisionModelAdapter

    class RecordingAdapter(VisionModelAdapter):
        def __init__(self, role: str) -> None:
            self.role = role
            self.contexts: list[dict[str, object]] = []
        def analyze(self, scene_id, image_paths, context):  # type: ignore[no-untyped-def]
            self.contexts.append(context)
            return ModelEvaluationResult(scene_id=scene_id, role=self.role, observations=[], confidence=0.8)

    folders = []
    for index, level in enumerate((80, 120)):
        folder = tmp_path / f"protocol-{index}"
        folder.mkdir()
        Image.new("RGB", (96, 96), (level, level, level)).save(folder / "1.jpg")
        folders.append(folder)
    database = Database(f"sqlite:///{tmp_path / 'protocol.sqlite'}")
    database.create_all()
    primary = RecordingAdapter("primary")
    reviewer = RecordingAdapter("reviewer")
    with database.session_factory() as session:
        repo = Repository(session)
        project = repo.create_project("protocol")
        for index, folder in enumerate(folders):
            repo.add_device(project.id, f"Device {index}", str(folder))
        pairing = repo.scan_and_pair(project.id)
        repo.confirm_pairing(project.id, pairing["version"])
        EvaluationPipeline(session, tmp_path / "workspace", primary=primary, reviewer=reviewer).run(project.id)
    assert [context["pass"] for context in primary.contexts] == ["visual", "metric_validation"]
    assert "metrics" not in primary.contexts[0]
    assert [context["pass"] for context in reviewer.contexts] == ["independent_visual", "challenge"]
    assert "primary" not in reviewer.contexts[0]


def test_pipeline_creates_review_item_when_reviewer_challenges_primary_observation(tmp_path: Path) -> None:
    from portrait_eval.models import ModelEvaluationResult, ModelObservation
    from portrait_eval.vlm import VisionModelAdapter

    class PrimaryAdapter(VisionModelAdapter):
        def analyze(self, scene_id, image_paths, context):  # type: ignore[no-untyped-def]
            observations = []
            if context["pass"] == "metric_validation":
                device_code = sorted(image_paths)[0]
                observations = [ModelObservation(device_id=device_code, dimension="highlight_retention", statement="This output has more retained highlight structure.", evidence_refs=[f"asset:{scene_id}:{device_code}"], certainty=0.8)]
            return ModelEvaluationResult(scene_id=scene_id, role="primary", observations=observations, confidence=0.8)

    class ReviewerAdapter(VisionModelAdapter):
        def analyze(self, scene_id, image_paths, context):  # type: ignore[no-untyped-def]
            return ModelEvaluationResult(scene_id=scene_id, role="reviewer", observations=[], confidence=0.8)

    folders = []
    for index, level in enumerate((80, 120)):
        folder = tmp_path / f"conflict-{index}"
        folder.mkdir()
        Image.new("RGB", (96, 96), (level, level, level)).save(folder / "1.jpg")
        folders.append(folder)
    database = Database(f"sqlite:///{tmp_path / 'conflict.sqlite'}")
    database.create_all()
    with database.session_factory() as session:
        repo = Repository(session)
        project = repo.create_project("conflict")
        for index, folder in enumerate(folders):
            repo.add_device(project.id, f"Device {index}", str(folder))
        pairing = repo.scan_and_pair(project.id)
        repo.confirm_pairing(project.id, pairing["version"])
        EvaluationPipeline(session, tmp_path / "workspace", primary=PrimaryAdapter(), reviewer=ReviewerAdapter()).run(project.id)
        categories = {item["category"] for item in repo.list_review_items(project.id)}
        assert "model_conflict" in categories
