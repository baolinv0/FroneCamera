from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from portrait_eval.adjudication import adjudicate_claim
from portrait_eval.attribution import attribute_output_claim
from portrait_eval.bias import assess_capture_bias
from portrait_eval.corroboration import CorroborationAdapter, HeuristicCorroborationAdapter
from portrait_eval.hardware import build_hardware_queries
from portrait_eval.imaging import analyze_image, audit_scene
from portrait_eval.models import (
    ClaimCandidate,
    ModelEvaluationResult,
    ModelObservation,
    ProjectStatus,
)
from portrait_eval.reporting import ReportPayload, render_report_bundle
from portrait_eval.repository import Repository
from portrait_eval.research import (
    DisabledSearchProvider,
    SearchProvider,
    build_corroboration_queries,
    grade_source,
)
from portrait_eval.strategy import infer_cross_scene_claims
from portrait_eval.vlm import HeuristicVisionAdapter, VisionModelAdapter


def _anonymous_mapping(
    scene_id: str, device_ids: list[str]
) -> tuple[dict[str, str], dict[str, str]]:
    ordered = sorted(
        device_ids, key=lambda value: hashlib.sha256(f"{scene_id}:{value}".encode()).hexdigest()
    )
    forward = {device_id: chr(65 + index) for index, device_id in enumerate(ordered)}
    reverse = {code: device_id for device_id, code in forward.items()}
    return forward, reverse


def _restore_device_ids(
    result: ModelEvaluationResult, reverse: dict[str, str]
) -> ModelEvaluationResult:
    observations = [
        ModelObservation(
            device_id=reverse.get(item.device_id, item.device_id),
            dimension=item.dimension,
            statement=item.statement,
            evidence_refs=item.evidence_refs,
            certainty=item.certainty,
        )
        for item in result.observations
    ]
    return result.model_copy(update={"observations": observations})


class EvaluationPipeline:
    def __init__(
        self,
        session: Session,
        workspace: Path,
        primary: VisionModelAdapter | None = None,
        reviewer: VisionModelAdapter | None = None,
        search: SearchProvider | None = None,
        corroborator: CorroborationAdapter | None = None,
    ) -> None:
        self.session = session
        self.repo = Repository(session)
        self.workspace = workspace
        self.primary = primary or HeuristicVisionAdapter("primary")
        self.reviewer = reviewer or HeuristicVisionAdapter("reviewer")
        self.search = search or DisabledSearchProvider()
        self.corroborator = corroborator or HeuristicCorroborationAdapter()

    def run(self, project_id: str) -> dict[str, Any]:
        project = self.repo.get_project(project_id)
        if project.status != ProjectStatus.PAIRING_CONFIRMED.value:
            raise ValueError("Pairing must be confirmed before analysis")
        project.status = ProjectStatus.ANALYZING.value
        self.session.commit()
        pairing = self.repo.get_pairing(project_id)
        scene_results: list[dict[str, Any]] = []
        all_findings: list[dict[str, Any]] = []

        for group in pairing["groups"]:
            metrics: dict[str, dict[str, object]] = {}
            paths: dict[str, Path] = {}
            for device_id, cell in group["cells"].items():
                if cell is None:
                    continue
                path = Path(cell["path"])
                artifact_dir = (
                    self.workspace
                    / "projects"
                    / project_id
                    / "diagnostics"
                    / group["group_id"]
                    / device_id
                )
                result = analyze_image(path, artifact_dir)
                metrics[device_id] = result
                paths[device_id] = path
                self.repo.save_analysis(
                    project_id,
                    "image_metrics",
                    result,
                    scene_group_id=group["id"],
                    image_id=cell["image_id"],
                )

            audit = audit_scene(metrics)
            self.repo.save_analysis(project_id, "scene_audit", audit, scene_group_id=group["id"])
            if len(paths) < 2:
                self.repo.create_review_item(
                    project_id,
                    "scene_not_comparable",
                    {"group_id": group["group_id"], "audit": audit},
                    priority="high",
                )
                scene_results.append(
                    {"group_id": group["group_id"], "audit": audit, "metrics": metrics}
                )
                continue

            forward, reverse = _anonymous_mapping(group["group_id"], list(paths))
            anonymous_paths = {forward[device_id]: path for device_id, path in paths.items()}
            anonymous_metrics = {
                forward[device_id]: payload for device_id, payload in metrics.items()
            }
            self.repo.save_analysis(
                project_id,
                "anonymous_mapping",
                {"mapping": forward},
                scene_group_id=group["id"],
            )

            primary_visual_raw = self.primary.analyze(
                group["group_id"],
                anonymous_paths,
                {"audit": audit, "pass": "visual"},
            )
            primary_raw = self.primary.analyze(
                group["group_id"],
                anonymous_paths,
                {
                    "metrics": anonymous_metrics,
                    "audit": audit,
                    "pass": "metric_validation",
                    "visual_observation": primary_visual_raw.model_dump(mode="json"),
                },
            )
            reviewer_independent_raw = self.reviewer.analyze(
                group["group_id"],
                anonymous_paths,
                {"audit": audit, "pass": "independent_visual"},
            )
            reviewer_challenge_raw = self.reviewer.analyze(
                group["group_id"],
                anonymous_paths,
                {
                    "metrics": anonymous_metrics,
                    "audit": audit,
                    "pass": "challenge",
                    "primary": primary_raw.model_dump(mode="json"),
                    "reviewer_independent": reviewer_independent_raw.model_dump(mode="json"),
                },
            )
            primary_visual = _restore_device_ids(primary_visual_raw, reverse)
            primary = _restore_device_ids(primary_raw, reverse)
            reviewer_independent = _restore_device_ids(reviewer_independent_raw, reverse)
            reviewer_challenge = _restore_device_ids(reviewer_challenge_raw, reverse)
            self.repo.save_analysis(
                project_id,
                "primary_vlm_visual",
                primary_visual.model_dump(mode="json"),
                scene_group_id=group["id"],
            )
            self.repo.save_analysis(
                project_id,
                "primary_vlm",
                primary.model_dump(mode="json"),
                scene_group_id=group["id"],
            )
            self.repo.save_analysis(
                project_id,
                "reviewer_independent",
                reviewer_independent.model_dump(mode="json"),
                scene_group_id=group["id"],
            )
            self.repo.save_analysis(
                project_id,
                "reviewer_challenge",
                reviewer_challenge.model_dump(mode="json"),
                scene_group_id=group["id"],
            )

            scene_findings = self._adjudicate_scene(
                project_id,
                group["id"],
                group["group_id"],
                audit,
                primary,
                reviewer_independent,
                reviewer_challenge,
            )
            all_findings.extend(scene_findings)
            scene_results.append(
                {
                    "group_id": group["group_id"],
                    "audit": audit,
                    "metrics": metrics,
                    "primary_visual": primary_visual.model_dump(mode="json"),
                    "primary": primary.model_dump(mode="json"),
                    "reviewer_independent": reviewer_independent.model_dump(mode="json"),
                    "reviewer_challenge": reviewer_challenge.model_dump(mode="json"),
                    "findings": scene_findings,
                }
            )

        strategy_claims = infer_cross_scene_claims(scene_results)
        for claim in strategy_claims:
            finding = claim.model_dump(mode="json")
            finding["evidence"] = claim.supporting_scene_ids
            finding["claim_type"] = "strategy"
            all_findings.append(finding)
            self.repo.save_analysis(project_id, "strategy_claim", finding)
            if claim.grade.value == "C":
                self.repo.create_review_item(project_id, "insufficient_scene_coverage", finding)

        # Freeze internal results before external sources are retrieved to prevent media anchoring.
        self.repo.save_analysis(
            project_id,
            "internal_result_snapshot",
            {"scene_results": scene_results, "findings": all_findings},
        )

        hardware_context = self._hardware_research(pairing)
        self.repo.save_analysis(project_id, "hardware_research", hardware_context)

        external = self._external_corroboration(pairing, all_findings)
        self.repo.save_analysis(project_id, "external_corroboration", external)
        for item in external:
            if item.get("verdict") == "unresolved":
                self.repo.create_review_item(
                    project_id,
                    "external_corroboration_review",
                    item,
                    priority="medium",
                )
        if not external and any(item.get("claim_type") == "strategy" for item in all_findings):
            self.repo.create_review_item(
                project_id,
                "external_corroboration_unavailable",
                {
                    "message": (
                        "No independent professional-review evidence was retrieved. Keep strategy claims "
                        "scoped to the submitted captures until external corroboration or a controlled reshoot."
                    )
                },
            )

        audit_by_group = {scene["group_id"]: scene.get("audit", {}) for scene in scene_results}
        bias_assessments: list[dict[str, Any]] = []
        for finding in all_findings:
            scene_ids = finding.get("supporting_scene_ids", finding.get("evidence", []))
            comparable_count = sum(
                1
                for scene_id in scene_ids
                if audit_by_group.get(scene_id, {}).get("status") == "FULLY_COMPARABLE"
            )
            related_external = [
                item for item in external if item.get("claim_statement") == finding.get("statement")
            ]
            assessment = assess_capture_bias(
                supporting_scene_count=len(scene_ids),
                comparable_scene_count=comparable_count,
                independent_support_count=sum(
                    item.get("supports_claim") is True for item in related_external
                ),
                independent_contradiction_count=sum(
                    item.get("supports_claim") is False for item in related_external
                ),
            ).model_dump(mode="json")
            assessment["claim_statement"] = finding.get("statement")
            bias_assessments.append(assessment)
            if assessment["status"] == "RESHOOT_REQUIRED_FOR_GENERALIZATION":
                self.repo.create_review_item(
                    project_id, "possible_capture_bias", assessment, priority="high"
                )
        self.repo.save_analysis(project_id, "capture_bias_assessment", bias_assessments)

        attribution_cases: list[dict[str, Any]] = []
        device_exif: dict[str, dict[str, Any]] = {}
        for group in pairing["groups"]:
            for device_id, cell in group["cells"].items():
                if cell and cell.get("exif") and device_id not in device_exif:
                    device_exif[device_id] = dict(cell["exif"])
        hardware_by_device: dict[str, list[dict[str, Any]]] = {}
        for source in hardware_context:
            hardware_by_device.setdefault(str(source.get("device_id")), []).append(source)
        for finding in all_findings:
            if finding.get("claim_type") != "strategy":
                continue
            attribution = attribute_output_claim(
                finding["statement"],
                hardware_context={"sources": hardware_by_device.get(finding["device_id"], [])},
                exif_context=device_exif.get(finding["device_id"], {}),
            ).model_dump(mode="json")
            attribution["device_id"] = finding["device_id"]
            attribution["supporting_scene_ids"] = finding.get("supporting_scene_ids", [])
            attribution_cases.append(attribution)
            self.repo.create_review_item(project_id, "mechanism_attribution_review", attribution)
        self.repo.save_analysis(project_id, "attribution_case", attribution_cases)

        report_payload = ReportPayload(
            project_name=project.name,
            devices=[device["name"] for device in pairing["devices"]],
            findings=[item for item in all_findings if item["grade"] in {"A", "B"}],
            scene_results=scene_results,
            external_validation=external,
            hardware_context=hardware_context,
            attributions=attribution_cases,
            capture_bias=bias_assessments,
            limitations=[
                "Results apply to the submitted captures and firmware versions.",
                "Rendered JPEGs do not uniquely identify internal ISP mechanisms.",
                "External reviews are corroboration only and do not overwrite internal sample findings.",
                "Unknown scene labels and single captures reduce cross-scene generalization confidence.",
            ],
        )
        report_dir = self.workspace / "projects" / project_id / "reports"
        bundle = render_report_bundle(report_payload, report_dir, version="0.1", status="draft")
        self.repo.save_report(project_id, "0.1", "draft", str(bundle["html"]))
        review_count = len(
            [item for item in self.repo.list_review_items(project_id) if item["status"] == "open"]
        )
        project.status = (
            ProjectStatus.HUMAN_REVIEW_REQUIRED.value
            if review_count
            else ProjectStatus.REPORT_DRAFT_READY.value
        )
        self.session.commit()
        return {
            "status": project.status,
            "report_path": str(bundle["html"]),
            "json_path": str(bundle["json"]),
            "csv_path": str(bundle["csv"]),
            "scene_count": len(scene_results),
            "review_items": review_count,
        }

    def _adjudicate_scene(
        self,
        project_id: str,
        scene_group_id: str,
        group_key: str,
        audit: dict[str, object],
        primary: ModelEvaluationResult,
        reviewer: ModelEvaluationResult,
        challenge: ModelEvaluationResult,
    ) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for observation in primary.observations:
            independent_match = any(
                item.device_id == observation.device_id and item.dimension == observation.dimension
                for item in reviewer.observations
            )
            challenge_match = any(
                item.device_id == observation.device_id and item.dimension == observation.dimension
                for item in challenge.observations
            )
            model_agreement = (
                1.0
                if independent_match and challenge_match
                else 0.75
                if independent_match or challenge_match
                else 0.35
            )
            if not independent_match or not challenge_match:
                self.repo.create_review_item(
                    project_id,
                    "model_conflict",
                    {
                        "group_id": group_key,
                        "primary_observation": observation.model_dump(mode="json"),
                        "independent_match": independent_match,
                        "challenge_match": challenge_match,
                        "reviewer_independent": reviewer.model_dump(mode="json"),
                        "reviewer_challenge": challenge.model_dump(mode="json"),
                    },
                    priority="high" if not independent_match and not challenge_match else "medium",
                )
            claim = ClaimCandidate(
                device_id=observation.device_id,
                statement=observation.statement,
                claim_type="observation",
                supporting_scene_ids=[group_key],
                contradicting_scene_ids=[],
                model_agreement=model_agreement,
                objective_support=1.0,
                scene_validity=1.0 if audit["status"] == "FULLY_COMPARABLE" else 0.6,
                alternative_explanations=["capture variation", "framing difference"],
            )
            adjudicated = adjudicate_claim(claim)
            finding = adjudicated.model_dump(mode="json")
            finding["evidence"] = [group_key]
            finding["claim_type"] = "observation"
            findings.append(finding)
            self.repo.save_analysis(project_id, "claim", finding, scene_group_id=scene_group_id)
            if adjudicated.grade.value == "C":
                self.repo.create_review_item(project_id, "low_confidence_claim", finding)
        return findings

    def _hardware_research(self, pairing: dict[str, Any]) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for device in pairing["devices"]:
            device_name = device.get("canonical_model") or device["name"]
            for query in build_hardware_queries(device_name):
                for item in self.search.search(query, limit=3):
                    if item.url in seen_urls:
                        continue
                    seen_urls.add(item.url)
                    record = item.model_dump()
                    record.update(grade_source(item.source_domain, item.title, item.snippet))
                    record["device_id"] = device["id"]
                    record["device_name"] = device_name
                    record["query"] = query
                    evidence.append(record)
        return evidence

    def _external_corroboration(
        self,
        pairing: dict[str, Any],
        findings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        external: list[dict[str, Any]] = []
        devices = {item["id"]: item for item in pairing["devices"]}
        seen_pairs: set[tuple[str, str]] = set()
        for finding in findings:
            if finding.get("grade") not in {"A", "B"}:
                continue
            device = devices.get(finding["device_id"])
            if not device:
                continue
            device_name = device.get("canonical_model") or device["name"]
            for query in build_corroboration_queries(device_name, finding["statement"]):
                for item in self.search.search(query, limit=3):
                    key = (item.url, finding["statement"])
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    record = item.model_dump()
                    record.update(grade_source(item.source_domain, item.title, item.snippet))
                    try:
                        decision = self.corroborator.classify(
                            finding["statement"], device_name, item
                        )
                    except Exception as exc:
                        decision = HeuristicCorroborationAdapter().classify(
                            finding["statement"], device_name, item
                        )
                        record["classifier_warning"] = f"{type(exc).__name__}: {exc}"
                    record.update(decision.model_dump(mode="json"))
                    record["supports_claim"] = decision.supports_claim
                    record["query"] = query
                    record["claim_statement"] = finding["statement"]
                    record["device_id"] = finding["device_id"]
                    external.append(record)
        return external
