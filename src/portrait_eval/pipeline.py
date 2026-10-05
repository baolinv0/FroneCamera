from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, update
from sqlalchemy.orm import Session

from portrait_eval.adjudication import adjudicate_claim, stable_claim_id
from portrait_eval.attribution import attribute_output_claim
from portrait_eval.bias import assess_capture_bias
from portrait_eval.corroboration import CorroborationAdapter, HeuristicCorroborationAdapter
from portrait_eval.database import AnalysisRow, ProjectRow, ReportRow, ReviewItemRow
from portrait_eval.hardware import build_hardware_queries
from portrait_eval.imaging import analyze_image, audit_scene
from portrait_eval.iqa_bridge import (
    dimension_evidence,
    evaluate_scene,
    model_evidence_context,
    scene_evidence_audit,
)
from portrait_eval.model_validation import (
    anonymous_context,
    semantic_relation,
    validate_model_result,
)
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
from portrait_eval.run_binding import validate_run_binding
from portrait_eval.strategy import infer_cross_scene_claims
from portrait_eval.vlm import HeuristicVisionAdapter, VisionModelAdapter, adapter_identity


def _anonymous_mapping(
    scene_id: str, device_ids: list[str]
) -> tuple[dict[str, str], dict[str, str]]:
    ordered = sorted(
        device_ids, key=lambda value: hashlib.sha256(f"{scene_id}:{value}".encode()).hexdigest()
    )
    forward = {device_id: chr(65 + index) for index, device_id in enumerate(ordered)}
    reverse = {code: device_id for device_id, code in forward.items()}
    return forward, reverse


def _restore_evidence_ref(reference: str, reverse: dict[str, str]) -> str:
    parts = reference.split(":")
    if len(parts) >= 3 and parts[0] in {"asset", "metric", "iqa"}:
        parts[2] = reverse.get(parts[2], parts[2])
    return ":".join(parts)


def _restore_device_ids(
    result: ModelEvaluationResult, reverse: dict[str, str]
) -> ModelEvaluationResult:
    observations = [
        ModelObservation(
            claim_id=stable_claim_id(
                reverse.get(item.device_id, item.device_id),
                item.dimension,
                item.statement,
                [result.scene_id],
            ),
            device_id=reverse.get(item.device_id, item.device_id),
            dimension=item.dimension,
            statement=item.statement,
            evidence_refs=[
                _restore_evidence_ref(reference, reverse) for reference in item.evidence_refs
            ],
            certainty=item.certainty,
        )
        for item in result.observations
    ]
    return result.model_copy(update={"observations": observations})


class _EvaluationRepository(Repository):
    """Track only output rows created through this evaluator's repository."""

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        self.created_ids: dict[type[Any], set[str]] = {
            model: set() for model in (AnalysisRow, ReviewItemRow, ReportRow)
        }

    def save_analysis(
        self,
        project_id: str,
        kind: str,
        payload: object,
        scene_group_id: str | None = None,
        image_id: str | None = None,
    ) -> AnalysisRow:
        row = super().save_analysis(project_id, kind, payload, scene_group_id, image_id)
        self.created_ids[AnalysisRow].add(row.id)
        return row

    def create_review_item(
        self, project_id: str, category: str, payload: object, priority: str = "medium"
    ) -> ReviewItemRow:
        row = super().create_review_item(project_id, category, payload, priority)
        self.created_ids[ReviewItemRow].add(row.id)
        return row


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
        self.repo = _EvaluationRepository(session)
        self.workspace = workspace
        self.primary = primary or HeuristicVisionAdapter("primary")
        self.reviewer = reviewer or HeuristicVisionAdapter("reviewer")
        self.model_identities = {
            "primary": adapter_identity(self.primary),
            "reviewer": adapter_identity(self.reviewer),
        }
        self.independent_models = (
            all(self.model_identities.values())
            and self.model_identities["primary"] != self.model_identities["reviewer"]
        )
        self.search = search or DisabledSearchProvider()
        self.corroborator = corroborator or HeuristicCorroborationAdapter()

    def run(self, project_id: str) -> dict[str, Any]:
        project = self.repo.get_project(project_id)
        if project.status not in {
            ProjectStatus.PAIRING_CONFIRMED.value,
            ProjectStatus.FAILED.value,
        }:
            raise ValueError("Pairing must be confirmed before analysis")
        snapshots = self.repo.list_pairing_snapshots(project_id)
        if not snapshots or snapshots[0]["version"] != project.version:
            raise ValueError(
                "Confirmed pairing snapshot is missing or stale; confirm pairing again"
            )
        snapshot = snapshots[0]
        pairing = snapshot["payload"]
        # Evaluate the immutable snapshot, never mutable live pairing rows.
        checksums = {}
        for group in pairing["groups"]:
            for cell in group["cells"].values():
                if cell:
                    checksum = cell.get("checksum")
                    actual = hashlib.sha256(Path(cell["path"]).read_bytes()).hexdigest()
                    if not checksum or actual != checksum:
                        raise ValueError(
                            "Source bytes changed since scan; re-scan and confirm pairing"
                        )
                    checksums[cell["image_id"]] = checksum
        for ids in self.repo.created_ids.values():
            ids.clear()
        self.run_binding = {
            "attempt_id": str(uuid4()),
            "pairing_snapshot_id": snapshot["id"],
            "pairing_version": snapshot["version"],
            "pairing_sha256": hashlib.sha256(
                json.dumps(pairing, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "source_checksums": checksums,
            "model_identities": self.model_identities,
            "independent_models": bool(self.independent_models),
        }
        self.active_project_id = project_id
        self.repo.save_analysis(project_id, "evaluation_run_binding", self.run_binding)
        try:
            return self._run_confirmed(project_id, pairing)
        except Exception as exc:
            self.session.rollback()
            # Partial outputs must not look like a successful run, or leak into a retry.
            for model, ids in self.repo.created_ids.items():
                if ids:
                    self.session.execute(
                        delete(model).where(model.project_id == project_id, model.id.in_(ids))
                    )
            # A stale worker must not replace the user's new pairing-required state.
            self.session.execute(
                update(ProjectRow)
                .where(
                    ProjectRow.id == project_id,
                    ProjectRow.version == self.run_binding["pairing_version"],
                )
                .values(status=ProjectStatus.FAILED.value)
            )
            self.session.commit()
            self.repo.save_analysis(
                project_id,
                "evaluation_failure",
                {
                    **self.run_binding,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "model_input_traces": [
                        getattr(adapter, "last_input_trace", {})
                        for adapter in (self.primary, self.reviewer)
                    ],
                },
            )
            raise

    def _analyze(
        self,
        adapter: VisionModelAdapter,
        scene_id: str,
        paths: dict[str, Path],
        context: dict[str, Any],
    ) -> ModelEvaluationResult:
        # These dictionaries already carry anonymous image codes. A user device
        # named A/B must not relabel those codes while redacting other identities.
        identities = {
            identity: code
            for identity, code in getattr(self, "identities", {}).items()
            if identity not in paths
        }
        context = anonymous_context(context, identities)
        result = validate_model_result(
            adapter.analyze(scene_id, paths, context), scene_id, list(paths), context
        )
        # A heuristic/custom adapter cannot hide the declared fallback provenance.
        if isinstance(adapter, HeuristicVisionAdapter) or result.raw.get("adapter") in {
            "heuristic",
            "synthetic",
        }:
            result = result.model_copy(update={"provisional": True})
        return result

    def _run_confirmed(self, project_id: str, pairing: dict[str, Any]) -> dict[str, Any]:
        project = self.repo.get_project(project_id)
        self._set_run_status(project_id, ProjectStatus.ANALYZING.value)
        self.session.commit()
        scene_results: list[dict[str, Any]] = []
        all_findings: list[dict[str, Any]] = []
        if not self.independent_models:
            self.repo.create_review_item(
                project_id,
                "independent_model_evidence_unavailable",
                {
                    "model_identities": self.model_identities,
                    "message": "Shared or unknown model identities provide no independent model consensus evidence.",
                },
            )

        for group in pairing["groups"]:
            metrics: dict[str, dict[str, object]] = {}
            paths: dict[str, Path] = {}
            for device_id, cell in group["cells"].items():
                if cell is None:
                    continue
                path = Path(cell["path"])
                if hashlib.sha256(path.read_bytes()).hexdigest() != cell["checksum"]:
                    raise ValueError(
                        "Source bytes changed during evaluation; re-scan and confirm pairing"
                    )
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

            iqa_evaluation = evaluate_scene(group["group_id"], paths, metrics)
            self.repo.save_analysis(
                project_id, "iqa_evaluation", iqa_evaluation, scene_group_id=group["id"]
            )
            audit = audit_scene(metrics)
            iqa_audit = scene_evidence_audit(iqa_evaluation, list(paths))
            audit["iqa_evidence"] = iqa_audit
            if not iqa_audit["valid"]:
                audit["status"] = "NOT_COMPARABLE"
                existing_warnings = audit.get("warnings", [])
                audit["warnings"] = (
                    existing_warnings if isinstance(existing_warnings, list) else []
                ) + iqa_audit["warnings"]
            self.repo.save_analysis(project_id, "scene_audit", audit, scene_group_id=group["id"])
            if len(paths) < 2 or audit["status"] == "NOT_COMPARABLE":
                self.repo.create_review_item(
                    project_id,
                    "scene_not_comparable",
                    {"group_id": group["group_id"], "audit": audit},
                    priority="high",
                )
                scene_results.append(
                    {
                        "group_id": group["group_id"],
                        "audit": audit,
                        "metrics": metrics,
                        "iqa_evaluation": iqa_evaluation,
                    }
                )
                continue

            forward, reverse = _anonymous_mapping(group["group_id"], list(paths))
            self.identities = {**forward}
            for device in pairing["devices"]:
                if device["id"] in forward:
                    for identity in (device["name"], device.get("canonical_model")):
                        if identity:
                            self.identities[identity] = forward[device["id"]]
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

            primary_visual_raw = self._analyze(
                self.primary,
                group["group_id"],
                anonymous_paths,
                {
                    "audit": audit,
                    "pass": "visual",
                    "iqa_evidence": model_evidence_context(iqa_evaluation, measurements=False),
                },
            )
            primary_raw = self._analyze(
                self.primary,
                group["group_id"],
                anonymous_paths,
                {
                    "metrics": anonymous_metrics,
                    "audit": audit,
                    "pass": "metric_validation",
                    "iqa_evidence": model_evidence_context(iqa_evaluation, measurements=True),
                    "visual_observation": primary_visual_raw.model_dump(mode="json"),
                },
            )
            reviewer_independent_raw = self._analyze(
                self.reviewer,
                group["group_id"],
                anonymous_paths,
                {
                    "audit": audit,
                    "pass": "independent_visual",
                    "iqa_evidence": model_evidence_context(iqa_evaluation, measurements=False),
                },
            )
            reviewer_challenge_raw = self._analyze(
                self.reviewer,
                group["group_id"],
                anonymous_paths,
                {
                    "metrics": anonymous_metrics,
                    "audit": audit,
                    "pass": "challenge",
                    "iqa_evidence": model_evidence_context(iqa_evaluation, measurements=True),
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
                metrics,
                iqa_evaluation,
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
                    "iqa_evaluation": iqa_evaluation,
                    "provisional": primary.provisional
                    or reviewer_independent.provisional
                    or reviewer_challenge.provisional,
                }
            )

        validate_run_binding(self.repo, project_id, self.run_binding)
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
            assessment["claim_id"] = finding["claim_id"]
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
            attribution["claim_id"] = finding["claim_id"]
            attribution["device_id"] = finding["device_id"]
            attribution["supporting_scene_ids"] = finding.get("supporting_scene_ids", [])
            attribution_cases.append(attribution)
            self.repo.create_review_item(project_id, "mechanism_attribution_review", attribution)
        self.repo.save_analysis(project_id, "attribution_case", attribution_cases)

        device_names = {device["id"]: device["name"] for device in pairing["devices"]}
        visual_assets: list[dict[str, Any]] = []
        for group in pairing["groups"]:
            selected_cells = [
                (device_id, cell) for device_id, cell in group["cells"].items() if cell is not None
            ][:6]
            visual_assets.extend(
                {
                    "scene_id": group["group_id"],
                    "device_id": device_id,
                    "device_name": device_names.get(device_id, device_id),
                    "path": cell["path"],
                    "filename": cell["filename"],
                }
                for device_id, cell in selected_cells
            )
        device_profiles = []
        for device in pairing["devices"]:
            strategy_findings = [
                item["statement"]
                for item in all_findings
                if item.get("device_id") == device["id"]
                and item.get("claim_type") == "strategy"
                and item.get("grade") in {"A", "B"}
            ]
            device_profiles.append(
                {
                    "device_id": device["id"],
                    "device_name": device["name"],
                    "label": "Evidence Profile",
                    "claim_ids": [
                        item["claim_id"]
                        for item in all_findings
                        if item.get("device_id") == device["id"]
                        and item.get("claim_type") == "strategy"
                        and item.get("grade") in {"A", "B"}
                    ],
                    "summary": "；".join(strategy_findings[:2])
                    or "需要更多跨场景证据形成稳定画像。",
                }
            )

        report_payload = ReportPayload(
            project_name=project.name,
            pairing_snapshot_id=self.run_binding["pairing_snapshot_id"],
            input_trace={
                "run_binding": self.run_binding,
                "models": [
                    {
                        "scene_id": scene["group_id"],
                        "pass": key,
                        "input_trace": scene[key].get("input_trace", {}),
                    }
                    for scene in scene_results
                    for key in (
                        "primary_visual",
                        "primary",
                        "reviewer_independent",
                        "reviewer_challenge",
                    )
                    if key in scene
                ],
            },
            provenance_notes=(
                [
                    "Heuristic/synthetic scene observations are provisional and require real model or human review."
                ]
                if any(scene.get("provisional") for scene in scene_results)
                else []
            )
            + (
                [
                    "Primary and reviewer share or lack verified model identity; their agreement is not independent model evidence."
                ]
                if not self.independent_models
                else []
            ),
            devices=[device["name"] for device in pairing["devices"]],
            findings=[item for item in all_findings if item["grade"] in {"A", "B"}],
            scene_results=scene_results,
            visual_assets=visual_assets,
            device_profiles=device_profiles,
            methodology_notes=[
                "同一场景内先进行匿名横向比较，再恢复设备身份。",
                "客观指标、主模型、独立审阅和挑战审阅共同进入裁决。",
                "只把 A/B 级证据写入正式结论，C 级保留为复核线索。",
                "Model roles and repeated passes are not independent model identities; shared or unknown model identities provide no independent consensus credit."
                if not self.independent_models
                else "Primary and reviewer have distinct configured model identities; agreement still requires matching semantics, evidence and uncertainty.",
            ],
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
        validate_run_binding(self.repo, project_id, self.run_binding)
        bundle = render_report_bundle(report_payload, report_dir, version="0.1", status="draft")
        review_count = len(
            [item for item in self.repo.list_review_items(project_id) if item["status"] == "open"]
        )
        status = (
            ProjectStatus.HUMAN_REVIEW_REQUIRED.value
            if review_count
            else ProjectStatus.REPORT_DRAFT_READY.value
        )
        validate_run_binding(self.repo, project_id, self.run_binding)
        self._set_run_status(project_id, status)
        # Publish the draft row and its project state in the same transaction.
        draft = ReportRow(
            id=str(uuid4()),
            project_id=project_id,
            version="0.1",
            status="draft",
            html_path=str(bundle["html"]),
        )
        self.repo.created_ids[ReportRow].add(draft.id)
        self.session.add(draft)
        self.session.commit()
        return {
            "status": status,
            "report_path": str(bundle["html"]),
            "json_path": str(bundle["json"]),
            "csv_path": str(bundle["csv"]),
            "scene_count": len(scene_results),
            "review_items": review_count,
        }

    def _set_run_status(self, project_id: str, status: str) -> None:
        changed = self.session.execute(
            update(ProjectRow)
            .where(
                ProjectRow.id == project_id,
                ProjectRow.version == self.run_binding["pairing_version"],
            )
            .values(status=status)
            .returning(ProjectRow.id)
        ).scalar_one_or_none()
        if changed is None:
            raise ValueError("Pairing changed during evaluation; confirm the new pairing snapshot")

    def _adjudicate_scene(
        self,
        project_id: str,
        scene_group_id: str,
        group_key: str,
        audit: dict[str, object],
        primary: ModelEvaluationResult,
        reviewer: ModelEvaluationResult,
        challenge: ModelEvaluationResult,
        metrics: dict[str, dict[str, Any]] | None = None,
        iqa_evaluation: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for observation in primary.observations:
            claim_id = observation.claim_id or stable_claim_id(
                observation.device_id, observation.dimension, observation.statement, [group_key]
            )

            def decision(
                model: ModelEvaluationResult, observation: ModelObservation = observation
            ) -> tuple[str, float]:
                matching = [
                    item
                    for item in model.observations
                    if item.device_id == observation.device_id
                    and item.dimension == observation.dimension
                ]
                # Only the same semantic text and declared evidence support an observation.
                # An unstructured difference is unknown, never presumed agreement.
                exact = [
                    item
                    for item in matching
                    if " ".join(item.statement.casefold().split())
                    == " ".join(observation.statement.casefold().split())
                    and item.evidence_refs
                ]
                if any(
                    semantic_relation(observation.statement, item.statement) == "opposition"
                    for item in matching
                ):
                    return "opposition", 0.0
                if exact and len(exact) == len(matching):
                    return "support", min(min(item.certainty for item in exact), model.confidence)
                return "unknown", 0.0

            independent_decision, independent_strength = decision(reviewer)
            challenge_decision, challenge_strength = decision(challenge)
            independent_match = independent_decision == "support"
            challenge_match = challenge_decision == "support"
            model_agreement = (
                min(
                    observation.certainty,
                    primary.confidence,
                    independent_strength,
                    challenge_strength,
                )
                if self.independent_models and observation.evidence_refs
                else 0.0
            )
            if not independent_match or not challenge_match:
                self.repo.create_review_item(
                    project_id,
                    "model_conflict",
                    {
                        "claim_id": claim_id,
                        "group_id": group_key,
                        "primary_observation": observation.model_dump(mode="json"),
                        "independent_match": independent_match,
                        "challenge_match": challenge_match,
                        "independent_decision": independent_decision,
                        "challenge_decision": challenge_decision,
                        "independent_models": bool(self.independent_models),
                        "model_identities": self.model_identities,
                        "reviewer_independent": reviewer.model_dump(mode="json"),
                        "reviewer_challenge": challenge.model_dump(mode="json"),
                    },
                    priority="high" if not independent_match and not challenge_match else "medium",
                )
            # Text with an asset citation does not establish quantitative objective support.
            objective_support = 0.0
            device_metrics = (metrics or {}).get(observation.device_id, {}).get("whole", {})
            if (
                observation.evidence_refs
                and observation.dimension == "global_exposure"
                and observation.statement
                == "This output has the highest display-referred mean luminance in the matched group."
            ):
                value = device_metrics.get("luma_mean")
                other_values = [
                    payload.get("whole", {}).get("luma_mean")
                    for payload in (metrics or {}).values()
                ]
                if (
                    value is not None
                    and all(current is not None for current in other_values)
                    and value == max(other_values)
                ):
                    objective_support = 1.0
            opposition = independent_decision == "opposition" or challenge_decision == "opposition"
            observed_evidence = (
                dimension_evidence(iqa_evaluation, observation.device_id, observation.dimension)
                if iqa_evaluation is not None
                else None
            )
            uncertainty = min(observation.certainty, primary.confidence)
            if observed_evidence is not None and observed_evidence.get("state") != "measured_proxy":
                model_agreement = objective_support = uncertainty = 0.0
                self.repo.create_review_item(
                    project_id,
                    "iqa_dimension_unobservable",
                    {
                        "claim_id": claim_id,
                        "device_id": observation.device_id,
                        "dimension": observation.dimension,
                        "group_id": group_key,
                        "iqa_evidence": observed_evidence,
                    },
                )
            if opposition:
                objective_support = 0.0
            claim = ClaimCandidate(
                claim_id=claim_id,
                dimension=observation.dimension,
                device_id=observation.device_id,
                statement=observation.statement,
                claim_type="observation",
                evidence_refs=observation.evidence_refs,
                provisional=primary.provisional or reviewer.provisional or challenge.provisional,
                uncertainty=uncertainty,
                supporting_scene_ids=[group_key],
                contradicting_scene_ids=[group_key] if opposition else [],
                model_agreement=model_agreement,
                objective_support=objective_support,
                scene_validity=1.0
                if audit["status"] == "FULLY_COMPARABLE"
                else 0.5
                if audit["status"] == "COMPARABLE_WITH_CONFOUNDERS"
                else 0.0,
                alternative_explanations=["capture variation", "framing difference"],
            )
            adjudicated = adjudicate_claim(claim)
            finding = adjudicated.model_dump(mode="json")
            finding["evidence"] = [group_key]
            finding["model_agreement"] = model_agreement
            finding["uncertainty"] = uncertainty
            finding["objective_support"] = objective_support
            finding["iqa_evidence"] = observed_evidence
            finding["independent_models"] = bool(self.independent_models)
            finding["model_identities"] = self.model_identities
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
            if finding.get("grade") not in {"A", "B"} and not (
                finding.get("claim_type") == "strategy" and finding.get("status") == "PROVISIONAL"
            ):
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
                    record["claim_id"] = finding["claim_id"]
                    record["claim_statement"] = finding["statement"]
                    record["device_id"] = finding["device_id"]
                    external.append(record)
        return external
