from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from sqlalchemy.orm import Session

from portrait_eval.adjudication import adjudicate_claim
from portrait_eval.annotations import latest_scene_annotation
from portrait_eval.attribution import attribute_output_claim
from portrait_eval.bias import assess_capture_bias
from portrait_eval.corroboration import CorroborationAdapter, HeuristicCorroborationAdapter
from portrait_eval.evidence import (
    claim_identity,
    measure_observation,
    measured_statement,
    related_external,
    review_relation,
)
from portrait_eval.hardware import build_hardware_queries
from portrait_eval.imaging import analyze_image, audit_scene
from portrait_eval.models import (
    ClaimCandidate,
    ModelEvaluationResult,
    ProjectStatus,
)
from portrait_eval.report_model import challenge_matrix, device_profiles
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
        item.model_copy(
            update={
                "device_id": reverse.get(item.device_id, item.device_id),
                "comparison": reverse.get(item.comparison, item.comparison),
                "evidence_refs": [_restore_reference(ref, reverse) for ref in item.evidence_refs],
            }
        )
        for item in result.observations
    ]
    viewed = [
        {
            **item,
            "evidence_id": _restore_reference(item.get("evidence_id", ""), reverse),
            "device_id": reverse.get(
                item.get("anonymous_device", ""), item.get("anonymous_device", "unknown")
            ),
        }
        for item in result.viewed_evidence
    ]
    return result.model_copy(update={"observations": observations, "viewed_evidence": viewed})


def _restore_reference(reference: str, reverse: dict[str, str]) -> str:
    parts = reference.split(":")
    if len(parts) >= 3 and parts[0] in {"asset", "metric", "crop"}:
        parts[2] = reverse.get(parts[2], parts[2])
    return ":".join(parts)


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
        snapshots = self.repo.list_pairing_snapshots(project_id)
        frozen = next(
            (row["payload"] for row in snapshots if row["version"] == project.version), None
        )
        if frozen is None:
            raise ValueError("Confirmed pairing snapshot is missing; confirm the pairing again")
        project.status = ProjectStatus.ANALYZING.value
        self.session.commit()
        pairing = frozen
        claim_version = f"0.2:{pairing['version']}:{uuid4().hex[:12]}"
        self.claim_version = claim_version
        run_key = claim_version.rsplit(":", 1)[-1]
        asset_manifest = []
        scene_results: list[dict[str, Any]] = []
        all_findings: list[dict[str, Any]] = []

        for group in pairing["groups"]:
            annotation = latest_scene_annotation(self.repo, project_id, group["group_id"])
            if annotation.get("expected_pairing_version") != pairing["version"]:
                annotation = {}
            metrics: dict[str, dict[str, Any]] = {}
            paths: dict[str, Path] = {}
            for device_id, cell in group["cells"].items():
                if cell is None:
                    asset_manifest.append(
                        {"scene_id": group["group_id"], "device_id": device_id, "status": "missing"}
                    )
                    continue
                path = Path(cell["path"])
                artifact_dir = (
                    self.workspace
                    / "projects"
                    / project_id
                    / "diagnostics"
                    / run_key
                    / group["group_id"]
                    / device_id
                )
                face_annotations = annotation.get("faces", {}).get(device_id)
                result = analyze_image(path, artifact_dir, face_annotations=face_annotations)
                if cell.get("checksum") and cell["checksum"] != result["rendering"]["input_sha256"]:
                    raise ValueError(
                        "Input changed after pairing confirmation; rescan and confirm before analysis"
                    )
                result["image_id"] = cell["image_id"]
                result["asset_id"] = f"asset:{group['group_id']}:{device_id}"
                result["annotation_revision"] = annotation.get("revision")
                asset_manifest.append(
                    {
                        "scene_id": group["group_id"],
                        "device_id": device_id,
                        "image_id": cell["image_id"],
                        "sha256": result["rendering"]["input_sha256"],
                        "status": "analyzed",
                        "rendering": result["rendering"],
                    }
                )
                metrics[device_id] = result
                paths[device_id] = Path(result["analysis_path"])
                self.repo.save_analysis(
                    project_id,
                    "image_metrics",
                    result,
                    scene_group_id=group["id"],
                    image_id=cell["image_id"],
                )

            audit = audit_scene(metrics)
            audit["pairing_basis"] = "user_confirmed_same_scene_not_pixel_aligned"
            if annotation:
                audit["challenge"] = {
                    "primary": annotation["challenge"],
                    "basis": "manual_annotation",
                    "author": annotation["author"],
                    "revision": annotation["revision"],
                }
                audit["attribute_comparability"].update(
                    annotation.get("attribute_comparability", {})
                )
            audit["scene_family"] = annotation.get("scene_family")
            self.repo.save_analysis(project_id, "scene_audit", audit, scene_group_id=group["id"])
            if len(paths) < 2:
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
                        "scene_family": annotation.get("scene_family"),
                        "missing_devices": [
                            key for key, value in group["cells"].items() if value is None
                        ],
                    }
                )
                continue

            forward, reverse = _anonymous_mapping(group["group_id"], list(paths))
            anonymous_paths = {forward[device_id]: path for device_id, path in paths.items()}
            anonymous_metrics = {
                forward[device_id]: {
                    "whole": payload["whole"],
                    "regions": payload.get("regions", {}),
                    "faces": [
                        {key: value for key, value in face.items() if key != "native_crop_path"}
                        for face in payload.get("faces", [])
                    ],
                    "color_status": payload["rendering"]["color_status"],
                    "color_claims_allowed": payload["rendering"]["color_claims_allowed"],
                    "warnings": payload.get("warnings", []),
                }
                for device_id, payload in metrics.items()
            }
            evidence_inputs = {
                forward[device]: [
                    {
                        "evidence_id": f"asset:{group['group_id']}:{forward[device]}",
                        "kind": "whole",
                        "path": str(paths[device]),
                        "rendering_version": metrics[device]["rendering"]["version"],
                    },
                    *[
                        {
                            "evidence_id": f"crop:{group['group_id']}:{forward[device]}:{face['face_id']}",
                            "kind": "native_face",
                            "path": face["native_crop_path"],
                            "bbox_xywh": face["bbox_xywh"],
                            "face_id": face["face_id"],
                        }
                        for face in metrics[device]["faces"]
                    ],
                ]
                for device in paths
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
                {
                    "audit": {key: audit[key] for key in ("challenge", "pairing_basis")},
                    "pass": "visual",
                    "evidence_inputs": evidence_inputs,
                },
            )
            primary_raw = self.primary.analyze(
                group["group_id"],
                anonymous_paths,
                {
                    "metrics": anonymous_metrics,
                    "audit": audit,
                    "pass": "metric_validation",
                    "evidence_inputs": evidence_inputs,
                    "visual_observation": primary_visual_raw.model_dump(mode="json"),
                },
            )
            reviewer_independent_raw = self.reviewer.analyze(
                group["group_id"],
                anonymous_paths,
                {
                    "audit": {key: audit[key] for key in ("challenge", "pairing_basis")},
                    "pass": "independent_visual",
                    "evidence_inputs": evidence_inputs,
                },
            )
            reviewer_challenge_raw = self.reviewer.analyze(
                group["group_id"],
                anonymous_paths,
                {
                    "metrics": anonymous_metrics,
                    "audit": audit,
                    "pass": "challenge",
                    "evidence_inputs": evidence_inputs,
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
            )
            all_findings.extend(scene_findings)
            scene_results.append(
                {
                    "group_id": group["group_id"],
                    "scene_family": annotation.get("scene_family"),
                    "missing_devices": [
                        key for key, value in group["cells"].items() if value is None
                    ],
                    "audit": audit,
                    "metrics": metrics,
                    "primary_visual": primary_visual.model_dump(mode="json"),
                    "primary": primary.model_dump(mode="json"),
                    "reviewer_independent": reviewer_independent.model_dump(mode="json"),
                    "reviewer_challenge": reviewer_challenge.model_dump(mode="json"),
                    "findings": scene_findings,
                }
            )

        self.repo.save_analysis(
            project_id,
            "asset_manifest",
            {
                "schema_version": "0.2",
                "claim_version": claim_version,
                "pairing_version": pairing["version"],
                "assets": asset_manifest,
            },
        )
        strategy_claims = infer_cross_scene_claims(scene_results, claim_version)
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
            related_records = related_external(finding, external)
            assessment = assess_capture_bias(
                supporting_scene_count=len(scene_ids),
                comparable_scene_count=comparable_count,
                independent_support_count=sum(
                    item.get("supports_claim") is True
                    and item.get("independent_support_eligible") is True
                    for item in related_records
                ),
                independent_contradiction_count=sum(
                    item.get("supports_claim") is False
                    and item.get("independent_support_eligible") is True
                    for item in related_records
                ),
            ).model_dump(mode="json")
            assessment.update(
                {
                    key: finding.get(key)
                    for key in ("claim_id", "claim_version", "device_id", "attribute")
                }
            )
            assessment["claim_statement"] = finding.get("statement")
            bias_assessments.append(assessment)
            if assessment["status"] == "RESHOOT_REQUIRED_FOR_GENERALIZATION":
                self.repo.create_review_item(
                    project_id, "possible_capture_bias", assessment, priority="high"
                )
        self.repo.save_analysis(project_id, "capture_bias_assessment", bias_assessments)

        attribution_cases: list[dict[str, Any]] = []
        cells_by_group = {group["group_id"]: group["cells"] for group in pairing["groups"]}
        for finding in all_findings:
            for scene_id in finding.get("supporting_scene_ids", []):
                cell = cells_by_group.get(scene_id, {}).get(finding["device_id"])
                if not cell:
                    continue
                attribution = attribute_output_claim(
                    finding["statement"], hardware_context={}, exif_context=cell.get("exif", {})
                ).model_dump(mode="json")
                attribution.update(
                    {
                        key: finding.get(key)
                        for key in ("claim_id", "claim_version", "device_id", "attribute")
                    }
                )
                attribution["scene_id"] = scene_id
                attribution["image_id"] = cell["image_id"]
                attribution["supporting_scene_ids"] = [scene_id]
                attribution_cases.append(attribution)
        self.repo.save_analysis(project_id, "attribution_case", attribution_cases)

        device_names = {device["id"]: device["name"] for device in pairing["devices"]}
        visual_assets: list[dict[str, Any]] = []
        for group in pairing["groups"]:
            selected_cells = [
                (device_id, cell) for device_id, cell in group["cells"].items() if cell is not None
            ]
            visual_assets.extend(
                {
                    "scene_id": group["group_id"],
                    "device_id": device_id,
                    "device_name": device_names.get(device_id, device_id),
                    "path": next(
                        (
                            scene["metrics"][device_id]["analysis_path"]
                            for scene in scene_results
                            if scene["group_id"] == group["group_id"]
                            and device_id in scene["metrics"]
                        ),
                        cell["path"],
                    ),
                    "filename": cell["filename"],
                }
                for device_id, cell in selected_cells
            )
        profiles = device_profiles(pairing["devices"], all_findings)

        report_payload = ReportPayload(
            project_name=project.name,
            schema_version="0.2",
            all_claims=all_findings,
            challenge_matrix=challenge_matrix(scene_results),
            coverage={
                "assets": len([item for item in asset_manifest if item["status"] == "analyzed"]),
                "missing_slots": len(
                    [item for item in asset_manifest if item["status"] == "missing"]
                ),
                "scene_positions": len(scene_results),
                "confirmed_families": len(
                    {scene["scene_family"] for scene in scene_results if scene.get("scene_family")}
                ),
                "pairing_version": pairing["version"],
            },
            review_provenance={
                "human_review": "pending",
                "calibration": "not_performed",
                "claim_version": claim_version,
                "primary_model": getattr(self.primary, "model", type(self.primary).__name__),
                "reviewer_model": getattr(self.reviewer, "model", type(self.reviewer).__name__),
                "independence": "not_verified",
                "prompt_version": "evidence-0.2",
            },
            devices=[device["name"] for device in pairing["devices"]],
            findings=[item for item in all_findings if item["grade"] in {"A", "B"}],
            scene_results=scene_results,
            visual_assets=visual_assets,
            device_profiles=profiles,
            methodology_notes=[
                "同一场景内先进行匿名横向比较，再恢复设备身份。",
                "主评与复核是否独立由模型来源记录决定；同模型多轮不作为独立证据。",
                "A/B/C 为未校准规则标签；完整主张与反例保留在附录，尚未完成人工验证。",
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
        report_version = f"0.2.{run_key}"
        report_dir = self.workspace / "projects" / project_id / "reports" / run_key
        bundle = render_report_bundle(
            report_payload, report_dir, version=report_version, status="draft"
        )
        self.repo.save_report(project_id, report_version, "draft", str(bundle["html"]))
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
        audit: dict[str, Any],
        primary: ModelEvaluationResult,
        reviewer: ModelEvaluationResult,
        challenge: ModelEvaluationResult,
        metrics: dict[str, Any],
    ) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for observation in primary.observations:
            if observation.device_id not in metrics:
                self.repo.create_review_item(
                    project_id, "unknown_observation_device", observation.model_dump(mode="json")
                )
                continue
            relations = []
            for review in (reviewer, challenge):
                candidates = [review_relation(observation, item) for item in review.observations]
                relations.append(
                    "contradicts"
                    if "contradicts" in candidates
                    else "supports"
                    if "supports" in candidates
                    else "unknown"
                )
            if relations != ["supports", "supports"]:
                self.repo.create_review_item(
                    project_id,
                    "model_conflict",
                    {
                        "group_id": group_key,
                        "primary_observation": observation.model_dump(mode="json"),
                        "review_relations": relations,
                    },
                    priority="high",
                )
            objective_support, measurements = measure_observation(observation, metrics, audit)
            validity = (
                0.0
                if audit["status"] == "NOT_COMPARABLE"
                else 1.0
                if audit["status"] == "FULLY_COMPARABLE"
                else 0.6
            )
            key = (
                observation.attribute
                if observation.attribute in audit.get("attribute_comparability", {})
                else observation.dimension
            )
            if audit.get("attribute_comparability", {}).get(key) in {
                "NOT_COMPARABLE",
                "INSUFFICIENT_EVIDENCE",
            }:
                objective_support = None
            claim = ClaimCandidate(
                device_id=observation.device_id,
                statement=observation.statement,
                claim_type="observation",
                claim_id=claim_identity(
                    observation.device_id,
                    group_key,
                    observation.statement,
                    observation.region,
                    observation.attribute,
                    observation.direction,
                    self.claim_version,
                    comparison=observation.comparison,
                    conditions=observation.conditions,
                ),
                claim_version=self.claim_version,
                region=observation.region,
                attribute=observation.attribute,
                direction=observation.direction,
                comparison=observation.comparison,
                conditions=observation.conditions,
                supporting_scene_ids=[group_key],
                contradicting_scene_ids=[group_key]
                if "contradicts" in relations or objective_support == 0
                else [],
                model_agreement=relations.count("supports") / 2,
                objective_support=objective_support,
                scene_validity=validity,
                measurement_evidence=measurements,
                review_relations=relations,
                reviewer_dependency="same_model_family"
                if primary.model_family != "unknown"
                and primary.model_family == reviewer.model_family
                else "independence_not_verified",
                alternative_explanations=["capture variation", "framing difference"],
            )
            if objective_support is not None:
                claim.statement = measured_statement(observation)
            adjudicated = adjudicate_claim(claim)
            finding = adjudicated.model_dump(mode="json")
            finding.update(
                evidence=[group_key],
                evidence_refs=observation.evidence_refs,
                model_statement=observation.statement,
                claim_type="observation",
            )
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
                    key = (item.url, finding["claim_id"])
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
                    record.update(
                        {
                            key: finding.get(key)
                            for key in ("claim_id", "claim_version", "device_id", "attribute")
                        }
                    )
                    record["scene_ids"] = finding.get("supporting_scene_ids", [])
                    record["evidence_scope"] = "search_excerpt_background"
                    record["independent_support_eligible"] = False
                    record["source_family"] = urlparse(item.url).netloc.casefold()
                    external.append(record)
        return external
