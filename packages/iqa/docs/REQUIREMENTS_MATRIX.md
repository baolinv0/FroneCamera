# Frozen Requirements Matrix — TMQA 0.2.0

| ID | Requirement | Primary implementation | Behavioral / mechanical evidence | Status gate |
|---|---|---|---|---|
| R-001 | Canonical nine-level discovery and strict completeness | `config.py`, `dataset.py` | strict missing/unknown/duplicate/baseline tests; E2E nine levels | CI PASS required |
| R-002 | Objective metrics, sequence metrics, and stage trace | `metrics.py`, `evaluation.py` | metric tests; six-scene output schema | CI PASS required |
| R-003 | Arbitrary independent judges and fault isolation | judge adapters, `evaluation.py` | partial failure and all-unavailable tests | CI PASS required |
| R-004 | Complete six-dimensional model records | `domain.py`, real/mock adapters | malformed/incomplete/invalid schema tests | CI PASS required |
| R-005 | Prompt Inspector and inference trace | `prompts.py`, engineering dashboard | prompt and visualization tests | CI PASS required |
| R-006 | Explicit image order and exact sent-payload lineage | `image_io.py`, `prompts.py`, real adapter | request-byte hash, EXIF, resize tests | CI PASS required |
| R-007 | Immutable prompt/schema versions | `PromptRegistry` | historical 3.2 fixture; 3.3/v4 tests | CI PASS required |
| R-008 | Safe final-decision precedence | `evaluation.py` | fatal, unavailable, disagreement, unanimous-decision tests | CI PASS required |
| R-009 | Complete UTF-8 engineering visualization | `dashboard.html`, `visualization.py` | dashboard functional and Node syntax tests | CI PASS required |
| R-010 | Data-layer blind review | `review.html`, reviewer payload, reveal API | forbidden-field scans and reveal authorization tests | CI PASS required |
| R-011 | Focused review queue on client and server | `review.py`, `server.py` | clean exclusion and HTTP 403 tests | CI PASS required |
| R-012 | Append-only, multi-reviewer, authoritative review records | `domain.py`, `review.py`, `server.py` | reviewer, timestamp, duplicate and order tests | CI PASS required |
| R-013 | Synthetic/real evidence isolation | calibration selection policy | real/synthetic/mixed policy tests | CI PASS required |
| R-014 | Reliability and calibration statistics | `review.py`, `cli.py` | MAE, agreement, gap, count and weight tests | CI PASS required |
| R-015 | Local review-server security and failure behavior | `server.py` | queue, schema, storage failure, protected-path tests | CI PASS required |
| R-016 | Eight CLI surfaces and deterministic experiment | `cli.py`, verification scripts | CLI contract plus six-scene E2E | CI PASS required |
| R-017 | Traceable nineteen-requirement audit | `audit.py`, `audit_requirements.py` | exact R-001..R-019 output and failure probes | CI PASS required |
| R-018 | Python 3.10/3.12 build and exact-head CI | `.github/workflows/ci.yml` | recorded head SHA, build, install, CLI, E2E artifacts | CI PASS required |
| R-019 | Evidence boundaries and non-misleading claims | README, report, UI/calibration labels | documentation review and secret/evidence scans | CI PASS required |

The machine-readable result is `runs/verified_experiment/requirements_audit.json`. A row is not considered complete because this table says PASS; the latest workflow must show success on the exact PR head.

## Non-claims

- The synthetic experiment does not prove real Qwen/InternVL/Ovis/MiniCPM accuracy.
- Synthetic fusion weights are not production recommendations.
- The current system is not a trained Reward Model.
- Physical HDR/RAW source-aware evaluation remains outside TMQA v0.2.0.
