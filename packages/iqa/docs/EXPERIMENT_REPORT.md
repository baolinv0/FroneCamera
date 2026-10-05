# Verified Experiment Report

## Purpose

Verify that TMQA v0.2.0 is executable and that every frozen MUST requirement R-001 through R-019 has implementation, behavioral tests, output artifacts, and exact-head CI evidence.

## Setup

- Six deterministic synthetic scenes.
- Nine canonical alpha levels per scene.
- Controlled patterns for normal progression, highlight clipping, color shift, structural mutation, response reversal, and positive-alpha dead zone.
- Four deterministic synthetic judge roles.
- Synthetic judge and synthetic human labels remain explicitly marked and cannot be used by production calibration without operator opt-in.

## Data-flow checks

The experiment verifies:

1. strict nine-directory and per-scene completeness;
2. deterministic objective and sequence metrics;
3. four independent judge records and isolated unavailable evidence;
4. six-score response contract;
5. immutable prompt 3.2 and current prompt 3.3/schema v4;
6. source and exact sent-payload SHA-256 lineage;
7. fatal, all-unavailable, disagreement, unanimous REJECT, unanimous REGENERATE, and unanimous REVIEW decision precedence;
8. focused queue selection;
9. reviewer-safe payload separation;
10. reveal denial before persistence and reveal authorization after matching persistence;
11. reviewer-only default route isolation, including encoded and normalized path aliases;
12. server-generated authoritative review time;
13. append-only multi-reviewer evidence;
14. real/synthetic calibration isolation;
15. reliability metrics and normalized fusion weights;
16. engineering and reviewer JavaScript syntax;
17. secret-leakage scan;
18. eight CLI surfaces and sdist/Wheel isolated import;
19. machine audit containing exactly R-001 through R-019.

## Mechanical evidence

The pull-request workflow executes the frozen verification plan on the exact head SHA for Python 3.10 and 3.12. The authoritative test count, focused-command logs, artifact hashes, and results are recorded by the latest exact-head CI run rather than hard-coded in this report.

Required successful steps:

```text
exact checked-out head: PASS
compileall: PASS
ruff check: PASS
full pytest: PASS
sdist build: PASS
wheel build: PASS
wheel isolated-target import: PASS
CLI contract: PASS
six-scene end-to-end experiment: PASS
engineering dashboard JavaScript: PASS
reviewer dashboard JavaScript: PASS
secret scan: PASS
requirements audit: 19/19 PASS
```

## Blind-review security boundary

The engineering page embeds the complete evaluation payload and is distributed as a separate self-contained artifact. The default review server exposes only the reviewer-safe page, image assets, review APIs, and reviewer-safe payload API. It rejects the engineering page, engineering JSON, reveal JSON, and URL-encoded or normalized aliases of those routes. The reviewer page embeds only scene ID, image URL, image index, role, alpha, and level. A reveal response requires a persisted `(scene_id, review_id)` match and is returned only after append succeeds. Storage failure returns no reveal authorization.

## Synthetic calibration boundary

The verified experiment invokes:

```text
--allow-synthetic --review-type synthetic
```

The resulting calibration is explicitly marked experimental. Default calibration rejects synthetic-only evidence, and implicit real/synthetic mixtures are rejected. These values only demonstrate calibration mechanics and are not recommended production weights.

## Remaining external validation

Deploy real model endpoints, collect blind expert labels, and rerun calibration before using scores or weights in a product decision or model-training workflow. Physical HDR/RAW source-aware evaluation remains outside v0.2.0.
