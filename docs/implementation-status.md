# Implementation Status

## v0.1 completed

The repository contains a runnable end-to-end MVP covering:

- project/device creation;
- image scan and editable generic pairing;
- immutable, versioned pairing snapshots;
- SQLite persistence and migrations;
- objective image metrics and diagnostic regions;
- two-stage primary visual/metric protocol and independent/challenge reviewer protocol;
- deterministic fallback analysis;
- evidence adjudication and cross-scene claims;
- internal-result freeze;
- external hardware/professional-review retrieval and text-evidence corroboration classification;
- capture-bias and attribution cases;
- atomic persistent worker tasks with bounded retry;
- browser review interfaces;
- review-aware immutable final report bundle with reference-derived DOCX, optional PDF, and project export;
- Docker and CI definitions;
- backend unit/integration tests.

## Deployment dependencies not stored in Git

- proprietary or open model weights;
- the actual vLLM/SGLang services;
- a SearxNG deployment;
- user image datasets;
- fonts or system packages needed by the optional PDF renderer.

## Expected next calibration work

The software workflow is complete, but formal use still requires empirical calibration on the target server and image set:

1. validate the selected face/ROI backend on diverse faces and lighting;
2. calibrate metric thresholds against manually reviewed samples;
3. validate prompt JSON compliance and position invariance for both VLM families;
4. test BF16/quantized deployment on the 98 GB GPU;
5. define the approved professional-source domain list for the organization;
6. conduct a controlled three-repeat pilot to quantify capture variance.

## Verification evidence

The current branch is verified with backend unit/integration tests, MyPy, Ruff, frontend Vitest, TypeScript/Vite production build, npm audit, and an end-to-end smoke run covering asynchronous worker execution, human review resolution, final report generation, and project export.
