# Simplified Four-Step Workflow Implementation Plan

## Goal

Make the default product experience match the user-facing workflow:

```text
Create test data
→ Run full evaluation
→ Generate report automatically
→ Open a read-only report from another computer
```

The existing evidence pipeline and advanced review console remain available behind an Advanced Review section.

## Tasks

1. Add failing API and pipeline tests for quick/professional modes, automatic quick-report finalization, and signed read-only report access.
2. Add `mode` support to the evaluation pipeline and background worker.
3. Add `POST /api/projects/{id}/run-full-evaluation` and signed report-share endpoints.
4. Replace the default React workspace with a four-step workflow and automatic task polling while retaining advanced pairing, analysis, and review controls.
5. Document LAN report sharing and the distinction between quick and professional modes.
6. Run backend tests, type checks, lint, frontend tests, and production build in CI.
