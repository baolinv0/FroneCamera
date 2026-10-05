# Task 1 independent Critic, round 1

Verdict: REVISE. Non-editing reviewer `critic_iqa_compare_r1` reran all four new test modules: 28 passed. Additional actual-image probes found:

## IQA-R1-01 — P1 — INT-002/003

TIFF EXIF is applied twice: OpenCV IMREAD_UNCHANGED already rotates TIFF, then `_orient` rotates again (`asset_io.py`). A 24x32 TIFF with orientation 6 differs from equivalent rotated PNG and falsely rejects geometry. Orientations 2–8 all fail. Fix one orientation operation per format while preserving precision; add reflected/rotated TIFF and comparison regressions.

## IQA-R1-02 — P1 — INT-003/004

ROI uniqueness uses `(asset_id,id)` but assessment stores by `id`. Implicit baseline scope and explicit baseline ID are equivalent yet accepted with identical ROI IDs. A destroyed 8x8 face first rejects `local_content_collapse:face`; adding the same-ID intact ROI hides it and returns REVIEW. Reject effective-scope collisions in both insertion orders, including device implicit/explicit collisions (`comparison_models.py`, `comparison.py`).

## IQA-R1-03 — P2 — INT-003

Empty `.npy` raises EOFError outside loader/evaluation exception boundary and aborts instead of finite `REJECT/invalid_candidate` with the empty byte SHA. Normalize expected NumPy errors to `AssetDecodeError`; add empty/truncated loader/evaluator/CLI tests (`asset_io.py`).

Findings establish software behavior, not image quality accuracy or training utility. Each requires fresh independent closure after Builder remediation.
