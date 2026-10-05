Read DESIGN.md for the shared contract.

### Task 2: IQA blind review and PGT repair

Owner: existing `packages/iqa/src/qwen_tmqa/{config.py,domain.py,review.py,server.py,cli.py,prompts.py,judges/*}` plus newly ported `{candidate_schema.py,lineage.py,pseudo_gt.py,training_data.py}`; tests for these paths. Do not edit Task 1 files, existing `evaluation.py`/visualization unless coordinated with root. Root owns packaging.

Port Canary selection/lineage selectively from `repo_review/IQA/61cbaed3` preserving P1 payload trace, decoder, review and real/synthetic isolation. Historical prompts 3.2 and 3.3 remain immutable; if candidate schema requires changed contract, add a new version explicitly. Reject duplicate Judge IDs and count unique evidence. Enforce train export in library and CLI; require explicit canonical metadata and real human confirmation bound to source/candidate bytes and a supported label scope. Keep candidate suggestions separately exportable; never call unconfirmed suggestions training labels.

Track reviewer/scene exposure server-side persistently. Before reveal, latest blind revisions are valid. After reveal, allow an explicitly non-gold adjudication path or reject blind resubmission; calibration must not replace the pre-reveal gold. Preserve append-only history and authorized queue behavior. Add meaningful HTTP/storage failure and restart tests. Keep production weights limited to selected review evidence; no unverified automatic fusion activation. Report `evidence/build/task-2.md`.


Verification interpreter: /workspace/scratch/617f8bd555c4/frone-dev-env/bin/python
