# Task 3 remediation, round 1

Builder remediation for every finding in the final `evidence/review/task-3-r1.md`. No historical tests were edited, deleted or weakened. No commit or push was performed. Independent closure remains a separate reviewer decision.

## Reproduction and closure changes

| Finding | Fail-before evidence | Builder change | Regression coverage |
|---|---|---|---|
| T3-R1-01, P1, INT-009 | Actual SQLite `_adjudicate_scene`, fully comparable .8/.4, primary exact highest statement and reviewers exact lowest: remained B=.633 with empty counter list. | Explicit semantic opposition removes objective claim support, records the scene in `contradicting_scene_ids`, and yields disputed C with zero confidence. Numerical image measurements remain separately persisted. | Direct actual SQLite repro; three-PNG actual pipeline and generated report excludes the disputed observation claim from formal findings. |
| T3-R1-02, P1, INT-009 | Three brighter scenes with opposite/zero-confidence C findings became A=1.0 and reused their claim IDs. Four initial parameterized cases failed. Positive actual pipeline control later exposed that nonzero source confidence also needed to cap agreement credit. | Cross-scene model credit requires the exact supported proposition, resolved A/B status, positive finite confidence/agreement/uncertainty, declared independent model provenance, no provisional/counterexample flags, and a luminance evidence reference from the same scene. Credit is capped by all three numeric strengths; only these usable claims supply `source_claim_ids`. Anonymous evidence references are restored with device identity so valid evidence remains usable. | Opposite text, zero confidence, zero uncertainty, disputed C helper cases; actual three-PNG pipeline with opposite C sources produces no A and no reused bad source IDs; positive helper and actual pipeline demonstrate bounded confidence and valid source IDs. |
| T3-R1-03, P2, INT-008/012 | Embedded UNC `Cannot open \\secret-server\private-share\capture.jpg` remained in recursively processed context. | Recursive string and key sanitization removes embedded UNC as well as POSIX/drive paths, including serialized inter-stage text, while preserving numeric metric context. | Nested helper repro plus captured actual outbound prompts in both OpenAI-compatible and Responses `httpx.MockTransport` paths, including serialized nested JSON. |
| T3-R1-04, P1, INT-013/014 | Forced real core result with invalid image evidence still invoked the model twice; actual supported HEIC decoded in Front but core used to report `image_decode_failed`. | Task 1 owner added precision/orientation-safe HEIC decoding. Front now consumes core asset/fatal validity in its real scene audit: invalid/fatal evidence makes the scene not comparable and skips model and claim production. Core dimension applicability enters blind model passes; core measurements enter metric/challenge passes. Unobservable/invalid/not-applicable canonical dimensions receive zero claim credit and an explicit review item. Declared core measurement references enter strict runtime evidence validation. | Actual HEIC pipeline assets valid in both Front/core, forced-invalid core rejects scene before model calls, core unobservable skin evidence reaches model context and yields zero-confidence review-required claim; blind pass retains applicability without measurement facts. |

The first six semantic/privacy regressions failed before their fixes; the forced-invalid-core regression failed independently, for **7 reproduced fail-before failures**. Positive directional control also caught and fixed uncapped nonzero model credit. New file `tests/test_task3_remediation_r1.py` contains **15 tests**, all additive.

## Verification

- New round-1 regressions: **15 passed**, 3.39 s after final evidence-ID restoration and nonzero-credit cap.
- Full Front suite: root independently reported **168 passed** with full Ruff/format/mypy clean; final additive positive control and bounded-credit adjustment were then verified by the fresh 15-test R1 run. Before that final cap, the Builder broad run exposed the positive-control failure alongside 168 passing tests; the unchanged assertion now passes.
- Owned 11-source-module mypy: success, no issues found.
- Ruff on owned source modules and both new regression files: clean.
- Earlier post-remediation integration subset: 56 passed, 1 existing Starlette/httpx deprecation warning.

Interpreter: `/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python`.

## Scope and limits

The independent final R1 record contains four findings and no additional IDs. This Builder addressed all four; root owns fresh independent closure dispatch and delivery. HEIC precision/orientation loader changes belong to Task 1 and are not misattributed to this Builder. Publication reservation/migrations and unified human-review projection belong to Task 4 and were not edited by this remediation.

Real image fixtures, SQLite, report rendering, actual adapter HTTP plumbing with mocked provider responses, and strict claim/evidence gates establish software behavior. No paid-model accuracy, real-mobile-capture accuracy, physical causation, human Gold, or training improvement is claimed. Core photometric measurements do not supply missing skin/reference/naturalness truth. Independent closure is required before declaring the review round accepted.
