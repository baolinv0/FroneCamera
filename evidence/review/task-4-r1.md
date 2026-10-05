# Independent Critic Task 4 round 1

Agent: /root/critic_front_publication_r1. Verdict: REVISE. Non-editing review against local source baseline 3430951. 24 backend tests and 5 React tests passed; TypeScript/Vite build passed. Actual SQLite/FastAPI probes checked professional open-review/malformedJSON gates. No weakened old tests identified.

## T4-R1-01 — P1 — INT-011 — OPEN

Rejected claims survive provider response strings. `report_projection.py:166–180` filters dictionaries but leaves strings; `reporting.py:48–52,149–152` displays complete scene JSON. Put serialized `{"observations":[{"device_id":"D1","statement":"removed-secret"}]}` in fixture_payload().scene_results[0].primary.raw.choices[0].message.content, reject D1 and finalize actual API. HTTP200; resolved top findings keep only B, but final HTML and JSON retain removed-secret.

Expected: no rejected claim in publication. Closure: exclude/project raw response bodies safely while preserving hashes/provenance; add actual API/renderer regression with realistic provider content. Retain immutable raw audit evidence in engineering storage, not the published report.

## T4-R1-02 — P2 — INT-011 — OPEN

Stable-ID review incorrectly also matches different IDs by text. `report_projection.py:80–88,135–163`: review containing claim_id/device/statement enters both ID and unrestricted legacy rejection. Probe a={claim_id:a,device_id:d,dimension:tone,statement:same,evidence:[G001]}, b={same values,claim_id:b,dimension:color}. Review rejected with payload {claim_id:a,device_id:d,statement:same}. Actual findings empty; expected b survives.

Closure: authoritative ID cannot reject a different ID; text fallback may resolve matching representations lacking IDs. Test both ID-bearing and historical records.

## T4-R1-03 — P2 — INT-011 — OPEN

Legacy ambiguity ignores scene/dimension scope. `report_projection.py:90–114` matching_devices scans statement alone. Findings a/same/tone/G001 and b/same/tone/G002; rejected payload {group_id:G001,primary_observation:{statement:same,dimension:tone}}. Expected remove only unique G001; actual ValueError ambiguous identity.

Closure: determine ambiguity within supplied scene/dimension/URL scope; genuinely ambiguous identity stays blocked.

## T4-R1-04 — P2 — INT-012 — OPEN

Embedded Windows UNC paths leak. `exporting.py:27–37` handles whole UNC but embedded regex only drive-letter/POSIX. Save actual analysis error string `Cannot open \\secret-server\private-share\capture.jpg`, then export_project; ZIP analysis.json retains complete UNC. Whole-string equivalent is anonymized.

Closure: sanitize embedded UNC recursively, add actual ZIP regression with existing drive/POSIX cases.

Evidence boundary: local software probes, no real endpoint/model-quality claim. All findings returned to original Builder for remediation; only a fresh Critic can close them.
