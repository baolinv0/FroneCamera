# Task 4 independent Critic, round 2

Verdict REVISE. Fresh non-editing critic_front_publication_r2 closed T4-R1-01..04 using actual professional/quick/rawprovider/DOCX, ID, scope and ZIP regressions. 29 backend and 5 React tests passed; TypeScript/Vite build passed.

T4-R2-01 — P1 — INT-011/014 — OPEN: api/workflow count final rows then render same final-v1.0 path nonatomically. Actual TestClient/SQLite: first rejected-a projection pauses; review approved; second HTTP200/version1.0/claims[a,b]; first resumes HTTP200/version1.0. Two final IDs reference samepath and previously successful second report changes to [b].

Require durable DB-atomic allocation or contention rejection shared across quick/professional, unique immutable artifact paths, complete-bundle publication; no successful row on failed render, safe retry. Process mutex insufficient for API/worker. Test pro/pro and quick/pro differing projections, verify published bytes unchanged. Fresh review required before Evaluator.
