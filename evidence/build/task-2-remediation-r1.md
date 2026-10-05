# Task 2 — Independent review round 1 remediation

## Finding and change

**T2-R1-01 · P2 · INT-008 / legacy R-004**: OpenAI-compatible runtime decoding accepted nonfinite or coercible issue `bbox` values and returned `available=True`.

Reproduced before the fix through actual `OpenAICompatibleJudge.evaluate` with mocked provider HTTP transport: NaN, positive infinity, negative infinity, string numeric coordinates and boolean coordinates all remained available. The targeted failing run showed five failures rather than silently checking already-correct behavior.

Changed only the owned runtime decoder and the newly added candidate-contract test file for this finding. Optional `bbox` is now either explicit null or a JSON array of exactly four finite numeric coordinates; booleans, strings, nonfinite values and malformed shapes fail validation, preserve the raw response/error provenance, and return unavailable/REVIEW. No undocumented normalized/pixel coordinate assumption is imposed; finite integer and float boxes are preserved. Existing prompt 3.1/3.2/3.3 contents and legacy tests remain unchanged.

Added 13 regression cases: seven malformed/nonfinite/coerced boxes, three positive null/finite box cases, and three historical 3.3 runtime cases. The new 3.4 contract and old 3.3 decoder both reject the same unsafe values.

## Verification

- New candidate-contract plus existing Judge tests: **54 passed**, exit 0.
- Fresh owned-path Ruff check: **all checks passed**, exit 0.
- Fresh owned-path Ruff format check: **18 files already formatted**, exit 0.
- Fresh full nested IQA suite: **240 passed in 40.43s**, exit 0.
- Full suite command:

```sh
PATH=/workspace/scratch/617f8bd555c4/frone-dev-env/bin:$PATH \
/workspace/scratch/617f8bd555c4/frone-dev-env/bin/python -m pytest -q
```

The independent Critic 2 reported 97 distinct scoped tests passing and separately verified historical prompt text byte identity and unchanged legacy tests. This remediation document records the builder evidence; final independent re-review remains the Critic's responsibility. No other round 1 Task 2 findings were reported. No commit or push performed.
