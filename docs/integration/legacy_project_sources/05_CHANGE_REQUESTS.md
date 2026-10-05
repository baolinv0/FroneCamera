# Change Requests

Document-ID: `WORK-CHANGE-REQUESTS-001`  
Document-Version: `1.0.0`

This file is an audit log. It does not directly override a frozen specification.

Any approved change requires a new specification version, updated source files, refreshed Project Sources, and new human approval.

## Change Request Template

### CR-001

- Status: `PROPOSED`
- Requested by:
- Requested at UTC:
- Affected Requirement IDs:
- Current requirement/scope/threshold:
- Proposed requirement/scope/threshold:
- Reason:
- Product impact:
- Acceptance impact:
- Verification impact:
- Regression risk:
- Human decision: `PENDING`
- Human approver:
- Decision at UTC:
- New specification version:
- Source files replaced in Project: `NO`

## Allowed statuses

- `PROPOSED`
- `APPROVED_PENDING_REBASELINE`
- `REJECTED`
- `SUPERSEDED`
- `APPLIED_TO_NEW_BASELINE`

## Authority rule

A change becomes authoritative only when:

1. affected specification files are updated;
2. `spec_version` is incremented;
3. Project Sources are replaced;
4. the new baseline is marked `FROZEN`;
5. human approval is recorded;
6. the workflow restarts from the Specification Gate.
