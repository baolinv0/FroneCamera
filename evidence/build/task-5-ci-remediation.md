# Task 5 — frontend clean-install remediation

Builder: `/root/build_ci_frontend`  
Finding: `T5-CI-01`  
Acceptance: `INT-014`, `INT-015`

## Observed failure and reproduction

The initial integration head `c9fa5adb06272c96fe43283391697d3a6c886d21` failed before any frontend tests ran. Actual GitHub Actions run [37284042327](https://github.com/baolinv0/FroneCamera/actions/runs/37284042327), frontend job `111678592054`, used Node `22.23.3` and npm `10.9.9`; `npm install --no-package-lock` exited 1 with `Cannot read properties of null (reading 'edgesOut')`.

The builder read the actual job log, downloaded the exact public Node/npm packages, and reproduced that same exit 1 in a separate clean frontend copy with no `node_modules` or lock. The original development `node_modules` directory was untouched. The reproduction log identifies npm Arborist's `#loadPeerSet` at `build-ideal-tree.js:1289`, which dereferences `node.parent.edgesOut` while resolving the Vitest optional peer graph. Its unfinished timer is `idealTree:node_modules/vitest`. This is a fresh dependency-resolution failure, not a failed product assertion or TypeScript build.

The repository ignored `web/package-lock.json` and CI/Docker/native checks requested a newly resolved dependency tree on every install. The installed development toolchain differed (Node `24.19.0`, npm `11.9.0`). The corrective boundary is to record one validated graph and install that graph reproducibly, without changing the existing pinned manifest or bypassing peer validation.

## Changes

- Add a public-registry v3 `web/package-lock.json`, generated with npm `11.9.0` on Node `22.23.3` from the unchanged `web/package.json`.
- Remove the lock's `.gitignore` exclusion.
- Change the frontend CI install, `make frontend`, root README checks, and integration QUICKSTART checks to `npm ci`.
- Docker explicitly copies both manifest and lock, and uses `npm ci` on its existing Node 22 image.
- Qualify the shared QUICKSTART's full workflow checks as integration-branch checks; the independent IQA branch does not promise the Front frontend installation.

No additional product files, frontend test files/assertions, dependency pins, npm versions in CI, or Node major versions were modified. The fix does not use `--legacy-peer-deps`, `--force`, test skipping, or installation retries.

## Controlled verification

Tooling and copies are under `/workspace/scratch/617f8bd555c4/frontend-ci-repro/`. Both the failing and passing experiments used the exact same Node `22.23.3` and npm `10.9.9`. In the commands below, `NODE22` is `tooling/node22/package`, `NPM10` is `tooling/npm10/package`, and the Node directory is first in `PATH`, including for child processes.

| Experiment | Command | Actual outcome |
| --- | --- | --- |
| Original installer, empty copy | `node22 npm10/bin/npm-cli.js install --no-package-lock` | Exit 1, same `edgesOut` resolver exception |
| Lock generation, same unchanged manifest | `node22 npm11/bin/npm-cli.js install --package-lock-only` | Exit 0; npm 11.9.0; 224 lock entries |
| Locked installer, empty `node_modules` | `node22 npm10/bin/npm-cli.js ci` | Exit 0; 174 installed packages; no resolver bypass flags |
| Actual full frontend tests from clean install | `node22 npm10/bin/npm-cli.js test -- --run` | 4 test files, **7 tests passed**, exit 0 |
| Actual TypeScript and Vite build from clean install | `node22 npm10/bin/npm-cli.js run build` | `tsc -b` passed; Vite 6.4.3 transformed 33 modules and produced the distribution, exit 0 |
| Lock contract inspection | Compare every direct installed version and both root dependency maps to `web/package.json` | All existing pins unchanged |
| Lock source/integrity inspection | Inspect every resolved entry | All resolved URLs use `https://registry.npmjs.org/`; all have cryptographic integrity fields |
| Docker input inspection | Explicit manifest+lock copy and musl native Rollup lock entry | Both files copied; Linux x64 musl entry present |

Validated lock SHA-256 after the successful npm 10 clean install (the repository copy is byte-identical):

`3558a26809e42e4977513522ee33c39909402dd35cf1d12f5fe4c6afbbc93b21`

The exact npm 10 reproduction, lock-generation, and successful clean-install debug logs are in `cache-npm10/_logs/`. Test/build output is in `frontend-test-node22-npm10.log` and `frontend-build-node22-npm10.log`. Docker is not installed in the execution environment, so this builder does not claim an actual Docker image build. Fresh independent review and the next real GitHub Actions run remain delivery verification steps; this report does not claim they have already passed.
