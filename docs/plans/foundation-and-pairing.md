# Foundation and Pairing Implementation Plan

## Goal

Build the first working vertical slice of the front-camera portrait evaluation system:

```text
create project
→ register device folders
→ scan images
→ propose matched scene groups
→ review and edit pairing
→ freeze a versioned pairing snapshot
→ reopen the project with the same state
```

This phase does not implement VLM inference, ROI models, objective image-quality metrics, external media search, or report generation.

## Architecture

- Python 3.12
- FastAPI
- Pydantic v2
- SQLAlchemy 2
- Alembic
- SQLite
- pytest
- React/Vite review console
- `uv` for native development
- Docker Compose-compatible layout

The core domain and pairing logic must not depend on FastAPI, SQLAlchemy, or the web UI. Infrastructure adapters implement domain interfaces.

## Global Constraints

- Support `N >= 2` device folders.
- Do not require the optional 20-scene template.
- Existing inputs may contain an arbitrary number of matched, unclassified scene groups.
- Natural ordering only proposes pairing; human confirmation is mandatory.
- Original source images are read-only.
- Missing images are explicit and must never cause silent sequence shifting.
- Repeats are optional and explicitly assigned.
- Every confirmed pairing is immutable and versioned.
- SQLite is the first database, behind repository abstractions.
- All behavior is implemented test-first.

---

## Task 1: Repository and Python Package Foundation

### Files

```text
pyproject.toml
README.md
.env.example
Makefile
src/fronecamera/__init__.py
src/fronecamera/config.py
tests/test_config.py
```

### Deliverable

A Python package installable with `uv`, with lint, type-check, and test commands.

### Tests

- default settings load without an environment file;
- workspace and database paths are resolved from configuration;
- invalid port and access mode values are rejected.

### Commands

```bash
uv sync
uv run pytest tests/test_config.py -v
uv run ruff check .
uv run mypy src
```

### Commit

```text
chore: initialize Python project and configuration
```

---

## Task 2: Domain Identifiers and Project State Machine

### Files

```text
src/fronecamera/domain/ids.py
src/fronecamera/domain/project.py
src/fronecamera/domain/states.py
src/fronecamera/domain/errors.py
tests/domain/test_project_state.py
```

### Required Interfaces

```python
ProjectId = NewType("ProjectId", str)
DeviceId = NewType("DeviceId", str)
SceneGroupId = NewType("SceneGroupId", str)
ImageAssetId = NewType("ImageAssetId", str)
PairingSnapshotId = NewType("PairingSnapshotId", str)
```

```python
class ProjectState(StrEnum):
    CREATED = "created"
    PAIRING_REQUIRED = "pairing_required"
    PAIRING_CONFIRMED = "pairing_confirmed"
    AUDITING = "auditing"
    AUDIT_REVIEW_REQUIRED = "audit_review_required"
    READY_FOR_ANALYSIS = "ready_for_analysis"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

```python
class Project:
    def transition_to(self, target: ProjectState) -> None: ...
```

### Tests

- `CREATED → PAIRING_REQUIRED` succeeds;
- `PAIRING_REQUIRED → PAIRING_CONFIRMED` succeeds only with a valid snapshot;
- illegal transitions raise `InvalidStateTransition`;
- cancelled projects cannot transition to active states.

### Commit

```text
feat: add project domain and state machine
```

---

## Task 3: Device, Image Asset, and Scene-Group Domain Models

### Files

```text
src/fronecamera/domain/device.py
src/fronecamera/domain/assets.py
src/fronecamera/domain/pairing.py
tests/domain/test_pairing_models.py
```

### Required Models

```python
@dataclass(frozen=True)
class Device:
    id: DeviceId
    display_name: str
    brand: str | None
    model: str | None
    source_folder: Path
    order_index: int
```

```python
@dataclass(frozen=True)
class ImageAsset:
    id: ImageAssetId
    device_id: DeviceId
    original_path: Path
    filename: str
    checksum: str
    natural_order_index: int
    width: int | None
    height: int | None
    capture_time: datetime | None
```

```python
@dataclass(frozen=True)
class PairingCell:
    scene_group_id: SceneGroupId
    device_id: DeviceId
    image_asset_id: ImageAssetId | None
    status: Literal["present", "missing", "invalid"]
```

```python
@dataclass(frozen=True)
class SceneGroup:
    id: SceneGroupId
    order_index: int
    display_name: str | None
    classification: str | None
    cells: tuple[PairingCell, ...]
```

### Tests

- a scene group can remain unclassified;
- a missing device cell is preserved explicitly;
- at least two present images are required for comparison eligibility;
- duplicate device cells in one scene group are rejected.

### Commit

```text
feat: define device image and scene-group models
```

---

## Task 4: Natural Sort and Read-Only Folder Scanner

### Files

```text
src/fronecamera/dataset/natural_sort.py
src/fronecamera/dataset/scanner.py
src/fronecamera/dataset/checksum.py
tests/dataset/test_natural_sort.py
tests/dataset/test_scanner.py
```

### Required Interfaces

```python
def natural_sort_key(value: str) -> tuple[object, ...]: ...
```

```python
@dataclass(frozen=True)
class ScanResult:
    device_id: DeviceId
    assets: tuple[ImageAsset, ...]
    warnings: tuple[str, ...]
```

```python
class FolderScanner(Protocol):
    def scan(self, device: Device) -> ScanResult: ...
```

### Supported Extensions

```text
.jpg .jpeg .png .webp .heic .heif
```

### Tests

- `1.jpg, 2.jpg, 10.jpg` sorts numerically;
- hidden files and unsupported extensions are ignored;
- recursive scanning is disabled in MVP;
- checksum is deterministic;
- scanning never modifies file timestamps or contents;
- unreadable files generate warnings rather than disappearing silently.

### Commit

```text
feat: add deterministic read-only folder scanner
```

---

## Task 5: Initial Pairing Proposal

### Files

```text
src/fronecamera/pairing/proposal.py
tests/pairing/test_proposal.py
```

### Required Interface

```python
@dataclass(frozen=True)
class PairingProposal:
    scene_groups: tuple[SceneGroup, ...]
    warnings: tuple[str, ...]


def propose_pairing(
    devices: Sequence[Device],
    scans: Mapping[DeviceId, ScanResult],
) -> PairingProposal: ...
```

### Rules

- pair by natural-order position;
- create `G001`, `G002`, ... identifiers;
- use the maximum folder length to preserve missing cells;
- never shift later images to hide a missing item;
- do not classify scenes automatically;
- do not infer repeats.

### Tests

- equal-length folders produce a complete matrix;
- shorter folders produce explicit missing cells;
- three-device and eight-device projects behave identically;
- scene IDs remain stable for the same input ordering.

### Commit

```text
feat: propose matched scene groups from folder order
```

---

## Task 6: Editable Pairing Draft

### Files

```text
src/fronecamera/pairing/draft.py
src/fronecamera/pairing/commands.py
tests/pairing/test_draft_commands.py
```

### Commands

```python
class SwapImages: ...
class MoveImage: ...
class SetMissing: ...
class ReplaceImage: ...
class RenameSceneGroup: ...
class SetSceneClassification: ...
class ApplyDeviceOffset: ...
```

### Required Behavior

- every command returns a new draft revision;
- the previous revision remains available;
- a device offset is explicit and previews all affected groups;
- duplicate use of the same image in one snapshot is rejected unless explicitly allowed for a diagnostic copy;
- classification remains optional.

### Tests

- swap, move, missing, replace, rename, and offset behavior;
- undo by selecting a previous revision;
- an invalid command does not mutate the draft;
- no image is silently dropped.

### Commit

```text
feat: add versioned pairing edit commands
```

---

## Task 7: Pairing Validation and Freeze Gate

### Files

```text
src/fronecamera/pairing/validation.py
src/fronecamera/pairing/snapshot.py
tests/pairing/test_snapshot.py
```

### Required Interfaces

```python
@dataclass(frozen=True)
class PairingValidationResult:
    valid: bool
    blocking_errors: tuple[str, ...]
    warnings: tuple[str, ...]
```

```python
@dataclass(frozen=True)
class PairingSnapshot:
    id: PairingSnapshotId
    project_id: ProjectId
    version: int
    created_at: datetime
    scene_groups: tuple[SceneGroup, ...]
    source_checksums: Mapping[ImageAssetId, str]
```

### Freeze Rules

- at least two devices are registered;
- every comparison-eligible scene contains at least two present valid images;
- source assets still match scanned checksums;
- blocking scan errors are resolved;
- the user explicitly confirms the draft;
- freezing creates a new immutable version.

### Tests

- valid drafts freeze successfully;
- changed source files block freezing;
- a second confirmation creates version 2 and preserves version 1;
- confirmed snapshots cannot be edited in place.

### Commit

```text
feat: validate and freeze immutable pairing snapshots
```

---

## Task 8: SQLAlchemy and Alembic Persistence

### Files

```text
src/fronecamera/infrastructure/db/base.py
src/fronecamera/infrastructure/db/models.py
src/fronecamera/infrastructure/db/session.py
src/fronecamera/infrastructure/repositories/projects.py
src/fronecamera/infrastructure/repositories/pairing.py
alembic.ini
alembic/env.py
alembic/versions/0001_foundation.py
tests/infrastructure/test_repositories.py
```

### Tables

```text
projects
devices
image_assets
pairing_drafts
pairing_draft_revisions
scene_groups
pairing_cells
pairing_snapshots
pairing_snapshot_groups
pairing_snapshot_cells
```

### Tests

- migration upgrades an empty SQLite database;
- project/device/assets round-trip correctly;
- draft revisions preserve history;
- snapshot retrieval is byte-for-byte stable at the domain serialization level;
- repository APIs do not expose SQLAlchemy models to the domain layer.

### Commit

```text
feat: persist projects assets and pairing snapshots
```

---

## Task 9: Application Services

### Files

```text
src/fronecamera/application/projects.py
src/fronecamera/application/scanning.py
src/fronecamera/application/pairing.py
tests/application/test_project_workflow.py
```

### Required Use Cases

```python
create_project(...)
register_device(...)
scan_project(project_id)
get_pairing_draft(project_id)
apply_pairing_command(project_id, command)
validate_pairing(project_id)
confirm_pairing(project_id)
```

### End-to-End Service Test

```text
create project
→ register four folders
→ scan
→ propose groups
→ mark one cell missing
→ rename one group
→ validate
→ confirm
→ close database
→ reopen
→ load identical snapshot
```

### Commit

```text
feat: add project scanning and pairing use cases
```

---

## Task 10: FastAPI Endpoints

### Files

```text
src/fronecamera/api/app.py
src/fronecamera/api/dependencies.py
src/fronecamera/api/schemas/projects.py
src/fronecamera/api/schemas/pairing.py
src/fronecamera/api/routes/projects.py
src/fronecamera/api/routes/pairing.py
tests/api/test_pairing_api.py
```

### Endpoints

```text
POST /projects
POST /projects/{project_id}/devices
POST /projects/{project_id}/scan
GET  /projects/{project_id}/pairing
POST /projects/{project_id}/pairing/commands
GET  /projects/{project_id}/pairing/validation
POST /projects/{project_id}/pairing/confirm
GET  /projects/{project_id}/pairing/snapshots
```

### API Rules

- clients pass asset IDs, never arbitrary server paths;
- validation errors use stable machine-readable codes;
- the confirm endpoint requires the expected draft revision to prevent stale writes;
- source folder paths are accepted only during local project setup and must resolve under configured allowed roots.

### Tests

- happy-path workflow;
- stale revision conflict returns HTTP 409;
- path outside allowed roots returns HTTP 422/403;
- invalid state transition returns HTTP 409;
- missing project returns HTTP 404.

### Commit

```text
feat: expose project and pairing API
```

---

## Task 11: Asset Thumbnail API

### Files

```text
src/fronecamera/assets/thumbnails.py
src/fronecamera/api/routes/assets.py
tests/api/test_asset_api.py
```

### Requirements

- generate cached JPEG thumbnails without changing source files;
- orient according to EXIF;
- restrict maximum dimensions and quality;
- serve by asset ID;
- reject arbitrary path traversal;
- cache key includes source checksum and thumbnail version.

### Commit

```text
feat: add safe thumbnail generation and asset serving
```

---

## Task 12: React Pairing Review Console

### Files

```text
apps/web/package.json
apps/web/src/main.tsx
apps/web/src/api/client.ts
apps/web/src/pages/ProjectCreatePage.tsx
apps/web/src/pages/PairingReviewPage.tsx
apps/web/src/components/PairingGrid.tsx
apps/web/src/components/DeviceColumn.tsx
apps/web/src/components/SceneRow.tsx
apps/web/src/components/PairingWarnings.tsx
apps/web/src/components/ConfirmPairingDialog.tsx
apps/web/src/**/*.test.tsx
```

### Required UI

- create a project and register `N` device folders;
- show the proposed scene-device matrix;
- show thumbnail, filename, index, capture time, and missing state;
- swap/move/replace/mark missing;
- apply a device offset with preview;
- rename a scene group;
- optionally assign a scene label;
- show blocking errors and warnings separately;
- require explicit confirmation;
- display snapshot version after confirmation.

### Tests

Use Vitest and Testing Library:

- renders four devices and arbitrary scene count;
- missing cells are visible;
- a move command sends the expected revision;
- stale revision response prompts reload rather than overwriting;
- confirmation is disabled while blocking errors remain.

### Commit

```text
feat: add browser-based pairing review console
```

---

## Task 13: Docker and Developer Commands

### Files

```text
Dockerfile
compose.yaml
Makefile
scripts/dev.sh
scripts/test.sh
```

### Services

```text
api
web
redis
worker-placeholder
```

Redis and the worker placeholder are included to preserve the approved deployment topology, but this phase does not enqueue heavy evaluation jobs.

### Requirements

- source folders are mounted read-only;
- workspace is writable;
- API does not receive GPU access;
- LAN binding is opt-in;
- default binding is localhost;
- health endpoints cover API and database.

### Commit

```text
chore: add native and Docker development workflows
```

---

## Task 14: Regression Fixture and Acceptance Test

### Files

```text
tests/fixtures/pairing_project/
tests/e2e/test_foundation_pairing.py
docs/operator/foundation-pairing.md
```

### Fixture

Create a synthetic project with:

- four device folders;
- six matched groups;
- one missing image;
- different filenames per device;
- filenames containing numeric ordering edge cases;
- no mandatory scene classification.

### Acceptance Test

The automated test must prove:

1. four folders scan deterministically;
2. six groups are proposed;
3. the missing cell is explicit;
4. a correction changes only the intended draft revision;
5. the pairing freezes as snapshot version 1;
6. the project reopens with the same snapshot;
7. source images are unchanged;
8. no scene taxonomy is required.

### Final Verification

```bash
uv run pytest -v
uv run ruff check .
uv run mypy src
pnpm --dir apps/web test
pnpm --dir apps/web build
docker compose config
```

### Commit

```text
test: add foundation and pairing acceptance workflow
```

---

## Completion Gate

The foundation-and-pairing slice is complete only when:

- a clean Linux environment can start the API and web console;
- the browser creates a project with arbitrary device folders;
- natural-order pairing is reviewable and editable;
- missing images remain explicit;
- scene classification is optional;
- confirmation creates an immutable versioned snapshot;
- a restart preserves all state;
- all tests, lint, type checks, and web build pass.

The next implementation plan begins with image decoding, EXIF normalization, comparability audit, ROI infrastructure, and diagnostic assets.
