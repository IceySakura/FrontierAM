# Automatic Host Module Dependencies

## Goal

Source-package executions must install the Host dependencies declared by every Nexus module selected by the submitted `RunInput`. Batch Task Lists use the first normalized record as the dependency representative. The feature must work without `NEXUS_AUTO_INSTALL`, must preserve the uploaded `uv.lock`, and must never provision Batch child managers before their dependency build is ready.

## Scope

- Supported code mode: `source_package`.
- Supported execution modes: Single, Live, local Task List Batch, and remote-materialized Batch.
- Host requirements are `ModuleMeta.base_deps + ModuleMeta.host_extras`.
- Runtime requirements remain owned by the worker-venv module mechanism.
- `image` submissions keep dependencies inside the submitted image.
- `config_only` submissions keep dependencies inside the predefined pool image.

## Module selection

The analyzer consumes a normalized `RunInput` and emits deterministic module selectors:

| RunInput value | Nexus metadata object table |
| --- | --- |
| `task_name` | `tasks` |
| `agent_name` | `agents` |
| `openai_clients.*.name` | `openai_clients` |
| `runtime_provider_name` | `runtime_providers` |
| `processors.*.name` | `span_processors` |
| `imports[]` | longest matching `ModuleMeta.module_path` |

Registry names and imports that resolve to wheel metadata contribute their Host dependencies. Valid selectors that cannot be
resolved from the uploaded wheel are logged as `module_dependency_selector_unresolved` warnings and skipped, so the
Builder can still install dependencies for every selector it understands. Ambiguous registry declarations remain errors,
and import strings must still be valid dotted Python module paths.

Module metadata is read from the exact uploaded `nexus-agent.whl`. Wheel discovery must support both directory `pyproject.toml` files and PEP 723 metadata in single Python files. Resolver code must not import extension modules.

Requirements are parsed as PEP 508. Editable, path, `file:` and direct-URL requirements are rejected. Exact duplicate requirements are removed; compatible constraints for the same distribution remain separate inputs so the resolver enforces their intersection.

## Dependency build

`WheelhouseBuildSpec` carries the normalized module selectors. They are part of the build key. The Builder performs these steps:

1. Validate the source archive, wheel and uploaded `uv.lock`, including the project name, version and selected extras. `nexus submit` builds the wheel from the local project on each source-package submission; the Builder uses that wheel's dependency metadata without reconstructing or comparing source dependency declarations.
2. Export base requirements with `uv export --frozen`, retaining hashes and markers.
3. Resolve selectors against metadata from the downloaded wheel.
4. Write the resulting Host requirements to `module-host.in`.
5. Seed `module-host.lock` with the exported versions, then run `uv pip compile --generate-hashes --no-sources` with the actual Nexus wheel, its selected extras and Host requirements as roots. The wheel supplies the dependency and extra-marker semantics used during installation. Existing versions are preferences; real declarations and explicit project constraints remain mandatory. Use the platform-controlled package index and ignore development overrides in `[tool.uv.sources]`.
6. Build wheels from the resulting complete dependency lock with `pip wheel --require-hashes`. Builds without module Host requirements keep the frozen base path.
7. Create an empty verification venv, install Nexus plus extras first and Host requirements second using only the wheelhouse, run `uv pip check`, and compare installed versions with the target lock. The installation requests enforce the requested roots; also verify Nexus import/version.
8. Archive wheels plus `locks/module-host.lock` and publish schema-v2 metadata containing selected modules, Host requirements, and the complete dependency lock's SHA256.

The Builder first posts the validated metadata document to
`POST /v1/internal/dependency-build/wheelhouse-metadata`, then uploads the archive with only the metadata schema version
and canonical SHA256 in headers. The Application Gateway stores the attempt-scoped document in the dependency-build row
and requires the upload digest to match it before binding the Artifact. This removes the HTTP-header size limit while
keeping metadata and archive publication fenced by the same build attempt.

During a rolling deployment, a new Application Gateway also accepts the previous complete-metadata header. It accepts
the older Nexus-wheel-path-only protocol only for builds without module selectors, because schema v1 cannot represent
module dependency metadata. A new Builder fails explicitly when its metadata POST reaches an old Gateway, so it cannot
silently publish an incomplete schema-v1 result.

The uploaded `uv.lock` is never modified. Compatible locked versions are preferred; a stale version may change when the declarations allow it. An actual conflict between declarations fails with `dependency_module_lock_failed`. The module-specific `module-host.lock` contains the full resolved dependency closure, including base and extra dependencies, at the existing archive path. Reproduction uses that lock and wheelhouse; a later fresh resolution can choose newer available versions.

Manager bootstrap retains staged installation and runs `uv pip check` before writing `.prepared`. The preparation identity includes the validation policy so an older cached environment is rebuilt and checked. Resolution policy changes also invalidate dependency-build cache keys. A rollout must update the code that computes build keys, the Builder image and the workload image containing bootstrap.

## Single and Live

Single and Live submissions already contain their normalized configuration. Submission extracts selectors and places them in the one dependency build used by the allocation. No new lifecycle state is required.

## Batch lifecycle

All source-package Batch submissions use the same two-stage lifecycle:

1. Submission creates a base dependency build and binds it to the orchestration allocation.
2. The execution allocation is inserted as a provisional requested allocation. `BatchRun.dependency_status=pending` gates the Environment Controller, so no provider resource is created.
3. Batch Control downloads or remotely materializes the Task List and validates it with the existing record parser.
4. The first normalized record is converted to module selectors. Later records are not inspected in v1.
5. Batch Control creates or reuses an augmented dependency build using the same source artifacts plus those selectors.
6. While holding the BatchRun and child-allocation rows, the repository verifies the child remains unprovisioned, replaces its dependency build identity and allocation-spec hash, writes the authoritative `BatchRun.child_template_json`, and changes the dependency state to `building`.
7. The Environment Controller continues to gate the child until the augmented build is `ready`. A failed or cancelled build changes the Batch dependency state to `failed` and prevents child provisioning.
8. Once ready, the child allocation provisions normally. `submit_window()` reads the authoritative BatchRun child template and still waits for all requested managers to be ready.

The provisional allocation may change exactly once, only while its status is `requested`, no provider resource exists, no manager is registered, and the Batch dependency state is `pending`. Existing Batch rows fall back to the child template stored in the parent execution spec.

An empty Source Package Task List records `dependency_status=skipped`, remains permanently gated, does not
provision child managers, and completes without a supplemental build. Image and config-only Batch submissions use
`dependency_status=not_required` because their execution image already owns Host dependencies.

## Task List contract

The first record means ordinal zero after existing JSONL, version, forbidden-field, and `RunInput` normalization checks. An invalid first line fails the Task List; it is not skipped. All records in one Task List must have the same Host dependency set. This is a producer contract in v1 and is not verified by scanning later records.

The representative item ID, selector digest, selected module paths, Host requirements and child dependency-build ID are persisted on `BatchRun` and exposed in the control response. They do not mutate the immutable Task List manifest.

## CLI

`nexus modules wheel` is a local wrapper around the same analyzer and joint-resolution builder. Its environment must include the `builder` extra (for example, `uv sync --extra dev --extra builder --extra distributed --extra runtime`) because locked dependencies may only publish an sdist. It accepts exactly one of `--run-input` and `--task-list`, plus `--lock`, `--nexus-wheel` and `--output`. Task List mode reads only the first validated record. The command writes a report, resolved dependency lock and wheels; it never installs module Host packages into the current interpreter. Cloud code calls the library, not the CLI.

## Failure codes

- `module_object_not_declared`
- `module_import_invalid`
- `module_requirement_invalid`
- `module_requirement_unsafe`
- `dependency_module_lock_failed`
- `dependency_wheel_build_failed`
- `dependency_wheelhouse_inconsistent`
- `batch_module_dependency_failed`

## Acceptance

Automated tests must cover selector extraction, wheel metadata discovery, unsafe requirements, lock compatibility/conflict, offline verification, Single/Live build specs, first-record-only Batch behavior, empty Task Lists, idempotent binding, fencing and Environment Controller gating.

The end-to-end acceptance target is DevCloud only. A one-item Frontend remote Batch must run with `NEXUS_AUTO_INSTALL` unset, resolve `boto3` and `litellm>=1.91.2,<1.92.0`, keep child managers absent until the augmented dependency build is ready, import both packages inside the child manager venv, and enter real task execution without a missing-package failure.
