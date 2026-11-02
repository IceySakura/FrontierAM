# Harbor Native Compose Phase 1 (Historical)

> This document records the original contract-only Phase 1 design. Native
> Compose runtime execution, Agent transport, and shared/separate Verifier
> support are now implemented; see `harbor-native-compose-runtime.md` and the
> Harbor Bridge README for the current behavior.

## Status

- Historical implementation branch: `feat/harbor-compose-plan`
- Scope: task contract, locked-definition parser, admission, identity, and fail-closed execution
- Runtime support at this phase: not enabled
- Verifier mode at this phase: `shared` only

## Goal

Phase 1 defines which prebuilt Docker Compose tasks Nexus may execute and freezes their runtime-relevant
inputs before a sandbox is allocated. It does not start Docker Compose. A task that passes compilation
produces a typed `ComposePlan`; Harbor Bridge still rejects execution until the controller runtime from
Phase 2 is connected.

The design keeps the Compose lock as the only authority for services, entrypoints, commands, networks,
volumes, dependencies, and service healthchecks. `task.toml` only selects the definition, binds Harbor's
logical main role to a Compose service, and declares lifecycle budgets that are not represented by the
Compose file itself.

## Task contract

```toml
schema_version = "1.2"

[environment]
docker_image = "mirrors.tencent.com/rl_task/nexus-compose-controller@sha256:..."
workdir = "/workdir"
cpus = 8
memory_mb = 16384
enable_dind = true
network_mode = "public"

[environment.env]
TASK_MODE = "evaluation"

[environment.healthcheck]
command = "test -f /workdir/.initialized"
timeout_sec = 10
interval_sec = 2
retries = 30

[agent]
timeout_sec = 3600
user = "root"

[verifier]
environment_mode = "shared"
timeout_sec = 1800

[extensions.nexus.compose]
definition = "environment/docker-compose.lock.yaml"
primary_service = "main"
startup_timeout_sec = 180
shutdown_timeout_sec = 30
one_shot_services = ["migrate", "seeder"]

[[verifier.collect]]
service = "db"
command = "pg_dump -U app app > /tmp/database.sql"
timeout_sec = 60

[[artifacts]]
service = "db"
source = "/tmp/database.sql"
destination = "db/database.sql"
```

Only the six fields shown under `[extensions.nexus.compose]` are accepted. Unknown fields fail compilation.

## Field semantics

### `definition`

Task-relative path to the production Compose lock. The path must remain inside the task bundle, must not
contain symlinks, and must resolve to a regular YAML file with a non-empty `services` mapping.

The task author does not provide a digest. The compiler computes SHA-256 internally, stores it in
`ComposePlan`, includes it in metadata, and the future runtime will verify it after TaskFiles materialization.

### `primary_service`

Compose service corresponding to Harbor's logical main service. In v1 it is the execution target for Agent,
shared Verifier, `environment.workdir`, `environment.env`, and the task-level healthcheck. Artifact and
collect entries without a sidecar service also resolve to it.

The service must exist, be long-lived, and not be a one-shot service. Its business image is defined by
the Compose lock and is independent of `environment.docker_image`; an explicit Compose `working_dir`,
when present, must equal `environment.workdir`.

### `startup_timeout_sec`

Total topology-start budget. It does not alter individual Compose healthcheck probes. The value must be
a positive integer; the outer Runner timeout and Sandbox TTL provide the overall execution bounds.

### `shutdown_timeout_sec`

Grace period for topology shutdown before forced cleanup. The value must be a positive integer.

### `one_shot_services`

Services for which exit code 0 is a successful terminal state. The compiler also derives one-shot services
from `depends_on.condition=service_completed_successfully`. One-shot services must exist, cannot be the
primary service, and cannot use an automatic restart policy.

### Proxy bypass

The Bridge does not synthesize proxy bypass configuration. A task that injects HTTP proxy variables must
set concrete `NO_PROXY` and `no_proxy` values in its Compose services for loopback and service DNS names.

## Existing Environment fields in Compose mode

| Field | Compose meaning |
|---|---|
| `docker_image` | Outer Gongfeng Sandbox / Compose Controller image |
| `workdir` | Agent/Verifier workdir inside `primary_service` |
| `cpus`, `memory_mb` | Total controller and topology scheduling budget |
| `enable_dind` | Required to be explicitly `true` for the Gongfeng controller implementation |
| `network_mode`, `allowed_hosts` | External topology egress policy; not internal Compose DNS |
| `environment.env` | Future trusted overlay for `primary_service`; never controller environment |
| `environment.healthcheck` | Additional check executed in `primary_service` after topology readiness |
| `environment.upload` | Future post-start upload target in `primary_service` |
| `environment.mcp_servers` | Agent-visible MCP configuration from the primary-service network view |

Native Compose executes `environment.boot` or `HARBOR_BOOT_CMD` in `primary_service` after topology
readiness and environment uploads, but before the task healthcheck and Agent. The Controller still owns
up/wait/down, so boot is a pre-Agent initialization hook rather than a second Compose lifecycle
owner. The v1 contract rejects `environment.browser` until a browser transport exists between the
Controller and the inner service.

## Prebuilt lock requirements

The production lock:

- contains no `build` or `pull_policy: build`;
- uses prebuilt images from an explicit registry; `@sha256` remains recommended but is optional;
- may use Compose interpolation, Controller environment inheritance, external resources, and fixed resource names;
- contains no Compose `include`;
- declares every named volume and network referenced by a service;
- keeps all relative bind sources inside the task bundle;
- does not contain symlinks in bind-source trees.

The admission allowlist also passes through `deploy`, `env_file`, `configs`, `cap_drop`, `shm_size`,
and `ulimits`. Their nested values are left to Docker Compose validation in the current phase;
relative files referenced by `env_file` or `configs.file` are not yet added to identity or staging.

Every inner service, including `primary_service`, defines its own prebuilt business image in the Compose
lock. The Controller image is independently supplied by `environment.docker_image`.

The compiler rejects task-controlled container names, privileged containers, Host/container/service
namespace sharing, Docker API sockets, `volumes_from`, device/cgroup/runtime controls, Host gateway
aliases, and unsupported service fields. Network and volume definitions may not override global names or
use driver options; only bridge/local drivers are accepted in v1.

Environment values must be concrete. Mapping entries with null values and list entries without `KEY=value`
are rejected because Compose would inherit values from the controller process.

## Dependency and health semantics

`depends_on` may use:

- `service_started`;
- `service_healthy`;
- `service_completed_successfully`.

Long-form `required` and `restart` flags must be booleans. Dependency cycles are rejected. A
`service_healthy` dependency must expose a non-disabled healthcheck with a concrete test; `disable: true`
and `test: ["NONE"]` do not count.

## Volume semantics

Short and long volume syntax are parsed separately:

- `/container/path` is an anonymous volume target;
- `name:/container/path` is a declared named volume;
- `./path:/container/path` is a task-local bind;
- absolute sources are passed through to the isolated Controller;
- home-relative, parent-relative, undeclared, or malformed relative sources are rejected.

Long syntax validates type-specific fields and requires an absolute container target. Task-local bind
paths are recorded in `ComposePlan`; absolute Controller paths, external resources, image tag contents,
and inherited environment values are not part of task identity.

## Artifact and collect references

Harbor's existing `artifacts` and `verifier.collect` schemas are reused. A missing service or logical
`main` resolves to `primary_service`; explicit sidecar names must exist in the lock. Phase 2/3 service
handles will provide the actual exec and transfer operations.

## Identity

Single-container identity is unchanged. Native Compose identity additionally hashes:

- the locked Compose definition;
- every referenced task-local bind file or directory;
- file and directory permission modes.

This makes topology, mounted configuration, and executable-bit changes produce a new run ID. Inputs are
hashed with a Compose-specific domain separator.

## Serialized plan

The compiler adds a strict, frozen plan to task config:

```json
{
  "schema_version": "nexus-compose-plan-v1",
  "definition": "environment/docker-compose.lock.yaml",
  "definition_sha256": "sha256:...",
  "controller_image": "registry/nexus-compose-controller@sha256:...",
  "primary_service": "main",
  "primary_image": "registry/main@sha256:...",
  "workdir": "/workdir",
  "environment": {"TASK_MODE": "evaluation"},
  "startup_timeout_sec": 180,
  "shutdown_timeout_sec": 30,
  "one_shot_services": ["migrate"],
  "services": {},
  "bind_paths": ["environment/config"]
}
```

The Dataset metadata records execution kind, definition digest, primary service, and service names.

## Fail-closed execution

Phase 1 deliberately does not execute Compose. A serialized plan causes Harbor Bridge to fail before
allocating a runtime. The task also re-reads the authoritative materialized `task.toml`; deleting or
overriding `task_config.compose_plan` cannot fall back to the single-container path.

## Legacy behavior

- A task without `[extensions.nexus.compose]` follows the existing single-container path unchanged.
- Existing hand-written `enable_dind + environment.boot` sidecar adapters remain valid as legacy tasks.
- Native Compose routes `environment.boot` into the primary service.
- Legacy Generator and DatasetReader share the same compiled body and identity behavior.
- `HARBOR_SKIP_INVALID_TASKS=1` skips definition, identity, and admission failures.

## Validation

Phase 1 includes unit coverage for valid plans, image/workdir/service cross-checks, one-shot derivation,
dependency cycles, healthcheck semantics, short/long volumes, network references, interpolation,
environment inheritance, unsafe fields, path traversal, symlinks, identity changes, Generator skipping,
and runtime fail-closed behavior.

The focused and Harbor regression suites passed before the Phase 1 commit. Runtime execution and real
13-task admission remain follow-up work; the comparison directory currently contains notes and checksums
but not the two referenced source archives.
