# Harbor Native Compose Phase 2 Runtime

## Status

- Implementation branch: `feat/harbor-compose-runtime`
- Depends on: `feat/harbor-compose-plan`
- Scope: provider-neutral topology/service handles and an outer-Runtime Compose controller
- Harbor Task integration: Agent/Oracle/NOP plus shared Verifier enabled

Phase 2 provides the execution substrate required by Harbor Bridge. Native Compose Agent execution uses
an explicit transport capability instead of pretending that an inner service is a full Nexus Runtime.

## Runtime model

```text
Host
└── outer Nexus Runtime
    ├── controller staging
    ├── Docker CLI / Compose v2
    └── task Compose project
        ├── primary service
        └── sidecars / one-shot services
```

The outer Runtime remains the trusted control plane. Inner services are represented by `ServiceHandle`;
they are not full Nexus Runtime instances and do not receive their own runtime server or have their
entrypoints replaced.

`environment.docker_image` selects this outer Controller image. Each inner service image, including the
primary service, remains independently defined in the locked Compose file.

## Public interfaces

`TopologyHandle` owns lifecycle:

- `stage()`
- `start()`
- `wait_ready()`
- `inspect()`
- `service(name)`
- `delete()`

`ServiceHandle` owns service-local operations:

- `exec()`
- `upload_file()` / `upload_dir()`
- `download_file()` / `download_dir()`

`ComposeTopologySpec` is provider-neutral and contains the locked definition digest, primary service,
startup/shutdown budgets, declared service expectations, and task-local bind paths.

## Controller lifecycle

1. Probe Docker Compose and require version 2.20 or newer.
2. Atomically claim an attempt-scoped controller directory.
3. Revalidate every staged path against the task root, rejecting links and special files.
4. Archive and upload only the definition and declared bind inputs.
5. Verify the staged definition SHA-256.
6. Write a mode-0600 Controller-generated override that injects `environment.env` only into the primary service.
7. For Agent runs, add trusted mounts for read-only `/nix` and an attempt-scoped Agent transport directory.
8. Pull service images.
9. Start the project with `up --detach --remove-orphans`.
10. Poll `compose ps --all --format json`.
11. Require daemon services to be running/healthy and one-shot services to exit with code 0.
12. Delete with `down --volumes --remove-orphans`, then remove controller and transport staging.

Pull, up, status polling, and readiness share one startup deadline. Deletion is single-flight and has one
connection-safe total deadline. Ordinary deletion errors remain retryable; a deadline abandonment is not
immediately restarted by deferred cleanup. Cancellation waits only for the remaining shielded deletion
budget before propagating, and the handle enters `deleted` only after Compose and controller cleanup succeed.

## Service execution

Service commands use `docker compose exec -T`. Workdir and user are explicit options. Environment values
are supplied through the outer Runtime's structured `extra_env`; command logs contain only variable names,
not values. Nonzero service exit codes are returned to callers rather than converted into transport errors.

Service operations are rejected before topology start and after failure/deletion.

## Agent transport

Compose-aware Agents receive the shared trusted `ComposeExecutionTarget`. The Nexus Runtime starts the LLM proxy on
Controller loopback, creates an outer Unix-socket relay, starts an inner loopback relay through
`docker compose exec`, and finally launches the Agent CLI in the primary service. The primary receives
neither the Docker socket nor a Controller HTTP endpoint.

The generated override mounts `/nix` read-only so the inner relay can use Runtime-owned binaries, and
mounts only a short, attempt-scoped writable transport directory at `/opt/nexus-agent`. Agent executables,
HOME, proxy sockets, and process metadata stay under that directory. Task Compose definitions cannot
bind-mount over either reserved path. Service tmpfs mounts may hide Runtime source/test subtrees for
isolation, but cannot cover `/nix`, `/nix/runtime`, the `/nix/runtime/nix-env` toolchain, or any part of
`/opt/nexus-agent`.

`claude_code`, `codex`, and `hycode` use the same `AgentExecutionTransport` as Harbor Trial. Harbor Bridge
checks Agent capability metadata before selecting the target, so an unsupported Agent is rejected instead
of running silently in the Controller.

Installed Agents receive an invocation-scoped writable HOME under `/opt/nexus-agent`. Other Harbor
commands that select an explicit non-default `user` use Docker Compose's `--user` semantics only; Bridge
does not emulate a login shell or infer a passwd HOME for arbitrary task images. Tasks whose verifier,
collect hook, Oracle, or direct exec command depends on HOME/profile state must declare HOME explicitly
in the service/call environment and mount that path writable. The transport writes the task workdir as
Git `safe.directory` in the invocation HOME so Agent Git commands use the same configuration they launch
with.

## File transfer

Transfers use attempt-scoped controller staging with `docker compose cp` as the fast path. If
`docker compose cp` cannot access a running service mount (for example, a writable `tmpfs` under a
read-only root filesystem), the service handle falls back to in-service `cat`/`tar` streaming via
`docker compose exec`.

- Host uploads are limited to 256 MiB and reject links/special files.
- Directory archives preserve modes and are validated before Host extraction.
- Service downloads reject links, special files, and archives larger than the transfer limit.
- Symlink files and directory trees containing symlinks are rejected to prevent service-controlled links
  from exposing controller files.
- Temporary paths use shielded best-effort cleanup with a total deadline and never replace the primary error.

This is an initial bounded transport. Large datasets should later use streaming object storage rather than
base64 file APIs.

## Project isolation

Project names must be lowercase, bounded identifiers. Staging claims the project directory with a
non-idempotent `mkdir`; a second handle cannot silently delete or reuse an existing project's control
files. Production callers must derive project names from run and attempt identity.

## Current limitations

- `claude_code`, `codex`, and `hycode` implement the shared `execution_target`
  transport; other Agents fail closed.
- Artifact retention outside the temporary Harbor trial directory is not connected yet.
- No real DinD/Compose integration test yet.
- The task still supplies one authoritative lock; the Controller adds one generated runtime override.

The next phase should reuse the transport for OpenClaw and implement retained sidecar artifacts. Proxy
bypass remains an explicit Compose service configuration rather than a Bridge runtime mode.
