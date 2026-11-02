# aidata confine kit

User-space deny-list sandbox for aidata's shell MCP — a **filesystem** path
deny-list plus a **network** host deny-list. Plug-and-play across any
glibc-based runtime image — no Dockerfile changes, no `image: tag` bumps. The
kit compiles itself inside the sandbox at task start.

## What it does

Replaces the implicit "agent can't reach internal dirs because cwd is pinned
to `/workspace/desktop`" hand-wave with a real syscall-level wall:

```
agent (via shell MCP run_command)
  → bash -c <command>           ← LD_PRELOAD=/workspace/.aidata/confine/libconfine.so
  → libc open / openat / stat / opendir / unlink / rename / chmod / execve
                                 → libconfine intercepts → EACCES if path is denied
```

**Default deny prefixes** (set by `engine/paths.py::CONFINE_DENY_DEFAULT`,
overridable via `AIDATA_CONFINE_DENY`):

| Prefix                       | Rationale                                                      |
| ---------------------------- | -------------------------------------------------------------- |
| `/workspace/.aidata`         | Engine-private root: `solution/`, `state/`, `context/`, `case/`, `servers/`, and the `confine/` kit itself. |
| `/var/lib/mock-server`       | MockServer state directory (`MOCK_SERVER_DATA_DIR`); contains the sqlite oracle that `rubric_sqlite_path` points at. |
| `/var/lib/jd_backend`        | woocommerce backend store + media for `channel_inventory_ppt`. |
| `/opt/mock-server/src`       | MockServer source tree (oracle answers live here); previously soft-hidden via `chmod -R o-rwx` in case `setup.sh`. |
| `/opt/mock-server/vendor`    | MockServer third-party vendor tree.                            |
| `/nix/runtime/nexus`         | Nexus runtime/control-plane source. The runtime server runs from here and exposes the `invoke`/`call_tool`/`call_executor` RPC; readable source let agents grep the tool registry + RPC contract and try to bypass MCP. PATH (`/nix/runtime/nix-env`) and the agent venv (`/opt/aio-deps`) are siblings/elsewhere, so denying this does not break the agent's shell. |

NOT denied (agent must keep using these): `/workspace/desktop/*`,
`/workspace/skills/*`, `/workspace/tests/*`, `/opt/aio-deps/*`,
`/opt/mcp-benchmarks/*`, `/opt/mock-server/.venv/*`,
`/opt/mock-server/scripts/*`, `/opt/wecom-env/*`, `/tmp/*`, `/proc/*`,
`/etc/*`, `/usr/*`, `/lib*/*`, `/bin/*`, `/sbin/*`.

## Network deny

A second, independent deny-list blocks **name resolution** of public hosts the
agent must not reach (today: the public SkillHub store — tasks must use the
in-sandbox `mcp-skillhub` mirror instead). Source of truth:
`engine/paths.py::CONFINE_NET_DENY_DEFAULT` (comma-separated host suffixes),
overridable via the `AIDATA_CONFINE_NET_DENY` env var.

The **one host list drives two cooperating layers**:

1. **`getaddrinfo()` hook (libconfine.c)** — returns `EAI_NONAME` for any host
   that equals, or is a subdomain of, a deny entry (suffix match: `skillhub.cn`
   covers `api.skillhub.cn` but NOT `evilskillhub.cn`). Every dynamically-linked
   resolver — curl, wget, python `socket`, node — fails with "could not resolve
   host". Re-injected across `execve`/`execvp` exactly like `AIDATA_CONFINE_DENY`,
   so `env -u AIDATA_CONFINE_NET_DENY curl …` cannot strip it.
2. **`no_proxy` force-direct (shell_server.py `_apply_net_deny_no_proxy`)** —
   the sandbox injects an `http(s)_proxy`; a bare `curl skillhub.cn` would
   otherwise tunnel via the proxy, which resolves the host on the *proxy* side,
   so layer 1's getaddrinfo hook never sees it. We append the deny hosts to
   `no_proxy` (MERGING with the inherited value, so `.woa.com` etc. stay direct)
   to force a direct connection, where layer 1 applies.

### What the network deny does NOT cover

This is a guard-rail against accidental/lazy public-store access, **not an
adversarial wall**. A determined agent can still:

* **connect to a raw IP** (`curl https://1.2.3.4 -H 'Host: …'`) — no name
  resolution happens, so layer 1 never sees it. (Blocking by IP is intentionally
  out of scope: the store is CDN-fronted, so IPs rotate and are shared.)
* **re-set its own `http_proxy`** to re-route through the proxy past `no_proxy`.
* **resolve via a non-NSS path** (`dig`/`nslookup` straight to a DNS server) and
  then connect by IP.
* **use a statically-linked binary or raw `syscall`** — same LD_PRELOAD ceiling
  as the filesystem hooks (no seccomp / NET_ADMIN in gongfeng).

A true hard wall requires **egress policy on the sandbox proxy itself**
(domain/SNI deny-list, outside the container, unbypassable). That is an
infra-level control, not something this kit can provide. Unlike the filesystem
deny, the network deny has **no mount-namespace fallback** (would need iptables
+ `CAP_NET_ADMIN`), so it is LD_PRELOAD-only.

## Time faking

The same `.so` also fakes the **wall clock** so the agent's shell sees the
mock's "today" instead of the container's real date. This closes a determinism
hole: MockServer freezes its clock at `MOCK_SERVER_NOW` (see
`MockServer/_common/clock.py`), but the agent learns "today" by running
`date` — which, unhooked, returns the container's real time. A case that pins
`MOCK_SERVER_NOW=2026-04-21` while the container runs on 2026-06-07 would have
its mock data and the agent's `date` disagree by weeks.

```
agent → bash -c "date"          ← LD_PRELOAD=…/libconfine.so, AIDATA_FAKE_NOW=<epoch>
  → libc time / gettimeofday / clock_gettime(CLOCK_REALTIME)
                                 → libconfine adds a fixed offset → mock "now"
```

* **Activation**: gated on `AIDATA_FAKE_NOW` (unix epoch seconds). The engine
  derives it host-side from the case's `MOCK_SERVER_NOW`
  (`mcp_catalog.fake_now_epoch`) and only injects it into the shell MCP env
  when a service actually pins a frozen clock (`task.py::_resolve_mock_now`).
  Cases with no frozen clock get **no** `AIDATA_FAKE_NOW`, so the wall clock is
  untouched and their behaviour is byte-for-byte unchanged.
* **Offset, not frozen**: `offset = desired_now − real_now` is captured once at
  `.so` load; the wrappers add it, so the faked clock keeps advancing at the
  real rate from the mock anchor. We deliberately do **not** freeze REALTIME —
  a frozen clock breaks `CLOCK_REALTIME`-deadline polling loops and
  `pthread_cond_timedwait` (the classic libfaketime hang). Over a single
  rollout the drift vs MockServer's frozen value is seconds — never a day
  boundary — so "today's date" stays consistent, which is the whole point.
* **REALTIME only**: `CLOCK_MONOTONIC` (and `*_RAW` / `BOOTTIME` / CPU clocks)
  pass through untouched, so `sleep`, subprocess timeouts, and benchmark timers
  still measure true elapsed time.
* **Survives `exec`**: `AIDATA_FAKE_NOW` is re-injected across
  `execve`/`execvp`/`execvpe` exactly like the deny lists, so
  `env -i date` / `env -u AIDATA_FAKE_NOW date` still observe the faked clock.
* **Override**: set `AIDATA_FAKE_NOW` (epoch seconds) in the environment to
  force a clock for ad-hoc testing regardless of the case.

What it does NOT cover: statically-linked binaries and raw `syscall` /
`vDSO`-only paths that skip the libc symbols (same LD_PRELOAD ceiling as the
filesystem hooks). `date`, Python `datetime.now()`, Node, and other
dynamically-linked tools all go through the hooked symbols.

## How it ships

This directory lives inside the engine. Three integration points:

1. **`engine/task.py`** uploads the whole dir into the sandbox via
   `upload_dir(CONFINE_DIR_HOST, CONFINE_DIR_IN)`, then runs `compile.sh`
   to produce `libconfine.so` against the runtime image's exact glibc.
2. **`engine/mcp_catalog.py`** wires `LD_PRELOAD` + `AIDATA_CONFINE_DENY`
   into the shell MCP server's environment.
3. **`engine/servers/shell_server.py`** has the Python-side fail-closed
   gate: if the `.so` is missing under `AIDATA_CONFINE_MODE=enforce`,
   `run_command` refuses to execute and returns exit 99 with a
   `[confine]` message in stderr.

## Files

| File              | Purpose                                                      |
| ----------------- | ------------------------------------------------------------ |
| `libconfine.c`    | All hook implementations + path matching.                    |
| `compile.sh`      | In-sandbox build (`cc -O2 -fPIC -shared`) + dlopen smoke test. Run at task start by `engine/task.py`. |
| `README.md`       | This file.                                                   |

## Modes

`AIDATA_CONFINE_MODE` (read by `shell_server.py`):

| Value     | Behaviour                                                    |
| --------- | ------------------------------------------------------------ |
| `enforce` (default in `mcp_catalog.py`) | `.so` missing → refuse to execute (exit 99). `.so` present → inject LD_PRELOAD + DENY into bash subprocess. |
| `warn`    | `.so` missing → warn on stderr, run unconfined. Use during dev / image-bring-up to avoid blocking the agent before the kit is wired through. |
| `off`     | Skip everything; pure baseline. Used by `tests/test_sandbox_confinement.py` for the no-confine attack-matrix runs. |

## Defence layers

The kit deploys two independent defences in priority order:

1. **LD_PRELOAD libc-hook layer** — primary. Hooks every libc path-taking
   entry (`open` / `openat` / `__open_2` / `__openat_2` / `stat` family /
   `opendir` / `unlink` / `rename` / `chmod` / `execve` / `execv` /
   `execvp` / `execvpe`). `dirfd`-relative variants resolve via
   `/proc/self/fd/<dirfd>` so `grep -r /workspace/` style traversal —
   which uses `openat(parent_dirfd, ".aidata", ...)` — still maps back
   to a deny prefix.

2. **Mount-namespace layer** — defence-in-depth, best-effort. When the
   sandbox grants `CAP_SYS_ADMIN` (gongfeng currently doesn't, most
   Docker daemons do for unprivileged containers), `confine_init`
   `unshare(CLONE_NEWNS)` and bind-mounts an empty tmpfs over each deny
   prefix. Any caller — including statically linked binaries that skip
   libc — sees an empty directory. Falls through silently when unshare
   returns EPERM.

## What it does NOT cover

* **Static binaries on a sandbox without CAP_SYS_ADMIN**: layer 1 only
  helps dynamically linked code. If the runtime image ships statically
  linked utilities AND the container blocks `unshare`, those tools
  bypass everything. Audit images:
  `find / -perm /x | xargs file 2>/dev/null | grep "statically linked"`.
* **Direct syscalls without CAP_SYS_ADMIN**: `python3 -c "import ctypes;
  ctypes.CDLL(None).syscall(SYS_openat, ...)"` skips libc. Mitigation
  needs seccomp.
* **`..`-only obfuscation crossing a deny boundary**: the kernel resolves
  `/foo/../workspace/.aidata/x` to `/workspace/.aidata/x` after libc has
  already passed the literal string down. Practically every `..` attack
  starts WITH a deny prefix and adds noise — those still match. If a
  case needs full canonicalisation, swap `is_denied_raw()` for a guarded
  `realpath()` call (template comment in `libconfine.c` explains).

## Bypass test matrix

`tests/test_sandbox_confinement.py` enumerates 30+ attack vectors covering
the shell MCP's exposed surface. Run with `AIDATA_CONFINE_MODE=enforce`
to verify each one is blocked. Any failure there is either a confine bug
or a new bypass class — both worth a PR.

## Trade-offs vs prebaked image

| Dimension            | Prebaked `/opt/aidata-confine/`           | This kit (in-engine)                 |
| -------------------- | ----------------------------------------- | ------------------------------------ |
| First-time setup     | Dockerfile change → build → push → bump tag | Add this dir, commit                 |
| Multi-image upkeep   | Bake into each image (aio-agent-rc / aidata-aio-agent / work_mcp_mirror / forks) | One source of truth in the engine   |
| `.so` upgrade        | Rebuild + repush + repin all images       | `git pull` + run                     |
| Per-task overhead    | 0                                         | ~150 ms compile + small upload       |
| Integrity protection | image readonly layer                      | `.so` lives under `/workspace/.aidata/`, which is itself denied — agent can't tamper |
| ABI compat           | Pinned at image build                     | Always matches the runtime's glibc (compiled in-place) |

## Editing the deny list

`engine/paths.py::CONFINE_DENY_DEFAULT` is the source of truth. Adding a
new path:

1. Append to `CONFINE_DENY_DEFAULT` in `engine/paths.py`.
2. Run `pytest tests/test_sandbox_confinement.py` baseline + enforce.
3. If the new path overlaps something the agent legitimately needs (e.g.
   you accidentally denied `/opt/aio-deps/.venv`), the test harness's
   sandbox prep will fail with `python3: command not found` or similar —
   trim back.

Per-case overrides go through the `AIDATA_CONFINE_DENY` env var that
`mcp_catalog.py` injects into the shell MCP; you can extend it with
service-startup additions without touching engine code.

To edit the **network** deny list, change `CONFINE_NET_DENY_DEFAULT` in
`engine/paths.py` (comma-separated host suffixes) or override per-run via the
`AIDATA_CONFINE_NET_DENY` env var. The same value feeds both the `getaddrinfo`
hook and the `no_proxy` force-direct merge — no other wiring needed.
