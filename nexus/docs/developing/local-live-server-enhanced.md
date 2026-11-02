# Local Live Server Enhanced Design

## Goals

The local live server should be useful as a durable, inspectable development gateway instead of a purely in-memory
single-process demo. This design covers two sets of enhancements:

- Persist live run metadata and events locally so a server restart can still list and inspect historical run IDs.
- Support pre-warmed cache runs so later requests with the same configuration can reuse an already-started live run.

## Current Baseline

The live server owns one `LiveRunManager` per aiohttp app process. `LiveRunManager` owns many in-memory `LiveRun`
instances. A live run starts `single.run` in a background task and drives human input through the `get_user_input`
hook. When the agent asks for input, `LiveRun` emits `input.required`, sets status to `idle`, and waits on the turn
queue used by `POST /v1/runs/{run_id}/turns`.

This means live runtime objects are process-local. SQLite can persist history, but it cannot restore a runtime after
process restart.

## Persisted Local History

`SQLiteLiveRunStore` stores control and lookup history in one table:

- `live_runs`: current run metadata, request JSON, status, runtime URL metadata, output, error, event cursor, and raw trace
  metadata.

The `live_events` projection table is not retained. Runs with local storage attach a shared
`nexus.runner.trace.local.RunTraceLocalSink`; active events come from `LiveRunTraceView`, while closed events are
reconstructed by `RunTraceProjectionEngine` from raw `SpanEventRecord` JSONL read by `RunTraceLocalSource`. SQLite stores the
trace metadata needed to locate that journal. `LiveRunManager` reads active runs first and falls back to this raw replay
path for historical events.

When the store opens, statuses that imply an in-process runtime is still alive are marked `failed` with an explanatory
error because the runtime object cannot survive process restart.

## Run List API And UI

The server exposes `GET /v1/runs?limit=N`, returning a `RunListResponse`. The web UI uses this API as the source of
truth for the sidebar list. Browser local storage is limited to UI-only state such as draft YAML and locally inferred
titles; it is not used as the run list database.

The CLI accepts a local database path:

```bash
nexus run live --database /tmp/nexus-live.sqlite
```

## Cache Run Semantics

A cache run is a real live run created ahead of time from a normal `CreateRunRequest`. It starts runtime setup, installs
worker state, and advances until the first user-input boundary. At that point the run is held in a `cached` state rather
than exposed as a normal idle user session.

Normal run creation can reuse a cached run when the requested configuration matches the cache run configuration. Reuse
returns the cached run's original `run_id`; it does not copy or rename the run.

## Cache Run API

Create a cache run:

```http
POST /v1/runs/cache
Content-Type: application/json

{ ...CreateRunRequest... }
```

The response returns immediately after scheduling the background run:

```json
{
  "run_id": "run_abc123",
  "status": "starting",
  "events_cursor": 1,
  "reused_cache": false
}
```

The caller can poll `GET /v1/runs/{run_id}` or `GET /v1/runs` until status becomes `cached`.

Create a normal run:

```http
POST /v1/runs
Content-Type: application/json

{ ...CreateRunRequest... }
```

If no matching cache exists, the behavior is unchanged and a new live run is created. If a matching cached run exists,
the manager removes one cached run from the matching cache queue, changes it to `idle`, and returns:

```json
{
  "run_id": "run_cached",
  "status": "idle",
  "events_cursor": 8,
  "reused_cache": true
}
```

If the normal create request explicitly supplies `run_id`, cache reuse is skipped. A caller that requires a specific
run ID gets a fresh run or the existing duplicate-run conflict behavior.

## Cache Key

The cache key is a stable hash of the final `CreateRunRequest` after openai config file merging. It excludes `run_id`
and includes task, agent, runtime, provider, OpenAI clients, tools, processors, runner, and live config. This makes YAML
field ordering irrelevant while avoiding reuse across semantically different configurations.

The manager stores a queue per key:

```text
cache_key -> deque[run_id]
```

This supports prewarming multiple cache runs for the same configuration and consuming them one at a time.

## Status Model

Add one live status:

```text
cached
```

No separate `active` status is needed. A reused run becomes `idle`, so existing turn submission semantics remain
unchanged.

Status flow:

```text
cache create miss: starting -> running -> idle -> cached
normal create hit: cached -> idle
normal create miss: starting -> running -> idle
turn submit: idle -> running -> idle|closed|failed
kill: starting|running|idle|cached -> killing -> killed
```

`POST /v1/runs/{run_id}/turns` accepts only `idle`; it rejects `cached`.

## Cleanup And Persistence

Cached runs are process-local. SQLite may record `cached` for list visibility, but store initialization must mark
`cached` as failed along with `starting`, `idle`, `running`, and `killing`.

The manager removes a run from the cache index when:

- The cached run is reused by normal create.
- The cached run is killed.
- The cached run fails or closes.
- A duplicate or stale cache index entry is discovered while looking up a cache key.

## Tests

Unit coverage should prove:

- `POST /v1/runs/cache` route is registered before `/v1/runs/{run_id}`.
- Cache creation returns immediately and later moves the run to `cached` at the first input boundary.
- Normal create reuses one cached run for identical config and does not call `run_single` a second time.
- Multiple cached runs for the same config are consumed one at a time.
- Different config does not reuse a cached run.
- Explicit `run_id` on normal create skips cache reuse.
- Cached runs reject turn submission.
- SQLite startup marks cached rows failed.

Integration coverage should start an aiohttp live app, create cache runs, verify `GET /v1/runs` shows them, create a
normal run with matching config, and verify the returned run ID is the cached run ID. The integration should run in mock
mode by default and support an environment-enabled real OpenAI chat completion mode using credentials from `.env`.
