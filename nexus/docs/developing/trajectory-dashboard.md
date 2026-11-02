# Offline trajectory dashboard

Nexus provides a final-only trajectory exporter and an independent, read-only
browser. The canonical Gateway `Trajectory` is the only source of facts:

```text
Gateway finalization → trajectory_file observer → JSON artifact + summary
                                                ↓
                             nexus dashboard --path OUTPUT_DIRECTORY
                                                ↓
                              read-only API → lanes / topology / details
```

HyTuner owns the experiment/run directory and passes it to Nexus. Nexus does not
invent experiment paths, create its own run hierarchy, or need HyTuner to remain
running. This MVP supports local files and mounted Ceph directories, not S3/RGW
URLs, central databases, or live/follow updates. Experiment-name resolution is
intentionally deferred until HyTuner provides a directory convention or resolver.

## Enable export

Merge this block into the existing run configuration. Supply a non-empty
`args.output_dir` explicitly; there is no environment-variable fallback.
HyTuner or another caller passes its chosen save directory through this Observer
configuration. In distributed deployments, the directory must be accessible to
the actual worker executing Nexus `run()`.

```yaml
gateway_trajectory_observers:
  files:
    name: trajectory_file
    args:
      output_dir: /mnt/ceph/hytuner/experiment-a/run-001
      compress: true
```

No Ceph root, credentials, experiment, or run ID is hard-coded in Nexus. There is
no change to Langfuse or to training/rollout serialization.

## Browse

Run from the independent worktree after `uv sync --extra dev`:

```bash
uv run nexus dashboard --path /mnt/ceph/hytuner/experiment-a/run-001
uv run nexus dashboard --path /mnt/ceph/hytuner/experiment-a/run-001/ARTIFACT.trajectory.json.gz
```

The default listener is `127.0.0.1:8766`. Override `--port` and `--host` explicitly.
There is no authentication: keep the loopback default, use SSH forwarding, or put
an authenticated reverse proxy in front of it. Do not expose private trajectories
on an untrusted network. Native request/message facts may contain sensitive data;
this exporter does not promise redaction. Frontend text is not interpreted as HTML
and it does not automatically fetch remote images or load external scripts.

The dashboard requires an explicit `--path` and never generates trajectories.
Enable the Observer above in a real run, then browse its output directory.
Synthetic fixtures are confined to the test suite and are not bundled in Nexus.

## Artifact contract

Each final callback creates a unique random artifact ID, independent of untrusted
session names. Multiple workers and reused session IDs cannot overwrite each
other. Files are created in the supplied directory with no extra run hierarchy:

```text
<artifact-id>.trajectory.json.gz
<artifact-id>.summary.json
```

`compress: false` emits `.trajectory.json` instead. The artifact envelope contains
artifact/session IDs, creation
time, state, final reward, termination reason, counts, explicit omissions, and a
`trajectory` field using the existing `gateway-trajectory-v1` schema. Only the
trajectory `schema_version` is versioned and validated; the envelope and summary
have no separate version. Node IDs,
physical parent/children, Chain membership roots, spawn/join evidence, segment
boundaries, messages, provider usage and recorded timestamps are retained.

A shallow serialization view excludes `segment_training_data` without modifying
the source or deep-copying tensors. `Trajectory.dump(include_payload_refs=False)`
excludes ephemeral token-store references. External token payloads, routed experts,
logprobs, and training tensors are not fetched. Inline token fields already
supported by `Trajectory.dump()` remain available. This is a visualization
artifact, not a replacement for the training checkpoint/serialization format.

Both files use same-directory temporary files, fsync, and atomic rename. The
summary is published last. The listing reads small summary files rather than
full trajectories; a complete artifact without a summary remains recoverable.
A failed/partial temporary write is not catalogued. Concurrent append to shared
JSONL files is deliberately avoided. Compression and filesystem writes are
synchronous on finalization; there is no background writer/retry queue yet.

Existing `TrajectoryObservationHub` deduplicates final publication per session,
isolates/logs Observer errors, and invokes this Observer only for final events.
Recorded failed/cancelled final trajectories can be exported too. A kill before
final publication, unavailable mount, or unsupported JSON value can still prevent
export. There is no guaranteed recovery or periodic snapshot in this MVP.

## Presentation semantics

- **Behavior lanes:** one lane per recorded Chain; unclaimed nodes use the model's
  default `main` membership but are labelled unclassified when no Chain exists.
- **Physical topology:** every physical parent edge is preserved, including cross-
  Chain edges and rejected branches. No edge is inferred from timestamp order.
- **Spawn/join:** resolved links connect the exact recorded parent node and an
  explicit Chain endpoint. A join endpoint represents the whole Chain, not a
  fabricated child completion node. Unresolved links remain in Chain details and
  produce a visible warning rather than guessed edges.
- **Compaction:** a recorded segment boundary is highlighted at its original node.
  It is not inferred from message text or turned into a synthetic subagent.
- **Aggregation:** only linear, same-label/type messages away from roots, branches,
  boundaries, cross-Chain edges and spawn/join anchors are grouped. Every group
  preserves the original node IDs and can be expanded.
- **Layout:** topological order, not elapsed time. Original timeline records are
  shown in details; the horizontal distance has no duration semantics.
- **Details:** native message blocks, thinking, tool calls/results, request kwargs,
  provider usage, segment evidence and complete raw node facts are inspectable.
  Conversation mode is Chain-scoped assembly order, not a claimed selected path.

The custom frontend uses bundled HTML/CSS/JS + SVG with no build step, CDN, or
third-party graphics dependency. The service uses the existing `aiohttp`
dependency, following Nexus's current live-service stack rather than adding
FastAPI solely to resemble Miles. It still requires the read-only API to run.

## API and limits

- `GET /api/trajectories?offset=0&limit=50&q=SESSION`: paginated catalog (limit ≤ 200),
  total count, and visible file read errors. Directories are scanned non-recursively.
- `GET /api/trajectories/{id}`: full trajectory envelope and derived graph. IDs are
  generated from catalogued filenames, JSONL line numbers and client selectors,
  never interpreted as user-supplied paths.
- `GET /`, `/assets/app.js`, `/assets/style.css`: bundled frontend. No arbitrary
  directory/static file serving and no mutation endpoints.

Accepted inputs are Observer artifacts, raw `Trajectory.dump()` JSON/JSON.gz,
run-result JSON/JSON.gz containing `processor_results`, and uncompressed batch
JSONL with one result object per line. A processor result (usually `gateway`)
may hold one trajectory or a mapping of client names to trajectories. Custom
processor keys are recognized by the Gateway schema at these same two levels;
the reader does not search arbitrary nested objects or accept JSON arrays of
runs. Each processor/client/line becomes a separate catalog item, even when
session IDs repeat. Source filename, 1-based line number, client and batch Run ID
are shown in the catalog and details. Inline `node_payloads` are retained without fetching
external payload references.

```bash
uv run nexus dashboard --path /path/to/single-result.json
uv run nexus dashboard --path /path/to/batch-results.jsonl
```

Directory discovery includes `*.json`, `*.json.gz` and `*.jsonl`, non-recursively,
excluding hidden files and `*.summary.json` sidecars. Observer artifacts still use
the small sidecar for fast listing. Unrelated/invalid JSON, missing Gateway
results, malformed lines and invalid clients are reported with their source
location without blocking other records. Blank JSONL lines are skipped.
External symlinks are rejected. Compressed JSONL must be decompressed first.

The default read limit remains 64 MiB per JSON document (both on disk and after
gzip decompression), or per JSONL line excluding its line terminator. The run
wrapper and all clients in one record count toward this limit; the entire JSONL
file may exceed it. Oversized lines are drained in bounded chunks and skipped.
Override the limit explicitly for trusted larger records:

```bash
uv run nexus dashboard --path /mnt/ceph/hytuner/experiment-a/run-001 --max-artifact-mib 1024
```

`--max-artifact-mib` accepts positive integer MiB (1 MiB = 1024 * 1024 bytes).
The example allows up to 1 GiB of JSON; it is not a memory budget or a claim of
validated 1 GiB performance. Both file and decompressed size checks remain active;
there is no unlimited mode. Summary files retain a separate fixed 1 MiB limit.
The Python application factory also accepts `create_app(path, max_bytes=...)`.

The reader keeps summary-based listing and whole-trajectory loading on selection.
Node inspection uses the already loaded trajectory, with no separate node/media
API or media extraction. The first JSONL scan parses one bounded line at a time
and caches only summaries, errors and byte offsets, not trajectory bodies.
Selecting a JSONL item seeks directly to its line; selecting a client in a JSON
file still parses the whole bounded document. Index memory grows with record count.
File metadata is checked before and after indexing/loading; changed or truncated
files require an explicit catalog refresh and re-selection, rather than reusing
stale offsets. No background follow/watch or persistent index is provided.

GUI trajectories of hundreds of MiB are an expected future workload to validate,
not automatically invalid data. JSON parsing, graph projection, response encoding,
and browser state can use substantially more memory than the artifact size, and
full graph/SVG rendering is not virtualized yet. Real GUI samples must be measured
before claiming usability at that scale or introducing node-level loading or
separate media storage.

## Inline images

Images remain embedded in the existing trajectory JSON/JSON.gz. No schema changes,
media uploads, external fetches, or separate media files are required. The UI
recognizes PNG, JPEG, WebP and GIF Base64 in these native message blocks:

- OpenAI Chat: `image_url.url = data:image/...;base64,...`.
- OpenAI Responses: `input_image.image_url = data:image/...;base64,...`.
- Anthropic: `image.source = {type: base64, media_type: image/png, data: ...}`,
  including images nested inside `tool_result.content`.

Both node inspector and Chain conversation show image cards with decoded pixel
size, encoded-file byte size, and click-to-enlarge dialogs (Escape to close).
Image sources are assigned when cards approach the visible area, and detached
pending cards are released on view changes. This defers image decoding, not the
whole-trajectory transfer/JSON parsing. Broken data shows an explicit error while
retaining the original facts. External URLs, local file references, SVG and other
unsupported types are not rendered or fetched.

Raw fact previews fold Base64 strings without modifying stored data. Expanding a
node's raw facts is lazy; a second explicit action displays the complete JSON,
which may be expensive for large payloads.

## Verification

```bash
uv run pytest tests/unit/dashboard tests/unit/gateway/test_file_trajectory_observer.py -q
uv run ruff check nexus/dashboard nexus/gateway/observers/file.py nexus/cli/commands/dashboard.py nexus/cli/main.py
uv run ty check nexus/dashboard nexus/gateway/observers/file.py nexus/cli/commands/dashboard.py
node --check nexus/dashboard/static/app.js
node --check nexus/dashboard/static/media.js
node --test tests/unit/dashboard/test_media.cjs
```

The Python suite generates its own test artifacts. Pytest automatically runs the
media JavaScript tests through Node.js, with a 30-second timeout and captured
output on failure. If Node.js is not installed, only that test is skipped; the
remaining Python tests do not require it. Install Node.js in CI to include this
coverage. The standalone Node.js command above remains available for focused runs.
These checks do not validate frontend interactions, real-browser rendering, or
large GUI trajectory performance.

The broader Gateway test directory additionally needs optional dependencies such
as NumPy; these are not introduced as dashboard runtime dependencies.
