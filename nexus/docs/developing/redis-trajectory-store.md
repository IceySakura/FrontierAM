# Redis trajectory store

`RedisTrajectoryRecordStore(namespace, config=None)` implements the asynchronous trajectory
record contract using the synchronous Redis client. Nexus also supports memory
and Ray trajectory stores. The store inherits `TrajectoryRecordStore` and owns one
`RedisClient`, reusing its connection pool. Every Redis client also connects to
the FS service for large-field storage.
Synchronous I/O runs in worker threads; each operation has its own pipeline.
Cancellation waits for that operation to finish before propagating, without
closing the shared client. After all operations finish, the owner can call
`close()` or leave the synchronous context manager to close the client.
`release_many` deletes records and keeps the client available for further calls.

Client initialization happens in the store constructor, including Redis Cluster
connection discovery and Controller topology discovery. Connection errors can therefore occur during construction.
Stores stay in the process that creates them; other processes construct their
own stores from configuration and exchange serializable refs.

```python
from nexus.gateway.trajectory_store import RedisTrajectoryRecordStore, NodePayloadRecord, SegmentDataRecord

store = RedisTrajectoryRecordStore('training-run-1')  # Ray is already initialized; defaults use its head.
node_ref, segment_ref = await store.put_many([
    NodePayloadRecord('trajectory-1', 3, token_ids=[10, 11], logprobs=[-0.1]),
    SegmentDataRecord('trajectory-1', 'main@3', {'reward': None}),
])
await store.put_fields_many({segment_ref: {'ref_log_prob': [-0.2]}})
assert await store.get_many([node_ref], fields=['token_ids']) == [{'token_ids': [10, 11]}]
await store.release_many([node_ref, segment_ref])
```

Ordinary `redis://` and `rediss://` URLs detect standalone or Cluster mode with
`INFO cluster` when the client is constructed. Cluster connections preserve the
URL's authentication, TLS and connection options and discover the remaining nodes.
The account needs permission to run `INFO`; connection, authentication and discovery
errors propagate without falling back to a direct client. Explicit
`redis+cluster://host:port/0?nodes=host2:port2,host3:port3` and Sentinel URLs remain supported.
Without a URL, clients discover the head of the Ray cluster already connected in
the current process and use `redis://<ray-head>:16379/0`. An explicit URL takes
precedence. Clients do not connect to or start Ray automatically; missing Ray
initialization or a discoverable head raises an error when a default address is
needed. The Controller address is configured independently and defaults to the
same head at port `17000`; an explicit Redis URL does not change it. `CHIEF_IP`
does not supply these defaults. Start both services with `nexus redis up` before
constructing clients. If a dedicated Redis host list excludes the Ray head, pass
the published Redis URL explicitly; the FS Controller still runs on the head.

## Data and identity

`NodeRecord` and `SegmentRecord` are frozen dataclasses inheriting both `Serializable`
and the corresponding Nexus Gateway record type. Their fields and constructor signatures
match the Nexus records exactly. They are storage representations in `trajectory_store_backends.redis`; callers can publish Nexus Gateway record types directly.

Pickle stores each class's module path. Readers and writers must use the same
Nexus version. Records and refs from the former external package are unsupported.

Node records keep `token_ids`, `logprobs`, `topk_indices` and `topk_values` in Redis;
non-`None` `routed_experts` and `multi_modal_items` values are stored in DataUnits. Node reads always include
`token_ids` and omit optional fields whose value is `None`.
Segment input `fields` contains Python objects. Before writing, the store creates a
separate `SegmentRecord` with each field pickled using `HIGHEST_PROTOCOL`; the input
record is unchanged. Reads decode the selected fields, preserving explicit `None`
and raw bytes. `SegmentRecord.with_fields` merges encoded fields on stored copies.
Generated-span boundaries and message content remain in the Nexus trajectory.

`Serializable` has no data fields. Its default `serialize()` and static `deserialize()`
methods use `pickle.dumps(..., protocol=pickle.HIGHEST_PROTOCOL)` and `pickle.loads`.
Writes call the payload instance's `serialize()`; reads call the explicitly
selected class's `deserialize()`. Classes may override both methods to use their
own byte format. Neither production Record currently overrides the pickle defaults.
Generic KV callers supply a concrete `Serializable` subclass whose module is importable
by the reader. Tensor metadata retains its existing codec.

Record IDs are client-generated `uuid4().hex` strings. Every publication creates
new records, including repeated items in one batch and retries. Payloads keep their
native `trajectory_id` and `node_id` / `segment_id`, which are not used for deduplication.
Namespace is a label on the Store configuration and Ref, not stored in the payload.
The store and configuration entrypoint reject empty namespaces.
Reads, appends and releases use the record ID in the configured Redis keyspace;
differences between Store and Ref namespace labels do not affect these operations.
`NodeRef` and `SegmentRef` contain
only `namespace` and `record_id`. `HybridNodeRef` extends `NodeRef` with optional
`routed_experts_url` and `multi_modal_items_url`. All three are frozen dataclasses supporting hashing
and pickle. Node and segment refs are distinct types. Redis connection
information belongs to the store configuration, not to individual refs.

Replaying an initial segment snapshot after appending fields creates an independent
record. Append retries on the same SegmentRef still compare field pickle bytes;
numerically equal independent Tensor objects are not guaranteed to pickle identically.

Callers supply correctly typed records and refs; the store does not revalidate
business identity types or require nonempty Ref IDs.
Refs can address hash IDs when using the new record format.
## Batch behavior

- `put_many` prepares the batch, then publishes it once. Payloads are
  encoded before publication, and results preserve input order.
- `get_many` fetches the Redis records. Inline nodes are fully unpickled before selecting
  fields; segments unpickle only requested field values. The
  returned flat mapping keeps Nexus field names and excludes identity metadata.
  The underlying client returns payloads directly, without an `AssembledNode` wrapper.
  `None` selects every stored field; an empty collection selects none. Missing
  fields are omitted from each result, while missing records still raise `KeyError`.
- `put_fields_many` accepts segment refs only. An existing field with identical
  bytes is a retry; different bytes conflict. Each record's check and append is
  atomic. The trajectory backend uses batched reads and pipelined single-key Lua
  compare-and-set, with at most eight total rounds (the initial attempt plus up
  to seven retries). Only mismatched records are reread and merged again.
- `release_many` deduplicates refs by record ID. Missing records succeed.
  Operational failures propagate and deletion can be retried.

Empty operations perform no I/O after store construction. Redis uses MGET (Cluster: `mget_nonatomic`) and
pipelines for writes and deletes (Cluster writes are nontransactional).

There is no cross-record transaction. Field-encoding and payload-serialization errors are
rejected before publishing, but a runtime failure can leave a partially completed
batch. Publication retries create new UUIDs and may leave stored copies from the
failed attempt; no automatic rollback is performed. Segment append retries remain
idempotent on the same refs. Refs do not
provide reference counting: release only after consumers finish reading.

## Install and configure

```bash
uv sync --extra redis
```

For published wheels, install `nexus-agent[redis]`. The extra provides
`redis>=8.1,<9` and `pyzmq>=25`; `redis-fs` remains a compatible extra with
the same dependencies. Importing Gateway or using memory/ray storage does not load
redis-py or ZMQ. Tensor APIs import torch only when called; install the CPU or GPU torch
build appropriate to the caller. The multi-host launcher separately requires Ray.

```yaml
trajectory_store:
  backend: redis
  namespace: training-run-1
  redis:
    url: redis://127.0.0.1:16379/0
    key_prefix: nexus:trajectory:training-run-1
    controller_endpoint:
      host: 127.0.0.1
      port: 17000
```

The default key prefix is `nexus:trajectory`. An explicit prefix overrides it;
namespace remains a label and is not automatically added to keys. All workers
reading the same refs must use the same Redis connection, Controller and prefix.
`launch_servers()` returns connection configuration without starting Redis.

The synchronous KV and Tensor client is `RedisClient`; `AsyncRedisClient` retains
its `async_put_node`, `async_get_node`, `async_put_tensor`, and `async_get_tensor`
facade. Both live in `nexus.gateway.trajectory_store_backends.redis_store.client`.
Payloads subclass `Serializable` from the sibling `types` module, which also
provides `NodeId` (a string alias), `CompareAndSetResult`, and `RedisClientError`.
KV and Tensor writes return string IDs; their reads accept IDs directly. Payload-validation and codec errors before publication keep their original exception types.
Publication I/O failures report partial results as described below.
`overwrite=False` uses NX and reports collisions in `RedisClientError(code='publication_failed')`. A failed batch
returns no ID list, and successful writes before the failure are not rolled back.
Existing read IDs must be strings, but need not be nonempty or UUID-shaped.
`put_node(..., node_id=...)` and `put_multi_node(..., node_ids=...)` accept preallocated
storage IDs, independent of the business `NodeRecord.node_id`. Supplied write IDs must
be nonempty strings, match the batch length, and be unique within the batch.

## FS storage

The `redis` extra includes Redis and ZMQ dependencies; `redis-fs` is also supported.
Ray is installed separately for actor deployment or default head discovery;
explicit connections can be used without it. Neither extra
adds torch. FS uses TCP ROUTER/DEALER sockets and stores opaque blobs
in DataUnit DRAM with node-local SSD offload enabled by default. Redis still owns record
publication, discovery and heartbeat metadata.

`nexus redis up` starts Redis Cluster and FS, placing one DataUnit per alive Ray
node, including the head. The Redis host list puts the head first only if that
list already contains it. When Redis is already running,
`nexus redis fs up` starts only FS. `--num-data-units`
overrides this count and distributes units round-robin across the alive nodes.
It discovers the connected Ray head and starts the Controller there, at port
`17000` by default, waits for readiness, and returns while Ray keeps the services
alive. Its Redis URL defaults to `redis://<ray-head>:16379/0`; `--redis-url` takes
precedence. `nexus redis fs down` stops FS (the Controller and all DataUnits);
`nexus redis down` stops FS first, then Redis. Repeating `up` with the same options
preserves existing blobs. If an older version placed FS on another node, stop it
with that version before upgrading and restarting on the Ray head; the new
`down` locates services on the head.

DataUnits default to a `200G` high waterline. An unspecified safe line is half the
high waterline, so the default is `100G`. Adjust these per-DataUnit budgets to suit
the nodes, for example with a 1 GiB high waterline and 512 MiB safe line:

```bash
nexus redis fs up --ssd-offload-dir /mnt/local-ssd/nexus \
  --memory-waterline-bytes 1G --memory-safe-line-bytes 512M
```

The two command-line capacity options accept integer bytes or an integer with a
single `K/M/G/T` suffix, ignoring case and using powers of 1024. For example,
`200G` equals `214748364800` bytes. Negative values, fractions and other suffixes
are unsupported. The stress script uses the same parsing rule; Python APIs and
`serve --config` JSON continue to use integer bytes.

The same `ssd_offload_dir`, `memory_waterline_bytes` and `memory_safe_line_bytes`
options work with `LocalCluster`, `RayCluster`, `serve --config` and standalone
DataUnit processes. All three defaults belong to the DataUnit: CLI arguments and
JSON fields that are omitted stay absent through the launcher and deployment layers;
explicit values are passed unchanged for DataUnit validation and normalization.
The high waterline is an integer byte count and does not accept
`None` to disable offload. An unspecified safe line is normalized once by the
DataUnit to `H // 2`; an explicit value must satisfy `0 <= S <= H`, with `H > 0`.
The default directory is `.hykvstore`, relative to each DataUnit worker's startup
working directory. Each DataUnit creates a unique subdirectory on its own node;
select a real SSD mount on every participating node when using this feature.
Normal shutdown cleans its own files; forced termination can leave orphan directories.
This budget covers resident blob bytes, rather than process RSS, queued payloads,
network copies or filesystem page cache. Multiple DataUnits on one node each have
their own budget. SSD errors propagate to callers without bypassing the waterline.
Resident blob bytes can equal the high waterline when the safe line equals it.

Values larger than the safe line `S` go directly to SSD; only values with size
`n <= S` enter memory. When resident bytes `R` plus the incoming value reach the
high waterline `H`, a DataUnit selects enough entries from the LRU prefix to reach
`R <= S - n`, then admits the value, leaving at most `S` resident bytes after this admission.
With `S = H`, an in-memory admission at `R + n = H` needs no eviction; it evicts
only when `R + n > H`.
SSD reads cache only values with `n <= S`, retaining their SSD files; values larger
than `S` remain on SSD without a memory cache. Cache hits update recency, and evicting
these cached copies only releases memory. Cache admission errors fail the GET and
preserve the source file.
DataUnits default to eight ordinary request workers (`request_workers=8`, or
`--request-workers 8` for `fs up` and `fs data-unit`). A separate single admission
worker serializes memory growth and eviction; no additional admission lock is
needed. Memory PUTs hand off admission work and return a Future. A cold GET instead
reads SSD in its ordinary request worker, releases the URL and metadata locks,
then submits any cache refill to the admission worker and waits for it to finish.
The ordinary worker remains occupied until the GET completes, so `request_workers`
also limits concurrent cold GETs without a separate cold-read queue or counter.
When all ordinary workers are reading or waiting for refill, hot GETs and DELETEs
also queue; stats and heartbeat use independent channels. Waiting for refill never
holds the URL or metadata lock. This limit counts requests, not bytes or total RSS.
Two independent file-I/O pools default to `ssd_workers=8` and `copy_workers=8`
(`--ssd-workers` and `--copy-workers`). The SSD pool prepares replicas and LRU
victims; the copy pool transfers individual files between SSD and the shared
checkpoint directory. Each pool bounds submitted file tasks across the whole DU;
Queued preparation and copy work retains metadata rather than eagerly reading blobs;
ordinary PUT queues can still retain already received payloads. Increasing these
settings can increase I/O concurrency, temporary files, and retained buffers;
eight is an adjustable starting point, not a measured optimum.
Each admission selects at most 1,000 LRU candidates under the metadata lock, then
submits them outside the lock. SSD execution remains bounded by `ssd_workers`; a
finished task frees its slot for the next candidate in the same batch. Admission
waits for the entire batch before checking the current memory budget and selecting
another batch. A slow final file can leave other workers idle at a batch boundary.
Task submission, file I/O and waiting never hold the metadata lock.
An SSD worker holds the candidate's URL lock while checking entry identity,
establishing its SSD replica and releasing its resident memory. GET recency changes
and memory freed by concurrent DELETEs do not cancel the selected candidates; admission
can evict more memory than ultimately required. A DELETE that wins the URL lock
first makes that candidate a no-op. Admission drains all submitted candidates before
admitting the new value or propagating a selection, submission or file-task failure.
After a failure, it does not select another group. Successful candidates
still release memory when another candidate fails. An earlier
victim can release memory while a later SSD write is still running.
One metadata lock protects the index, LRU and counters; per-URL locks serialize
operations on the same blob. File workers acquire the URL lock before the metadata
lock. A memory GET takes a strong reference to immutable bytes under the same
URL lock, so an eviction after that lock is released cannot invalidate its reply.
A GET that follows eviction reads the completed SSD replica instead.
Disk I/O never holds the metadata lock. PUT and GET return after their
required migrations complete; different URLs can overlap SSD operations, while
thread-pool, disk and network saturation can still queue ordinary requests.
An RPC timeout can leave an unknown write outcome and does not trigger automatic replay.

DataUnit `stats` report logical `objects` and `used_bytes`, plus `resident_bytes`,
`ssd_bytes`, `resident_objects`, `ssd_objects`, `cached_bytes` and `cached_objects`.
Cached values have both copies, so `used_bytes = resident_bytes + ssd_bytes - cached_bytes`;
the object counts follow the same formula.
Stats reads and counter updates share the short metadata lock; reading copies the
eight counters and never waits for a lock held across blob I/O. The independent
heartbeat thread sends these counters. Controller `status` returns the latest
complete heartbeat report alongside each unit's
`data_unit_id` and `alive`, without querying the
DataUnit. Heartbeats use a separate controller handler from status and discovery;
they still depend on the Redis connection and network completing. Reports expire
with the heartbeat TTL (15 seconds by default), after which `alive` is false and
`stats` is `null`. The metadata counters describe committed state and do not
indicate whether a disk operation is in progress. Logically deleted blobs are
excluded even when an export pin delays physical SSD reclamation.

```yaml
trajectory_store:
  backend: redis
  namespace: training-run-1
  redis:
    url: redis://127.0.0.1:16379/0
    key_prefix: nexus:trajectory:training-run-1
    controller_endpoint:
      host: 127.0.0.1
      port: 17000
    fs_timeout: 120
    io_workers: 32
```

Use a reachable advertised Controller endpoint from the deployment output. The same
options work on `RedisClient`, `AsyncRedisClient` and `RedisTrajectoryRecordStore`.
`controller_endpoint` accepts an `Endpoint` or a mapping. Its mapping fields are
filled independently: omitted `host` discovers the connected Ray head, and omitted
`port` uses `17000`. Strings are not accepted. Python callers can instead inject
`fs_client`; specifying both is an error. Omitting both creates a client for the
default Controller. An explicit Controller host/Endpoint or borrowed `fs_client`
skips FS address discovery. An injected `redis_client` also skips Redis URL
discovery. Supplying either an explicit Redis URL or `redis_client`, together with
a Controller host/Endpoint or `fs_client`, requires no Ray installation or
connection. Omitting the Redis URL without injecting `redis_client` still requires
head discovery for that URL.
Internally created FS clients close with the Redis client; injected FS clients remain
owned by their caller. `fs_timeout` defaults to 120 seconds and accepts `None` to wait
indefinitely. `io_workers` defaults to 32, with a minimum of 2; `fs_io_threads` defaults
to 32 for the FS ZMQ context.
FS close rejects new operations and drains admitted work before releasing connections.

Only non-`None` `routed_experts` and `multi_modal_items` values are offloaded; empty
bytes and empty containers are still stored as blobs. The other fields remain inline.
The Redis envelope stores the payload's own encoded bytes and the blob URLs.
Reads continue to invoke the explicitly selected payload decoder. Ordinary payloads
without external fields keep their original Redis bytes. Store projections avoid
reading unselected blobs. `HybridNodeRef` carries both URLs so Redis and requested
blob reads can begin concurrently; string-ID client reads discover URLs from Redis.
Existing `NodeRef` and `SegmentRef` fields and module paths remain unchanged.
Refs can be pickled or reconstructed from `dataclasses.asdict()` output; there is no
new `dump()`/`load()` API on trajectory refs.

FS publications with external fields or preallocated IDs are immutable: allocate a new
storage ID for replacements, and do not pass `overwrite=True`. Publication reserves
each Redis key with NX, uploads its blobs, then publishes the committed envelope with
the existing per-key CAS primitive. A reservation conflict never uploads that record's
blobs. Successful siblings stay published when another item fails.

Publication I/O failures raise `RedisClientError(code='publication_failed')`.
The Store error body contains `partial_refs`, aligned with the input with `None` at
failed positions, and `errors`, keyed by zero-based input index. The string-ID client
provides the analogous `partial_ids`. IDs, URLs and errors travel directly from the
publication operation to the Store; refs are not reconstructed by rereading Redis.
Validation and encoding errors before I/O retain their original exception types.
Unknown write outcomes are not automatically replayed or rolled back. Failed or
uncertain operations may leave writing envelopes and blobs for diagnosis; a writing
record cannot be read or deleted through the normal record API.
The publication layer disables redis-py retries on its Redis connection, including an injected
Redis client. Its Cluster mutation pipelines also disable the driver's redirection
replay, which otherwise retries connection failures even with zero retries. Cluster
topology errors are reported to the caller. This uses the pinned redis-py 8.x routing
queue.

Release validates URL ownership against every current record before destructive calls.
Its per-record Lua compares the bytes actually read and checks the dump-protection
set. Without an active dump owner it `UNLINK`s the record; with an owner it atomically
`RENAME`s the original bytes to a deferred key in the same Cluster slot. Both outcomes
complete logical deletion immediately: ordinary GET/EXISTS no longer see the record,
and a new dump cannot register it. Protected exporters may still read the deferred
metadata until they have captured their metadata and established DU pins.
Missing, changed and uncertain deletion results are not automatically replayed.
Only a confirmed immediate deletion permits that caller to delete the corresponding
DU blobs. A confirmed deferred deletion transfers cleanup to the last dump owner;
release does not wait for its preparation or Ceph copying.
The last owner cleans the canonical URLs of all fields, including unselected fields,
then `UNLINK`s deferred metadata. Failed cleanup retains that metadata and reports an
error; an explicit repeated release can retry it. Unknown Redis or DU outcomes retain
the metadata and blobs needed for diagnosis. There is no background garbage collector.
Runtime record IDs and blob UUIDs are not reused.

FS topology is immutable for one run: `hykvfs://` URLs use stable sharding across
256 slots. Heartbeat expiry does not remap slots. DataUnits have no persistence,
replicas, automatic recovery or migration. SSD offload expands capacity for the
current run; its index remains in memory, and restarting a DataUnit does not recover
its blobs. Old SSD directories are never automatically loaded or garbage-collected.
Use a new key prefix and matching reader/writer Nexus versions for each deployment.
Old HyKVStore data is not converted. Clients receive connection configuration only;
start services explicitly via [the deployment CLI](redis-deployment.md).
For local development, `LocalCluster(redis_url)` starts TCP services in threads and
`RayCluster(redis_url)` schedules CPU actors in an already initialized Ray cluster.
Both expose `client()`, `fs_client()` and serializable `ClusterInfo` connection data.
See [the runnable example](../examples/redis_fs.py) and
[configuration example](../examples/configs/redis_fs.yaml).

## Parquet payload checkpoints

Redis stores implement the existing two-stage export interface:

```python
dump_refs, writing = store.dump_payloads(refs, dump_dir='/mnt/ceph/checkpoint/trajectory_store')
await writing
```

The synchronous call only captures reference metadata and allocates persistent
references; it does not query Redis, create files or launch tasks. Awaiting
`writing` registers deletion protection for the selected records, then snapshots
metadata in batches of 16. Each registration uses Lua to check that the record
exists and add the dump ID to its protection set atomically; the commands are
pipelined. Multiple dumps can protect the same record without waiting for each
other. The auxiliary set shares the record's Redis Cluster hash slot and follows
the actual Redis key, not the caller's Store namespace.
Each batch removes its own protection as soon as metadata is captured and all
selected DU blobs have acknowledged stable SSD replicas and pins, before Arrow
conversion and local Parquet writing. Records in later batches stay protected
until their own preparation completes. A concurrent release returns after moving
protected metadata to its deferred key; the last dump owner later reclaims the DU
objects. PUT, publication commit and Segment CAS neither
register protection nor wait for it, so ordinary writes add no gate commands.
Segment updates replace complete metadata atomically; export captures the version
actually read, without promising a simultaneous snapshot of the whole trajectory.

Node numeric fields, including `topk_indices` and `topk_values`, are typed Parquet
columns: `token_ids` and `topk_indices` use int32, while `logprobs` and
`topk_values` use float32, following the Gateway producer contract. Each column
stores only `shape` and flattened `values`; readers restore NumPy arrays. Export
rejects non-ndarray inputs or mismatched dtypes rather than casting them. Optional
null fields and empty arrays retain their meaning. Checkpoints from the earlier
experimental schema must be exported again; no legacy reader is provided.
Segment fields keep their existing per-field encoding. The layout is:

```text
trajectory_store/
    trajectory.parquet
    data/
        <uuid>.blob
```

Existing hybrid blobs stay on their owning DUs: each DU prepares an SSD replica
and pins it, then copies it directly to the shared target path. The coordinator
never fetches those large blobs. A selected non-null `routed_experts` or
`multi_modal_items` field must have a canonical DU URL; exporting an older Redis
record with an inline large field raises `ValueError`. Rewrite such records through
the hybrid store before exporting them. Unselected inline fields do not prevent
exporting other fields. `ensure_ssd_copy(url)` ensures the SSD replica without
evicting memory or changing LRU recency; existing SSD replicas make it idempotent.
`import_file_to_ssd(url, source_path)` imports a new object directly from a shared filesystem
path into SSD. Callers allocate the target URL in the same way as ordinary PUT:

```python
url = fs.allocate_url()
fs.import_file_to_ssd(url, '/mnt/ceph/checkpoint/trajectory_store/data/source.blob')
```

The DU reads the source itself; the RPC carries no blob bytes. The method returns
only after the file and disk-only metadata are published. It neither admits the
object to resident memory nor adds it to LRU. An identical target URL and source
reuses the original import; changing the source for that target is a conflict.

Metadata is staged as local temporary Parquet. All requested records are registered
before metadata batching begins, without loading payloads during registration.
Deletion that wins before registration makes export fail with `not_found`. Deletion
after registration moves metadata to a deferred key and returns immediately. Only
already-protected dump reads fall back to that key when the original is missing;
normal reads do not. Updates can proceed throughout.
After each batch captures metadata and establishes its selected blob pins, a release
Lua removes only this dump ID. Other dump owners remain intact. The Lua also checks
whether this was the last owner and deferred metadata needs cleanup. An acknowledged
owner removal leaves the lease's pending set before physical cleanup; an unconfirmed
removal remains pending for idempotent retry on context exit. A cleanup failure after
owner removal leaves the deferred key for an explicit release retry.

Ceph copying still begins only after all batches are prepared and the local
Parquet is closed. The DUs copy in a dedicated executor while the coordinator
copies staged metadata. Only the exported metadata replaces runtime URLs with
`ckpt://data/<uuid>.blob` references; the part after `ckpt://` is relative to the
checkpoint directory. Live Redis values and borrowed refs are unchanged.
The final `trajectory.parquet` is published last, after all blob copies succeed.
Input order, duplicate refs and field projection are preserved; unselected blobs
are not copied.
When exporting Nodes, an explicit field selection must include `token_ids`;
omitting it raises `ValueError` synchronously before any I/O. Empty exports and
Segment-only exports are exempt. A zero-length token array is valid.

DELETE immediately removes a pinned blob logically, and its logical stats fall
immediately. Its SSD file survives until the last export pin is released. Waiting
for Ceph therefore occupies neither a DELETE worker nor the short metadata lock.
Failed copy jobs retain pins for explicit retry or termination. `FSClient` exposes
`start_dump_copy(dump_id, dump_dir)`, `get_dump_status(dump_id)` and `finalize_dump(dump_id)`;
finish rejects a still-running copy. A failed or cancelled prepare first exits the
protection context, then drains actual remote preparation and aborts its pins.
The last Redis owner may now reclaim a deferred record in that context exit.
DU per-URL locks serialize delayed preparation with DELETE: DELETE first makes
preparation fail, while preparation first establishes a pin that protects the SSD
source after logical deletion. Failed preparation never publishes final Parquet.
Owner removal and deferred cleanup are separate acknowledgements: losing the former
retains the pending owner, while failing the latter retains deferred metadata without
reacquiring or repeatedly releasing the owner. No polling or Pub/Sub connection is
used for deletion protection.
Export failures include `dump_id` in their error details where remote work can
remain. Cancellation waits for actual I/O to finish before releasing pins. If
remote status cannot be confirmed, pins remain held and an error is reported; cancelling a future alone does not authorize
deleting its source.

Checkpoint code is grouped under `redis_store/checkpoint/`: `format.py` defines
schema and array/reference encoding, `dump.py` coordinates exports, `load.py`
restores records, and `reader.py` supports offline inspection. The package keeps
the existing `iter_payloads` / `read_dump` imports and the module CLI below;
ordinary release does not import this package.

The offline reader needs PyArrow (`redis` and `redis-fs` extras include it), but
does not connect to Redis or any DU:

```python
from nexus.gateway.trajectory_store_backends.redis_store.checkpoint import iter_payloads, read_dump

for row in iter_payloads('/mnt/ceph/checkpoint/trajectory_store', fields=['token_ids', 'topk_values']):
    print(row['trajectory_id'], row['fields'])

# Restore requested ordering and duplicates from persistent references.
rows = read_dump('/mnt/ceph/checkpoint/trajectory_store', dump_refs)
```

`iter_payloads(..., load_blobs=False)` returns the complete `ckpt://` reference
after checking its prefix and relative-path syntax, without accessing the blob
file. Selected references must use this prefix; checkpoints with bare
`data/<uuid>.blob` references must be exported again. Unselected blob fields are
skipped. The directory can be moved as a whole without rewriting its references.

For compact inspection, leave blob values unloaded and request numeric summaries:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" uv run --no-sync python -m \
  nexus.gateway.trajectory_store_backends.redis_store.checkpoint \
  /mnt/ceph/checkpoint/trajectory_store --fields token_ids topk_values
```

For online restore, use the persistent references saved with the trajectory:

```python
restored_refs = await target_store.load_payloads(
    dump_refs, dump_dir='/mnt/ceph/checkpoint/trajectory_store'
)
```

Redis checkpoint dump/load always uses the hybrid store. Target DUs pull each blob
directly into SSD, then publish Redis metadata with newly allocated runtime URLs.
Numeric data is read in batches of 16. Segment fields retain their existing
encoding; Segments and Nodes without blobs still use plain Redis record values.
The file layout and column types remain unchanged. Restore resolves each selected
`ckpt://` reference to an actual shared file path before passing it to the DU;
the DU RPC does not receive the checkpoint URI. No old Redis, DU, or slot mapping
is needed.

Restore preserves business trajectory/node/segment identities and uses the target
Store namespace, fresh record IDs, and fresh DU URLs. Duplicate persistent entries
share the newly restored object while returning independent ref wrappers in input
order. Separate load calls create independent objects. Preflight checks the refs,
format, required Node tokens and selected source files before target writes.
`exclude_node_id` is a Ray scheduling hint: Redis accepts and ignores it, logging
one warning per call with a non-`None` value for compatibility with shared callers.

Ordinary runtime failures retain successful records and raise
`RedisClientError(code='load_failed')`, with input-aligned `partial_refs`, errors,
and unknown outcomes in the error body. Cancellation stops subsequent batches and
new commits, drains admitted I/O, and conditionally removes confirmed writes still
owned by this load. Changed
or uncertain records and their blobs are retained and reported. Redis writes with
unknown outcomes are not automatically replayed. A cancellation delivered after
the worker has already completed returns the completed outcome. Dump/load workers
hold a publication operation through preflight, I/O, cleanup and protection exit.
`Store.close()` delegates to the client/publication lifecycle, which waits for
these operations before disconnecting; the Store keeps no second load counter or
closing state.

DU import bookkeeping uses target URLs internally; callers supply no extra task
ID. Termination fences delayed requests, and cleanup waits for actual file I/O,
not merely a cancelled Future or a marker queued behind copies. Checkpoint source
files are never removed by restore cleanup. This implementation uses shared
filesystem paths rather than a Ceph object-store SDK, and does not persist import
jobs across service restarts.

Dump-protection sets deliberately have no expiry. A transport failure during
registration or owner removal can leave a dump ID registered; unconfirmed attempts
remain tracked for idempotent removal. Protection and deferred keys use distinct
prefixes and the record's Cluster hash slot, including prefixes with hash-tag braces.
Ordinary publication failures do not leave writer registrations.
Load cancellation uses the same deferred-deletion decision alongside its original
byte-digest ownership test. A deferred outcome hands the record and target URLs to
the last dump owner, so load does not independently keep or abort those imports.
The deferred bytes retain the original `StoredNode` state. A committed envelope can
clean its DU URLs directly; a writing envelope first fences and drains imports via
`finalize_ssd_imports` with no kept URLs, then deletes them. This prevents a late import
from recreating an aborted object. The checkpoint source files are never removed.
After all required DU cleanup acknowledgements, the deferred key is unlinked. A
failed cleanup retains it for an explicit retry of release after owners have left.
Deletion protection uses no TTL unlock, polling loop, notification subscription or recovery worker.
DU prepare/status checks still use their existing 20 ms status wait; this change does not alter those checks.
Unconfirmed DU pin cleanup may require explicit `fs.finalize_dump(dump_id)` after the
actual tasks have ended. Before manually removing a stranded owner, verify that its
dump has ended; remove only that owner, then run the deferred-cleanup path rather than
blindly deleting the whole protection set or deferred metadata.
All dump/release participants must use this protocol version. Drain old dump tasks
before upgrading: an old client that only executes `SREM` cannot reclaim deferred
metadata. Abandoned protection and deferred state are not automatically recovered
after process or DU restart.

## Payload codecs

KV reads require the expected payload class. Single and asynchronous reads use
the keyword-only `payload_type`; batch reads accept `(node_id, payload_type)`
pairs, including mixed types and repeated IDs:

```python
from nexus.gateway.trajectory_store_backends.redis import NodeRecord, SegmentRecord

node = client.get_node(node_id, payload_type=NodeRecord)
records = client.get_multi_node([
    (node_id, NodeRecord),
    (segment_id, SegmentRecord),
    (node_id, NodeRecord),
])
node = await async_client.async_get_node(node_id, payload_type=NodeRecord)
```

Here the IDs are strings returned by `put_node` or `put_multi_node`.
Trajectory callers continue to use `store.get_many(refs)`;
the trajectory backend selects `NodeRecord` or `SegmentRecord` from the ref type.
Its field-update preflight and CAS retries also use `SegmentRecord.deserialize`.

A custom payload subclasses `Serializable` and implements `serialize(self) -> bytes`
and a matching static `deserialize(blob: bytes)` returning its own class. There
is no codec registry or type metadata on refs. Only offloaded records use an FS
envelope; its record bytes still use the selected payload codec.
The client validates the entire request batch before I/O, then fetches once with
MGET or Cluster `mget_nonatomic` and decodes in input order. Results must be
instances of the selected classes; otherwise `RedisClientError` includes the
record ID and expected/actual types. Missing records retain `not_found`; decoder
and network exceptions propagate without fallback. Empty batches perform no I/O.

The former lower-level reference arguments and omitted payload types are unsupported.
`Reference` and `TensorReference` imports and their direct pickles have no compatibility
aliases. Trajectory `NodeRef` and `SegmentRef` retain their names, fields and module
paths, including support for existing pickles, but no longer expose `dump()` / `load()`.
Payload encoding and Redis key layouts are unchanged; no data conversion is needed.
Future codec changes require coordinated readers and writers and a data cutover. CAS always compares the original stored bytes,
even when a custom codec produces different valid bytes on each serialization.

## Layering and CAS

`trajectory_store_backends/redis.py` owns the trajectory records and refs, field
encoding, conflict rules, the eight-round contention limit, and cancellation.
The store inherits only `TrajectoryRecordStore` and owns one `RedisClient` in
`self._client`, reusing its connection pool for all operations and CAS rounds.
Its `close()` and context manager close that client, including an injected Redis
connection. KV/Tensor methods are available on the client, not on the store.
It depends on `trajectory_store_backends/redis_store/`, whose `client.py`,
`types.py`, and `local.py` provide generic storage primitives and local startup.
The lower package exports only `RedisClient` and `AsyncRedisClient` for use within
Nexus. The client provides generic storage primitives; its publication component
recognizes NodeRecord to offload large fields to DataUnits. Normal Nexus package initialization may load trajectory modules.

`RedisClient.get_node_blobs(node_ids)` returns raw bytes or `None`, preserving order
and supporting Cluster keys across slots. `compare_and_set_many(updates)` accepts
`(node_id, expected_bytes, replacement_bytes)` tuples and returns
`CompareAndSetResult` values from `redis_store.types`: `UPDATED=1`, `MISMATCH=0`,
or `MISSING=-1`. It uses a nontransactional pipeline of atomic single-key Lua
operations. These primitives do not decode payloads, merge fields, or retry.
Empty batches perform no I/O, non-string IDs fail before execution, and network
errors propagate. There is no cross-record transaction or rollback.

The trajectory backend first encodes and validates the entire update batch,
including known field conflicts. Each attempt merges against freshly read
records and compares the **original stored bytes**, not a reserialized expected
record. Byte-identical field updates skip writes. Missing records, including
deletion during CAS, raise `KeyError`; contention exhaustion retains the
`RedisClientError` code `contention`. Partial success is retained and can be
retried idempotently. The entire synchronous process stays in one worker task;
cancellation waits for that process to finish.

The earlier move from the `redis/` package to `redis.py` changed pickle paths.
That migration requires readers and writers to upgrade together after current runs
finish, then use a new run namespace and key prefix.
The former `redis.record_store`, `redis.client`, `redis.types`, and `redis.local`
internal imports and their pickles have no compatibility aliases. Redis data
is not cleared or converted.

## Migration from HyKVStore

The Redis implementation was migrated from HyKVStore revision
`46b7d5aacae6818300a30fe1cfa9c64c5e156e34`. The FS services and hybrid publication
increment from `hybrid_client_dev@5853bb859e84842bee9271fbe030a1fff1051e92` now use
the existing Nexus Redis client, types, record store, local Redis launcher and fixtures.
The old `data_control_seperate_mode`, multi-backend factory, package root and unchanged
HyKVStore base modules are not copied. The Redis Cluster launcher is existing Nexus
code moved into the package, retaining the old deployment script as a thin entrypoint.

This is a coordinated cutover, without compatibility aliases or old pickle decoding:

1. Complete old runs before upgrading readers and writers together.
2. Replace the `hykv` extra and backend with `redis`; move `trajectory_store.hykv`
   options to `trajectory_store.redis`.
3. Replace `HyKVTrajectoryRecordStore`/`HyKVRecordStore` with
   `RedisTrajectoryRecordStore`, and `RedisHyKVClient`/`HyKVClientError` with
   `RedisClient`/`RedisClientError`. Use the Nexus imports above.
4. Remove HyKVStore paths from driver and worker `PYTHONPATH`; ensure every worker
   receives the same Nexus wheel or checkout. External training repositories must
   update their dependency installation and launch configuration together.
5. Start a new run with a new namespace and key prefix. Existing Redis data is
   not cleared or rewritten. Old refs and data must stay with the old application.
6. Pass launcher settings as command-line arguments, such as `--primary-count`,
   `--replicas-per-primary` and `--redis-dir`, and use endpoint output `redis_url`.
   Start FS before clients. Keep the original persistent directory when
   reusing an existing cluster: changing it can lose its `nodes.conf` identity.

## Validation

From the Nexus checkout, run `uv sync --extra dev --extra redis`, then install
Ray, NumPy and CPU torch into that environment.
`redis-server` and `redis-cli` must be available; alternatively set
`NEXUS_REDIS_SERVER` to the server binary with redis-cli beside it.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run --no-sync pytest -o addopts="" -v -s \
  tests/unit/gateway/redis_store \
  tests/integration/gateway/redis \
  tests/unit/gateway/test_trajectory_store.py \
  tests/unit/gateway/test_session_record_store.py \
  tests/unit/gateway/test_gateway_run_finalization.py \
  tests/unit/test_redis_store_installation.py
```

Fixtures start isolated Standalone and three-primary Cluster processes and stop
them afterwards. Missing optional dependencies are reported as skips; a complete
Redis validation run must include the CPU Tensor and live Cluster cases.

CLI and launcher validation:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" uv run --no-sync pytest -o addopts="" -v -s \
  tests/unit/cli/test_redis.py tests/unit/test_redis*launcher.py
```

The deferred-deletion validation on 2026-10-26 passed 401 Redis store unit cases
and 77 FS/deletion-protection integration cases with Redis 7.2.6. Standalone and
Cluster restore cases ran after the source DU stopped and the checkpoint directory
moved. Cross-process tests confirmed logical release completes while dump owners
remain, and that the last owner can reclaim metadata from another process.
Before-read and before-pin races retained canonical URLs, including unselected
blobs; load cancellation handed both writing and committed records to dump cleanup.
Two unit and six integration cases requiring Ray were skipped.
`UV_NO_SYNC=1 make ci-check` passed format, lint and production type checks.
Shared-file copies used local temporary directories; this run did not validate a
real Ceph deployment or production throughput.

Redis benchmarks (no GPU required; `uv sync --extra redis` supplies redis-py and ZMQ):

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" uv run --no-sync python -m tests.stress.redis.bench_put_get --ops 200 --clients 4
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" uv run --no-sync python -m tests.stress.redis.bench_put_get --ops 200 --tensor-elements 16384 --tensor-count 5
```

Omitting `--redis-url` starts and stops isolated local Redis and FS services.
An explicit URL borrows that Redis server; the script still starts and stops local
FS. Neither mode needs CHIEF_IP or Ray. The external Redis retains the benchmark's
UUID-prefixed data (or `--redis-prefix`) and FS topology metadata.
Multi-host deployment and destructive failover tests are documented in
[the Redis deployment guide](redis-deployment.md). GPU/Megatron
training regression belongs in the training environment, not these CPU tests.
