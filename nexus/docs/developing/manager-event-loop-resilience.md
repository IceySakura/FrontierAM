# Manager Event-Loop Resilience

## Scope

This change protects distributed Instance Managers from event-loop stalls while preserving the existing execution
model. `AsyncioExecutor` remains the default. `ThreadExecutor` continues to support `pooled` and `per_execution`.

The implementation has three mechanisms:

1. **Health** controls admission and decides when degraded execution can recover.
2. **Reconciliation** repairs disagreements between the database, the Manager registry, and the Executor.
3. **Fencing** stops an unsafe Manager and guarantees bounded process termination.

Batch offloading and cross-loop deadlines feed these mechanisms; they are not separate recovery protocols.

## Timing

| Boundary | Default | Result |
| --- | ---: | --- |
| Manager heartbeat or main-loop stall | 30 seconds | Self-fence |
| Executor or Batch blocking-operation hang | 3 minutes | Recover, quarantine, or self-fence |
| Cleanup after self-fence | 5 seconds | `os._exit(70)` |

Supporting intervals remain five seconds for heartbeat and Executor probes, with a ten-second Executor probe
deadline. The control plane considers a Manager unhealthy after 15 seconds and dead after 45 seconds. These TTLs
govern scheduling and recovery; they are not additional Manager watchdogs.

## Health and Capacity

The Manager never awaits `Executor.capacity_snapshot()` from registration, heartbeat, admission, or polling. It
maintains only running assignment slot costs and the latest `ExecutorHealthSnapshot`.

The Executor owns worker health and reports:

- whether the Executor loop is responsive;
- effective capacity after suspect or quarantined workers are excluded;
- assignments lost with a worker;
- whether the process should recycle after degraded work drains.

The Manager derives admission locally:

```python
running_slot_cost = sum(assignment.slot_cost for assignment in running_assignments)
available_slots = max(0, health.effective_capacity - running_slot_cost)
```

An unknown start result or an unresponsive shared Executor sets effective capacity to zero. Admission reopens only
after a successful probe and assignment reconciliation.

### AsyncioExecutor

All assignments share one child-thread event loop. A failed probe immediately closes admission. Recovery within
three minutes is allowed; three minutes of continuous failure self-fences the Manager.

### ThreadExecutor

Each worker owns its event loop. Worker health follows one lifecycle:

```text
healthy -> suspect -> quarantined
```

A suspect idle worker is not scheduled. A worker that recovers within three minutes returns to service. A worker
that remains unresponsive, or closes unexpectedly while still owning an assignment, is quarantined and its
assignment is reported lost.

Quarantined workers are never reused. The Manager self-fences immediately when effective capacity is strictly below
50 percent. Exactly 50 percent may continue draining healthy assignments. Any quarantine requests a process recycle
after healthy work drains because Python cannot safely terminate a stuck thread.

## Assignment Reconciliation

Reconciliation keeps two explicit directions because they repair different ownership failures:

- **Database to local:** the database no longer owns an assignment that the Manager still runs. Stop the Executor
  assignment, then remove the local reservation only after absence is confirmed.
- **Executor to database:** the Manager registry contains an active assignment that disappeared from the Executor.
  Mark it lost under Manager/session fencing, remove the local reservation, and wake scheduling for retry.

Start and stop bridge timeouts use the same presence check. If presence cannot be confirmed, the local reservation
is retained and admission remains closed. This prevents a late start from becoming an untracked ghost assignment.

## Cross-Loop Calls

All `asyncio.run_coroutine_threadsafe()` bridges use `await_threadsafe()` with a caller-owned deadline and
best-effort cancellation:

- ordinary start, command, and information calls: 5 seconds;
- reporter to Manager: 30 seconds;
- Executor health probe: 10 seconds;
- stop: execution stop timeout plus 1 second;
- SSE: long-poll timeout plus 2 seconds;
- Batch preparation: generation timeout plus 2 seconds.

A single bridge timeout closes or reconciles admission state; it does not by itself prove a three-minute hang.

## Batch Isolation

`BatchControlService` owns one semaphore-limited blocking runner with concurrency two. Every Manager-created
`BatchRepository` receives this runner once at construction; individual repository calls do not carry separate
runner policy.

The runner uses `asyncio.to_thread()` for bounded but potentially expensive local operations:

- task-list reads, JSON parsing, and hashes;
- submit-window validation, copying, and hashes;
- representative-item and module-selector parsing;
- workspace scans, fingerprints, and recursive cleanup.

Only plain values and paths cross the thread boundary. `AsyncSession`, ORM rows, repositories, and artifact
transports stay on the Manager loop. A timed-out caller does not release the concurrency slot until its underlying
thread completes. A thread still blocked after three minutes requests the same Manager self-fence path.

Compose materialization remains a bounded subprocess. Credentials pass through the subprocess environment, never
the request file. Timeout shutdown sends TERM, waits for a bounded grace period, then sends KILL.

## Fencing

Heartbeat failure, a 30-second Manager-loop stall, a continuous Executor hang, insufficient Thread capacity, and a
stuck Batch helper all converge on one self-fence outcome.

The CLI and `InstanceManagerService` share one `HardExitGuard`. In-loop fencing arms the guard, closes admission,
and attempts to stop local assignments. The OS-thread Manager-loop monitor can arm the same guard when the loop
itself cannot run cleanup. Successful shutdown disarms it; otherwise the guard calls `os._exit(70)` after five
seconds and lets the workload supervisor restart the process.

## Non-goals

This MVP does not add multi-loop sharding, per-assignment subprocesses, new database state, two-phase Batch CAS, or
new public recovery settings. Thread quarantine fences Nexus ownership but cannot undo external side effects already
performed by user code.

## Verification

Tests cover Manager-loop escalation, shared hard-exit ownership, bridge cancellation, unknown start reconciliation,
Asyncio recovery, Thread suspect/quarantine/closed-worker handling, the exact 50-percent threshold, Batch blocking
concurrency, caller cancellation, thread-boundary purity, and Compose subprocess termination.
