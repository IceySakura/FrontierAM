# Allocation Manager liveness

Dynamic Single/Live submissions and Batch execution Managers can request an immutable
liveness policy under `resources.managers.liveness`. Batch orchestration Managers and
predefined shared pools keep their platform policy. Omission preserves existing
submission hashes and behavior; an explicit `null` is rejected.

```json
{
  "resources": {
    "managers": {
      "count": 1,
      "cpu_cores": 2,
      "memory_gib": 4,
      "slots": 1,
      "liveness": {
        "self_fence_timeout_seconds": 90,
        "executor_hang_timeout_seconds": 180,
        "dead_timeout_seconds": 120
      }
    }
  }
}
```

The equivalent optional CLI flags, supplied together, are:

```sh
--manager-self-fence-timeout-seconds 90 \
--manager-executor-hang-timeout-seconds 180 \
--manager-dead-timeout-seconds 120
```

All three values must be integers in 1–600 seconds. Admission and final Manager startup
validate the platform heartbeat interval, healthy TTL and hard-exit grace:

- `heartbeat_interval < healthy_ttl < self_fence`
- `self_fence + hard_exit_timeout < dead`
- `self_fence <= executor_hang`

The 120-second deadline is measured from the last effective heartbeat, not added to
90 seconds. A Manager explicitly fenced or replaced can lose its assignments earlier.
Queue, assignment acknowledgement, execution start and task timeouts remain independent.
Self-fence is irreversible; only a pause ending before self-fence may recover. The
hard-exit grace is still platform controlled. These settings do not fix GIL starvation
or large trace serialization.

## Persistence and startup

The Allocation spec contains `manager_liveness` with `version: 1` and the three fields,
included in the Allocation hash. There is no schema migration. Dependency-build
snapshots retain this subtree. Allocation responses expose only
`configured_manager_liveness` and `manager_liveness_source` (`allocation`, `platform`,
or `invalid`); `platform` does not claim to describe a running Manager's actual config.
Startup logs report the actual effective values.

The existing `NEXUS_DIST_ALLOCATION_SPEC_JSON` is the single policy source. Renderer,
bootstrap and Manager CLI pass `--require-allocation-liveness-v1` as a capability
check. Missing/mismatched markers, identities, versions or invalid policies fail
before registration. The Manager applies the three settings after DB config scopes
and validates against its final platform config. An older bootstrap or Manager rejects
the unknown flag rather than silently registering with shorter deadlines.

Reconciler joins the Allocation by its primary key, calculates the per-row cutoff in
SQL before pagination, then revalidates after locks and uses the same cutoff for the
liveness fence CAS. Effective heartbeat compatibility with legacy Manager rows is
preserved. Missing/invalid referenced policies are excluded from heartbeat expiry;
explicit fences and missing/replaced Managers retain their existing handling. Invalid
policies are reported on startup and Allocation reads instead of using a shorter default.

## Rollout and validation

Upgrade all Reconcilers first, then bootstrap/Manager images and source-package wheels,
then expose submission overrides. Do not roll Reconcilers back while custom-policy
Allocations remain active. A source package carrying an older Manager wheel must be
rebuilt; the startup capability check intentionally rejects it.

Local coverage:

```sh
uv run pytest tests/unit/distributed/test_allocation_liveness.py -q
```

For DevCloud use a disposable test Pod, isolated MySQL instance and current source.
Set `NEXUS_LIVENESS_TEST_DATABASE_URL` through a test Secret. The unit test file also
runs against that database and drops its tables during fixture cleanup: never point it
at shared or production data.

```sh
python -m pytest tests/unit/distributed/test_allocation_liveness.py -q
python tests/integration/distributed/run_allocation_liveness_acceptance.py
```

The acceptance runner creates a separate `liveness_runtime_acceptance` database on
that disposable server, runs real bootstrap/Manager processes with 90/180/120 despite
30/180/45 stored DB defaults, submits three sleep tasks and injects process pauses:
55 seconds recovers and succeeds, 95 seconds self-fences, and 123 seconds expires.
It saves startup logs and `summary.json` under `/tmp/liveness-acceptance`. Delete the
test Pod and Secret after collecting evidence; the runner kills its Manager processes.
