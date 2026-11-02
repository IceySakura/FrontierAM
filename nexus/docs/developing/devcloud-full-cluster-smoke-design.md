# DevCloud Full Distributed Cluster Smoke Design

## Goal

Deploy every Nexus distributed role in the DevCloud test namespace
`nexus-online-service`, prove that each service loads database-backed
`DistributedConfig`, and exercise the scheduling and runtime-proxy paths.

This is a disposable functional cluster, not a production availability or
performance test.

## Chosen Approach

Use the repository Helm chart with one replica of every role and a dedicated
MySQL 8 instance:

- MySQL 8.0.45 with disposable `emptyDir` storage;
- DB Init Job;
- Control Gateway;
- Application Gateway;
- Scheduler;
- Reconciler;
- predefined Instance Manager;
- fake Runtime.

The MySQL image is built from the internal Tlinux 4 base image and its
AppStream `mysql-server` package, then pushed to
`mirrors.tencent.com/nexus_utopia`. The deployment never pulls a public image.

This approach is preferred over:

- putting every role in one Pod with SQLite, which does not exercise normal
  cross-Pod service and database boundaries;
- using the chart's default high-availability replica counts, which adds image
  pull time and resource use without improving this functional validation.

## Images and Secrets

Build uniquely tagged control and workload images from the current worktree.
The workload image is rebuilt because Instance Manager is itself a DB Config
consumer and must contain the new code.

Create one ephemeral Kubernetes Secret outside Helm release values. It
contains a generated hexadecimal MySQL root password and internal token.
Unused runtime and OTel keys are present with empty values only because the
current chart references those keys. No secret value is printed, committed,
or included in test evidence.

OpenTelemetry remains disabled for this test.

## Deployment Sequence

1. Build, locally smoke-test, and push the MySQL, control, and workload images.
2. Create the ephemeral Secret.
3. Install the Helm release with MySQL and DB Init enabled, while all
   long-running Nexus replicas are initially zero.
4. Wait for MySQL readiness and successful schema migration.
5. Run a Config Seed Job that writes:
   - global `*/*`: `manager_poll_interval=7.0`;
   - `control-gateway/*`: `gateway_route_cache_ttl=2.5`;
   - `application-gateway/*`: `gateway_route_cache_ttl=2.0`;
   - `scheduler/*`: `scheduler_poll_interval=0.5`;
   - `reconciler/*`: `reconciler_poll_interval=1.5`;
   - `instance-manager/<stateful-pod-name>`:
     `manager_poll_interval=0.5`.
6. Run a Config Verify Job that loads each scope with
   `initialize_distributed_config`, asserts the expected values, verifies an
   explicit override wins, and confirms the bootstrap database URL is
   unchanged without printing it.
7. Upgrade the Helm release to one replica of every long-running role.
8. Wait for all Deployments, the Instance Manager StatefulSet, Services, and
   endpoint slices to become healthy.

This order ensures the real service processes read the seeded DB configuration
during their first startup instead of requiring a restart after seeding.

## Functional Verification

The test passes only when all of the following hold:

- MySQL is Ready and DB Init, Config Seed, and Config Verify exit zero;
- Control and Application health endpoints return HTTP 200 through their
  Services;
- Scheduler owns all configured shard leases;
- Reconciler remains running without restart;
- Instance Manager is registered, healthy, and exposes one available slot;
- one deterministic `nexus_single` execution using the repository integration
  fixture reaches `succeeded` with the expected output;
- one cached `nexus_live` execution reaches its active state, and Application
  Gateway proxies `/ping` to the fake Runtime successfully;
- the live execution is then stopped cleanly;
- bounded logs contain no unexpected error, exception, or traceback.

The workload image already includes `tests/integration/distributed/fixtures.py`,
which supplies the deterministic task, interactive agent, and fake Runtime
provider used by these checks. No model or Gongfeng credential is required.

## Resources and Failure Handling

Each role uses one replica with explicit small CPU and memory requests and
bounded limits. Instance Manager capacity is one and the cluster uses 16
scheduler shards.

On failure, collect bounded status, events, Job logs, service logs, execution
records, and manager state before cleanup. Do not print Secret contents or
full container environments.

After evidence collection, uninstall only this Helm release and delete its
ephemeral Secret and auxiliary Config Jobs. Verify that no object with the
test release labels remains. The three uniquely tagged internal images remain
available for diagnosis and reproduction.
