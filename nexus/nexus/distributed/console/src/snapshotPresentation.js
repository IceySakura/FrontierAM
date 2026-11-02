import { snapshotRows } from "./platformStatus.js";

export const SNAPSHOT_GROUPS = {
  queues: { domain: "execution", title: "Execution state", fields: ["state", "count", "schedulable", "preparing", "retry_wait", "oldest_wait_seconds", "unknown_wait_start", "overdue"] },
  allocations: { domain: "resources", title: "Allocation state", fields: ["state", "count", "desired", "ready", "manager_gap", "provisioning_overdue", "unknown_provisioning_budget", "release_overdue", "unknown_release_start"] },
  builds: { domain: "preparation", title: "Dependency builds", fields: ["state", "count", "oldest_age_seconds", "overdue", "unknown_budget", "release_overdue"] },
  batches: { domain: "execution", title: "Batch progress", fields: ["state", "stage", "count", "active_items", "terminal_items", "oldest_progress_age_seconds", "preparation_overdue", "unknown_preparation_budget", "finalization_overdue", "unbounded_finalization"] },
  commands: { domain: "execution", title: "Control commands", fields: ["operation", "state", "count", "oldest_age_seconds"] },
  uploads: { domain: "delivery", title: "Artifact uploads", fields: ["state", "count", "oldest_age_seconds", "overdue"] },
  leases: { domain: "execution", title: "Control leases", fields: ["component", "total", "active", "unowned"] },
};

function sum(rows, field) {
  return rows === null || rows.some((row) => !Number.isFinite(row[field])) ? null : rows.reduce((n, row) => n + row[field], 0);
}

export function snapshotCore(snapshot) {
  const queues = snapshotRows(snapshot?.groups?.queues);
  const allocations = snapshotRows(snapshot?.groups?.allocations);
  const builds = snapshotRows(snapshot?.groups?.builds);
  const knownWait = queues?.map((row) => row.oldest_wait_seconds).filter(Number.isFinite) || [];
  const missingWait = sum(queues, "unknown_wait_start");
  return {
    running: sum(queues?.filter((row) => row.state === "running") ?? null, "count"),
    waitingInput: sum(queues?.filter((row) => row.state === "waiting_input") ?? null, "count"),
    preparing: sum(queues, "preparing"),
    schedulable: sum(queues, "schedulable"),
    oldestWait: queues === null ? null : knownWait.length ? Math.max(...knownWait) : missingWait ? null : 0,
    missingWait,
    overdue: sum(queues, "overdue"),
    allocationRelease: sum(allocations, "release_overdue"),
    builderRelease: sum(builds, "release_overdue"),
    missingBuildDeadline: sum(builds, "unknown_budget"),
  };
}

export function snapshotTimingIssues(snapshot) {
  const result = [];
  for (const [group, field, label] of [
    ["queues", "unknown_wait_start", "Queue wait start missing"],
    ["builds", "unknown_budget", "Build deadlines missing"],
    ["allocations", "unknown_provisioning_budget", "Provisioning deadlines missing"],
    ["allocations", "unknown_release_start", "Release start missing"],
    ["batches", "unknown_preparation_budget", "Preparation deadlines missing"],
  ]) {
    const count = sum(snapshotRows(snapshot?.groups?.[group]), field);
    if (count > 0) result.push({ label, count, domain: SNAPSHOT_GROUPS[group].domain });
  }
  const server = snapshot?.groups?.inventory?.values?.database_server;
  if (server?.data_status === "error") result.push({ label: "Database server collection failed", domain: "database" });
  return result;
}

export function snapshotDomainSummary(snapshot, domain) {
  const spec = {
    preparation: ["builds", "count", "build records"],
    execution: ["commands", "count", "open commands"],
    delivery: ["uploads", "overdue", "overdue upload records"],
    resources: ["allocations", "count", "allocation records"],
  }[domain];
  if (!spec) return "";
  const value = sum(snapshotRows(snapshot?.groups?.[spec[0]]), spec[1]);
  return value > 0 ? `${value.toLocaleString()} ${spec[2]}` : "";
}
