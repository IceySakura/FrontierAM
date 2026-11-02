import { useEffect, useState } from "react";
import ConsoleLink from "./ConsoleLink.jsx";
import { managerRoleLabel } from "./executionPresentation.js";
import { BuildReference } from "./BuildViews.jsx";
import { useResourceQuery } from "./resourceQuery.js";
import { updateRouteQuery } from "./routes.js";
import {
  CopyId,
  ResourceIdLink,
  RefreshStatus,
  ResourceBack,
  ResourceBadge,
  ResourceFacts,
  ResourceNotice,
  ResourcePager,
  countText,
  timeText,
} from "./ResourceUI.jsx";

export function AllocationTable({
  allocations,
  isAdmin = false,
  showRun = true,
}) {
  return (
    <table>
      <thead>
        <tr>
          <th>{showRun ? "Allocation / Run" : "Allocation"}</th>
          <th>Environment</th>
          <th>Role</th>
          <th>Allocation status</th>
          <th>Dependency build</th>
          <th>Managers ready / desired</th>
          <th title="Configured slots for this allocation; released allocations no longer reserve capacity">Slot capacity</th>
        </tr>
      </thead>
      <tbody>
        {allocations.map((a) => (
          <tr key={a.allocation_id}>
            <td>
              <ResourceIdLink
                to={{ section: "allocation", allocationId: a.allocation_id }}
                value={a.allocation_id}
              />
              {showRun ? (
                <div>
                  <ResourceIdLink
                    to={{
                      section: "execution",
                      executionId: a.top_level_execution_id,
                    }}
                    value={a.top_level_execution_id}
                  />
                </div>
              ) : null}
            </td>
            <td>{a.environment_id}</td>
            <td>
              {managerRoleLabel(a.workload)}
            </td>
            <td>
              <ResourceBadge>
                {a.degraded ? `${a.status} (degraded)` : a.status}
              </ResourceBadge>
            </td>
            <td>
              <BuildReference build={a.dependency_build} />
            </td>
            <td>
              {countText(a.ready_manager_count)} /{" "}
              {countText(a.desired_manager_count)}
              {isAdmin ? (
                <div>
                  <ConsoleLink
                    to={{
                      section: "infrastructure",
                      infrastructureView: "managers",
                      query: { allocation_id: a.allocation_id, health: "all" },
                    }}
                  >
                    View managers
                  </ConsoleLink>
                </div>
              ) : null}
            </td>
            <td>{countText(a.slots)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
export function AllocationList({ request, controlPath, route, isAdmin }) {
  const q = route.query || {},
    key = JSON.stringify(q),
    [draft, setDraft] = useState(q);
  useEffect(() => setDraft(q), [key]);
  const filters = Object.fromEntries(
    ["environment_id", "status", "owner", "execution_id", "cursor"]
      .filter((k) => q[k])
      .map((k) => [k, q[k]]),
  );
  const type = q.allocation_type || "dynamic";
  if (type !== "all") filters.allocation_type = type;
  const path = controlPath(
    `/v1/allocations?${new URLSearchParams({ ...filters, limit: q.page_size || "50" })}`,
  );
  const query = useResourceQuery(
    path,
    (signal) => request("GET", path, undefined, { signal }),
    { interval: 10000 },
  );
  return (
    <main id="allocations-view" className="panel single-view">
      <div className="panel-header">
        <h2>Allocations</h2>
        <RefreshStatus query={query} interval={10} />
      </div>
      <form
        className="filters resource-filters"
        onSubmit={(e) => {
          e.preventDefault();
          updateRouteQuery(
            route,
            { ...draft, cursor: undefined, page: undefined },
            { resetPage: true },
          );
        }}
      >
        <select
          aria-label="Allocation type"
          value={draft.allocation_type || "dynamic"}
          onChange={(e) =>
            setDraft({ ...draft, allocation_type: e.target.value })
          }
        >
          {["dynamic", "predefined", "all"].map((x) => (
            <option key={x}>{x}</option>
          ))}
        </select>
        <select
          aria-label="Allocation status"
          value={draft.status || ""}
          onChange={(e) => setDraft({ ...draft, status: e.target.value })}
        >
          <option value="">All statuses</option>
          {[
            "requested",
            "provisioning",
            "ready",
            "draining",
            "deleting",
            "released",
          ].map((x) => (
            <option key={x}>{x}</option>
          ))}
        </select>
        {[
          ["environment_id", "Environment ID"],
          ["execution_id", "Run ID"],
          ["owner", "Owner"],
        ].map(([k, label]) => (
          <input
            key={k}
            aria-label={label}
            placeholder={label}
            value={draft[k] || ""}
            onChange={(e) => setDraft({ ...draft, [k]: e.target.value })}
          />
        ))}
        <button type="submit" className="row-action">
          Apply filters
        </button>
      </form>
      <ResourceNotice query={query} noun="allocations" />
      <p className="muted resource-scope">
        Manager readiness reflects allocation provisioning. Use View managers for
        current health. Slot capacity is the configured total, not live reservations.
      </p>
      <div className="table-wrap">
        <AllocationTable
          allocations={query.data?.items || []}
          isAdmin={isAdmin}
        />
      </div>
      {query.data && !query.data.items.length ? (
        <p className="empty">No allocations match these filters.</p>
      ) : null}
      <ResourcePager
        pageSize
        route={route}
        nextCursor={query.data?.next_cursor}
        loading={query.loading}
      />
    </main>
  );
}
export function AllocationDetail({ request, controlPath, route, isAdmin }) {
  const path = controlPath(
    `/v1/allocations/${encodeURIComponent(route.allocationId)}`,
  );
  const query = useResourceQuery(
    path,
    (signal) => request("GET", path, undefined, { signal }),
    { interval: 10000 },
  );
  const a = query.data,
    [busy, setBusy] = useState(false),
    [error, setError] = useState(null);
  const cleanup = async () => {
    if (
      !a ||
      !window.confirm(
        `Request cleanup of allocation ${a.allocation_id}?\nRun: ${a.top_level_execution_id}\nEnvironment: ${a.environment_id}\nStatus: ${a.status}\nActive assignments: ${a.active_assignments}\nThe allocation will enter draining and its resources will be released asynchronously. Active work may be interrupted.`,
      )
    )
      return;
    setBusy(true);
    setError(null);
    try {
      await request(
        "POST",
        path +
          `/force-cleanup?expected_version=${encodeURIComponent(a.status_version)}`,
        {},
      );
      query.refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <main className="panel single-view detail-page">
      <ResourceBack
        route={route}
        fallback={
          isAdmin
            ? { section: "infrastructure", infrastructureView: "allocations" }
            : { section: "runs" }
        }
        label="Back to source"
      />
      <div className="detail-head">
        <div>
          <h2>Allocation</h2>
          <CopyId value={route.allocationId} />
        </div>
        <RefreshStatus query={query} interval={10} />
      </div>
      <ResourceNotice query={query} noun="allocation" />
      {a ? (
        <div className="detail-body">
          <ResourceFacts
            values={[
              [
                "Run",
                <ConsoleLink
                  to={{
                    section: "execution",
                    executionId: a.top_level_execution_id,
                  }}
                >
                  {a.top_level_execution_id}
                </ConsoleLink>,
              ],
              ["Environment", a.environment_id],
              ["Owner", a.owner_principal_id],
              ["Role", managerRoleLabel(a.workload)],
              ["Status", <ResourceBadge>{a.status}</ResourceBadge>],
              ["Active assignments", a.active_assignments],
              [
                "Managers ready / desired",
                `${countText(a.ready_manager_count)} / ${countText(a.desired_manager_count)}`,
              ],
              ["Slot capacity", a.slots],
              ["Created", timeText(a.created_at)],
              ["Last progress", timeText(a.last_progress_at)],
              ["Released", timeText(a.released_at)],
            ]}
          />
          <section className="detail-section">
            <h3>Dependency build</h3>
            <BuildReference build={a.dependency_build} />
          </section>
          {isAdmin ? (
            <section className="detail-section">
              <h3>Managers and provider resources</h3>
              <ConsoleLink
                to={{
                  section: "infrastructure",
                  infrastructureView: "managers",
                  query: { allocation_id: a.allocation_id, health: "all" },
                }}
              >
                View managers for this allocation
              </ConsoleLink>
              <pre className="code-block">
                {JSON.stringify(a.provider_resource_ref || {}, null, 2)}
              </pre>
              {a.last_error_code ? (
                <div className="resource-query-error">
                  {a.last_error_code}: {a.last_error_message || ""}
                </div>
              ) : null}
            </section>
          ) : null}
          {["draining", "deleting"].includes(a.status) ? (
            <p role="status">
              Cleanup is in progress. Resources have not been confirmed released
              yet.
            </p>
          ) : a.status === "released" ? (
            <p role="status">Resource release is confirmed.</p>
          ) : null}
          {error ? (
            <div className="alert" role="alert">
              {error}
            </div>
          ) : null}
          {isAdmin && a.status !== "released" ? (
            <button
              className="row-action danger"
              disabled={busy}
              onClick={cleanup}
            >
              {busy ? "Requesting cleanup…" : "Request force cleanup"}
            </button>
          ) : null}
        </div>
      ) : null}
    </main>
  );
}
export function RunPreparation({
  allocations = [],
  batchSummary,
  isAdmin,
  compact = false,
  openResources,
}) {
  const childBuild = batchSummary?.child_dependency_build_id;
  const roleOrder = { orchestration: 0, execution: 1 };
  const orderedAllocations = [...allocations].sort(
    (left, right) => (roleOrder[left.workload] ?? 2) - (roleOrder[right.workload] ?? 2),
  );
  if (!allocations.length && !childBuild) return null;
  if (compact)
    return (
      <section className="preparation-overview">
        <div className="section-heading">
          <h3>Resources</h3>
          <button className="row-action" onClick={openResources}>
            View details →
          </button>
        </div>
        <div className="preparation-cards">
          {orderedAllocations.map((a) => (
            <div
              className={`preparation-card ${a.dependency_build?.status === "failed" ? "has-error" : ""}`}
              key={a.allocation_id}
            >
              <div className="section-heading">
                <strong>
                  {managerRoleLabel(a.workload)}
                </strong>
                <span className="preparation-allocation-state">
                  <span className="muted">Allocation</span>
                  <ResourceBadge>{a.status}</ResourceBadge>
                </span>
              </div>
              <div className="muted preparation-build-label">Dependency build</div>
              <BuildReference build={a.dependency_build} />
            </div>
          ))}
        </div>
        {childBuild &&
        !allocations.some(
          (a) => a.dependency_build?.dependency_build_id === childBuild,
        ) ? (
          <BuildReference id={childBuild} />
        ) : null}
      </section>
    );
  return (
    <section className="detail-section">
      <h3>Allocations and dependency builds</h3>

      <div className="table-wrap">
        <AllocationTable
          allocations={orderedAllocations}
          isAdmin={isAdmin}
          showRun={false}
        />
      </div>
      {childBuild &&
      !allocations.some(
        (a) => a.dependency_build?.dependency_build_id === childBuild,
      ) ? (
        <div>
          <h4>Execution dependencies</h4>
          <BuildReference id={childBuild} />
        </div>
      ) : null}
    </section>
  );
}
