import { useEffect, useState } from "react";
import ConsoleLink from "./ConsoleLink.jsx";
import { managerRoleLabel } from "./executionPresentation.js";
import { useResourceQuery } from "./resourceQuery.js";
import { updateRouteQuery } from "./routes.js";
import {
  CopyId,
  RefreshStatus,
  ResourceIdLink,
  shortResourceId,
  ResourceBadge,
  ResourceFacts,
  ResourceNotice,
  ResourcePager,
  countText,
  timeText,
} from "./ResourceUI.jsx";

const statusQuery = (value) => ({
  health: value === "lost" ? "lost" : value === "all" ? "all" : "healthy",
  draining:
    value === "accepting" ? "0" : value === "draining" ? "1" : undefined,
});

function InventoryCount({ value }) {
  if (!Number.isFinite(value)) return countText(value);
  return new Intl.NumberFormat().formatToParts(value).map((part, index) => (
    <span key={index}>{part.value}{part.type === "group" ? <wbr /> : null}</span>
  ));
}

export default function ManagerInventory({
  request,
  requestText,
  controlPath,
  route,
  environments = [],
}) {
  const q = route.query || {};
  const scope = Object.fromEntries(
    ["environment_id", "pool_id", "workload", "allocation_id"]
      .filter((k) => q[k])
      .map((k) => [k, q[k]]),
  );
  const scopeKey = JSON.stringify(scope);
  const currentStatus =
    q.draining === "1"
      ? "draining"
      : q.draining === "0"
        ? "accepting"
        : q.health || "healthy";
  const [draft, setDraft] = useState({ ...scope, state: currentStatus });
  useEffect(
    () => setDraft({ ...scope, state: currentStatus }),
    [scopeKey, currentStatus],
  );
  const listPath = controlPath(
    `/v1/managers?${new URLSearchParams({ ...scope, limit: q.page_size || "50", health: q.health || "healthy", ...(q.draining !== undefined ? { draining: q.draining } : {}), ...(q.cursor ? { cursor: q.cursor } : {}) })}`,
  );
  const list = useResourceQuery(
    listPath,
    (signal) => request("GET", listPath, undefined, { signal }),
    { interval: 10000 },
  );
  const summaryPath = controlPath(
    `/v1/managers/summary?${new URLSearchParams(scope)}`,
  );
  const summary = useResourceQuery(
    summaryPath,
    (signal) => request("GET", summaryPath, undefined, { signal }),
    { interval: 10000 },
  );
  const poolsPath = controlPath(
    `/v1/manager-pools?${new URLSearchParams({ ...(draft.environment_id ? { environment_id: draft.environment_id } : {}), ...(draft.pool_id ? { q: draft.pool_id } : {}), limit: "50" })}`,
  );
  const pools = useResourceQuery(poolsPath, (signal) =>
    request("GET", poolsPath, undefined, { signal }),
  );
  const detailPath = q.manager_id
    ? controlPath(`/v1/managers/${encodeURIComponent(q.manager_id)}`)
    : "";
  const detail = useResourceQuery(
    detailPath,
    (signal) => request("GET", detailPath, undefined, { signal }),
    { enabled: Boolean(detailPath), interval: 10000 },
  );
  const manager = detail.data;
  const session = q.session_id || manager?.session_id;
  const logsPath =
    q.manager_id && session
      ? controlPath(
          `/v1/managers/${encodeURIComponent(q.manager_id)}/logs?${new URLSearchParams({ session_id: session, lines: q.lines || "200" })}`,
        )
      : "";
  const logs = useResourceQuery(
    logsPath,
    (signal) => requestText("GET", logsPath, undefined, { signal }),
    { enabled: Boolean(logsPath) },
  );
  useEffect(() => {
    if (!q.manager_id) return;
    const timer = setTimeout(
      () =>
        document
          .getElementById("manager-log-viewer")
          ?.scrollIntoView({ block: "start", behavior: "smooth" }),
      100,
    );
    return () => clearTimeout(timer);
  }, [q.manager_id]);
  const [action, setAction] = useState(null),
    [actionError, setActionError] = useState(null);
  const act = async (row, operation) => {
    const draining = !row.draining;
    const prompt =
      operation === "delete"
        ? `Delete stale manager record ${row.manager_id} (session ${row.session_id})? Active assignments prevent deletion.`
        : `${draining ? "Drain" : "Resume"} manager ${row.manager_id} (session ${row.session_id})? ${draining ? "It will stop accepting new assignments; active tasks continue." : "It will accept new assignments again."}`;
    if (!window.confirm(prompt)) return;
    setAction(row.manager_id);
    setActionError(null);
    try {
      if (operation === "delete")
        await request(
          "DELETE",
          controlPath(
            `/v1/managers/${encodeURIComponent(row.manager_id)}?session_id=${encodeURIComponent(row.session_id)}`,
          ),
        );
      else
        await request(
          "PATCH",
          controlPath(`/v1/managers/${encodeURIComponent(row.manager_id)}`),
          { draining, expected_session_id: row.session_id },
        );
      list.refresh();
      summary.refresh();
      detail.refresh();
      if (operation === "delete" && row.manager_id === q.manager_id)
        updateRouteQuery(route, {
          manager_id: undefined,
          session_id: undefined,
        });
    } catch (error) {
      setActionError(`${row.manager_id}: ${error.message}`);
    } finally {
      setAction(null);
    }
  };
  const matching = summary.data
    ? {
        healthy: summary.data.healthy,
        all: summary.data.total,
        lost: summary.data.lost,
        accepting: summary.data.accepting,
        draining: summary.data.draining,
      }[currentStatus]
    : null;
  const apply = (event) => {
    event.preventDefault();
    updateRouteQuery(
      route,
      {
        ...draft,
        state: undefined,
        ...statusQuery(draft.state),
        manager_id: undefined,
        session_id: undefined,
      },
      { resetPage: true },
    );
  };
  return (
    <main id="managers-view" className="panel single-view">
      <div className="panel-header">
        <h2>Managers</h2>
        <RefreshStatus queries={[list, summary]} interval={10} />
      </div>
      <form className="filters resource-filters" onSubmit={apply}>
        <select
          aria-label="Manager environment"
          value={draft.environment_id || ""}
          onChange={(e) =>
            setDraft({ ...draft, environment_id: e.target.value, pool_id: "" })
          }
        >
          <option value="">All environments</option>
          {environments.map((e) => (
            <option key={e.environment_id} value={e.environment_id}>
              {e.name || e.environment_id}
            </option>
          ))}
        </select>
        <input
          aria-label="Manager pool"
          list="manager-pools"
          placeholder="Pool"
          value={draft.pool_id || ""}
          onChange={(e) => setDraft({ ...draft, pool_id: e.target.value })}
        />
        <datalist id="manager-pools">
          {(pools.data?.items || []).map((p) => (
            <option value={p} key={p} />
          ))}
        </datalist>
        <select
          aria-label="Manager state"
          value={draft.state}
          onChange={(e) => setDraft({ ...draft, state: e.target.value })}
        >
          {[
            ["healthy", "Healthy · all online"],
            ["accepting", "Accepting work"],
            ["draining", "Draining · online"],
            ["lost", "Lost / stale"],
            ["all", "All states"],
          ].map(([v, label]) => (
            <option key={v} value={v}>
              {label}
            </option>
          ))}
        </select>
        <select
          aria-label="Manager role"
          value={draft.workload || ""}
          onChange={(e) => setDraft({ ...draft, workload: e.target.value })}
        >
          <option value="">All manager roles</option>
          {["orchestration", "execution"].map((role) => (
            <option key={role} value={role}>{managerRoleLabel(role)}</option>
          ))}
          {draft.workload && !["orchestration", "execution"].includes(draft.workload) ? (
            <option value={draft.workload}>{managerRoleLabel(draft.workload)}</option>
          ) : null}
        </select>
        <input
          aria-label="Manager allocation"
          placeholder="Allocation ID"
          value={draft.allocation_id || ""}
          onChange={(e) =>
            setDraft({ ...draft, allocation_id: e.target.value })
          }
        />
        <button type="submit" className="row-action">
          Apply filters
        </button>
      </form>
      <div className="summary-grid inventory-summary">
        {[
          ["Total inventory", "total"],
          ["Healthy", "healthy"],
          ["Accepting work", "accepting"],
          ["Draining · online", "draining"],
          ["Lost / stale", "lost"],
          ["Live capacity", "capacity_total"],
          ["Live reserved slots", "reserved_slots"],
          ["Available slots", "available_slots"],
        ].map(([label, key]) => (
          <div className="summary-card" key={key}>
            <div className="summary-label">{label}</div>
            <div className="summary-value">
              <InventoryCount value={summary.data?.[key]} />
            </div>
          </div>
        ))}
      </div>
      <p className="muted resource-scope">
        Inventory totals cover the selected environment, pool, manager role and
        allocation, across all pages. Healthy includes online draining managers.
        Capacity figures cover healthy current sessions.
      </p>
      <ResourceNotice query={summary} noun="inventory totals" />
      {summary.data?.data_status === "stale" ? (
        <p role="status" className="resource-query-status resource-query-error">
          Inventory totals could not refresh. Showing the last successful sample
          from {timeText(summary.data.queried_at)}.
        </p>
      ) : null}
      <ResourceNotice query={list} noun="managers" />
      {actionError ? (
        <div role="alert" className="alert">
          {actionError}
        </div>
      ) : null}
      <p className="muted resource-scope">
        Showing {list.data?.items?.length ?? 0} on this page
        {matching != null ? ` · ${countText(matching)} matching managers` : ""}
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Manager / session</th>
              <th>Environment / pool</th>
              <th>Health</th>
              <th>Scheduling</th>
              <th>Allocation</th>
              <th>Capacity</th>
              <th>Reserved (session)</th>
              <th>Available</th>
              <th>Heartbeat</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {(list.data?.items || []).map((row) => (
              <tr key={row.manager_id}>
                <td>
                  <CopyId value={row.manager_id} compact />
                  <div className="muted" title={row.session_id}>
                    {shortResourceId(row.session_id)}
                  </div>
                </td>
                <td>
                  {row.environment_id}
                  <div className="muted" title={row.pool_id}>
                    {shortResourceId(row.pool_id)}
                  </div>
                  <div className="muted">{managerRoleLabel(row.workload)}</div>
                </td>
                <td>
                  <ResourceBadge tone={row.healthy ? "good" : "bad"}>
                    {row.healthy ? "Healthy" : "Lost / stale"}
                  </ResourceBadge>
                </td>
                <td>
                  {row.draining
                    ? "Draining"
                    : row.healthy
                      ? "Accepting"
                      : "Not accepting"}
                </td>
                <td>
                  {row.allocation_id ? (
                    <ResourceIdLink
                      to={{
                        section: "allocation",
                        allocationId: row.allocation_id,
                      }}
                      value={row.allocation_id}
                    />
                  ) : (
                    "Predefined"
                  )}
                </td>
                <td>{countText(row.capacity_total)}</td>
                <td>{countText(row.reserved_slots)}</td>
                <td>{countText(row.available_slots)}</td>
                <td>{timeText(row.last_heartbeat_at)}</td>
                <td>
                  <div className="row-actions">
                    <button
                      className="row-action"
                      onClick={() =>
                        updateRouteQuery(route, {
                          manager_id: row.manager_id,
                          session_id: row.session_id,
                        })
                      }
                    >
                      Logs
                    </button>
                    <button
                      className="row-action"
                      disabled={
                        Boolean(action) ||
                        (!row.healthy && row.reserved_slots > 0)
                      }
                      onClick={() =>
                        act(row, row.healthy ? "draining" : "delete")
                      }
                    >
                      {action === row.manager_id
                        ? "Working…"
                        : row.healthy
                          ? row.draining
                            ? "Resume"
                            : "Drain"
                          : "Delete stale record"}
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {list.data && !list.data.items.length ? (
        <p className="empty">No managers match these filters.</p>
      ) : null}
      <ResourcePager
        pageSize
        route={route}
        nextCursor={list.data?.next_cursor}
        loading={list.loading}
      />
      {q.manager_id ? (
        <section id="manager-log-viewer" className="detail-section">
          <div className="panel-header">
            <h3>Manager logs</h3>
            <button
              className="row-action"
              onClick={() =>
                updateRouteQuery(route, {
                  manager_id: undefined,
                  session_id: undefined,
                })
              }
            >
              Close logs
            </button>
          </div>
          <CopyId value={q.manager_id} />
          <p className="muted">Session {session || "Unknown"}</p>
          <ResourceNotice query={detail} noun="manager details" />
          {manager && session !== manager.session_id ? (
            <div className="alert">
              This manager has restarted. The selected session is no longer
              current.{" "}
              <button
                onClick={() =>
                  updateRouteQuery(route, { session_id: manager.session_id })
                }
              >
                View current session
              </button>
            </div>
          ) : null}
          <ResourceFacts
            values={[
              ["Environment", manager?.environment_id],
              ["Pool", manager?.pool_id],
              ["Role", managerRoleLabel(manager?.workload)],
              ["Endpoint", manager?.endpoint_url],
              ["Bootstrap version", manager?.bootstrap_version],
              [
                "Supported spec versions",
                JSON.stringify(manager?.supported_spec_versions || {}),
              ],
            ]}
          />
          <div className="toolbar dense">
            <select
              aria-label="Manager log lines"
              value={q.lines || "200"}
              onChange={(e) =>
                updateRouteQuery(
                  route,
                  { lines: e.target.value },
                  { replace: true },
                )
              }
            >
              {[100, 200, 500, 1000].map((n) => (
                <option key={n} value={n}>
                  {n} lines
                </option>
              ))}
            </select>
            <button disabled={logs.loading} onClick={logs.refresh}>
              Refresh logs
            </button>
          </div>
          <ResourceNotice query={logs} noun="logs for this session" />
          <pre className="log-output">
            {logs.data ??
              (logs.loading
                ? "Loading logs…"
                : logs.error
                  ? "Logs are unavailable."
                  : "")}
            {logs.data === "" ? "No log lines returned." : ""}
          </pre>
        </section>
      ) : null}
    </main>
  );
}
