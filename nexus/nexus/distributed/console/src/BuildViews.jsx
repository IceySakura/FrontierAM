import { useEffect, useState } from "react";
import ConsoleLink from "./ConsoleLink.jsx";
import { managerRoleLabel } from "./executionPresentation.js";
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
  timeText,
} from "./ResourceUI.jsx";

export function buildSchedule(build) {
  if (build?.status === "failed" && build.next_retry_at)
    return `Retry scheduled ${timeText(build.next_retry_at)}`;
  if (build?.status === "ready" && build.next_retry_at)
    return `Cleanup scheduled ${timeText(build.next_retry_at)}`;
  return "";
}
export function BuildReference({ build, id }) {
  const identity = id || build?.dependency_build_id;
  if (!identity) return <span>Build not required</span>;
  return (
    <div>
      <ResourceIdLink
        to={{ section: "build", buildId: identity }}
        value={identity}
      />
      {build ? (
        <>
          <div>
            <ResourceBadge>{build.status}</ResourceBadge> · attempt{" "}
            {build.attempt ?? "Unknown"}
          </div>
          <div className="muted">{buildSchedule(build)}</div>
          {build.error_code ? (
            <div className="resource-query-error">{build.error_code}</div>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
export function BuildsView({ request, controlPath, route }) {
  const q = route.query || {},
    key = JSON.stringify(q);
  const [draft, setDraft] = useState(q);
  useEffect(() => setDraft(q), [key]);
  const params = Object.fromEntries(
    ["q", "status", "environment_id", "owner", "execution_id", "cursor"]
      .filter((k) => q[k])
      .map((k) => [k, q[k]]),
  );
  const path = controlPath(
    `/v1/builds?${new URLSearchParams({ ...params, limit: q.page_size || "50" })}`,
  );
  const query = useResourceQuery(
    path,
    (signal) => request("GET", path, undefined, { signal }),
    { interval: 3000 },
  );
  return (
    <main className="panel single-view">
      <div className="panel-header">
        <div>
          <h2>Build diagnostics</h2>
          <span className="muted">
            Dependency builds can be reused by multiple Runs.
          </span>
        </div>
        <RefreshStatus query={query} interval={3} />
      </div>
      {q.return_to ? <ResourceBack route={route} /> : null}
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
        {[
          ["q", "Build ID prefix"],
          ["environment_id", "Environment ID"],
          ["owner", "Owner"],
          ["execution_id", "Run ID"],
        ].map(([k, label]) => (
          <input
            key={k}
            aria-label={label}
            placeholder={label}
            value={draft[k] || ""}
            onChange={(e) =>
              setDraft({
                ...draft,
                [k]: e.target.value,
                cursor: undefined,
                page: undefined,
              })
            }
          />
        ))}
        <select
          aria-label="Build status"
          value={draft.status || ""}
          onChange={(e) =>
            setDraft({
              ...draft,
              status: e.target.value,
              cursor: undefined,
              page: undefined,
            })
          }
        >
          <option value="">All statuses</option>
          {["requested", "building", "failed", "ready"].map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
        <button type="submit" className="row-action">
          Apply filters
        </button>
      </form>
      <ResourceNotice query={query} noun="builds" />
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Build</th>
              <th>Environment</th>
              <th>Status / schedule</th>
              <th>Attempt</th>
              <th>Created</th>
              <th>Finished</th>
            </tr>
          </thead>
          <tbody>
            {(query.data?.items || []).map((build) => (
              <tr key={build.dependency_build_id}>
                <td>
                  <ConsoleLink
                    to={{
                      section: "build",
                      buildId: build.dependency_build_id,
                    }}
                  >
                    {build.dependency_build_id}
                  </ConsoleLink>
                </td>
                <td>{build.environment_id}</td>
                <td>
                  <ResourceBadge>{build.status}</ResourceBadge>
                  <div className="muted">{buildSchedule(build)}</div>
                  {build.error_code}
                </td>
                <td>{build.attempt}</td>
                <td>{timeText(build.created_at)}</td>
                <td>{timeText(build.finished_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {query.data && !query.data.items.length ? (
        <p className="empty">No builds match these filters.</p>
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
export function BuildDetail({
  request,
  controlPath,
  route,
  renderLogs,
  isAdmin,
}) {
  const q = route.query || {},
    path = controlPath(`/v1/builds/${encodeURIComponent(route.buildId)}`);
  const [interval, setIntervalMs] = useState(3000);
  const metadata = useResourceQuery(
    path,
    (signal) => request("GET", path, undefined, { signal }),
    { interval },
  );
  const build = metadata.data;
  useEffect(
    () =>
      setIntervalMs(
        !build || ["requested", "building"].includes(build.status)
          ? 3000
          : build.status === "failed" && build.next_retry_at
            ? 10000
            : 0,
      ),
    [build?.status, build?.next_retry_at],
  );
  const usagePath =
    path +
    `/usage?${new URLSearchParams({ limit: "50", ...(q.cursor ? { cursor: q.cursor } : {}) })}`;
  const usage = useResourceQuery(
    usagePath,
    (signal) => request("GET", usagePath, undefined, { signal }),
    { enabled: q.tab === "usage", interval: 10000 },
  );
  return (
    <main className="panel single-view detail-page">
      <ResourceBack
        route={route}
        fallback={
          isAdmin
            ? { section: "infrastructure", infrastructureView: "builds" }
            : { section: "runs" }
        }
        label="Back to source"
      />
      <div className="detail-head">
        <div>
          <h2>Build diagnostics</h2>
          <CopyId value={route.buildId} />
          {build ? (
            <div className="detail-subtitle">
              <ResourceBadge>{build.status}</ResourceBadge>
              <span>Attempt {build.attempt}</span>
              <span>{buildSchedule(build)}</span>
            </div>
          ) : null}
        </div>
        <RefreshStatus query={metadata} interval={interval / 1000} />
      </div>
      <ResourceNotice query={metadata} noun="build metadata" />
      <div className="detail-tabs run-secondary-tabs">
        {[
          ["overview", "Overview"],
          ["logs", "Logs"],
          ["usage", "Related Runs / allocations"],
        ].map(([key, label]) => (
          <button
            key={key}
            className={(q.tab || "overview") === key ? "active" : ""}
            onClick={() =>
              updateRouteQuery(
                route,
                { tab: key, cursor: undefined, page: undefined },
                { replace: true },
              )
            }
          >
            {label}
          </button>
        ))}
      </div>
      {build && (!q.tab || q.tab === "overview") ? (
        <div className="detail-body">
          <ResourceFacts
            values={[
              ["Status", <ResourceBadge>{build.status}</ResourceBadge>],
              ["Environment", build.environment_id],
              ["Owner", build.owner_principal_id],
              ["Attempt", build.attempt],
              ["Created", timeText(build.created_at)],
              ["Started", timeText(build.started_at)],
              ["Finished", timeText(build.finished_at)],
              ["Next action", buildSchedule(build) || "None scheduled"],
            ]}
          />
          {build.error_code ? (
            <div role="status" className="resource-query-error">
              {build.error_code}
              {build.error_message ? `: ${build.error_message}` : ""}
            </div>
          ) : null}
        </div>
      ) : null}
      {!q.tab || q.tab === "overview" ? (
        <p className="resource-scope muted">
          Build status and retry scheduling come from the build record. Log
          availability is separate; expired logs do not mean the build failed.
        </p>
      ) : null}
      {q.tab === "logs" ? renderLogs(build) : null}
      {q.tab === "usage" ? (
        <section className="detail-body">
          <ResourceNotice query={usage} noun="related resources" />
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Type</th>
                  <th>Run status</th>
                  <th>Allocation</th>
                  <th>Role</th>
                  <th>Allocation status</th>
                </tr>
              </thead>
              <tbody>
                {(usage.data?.items || []).map((row) => (
                  <tr key={row.allocation_id}>
                    <td>
                      <ConsoleLink
                        to={{
                          section: "execution",
                          executionId: row.top_level_execution_id,
                        }}
                      >
                        {row.top_level_execution_id}
                      </ConsoleLink>
                    </td>
                    <td>{row.spec_type}</td>
                    <td>{row.execution_status}</td>
                    <td>
                      <ConsoleLink
                        to={{
                          section: "allocation",
                          allocationId: row.allocation_id,
                        }}
                      >
                        {row.allocation_id}
                      </ConsoleLink>
                    </td>
                    <td>{managerRoleLabel(row.workload)}</td>
                    <td>{row.allocation_status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {usage.data && !usage.data.items.length ? (
            <p className="empty">
              No visible allocations reference this build.
            </p>
          ) : null}
          <ResourcePager
            route={route}
            nextCursor={usage.data?.next_cursor}
            loading={usage.loading}
          />
        </section>
      ) : null}
    </main>
  );
}
