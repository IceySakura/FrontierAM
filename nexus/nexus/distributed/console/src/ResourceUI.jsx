import { Copy, Check } from "lucide-react";
import { useState } from "react";
import ConsoleLink from "./ConsoleLink.jsx";
import {
  cursorNavigation,
  savedListRoute,
  routeFromPath,
  updateRouteQuery,
} from "./routes.js";

export const timeText = (value) => {
  if (!value) return "—";
  const normalized =
    typeof value === "string" &&
    /^\d{4}-\d{2}-\d{2}T/.test(value) &&
    !/(?:Z|[+-]\d{2}:?\d{2})$/.test(value)
      ? `${value}Z`
      : value;
  return new Date(normalized).toLocaleString();
};
export const countText = (value) =>
  Number.isFinite(value) ? value.toLocaleString() : "Unknown";
export function ResourceBadge({ children, tone }) {
  tone ||= ["ready", "running", "succeeded", "active"].includes(children)
    ? "good"
    : ["failed", "lost", "expired", "killed", "cancelled"].includes(children)
      ? "bad"
      : ["requested", "building", "queued", "starting"].includes(children)
        ? "info"
        : ["draining", "deleting", "stopping"].includes(children)
          ? "warn"
          : "neutral";
  return <span className={`badge ${tone}`}>{children ?? "Unknown"}</span>;
}
export function RefreshStatus({ query, queries, interval = 0 }) {
  const states = queries || [query];
  const failed = states.some((q) => q?.error);
  const loading = states.some((q) => q?.loading);
  const stamps = states.map((q) => q?.updatedAt).filter(Boolean);
  const title = stamps.length
    ? `Last successful refresh: ${timeText(Math.min(...stamps))}`
    : "Waiting for data";
  return (
    <span
      className={`refresh-status ${failed ? "resource-query-error" : "muted"}`}
      title={title}
    >
      <span className={`refresh-dot ${loading ? "refreshing" : ""}`} />
      {failed
        ? "Update incomplete"
        : interval
          ? `Auto-refresh ${interval}s`
          : loading
            ? "Loading"
            : stamps.length
              ? "Loaded"
              : ""}
    </span>
  );
}
export function ResourceNotice({ query, noun = "data" }) {
  if (!query.error && (!query.loading || query.data || query.updatedAt))
    return null;
  return (
    <div
      role="status"
      className={`resource-query-status ${query.error ? "resource-query-error" : "muted"}`}
    >
      {query.error
        ? `${query.stale ? `Showing retained ${noun} from ${timeText(query.updatedAt)}. ` : ""}Could not load ${noun}: ${query.error.message}`
        : `Loading ${noun}…`}
    </div>
  );
}
export const shortResourceId = (value) =>
  value && value.length > 25
    ? `${value.slice(0, 13)}…${value.slice(-8)}`
    : value || "—";
export function ResourceIdLink({ to, value }) {
  return (
    <span className="resource-id">
      <ConsoleLink to={to} title={value}>
        {shortResourceId(value)}
      </ConsoleLink>
      <CopyId value={value} buttonOnly />
    </span>
  );
}
export function ResourcePager({
  route,
  nextCursor,
  loading = false,
  pageSize = false,
}) {
  const pager = cursorNavigation(route, nextCursor);
  return (
    <div className="pagination-footer">
      {pageSize ? (
        <label className="page-size">
          Rows per page{" "}
          <select
            aria-label="Rows per page"
            value={route.query?.page_size || "50"}
            onChange={(e) =>
              updateRouteQuery(
                route,
                { page_size: e.target.value },
                { resetPage: true },
              )
            }
          >
            {[25, 50, 100].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      <span>{pager.page ? `Page ${pager.page}` : "Current page"}</span>
      {route.query?.cursor ? (
        <button onClick={pager.first} disabled={loading}>
          First page
        </button>
      ) : null}
      <button
        aria-label="previous page"
        disabled={loading || !pager.canPrevious}
        onClick={pager.previous}
      >
        Previous
      </button>
      <button
        aria-label="next page"
        disabled={loading || !pager.canNext}
        onClick={pager.next}
      >
        Next
      </button>
    </div>
  );
}
export function CopyId({ value, buttonOnly = false, compact = false }) {
  const [copied, setCopied] = useState(false);
  return (
    <span className="copy-identity">
      {!buttonOnly ? (
        <code title={value}>
          {compact ? shortResourceId(value) : value || "—"}
        </code>
      ) : null}
      {value ? (
        <button
          className="copy-id-button"
          title={copied ? "Copied" : "Copy full ID"}
          aria-label={`Copy ${value}`}
          onClick={async (event) => {
            event.stopPropagation();
            try {
              await navigator.clipboard.writeText(value);
              setCopied(true);
            } catch {
              setCopied(false);
            }
          }}
        >
          {copied ? <Check size={13} /> : <Copy size={13} />}
        </button>
      ) : null}
    </span>
  );
}
export function ResourceBack({
  route,
  fallback = savedListRoute("runs"),
  label = "Back",
}) {
  const source = route.query?.return_to
    ? routeFromPath(route.query.return_to)
    : fallback;
  const name =
    source.section === "execution"
      ? source.query?.item_id
        ? "Back to subtask"
        : "Back to Run"
      : source.section === "build"
        ? "Back to Build"
        : source.section === "infrastructure"
          ? `Back to ${source.infrastructureView || "Managers"}`
          : source.section === "allocation"
            ? "Back to Allocation"
            : source.section === "builds"
              ? "Back to Builds"
              : "Back to Runs";
  return (
    <ConsoleLink
      className="back-link"
      origin={false}
      to={route.query?.return_to || fallback}
    >
      {label === "Back" || label === "Back to source" ? `← ${name}` : label}
    </ConsoleLink>
  );
}
export function ResourceFacts({ values }) {
  return (
    <div className="section-grid">
      {values.map(([label, value]) => (
        <div className="kv" key={label}>
          <div className="key">{label}</div>
          <div className="value">{value ?? "Unknown"}</div>
        </div>
      ))}
    </div>
  );
}
