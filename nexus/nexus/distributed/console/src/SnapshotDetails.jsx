import { numberValue, snapshotRows } from "./platformStatus";
import { SNAPSHOT_GROUPS } from "./snapshotPresentation";

export function snapshotTime(value) {
  return value ? new Date(/Z$|[+-]\d\d:\d\d$/.test(value) ? value : `${value}Z`).toLocaleString() : "Not collected";
}

export default function SnapshotDetails({ snapshot, domain, environmentId }) {
  const fieldsForUnknown = { overdue: "unknown_budget", provisioning_overdue: "unknown_provisioning_budget", release_overdue: "unknown_release_start", preparation_overdue: "unknown_preparation_budget" };
  return <div className="snapshot-detail-groups">{Object.entries(SNAPSHOT_GROUPS).filter(([key, group]) => group.domain === domain && (!environmentId || key !== "leases")).map(([key, { title, fields }]) => {
    const group = snapshot?.groups?.[key];
    const rows = snapshotRows(group);
    return <section className="snapshot-group" key={key} aria-label={title}>
      <div className="snapshot-group-header"><h4>{title}</h4><span className="muted">{group?.data_status || "Not collected"}</span></div>
      <p className="muted">Sampled {snapshotTime(group?.sampled_at)}{group?.error_code ? ` · ${group.error_code}` : ""}</p>
      {group?.values?.release_grace_seconds ? <p className="muted">Release overdue means no control-plane confirmation after {group.values.release_grace_seconds}s.</p> : null}
      {key === "builds" ? <p className="muted">Failed builds await retry; ready builds await cleanup.</p> : null}
      {rows === null ? <p>No complete sample available.</p> : !rows.length ? <p>No active records.</p> : <div className="snapshot-records">{rows.map((row, index) => <dl className="snapshot-record" key={index}>{fields.map((field) => {
        const unknown = row[fieldsForUnknown[field]] || 0;
        const value = typeof row[field] === "number" ? numberValue(row[field]) : row[field] ?? "Not recorded";
        return <div key={field}><dt>{field.replaceAll("_", " ")}</dt><dd>{unknown ? `${value} confirmed · ${unknown} missing timing` : value}</dd></div>;
      })}</dl>)}</div>}
    </section>;
  })}</div>;
}
