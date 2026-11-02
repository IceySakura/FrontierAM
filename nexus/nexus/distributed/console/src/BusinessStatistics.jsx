import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BarChart3, Download, X } from "lucide-react";
import { BASIS_LABELS, STATUS_LABELS, TASK_KIND_LABELS, beijingTime, initialStatisticsForm, percent, statisticsCsv, statisticsError, statisticsQuery, taskLabel } from "./businessStatistics";

const COLORS = { succeeded: "#267b63", failed: "#bd5d54", cancelled: "#93a39a", unfinished: "#4b7da5" };
const count = (value) => Number.isFinite(value) ? value.toLocaleString("en-GB") : "—";

function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a"); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function TaskTrend({ buckets }) {
  const [hovered, setHovered] = useState(null);
  const maximum = Math.max(1, ...buckets.map((item) => item.total));
  const lastPopulated = buckets.findLastIndex((item) => item.total > 0);
  const selected = buckets[Math.min(hovered ?? (lastPopulated < 0 ? buckets.length - 1 : lastPopulated), buckets.length - 1)];
  const start = Date.parse(buckets[0]?.start_at), end = Date.parse(buckets.at(-1)?.end_at);
  const x = (value) => (Date.parse(value) - start) / Math.max(1, end - start) * 910;
  const y = (value) => 155 - value / maximum * 155;
  return <div className="business-chart">
    <div className="business-chart-plot">
      <div className="business-chart-axis">{[1, 0.5, 0].map((fraction) => <span key={fraction}>{count(maximum * fraction)}</span>)}</div>
      <svg viewBox="0 0 910 155" preserveAspectRatio="none" role="img" aria-label="Task volume by status over time" onMouseMove={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const stamp = start + Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * (end - start);
        const found = buckets.findIndex((item) => Date.parse(item.start_at) <= stamp && Date.parse(item.end_at) > stamp);
        setHovered(found < 0 ? buckets.length - 1 : found);
      }} onMouseLeave={() => setHovered(null)}>
        {[0, 0.5, 1].map((fraction) => <line key={fraction} x1="0" x2="910" y1={y(maximum * fraction)} y2={y(maximum * fraction)} stroke="#dfe7e2" strokeDasharray={fraction ? "3 5" : undefined} />)}
        {buckets.map((item) => {
          let height = 0;
          return <g key={item.start_at}>{Object.keys(COLORS).map((status) => {
            const base = height; height += item[status];
            return item[status] ? <rect key={status} x={x(item.start_at)} y={y(height)} width={Math.max(1, x(item.end_at) - x(item.start_at) - 1)} height={(height - base) / maximum * 155} fill={COLORS[status]} opacity={item.partial_period ? 0.65 : 1} /> : null;
          })}</g>;
        })}
      </svg>
    </div>
    <div className="business-chart-times muted"><span>{beijingTime(buckets[0]?.start_at)}</span><span>{beijingTime(buckets.at(-1)?.end_at)}</span></div>
    <div className="business-chart-legend">{Object.entries(COLORS).map(([key, color]) => <span key={key}><i style={{ background: color }} />{STATUS_LABELS[key]}</span>)}</div>
    {selected ? <div className="business-chart-readout muted">{beijingTime(selected.start_at)} — {beijingTime(selected.end_at)}{selected.partial_period ? " · Partial period" : ""}<span>Tasks {count(selected.total)} · Succeeded {count(selected.succeeded)} · Failed {count(selected.failed)} · Cancelled {count(selected.cancelled)} · Unfinished {count(selected.unfinished)}</span></div> : null}
  </div>;
}

export default function BusinessStatistics({ request, controlPath, environments = [], active }) {
  const [form, setForm] = useState(initialStatisticsForm);
  const [data, setData] = useState(null);
  const [submittedForm, setSubmittedForm] = useState(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState("");
  const [grain, setGrain] = useState("daily");
  const [page, setPage] = useState(0);
  const [sort, setSort] = useState("total");
  const controller = useRef(null);
  const version = useRef(0);
  const cancel = useCallback(() => {
    version.current++;
    const current = controller.current;
    controller.current = null;
    if (current) { current.abort(); setPending(false); setNotice("Query cancelled. You can generate statistics again."); }
  }, []);
  const generate = useCallback(async () => {
    if (!active || document.hidden || controller.current) return;
    let query;
    try { query = statisticsQuery(form); } catch (failure) { setError(failure.message); return; }
    const current = new AbortController(), id = ++version.current;
    controller.current = current; setPending(true); setError(null); setNotice("");
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; current.abort(); }, 20000);
    try {
      const result = await request("GET", controlPath(`/v1/system/business-statistics?${new URLSearchParams(query)}`), undefined, { signal: current.signal });
      if (!current.signal.aborted && version.current === id) {
        setData(result); setSubmittedForm(JSON.stringify(form)); setPage(0);
      }
    } catch (failure) {
      if (version.current === id && (!current.signal.aborted || timedOut)) setError(statisticsError(failure));
    } finally {
      clearTimeout(timer);
      if (controller.current === current) { controller.current = null; setPending(false); }
    }
  }, [active, controlPath, form, request]);
  useEffect(() => {
    if (!active) { cancel(); return; }
    const hide = () => { if (document.hidden) cancel(); };
    document.addEventListener("visibilitychange", hide);
    window.addEventListener("refresh-business-statistics", generate);
    return () => { document.removeEventListener("visibilitychange", hide); window.removeEventListener("refresh-business-statistics", generate); };
  }, [active, cancel, generate]);
  useEffect(() => () => { version.current++; controller.current?.abort(); }, []);
  const update = (key) => (event) => setForm((current) => ({ ...current, [key]: event.target.value }));
  const rows = useMemo(() => [...(data?.task_names.items || [])].sort((a, b) => sort === "name" ? taskLabel(a.task_name).localeCompare(taskLabel(b.task_name)) : b.total - a.total), [data, sort]);
  const pages = Math.max(1, Math.ceil(rows.length / 20));
  const currentPage = Math.min(page, pages - 1);
  const dirty = data && JSON.stringify(form) !== submittedForm;
  const distribution = data ? [...data.task_names.top.map((item) => ({ ...item, label: item.task_name })), ...(data.task_names.other.total ? [{ ...data.task_names.other, label: "Other names", share: data.task_names.other.total / data.summary.total }] : []), ...(data.task_names.unknown.total ? [{ ...data.task_names.unknown, label: "Missing name" }] : [])] : [];
  return <div className="workspace-stack" id="business-statistics">
    <section className="panel">
      <div className="panel-header"><div><h2>Business statistics</h2><span className="muted">Generate on demand · Beijing time (UTC+8) · Retries count as one logical task</span></div></div>
      <form onSubmit={(event) => { event.preventDefault(); generate(); }}>
        <div className="business-filters">
          <label>Time range<select aria-label="Statistics time range" value={form.preset} onChange={update("preset")}><option value="7d">Last 7 days</option><option value="30d">Last 30 days</option><option value="month">Calendar month</option><option value="custom">Custom</option></select></label>
          <label>Time basis<select aria-label="Statistics time basis" value={form.basis} onChange={update("basis")}>{Object.entries(BASIS_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          <label>Environment<select aria-label="Statistics environment" value={form.environment} onChange={update("environment")}><option value="">All environments</option>{environments.map((item) => <option key={item.environment_id} value={item.environment_id}>{item.name || item.environment_id}</option>)}</select></label>
          <label>Task type<select aria-label="Statistics task type" value={form.taskKind} onChange={update("taskKind")}>{Object.entries(TASK_KIND_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          <label className="business-name-filter">Registered task name<input aria-label="Statistics task name" value={form.taskName} onChange={update("taskName")} maxLength={1024} placeholder="Exact task_name; leave blank for all" /></label>
          {form.preset === "month" ? <label>Month<input aria-label="Statistics month" type="month" value={form.month} onChange={update("month")} /></label> : null}
          {form.preset === "custom" ? <><label>Start time (Beijing)<input aria-label="Statistics start time" type="datetime-local" value={form.start} onChange={update("start")} /></label><label>End time (exclusive)<input aria-label="Statistics end time" type="datetime-local" value={form.end} onChange={update("end")} /></label></> : null}
        </div>
        <div className="business-actions"><button className="icon-button primary" type="submit" disabled={pending}><BarChart3 size={16} aria-hidden="true" />{pending ? "Generating…" : "Generate statistics"}</button>{pending ? <button className="icon-button" type="button" onClick={cancel}><X size={15} aria-hidden="true" />Cancel query</button> : null}<span className="muted">Click Generate after changing filters. Results do not refresh automatically.</span></div>
      </form>
      {error ? <p className="business-message business-error" role="alert">{error}</p> : null}
      {notice ? <p className="business-message muted" role="status">{notice}</p> : null}
      {dirty ? <p className="business-message" role="status">Filters changed. Generate statistics to update the results below.</p> : null}
    </section>
    {!data ? <section className="panel business-empty"><BarChart3 size={30} aria-hidden="true" /><h3>{pending ? "Loading task statistics" : "Select a range and generate statistics"}</h3><p className="muted">View task counts, hourly and daily peaks, task names, and execution success rates.</p></section> : <>
      <section className="panel business-result">
        <div className="panel-header"><div><h2>{BASIS_LABELS[data.filters.basis]} · {TASK_KIND_LABELS[data.filters.task_kind]}</h2><p className="muted">{beijingTime(data.filters.start_at)} ≤ time &lt; {beijingTime(data.filters.end_at)} · Beijing time (UTC+8)</p><p className="muted">Environment: {data.filters.environment_id || "All"} · Task name: {data.filters.task_name ?? "All"} · Generated {beijingTime(data.generated_at)}{data.cached ? " · Cached result" : ""}</p></div><div className="row-actions business-export"><button className="icon-button" type="button" onClick={() => download("business-statistics.csv", statisticsCsv(data), "text/csv;charset=utf-8")}><Download size={15} aria-hidden="true" />Export CSV</button><button className="icon-button" type="button" onClick={() => download("business-statistics.json", JSON.stringify(data, null, 2), "application/json")}><Download size={15} aria-hidden="true" />JSON</button></div></div>
        {data.coverage.missing_time_records || data.coverage.excluded_unknown_name_records ? <p className="business-message business-error" role="status">Statistics are incomplete. Counts below cover known records only. {data.coverage.missing_time_records ? `${count(data.coverage.missing_time_records)} retained records in the selected environment and task types lack the selected timestamp. Their membership in this time range is unknown. ` : ""}{data.coverage.excluded_unknown_name_records ? `Another ${count(data.coverage.excluded_unknown_name_records)} records in this range have no task name, so their match to the name filter is unknown.` : ""}</p> : null}
        {data.coverage.estimated_time_records ? <p className="business-message" role="status">{count(data.coverage.estimated_time_records)} tasks use their last update time as an estimated finish time. Counts and time placement for these historical tasks are approximate.</p> : null}
        {data.coverage.unknown_name_records ? <p className="business-message" role="status">{count(data.coverage.unknown_name_records)} tasks have no recorded name and are shown separately. Name coverage: {percent(data.coverage.name_coverage_ratio)}.</p> : null}
        <div className="summary-grid business-summary">{[["Total tasks", data.summary.total], ["Succeeded", data.summary.succeeded], ["Failed", data.summary.failed], ["Cancelled", data.summary.cancelled], ["Unfinished", data.summary.unfinished], ["Execution success rate", data.summary.success_denominator ? percent(data.summary.success_rate) : "No decided tasks"]].map(([label, value]) => <div className="summary-card" key={label}><div className="summary-label">{label}</div><div className={`summary-value ${value === "No decided tasks" ? "summary-text" : ""}`}>{typeof value === "number" ? count(value) : value}</div></div>)}</div>
        <p className="muted business-footnote">Execution success rate = Succeeded / (Succeeded + Failed). Cancelled and unfinished tasks are excluded from the denominator. Status is captured at query time; agent and evaluation outcomes use separate definitions.</p>
        <div className="business-peaks">{[["Hourly peak", data.peaks.hourly], ["Daily peak", data.peaks.daily]].map(([label, value]) => <div key={label}><span className="muted">{label}</span><strong>{count(value.count)} <small>tasks</small></strong><span className="muted">{value.start_at ? `${beijingTime(value.start_at)} — ${beijingTime(value.end_at)}${value.partial_period ? " · Partial period" : ""}${value.tied_periods > 1 ? ` · ${value.tied_periods} tied periods` : ""}` : "No tasks or peak period"}</span></div>)}</div>
      </section>
      <section className="panel"><div className="panel-header"><div><h2>Task volume over time</h2><span className="muted">Peaks show task counts per period. Partial first and last periods use the selected time boundaries.</span></div><div className="segmented-control" role="group" aria-label="Statistics chart granularity">{[["hourly", "Hourly"], ["daily", "Daily"]].map(([value, label]) => <button key={value} className={grain === value ? "active" : ""} type="button" onClick={() => setGrain(value)}>{label}</button>)}</div></div><TaskTrend buckets={data[grain]} /></section>
      <section className="panel"><div className="panel-header"><div><h2>Task name distribution</h2><span className="muted">Exact registered names · Top 10 and full name breakdown</span></div></div><div className="business-distribution"><div className="business-bars">{distribution.length ? distribution.map((item, index) => <div className="business-bar" key={index}><div><span title={item.label}>{item.label}</span><strong>{count(item.total)} <small>{percent(item.share)}</small></strong></div><div className="business-bar-track"><i style={{ width: `${(item.share || 0) * 100}%`, background: item.task_name === null ? COLORS.cancelled : COLORS.succeeded }} /></div></div>) : <p className="muted">No tasks in the selected range.</p>}</div><div className="business-name-table"><div className="business-table-actions"><label>Sort <select aria-label="Task name sort order" value={sort} onChange={(event) => { setSort(event.target.value); setPage(0); }}><option value="total">Task count, highest first</option><option value="name">Task name</option></select></label></div><div className="table-scroll"><table><thead><tr><th>task_name</th><th>Count</th><th>Share</th><th>Succeeded / Failed</th><th>Success rate</th></tr></thead><tbody>{rows.slice(currentPage * 20, (currentPage + 1) * 20).map((item) => <tr key={JSON.stringify([item.task_name])}><td><span className="business-task-name" title={taskLabel(item.task_name)}>{taskLabel(item.task_name)}</span></td><td>{count(item.total)}</td><td>{percent(item.share)}</td><td>{count(item.succeeded)} / {count(item.failed)}</td><td>{percent(item.success_rate)}</td></tr>)}</tbody></table></div><div className="business-pagination"><span className="muted">{count(rows.length)} groups · {currentPage + 1} / {pages}</span><button className="row-action" type="button" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>Previous</button><button className="row-action" type="button" disabled={currentPage + 1 >= pages} onClick={() => setPage(currentPage + 1)}>Next</button></div></div></div></section>
      <p className="muted business-retention">Statistics use task records currently retained in the database. Deleted history cannot be reconstructed. Submitted Batch subtasks are counted when materialized; each Live session counts as one task.</p>
    </>}
  </div>;
}
