import { memo, useCallback, useEffect, useMemo, useState } from "react";

import { startVisiblePolling } from "./platformStatus";
import { DETAIL_GROUPS, loadMetricDashboard, metricDashboardCards, visibleMetricRequests } from "./metricDashboard";
import { failureRateValue, metricState, metricValue } from "./metricPresentation";
import { TrendChart } from "./MetricCharts";
import CoreMetrics from "./CoreMetrics";
import SnapshotDetails from "./SnapshotDetails";
import { snapshotDomainSummary } from "./snapshotPresentation";
import { grainLabel, METRIC_RANGES, metricRange } from "./metricRanges";

function sampleTime(value) {
  return value ? new Date(`${value}Z`).toLocaleString() : "No samples";
}

const MetricPanel = memo(function MetricPanel({ card, entry, snapshot }) {
  const data = entry?.data;
  const samples = data?.series?.some((series) => series.points.some((point) => Number.isFinite(point.value)));
  const missingBuilds = snapshot?.groups?.builds?.values?.items?.reduce((count, row) => count + (row.unknown_budget || 0), 0) || 0;
  const server = snapshot?.groups?.inventory?.values?.database_server;
  const contextNote = card.panel === "stage_overdue" && card.view === "dependency_build" && missingBuilds ? `${missingBuilds} build records lack a deadline; their overdue state cannot be determined.` : card.panel === "database_server" && server?.data_status === "error" ? "Database server collection failed; retained samples may be stale." : card.panel === "database_server" && server?.data_status === "not_supported" ? "Server counters are not supported by this database." : null;
  const status = !samples && contextNote ? (card.panel === "database_server" ? "Collection unavailable" : "Missing deadlines") : metricState(entry);
  const latest = data?.series?.[0]?.points?.filter((point) => Number.isFinite(point.value)).at(-1);
  const timestampMetric = card.panel.endsWith("_timestamp");
  return <article className={`metric-panel ${card.summary ? "metric-panel-summary" : "metric-panel-single"}`} data-metric-card={card.id}>
    <div className="metric-panel-header"><div><h4>{card.title}</h4>{card.subtitle ? <p className="muted">{card.subtitle}</p> : null}</div><span className="muted" role="status">{status}</span></div>
    {card.summary ? <dl className="metric-totals">
      <div><dt>{card.panel === "terminal" ? "Terminal executions" : "Completed operations"}</dt><dd>{metricValue(data?.completed, entry)}</dd></div>
      <div><dt>{card.panel === "terminal" ? "Known platform failures" : "Failed operations"}</dt><dd>{metricValue(data?.failed, entry)}</dd></div>
      <div><dt>{card.panel === "terminal" ? "Known platform failure rate" : "Failure rate"}</dt><dd>{failureRateValue(entry)}</dd></div>
    </dl> : <div className="metric-latest"><strong>{timestampMetric && latest ? new Date(latest.value * 1000).toLocaleString() : (!samples && contextNote ? status : metricValue(latest?.value, entry))}</strong><span className="muted">{timestampMetric ? "Last success" : card.unit} · {card.aggregation} · {sampleTime(latest?.time)}</span></div>}
    <div className={card.summary ? "metric-charts" : "metric-single-chart"}>{card.metrics.map((metric, index) => {
      const series = data?.series?.find((item) => item.metric === metric) || { metric, points: [] };
      const title = card.summary ? (index === 0 ? (card.panel === "terminal" ? "Terminal executions" : "Completed") : index === 1 ? (card.panel === "terminal" ? "Known platform failures" : "Failed") : card.panel === "terminal" ? "Unattributed outcomes" : "P95 duration") : null;
      const seriesStatus = series.data_status === "not_supported" ? "Not supported" : status;
      return <div key={metric} data-metric={metric}>{title ? <h5>{title}</h5> : null}{status === "Loading" ? <div className="metric-placeholder">Loading…</div> : <TrendChart series={[card.summary ? series : { ...series, unit: card.unit }]} emptyLabel={seriesStatus === "Available" ? "No samples" : seriesStatus} />}</div>;
    })}</div>
    {contextNote ? <p className="muted">{contextNote}</p> : null}
    {data?.failed_intervals?.length ? <p className="muted">Some time intervals could not be read; totals cover available samples.</p> : null}
    {entry?.error ? <p className="muted">Latest query failed{data ? "; showing the previous response" : ""}.</p> : null}
  </article>;
});

export default function MetricHistory({ request, controlPath, environmentId, exportState, active = true, snapshot, expanded, onToggle, onMore }) {
  const [hours, setHours] = useState("1");
  const [cards, setCards] = useState(() => metricDashboardCards([]));
  const [results, setResults] = useState({});
  const [error, setError] = useState(null);
  const [pending, setPending] = useState(false);
  const [queriedAt, setQueriedAt] = useState(null);
  const range = metricRange(hours);
  const load = useCallback(async ({ signal }) => {
    const catalog = await request("GET", controlPath("/v1/system/metrics?catalog=1"), undefined, { signal });
    if (signal.aborted) return;
    const nextCards = metricDashboardCards(catalog.items || []);
    setCards(nextCards);
    await loadMetricDashboard(visibleMetricRequests(nextCards, expanded), async (card, requestSignal) => {
      const query = new URLSearchParams({ panel: card.panel, hours, ...card.filters });
      if (environmentId) query.set("environment_id", environmentId);
      if (card.view) query.set("view", card.view);
      return request("GET", controlPath(`/v1/system/metrics?${query}`), undefined, { signal: requestSignal });
    }, (card, entry) => {
      setResults((current) => ({ ...current, [card.id]: { ...current[card.id], ...entry } }));
      setQueriedAt(new Date().toLocaleString());
    }, signal, 3, Number(hours) > 24 ? 6500 : 5000);
  }, [request, controlPath, environmentId, hours, expanded]);
  useEffect(() => { setResults({}); setError(null); setQueriedAt(null); }, [environmentId, hours]);
  useEffect(() => {
    if (!active) return;
    const poll = startVisiblePolling(load, { interval: range.interval, timeoutMs: range.loadTimeoutMs, fixedCadence: true, onError: setError, onPending: setPending });
    window.addEventListener("refresh-platform-observability", poll.refresh);
    return () => { poll.stop(); window.removeEventListener("refresh-platform-observability", poll.refresh); };
  }, [load, active, range.interval, range.loadTimeoutMs]);
  const visibleCards = visibleMetricRequests(cards, expanded);
  const unavailable = visibleCards.filter((card) => results[card.id]?.error || ["stale", "unavailable"].includes(results[card.id]?.data?.data_status) || results[card.id]?.data?.failed_intervals?.length).length;
  const checked = visibleCards.filter((card) => results[card.id]).length;
  const windowData = results.gateway?.data || Object.values(results).find((entry) => entry.data)?.data;
  const groups = useMemo(() => DETAIL_GROUPS.map(([id, title]) => ({ id, title, cards: cards.filter((card) => card.group === id) })), [cards]);
  return <section className="panel" id="platform-trends">
    <div className="panel-header"><div><h2>Metrics · core trends</h2><span className="muted">Six core charts · {grainLabel(range.grain)} buckets · refreshes every {range.interval / 60000} minute{range.interval > 60000 ? "s" : ""}</span></div>
      <label className="history-range">Time range <select aria-label="History range" value={hours} onChange={(event) => setHours(event.target.value)}>{METRIC_RANGES.map((option) => <option key={option.hours} value={option.hours}>{option.label}</option>)}</select></label>
    </div>
    <p className="muted" role="status">{error ? "Metric catalog query failed; retained data may be stale." : `${checked} / ${visibleCards.length} visible queries checked${active && pending ? " · refreshing" : ""}${unavailable ? ` · ${unavailable} have failed reads or retained data` : ""}`}. Last update: {queriedAt || "Not collected"}.</p>
    <CoreMetrics results={results} onMore={onMore} />
    <p className="muted">Charts cover received samples; unobserved intervals remain empty. P95 uses each bucket’s merged distribution; rates with no operations are not calculated.</p>
    {Number(hours) > 24 ? <p className="muted">Complete {grainLabel(range.grain)} buckets · {sampleTime(windowData?.window_start)} — {sampleTime(windowData?.window_end)}. Historical availability follows source retention; selecting a longer range does not create missing history.</p> : null}
    <div className="metric-details">{groups.map((group) => <details key={group.id} id={`metric-detail-${group.id}`} open={expanded.has(group.id)} onToggle={(event) => onToggle(group.id, event.currentTarget.open)}>
      <summary><span>{group.title}</span><span className="muted">{[snapshotDomainSummary(snapshot, group.id), `${group.cards.length} metric panels`].filter(Boolean).join(" · ")}</span></summary>
      {expanded.has(group.id) ? <div className="metric-detail-body">
        <SnapshotDetails snapshot={snapshot} domain={group.id} environmentId={environmentId} />
        {group.id === "monitoring" ? <p className="muted">Exporter: {exportState?.enabled ? (exportState.available ? "Initialized" : "Unavailable") : "Disabled"}. Last successful export: {exportState?.last_export_success_at ? new Date(exportState.last_export_success_at * 1000).toLocaleString() : "No confirmed export"}.</p> : null}
        <div className="metric-detail-panels">{group.cards.map((card) => <MetricPanel key={card.id} card={card} entry={results[card.id]} snapshot={snapshot} />)}</div>
      </div> : null}
    </details>)}</div>
  </section>;
}
