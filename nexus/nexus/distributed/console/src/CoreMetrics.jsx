import { memo, useMemo } from "react";
import { TrendChart } from "./MetricCharts";
import { failureRateValue, metricState, metricValue, ratioPoints, totalSamples } from "./metricPresentation";
import { latestBucketLabel } from "./metricRanges";

const CARDS = [
  { id: "api-volume", title: "API traffic", domain: "execution", entry: "gateway", metrics: ["gateway_requests", "gateway_server_errors"], labels: ["Requests", "5xx"] },
  { id: "api-latency", title: "API latency · P95", domain: "execution", entry: "gateway", metrics: ["gateway_duration_p95"], labels: ["P95"] },
  { id: "terminal-volume", title: "Task terminal outcomes", domain: "execution", entry: "terminal", metrics: ["platform_terminal_results", "platform_terminal_failures", "platform_terminal_unknown"], labels: ["Terminal", "Platform failures", "Unattributed"] },
  { id: "queue", title: "Scheduling queue", domain: "execution", entry: "metric:queue_depth:schedulable", second: "metric:queue_depth:preparing", labels: ["Schedulable", "Preparing"] },
  { id: "slots", title: "Execution slots", domain: "resources", entry: "metric:slots:available", second: "metric:slots:total", labels: ["Available", "Total"] },
  { id: "delivery-rate", title: "Delivery operation failure rate", domain: "delivery", entry: "delivery", labels: ["Failure rate"] },
];
const latest = (series) => series?.points?.filter((point) => Number.isFinite(point.value)).at(-1)?.value;

const CoreCard = memo(function CoreCard({ card, entry, second, onMore }) {
  const data = entry?.data;
  const series = useMemo(() => {
    if (card.id === "delivery-rate") return [{ metric: "delivery_failure_rate", label: card.labels[0], unit: "%", granularity_seconds: data?.granularity_seconds, points: ratioPoints(data?.series?.[0]?.points, data?.series?.[1]?.points) }];
    if (card.second) return [entry, second].map((item, index) => ({ ...item?.data?.series?.[0], metric: `${card.id}-${index}`, label: card.labels[index] }));
    return card.metrics.map((metric, index) => ({ metric, ...data?.series?.find((item) => item.metric === metric), label: card.labels[index] }));
  }, [card, data, entry, second]);
  let value, note;
  if (card.id === "api-latency") { value = series[0]?.data_status === "not_supported" ? "Not supported" : metricValue(latest(series[0]), entry, 1, " s"); note = `Latest ${latestBucketLabel(series[0], data?.granularity_seconds)} P95`; }
  else if (card.id === "delivery-rate") { value = failureRateValue(entry); note = `Failed operations: ${metricValue(data?.failed, entry)} / ${metricValue(data?.completed, entry)}`; }
  else if (card.second) { const left = metricValue(latest(series[0]), entry), right = metricValue(latest(series[1]), second); value = left === right && !Number.isFinite(latest(series[0])) ? left : `${left} / ${right}`; note = `${card.labels[0]} / ${card.labels[1]} · latest ${latestBucketLabel(series[0], data?.granularity_seconds)}`; }
  else { value = metricValue(data?.completed, entry); note = card.id === "api-volume" ? `5xx rate: ${failureRateValue(entry)}` : `Known platform failures: ${metricValue(data?.failed, entry)} · unattributed: ${metricValue(totalSamples(data?.series?.[2]?.points), entry)}`; }
  const states = [metricState(entry), ...(card.second ? [metricState(second)] : [])];
  const state = states.every((item) => item === states[0]) ? states[0] : "Partial data";
  return <article className="metric-panel core-metric" data-core-metric={card.id}>
    <div className="metric-panel-header"><h3>{card.title}</h3><button className="row-action" type="button" onClick={() => onMore(card.domain)}>Details</button></div>
    <div className="core-metric-value">{value}</div><p className="muted core-metric-note">{note}</p>
    <TrendChart series={series} emptyLabel={state === "Available" ? "No operations in these samples" : state} />
    <div className="muted" role="status">{state}</div>
  </article>;
});

export default function CoreMetrics({ results, onMore }) {
  return <div className="core-metric-grid">{CARDS.map((card) => <CoreCard key={card.id} card={card} entry={results[card.entry]} second={card.second ? results[card.second] : undefined} onMore={onMore} />)}</div>;
}
