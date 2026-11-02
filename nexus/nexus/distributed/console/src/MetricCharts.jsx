import { memo, useMemo, useState } from "react";
import { numberValue } from "./platformStatus";
import { nearestTick, sampleEpoch, trendPlot } from "./metricRanges";

const COLORS = ["#3978dd", "#be544c", "#7b8794"];

export const TrendChart = memo(function TrendChart({ series, emptyLabel = "No samples" }) {
  const [hovered, setHovered] = useState(null);
  const plot = useMemo(() => trendPlot(series), [series]);
  if (!plot) return <div className="metric-empty">{emptyLabel}</div>;
  if (plot.error) return <div className="metric-empty">{plot.error}</div>;
  const index = Math.min(hovered ?? plot.last, plot.ticks.length - 1);
  const axisTime = (time) => new Date(time).toLocaleString([], { timeZone: "Asia/Shanghai", hour: "2-digit", minute: "2-digit", ...(plot.end - plot.start > 86400000 ? { month: "short", day: "numeric" } : {}) });
  const detail = plot.records.map((item) => item.detailsByTime.get(plot.ticks[index])).find((item) => item?.bucket_end);
  return <div className="trend-chart">
    <div className="trend-scale muted">Max <strong>{numberValue(plot.maximum)}</strong></div>
    <svg viewBox="0 0 420 142" role="img" aria-label={series.map((item) => item.label || item.metric).join(", ") + " trend"}
      onMouseMove={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const fraction = Math.max(0, Math.min(1, ((event.clientX - rect.left) / rect.width * 420 - 42) / 362));
        setHovered(nearestTick(plot.ticks, plot.start + fraction * (plot.end - plot.start)));
      }} onMouseLeave={() => setHovered(null)}>
      <line x1="42" y1="112" x2="404" y2="112" stroke="currentColor" opacity="0.2" />
      <text x="24" y="112" fontSize="10" fill="currentColor">0</text>
      {plot.records.map((item, n) => <g key={item.metric}>
        <path d={item.path} fill="none" stroke={COLORS[n % COLORS.length]} strokeWidth="2" />
        {item.isolated.map((point) => <circle key={point.index} cx={plot.x(point.index)} cy={plot.y(point.value)} r="3" fill={COLORS[n % COLORS.length]} />)}
      </g>)}
      <text x="42" y="136" fontSize="10" fill="currentColor">{axisTime(plot.ticks[0])}</text>
      <text x="404" y="136" textAnchor="end" fontSize="10" fill="currentColor">{axisTime(plot.ticks.at(-1))}</text>
    </svg>
    <div className="trend-legend">{plot.records.map((item, n) => <span key={item.metric}><i style={{ background: COLORS[n % COLORS.length] }} />{item.label || item.metric}: {Number.isFinite(item.pointsByTime.get(plot.ticks[index])) ? numberValue(item.pointsByTime.get(plot.ticks[index])) : "—"} {item.unit || ""}</span>)}</div>
    <div className="muted trend-sample-time">{new Date(plot.ticks[index]).toLocaleString([], { timeZone: "Asia/Shanghai" })}{detail ? ` – ${axisTime(sampleEpoch(detail.bucket_end))}${detail.partial_period ? " · partial bucket" : ""}` : ""}</div>
  </div>;
});
