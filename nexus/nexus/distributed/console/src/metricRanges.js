export const METRIC_RANGES = [
  { hours: "1", label: "Last hour", grain: 60, interval: 60000, loadTimeoutMs: 60000 },
  { hours: "24", label: "Last 24 hours", grain: 60, interval: 60000, loadTimeoutMs: 60000 },
  { hours: "168", label: "Last 7 days", grain: 1800, interval: 300000, loadTimeoutMs: 60000 },
  { hours: "720", label: "Last 30 days", grain: 3600, interval: 300000, loadTimeoutMs: 180000 },
];

export const metricRange = (hours) => METRIC_RANGES.find((range) => range.hours === String(hours)) || METRIC_RANGES[0];
export const sampleEpoch = (value) => Date.parse(/Z$|[+-]\d\d:\d\d$/.test(value) ? value : `${value}Z`);
export const grainLabel = (seconds = 60) => seconds >= 3600 ? `${seconds / 3600}-hour` : `${seconds / 60}-minute`;

export function latestBucketLabel(series, fallback = 60) {
  const point = series?.points?.filter((item) => Number.isFinite(item.value)).at(-1);
  const seconds = point?.bucket_end ? (sampleEpoch(point.bucket_end) - sampleEpoch(point.time)) / 1000 : series?.granularity_seconds || fallback;
  return `${point?.partial_period ? "partial " : ""}${grainLabel(seconds)} bucket`;
}

export function nearestTick(ticks, stamp) {
  let left = 0, right = ticks.length - 1;
  while (left < right) {
    const middle = Math.floor((left + right) / 2);
    if (ticks[middle] < stamp) left = middle + 1; else right = middle;
  }
  return left > 0 && stamp - ticks[left - 1] < ticks[left] - stamp ? left - 1 : left;
}

export function trendPlot(series) {
  const records = series.map((item) => ({ ...item,
    pointsByTime: new Map((item.points || []).map((point) => [sampleEpoch(point.time), point.value])),
    detailsByTime: new Map((item.points || []).map((point) => [sampleEpoch(point.time), point])),
  }));
  const times = records.flatMap((item) => [...item.pointsByTime.keys()]).filter(Number.isFinite);
  const values = records.flatMap((item) => [...item.pointsByTime.values()]).filter(Number.isFinite);
  if (!times.length || !values.length) return null;
  const start = Math.min(...times), end = Math.max(...times);
  const step = Math.min(...series.map((item) => item.granularity_seconds || 60)) * 1000;
  if (!Number.isFinite(step) || step <= 0 || (end - start) / step > 2000) return { error: "Unsupported sample interval" };
  const allTimes = new Set(times);
  for (let time = Math.ceil(start / step) * step; time <= end; time += step) allTimes.add(time);
  const ticks = [...allTimes].sort((a, b) => a - b);
  const maximum = Math.max(...values, 1e-9);
  const x = (index) => 42 + (ticks[index] - start) * 362 / Math.max(1, end - start);
  const y = (value) => 112 - value * 94 / maximum;
  for (const item of records) {
    let drawing = false;
    item.path = ticks.map((time, index) => {
      const value = item.pointsByTime.get(time);
      if (!Number.isFinite(value)) { drawing = false; return ""; }
      const command = drawing ? "L" : "M";
      drawing = true;
      return `${command}${x(index)},${y(value)}`;
    }).join(" ");
    item.isolated = ticks.flatMap((time, index) => Number.isFinite(item.pointsByTime.get(time))
      && !Number.isFinite(item.pointsByTime.get(ticks[index - 1])) && !Number.isFinite(item.pointsByTime.get(ticks[index + 1]))
      ? [{ index, value: item.pointsByTime.get(time) }] : []);
  }
  const last = ticks.findLastIndex((time) => records.some((item) => Number.isFinite(item.pointsByTime.get(time))));
  return { records, ticks, maximum, x, y, last, start, end };
}
