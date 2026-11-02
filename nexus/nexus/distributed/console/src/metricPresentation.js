import { numberValue } from "./platformStatus.js";

export function metricState(entry) {
  const data = entry?.data;
  if (entry?.error) return data ? "Stale" : "Query failed";
  if (!data) return "Loading";
  if (data.error_code === "history_not_configured") return "Not configured";
  return ({ ready: "Available", no_samples: "No samples", partial: "Incomplete samples", stale: "Stale", unavailable: "Query failed", not_configured: "Not enabled", not_supported: "Not supported" })[data.data_status] || "Not collected";
}

export function metricValue(value, entry, scale = 1, suffix = "") {
  if (Number.isFinite(value)) return numberValue(value, scale, suffix);
  const state = metricState(entry);
  return state === "Available" ? "Insufficient samples" : state;
}

export function failureRateValue(entry) {
  if (entry?.data?.completed === 0) return "No operations";
  return metricValue(entry?.data?.failure_ratio, entry, 0.01, "%");
}

export function ratioPoints(completed = [], failed = []) {
  const failures = new Map(failed.map((point) => [point.time, point.value]));
  return completed.map((point) => {
    const numerator = failures.get(point.time);
    return { ...point, value: Number.isFinite(point.value) && point.value > 0 && Number.isFinite(numerator) ? 100 * numerator / point.value : null };
  });
}

export function totalSamples(points = []) {
  const values = points.map((point) => point.value).filter(Number.isFinite);
  return values.length ? values.reduce((sum, value) => sum + value, 0) : null;
}
