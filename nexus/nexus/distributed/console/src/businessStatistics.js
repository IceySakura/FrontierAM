const DAY = 86400000;
const OFFSET = 8 * 3600000;

export const STATUS_LABELS = { succeeded: "Succeeded", failed: "Failed", cancelled: "Cancelled", unfinished: "Unfinished" };
export const BASIS_LABELS = { finished: "Completed", created: "Submitted", started: "First started" };
export const TASK_KIND_LABELS = { all: "All tasks", single: "Single", live: "Live", batch: "Batch subtasks" };

export function beijingInput(value) {
  return new Date(new Date(value).getTime() + OFFSET).toISOString().slice(0, 16);
}

export function beijingTime(value) {
  if (!value) return "—";
  return new Date(value).toLocaleString("en-GB", { timeZone: "Asia/Shanghai", hour12: false });
}

export function initialStatisticsForm(now = new Date()) {
  return { preset: "7d", basis: "finished", environment: "", taskKind: "all", taskName: "",
    month: beijingInput(now).slice(0, 7), start: beijingInput(new Date(+now - 7 * DAY)), end: beijingInput(now) };
}

function parseBeijing(value) {
  if (!/^\d{4}-\d\d-\d\dT\d\d:\d\d$/.test(value)) throw new Error("Select a valid Beijing time (UTC+8)");
  const parsed = Date.parse(`${value}:00+08:00`);
  if (!Number.isFinite(parsed) || beijingInput(parsed) !== value) throw new Error("Select a valid Beijing time (UTC+8)");
  return parsed;
}

export function statisticsQuery(form, now = new Date()) {
  const current = Math.floor(+now / 60000) * 60000;
  let start, end = current;
  if (form.preset === "7d" || form.preset === "30d") start = end - Number.parseInt(form.preset, 10) * DAY;
  else if (form.preset === "month") {
    if (!/^\d{4}-(0[1-9]|1[0-2])$/.test(form.month)) throw new Error("Select a valid month");
    const [year, month] = form.month.split("-").map(Number);
    start = Date.UTC(year, month - 1, 1) - OFFSET;
    end = Math.min(Date.UTC(year, month, 1) - OFFSET, current);
  } else if (form.preset === "custom") {
    start = parseBeijing(form.start); end = parseBeijing(form.end);
    if (end > current) throw new Error("The end time cannot be in the future");
  } else throw new Error("Select a time range");
  if (!(end > start) || end - start > 31 * DAY) throw new Error("The time range must be positive and at most 31 days");
  if (!BASIS_LABELS[form.basis] || !TASK_KIND_LABELS[form.taskKind]) throw new Error("Select a valid time basis and task type");
  return { start_at: new Date(start).toISOString(), end_at: new Date(end).toISOString(), basis: form.basis,
    task_kind: form.taskKind, ...(form.environment ? { environment_id: form.environment } : {}),
    ...(form.taskName !== "" ? { task_name: form.taskName } : {}) };
}

export const percent = (value) => Number.isFinite(value) ? `${(value * 100).toLocaleString("en-GB", { maximumFractionDigits: 2 })}%` : "—";
export const taskLabel = (name) => name === null ? "Missing name" : name;

export function statisticsError(error) {
  const message = error.message || String(error);
  if (message.includes("statistics_busy")) return "Statistics are busy. Please try again shortly.";
  if (message.includes("statistics_timeout") || error.name === "AbortError") return "Statistics timed out. Narrow the scope or try again later.";
  if (/statistics_(too_many_task_names|response_too_large)/.test(message)) return "Statistics exceed the result limit. Narrow the environment, task name, or time range.";
  if (message.includes("statistics_disabled")) return "Business statistics are currently disabled.";
  if (/^(400|422)/.test(message)) return "Invalid statistics query. Check the time range and filters.";
  if (/^(401|403)/.test(message)) return "You do not have permission to query this scope.";
  if (/^5\d\d/.test(message)) return "Statistics could not be loaded. Previous results are retained.";
  return message;
}

export function statisticsCsv(data) {
  const columns = ["section", "name", "start_at", "end_at", "total", "succeeded", "failed", "cancelled", "unfinished", "success_rate", "share", "partial_period", "status_counts", "metadata_key", "metadata_value"];
  const rows = [{ section: "summary", start_at: data.filters.start_at, end_at: data.filters.end_at, ...data.summary }];
  rows.push(...data.hourly.map((row) => ({ section: "hourly", ...row })), ...data.daily.map((row) => ({ section: "daily", name: row.date, ...row })),
    ...data.task_names.items.map((row) => ({ section: row.task_name === null ? "unknown_task_name" : "task_name", name: row.task_name, ...row })));
  for (const [key, value] of Object.entries({ definition_version: data.definition_version, generated_at: data.generated_at, timezone: data.timezone, filters: data.filters, coverage: data.coverage, peaks: data.peaks })) {
    rows.push({ section: "metadata", metadata_key: key, metadata_value: typeof value === "object" ? JSON.stringify(value) : value });
  }
  const quote = (value) => {
    let text = value == null ? "" : typeof value === "object" ? JSON.stringify(value) : String(value);
    if (typeof value === "string" && (/^[\s]*[=+\-@]/.test(text) || /^[\t\r\n]/.test(text))) text = `'${text}`;
    return `"${text.replaceAll('"', '""')}"`;
  };
  return "\ufeff" + [columns.map(quote).join(","), ...rows.map((row) => columns.map((key) => quote(row[key])).join(","))].join("\r\n");
}
