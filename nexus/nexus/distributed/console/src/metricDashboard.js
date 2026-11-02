export const HISTORY_PANELS = [
  { id: "terminal", panel: "terminal", title: "Platform terminal outcomes", metrics: ["platform_terminal_results", "platform_terminal_failures", "platform_terminal_unknown"] },
  { id: "gateway", panel: "gateway", title: "API requests", metrics: ["gateway_requests", "gateway_server_errors", "gateway_duration_p95"] },
  { id: "preparation", panel: "preparation", title: "Preparation", metrics: ["stage_operations", "stage_failures", "stage_duration_p95"] },
  { id: "delivery", panel: "delivery", title: "Data delivery", metrics: ["persistence_operations", "persistence_failures", "persistence_duration_p95"] },
  { id: "database", panel: "database", title: "Database", metrics: ["database_operations", "database_errors", "database_sql_duration_p95"] },
  { id: "control", panel: "control", title: "Terminal reports", metrics: ["terminal_operations", "terminal_failures", "terminal_report_p95"] },
];

export const DETAIL_GROUPS = [
  ["preparation", "Preparation and startup"],
  ["execution", "Execution and scheduling"],
  ["delivery", "Results and traces"],
  ["database", "Database"],
  ["resources", "Resources and cleanup"],
  ["monitoring", "Monitoring health"],
];

export const CORE_REQUEST_IDS = new Set([
  "gateway", "terminal", "delivery",
  "metric:queue_depth:schedulable", "metric:queue_depth:preparing",
  "metric:slots:available", "metric:slots:total",
]);

export function visibleMetricRequests(cards, expanded) {
  return [...cards.filter((card) => CORE_REQUEST_IDS.has(card.id)), ...cards.filter((card) => !CORE_REQUEST_IDS.has(card.id) && expanded.has(card.group))];
}

function metricGroup(name, view) {
  if (/^database_/.test(name)) return "database";
  if (/^(observer_|snapshot_|metrics_|process_|event_loop_)/.test(name)) return "monitoring";
  if (/^(persistence_|data_integrity_)/.test(name)) return "delivery";
  if (name === "objects_by_state") return view?.filters?.category === "builds" ? "preparation" : view?.filters?.category === "uploads" ? "delivery" : "resources";
  if (/^(resource_|slots$|manager_gap|manager_heartbeat_age|object_oldest_age)/.test(name)) return "resources";
  if (/^(stage_|operation_|instance_exits)/.test(name)) return /release|provisioning/.test(view?.id || "") ? "resources" : "preparation";
  return "execution";
}

export function metricDashboardCards(catalog) {
  const covered = new Set(HISTORY_PANELS.flatMap((panel) => panel.metrics));
  const cards = HISTORY_PANELS.map((panel) => ({ ...panel, group: { terminal: "execution", gateway: "execution", preparation: "preparation", delivery: "delivery", database: "database", control: "execution" }[panel.panel], summary: true }));
  for (const [panel, stages] of [
    ["preparation", ["dependency_build", "task_generation", "materialization", "manager_start", "runtime_create", "worker_start"]],
    ["delivery", ["trace_write", "trace_flush", "trace_close", "result_publish", "result_read", "artifact_put", "artifact_get"]],
  ]) {
    const base = HISTORY_PANELS.find((item) => item.panel === panel);
    for (const stage of stages) cards.push({ ...base, id: `${panel}:${stage}`, title: stage.replaceAll("_", " "), filters: { stage }, group: panel, summary: true });
  }
  for (const source of ["control_db", "snapshot_db", "trace_db"]) {
    const base = HISTORY_PANELS.find((item) => item.panel === "database");
    cards.push({ ...base, id: `database:${source}`, title: source.replaceAll("_", " "), filters: { source }, group: "database", summary: true });
  }
  for (const metric of catalog) {
    if (covered.has(metric.name)) continue;
    for (const view of metric.views?.length ? metric.views : [null]) {
      cards.push({ id: `metric:${metric.name}:${view?.id || "all"}`, panel: metric.name, view: view?.id, title: metric.aggregation === "distribution" ? `${metric.label.replace(/ distribution$/, "")} · sample count` : metric.label, subtitle: [view?.label, metric.scope === "global" ? "Global control database" : null].filter(Boolean).join(" · "), unit: metric.aggregation === "distribution" ? "samples" : metric.name === "resource_cpu_millicores" ? "millicores" : metric.unit, aggregation: metric.aggregation, metrics: [metric.name], group: metricGroup(metric.name, view), summary: false });
    }
  }
  return cards;
}

export async function loadMetricDashboard(cards, load, onResult, signal, concurrency = 3, requestTimeoutMs = 5000) {
  const pause = (milliseconds) => new Promise((resolve) => {
    const finish = () => { clearTimeout(timer); signal.removeEventListener("abort", finish); resolve(); };
    const timer = setTimeout(finish, milliseconds);
    signal.addEventListener("abort", finish, { once: true });
  });
  async function read(card, attempts) {
    for (let attempt = 0; attempt < attempts && !signal.aborted; attempt++) {
      const controller = new AbortController();
      const abort = () => controller.abort();
      signal.addEventListener("abort", abort, { once: true });
      const timeout = setTimeout(abort, requestTimeoutMs);
      let retry = false;
      try {
        const data = await load(card, controller.signal);
        if (!signal.aborted) onResult(card, { data, error: null });
        retry = data?.error_code === "history_query_failed" && data?.failed_intervals?.length > 0;
        if (!retry) return false;
      } catch (error) {
        if (signal.aborted) return false;
        retry = error.name === "AbortError" || /^(429|502|503|504)\b/.test(error.message);
        if (!retry || attempt === attempts - 1) { onResult(card, { error }); return retry; }
      } finally {
        clearTimeout(timeout);
        signal.removeEventListener("abort", abort);
      }
      if (attempt === attempts - 1) return retry;
      if (!signal.aborted) await pause(500);
    }
    return false;
  }
  async function run(queue, attempts) {
    let cursor = 0;
    const failed = [];
    async function worker() {
      while (!signal.aborted && cursor < queue.length) {
        const card = queue[cursor++];
        if (await read(card, attempts)) failed.push(card);
      }
    }
    await Promise.all(Array.from({ length: Math.min(concurrency, queue.length) }, worker));
    return failed;
  }
  const failed = await run(cards, 2);
  if (failed.length && !signal.aborted) {
    // Finish healthy cards before retrying failures that outlasted the quick retry.
    await pause(5000);
    if (!signal.aborted) await run(failed, 1);
  }
}
