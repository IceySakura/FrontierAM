import test from "node:test";
import assert from "node:assert/strict";

import { loadMetricDashboard, metricDashboardCards } from "./metricDashboard.js";

test("all catalog metrics and distinct quantities remain visible without selecting a metric", () => {
  const catalog = [
    { name: "gateway_requests", label: "Requests" },
    { name: "slots", label: "Slots", views: [{ id: "available", label: "Available" }, { id: "reserved", label: "Reserved" }, { id: "total", label: "Total" }] },
    { name: "metrics_exports", label: "Exports" },
  ];
  const cards = metricDashboardCards(catalog);
  const visible = new Set(cards.flatMap((card) => card.metrics));
  assert.ok(catalog.every((metric) => visible.has(metric.name)));
  assert.deepEqual(cards.filter((card) => card.panel === "slots").map((card) => card.view), ["available", "reserved", "total"]);
  assert.equal(new Set(cards.map((card) => card.id)).size, cards.length);
});

test("dashboard bounds concurrency and a failed metric does not suppress other metrics", async () => {
  const cards = Array.from({ length: 12 }, (_, id) => ({ id }));
  let pending = 0, maximum = 0;
  const results = [];
  await loadMetricDashboard(cards, async (card) => {
    maximum = Math.max(maximum, ++pending);
    await new Promise((resolve) => setTimeout(resolve, 2));
    pending--;
    if (card.id === 2) throw new Error("upstream unavailable");
    return card.id;
  }, (card, result) => results.push([card.id, result]), new AbortController().signal);
  assert.equal(maximum, 3);
  assert.equal(results.length, 12);
  assert.equal(results.filter(([, value]) => value.error).length, 1);
  assert.equal(results.filter(([, value]) => value.data !== undefined).length, 11);
});

test("leaving the visible dashboard aborts active requests and does not drain the waiting list", async () => {
  const controller = new AbortController();
  let started = 0, aborted = 0, published = 0;
  const loading = loadMetricDashboard(Array.from({ length: 30 }, (_, id) => ({ id })), async (_, signal) => {
    started++;
    await new Promise((_, reject) => signal.addEventListener("abort", () => { aborted++; reject(new Error("aborted")); }, { once: true }));
  }, () => published++, controller.signal);
  controller.abort();
  await loading;
  assert.equal(started, 3);
  assert.equal(aborted, 3);
  assert.equal(published, 0);
});

test("a transient query failure is retried once and publishes its successful response", async () => {
  let attempts = 0;
  const results = [];
  await loadMetricDashboard([{ id: "transient" }], async () => {
    if (++attempts === 1) throw new Error("503 Service Unavailable");
    return { value: 7 };
  }, (_, result) => results.push(result), new AbortController().signal);
  assert.equal(attempts, 2);
  assert.deepEqual(results, [{ data: { value: 7 }, error: null }]);
});

test("leaving the dashboard during retry delay prevents another request", async () => {
  const controller = new AbortController();
  let attempts = 0, published = 0;
  const loading = loadMetricDashboard([{ id: "transient" }], async () => {
    attempts++;
    throw new Error("503 Service Unavailable");
  }, () => published++, controller.signal);
  await new Promise((resolve) => setTimeout(resolve, 10));
  controller.abort();
  await loading;
  assert.equal(attempts, 1);
  assert.equal(published, 0);
});

test("HTTP 200 with failed history intervals retries while retaining available samples", async () => {
  let calls = 0;
  const partial = { data_status: "partial", error_code: "history_query_failed", failed_intervals: [{ error_code: "history_upstream_http_error" }], completed: 3 };
  const complete = { data_status: "ready", failed_intervals: [], completed: 5 };
  const seen = [];
  await loadMetricDashboard([{ id: "partial" }], async () => ++calls === 1 ? partial : complete,
    (_, value) => seen.push(value), new AbortController().signal);
  assert.equal(calls, 2);
  assert.deepEqual(seen.map((entry) => entry.data), [partial, complete]);
});

test("incomplete sample coverage without source failures is not retried", async () => {
  let calls = 0;
  await loadMetricDashboard([{ id: "coverage" }], async () => {
    calls++;
    return { data_status: "partial", failed_intervals: [] };
  }, () => {}, new AbortController().signal);
  assert.equal(calls, 1);
});

test("a longer interruption recovers after healthy cards have loaded", async () => {
  const order = [], seen = [];
  let failures = 0, lastFailureAt = 0, recoveredAt = 0;
  await loadMetricDashboard([{ id: "slow" }, { id: "healthy" }], async (card) => {
    order.push(card.id);
    if (card.id === "slow" && ++failures <= 2) {
      lastFailureAt = performance.now();
      throw new Error("503 Service Unavailable");
    }
    if (card.id === "slow") recoveredAt = performance.now();
    return { value: card.id };
  }, (card, result) => seen.push([card.id, result]), new AbortController().signal, 1);
  assert.deepEqual(order, ["slow", "slow", "healthy", "slow"]);
  assert.ok(recoveredAt - lastFailureAt >= 4900);
  assert.equal(seen[0][1].error.message, "503 Service Unavailable");
  assert.deepEqual(seen.at(-1), ["slow", { data: { value: "slow" }, error: null }]);
});

test("persistent failed intervals remain visible after bounded recovery", async () => {
  let calls = 0;
  const partial = { data_status: "partial", error_code: "history_query_failed", failed_intervals: [{ error_code: "history_segment_timeout" }], completed: 3 };
  const seen = [];
  await loadMetricDashboard([{ id: "partial" }], async () => { calls++; return partial; },
    (_, result) => seen.push(result), new AbortController().signal);
  assert.equal(calls, 3);
  assert.equal(seen.length, 3);
  assert.equal(seen.at(-1).data, partial);
});

test("leaving during delayed recovery prevents its request", async () => {
  const controller = new AbortController();
  let calls = 0, failed;
  const failureSeen = new Promise((resolve) => { failed = resolve; });
  const loading = loadMetricDashboard([{ id: "transient" }], async () => {
    calls++;
    throw new Error("503 Service Unavailable");
  }, () => failed(), controller.signal);
  await failureSeen;
  await new Promise((resolve) => setImmediate(resolve));
  controller.abort();
  await loading;
  assert.equal(calls, 2);
});
