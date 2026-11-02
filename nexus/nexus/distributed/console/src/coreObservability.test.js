import test from "node:test";
import assert from "node:assert/strict";
import { metricDashboardCards, visibleMetricRequests } from "./metricDashboard.js";
import { failureRateValue, metricState, metricValue, ratioPoints, totalSamples } from "./metricPresentation.js";
import { snapshotCore, snapshotTimingIssues } from "./snapshotPresentation.js";

test("core dashboard loads seven unique queries and only expanded domains add work", () => {
  const catalog = [
    { name: "slots", views: [{ id: "available" }, { id: "total" }, { id: "reserved" }] },
    { name: "queue_depth", views: [{ id: "schedulable" }, { id: "preparing" }, { id: "retry_wait" }] },
    { name: "database_pool", views: [{ id: "used" }] },
  ];
  const cards = metricDashboardCards(catalog);
  assert.equal(visibleMetricRequests(cards, new Set()).length, 7);
  const opened = visibleMetricRequests(cards, new Set(["database", "delivery"]));
  assert.ok(opened.some((card) => card.panel === "database_pool"));
  assert.ok(opened.some((card) => card.filters?.stage === "trace_write"));
  assert.ok(!opened.some((card) => card.filters?.stage === "worker_start"));
  assert.equal(visibleMetricRequests(cards, new Set()).length, 7);
});

test("missing samples, disabled sources, failed queries and zero are distinct", () => {
  assert.equal(metricValue(null, { data: { data_status: "no_samples" } }), "No samples");
  assert.equal(metricValue(null, { data: { data_status: "not_configured" } }), "Not enabled");
  assert.equal(metricValue(null, { error: new Error("failed") }), "Query failed");
  assert.equal(metricState({ data: { data_status: "stale" } }), "Stale");
  assert.equal(metricValue(0, { data: { data_status: "ready" } }), "0");
  assert.equal(failureRateValue({ data: { data_status: "ready", completed: 0, failure_ratio: null } }), "No operations");
  assert.equal(totalSamples([]), null);
  assert.equal(totalSamples([{ value: 0 }]), 0);
});

test("rate trends require matching numerator samples and nonzero denominators", () => {
  assert.deepEqual(ratioPoints([{ time: "a", value: 10 }, { time: "b", value: 0 }, { time: "c", value: 3 }], [{ time: "a", value: 1 }, { time: "b", value: 0 }]), [{ time: "a", value: 10 }, { time: "b", value: null }, { time: "c", value: null }]);
});

test("snapshot summary distinguishes missing snapshots from confirmed empty queues", () => {
  assert.equal(snapshotCore(null).running, null);
  const snapshot = { groups: {
    queues: { data_status: "ready", values: { items: [] } },
    builds: { data_status: "ready", values: { items: [{ count: 5, unknown_budget: 5, release_overdue: 0 }] } },
    allocations: { data_status: "ready", values: { items: [] } },
  } };
  assert.equal(snapshotCore(snapshot).running, 0);
  assert.equal(snapshotCore(snapshot).oldestWait, 0);
  assert.deepEqual(snapshotTimingIssues(snapshot), [{ label: "Build deadlines missing", count: 5, domain: "preparation" }]);
});
