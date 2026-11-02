import test from "node:test";
import assert from "node:assert/strict";
import { initialStatisticsForm, statisticsQuery, statisticsCsv } from "./businessStatistics.js";

const NOW = new Date("2026-09-14T09:41:37Z");

test("business presets use Beijing month boundaries and explicit UTC times", () => {
  const form = initialStatisticsForm(NOW);
  assert.equal(form.end, "2026-09-14T17:41");
  assert.equal(statisticsQuery(form, NOW).start_at, "2026-09-07T09:41:00.000Z");
  const month = statisticsQuery({ ...form, preset: "month", month: "2026-08" }, NOW);
  assert.equal(month.start_at, "2026-07-31T16:00:00.000Z");
  assert.equal(month.end_at, "2026-08-31T16:00:00.000Z");
  assert.equal(statisticsQuery({ ...form, preset: "month", month: "2026-09" }, NOW).end_at, "2026-09-14T09:41:00.000Z");
});

test("custom ranges are validated and exact registered names are preserved", () => {
  const form = { ...initialStatisticsForm(NOW), taskName: " exact_task_2 " };
  assert.equal(statisticsQuery(form, NOW).task_name, " exact_task_2 ");
  assert.throws(() => statisticsQuery({ ...form, preset: "month", month: "2026-10" }, NOW));
  assert.throws(() => statisticsQuery({ ...form, preset: "custom", start: "2026-02-31T00:00" }, NOW));
  assert.throws(() => statisticsQuery({ ...form, preset: "custom", start: "2026-01-01T00:00" }, NOW));
  assert.throws(() => statisticsQuery({ ...form, preset: "custom", end: "2026-09-15T00:00" }, NOW));
});

test("CSV preserves the generated result metadata and protects spreadsheet formula cells", () => {
  const data = { definition_version: 1, generated_at: "2026-09-14T00:00:00Z", timezone: "Asia/Shanghai", filters: { start_at: "a", end_at: "b" },
    summary: { total: 1, succeeded: 1 }, hourly: [], daily: [], coverage: { count_complete: true }, peaks: {},
    task_names: { items: [{ task_name: '=HYPERLINK("example")', total: 1, succeeded: 1 }] } };
  const csv = statisticsCsv(data);
  assert.ok(csv.startsWith("\ufeff"));
  assert.ok(csv.includes("'=HYPERLINK"));
  assert.ok(csv.includes('"generated_at","2026-09-14T00:00:00Z"'));
  assert.ok(csv.includes('"filters"'));
  assert.equal(data.task_names.items[0].task_name, '=HYPERLINK("example")');
});
