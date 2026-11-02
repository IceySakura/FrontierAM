import test from "node:test";
import assert from "node:assert/strict";
import { METRIC_RANGES, latestBucketLabel, nearestTick, trendPlot } from "./metricRanges.js";

test("monthly plots retain all hours including the final day", () => {
  const start = Date.parse("2026-08-15T00:00:00Z");
  const points = Array.from({ length: 721 }, (_, i) => ({ time: new Date(start + i * 3600000).toISOString(), value: i === 720 ? null : i }));
  const plot = trendPlot([{ metric: "count", granularity_seconds: 3600, points }]);
  assert.equal(plot.ticks.length, 721);
  assert.equal(plot.end - plot.start, 30 * 86400000);
  assert.equal(plot.last, 719);
  assert.equal(plot.x(720), 404);
  assert.equal(nearestTick(plot.ticks, plot.end), 720);
});

test("missing hours break paths and isolated samples remain visible", () => {
  const plot = trendPlot([{ metric: "count", granularity_seconds: 3600, points: [
    { time: "2026-09-14T00:00:00Z", value: 0 }, { time: "2026-09-14T02:00:00Z", value: 4 },
  ] }]);
  assert.equal(plot.ticks.length, 3);
  assert.equal(plot.records[0].path.includes("L"), false);
  assert.equal(plot.records[0].isolated.length, 2);
  assert.equal(latestBucketLabel({ granularity_seconds: 3600, points: [] }), "1-hour bucket");
});

test("only long views reduce polling frequency; source grain remains explicit", () => {
  assert.deepEqual(METRIC_RANGES.map((r) => [r.hours, r.grain, r.interval]), [["1", 60, 60000], ["24", 60, 60000], ["168", 1800, 300000], ["720", 3600, 300000]]);
});
