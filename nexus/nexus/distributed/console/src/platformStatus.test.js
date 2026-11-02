import test from "node:test";
import assert from "node:assert/strict";
import { mergeExecutionSummary, numberValue, snapshotRows, startVisiblePolling } from "./platformStatus.js";
import { metricRange } from "./metricRanges.js";
const settled = () => new Promise((resolve) => setImmediate(resolve));
function controls() {
  const listeners = new Map(), timers = new Map();
  let next = 0;
  const visibility = {
    hidden: false,
    addEventListener: (name, callback) => listeners.set(name, callback),
    removeEventListener: (name) => listeners.delete(name),
  };
  return {
    visibility, timers,
    setTimer: (callback, milliseconds) => { const id = ++next; timers.set(id, { callback, milliseconds }); return id; },
    clearTimer: (id) => timers.delete(id),
    hide(value) { visibility.hidden = value; listeners.get("visibilitychange")?.(); },
  };
}
test("unknown counts and missing snapshots do not turn into zero", () => {
  assert.equal(numberValue(null), "Unknown");
  assert.equal(numberValue(0), "0");
  assert.equal(snapshotRows({ data_status: "unavailable", values: { items: [] } }), null);
  assert.deepEqual(snapshotRows({ data_status: "ready", values: { items: [] } }), []);
});
test("polling pauses while hidden and never overlaps a pending request", async () => {
  const clock = controls();
  let calls = 0, resolve;
  const signals = [];
  const poll = startVisiblePolling(({ signal }) => {
    calls += 1; signals.push(signal);
    return new Promise((done) => { resolve = done; });
  }, { interval: 2000, ...clock });
  assert.equal(calls, 1);
  await poll.refresh();
  assert.equal(calls, 1);
  clock.hide(true);
  assert.equal(signals[0].aborted, true);
  resolve(); await settled();
  await poll.refresh();
  assert.equal(calls, 1);
  clock.hide(false);
  assert.equal(calls, 2);
  poll.stop();
  assert.equal(signals[1].aborted, true);
  resolve(); await settled();
  assert.equal(clock.timers.size, 0);
});
test("a slow history partition does not delay live polling", async () => {
  const liveClock = controls(), slowClock = controls();
  let liveCalls = 0, finishSlow;
  const slow = startVisiblePolling(() => new Promise((resolve) => { finishSlow = resolve; }), { interval: 60000, ...slowClock });
  const live = startVisiblePolling(async () => { liveCalls += 1; }, { interval: 2000, ...liveClock });
  await settled(); await live.refresh();
  assert.equal(liveCalls, 2);
  live.stop(); slow.stop(); finishSlow(); await settled();
});
test("monthly details can finish after a minute while their total wait stays bounded", async () => {
  const clock = controls(), range = metricRange("720");
  let signal, finish;
  const poll = startVisiblePolling((request) => {
    signal = request.signal;
    return new Promise((resolve) => { finish = resolve; });
  }, { interval: range.interval, timeoutMs: range.loadTimeoutMs, ...clock });
  for (const timer of [...clock.timers.values()]) if (timer.milliseconds <= 90000) timer.callback();
  assert.equal(signal.aborted, false);
  const deadline = [...clock.timers.values()][0];
  assert.ok(deadline.milliseconds < range.interval);
  deadline.callback();
  assert.equal(signal.aborted, true);
  finish(); await settled(); poll.stop();
});
test("status merges retain details only for the same assignment", () => {
  const previous = { execution_id: "one", spec: { input: 1 }, current_assignment: { assignment_id: "a", runtime_info: { endpoint: "private" } } };
  const same = mergeExecutionSummary(previous, { execution_id: "one", status: "running", current_assignment: { assignment_id: "a", status: "active" } });
  assert.deepEqual(same.spec, previous.spec);
  assert.deepEqual(same.current_assignment.runtime_info, previous.current_assignment.runtime_info);
  const retry = mergeExecutionSummary(previous, { execution_id: "one", current_assignment: { assignment_id: "b" } });
  assert.equal(retry.current_assignment.runtime_info, undefined);
});

test("returning to a visible tab during abort resumes after the pending request settles", async () => {
  const clock = controls(); let calls = 0, finish;
  const poll = startVisiblePolling(() => { calls++; return new Promise(resolve => { finish = resolve; }); }, { interval: 2147483647, ...clock });
  clock.hide(true); clock.hide(false);
  assert.equal(calls, 1);
  finish(); await settled();
  assert.equal(calls, 2);
  poll.stop(); finish(); await settled();
});
