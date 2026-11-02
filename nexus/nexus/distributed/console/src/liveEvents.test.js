import assert from "node:assert/strict";
import test from "node:test";

import {
  canSubmitLiveCommand,
  isTerminalExecution,
  liveEventReadMode,
  parseSseEvents,
} from "./liveEvents.js";

test("liveEventReadMode streams active live executions", () => {
  assert.equal(
    liveEventReadMode({ mode: "live", status: "running", route_status: "open" }, false),
    "stream",
  );
  assert.equal(
    liveEventReadMode({ mode: "live", status: "waiting_input", route_status: "open" }, false),
    "stream",
  );
});

test("liveEventReadMode uses history for terminal live executions", () => {
  assert.equal(
    liveEventReadMode({ mode: "live", status: "succeeded", route_status: "closed" }, false),
    "history",
  );
  assert.equal(
    liveEventReadMode({ mode: "live", status: "running", route_status: "closed" }, false),
    "history",
  );
});

test("liveEventReadMode pauses only active streams", () => {
  assert.equal(
    liveEventReadMode({ mode: "live", status: "running", route_status: "open" }, true),
    "paused",
  );
  assert.equal(
    liveEventReadMode({ mode: "live", status: "failed", route_status: "closed" }, true),
    "history",
  );
});

test("canSubmitLiveCommand only allows commandable live executions", () => {
  assert.equal(canSubmitLiveCommand({ mode: "live", status: "running", route_status: "open" }), true);
  assert.equal(canSubmitLiveCommand({ mode: "live", status: "waiting_input", route_status: "open" }), true);
  assert.equal(canSubmitLiveCommand({ mode: "live", status: "succeeded", route_status: "closed" }), false);
  assert.equal(canSubmitLiveCommand({ mode: "live", status: "running", route_status: "closed" }), false);
  assert.equal(canSubmitLiveCommand({ mode: "job", status: "running", route_status: "open" }), false);
});

test("isTerminalExecution includes route closed", () => {
  assert.equal(isTerminalExecution({ status: "running", route_status: "closed" }), true);
  assert.equal(isTerminalExecution({ status: "lost", route_status: "open" }), true);
  assert.equal(isTerminalExecution({ status: "running", route_status: "open" }), false);
});

test("parseSseEvents parses ids, types, comments, and multiline data", () => {
  const events = parseSseEvents([
    ": keep-alive",
    "",
    "id: 41",
    "event: trace.event",
    'data: {"message":"hello"}',
    "",
    "id: 42",
    "event: run.status",
    'data: {"status":"closed"',
    'data: ,"ok":true}',
    "",
  ].join("\n"));

  assert.deepEqual(events, [
    {
      id: "41",
      type: "trace.event",
      created_at: events[0].created_at,
      payload: { message: "hello" },
    },
    {
      id: "42",
      type: "run.status",
      created_at: events[1].created_at,
      payload: { status: "closed", ok: true },
    },
  ]);
});
