import test from "node:test";
import assert from "node:assert/strict";

import * as executionPresentation from "./executionPresentation.js";
import {
  executionDisplayStatus,
  executionLogRetryDelay,
  executionLogSourceLabel,
  executionLogStateText,
} from "./executionPresentation.js";


test("displays a batch outcome instead of its parent execution status", () => {
  assert.equal(
    executionDisplayStatus({ status: "failed", outcome: "partial_success" }),
    "partial_success",
  );
});


test("falls back to execution status while a batch outcome is unavailable", () => {
  assert.equal(executionDisplayStatus({ status: "running", outcome: null }), "running");
  assert.equal(executionDisplayStatus({ status: "failed" }), "failed");
});


test("labels normalized log sources and terminal ingest state", () => {
  assert.equal(executionLogSourceLabel("instance_manager"), "Instance Manager");
  assert.equal(executionLogSourceLabel("zhiyan"), "Zhiyan");
  assert.equal(executionLogSourceLabel("none"), "Not started");
  assert.equal(executionLogStateText({ availability: "not_started" }), "Manager has not started");
  assert.equal(
    executionLogStateText({ availability: "ingest_pending" }),
    "Execution finished; Zhiyan logs are still arriving",
  );
});


test("bounds ingest retries to the three documented delays", () => {
  assert.equal(executionLogRetryDelay("ingest_pending", 0), 2000);
  assert.equal(executionLogRetryDelay("ingest_pending", 1), 5000);
  assert.equal(executionLogRetryDelay("ingest_pending", 2), 10000);
  assert.equal(executionLogRetryDelay("ingest_pending", 3), null);
  assert.equal(executionLogRetryDelay("ready", 0), null);
});


test("formats local and Zhiyan records with the same structured line", () => {
  assert.equal(typeof executionPresentation.formatExecutionLogLine, "function");
  assert.equal(
    executionPresentation.formatExecutionLogLine(
      {
        timestamp: "2026-08-28T10:00:00Z",
        level: "ERROR",
        logger: "nexus.worker",
        message: "failed operation",
        exception: { traceback: "Traceback: failed operation\n" },
      },
      () => "2026-08-28 18:00:00",
    ),
    "2026-08-28 18:00:00 | ERROR | nexus.worker | failed operation\nTraceback: failed operation",
  );
});
