import assert from "node:assert/strict";
import test from "node:test";

import {
  buildRunTraceBackendGrantPayload,
  mergeRunTraceBackendGrantDraft,
  runTraceBackendAccessRows,
} from "./runTraceBackends.js";

test("mergeRunTraceBackendGrantDraft combines the safe catalog with persisted grants", () => {
  assert.deepEqual(
    mergeRunTraceBackendGrantDraft(
      [
        { name: "archive", kind: "cos", user_selectable: true },
        { name: "primary", kind: "tdsql", user_selectable: true },
        { name: "hidden", kind: "custom", user_selectable: false },
        { name: "system", kind: "custom", user_selectable: true, system_managed: true },
      ],
      [
        {
          backend_name: "primary",
          enabled: true,
          default_order: 0,
          updated_at: "2026-08-10T12:00:00",
        },
        { backend_name: "archive", enabled: false, default_order: null },
      ],
    ),
    [
      { backend_name: "archive", enabled: false, default_order: null },
      {
        backend_name: "primary",
        enabled: true,
        default_order: 0,
        updated_at: "2026-08-10T12:00:00",
      },
    ],
  );
});

test("mergeRunTraceBackendGrantDraft ignores grants loaded for another principal", () => {
  assert.deepEqual(
    mergeRunTraceBackendGrantDraft(
      [{ name: "primary", kind: "tdsql", user_selectable: true }],
      [{ backend_name: "primary", enabled: true, default_order: 0 }],
      "principal-a",
      "principal-b",
    ),
    [{ backend_name: "primary", enabled: false, default_order: null }],
  );
});

test("buildRunTraceBackendGrantPayload preserves grants and validates continuous defaults", () => {
  assert.deepEqual(
    buildRunTraceBackendGrantPayload([
      { backend_name: "archive", enabled: true, default_order: 1 },
      { backend_name: "primary", enabled: true, default_order: 0 },
      { backend_name: "unused", enabled: false, default_order: null },
    ]),
    {
      grants: [
        { backend_name: "archive", enabled: true, default_order: 1 },
        { backend_name: "primary", enabled: true, default_order: 0 },
      ],
    },
  );

  assert.throws(
    () => buildRunTraceBackendGrantPayload([
      { backend_name: "archive", enabled: true, default_order: 1 },
    ]),
    /continuous from zero/,
  );
  assert.throws(
    () => buildRunTraceBackendGrantPayload([
      { backend_name: "archive", enabled: false, default_order: 0 },
    ]),
    /disabled backend cannot be a default/,
  );
});

test("runTraceBackendAccessRows exposes only API-declared claimable backends as available", () => {
  const rows = runTraceBackendAccessRows({
    items: [{ name: "primary", kind: "tdsql", default_order: 0 }],
    claimable_sinks: [
      { name: "archive", kind: "cos" },
      { name: "primary", kind: "tdsql" },
    ],
    required_sinks: [{ name: "artifact", kind: "artifact" }],
  });

  assert.deepEqual(
    rows.map((row) => [row.name, row.access]),
    [
      ["primary", "granted"],
      ["archive", "available"],
      ["artifact", "required"],
    ],
  );
});
