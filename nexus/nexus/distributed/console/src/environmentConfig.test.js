import assert from "node:assert/strict";
import test from "node:test";

import * as environmentConfig from "./environmentConfig.js";
import {
  ADVANCED_RESOURCE_LIMIT_FIELDS,
  COMMON_RESOURCE_LIMIT_FIELDS,
  DEFAULT_QUOTA_FIELDS,
  DEFAULT_TIMEOUT_FIELDS,
  RESOURCE_LIMIT_FIELDS,
  configToDraft,
  draftToConfig,
  draftToCompleteConfig,
  emptyConfigDraft,
} from "./environmentConfig.js";


test("formats environment quotas for compact human-readable display", () => {
  assert.equal(typeof environmentConfig.formatQuotaSummary, "function");
  assert.equal(
    environmentConfig.formatQuotaSummary({
      manager_count: 60,
      cpu_millicores: 60000,
      memory_bytes: 68719476736,
      slots: 1200,
      active_batches: 4,
    }),
    "60 managers · 60 cores · 64 GiB · 1200 slots · 4 batches",
  );
  assert.equal(environmentConfig.formatQuotaSummary(null), "-");
  assert.equal(
    environmentConfig.formatQuotaSummary({
      manager_count: 3,
      cpu_millicores: 10000,
      memory_bytes: 21474836480,
    }),
    "3 managers · 10 cores · 20 GiB",
  );
  assert.equal(environmentConfig.formatQuotaSummary({}), "No additional limits");
  assert.equal(
    environmentConfig.formatQuotaSummary({
      manager_count: 1,
      cpu_millicores: 1000,
      memory_bytes: 1024 ** 3,
      slots: 1,
      active_batches: 1,
    }),
    "1 manager · 1 core · 1 GiB · 1 slot · 1 batch",
  );
});


test("formats effective resource limits in their declared display units", () => {
  assert.equal(typeof environmentConfig.formatResourceLimitValue, "function");
  const cpu = RESOURCE_LIMIT_FIELDS.find((field) => field.key === "cpu_millicores");
  const memory = RESOURCE_LIMIT_FIELDS.find((field) => field.key === "memory_bytes");
  assert.equal(environmentConfig.formatResourceLimitValue(cpu, 10000), "10");
  assert.equal(environmentConfig.formatResourceLimitValue(memory, 20 * 1024 ** 3), "20480");
  assert.equal(environmentConfig.formatResourceLimitValue(cpu, "inherited"), "inherited");
});


test("keeps the default environment form focused on common capacity fields", () => {
  assert.deepEqual(COMMON_RESOURCE_LIMIT_FIELDS.map((field) => field.key), [
    "manager_count",
    "cpu_millicores",
    "memory_bytes",
    "slots",
  ]);
  assert.equal(ADVANCED_RESOURCE_LIMIT_FIELDS.some((field) => field.key === "max_items"), true);
  assert.equal(ADVANCED_RESOURCE_LIMIT_FIELDS.some((field) => field.key === "manager_count"), false);
  assert.equal(COMMON_RESOURCE_LIMIT_FIELDS.find((field) => field.key === "cpu_millicores").label, "CPU (cores)");
  assert.equal(COMMON_RESOURCE_LIMIT_FIELDS.find((field) => field.key === "memory_bytes").label, "Memory (MiB)");
});

test("default quota uses the complete self-service capacity contract", () => {
  assert.deepEqual(DEFAULT_QUOTA_FIELDS.map((field) => field.key), [
    "manager_count",
    "cpu_millicores",
    "memory_bytes",
    "slots",
    "active_batches",
  ]);
  const draft = configToDraft({
    manager_count: 2,
    cpu_millicores: 4000,
    memory_bytes: 8589934592,
    slots: 20,
    active_batches: 1,
  }, DEFAULT_QUOTA_FIELDS);
  assert.equal(draft.cpu_millicores, "4");
  assert.equal(draft.memory_bytes, "8192");
  assert.deepEqual(draftToCompleteConfig(draft, DEFAULT_QUOTA_FIELDS), {
    manager_count: 2,
    cpu_millicores: 4000,
    memory_bytes: 8589934592,
    slots: 20,
    active_batches: 1,
  });
  assert.throws(
    () => draftToCompleteConfig({ ...draft, slots: "" }, DEFAULT_QUOTA_FIELDS),
    /slots is required/,
  );
});

test("converts environment capacity display units without changing the API contract", () => {
  const draft = configToDraft({
    cpu_millicores: 125,
    memory_bytes: 512 * 1024 ** 2,
  }, COMMON_RESOURCE_LIMIT_FIELDS);
  assert.equal(draft.cpu_millicores, "0.125");
  assert.equal(draft.memory_bytes, "512");
  assert.deepEqual(draftToConfig(draft, COMMON_RESOURCE_LIMIT_FIELDS), {
    cpu_millicores: 125,
    memory_bytes: 512 * 1024 ** 2,
  });
  assert.throws(
    () => draftToConfig({ ...draft, cpu_millicores: "0.0005" }, COMMON_RESOURCE_LIMIT_FIELDS),
    /cpu_millicores/,
  );
});

test("resource and timeout fields cover the server schema without duplicates", () => {
  assert.equal(new Set(RESOURCE_LIMIT_FIELDS.map((field) => field.key)).size, RESOURCE_LIMIT_FIELDS.length);
  assert.equal(new Set(DEFAULT_TIMEOUT_FIELDS.map((field) => field.key)).size, DEFAULT_TIMEOUT_FIELDS.length);
  assert.ok(RESOURCE_LIMIT_FIELDS.some((field) => field.key === "manager_count" && field.min === 0));
  assert.ok(RESOURCE_LIMIT_FIELDS.some((field) => field.key === "max_workspace_bytes" && field.min === 1));
  assert.ok(DEFAULT_TIMEOUT_FIELDS.every((field) => field.min === 1));
});

test("config drafts preserve exact integers and omit empty fields", () => {
  const draft = configToDraft(
    { manager_count: 0, memory_bytes: 4294967296 },
    RESOURCE_LIMIT_FIELDS,
  );
  assert.equal(draft.manager_count, "0");
  assert.equal(draft.memory_bytes, "4096");
  assert.deepEqual(draftToConfig(draft, RESOURCE_LIMIT_FIELDS), {
    manager_count: 0,
    memory_bytes: 4294967296,
  });
  assert.deepEqual(draftToConfig(emptyConfigDraft(RESOURCE_LIMIT_FIELDS), RESOURCE_LIMIT_FIELDS), {});
});

test("config drafts reject invalid, fractional, negative, and unsafe values", () => {
  const field = RESOURCE_LIMIT_FIELDS.find((item) => item.key === "max_items");
  assert.ok(field);
  for (const value of ["abc", "1.5", "-1", "0", "9007199254740992"]) {
    assert.throws(
      () => draftToConfig({ ...emptyConfigDraft(RESOURCE_LIMIT_FIELDS), max_items: value }, RESOURCE_LIMIT_FIELDS),
      /max_items/,
    );
  }
});
