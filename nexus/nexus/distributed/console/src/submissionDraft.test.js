import test from "node:test";
import assert from "node:assert/strict";

import * as submissionDraft from "./submissionDraft.js";
import { buildSubmission, changeSubmissionType, defaultSubmissionDraft } from "./submissionDraft.js";


test("builds a minimal config-only single submission from defaults", () => {
  const draft = defaultSubmissionDraft("single", "environment-a");
  draft.runInput = '{"task_name":"identity"}';

  assert.deepEqual(buildSubmission(draft), {
    submission_version: 1,
    idempotency_key: draft.idempotencyKey,
    environment_id: "environment-a",
    code: { mode: "config_only" },
    resources: {
      managers: { count: 1, cpu_cores: 1, memory_gib: 2, slots: 20 },
    },
    run_input: { task_name: "identity" },
  });
});


test("builds ordered host environment entries from KEY=VALUE lines", () => {
  const draft = defaultSubmissionDraft("single", "environment-a");
  draft.hostEnv = "HYTUNER=1\nFOO=value=with=equals\n\n";

  assert.deepEqual(buildSubmission(draft).host_env, [
    { name: "HYTUNER", value: "1" },
    { name: "FOO", value: "value=with=equals" },
  ]);
});


test("rejects malformed host environment lines", () => {
  const draft = defaultSubmissionDraft("single", "environment-a");
  draft.hostEnv = "HYTUNER\n";

  assert.throws(() => buildSubmission(draft), /Host environment line 1 must use KEY=VALUE/);
});


test("rejects host environment on predefined submissions", () => {
  const draft = defaultSubmissionDraft("single", "environment-a");
  draft.pool = "pool-a";
  draft.hostEnv = "HYTUNER=1";

  assert.throws(() => buildSubmission(draft), /Host environment requires dynamic allocation/);
});


test("predefined pool suppresses stale dynamic resource values", () => {
  const draft = defaultSubmissionDraft("live", "environment-a");
  draft.runInput = '{"task_name":"identity"}';
  draft.pool = "interactive";
  draft.managerCpuCores = "4";
  draft.managerMemoryGiB = "8";
  draft.queueTtlSeconds = "600";

  const submission = buildSubmission(draft);
  assert.equal("resources" in submission, false);
  assert.equal(submission.pool, "interactive");
  assert.equal(submission.queue_ttl_seconds, 600);
  assert.deepEqual(submission.live, {});
  assert.equal("execution_ttl_seconds" in submission, false);
});


test("changing submission type preserves the current draft", () => {
  const draft = defaultSubmissionDraft("single", "environment-a");
  draft.managerCpuCores = "1";
  draft.managerMemoryGiB = "2";
  draft.managerCount = "25";
  draft.managerSlots = "20";
  draft.batchManagerCpuCores = "4";
  draft.batchManagerMemoryGiB = "4";
  draft.queueTtlSeconds = "600";

  const changed = changeSubmissionType(draft, "batch");

  assert.deepEqual(changed, { ...draft, type: "batch" });
  assert.notEqual(changed, draft);
});


test("builds a dynamic batch with required max inflight and inferred manager capacity", () => {
  const draft = defaultSubmissionDraft("batch", "environment-a");

  assert.deepEqual(buildSubmission(draft), {
    submission_version: 1,
    idempotency_key: draft.idempotencyKey,
    environment_id: "environment-a",
    code: { mode: "config_only" },
    generation: { mode: "remote_materialize", source: { type: "compose", compose: { parts: [] } } },
    batch: { max_inflight: 20 },
    resources: {
      batch_manager: { cpu_cores: 1, memory_gib: 4 },
      managers: { cpu_cores: 1, memory_gib: 2 },
    },
  });
});


test("batch manager count and slots can be supplied independently", () => {
  const withCount = defaultSubmissionDraft("batch", "environment-a");
  withCount.managerCount = "3";
  assert.deepEqual(buildSubmission(withCount).resources.managers, {
    count: 3,
    cpu_cores: 1,
    memory_gib: 2,
  });

  const withSlots = defaultSubmissionDraft("batch", "environment-a");
  withSlots.managerSlots = "8";
  assert.deepEqual(buildSubmission(withSlots).resources.managers, {
    cpu_cores: 1,
    memory_gib: 2,
    slots: 8,
  });
});


test("batch requires max inflight", () => {
  const draft = defaultSubmissionDraft("batch", "environment-a");
  draft.maxInflight = "";

  assert.throws(() => buildSubmission(draft), /max_inflight must be a positive integer/);
});


test("resource inputs use representable units and reject misaligned values", () => {
  assert.deepEqual(submissionDraft.CPU_CORES_INPUT, { min: 0.001, step: 0.001 });
  assert.deepEqual(submissionDraft.MEMORY_GIB_INPUT, { min: 0.125, step: 0.125 });

  const invalidMemory = defaultSubmissionDraft("single", "environment-a");
  invalidMemory.managerCpuCores = "1";
  invalidMemory.managerMemoryGiB = "0.001";
  assert.throws(() => buildSubmission(invalidMemory), /align to bytes/);

  const invalidCpu = defaultSubmissionDraft("single", "environment-a");
  invalidCpu.managerCpuCores = "0.0015";
  invalidCpu.managerMemoryGiB = "1";
  assert.throws(() => buildSubmission(invalidCpu), /align to millicores/);

  const valid = defaultSubmissionDraft("single", "environment-a");
  valid.managerCpuCores = "0.001";
  valid.managerMemoryGiB = "0.125";
  valid.managerSlots = "7";
  assert.deepEqual(buildSubmission(valid).resources, {
    managers: { count: 1, cpu_cores: 0.001, memory_gib: 0.125, slots: 7 },
  });
});


test("builds image code from a workload image version", () => {
  const draft = defaultSubmissionDraft("single", "environment-a");
  draft.codeMode = "image";
  draft.imageVersion = "user-build-20260802-001";
  draft.runInput = '{"task_name":"identity"}';

  assert.deepEqual(buildSubmission(draft).code, {
    mode: "image",
    image_version: "user-build-20260802-001",
  });
});


test("image code requires a workload image version", () => {
  const draft = defaultSubmissionDraft("single", "environment-a");
  draft.codeMode = "image";
  draft.imageVersion = "   ";

  assert.throws(() => buildSubmission(draft), /image version is required/i);
});
