import test from "node:test";
import assert from "node:assert/strict";

import {
  buildConfigOverrideDeletePayload,
  buildConfigOverridePayload,
  diffConfigValues,
  parseConfigValues,
} from "./systemConfig.js";


test("parses an ordinary configuration object", () => {
  assert.deepEqual(parseConfigValues('{"scheduler_poll_interval":0.25}'), {
    scheduler_poll_interval: 0.25,
  });
});


test("rejects non-object configuration JSON", () => {
  assert.throws(() => parseConfigValues("[]"), /Configuration must be a JSON object/);
});


test("builds a versioned configuration update payload", () => {
  assert.deepEqual(buildConfigOverridePayload({
    role: "scheduler",
    identity: "*",
    expectedVersion: "3",
    values: { scheduler_poll_interval: 0.25 },
  }), {
    role: "scheduler",
    identity: "*",
    expected_version: 3,
    values: { scheduler_poll_interval: 0.25 },
    secret: false,
  });
});


test("builds a secret update payload without changing its shape", () => {
  assert.equal(buildConfigOverridePayload({
    role: "*",
    identity: "*",
    expectedVersion: 1,
    values: { internal_token: "new-secret" },
    secret: true,
  }).secret, true);
});


test("builds a versioned configuration delete payload", () => {
  assert.deepEqual(buildConfigOverrideDeletePayload({
    role: "scheduler",
    identity: "*",
    expectedVersion: 3,
    secret: false,
  }), {
    role: "scheduler",
    identity: "*",
    expected_version: 3,
    secret: false,
  });
});


test("reports added, removed, and changed configuration fields", () => {
  assert.deepEqual(diffConfigValues(
    {
      unchanged: "same",
      nested: { old: 1, removed: true },
      array: [1, 2],
    },
    {
      unchanged: "same",
      nested: { old: 2, added: "new" },
      array: [1, 3],
    },
  ), [
    { path: "array", kind: "changed", before: [1, 2], after: [1, 3] },
    { path: "nested.added", kind: "added", before: undefined, after: "new" },
    { path: "nested.old", kind: "changed", before: 1, after: 2 },
    { path: "nested.removed", kind: "removed", before: true, after: undefined },
  ]);
});


test("returns no changes for equivalent configuration objects", () => {
  assert.deepEqual(diffConfigValues({ nested: { value: 1 }, list: ["a"] }, { list: ["a"], nested: { value: 1 } }), []);
});
