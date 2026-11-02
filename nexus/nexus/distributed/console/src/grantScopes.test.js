import test from "node:test";
import assert from "node:assert/strict";

import {
  ALL_SUBMISSION_SCOPES,
  DEFAULT_GRANT_SCOPE_PRESET,
  GRANT_SCOPE_PRESETS,
  grantScopesForPreset,
  inferGrantScopePreset,
} from "./grantScopes.js";


test("all submissions includes every scope required by submission workflows", () => {
  assert.equal(DEFAULT_GRANT_SCOPE_PRESET, "all-submissions");
  assert.deepEqual(grantScopesForPreset(DEFAULT_GRANT_SCOPE_PRESET), [
    "submission:single",
    "submission:live",
    "submission:batch",
    "artifact:write",
    "execution:read",
    "execution:command",
    "execution:stop",
  ]);
  assert.deepEqual(ALL_SUBMISSION_SCOPES, grantScopesForPreset(DEFAULT_GRANT_SCOPE_PRESET));
});


test("grant scope presets expose only all submissions and custom", () => {
  assert.deepEqual(Object.keys(GRANT_SCOPE_PRESETS), ["all-submissions", "custom"]);
  assert.equal(inferGrantScopePreset(ALL_SUBMISSION_SCOPES), "all-submissions");
  assert.equal(inferGrantScopePreset(["submission:single"]), "custom");
});
