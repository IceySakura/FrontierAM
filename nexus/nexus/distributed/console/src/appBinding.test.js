import assert from "node:assert/strict";
import test from "node:test";

import {
  buildUserAppBindingRequest,
  isValidAppId,
  normalizeAppId,
} from "./appBinding.js";


test("builds a normalized system-default app binding request for one user", () => {
  assert.deepEqual(buildUserAppBindingRequest("tai/default/user 1", "  MCP-App.01  "), {
    method: "PUT",
    path: "/v1/access/users/tai%2Fdefault%2Fuser%201/app-binding",
    body: { app_id: "mcp-app.01" },
  });
});


test("builds an explicit unbind request without a body", () => {
  assert.deepEqual(buildUserAppBindingRequest("principal-1", null), {
    method: "DELETE",
    path: "/v1/access/users/principal-1/app-binding",
    body: undefined,
  });
});


test("validates app ids before sending them", () => {
  assert.equal(normalizeAppId("  MCP_App-1  "), "mcp_app-1");
  assert.equal(isValidAppId("mcp_app-1"), true);
  assert.equal(isValidAppId("invalid app"), false);
  assert.throws(() => buildUserAppBindingRequest("principal-1", "invalid app"), /app_id/);
});
