import assert from "node:assert/strict";
import test from "node:test";

import {
  accountAction,
  dedupeTraceDatabaseAccounts,
  eligibleTraceDatabaseEnvironmentId,
} from "./traceDatabaseAccess.js";


test("dedupes trace database accounts by backend and prefers a created account", () => {
  const accounts = dedupeTraceDatabaseAccounts([
    { backend_name: "tdsql", status: "not_created", username: null },
    { backend_name: "archive", status: "ready", username: "archive-user" },
    { backend_name: "tdsql", status: "ready", username: "tdsql-user" },
  ]);

  assert.deepEqual(accounts, [
    { backend_name: "archive", status: "ready", username: "archive-user" },
    { backend_name: "tdsql", status: "ready", username: "tdsql-user" },
  ]);
});


test("selects only an enabled granted environment for credential operations", () => {
  assert.equal(eligibleTraceDatabaseEnvironmentId([
    { environment_id: "disabled-grant", status: "enabled", grant: { enabled: false } },
    { environment_id: "disabled-env", status: "disabled", grant: { enabled: true } },
    { environment_id: "eligible", status: "enabled", grant: { enabled: true } },
  ]), "eligible");
  assert.equal(eligibleTraceDatabaseEnvironmentId([]), null);
});


test("maps account state to the only user-safe action", () => {
  assert.equal(accountAction({ status: "not_created" }), "create");
  assert.equal(accountAction({ status: "ready" }), "reset");
  assert.equal(accountAction({ status: "error" }), "reset");
  assert.equal(accountAction({ status: "disabled" }), null);
  assert.equal(accountAction({ status: "creating" }), null);
});
