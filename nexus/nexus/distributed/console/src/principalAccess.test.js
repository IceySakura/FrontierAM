import test from "node:test";
import assert from "node:assert/strict";

import {
  buildPrincipalGrantRequest,
  partitionPrincipalGrants,
  visiblePrincipalGrants,
} from "./principalAccess.js";


test("builds encoded principal grant requests and rejects self grants", () => {
  assert.deepEqual(buildPrincipalGrantRequest("owner/a", "grantee b", "PUT"), {
    method: "PUT",
    path: "/v1/access/principal-grants/owner%2Fa/grantee%20b",
  });
  assert.throws(() => buildPrincipalGrantRequest("same", "same", "PUT"), /must differ/);
  assert.throws(() => buildPrincipalGrantRequest("owner", "", "DELETE"), /required/);
});


test("partitions direct grants into outgoing and incoming rows", () => {
  const grants = [
    { owner_principal_id: "a", grantee_principal_id: "c" },
    { owner_principal_id: "c", grantee_principal_id: "d" },
    { owner_principal_id: "b", grantee_principal_id: "c" },
  ];

  assert.deepEqual(partitionPrincipalGrants(grants, "c"), {
    outgoing: [{ owner_principal_id: "c", grantee_principal_id: "d" }],
    incoming: [
      { owner_principal_id: "a", grantee_principal_id: "c" },
      { owner_principal_id: "b", grantee_principal_id: "c" },
    ],
  });
});


test("ordinary users only see grants they own", () => {
  const grants = [
    { owner_principal_id: "a", grantee_principal_id: "c" },
    { owner_principal_id: "c", grantee_principal_id: "d" },
    { owner_principal_id: "b", grantee_principal_id: "c" },
  ];

  assert.deepEqual(visiblePrincipalGrants(grants, "c", false), [
    { owner_principal_id: "c", grantee_principal_id: "d" },
  ]);
  assert.deepEqual(visiblePrincipalGrants(grants, "c", true), grants);
});
