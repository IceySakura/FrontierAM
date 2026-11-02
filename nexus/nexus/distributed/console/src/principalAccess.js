function requiredPrincipal(value, label) {
  const normalized = String(value || "").trim();
  if (!normalized) throw new Error(`${label} is required`);
  return normalized;
}


export function buildPrincipalGrantRequest(ownerPrincipalId, granteePrincipalId, method = "PUT") {
  const owner = requiredPrincipal(ownerPrincipalId, "owner_principal_id");
  const grantee = requiredPrincipal(granteePrincipalId, "grantee_principal_id");
  if (owner === grantee) throw new Error("owner and grantee must differ");
  return {
    method,
    path: `/v1/access/principal-grants/${encodeURIComponent(owner)}/${encodeURIComponent(grantee)}`,
  };
}


export function partitionPrincipalGrants(grants, principalId) {
  const sorted = [...(grants || [])].sort((left, right) => (
    left.owner_principal_id.localeCompare(right.owner_principal_id)
    || left.grantee_principal_id.localeCompare(right.grantee_principal_id)
  ));
  return {
    outgoing: sorted.filter((grant) => grant.owner_principal_id === principalId),
    incoming: sorted.filter((grant) => grant.grantee_principal_id === principalId),
  };
}


export function visiblePrincipalGrants(grants, principalId, isAdmin) {
  if (isAdmin) return [...(grants || [])];
  return partitionPrincipalGrants(grants, principalId).outgoing;
}
