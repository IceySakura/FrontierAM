export function mergeRunTraceBackendGrantDraft(catalog, grants, grantsPrincipalId, selectedPrincipalId) {
  const currentGrants = grantsPrincipalId && selectedPrincipalId && grantsPrincipalId !== selectedPrincipalId
    ? []
    : grants || [];
  const grantsByName = new Map(currentGrants.map((grant) => [grant.backend_name, grant]));
  return (catalog || [])
    .filter((backend) => !backend.system_managed && backend.user_selectable)
    .map((backend) => {
      const grant = grantsByName.get(backend.name);
      return {
        backend_name: backend.name,
        enabled: Boolean(grant?.enabled),
        default_order: grant?.default_order ?? null,
        ...(grant?.updated_at ? { updated_at: grant.updated_at } : {}),
      };
    })
    .sort((left, right) => left.backend_name.localeCompare(right.backend_name));
}

export function buildRunTraceBackendGrantPayload(draft) {
  const grants = draft || [];
  if (grants.some((grant) => grant.default_order != null && !grant.enabled)) {
    throw new Error("A disabled backend cannot be a default");
  }
  const enabled = grants.filter((grant) => grant.enabled);
  const defaultOrders = enabled
    .filter((grant) => grant.default_order != null)
    .map((grant) => Number(grant.default_order))
    .sort((left, right) => left - right);
  if (defaultOrders.some((value, index) => !Number.isInteger(value) || value !== index)) {
    throw new Error("Default order must be unique and continuous from zero");
  }
  return {
    grants: enabled.map((grant) => ({
      backend_name: grant.backend_name,
      enabled: true,
      default_order: grant.default_order == null ? null : Number(grant.default_order),
    })),
  };
}

export function runTraceBackendAccessRows(access) {
  const seen = new Set();
  const rows = [];
  for (const [items, status] of [
    [access?.items || [], "granted"],
    [access?.claimable_sinks || [], "available"],
    [access?.required_sinks || [], "required"],
  ]) {
    for (const item of items) {
      if (seen.has(item.name)) continue;
      seen.add(item.name);
      rows.push({ ...item, access: status });
    }
  }
  return rows;
}
