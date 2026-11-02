export function dedupeTraceDatabaseAccounts(accounts) {
  const byBackend = new Map();
  for (const account of accounts || []) {
    if (!account?.backend_name) continue;
    const current = byBackend.get(account.backend_name);
    if (!current || (current.status === "not_created" && account.status !== "not_created")) {
      byBackend.set(account.backend_name, account);
    }
  }
  return [...byBackend.values()].sort((left, right) => left.backend_name.localeCompare(right.backend_name));
}


export function eligibleTraceDatabaseEnvironmentId(environments) {
  return (environments || []).find(
    (environment) => environment?.status === "enabled" && environment.grant?.enabled,
  )?.environment_id || null;
}


export function accountAction(account) {
  if (account?.status === "not_created") return "create";
  if (["ready", "error"].includes(account?.status)) return "reset";
  return null;
}
