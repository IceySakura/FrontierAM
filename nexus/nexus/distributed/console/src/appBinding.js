const APP_ID_PATTERN = /^[a-z0-9][a-z0-9._-]{0,127}$/;


export function normalizeAppId(value) {
  return String(value ?? "").trim().toLowerCase();
}


export function isValidAppId(value) {
  return APP_ID_PATTERN.test(normalizeAppId(value));
}


export function buildUserAppBindingRequest(principalId, appId) {
  if (!principalId) throw new Error("principal_id is required");
  const path = `/v1/access/users/${encodeURIComponent(principalId)}/app-binding`;
  if (appId === null) return { method: "DELETE", path, body: undefined };

  const normalized = normalizeAppId(appId);
  if (!isValidAppId(normalized)) throw new Error("invalid app_id");
  return { method: "PUT", path, body: { app_id: normalized } };
}
