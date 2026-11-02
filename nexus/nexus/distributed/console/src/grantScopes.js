export const ALL_SUBMISSION_SCOPES = Object.freeze([
  "submission:single",
  "submission:live",
  "submission:batch",
  "artifact:write",
  "execution:read",
  "execution:command",
  "execution:stop",
]);

export const GRANT_SCOPE_PRESETS = Object.freeze({
  "all-submissions": { label: "All submissions", scopes: ALL_SUBMISSION_SCOPES },
  custom: { label: "Custom", scopes: [] },
});

export const DEFAULT_GRANT_SCOPE_PRESET = "all-submissions";

export function grantScopesForPreset(preset) {
  const definition = GRANT_SCOPE_PRESETS[preset];
  if (!definition) throw new Error(`Unknown grant scope preset: ${preset}`);
  return [...definition.scopes];
}

export function inferGrantScopePreset(scopes) {
  const values = new Set(scopes || []);
  return values.size === ALL_SUBMISSION_SCOPES.length && ALL_SUBMISSION_SCOPES.every((scope) => values.has(scope))
    ? "all-submissions"
    : "custom";
}
