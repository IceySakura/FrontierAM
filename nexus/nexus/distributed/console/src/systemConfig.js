export function parseConfigValues(value) {
  let parsed;
  try {
    parsed = JSON.parse(value);
  } catch (error) {
    throw new Error(`Configuration must be valid JSON: ${error.message}`);
  }
  if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
    throw new Error("Configuration must be a JSON object");
  }
  return parsed;
}


function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}


function valuesEqual(left, right) {
  return Object.is(left, right) || JSON.stringify(left) === JSON.stringify(right);
}


function formatConfigPath(path) {
  return path.length ? path.join(".") : "$";
}


export function diffConfigValues(before, after) {
  const changes = [];

  function visit(path, left, right, leftExists, rightExists) {
    if (!leftExists) {
      changes.push({ path: formatConfigPath(path), kind: "added", before: undefined, after: right });
      return;
    }
    if (!rightExists) {
      changes.push({ path: formatConfigPath(path), kind: "removed", before: left, after: undefined });
      return;
    }
    if (isObject(left) && isObject(right)) {
      const keys = [...new Set([...Object.keys(left), ...Object.keys(right)])].sort();
      for (const key of keys) {
        visit(path.concat(key), left[key], right[key], Object.hasOwn(left, key), Object.hasOwn(right, key));
      }
      return;
    }
    if (!valuesEqual(left, right)) {
      changes.push({ path: formatConfigPath(path), kind: "changed", before: left, after: right });
    }
  }

  visit([], before, after, true, true);
  return changes;
}


function expectedVersion(value) {
  const version = Number(value);
  if (!Number.isSafeInteger(version) || version < 0) {
    throw new Error("Expected version must be a non-negative integer");
  }
  return version;
}


export function buildConfigOverridePayload({ role, identity, expectedVersion: version, values, secret = false }) {
  return {
    role,
    identity,
    expected_version: expectedVersion(version),
    values,
    secret: Boolean(secret),
  };
}


export function buildConfigOverrideDeletePayload({ role, identity, expectedVersion: version, secret = false }) {
  return {
    role,
    identity,
    expected_version: expectedVersion(version),
    secret: Boolean(secret),
  };
}
