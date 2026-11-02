export const RESOURCE_LIMIT_FIELDS = [
  { key: "manager_count", label: "Managers", min: 0 },
  {
    key: "cpu_millicores",
    label: "CPU (cores)",
    min: 0,
    displayScale: 1000,
    displayMin: 0,
    displayStep: 0.001,
  },
  {
    key: "memory_bytes",
    label: "Memory (MiB)",
    min: 0,
    displayScale: 1024 ** 2,
    displayMin: 0,
    displayStep: 1,
  },
  { key: "slots", label: "Slots", min: 0 },
  { key: "active_batches", label: "Active batches", min: 0 },
  { key: "batch_max_inflight", label: "Batch max inflight", min: 1 },
  { key: "max_artifact_size_bytes", label: "Artifact bytes", min: 1 },
  { key: "max_source_files", label: "Source files", min: 1 },
  { key: "max_source_file_bytes", label: "Source file bytes", min: 1 },
  { key: "max_source_unpacked_bytes", label: "Source unpacked bytes", min: 1 },
  { key: "max_workspace_bytes", label: "Workspace bytes", min: 1 },
  { key: "max_items", label: "Batch items", min: 1 },
  { key: "max_task_list_bytes", label: "Task list bytes", min: 1 },
  { key: "max_line_bytes", label: "Task line bytes", min: 1 },
  { key: "max_provisioning_timeout_seconds", label: "Provisioning max (seconds)", min: 1 },
  { key: "max_queue_ttl_seconds", label: "Queue TTL max (seconds)", min: 1 },
  { key: "max_execution_ttl_seconds", label: "Execution TTL max (seconds)", min: 1 },
  { key: "max_stop_timeout_seconds", label: "Stop max (seconds)", min: 1 },
  { key: "max_child_queue_ttl_seconds", label: "Child queue TTL max (seconds)", min: 1 },
  { key: "max_timeout_per_run", label: "Run timeout max (seconds)", min: 1 },
  { key: "max_generation_timeout_seconds", label: "Generation max (seconds)", min: 1 },
];

const COMMON_RESOURCE_LIMIT_KEYS = new Set(["manager_count", "cpu_millicores", "memory_bytes", "slots"]);
export const COMMON_RESOURCE_LIMIT_FIELDS = RESOURCE_LIMIT_FIELDS.filter((field) =>
  COMMON_RESOURCE_LIMIT_KEYS.has(field.key)
);
export const ADVANCED_RESOURCE_LIMIT_FIELDS = RESOURCE_LIMIT_FIELDS.filter(
  (field) => !COMMON_RESOURCE_LIMIT_KEYS.has(field.key)
);

const DEFAULT_QUOTA_KEYS = new Set([
  "manager_count",
  "cpu_millicores",
  "memory_bytes",
  "slots",
  "active_batches",
]);
export const DEFAULT_QUOTA_FIELDS = RESOURCE_LIMIT_FIELDS
  .filter((field) => DEFAULT_QUOTA_KEYS.has(field.key))
  .map((field) => ({
    ...field,
    min: 1,
    displayMin: field.key === "cpu_millicores" ? 0.001 : 1,
  }));

export const DEFAULT_TIMEOUT_FIELDS = [
  { key: "provisioning_timeout_seconds", label: "Provisioning (seconds)", min: 1 },
  { key: "queue_ttl_seconds", label: "Queue TTL (seconds)", min: 1 },
  { key: "execution_ttl_seconds", label: "Execution TTL (seconds)", min: 1 },
  { key: "stop_timeout_seconds", label: "Stop (seconds)", min: 1 },
  { key: "child_queue_ttl_seconds", label: "Child queue TTL (seconds)", min: 1 },
  { key: "timeout_per_run", label: "Per run (seconds)", min: 1 },
  { key: "generation_timeout_seconds", label: "Generation (seconds)", min: 1 },
];

export function emptyConfigDraft(fields) {
  return Object.fromEntries(fields.map((field) => [field.key, ""]));
}

export function configToDraft(config, fields) {
  const source = config && typeof config === "object" ? config : {};
  return Object.fromEntries(
    fields.map((field) => [
      field.key,
      source[field.key] === undefined || source[field.key] === null
        ? ""
        : String(source[field.key] / (field.displayScale || 1)),
    ]),
  );
}

export function draftToConfig(draft, fields) {
  const result = {};
  for (const field of fields) {
    const raw = String(draft?.[field.key] ?? "").trim();
    if (!raw) continue;
    if (!field.displayScale && !/^\d+$/.test(raw)) {
      throw new Error(`${field.key} must be an integer`);
    }
    const displayValue = Number(raw);
    const value = displayValue * (field.displayScale || 1);
    const displayMin = field.displayMin ?? field.min;
    if (
      !Number.isFinite(displayValue)
      || displayValue < displayMin
      || !Number.isSafeInteger(value)
      || value < field.min
    ) {
      throw new Error(`${field.key} must be a safe integer greater than or equal to ${field.min}`);
    }
    result[field.key] = value;
  }
  return result;
}

export function draftToCompleteConfig(draft, fields) {
  for (const field of fields) {
    if (!String(draft?.[field.key] ?? "").trim()) {
      throw new Error(`${field.key} is required`);
    }
  }
  return draftToConfig(draft, fields);
}

export function formatResourceLimitValue(field, value) {
  if (typeof value !== "number") return String(value);
  return String(value / (field.displayScale || 1));
}

function formatCount(value, singular, plural = `${singular}s`) {
  return `${value} ${value === 1 ? singular : plural}`;
}

function formatCpu(millicores) {
  return millicores % 1000 === 0 ? formatCount(millicores / 1000, "core") : `${millicores}m CPU`;
}

function formatMemory(bytes) {
  const gibibyte = 1024 ** 3;
  const mebibyte = 1024 ** 2;
  if (bytes % gibibyte === 0) return `${bytes / gibibyte} GiB`;
  if (bytes % mebibyte === 0) return `${bytes / mebibyte} MiB`;
  return `${bytes} bytes`;
}

export function formatQuotaSummary(quota) {
  if (!quota) return "-";
  const parts = [];
  if (typeof quota.manager_count === "number") parts.push(formatCount(quota.manager_count, "manager"));
  if (typeof quota.cpu_millicores === "number") parts.push(formatCpu(quota.cpu_millicores));
  if (typeof quota.memory_bytes === "number") parts.push(formatMemory(quota.memory_bytes));
  if (typeof quota.slots === "number") parts.push(formatCount(quota.slots, "slot"));
  if (typeof quota.active_batches === "number") parts.push(formatCount(quota.active_batches, "batch", "batches"));
  return parts.length ? parts.join(" · ") : "No additional limits";
}
