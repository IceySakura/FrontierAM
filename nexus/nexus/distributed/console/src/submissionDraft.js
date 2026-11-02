const TIMEOUT_FIELDS = [
  ["provisioningTimeoutSeconds", "provisioning_timeout_seconds"],
  ["queueTtlSeconds", "queue_ttl_seconds"],
  ["executionTtlSeconds", "execution_ttl_seconds"],
  ["stopTimeoutSeconds", "stop_timeout_seconds"],
];

const CPU_MILLICORES_PER_CORE = 1000;
const MEMORY_BYTES_PER_GIB = 1024 ** 3;

export const CPU_CORES_INPUT = { min: 0.001, step: 0.001 };
export const MEMORY_GIB_INPUT = { min: 0.125, step: 0.125 };


function newIdempotencyKey() {
  const suffix = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `console-${suffix}`;
}


export function defaultSubmissionDraft(type = "single", environmentId = "") {
  return {
    type,
    idempotencyKey: newIdempotencyKey(),
    environmentId,
    pool: "",
    hostEnv: "",
    codeMode: "config_only",
    codeJson: '{"mode":"config_only"}',
    imageVersion: "",
    runInput: '{"task_name":"identity"}',
    liveConfig: "{}",
    generation: '{"mode":"remote_materialize","source":{"type":"compose","compose":{"parts":[]}}}',
    batchPolicy: "{}",
    maxInflight: "20",
    managerCpuCores: "1",
    managerMemoryGiB: "2",
    batchManagerCpuCores: "1",
    batchManagerMemoryGiB: "4",
    managerCount: "",
    managerSlots: type === "batch" ? "" : "20",
    provisioningTimeoutSeconds: "",
    queueTtlSeconds: "",
    executionTtlSeconds: "",
    stopTimeoutSeconds: "",
  };
}


export function changeSubmissionType(draft, type) {
  return { ...draft, type };
}


function parseObject(value, field) {
  let parsed;
  try {
    parsed = JSON.parse(value);
  } catch (error) {
    throw new Error(`${field} must be valid JSON: ${error.message}`);
  }
  if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
    throw new Error(`${field} must be a JSON object`);
  }
  return parsed;
}


function positiveNumber(value, field, { integer = false } = {}) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0 || (integer && !Number.isSafeInteger(number))) {
    throw new Error(`${field} must be a positive ${integer ? "integer" : "number"}`);
  }
  return number;
}

function positiveUnitNumber(value, field, scale, unit) {
  const number = positiveNumber(value, field);
  if (!Number.isSafeInteger(number * scale)) {
    throw new Error(`${field} must align to ${unit}`);
  }
  return number;
}


function applyTimeouts(payload, draft) {
  for (const [draftKey, payloadKey] of TIMEOUT_FIELDS) {
    const value = String(draft[draftKey] || "").trim();
    if (value) payload[payloadKey] = positiveNumber(value, payloadKey, { integer: true });
  }
}


function parseHostEnv(value) {
  const lines = String(value || "").split(/\r?\n/);
  const entries = [];
  for (const [index, line] of lines.entries()) {
    if (!line.trim()) continue;
    const separator = line.indexOf("=");
    if (separator <= 0) {
      throw new Error(`Host environment line ${index + 1} must use KEY=VALUE`);
    }
    const name = line.slice(0, separator).trim();
    if (!name) {
      throw new Error(`Host environment line ${index + 1} must use KEY=VALUE`);
    }
    entries.push({ name, value: line.slice(separator + 1) });
  }
  return entries;
}


function singleResources(draft) {
  const cpu = String(draft.managerCpuCores || "").trim();
  const memory = String(draft.managerMemoryGiB || "").trim();
  const slots = String(draft.managerSlots || "").trim();
  if (!cpu && !memory && !slots) return undefined;
  if (!cpu || !memory || !slots) throw new Error("Manager CPU, memory, and slots must be provided together");
  return {
    managers: {
      count: 1,
      cpu_cores: positiveUnitNumber(cpu, "manager_cpu_cores", CPU_MILLICORES_PER_CORE, "millicores"),
      memory_gib: positiveUnitNumber(memory, "manager_memory_gib", MEMORY_BYTES_PER_GIB, "bytes"),
      slots: positiveNumber(slots, "manager_slots", { integer: true }),
    },
  };
}


function batchResources(draft) {
  const batchManagerCpu = String(draft.batchManagerCpuCores || "").trim();
  const batchManagerMemory = String(draft.batchManagerMemoryGiB || "").trim();
  const managerCount = String(draft.managerCount || "").trim();
  const managerCpu = String(draft.managerCpuCores || "").trim();
  const managerMemory = String(draft.managerMemoryGiB || "").trim();
  const managerSlots = String(draft.managerSlots || "").trim();
  const required = [batchManagerCpu, batchManagerMemory, managerCpu, managerMemory];
  if (required.every((value) => !value) && !managerCount && !managerSlots) return undefined;
  if (required.some((value) => !value)) {
    throw new Error("Orchestration and Execution CPU and memory must be provided together");
  }
  const managers = {
    cpu_cores: positiveUnitNumber(managerCpu, "manager_cpu_cores", CPU_MILLICORES_PER_CORE, "millicores"),
    memory_gib: positiveUnitNumber(managerMemory, "manager_memory_gib", MEMORY_BYTES_PER_GIB, "bytes"),
  };
  if (managerCount) managers.count = positiveNumber(managerCount, "manager_count", { integer: true });
  if (managerSlots) managers.slots = positiveNumber(managerSlots, "manager_slots", { integer: true });
  return {
    batch_manager: {
      cpu_cores: positiveUnitNumber(
        batchManagerCpu, "batch_manager_cpu_cores", CPU_MILLICORES_PER_CORE, "millicores",
      ),
      memory_gib: positiveUnitNumber(
        batchManagerMemory, "batch_manager_memory_gib", MEMORY_BYTES_PER_GIB, "bytes",
      ),
    },
    managers,
  };
}


export function buildSubmission(draft) {
  if (!draft.environmentId) throw new Error("Environment is required");
  if (!draft.idempotencyKey) throw new Error("Idempotency key is required");
  let code;
  if (draft.codeMode === "config_only") {
    code = { mode: "config_only" };
  } else if (draft.codeMode === "image") {
    const imageVersion = String(draft.imageVersion || "").trim();
    if (!imageVersion) throw new Error("Workload image version is required");
    code = { mode: "image", image_version: imageVersion };
  } else {
    code = parseObject(draft.codeJson, "Code");
  }
  const payload = {
    submission_version: 1,
    idempotency_key: draft.idempotencyKey,
    environment_id: draft.environmentId,
    code,
  };
  if (draft.pool) payload.pool = draft.pool;
  const hostEnv = parseHostEnv(draft.hostEnv);
  if (hostEnv.length) {
    if (draft.pool) throw new Error("Host environment requires dynamic allocation");
    payload.host_env = hostEnv;
  }
  applyTimeouts(payload, draft);
  if (draft.type === "batch") {
    payload.generation = parseObject(draft.generation, "Generation");
    payload.batch = parseObject(draft.batchPolicy || "{}", "Batch policy");
    payload.batch.max_inflight = positiveNumber(draft.maxInflight, "max_inflight", { integer: true });
    const resources = draft.pool ? undefined : batchResources(draft);
    if (resources) payload.resources = resources;
    return payload;
  }
  payload.run_input = parseObject(draft.runInput, "Run input");
  const resources = draft.pool ? undefined : singleResources(draft);
  if (resources) payload.resources = resources;
  if (draft.type === "live") payload.live = parseObject(draft.liveConfig || "{}", "Live config");
  return payload;
}
