export function managerRoleLabel(workload) {
  if (workload === "orchestration") return "Orchestration";
  if (workload === "execution") return "Execution";
  return workload || "-";
}

export function executionDisplayStatus(execution, outcome = execution?.outcome) {
  return outcome || execution?.status || "unknown";
}

export function executionLogSourceLabel(source) {
  if (source === "instance_manager") return "Instance Manager";
  if (source === "zhiyan") return "Zhiyan";
  return "Not started";
}

export function executionLogStateText(payload) {
  if (!payload) return "Loading...";
  if (payload.availability === "not_started") return "Manager has not started";
  if (payload.availability === "ingest_pending") {
    return "Execution finished; Zhiyan logs are still arriving";
  }
  if (payload.availability === "unavailable") {
    return payload.error_code || "Log source unavailable";
  }
  return "No log lines";
}

export function executionLogRetryDelay(availability, attempt) {
  if (availability !== "ingest_pending") return null;
  return [2000, 5000, 10000][attempt] ?? null;
}

export function formatExecutionLogLine(line, formatTimestamp = (value) => String(value)) {
  const parts = [];
  if (line?.timestamp) parts.push(formatTimestamp(line.timestamp));
  if (line?.level) parts.push(line.level);
  if (line?.logger) parts.push(line.logger);
  if (line?.message) parts.push(line.message);

  const exception = line?.exception;
  let exceptionText = "";
  if (typeof exception?.traceback === "string") {
    exceptionText = exception.traceback.trimEnd();
  } else if (exception) {
    exceptionText = [exception.type, exception.message].filter(Boolean).join(": ");
  }
  return [parts.join(" | "), exceptionText].filter(Boolean).join("\n");
}
