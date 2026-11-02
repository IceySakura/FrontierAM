export const COMMANDABLE_EXECUTION_STATUSES = new Set(["running", "waiting_input"]);
export const TERMINAL_EXECUTION_STATUSES = new Set(["succeeded", "failed", "killed", "lost", "expired", "closed"]);
export const TERMINAL_LIVE_EVENT_TYPES = new Set(["run.output", "run.error", "run.killed"]);

export function isTerminalExecution(item) {
  return Boolean(item && (TERMINAL_EXECUTION_STATUSES.has(item.status) || item.route_status === "closed"));
}

export function canSubmitLiveCommand(item) {
  return Boolean(
    item &&
      item.mode === "live" &&
      item.route_status !== "closed" &&
      COMMANDABLE_EXECUTION_STATUSES.has(item.status),
  );
}

export function liveEventReadMode(item, eventStreamPaused = false) {
  if (!item || item.mode !== "live") return "none";
  if (isTerminalExecution(item)) return "history";
  if (eventStreamPaused) return "paused";
  return "stream";
}

export function isTerminalLiveEvent(type) {
  return TERMINAL_LIVE_EVENT_TYPES.has(type);
}

export function safeParseEventData(data) {
  try {
    return JSON.parse(data);
  } catch {
    return { data };
  }
}

export function parseSseEvents(text, now = () => new Date().toISOString()) {
  const events = [];
  let id = "";
  let type = "message";
  let dataLines = [];

  const flush = () => {
    if (!dataLines.length) {
      id = "";
      type = "message";
      return;
    }
    const data = dataLines.join("\n");
    events.push({
      id: id || undefined,
      type,
      created_at: now(),
      payload: safeParseEventData(data),
    });
    id = "";
    type = "message";
    dataLines = [];
  };

  for (const rawLine of text.split(/\r?\n/)) {
    if (rawLine === "") {
      flush();
      continue;
    }
    if (rawLine.startsWith(":")) continue;
    const colon = rawLine.indexOf(":");
    const field = colon === -1 ? rawLine : rawLine.slice(0, colon);
    const value = colon === -1 ? "" : rawLine.slice(colon + 1).replace(/^ /, "");
    if (field === "id") {
      id = value;
    } else if (field === "event") {
      type = value || "message";
    } else if (field === "data") {
      dataLines.push(value);
    }
  }
  flush();
  return events;
}
