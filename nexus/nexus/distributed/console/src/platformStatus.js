export function numberValue(value, scale = 1, suffix = "") {
  if (typeof value !== "number" || !Number.isFinite(value)) return "Unknown";
  return `${Number((value / scale).toFixed(2)).toLocaleString()}${suffix}`;
}

export function snapshotRows(group) {
  if (!group || group.data_status === "unavailable" || !Array.isArray(group.values?.items)) return null;
  return group.values.items;
}

export function startVisiblePolling(load, {
  interval,
  timeoutMs = 5000,
  fixedCadence = false,
  visibility = document,
  onError = () => {},
  onPending = () => {},
  setTimer = setTimeout,
  clearTimer = clearTimeout,
}) {
  let stopped = false;
  let resumePending = false;
  let pending = false;
  let timer = null;
  let controller = null;
  const refresh = async () => {
    if (stopped || pending || visibility.hidden) return;
    clearTimer(timer);
    pending = true;
    const startedAt = performance.now();
    onPending(true);
    controller = new AbortController();
    const timeout = setTimer(() => controller?.abort(), timeoutMs);
    try {
      await load({ signal: controller.signal });
      if (!stopped) onError(null);
    } catch (error) {
      if (!stopped && !visibility.hidden) onError(error);
    } finally {
      clearTimer(timeout);
      pending = false;
      if (!stopped) {
        onPending(false);
        if (resumePending && !visibility.hidden) { resumePending = false; refresh(); return; }
        timer = setTimer(refresh, fixedCadence ? Math.max(0, interval - (performance.now() - startedAt)) : interval);
      }
    }
  };
  const visibilityChanged = () => {
    clearTimer(timer);
    if (visibility.hidden) controller?.abort();
    else if (pending) resumePending = true;
    else refresh();
  };
  visibility.addEventListener("visibilitychange", visibilityChanged);
  refresh();
  return {
    refresh,
    stop() {
      stopped = true;
      clearTimer(timer);
      controller?.abort();
      visibility.removeEventListener("visibilitychange", visibilityChanged);
    },
  };
}

export function mergeExecutionSummary(previous, summary) {
  if (previous?.execution_id !== summary.execution_id) return summary;
  const current = summary.current_assignment;
  return {
    ...previous,
    ...summary,
    current_assignment: current
      ? previous.current_assignment?.assignment_id === current.assignment_id
        ? { ...previous.current_assignment, ...current }
        : current
      : null,
  };
}
