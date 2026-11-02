import { useEffect, useRef, useState } from "react";
import { startVisiblePolling } from "./platformStatus.js";

export function useResourceQuery(
  key,
  load,
  { interval = 0, enabled = true } = {},
) {
  const [state, setState] = useState({
    key: null,
    data: null,
    error: null,
    loading: false,
    updatedAt: null,
  });
  const [revision, refresh] = useState(0);
  const loader = useRef(load),
    generation = useRef(0);
  loader.current = load;
  useEffect(() => {
    if (!enabled) return undefined;
    const id = ++generation.current;
    const run = async ({ signal }) => {
      if (signal.aborted) return;
      setState((previous) => ({
        ...(previous.key === key ? previous : { data: null, updatedAt: null }),
        key,
        error: null,
        loading: true,
      }));
      try {
        const data = await loader.current(signal);
        if (!signal.aborted && id === generation.current)
          setState({
            key,
            data,
            error: null,
            loading: false,
            updatedAt: Date.now(),
          });
      } catch (error) {
        if (id === generation.current)
          setState((previous) => ({
            ...previous,
            key,
            loading: false,
            error: document.hidden
              ? null
              : signal.aborted
                ? new Error("Request timed out; refresh to retry.")
                : error,
          }));
        throw error;
      }
    };
    const poll = startVisiblePolling(run, {
      interval: interval || 2147483647,
      timeoutMs: 15000,
      onError: () => {},
    });
    const manual = () => refresh((value) => value + 1);
    window.addEventListener("refresh-console-resources", manual);
    return () => {
      generation.current++;
      poll.stop();
      window.removeEventListener("refresh-console-resources", manual);
    };
  }, [key, enabled, interval, revision]);
  const current =
    state.key === key && enabled
      ? state
      : { key, data: null, error: null, loading: enabled, updatedAt: null };
  return {
    ...current,
    stale: Boolean(current.error && current.data),
    refresh: () => refresh((value) => value + 1),
  };
}
