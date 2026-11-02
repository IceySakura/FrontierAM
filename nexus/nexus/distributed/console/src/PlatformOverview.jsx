import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUpRight, RefreshCw } from "lucide-react";

import { numberValue, startVisiblePolling } from "./platformStatus";
import MetricHistory from "./MetricHistory";
import { snapshotCore, snapshotTimingIssues } from "./snapshotPresentation";

function timestamp(value) {
  if (!value) return "Not collected";
  const normalized = /Z$|[+-]\d\d:\d\d$/.test(value) ? value : `${value}Z`;
  return new Date(normalized).toLocaleString();
}

function usePoll(load, interval, active) {
  const [error, setError] = useState(null);
  const [pending, setPending] = useState(false);
  const poller = useRef(null);
  useEffect(() => {
    if (!active) return;
    const poll = startVisiblePolling(load, { interval, onError: setError, onPending: setPending });
    poller.current = poll;
    window.addEventListener("refresh-platform-observability", poll.refresh);
    return () => {
      poll.stop();
      poller.current = null;
      window.removeEventListener("refresh-platform-observability", poll.refresh);
    };
  }, [load, interval, active]);
  return { error, pending, refresh: () => poller.current?.refresh() };
}

function MetricCard({ label, value, note, onMore }) {
  return <div className="summary-card"><div><div className="summary-label">{onMore ? <button type="button" className="summary-link" onClick={onMore}>{label}<ArrowUpRight size={13} aria-hidden="true" /></button> : label}</div><div className="summary-value">{value}</div>{note ? <div className="muted summary-note">{note}</div> : null}</div></div>;
}

export default function PlatformOverview({ request, controlPath, environments = [], environmentId: fixedEnvironment, active = true }) {
  const [selected, setSelected] = useState("");
  const environmentId = fixedEnvironment ?? selected;
  const [live, setLive] = useState(null);
  const [snapshot, setSnapshot] = useState(null);
  const [expanded, setExpanded] = useState(() => new Set());
  const openDomain = useCallback((domain) => {
    setExpanded((current) => current.has(domain) ? current : new Set([...current, domain]));
    requestAnimationFrame(() => document.getElementById(`metric-detail-${domain}`)?.scrollIntoView({ behavior: "smooth", block: "start" }));
  }, []);
  const toggleDomain = useCallback((domain, open) => setExpanded((current) => {
    if (current.has(domain) === open) return current;
    const next = new Set(current);
    if (open) next.add(domain); else next.delete(domain);
    return next;
  }), []);
  useEffect(() => { setLive(null); setSnapshot(null); }, [environmentId]);
  const loadLive = useCallback(async ({ signal }) => {
    const path = environmentId
      ? `/v1/environments/${encodeURIComponent(environmentId)}/stats`
      : "/v1/system/summary";
    const value = await request("GET", controlPath(path), undefined, { signal });
    if (!signal.aborted) setLive(value);
  }, [request, controlPath, environmentId]);
  const loadSnapshot = useCallback(async ({ signal }) => {
    const query = environmentId ? `?environment_id=${encodeURIComponent(environmentId)}` : "";
    const value = await request("GET", controlPath(`/v1/system/status${query}`), undefined, { signal });
    if (!signal.aborted) setSnapshot(value);
  }, [request, controlPath, environmentId]);
  const livePoll = usePoll(loadLive, 2000, active);
  const snapshotPoll = usePoll(loadSnapshot, 30000, active);
  const managers = live?.managers || {};
  const usage = live?.resource_usage || {};
  const core = snapshotCore(snapshot);
  const timingIssues = snapshotTimingIssues(snapshot);
  const shown = (value, suffix = "") => Number.isFinite(value) ? numberValue(value, 1, suffix) : "Not collected";
  const latestSnapshot = Object.values(snapshot?.groups || {}).map((group) => group.sampled_at).filter(Boolean).sort().at(0);
  return <div className="workspace-stack" id="platform-overview">
    <div className="observability-quality" role="status">
      <span>Live: {livePoll.error ? "Query failed" : live?.data_status || "Collecting"}</span>
      <span>Snapshot: {snapshotPoll.error ? "Query failed" : snapshot?.data_status || "Collecting"} · oldest sample {timestamp(latestSnapshot)}</span>
      <span>Metrics export: {live?.metrics?.last_export_error_code || (live?.metrics?.last_export_success_at ? timestamp(new Date(live.metrics.last_export_success_at * 1000).toISOString()) : "No confirmed export")}</span>
      {timingIssues.map((issue) => <button className="row-action" key={issue.label} onClick={() => openDomain(issue.domain)}>{issue.label}{issue.count ? `: ${issue.count}` : ""}</button>)}
    </div>
    <section className="panel">
      <div className="panel-header">
        <div><h2>Live platform state</h2><span className="muted">Refreshes every 2 seconds · queried {timestamp(live?.queried_at)}</span></div>
        <div className="row-actions">
          {fixedEnvironment == null ? <select aria-label="Platform environment" value={selected} onChange={(event) => setSelected(event.target.value)}>
            <option value="">All environments</option>
            {environments.map((item) => <option key={item.environment_id} value={item.environment_id}>{item.name || item.environment_id}</option>)}
          </select> : <span>{fixedEnvironment}</span>}
          <button className="icon-button" type="button" onClick={livePoll.refresh} disabled={livePoll.pending}><RefreshCw size={14} aria-hidden="true" />Refresh live state</button>
        </div>
      </div>
      <p className="muted" role="status">{livePoll.error ? "Latest read failed; showing the last response." : `Data: ${live?.data_status || "not collected"}`}</p>
      <div className="summary-grid system-summary-grid">
        <MetricCard label="Online managers" value={shown(managers.healthy)} />
        <MetricCard label="Available / total slots" value={`${shown(managers.available_slots)} / ${shown(managers.capacity_total)}`} />
        <MetricCard label="Assignment reserved slots" value={shown(managers.reserved_slots)} />
        <MetricCard label="Reserved CPU" value={Number.isFinite(usage.cpu_millicores) ? numberValue(usage.cpu_millicores, 1000, " cores") : "Not collected"} />
        <MetricCard label="Reserved memory" value={Number.isFinite(usage.memory_bytes) ? numberValue(usage.memory_bytes, 1024 ** 3, " GiB") : "Not collected"} />
        <MetricCard label="Reserved allocation slots" value={shown(usage.slots)} />
      </div>
      <p className="muted">Reservation accounting · cache age {shown(live?.cache_age_seconds, " s")}. Last healthy heartbeat: {timestamp(live?.liveness?.last_heartbeat_at)}. Heartbeat interval {shown(live?.liveness?.heartbeat_interval_seconds, " s")}; health window {shown(live?.liveness?.healthy_ttl_seconds, " s")}. Actual CPU and memory utilization are measured separately by infrastructure monitoring.</p>
    </section>
    <section className="panel" id="platform-snapshots">
      <div className="panel-header">
        <div><h2>Snapshot · current work</h2><span className="muted">Refreshes every 30 seconds · oldest sample {timestamp(latestSnapshot)}</span></div>
        <button className="icon-button" type="button" onClick={snapshotPoll.refresh} disabled={snapshotPoll.pending}><RefreshCw size={14} aria-hidden="true" />Refresh collected state</button>
      </div>
      <div className="summary-grid system-summary-grid snapshot-core">
        <MetricCard label="Running tasks" value={shown(core.running)} note={`Waiting for input: ${shown(core.waitingInput)}`} onMore={() => openDomain("execution")} />
        <MetricCard label="Preparing tasks" value={shown(core.preparing)} onMore={() => openDomain("preparation")} />
        <MetricCard label="Schedulable queue" value={shown(core.schedulable)} onMore={() => openDomain("execution")} />
        <MetricCard label="Oldest scheduling wait" value={core.missingWait && core.oldestWait === null ? "Missing wait start" : shown(core.oldestWait, " s")} note={core.missingWait ? `${core.missingWait} records lack a wait start` : "Schedulable tasks only"} onMore={() => openDomain("execution")} />
        <MetricCard label="Overdue tasks" value={shown(core.overdue)} note="Configured execution deadlines" onMore={() => openDomain("execution")} />
        <MetricCard label="Overdue release confirmation" value={`${shown(core.allocationRelease)} / ${shown(core.builderRelease)}`} note="Allocations / builders" onMore={() => openDomain("resources")} />
      </div>
    </section>
    <MetricHistory request={request} controlPath={controlPath} environmentId={environmentId} exportState={live?.metrics} active={active} snapshot={snapshot} expanded={expanded} onToggle={toggleDomain} onMore={openDomain} />
  </div>;
}
