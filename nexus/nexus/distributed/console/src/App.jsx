import ConsoleLink from "./ConsoleLink.jsx";
import { UserGuideLink, UserGuideWelcome } from "./UserGuide.jsx";
import { BuildsView, BuildDetail } from "./BuildViews.jsx";
import { AllocationList, AllocationDetail, RunPreparation } from "./AllocationViews.jsx";
import ManagerInventory from "./ManagerInventory.jsx";
import { useResourceQuery } from "./resourceQuery.js";
import { ResourceBack, ResourceNotice, ResourcePager, RefreshStatus, CopyId } from "./ResourceUI.jsx";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Boxes,
  ChevronLeft,
  ChevronRight,
  CircleStop,
  ClipboardPlus,
  Copy,
  Database,
  Eye,
  EyeOff,
  Gauge,
  Layers,
  Link2,
  ListFilter,
  MoreHorizontal,
  Pause,
  Pencil,
  Play,
  Plus,
  Radio,
  RefreshCcw,
  Route,
  Send,
  Server,
  ShieldCheck,
  SquareTerminal,
  Trash2,
  Unlink,
  Users,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import PlatformOverview from "./PlatformOverview";
import BusinessStatistics from "./BusinessStatistics";
import { mergeExecutionSummary, startVisiblePolling } from "./platformStatus";

import {
  canSubmitLiveCommand,
  isTerminalLiveEvent,
  liveEventReadMode,
  parseSseEvents,
  safeParseEventData,
} from "./liveEvents.js";
import {
  executionDisplayStatus,
  formatExecutionLogLine,
  executionLogRetryDelay,
  executionLogSourceLabel,
  executionLogStateText,
} from "./executionPresentation.js";
import {
  ADVANCED_RESOURCE_LIMIT_FIELDS,
  COMMON_RESOURCE_LIMIT_FIELDS,
  DEFAULT_QUOTA_FIELDS,
  DEFAULT_TIMEOUT_FIELDS,
  RESOURCE_LIMIT_FIELDS,
  configToDraft,
  draftToConfig,
  draftToCompleteConfig,
  emptyConfigDraft,
  formatQuotaSummary,
  formatResourceLimitValue,
} from "./environmentConfig.js";
import {
  CPU_CORES_INPUT,
  MEMORY_GIB_INPUT,
  buildSubmission,
  changeSubmissionType,
  defaultSubmissionDraft,
} from "./submissionDraft.js";
import { consolePath, navigationItems, parseConsoleLocation, navigateConsole, savedListRoute, setNavigationIdentity, updateRouteQuery, restoreConsoleScroll } from "./routes.js";
import {
  ALL_SUBMISSION_SCOPES,
  DEFAULT_GRANT_SCOPE_PRESET,
  GRANT_SCOPE_PRESETS,
  grantScopesForPreset,
  inferGrantScopePreset,
} from "./grantScopes.js";
import {
  emptyProviderConfigDraft,
  providerConfigToDraft,
  providerConfigToPayload,
} from "./providerConfig.js";
import {
  buildUserAppBindingRequest,
  isValidAppId,
  normalizeAppId,
} from "./appBinding.js";
import {
  accountAction,
  dedupeTraceDatabaseAccounts,
  eligibleTraceDatabaseEnvironmentId,
} from "./traceDatabaseAccess.js";
import {
  buildRunTraceBackendGrantPayload,
  mergeRunTraceBackendGrantDraft,
  runTraceBackendAccessRows,
} from "./runTraceBackends.js";
import {
  buildPrincipalGrantRequest,
  visiblePrincipalGrants,
} from "./principalAccess.js";
import {
  buildConfigOverrideDeletePayload,
  buildConfigOverridePayload,
  diffConfigValues,
  parseConfigValues,
} from "./systemConfig.js";

const CONFIG = window.__NEXUS_DISTRIBUTED_CONSOLE__ || {};
const STOPPABLE_STATUSES = new Set(["queued", "assigned", "starting", "running", "waiting_input"]);
const LIVE_EVENT_TYPES = [
  "runtime.ready",
  "input.required",
  "turn.submitted",
  "turn.started",
  "run.status",
  "run.output",
  "run.error",
  "run.killed",
  "agent.event",
  "driver.event",
  "trace.event",
  "log.appended",
  "progress.updated",
];
const NAVIGATION_ICONS = {
  runs: Activity,
  builds: Boxes,
  batches: Layers,
  "my-environments": Boxes,
  "shared-access": ShieldCheck,
  access: Users,
  environments: Boxes,
  infrastructure: Server,
  system: Database,
};

function navigationRoute(key) { return savedListRoute(key, key === "infrastructure" ? "managers" : undefined); }

function routeTitle(route) {
  if (route.section === "build") return "Build Detail";
  if (route.section === "allocation") return "Allocation Detail";
  if (route.section === "builds") return "Build Diagnostics";
  if (route.section === "execution") return "Run Detail";
  if (route.section === "my-environments") return "My Environments";
  return navigationItems(true).find((item) => item.key === route.section)?.label || "Runs";
}

function trimSlash(value) {
  return (value || "").replace(/\/+$/, "");
}

function controlBaseUrl() {
  return trimSlash(window.location.origin);
}

function applicationBaseUrl() {
  return trimSlash(CONFIG.applicationBaseUrl || window.location.origin);
}

function controlPath(path) {
  return `${controlBaseUrl()}${path}`;
}

function applicationPath(path) {
  return `${applicationBaseUrl()}${path}`;
}

let csrfToken = "";

async function request(method, url, body, options = {}) {
  const headers = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (["POST", "PUT", "PATCH", "DELETE"].includes(method) && csrfToken) {
    headers["X-Nexus-CSRF-Token"] = csrfToken;
  }
  const response = await fetch(url, {
    method,
    credentials: "include",
    headers: Object.keys(headers).length ? headers : undefined,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: options.signal,
  });
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { body: text };
    }
  }
  if (!response.ok) {
    const detail = payload.error || payload.detail || response.statusText;
    throw new Error(`${response.status} ${detail}`);
  }
  return payload;
}

async function requestText(method, url, body, options = {}) {
  const response = await fetch(url, { method, credentials: "include", signal: options.signal });
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`${response.status} ${text || response.statusText}`);
  }
  return text;
}

function formatTime(value) {
  if (!value) return "-";
  const normalized = (
    typeof value === "string"
    && /^\d{4}-\d{2}-\d{2}T/.test(value)
    && !/(?:Z|[+-]\d{2}:\d{2})$/i.test(value)
  ) ? `${value}Z` : value;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

function formatLogTime(value) {
  if (!value) return "-";
  const normalized = (
    typeof value === "string"
    && /^\d{4}-\d{2}-\d{2}T/.test(value)
    && !/(?:Z|[+-]\d{2}:\d{2})$/i.test(value)
  ) ? `${value}Z` : value;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return String(value);
  const pad = (part) => String(part).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`
    + ` ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

function humanizeStatus(value) {
  return value ? String(value).replaceAll("_", " ") : "-";
}

function displayAttempt(value) {
  return value == null ? "-" : Number(value) + 1;
}

function shortId(value) {
  if (!value) return "-";
  if (value.length <= 18) return value;
  return `${value.slice(0, 10)}…${value.slice(-6)}`;
}

function statusTone(status) {
  if (["running", "waiting_input", "active", "open", "ready", "succeeded"].includes(status)) return "good";
  if (["queued", "assigned", "starting", "cached", "requested", "building"].includes(status)) return "info";
  if (["failed", "lost", "expired", "killed", "closed", "cancelled"].includes(status)) return "bad";
  if (["stopping", "draining", "partial_success"].includes(status)) return "warn";
  return "neutral";
}

function redact(value) {
  if (Array.isArray(value)) return value.map(redact);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [
      key,
      /token|api[_-]?key|secret|password|credential/i.test(key) ? "**" : redact(item),
    ])
  );
}

function prettyJson(value) {
  return JSON.stringify(redact(value ?? {}), null, 2);
}

function canStopExecution(item) {
  return item && STOPPABLE_STATUSES.has(item.status);
}

function executionPhase(item) {
  if (!item) return "-";
  return item.phase || item.status || "unknown";
}

function isRuntimeDebuggable(runtime) {
  return Boolean(
    runtime &&
      typeof runtime === "object" &&
      (runtime.base_url || runtime.proxy_url || runtime.debug_gui_url || runtime.runtime_id),
  );
}

function getRuntimeEntries(runtimeInfo) {
  const runtimes = runtimeInfo?.runtimes || {};
  return Object.entries(runtimes).filter(([, runtime]) => isRuntimeDebuggable(runtime));
}

function Badge({ children, tone }) {
  return <span className={`badge ${tone || statusTone(children)}`}>{children ?? "-"}</span>;
}

function ExecutionStatusBadge({ item, outcome }) {
  const status = executionDisplayStatus(item, outcome);
  return <Badge tone={statusTone(status)}>{humanizeStatus(status)}</Badge>;
}

function IconButton({ children, icon: Icon, variant = "default", ...props }) {
  return (
    <button className={`icon-button ${variant}`} {...props}>
      {Icon ? <Icon size={15} /> : null}
      <span>{children}</span>
    </button>
  );
}

function SummaryCard({ label, value, icon: Icon, tone }) {
  return (
    <div className={`summary-card ${tone || ""}`}>
      <div className="summary-icon">{Icon ? <Icon size={18} /> : null}</div>
      <div>
        <div className="summary-label">{label}</div>
        <div className="summary-value">{value ?? "-"}</div>
      </div>
    </div>
  );
}

function Empty({ children }) {
  return <div className="empty">{children}</div>;
}

function CodeBlock({ value }) {
  return <pre className="code-block">{prettyJson(value)}</pre>;
}

function CursorPager({
  pageNumber,
  canPrevious,
  canNext,
  previous,
  next,
  pageSize,
  setPageSize,
}) {
  return (
    <div className="pagination-footer">
      {setPageSize ? (
        <select aria-label="page size" value={pageSize} onChange={(event) => setPageSize(Number(event.target.value))}>
          {[25, 50, 100, 250].map((size) => <option key={size} value={size}>{size} rows</option>)}
        </select>
      ) : null}
      <span>Page {pageNumber}</span>
      <button aria-label="previous page" disabled={!canPrevious} onClick={previous} title="Previous page">
        <ChevronLeft size={15} />
      </button>
      <button aria-label="next page" disabled={!canNext} onClick={next} title="Next page">
        <ChevronRight size={15} />
      </button>
    </div>
  );
}

function ViewSwitcher({ value, options, onChange }) {
  return (
    <div className="segmented-control workspace-switcher">
      {options.map(([key, label]) => (
        <button key={key} className={value === key ? "active" : ""} onClick={() => onChange(key)}>{label}</button>
      ))}
    </div>
  );
}

function IntegerConfigFields({ legend, fields, value, onChange }) {
  return (
    <fieldset className="config-fieldset">
      <legend>{legend}</legend>
      <div className="config-grid">
        {fields.map((field) => (
          <label key={field.key}>
            <span>{field.label}</span>
            <input
              type="number"
              min={field.displayMin ?? field.min}
              step={field.displayStep ?? 1}
              value={value[field.key] ?? ""}
              onChange={(event) => onChange({ ...value, [field.key]: event.target.value })}
            />
          </label>
        ))}
      </div>
    </fieldset>
  );
}

function App() {
  const [route, setRoute] = useState(() => parseConsoleLocation(window.location.pathname, window.location.search));
  const detailTab = route.query?.view === "subtasks" || route.query?.tab === "items" ? "items" : route.query?.tab || "overview";
  const setDetailTab = (tab) => updateRouteQuery(route, { tab, view: undefined }, { replace: true });
  const [selectedExecutionId, setSelectedExecutionId] = useState(null);
  const [selectedExecution, setSelectedExecution] = useState(null);
  const [executionLoad, setExecutionLoad] = useState({});
  const [controlEvents, setControlEvents] = useState([]);
  const [liveEvents, setLiveEvents] = useState([]);
  const [liveEventFilter, setLiveEventFilter] = useState("");
  const [eventStreamPaused, setEventStreamPaused] = useState(false);
  const [liveEventStatus, setLiveEventStatus] = useState("");
  const [liveHistoryLoading, setLiveHistoryLoading] = useState(false);
  const [liveHistoryReloadToken, setLiveHistoryReloadToken] = useState(0);
  const [autoScrollEvents, setAutoScrollEvents] = useState(true);
  const [eventSourceExecutionId, setEventSourceExecutionId] = useState(null);
  const [commandPayload, setCommandPayload] = useState('{"type":"input.submit","payload":{"text":"Alpha"}}');
  const [createOpen, setCreateOpen] = useState(false);
  const [environments, setEnvironments] = useState([]);
  const [providerConfigs, setProviderConfigs] = useState([]);
  const [environmentAdminView, setEnvironmentAdminView] = useState("environments");
  const [myEnvironments, setMyEnvironments] = useState([]);
  const [environmentCatalog, setEnvironmentCatalog] = useState([]);
  const [traceDatabaseAccounts, setTraceDatabaseAccounts] = useState([]);
  const [traceDatabaseAdminAccounts, setTraceDatabaseAdminAccounts] = useState([]);
  const [myRunTraceBackends, setMyRunTraceBackends] = useState({
    items: [],
    claimable_sinks: [],
    required_sinks: [],
  });
  const [runTraceBackendCatalog, setRunTraceBackendCatalog] = useState([]);
  const [runTraceBackendGrants, setRunTraceBackendGrants] = useState({ principalId: "", items: [] });
  const [runTraceBackendPrincipalId, setRunTraceBackendPrincipalId] = useState("");
  const [issuedTraceCredential, setIssuedTraceCredential] = useState(null);
  const [environmentGrants, setEnvironmentGrants] = useState([]);
  const [configOverrides, setConfigOverrides] = useState({ config_overrides: [], secret_overrides: [] });
  const [me, setMe] = useState(null);
  const [accessUsers, setAccessUsers] = useState([]);
  const [principalAccessGrants, setPrincipalAccessGrants] = useState([]);
  const [accessView, setAccessView] = useState("users");
  const [executionResult, setExecutionResult] = useState(null);
  const [error, setError] = useState("");
  const liveEventsRef = useRef(null);
  const liveEventIdsRef = useRef(new Set());
  const terminalEventSourceIdsRef = useRef(new Set());
  const loadedLiveHistoryKeyRef = useRef("");
  const eventSourceRef = useRef(null);
  const selectedExecutionIdRef = useRef(null);
  const runTraceBackendPrincipalIdRef = useRef("");

  const currentExecution = route.section === "execution" && selectedExecution?.execution_id === route.executionId ? selectedExecution : null;
  const isAdmin = Boolean(me?.roles?.includes("admin"));
  const tab = route.section;
  const liveEventItems = useMemo(
    () => liveEvents.filter((item) => !liveEventFilter || item.type === liveEventFilter),
    [liveEvents, liveEventFilter]
  );

  const showError = useCallback((message) => {
    if (message?.name === "AbortError") return;
    setError(message instanceof Error ? message.message : String(message));
  }, []);

  const navigate = useCallback(navigateConsole, []);

  const closeLiveEventSource = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setEventSourceExecutionId(null);
  }, []);

  const resetLiveEvents = useCallback(() => {
    setLiveEvents([]);
    setLiveEventStatus("");
    setLiveHistoryLoading(false);
    liveEventIdsRef.current.clear();
    loadedLiveHistoryKeyRef.current = "";
    closeLiveEventSource();
  }, [closeLiveEventSource]);

  const selectExecution = useCallback(
    (executionId) => {
      if (selectedExecutionIdRef.current === executionId) return;
      selectedExecutionIdRef.current = executionId;
      setSelectedExecutionId(executionId);
      setSelectedExecution(null);
      setControlEvents([]);
      setExecutionResult(null);
      resetLiveEvents();
    },
    [resetLiveEvents]
  );

  const openExecution = useCallback(
    (executionId, fromBatch = null) => {
      selectExecution(executionId);
      navigate({ section: "execution", executionId, query: { return_to: window.location.pathname + window.location.search } });
    },
    [navigate, selectExecution]
  );

  const refreshConfigOverrides = useCallback(async ({ signal } = {}) => {
    const payload = await request("GET", controlPath("/v1/system/config-overrides"), undefined, { signal });
    setConfigOverrides(payload);
  }, []);

  const saveConfigOverride = useCallback(async (payload) => {
    const result = await request("PUT", controlPath("/v1/system/config-overrides"), payload);
    await refreshConfigOverrides();
    return result;
  }, [refreshConfigOverrides]);

  const removeConfigOverride = useCallback(async (payload) => {
    const result = await request("DELETE", controlPath("/v1/system/config-overrides"), payload);
    await refreshConfigOverrides();
    return result;
  }, [refreshConfigOverrides]);

  const refreshMe = useCallback(async () => {
    const payload = await request("GET", controlPath("/v1/me"));
    csrfToken = payload.csrf_token || "";
    setNavigationIdentity(payload.principal_id);
    setMe(payload);
  }, []);

  const refreshAccess = useCallback(async ({ signal } = {}) => {
    const usersPayload = await request("GET", controlPath("/v1/access/users"), undefined, { signal });
    setAccessUsers(usersPayload.items || []);
  }, []);

  const refreshPrincipalAccess = useCallback(async ({ signal } = {}) => {
    const payload = await request("GET", controlPath("/v1/access/principal-grants"), undefined, { signal });
    setPrincipalAccessGrants(payload.items || []);
  }, []);

  const putPrincipalAccessGrant = useCallback(async (ownerPrincipalId, granteePrincipalId) => {
    const operation = buildPrincipalGrantRequest(ownerPrincipalId, granteePrincipalId, "PUT");
    await request(operation.method, controlPath(operation.path));
    await refreshPrincipalAccess();
  }, [refreshPrincipalAccess]);

  const deletePrincipalAccessGrant = useCallback(async (ownerPrincipalId, granteePrincipalId) => {
    const operation = buildPrincipalGrantRequest(ownerPrincipalId, granteePrincipalId, "DELETE");
    await request(operation.method, controlPath(operation.path));
    await refreshPrincipalAccess();
  }, [refreshPrincipalAccess]);

  const refreshTraceDatabaseAdminAccounts = useCallback(async ({ signal } = {}) => {
    if (!isAdmin) return;
    const payload = await request(
      "GET",
      controlPath("/v1/access/trace-database-accounts"),
      undefined,
      { signal },
    );
    setTraceDatabaseAdminAccounts(payload.items || []);
  }, [isAdmin]);

  const refreshRunTraceBackendCatalog = useCallback(async ({ signal } = {}) => {
    if (!isAdmin) return;
    const payload = await request("GET", controlPath("/v1/admin/run-trace-backends"), undefined, { signal });
    setRunTraceBackendCatalog(payload.items || []);
  }, [isAdmin]);

  const selectRunTraceBackendPrincipal = useCallback((principalId) => {
    runTraceBackendPrincipalIdRef.current = principalId;
    setRunTraceBackendPrincipalId(principalId);
    setRunTraceBackendGrants({ principalId: "", items: [] });
  }, []);

  const refreshPrincipalRunTraceBackendGrants = useCallback(async (principalId, { signal } = {}) => {
    if (!isAdmin || !principalId) {
      setRunTraceBackendGrants({ principalId: "", items: [] });
      return;
    }
    const payload = await request(
      "GET",
      controlPath(`/v1/admin/principals/${encodeURIComponent(principalId)}/run-trace-backends`),
      undefined,
      { signal },
    );
    if (runTraceBackendPrincipalIdRef.current === principalId) {
      setRunTraceBackendGrants({ principalId, items: payload.items || [] });
    }
  }, [isAdmin]);

  const replacePrincipalRunTraceBackendGrants = useCallback(async (principalId, draft) => {
    const payload = await request(
      "PUT",
      controlPath(`/v1/admin/principals/${encodeURIComponent(principalId)}/run-trace-backends`),
      buildRunTraceBackendGrantPayload(draft),
    );
    if (runTraceBackendPrincipalIdRef.current === principalId) {
      setRunTraceBackendGrants({ principalId, items: payload.items || [] });
    }
  }, []);

  const updateUserRoles = useCallback(async (user, roles) => {
    await request("PUT", controlPath(`/v1/access/users/${encodeURIComponent(user.principal_id)}/roles`), {
      issuer: user.issuer,
      tenant: user.tenant,
      subject: user.subject,
      display_name: user.display_name,
      roles,
    });
    await refreshAccess();
  }, [refreshAccess]);

  const updateUserAppBinding = useCallback(async (user, appId) => {
    const operation = buildUserAppBindingRequest(user.principal_id, appId);
    await request(operation.method, controlPath(operation.path), operation.body);
    await refreshAccess();
  }, [refreshAccess]);

  const refreshSelectedExecution = useCallback(async (executionId = selectedExecutionIdRef.current, { signal, summaryOnly = false } = {}) => {
    if (!executionId) return;
    setExecutionLoad(previous => ({...previous, id: executionId, loading: true, error: null}));
    try {
      const payload = await request("GET", controlPath(`/v1/executions/${encodeURIComponent(executionId)}${summaryOnly ? "?summary=1" : ""}`), undefined, {signal});
      if (signal?.aborted || selectedExecutionIdRef.current !== executionId) return;
      setSelectedExecution(previous => summaryOnly ? mergeExecutionSummary(previous, payload) : payload);
      setExecutionLoad({id:executionId,loading:false,error:null,updatedAt:Date.now()});
    } catch (error) {
      if (selectedExecutionIdRef.current === executionId && !signal?.aborted) setExecutionLoad(previous => ({...previous,id:executionId,loading:false,error}));
      throw error;
    }
  }, []);

  const refreshControlEvents = useCallback(
    async (executionId = selectedExecutionIdRef.current) => {
      if (!executionId) {
        setControlEvents([]);
        return;
      }
      const payload = await request(
        "GET",
        controlPath(`/v1/executions/${encodeURIComponent(executionId)}/control-events?cursor=0&limit=200`)
      );
      if (selectedExecutionIdRef.current !== executionId) return;
      setControlEvents(payload.items || []);
    },
    []
  );

  const loadExecutionResult = useCallback(async (item = currentExecution) => {
    if (!item) return;
    const payload = await request(
      "GET",
      applicationPath(`/v1/executions/${encodeURIComponent(item.execution_id)}/result`)
    );
    if (selectedExecutionIdRef.current === item.execution_id) setExecutionResult(payload);
  }, [currentExecution]);

  const refreshEnvironments = useCallback(async ({ signal } = {}) => {
    const payload = await request("GET", controlPath("/v1/environments"), undefined, { signal });
    setEnvironments(payload.items || []);
  }, []);

  const refreshProviderConfigs = useCallback(async ({ signal } = {}) => {
    const payload = await request("GET", controlPath("/v1/provider-configs"), undefined, { signal });
    setProviderConfigs(payload.items || []);
  }, []);

  const refreshMyEnvironments = useCallback(async ({ signal } = {}) => {
    const [catalogPayload, myPayload, traceBackendPayload] = await Promise.all([
      request("GET", controlPath("/v1/environment-catalog"), undefined, { signal }),
      request("GET", controlPath("/v1/my-environments"), undefined, { signal }),
      request("GET", controlPath("/v1/access/run-trace-backends"), undefined, { signal }),
    ]);
    const catalog = catalogPayload.items || [];
    setEnvironmentCatalog(catalog);
    setMyEnvironments((myPayload.items || []).filter((item) => item.status === "enabled" && item.grant?.enabled));
    setTraceDatabaseAccounts(dedupeTraceDatabaseAccounts(myPayload.trace_database_accounts || []));
    setMyRunTraceBackends({
      items: traceBackendPayload.items || [],
      claimable_sinks: traceBackendPayload.claimable_sinks || [],
      required_sinks: traceBackendPayload.required_sinks || [],
    });
  }, []);

  const grantRunTraceBackend = useCallback(async (backendName) => {
    await request(
      "PUT",
      controlPath(`/v1/access/run-trace-backends/${encodeURIComponent(backendName)}`),
    );
    await refreshMyEnvironments();
  }, [refreshMyEnvironments]);

  const issueTraceDatabaseCredential = useCallback(async (account, operation) => {
    const environmentId = eligibleTraceDatabaseEnvironmentId(myEnvironments);
    if (!environmentId) throw new Error("No granted environment is available for this database backend");
    const credential = await request(
      "POST",
      controlPath(`/v1/my-environments/${encodeURIComponent(environmentId)}/trace-database-credentials`),
      { backend_name: account.backend_name, operation },
    );
    setIssuedTraceCredential(credential);
    await refreshMyEnvironments();
  }, [myEnvironments, refreshMyEnvironments]);

  const manageTraceDatabaseAccount = useCallback(async (account, operation) => {
    if (!window.confirm(`${humanizeStatus(operation)} database account for ${account.owner_principal_id}?`)) return;
    const payload = await request(
      "POST",
      controlPath(
        `/v1/access/trace-database-accounts/${encodeURIComponent(account.owner_principal_id)}`
        + `/${encodeURIComponent(account.backend_name)}/${operation}`
      ),
    );
    if (operation === "reset") setIssuedTraceCredential(payload);
    await refreshTraceDatabaseAdminAccounts();
  }, [refreshTraceDatabaseAdminAccounts]);

  const claimDefaultQuota = useCallback(async (environmentId) => {
    await request("PUT", controlPath(`/v1/my-environments/${encodeURIComponent(environmentId)}/quota`));
    await refreshMyEnvironments();
  }, [refreshMyEnvironments]);

  const refreshEnvironmentGrants = useCallback(async ({ signal } = {}) => {
    const payload = await request("GET", controlPath("/v1/environment-grants"), undefined, { signal });
    setEnvironmentGrants(payload.items || []);
  }, []);

  const createEnvironment = useCallback(async (payload) => {
    await request("POST", controlPath("/v1/environments"), payload);
    await refreshEnvironments();
  }, [refreshEnvironments]);

  const updateEnvironment = useCallback(async (environmentId, payload) => {
    await request("PUT", controlPath(`/v1/environments/${encodeURIComponent(environmentId)}`), payload);
    await refreshEnvironments();
  }, [refreshEnvironments]);

  const createProviderConfig = useCallback(async (payload) => {
    const created = await request("POST", controlPath("/v1/provider-configs"), payload);
    await refreshProviderConfigs();
    return created;
  }, [refreshProviderConfigs]);

  const updateProviderConfig = useCallback(async (providerConfigRef, payload) => {
    const updated = await request(
      "PUT",
      controlPath(`/v1/provider-configs/${encodeURIComponent(providerConfigRef)}`),
      payload,
    );
    await refreshProviderConfigs();
    return updated;
  }, [refreshProviderConfigs]);

  const upsertEnvironmentGrant = useCallback(async (environmentId, principalId, limits, scopes, enabled) => {
    await request(
      "PUT",
      controlPath(
        `/v1/environments/${encodeURIComponent(environmentId)}/grants/${encodeURIComponent(principalId)}`
      ),
      { limits, scopes, enabled }
    );
    await Promise.all([refreshEnvironmentGrants(), refreshMyEnvironments()]);
  }, [refreshEnvironmentGrants, refreshMyEnvironments]);

  const stopExecution = useCallback(
    async (item) => {
      if (!item || !canStopExecution(item)) return;
      const queued = ["queued", "assigned"].includes(item.status);
      const message = queued
        ? `Stop queued execution ${item.execution_id}? It will be marked killed.`
        : `Send stop.request to execution ${item.execution_id}?`;
      if (!window.confirm(message)) return;
      await request("POST", controlPath(`/v1/executions/${encodeURIComponent(item.execution_id)}/stop`));
      await refreshSelectedExecution(item.execution_id);
    },
    [refreshSelectedExecution]
  );

  const activateExecution = useCallback(
    async (item) => {
      if (!item) return;
      await request("POST", controlPath(`/v1/executions/${encodeURIComponent(item.execution_id)}/activate`));
      await refreshSelectedExecution(item.execution_id);
    },
    [refreshSelectedExecution]
  );

  const submitTypedSubmission = useCallback(async (type, payload) => {
    const created = await request("POST", controlPath(`/v1/submissions/${encodeURIComponent(type)}`), payload);
    setCreateOpen(false);
    if (type === "batch") navigate({ section: "batch", batchId: created.execution_id });
    else openExecution(created.execution_id);
  }, [navigate, openExecution]);

  const submitCommand = useCallback(async () => {
    if (!currentExecution) return;
    if (!canSubmitLiveCommand(currentExecution)) {
      showError("Commands are only available while the live execution is running or waiting for input.");
      return;
    }
    let payload;
    try {
      payload = JSON.parse(commandPayload);
    } catch (err) {
      showError(`Invalid command JSON: ${err.message}`);
      return;
    }
    await request(
      "POST",
      applicationPath(`/v1/executions/${encodeURIComponent(currentExecution.execution_id)}/commands`),
      payload
    );
    await refreshSelectedExecution(currentExecution.execution_id);
    await refreshControlEvents(currentExecution.execution_id);
  }, [commandPayload, currentExecution, refreshControlEvents, refreshSelectedExecution, showError]);

  const appendLiveEvent = useCallback((event) => {
    setLiveEvents((items) => [...items.slice(-399), event]);
  }, []);

  const loadLiveEventHistory = useCallback(
    async (item, reloadToken) => {
      if (!item || liveEventReadMode(item, false) !== "history") return;
      const historyKey = `${item.execution_id}:${reloadToken}`;
      if (loadedLiveHistoryKeyRef.current === historyKey) return;
      closeLiveEventSource();
      setLiveHistoryLoading(true);
      setLiveEventStatus("Loading history...");
      try {
        const text = await requestText(
          "GET",
          applicationPath(`/v1/executions/${encodeURIComponent(item.execution_id)}/live/events?cursor=0`)
        );
        if (selectedExecutionIdRef.current !== item.execution_id) return;
        const events = parseSseEvents(text);
        liveEventIdsRef.current = new Set(events.map((event) => `${event.type}:${event.id}`).filter(Boolean));
        setLiveEvents(events.slice(-400));
        setLiveEventStatus(`History loaded: ${events.length} event${events.length === 1 ? "" : "s"}`);
        loadedLiveHistoryKeyRef.current = historyKey;
      } catch (err) {
        if (selectedExecutionIdRef.current !== item.execution_id) return;
        setLiveEvents([]);
        setLiveEventStatus(`History unavailable: ${err instanceof Error ? err.message : String(err)}`);
        loadedLiveHistoryKeyRef.current = historyKey;
      } finally {
        if (selectedExecutionIdRef.current === item.execution_id) {
          setLiveHistoryLoading(false);
        }
      }
    },
    [closeLiveEventSource]
  );

  const reloadLiveEventHistory = useCallback(() => {
    loadedLiveHistoryKeyRef.current = "";
    setLiveHistoryReloadToken((value) => value + 1);
  }, []);

  const connectLiveEvents = useCallback(
    (item) => {
      if (liveEventReadMode(item, eventStreamPaused) !== "stream") return;
      if (terminalEventSourceIdsRef.current.has(item.execution_id)) return;
      if (eventSourceExecutionId === item.execution_id && eventSourceRef.current) return;
      closeLiveEventSource();
      setLiveEvents([]);
      setLiveEventStatus("Streaming live events...");
      liveEventIdsRef.current.clear();
      setEventSourceExecutionId(item.execution_id);
      const source = new EventSource(
        applicationPath(`/v1/executions/${encodeURIComponent(item.execution_id)}/live/events?cursor=0`),
        { withCredentials: true }
      );
      eventSourceRef.current = source;
      for (const type of LIVE_EVENT_TYPES) {
        source.addEventListener(type, (event) => {
          if (eventSourceRef.current !== source || selectedExecutionIdRef.current !== item.execution_id) return;
          const eventId = event.lastEventId ? `${type}:${event.lastEventId}` : "";
          if (eventId) {
            if (liveEventIdsRef.current.has(eventId)) return;
            liveEventIdsRef.current.add(eventId);
          }
          if (isTerminalLiveEvent(type)) {
            terminalEventSourceIdsRef.current.add(item.execution_id);
            setLiveEventStatus("Execution finished. History can be reloaded from artifacts.");
            source.close();
            if (eventSourceRef.current === source) {
              eventSourceRef.current = null;
              setEventSourceExecutionId(null);
            }
          }
          appendLiveEvent({
            id: event.lastEventId || undefined,
            type,
            created_at: new Date().toISOString(),
            payload: safeParseEventData(event.data),
          });
        });
      }
      source.onerror = () => {
        if (eventSourceRef.current !== source || selectedExecutionIdRef.current !== item.execution_id) return;
        if (liveEventReadMode(item, false) === "history") {
          closeLiveEventSource();
          return;
        }
        appendLiveEvent({
          type: "stream.error",
          created_at: new Date().toISOString(),
          payload: { message: "event stream interrupted" },
        });
      };
    },
    [appendLiveEvent, closeLiveEventSource, eventSourceExecutionId, eventStreamPaused]
  );

  useEffect(() => {
    refreshMe().catch(showError);
    const onPopState = () => setRoute(parseConsoleLocation(window.location.pathname, window.location.search));
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [refreshMe, showError]);

  useEffect(() => restoreConsoleScroll(), [route]);
  useEffect(() => {
    if (window.innerWidth > 800) return;
    for (const selector of ["nav.tabs", ".workspace-switcher"]) {
      const bar = document.querySelector(selector), active = bar?.querySelector(".active");
      if (active && bar.scrollWidth > bar.clientWidth) bar.scrollLeft = Math.max(0, active.offsetLeft - bar.offsetLeft - 12);
    }
  }, [route.section, route.infrastructureView, isAdmin]);

  useEffect(() => {
    selectedExecutionIdRef.current = selectedExecutionId;
  }, [selectedExecutionId]);

  useEffect(() => {
    if (!me || isAdmin || !["access", "environments", "infrastructure", "system"].includes(route.section)) return;
    navigate({ section: "runs" }, { replace: true });
  }, [isAdmin, me, navigate, route.section]);

  useEffect(() => {
    const executionId = route.section === "execution" ? route.executionId : null;
    if (!executionId) return undefined;
    selectExecution(executionId);
    const controller = new AbortController();
    refreshSelectedExecution(executionId, { signal: controller.signal }).catch(showError);
    return () => controller.abort();
  }, [refreshSelectedExecution, route.section, route.executionId, selectExecution, showError]);

  const pollingExecutionId = route.section === "execution" ? route.executionId : null;
  const pollExecutionStatus = Boolean(pollingExecutionId && selectedExecution?.execution_id === pollingExecutionId && canStopExecution(selectedExecution));
  useEffect(() => {
    if (!pollExecutionStatus) return undefined;
    const poll = startVisiblePolling(
      ({ signal }) => refreshSelectedExecution(pollingExecutionId, { summaryOnly: true, signal }),
      { interval: currentExecution?.spec_type === "nexus_batch" ? 10000 : 3000, onError: (error) => { if (error) showError(error); } },
    );
    return () => poll.stop();
  }, [pollExecutionStatus, pollingExecutionId, refreshSelectedExecution, currentExecution?.spec_type, showError]);

  useEffect(() => {
    if (route.section !== "execution" || detailTab !== "control" || !route.executionId) return;
    refreshControlEvents(route.executionId).catch(showError);
  }, [detailTab, refreshControlEvents, route.executionId, route.section, showError]);

  useEffect(() => {
    const controller = new AbortController();
    const options = { signal: controller.signal };
    let operation = null;
    if (tab === "my-environments") operation = refreshMyEnvironments(options);
    if (tab === "shared-access") {
      operation = Promise.all([refreshPrincipalAccess(options), refreshAccess(options)]);
    }
    if (tab === "environments") {
      operation = environmentAdminView === "provider-configs"
        ? refreshProviderConfigs(options)
        : Promise.all([refreshEnvironments(options), refreshProviderConfigs(options)]);
    }
    if (tab === "infrastructure") {
      operation = refreshEnvironments(options);
    }
    if (tab === "system") operation = Promise.all([
      refreshEnvironments(options),
      refreshConfigOverrides(options),
    ]);
    if (tab === "access") {
      if (accessView === "users") operation = refreshAccess(options);
      else if (accessView === "grants") {
        operation = Promise.all([refreshAccess(options), refreshEnvironments(options), refreshEnvironmentGrants(options)]);
      } else if (accessView === "trace-backends") {
        operation = Promise.all([
          refreshAccess(options),
          refreshRunTraceBackendCatalog(options),
          refreshPrincipalRunTraceBackendGrants(runTraceBackendPrincipalId, options),
        ]);
      } else {
        operation = Promise.all([refreshAccess(options), refreshTraceDatabaseAdminAccounts(options)]);
      }
    }
    if (operation) operation.catch(showError);
    return () => controller.abort();
  }, [
    refreshAccess,
    refreshEnvironmentGrants,
    refreshEnvironments,
    refreshMyEnvironments,
    refreshProviderConfigs,
    refreshPrincipalAccess,
    refreshPrincipalRunTraceBackendGrants,
    refreshRunTraceBackendCatalog,
    refreshConfigOverrides,
    refreshTraceDatabaseAdminAccounts,
    accessView,
    environmentAdminView,
    isAdmin,
    route.infrastructureView,
    runTraceBackendPrincipalId,
    showError,
    tab,
  ]);

  useEffect(() => {
    if (!createOpen) return undefined;
    const controller = new AbortController();
    refreshMyEnvironments({ signal: controller.signal }).catch(showError);
    return () => controller.abort();
  }, [createOpen, refreshMyEnvironments, showError]);

  useEffect(() => {
    if (detailTab === "live" && currentExecution?.mode === "live") {
      const mode = liveEventReadMode(currentExecution, eventStreamPaused);
      if (mode === "stream") {
        connectLiveEvents(currentExecution);
      } else if (mode === "history") {
        loadLiveEventHistory(currentExecution, liveHistoryReloadToken).catch(showError);
      } else if (mode === "paused") {
        setLiveEventStatus("Event stream paused.");
      }
    }
  }, [
    connectLiveEvents,
    currentExecution,
    detailTab,
    eventStreamPaused,
    liveHistoryReloadToken,
    loadLiveEventHistory,
    showError,
  ]);

  useEffect(() => {
    if (eventStreamPaused) closeLiveEventSource();
  }, [closeLiveEventSource, eventStreamPaused]);

  useEffect(() => {
    setIssuedTraceCredential(null);
  }, [route.section]);

  useEffect(() => {
    if (autoScrollEvents && liveEventsRef.current) {
      liveEventsRef.current.scrollTop = liveEventsRef.current.scrollHeight;
    }
  }, [autoScrollEvents, liveEventItems]);

  useEffect(
    () => () => {
      if (eventSourceRef.current) eventSourceRef.current.close();
    },
    []
  );

  const refreshCurrentView = () => {
    window.dispatchEvent(new Event("refresh-console-resources"));
    if (route.section === "runs") return Promise.resolve();
    if (route.section === "execution") {
      return Promise.all([
        refreshSelectedExecution(route.executionId),
      ]);
    }
    if (["builds", "build"].includes(route.section)) { window.dispatchEvent(new Event("refresh-builds")); return Promise.resolve(); }
    if (route.section === "my-environments") return refreshMyEnvironments();
    if (route.section === "shared-access") {
      return Promise.all([refreshPrincipalAccess(), refreshAccess()]);
    }
    if (route.section === "access") {
      if (accessView === "users") return refreshAccess();
      if (accessView === "grants") {
        return Promise.all([refreshAccess(), refreshEnvironments(), refreshEnvironmentGrants()]);
      }
      if (accessView === "trace-backends") {
        return Promise.all([
          refreshAccess(),
          refreshRunTraceBackendCatalog(),
          refreshPrincipalRunTraceBackendGrants(runTraceBackendPrincipalId),
        ]);
      }
      return Promise.all([refreshAccess(), refreshTraceDatabaseAdminAccounts()]);
    }
    if (route.section === "environments") {
      return environmentAdminView === "provider-configs"
        ? refreshProviderConfigs()
        : Promise.all([refreshEnvironments(), refreshProviderConfigs()]);
    }
    if (route.section === "infrastructure") {
      return Promise.resolve();
    }
    if (route.section === "system") {
      window.dispatchEvent(new Event("refresh-system"));
      return Promise.resolve();
    }
    return Promise.resolve();
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">Nexus Distributed</div>
          <h1>{routeTitle(route)}</h1>
        </div>
        <div className="topbar-actions">
          <UserGuideWelcome key={me?.principal_id || "anonymous"} principalId={me?.principal_id}
            enabled={CONFIG.welcomeAnimationEnabled === true} />
          <IconButton icon={RefreshCcw} onClick={() => refreshCurrentView().catch(showError)}>
            Refresh
          </IconButton>
          {["runs", "batches"].includes(route.section) ? (
            <IconButton icon={ClipboardPlus} variant="primary" onClick={() => setCreateOpen((value) => !value)}>
              New Run
            </IconButton>
          ) : null}
        </div>
      </header>

      {error ? (
        <div className="alert">
          <AlertTriangle size={16} />
          <span>{error}</span>
          <button aria-label="dismiss error" onClick={() => setError("")}>
            <X size={15} />
          </button>
        </div>
      ) : null}

      <nav className="tabs" aria-label="console sections">
        {navigationItems(isAdmin).map(({ key, label, admin }) => {
          const Icon = NAVIGATION_ICONS[key];
          const diagnostic = ["build", "builds", "allocation"].includes(route.section);
          const diagnosticSection = route.query?.return_to ? (route.query.return_to.includes("/admin/infrastructure") ? "infrastructure" : "runs") : isAdmin ? "infrastructure" : "runs";
          const active = route.section === key || (key === "runs" && route.section === "execution") || (diagnostic && key === diagnosticSection);
          return (
          <button
            key={key}
            className={`${active ? "tab active" : "tab"}${admin ? " admin-tab" : ""}`}
            data-tab={key}
            onClick={() => navigate(navigationRoute(key))}
          >
            <Icon size={16} />
            <span>{label}</span>
          </button>
          );
        })}
      </nav>

      {route.section === "builds" ? <BuildsView request={request} controlPath={controlPath} route={route} /> : null}
      {route.section === "build" ? <BuildDetail key={route.buildId} request={request} controlPath={controlPath} route={route} isAdmin={isAdmin} renderLogs={build => <ExecutionManagerLogsTab item={{}} buildId={route.buildId} buildMetadata={build} />} /> : null}
      {route.section === "allocation" ? <AllocationDetail request={request} controlPath={controlPath} route={route} isAdmin={isAdmin} /> : null}

      {createOpen ? (
        <SubmissionPanel
          environments={myEnvironments}
          initialType={route.query?.type === "batch" ? "batch" : "single"}
          submit={submitTypedSubmission}
          close={() => setCreateOpen(false)}
          onError={showError}
        />
      ) : null}

      {tab === "runs" ? <RunsWorkspace route={route} stop={item => stopExecution(item).catch(showError)} activate={item => activateExecution(item).catch(showError)} /> : null}

      {tab === "execution" ? (
        <main className={`panel single-view detail-page ${detailTab === "items" ? "subtasks-page" : ""}`}>
          <ResourceBack route={route} fallback={currentExecution?.batch_context ? { section: "execution", executionId: currentExecution.batch_context.parent_execution_id, query: { tab: "items", item_id: currentExecution.batch_context.item_id } } : savedListRoute("runs")} label="Back to source" />
          {currentExecution?.batch_context ? <div className="resource-scope">Batch <ConsoleLink origin={false} to={{section:"execution",executionId:currentExecution.batch_context.parent_execution_id,query:{tab:"items",item_id:currentExecution.batch_context.item_id}}}>{currentExecution.batch_context.parent_execution_id}</ConsoleLink> · Item {currentExecution.batch_context.item_id || "Unknown"} · Attempt {displayAttempt(currentExecution.batch_context.attempt_no)}</div> : null}
          <ResourceNotice query={executionLoad.id === route.executionId ? {...executionLoad, stale: Boolean(currentExecution && executionLoad.error)} : {loading:true}} noun="Run details" />
          <ExecutionDetail
            item={currentExecution}
            route={route}
            loadState={executionLoad}
            isAdmin={isAdmin}
            batchContent={currentExecution?.spec_type === "nexus_batch" ? <BatchPanel route={route} parent={currentExecution} stop={() => stopExecution(currentExecution).catch(showError)} /> : null}
            detailTab={detailTab}
            setDetailTab={setDetailTab}
            controlEvents={controlEvents}
            liveEvents={liveEventItems}
            liveEventFilter={liveEventFilter}
            setLiveEventFilter={setLiveEventFilter}
            eventStreamPaused={eventStreamPaused}
            setEventStreamPaused={setEventStreamPaused}
            liveEventStatus={liveEventStatus}
            liveHistoryLoading={liveHistoryLoading}
            liveEventMode={liveEventReadMode(currentExecution, eventStreamPaused)}
            reloadLiveEventHistory={reloadLiveEventHistory}
            clearLiveEvents={() => {
              liveEventIdsRef.current.clear();
              setLiveEvents([]);
            }}
            autoScrollEvents={autoScrollEvents}
            setAutoScrollEvents={setAutoScrollEvents}
            liveEventsRef={liveEventsRef}
            commandPayload={commandPayload}
            setCommandPayload={setCommandPayload}
            submitCommand={() => submitCommand().catch(showError)}
            stopExecution={() => stopExecution(currentExecution).catch(showError)}
            activateExecution={() => activateExecution(currentExecution).catch(showError)}
            executionResult={executionResult}
            loadExecutionResult={() => loadExecutionResult().catch(showError)}
            openBatch={currentExecution?.spec_type === "nexus_batch"
              ? () => setDetailTab("items")
              : null}
          />
        </main>
      ) : null}

      {tab === "my-environments" ? (
        <MyEnvironmentsView
          environments={environmentCatalog}
          runTraceBackends={myRunTraceBackends}
          traceDatabaseAccounts={traceDatabaseAccounts}
          grantRunTraceBackend={(backendName) => grantRunTraceBackend(backendName).catch(showError)}
          claimQuota={(environmentId) => claimDefaultQuota(environmentId).catch(showError)}
          issueTraceDatabaseCredential={(account, operation) => (
            issueTraceDatabaseCredential(account, operation).catch(showError)
          )}
          refresh={() => refreshMyEnvironments().catch(showError)}
        />
      ) : null}
      {tab === "shared-access" ? (
        <PrincipalAccessView
          me={me}
          users={accessUsers}
          grants={principalAccessGrants}
          putGrant={putPrincipalAccessGrant}
          deleteGrant={deletePrincipalAccessGrant}
          onError={showError}
          refresh={() => Promise.all([refreshPrincipalAccess(), refreshAccess()]).catch(showError)}
        />
      ) : null}
      {tab === "environments" ? (
        <div className="workspace-stack">
          <ViewSwitcher
            value={environmentAdminView}
            options={[["environments", "Environments"], ["provider-configs", "Provider configs"]]}
            onChange={setEnvironmentAdminView}
          />
          {environmentAdminView === "provider-configs" ? (
            <ProviderConfigsView
              providerConfigs={providerConfigs}
              createProviderConfig={createProviderConfig}
              updateProviderConfig={updateProviderConfig}
              onError={showError}
              refresh={() => refreshProviderConfigs().catch(showError)}
            />
          ) : (
            <EnvironmentsView
              environments={environments}
              providerConfigs={providerConfigs}
              createEnvironment={createEnvironment}
              updateEnvironment={updateEnvironment}
              onError={showError}
              refresh={() => Promise.all([refreshEnvironments(), refreshProviderConfigs()]).catch(showError)}
            />
          )}
        </div>
      ) : null}
      {tab === "access" ? (
        <div className="workspace-stack">
          <ViewSwitcher
            value={accessView}
            options={[
              ["users", "Users"],
              ["grants", "Grants"],
              ["trace-backends", "Trace backends"],
              ["database-accounts", "Database accounts"],
            ]}
            onChange={setAccessView}
          />
          {accessView === "users" ? (
            <AccessView
              me={me}
              users={accessUsers}
              updateUserRoles={(user, roles) => updateUserRoles(user, roles).catch(showError)}
              updateUserAppBinding={(user, appId) => updateUserAppBinding(user, appId).catch(showError)}
              refresh={() => refreshAccess().catch(showError)}
            />
          ) : accessView === "grants" ? (
            <GrantsView
              grants={environmentGrants}
              environments={environments}
              users={accessUsers}
              upsertGrant={upsertEnvironmentGrant}
              onError={showError}
              refresh={() => refreshEnvironmentGrants().catch(showError)}
            />
          ) : accessView === "trace-backends" ? (
            <RunTraceBackendGrantsView
              catalog={runTraceBackendCatalog}
              grants={runTraceBackendGrants.items}
              grantsPrincipalId={runTraceBackendGrants.principalId}
              principalId={runTraceBackendPrincipalId}
              users={accessUsers}
              selectPrincipal={selectRunTraceBackendPrincipal}
              save={(principalId, draft) => (
                replacePrincipalRunTraceBackendGrants(principalId, draft).catch(showError)
              )}
              refresh={() => Promise.all([
                refreshRunTraceBackendCatalog(),
                refreshPrincipalRunTraceBackendGrants(runTraceBackendPrincipalId),
              ]).catch(showError)}
            />
          ) : (
            <DatabaseAccountsView
              accounts={traceDatabaseAdminAccounts}
              users={accessUsers}
              manage={(account, operation) => manageTraceDatabaseAccount(account, operation).catch(showError)}
              refresh={() => refreshTraceDatabaseAdminAccounts().catch(showError)}
            />
          )}
        </div>
      ) : null}
      {tab === "infrastructure" ? <div className="workspace-stack">
        <ViewSwitcher value={route.infrastructureView || "managers"} options={[["managers", "Managers"], ["allocations", "Allocations"], ["builds", "Advanced · Builds"]]} onChange={view => navigate(savedListRoute("infrastructure", view))} />
        {route.infrastructureView === "allocations" ? <AllocationList isAdmin={isAdmin} request={request} controlPath={controlPath} route={route} /> : route.infrastructureView === "builds" ? <BuildsView request={request} controlPath={controlPath} route={route} /> : <ManagerInventory requestText={requestText} request={request} controlPath={controlPath} route={route} environments={environments} />}
      </div> : null}
      {tab === "system" ? (
        <SystemView
          environments={environments}
          configOverrides={configOverrides}
          refreshSystem={() => refreshConfigOverrides().catch(showError)}
          saveConfigOverride={saveConfigOverride}
          removeConfigOverride={removeConfigOverride}
          onError={showError}
        />
      ) : null}
      {issuedTraceCredential ? (
        <TraceDatabaseCredentialModal
          credential={issuedTraceCredential}
          close={() => setIssuedTraceCredential(null)}
        />
      ) : null}
    </div>
  );
}

function RunsWorkspace({ route, stop, activate }) {
  const q = route.query || {};
  const params = new URLSearchParams({
    summary: "1",
    top_level: "1",
    limit: q.page_size || "50",
  });
  for (const key of ["q", "status", "cursor", "mode"])
    if (q[key]) params.set(key, q[key]);
  if (q.type) params.set("spec_type", `nexus_${q.type}`);
  const path = controlPath(`/v1/executions?${params}`);
  const query = useResourceQuery(
    path,
    (signal) => request("GET", path, undefined, { signal }),
    { interval: 10000 },
  );
  return (
    <main className="panel single-view">
      <div className="panel-header">
        <div>
          <h2>Runs</h2>
          <span className="muted">Single, Live and Batch submissions</span>
        </div>
        <RefreshStatus query={query} interval={10} />
      </div>
      <div className="filters run-filters">
        <input
          aria-label="Search runs"
          placeholder="Run ID / manager ID"
          value={q.q || ""}
          onChange={(e) =>
            updateRouteQuery(
              route,
              { q: e.target.value },
              { resetPage: true, replace: true },
            )
          }
        />
        <select
          aria-label="Run type"
          value={q.type || ""}
          onChange={(e) =>
            updateRouteQuery(
              route,
              { type: e.target.value, mode: undefined },
              { resetPage: true },
            )
          }
        >
          <option value="">All types</option>
          {["single", "live", "batch"].map((t) => (
            <option key={t}>{t}</option>
          ))}
        </select>
        <select
          aria-label="Run status"
          value={q.status || ""}
          onChange={(e) =>
            updateRouteQuery(
              route,
              { status: e.target.value },
              { resetPage: true },
            )
          }
        >
          <option value="">All statuses</option>
          {[
            "queued",
            "assigned",
            "starting",
            "running",
            "waiting_input",
            "stopping",
            "succeeded",
            "failed",
            "killed",
            "lost",
            "expired",
          ].map((t) => (
            <option key={t}>{t}</option>
          ))}
        </select>

      </div>
      <ResourceNotice query={query} noun="runs" />
      {query.data ? (
        <ExecutionTable
          items={query.data.items || []}
          emptyMessage={query.error ? "No executions" : ["q", "status", "type", "mode", "cursor"].some((key) => q[key]) ? (
            "No runs match these filters or page."
          ) : (
            <>
              <div>还没有任务？按照使用指引完成环境准备并提交第一个任务。</div>
              <UserGuideLink />
            </>
          )}
          onStop={async (item) => {
            await stop(item);
            query.refresh();
          }}
          onActivate={async (item) => {
            await activate(item);
            query.refresh();
          }}
        />
      ) : null}
      <ResourcePager
        pageSize
        route={route}
        nextCursor={query.data?.next_cursor}
        loading={query.loading}
      />
    </main>
  );
}

function RunResources({ item, isAdmin, compact = false, openResources, openSubtasks }) {
  const path = controlPath(
    `/v1/executions/${encodeURIComponent(item.execution_id)}/allocations`,
  );
  const resources = useResourceQuery(
    path,
    (signal) => request("GET", path, undefined, { signal }),
    { interval: canStopExecution(item) ? 5000 : 0 },
  );
  const batchPath = controlPath(
    `/v1/executions/${encodeURIComponent(item.execution_id)}/batch?include_status_counts=true`,
  );
  const batch = useResourceQuery(
    batchPath,
    (signal) => request("GET", batchPath, undefined, { signal }),
    {
      enabled: item.spec_type === "nexus_batch",
      interval: canStopExecution(item) ? 10000 : 0,
    },
  );
  return (
    <>
      <ResourceNotice query={resources} noun="Run resources" />
      {item.spec_type === "nexus_batch" ? (
        <>
          <ResourceNotice query={batch} noun="Batch progress" />
          {compact && batch.data ? <section className="overview-progress"><div className="section-heading"><h3>Subtask progress</h3><button className="row-action" onClick={openSubtasks}>View subtasks →</button></div><div className="progress-metrics">{Object.entries(batch.data.item_status_counts || batch.data.summary || {}).map(([status,count]) => <div key={status}><span className="muted">{humanizeStatus(status)}</span><strong>{count}</strong></div>)}</div></section> : null}
        </>
      ) : null}
      <RunPreparation
        allocations={resources.data?.items || []}
        batchSummary={batch.data}
        isAdmin={isAdmin}
        compact={compact}
        openResources={openResources}
      />
      {resources.data &&
      !resources.data.items?.length &&
      !batch.data?.child_dependency_build_id ? (
        <p className="empty">No allocations recorded for this Run.</p>
      ) : null}
    </>
  );
}

function BatchPanel({ route, parent, stop }) {
  const q = route.query || {},
    active = canStopExecution(parent);
  const root = controlPath(
    `/v1/executions/${encodeURIComponent(route.executionId)}/batch`,
  );
  const summary = useResourceQuery(
    root,
    (signal) =>
      request("GET", root + "?include_status_counts=true", undefined, {
        signal,
      }),
    { interval: active ? 10000 : 0 },
  );
  const params = new URLSearchParams({ limit: "100" });
  for (const [url, api] of [
    ["item_status", "status"],
    ["item_prefix", "item_id_prefix"],
    ["child_execution_id", "child_execution_id"],
    ["error_code", "error_code"],
    ["cursor", "cursor"],
  ])
    if (q[url]) params.set(api, q[url]);
  const listPath = `${root}/items?${params}`;
  const items = useResourceQuery(
    listPath,
    (signal) => request("GET", listPath, undefined, { signal }),
    { interval: active ? 5000 : 0 },
  );
  const itemPath = q.item_id
    ? `${root}/items/${encodeURIComponent(q.item_id)}`
    : "";
  const selected = useResourceQuery(
    itemPath,
    (signal) => request("GET", itemPath, undefined, { signal }),
    { enabled: Boolean(itemPath), interval: active ? 5000 : 0 },
  );
  useEffect(() => {
    if (!selected.data?.item_id || window.innerWidth > 1100) return;
    const timer = setTimeout(() => document.querySelector(".batch-item-drawer")?.scrollIntoView({block:"start",behavior:"smooth"}), 100);
    return () => clearTimeout(timer);
  }, [selected.data?.item_id]);
  const filterKey = JSON.stringify([
    q.item_status,
    q.item_prefix,
    q.child_execution_id,
    q.error_code,
  ]);
  const values = () => ({
    status: q.item_status || "",
    itemId: q.item_prefix || "",
    childExecutionId: q.child_execution_id || "",
    errorCode: q.error_code || "",
  });
  const [draft, setDraft] = useState(values);
  useEffect(() => setDraft(values()), [filterKey]);
  const openChild = (executionId) =>
    navigateConsole({
      section: "execution",
      executionId,
      query: { return_to: consolePath(route) },
    });
  return (
    <BatchWorkspace
      route={route}
      parent={parent}
      summary={summary.data}
      items={items.data?.items || []}
      filter={draft}
      setFilter={setDraft}
      applyFilter={() =>
        updateRouteQuery(
          route,
          {
            item_status: draft.status,
            item_prefix: draft.itemId,
            child_execution_id: draft.childExecutionId,
            error_code: draft.errorCode,
            item_id: undefined,
          },
          { resetPage: true },
        )
      }
      refresh={() => {
        summary.refresh();
        items.refresh();
        selected.refresh();
      }}
      selectedItem={selected.data}
      selectItem={(item_id) => updateRouteQuery(route, { item_id })}
      closeItem={() => updateRouteQuery(route, { item_id: undefined })}
      openChild={openChild}
      stop={stop}
      notice={
        <>
          <ResourceNotice query={summary} noun="Batch progress" />
          <ResourceNotice query={items} noun="subtasks" />
          {q.item_id ? (
            <ResourceNotice query={selected} noun="selected subtask" />
          ) : null}
          {items.data && !items.data.items.length ? (
            <p className="empty">No subtasks match these filters.</p>
          ) : null}
        </>
      }
      pager={
        <ResourcePager
          route={route}
          nextCursor={items.data?.next_cursor}
          loading={items.loading}
        />
      }
    />
  );
}

function SubmissionPanel({ environments, initialType, submit, close, onError }) {
  const [draft, setDraft] = useState(() => defaultSubmissionDraft(initialType, environments[0]?.environment_id || ""));
  const [rawPayload, setRawPayload] = useState("{}");
  const selectedEnvironment = environments.find((item) => item.environment_id === draft.environmentId) || null;
  const predefined = selectedEnvironment?.allocation_type === "predefined";
  const dynamicResourcesReady = Boolean(
    draft.managerCpuCores &&
    draft.managerMemoryGiB &&
    (draft.type === "batch" ? (
      draft.maxInflight &&
      draft.batchManagerCpuCores &&
      draft.batchManagerMemoryGiB
    ) : draft.managerSlots)
  );
  const updateDraft = (field, value) => setDraft((current) => ({ ...current, [field]: value }));

  useEffect(() => {
    if (!draft.environmentId && environments.length) {
      updateDraft("environmentId", environments[0].environment_id);
    }
  }, [draft.environmentId, environments]);

  useEffect(() => {
    if (!selectedEnvironment) return;
    if (predefined) {
      setDraft((current) => ({
        ...current,
        codeMode: "config_only",
        pool: current.pool || selectedEnvironment.pools?.[0] || "",
      }));
      return;
    }
    setDraft((current) => ({
      ...current,
      pool: "",
      managerCpuCores: current.managerCpuCores || "1",
      managerMemoryGiB: current.managerMemoryGiB || "2",
      managerSlots: current.type === "batch" ? current.managerSlots : current.managerSlots || "20",
      batchManagerCpuCores: current.batchManagerCpuCores || "1",
      batchManagerMemoryGiB: current.batchManagerMemoryGiB || "4",
    }));
  }, [draft.type, predefined, selectedEnvironment?.environment_id]);

  const changeType = (type) => {
    setDraft((current) => changeSubmissionType(current, type));
  };
  const submitForm = () => submit(draft.type, buildSubmission(draft));
  const prepareRawPayload = () => {
    try {
      const payload = buildSubmission(draft);
      setRawPayload(JSON.stringify(payload, null, 2));
    } catch (error) {
      onError(error);
    }
  };
  const submitRaw = () => {
    let payload;
    try {
      payload = JSON.parse(rawPayload);
    } catch (error) {
      throw new Error(`Submission JSON must be valid JSON: ${error.message}`);
    }
    if (!payload || Array.isArray(payload) || typeof payload !== "object") {
      throw new Error("Submission JSON must be an object");
    }
    return submit(draft.type, payload);
  };

  return (
    <section id="new-submission" className="panel submission-panel">
      <div className="panel-header compact">
        <div>
          <h2>New Submission</h2>
          <span className="muted">{selectedEnvironment?.name || "Select an environment"}</span>
        </div>
        <button className="icon-only-button" aria-label="close submission" title="Close" onClick={close}>
          <X size={16} />
        </button>
      </div>
      <div className="segmented-control" aria-label="submission type">
        {["single", "live", "batch"].map((type) => (
          <button key={type} className={draft.type === type ? "active" : ""} onClick={() => changeType(type)}>
            {type[0].toUpperCase() + type.slice(1)}
          </button>
        ))}
      </div>
      <div className="action-form submission-form">
        <label>
          <span>Environment</span>
          <select value={draft.environmentId} onChange={(event) => updateDraft("environmentId", event.target.value)}>
            <option value="">Select environment</option>
            {environments.map((environment) => (
              <option key={environment.environment_id} value={environment.environment_id}>
                {environment.name || environment.environment_id}
              </option>
            ))}
          </select>
        </label>
        {predefined ? (
          <label>
            <span>Pool</span>
            <input value={draft.pool} onChange={(event) => updateDraft("pool", event.target.value)} />
          </label>
        ) : null}
        {draft.type === "batch" ? (
          <label><span>Max inflight</span><input type="number" min="1" required value={draft.maxInflight} onChange={(event) => updateDraft("maxInflight", event.target.value)} /></label>
        ) : null}
        {draft.type === "batch" && !predefined ? (
          <label><span>Execution managers (optional)</span><input type="number" min="1" placeholder="Inferred" value={draft.managerCount} onChange={(event) => updateDraft("managerCount", event.target.value)} /></label>
        ) : null}
        {!predefined ? (
          <>
            <label><span>Execution CPU cores / manager</span><input type="number" min={CPU_CORES_INPUT.min} step={CPU_CORES_INPUT.step} value={draft.managerCpuCores} onChange={(event) => updateDraft("managerCpuCores", event.target.value)} /></label>
            <label><span>Execution memory GiB / manager</span><input type="number" min={MEMORY_GIB_INPUT.min} step={MEMORY_GIB_INPUT.step} value={draft.managerMemoryGiB} onChange={(event) => updateDraft("managerMemoryGiB", event.target.value)} /></label>
            <label><span>{draft.type === "batch" ? "Execution slots / manager (optional)" : "Execution slots / manager"}</span><input type="number" min="1" placeholder={draft.type === "batch" ? "20 default" : undefined} value={draft.managerSlots} onChange={(event) => updateDraft("managerSlots", event.target.value)} /></label>
          </>
        ) : null}
        {draft.type === "batch" ? (
          <label className="json-field full-field">
            <span>Generation</span>
            <textarea value={draft.generation} onChange={(event) => updateDraft("generation", event.target.value)} />
          </label>
        ) : (
          <label className="json-field full-field">
            <span>Run input</span>
            <textarea value={draft.runInput} onChange={(event) => updateDraft("runInput", event.target.value)} />
          </label>
        )}
      </div>
      <details className="advanced-section">
        <summary>Advanced overrides</summary>
        <div className="action-form submission-form">
          <label><span>Idempotency key</span><input value={draft.idempotencyKey} onChange={(event) => updateDraft("idempotencyKey", event.target.value)} /></label>
          <label>
            <span>Code</span>
            <select disabled={predefined} value={draft.codeMode} onChange={(event) => updateDraft("codeMode", event.target.value)}>
              <option value="config_only">config only</option>
              <option value="image">workload image</option>
              <option value="source_package">source package</option>
            </select>
          </label>
          {draft.codeMode === "image" ? (
            <label><span>Workload image version</span><input value={draft.imageVersion} onChange={(event) => updateDraft("imageVersion", event.target.value)} /></label>
          ) : null}
          {draft.codeMode === "source_package" ? (
            <label className="json-field full-field"><span>Code artifacts</span><textarea value={draft.codeJson} onChange={(event) => updateDraft("codeJson", event.target.value)} /></label>
          ) : null}
          <label className="json-field full-field">
            <span>Host environment (one KEY=VALUE per line)</span>
            <textarea
              disabled={predefined}
              placeholder={predefined ? "Only available for dynamic environments" : "HYTUNER=1\nMY_FLAG=value"}
              value={draft.hostEnv}
              onChange={(event) => updateDraft("hostEnv", event.target.value)}
            />
          </label>
          {draft.type === "live" ? (
            <label className="json-field full-field"><span>Live config</span><textarea value={draft.liveConfig} onChange={(event) => updateDraft("liveConfig", event.target.value)} /></label>
          ) : null}
          {draft.type === "batch" ? (
            <>
              <label><span>Orchestration CPU cores</span><input type="number" min={CPU_CORES_INPUT.min} step={CPU_CORES_INPUT.step} disabled={predefined} value={draft.batchManagerCpuCores} onChange={(event) => updateDraft("batchManagerCpuCores", event.target.value)} /></label>
              <label><span>Orchestration memory GiB</span><input type="number" min={MEMORY_GIB_INPUT.min} step={MEMORY_GIB_INPUT.step} disabled={predefined} value={draft.batchManagerMemoryGiB} onChange={(event) => updateDraft("batchManagerMemoryGiB", event.target.value)} /></label>
              <label className="json-field full-field"><span>Batch policy</span><textarea value={draft.batchPolicy} onChange={(event) => updateDraft("batchPolicy", event.target.value)} /></label>
            </>
          ) : null}
          <TimeoutFields draft={draft} updateDraft={updateDraft} />
        </div>
      </details>
      <details className="advanced-section" onToggle={(event) => event.currentTarget.open && prepareRawPayload()}>
        <summary>Raw submission JSON</summary>
        <div className="raw-submit-editor">
          <textarea value={rawPayload} spellCheck={false} onChange={(event) => setRawPayload(event.target.value)} />
          <IconButton icon={Send} onClick={() => Promise.resolve().then(submitRaw).catch(onError)}>Submit JSON</IconButton>
        </div>
      </details>
      <div className="submission-actions">
        <IconButton
          icon={Send}
          variant="primary"
          disabled={!draft.environmentId || (draft.type === "batch" && !draft.maxInflight) || (predefined ? !draft.pool : !dynamicResourcesReady)}
          onClick={() => Promise.resolve().then(submitForm).catch(onError)}
        >
          Submit {draft.type}
        </IconButton>
      </div>
    </section>
  );
}

function TimeoutFields({ draft, updateDraft }) {
  const fields = [
    ["provisioningTimeoutSeconds", "Provisioning timeout"],
    ["queueTtlSeconds", "Queue TTL"],
    ["executionTtlSeconds", "Execution TTL"],
    ["stopTimeoutSeconds", "Stop timeout"],
  ];
  return (
    <fieldset className="config-fieldset">
      <legend>Timeouts</legend>
      <div className="config-grid">
        {fields.map(([key, label]) => (
          <label key={key}><span>{label} (seconds)</span><input type="number" min="1" value={draft[key]} onChange={(event) => updateDraft(key, event.target.value)} /></label>
        ))}
      </div>
    </fieldset>
  );
}

function BatchWorkspace({
  route,
  parent,
  summary,
  items,
  pager,
  notice,
  filter,
  setFilter,
  applyFilter,
  refresh,
  loadMore,
  selectedItem,
  selectItem,
  closeItem,
  openChild,
  stop,
  back,
}) {
  const counts = summary?.summary || summary?.item_status_counts || {};
  const total = summary?.total_items ?? summary?.materialized_count ?? 0;
  const terminal = ["succeeded", "failed", "killed", "lost", "expired", "cancelled"]
    .reduce((sum, status) => sum + (counts[status] ?? 0), 0);
  const active = summary?.item_status_counts
    ? (counts.submitted ?? 0) + (counts.running ?? 0)
    : (summary?.active_items ?? 0);
  const pending = summary?.item_status_counts
    ? (counts.pending ?? 0)
    : Math.max(0, total - terminal - active);
  const progress = total > 0 ? Math.min(100, Math.round((terminal / total) * 100)) : 0;
  return (
    <section className="batch-workspace">
      <div className="subtasks-toolbar"><span className="muted">{summary ? `${total.toLocaleString()} subtasks · Generation ${summary.generation_status || "unknown"}` : "Loading subtasks"}</span></div>
      {summary ? <section className="batch-summary-band">
        <div className="progress-block">
          <div className="progress-heading"><span>Completed</span><strong>{terminal} / {total}</strong></div>
          <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
        </div>
        {[
          ["Pending", pending],
          ["Active", active],
          ["Succeeded", counts.succeeded ?? 0],
          ["Failed", counts.failed ?? 0],
          ["Killed", counts.killed ?? 0],
          ["Lost", counts.lost ?? 0],
          ["Expired", counts.expired ?? 0],
          ["Cancelled", counts.cancelled ?? 0],
        ].map(([label, value]) => <KeyValue key={label} label={label} value={value} />)}
      </section> : null}
      {notice}
      <div className={`batch-content ${selectedItem ? "has-drawer" : ""}`}>
        <section className="batch-items">
          <div className="filters batch-filters">
            <select value={filter.status} onChange={(event) => setFilter((current) => ({ ...current, status: event.target.value }))}>
              <option value="">all status</option>
              {["pending", "submitted", "running", "succeeded", "failed", "killed", "lost", "expired", "cancelled"].map((status) => <option key={status} value={status}>{status}</option>)}
            </select>
            <input placeholder="item ID prefix" value={filter.itemId} onChange={(event) => setFilter((current) => ({ ...current, itemId: event.target.value }))} />
            <input placeholder="child execution ID" value={filter.childExecutionId} onChange={(event) => setFilter((current) => ({ ...current, childExecutionId: event.target.value }))} />
            <input placeholder="error code" value={filter.errorCode} onChange={(event) => setFilter((current) => ({ ...current, errorCode: event.target.value }))} />
            <IconButton icon={ListFilter} onClick={applyFilter}>Apply</IconButton>
          </div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Item</th><th>Status</th><th>Attempt</th><th>Child execution</th></tr></thead>
              <tbody>{items.map((item) => (
                <tr key={item.item_id} className={selectedItem?.item_id === item.item_id ? "selected" : ""}>
                  <td><ConsoleLink className="row-action" origin={false} preserveScroll to={{...route,query:{...route.query,item_id:item.item_id}}}>{item.item_id}</ConsoleLink></td>
                  <td><Badge>{item.status}</Badge></td>
                  <td>{displayAttempt(item.current_attempt_no)}</td>
                  <td>{item.current_child_execution_id ? <ConsoleLink className="row-action" to={{section:"execution",executionId:item.current_child_execution_id,query:{return_to:consolePath({...route,query:{...route.query,item_id:item.item_id}})}}}>{shortId(item.current_child_execution_id)}</ConsoleLink> : "-"}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          {pager}
        </section>
        {selectedItem ? <BatchItemDrawer item={selectedItem} close={closeItem} openChild={openChild} /> : null}
      </div>
    </section>
  );
}

function BatchItemDrawer({ item, close, openChild }) {
  return (
    <aside className="batch-item-drawer">
      <div className="panel-header compact">
        <div><h2>{item.item_id}</h2><Badge>{item.status}</Badge></div>
        <button className="icon-only-button" aria-label="close item detail" title="Close" onClick={close}><X size={16} /></button>
      </div>
      <div className="section-grid compact-grid">
        <KeyValue label="ordinal" value={item.task_ordinal} />
        <KeyValue label="attempt" value={displayAttempt(item.current_attempt_no)} />
        <KeyValue label="spec sha256" value={item.spec_sha256} />
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Attempt</th><th>Status</th><th>Execution</th><th>Error</th></tr></thead>
          <tbody>{(item.attempts || []).map((attempt) => (
            <tr key={attempt.attempt_no}>
              <td>{displayAttempt(attempt.attempt_no)}</td>
              <td><Badge>{attempt.status}</Badge></td>
              <td>{attempt.child_execution_id ? <ConsoleLink className="row-action" to={{section:"execution",executionId:attempt.child_execution_id}}>{shortId(attempt.child_execution_id)}</ConsoleLink> : "-"}</td>
              <td><div>{attempt.error_code || "-"}</div><div className="muted">{attempt.error_message || ""}</div></td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <details className="advanced-section"><summary>Item payload</summary><CodeBlock value={item} /></details>
    </aside>
  );
}

function ExecutionTable({ items, selectedExecutionId, statusLabel = "Status", onSelect, onStop, onActivate, emptyMessage = "No executions" }) {
  if (!items.length) return <Empty>{emptyMessage}</Empty>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Execution</th>
            <th>Owner</th>
            <th>{statusLabel}</th>

            <th>Type</th>
            <th>Manager</th>
            <th>Slots</th>
            <th>State updated</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr
              key={item.execution_id}
              className={selectedExecutionId === item.execution_id ? "selected" : ""}

            >
              <td>
                <ConsoleLink className="primary-cell" to={{section:"execution",executionId:item.execution_id}}>{shortId(item.execution_id)}</ConsoleLink>
                <div className="muted">{item.shard_id == null ? "-" : `shard ${item.shard_id}`}</div>
              </td>
              <td><code title={item.owner_principal_id}>{shortId(item.owner_principal_id)}</code></td>
              <td>
                <ExecutionStatusBadge item={item} />
              </td>

              <td>
                <div>{item.spec_type === "nexus_batch" ? "Batch" : item.mode === "live" ? "Live" : "Single"}</div>
              </td>
              <td>{shortId(item.current_assignment?.manager_id)}</td>
              <td>{item.slot_cost || 1}</td>
              <td>{formatTime(item.updated_at || item.next_schedule_at)}</td>
              <td>
                <div className="row-actions">
                  {item.route_status === "cached" ? (
                    <button className="row-action" onClick={(event) => actionClick(event, () => onActivate(item))}>
                      Activate
                    </button>
                  ) : null}
                  {canStopExecution(item) ? (
                    <button className="row-action danger" onClick={(event) => actionClick(event, () => onStop(item))}>
                      Stop
                    </button>
                  ) : null}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function actionClick(event, fn) {
  event.preventDefault();
  event.stopPropagation();
  fn();
}

function ExecutionDetail(props) {
  const { item, detailTab, setDetailTab, route } = props;
  if (!item) return null;
  const isBatch = item.spec_type === "nexus_batch";
  const isSubtasks = isBatch && detailTab === "items";
  const tabs = [
    ["overview", "Overview"],
    ["manager-logs", "Logs"],
    ["control", "Events"],
    ["runtime", "Runtime"],
    ["resources", "Resources"],
    ["spec", "Spec / Result"],
    ...(item.mode === "live" ? [["live", "Live Events"]] : []),
  ];
  const switchView = (view) =>
    updateRouteQuery(
      route,
      {
        view: view === "subtasks" ? view : undefined,
        tab: route.query?.tab === "items" ? "overview" : route.query?.tab,
      },
      { replace: true },
    );
  const type = isBatch ? "Batch" : item.mode === "live" ? "Live" : "Single";
  const phase = executionPhase(item);
  const error = item.error_json;
  const errorText =
    typeof error === "string"
      ? error
      : error?.message || error?.error_message || error?.code;
  return (
    <>
      <div className="detail-head run-heading">
        <div className="run-identity">
          <div className="run-title">
            <h2 title={item.execution_id}>{item.execution_id}</h2>
            <CopyId value={item.execution_id} buttonOnly />
          </div>
          <div className="detail-subtitle">
            <ExecutionStatusBadge item={item} />
            <span className="run-type">{type}</span>
            {item.route_status && item.route_status !== "none" ? (
              <Badge>{item.route_status}</Badge>
            ) : null}
            {phase && phase !== item.status ? (
              <span>{humanizeStatus(phase)}</span>
            ) : null}
          </div>
        </div>
        <div className="detail-actions">
          <RefreshStatus
            query={props.loadState}
            interval={canStopExecution(item) ? (isBatch ? 10 : 3) : 0}
          />
          {item.route_status === "cached" ? (
            <IconButton icon={Play} onClick={props.activateExecution}>
              Activate
            </IconButton>
          ) : null}
          {canStopExecution(item) ? (
            <IconButton
              icon={CircleStop}
              variant="danger"
              onClick={props.stopExecution}
            >
              Stop Run
            </IconButton>
          ) : null}
        </div>
      </div>
      {item.status === "failed" ? (
        <div className="run-failure" role="status">
          <AlertTriangle size={16} />
          <span>
            {typeof errorText === "string" && errorText
              ? errorText
              : "This Run failed. Inspect its logs and preparation details."}
          </span>
          <button
            className="row-action"
            onClick={() => setDetailTab("manager-logs")}
          >
            Logs
          </button>
          <button
            className="row-action"
            onClick={() => setDetailTab("resources")}
          >
            Resources
          </button>
        </div>
      ) : null}
      {isBatch ? (
        <div
          className="run-primary-tabs"
          role="tablist"
          aria-label="Batch views"
          onKeyDown={(event) => {
            const direction =
              event.key === "ArrowRight" || event.key === "End"
                ? "subtasks"
                : event.key === "ArrowLeft" || event.key === "Home"
                  ? "main"
                  : null;
            if (!direction) return;
            event.preventDefault();
            switchView(direction);
            event.currentTarget
              .querySelectorAll("button")
              [direction === "main" ? 0 : 1].focus();
          }}
        >
          <button
            role="tab"
            tabIndex={isSubtasks ? -1 : 0}
            aria-selected={!isSubtasks}
            onClick={() => switchView("main")}
          >
            Main
          </button>
          <button
            role="tab"
            tabIndex={isSubtasks ? 0 : -1}
            aria-selected={isSubtasks}
            onClick={() => switchView("subtasks")}
          >
            Subtasks
          </button>
        </div>
      ) : null}
      {!isSubtasks ? (
        <div
          className="detail-tabs run-secondary-tabs"
          aria-label="Run sections"
        >
          {tabs.map(([key, label]) => (
            <button
              key={key}
              data-detail-tab={key}
              className={detailTab === key ? "active" : ""}
              aria-current={detailTab === key ? "page" : undefined}
              onClick={() => setDetailTab(key)}
            >
              {label}
            </button>
          ))}
        </div>
      ) : null}
      <div
        className={
          detailTab === "live"
            ? "detail-body live-body"
            : isSubtasks
              ? "detail-body subtasks-body"
              : "detail-body"
        }
      >
        {detailTab === "overview" ? (
          <OverviewTab
            item={item}
            isAdmin={props.isAdmin}
            openResources={() => setDetailTab("resources")}
            openSubtasks={() => switchView("subtasks")}
          />
        ) : null}
        {isSubtasks ? props.batchContent : null}
        {detailTab === "resources" ? (
          <RunResources item={item} isAdmin={props.isAdmin} />
        ) : null}
        {detailTab === "control" ? (
          <ControlEventsTab events={props.controlEvents} />
        ) : null}
        {detailTab === "live" ? <LiveEventsTab item={item} {...props} /> : null}
        {detailTab === "runtime" ? <RuntimeTab item={item} /> : null}
        {detailTab === "manager-logs" ? (
          <ExecutionManagerLogsTab item={item} />
        ) : null}
        {detailTab === "spec" ? (
          <SpecTab
            item={item}
            loadedResult={props.executionResult}
            loadResult={props.loadExecutionResult}
          />
        ) : null}
      </div>
    </>
  );
}

function OverviewTab({ item, isAdmin, openResources, openSubtasks }) {
  const assignment = item.current_assignment || {};
  return (
    <div className="run-overview">
      <dl className="run-facts">
        {[
          [
            "Owner",
            <span title={item.owner_principal_id}>
              {shortId(item.owner_principal_id)}{" "}
              <CopyId value={item.owner_principal_id} buttonOnly />
            </span>,
          ],
          ["Environment", item.environment_id || item.placement?.environment],
          ["Attempt", `${item.attempt_count || 0} / ${item.max_attempts || 1}`],
          ["Created", formatTime(item.created_at)],
          ["State updated", formatTime(item.updated_at)],
        ].map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value || "—"}</dd>
          </div>
        ))}
      </dl>
      <RunResources
        item={item}
        isAdmin={isAdmin}
        compact
        openResources={openResources}
        openSubtasks={openSubtasks}
      />
      <details className="scheduling-details">
        <summary>Scheduling details</summary>
        <div className="section-grid">
          <KeyValue
            label="Assignment"
            value={assignment.assignment_id || "—"}
          />
          <KeyValue
            label="Attempt ID"
            value={item.current_attempt_id || assignment.attempt_id || "—"}
          />
          <KeyValue label="Manager" value={assignment.manager_id || "—"} />
          <KeyValue
            label="Endpoint ready"
            value={String(Boolean(assignment.endpoint_ready))}
          />
          <KeyValue
            label="Next schedule"
            value={formatTime(item.next_schedule_at)}
          />
          <KeyValue
            label="Stop timeout"
            value={`${item.stop_timeout_seconds || 60}s`}
          />
        </div>
        <CodeBlock value={item.placement || {}} />
      </details>
    </div>
  );
}

function KeyValue({ label, value }) {
  return (
    <div className="kv">
      <div className="key">{label}</div>
      <div className="value">{value}</div>
    </div>
  );
}

function ControlEventsTab({ events }) {
  if (!events.length) return <Empty>No control events</Empty>;
  return (
    <div className="event-list">
      {events.map((event) => (
        <EventItem key={event.seq || event.sequence || `${event.type}-${event.created_at}`} type={event.type || event.event_type} event={event} />
      ))}
    </div>
  );
}

function LiveEventsTab({
  item,
  liveEvents,
  liveEventFilter,
  setLiveEventFilter,
  eventStreamPaused,
  setEventStreamPaused,
  liveEventStatus,
  liveHistoryLoading,
  liveEventMode,
  reloadLiveEventHistory,
  clearLiveEvents,
  autoScrollEvents,
  setAutoScrollEvents,
  liveEventsRef,
  commandPayload,
  setCommandPayload,
  submitCommand,
}) {
  if (item.mode !== "live") return <Empty>Live events are available for live executions</Empty>;
  const commandEnabled = canSubmitLiveCommand(item);
  const historyMode = liveEventMode === "history";
  return (
    <div className="live-layout">
      <div className="toolbar dense">
        <select id="event-type-filter" value={liveEventFilter} onChange={(event) => setLiveEventFilter(event.target.value)}>
          <option value="">all event types</option>
          {LIVE_EVENT_TYPES.map((type) => (
            <option key={type} value={type}>
              {type}
            </option>
          ))}
        </select>
        {historyMode ? (
          <button id="reload-live-history" disabled={liveHistoryLoading} onClick={reloadLiveEventHistory}>
            <RefreshCcw size={14} />
            <span>{liveHistoryLoading ? "Loading" : "Reload history"}</span>
          </button>
        ) : (
          <button id="toggle-event-stream" onClick={() => setEventStreamPaused(!eventStreamPaused)}>
            {eventStreamPaused ? <Play size={14} /> : <Pause size={14} />}
            <span>{eventStreamPaused ? "Resume" : "Pause"}</span>
          </button>
        )}
        <button id="clear-live-events" onClick={clearLiveEvents}>
          Clear
        </button>
        <Badge tone={historyMode ? "neutral" : "good"}>{historyMode ? "History" : liveEventMode}</Badge>
        <label className="check">
          <input type="checkbox" checked={autoScrollEvents} onChange={(event) => setAutoScrollEvents(event.target.checked)} />
          auto scroll
        </label>
      </div>
      <div className="live-status">{liveEventStatus || " "}</div>
      <div className="command-box">
        <div className="command-head">
          <SquareTerminal size={15} />
          <span>Command</span>
        </div>
        <textarea
          value={commandPayload}
          spellCheck={false}
          disabled={!commandEnabled}
          onChange={(event) => setCommandPayload(event.target.value)}
        />
        <IconButton icon={Send} variant="primary" disabled={!commandEnabled} onClick={submitCommand}>
          {commandEnabled ? "Submit Command" : "Command disabled"}
        </IconButton>
        {!commandEnabled ? (
          <div className="command-disabled">Commands are only available while the live execution is running or waiting for input.</div>
        ) : null}
      </div>
      <div className="event-list live-events" ref={liveEventsRef}>
        {liveEvents.length ? liveEvents.map((event, index) => <EventItem key={`${event.type}-${index}`} type={event.type} event={event} />) : <Empty>No live events</Empty>}
      </div>
    </div>
  );
}

function EventItem({ type, event }) {
  const open = type !== "trace.event";
  return (
    <details className="event-item" open={open}>
      <summary>
        <span className="event-type">{type || "event"}</span>
        <span className="event-time">{formatTime(event.created_at)}</span>
      </summary>
      <CodeBlock value={event.payload || event.payload_json || event} />
    </details>
  );
}

function RuntimeTab({ item }) {
  const assignment = item.current_assignment || {};
  const runtimeInfo = assignment.runtime_info || item.runtime_info || {};
  const runtimeEntries = getRuntimeEntries(runtimeInfo);
  return (
    <div className="runtime-tab">
      <div className="section-grid">
        <KeyValue label="endpoint" value={assignment.endpoint_url || "-"} />
        <KeyValue label="endpoint ready" value={String(Boolean(assignment.endpoint_ready))} />
        <KeyValue label="assignment status" value={assignment.status || "-"} />
        <KeyValue label="manager" value={assignment.manager_id || "-"} />
      </div>
      {item.mode === "live" && runtimeEntries.length ? (
        <div className="runtime-links">
          {runtimeEntries.map(([name]) => (
            <a
              key={name}
              href={applicationPath(`/v1/executions/${encodeURIComponent(item.execution_id)}/runtimes/${encodeURIComponent(name)}/proxy`)}
              target="_blank"
              rel="noreferrer"
            >
              <Radio size={15} />
              <span>{name} debug</span>
            </a>
          ))}
        </div>
      ) : null}
      <CodeBlock value={runtimeInfo} />
    </div>
  );
}

function ExecutionManagerLogsTab({ item, buildId = null, buildMetadata = null }) {
  const buildLogs = Boolean(buildId);
  const [lines, setLines] = useState(200);
  const [logPayload, setLogPayload] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [retryAttempt, setRetryAttempt] = useState(0);
  const requestSeqRef = useRef(0);
  const abortRef = useRef(null);
  const retryTimerRef = useRef(null);

  const refreshLogs = useCallback(async (automatic = false) => {
    const requestSeq = ++requestSeqRef.current;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    if (!automatic) setRetryAttempt(0);
    setLoading(true);
    setLoadError("");
    try {
      const payload = await request(
        "GET",
        controlPath(
          buildId ? `/v1/builds/${encodeURIComponent(buildId)}/logs?lines=${lines}`
            : `/v1/executions/${encodeURIComponent(item.execution_id)}/logs?lines=${lines}`
        ), undefined, { signal: controller.signal }
      );
      if (requestSeqRef.current === requestSeq) setLogPayload(payload);
    } catch (error) {
      if (requestSeqRef.current === requestSeq) {
        setLogPayload(null);
        setLoadError(error instanceof Error ? error.message : String(error));
      }
    } finally {
      if (requestSeqRef.current === requestSeq) setLoading(false);
    }
  }, [item.execution_id, item.status, lines, buildLogs, buildId, buildMetadata?.attempt, buildMetadata?.status]);

  useEffect(() => {
    refreshLogs();
    return () => {
      requestSeqRef.current += 1;
      abortRef.current?.abort();
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
    };
  }, [refreshLogs]);

  useEffect(() => {
    const pending = buildLogs ? logPayload?.items?.some((entry) => entry.availability === "ingest_pending")
      : logPayload?.availability === "ingest_pending";
    const activeBuild = buildLogs && (buildMetadata ? ["requested", "building"].includes(buildMetadata.status) : logPayload?.items?.some((entry) => ["requested", "building"].includes(entry.status)));
    const retryScheduled = buildLogs && buildMetadata?.status === "failed" && buildMetadata.next_retry_at;
    const delay = activeBuild ? 3000 : retryScheduled ? 10000 : executionLogRetryDelay(pending ? "ingest_pending" : null, retryAttempt);
    if (delay === null || loading) return undefined;
    const poll = () => {
      if (document.hidden) { retryTimerRef.current = setTimeout(poll, delay); return; }
      if (!activeBuild) setRetryAttempt((attempt) => attempt + 1);
      refreshLogs(true);
    };
    retryTimerRef.current = setTimeout(poll, delay);
    return () => clearTimeout(retryTimerRef.current);
  }, [logPayload, loading, refreshLogs, retryAttempt, buildLogs, item.status, buildId, buildMetadata?.status, buildMetadata?.next_retry_at]);

  useEffect(() => {
    if (!buildId) return undefined;
    const refresh = () => refreshLogs();
    window.addEventListener("refresh-builds", refresh);
    return () => window.removeEventListener("refresh-builds", refresh);
  }, [buildId, refreshLogs]);

  const entries = buildLogs ? (logPayload?.items || []) : [logPayload];
  const output = loading && !logPayload ? "Loading..." : loadError
    || (logPayload?.lines || []).map((line) => formatExecutionLogLine(line, formatLogTime)).join("\n")
    || executionLogStateText(logPayload);
  const buildOutput = entries.flatMap((entry) => entry?.lines || [])
    .map((line) => formatExecutionLogLine(line, formatLogTime)).join("\n");
  const source = buildLogs ? (entries.some((entry) => entry?.source === "zhiyan") ? "Zhiyan" : "Builder file") : executionLogSourceLabel(logPayload?.source);
  return (
    <section className="execution-manager-logs">
      <div className="manager-log-head">
        <div>
          <h3>{buildLogs ? "Build Logs" : "Logs"}</h3>
          <div className="muted">Source: {source}</div>
        </div>
        <div className="toolbar dense inline-toolbar">
          <select value={lines} onChange={(event) => setLines(Number(event.target.value))}>
            <option value={100}>100 lines</option>
            <option value={200}>200 lines</option>
            <option value={500}>500 lines</option>
            <option value={1000}>1000 lines</option>
          </select>
          <button disabled={loading} onClick={() => refreshLogs()}>
            <RefreshCcw size={14} />
            <span>{loading ? "Loading" : "Refresh"}</span>
          </button>
        </div>
      </div>
      {buildLogs ? <div className="build-log-status" role="status" aria-live="polite">
        {loadError ? <span>{loadError}</span> : entries.length ? entries.map((entry) => <div key={entry.dependency_build_id}>
          <div className="detail-subtitle"><Badge>{{ preparing: "Pod initializing", starting: "Container starting", finalizing: "Finalizing" }[entry.phase] || entry.phase || entry.status}</Badge><span>{entry.dependency_build_id} · attempt {entry.attempt}</span></div>
          {entry.availability === "ingest_pending" ? <div>Waiting for log ingestion</div> : null}
          {entry.availability === "unavailable" ? <div>{entry.error_code === "build_log_expired"
            ? "Build has finished. Local build logs are no longer available."
            : executionLogStateText(entry)}</div> : null}
        </div>) : <span>{loading ? "Loading build status..." : "No dependency builds yet"}</span>}
      </div> : null}
      <pre className="log-output">{buildLogs ? buildOutput : output}</pre>
    </section>
  );
}

function SpecTab({ item, loadedResult, loadResult }) {
  return (
    <div className="spec-grid">
      <div>
        <h3>Spec</h3>
        <CodeBlock value={item.spec} />
      </div>
      <div>
        <div className="section-heading">
          <h3>Result</h3>
          <IconButton icon={RefreshCcw} onClick={loadResult}>Load result</IconButton>
        </div>
        <CodeBlock value={loadedResult ?? item.result_json ?? item.error_json ?? {}} />
      </div>
    </div>
  );
}

function PrincipalIdentity({ principalId, user = null }) {
  const [copied, setCopied] = useState(false);
  const name = user ? (user.display_name || user.subject) : "Unknown user";
  const subjectLabel = user?.subject
    ? `${user.issuer === "tai" ? "StaffId" : "Subject"} ${user.subject}`
    : null;
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(principalId);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      window.prompt("Copy full principal", principalId);
    }
  };
  return (
    <div className="principal-identity">
      <div className="primary-cell">{name}</div>
      {subjectLabel ? <div className="muted">{subjectLabel}</div> : null}
      <div className="principal-value">
        <code title={principalId}>{principalId}</code>
        <button
          type="button"
          className="icon-only-button"
          aria-label="Copy full principal"
          title={copied ? "Copied" : "Copy full principal"}
          onClick={() => copy()}
        >
          <Copy size={14} />
        </button>
      </div>
    </div>
  );
}

function CopyableValue({ value, label, secret = false }) {
  const [copied, setCopied] = useState(false);
  const [revealed, setRevealed] = useState(!secret);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(String(value));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      window.prompt(`Copy ${label}`, String(value));
    }
  };
  return (
    <div className="copyable-value">
      <code title={String(value)}>{revealed ? value : "••••••••••••"}</code>
      {secret ? (
        <button
          type="button"
          className="icon-only-button"
          aria-label={revealed ? `Hide ${label}` : `Show ${label}`}
          title={revealed ? `Hide ${label}` : `Show ${label}`}
          onClick={() => setRevealed((current) => !current)}
        >
          {revealed ? <EyeOff size={14} /> : <Eye size={14} />}
        </button>
      ) : null}
      <button
        type="button"
        className="icon-only-button"
        aria-label={`Copy ${label}`}
        title={copied ? "Copied" : `Copy ${label}`}
        onClick={copy}
      >
        <Copy size={14} />
      </button>
    </div>
  );
}

function TraceDatabaseCredentialModal({ credential, close }) {
  const rows = [
    ["Host", credential.read_host],
    ["Port", credential.read_port],
    ["Database", credential.database],
    ["Username", credential.username],
    ["Password", credential.password, true],
    ["Trace table", credential.trace_table_name || credential.batch_table_name],
  ];
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget) close();
    }}>
      <section className="credential-modal" role="dialog" aria-modal="true" aria-labelledby="credential-modal-title">
        <div className="panel-header">
          <div>
            <h2 id="credential-modal-title">Trace database credentials</h2>
            <span className="muted">Shown once</span>
          </div>
          <button type="button" className="icon-only-button" aria-label="Close credentials" onClick={close}>
            <X size={16} />
          </button>
        </div>
        <div className="credential-fields">
          {rows.map(([label, value, secret]) => (
            <div className="credential-field" key={label}>
              <span>{label}</span>
              <CopyableValue value={value ?? "-"} label={label.toLowerCase()} secret={Boolean(secret)} />
            </div>
          ))}
        </div>
        <div className="modal-actions">
          <IconButton icon={X} onClick={close}>Close</IconButton>
        </div>
      </section>
    </div>
  );
}

function TraceDatabaseAccessTable({ accounts, issueCredential }) {
  if (!accounts.length) return null;
  return (
    <section className="panel-section trace-database-section">
      <div className="section-heading"><h3>Trace Database Access</h3></div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Backend</th><th>Connection</th><th>Username</th><th>Trace table</th><th>Scope</th><th>Status</th><th>Action</th></tr>
          </thead>
          <tbody>
            {accounts.map((account) => {
              const action = accountAction(account);
              return (
                <tr key={account.backend_name}>
                  <td><span className="primary-cell">{account.backend_name}</span></td>
                  <td><CopyableValue value={`${account.read_host}:${account.read_port}`} label="connection" /></td>
                  <td>{account.username ? <CopyableValue value={account.username} label="username" /> : "-"}</td>
                  <td><CopyableValue value={account.trace_table_name || account.batch_table_name} label="trace table" /></td>
                  <td>All trace tables</td>
                  <td><Badge tone={statusTone(account.status)}>{humanizeStatus(account.status)}</Badge></td>
                  <td>
                    {action ? (
                      <IconButton
                        icon={action === "create" ? Plus : RefreshCcw}
                        variant={action === "create" ? "primary" : "default"}
                        onClick={() => issueCredential(account, action)}
                      >
                        {action === "create" ? "Create credentials" : "Reset password"}
                      </IconButton>
                    ) : "-"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function MyRunTraceBackendsTable({ access, grantBackend }) {
  const items = runTraceBackendAccessRows(access);
  return (
    <section className="panel-section">
      <div className="section-heading"><h3>Run Trace Backends</h3></div>
      {items.length ? (
        <div className="table-wrap">
          <table>
            <thead><tr><th>Backend</th><th>Kind</th><th>Description</th><th>Default order</th><th>Access</th><th>Action</th></tr></thead>
            <tbody>{items.map((item) => (
              <tr key={item.name}>
                <td><div className="primary-cell">{item.display_name || item.name}</div><div className="muted">{item.name}</div></td>
                <td>{item.kind}</td>
                <td>{item.description || "-"}</td>
                <td>{item.default_order == null ? "-" : item.default_order}</td>
                <td>
                  <Badge tone={item.access === "granted" ? "good" : "info"}>{item.access}</Badge>
                </td>
                <td>
                  {item.access === "available" ? (
                    <IconButton icon={Plus} variant="primary" onClick={() => grantBackend(item.name)}>
                      Grant
                    </IconButton>
                  ) : "-"}
                </td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      ) : <Empty>No Run Trace Backends granted</Empty>}
    </section>
  );
}

function RunTraceBackendGrantsView({
  catalog,
  grants,
  grantsPrincipalId,
  principalId,
  users,
  selectPrincipal,
  save,
  refresh,
}) {
  const [draft, setDraft] = useState([]);
  const [pending, setPending] = useState(false);
  const catalogByName = new Map(catalog.map((backend) => [backend.name, backend]));
  const grantsLoaded = Boolean(principalId) && grantsPrincipalId === principalId;

  useEffect(() => {
    setDraft(mergeRunTraceBackendGrantDraft(catalog, grants, grantsPrincipalId, principalId));
  }, [catalog, grants, grantsPrincipalId, principalId]);

  const updateGrant = (backendName, patch) => {
    setDraft((current) => current.map((grant) => (
      grant.backend_name === backendName
        ? { ...grant, ...patch, ...(!patch.enabled && patch.enabled != null ? { default_order: null } : {}) }
        : grant
    )));
  };

  const submit = async () => {
    if (!grantsLoaded) return;
    setPending(true);
    try {
      await save(principalId, draft);
    } finally {
      setPending(false);
    }
  };

  return (
    <main className="panel single-view trace-backend-grants-view">
      <div className="panel-header">
        <div><h2>Run trace backend grants</h2><span className="muted">Choose which trace backends this principal can use for new submissions</span></div>
        <IconButton icon={RefreshCcw} onClick={refresh}>Refresh</IconButton>
      </div>
      <section className="action-form trace-backend-grants-form">
        <label>
          <span>Principal</span>
          <select value={principalId} onChange={(event) => selectPrincipal(event.target.value)}>
            <option value="">Select principal</option>
            {users.map((user) => (
              <option key={user.principal_id} value={user.principal_id}>
                {user.display_name || user.subject} · {shortId(user.principal_id)}
              </option>
            ))}
          </select>
        </label>
        <IconButton icon={ShieldCheck} variant="primary" disabled={!grantsLoaded || pending} onClick={submit}>
          Save grants
        </IconButton>
      </section>
      {grantsLoaded && draft.length ? (
        <div className="table-wrap">
          <table>
            <thead><tr><th>Backend</th><th>Kind</th><th>Granted</th><th>Default order</th><th>State updated</th><th>Description</th></tr></thead>
            <tbody>{draft.map((grant) => {
              const backend = catalogByName.get(grant.backend_name) || {};
              return (
                <tr key={grant.backend_name}>
                  <td><div className="primary-cell">{backend.display_name || grant.backend_name}</div><div className="muted">{grant.backend_name}</div></td>
                  <td>{backend.kind || "-"}</td>
                  <td><input type="checkbox" checked={grant.enabled} onChange={(event) => updateGrant(grant.backend_name, { enabled: event.target.checked })} /></td>
                  <td>
                    <input
                      aria-label={`Default order for ${grant.backend_name}`}
                      disabled={!grant.enabled}
                      min="0"
                      type="number"
                      value={grant.default_order ?? ""}
                      onChange={(event) => updateGrant(grant.backend_name, {
                        default_order: event.target.value === "" ? null : Number(event.target.value),
                      })}
                    />
                  </td>
                  <td>{formatTime(grant.updated_at)}</td>
                  <td>{backend.description || "-"}</td>
                </tr>
              );
            })}</tbody>
          </table>
        </div>
      ) : <Empty>{principalId && !grantsLoaded ? "Loading grants" : principalId ? "No selectable backends" : "Select a principal"}</Empty>}
    </main>
  );
}

function DatabaseAccountsView({ accounts, users, manage, refresh }) {
  const usersByPrincipal = new Map(users.map((user) => [user.principal_id, user]));
  return (
    <main className="panel single-view">
      <div className="panel-header">
        <h2>Database accounts</h2>
        <IconButton icon={RefreshCcw} onClick={refresh}>Refresh</IconButton>
      </div>
      {accounts.length ? (
        <div className="table-wrap database-accounts-table">
          <table>
            <thead>
              <tr><th>Principal</th><th>Backend</th><th>Database</th><th>Username</th><th>Trace table</th><th>Status</th><th>State updated</th><th>Action</th></tr>
            </thead>
            <tbody>
              {accounts.map((account) => (
                <tr key={`${account.owner_principal_id}:${account.backend_name}`}>
                  <td><PrincipalIdentity principalId={account.owner_principal_id} user={usersByPrincipal.get(account.owner_principal_id)} /></td>
                  <td>{account.backend_name}</td>
                  <td><CopyableValue value={account.database} label="database" /></td>
                  <td><CopyableValue value={account.username} label="username" /></td>
                  <td><CopyableValue value={account.trace_table_name || account.batch_table_name} label="trace table" /></td>
                  <td><Badge tone={statusTone(account.status)}>{humanizeStatus(account.status)}</Badge></td>
                  <td>{formatTime(account.updated_at)}</td>
                  <td>
                    <details className="action-menu">
                      <summary aria-label="Database account actions" title="Database account actions">
                        <MoreHorizontal size={16} />
                      </summary>
                      <div className="action-menu-popover">
                        {account.status === "disabled" ? (
                          <button type="button" onClick={() => manage(account, "restore")}>Restore</button>
                        ) : (
                          <button type="button" className="danger" onClick={() => manage(account, "disable")}>Disable</button>
                        )}
                        <button type="button" onClick={() => manage(account, "reset")}>Reset password</button>
                        <button type="button" onClick={() => manage(account, "reconcile")}>Reconcile</button>
                      </div>
                    </details>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <Empty>No database accounts</Empty>}
    </main>
  );
}

function AppBindingEditor({ user, updateUserAppBinding }) {
  const current = user.app_id || "";
  const [draft, setDraft] = useState(current);
  const [pending, setPending] = useState(false);
  const normalized = normalizeAppId(draft);
  const valid = isValidAppId(draft);
  const changed = normalized !== current;

  useEffect(() => setDraft(current), [current, user.principal_id]);

  const update = async (appId) => {
    setPending(true);
    try {
      await updateUserAppBinding(user, appId);
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="app-binding-editor">
      <input
        aria-label={`App ID for ${user.display_name || user.subject}`}
        aria-invalid={Boolean(draft.trim()) && !valid}
        autoCapitalize="none"
        autoComplete="off"
        maxLength={128}
        placeholder="mcp-app-id"
        spellCheck={false}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
      />
      <button
        type="button"
        className="row-action"
        disabled={pending || !valid || !changed}
        onClick={() => update(normalized)}
      >
        <Link2 size={13} />
        {current ? "Update" : "Bind"}
      </button>
      <button
        type="button"
        className="row-action danger"
        disabled={pending || !current}
        onClick={() => update(null)}
      >
        <Unlink size={13} />
        Unbind
      </button>
    </div>
  );
}

function AccessView({ me, users, updateUserRoles, updateUserAppBinding, refresh }) {
  const setRole = (user, role, enabled) => {
    const roles = new Set(user.roles || []);
    roles.add("user");
    if (enabled) roles.add(role);
    else roles.delete(role);
    updateUserRoles(user, [...roles].sort());
  };
  return (
    <main className="panel single-view">
      <div className="panel-header">
        <div><h2>Access</h2><span className="muted">{me?.subject || "-"}</span></div>
        <IconButton icon={RefreshCcw} onClick={refresh}>Refresh</IconButton>
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Identity</th><th>Tenant</th><th>App ID</th><th>Roles</th></tr></thead>
          <tbody>{users.map((user) => <tr key={user.principal_id}>
            <td><PrincipalIdentity principalId={user.principal_id} user={user} /></td><td>{user.tenant}</td>
            <td><AppBindingEditor user={user} updateUserAppBinding={updateUserAppBinding} /></td>
            <td><div className="role-options">
              <label className="role-option">
                <input
                  type="checkbox"
                  checked={Boolean(user.roles?.includes("admin"))}
                  onChange={(event) => setRole(user, "admin", event.target.checked)}
                />
                <span>admin</span>
              </label>
            </div></td>
          </tr>)}</tbody>
        </table>
      </div>
    </main>
  );
}


function PrincipalAccessView({ me, users, grants, putGrant, deleteGrant, onError, refresh }) {
  const isAdmin = Boolean(me?.roles?.includes("admin"));
  const [ownerDraft, setOwnerDraft] = useState("");
  const [granteeDraft, setGranteeDraft] = useState("");
  const [pending, setPending] = useState(false);
  const usersById = useMemo(
    () => new Map((users || []).map((user) => [user.principal_id, user])),
    [users]
  );
  const visibleRows = visiblePrincipalGrants(grants, me?.principal_id || "", isAdmin);
  const ownerPrincipalId = isAdmin ? ownerDraft : (me?.principal_id || "");
  const ownerUser = usersById.get(ownerPrincipalId);

  const createGrant = async (event) => {
    event.preventDefault();
    setPending(true);
    try {
      await putGrant(ownerPrincipalId, granteeDraft);
      setGranteeDraft("");
    } catch (error) {
      onError(error);
    } finally {
      setPending(false);
    }
  };

  const revokeGrant = async (grant) => {
    if (!window.confirm(`Revoke access from ${grant.grantee_principal_id}?`)) return;
    try {
      await deleteGrant(grant.owner_principal_id, grant.grantee_principal_id);
    } catch (error) {
      onError(error);
    }
  };

  return (
    <main className="panel single-view">
      <div className="panel-header">
        <div>
          <h2>Shared Access</h2>
          <span className="muted">Direct access to existing runs and batches; submissions remain personal.</span>
        </div>
        <IconButton icon={RefreshCcw} onClick={refresh}>Refresh</IconButton>
      </div>
      <form className="action-form grant-form" onSubmit={createGrant}>
        <label>
          Owner
          {isAdmin ? (
            <select value={ownerDraft} onChange={(event) => setOwnerDraft(event.target.value)} required>
              <option value="">Select owner</option>
              {users.map((user) => (
                <option key={user.principal_id} value={user.principal_id}>
                  {user.display_name || user.subject} · {user.principal_id}
                </option>
              ))}
            </select>
          ) : (
            <input
              value={ownerUser ? `${ownerUser.display_name || ownerUser.subject} · ${ownerPrincipalId}` : ownerPrincipalId}
              readOnly
            />
          )}
        </label>
        <label>
          Grantee
          <select value={granteeDraft} onChange={(event) => setGranteeDraft(event.target.value)} required>
            <option value="">Select grantee</option>
            {users.filter((user) => user.principal_id !== ownerPrincipalId).map((user) => (
              <option key={user.principal_id} value={user.principal_id}>
                {user.display_name || user.subject} · {user.principal_id}
              </option>
            ))}
          </select>
        </label>
        <div className="form-actions">
          <IconButton icon={Plus} variant="primary" type="submit" disabled={pending}>
            Grant access
          </IconButton>
        </div>
      </form>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Owner</th><th>Grantee</th><th>Created by</th><th>Created</th><th>Action</th></tr></thead>
          <tbody>{visibleRows.map((grant) => {
            const canRevoke = isAdmin || grant.owner_principal_id === me?.principal_id;
            return (
              <tr key={`${grant.owner_principal_id}:${grant.grantee_principal_id}`}>
                <td>
                  <PrincipalIdentity
                    principalId={grant.owner_principal_id}
                    user={usersById.get(grant.owner_principal_id)}
                  />
                </td>
                <td>
                  <PrincipalIdentity
                    principalId={grant.grantee_principal_id}
                    user={usersById.get(grant.grantee_principal_id)}
                  />
                </td>
                <td>
                  <PrincipalIdentity
                    principalId={grant.created_by}
                    user={usersById.get(grant.created_by)}
                  />
                </td>
                <td>{formatTime(grant.created_at)}</td>
                <td>
                  {canRevoke ? (
                    <IconButton icon={Trash2} variant="danger" onClick={() => revokeGrant(grant)}>
                      Revoke
                    </IconButton>
                  ) : "-"}
                </td>
              </tr>
            );
          })}</tbody>
        </table>
      </div>
      {!visibleRows.length ? <Empty>No principal access grants</Empty> : null}
    </main>
  );
}

function effectiveLimitRows(environment, grant) {
  const environmentLimits = environment?.resource_limits || {};
  const grantLimits = grant?.limits || {};
  return RESOURCE_LIMIT_FIELDS.map((field) => {
    const environmentValue = environmentLimits[field.key];
    const grantValue = grantLimits[field.key];
    if (environmentValue == null && grantValue == null) {
      return { ...field, value: "inherited", source: "deployment" };
    }
    if (environmentValue == null) return { ...field, value: grantValue, source: "grant" };
    if (grantValue == null) return { ...field, value: environmentValue, source: "environment" };
    return {
      ...field,
      value: Math.min(environmentValue, grantValue),
      source: grantValue < environmentValue ? "grant" : "environment",
    };
  });
}

function EffectiveConfigDetails({ environment, grant = null }) {
  const limitRows = effectiveLimitRows(environment, grant);
  return (
    <details className="inline-details">
      <summary>Effective config</summary>
      <div className="effective-config-table">
        {limitRows.map((row) => (
          <div key={row.key}><span>{row.label}</span><strong>{formatResourceLimitValue(row, row.value)}</strong><small>{row.source}</small></div>
        ))}
        {DEFAULT_TIMEOUT_FIELDS.map((field) => {
          const value = environment?.default_timeouts?.[field.key];
          return (
            <div key={field.key}><span>{field.label}</span><strong>{value ?? "inherited"}</strong><small>{value == null ? "deployment" : "environment"}</small></div>
          );
        })}
      </div>
    </details>
  );
}

function MyEnvironmentsView({
  environments,
  runTraceBackends,
  traceDatabaseAccounts,
  grantRunTraceBackend,
  claimQuota,
  issueTraceDatabaseCredential,
  refresh,
}) {
  return (
    <main id="my-environments-view" className="panel single-view">
      <div className="panel-header">
        <h2>My Environments</h2>
        <IconButton icon={RefreshCcw} onClick={refresh}>Refresh</IconButton>
      </div>
      <TraceDatabaseAccessTable
        accounts={traceDatabaseAccounts}
        issueCredential={issueTraceDatabaseCredential}
      />
      <MyRunTraceBackendsTable access={runTraceBackends} grantBackend={grantRunTraceBackend} />
      {environments.length ? (
        <div className="table-wrap">
          <table>
            <thead><tr><th>Environment</th><th>Allocation</th><th>Default quota</th><th>Access</th><th>Granted limits</th><th>Action</th></tr></thead>
            <tbody>
              {environments.map((environment) => (
                <tr key={environment.environment_id}>
                  <td>
                    <div className="primary-cell">{environment.name}</div><div className="muted">{environment.environment_id}</div>
                    <EffectiveConfigDetails environment={environment} grant={environment.grant} />
                  </td>
                  <td>{environment.allocation_type}</td>
                  <td><span className="muted quota-summary">{formatQuotaSummary(environment.default_quota)}</span></td>
                  <td>
                    {environment.grant ? (
                      <Badge tone={environment.grant.enabled ? "good" : "bad"}>
                        {environment.grant.enabled ? "granted" : "disabled"}
                      </Badge>
                    ) : <Badge tone="info">available</Badge>}
                  </td>
                  <td><span className="muted quota-summary">{formatQuotaSummary(environment.grant?.limits)}</span></td>
                  <td>
                    {environment.claimable ? (
                      <IconButton icon={Plus} variant="primary" onClick={() => claimQuota(environment.environment_id)}>
                        Claim quota
                      </IconButton>
                    ) : "-"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty>No available environments</Empty>
      )}
    </main>
  );
}

function emptyEnvironmentDraft() {
  return {
    environment_id: "",
    name: "",
    allocation_type: "dynamic",
    provider_type: "kubernetes",
    provider_config_ref: "",
    resource_limits: emptyConfigDraft(RESOURCE_LIMIT_FIELDS),
    default_quota_enabled: false,
    default_quota: emptyConfigDraft(DEFAULT_QUOTA_FIELDS),
    default_timeouts: emptyConfigDraft(DEFAULT_TIMEOUT_FIELDS),
    status: "disabled",
  };
}

function EnvironmentsView({
  environments,
  providerConfigs,
  createEnvironment,
  updateEnvironment,
  refresh,
  onError,
}) {
  const [editingId, setEditingId] = useState(null);
  const [viewingId, setViewingId] = useState(null);
  const [draft, setDraft] = useState(emptyEnvironmentDraft);
  const updateDraft = (field, value) => setDraft((current) => ({ ...current, [field]: value }));
  const resetDraft = () => {
    setEditingId(null);
    setDraft(emptyEnvironmentDraft());
  };
  const editEnvironment = (environment) => {
    setEditingId(environment.environment_id);
    setDraft({
      environment_id: environment.environment_id,
      name: environment.name,
      allocation_type: environment.allocation_type,
      provider_type: environment.provider_type,
      provider_config_ref: environment.provider_config_ref || "",
      resource_limits: configToDraft(environment.resource_limits, RESOURCE_LIMIT_FIELDS),
      default_quota_enabled: Boolean(environment.default_quota),
      default_quota: configToDraft(environment.default_quota, DEFAULT_QUOTA_FIELDS),
      default_timeouts: configToDraft(environment.default_timeouts, DEFAULT_TIMEOUT_FIELDS),
      status: environment.status,
    });
  };
  const updateAllocationType = (allocationType) => {
    setDraft((current) => ({
      ...current,
      allocation_type: allocationType,
      provider_type:
        allocationType === "predefined"
          ? "predefined"
          : current.provider_type === "predefined" ? "kubernetes" : current.provider_type,
      provider_config_ref: allocationType === "predefined" ? "" : current.provider_config_ref,
    }));
  };
  const updateProviderType = (providerType) => {
    setDraft((current) => ({
      ...current,
      provider_type: providerType,
      provider_config_ref: providerConfigs.some(
        (item) => item.provider_config_ref === current.provider_config_ref && item.provider_type === providerType
      ) ? current.provider_config_ref : "",
    }));
  };
  const selectableProviderConfigs = providerConfigs.filter(
    (item) => item.provider_type === draft.provider_type &&
      (item.status === "enabled" || item.provider_config_ref === draft.provider_config_ref)
  );
  const saveEnvironment = async () => {
    try {
      const payload = {
        name: draft.name,
        allocation_type: draft.allocation_type,
        provider_type: draft.provider_type,
        provider_config_ref: draft.provider_config_ref || null,
        resource_limits: draftToConfig(draft.resource_limits, RESOURCE_LIMIT_FIELDS),
        default_quota: draft.default_quota_enabled
          ? draftToCompleteConfig(draft.default_quota, DEFAULT_QUOTA_FIELDS)
          : null,
        default_timeouts: draftToConfig(draft.default_timeouts, DEFAULT_TIMEOUT_FIELDS),
        status: draft.status,
      };
      if (editingId) await updateEnvironment(editingId, payload);
      else await createEnvironment({ ...payload, environment_id: draft.environment_id });
      resetDraft();
    } catch (error) {
      onError(error);
    }
  };
  return (
    <main id="environments-view" className="panel single-view">
      <div className="panel-header"><h2>Environments</h2><IconButton icon={RefreshCcw} onClick={refresh}>Refresh</IconButton></div>
      {viewingId ? <section id="environment-state"><div className="panel-header"><h2>Environment state · {viewingId}</h2><button type="button" onClick={() => setViewingId(null)}>Close state</button></div><PlatformOverview request={request} controlPath={controlPath} environmentId={viewingId} /></section> : null}
      <section className="action-form environment-form">
        <label><span>ID</span><input disabled={Boolean(editingId)} value={draft.environment_id} onChange={(event) => updateDraft("environment_id", event.target.value)} /></label>
        <label><span>Name</span><input value={draft.name} onChange={(event) => updateDraft("name", event.target.value)} /></label>
        <label><span>Allocation</span><select value={draft.allocation_type} onChange={(event) => updateAllocationType(event.target.value)}><option value="predefined">predefined</option><option value="dynamic">dynamic</option></select></label>
        <label><span>Provider</span><select value={draft.provider_type} onChange={(event) => updateProviderType(event.target.value)} disabled={draft.allocation_type === "predefined"}>{draft.allocation_type === "predefined" ? <option value="predefined">predefined</option> : <><option value="kubernetes">kubernetes</option><option value="tkex">tkex</option></>}</select></label>
        <label><span>Provider config ref</span><select disabled={draft.allocation_type === "predefined"} value={draft.provider_config_ref} onChange={(event) => updateDraft("provider_config_ref", event.target.value)}><option value="">Select provider config</option>{selectableProviderConfigs.map((item) => <option key={item.provider_config_ref} value={item.provider_config_ref}>{item.provider_config_ref}{item.status === "disabled" ? " (disabled)" : ""}</option>)}</select></label>
        <label><span>Status</span><select value={draft.status} onChange={(event) => updateDraft("status", event.target.value)}><option value="disabled">disabled</option><option value="enabled">enabled</option></select></label>
        <IntegerConfigFields
          legend="Common resource limits"
          fields={COMMON_RESOURCE_LIMIT_FIELDS}
          value={draft.resource_limits}
          onChange={(value) => updateDraft("resource_limits", value)}
        />
        <label className="check">
          <input
            type="checkbox"
            checked={draft.default_quota_enabled}
            onChange={(event) => updateDraft("default_quota_enabled", event.target.checked)}
          />
          Enable default quota
        </label>
        {draft.default_quota_enabled ? (
          <IntegerConfigFields
            legend="Default quota"
            fields={DEFAULT_QUOTA_FIELDS}
            value={draft.default_quota}
            onChange={(value) => updateDraft("default_quota", value)}
          />
        ) : null}
        <details className="advanced-section form-advanced">
          <summary>Advanced overrides</summary>
          <IntegerConfigFields
            legend="Additional resource limits"
            fields={ADVANCED_RESOURCE_LIMIT_FIELDS}
            value={draft.resource_limits}
            onChange={(value) => updateDraft("resource_limits", value)}
          />
          <IntegerConfigFields
            legend="Default timeouts"
            fields={DEFAULT_TIMEOUT_FIELDS}
            value={draft.default_timeouts}
            onChange={(value) => updateDraft("default_timeouts", value)}
          />
        </details>
        <div className="form-actions">
          {editingId ? <IconButton icon={X} onClick={resetDraft}>Cancel</IconButton> : null}
          <IconButton icon={editingId ? Pencil : Plus} variant="primary" onClick={() => saveEnvironment()}>
            {editingId ? "Save" : "Create"}
          </IconButton>
        </div>
      </section>
      <div className="table-wrap"><table>
        <thead><tr><th>Environment</th><th>Allocation</th><th>Provider</th><th>Default quota</th><th>Status</th><th>Version</th><th>Action</th></tr></thead>
        <tbody>{environments.map((environment) => <tr key={environment.environment_id}>
          <td>
            <div className="primary-cell">{environment.name}</div><div className="muted">{environment.environment_id}</div>
            <EffectiveConfigDetails environment={environment} />
          </td>
          <td>{environment.allocation_type}</td><td>{environment.provider_type}</td>
          <td><span className="muted quota-summary">{formatQuotaSummary(environment.default_quota)}</span></td>
          <td><Badge>{environment.status}</Badge></td>
          <td>{environment.config_version}</td><td><div className="row-actions">
            <button className="row-action" onClick={() => { setViewingId(environment.environment_id); requestAnimationFrame(() => document.getElementById("environment-state")?.scrollIntoView({ behavior: "smooth" })); }}>View state</button>
            <button className="row-action" onClick={() => editEnvironment(environment)}>Edit</button>
            <button className="row-action" onClick={() => updateEnvironment(environment.environment_id, { status: environment.status === "enabled" ? "disabled" : "enabled" }).catch(onError)}>
              {environment.status === "enabled" ? "Disable" : "Enable"}
            </button>
          </div></td>
        </tr>)}</tbody>
      </table></div>
    </main>
  );
}

function ProviderConfigsView({
  providerConfigs,
  createProviderConfig,
  updateProviderConfig,
  refresh,
  onError,
}) {
  const [editingRef, setEditingRef] = useState(null);
  const [draft, setDraft] = useState(() => emptyProviderConfigDraft());
  const [restartRequired, setRestartRequired] = useState(false);
  const updateDraft = (field, value) => setDraft((current) => ({ ...current, [field]: value }));
  const resetDraft = () => {
    setEditingRef(null);
    setDraft(emptyProviderConfigDraft());
  };
  const editProviderConfig = (record) => {
    setRestartRequired(false);
    setEditingRef(record.provider_config_ref);
    setDraft(providerConfigToDraft(record));
  };
  const updateProviderType = (providerType) => {
    setDraft((current) => ({
      ...emptyProviderConfigDraft(providerType),
      provider_config_ref: current.provider_config_ref,
      status: current.status,
    }));
  };
  const saveProviderConfig = async () => {
    try {
      const payload = providerConfigToPayload(draft);
      let saved;
      if (editingRef) {
        const updatePayload = { config: payload.config, status: payload.status };
        if (payload.credential) updatePayload.credential = payload.credential;
        saved = await updateProviderConfig(editingRef, updatePayload);
      } else {
        saved = await createProviderConfig(payload);
      }
      resetDraft();
      setRestartRequired(Boolean(saved?.credential_restart_required));
    } catch (error) {
      onError(error);
    }
  };
  return (
    <main id="provider-configs-view" className="panel single-view">
      <div className="panel-header"><h2>Provider configs</h2><IconButton icon={RefreshCcw} onClick={refresh}>Refresh</IconButton></div>
      <section className="action-form provider-config-form">
        <label><span>Ref</span><input disabled={Boolean(editingRef)} value={draft.provider_config_ref} onChange={(event) => updateDraft("provider_config_ref", event.target.value)} /></label>
        <label><span>Provider</span><select disabled={Boolean(editingRef)} value={draft.provider_type} onChange={(event) => updateProviderType(event.target.value)}><option value="kubernetes">kubernetes</option><option value="tkex">tkex</option></select></label>
        <label><span>Namespace</span><input value={draft.namespace} onChange={(event) => updateDraft("namespace", event.target.value)} /></label>
        <label><span>Status</span><select value={draft.status} onChange={(event) => updateDraft("status", event.target.value)}><option value="disabled">disabled</option><option value="enabled">enabled</option></select></label>
        {draft.provider_type === "kubernetes" ? (
          <>
            <label><span>Authentication</span><select value={draft.auth_type} onChange={(event) => updateDraft("auth_type", event.target.value)}><option value="in_cluster">in_cluster</option><option value="secret">secret</option></select></label>
            {draft.auth_type === "secret" ? (
              <>
                <label><span>API URL</span><input value={draft.api_url} onChange={(event) => updateDraft("api_url", event.target.value)} /></label>
                <label><span>Credential ref</span><input value={draft.credential_ref} onChange={(event) => updateDraft("credential_ref", event.target.value)} /></label>
                <label><span>Bearer token</span><input type="password" autoComplete="new-password" spellCheck={false} value={draft.credential_token} onChange={(event) => updateDraft("credential_token", event.target.value)} /></label>
                <label className="json-field"><span>CA certificate</span><textarea rows="4" spellCheck={false} value={draft.credential_ca_certificate} onChange={(event) => updateDraft("credential_ca_certificate", event.target.value)} /></label>
              </>
            ) : null}
            <label><span>Request ratio</span><input type="number" min="0.01" max="1" step="0.01" value={draft.manager_resource_request_ratio} onChange={(event) => updateDraft("manager_resource_request_ratio", event.target.value)} /></label>
            <label className="check"><input type="checkbox" checked={draft.tls_strict} onChange={(event) => updateDraft("tls_strict", event.target.checked)} />TLS strict</label>
          </>
        ) : (
          <>
            <label><span>Base URL</span><input value={draft.base_url} onChange={(event) => updateDraft("base_url", event.target.value)} /></label>
            <label><span>API account</span><input value={draft.api_account} onChange={(event) => updateDraft("api_account", event.target.value)} /></label>
            <label><span>Credential ref</span><input value={draft.credential_ref} onChange={(event) => updateDraft("credential_ref", event.target.value)} /></label>
            <label><span>API token</span><input type="password" autoComplete="new-password" spellCheck={false} value={draft.credential_token} onChange={(event) => updateDraft("credential_token", event.target.value)} /></label>
            <label><span>Cluster</span><input value={draft.cluster_name} onChange={(event) => updateDraft("cluster_name", event.target.value)} /></label>
            <label><span>Project</span><input value={draft.project_name} onChange={(event) => updateDraft("project_name", event.target.value)} /></label>
            <label className="check"><input type="checkbox" checked={draft.zhiyan_log_enabled} onChange={(event) => updateDraft("zhiyan_log_enabled", event.target.checked)} />Zhiyan logs</label>
            {draft.zhiyan_log_enabled ? (
              <>
                <label><span>Zhiyan log name</span><input value={draft.zhiyan_log_name} onChange={(event) => updateDraft("zhiyan_log_name", event.target.value)} /></label>
                <label><span>Agent image</span><input value={draft.zhiyan_agent_image} onChange={(event) => updateDraft("zhiyan_agent_image", event.target.value)} /></label>
                <label><span>Log volume (MB)</span><input type="number" min="1" step="1" value={draft.zhiyan_log_volume_mb} onChange={(event) => updateDraft("zhiyan_log_volume_mb", event.target.value)} /></label>
              </>
            ) : null}
          </>
        )}
        <div className="form-actions">
          {editingRef ? <IconButton icon={X} onClick={resetDraft}>Cancel</IconButton> : null}
          <IconButton icon={editingRef ? Pencil : Plus} variant="primary" onClick={saveProviderConfig}>
            {editingRef ? "Save" : "Create"}
          </IconButton>
        </div>
      </section>
      {restartRequired ? <div className="provider-config-notice" role="status">Credential saved. Restart Reconciler before creating new allocations.</div> : null}
      <div className="table-wrap"><table>
        <thead><tr><th>Ref</th><th>Provider</th><th>Namespace</th><th>Endpoint</th><th>Credential</th><th>Status</th><th>Version</th><th>Action</th></tr></thead>
        <tbody>{providerConfigs.map((record) => <tr key={record.provider_config_ref}>
          <td className="primary-cell">{record.provider_config_ref}</td>
          <td>{record.provider_type}</td>
          <td>{record.config?.namespace}</td>
          <td>{record.config?.api_url || record.config?.base_url || "in-cluster"}</td>
          <td>{record.config?.credential_ref ? <><div>{record.config.credential_ref}</div><Badge tone={record.credential_configured ? "good" : "bad"}>{record.credential_configured ? "configured" : "missing"}</Badge></> : "in-cluster"}</td>
          <td><Badge>{record.status}</Badge></td>
          <td>{record.version}</td>
          <td><div className="row-actions">
            <button className="row-action" onClick={() => editProviderConfig(record)}>Edit</button>
            <button className="row-action" onClick={() => updateProviderConfig(record.provider_config_ref, { status: record.status === "enabled" ? "disabled" : "enabled" }).catch(onError)}>{record.status === "enabled" ? "Disable" : "Enable"}</button>
          </div></td>
        </tr>)}</tbody>
      </table></div>
    </main>
  );
}

function GrantsView({ grants, environments, users, upsertGrant, refresh, onError }) {
  const [environmentId, setEnvironmentId] = useState("");
  const [principalId, setPrincipalId] = useState("");
  const [limits, setLimits] = useState(() => emptyConfigDraft(RESOURCE_LIMIT_FIELDS));
  const [preset, setPreset] = useState("standard");
  const [scopePreset, setScopePreset] = useState(DEFAULT_GRANT_SCOPE_PRESET);
  const [scopes, setScopes] = useState(() => grantScopesForPreset(DEFAULT_GRANT_SCOPE_PRESET));
  const [enabled, setEnabled] = useState(true);
  const usersByPrincipal = useMemo(
    () => new Map(users.map((user) => [user.principal_id, user])),
    [users]
  );
  const batchKeys = new Set(["active_batches", "batch_max_inflight", "max_items", "max_task_list_bytes"]);
  const presetFields = preset === "custom"
    ? RESOURCE_LIMIT_FIELDS
    : preset === "batch"
      ? RESOURCE_LIMIT_FIELDS.filter((field) => COMMON_RESOURCE_LIMIT_FIELDS.includes(field) || batchKeys.has(field.key))
      : COMMON_RESOURCE_LIMIT_FIELDS;
  const changePreset = (nextPreset) => {
    setPreset(nextPreset);
    if (nextPreset === "custom") return;
    const allowed = new Set(
      (nextPreset === "batch"
        ? RESOURCE_LIMIT_FIELDS.filter((field) => COMMON_RESOURCE_LIMIT_FIELDS.includes(field) || batchKeys.has(field.key))
        : COMMON_RESOURCE_LIMIT_FIELDS
      ).map((field) => field.key)
    );
    setLimits((current) => Object.fromEntries(
      RESOURCE_LIMIT_FIELDS.map((field) => [field.key, allowed.has(field.key) ? current[field.key] : ""])
    ));
  };
  const editGrant = (grant) => {
    setEnvironmentId(grant.environment_id);
    setPrincipalId(grant.principal_id);
    setLimits(configToDraft(grant.limits, RESOURCE_LIMIT_FIELDS));
    setPreset("custom");
    setScopePreset(inferGrantScopePreset(grant.scopes));
    setScopes([...(grant.scopes || [])]);
    setEnabled(Boolean(grant.enabled));
  };
  const changeScopePreset = (nextPreset) => {
    setScopePreset(nextPreset);
    if (nextPreset !== "custom") setScopes(grantScopesForPreset(nextPreset));
  };
  const toggleScope = (scope) => {
    setScopes((current) => current.includes(scope)
      ? current.filter((item) => item !== scope)
      : [...current, scope]);
  };
  const saveGrant = async () => {
    try {
      await upsertGrant(environmentId, principalId, draftToConfig(limits, RESOURCE_LIMIT_FIELDS), scopes, enabled);
    } catch (error) {
      onError(error);
    }
  };
  return (
    <main id="grants-view" className="panel single-view">
      <div className="panel-header"><h2>Environment Grants</h2><IconButton icon={RefreshCcw} onClick={refresh}>Refresh</IconButton></div>
      <section className="action-form grant-form">
        <label><span>Environment</span><select value={environmentId} onChange={(event) => setEnvironmentId(event.target.value)}><option value="">Select environment</option>{environments.map((environment) => <option key={environment.environment_id} value={environment.environment_id}>{environment.name}</option>)}</select></label>
        <label><span>Principal</span><select value={principalId} onChange={(event) => setPrincipalId(event.target.value)}><option value="">Select principal</option>{users.map((user) => <option key={user.principal_id} value={user.principal_id}>{user.display_name || user.subject} · {user.subject} · {shortId(user.principal_id)}</option>)}</select></label>
        <label><span>Resource preset</span><select value={preset} onChange={(event) => changePreset(event.target.value)}><option value="standard">Standard</option><option value="batch">Batch</option><option value="custom">Custom</option></select></label>
        <label><span>Permission preset</span><select value={scopePreset} onChange={(event) => changeScopePreset(event.target.value)}>{Object.entries(GRANT_SCOPE_PRESETS).map(([key, item]) => <option key={key} value={key}>{item.label}</option>)}</select></label>
        <IntegerConfigFields
          legend="Limits"
          fields={presetFields}
          value={limits}
          onChange={setLimits}
        />
        {scopePreset === "custom" ? (
          <fieldset className="scope-field">
            <legend>Scopes</legend>
            <div className="scope-options">
              {ALL_SUBMISSION_SCOPES.map((scope) => (
                <label key={scope} className="scope-option">
                  <input type="checkbox" checked={scopes.includes(scope)} onChange={() => toggleScope(scope)} />
                  <span>{scope}</span>
                </label>
              ))}
            </div>
          </fieldset>
        ) : <div className="preset-scopes">{scopes.join(", ")}</div>}
        <label className="check"><input type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} />enabled</label>
        <IconButton icon={ShieldCheck} variant="primary" disabled={!environmentId || !principalId} onClick={saveGrant}>Save grant</IconButton>
      </section>
      <div className="table-wrap"><table>
        <thead><tr><th>Environment</th><th>Principal</th><th>Status</th><th>Scopes</th><th>Limits</th><th>Version</th><th>Action</th></tr></thead>
        <tbody>{grants.map((grant) => <tr key={`${grant.environment_id}:${grant.principal_id}`}>
          <td>{grant.environment_id}</td><td><PrincipalIdentity principalId={grant.principal_id} user={usersByPrincipal.get(grant.principal_id)} /></td><td><Badge>{grant.enabled ? "enabled" : "disabled"}</Badge></td>
          <td><code>{(grant.scopes || []).join(", ")}</code></td><td><code>{JSON.stringify(grant.limits || {})}</code></td><td>{grant.version}</td>
          <td><button className="row-action" onClick={() => editGrant(grant)}>Edit</button></td>
        </tr>)}</tbody>
      </table></div>
    </main>
  );
}

function SystemView({
  environments,
  configOverrides,
  refreshSystem,
  saveConfigOverride,
  removeConfigOverride,
  onError,
}) {
  const [activeTab, setActiveTab] = useState("observability");
  useEffect(() => {
    const refresh = () => {
      if (activeTab === "configuration") refreshSystem();
      else if (activeTab === "business-statistics") window.dispatchEvent(new Event("refresh-business-statistics"));
      else window.dispatchEvent(new Event("refresh-platform-observability"));
    };
    window.addEventListener("refresh-system", refresh);
    return () => window.removeEventListener("refresh-system", refresh);
  }, [activeTab, refreshSystem]);
  const ordinaryOverrides = configOverrides?.config_overrides || [];
  const secretOverrides = configOverrides?.secret_overrides || [];
  const [editor, setEditor] = useState({
    secret: false,
    role: "*",
    identity: "*",
    expectedVersion: "0",
    values: "{}",
    originalValues: {},
  });
  const [editorStatus, setEditorStatus] = useState("");
  const [editorSaving, setEditorSaving] = useState(false);

  const resetEditor = () => {
    setEditor({ secret: false, role: "*", identity: "*", expectedVersion: "0", values: "{}", originalValues: {} });
    setEditorStatus("");
  };
  const editRecord = (record, secret) => {
    setEditor({
      secret,
      role: record.role,
      identity: record.identity,
      expectedVersion: String(record.version),
      values: secret ? "{}" : JSON.stringify(record.values || {}, null, 2),
      originalValues: secret ? null : record.values || {},
    });
    setEditorStatus(secret ? "Secret values are write-only; enter the complete replacement object." : "");
  };
  const editorDiff = useMemo(() => {
    if (editor.secret || editor.originalValues === null) return null;
    try {
      return { changes: diffConfigValues(editor.originalValues, parseConfigValues(editor.values)), error: null };
    } catch (error) {
      return { changes: [], error };
    }
  }, [editor.secret, editor.originalValues, editor.values]);
  const saveEditor = async () => {
    setEditorSaving(true);
    setEditorStatus("");
    try {
      const values = parseConfigValues(editor.values);
      await saveConfigOverride(buildConfigOverridePayload({ ...editor, values }));
      setEditorStatus("Saved. Restart the affected service(s) to load the new startup snapshot.");
      setEditor({
        ...editor,
        expectedVersion: String(Number(editor.expectedVersion) + 1),
        values: editor.secret ? "{}" : JSON.stringify(values, null, 2),
        originalValues: editor.secret ? null : values,
      });
    } catch (error) {
      setEditorStatus(`Save failed: ${error.message}`);
      onError(error);
    } finally {
      setEditorSaving(false);
    }
  };
  const deleteRecord = async (record, secret) => {
    if (!window.confirm(`Delete ${record.role}/${record.identity} configuration?`)) return;
    try {
      await removeConfigOverride(buildConfigOverrideDeletePayload({
        role: record.role,
        identity: record.identity,
        expectedVersion: record.version,
        secret,
      }));
      if (editor.role === record.role && editor.identity === record.identity && editor.secret === secret) {
        resetEditor();
      }
    } catch (error) {
      onError(error);
    }
  };
  return (
    <main id="system-view" className="workspace-stack">
      <div className="detail-tabs system-tabs" role="tablist" aria-label="System views"
        onKeyDown={(event) => {
          if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          const keys = ["observability", "business-statistics", "configuration"];
          const next = event.key === "Home" ? keys[0] : event.key === "End" ? keys.at(-1) : keys[(keys.indexOf(activeTab) + (event.key === "ArrowRight" ? 1 : keys.length - 1)) % keys.length];
          setActiveTab(next);
          document.getElementById(`system-tab-${next}`)?.focus();
        }}>
        {[["observability", "Observability"], ["business-statistics", "Business statistics"], ["configuration", "Configuration"]].map(([key, label]) => (
          <button key={key} type="button" role="tab" id={`system-tab-${key}`} aria-controls={`system-panel-${key}`}
            aria-selected={activeTab === key} tabIndex={activeTab === key ? 0 : -1}
            className={activeTab === key ? "active" : ""} onClick={() => setActiveTab(key)}>{label}</button>
        ))}
      </div>
      <div className="system-tab-panel" role="tabpanel" id="system-panel-observability" aria-labelledby="system-tab-observability" hidden={activeTab !== "observability"}>
        <PlatformOverview request={request} controlPath={controlPath} environments={environments} active={activeTab === "observability"} />
      </div>
      <div className="system-tab-panel" role="tabpanel" id="system-panel-business-statistics" aria-labelledby="system-tab-business-statistics" hidden={activeTab !== "business-statistics"}>
        <BusinessStatistics request={request} controlPath={controlPath} environments={environments} active={activeTab === "business-statistics"} />
      </div>
      <div className="system-tab-panel workspace-stack" role="tabpanel" id="system-panel-configuration" aria-labelledby="system-tab-configuration" hidden={activeTab !== "configuration"}>
      <div className="row-actions"><IconButton icon={RefreshCcw} onClick={refreshSystem}>Refresh configuration</IconButton></div>
      <section className="panel config-overrides-panel">
        <div className="panel-header">
          <div>
            <h2>Stored configuration</h2>
            <span className="muted">Database overlays only. Services load an immutable snapshot at startup.</span>
          </div>
        </div>
        <div className="config-overrides-section config-editor">
          <h3>{editor.secret ? "Write secret configuration" : "Edit ordinary configuration"}</h3>
          <p className="muted">Only administrators can save configuration. Saving replaces the complete scope object and uses its current version.</p>
          <div className="action-form submission-form">
            <label>
              <span>Kind</span>
              <select
                value={editor.secret ? "secret" : "ordinary"}
                onChange={(event) => setEditor((current) => ({
                  ...current,
                  secret: event.target.value === "secret",
                  values: "{}",
                  originalValues: {},
                }))}
              >
                <option value="ordinary">Ordinary</option>
                <option value="secret">Secret (write-only)</option>
              </select>
            </label>
            <label><span>Role</span><input value={editor.role} onChange={(event) => setEditor((current) => ({ ...current, role: event.target.value }))} /></label>
            <label><span>Identity</span><input value={editor.identity} onChange={(event) => setEditor((current) => ({ ...current, identity: event.target.value }))} /></label>
            <label><span>Expected version</span><input type="number" min="0" value={editor.expectedVersion} onChange={(event) => setEditor((current) => ({ ...current, expectedVersion: event.target.value }))} /></label>
            <label className="json-field full-field">
              <span>{editor.secret ? "Secret values JSON (complete replacement; never returned)" : "Values JSON"}</span>
              <textarea value={editor.values} spellCheck={false} onChange={(event) => setEditor((current) => ({ ...current, values: event.target.value }))} />
            </label>
          </div>
          {!editor.secret && editorDiff ? (
            <div className="config-diff" aria-live="polite">
              <div className="config-diff-header">
                <h4>Pending changes</h4>
                {!editorDiff.error ? <span className="muted">{editorDiff.changes.length} change{editorDiff.changes.length === 1 ? "" : "s"}</span> : null}
              </div>
              {editorDiff.error ? <p className="muted">Diff unavailable: {editorDiff.error.message}</p> : editorDiff.changes.length ? (
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>Change</th><th>Path</th><th>Original</th><th>Current</th></tr></thead>
                    <tbody>{editorDiff.changes.map((change) => (
                      <tr key={`${change.kind}:${change.path}`}>
                        <td><span className={`config-diff-kind config-diff-${change.kind}`}>{change.kind}</span></td>
                        <td className="primary-cell"><code>{change.path}</code></td>
                        <td><pre className="config-diff-value">{change.before === undefined ? "—" : JSON.stringify(change.before, null, 2)}</pre></td>
                        <td><pre className="config-diff-value">{change.after === undefined ? "—" : JSON.stringify(change.after, null, 2)}</pre></td>
                      </tr>
                    ))}</tbody>
                  </table>
                </div>
              ) : <p className="muted">No changes from the loaded version.</p>}
            </div>
          ) : null}
          <div className="row-actions">
            <IconButton icon={Pencil} variant="primary" disabled={editorSaving} onClick={() => saveEditor()}>
              {editorSaving ? "Saving..." : "Save configuration"}
            </IconButton>
            <IconButton icon={X} onClick={resetEditor}>Clear</IconButton>
          </div>
          {editorStatus ? <p className="muted config-editor-status">{editorStatus}</p> : null}
        </div>
        <div className="config-overrides-section">
          <h3>Ordinary overrides</h3>
          {ordinaryOverrides.length ? (
            <div className="table-wrap">
              <table>
                <thead><tr><th>Scope</th><th>Version</th><th>Values</th><th>Actions</th></tr></thead>
                <tbody>{ordinaryOverrides.map((record) => (
                  <tr key={`${record.role}/${record.identity}`}>
                    <td className="primary-cell">{record.role}/{record.identity}</td>
                    <td>{record.version}</td>
                    <td><pre className="config-overrides-json">{JSON.stringify(record.values, null, 2)}</pre></td>
                    <td><div className="row-actions">
                      <button className="row-action" onClick={() => editRecord(record, false)}>Edit</button>
                      <button className="row-action danger" onClick={() => deleteRecord(record, false)}>Delete</button>
                    </div></td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          ) : <Empty>No ordinary overrides</Empty>}
        </div>
        <div className="config-overrides-section">
          <h3>Secret metadata</h3>
          <p className="muted">Secret values are never returned to the browser.</p>
          {secretOverrides.length ? (
            <div className="table-wrap">
              <table>
                <thead><tr><th>Scope</th><th>Version</th><th>Configured fields</th><th>Actions</th></tr></thead>
                <tbody>{secretOverrides.map((record) => (
                  <tr key={`${record.role}/${record.identity}`}>
                    <td className="primary-cell">{record.role}/{record.identity}</td>
                    <td>{record.version}</td>
                    <td>{record.fields.join(", ") || "-"}</td>
                    <td><div className="row-actions">
                      <button className="row-action" onClick={() => editRecord(record, true)}>Replace</button>
                      <button className="row-action danger" onClick={() => deleteRecord(record, true)}>Delete</button>
                    </div></td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          ) : <Empty>No secret overrides</Empty>}
        </div>
      </section>
      </div>
    </main>
  );
}

export default App;
