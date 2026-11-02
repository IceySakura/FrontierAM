const BASE_PATH = "/v1/console";
const KEYS = new Set([
  "q",
  "status",
  "type",
  "mode",
  "environment_id",
  "pool_id",
  "workload",
  "allocation_id",
  "allocation_type",
  "owner",
  "execution_id",
  "health",
  "draining",
  "cursor",
  "page_size",
  "page",
  "tab",
  "view",
  "item_id",
  "item_status",
  "item_prefix",
  "child_execution_id",
  "error_code",
  "manager_id",
  "session_id",
  "lines",
  "return_to",
]);
let identity = "anonymous";

function decodePart(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return null;
  }
}
export function safeReturnPath(value) {
  if (
    typeof value !== "string" ||
    value.length > 6000 ||
    !value.startsWith(`${BASE_PATH}/`) ||
    /[\r\n\\]/.test(value)
  )
    return null;
  try {
    if (
      !new URL(value, "https://console.invalid").pathname.startsWith(
        `${BASE_PATH}/`,
      )
    )
      return null;
  } catch {
    return null;
  }
  return value;
}
function cleanQuery(query = {}) {
  const result = {};
  for (const [key, value] of Object.entries(query)) {
    if (!KEYS.has(key) || value === undefined || value === null || value === "")
      continue;
    if (key === "return_to" && !safeReturnPath(value)) continue;
    if (
      ["page", "page_size", "lines"].includes(key) &&
      (!/^\d+$/.test(String(value)) ||
        Number(value) < 1 ||
        Number(value) >
          (key === "page" ? 1000000 : key === "lines" ? 1000 : 100))
    )
      continue;
    result[key] = String(value);
  }
  return result;
}
export function parseConsoleLocation(pathname, search = "") {
  const relative =
    pathname === BASE_PATH || pathname.startsWith(`${BASE_PATH}/`)
      ? pathname.slice(BASE_PATH.length)
      : pathname;
  const parts = relative.split("/").filter(Boolean).map(decodePart);
  if (parts.some((part) => part === null)) return { section: "runs" };
  const query = cleanQuery(Object.fromEntries(new URLSearchParams(search)));
  let route;
  if (!parts.length || parts[0] === "runs") route = { section: "runs" };
  else if (parts[0] === "executions" && parts[1]) {
    route = { section: "execution", executionId: parts[1] };
    const legacyParent = new URLSearchParams(search).get("from_batch");
    if (legacyParent && !query.return_to)
      query.return_to = consolePath({
        section: "execution",
        executionId: legacyParent,
        query: { tab: "items" },
      });
  } else if (parts[0] === "batches") {
    if (!parts[1]) {
      route = { section: "runs" };
      query.type = "batch";
    } else {
      route = { section: "execution", executionId: parts[1] };
      query.tab = "items";
      if (parts[2] === "items" && parts[3]) query.item_id = parts[3];
    }
  } else if (parts[0] === "builds")
    route = parts[1]
      ? { section: "build", buildId: parts[1] }
      : { section: "builds" };
  else if (parts[0] === "allocations" && parts[1])
    route = { section: "allocation", allocationId: parts[1] };
  else if (["my-environments", "shared-access"].includes(parts[0]))
    route = { section: parts[0] };
  else if (
    parts[0] === "admin" &&
    ["access", "environments", "system"].includes(parts[1])
  )
    route = { section: parts[1] };
  else if (parts[0] === "admin" && parts[1] === "infrastructure")
    route = {
      section: "infrastructure",
      infrastructureView: ["allocations", "builds"].includes(parts[2])
        ? parts[2]
        : "managers",
    };
  else route = { section: "runs" };
  if (route.section === "runs") {
    if (!["single", "live", "batch"].includes(query.type)) delete query.type;
    if (query.type === "batch") delete query.mode;
  }
  if (Object.keys(query).length) route.query = query;
  return route;
}
export function consolePath(route) {
  const encode = encodeURIComponent;
  const query = cleanQuery(route.query);
  let path;
  switch (route.section) {
    case "execution":
      path = `${BASE_PATH}/executions/${encode(route.executionId)}`;
      if (route.fromBatch && !query.return_to)
        query.return_to = consolePath({
          section: "execution",
          executionId: route.fromBatch,
          query: { tab: "items" },
        });
      break;
    case "batch":
      path = `${BASE_PATH}/executions/${encode(route.batchId)}`;
      query.tab = "items";
      if (route.itemId) query.item_id = route.itemId;
      break;
    case "batches":
      path = `${BASE_PATH}/runs`;
      query.type = "batch";
      break;
    case "builds":
      path = `${BASE_PATH}/builds`;
      break;
    case "build":
      path = `${BASE_PATH}/builds/${encode(route.buildId)}`;
      break;
    case "allocation":
      path = `${BASE_PATH}/allocations/${encode(route.allocationId)}`;
      break;
    case "my-environments":
    case "shared-access":
      path = `${BASE_PATH}/${route.section}`;
      break;
    case "access":
    case "environments":
    case "system":
      path = `${BASE_PATH}/admin/${route.section}`;
      break;
    case "infrastructure":
      path = `${BASE_PATH}/admin/infrastructure/${["allocations", "builds"].includes(route.infrastructureView) ? route.infrastructureView : "managers"}`;
      break;
    default:
      path = `${BASE_PATH}/runs`;
  }
  const params = new URLSearchParams(query).toString();
  return path + (params ? `?${params}` : "");
}
export function routeFromPath(path) {
  const safe = safeReturnPath(path);
  if (!safe) return { section: "runs" };
  const [pathname, ...search] = safe.split("?");
  return parseConsoleLocation(pathname, search.join("?"));
}
export function setNavigationIdentity(value) {
  identity = value || "anonymous";
}
function listKey(route) {
  return route.section === "infrastructure"
    ? `infrastructure:${route.infrastructureView || "managers"}`
    : route.section === "runs" || route.section === "builds"
      ? route.section
      : null;
}
function storageGet(key) {
  try {
    return JSON.parse(
      sessionStorage.getItem(`nexus-console:${identity}:${key}`) || "null",
    );
  } catch {
    return null;
  }
}
function storagePut(key, value) {
  try {
    sessionStorage.setItem(
      `nexus-console:${identity}:${key}`,
      JSON.stringify(value),
    );
  } catch {
    /* Storage is optional. */
  }
}
export function savedListRoute(section, infrastructureView) {
  const fallback = {
    section,
    ...(infrastructureView ? { infrastructureView } : {}),
  };
  return storageGet(`list:${listKey(fallback)}`) || fallback;
}
export function navigateConsole(
  next,
  { replace = false, state = {}, restore = false } = {},
) {
  const current = parseConsoleLocation(
    window.location.pathname,
    window.location.search,
  );
  if (listKey(current)) storagePut(`list:${listKey(current)}`, current);
  storagePut(`scroll:${consolePath(current)}`, window.scrollY);
  window.history.replaceState(
    { ...window.history.state, scrollY: window.scrollY },
    "",
  );
  const route = typeof next === "string" ? routeFromPath(next) : next;
  const path = consolePath(route);
  window.history[replace ? "replaceState" : "pushState"](
    {
      ...state,
      ...(restore
        ? { scrollY: storageGet(`scroll:${path}`) || 0 }
        : replace
          ? { scrollY: window.scrollY }
          : {}),
    },
    "",
    path,
  );
  window.dispatchEvent(new PopStateEvent("popstate"));
  if (!replace && !restore) window.scrollTo(0, state.scrollY || 0);
}
export function updateRouteQuery(
  route,
  patch,
  { resetPage = false, ...options } = {},
) {
  const query = {
    ...route.query,
    ...(resetPage ? { cursor: undefined, page: undefined } : {}),
    ...patch,
  };
  navigateConsole({ ...route, query }, options);
}
function cursorPath(route) {
  const query = { ...route.query };
  for (const key of [
    "item_id",
    "manager_id",
    "session_id",
    "lines",
    "tab",
    "view",
    "return_to",
  ])
    delete query[key];
  return consolePath({ ...route, query });
}
export function cursorNavigation(route, nextCursor) {
  const path = cursorPath(route),
    query = route.query || {};
  const stack =
    window.history.state?.cursors || storageGet(`cursors:${path}`) || [];
  const page = query.page ? Number(query.page) : query.cursor ? null : 1;
  const move = (cursor, nextStack, number) => {
    const next = {
      ...route,
      query: {
        ...query,
        cursor: cursor || undefined,
        page: number > 1 ? number : undefined,
      },
    };
    storagePut(`cursors:${cursorPath(next)}`, nextStack);
    navigateConsole(next, { state: { cursors: nextStack } });
  };
  return {
    page,
    canPrevious: stack.length > 0,
    canNext: Boolean(nextCursor),
    previous: () =>
      move(stack.at(-1), stack.slice(0, -1), page ? page - 1 : undefined),
    next: () => {
      if (nextCursor)
        move(
          nextCursor,
          [...stack, query.cursor || null],
          page ? page + 1 : undefined,
        );
    },
    first: () => move(null, [], 1),
  };
}
export function navigationItems(isAdmin) {
  const work = [
    { key: "runs", label: "Runs", admin: false },
    { key: "my-environments", label: "My Environments", admin: false },
    { key: "shared-access", label: "Shared Access", admin: false },
  ];
  return isAdmin
    ? [
        ...work,
        ...[
          ["access", "Access"],
          ["environments", "Environments"],
          ["infrastructure", "Infrastructure"],
          ["system", "System"],
        ].map(([key, label]) => ({ key, label, admin: true })),
      ]
    : work;
}
export function loadPlanForRoute(route) {
  switch (route.section) {
    case "builds":
      return ["builds"];
    case "build":
      return ["build-detail"];
    case "allocation":
      return ["allocation-detail"];
    case "execution":
      return ["execution-detail", "execution-allocations"];
    case "batch":
      return ["batch-summary", "batch-items"];
    case "my-environments":
      return ["my-environments"];
    case "shared-access":
      return ["principal-access"];
    case "access":
      return ["access"];
    case "environments":
      return ["environments"];
    case "infrastructure":
      return [
        route.infrastructureView === "allocations"
          ? "allocations"
          : route.infrastructureView === "builds"
            ? "builds"
            : "managers",
      ];
    case "system":
      return ["system-summary", "config-overrides"];
    default:
      return ["execution-list"];
  }
}
export { BASE_PATH };

export function restoreConsoleScroll() {
  const target = window.history.state?.scrollY;
  if (!Number.isFinite(target)) return () => {};
  let frame,
    stopped = false;
  const start = performance.now();
  const stop = () => {
    stopped = true;
    cancelAnimationFrame(frame);
    window.removeEventListener("wheel", stop);
    window.removeEventListener("touchstart", stop);
  };
  const restore = () => {
    if (stopped) return;
    window.scrollTo(0, target);
    if (
      Math.abs(window.scrollY - target) < 2 ||
      performance.now() - start > 2000
    )
      stop();
    else frame = requestAnimationFrame(restore);
  };
  window.addEventListener("wheel", stop, { passive: true });
  window.addEventListener("touchstart", stop, { passive: true });
  frame = requestAnimationFrame(restore);
  return stop;
}
