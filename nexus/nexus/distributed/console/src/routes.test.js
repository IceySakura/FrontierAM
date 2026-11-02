import test from "node:test";
import assert from "node:assert/strict";
import {
  consolePath,
  parseConsoleLocation,
  navigationItems,
  loadPlanForRoute,
  safeReturnPath,
  routeFromPath,
  cursorNavigation,
  navigateConsole,
  setNavigationIdentity,
  savedListRoute,
} from "./routes.js";

test("legacy batch URLs resolve to the unified Run and selected subtask", () => {
  assert.deepEqual(parseConsoleLocation("/v1/console/batches"), {
    section: "runs",
    query: { type: "batch" },
  });
  const route = parseConsoleLocation(
    "/v1/console/batches/batch-1/items/item-1",
    "?item_status=failed&cursor=page2",
  );
  assert.deepEqual(route, {
    section: "execution",
    executionId: "batch-1",
    query: {
      item_status: "failed",
      cursor: "page2",
      tab: "items",
      item_id: "item-1",
    },
  });
  assert.equal(
    consolePath({ section: "batches" }),
    "/v1/console/runs?type=batch",
  );
  assert.equal(
    consolePath({ section: "batch", batchId: "a/b", itemId: "item 1" }),
    "/v1/console/executions/a%2Fb?tab=items&item_id=item+1",
  );
});
test("origin retains the complete Batch filter, page and selected item through an attempt", () => {
  const origin =
    "/v1/console/executions/batch?tab=items&item_id=task&cursor=opaque&page=2&item_status=failed";
  const child = {
    section: "execution",
    executionId: "attempt",
    query: { return_to: origin, tab: "manager-logs" },
  };
  assert.deepEqual(routeFromPath(consolePath(child)), child);
  assert.equal(
    parseConsoleLocation("/v1/console/executions/child", "?from_batch=parent")
      .query.return_to,
    "/v1/console/executions/parent?tab=items",
  );
});
test("normal navigation has one Run entry and hides global build diagnostics", () => {
  assert.deepEqual(
    navigationItems(false).map((x) => x.key),
    ["runs", "my-environments", "shared-access"],
  );
  assert.deepEqual(
    loadPlanForRoute({
      section: "infrastructure",
      infrastructureView: "builds",
    }),
    ["builds"],
  );
  assert.deepEqual(
    loadPlanForRoute({ section: "allocation", allocationId: "a" }),
    ["allocation-detail"],
  );
});
test("query validation rejects unsafe return paths and invalid numeric sizes", () => {
  for (const path of [
    "https://evil.test/v1/console/runs",
    "//evil.test",
    "/v1/console/../../admin",
    "/v1/console/\\evil",
    "/v1/console/%2e%2e/%2e%2e/admin",
  ])
    assert.equal(safeReturnPath(path), null);
  assert.deepEqual(
    parseConsoleLocation(
      "/v1/console/runs",
      "?type=invalid&page_size=100000&token=secret&page=-3",
    ),
    { section: "runs" },
  );
  assert.equal(
    parseConsoleLocation(
      "/v1/console/admin/infrastructure/allocations",
      "?allocation_type=all",
    ).query.allocation_type,
    "all",
  );
});
test("cursor navigation survives detail round trips and is isolated per principal", () => {
  const storage = new Map();
  globalThis.sessionStorage = {
    getItem: (k) => storage.get(k),
    setItem: (k, v) => storage.set(k, v),
  };
  globalThis.PopStateEvent = class {
    constructor(type) {
      this.type = type;
    }
  };
  let url = new URL("https://console.test/v1/console/runs?type=batch");
  globalThis.window = {
    get location() {
      return url;
    },
    scrollY: 240,
    scrollTo() {},
    dispatchEvent() {},
    history: {
      state: {},
      replaceState(state, _, path) {
        this.state = state;
        if (path) url = new URL(path, url);
      },
      pushState(state, _, path) {
        this.state = state;
        url = new URL(path, url);
      },
    },
  };
  setNavigationIdentity("alice");
  let route = () => parseConsoleLocation(url.pathname, url.search);
  cursorNavigation(route(), "cursor2").next();
  assert.equal(route().query.cursor, "cursor2");
  assert.equal(cursorNavigation(route(), null).page, 2);
  const origin = consolePath(route());
  navigateConsole({
    section: "execution",
    executionId: "b",
    query: { return_to: origin },
  });
  assert.equal(savedListRoute("runs").query.cursor, "cursor2");
  navigateConsole(origin, { restore: true });
  assert.equal(window.history.state.scrollY, 240);
  assert.equal(cursorNavigation(route(), null).canPrevious, true);
  cursorNavigation(route(), null).previous();
  assert.equal(route().query.cursor, undefined);
  navigateConsole({
    section: "execution",
    executionId: "batch",
    query: { tab: "items" },
  });
  cursorNavigation(route(), "items-page-2").next();
  navigateConsole({ ...route(), query: { ...route().query, item_id: "task" } });
  assert.equal(cursorNavigation(route(), null).canPrevious, true);
  navigateConsole(
    { ...route(), query: { ...route().query, tab: "resources" } },
    { replace: true },
  );
  navigateConsole(
    { ...route(), query: { ...route().query, tab: "items" } },
    { replace: true },
  );
  assert.equal(cursorNavigation(route(), null).canPrevious, true);
  cursorNavigation(route(), null).previous();
  assert.equal(route().query.cursor, undefined);
  setNavigationIdentity("bob");
  assert.deepEqual(savedListRoute("runs"), { section: "runs" });
  delete globalThis.window;
  delete globalThis.sessionStorage;
  delete globalThis.PopStateEvent;
});

test("Batch view URLs retain the Main section and selected subtask", () => {
  const route = {
    section: "execution",
    executionId: "batch",
    query: {
      view: "subtasks",
      tab: "resources",
      item_id: "task/2",
      cursor: "page2",
      page: "2",
    },
  };
  assert.deepEqual(routeFromPath(consolePath(route)), route);
});
