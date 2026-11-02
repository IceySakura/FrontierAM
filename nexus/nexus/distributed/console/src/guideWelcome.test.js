import test from "node:test";
import assert from "node:assert/strict";
import { beginGuideVisit, recordGuideVisit, guideVisitKey, GUIDE_RETURN_AFTER_MS } from "./guideWelcome.js";

const now = 1800000000000;
function storage() {
  const values = new Map();
  return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
}

test("welcomes a first visit but not a reload, independently for each account", () => {
  const store = storage();
  assert.equal(beginGuideVisit(store, "alice", now), "first");
  assert.equal(beginGuideVisit(store, "alice", now + 1000), null);
  assert.equal(beginGuideVisit(store, "bob", now + 1000), "first");
});

test("welcomes a return after seven days of inactivity", () => {
  const store = storage();
  recordGuideVisit(store, "alice", now);
  assert.equal(beginGuideVisit(store, "alice", now + GUIDE_RETURN_AFTER_MS), "return");
  assert.equal(beginGuideVisit(store, "alice", now + GUIDE_RETURN_AFTER_MS + 1000), null);
});

test("recent activity resets the inactivity window", () => {
  const store = storage();
  beginGuideVisit(store, "alice", now);
  recordGuideVisit(store, "alice", now + GUIDE_RETURN_AFTER_MS - 1000);
  assert.equal(beginGuideVisit(store, "alice", now + GUIDE_RETURN_AFTER_MS), null);
});

test("invalid saved times recover and disabled storage does not break the page", () => {
  const store = storage();
  store.setItem(guideVisitKey("alice"), "invalid");
  assert.equal(beginGuideVisit(store, "alice", now), "first");
  const blocked = { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); } };
  assert.equal(beginGuideVisit(blocked, "alice", now), "first");
  assert.equal(beginGuideVisit(null, "alice", now), "first");
});

test("waits for an authenticated account and tolerates a clock moving backwards", () => {
  const store = storage();
  assert.equal(beginGuideVisit(store, null, now), null);
  recordGuideVisit(store, "alice", now + 10000);
  assert.equal(beginGuideVisit(store, "alice", now), null);
});
