export const GUIDE_RETURN_AFTER_MS = 7 * 24 * 60 * 60 * 1000;

export function guideVisitKey(principalId) {
  return `nexus.console.guide.last-visit:${encodeURIComponent(principalId)}`;
}

export function recordGuideVisit(storage, principalId, now = Date.now()) {
  try {
    storage?.setItem(guideVisitKey(principalId), String(now));
  } catch {
    // A disabled/full browser store must not prevent using the Console.
  }
}

export function beginGuideVisit(storage, principalId, now = Date.now()) {
  if (!principalId) return null;
  let lastVisit;
  try {
    lastVisit = Number(storage?.getItem(guideVisitKey(principalId)));
  } catch {
    lastVisit = 0;
  }
  recordGuideVisit(storage, principalId, now);
  if (!Number.isFinite(lastVisit) || lastVisit <= 0) return "first";
  return now - lastVisit >= GUIDE_RETURN_AFTER_MS ? "return" : null;
}
