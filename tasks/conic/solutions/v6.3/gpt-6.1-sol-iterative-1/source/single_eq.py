"""Scalar-dual solution of a strictly convex separable resource-allocation QP."""
import numpy as np


def norm(a):
    return float(np.max(np.abs(a), initial=0.0))


def solve(P, q, A, b, cones, check):
    n, m = len(q), len(b)
    if len(cones.eq) != 1 or not len(cones.pos):
        return None
    pc = np.diff(P.indptr)
    if np.any(pc > 1) or not np.array_equal(P.indices, np.flatnonzero(pc)):
        return None
    p = P.diagonal()
    if np.any(p <= 0):
        return None
    R = A[cones.pos].tocsr()
    counts = np.diff(R.indptr)
    if np.any(counts > 1) or np.any(R.data == 0):
        return None
    rows = cones.pos[np.flatnonzero(counts)]
    cols, vals = R.indices, R.data
    bound = b[rows] / vals
    low, high = vals < 0, vals > 0
    lb, ub = np.full(n, -np.inf), np.full(n, np.inf)
    np.maximum.at(lb, cols[low], bound[low])
    np.minimum.at(ub, cols[high], bound[high])
    if np.any(lb > ub):
        return None
    lr, ur = np.full(n, m, dtype=np.int64), np.full(n, m, dtype=np.int64)
    lm = low & (bound == lb[cols])
    um = high & (bound == ub[cols])
    np.minimum.at(lr, cols[lm], rows[lm])
    np.minimum.at(ur, cols[um], rows[um])
    erow = int(cones.eq[0])
    e0 = A[erow].toarray().ravel()
    escale = norm(e0)
    if not escale:
        return None
    objective_scale = max(norm(p), norm(q), 1e-150)
    pp, qq, e = p / objective_scale, q / objective_scale, e0 / escale
    target = b[erow] / escale
    if np.any(pp <= 0) or not np.isfinite(target):
        return None
    tolerance = 2e-11 * (1 + norm(b)) / escale
    positive, negative = e > 0, e < 0
    minimum = float(e[positive] @ lb[positive] + e[negative] @ ub[negative])
    maximum = float(e[positive] @ ub[positive] + e[negative] @ lb[negative])
    if target < minimum - tolerance or target > maximum + tolerance:
        return None

    def evaluate(lam):
        xx = np.clip((-qq - lam * e) / pp, lb, ub)
        return xx, float(e @ xx) - target

    x, f = evaluate(0.0)
    if not np.isfinite(x).all() or not np.isfinite(f):
        return None
    lam = 0.0
    if abs(f) > tolerance:
        direction = 1.0 if f > 0 else -1.0
        endpoint = direction
        found = False
        for _ in range(80):
            xt, ft = evaluate(endpoint)
            if not np.isfinite(xt).all() or not np.isfinite(ft):
                return None
            if ft * direction <= 0:
                found = True
                break
            endpoint *= 2
        if not found:
            return None
        lo, hi = (0.0, endpoint) if direction > 0 else (endpoint, 0.0)
        lam = .5 * (lo + hi)
        for _ in range(60):
            x, f = evaluate(lam)
            if abs(f) <= tolerance:
                break
            if f > 0:
                lo = lam
            else:
                hi = lam
            free = (x > lb) & (x < ub) & (e != 0)
            slope = float(np.sum(e[free] ** 2 / pp[free]))
            step = lam + f / slope if slope > 0 and np.isfinite(slope) else np.nan
            if not lo < step < hi:
                step = .5 * (lo + hi)
            if abs(step - lam) <= 1e-15 * (1 + abs(lam)):
                lam = step
                break
            lam = step
        x, f = evaluate(lam)
    multiplier = lam * objective_scale / escale
    g = p * x + q + e0 * multiplier
    coeff = np.ones(m)
    coeff[rows] = vals
    z = np.zeros(m)
    lm = (x == lb) & (g > 0) & (lr < m)
    um = (x == ub) & (g < 0) & (ur < m)
    z[lr[lm]] = -g[lm] / coeff[lr[lm]]
    z[ur[um]] = -g[um] / coeff[ur[um]]
    z[erow] = multiplier
    s = b - A @ x
    s[cones.pos] = np.maximum(s[cones.pos], 0)
    s[cones.eq] = 0
    out = {"status": "solved", "x": x, "s": s, "z": z}
    return out if check(out) else None
