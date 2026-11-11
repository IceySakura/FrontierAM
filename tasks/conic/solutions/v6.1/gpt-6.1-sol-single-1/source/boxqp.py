"""Semismooth dual Newton for diagonal quadratic programs with box bounds."""

import numpy as np
from scipy import linalg, sparse


def solve_box(p):
    from solver import accepted, feasible_slack
    diag = p.P.diagonal()
    if (p.n == 0 or np.any(diag <= 0) or p.P.nnz != np.count_nonzero(diag)
            or len(p.eq) > 30):
        return None
    B = p.Ar[p.ineq]
    if np.any(np.diff(B.indptr) != 1):
        return None
    cols, vals = B.indices, B.data
    bounds = p.b[p.ineq] / vals
    lower, upper = np.full(p.n, -np.inf), np.full(p.n, np.inf)
    positive = vals > 0
    np.minimum.at(upper, cols[positive], bounds[positive])
    np.maximum.at(lower, cols[~positive], bounds[~positive])
    if np.any(lower > upper):
        return None
    F = p.A[p.eq].tocsc()
    b = p.b[p.eq].copy()
    sc = np.maximum(np.asarray(abs(F).max(axis=1).toarray()).ravel(), 1e-12)
    F = sparse.diags(1 / sc) @ F
    b /= sc
    y = np.zeros(len(p.eq))
    q = p.q

    def point(y):
        c = q + F.T @ y
        x = np.clip(-c / diag, lower, upper)
        g = F @ x - b
        obj = float(np.sum((0.5 * diag * x + q) * x) + y @ g)
        return x, g, obj, c

    for it in range(100):
        x, g, obj, c = point(y)
        z = np.zeros(p.m)
        z[p.eq] = y / sc
        station = diag * x + c
        used = np.zeros(p.n, dtype=bool)
        for r, col, a, bound in zip(p.ineq, cols, vals, bounds):
            if not used[col] and abs(x[col] - bound) <= 1e-7 * max(1, abs(bound)):
                v = -station[col] / a
                if v >= -1e-9:
                    z[r] = max(0.0, v)
                    used[col] = True
        result = {"status": "solved", "x": x, "s": feasible_slack(p, x), "z": z}
        if accepted(p, result, tol=3e-9):
            return result
        raw = -c / diag
        active = (raw >= lower - 1e-12) & (raw <= upper + 1e-12) & (upper > lower)
        H = (F.multiply((active / diag)[None, :]) @ F.T).toarray()
        if len(y) == 0:
            break
        try:
            dy = linalg.solve(H, g, assume_a="sym", check_finite=False)
        except linalg.LinAlgError:
            dy = linalg.lstsq(H, g, check_finite=False)[0]
        if not np.isfinite(dy).all():
            break
        slope = float(g @ dy)
        alpha = 1.0
        for _ in range(40):
            yn = y + alpha * dy
            if point(yn)[2] >= obj + 1e-4 * alpha * slope - 1e-10 * max(1, abs(obj)):
                break
            alpha *= 0.5
        y = yn
    return None
