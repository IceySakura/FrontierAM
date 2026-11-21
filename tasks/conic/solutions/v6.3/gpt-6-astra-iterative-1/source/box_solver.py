import numpy as np
from problem import assess


def solve(problem):
    """Separable convex objectives with one variable per affine row."""
    P, q, A, b = problem.P, problem.q, problem.A, problem.b
    diag = P.diagonal()
    if P.nnz != np.count_nonzero(diag):
        return None
    Ar = A.tocsr()
    counts = np.diff(Ar.indptr)
    if np.max(counts, initial=0) > 1:
        return None
    n, m = problem.n, problem.m
    eq = np.zeros(m, dtype=bool)
    start = 0
    for cone in problem.cones:
        if cone["type"] not in ("zero", "nonnegative"):
            return None
        end = start+cone["dim"]
        if cone["type"] == "zero":
            eq[start:end] = True
        start = end
    empty = counts == 0
    bad = np.flatnonzero(empty & ((eq & (b != 0)) | (~eq & (b < 0))))
    if len(bad):
        row = bad[0]
        z = np.zeros(m)
        z[row] = -np.sign(b[row])
        result = {"status": "primal_infeasible", "z": z}
        if assess(problem, result)[0]:
            return result
        return None
    rows = np.flatnonzero(~empty)
    ptr = Ar.indptr[rows]
    cols, coef = Ar.indices[ptr], Ar.data[ptr]
    value = b[rows]/coef
    lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
    lower_rows = (coef < 0) | eq[rows]
    upper_rows = (coef > 0) | eq[rows]
    np.maximum.at(lo, cols[lower_rows], value[lower_rows])
    np.minimum.at(hi, cols[upper_rows], value[upper_rows])
    lorow, hirow = np.full(n, -1, dtype=int), np.full(n, -1, dtype=int)
    match = lower_rows & (value == lo[cols])
    lorow[cols[match]] = rows[match]
    match = upper_rows & (value == hi[cols])
    hirow[cols[match]] = rows[match]
    bad = np.flatnonzero(lo > hi)
    if len(bad):
        col = bad[0]
        lr, ur = lorow[col], hirow[col]
        z = np.zeros(m)
        z[lr] -= 1/Ar.data[Ar.indptr[lr]]
        z[ur] += 1/Ar.data[Ar.indptr[ur]]
        result = {"status": "primal_infeasible", "z": z}
        return result if assess(problem, result)[0] else None
    unbounded = (diag == 0) & (((q < 0) & ~np.isfinite(hi)) |
                               ((q > 0) & ~np.isfinite(lo)))
    ids = np.flatnonzero(unbounded)
    if len(ids):
        x = np.zeros(n)
        x[ids[0]] = -np.sign(q[ids[0]])
        return {"status": "dual_infeasible", "x": x, "s": -A@x}
    x = np.zeros(n)
    curved = diag > 0
    with np.errstate(over="ignore", divide="ignore"):
        x[curved] = -q[curved]/diag[curved]
    x[(~curved) & (q > 0)] = lo[(~curved) & (q > 0)]
    x[(~curved) & (q < 0)] = hi[(~curved) & (q < 0)]
    x = np.clip(x, lo, hi)
    gradient = diag*x+q
    z = np.zeros(m)
    ids = np.flatnonzero((gradient > 0) & (lorow >= 0))
    active = lorow[ids]
    z[active] = -gradient[ids]/Ar.data[Ar.indptr[active]]
    ids = np.flatnonzero((gradient < 0) & (hirow >= 0))
    active = hirow[ids]
    z[active] = -gradient[ids]/Ar.data[Ar.indptr[active]]
    s = np.maximum(b-A@x, 0.)
    s[eq] = 0.
    return {"status": "solved", "x": x, "s": s, "z": z}
