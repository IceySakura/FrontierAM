"""Exact structural reductions for separable polyhedral problems."""

import numpy as np


def free_direction(P, q, A, b):
    unused = (np.diff(A.indptr) == 0) & (np.diff(P.indptr) == 0) & (q != 0)
    idx = np.flatnonzero(unused)
    if not len(idx):
        return None
    j = idx[np.argmax(np.abs(q[idx]))]
    x = np.zeros(len(q))
    x[j] = -np.sign(q[j])
    return {"status": "dual_infeasible", "x": x, "s": np.zeros(len(b))}


def solve(P, q, A, b, cones):
    if any(c["type"] not in ("zero", "nonnegative") for c in cones):
        return None
    n, m = len(q), len(b)
    if P.nnz > n:
        return None
    if P.nnz and np.any(P.indices != np.repeat(np.arange(n), np.diff(P.indptr))):
        return None
    counts = np.bincount(A.indices, minlength=m)
    if np.any(counts > 1):
        return None
    eq = np.zeros(m, dtype=bool)
    off = 0
    for cone in cones:
        if cone["type"] == "zero":
            eq[off:off + cone["dim"]] = True
        off += cone["dim"]
    bad = np.flatnonzero((counts == 0) & np.where(eq, b != 0, b < 0))
    if len(bad):
        i = bad[np.argmax(np.abs(b[bad]))]
        z = np.zeros(m)
        z[i] = -np.sign(b[i]) if eq[i] else 1.
        return {"status": "primal_infeasible", "z": z}
    rows = A.indices
    cols = np.repeat(np.arange(n), np.diff(A.indptr))
    coeff = A.data
    values = b[rows] / coeff
    lower = eq[rows] | (coeff < 0)
    upper = eq[rows] | (coeff > 0)
    lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
    np.maximum.at(lo, cols[lower], values[lower])
    np.minimum.at(hi, cols[upper], values[upper])
    il, iu = np.full(n, m, dtype=np.int64), np.full(n, m, dtype=np.int64)
    select = lower & (values == lo[cols])
    np.minimum.at(il, cols[select], rows[select])
    select = upper & (values == hi[cols])
    np.minimum.at(iu, cols[select], rows[select])
    bad = np.flatnonzero(lo > hi)
    byrow = np.zeros(m)
    byrow[rows] = coeff
    if len(bad):
        j = bad[np.argmax(lo[bad] - hi[bad])]
        z = np.zeros(m)
        z[il[j]] = -1 / byrow[il[j]]
        z[iu[j]] += 1 / byrow[iu[j]]
        return {"status": "primal_infeasible", "z": z}
    diag = P.diagonal()
    unbounded = (diag == 0) & (((q < 0) & np.isposinf(hi)) | ((q > 0) & np.isneginf(lo)))
    idx = np.flatnonzero(unbounded)
    if len(idx):
        j = idx[np.argmax(np.abs(q[idx]))]
        x = np.zeros(n)
        x[j] = -np.sign(q[j])
        return {"status": "dual_infeasible", "x": x, "s": -A @ x}
    x = np.zeros(n)
    curved = diag > 0
    x[curved] = -q[curved] / diag[curved]
    flat = ~curved
    x[flat & (q > 0)] = lo[flat & (q > 0)]
    x[flat & (q < 0)] = hi[flat & (q < 0)]
    x = np.maximum(lo, np.minimum(hi, x))
    grad = diag * x + q
    z = np.zeros(m)
    # A fixed variable can use either equality orientation; all other active
    # multipliers have the sign implied by the selected one-sided bound.
    positive, negative = grad > 0, grad < 0
    selected = np.where(positive, il, iu)
    use = (positive | negative) & (selected < m)
    z[selected[use]] = -grad[use] / byrow[selected[use]]
    s = b - A @ x
    s[eq] = 0
    s[~eq] = np.maximum(s[~eq], 0)
    return {"status": "solved", "x": x, "s": s, "z": z}
