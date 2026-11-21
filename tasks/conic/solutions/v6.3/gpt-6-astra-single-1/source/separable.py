import numpy as np
from common import blocks


def solve(P, q, A, b, cones):
    n, m = len(q), len(b)
    if any(c["type"] not in ("zero", "nonnegative") for c in cones):
        return None
    if P.nnz > n or (P.nnz and np.any(P.indices != np.repeat(np.arange(n), np.diff(P.indptr)))):
        return None
    counts = np.bincount(A.indices, minlength=m)
    if np.max(counts, initial=0) > 1:
        return None
    eq = np.zeros(m, dtype=bool)
    for kind, d, i, j in blocks(cones):
        if kind == "zero":
            eq[i:j] = True
    empty_bad = (counts == 0) & np.where(eq, b != 0, b < 0)
    if np.any(empty_bad):
        row = np.flatnonzero(empty_bad)[0]
        z = np.zeros(m)
        z[row] = -np.sign(b[row])
        return {"status": "primal_infeasible", "z": z}
    rows = A.indices
    cols = np.repeat(np.arange(n), np.diff(A.indptr))
    aa = A.data
    values = b[rows] / aa
    lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
    low = (aa < 0) | eq[rows]
    upp = (aa > 0) | eq[rows]
    np.maximum.at(lower, cols[low], values[low])
    np.minimum.at(upper, cols[upp], values[upp])
    lrow, urow = np.full(n, -1, dtype=int), np.full(n, -1, dtype=int)
    lowactive = low & (values == lower[cols])
    upactive = upp & (values == upper[cols])
    np.maximum.at(lrow, cols[lowactive], rows[lowactive])
    np.maximum.at(urow, cols[upactive], rows[upactive])
    coeff = np.zeros(m)
    coeff[rows] = aa
    if np.any(lower > upper):
        variable = np.flatnonzero(lower > upper)[0]
        r1, r2 = lrow[variable], urow[variable]
        scale = min(abs(coeff[r1]), abs(coeff[r2]))
        z = np.zeros(m)
        z[r1], z[r2] = -scale / coeff[r1], scale / coeff[r2]
        return {"status": "primal_infeasible", "z": z}
    h = P.diagonal()
    curved = h > 0
    down = (~curved) & (((q > 0) & ~np.isfinite(lower)) | ((q < 0) & ~np.isfinite(upper)))
    if np.any(down):
        x = np.zeros(n)
        x[down] = -np.sign(q[down])
        return {"status": "dual_infeasible", "x": x, "s": -A @ x}
    x = np.zeros(n)
    np.divide(-q, h, out=x, where=curved)
    choose_low, choose_up = (~curved) & (q > 0), (~curved) & (q < 0)
    x[choose_low], x[choose_up] = lower[choose_low], upper[choose_up]
    x = np.clip(x, lower, upper)
    gradient = h * x + q
    z = np.zeros(m)
    pos, neg = (gradient > 0) & (lrow >= 0), (gradient < 0) & (urow >= 0)
    z[lrow[pos]] = -gradient[pos] / coeff[lrow[pos]]
    z[urow[neg]] = -gradient[neg] / coeff[urow[neg]]
    s = b - A @ x
    s[eq] = 0
    s[~eq] = np.maximum(s[~eq], 0)
    return {"status": "solved", "x": x, "s": s, "z": z}
