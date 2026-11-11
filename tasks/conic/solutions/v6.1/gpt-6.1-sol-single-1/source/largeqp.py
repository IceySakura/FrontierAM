"""Sparse least-squares feasibility shortcut for zero-loss quadratic programs."""

import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr


def zero_loss(p):
    from solver import accepted, residuals, feasible_slack, log, norm
    begin = time.monotonic()
    z = np.zeros(p.m)
    rows = p.Ar[p.ineq]
    singleton = np.flatnonzero((np.diff(rows.indptr) == 1) & (p.b[p.ineq] == 0))
    ix = rows.indptr[singleton]
    cols, vals = rows.indices[ix], rows.data[ix]
    needed = -p.q[cols] / vals
    use = (needed > 0)
    cols, rr, needed = cols[use], p.ineq[singleton[use]], needed[use]
    if len(cols):
        _, unique = np.unique(cols, return_index=True)
        z[rr[unique]] = needed[unique]
    if norm(p.q + p.A.T @ z) > 1e-11 * max(1.0, norm(p.q)):
        return None
    free = np.flatnonzero((p.q == 0) & (np.diff(p.P.indptr) == 0) &
                          (np.diff(p.A.indptr) != 0))
    if len(free) == 0:
        return None
    C = p.A[p.eq][:, free].tocsr()
    rhs = p.b[p.eq]
    cs = np.sqrt(np.asarray(C.multiply(C).sum(axis=0)).ravel())
    cs = 1 / np.maximum(cs, 1e-10)
    C = C @ sparse.diags(cs)
    rs = np.sqrt(np.asarray(C.multiply(C).sum(axis=1)).ravel())
    rs = 1 / np.maximum(rs, 1e-10)
    C = sparse.diags(rs) @ C
    br = rs * rhs
    sol = np.zeros(len(free))
    for it in range(5):
        sol = lsmr(C, br, atol=1e-12, btol=1e-12, maxiter=150, x0=sol)[0]
        x = np.zeros(p.n)
        x[free] = cs * sol
        r = {"status": "solved", "x": x, "s": feasible_slack(p, x), "z": z}
        vals = residuals(p, r)
        log("zero-loss", it, vals, "t", time.monotonic() - begin)
        if max(vals) < 6e-9:
            return r
        if time.monotonic() - begin > 15:
            break
    return None


def solve_lasso(p):
    from solver import accepted, residuals, feasible_slack, log, norm
    from native import Lasso
    from fast import remaining
    begin = time.monotonic()
    if p.nonlinear or len(p.eq) == 0:
        return None
    diagonal = p.P.diagonal()
    residual_cols = np.flatnonzero(diagonal > 0)
    if (len(residual_cols) != len(p.eq) or
            p.P.nnz != len(residual_cols) or np.any(p.q[residual_cols] != 0)):
        return None
    F = p.A[p.eq].tocsc()
    eqdegree = np.diff(F.indptr)
    if np.any(eqdegree[residual_cols] != 1):
        return None
    positions = F.indptr[residual_cols]
    residual_rows, coefficients = F.indices[positions], F.data[positions]
    if len(np.unique(residual_rows)) != len(p.eq):
        return None
    R = p.Ar[p.ineq]
    if np.any(p.b[p.ineq] != 0) or np.any(np.diff(R.indptr) != 2):
        return None
    ci, av = R.indices.reshape(-1, 2), R.data.reshape(-1, 2)
    swap = ((p.q[ci[:, 0]] > 0) |
            ((eqdegree[ci[:, 0]] == 0) & (eqdegree[ci[:, 1]] > 0)))
    choice = swap.astype(np.int64)
    rows = np.arange(len(p.ineq))
    feature = ci[rows, choice]
    auxiliary = ci[rows, 1 - choice]
    a_feature, a_aux = av[rows, choice], av[rows, 1 - choice]
    if (np.any(p.q[feature] != 0) or np.any(p.q[auxiliary] < 0) or
            np.any(diagonal[feature] != 0) or np.any(diagonal[auxiliary] != 0) or
            np.any(eqdegree[auxiliary] != 0) or np.any(a_aux >= 0)):
        return None
    counts = np.bincount(feature, minlength=p.n)
    feature_cols = np.flatnonzero(counts)
    if np.any(counts[feature_cols] != 2):
        return None
    upper, lower = a_feature > 0, a_feature < 0
    if np.sum(upper) != len(feature_cols) or np.sum(lower) != len(feature_cols):
        return None
    up_order, low_order = np.argsort(feature[upper]), np.argsort(feature[lower])
    uf, lf = feature[upper][up_order], feature[lower][low_order]
    if not np.array_equal(uf, feature_cols) or not np.array_equal(lf, feature_cols):
        return None
    aux = auxiliary[upper][up_order]
    if not np.array_equal(aux, auxiliary[lower][low_order]) or len(np.unique(aux)) != len(aux):
        return None
    used = np.zeros(p.n, dtype=bool)
    used[feature_cols], used[aux], used[residual_cols] = True, True, True
    if np.any((~used) & ((np.diff(p.A.indptr) > 0) | (p.q != 0))):
        return None
    au, al = a_feature[upper][up_order], a_feature[lower][low_order]
    tu, tl = -a_aux[upper][up_order], -a_aux[lower][low_order]
    cu, cl = au / tu, -al / tl
    z_upper_rows = p.ineq[upper][up_order]
    z_lower_rows = p.ineq[lower][low_order]
    observed = feature_cols[eqdegree[feature_cols] > 0]
    obs_pos = np.searchsorted(feature_cols, observed)
    X = F[:, observed].tocsc()
    weights = np.zeros(len(p.eq))
    weights[residual_rows] = diagonal[residual_cols] / coefficients ** 2
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        return None
    dscale, bscale = norm(weights), norm(p.b[p.eq]) or 1.
    factor = dscale * bscale * bscale
    if not np.isfinite(factor) or factor == 0:
        return None
    lam = p.q[aux]
    lp = lam[obs_pos] * cu[obs_pos] / (dscale * bscale)
    lm = lam[obs_pos] * cl[obs_pos] / (dscale * bscale)
    rhs = p.b[p.eq] / bscale
    cd = Lasso(X, weights / dscale, lp, lm, rhs)
    for iteration in range(250):
        cd.run(1 if iteration < 4 else 5)
        cd.residual[:] = rhs - X @ cd.x
        violation, gap, obj = cd.metrics()
        if iteration % 10 == 0:
            log("lasso", iteration, "violation", violation, "gap", gap,
                "t", time.monotonic() - begin)
        if (violation * dscale * bscale <= 1e-9 * (1 + norm(p.q)) and
                abs(gap) * factor <= 1e-9 * max(1., abs(obj) * factor)):
            x = np.zeros(p.n)
            x[observed] = cd.x * bscale
            r = p.b[p.eq] - X @ x[observed]
            x[residual_cols] = r[residual_rows] / coefficients
            values = x[feature_cols]
            x[aux] = np.where(values >= 0, cu * values, -cl * values)
            z = np.zeros(p.m)
            z[p.eq] = -weights * r
            gradient = np.zeros(len(feature_cols))
            gradient[obs_pos] = X.T @ z[p.eq]
            z[z_upper_rows] = np.maximum(0., (cl * lam - gradient) / ((cu + cl) * tu))
            z[z_lower_rows] = np.maximum(0., (cu * lam + gradient) / ((cu + cl) * tl))
            answer = {"status": "solved", "x": x, "s": feasible_slack(p, x), "z": z}
            if accepted(p, answer, tol=8e-9):
                return answer
            log("lasso residuals", residuals(p, answer))
        if time.monotonic() - begin > 25. or remaining() < 2.:
            break
    return None
