"""HiGHS LP/QP interface with certificate construction and KKT refinement."""

import time
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla
from scipy.optimize._highspy import _core as hc
from common import norm, Problem, assess, equilibrate


def highs(P, q, A, lower, upper, col_lower=None, col_upper=None,
          method="choose", time_limit=260.0, return_solver=False):
    n, m = len(q), len(lower)
    h = hc._Highs()
    for k, v in (("output_flag", False), ("threads", 1), ("parallel", "off"),
                 ("primal_feasibility_tolerance", 1e-10),
                 ("dual_feasibility_tolerance", 1e-10),
                 ("ipm_optimality_tolerance", 1e-12),
                 ("small_matrix_value", 1e-12), ("time_limit", time_limit)):
        h.setOptionValue(k, v)
    if method != "choose":
        h.setOptionValue("solver", method)
    lp = hc.HighsLp()
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = np.asarray(q, dtype=float)
    lp.col_lower_ = np.full(n, -hc.kHighsInf) if col_lower is None else col_lower
    lp.col_upper_ = np.full(n, hc.kHighsInf) if col_upper is None else col_upper
    lp.row_lower_, lp.row_upper_ = lower, upper
    am = lp.a_matrix_
    am.num_col_, am.num_row_ = n, m
    am.format_ = hc.MatrixFormat.kColwise
    am.start_, am.index_, am.value_ = A.indptr, A.indices, A.data
    model = hc.HighsModel()
    model.lp_ = lp
    if P is not None and P.nnz:
        T = sparse.tril(P, format="csc")
        hh = model.hessian_
        hh.dim_ = n
        hh.format_ = hc.HessianFormat.kTriangular
        hh.start_, hh.index_, hh.value_ = T.indptr, T.indices, T.data
        h.setOptionValue("qp_regularization_value", 1e-12)
        h.setOptionValue("qp_iteration_limit", 200000)
    h.passModel(model)
    h.run()
    if return_solver:
        return h
    sol = h.getSolution()
    return h.getModelStatus(), np.asarray(sol.col_value), -np.asarray(sol.row_dual)


def polish(p, x, z, tries=4):
    """Solve the exact active KKT equations, without objective regularization."""
    s = p.b - p.A @ x
    best = (x, s, z)
    _, besterr = assess(p, *best)
    if besterr < 2e-10:
        return best
    for threshold in (1e-7, 1e-9, 1e-5, 1e-11)[:tries]:
        active = p.eq | ((s < threshold * (1 + norm(s))) & (z > threshold * 0.01))
        ids = np.flatnonzero(active)
        C = p.A[ids].tocsc()
        K = sparse.bmat([[p.P, C.T], [C, None]], format="csc")
        rhs = np.r_[-p.q, p.b[ids]]
        try:
            from ipm import LinearSystem, BandedSystem
            if p.n >= 3000 and p.P.nnz and np.min(p.P.diagonal()) > 0:
                fac = BandedSystem(p.P, C)
            else:
                fac = LinearSystem(p.P, C, reg=1e-12)
            v = fac.solve(rhs)
        except (RuntimeError, ValueError):
            reg = 1e-12 * max(1.0, norm(K.data))
            Kreg = K + sparse.diags(np.r_[np.full(p.n, reg), np.full(len(ids), -reg)])
            try:
                fac = sla.splu(Kreg.tocsc())
                v = np.r_[x, z[ids]]
                for _ in range(5):
                    v += fac.solve(rhs - K @ v)
            except (RuntimeError, ValueError):
                continue
        zz = np.zeros(p.m)
        zz[ids] = v[p.n:]
        xx = v[:p.n]
        ss = p.b - p.A @ xx
        ok, err = assess(p, xx, ss, zz)
        if err < besterr:
            best, besterr = (xx, ss, zz), err
        if ok:
            return best
    return best


def certificates(p, hint=None, time_limit=60.0):
    started = time.monotonic()
    a, h = norm(p.A.data) or 1.0, norm(p.P.data) or 1.0
    if norm(p.b):
        # The box makes the homogeneous Farkas problem bounded.
        lo = np.where(p.eq, -1.0, 0.0)
        status, z, _ = highs(None, p.b / norm(p.b), (p.A.T / a).tocsc(),
                             np.zeros(p.n), np.zeros(p.n), lo, np.ones(p.m),
                             time_limit=max(0.1, time_limit * 0.55))
        if z.size == p.m and assess(p, z=z, status="primal_infeasible")[0]:
            return {"status": "primal_infeasible", "z": z}
    if norm(p.q):
        mat = sparse.vstack((p.A / a, p.P / h), format="csc")
        upper = np.zeros(p.m + p.n)
        lower = np.r_[np.where(p.eq, 0.0, -hc.kHighsInf), np.zeros(p.n)]
        status, x, _ = highs(None, p.q / norm(p.q), mat, lower, upper,
                             -np.ones(p.n), np.ones(p.n),
                             time_limit=max(0.1, time_limit - (time.monotonic() - started)))
        if x.size == p.n:
            s = -p.A @ x
            if assess(p, x=x, s=s, status="dual_infeasible")[0]:
                return {"status": "dual_infeasible", "x": x, "s": s}
    return {"status": "unresolved"}


def zero_lower_bound(p, time_limit):
    diagonal = p.P.diagonal()
    if p.P.nnz != np.count_nonzero(diagonal):
        return None
    A = p.A.tocsr()
    degree = np.diff(A.indptr)
    rows = np.flatnonzero(~p.eq & (degree == 1) & (p.b == 0))
    cols = A.indices[A.indptr[rows]]
    vals = A.data[A.indptr[rows]]
    eligible = vals * p.q[cols] < 0
    rows, cols, vals = rows[eligible], cols[eligible], vals[eligible]
    chosen = np.full(p.n, p.m, dtype=int)
    np.minimum.at(chosen, cols, rows)
    cost = np.flatnonzero(p.q != 0)
    if np.any(chosen[cost] == p.m):
        return None
    z = np.zeros(p.m)
    if len(cost):
        rr = chosen[cost]
        z[rr] = -p.q[cost] / A.data[A.indptr[rr]]
    fixed = (diagonal > 0) | (p.q != 0)
    lo, up = np.full(p.n, -hc.kHighsInf), np.full(p.n, hc.kHighsInf)
    lo[fixed], up[fixed] = 0, 0
    status, x, _ = highs(None, np.zeros(p.n), p.A, np.where(p.eq, p.b, -hc.kHighsInf), p.b,
                         lo, up, method="ipm" if p.n > 2000 else "simplex",
                         time_limit=min(3.0, max(0.1, time_limit * 0.15)))
    if x.size == p.n:
        x[fixed] = 0
        for _ in range(3):
            slack = p.b - p.A @ x
            violations = np.flatnonzero(~p.eq & (degree == 1) & (slack < 0))
            if not len(violations):
                break
            cc = A.indices[A.indptr[violations]]
            vv = A.data[A.indptr[violations]]
            good = ~fixed[cc]
            x[cc[good]] = p.b[violations[good]] / vv[good]
        s = p.b - p.A @ x
        if assess(p, x, s, z)[0]:
            return {"status": "solved", "x": x, "s": s, "z": z}
    return None


def solve_poly(p, time_limit=260.0):
    started = time.monotonic()
    if p.n == 0:
        z = np.zeros(p.m)
        if assess(p, np.zeros(0), p.b, z)[0]:
            return {"status": "solved", "x": np.zeros(0), "s": p.b, "z": z}
        return certificates(p, time_limit=min(time_limit, 60))
    if p.P.nnz and p.n >= 100:
        zero = zero_lower_bound(p, time_limit)
        if zero is not None:
            return zero
    if p.P.nnz and p.n >= 2000 and not np.any(p.q):
        mat = sparse.vstack((p.A, p.P), format="csc")
        lower = np.r_[np.where(p.eq, p.b, -hc.kHighsInf), np.zeros(p.n)]
        upper = np.r_[p.b, np.zeros(p.n)]
        st, x, _ = highs(None, np.zeros(p.n), mat, lower, upper, method="ipm",
                         time_limit=min(5, time_limit * 0.1))
        if x.size == p.n:
            s, z = p.b - p.A @ x, np.zeros(p.m)
            if assess(p, x, s, z)[0]:
                return {"status": "solved", "x": x, "s": s, "z": z}
    if p.P.nnz and p.n < 2000:
        try:
            x, s, z = polish(p, np.zeros(p.n), np.zeros(p.m), tries=1)
            if assess(p, x, s, z)[0]:
                return {"status": "solved", "x": x, "s": s, "z": z}
        except (RuntimeError, ValueError, MemoryError):
            pass
    if p.P.nnz and not np.all(p.eq):
        from ipm import solve_conic
        r = solve_conic(p, time_limit=min(time_limit, 220))
        if r["status"] != "unresolved" and assess(
                p, status=r["status"], **{k: r[k] for k in ("x", "s", "z") if k in r})[0]:
            return r
    sp, D, E, c = equilibrate(p, 4)
    remaining = time_limit - (time.monotonic() - started)
    if remaining <= 0.1:
        return {"status": "unresolved"}
    lower = np.where(sp.eq, sp.b, -hc.kHighsInf)
    method = "ipm" if not p.P.nnz and p.n >= 3000 else "choose"
    status, x, z = highs(sp.P, sp.q, sp.A, lower, sp.b, method=method, time_limit=remaining)
    if x.size == p.n and z.size == p.m and np.isfinite(x).all():
        x, z = x * D, z * E / c
        s = p.b - p.A @ x
        if assess(p, x, s, z)[0]:
            return {"status": "solved", "x": x, "s": s, "z": z}
        if (status == hc.HighsModelStatus.kOptimal and p.n + p.m < 12000) or (
                p.P.nnz and p.n <= 20000 and np.min(p.P.diagonal(), initial=1) > 0):
            x, s, z = polish(p, x, z)
            if assess(p, x, s, z)[0]:
                return {"status": "solved", "x": x, "s": s, "z": z}
    remaining = time_limit - (time.monotonic() - started)
    return certificates(p, time_limit=min(remaining, 75)) if remaining > 0.1 else {"status": "unresolved"}
