"""HiGHS LP/QP backend, KKT polishing, and normalized ray problems."""
import os
import sys
import importlib.machinery
import importlib.util
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import splu

from accuracy import assess, norm

# Loading the native interface directly avoids importing unrelated optimizers.
_name = "scipy.optimize._highspy._core"
if _name in sys.modules:
    h = sys.modules[_name]
else:
    _path = os.path.join(os.path.dirname(scipy.__file__), "optimize", "_highspy",
                         "_core" + importlib.machinery.EXTENSION_SUFFIXES[0])
    _spec = importlib.util.spec_from_file_location(_name, _path)
    h = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = h
    _spec.loader.exec_module(h)


def highs(P, q, A, lower, upper, lb=None, ub=None, ipm=False):
    n, m = q.size, upper.size
    model = h.HighsModel()
    lp = model.lp_
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = q
    lp.col_lower_ = np.full(n, -np.inf) if lb is None else lb
    lp.col_upper_ = np.full(n, np.inf) if ub is None else ub
    lp.row_lower_, lp.row_upper_ = lower, upper
    lp.a_matrix_.format_ = h.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, m
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    if P is not None and P.nnz:
        L = sparse.tril(P, format="csc")
        model.hessian_.dim_ = n
        model.hessian_.format_ = h.HessianFormat.kTriangular
        model.hessian_.start_ = L.indptr
        model.hessian_.index_ = L.indices
        model.hessian_.value_ = L.data
    sol = h._Highs()
    opts = h.HighsOptions()
    opts.output_flag = False
    opts.log_to_console = False
    opts.threads = 1
    opts.parallel = "off"
    opts.primal_feasibility_tolerance = 1e-9
    opts.dual_feasibility_tolerance = 1e-9
    opts.ipm_optimality_tolerance = 1e-11
    opts.small_matrix_value = 1e-12
    opts.large_matrix_value = 1e20
    opts.time_limit = 250.0
    if ipm:
        opts.solver = "ipm"
    sol.passOptions(opts)
    sol.passModel(model)
    sol.run()
    answer = sol.getSolution()
    if os.environ.get("SOLVER_DEBUG"):
        print("highs", sol.getModelStatus(), sol.getInfo().simplex_iteration_count,
              file=sys.stderr)
    return sol, answer


def polish(P, q, A, b, K, x, z):
    eq = K.zero
    nn = K.nonnegative
    s = b - A @ x
    scale = max(1.0, norm(s), norm(b))
    active = eq | (nn & ((np.abs(s) < 1e-6 * scale) | (z > 1e-6)))
    best = ("solved", x, s, z)
    best_score = assess(P, q, A, b, K, best)[0]
    for it in range(6):
        ids = np.flatnonzero(active)
        B = A[ids, :].tocsc()
        H = sparse.bmat([[P, B.T], [B, None]], format="csc")
        rhs = np.concatenate((-q, b[ids]))
        try:
            lu = splu(H, permc_spec="COLAMD")
        except RuntimeError:
            delta = 1e-11 * max(1.0, norm(P.data), norm(A.data))
            reg = np.r_[np.full(q.size, delta), np.full(ids.size, -delta)]
            try:
                lu = splu(H + sparse.diags(reg, format="csc"))
            except RuntimeError:
                break
        v = np.r_[x, z[ids]]
        for _ in range(4):
            dv = lu.solve(rhs - H @ v)
            v += dv
            if norm(rhs - H @ v) < 1e-11 * (1 + norm(rhs)):
                break
        xn = v[:q.size]
        zn = np.zeros(b.size)
        zn[ids] = v[q.size:]
        sn = b - A @ xn
        trial = ("solved", xn, sn, zn)
        score = assess(P, q, A, b, K, trial)[0]
        if score < best_score:
            best, best_score = trial, score
        if best_score < 0.5:
            return best
        drop = nn & active & (zn < -1e-9 * max(1.0, norm(zn)))
        add = nn & ~active & (sn < -1e-9 * max(1.0, norm(sn)))
        if not np.any(drop | add):
            break
        active[drop] = False
        active[add] = True
        x, z = xn, zn
    return best


def certificate(P, q, A, b, K, primal):
    n, m = q.size, b.size
    if primal:
        lower = np.zeros(m)
        lower[K.zero] = -1.0
        upper = np.ones(m)
        zmodel, sol = highs(None, b / max(1.0, norm(b)), A.T.tocsc(),
                            np.zeros(n), np.zeros(n), lower, upper)
        if len(sol.col_value) == m:
            z = np.asarray(sol.col_value)
            return ("primal_infeasible", None, None, z)
    else:
        AA = A
        low = np.full(m, -np.inf)
        low[K.zero] = 0.0
        up = np.zeros(m)
        if P.nnz:
            scale = norm(P.data) or 1
            AA = sparse.vstack([A, P / scale], format="csc")
            low = np.r_[low, np.zeros(n)]
            up = np.r_[up, np.zeros(n)]
        zmodel, sol = highs(None, q / max(1.0, norm(q)), AA, low, up,
                            -np.ones(n), np.ones(n))
        if len(sol.col_value) == n:
            x = np.asarray(sol.col_value)
            return ("dual_infeasible", x, -A @ x, None)
    return ("unresolved", None, None, None)


def solve_box(P, q, A, b, K, lb, ub, li, ui, rowval):
    n, m = q.size, b.size
    diag = P.diagonal()
    scale = 1 / np.sqrt(np.maximum(diag, max(1e-16, norm(diag)*1e-14)))
    cols = np.repeat(np.arange(n), np.diff(P.indptr))
    H = P.copy()
    H.data *= scale[H.indices] * scale[cols]
    lip = float(np.max(np.bincount(H.indices, weights=np.abs(H.data), minlength=n), initial=1.))
    qq, lo, hi = q*scale, lb/scale, ub/scale
    y = np.clip(np.zeros(n), lo, hi)
    v, t = y.copy(), 1.

    def candidate(xx):
        g = P @ xx + q
        zz = np.zeros(m)
        atlo = (xx <= lb + 1e-10*(1+np.abs(lb))) & (g > 0) & (li < m)
        athi = (xx >= ub - 1e-10*(1+np.abs(ub))) & (g < 0) & (ui < m)
        zz[li[atlo]] = -g[atlo]/rowval[li[atlo]]
        zz[ui[athi]] = -g[athi]/rowval[ui[athi]]
        return "solved", xx, b-A@xx, zz

    best, bestscore = ("unresolved", None, None, None), float("inf")
    for it in range(300):
        yn = np.clip(v - (H @ v + qq)/lip, lo, hi)
        tn = .5*(1+np.sqrt(1+4*t*t))
        momentum = (t-1)/tn
        if float((v-yn) @ (yn-y)) > 0:
            tn, momentum = 1., 0.
        v = yn + momentum*(yn-y)
        y, t = yn, tn
        if it % 15 == 14:
            result = candidate(scale*y)
            score = assess(P, q, A, b, K, result)[0]
            if score < bestscore:
                best, bestscore = result, score
            if score <= .5:
                return result
            if it in (29, 89, 179, 299):
                result = polish(P, q, A, b, K, result[1], result[3])
                score = assess(P, q, A, b, K, result)[0]
                if score < bestscore:
                    best, bestscore = result, score
                if score <= .5:
                    return result
    return best


def solve_polyhedral(P, q, A, b, K):
    m = b.size
    R = A.tocsr()
    counts = np.diff(R.indptr)
    singleton = np.flatnonzero(counts == 1)
    lb, ub = np.full(q.size, -np.inf), np.full(q.size, np.inf)
    li, ui = np.full(q.size, m, dtype=int), np.full(q.size, m, dtype=int)
    rowval = np.zeros(m)
    if singleton.size:
        vals = R.data[R.indptr[singleton]]
        cols = R.indices[R.indptr[singleton]]
        rhs = b[singleton] / vals
        islo = (vals < 0) | K.zero[singleton]
        ishi = (vals > 0) | K.zero[singleton]
        np.maximum.at(lb, cols[islo], rhs[islo])
        np.minimum.at(ub, cols[ishi], rhs[ishi])
        mask = islo & (rhs == lb[cols])
        np.minimum.at(li, cols[mask], singleton[mask])
        mask = ishi & (rhs == ub[cols])
        np.minimum.at(ui, cols[mask], singleton[mask])
        rowval[singleton] = vals
        if np.any(lb > ub):
            j = int(np.flatnonzero(lb > ub)[0])
            zz = np.zeros(m)
            zz[ui[j]] = 1 / rowval[ui[j]]
            zz[li[j]] -= 1 / rowval[li[j]]
            result = ("primal_infeasible", None, None, zz)
            if assess(P, q, A, b, K, result)[0] <= .5:
                return result
            singleton = np.empty(0, dtype=int)
            lb.fill(-np.inf)
            ub.fill(np.inf)
    keep = np.ones(m, dtype=bool)
    keep[singleton] = False
    keep = np.flatnonzero(keep)
    AA = A[keep, :].tocsc() if singleton.size else A
    if (q.size >= 500 and P.nnz and keep.size == 0
            and np.isfinite(lb).all() and np.isfinite(ub).all()):
        result = solve_box(P, q, A, b, K, lb, ub, li, ui, rowval)
        if assess(P, q, A, b, K, result)[0] <= 1.:
            return result
    lower = np.full(keep.size, -np.inf)
    eq = K.zero[keep]
    lower[eq] = b[keep[eq]]
    high, sol = highs(P, q, AA, lower, b[keep], lb, ub)
    status = high.getModelStatus()
    if status == h.HighsModelStatus.kOptimal:
        x = np.asarray(sol.col_value)
        z = np.zeros(m)
        z[keep] = -np.asarray(sol.row_dual)
        if singleton.size:
            cd = np.asarray(sol.col_dual)
            use = (cd > 0) & (li < m)
            z[li[use]] = -cd[use] / rowval[li[use]]
            use = (cd < 0) & (ui < m)
            z[ui[use]] = -cd[use] / rowval[ui[use]]
        result = ("solved", x, b - A @ x, z)
        if assess(P, q, A, b, K, result)[0] <= 0.8:
            return result
        result = polish(P, q, A, b, K, x, z)
        if assess(P, q, A, b, K, result)[0] <= 1.0:
            return result
    elif status in (h.HighsModelStatus.kInfeasible,
                    h.HighsModelStatus.kUnboundedOrInfeasible,
                    h.HighsModelStatus.kUnbounded):
        primal_first = status != h.HighsModelStatus.kUnbounded
        for primal in (primal_first, not primal_first):
            cert = certificate(P, q, A, b, K, primal)
            if assess(P, q, A, b, K, cert)[0] <= 1.0:
                return cert
    return ("unresolved", None, None, None)
