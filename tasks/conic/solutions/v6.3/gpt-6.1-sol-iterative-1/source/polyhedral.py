"""LP/QP path using the offline HiGHS library bundled with SciPy."""

import os
import numpy as np
from scipy import sparse
from numeric import linalg, splu
from fastpoly import highs_module

h = highs_module()


def model(P, q, A, lower, upper, col_lower=None, col_upper=None):
    n, m = len(q), len(upper)
    obj = h.HighsModel()
    lp = obj.lp_
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = q
    lp.col_lower_ = np.full(n, -h.kHighsInf) if col_lower is None else col_lower
    lp.col_upper_ = np.full(n, h.kHighsInf) if col_upper is None else col_upper
    lp.row_lower_, lp.row_upper_ = lower, upper
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, m
    lp.a_matrix_.format_ = h.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    if P.nnz:
        H = sparse.tril(P, format="csc")
        obj.hessian_.dim_ = n
        obj.hessian_.format_ = h.HessianFormat.kTriangular
        obj.hessian_.start_, obj.hessian_.index_, obj.hessian_.value_ = H.indptr, H.indices, H.data
    highs = h._Highs()
    for key, val in (("output_flag", False), ("threads", 1), ("parallel", "off"),
                     ("primal_feasibility_tolerance", 1e-10), ("dual_feasibility_tolerance", 1e-10),
                     ("ipm_optimality_tolerance", 1e-12), ("small_matrix_value", 1e-12),
                     ("qp_regularization_value", 1e-12)):
        highs.setOptionValue(key, val)
    if not P.nnz:
        highs.setOptionValue("solver", os.getenv("HIGHS_SOLVER", "ipm" if n >= 300 else "choose"))
        highs.setOptionValue("run_crossover", os.getenv("HIGHS_CROSSOVER", "on"))
    highs.passModel(obj)
    highs.run()
    return highs


def polish(P, q, A, b, eq, x, z):
    slack = b - A @ x
    active = eq | ((np.abs(slack) < 1e-6 * (1 + np.abs(b))) & (z > 1e-9))
    idx = np.flatnonzero(active)
    B = A[idx, :].tocsc()
    K = sparse.bmat([[P, B.T], [B, None]], format="csc")
    v = np.concatenate((x, z[idx]))
    rhs = np.concatenate((-q, b[idx]))
    if K.shape[0] <= 700:
        correction = linalg.lstsq(K.toarray(), rhs - K @ v, cond=1e-13,
                                  lapack_driver="gelsy", check_finite=False)[0]
    else:
        reg = np.concatenate((np.full(len(x), 1e-10), np.full(len(idx), -1e-10)))
        try:
            lu = splu(K + sparse.diags(reg, format="csc"))
            correction = lu.solve(rhs - K @ v)
            for _ in range(3):
                residual = rhs - K @ (v + correction)
                correction += lu.solve(residual)
        except RuntimeError:
            from scipy.sparse.linalg import lsmr
            correction = lsmr(K, rhs - K @ v, atol=1e-13, btol=1e-13, maxiter=300)[0]
    v += correction
    zz = np.zeros_like(z)
    zz[idx] = v[len(x):]
    if np.min(zz[~eq], initial=0) >= -1e-9:
        return v[:len(x)], zz
    return x, z


def solve(P, q, A, b, cones):
    n, m = len(q), len(b)
    eq = np.zeros(m, dtype=bool)
    pos = 0
    for c in cones:
        if c["type"] == "zero":
            eq[pos:pos + c["dim"]] = True
        pos += c["dim"]
    lower = np.where(eq, b, -h.kHighsInf)
    hi = model(P, q, A, lower, b)
    status = hi.getModelStatus()
    if status == h.HighsModelStatus.kOptimal:
        sol = hi.getSolution()
        x, z = np.asarray(sol.col_value), -np.asarray(sol.row_dual)
        if P.nnz:
            x, z = polish(P, q, A, b, eq, x, z)
        s = b - A @ x
        s[eq] = 0.
        s[~eq] = np.maximum(s[~eq], 0.)
        return {"status": "solved", "x": x, "s": s, "z": z}
    if status in (h.HighsModelStatus.kInfeasible, h.HighsModelStatus.kUnboundedOrInfeasible):
        # Bounded alternative LP: a Farkas ray without relying on private ray APIs.
        AA = A.T.tocsc() / max(float(np.max(np.abs(A.data), initial=0)), 1e-300)
        qq = b / max(float(np.max(np.abs(b), initial=0)), 1e-300)
        alt = model(sparse.csc_matrix((m, m)), qq, AA, np.zeros(n), np.zeros(n),
                    np.where(eq, -1., 0.), np.ones(m))
        if alt.getModelStatus() == h.HighsModelStatus.kOptimal:
            zz = np.asarray(alt.getSolution().col_value)
            if b @ zz < 0:
                return {"status": "primal_infeasible", "z": zz}
    if status in (h.HighsModelStatus.kUnbounded, h.HighsModelStatus.kUnboundedOrInfeasible):
        AA = sparse.vstack((A, P), format="csc") if P.nnz else A
        lo = np.concatenate((np.where(eq, 0., -h.kHighsInf), np.zeros(n))) if P.nnz else np.where(eq, 0., -h.kHighsInf)
        qq = q / max(float(np.max(np.abs(q), initial=0)), 1e-300)
        alt = model(sparse.csc_matrix((n, n)), qq, AA, lo, np.zeros(AA.shape[0]),
                    -np.ones(n), np.ones(n))
        if alt.getModelStatus() == h.HighsModelStatus.kOptimal:
            xx = np.asarray(alt.getSolution().col_value)
            if q @ xx < 0:
                return {"status": "dual_infeasible", "x": xx, "s": -A @ xx}
    return {"status": "unresolved"}
