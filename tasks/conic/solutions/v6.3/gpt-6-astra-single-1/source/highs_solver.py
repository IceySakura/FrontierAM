import numpy as np
from scipy import sparse
from scipy.optimize._highspy import _core as hs
from common import blocks, norm, assess


def run_highs(P, q, A, lo, hi, col_lo=None, col_hi=None, presolve=True):
    n, m = len(q), len(hi)
    lp = hs.HighsLp()
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = q
    lp.col_lower_ = np.full(n, -np.inf) if col_lo is None else col_lo
    lp.col_upper_ = np.full(n, np.inf) if col_hi is None else col_hi
    lp.row_lower_, lp.row_upper_ = lo, hi
    lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, m
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    h = hs._Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("parallel", "off")
    h.setOptionValue("primal_feasibility_tolerance", 1e-9)
    h.setOptionValue("dual_feasibility_tolerance", 1e-9)
    h.setOptionValue("ipm_optimality_tolerance", 1e-11)
    h.setOptionValue("small_matrix_value", 1e-12)
    h.setOptionValue("large_matrix_value", 1e30)
    if not presolve:
        h.setOptionValue("presolve", "off")
    h.passModel(lp)
    if P is not None and P.nnz:
        T = sparse.tril(P, format="csc")
        H = hs.HighsHessian()
        H.dim_ = n
        H.format_ = hs.HessianFormat.kTriangular
        H.start_, H.index_, H.value_ = T.indptr, T.indices, T.data
        h.passHessian(H)
        # These options are available in the bundled HiGHS even when absent
        # from the Python HighsOptions attribute list.
        h.setOptionValue("qp_regularization_value", 1e-12)
        h.setOptionValue("qp_nullspace_limit", 1000000)
        h.setOptionValue("qp_iteration_limit", 100000)
    h.run()
    return h


def polish(P, q, A, b, cones, x, z):
    from scipy.sparse.linalg import splu, lsmr
    slack = b - A @ x
    eq = np.zeros(len(b), dtype=bool)
    for kind, d, i, j in blocks(cones):
        if kind == "zero":
            eq[i:j] = True
    active = eq | ((slack < 1e-7 * (1 + np.abs(b))) & (z > 1e-8))
    B = A[active].tocsc()
    K = sparse.bmat([[P, B.T], [B, None]], format="csc")
    rhs = np.concatenate((-q, b[active]))
    try:
        lu = splu(K)
        sol = lu.solve(rhs)
        for _ in range(2):
            residual = rhs - K @ sol
            sol += lu.solve(residual)
    except RuntimeError:
        sol = lsmr(K, rhs, atol=1e-13, btol=1e-13,
                   maxiter=min(5000, 3 * K.shape[0]))[0]
    n = len(q)
    zz = np.zeros(len(b))
    zz[active] = sol[n:]
    return {"status": "solved", "x": sol[:n], "s": b - A @ sol[:n], "z": zz}


def solve(P, q, A, b, cones):
    m, n = A.shape
    lo = np.full(m, -np.inf)
    eq = np.zeros(m, dtype=bool)
    for kind, d, i, j in blocks(cones):
        if kind == "zero":
            lo[i:j] = b[i:j]
            eq[i:j] = True
    h = run_highs(P, q, A, lo, b)
    status = h.getModelStatus()
    sol = h.getSolution()
    candidate = {"status": "unresolved"}
    failed_optimal = False
    if status == hs.HighsModelStatus.kOptimal:
        x, z = np.array(sol.col_value), -np.array(sol.row_dual)
        slack = b - A @ x
        slack[eq] = 0.0
        slack[~eq] = np.maximum(slack[~eq], 0.0)
        z[~eq] = np.maximum(z[~eq], 0.0)
        result = {"status": "solved", "x": x, "s": slack, "z": z}
        if assess(P, q, A, b, cones, result)[0]:
            return result
        if P.nnz or n < 20000:
            improved = polish(P, q, A, b, cones, x, z)
            if assess(P, q, A, b, cones, improved)[0]:
                return improved
        candidate = result
        failed_optimal = True
    if failed_optimal or status in (hs.HighsModelStatus.kInfeasible,
                  hs.HighsModelStatus.kUnboundedOrInfeasible):
        C = A.T.tocsc()
        hz = run_highs(None, b / max(1.0, norm(b)), C, np.zeros(n),
                       np.zeros(n), np.where(eq, -1.0, 0.0), np.ones(m))
        if hz.getModelStatus() == hs.HighsModelStatus.kOptimal:
            z = np.array(hz.getSolution().col_value)
            result = {"status": "primal_infeasible", "z": z}
            if assess(P, q, A, b, cones, result)[0]:
                return result
    if failed_optimal or status in (hs.HighsModelStatus.kUnbounded,
                  hs.HighsModelStatus.kUnboundedOrInfeasible,
                  hs.HighsModelStatus.kInfeasible):
        C = sparse.vstack([A, P], format="csc") if P.nnz else A
        clo = np.concatenate([np.where(eq, 0.0, -np.inf), np.zeros(n)]) if P.nnz else np.where(eq, 0.0, -np.inf)
        ch = np.zeros(C.shape[0])
        hx = run_highs(None, q / max(1.0, norm(q)), C, clo, ch,
                       -np.ones(n), np.ones(n))
        if hx.getModelStatus() == hs.HighsModelStatus.kOptimal:
            x = np.array(hx.getSolution().col_value)
            result = {"status": "dual_infeasible", "x": x, "s": -A @ x}
            if assess(P, q, A, b, cones, result)[0]:
                return result
    from ipm import solve as interior_solve
    alternative = interior_solve(P, q, A, b, cones)
    if assess(P, q, A, b, cones, alternative, tol=1e-8)[0]:
        return alternative
    return candidate
