import numpy as np
from scipy import sparse

from model import assess, norm, repair_slack


def bounds(problem):
    lower = np.full(problem.m, -np.inf)
    start = 0
    for cone in problem.cones:
        dim = cone["dim"]
        if cone["type"] == "zero":
            lower[start:start + dim] = problem.b[start:start + dim]
        start += dim
    return lower, problem.b.copy()


def highs(P, q, A, lower, upper, col_lower=None, col_upper=None,
          solver="choose", time_limit=250.0):
    h = highs_core()
    n, m = A.shape[1], A.shape[0]
    model = h.HighsModel()
    lp = model.lp_
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = q
    lp.col_lower_ = np.full(n, -np.inf) if col_lower is None else col_lower
    lp.col_upper_ = np.full(n, np.inf) if col_upper is None else col_upper
    lp.row_lower_, lp.row_upper_ = lower, upper
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, m
    lp.a_matrix_.format_ = h.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    if P is not None and P.nnz:
        triangle = sparse.tril(P, format="csc")
        H = model.hessian_
        H.dim_ = n
        H.format_ = h.HessianFormat.kTriangular
        H.start_, H.index_, H.value_ = (
            triangle.indptr, triangle.indices, triangle.data)
    engine = h._Highs()
    for key, value in (
        ("output_flag", False), ("threads", 1), ("parallel", "off"),
        ("primal_feasibility_tolerance", 1e-9),
        ("dual_feasibility_tolerance", 1e-9),
        ("ipm_optimality_tolerance", 1e-11),
        ("small_matrix_value", 1e-12),
        ("time_limit", time_limit), ("solver", solver),
    ):
        engine.setOptionValue(key, value)
    if solver == "ipm":
        engine.setOptionValue("run_crossover", "off")
    engine.passModel(model)
    engine.run()
    solution = engine.getSolution()
    return (str(engine.getModelStatus()).split(".")[-1],
            np.asarray(solution.col_value), -np.asarray(solution.row_dual),
            engine)


def polish(problem, x, z, max_iter=8):
    from scipy import linalg
    from scipy.sparse.linalg import splu
    P, q, A, b = problem.P, problem.q, problem.A, problem.b
    equality = np.zeros(problem.m, dtype=bool)
    start = 0
    for cone in problem.cones:
        width = cone["dim"]
        if cone["type"] == "zero":
            equality[start:start + width] = True
        start += width
    s = b - A @ x
    active = equality | (s < 1e-7 * (1 + norm(b))) | (z > 1e-8)
    z = z.copy()
    z[~active] = 0
    best = dict(status="solved", x=x, s=repair_slack(problem, x, s), z=z)
    for unused in range(max_iter):
        ids = np.flatnonzero(active)
        B = A[ids, :].tocsc()
        K = sparse.bmat([[P, B.T], [B, None]], format="csc")
        rhs = np.r_[-q, b[ids]]
        try:
            if K.shape[0] <= 250:
                sol = linalg.lstsq(K.toarray(), rhs, cond=1e-13,
                                  lapack_driver="gelsy", check_finite=False)[0]
            else:
                reg = np.r_[np.full(problem.n, 1e-11),
                            np.full(ids.size, -1e-11)]
                lu = splu(K + sparse.diags(reg, format="csc"))
                sol = np.r_[x, z[ids]]
                for refinement in range(5):
                    residual = rhs - K @ sol
                    if norm(residual) < 1e-12 * (1 + norm(rhs)):
                        break
                    sol += lu.solve(residual)
            xn = sol[:problem.n]
            zn = np.zeros(problem.m)
            zn[ids] = sol[problem.n:]
            sn = b - A @ xn
            candidate = dict(status="solved", x=xn, s=repair_slack(problem, xn, sn), z=zn)
            good, report = assess(problem, candidate)
            best = candidate
            if good:
                return candidate
            negative = (~equality) & active & (zn < -1e-9)
            violated = (~equality) & (~active) & (sn < -1e-9)
            if not np.any(negative | violated):
                return candidate
            if np.any(negative):
                remove = np.flatnonzero(negative)
                active[remove[np.argmin(zn[remove])]] = False
            active[violated] = True
            x, z = xn, zn
        except (np.linalg.LinAlgError, RuntimeError, ValueError):
            break
    return best


_highs_module = None


def highs_core():
    global _highs_module
    if _highs_module is None:
        import importlib.util
        import os
        import sys
        import scipy
        folder = os.path.join(os.path.dirname(scipy.__file__), "optimize", "_highspy")
        binary = next(name for name in os.listdir(folder)
                      if name.startswith("_core.") and name.endswith(".so"))
        spec = importlib.util.spec_from_file_location("_core", os.path.join(folder, binary))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["scipy.optimize._highspy._core"] = module
        _highs_module = module
    return _highs_module


def certificate(problem, status):
    P, q, A, b = problem.P, problem.q, problem.A, problem.b
    if status in ("kInfeasible", "kUnboundedOrInfeasible"):
        equal = np.zeros(problem.m, dtype=bool)
        start = 0
        for cone in problem.cones:
            if cone["type"] == "zero":
                equal[start:start + cone["dim"]] = True
            start += cone["dim"]
        lb = np.where(equal, -1.0, 0.0)
        bn, an = norm(b), norm(A.data)
        if bn:
            st, z, unused, engine = highs(
                None, b / bn, (A.T / (an or 1.0)).tocsc(),
                np.zeros(problem.n), np.zeros(problem.n),
                lb, np.ones(problem.m), time_limit=15.0)
            result = dict(status="primal_infeasible", z=z)
            if st == "kOptimal" and assess(problem, result)[0]:
                return result
    if status in ("kUnbounded", "kUnboundedOrInfeasible") and norm(q):
        lo, hi = bounds(problem)
        hi[:] = 0.0
        lo[np.isfinite(lo)] = 0.0
        an, pn = norm(A.data) or 1, norm(P.data) or 1
        direction_A = A / an
        if P.nnz:
            direction_A = sparse.vstack((direction_A, P / pn), format="csc")
            lo = np.r_[lo, np.zeros(problem.n)]
            hi = np.r_[hi, np.zeros(problem.n)]
        st, x, unused, engine = highs(
            None, q / norm(q), direction_A, lo, hi,
            -np.ones(problem.n), np.ones(problem.n), time_limit=15.0)
        result = dict(status="dual_infeasible", x=x, s=-A @ x)
        if st == "kOptimal" and assess(problem, result)[0]:
            return result
    return None


def solve(problem):
    lower, upper = bounds(problem)
    strategy = "ipm" if not problem.P.nnz and problem.n >= 500 else "choose"
    status, x, z, engine = highs(
        problem.P, problem.q, problem.A, lower, upper, solver=strategy)
    if status in ("kOptimal", "kModelEmpty") and x.size == problem.n:
        result = dict(status="solved", x=x, s=repair_slack(problem, x), z=z)
        if assess(problem, result)[0]:
            return result
        result = polish(problem, x, z)
        if assess(problem, result)[0]:
            return result
    cert = certificate(problem, status)
    return cert
