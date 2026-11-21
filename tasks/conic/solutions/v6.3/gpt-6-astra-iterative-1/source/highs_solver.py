import os
import numpy as np
from scipy import sparse
from problem import assess, norm, equilibrate
from native_highs import core as highs_core, lp as new_lp


def make_highs(P, q, A, lower, upper, col_lower=None, col_upper=None, options=None):
    m, n = A.shape
    solver = new_lp(q, A.indptr, A.indices, A.data, lower, upper,
                    col_lower, col_upper, options)
    if P is not None and P.nnz:
        L = sparse.tril(P, format="csc")
        solver.passHessian(n, L.nnz, 1, L.indptr, L.indices, L.data)
    return solver


def solve(problem, already_scaled=False):
    from box_solver import solve as separable
    box = separable(problem)
    if box is not None and assess(problem, box)[0]:
        return box
    P, q, A, b = problem.P, problem.q, problem.A, problem.b
    if not already_scaled:
        largest = max(norm(P.data), norm(A.data), norm(q), norm(b))
        smallest = min(np.min(np.abs(A.data), initial=np.inf),
                       np.min(np.abs(P.data), initial=np.inf))
        if largest > 1e7 or smallest < 1e-8:
            reduced, D, E, cost = equilibrate(problem, 8)
            scaled_result = solve(reduced, already_scaled=True)
            result = {"status": scaled_result["status"]}
            if "x" in scaled_result:
                result["x"] = D*scaled_result["x"]
            if "s" in scaled_result:
                result["s"] = scaled_result["s"]/E
            if "z" in scaled_result:
                result["z"] = E*scaled_result["z"]/cost
            return result
    lower = np.full(problem.m, -np.inf)
    equality = np.zeros(problem.m, dtype=bool)
    start = 0
    for cone in problem.cones:
        end = start + cone["dim"]
        if cone["type"] == "zero":
            lower[start:end] = b[start:end]
            equality[start:end] = True
        start = end
    use_ipm = (not P.nnz and problem.n >= 200
               and os.environ.get("CONIC_LP_METHOD","ipm") != "simplex")
    options = {"solver": "ipm", "run_crossover": "off"} if use_ipm else None
    solver = make_highs(P, q, A, lower, b, options=options)
    solver.run()
    status = str(solver.getModelStatus())
    if "Optimal" in status:
        sol = solver.getSolution()
        x = np.asarray(sol.col_value)
        z = -np.asarray(sol.row_dual)
        s = np.maximum(b-A@x, 0.)
        s[equality] = 0.
        result = {"status": "solved", "x": x, "s": s, "z": z}
        if assess(problem, result)[0]:
            return result
        if use_ipm:
            try:
                solver.crossover(sol)
                sol = solver.getSolution()
                x = np.asarray(sol.col_value)
                z = -np.asarray(sol.row_dual)
                s = np.maximum(b-A@x, 0.)
                s[equality] = 0.
                result = {"status": "solved", "x": x, "s": s, "z": z}
                if assess(problem, result)[0]:
                    return result
            except (ValueError, RuntimeError):
                pass
        result = polish(problem, result, equality)
        if assess(problem, result)[0]:
            return result
        return result
    if "Infeasible" in status or "Unbounded" in status:
        # The bundled Python bindings do not expose HiGHS' ray getters.
        # A bounded Farkas LP computes a direction in the original coordinates.
        if "Infeasible" in status:
            B = A.T.tocsc()
            bound = np.ones(problem.m)
            lb = np.where(equality, -1., 0.)
            cert = make_highs(None, b / (norm(b) or 1), B,
                              np.zeros(problem.n), np.zeros(problem.n), lb, bound)
            cert.run()
            if "Optimal" in str(cert.getModelStatus()):
                z = np.asarray(cert.getSolution().col_value)
                result = {"status": "primal_infeasible", "z": z}
                if assess(problem, result)[0]:
                    return result
        if "Unbounded" in status or "Infeasible" in status:
            B = sparse.vstack((A, P), format="csc") if P.nnz else A
            lo = np.where(equality, 0., -np.inf)
            if P.nnz:
                lo = np.r_[lo, np.zeros(problem.n)]
            cert = make_highs(None, q / (norm(q) or 1), B, lo,
                              np.zeros(B.shape[0]), -np.ones(problem.n),
                              np.ones(problem.n))
            cert.run()
            if "Optimal" in str(cert.getModelStatus()):
                x = np.asarray(cert.getSolution().col_value)
                s = np.maximum(-A@x, 0.)
                s[equality] = 0.
                result = {"status": "dual_infeasible", "x": x, "s": s}
                if assess(problem, result)[0]:
                    return result
    return {"status": "unresolved"}


def polish(problem, result, equality):
    from scipy.sparse.linalg import splu
    P, q, A, b = problem.P, problem.q, problem.A, problem.b
    x, z, s = result["x"], result["z"], result["s"]
    active = equality | ((s < 1e-6 * (1 + norm(s))) & (z > 1e-8))
    ids = np.flatnonzero(active)
    if len(ids) > 2 * problem.n + 100:
        return result
    B = A[ids, :].tocsc()
    reg = 1e-12
    K = sparse.bmat([[P + sparse.eye(problem.n, format="csc") * reg, B.T],
                     [B, -sparse.eye(len(ids), format="csc") * reg]], format="csc")
    rhs = np.r_[-q, b[ids]]
    try:
        lu = splu(K)
        u = np.r_[x, z[ids]]
        for _ in range(4):
            residual = rhs - np.r_[P @ u[:problem.n] + B.T @ u[problem.n:],
                                   B @ u[:problem.n]]
            u += lu.solve(residual)
        x = u[:problem.n]
        z = np.zeros(problem.m)
        z[ids] = u[problem.n:]
        s = np.maximum(b-A@x, 0.)
        s[equality] = 0.
        return {"status": "solved", "x": x, "s": s, "z": z}
    except (RuntimeError, ValueError):
        return result
