"""HiGHS backend, certificate construction, and active-set refinement."""

import time

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla
from scipy.optimize._highspy import _core as hc

from problem import blocks
from quality import acceptable, norm


def highs_model(P, q, A, lower, upper, col_lower=None, col_upper=None,
                seconds=270.0, solver="choose", presolve=True, crossover=True,
                tolerance=1e-10):
    n = q.size
    lp = hc.HighsLp()
    lp.num_col_, lp.num_row_ = n, upper.size
    lp.col_cost_ = q
    lp.col_lower_ = np.full(n, -hc.kHighsInf) if col_lower is None else col_lower
    lp.col_upper_ = np.full(n, hc.kHighsInf) if col_upper is None else col_upper
    lp.row_lower_, lp.row_upper_ = lower, upper
    A = sparse.csc_matrix(A)
    lp.a_matrix_.num_col_ = n
    lp.a_matrix_.num_row_ = upper.size
    lp.a_matrix_.format_ = hc.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32, copy=False)
    lp.a_matrix_.index_ = A.indices.astype(np.int32, copy=False)
    lp.a_matrix_.value_ = A.data
    model = hc.HighsModel()
    model.lp_ = lp
    if P is not None and P.nnz:
        L = sparse.tril(P, format="csc")
        h = hc.HighsHessian()
        h.dim_ = n
        h.format_ = hc.HessianFormat.kTriangular
        h.start_ = L.indptr.astype(np.int32, copy=False)
        h.index_ = L.indices.astype(np.int32, copy=False)
        h.value_ = L.data
        model.hessian_ = h
    engine = hc._Highs()
    for key, value in {
        "output_flag": False, "threads": 1, "parallel": "off",
        "primal_feasibility_tolerance": tolerance,
        "dual_feasibility_tolerance": tolerance,
        "ipm_optimality_tolerance": 1e-11,
        "small_matrix_value": 1e-12,
        "time_limit": max(0.01, seconds),
        "solver": solver, "presolve": "on" if presolve else "off",
        "run_crossover": "on" if crossover else "off",
        "qp_iteration_limit": 2000,
    }.items():
        engine.setOptionValue(key, value)
    engine.passModel(model)
    engine.run()
    return engine


def masks(p):
    eq = np.zeros(p.m, dtype=bool)
    for kind, dim, sl in blocks(p.cones):
        if kind == "zero":
            eq[sl] = True
    return eq


def certificate(p, status, deadline):
    eq = masks(p)
    if status == "primal_infeasible":
        if not norm(p.b):
            return None
        obj = p.b / norm(p.b)
        A = p.A.T.tocsc() / (norm(p.A.data) or 1.0)
        lo = np.where(eq, -1.0, 0.0)
        hi = np.ones(p.m)
        engine = highs_model(None, obj, A, np.zeros(p.n), np.zeros(p.n), lo, hi,
                             seconds=deadline - time.monotonic())
        sol = engine.getSolution()
        if sol.value_valid:
            result = {"status": status, "z": np.asarray(sol.col_value)}
            if acceptable(p, result):
                return result
    elif status == "dual_infeasible":
        if not norm(p.q):
            return None
        A = p.A / (norm(p.A.data) or 1.0)
        lo = np.where(eq, 0.0, -hc.kHighsInf)
        hi = np.zeros(p.m)
        if p.P.nnz:
            P = p.P / (norm(p.P.data) or 1.0)
            idx = np.flatnonzero(np.diff(P.tocsr().indptr))
            A = sparse.vstack((A, P[idx]), format="csc")
            lo = np.r_[lo, np.zeros(idx.size)]
            hi = np.r_[hi, np.zeros(idx.size)]
        engine = highs_model(None, p.q / norm(p.q), A, lo, hi,
                             -np.ones(p.n), np.ones(p.n),
                             seconds=deadline - time.monotonic())
        sol = engine.getSolution()
        if sol.value_valid:
            x = np.asarray(sol.col_value)
            result = {"status": status, "x": x, "s": -p.A @ x}
            if acceptable(p, result):
                return result
    return None


def polish(p, r, eq, max_iter=5):
    if p.n + p.m > 150000:
        return r
    x, z = r["x"].copy(), r["z"].copy()
    s = p.b - p.A @ x
    active = eq | ((s < 1e-6 * (1 + np.abs(p.b))) & (z > 1e-8))
    for unused in range(max_iter):
        indices = np.flatnonzero(active)
        B = p.A[indices].tocsc()
        K = sparse.bmat([[p.P, B.T], [B, None]], format="csc")
        rhs = np.r_[-p.q, p.b[indices]]
        try:
            factor = sla.splu(K)
            sol = factor.solve(rhs)
            for j in range(3):
                delta = rhs - K @ sol
                if norm(delta) < 1e-13 * max(1, norm(rhs)):
                    break
                sol += factor.solve(delta)
        except RuntimeError:
            try:
                diag = np.r_[np.full(p.n, 1e-10), np.full(indices.size, -1e-10)]
                factor = sla.splu(K + sparse.diags(diag, format="csc"))
                sol = np.r_[x, z[indices]]
                for j in range(8):
                    sol += factor.solve(rhs - K @ sol)
            except RuntimeError:
                return r
        x = sol[:p.n]
        z = np.zeros(p.m)
        z[indices] = sol[p.n:]
        s = p.b - p.A @ x
        candidate = {"status": "solved", "x": x, "s": s, "z": z}
        if acceptable(p, candidate):
            return candidate
        wrong = (z < -1e-8) & ~eq
        violated = (s < -1e-8 * (1 + np.abs(p.b))) & ~active
        if not np.any(wrong | violated):
            return r
        active[wrong] = False
        active[violated] = True
    return r


def solve(p, deadline):
    if p.P.nnz and p.n > 30000:
        from qpipm import solve as ipm_solve
        if p.n < 100000:
            return ipm_solve(p, deadline)
        from large_qp import solve as large_solve
        return large_solve(p, deadline)
    eq = masks(p)
    lower = np.where(eq, p.b, -hc.kHighsInf)
    if p.P.nnz and p.n > 4000 and p.P.nnz <= 10 * p.n:
        from qpipm import solve as ipm_solve
        result = ipm_solve(p, min(deadline, time.monotonic() + 45))
        if result["status"] != "unresolved":
            return result
    engine = highs_model(p.P, p.q, p.A, lower, p.b,
                         seconds=deadline - time.monotonic(),
                         solver="ipm" if not p.P.nnz and p.n > 8000 else "choose",
                         crossover=not (not p.P.nnz and p.n > 8000))
    status = engine.getModelStatus()
    sol = engine.getSolution()
    if __import__("os").environ.get("CONIC_DEBUG"):
        print("HiGHS", status, file=__import__("sys").stderr)
    if sol.value_valid and sol.dual_valid:
        x = np.asarray(sol.col_value)
        z = -np.asarray(sol.row_dual)
        r = {"status": "solved", "x": x, "s": p.b - p.A @ x, "z": z}
        r["s"][eq] = 0
        if acceptable(p, r):
            return r
        refined = polish(p, r, eq)
        if acceptable(p, refined):
            return refined
        if __import__("os").environ.get("CONIC_DEBUG"):
            from quality import residuals
            print("HiGHS residuals", residuals(p, r), file=__import__("sys").stderr)
    if status == hc.HighsModelStatus.kInfeasible:
        r = certificate(p, "primal_infeasible", deadline)
        if r is not None:
            return r
    if status in (hc.HighsModelStatus.kUnbounded, hc.HighsModelStatus.kUnboundedOrInfeasible):
        r = certificate(p, "dual_infeasible", deadline)
        if r is not None:
            return r
        r = certificate(p, "primal_infeasible", deadline)
        if r is not None:
            return r
    if time.monotonic() < deadline:
        if p.P.nnz:
            from qpipm import solve as ipm_solve
            warm = r if "r" in locals() and isinstance(r, dict) and r.get("status") == "solved" else None
            if warm is not None and not acceptable(p, warm) and p.n < 3000:
                feasibility = highs_model(None, np.zeros(p.n), p.A, lower, p.b,
                                           seconds=min(5, deadline - time.monotonic()))
                fs = feasibility.getSolution()
                if fs.value_valid and feasibility.getModelStatus() == hc.HighsModelStatus.kOptimal:
                    warm = {"x": np.asarray(fs.col_value)}
            result = ipm_solve(p, deadline, warm=warm)
            if result["status"] != "unresolved":
                return result
            if time.monotonic() < deadline:
                result = certificate(p, "primal_infeasible", deadline)
                if result is not None:
                    return result
        else:
            from scaling import equilibrate, recover
            scaled, D, E, cost = equilibrate(p)
            engine = highs_model(None, scaled.q, scaled.A,
                                 np.where(eq, scaled.b, -hc.kHighsInf), scaled.b,
                                 seconds=deadline - time.monotonic(), solver="ipm")
            sol = engine.getSolution()
            if sol.value_valid and sol.dual_valid:
                x = np.asarray(sol.col_value)
                r = recover({"status": "solved", "x": x, "s": scaled.b - scaled.A @ x,
                             "z": -np.asarray(sol.row_dual)}, D, E, cost)
                if acceptable(p, r):
                    return r
            if engine.getModelStatus() == hc.HighsModelStatus.kInfeasible:
                r = certificate(p, "primal_infeasible", deadline)
                if r is not None:
                    return r
    return {"status": "unresolved"}
