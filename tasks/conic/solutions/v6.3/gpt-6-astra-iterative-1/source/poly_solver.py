import os
import numpy as np
from problem import assess, equilibrate


def solve(problem, validate=None):
    if validate is None:
        validate = lambda result, cones=True: assess(problem, result, cones=cones)
    from box_solver import solve as box_solve
    result = box_solve(problem)
    if result is not None and validate(result)[0]:
        return result
    if all(c["type"] == "zero" for c in problem.cones):
        result = equality_solve(problem)
        if validate(result)[0]:
            return result
    large_qp = bool(problem.P.nnz)
    if large_qp:
        method = os.environ.get("CONIC_QP_METHOD", "auto")
        dense_bounds = False
        if problem.n <= 4000 and problem.P.nnz > .05*problem.n**2:
            eq = np.zeros(problem.m,dtype=bool)
            start = 0
            for cone in problem.cones:
                end = start+cone["dim"]
                if cone["type"] == "zero":
                    eq[start:end] = True
                start = end
            rows = np.bincount(problem.A.indices,minlength=problem.m)
            dense_bounds = (np.count_nonzero(eq) <= max(16,problem.n//8)
                            and np.max(rows[~eq],initial=0) <= 1)
        use_pd = method == "pd" or (method == "auto" and dense_bounds)
        if use_pd:
            from qp_pd import solve as interior
        else:
            from ipm import solve as interior
        if problem.n < 50000:
            from presolve import prepare
            prep = prepare(problem,validate)
            if prep.result is not None:
                return prep.result
            check = lambda result, cones=True: validate(prep.recover(result), cones=cones)
            feasible = zero_cost_feasible(prep.reduced,check)
            if feasible is not None:
                return prep.recover(feasible)
            result = prep.recover(interior(prep.reduced, validate=check))
        else:
            result = interior(problem, validate=validate)
        if validate(result)[0]:
            return result
        if use_pd:
            from ipm import solve as homogeneous
            other = homogeneous(problem, validate=validate)
            if validate(other)[0]:
                return other
        from highs_solver import solve as native
        other = native(problem)
        return other if validate(other)[0] else result
    from highs_solver import solve as native
    result = native(problem)
    if validate(result)[0]:
        return result
    from ipm import solve as interior
    other = interior(problem, validate=validate)
    return other if validate(other)[0] else result


def zero_cost_feasible(problem,validate):
    if problem.q.any() or problem.n > 20000:
        return None
    diagonal = problem.P.diagonal()
    if problem.P.nnz != np.count_nonzero(diagonal):
        return None
    free = np.flatnonzero(diagonal == 0)
    if not len(free):
        return None
    eq = np.zeros(problem.m,dtype=bool)
    start = 0
    for cone in problem.cones:
        end = start+cone["dim"]
        if cone["type"] == "zero":
            eq[start:end] = True
        start = end
    from highs_solver import make_highs
    B = problem.A[:,free].tocsc()
    h = make_highs(None,np.zeros(len(free)),B,np.where(eq,problem.b,-np.inf),
                   problem.b,options={"time_limit":.05,"solver":"simplex"})
    h.run()
    values = np.asarray(h.getSolution().col_value)
    if len(values) == len(free) and np.isfinite(values).all():
        x = np.zeros(problem.n)
        x[free] = values
        s = np.maximum(problem.b-problem.A@x,0.)
        s[eq] = 0.
        result = {"status":"solved","x":x,"s":s,"z":np.zeros(problem.m)}
        if validate(result)[0]:
            return result
    return None


def equality_solve(problem):
    from cones import ConeProduct
    from ipm import KKT
    try:
        p, D, E, cost = equilibrate(problem)
        cone = ConeProduct(p.cones, p.m)
        s, z = cone.initial()
        kkt = KKT(p.P, p.A, cone.scaling(s, z, 1.))
        x, z = kkt.solve(-p.q, p.b)
        x, z = D*x, E*z/cost
        result = {"status": "solved", "x": x, "s": np.zeros(problem.m), "z": z}
        if assess(problem, result)[0]:
            return result
        pi = {"status": "primal_infeasible", "z": z}
        if assess(problem, pi)[0]:
            return pi
        di = {"status": "dual_infeasible", "x": x, "s": -problem.A@x}
        if assess(problem, di)[0]:
            return di
    except (ValueError, ArithmeticError, RuntimeError):
        pass
    return {"status": "unresolved"}
