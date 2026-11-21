import os
import numpy as np
from native_ldl import library
from native_highs import lp


def norm(x):
    return float(np.max(np.abs(x), initial=0.))


def solve(q, b, ap, ai, av, eq, cones):
    lib = library()
    n, m = len(q), len(b)
    if not lib.conic_csc_sorted(n, ap, ai, av):
        return None
    amin = np.min(np.abs(av), initial=np.inf)
    amax = norm(av) or 1.
    if amin < 1e-8 or max(amax, norm(q), norm(b)) > 1e7:
        return None

    def multiply(x, trans=False):
        out = np.empty(n if trans else m)
        lib.conic_csc_matvec(m, n, ap, ai, av, x, out, int(trans))
        return out

    def dual_defect(z):
        minimum = np.min(z[~eq], initial=0.)
        if minimum >= -8e-9:
            return max(0., -minimum)
        worst, start = 0., 0
        for cone in cones:
            end = start+cone["dim"]
            if cone["type"] == "nonnegative":
                block = z[start:end]
                worst = max(worst, -np.min(block)/max(1., norm(block)))
            start = end
        return worst

    def solution(solver):
        sol = solver.getSolution()
        x, z = np.asarray(sol.col_value), -np.asarray(sol.row_dual)
        if len(x) != n or len(z) != m or not np.isfinite(x).all() or not np.isfinite(z).all():
            return None
        Ax, Atz = multiply(x), multiply(z, True)
        s = np.maximum(b-Ax, 0.)
        s[eq] = 0.
        rp = norm(Ax+s-b)/(1+max(norm(Ax), norm(s), norm(b)))
        rd = norm(q+Atz)/(1+max(norm(q), norm(Atz)))
        primal, dual = q.dot(x), -b.dot(z)
        gap = abs(primal-dual)/max(1., abs(primal), abs(dual))
        if np.isfinite([rp,rd,gap]).all() and max(rp,rd,gap,dual_defect(z)) <= 8e-9:
            return {"status":"solved", "x":x, "s":s, "z":z}
        return None

    lower = np.where(eq, b, -np.inf)
    use_ipm = n >= 200 and os.environ.get("CONIC_LP_METHOD","ipm") != "simplex"
    options = {"solver":"ipm","run_crossover":"off"} if use_ipm else None
    solver = lp(q, ap, ai, av, lower, b, options=options)
    solver.run()
    status = str(solver.getModelStatus())
    if "Optimal" in status:
        result = solution(solver)
        if result is None and use_ipm:
            solver.crossover(solver.getSolution())
            result = solution(solver)
        return result
    if "Infeasible" in status and norm(b):
        bp = np.empty(m+1, dtype=np.int32)
        bi = np.empty(len(ai), dtype=np.int32)
        bv = np.empty(len(av))
        work = np.empty(m, dtype=np.int32)
        lib.conic_csc_transpose(m, n, ap, ai, av, bp, bi, bv, work)
        cert = lp(b/norm(b), bp, bi, bv, np.zeros(n), np.zeros(n),
                  np.where(eq,-1.,0.), np.ones(m))
        cert.run()
        if "Optimal" in str(cert.getModelStatus()):
            z = np.asarray(cert.getSolution().col_value)
            if norm(z):
                z /= norm(z)
                margin = -(b/norm(b)).dot(z)
                if (margin > 1e-8 and norm(multiply(z, True))/amax <= 8e-9*margin
                        and dual_defect(z) <= 8e-9*margin):
                    return {"status":"primal_infeasible","z":z}
    if ("Unbounded" in status or "Infeasible" in status) and norm(q):
        cert = lp(q/norm(q), ap, ai, av, np.where(eq,0.,-np.inf), np.zeros(m),
                  -np.ones(n), np.ones(n))
        cert.run()
        if "Optimal" in str(cert.getModelStatus()):
            x = np.asarray(cert.getSolution().col_value)
            if norm(x):
                x /= norm(x)
                margin = -(q/norm(q)).dot(x)
                Ax = multiply(x)
                s = np.maximum(-Ax,0.)
                s[eq] = 0.
                if margin > 1e-8 and norm(Ax+s)/amax <= 8e-9*margin:
                    return {"status":"dual_infeasible","x":x,"s":s}
    return None
