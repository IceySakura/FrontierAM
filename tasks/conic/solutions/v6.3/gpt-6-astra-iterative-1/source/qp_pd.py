import time
import os
import numpy as np
from problem import assess, norm, equilibrate
from cones import ConeProduct
from ipm import KKT


def solve(problem, validate=None, max_iter=100, max_time=260.):
    if validate is None:
        validate = lambda result, cones=True: assess(problem, result, cones=cones)
    started = time.monotonic()
    p, D, E, cost = equilibrate(problem)
    cone = ConeProduct(p.cones, p.m)
    if cone.soc or cone.psd or len(cone.exp) or not len(cone.nn):
        return {"status":"unresolved"}
    P, q, A, b = p.P, p.q, p.A, p.b
    x = np.zeros(p.n)
    s = np.zeros(p.m)
    z = np.zeros(p.m)
    s[cone.nn] = np.maximum(1., b[cone.nn])
    z[cone.nn] = 1/s[cone.nn]
    count = len(cone.nn)
    cache = {}
    best, besterr = {"status":"unresolved"}, np.inf
    polished_at = np.inf
    equality = np.zeros(p.m, dtype=bool)
    equality[cone.zero] = True
    verbose = bool(os.environ.get("CONIC_VERBOSE"))

    def maxstep(v, dv):
        negative = dv < 0
        return min(1., np.min(-v[negative]/dv[negative], initial=1.))

    for iteration in range(max_iter):
        Px, Ax, Atz = P@x, A@x, A.T@z
        rd, rp = Px+q+Atz, Ax+s-b
        mu = s[cone.nn].dot(z[cone.nn])/count
        result = {"status":"solved","x":D*x,"s":s/E,"z":E*z/cost}
        passed, errors = validate(result, cones=False)
        error = max(errors)
        if error < besterr:
            best, besterr = result, error
        if verbose:
            print("qp",iteration,"err",error,"mu",mu,
                  "time",time.monotonic()-started,flush=True)
        if passed and validate(result)[0]:
            return result
        if error < min(1e-5, polished_at*.1):
            from highs_solver import polish
            polished = polish(problem, result, equality)
            polished_at = error
            if validate(polished)[0]:
                return polished
        if iteration > 4 and (norm(z) > 1e4 or norm(x) > 1e4):
            pi = {"status":"primal_infeasible","z":E*z/cost}
            if validate(pi)[0]:
                return pi
            di = {"status":"dual_infeasible","x":D*x,"s":s/E}
            if validate(di)[0]:
                return di
        if time.monotonic()-started > max_time or not np.isfinite(mu) or mu <= 0:
            break
        try:
            scaling = cone.scaling(s,z,mu)
            kkt = KKT(P,A,scaling,cache=cache)
            dxa, dza = kkt.solve(-rd, b-Ax)
            dsa = -rp-A@dxa
            aa = maxstep(s[cone.nn],dsa[cone.nn])
            ad = maxstep(z[cone.nn],dza[cone.nn])
            mua = (s[cone.nn]+aa*dsa[cone.nn]).dot(z[cone.nn]+ad*dza[cone.nn])/count
            sigma = np.clip(mua/mu,0.,1.)**3
            rc = np.zeros(p.m)
            ids = cone.nn
            rc[ids] = -s[ids]+(sigma*mu-dsa[ids]*dza[ids])/z[ids]
            dx, dz = kkt.solve(-rd,-rp-rc)
            ds = -rp-A@dx
            alpha = .995*min(maxstep(s[ids],ds[ids]),maxstep(z[ids],dz[ids]))
            if not np.isfinite(alpha) or alpha < 1e-12:
                break
            x += alpha*dx
            s += alpha*ds
            z += alpha*dz
            s[cone.zero] = 0.
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            break
    return best
