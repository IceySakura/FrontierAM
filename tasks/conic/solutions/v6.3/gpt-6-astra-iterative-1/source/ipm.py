import os
import time
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import splu
from problem import Problem, norm, assess, equilibrate
from cones import ConeProduct, exp_centrality, exp_hsolve, EXPMODE

VERBOSE = bool(os.environ.get("CONIC_VERBOSE"))
CENTER = float(os.environ.get("CONIC_CENTER", ".5"))
NEIGHBORHOOD = float(os.environ.get("CONIC_NEIGHBORHOOD", "1.5"))
ORDER_CACHE = os.environ.get("CONIC_ORDER_CACHE", "1") != "0"
NATIVE_LDL = os.environ.get("CONIC_LDL", "1") == "1"
SOC_LDL = os.environ.get("CONIC_SOC_LDL", "0") == "1"
SYMMETRIC_INIT = os.environ.get("CONIC_INIT", "0") == "1"
WIDE_EXP = os.environ.get("CONIC_WIDE_EXP", "0") == "1"


class KKT:
    def __init__(self, P, A, scaling, force_full=False, cache=None):
        self.P, self.A, self.scaling = P, A, scaling
        self.cache = cache
        self.poly = None
        self.box = None
        n, m = A.shape[1], A.shape[0]
        self.n, self.m = n, m
        largest = max((p.width for p, *_ in scaling.psd), default=0)
        dense_cone_size = sum(p.width*p.width for p, *_ in scaling.psd)
        self.condensed = bool(largest > 100 and n*n <= 4*dense_cone_size
                              and (not force_full or largest > 2000))
        self.reg = 1e-12
        if self.condensed:
            self._condense()
            return
        cone = scaling.cone
        if (cache is not None and not cache.get("skip_box",False)
                and not cone.soc and not cone.psd and not len(cone.exp)
                and len(cone.nn) and n <= 4000 and P.nnz > .05*n*n
                and len(cone.zero) <= max(16,n//8)):
            try:
                from box_kkt import BoxKKT
                box = cache.get("box")
                if box is None:
                    box = BoxKKT(P,A,cone)
                    cache["box"] = box
                box.factor(scaling)
                self.box = box
                return
            except (MemoryError, ArithmeticError, ValueError, linalg.LinAlgError):
                cache["skip_box"] = True
                cache.pop("box",None)
        if (NATIVE_LDL and cache is not None and not cache.get("skip_poly", False)
                and not cone.soc and not cone.psd and not len(cone.exp)
                and n+m >= 500 and P.nnz+2*A.nnz < 20*(n+m)):
            try:
                from poly_kkt import PolyKKT
                poly = cache.get("poly")
                if poly is None:
                    poly = PolyKKT(P, A, cone)
                    cache["poly"] = poly
                poly.factor(scaling)
                self.poly = poly
                return
            except (OSError, MemoryError, ArithmeticError, ValueError):
                cache["skip_poly"] = True
                cache.pop("poly", None)
        native_ok = (NATIVE_LDL and (SOC_LDL or
                     (all(ids.shape[1] <= 8 for ids in scaling.cone.soc)
                      and all(np.max(w[:, 0]) < 50 for _, _, w, _ in scaling.soc))))
        W = scaling.sparse_w(dense_soc=native_ok and not SOC_LDL)
        ne = W.shape[0] - m
        AA = sparse.vstack([A, sparse.csc_matrix((ne, n))], format="csc") if ne else A
        K = sparse.bmat([[P, AA.T], [AA, W]], format="csc")
        self.K = K
        self.extra = ne
        self.order = None
        self.native = None
        self.cache = cache
        key = (K.shape, K.nnz)
        if ORDER_CACHE and cache is not None and cache.get("key") == key:
            self.order = cache["order"]
        diag = np.r_[np.ones(n), -np.ones(m), np.ones(ne)]
        for retry in range(4):
            try:
                regularized = K.copy()
                regularized.setdiag(K.diagonal()+self.reg*diag)
                if self.order is not None:
                    regularized = regularized[self.order, :][:, self.order].tocsc()
                self.regularized = regularized
                if (native_ok and self.order is not None and cache is not None
                        and not cache.get("skip_native", False)):
                    try:
                        from native_ldl import LDL
                        upper = sparse.triu(regularized, format="csc")
                        upper.sort_indices()
                        factor = cache.get("native")
                        if factor is None or not factor.matches(upper):
                            signs = (np.zeros(len(diag)) if SOC_LDL and scaling.soc
                                     else diag[self.order])
                            factor = LDL(upper, signs)
                            cache["native"] = factor
                        factor.factor(upper)
                        self.native = factor
                        return
                    except (OSError, MemoryError, ArithmeticError, ValueError):
                        cache["skip_native"] = True
                self.lu = splu(regularized,
                               permc_spec="NATURAL" if self.order is not None else "MMD_AT_PLUS_A",
                               diag_pivot_thresh=.1,
                               options={"Equil": False})
                break
            except RuntimeError:
                self.reg *= 100
        else:
            raise ArithmeticError("KKT factorization failed")
        if ORDER_CACHE and cache is not None and self.order is None:
            cache["key"] = key
            cache["order"] = np.argsort(self.lu.perm_c)

    def linear_solve(self, rhs):
        solver = self.native if self.native is not None else self.lu
        if self.order is None:
            return solver.solve(rhs)
        result = np.empty_like(rhs)
        result[self.order] = (solver.solve(rhs[self.order], overwrite=True)
                              if self.native is not None else solver.solve(rhs[self.order]))
        return result

    def _condense(self):
        sc = self.scaling
        c = sc.cone
        n, m = self.n, self.m
        self.eq = c.zero
        Ae = self.A[self.eq, :].tocsc()
        self.Ae = Ae
        if n > 6000:
            raise ArithmeticError("Condensed PSD system too large")
        H = self.P.toarray()
        if len(c.nn):
            block = self.A[c.nn, :].tocsc()
            H += (block.T@block.multiply((1/sc.diag)[:, None])).toarray()
        for ids, beta, w, lam in sc.soc:
            jw = w.copy()
            jw[:, 1:] *= -1
            count, dim = ids.shape
            block = self.A[ids.ravel(), :].tocsc()
            weights = np.ones_like(w)/beta[:, None]
            weights[:, 0] *= -1
            H += (block.T@block.multiply(weights.ravel()[:, None])).toarray()
            reducer = sparse.csc_matrix(
                ((jw/np.sqrt(beta)[:, None]).ravel(),
                 (np.repeat(np.arange(count), dim), np.arange(ids.size))),
                shape=(count, ids.size))
            U = reducer@block
            H += 2*(U.T@U).toarray()
        if len(c.exp):
            count = len(c.exp)
            inv = exp_hsolve(sc.z[c.exp], np.broadcast_to(np.eye(3), (count,3,3)))/sc.mu
            ids = np.arange(count*3).reshape(count,3)
            C = sparse.csc_matrix((inv.ravel(),
                (np.repeat(ids,3,axis=1).ravel(), np.tile(ids,(1,3)).ravel())),
                shape=(3*count,3*count))
            block = self.A[c.exp.ravel(), :].tocsc()
            H += (block.T@C@block).toarray()
        for p, R, Ri, lam, D, Di in sc.psd:
            block = self.A[p.sl, :].tocsc()
            active = np.unique(block.indices)
            if len(active) <= 3000 and len(active)**2 <= max(n*p.dim**2*4, 10000):
                # Sparse coefficient matrices need only the active svec rows.
                r, t = p.r[active], p.c[active]
                local = Di[r[:,None], r]*Di[t[:,None], t]
                local += Di[r[:,None], t]*Di[t[:,None], r]
                weight = np.where(r == t, 1/np.sqrt(2.), 1.)
                local *= weight[:,None]
                local *= weight[None,:]
                B = block[active, :].tocsc()
                projected = B.T@local
                H += (B.T@projected.T).T
            else:
                for first in range(0, n, 32):
                    last = min(n, first+32)
                    mats = p.batch_mat(block[:, first:last].toarray())
                    columns = p.batch_vec(Di@mats@Di)
                    H[:, first:last] += block.T@columns
        H = .5 * (H + H.T)
        self.H = H
        if len(self.eq):
            ae = Ae.toarray()
            M = np.block([[H, ae.T], [ae, np.zeros((len(self.eq), len(self.eq)))]])
        else:
            M = H
        self.M = M
        reg = max(1e-12, norm(np.diag(H)) * 1e-14)
        diag = np.r_[np.ones(n) * reg, -np.ones(len(self.eq)) * 1e-12]
        self.lu = linalg.lu_factor(M + np.diag(diag), check_finite=False)

    def solve(self, rx, rz):
        if self.box is not None:
            try:
                return self.box.solve(rx,rz)
            except ArithmeticError:
                cache = self.cache
                cache["skip_box"] = True
                cache.pop("box",None)
                self.__init__(self.P,self.A,self.scaling,cache=cache)
                return self.solve(rx,rz)
        if self.poly is not None:
            try:
                return self.poly.solve(rx, rz)
            except ArithmeticError:
                cache = self.cache
                cache["skip_poly"] = True
                cache.pop("poly", None)
                self.__init__(self.P, self.A, self.scaling, cache=cache)
                return self.solve(rx, rz)
        if self.condensed:
            # A dx - W dz = rz, so dz = H (A dx - rz).
            hr = self.scaling.hmul(rz)
            rhs = np.r_[rx + self.A.T @ hr, rz[self.eq]]
            u = linalg.lu_solve(self.lu, rhs, check_finite=False)
            for _ in range(3):
                err = rhs - self.M @ u
                if norm(err) <= 1e-11 * (1 + norm(rhs)):
                    break
                u += linalg.lu_solve(self.lu, err, check_finite=False)
            x = u[:self.n]
            z = self.scaling.hmul(self.A @ x - rz)
            z[self.eq] = u[self.n:]
            return x, z
        rhs = np.r_[rx, rz, np.zeros(self.extra)]
        u = self.linear_solve(rhs)
        olderr = np.inf
        for _ in range(4):
            residual = rhs - self.K @ u
            err = norm(residual)
            if err < 2e-12 * (1 + norm(rhs)) or err >= olderr * .95:
                break
            olderr = err
            u += self.linear_solve(residual)
        final_residual = rhs-self.K@u if self.native is not None else None
        if self.native is not None and (
                not np.isfinite(u).all() or
                norm(final_residual) > 2e-10*(1+norm(rhs)) or
                norm(final_residual[:self.n]) > 2e-11*(1+norm(rx))):
            self.native = None
            self.cache["skip_native"] = True
            self.lu = splu(self.regularized, permc_spec="NATURAL",
                          diag_pivot_thresh=.1, options={"Equil": False})
            return self.solve(rx, rz)
        return u[:self.n], u[self.n:self.n + self.m]


def solve(problem, max_iter=250, scaling=True, max_time=280.0, validate=None):
    started = time.monotonic()
    if validate is None:
        validate = lambda result, cones=True: assess(problem, result, cones=cones)
    if scaling:
        prob, D, E, cost = equilibrate(problem)
    else:
        prob, D, E, cost = problem, np.ones(problem.n), np.ones(problem.m), 1.
    P, q, A, b = prob.P, prob.q, prob.A, prob.b
    cones = ConeProduct(prob.cones, prob.m)
    x = np.zeros(prob.n)
    s, z = cones.initial()
    kkt_cache = {}
    if SYMMETRIC_INIT and not len(cones.exp):
        try:
            initial = KKT(P, A, cones.scaling(s, z, 1.), cache=kkt_cache)
            xi, zi = initial.solve(-q, b)
            if max(norm(xi), norm(zi)) < 1e8:
                x = xi
                s = cones.interior_shift(-zi)
                z = cones.interior_shift(zi, dual=True)
        except (ArithmeticError, RuntimeError, ValueError, linalg.LinAlgError):
            pass
    tau, kappa = 1., 1.
    best, besterr = {"status": "unresolved"}, np.inf
    last_projection = np.inf
    force_full = False
    for iteration in range(max_iter):
        Px, Ax, Atz = P @ x, A @ x, A.T @ z
        rd = Px + Atz + q * tau
        rp = Ax + s - b * tau
        xPx = x.dot(Px)
        rg = q.dot(x) + b.dot(z) + xPx / tau + kappa
        mu = (s.dot(z) + tau * kappa) / (cones.degree + 1)
        result = {"status": "solved", "x": D * x / tau,
                  "s": s / E / tau, "z": E * z / (cost * tau)}
        ok, errors = validate(result, cones=False)
        err = max(errors)
        if err < besterr:
            best, besterr = result, err
        if VERBOSE:
            print(iteration, "res", "%.2e" % err, "mu", "%.2e" % mu,
                  "tau", "%.2e" % tau, "k", "%.2e" % kappa,
                  "time", "%.3f" % (time.monotonic() - started), flush=True)
        if ok:
            valid, _ = validate(result)
            if valid:
                return result
        if cones.psd and err < min(5e-7, last_projection*.4):
            from refine import project_solution
            polished = project_solution(problem, result)
            last_projection = err
            if validate(polished)[0]:
                return polished
        if tau < 1e-5 * max(1, kappa) or (iteration > 5 and err > 1):
            pi = {"status": "primal_infeasible", "z": E * z / cost}
            if validate(pi)[0]:
                return pi
            di = {"status": "dual_infeasible", "x": D * x, "s": s / E}
            if validate(di)[0]:
                return di
        if time.monotonic() - started > max_time:
            break
        try:
            sc = cones.scaling(s, z, mu)
            kkt = KKT(P, A, sc, force_full, kkt_cache)
            hx, hz = kkt.solve(q, -b)
            v = q + 2 * Px / tau
            denom = -xPx / tau**2 - kappa / tau - v.dot(hx) - b.dot(hz)

            def direction(rc, rt, gapcorr=0, residual_scale=1.):
                ux, uz = kkt.solve(-rd * residual_scale, -rp * residual_scale - rc)
                dt = (-rg * residual_scale - gapcorr - rt / tau
                      - v.dot(ux) - b.dot(uz)) / denom
                dx, dz = ux - hx * dt, uz - hz * dt
                ds = -rp * residual_scale - A @ dx + b * dt
                dk = (rt - kappa * dt) / tau
                return dx, ds, dz, dt, dk

            def maxstep(ds, dz, dt, dk):
                alpha = min(cones.step(s, ds), cones.step(z, dz, True))
                if dt < 0:
                    alpha = min(alpha, -tau / dt)
                if dk < 0:
                    alpha = min(alpha, -kappa / dk)
                return alpha

            if len(cones.exp) and EXPMODE in ("path", "pc"):
                wide = WIDE_EXP and not cones.psd
                if wide:
                    total_centrality,centrality = cones.centrality(
                        s,z,mu,tau,kappa,individual=True)
                else:
                    centrality = cones.centrality(s, z, mu, tau, kappa)
                centering = centrality > CENTER
                if centering:
                    dx, ds, dz, dt, dk = direction(sc.rhs(mu), mu - tau*kappa,
                                                  residual_scale=0.)
                elif EXPMODE == "path":
                    dx, ds, dz, dt, dk = direction(-s, -tau*kappa)
                else:
                    dxa, dsa, dza, dta, dka = direction(-s, -tau*kappa)
                    aa = maxstep(dsa, dza, dta, dka)
                    mua = ((s+aa*dsa).dot(z+aa*dza)
                           + (tau+aa*dta)*(kappa+aa*dka))/(cones.degree+1)
                    sigma = min(1., max(0., mua/mu))**3
                    rc = sc.rhs(sigma*mu, dsa, dza)
                    a = dxa - x*(dta/tau)
                    dx, ds, dz, dt, dk = direction(
                        rc, sigma*mu-tau*kappa-dta*dka, a.dot(P@a)/tau)
                alpha = .99 * maxstep(ds, dz, dt, dk)
                for backtrack in range(40):
                    sn, zn = s + alpha*ds, z + alpha*dz
                    tn, kn = tau + alpha*dt, kappa + alpha*dk
                    mun = (sn.dot(zn) + tn*kn) / (cones.degree + 1)
                    if wide:
                        total,cn = cones.centrality(sn,zn,mun,tn,kn,individual=True)
                        accepted = ((cn <= CENTER or
                            (total <= total_centrality*(1-.001*alpha)
                             and cn <= max(2.,centrality))) if centering
                            else cn <= NEIGHBORHOOD)
                    else:
                        cn = cones.centrality(sn, zn, mun, tn, kn)
                        limit = (max(CENTER, centrality * (1 - .001*alpha))
                                 if centering else NEIGHBORHOOD)
                        accepted = cn <= limit
                    if accepted:
                        break
                    alpha *= .7
                aa, sigma = centrality, int(centering)
            else:
                dxa, dsa, dza, dta, dka = direction(-s, -tau * kappa)
                aa = maxstep(dsa, dza, dta, dka)
                mua = ((s + aa * dsa).dot(z + aa * dza)
                       + (tau + aa * dta) * (kappa + aa * dka)) / (cones.degree + 1)
                sigma = min(1., max(0., mua / mu))**3
                sigma = max(sigma, .001 if len(cones.exp) else 0.)
                rc = sc.rhs(sigma * mu, dsa, dza)
                rtk = sigma * mu - tau * kappa - dta * dka
                a = dxa - x * (dta / tau)
                gapcorr = a.dot(P @ a) / tau
                dx, ds, dz, dt, dk = direction(rc, rtk, gapcorr)
                alpha = .99 * maxstep(ds, dz, dt, dk)
            if VERBOSE:
                print("  step", alpha, "aff", aa, "sig", sigma, flush=True)
            if not np.isfinite(alpha) or alpha < 1e-11:
                break
            if alpha < 1e-3 and cones.psd:
                force_full = True
            x += alpha * dx
            s += alpha * ds
            s[cones.zero] = 0.
            z += alpha * dz
            tau += alpha * dt
            kappa += alpha * dk
            if not np.isfinite(x).all() or tau <= 0:
                break
        except (ArithmeticError, RuntimeError, ValueError, linalg.LinAlgError) as exc:
            if VERBOSE:
                print("factorization/step failure:", repr(exc), flush=True)
            break
    return best
