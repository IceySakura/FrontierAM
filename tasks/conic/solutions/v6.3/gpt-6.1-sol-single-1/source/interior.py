"""Homogeneous primal-dual predictor/corrector method."""
import os
import sys
import time
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import splu

from accuracy import assess, norm
from cones import exp_centrality


def equilibrate(P, q, A, b, K):
    P, A, q, b = P.copy(), A.copy(), q.copy(), b.copy()
    n, m = q.size, b.size
    D, E = np.ones(n), np.ones(m)
    ac = np.repeat(np.arange(n), np.diff(A.indptr))
    pc = np.repeat(np.arange(n), np.diff(P.indptr))
    for _ in range(5):
        cn = np.zeros(n)
        np.maximum.at(cn, ac, np.abs(A.data))
        np.maximum.at(cn, pc, np.abs(P.data))
        cn = np.maximum(cn, 1e-4)
        dc = np.clip(1 / np.sqrt(cn), 1e-3, 1e3)
        P.data *= dc[P.indices] * dc[pc]
        A.data *= dc[ac]
        q *= dc
        D *= dc
        rn = np.zeros(m)
        np.maximum.at(rn, A.indices, np.abs(A.data))
        rn = np.maximum(rn, 1e-4)
        de = K.row_scales(np.clip(1 / np.sqrt(rn), 1e-3, 1e3))
        A.data *= de[A.indices]
        b *= de
        E *= de
    c = 1 / max(1., norm(q), norm(P.data))
    P.data *= c
    q *= c
    return P, q, A, b, D, E, c


def psd_coefficients(A, K):
    out = []
    for block in K.psd:
        sl, d, ii, jj, weight = block
        AB = A[sl, :].tocsc()
        cols = np.flatnonzero(np.diff(AB.indptr))
        used = np.unique(AB.indices)
        if used.size <= 4000 and used.size**2 < max(1, cols.size*d*d*2):
            reduced = AB[used, :][:, cols].tocsc()
            out.append(("kernel", cols, used, reduced))
        else:
            B = AB[:, cols].toarray().T
            mats = np.zeros((cols.size, d, d))
            mats[:, ii, jj] = B / weight[None, :]
            mats[:, jj, ii] = mats[:, ii, jj]
            out.append(("dense", cols, None, mats))
    return out


class LinearSystem:
    def __init__(self, P, A, K, scaling, regularization, coefficients=None):
        self.P, self.A, self.K, self.scaling = P, A, K, scaling
        n, m = P.shape[0], A.shape[0]
        self.n, self.m = n, m
        ordering = os.environ.get("SOLVER_KKT_ORDER", "COLAMD")
        self.permutation = None
        self.reduced = bool(K.psd) and not K.small_psd
        if not self.reduced:
            H = scaling.matrix()
            U = scaling.auxiliaries()
            if U is None:
                M = sparse.bmat([[P, A.T], [A, -H]], format="csc")
                reg = np.r_[np.full(n, regularization),
                            np.full(m, -regularization)]
            else:
                M = sparse.bmat([[P, A.T, None],
                                 [A, -H, U],
                                 [None, U.T, sparse.eye(scaling.naux)]], format="csc")
                reg = np.r_[np.full(n, regularization),
                            np.full(m, -regularization), np.zeros(scaling.naux)]
            self.M = M
            self.regularized = M + sparse.diags(reg, format="csc")
            if ordering == "STATIC" or (getattr(K, "lu_static", False) and
                                       "SOLVER_KKT_ORDER" not in os.environ):
                perm = np.r_[n + np.flatnonzero(~K.zero), np.arange(n),
                             n + K.zi, np.arange(n+m, M.shape[0])]
                self.permutation = perm
                self.lu = splu(self.regularized[perm, :][:, perm].tocsc(),
                                permc_spec="NATURAL", diag_pivot_thresh=0.,
                                options={"SymmetricMode": True})
            else:
                self.lu = splu(self.regularized, permc_spec=ordering)
        else:
            Hi = scaling.matrix(inverse=True)
            C = (P + A.T @ Hi @ A).tocsc()
            for ids, beta, w, _, _, _ in scaling.soc:
                if ids.shape[1] > 48:
                    for rows, be, ww in zip(ids, beta, w):
                        B = A[rows, :]
                        diag = np.ones(rows.size) / be ** 2
                        diag[0] *= -1
                        vv = ww.copy()
                        vv[1:] *= -1
                        u = np.asarray(B.T @ vv).ravel() * (np.sqrt(2) / be)
                        C += B.T @ sparse.diags(diag) @ B
                        C += sparse.csc_matrix(np.outer(u, u))
            for bi, (block, R, Ri, lam, Wi) in enumerate(scaling.psd):
                sl, d, ii, jj, weight = block
                mode, cols, used, coeff = coefficients[bi]
                if cols.size == 0:
                    continue
                if mode == "kernel":
                    ai, aj, wt = ii[used], jj[used], weight[used]
                    factor = wt[:, None] / wt[None, :]
                    factor[:, ai == aj] *= .5
                    H = (Wi[ai[:, None], ai] * Wi[aj[:, None], aj] +
                         Wi[ai[:, None], aj] * Wi[aj[:, None], ai]) * factor
                    temp = coeff.T @ H
                    gram = coeff.T @ temp.T
                else:
                    BB = (Ri @ coeff) @ Ri.T
                    BB = BB.reshape(cols.size, d*d)
                    gram = BB @ BB.T
                r, c = np.meshgrid(cols, cols, indexing="ij")
                C += sparse.csc_matrix((gram.ravel(), (r.ravel(), c.ravel())),
                                       shape=(n, n))
            E = A[K.zi, :].tocsc()
            M = sparse.bmat([[C, E.T], [E, None]], format="csc")
            self.M = M
            reg = np.r_[np.full(n, regularization),
                        np.full(K.zi.size, -regularization)]
            self.lu = splu(M + sparse.diags(reg, format="csc"),
                            permc_spec="COLAMD" if ordering == "STATIC" else ordering)

    def raw_solve(self, rhs):
        if self.permutation is None:
            return self.lu.solve(rhs)
        out = np.empty_like(rhs)
        out[self.permutation] = self.lu.solve(rhs[self.permutation])
        return out

    def solve(self, r1, r2):
        if not self.reduced:
            rhs = np.r_[r1, r2, np.zeros(self.scaling.naux)]
        else:
            rhs = np.r_[r1 + self.A.T @ self.scaling.apply_inverse(r2), r2[self.K.zi]]
        v = self.raw_solve(rhs)
        for _ in range(6):
            residual = rhs - self.M @ v
            if norm(residual) <= 1e-13 * (1 + norm(rhs)):
                break
            dv = self.raw_solve(residual)
            v += dv
            if norm(dv) < 1e-13 * max(1., norm(v)):
                break
        if self.permutation is not None and (not np.isfinite(v).all() or
                norm(rhs-self.M@v) > 1e-8*(1+norm(rhs))):
            self.K.lu_static = False
            self.permutation = None
            self.lu = splu(self.regularized, permc_spec="COLAMD")
            v = self.lu.solve(rhs)
            for _ in range(3):
                v += self.lu.solve(rhs-self.M@v)
        if not self.reduced:
            return v[:self.n], v[self.n:self.n+self.m]
        x = v[:self.n]
        z = self.scaling.apply_inverse(self.A @ x - r2)
        z[self.K.zi] = v[self.n:]
        return x, z


def solve_conic(P0, q0, A0, b0, K, start=None):
    debug = bool(os.environ.get("SOLVER_DEBUG"))
    n, m = q0.size, b0.size
    P, q, A, b, D, E, c = equilibrate(P0, q0, A0, b0, K)
    K.lu_static = (K.ei.shape[0] >= 700 and 0 < K.zi.size <= 10 and A.nnz < 4*m)
    if K.small_psd and m > 3*n and A.nnz > 4*m:
        K.small_psd = False
    coefficients = psd_coefficients(A, K) if K.psd and not K.small_psd else None
    x = np.zeros(n)
    s, z = K.initial()
    tau, kap = 1., 1.
    best = ("unresolved", None, None, None)
    bestscore = float("inf")
    best_interior = None
    interior_score = float("inf")
    previous = float("inf")
    stall = 0
    last_refine_score = float("inf")
    for it in range(180):
        if 1e-8 < tau < 1e8 and tau != 1.:
            scale = 1. / tau
            x *= scale
            s *= scale
            z *= scale
            kap *= scale
            tau = 1.
        Px, Ax, Atz = P @ x, A @ x, A.T @ z
        xPx = float(x @ Px)
        rd = Px + Atz + q * tau
        rp = Ax + s - b * tau
        rg = float(q @ x + b @ z + xPx / tau + kap)
        mu = float(s @ z + tau * kap) / (K.degree + 1)
        if not np.isfinite(mu) or mu <= 0 or tau <= 0 or kap <= 0:
            break
        if tau > 1e-12:
            candidate = ("solved", D * x / tau, s / E / tau, E * z / (c * tau))
            # Linear tests first avoid eigenvalue work on early iterates.
            quick = max(norm(rp) / (1 + norm(b) * tau),
                        norm(rd) / (1 + norm(q) * tau),
                        abs(xPx / tau ** 2 + (q @ x + b @ z) / tau)
                        / max(1., abs(xPx / tau ** 2), abs(q @ x / tau), abs(b @ z / tau)))
            if quick < 1e-2 or it % 5 == 0:
                score, res = assess(P0, q0, A0, b0, K, candidate)
                if score < interior_score:
                    best_interior, interior_score = candidate, score
                if score < bestscore:
                    best, bestscore = candidate, score
                if score <= .5:
                    return candidate
                threshold = 5. if K.psd and max(p[1] for p in K.psd) > 12 else 1e3
                if score < threshold and score < .05 * last_refine_score:
                    last_refine_score = score
                    from refine import refine
                    initial = ("solved", candidate[1] / D, candidate[2] * E,
                               candidate[3] * c / E)
                    refined = refine(P, q, A, b, K, initial,
                                     original=(P0, q0, A0, b0, D, E, c), maxiter=6)
                    refined_score = assess(P0, q0, A0, b0, K, refined)[0]
                    if refined_score < bestscore:
                        best, bestscore = refined, refined_score
                    if refined_score <= .5:
                        return refined
        if it >= 4:
            pi = ("primal_infeasible", None, None, E * z / c)
            di = ("dual_infeasible", D * x, s / E, None)
            for candidate in (pi, di):
                if candidate[0] == "primal_infeasible" and b @ z >= -1e-8:
                    continue
                if candidate[0] == "dual_infeasible" and q @ x >= -1e-8:
                    continue
                score, res = assess(P0, q0, A0, b0, K, candidate)
                if score <= .25:
                    return candidate
        if debug:
            print("ipm", it, "mu", "%.3e" % mu, "tau", "%.3e" % tau,
                  "kap", "%.3e" % kap, "rp", "%.3e" % norm(rp),
                  "rd", "%.3e" % norm(rd), "best", "%.3e" % bestscore, file=sys.stderr)
        try:
            sc = K.scaling(s, z, mu)
            reg = 1e-12
            try:
                ls = LinearSystem(P, A, K, sc, reg, coefficients)
            except RuntimeError:
                ls = LinearSystem(P, A, K, sc, 1e-7, coefficients)
            xt, zt = ls.solve(-q, b)
            g = q + 2 * Px / tau
            denom = float(g @ xt + b @ zt - xPx / tau**2 - kap / tau)

            def direction(sigma, ds_aff=None, dz_aff=None, dt_aff=0., dk_aff=0.):
                eta = 1 - sigma
                rc = sc.rhs(sigma, ds_aff, dz_aff)
                rt = -tau * kap + sigma * mu - dt_aff * dk_aff
                xx, zz = ls.solve(-eta * rd, -eta * rp - rc)
                extra = 0.
                if ds_aff is not None and P.nnz:
                    v = dx_aff - dt_aff * x / tau
                    extra = float(v @ (P @ v)) / tau
                dt = float(-eta * rg - extra - rt / tau - g @ xx - b @ zz) / denom
                dx, dz = xx + xt * dt, zz + zt * dt
                dk = rt / tau - kap / tau * dt
                ds = -eta * rp - A @ dx + b * dt
                return dx, ds, dz, dt, dk

            oldcentral = exp_centrality(s[K.ei], z[K.ei], mu) if K.ei.size and debug else 0.
            dx_aff, ds_aff, dz_aff, dt_aff, dk_aff = direction(0.)
            aa = sc.step(ds_aff, dz_aff)
            if dt_aff < 0:
                aa = min(aa, -tau / dt_aff)
            if dk_aff < 0:
                aa = min(aa, -kap / dk_aff)
            aa = max(0., min(1., aa))
            mu_aff = float((s + aa * ds_aff) @ (z + aa * dz_aff)
                           + (tau + aa * dt_aff) * (kap + aa * dk_aff)) / (K.degree + 1)
            sigma = np.clip((max(0., mu_aff) / mu)**3, 0., .99)
            if K.ei.size:
                sigma = max(sigma, .05)
            dx, ds, dz, dt, dk = direction(sigma, ds_aff, dz_aff, dt_aff, dk_aff)
            alpha = sc.step(ds, dz)
            if dt < 0:
                alpha = min(alpha, -tau / dt)
            if dk < 0:
                alpha = min(alpha, -kap / dk)
            alpha = min(1., .99 * alpha)
            if K.ei.size and alpha < .05:
                for center in (.5, .9, 1.):
                    trial = direction(center)
                    tx, ts, tz, tt, tk = trial
                    ta = sc.step(ts, tz)
                    if tt < 0:
                        ta = min(ta, -tau / tt)
                    if tk < 0:
                        ta = min(ta, -kap / tk)
                    ta = .99 * min(1., ta)
                    if ta > max(.05, 2 * alpha):
                        dx, ds, dz, dt, dk = trial
                        alpha = ta
                        sigma = center
                        break
            if debug:
                print("step", "%.4e" % alpha, "aff", "%.4e" % aa,
                      "sigma", "%.4e" % sigma, "central", "%.3e" % oldcentral,
                      file=sys.stderr)
            if not np.isfinite(alpha) or alpha < 1e-10:
                break
            x += alpha * dx
            s += alpha * ds
            z += alpha * dz
            tau += alpha * dt
            kap += alpha * dk
            if mu < previous * .999:
                stall = 0
            else:
                stall += 1
            previous = mu
            if stall > 20:
                break
        except (RuntimeError, np.linalg.LinAlgError, ValueError, FloatingPointError):
            if debug:
                import traceback
                traceback.print_exc()
            break
    if bestscore <= 1.:
        return best
    if best_interior is not None and interior_score < 1e7:
        from refine import refine
        initial = ("solved", best_interior[1] / D, best_interior[2] * E,
                   best_interior[3] * c / E)
        refined = refine(P, q, A, b, K, initial,
                         original=(P0, q0, A0, b0, D, E, c), maxiter=12)
        if assess(P0, q0, A0, b0, K, refined)[0] < bestscore:
            return refined
    return best
