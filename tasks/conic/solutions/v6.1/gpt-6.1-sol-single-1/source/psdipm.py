"""Nesterov-Todd semidefinite path following with sparse-support Schur systems."""

import time
import warnings
import numpy as np
from scipy import sparse, linalg

from coneipm import Product
from ipm import equilibrate, infnorm


class PSD:
    def __init__(self, d, ix, B):
        self.d, self.ix = d, ix
        self.c = np.repeat(np.arange(d), np.arange(1, d + 1))
        self.r = np.arange(len(ix)) - self.c * (self.c + 1) // 2
        self.f = np.where(self.r == self.c, 1.0, np.sqrt(2.0))
        self.support = np.flatnonzero(np.asarray(B.getnnz(axis=1)).ravel())
        self.B = B[self.support].tocsr()
        self.br, self.bc = self.r[self.support], self.c[self.support]
        self.bf = np.where(self.br == self.bc, 1 / np.sqrt(2.0), 1.0)
        self.D = None

    def mat(self, v):
        M = np.zeros((self.d, self.d))
        w = v / self.f
        M[self.r, self.c] = w
        M[self.c, self.r] = w
        return M

    def vec(self, M):
        return M[self.r, self.c] * self.f

    def initial(self, b):
        S = self.mat(b)
        ev = linalg.eigh(S, subset_by_index=[0, 0], eigvals_only=True, check_finite=False)[0]
        S.flat[::self.d + 1] += max(1, 1 - ev)
        Z = linalg.cho_solve(linalg.cho_factor(S, lower=True, check_finite=False),
                             np.eye(self.d), check_finite=False)
        return self.vec(S), self.vec(Z)

    def scaling(self, s, z):
        S, Z = self.mat(s), self.mat(z)
        R = linalg.cholesky(S, lower=True, check_finite=False)
        L = linalg.cholesky(Z, lower=True, check_finite=False)
        _, lam, VT = linalg.svd(L.T @ R, full_matrices=False,
                               check_finite=False, lapack_driver="gesvd")
        U = VT.T
        lam = np.maximum(lam, 1e-300)
        Ri = linalg.solve_triangular(R, np.eye(self.d), lower=True, check_finite=False)
        self.G = (R @ U) * lam[None, :] ** -0.5
        self.Gi = lam[:, None] ** 0.5 * (U.T @ Ri)
        self.lam = lam
        self.D = self.Gi.T @ self.Gi
        self.Si = Ri.T @ Ri
        self.Z = Z

    def apply(self, v):
        return self.vec(self.D @ self.mat(v) @ self.D)

    def schur(self):
        D, r, c, f, B = self.D, self.br, self.bc, self.bf, self.B
        k, n = B.shape
        U = np.empty((k, n))
        chunk = min(256, max(1, 2000000 // max(1, k)))
        for lo in range(0, k, chunk):
            hi = min(k, lo + chunk)
            rr, cc = r[lo:hi], c[lo:hi]
            V = (D[rr[:, None], r] * D[cc[:, None], c] +
                 D[rr[:, None], c] * D[cc[:, None], r])
            V *= f[lo:hi, None] * f[None, :]
            U[lo:hi] = (B.T @ V.T).T
        return np.asarray(B.T @ U)

    def rhs(self, mu, sigma, ds=None, dz=None):
        R = -self.Z + sigma * mu * self.Si
        if ds is not None:
            X = self.Gi @ self.mat(ds) @ self.Gi.T
            Y = self.G.T @ self.mat(dz) @ self.G
            C = X @ Y
            C = (C + C.T) / (self.lam[:, None] + self.lam[None, :])
            R -= self.Gi.T @ C @ self.Gi
        return self.vec(R)

    def step(self, s, ds):
        S, DS = self.mat(s), self.mat(ds)
        R = linalg.cholesky(S, lower=True, check_finite=False)
        V = linalg.solve_triangular(R, DS, lower=True, check_finite=False)
        V = linalg.solve_triangular(R, V.T, lower=True, check_finite=False)
        ev = linalg.eigh(V, subset_by_index=[0, 0], eigvals_only=True, check_finite=False)[0]
        return min(1.0, -1 / ev) if ev < 0 else 1.0


def solve_psd(p, maxiter=180, timelimit=250):
    from solver import residuals, accepted, log
    from fast import remaining
    if p.n > 4500 or len(p.eq) > min(2000, 2 * p.n + 50):
        return None
    start = time.monotonic()
    prod = Product(p)
    P, A, q, b, D, E, cost = equilibrate(p.P, p.A, p.q, p.b, blocks=p.blocks)
    eq, nz = p.eq, prod.nz
    B, F = A[nz].tocsc(), A[eq].tocsc()
    Bt, Ft = B.T.tocsc(), F.T.tocsc()
    beq, biq = b[eq], b[nz]
    pos = np.full(p.m, -1, dtype=np.int64)
    pos[nz] = np.arange(len(nz))
    blocks = []
    for t, d, lo, hi in p.nonlinear:
        if t == "psd":
            ix = np.arange(pos[lo], pos[lo] + hi - lo)
            blocks.append(PSD(d, ix, B[ix]))
            if blocks[-1].B.shape[0] * p.n > 100000000:
                return None
            prod.nu += d
    n, k = p.n, len(eq)
    x, y = np.zeros(n), np.zeros(k)
    s, z = prod.initial(biq)
    for blk in blocks:
        s[blk.ix], z[blk.ix] = blk.initial(biq[blk.ix])
    delta = 1e-12
    best = np.inf
    for it in range(maxiter):
        rd = P @ x + q + Ft @ y + Bt @ z
        rp, ri = F @ x - beq, B @ x + s - biq
        mu = float(s @ z) / max(prod.nu, 1)
        if it >= 2:
            xo = D * x
            zo, so = np.zeros(p.m), np.zeros(p.m)
            zo[eq], zo[nz] = E[eq] * y / cost, E[nz] * z / cost
            so[nz] = s / E[nz]
            r = {"status": "solved", "x": xo, "s": so, "z": zo}
            vals = residuals(p, r)
            err = max(vals)
            if it % 5 == 0:
                log("psd-ipm", it, vals, "mu", mu, "t", time.monotonic() - start)
            if err < 6e-9:
                return r
            best = min(best, err)
            if infnorm(zo) > 100:
                cert = {"status": "primal_infeasible", "z": zo}
                if accepted(p, cert, tol=4e-9):
                    return cert
            if infnorm(xo) > 100:
                cert = {"status": "dual_infeasible", "x": xo, "s": -p.A @ xo}
                if accepted(p, cert, tol=4e-9):
                    return cert
            if it > 15 and err > best * 1000:
                break
        if not np.isfinite(mu) or mu <= 0 or time.monotonic() - start > timelimit or remaining() < 2.:
            break
        try:
            H = prod.scaling(s, z, mu)
            G = (P + Bt @ H @ B).toarray()
            for blk in blocks:
                blk.scaling(s[blk.ix], z[blk.ix])
                G += blk.schur()
            diagsc = 1 / np.sqrt(np.maximum(np.abs(G.diagonal()), 1e-30))
            GG = G * diagsc[:, None] * diagsc[None, :]
            regularization = 1e-15 if mu < 1e-4 else delta
            GG.flat[::n + 1] += regularization
            try:
                factor = linalg.cho_factor(GG, lower=True, check_finite=False)
                chol = True
            except np.linalg.LinAlgError:
                factor = linalg.lu_factor(GG, check_finite=False)
                chol = False

            def gsolve(rhs):
                v = (linalg.cho_solve(factor, diagsc * rhs, check_finite=False) if chol else
                     linalg.lu_solve(factor, diagsc * rhs, check_finite=False))
                return diagsc * v

            if k:
                GF = np.column_stack([gsolve(v) for v in Ft.toarray().T])
                Schur = F @ GF + delta * np.eye(k)

            def solve(rhs):
                u = gsolve(rhs[:n])
                if k:
                    dy = linalg.solve(Schur, F @ u - rhs[n:], assume_a="sym", check_finite=False)
                    return np.r_[u - GF @ dy, dy]
                return u

            def apply(v):
                out = H @ v
                for blk in blocks:
                    out[blk.ix] = blk.apply(v[blk.ix])
                return out

            def center(sigma, da=None):
                out = prod.rhs(s, z, mu, sigma, None if da is None else (da[2], da[3]))
                for blk in blocks:
                    out[blk.ix] = blk.rhs(mu, sigma,
                        None if da is None else da[2][blk.ix],
                        None if da is None else da[3][blk.ix])
                return out

            def step(v, dv, dual=False):
                alpha = prod.step(v, dv, dual=dual)
                for blk in blocks:
                    alpha = min(alpha, blk.step(v[blk.ix], dv[blk.ix]))
                return alpha

            def direction(rcent):
                rhs = np.r_[-rd - Bt @ (rcent + apply(ri)), -rp]
                u = solve(rhs)
                for _ in range(12 if mu < 1e-4 else 4):
                    rr = rhs - np.r_[G @ u[:n] + Ft @ u[n:], F @ u[:n]]
                    if infnorm(rr) < 1e-12 * max(1, infnorm(rhs)):
                        break
                    du = solve(rr)
                    if infnorm(du) > 1e5 * max(1, infnorm(u)):
                        break
                    u += du
                dx, dy = u[:n], u[n:]
                ds = -ri - B @ dx
                dz = rcent - apply(ds)
                return dx, dy, ds, dz

            da = direction(center(0))
            ap, ad = step(s, da[2]), step(z, da[3], dual=True)
            mua = float((s + ap * da[2]) @ (z + ad * da[3])) / max(prod.nu, 1)
            sigma = min(1.0, max(0.0, mua / mu) ** 3)
            dx, dy, ds, dz = direction(center(sigma, da))
            ap, ad = 0.99 * step(s, ds), 0.99 * step(z, dz, dual=True)
            if min(ap, ad) < 1e-13:
                break
            x += ap * dx
            s += ap * ds
            y += ad * dy
            z += ad * dz
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
            log("psd failure", str(e))
            break
    return None
