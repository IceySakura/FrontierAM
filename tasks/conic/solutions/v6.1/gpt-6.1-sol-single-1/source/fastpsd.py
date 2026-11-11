"""Low-startup dense NT solver for small semidefinite programs."""

import time
import numpy as np

from dense import Cholesky


def norm(v):
    return float(np.max(np.abs(v), initial=0.))


class Block:
    def __init__(self, d, lo, hi):
        self.d, self.lo, self.hi = d, lo, hi
        self.c = np.repeat(np.arange(d), np.arange(1, d + 1))
        self.r = np.arange(hi - lo) - self.c * (self.c + 1) // 2
        self.f = np.where(self.r == self.c, 1., np.sqrt(2.))
        self.hf = np.where(self.r == self.c, 1 / np.sqrt(2.), 1.)

    def mat(self, v):
        a = v / self.f
        M = np.zeros((self.d, self.d))
        M[self.r, self.c], M[self.c, self.r] = a, a
        return M

    def vec(self, M):
        return self.f * M[self.r, self.c]

    def scaling(self, sv, zv):
        self.S, self.Z = self.mat(sv), self.mat(zv)
        R, L = np.linalg.cholesky(self.S), np.linalg.cholesky(self.Z)
        _, lam, VT = np.linalg.svd(L.T @ R, full_matrices=False)
        lam = np.maximum(lam, 1e-300)
        Ri = np.linalg.solve(R, np.eye(self.d))
        self.g = (R @ VT.T) * lam[None, :] ** -.5
        self.gi = lam[:, None] ** .5 * (VT @ Ri)
        self.lam = lam
        D = self.gi.T @ self.gi
        self.si = Ri.T @ Ri
        r, c = self.r, self.c
        H = (D[r[:, None], r] * D[c[:, None], c] +
             D[r[:, None], c] * D[c[:, None], r])
        H *= self.hf[:, None] * self.hf[None, :]
        return H

    def rhs(self, mu, sigma, ds=None, dz=None):
        V = -self.Z + sigma * mu * self.si
        if ds is not None:
            X = self.gi @ self.mat(ds) @ self.gi.T
            Y = self.g.T @ self.mat(dz) @ self.g
            C = X @ Y
            C = (C + C.T) / (self.lam[:, None] + self.lam[None, :])
            V -= self.gi.T @ C @ self.gi
        return self.vec(V)

    def step(self, v, dv):
        R = np.linalg.cholesky(self.mat(v))
        V = np.linalg.solve(R, self.mat(dv))
        V = np.linalg.solve(R, V.T)
        ev = np.linalg.eigvalsh(V)[0]
        return min(1., -1 / ev) if ev < 0 else 1.


def solve(p, maxiter=140):
    from fast import log
    if p.n > 200 or p.m > 1500 or any(t != "psd" for t, _, _, _ in p.blocks):
        return None
    ad, ar, ap, ac = p.A
    pd, pr, pp, pc = p.P
    A, P = np.zeros((p.m, p.n)), np.zeros((p.n, p.n))
    np.add.at(A, (ar, ac), ad)
    np.add.at(P, (pr, pc), pd)
    blocks = [Block(d, lo, hi) for t, d, lo, hi in p.blocks]
    q, b, D, E = p.q.copy(), p.b.copy(), np.ones(p.n), np.ones(p.m)
    for _ in range(5):
        dc = 1 / np.sqrt(np.maximum.reduce((
            np.max(np.abs(A), axis=0), np.max(np.abs(P), axis=0), np.full(p.n, 1e-6))))
        A *= dc[None, :]
        P *= dc[:, None] * dc[None, :]
        q *= dc
        D *= dc
        for blk in blocks:
            sl = slice(blk.lo, blk.hi)
            dr = 1 / np.sqrt(max(np.max(np.abs(A[sl])), 1e-6))
            A[sl] *= dr
            b[sl] *= dr
            E[sl] *= dr
    cost = 1 / max(norm(q), norm(P), 1e-3)
    P *= cost
    q *= cost
    s, z = np.zeros(p.m), np.zeros(p.m)
    for blk in blocks:
        sl = slice(blk.lo, blk.hi)
        S = blk.mat(b[sl])
        mineig = np.linalg.eigvalsh(S)[0]
        S.flat[::blk.d + 1] += max(1., 1. - mineig)
        s[sl] = blk.vec(S)
        z[sl] = blk.vec(np.linalg.solve(S, np.eye(blk.d)))
    x = np.zeros(p.n)
    nu = sum(blk.d for blk in blocks)
    begin = time.monotonic()
    for it in range(maxiter):
        rd, rp = P @ x + q + A.T @ z, A @ x + s - b
        mu = float(s @ z) / nu
        if it >= 3:
            r = {"status": "solved", "x": D * x, "s": s / E, "z": E * z / cost}
            if p.valid(r, tol=7e-9):
                log("fast psd solved", it, time.monotonic() - begin)
                return r
            if norm(r["z"]) > 1e4:
                cert = {"status": "primal_infeasible", "z": r["z"]}
                if p.valid(cert):
                    return cert
        if not np.isfinite(mu) or mu <= 0:
            break
        try:
            H = np.zeros((p.m, p.m))
            for blk in blocks:
                sl = slice(blk.lo, blk.hi)
                H[sl, sl] = blk.scaling(s[sl], z[sl])
            G = P + A.T @ H @ A
            scale = 1 / np.sqrt(np.maximum(np.abs(G.diagonal()), 1e-30))
            GR = G * scale[:, None] * scale[None, :]
            GR.flat[::p.n + 1] += 1e-12
            factor = Cholesky(GR)

            def direction(sigma, da=None):
                rc = np.empty(p.m)
                for blk in blocks:
                    sl = slice(blk.lo, blk.hi)
                    rc[sl] = blk.rhs(mu, sigma,
                        None if da is None else da[1][sl],
                        None if da is None else da[2][sl])
                rhs = -rd - A.T @ (rc + H @ rp)
                dx = scale * factor.solve(scale * rhs)
                for _ in range(4):
                    err = rhs - G @ dx
                    if norm(err) < 1e-12 * max(1, norm(rhs)):
                        break
                    dx += scale * factor.solve(scale * err)
                ds = -rp - A @ dx
                dz = rc - H @ ds
                return dx, ds, dz

            def step(v, dv):
                return min(blk.step(v[blk.lo:blk.hi], dv[blk.lo:blk.hi]) for blk in blocks)

            da = direction(0)
            ap, az = step(s, da[1]), step(z, da[2])
            mua = float((s + ap * da[1]) @ (z + az * da[2])) / nu
            sigma = min(1., max(0., mua / mu) ** 3)
            dx, ds, dz = direction(sigma, da)
            ap, az = .99 * step(s, ds), .99 * step(z, dz)
            x += ap * dx
            s += ap * ds
            z += az * dz
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            break
    return None
