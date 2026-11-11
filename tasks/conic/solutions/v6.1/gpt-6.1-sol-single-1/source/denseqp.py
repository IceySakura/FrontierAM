"""Dense predictor-corrector fallback for small linear-cone QPs."""

import time
import numpy as np

from dense import LU, Cholesky


def step(v, dv):
    ix = dv < 0
    return min(1.0, float(np.min(-v[ix] / dv[ix]))) if np.any(ix) else 1.0


def solve_dense(p, maxiter=65):
    if p.n > 650 or p.m > 4000 or p.n == 0:
        return None
    begin = time.monotonic()
    ad, ar, ap, ac = p.A
    pd, pr, pp, pc = p.P
    A, P = np.zeros((p.m, p.n)), np.zeros((p.n, p.n))
    np.add.at(A, (ar, ac), ad)
    np.add.at(P, (pr, pc), pd)
    q, b = p.q.copy(), p.b.copy()
    D, E = np.ones(p.n), np.ones(p.m)
    for _ in range(5):
        dc = 1 / np.sqrt(np.maximum.reduce((
            np.max(np.abs(A), axis=0, initial=0),
            np.max(np.abs(P), axis=0, initial=0), np.full(p.n, 1e-6))))
        A *= dc[None, :]
        P *= dc[:, None] * dc[None, :]
        q *= dc
        D *= dc
        dr = 1 / np.sqrt(np.maximum(np.max(np.abs(A), axis=1, initial=0), 1e-6))
        A *= dr[:, None]
        b *= dr
        E *= dr
    cost = 1 / max(np.max(np.abs(P), initial=0), np.max(np.abs(q), initial=0), 1e-3)
    P *= cost
    q *= cost
    eq, iq = np.flatnonzero(p.eq), np.flatnonzero(~p.eq)
    F, B = A[eq], A[iq]
    n, k, m = p.n, len(eq), len(iq)
    x, y = np.zeros(n), np.zeros(k)
    s = np.maximum(1.0, np.abs(b[iq]))
    z = 1 / s
    best = np.inf
    for it in range(maxiter):
        rd = P @ x + q + F.T @ y + B.T @ z
        rp, ri = F @ x - b[eq], B @ x + s - b[iq]
        mu = float(s @ z) / max(1, m)
        if it >= 3:
            xo, zo = D * x, np.zeros(p.m)
            zo[eq], zo[iq] = E[eq] * y / cost, E[iq] * z / cost
            r = p.result(xo, zo)
            if p.valid(r, tol=7e-9):
                return r
            if np.max(np.abs(zo), initial=0) > 1e6:
                cert = {"status": "primal_infeasible", "z": zo}
                if p.valid(cert):
                    return cert
        if (not np.isfinite(mu) or mu <= 0 or
                time.monotonic() - begin > 1.0):
            break
        w = z / s
        H = P + B.T @ (w[:, None] * B)
        K = np.block([[H, F.T], [F, np.zeros((k, k))]])
        scale = 1 / np.sqrt(np.maximum(np.max(np.abs(K), axis=1), 1e-20))
        KR = K * scale[:, None] * scale[None, :]
        KR.flat[::n + k + 1] += np.r_[np.full(n, 1e-11), np.full(k, -1e-11)]
        try:
            factor = LU(KR)

            def direction(center):
                rhs = np.r_[-rd - B.T @ (center / s + w * ri), -rp]
                v = scale * factor.solve(scale * rhs)
                for _ in range(4):
                    rr = scale * (rhs - K @ v)
                    if np.max(np.abs(rr) / (1 + np.abs(scale * rhs)), initial=0) < 1e-13:
                        break
                    dv = scale * factor.solve(rr)
                    if np.max(np.abs(dv), initial=0) > 1e5 * max(1, np.max(np.abs(v), initial=0)):
                        break
                    v += dv
                dx, dy = v[:n], v[n:]
                ds = -ri - B @ dx
                dz = center / s - w * ds
                return dx, dy, ds, dz

            da = direction(-s * z)
            ap, az = step(s, da[2]), step(z, da[3])
            mua = float((s + ap * da[2]) @ (z + az * da[3])) / max(1, m)
            sigma = min(1.0, max(0., mua / mu) ** 3)
            dx, dy, ds, dz = direction(-s * z + sigma * mu - da[2] * da[3])
            ap, az = 0.995 * step(s, ds), 0.995 * step(z, dz)
            x += ap * dx
            s += ap * ds
            y += az * dy
            z += az * dz
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            break
    return None
