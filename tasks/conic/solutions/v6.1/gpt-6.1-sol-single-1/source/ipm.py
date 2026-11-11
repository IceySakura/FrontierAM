"""Sparse infeasible-start Mehrotra predictor-corrector for convex QPs."""

import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def infnorm(v):
    return np.max(np.abs(v), initial=0.0)


def maxstep(v, dv):
    ix = dv < 0
    if not np.any(ix):
        return 1.0
    return min(1.0, float(np.min(-v[ix] / dv[ix])))


def equilibrate(P, A, q, b, iterations=5, blocks=None):
    n, m = len(q), len(b)
    P, A = P.copy(), A.copy()
    q, b = q.copy(), b.copy()
    D, E = np.ones(n), np.ones(m)
    blockidx = []
    if blocks is not None:
        groups = {}
        for kind, dim, lo, hi in blocks:
            if kind not in ("zero", "nonnegative"):
                groups.setdefault(hi - lo, []).append(lo)
        blockidx = [np.array(starts)[:, None] + np.arange(width)
                    for width, starts in groups.items()]
    for _ in range(iterations):
        ac = np.asarray(abs(A).max(axis=0).toarray()).ravel()
        pc = np.asarray(abs(P).max(axis=0).toarray()).ravel()
        dc = 1 / np.sqrt(np.maximum(np.maximum(ac, pc), 1e-6))
        dc = np.clip(dc, 1e-3, 1e3)
        D *= dc
        A = A @ sparse.diags(dc)
        P = sparse.diags(dc) @ P @ sparse.diags(dc)
        q *= dc
        ar = np.asarray(abs(A).max(axis=1).toarray()).ravel()
        dr = 1 / np.sqrt(np.maximum(ar, 1e-6))
        dr = np.clip(dr, 1e-3, 1e3)
        for ix in blockidx:
            dr[ix] = (1 / np.sqrt(np.maximum(np.max(ar[ix], axis=1), 1e-6)))[:, None]
        E *= dr
        A = sparse.diags(dr) @ A
        b *= dr
    cost = 1 / max(infnorm(q), infnorm(P.data), 1e-3)
    P *= cost
    q *= cost
    return P.tocsc(), A.tocsr(), q, b, D, E, cost


def solve_qp(p, maxiter=70, timelimit=20):
    # Delayed import also permits this module to be tested independently.
    from solver import accepted, residuals, feasible_slack, log, polish_qp
    start = time.monotonic()
    if p.n == 0:
        return None
    P, A, q, b, D, E, cost = equilibrate(p.P, p.A, p.q, p.b)
    eq, iq = p.eq, p.ineq
    B, F = A[iq].tocsc(), A[eq].tocsc()
    Bt, Ft = B.T.tocsc(), F.T.tocsc()
    beq, biq = b[eq], b[iq]
    n, k, m = p.n, len(eq), len(iq)
    x, y = np.zeros(n), np.zeros(k)
    s = np.maximum(1.0, np.abs(biq))
    z = 1 / s
    delta = 1e-11
    previous = np.inf
    best = None
    for it in range(maxiter):
        px, bx, fx = P @ x, B @ x, F @ x
        rd, rp, ri = px + q + Ft @ y + Bt @ z, fx - beq, bx + s - biq
        mu = float(s @ z) / max(1, m)
        if it >= 3:
            xo = D * x
            zo = np.zeros(p.m)
            zo[eq], zo[iq] = E[eq] * y / cost, E[iq] * z / cost
            r = {"status": "solved", "x": xo, "s": feasible_slack(p, xo), "z": zo}
            vals = residuals(p, r)
            error = max(vals)
            if it % 5 == 0:
                log("qp-ipm", it, vals, "mu", mu, "t", time.monotonic() - start)
            if error < 5e-9:
                return r
            if best is None or error < best[0]:
                best = (error, r)
            if it > 8 and error > previous * 100:
                break
            previous = min(previous, error)
            if it > 12 and mu < 1e-13:
                r2 = polish_qp(p, xo, zo, max_iter=1)
                if r2 is not None:
                    return r2
            if infnorm(zo) > 1e5:
                cert = {"status": "primal_infeasible", "z": zo}
                if accepted(p, cert):
                    return cert
            if infnorm(xo) > 1e5:
                cert = {"status": "dual_infeasible", "x": xo, "s": -p.A @ xo}
                if accepted(p, cert):
                    return cert
        if not np.isfinite(mu) or mu <= 0 or time.monotonic() - start > timelimit:
            break
        w = np.minimum(z / s, 1e30)
        H = (P + Bt @ B.multiply(w[:, None])).tocsc()
        hd = np.maximum(np.abs(H.diagonal()), 1.0)
        reg = delta * hd
        try:
            if k <= 20:
                from scipy import linalg
                hlu = splu(H + sparse.diags(reg), permc_spec="MMD_AT_PLUS_A",
                            diag_pivot_thresh=0.01)
                HF = hlu.solve(Ft.toarray())
                Schur = F @ HF + delta * np.eye(k)

                def solve(rhs):
                    u = hlu.solve(rhs[:n])
                    dy = linalg.solve(Schur, F @ u - rhs[n:],
                                      assume_a="sym", check_finite=False)
                    return np.r_[u - HF @ dy, dy]
            else:
                K = sparse.bmat([
                    [H + sparse.diags(reg), Ft],
                    [F, -delta * sparse.eye(k, format="csc")],
                ], format="csc")
                lu = splu(K, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.01)
                solve = lu.solve
        except (RuntimeError, ValueError, MemoryError):
            delta *= 100
            if delta > 1e-3:
                break
            continue

        def direction(center):
            v = center / s + w * ri
            rhs = np.r_[-rd - Bt @ v, -rp]
            u = solve(rhs)
            # Correct static regularization in the original KKT equations.
            for _ in range(3):
                dx, dy = u[:n], u[n:]
                rlin = rhs - np.r_[H @ dx + Ft @ dy, F @ dx]
                if infnorm(rlin) < 1e-11 * max(1.0, infnorm(rhs)):
                    break
                du = solve(rlin)
                if infnorm(du) > 1e6 * max(1.0, infnorm(u)):
                    break
                u += du
            dx, dy = u[:n], u[n:]
            ds = -ri - B @ dx
            dz = center / s - w * ds
            return dx, dy, ds, dz

        try:
            da = direction(-s * z)
            ap, ad = maxstep(s, da[2]), maxstep(z, da[3])
            mua = float((s + ap * da[2]) @ (z + ad * da[3])) / max(1, m)
            sigma = min(1.0, max(0.0, mua / mu) ** 3)
            dx, dy, ds, dz = direction(-s * z + sigma * mu - da[2] * da[3])
            ap, ad = 0.995 * maxstep(s, ds), 0.995 * maxstep(z, dz)
            x += ap * dx
            s += ap * ds
            y += ad * dy
            z += ad * dz
            if not all(np.isfinite(v).all() for v in (x, s, y, z)):
                break
        except (ValueError, FloatingPointError):
            break
    if best is not None and best[0] < 8e-9:
        return best[1]
    return None
