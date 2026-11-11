"""Analytic exponential-epigraph condensation and dense primal-dual Newton."""

import math
import time
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import lsmr

from ipm import infnorm, maxstep


def solve_gp(p, maxiter=150):
    from solver import accepted, residuals, log, feasible_slack
    if (p.P.nnz or len(p.eq) or
            any(t != "exponential" for t, _, _, _ in p.nonlinear)):
        return None
    blocks = np.array([[lo, lo + 1, lo + 2] for t, d, lo, hi in p.nonlinear], dtype=int)
    if not len(blocks):
        return None
    ur, vr, wr = blocks.T
    W = p.Ar[wr]
    if np.any(np.diff(W.indptr) != 1) or p.Ar[vr].nnz or np.any(p.b[vr] <= 0):
        return None
    wc, kval = W.indices, -W.data
    if len(np.unique(wc)) != len(wc) or np.any(kval <= 0):
        return None
    if p.A[ur][:, wc].nnz:
        return None
    base_mask = np.ones(p.n, dtype=bool)
    base_mask[wc] = False
    base = np.flatnonzero(base_mask)
    if len(base) > 1500 or len(base) == 0:
        return None
    yconst = p.b[vr]
    M = (-sparse.diags(1 / yconst) @ p.A[ur][:, base]).tocsr()
    offset = p.b[ur] / yconst
    B = p.A[p.ineq][:, base].tocsr()
    C = (p.A[p.ineq][:, wc] @ sparse.diags(1 / kval)).tocsr()
    qo = p.q[wc] / kval
    if np.any(C.data < -1e-13) or np.any(qo < 0):
        return None
    C.data = np.maximum(C.data, 0)
    rhs = p.b[p.ineq] + C @ p.b[wr]
    qb = p.q[base].copy()
    mc = np.asarray(abs(M).max(axis=0).toarray()).ravel()
    bc = np.asarray(abs(B).max(axis=0).toarray()).ravel()
    colscale = 1 / np.maximum.reduce((mc, bc, np.abs(qb), np.full(len(base), 1e-6)))
    M = (M @ sparse.diags(colscale)).tocsr()
    B = (B @ sparse.diags(colscale)).tocsr()
    qb *= colscale
    # A logarithmic least-squares start avoids unnecessarily enormous exp values.
    v = -lsmr(M, offset, atol=1e-8, btol=1e-8, maxiter=100)[0]

    def values(v):
        theta = offset + M @ v
        if np.max(theta, initial=-np.inf) > 700:
            raise FloatingPointError("exponential step overflow")
        ev = yconst * np.exp(np.maximum(theta, -745))
        return theta, ev

    theta, ev = values(v)
    rowscale = 1 / np.maximum.reduce((
        np.ones(len(rhs)), np.abs(rhs), np.abs(B @ v), np.abs(C @ ev)))
    B = (sparse.diags(rowscale) @ B).tocsr()
    C = (sparse.diags(rowscale) @ C).tocsr()
    rhs *= rowscale
    s = np.maximum(1.0, rhs - B @ v - C @ ev)
    z = 1 / s
    begin = time.monotonic()
    n, m = len(v), len(s)
    log("gp dimensions", n, m, len(ev), "M", M.nnz, "C", C.nnz)
    costscale = max(1.0, infnorm(qb))

    def evaluation(v, s, z, target=0.0):
        theta, ev = values(v)
        weight = qo + C.T @ z
        rd = qb + M.T @ (ev * weight) + B.T @ z
        f = B @ v + C @ ev - rhs
        rp = f + s
        rc = s * z - target
        return theta, ev, weight, rd, rp, rc

    def result(v, z, theta, ev):
        x = np.zeros(p.n)
        x[base] = colscale * v
        x[wc] = (ev - p.b[wr]) / kval
        dual = np.zeros(p.m)
        dual[p.ineq] = rowscale * z
        zw = qo + C.T @ z
        et = ev / yconst
        dual[ur], dual[vr], dual[wr] = -et * zw, (theta - 1) * et * zw, zw
        return {"status": "solved", "x": x, "s": feasible_slack(p, x), "z": dual}

    def polish(v, z, full=False):
        active = np.flatnonzero(z > max(1e-9, 1e-7 * infnorm(z)))
        if not len(active):
            return None
        for face_it in range(5 if full else 1):
            theta, ev = values(v)
            J = (B + C.multiply(ev[None, :]) @ M).toarray()
            grad = qb + M.T @ (ev * qo)
            zp = np.zeros(m)
            zp[active] = linalg.lstsq(J[active].T, -grad,
                cond=1e-13, check_finite=False, lapack_driver="gelsy")[0]
            if np.min(zp) >= -1e-10 * max(1, infnorm(zp)):
                zp = np.maximum(zp, 0)
                r = result(v, zp, theta, ev)
                if accepted(p, r, tol=8e-9):
                    return r
            if not full:
                break
            weight = qo + C.T @ z
            H = (M.T @ M.multiply((ev * weight)[:, None])).toarray()
            Ja = J[active]
            K = np.block([[H, Ja.T], [Ja, np.zeros((len(active), len(active)))]])
            rd = grad + J.T @ z
            fa = (B @ v + C @ ev - rhs)[active]
            krhs = -np.r_[rd, fa]
            sc = 1 / np.sqrt(np.maximum(np.max(np.abs(K), axis=1), 1e-12))
            step = sc * linalg.lstsq(K * sc[:, None] * sc[None, :], sc * krhs,
                cond=1e-13, check_finite=False, lapack_driver="gelsy")[0]
            dv, dz = step[:n], step[n:]
            zn = np.zeros(m)
            zn[active] = z[active] + dz
            neg = zn[active] < -1e-10 * max(1, infnorm(zn))
            if np.any(neg):
                active = active[~neg]
                z = np.maximum(zn, 0)
                continue
            v, z = v + dv, np.maximum(zn, 0)
        return None

    polished_at = -100
    for it in range(maxiter):
        theta, ev, weight, rd, rp, _ = evaluation(v, s, z)
        mu = float(s @ z) / max(1, m)
        pobj = float(qb @ v + qo @ (ev - p.b[wr]))
        if it % 5 == 0:
            log("gp", it, "rd", infnorm(rd), "rp", infnorm(rp),
                "gap", mu * m, "obj", pobj, "t", time.monotonic() - begin)
        if ((infnorm(rd) < 5e-8 * costscale and
             infnorm(rp) < 5e-9 and mu * m < 5e-9 * max(1, abs(pobj))) or
                (it > 5 and infnorm(z) > 1e8)):
            r = result(v, z, theta, ev)
            if accepted(p, r, tol=5e-9):
                return r
            cert = {"status": "primal_infeasible", "z": r["z"]}
            if accepted(p, cert, tol=4e-9):
                return cert
            if it % 5 == 0:
                log("gp conic", residuals(p, r))
        if not np.isfinite(mu) or mu <= 0:
            break
        if mu * m < 1e-7 * max(1, abs(pobj)) and it - polished_at >= 4:
            polished_at = it
            r = polish(v.copy(), z.copy(), full=(it > 35))
            if r is not None:
                return r
        try:
            J = (B + C.multiply(ev[None, :]) @ M).toarray()
            H = (M.T @ M.multiply((ev * weight)[:, None])).toarray()
            w = z / s
            G = H + J.T @ (w[:, None] * J)
            dscl = 1 / np.sqrt(np.maximum(np.abs(G.diagonal()), 1e-30))
            GG = G * dscl[:, None] * dscl[None, :]
            GG.flat[::n + 1] += 1e-13
            factor = linalg.cho_factor(GG, lower=True, check_finite=False)

            def solve(rhs):
                u = dscl * linalg.cho_solve(factor, dscl * rhs, check_finite=False)
                for _ in range(2):
                    rlin = rhs - G @ u
                    if infnorm(rlin) < 1e-12 * max(1, infnorm(rhs)):
                        break
                    u += dscl * linalg.cho_solve(factor, dscl * rlin, check_finite=False)
                return u

            def direction(center):
                dv = solve(-rd - J.T @ (center / s + w * rp))
                ds = -rp - J @ dv
                dz = center / s - w * ds
                return dv, ds, dz

            da = direction(-s * z)
            ap, ad = maxstep(s, da[1]), maxstep(z, da[2])
            mua = float((s + ap * da[1]) @ (z + ad * da[2])) / max(1, m)
            sigma = min(0.5, max(0.01, max(0.0, mua / mu) ** 3))
            dv, ds, dz = direction(-s * z + sigma * mu - da[1] * da[2])
            alpha = 0.99 * min(maxstep(s, ds), maxstep(z, dz))
            target = sigma * mu
            drscale = max(costscale, infnorm(qb + M.T @ (ev * weight)), infnorm(B.T @ z), 1.0)
            old = np.linalg.norm(np.r_[rd / drscale, rp, (s * z - target) / max(mu, 1e-10)])
            for attempt in range(2):
                if attempt:
                    sigma = 0.2
                    target = sigma * mu
                    dv, ds, dz = direction(-s * z + target)
                    alpha = 0.99 * min(maxstep(s, ds), maxstep(z, dz))
                    old = np.linalg.norm(np.r_[rd / drscale, rp,
                                                (s * z - target) / max(mu, 1e-10)])
                for ls in range(50):
                    vn, sn, zn = v + alpha * dv, s + alpha * ds, z + alpha * dz
                    try:
                        en = evaluation(vn, sn, zn, target)
                        merit = np.linalg.norm(np.r_[en[3] / drscale, en[4],
                                                     en[5] / max(mu, 1e-10)])
                        if merit <= old * (1 - 1e-4 * alpha):
                            break
                    except (FloatingPointError, OverflowError):
                        pass
                    alpha *= 0.5
                if alpha > 1e-7:
                    break
            if alpha < 1e-14:
                log("gp line search failed", it, alpha, "direction", infnorm(dv))
                break
            v, s, z = vn, sn, zn
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            log("gp failure", str(error))
            break
    return None
