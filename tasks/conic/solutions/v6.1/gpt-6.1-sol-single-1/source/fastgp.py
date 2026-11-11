"""NumPy-only exponential epigraph elimination with compiled sparse kernels."""

import time
import numpy as np

from dense import Cholesky
from native import affine_product, Derivatives


def infnorm(v):
    return float(np.max(np.abs(v), initial=0))


def maxstep(v, dv):
    use = dv < 0
    return min(1., float(np.min(-v[use] / dv[use]))) if np.any(use) else 1.


def solve(p, maxiter=100):
    from fast import log
    if len(p.P[0]) or any(t not in ("zero", "nonnegative", "exponential")
                          for t, _, _, _ in p.blocks):
        return None
    ad, ar, ap, ac = p.A
    eqentry = p.eq[ar]
    edeg = np.bincount(ac[eqentry], minlength=p.n)
    cand = np.flatnonzero(eqentry & (edeg[ac] == 1))
    if not len(cand):
        return None
    score = np.diff(ap)[ac[cand]]
    order = np.lexsort((ac[cand], score, ar[cand]))
    cand = cand[order]
    first = np.r_[True, ar[cand][1:] != ar[cand][:-1]]
    cand = cand[first]
    pivrows, pivcols, coeff = ar[cand], ac[cand], ad[cand]
    if len(pivrows) != np.sum(p.eq):
        return None
    keepcol = np.ones(p.n, dtype=bool)
    keepcol[pivcols] = False
    free = np.flatnonzero(keepcol)
    newcol = np.full(p.n, -1, dtype=np.int32)
    newcol[free] = np.arange(len(free))
    rowpiv = np.full(p.m, -1, dtype=np.int32)
    rowpiv[pivrows] = pivcols
    rowcoef = np.ones(p.m)
    rowcoef[pivrows] = coeff
    use = p.eq[ar] & keepcol[ac]
    tr = np.r_[free, rowpiv[ar[use]]]
    tc = np.r_[np.arange(len(free), dtype=np.int32), newcol[ac[use]]]
    tv = np.r_[np.ones(len(free)), -ad[use] / rowcoef[ar[use]]]
    order = np.argsort(tr, kind="stable")
    tr, tc, tv = tr[order], tc[order], tv[order]
    tp = np.r_[0, np.cumsum(np.bincount(tr, minlength=p.n))].astype(np.int64)
    offset = np.zeros(p.n)
    offset[pivcols] = p.b[pivrows] / coeff
    b = p.b - p.mul("A", offset)
    q = np.bincount(tc, weights=tv * p.q[tr], minlength=len(free))
    capacity = int(np.diff(ap).astype(np.int64) @ np.diff(tp))
    if capacity > 12000000:
        return None
    rr, cc, aa = affine_product(p.A, (tp, tc.astype(np.int32), tv),
                                (~p.eq).astype(np.uint8), capacity)
    if not len(aa):
        return None
    keys = rr.astype(np.int64) * len(free) + cc
    order = np.argsort(keys, kind="stable")
    keys, rr, cc, aa = keys[order], rr[order], cc[order], aa[order]
    first = np.r_[True, keys[1:] != keys[:-1]]
    starts = np.flatnonzero(first)
    aa = np.add.reduceat(aa, starts)
    rr, cc = rr[starts], cc[starts]
    use = aa != 0
    rr, cc, aa = rr[use], cc[use], aa[use]
    ptr = np.r_[0, np.cumsum(np.bincount(rr, minlength=p.m))]
    exp = np.array([[lo, lo + 1, lo + 2] for t, d, lo, hi in p.blocks
                    if t == "exponential"], dtype=int)
    ur, vr, wr = exp.T
    if np.any(ptr[vr + 1] != ptr[vr]) or np.any(b[vr] <= 0):
        return None
    if np.any(ptr[wr + 1] - ptr[wr] != 1):
        return None
    wi = ptr[wr]
    wc, kval = cc[wi], -aa[wi]
    if np.any(kval <= 0) or len(np.unique(wc)) != len(wc):
        return None
    basemask = np.ones(len(free), dtype=bool)
    basemask[wc] = False
    base = np.flatnonzero(basemask)
    nn = np.flatnonzero(p.nn)
    nt, n, m = len(wc), len(base), len(nn)
    if n == 0 or n > 500 or m > 2000:
        return None
    basepos = np.full(len(free), -1, dtype=np.int32)
    basepos[base] = np.arange(n)
    wpos = np.full(len(free), -1, dtype=np.int32)
    wpos[wc] = np.arange(nt)
    rupos = np.full(p.m, -1, dtype=np.int32)
    rupos[ur] = np.arange(nt)
    nnpos = np.full(p.m, -1, dtype=np.int32)
    nnpos[nn] = np.arange(m)
    usem = rupos[rr] >= 0
    if np.any(wpos[cc[usem]] >= 0):
        return None
    mr, mi = rupos[rr[usem]], basepos[cc[usem]]
    mv = -aa[usem] / b[vr][mr]
    mp = np.r_[0, np.cumsum(np.bincount(mr, minlength=nt))]
    usec = (nnpos[rr] >= 0) & (wpos[cc] >= 0)
    cr, ci = nnpos[rr[usec]], wpos[cc[usec]]
    cv = aa[usec] / kval[ci]
    if np.any(cv < -1e-13) or np.any(q[wc] < 0):
        return None
    cv = np.maximum(cv, 0)
    cp = np.r_[0, np.cumsum(np.bincount(cr, minlength=m))]
    B = np.zeros((m, n))
    useb = (nnpos[rr] >= 0) & (basepos[cc] >= 0)
    np.add.at(B, (nnpos[rr[useb]], basepos[cc[useb]]), aa[useb])
    qo, qb = q[wc] / kval, q[base].copy()
    rhs = b[nn] + np.bincount(cr, weights=cv * b[wr][ci], minlength=m)
    theta0, yc = b[ur] / b[vr], b[vr]
    colnorm = np.zeros(n)
    np.maximum.at(colnorm, mi, np.abs(mv))
    colscale = 1 / np.maximum.reduce((colnorm, np.max(np.abs(B), axis=0, initial=0),
                                     np.abs(qb), np.full(n, 1e-6)))
    mv *= colscale[mi]
    B *= colscale[None, :]
    qb *= colscale

    def M(v):
        return np.bincount(mr, weights=mv * v[mi], minlength=nt)

    def MT(v):
        return np.bincount(mi, weights=mv * v[mr], minlength=n)

    def C(v):
        return np.bincount(cr, weights=cv * v[ci], minlength=m)

    def CT(v):
        return np.bincount(ci, weights=cv * v[cr], minlength=nt)

    deriv = Derivatives(mp, mi, mv, cp, ci, cv, B)
    _, gram = deriv(np.ones(nt), np.ones(nt))
    gs = 1 / np.sqrt(np.maximum(np.abs(gram.diagonal()), 1e-20))
    GG = gram * gs[:, None] * gs[None, :]
    GG.flat[::n + 1] += 1e-9
    v = gs * Cholesky(GG).solve(-gs * MT(theta0))

    def exponential(v):
        theta = theta0 + M(v)
        if np.max(theta, initial=-np.inf) > 700:
            raise FloatingPointError("GP exponential overflow")
        return theta, yc * np.exp(np.maximum(theta, -745))

    theta, ev = exponential(v)
    rowscale = 1 / np.maximum.reduce((np.ones(m), np.abs(rhs), np.abs(B @ v), np.abs(C(ev))))
    cv *= rowscale[cr]
    B *= rowscale[:, None]
    rhs *= rowscale
    deriv = Derivatives(mp, mi, mv, cp, ci, cv, B)
    s = np.maximum(1, rhs - B @ v - C(ev))
    z = 1 / s
    begin = time.monotonic()
    costscale = max(1, infnorm(qb))
    log("fast gp", "dimensions", n, m, nt)
    native_evaluation = True

    def evaluate(v, s, z, target=0.):
        if native_evaluation:
            return deriv.evaluate(v, s, z, qb, qo, theta0, yc, rhs, target)
        theta, ev = exponential(v)
        weight = qo + CT(z)
        rd = qb + MT(ev * weight) + B.T @ z
        rp = B @ v + C(ev) - rhs + s
        return theta, ev, weight, rd, rp, s * z - target

    def result(v, z, theta, ev):
        xr = np.zeros(len(free))
        xr[base], xr[wc] = colscale * v, (ev - b[wr]) / kval
        x = offset + np.bincount(tr, weights=tv * xr[tc], minlength=p.n)
        zo = np.zeros(p.m)
        zo[nn] = rowscale * z
        zw = qo + CT(z)
        et = ev / yc
        zo[ur], zo[vr], zo[wr] = -et * zw, (theta - 1) * et * zw, zw
        grad = p.q + p.mul("A", zo, True)
        zo[pivrows] = -grad[pivcols] / coeff
        return p.result(x, zo)

    def recession():
        from fast import core, engine, pass_raw
        from minisparse import CSC
        br, bc = np.nonzero(B)
        A = CSC.coo(nt + m, n, np.r_[mr, nt + br],
                    np.r_[mi, bc], np.r_[mv, B[br, bc]])
        mod, h = core(), engine()
        h.setOptionValue("primal_feasibility_tolerance", 1e-10)
        h.setOptionValue("dual_feasibility_tolerance", 1e-10)
        h.setOptionValue("time_limit", 2.)
        pass_raw(h, n, nt + m, qb, np.full(nt + m, -mod.kHighsInf),
                 np.zeros(nt + m), A.data, A.rows, A.ptr,
                 col_lower=-np.ones(n), col_upper=np.ones(n))
        h.run()
        solution = h.getSolution()
        if len(solution.col_value) != n:
            return None
        xr = np.zeros(len(free))
        xr[base] = colscale * np.asarray(solution.col_value)
        x = np.bincount(tr, weights=tv * xr[tc], minlength=p.n)
        r = {"status": "dual_infeasible", "x": x, "s": -p.mul("A", x)}
        return r if p.valid(r, tol=8e-9) else None

    def polish(v, z, full=False):
        from fast import engine, core, pass_raw
        theta, ev = exponential(v)
        J, H = deriv(ev, qo + CT(z))
        grad = qb + MT(ev * qo)
        f = B @ v + C(ev) - rhs
        mod, h = core(), engine()
        h.setOptionValue("primal_feasibility_tolerance", 1e-10)
        h.setOptionValue("dual_feasibility_tolerance", 1e-10)
        h.setOptionValue("time_limit", 1.)
        values = J.ravel()
        indices = np.tile(np.arange(n, dtype=np.int32), m)
        pointers = np.arange(0, n * m + 1, n, dtype=np.int32)
        pass_raw(h, m, n, -f / max(infnorm(f), 1e-15),
                 -grad, -grad, values, indices, pointers,
                 col_lower=np.zeros(m),
                 col_upper=np.full(m, 1e6 * max(1, infnorm(z))))
        h.run()
        solution = h.getSolution()
        if len(solution.col_value) == m:
            zp = np.maximum(np.asarray(solution.col_value), 0)
            r = result(v, zp, theta, ev)
            if p.valid(r, tol=8e-9):
                return r
        active = np.flatnonzero(z > max(1e-9, 1e-7 * infnorm(z)))
        for face in range(8 if full else 1):
            if not len(active):
                return None
            theta, ev = exponential(v)
            J, H = deriv(ev, qo + CT(z))
            grad = qb + MT(ev * qo)
            zp = np.zeros(m)
            zp[active] = np.linalg.lstsq(J[active].T, -grad, rcond=1e-13)[0]
            if np.min(zp) >= -1e-10 * max(1, infnorm(zp)):
                r = result(v, np.maximum(zp, 0), theta, ev)
                if p.valid(r, tol=8e-9):
                    return r
            if not full:
                return None
            Ja = J[active]
            K = np.block([[H, Ja.T], [Ja, np.zeros((len(active), len(active)))]])
            f = B @ v + C(ev) - rhs
            krhs = -np.r_[grad + J.T @ z, f[active]]
            ksc = 1 / np.sqrt(np.maximum(np.max(np.abs(K), axis=1), 1e-12))
            du = ksc * np.linalg.lstsq(K * ksc[:, None] * ksc[None, :],
                                     ksc * krhs, rcond=1e-13)[0]
            zn = np.zeros(m)
            zn[active] = z[active] + du[n:]
            neg = zn[active] < -1e-9 * max(1, infnorm(zn))
            if np.any(neg):
                active = active[~neg]
                z = np.maximum(zn, 0)
                continue
            v += du[:n]
            z = np.maximum(zn, 0)
        return None

    last_polish = -100
    checked_recession = False
    for it in range(maxiter):
        theta, ev, weight, rd, rp, _ = evaluate(v, s, z)
        mu = float(s @ z) / max(1, m)
        obj = float(qb @ v + qo @ (ev - b[wr]))
        if native_evaluation and mu * m < 1e-5:
            native_evaluation = False
            theta, ev, weight, rd, rp, _ = evaluate(v, s, z)
        if it % 10 == 0:
            log("fast gp", it, infnorm(rd), infnorm(rp), mu * m, obj,
                time.monotonic() - begin)
        if (infnorm(rd) < 5e-8 * costscale and infnorm(rp) < 5e-9
                and mu * m < 8e-9 * max(1, abs(obj))):
            r = result(v, z, theta, ev)
            if p.valid(r, tol=8e-9):
                return r
        if it > 5 and infnorm(z) > 1e8:
            r = result(v, z, theta, ev)
            cert = {"status": "primal_infeasible", "z": r["z"]}
            if p.valid(cert):
                return cert
        if mu * m < 1e-7 * max(1, abs(obj)) and it - last_polish >= 4:
            last_polish = it
            r = polish(v.copy(), z.copy(), full=it > 32)
            if r is not None:
                return r
        if not np.isfinite(mu) or mu <= 0:
            break
        if obj < 0 and mu * m < 1e-5 and not checked_recession:
            checked_recession = True
            ray = recession()
            if ray is not None:
                return ray
        try:
            J, H = deriv(ev, weight)
            w = z / s
            G = H + J.T @ (w[:, None] * J)
            dscl = 1 / np.sqrt(np.maximum(np.abs(G.diagonal()), 1e-30))
            GG = G * dscl[:, None] * dscl[None, :]
            GG.flat[::n + 1] += 1e-13
            factor = Cholesky(GG)

            def linear(rhs):
                sol = dscl * factor.solve(dscl * rhs)
                for _ in range(3):
                    rr = rhs - G @ sol
                    if infnorm(rr) < 1e-12 * max(1, infnorm(rhs)):
                        break
                    sol += dscl * factor.solve(dscl * rr)
                return sol

            def direction(center):
                dv = linear(-rd - J.T @ (center / s + w * rp))
                ds = -rp - J @ dv
                dz = center / s - w * ds
                return dv, ds, dz

            da = direction(-s * z)
            ap, az = maxstep(s, da[1]), maxstep(z, da[2])
            mua = float((s + ap * da[1]) @ (z + az * da[2])) / max(1, m)
            sigma = min(.5, max(.01, max(0., mua / mu) ** 3))
            target_floor = 1e-11 * max(1, abs(obj)) / max(1, m)
            sigma = max(sigma, min(5., target_floor / mu))
            dv, ds, dz = direction(-s * z + sigma * mu - da[1] * da[2])
            drscale = max(costscale, infnorm(qb + MT(ev * weight)), infnorm(B.T @ z), 1)
            rc = s * z - sigma * mu
            slope = (-np.sum((rd / drscale) ** 2) - np.sum(rp ** 2) -
                     float(rc @ (rc + da[1] * da[2])) / max(mu, 1e-10) ** 2)
            for attempt in range(1 if slope >= 0 else 0, 2):
                if attempt:
                    sigma = max(.2, min(5., target_floor / mu))
                    dv, ds, dz = direction(-s * z + sigma * mu)
                alpha = .99 * min(maxstep(s, ds), maxstep(z, dz))
                target = sigma * mu
                old = np.linalg.norm(np.r_[rd / drscale, rp,
                                             (s * z - target) / max(mu, 1e-10)])
                for ls in range(24 if attempt == 0 else 50):
                    vn, sn, zn = v + alpha * dv, s + alpha * ds, z + alpha * dz
                    try:
                        en = evaluate(vn, sn, zn, target)
                        merit = np.linalg.norm(np.r_[en[3] / drscale, en[4],
                                                     en[5] / max(mu, 1e-10)])
                        if merit <= old * (1 - 1e-4 * alpha):
                            break
                    except FloatingPointError:
                        pass
                    alpha *= .5
                if alpha > 1e-7:
                    break
            if alpha < 1e-14:
                break
            v, s, z = vn, sn, zn
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            break
    return None
