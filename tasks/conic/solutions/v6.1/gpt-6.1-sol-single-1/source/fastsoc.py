"""Sparse primal-dual SOCP Newton without high-level sparse imports."""

import time
import os
import numpy as np

from minisparse import CSC, block_matrix


def norm(v):
    return float(np.max(np.abs(v), initial=0))


def nnstep(v, dv):
    use = dv < 0
    return min(1., float(np.min(-v[use] / dv[use]))) if np.any(use) else 1.


def jordan(a, b):
    v = a * b[:, :1] + b * a[:, :1]
    v[:, 0] = np.sum(a * b, axis=1)
    return v


class Cones:
    def __init__(self, p, socrows):
        pos = np.full(p.m, -1, np.int32)
        pos[socrows] = np.arange(len(socrows))
        grouped = {}
        for t, d, lo, hi in p.blocks:
            if t == "soc":
                grouped.setdefault(d, []).append(pos[lo])
        self.groups = {d: np.array(starts)[:, None] + np.arange(d)
                       for d, starts in grouped.items()}
        self.nu = sum(len(ix) for ix in self.groups.values())
        rr, cc, source = [], [], []
        for d, ix in self.groups.items():
            rr.extend((ix.ravel(), np.repeat(ix[:, 0], d - 1), ix[:, 1:].ravel()))
            cc.extend((ix.ravel(), ix[:, 1:].ravel(), np.repeat(ix[:, 0], d - 1)))
            source.extend((np.repeat(ix[:, 0], d), ix[:, 1:].ravel(), ix[:, 1:].ravel()))
        if not rr:
            self.rows = self.source = np.array([], dtype=np.int32)
            self.ptr = np.zeros(1, dtype=np.int32)
            self.size = 0
            return
        rr, cc, source = np.concatenate(rr), np.concatenate(cc), np.concatenate(source)
        order = np.lexsort((rr, cc))
        self.rows, self.source = rr[order].astype(np.int32), source[order]
        self.ptr = np.r_[0, np.cumsum(np.bincount(cc, minlength=len(socrows)))].astype(np.int32)
        self.size = len(socrows)

    def matrix(self, v):
        return CSC(self.size, self.size, v[self.source], self.rows, self.ptr)

    def initial(self, value, margin=1., mu=1.):
        s, z = value.copy(), np.zeros_like(value)
        for ix in self.groups.values():
            v = s[ix]
            r = np.linalg.norm(v[:, 1:], axis=1)
            v[:, 0] = np.maximum(np.abs(v[:, 0]), r) + margin
            det = (v[:, 0] - r) * (v[:, 0] + r)
            zi = -v.copy()
            zi[:, 0] = v[:, 0]
            zi *= (mu / det)[:, None]
            s[ix], z[ix] = v, zi
        return s, z

    def center(self, s, z, mu, sigma, affine=None):
        rc = np.zeros_like(s)
        for ix in self.groups.values():
            v = -jordan(s[ix], z[ix])
            v[:, 0] += sigma * mu
            if affine is not None:
                v -= jordan(affine[0][ix], affine[1][ix])
            rc[ix] = v
        return rc

    def step(self, v, dv):
        alpha = 1.
        for ix in self.groups.values():
            a, da = v[ix], dv[ix]
            r = np.linalg.norm(a[:, 1:], axis=1)
            cc = np.maximum((a[:, 0] - r) * (a[:, 0] + r), 1e-300)
            aa = da[:, 0] ** 2 - np.sum(da[:, 1:] ** 2, axis=1)
            bb = 2 * (a[:, 0] * da[:, 0] - np.sum(a[:, 1:] * da[:, 1:], axis=1))
            disc = bb ** 2 - 4 * aa * cc
            den = -bb + np.sqrt(np.maximum(disc, 0))
            use = (disc > 0) & (den > 0)
            if np.any(use):
                alpha = min(alpha, float(np.min(2 * cc[use] / den[use])))
            alpha = min(alpha, nnstep(a[:, 0], da[:, 0]))
        return max(0., min(1., alpha))


def solve(p, maxiter=100):
    from fast import log, engine, core, pass_raw, phase1, Raw, remaining
    if any(t not in ("zero", "nonnegative", "soc") for t, _, _, _ in p.blocks):
        return None
    ad, ai, ap, ac = p.A
    pd, pi, pp, pc = p.P
    originalA = CSC(p.m, p.n, ad.copy(), ai.copy(), ap.copy()).canonical()
    originalP = CSC(p.n, p.n, pd.copy(), pi.copy(), pp.copy()).canonical()
    eq, ni = np.flatnonzero(p.eq), np.flatnonzero(p.nn)
    si = np.flatnonzero(~(p.eq | p.nn))
    cones = Cones(p, si)
    warm = None
    if os.environ.get("SOLVER_SOC_WARM") == "1" and p.n > 10 and len(p.P[0]) == 0:
        li = np.flatnonzero(p.eq | p.nn)
        L = originalA.select_rows(li)
        h, mod = engine(), core()
        pass_raw(h, p.n, len(li), p.q, np.where(p.eq[li], p.b[li], -mod.kHighsInf),
                 p.b[li], L.data, L.rows, L.ptr)
        h.run()
        st, sol = h.getModelStatus(), h.getSolution()
        log("soc linear relaxation", st)
        if st in (mod.HighsModelStatus.kInfeasible, mod.HighsModelStatus.kUnknown):
            lp = object.__new__(Raw)
            lp.n, lp.m, lp.q, lp.b = p.n, len(li), p.q, p.b[li]
            lp.eq, lp.nn, lp.nonlinear = p.eq[li], p.nn[li], False
            lp.A = (L.data, L.rows, L.ptr, L.cols)
            lp.P = p.P
            cert = phase1(lp)
            if cert is not None:
                zo = np.zeros(p.m)
                zo[li] = cert["z"]
                r = {"status": "primal_infeasible", "z": zo}
                if p.valid(r):
                    return r
        if st == mod.HighsModelStatus.kOptimal and len(sol.col_value) == p.n:
            warm = np.asarray(sol.col_value), np.zeros(p.m)
            warm[1][li] = -np.asarray(sol.row_dual)
            initial = p.result(warm[0], warm[1])
            if p.valid(initial):
                return initial
    A, P = originalA, originalP
    q, b = p.q.copy(), p.b.copy()
    D, E = np.ones(p.n), np.ones(p.m)
    for _ in range(5):
        dc = 1 / np.sqrt(np.maximum.reduce((A.maxcol(), P.maxcol(), np.full(p.n, 1e-6))))
        dc = np.clip(dc, 1e-3, 1e3)
        A, P = A.scale(col=dc), P.scale(row=dc, col=dc)
        q *= dc
        D *= dc
        rowmax = A.maxrow()
        dr = 1 / np.sqrt(np.maximum(rowmax, 1e-6))
        dr = np.clip(dr, 1e-3, 1e3)
        for ix in cones.groups.values():
            global_ix = si[ix]
            dr[global_ix] = (1 / np.sqrt(np.maximum(np.max(rowmax[global_ix], axis=1), 1e-6)))[:, None]
        A = A.scale(row=dr)
        b *= dr
        E *= dr
    cost = 1 / max(norm(q), norm(P.data), 1e-3)
    P = P.scale(scalar=cost)
    q *= cost
    F, N, S = A.select_rows(eq), A.select_rows(ni), A.select_rows(si)
    Ft, Nt, St = F.T, N.T, S.T
    n, k, ns = p.n, len(eq), len(si)
    x, y = np.zeros(n), np.zeros(k)
    sn = np.maximum(1., np.abs(b[ni]))
    zn = 1 / sn
    ss, zs = cones.initial(b[si])
    if warm is not None:
        x = warm[0] / D
        y = warm[1][eq] * cost / E[eq]
        sn = np.maximum(b[ni] - N.mul(x), .1)
        zn = np.maximum(warm[1][ni] * cost / E[ni], .01 / sn)
        ss, zs = cones.initial(b[si] - S.mul(x), margin=.1, mu=.01)
    nu = len(ni) + cones.nu
    begin = time.monotonic()
    best = np.inf
    for it in range(maxiter):
        rd = P.mul(x) + q + Ft.mul(y) + Nt.mul(zn) + St.mul(zs)
        re, rn, rs = F.mul(x) - b[eq], N.mul(x) + sn - b[ni], S.mul(x) + ss - b[si]
        mu = float(sn @ zn + ss @ zs) / max(1, nu)
        if it >= 2:
            zo, so = np.zeros(p.m), np.zeros(p.m)
            zo[eq], zo[ni], zo[si] = y * E[eq] / cost, zn * E[ni] / cost, zs * E[si] / cost
            so[ni], so[si] = sn / E[ni], ss / E[si]
            r = {"status": "solved", "x": D * x, "s": so, "z": zo}
            if p.valid(r, tol=8e-9):
                log("fast soc solved", it, time.monotonic() - begin)
                return r
            if norm(zo) > 1e5:
                cert = {"status": "primal_infeasible", "z": zo}
                if p.valid(cert):
                    return cert
            if norm(D * x) > 1e5 and norm(p.q):
                xo = D * x
                cert = {"status": "dual_infeasible", "x": xo, "s": -p.mul("A", xo)}
                if p.valid(cert):
                    return cert
        if not np.isfinite(mu) or mu <= 0 or time.monotonic() - begin > 180 or remaining() < 1.:
            break
        if it % 10 == 0:
            log("fast soc", it, norm(rd), max(norm(re), norm(rn), norm(rs)), mu,
                time.monotonic() - begin)
        try:
            Ls, Lz = cones.matrix(ss), cones.matrix(zs)
            w = zn / sn
            G = P.add(Nt.matmul(N.scale(row=w)))
            LS = Lz.matmul(S).scale(scalar=-1)
            K = block_matrix([[G, Ft, St], [F, None, None], [LS, None, Ls]],
                             [n, k, ns], [n, k, ns])
            rsc = 1 / np.sqrt(np.maximum(K.maxrow(), 1e-12))
            csc = 1 / np.sqrt(np.maximum(K.maxcol(), 1e-12))
            reg = np.r_[np.full(n, 1e-12), np.full(k, -1e-12), np.full(ns, 1e-12)]
            KR = K.scale(row=rsc, col=csc).add(CSC.diagonal(reg))
            factor = KR.factor(order="MMD_AT_PLUS_A" if ns == 0 else "COLAMD",
                               pivot=0. if ns == 0 else .1)

            def direction(sigma, da=None):
                rc_n = -sn * zn + sigma * mu
                if da is not None:
                    rc_n -= da[2] * da[3]
                center_n = rc_n / sn
                rc_s = cones.center(ss, zs, mu, sigma,
                                    None if da is None else (da[4], da[5]))
                rhs = np.r_[-rd - Nt.mul(center_n + w * rn), -re,
                            rc_s + Lz.mul(rs)]
                rscaled = rsc * rhs
                u = csc * factor.solve(rscaled)
                for _ in range(5):
                    rr = rsc * (rhs - K.mul(u))
                    if norm(rr / (1 + np.abs(rscaled))) < 1e-13:
                        break
                    du = csc * factor.solve(rr)
                    if norm(du) > 1e5 * max(1, norm(u)):
                        break
                    u += du
                dx, dy, dzs = u[:n], u[n:n + k], u[n + k:]
                dsn, dss = -rn - N.mul(dx), -rs - S.mul(dx)
                dzn = center_n - w * dsn
                return dx, dy, dsn, dzn, dss, dzs

            da = direction(0)
            ap = min(nnstep(sn, da[2]), cones.step(ss, da[4]))
            az = min(nnstep(zn, da[3]), cones.step(zs, da[5]))
            mua = float((sn + ap * da[2]) @ (zn + az * da[3]) +
                        (ss + ap * da[4]) @ (zs + az * da[5])) / max(1, nu)
            sigma = min(1., max(0., mua / mu) ** 3)
            dx, dy, dsn, dzn, dss, dzs = direction(sigma, da)
            ap = .99 * min(nnstep(sn, dsn), cones.step(ss, dss))
            az = .99 * min(nnstep(zn, dzn), cones.step(zs, dzs))
            if min(ap, az) < 1e-13:
                break
            x += ap * dx
            sn += ap * dsn
            ss += ap * dss
            y += az * dy
            zn += az * dzn
            zs += az * dzs
        except (ValueError, RuntimeError, FloatingPointError, MemoryError):
            break
    return None
