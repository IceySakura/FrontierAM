#!/usr/bin/env python3
"""Budgeted black-box cubature on [0,1]^d using only the JSON oracle."""
import json
import math
import sys

import numpy as np
from numpy.polynomial.chebyshev import chebvander, cheb2poly
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BarycentricInterpolator
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import qmc


class Oracle:
    def __init__(self):
        init = json.loads(sys.stdin.readline())
        if init.get("type") != "init":
            raise SystemExit("missing init")
        self.d = int(init["dimension"])
        self.budget = int(init["budget"])
        self.seed = int(init.get("seed", 0))
        self.used = 0
        self.cache = {}

    def query(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        n = len(points)
        if n == 0:
            return np.empty(0)
        if self.used + n > self.budget:
            raise RuntimeError("budget exceeded")
        out = np.empty(n)
        need = []
        need_idx = []
        for i, p in enumerate(points):
            key = tuple(p)
            if key in self.cache:
                out[i] = self.cache[key]
            else:
                need.append(p)
                need_idx.append(i)
        for start in range(0, len(need), 512):
            chunk = np.asarray(need[start:start + 512])
            sys.stdout.write(json.dumps({"type": "query", "points": chunk.tolist()}) + "\n")
            sys.stdout.flush()
            ans = json.loads(sys.stdin.readline())
            vals = np.asarray(ans.get("values", []), dtype=float)
            if len(vals) != len(chunk) or not np.all(np.isfinite(vals)):
                raise SystemExit("invalid oracle values")
            self.used += len(chunk)
            for p, v in zip(chunk, vals):
                self.cache[tuple(p)] = float(v)
            for i, v in zip(need_idx[start:start + 512], vals):
                out[i] = v
        return out


def cc_nodes(m):
    k = np.arange(m)
    return (1.0 - np.cos(np.pi * k / (m - 1))) / 2.0


def gl_panel(a, b, m):
    if b - a <= 1e-12:
        return np.empty(0), np.empty(0)
    z, w = leggauss(m)
    return a + (z + 1.0) * 0.5 * (b - a), w * 0.5 * (b - a)


def axis_break(x, logy):
    i = int(np.argmax(logy))
    if i == 0 or i == len(x) - 1:
        return float(x[i]), False, i
    left = np.arange(max(0, i - 6), max(0, i - 1))
    right = np.arange(min(len(x) - 1, i + 2), min(len(x), i + 8))
    u = float(x[i])
    cusp = False
    if len(left) >= 3 and len(right) >= 3:
        al, bl = np.polyfit(x[left], logy[left], 1)
        ar, br = np.polyfit(x[right], logy[right], 1)
        if al > 0.0 and ar < 0.0:
            uu = (br - bl) / (al - ar)
            if x[i - 1] < uu < x[i + 1]:
                rl = np.sqrt(np.mean((al * x[left] + bl - logy[left]) ** 2))
                rr = np.sqrt(np.mean((ar * x[right] + br - logy[right]) ** 2))
                span = max(1e-12, np.max(logy) - np.min(logy))
                if max(rl, rr) < 1e-5 * max(1.0, span):
                    u = float(uu)
                    cusp = True
    if not cusp:
        a, b, c = np.polyfit(x[i - 1:i + 2], logy[i - 1:i + 2], 2)
        if a < 0.0:
            uu = -b / (2.0 * a)
            if x[i - 1] < uu < x[i + 1]:
                u = float(uu)
    return float(np.clip(u, 0.0, 1.0)), cusp, i


class MultiLog:
    def __init__(self, knots, xs, ys):
        self.knots = np.asarray(knots)
        self.interps = [BarycentricInterpolator(x, y) for x, y in zip(xs, ys) if len(x) >= 2]
        self.n = len(self.interps)

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        idx = np.searchsorted(self.knots, t, side="right") - 1
        idx = np.clip(idx, 0, self.n - 1)
        out = np.empty_like(t)
        for k in range(self.n):
            m = idx == k
            if np.any(m):
                out[m] = self.interps[k](t[m])
        return out


def separable_candidate(orc, X, y, sob):
    d = orc.d
    if np.max(y) <= 0.0 or np.any(y < 0.0):
        return None
    m0 = 65 if d >= 5 else 129
    if orc.used + d * m0 > orc.budget:
        return None
    base = X[int(np.argmax(y))].copy()
    fb = float(np.max(y))
    logfb = math.log(fb)
    xc = cc_nodes(m0)
    pts = []
    for j in range(d):
        p = np.tile(base, (m0, 1))
        p[:, j] = xc
        pts.append(p)
    vals = orc.query(np.vstack(pts))
    A = vals.reshape(d, m0)
    if np.any(A < 0.0) or not np.all(np.isfinite(A)):
        return None
    floors = np.maximum(1e-300, np.max(A, axis=1) * 1e-14)
    logA = np.log(np.maximum(A, floors[:, None]))
    jump = 0.0
    for j in range(d):
        df = np.sort(np.abs(np.diff(A[j])))[::-1]
        if len(df) >= 2 and df[0] > 1e-12:
            jump = max(jump, float(df[0] / (df[0] + df[1])))
    orc.axis_jump = jump
    if jump > 0.85:
        return None
    breaks = []
    coarse_interp = []
    for j in range(d):
        u, cusp, imax = axis_break(xc, logA[j])
        breaks.append((u, cusp, imax))
        if cusp:
            li = np.arange(max(0, imax - 8), imax)
            ri = np.arange(imax + 1, min(m0, imax + 9))
            if len(li) >= 2 and len(ri) >= 2:
                al, bl = np.polyfit(xc[li], logA[j, li], 1)
                ar, br = np.polyfit(xc[ri], logA[j, ri], 1)
                coarse_interp.append((al, bl, ar, br))
            else:
                coarse_interp.append(BarycentricInterpolator(xc, logA[j]))
        else:
            coarse_interp.append(BarycentricInterpolator(xc, logA[j]))

    def coarse_logpred(P):
        out = np.full(len(P), logfb)
        for j in range(d):
            it = coarse_interp[j]
            t = P[:, j]
            if isinstance(it, tuple):
                al, bl, ar, br = it
                v = np.where(t <= breaks[j][0], al * t + bl, ar * t + br)
            else:
                v = it(t)
            bj = base[j]
            if isinstance(it, tuple):
                al, bl, ar, br = it
                vb = al * bj + bl if bj <= breaks[j][0] else ar * bj + br
            else:
                vb = it(bj)
            out += v - vb
        return out

    pred = np.exp(coarse_logpred(X))
    rel = np.sqrt(np.mean((pred - y) ** 2)) / max(1e-12, np.sqrt(np.mean(y ** 2)))
    if rel > (0.30 if d <= 3 else 0.15):
        return None
    if np.any(y == 0.0) and rel > 0.01:
        return None
    if np.any(A == 0.0) and rel > 0.01:
        return None

    # Refine smooth peak locations by a short golden-section search.
    golden = 0.6180339887498949
    for j in range(d):
        u, cusp, imax = breaks[j]
        if cusp or imax == 0 or imax == m0 - 1:
            continue
        a = xc[imax - 1]
        b = xc[imax + 1]
        if b - a < 1e-8:
            continue
        c = b - golden * (b - a)
        dd = a + golden * (b - a)
        def fv(t):
            p = base.copy()
            p[j] = t
            return math.log(max(float(orc.query(p)[0]), floors[j]))
        fc = fv(c)
        fd = fv(dd)
        for _ in range(12):
            if fc < fd:
                a, c, fc = c, dd, fd
                dd = a + golden * (b - a)
                fd = fv(dd)
            else:
                b, dd, fd = dd, c, fc
                c = b - golden * (b - a)
                fc = fv(c)
        breaks[j] = (float(np.clip((a + b) / 2.0, 0.0, 1.0)), False, imax)

    m_inner = 48 if d <= 3 else 32
    m_outer = 32 if d <= 3 else 20
    panels = []
    pts = []
    for j in range(d):
        u = breaks[j][0]
        # Estimate a local width and put extra Gauss nodes around the peak/cusp.
        width = 0.1
        imax = breaks[j][2]
        if 0 < imax < m0 - 1:
            aa, bb, cc = np.polyfit(xc[imax - 1:imax + 2], logA[j, imax - 1:imax + 2], 2)
            if aa < -1e-8:
                width = 1.0 / math.sqrt(-aa)
            elif breaks[j][1]:
                drop = max(1e-6, logA[j, imax] - min(logA[j, imax - 1], logA[j, imax + 1]))
                dx = min(xc[imax] - xc[imax - 1], xc[imax + 1] - xc[imax])
                width = dx / drop
        r = float(np.clip(3.0 * width, 0.01, 0.25))
        knots = [0.0]
        for z in (u - r, u, u + r):
            z = float(np.clip(z, 0.0, 1.0))
            if z - knots[-1] > 1e-12:
                knots.append(z)
        if 1.0 - knots[-1] > 1e-12:
            knots.append(1.0)
        xs = []
        ws = []
        xall = []
        wall = []
        for k in range(len(knots) - 1):
            a, b = knots[k], knots[k + 1]
            near = abs(a - u) < 1e-12 or abs(b - u) < 1e-12
            m = m_inner if near else m_outer
            x, w = gl_panel(a, b, m)
            if len(x):
                xs.append(x)
                ws.append(w)
                xall.append(x)
                wall.append(w)
        x = np.concatenate(xall)
        w = np.concatenate(wall)
        p = np.tile(base, (len(x), 1))
        p[:, j] = x
        panels.append((knots, xs, ws, w))
        pts.append(p)
    vals = orc.query(np.vstack(pts))
    offset = 0
    interps = []
    logI = []
    for j in range(d):
        knots, xs, ws, w = panels[j]
        yj = vals[offset:offset + len(w)]
        offset += len(w)
        Ij = float(np.dot(w, yj))
        if Ij <= 0.0 or not math.isfinite(Ij):
            return None
        logI.append(math.log(Ij))
        ly = np.log(np.maximum(yj, floors[j]))
        ys = []
        off = 0
        for x in xs:
            ys.append(ly[off:off + len(x)])
            off += len(x)
        interps.append(MultiLog(knots, xs, ys))
    logsep = float(np.sum(logI) - (d - 1) * logfb)
    if not math.isfinite(logsep):
        return None
    Ab = np.array([interps[j](base[j]) for j in range(d)])

    def logpred(P):
        out = np.full(len(P), logfb)
        for j in range(d):
            out += interps[j](P[:, j]) - Ab[j]
        return out

    pred = np.exp(logpred(X))
    rel = np.sqrt(np.mean((pred - y) ** 2)) / max(1e-12, np.sqrt(np.mean(y ** 2)))
    if rel < 0.03:
        return float(math.exp(logsep))
    return None


def ridge_moments(alpha, K):
    dp = np.zeros(K + 1)
    dp[0] = 1.0
    for a in alpha:
        nd = np.zeros(K + 1)
        for k in range(K + 1):
            s = 0.0
            for r in range(0, k + 1, 2):
                s += math.comb(k, r) * dp[k - r] * a ** r / (r + 1)
            nd[k] = s
        dp = nd
    return dp


def ridge_candidate(orc, X, y):
    d = orc.d
    n = len(X)
    if n < 32 or orc.used + 16 * d + 32 > orc.budget:
        return None
    interior = np.where((X.min(axis=1) > 0.05) & (X.max(axis=1) < 0.95))[0]
    nb = 8
    if len(interior) >= nb:
        bases = interior[np.linspace(0, len(interior) - 1, nb).astype(int)]
    else:
        bases = np.arange(min(nb, n))
    h = 1e-7
    G = []
    for i in bases:
        b = X[i]
        g = np.empty(d)
        for j in range(d):
            p = b.copy()
            m = b.copy()
            if b[j] + h <= 1.0 and b[j] - h >= 0.0:
                p[j] += h
                m[j] -= h
                g[j] = (orc.query(p)[0] - orc.query(m)[0]) / (2 * h)
            elif b[j] + h <= 1.0:
                p[j] += h
                g[j] = (orc.query(p)[0] - orc.query(b)[0]) / h
            else:
                m[j] -= h
                g[j] = (orc.query(b)[0] - orc.query(m)[0]) / h
        G.append(g)
    G = np.asarray(G)
    norms = np.linalg.norm(G, axis=1)
    keep = norms > max(1e-10, 0.02 * np.max(norms))
    if np.sum(keep) < 2:
        return None
    Gn = G[keep] / norms[keep, None]
    M = Gn.T @ Gn
    evals, evecs = np.linalg.eigh(M)
    w = evecs[:, -1]
    conc = evals[-1] / max(1e-12, evals.sum())
    if conc < 0.80:
        return None
    sw = float(np.sum(np.abs(w)))
    if sw < 1e-12:
        return None
    t0 = float(np.sum(w)) / 2.0
    sc = sw / 2.0
    s = (X @ w - t0) / sc
    alpha = w / sw
    rms = max(1e-12, np.sqrt(np.mean(y ** 2)))

    # Pure periodic ridge: fit A cos(k s) + B sin(k s) + C.
    def sin_fit(k):
        V = np.column_stack((np.cos(k * s), np.sin(k * s), np.ones(n)))
        coef = np.linalg.lstsq(V, y, rcond=1e-12)[0]
        r = V @ coef - y
        return float(np.dot(r, r)), coef

    grid = np.linspace(0.05, 600.0, 1200)
    best = None
    for k in grid:
        sse, coef = sin_fit(k)
        if best is None or sse < best[0]:
            best = (sse, float(k), coef)
    sse, k0, coef = best
    step = grid[1] - grid[0]
    lo = max(0.01, k0 - step)
    hi = k0 + step
    res = minimize_scalar(lambda k: sin_fit(k)[0], bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-10})
    if res.success and res.fun < sse:
        k = float(res.x)
        sse, coef = sin_fit(k)
    else:
        k = k0
    rel = math.sqrt(sse / n) / rms
    if rel < 0.1:
        # Refine the ridge direction and frequency by variable projection.
        def obj(par):
            v = par[:d]
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                return 1e30
            v = v / nv
            swv = float(np.sum(np.abs(v)))
            if swv < 1e-12:
                return 1e30
            ss = (X @ v - float(np.sum(v)) / 2.0) / (swv / 2.0)
            kk = abs(par[d])
            V = np.column_stack((np.cos(kk * ss), np.sin(kk * ss), np.ones(n)))
            cc = np.linalg.lstsq(V, y, rcond=1e-12)[0]
            rr = V @ cc - y
            return float(np.dot(rr, rr))
        start = np.concatenate((w, [k]))
        res2 = minimize(obj, start, method="Nelder-Mead",
                        options={"maxiter": 2500, "xatol": 1e-12, "fatol": 1e-15})
        if res2.success and res2.fun < sse:
            w = res2.x[:d]
            w = w / np.linalg.norm(w)
            k = abs(float(res2.x[d]))
            sw = float(np.sum(np.abs(w)))
            t0 = float(np.sum(w)) / 2.0
            sc = sw / 2.0
            s = (X @ w - t0) / sc
            alpha = w / sw
            sse, coef = sin_fit(k)
            rel = math.sqrt(sse / n) / rms
    if rel < 1e-5:
        # E[cos(k s)] = prod sinc(k alpha_j); E[sin(k s)] = 0.
        factor = float(np.prod(np.sinc(k * alpha / np.pi)))
        return float(coef[2] + coef[0] * factor)

    # General smooth ridge: Chebyshev polynomial in the projected coordinate.
    K = min(48, max(8, n // 5))
    V = chebvander(s, K)
    coef = np.linalg.lstsq(V, y, rcond=1e-12)[0]
    r = V @ coef - y
    rel = np.sqrt(np.mean(r ** 2)) / rms
    if rel < 1e-6:
        mom = ridge_moments(alpha, K)
        pw = cheb2poly(coef)
        return float(np.dot(pw, mom[:len(pw)]))
    return None


def main():
    orc = Oracle()
    d = orc.d
    budget = orc.budget
    if budget <= 0:
        sys.stdout.write(json.dumps({"type": "result", "estimate": 0.0}) + "\n")
        return
    sob = qmc.Sobol(d, scramble=True, seed=orc.seed).random(budget)
    n0 = min(256, budget)
    X = sob[:n0]
    y = orc.query(X)
    sob_used = n0
    sob_vals = list(y)

    try:
        est = separable_candidate(orc, X, y, sob)
    except Exception:
        est = None
    if est is not None and math.isfinite(est):
        sys.stdout.write(json.dumps({"type": "result", "estimate": float(est)}) + "\n")
        return

    try:
        est = ridge_candidate(orc, X, y)
    except Exception:
        est = None
    if est is not None and math.isfinite(est):
        sys.stdout.write(json.dumps({"type": "result", "estimate": float(est)}) + "\n")
        return

    # Low-dimensional smooth fallback: a tensor Gauss rule is often stronger
    # than plain QMC when the separable/ridge models do not apply.
    if d <= 3 and np.all(y > 0.0) and getattr(orc, "axis_jump", 0.0) < 0.8:
        rem = budget - orc.used
        n = int(rem ** (1.0 / d))
        if n >= 2 and n ** d <= rem:
            try:
                z, w = leggauss(n)
                z = (z + 1.0) / 2.0
                w = w / 2.0
                grids = np.meshgrid(*([z] * d), indexing="ij")
                pts = np.stack([g.ravel() for g in grids], axis=1)
                wg = np.meshgrid(*([w] * d), indexing="ij")
                weights = np.prod(np.stack(wg), axis=0).ravel()
                vals = orc.query(pts)
                est = float(np.dot(vals, weights))
                if math.isfinite(est):
                    sys.stdout.write(json.dumps({"type": "result", "estimate": est}) + "\n")
                    return
            except Exception:
                pass

    remain = budget - orc.used
    take = min(remain, budget - sob_used)
    if take > 0:
        P = sob[sob_used:sob_used + take]
        vals = orc.query(P)
        sob_vals.extend(vals.tolist())
        sob_used += take
    estimate = float(np.mean(sob_vals)) if sob_vals else 0.0
    if not math.isfinite(estimate):
        estimate = 0.0
    sys.stdout.write(json.dumps({"type": "result", "estimate": estimate}) + "\n")


if __name__ == "__main__":
    main()
