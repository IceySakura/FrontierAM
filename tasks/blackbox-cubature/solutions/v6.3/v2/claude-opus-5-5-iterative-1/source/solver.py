#!/usr/bin/env python3
"""Black-box cubature over [0,1]^d under a fixed point budget (JSON Lines).

Strategy: spend a few points testing for exploitable structure, validate every
structural model on fresh points, and otherwise fall back to strong generic rules.
  * separable (rank-1) or additive integrands (cross-ratio / cross-sum tests)
      -> adaptive 1-D Gauss-Kronrod integrals along slices through an anchor point;
  * ridge integrands g(w.x) (parallel gradients)
      -> direction from high-order finite differences refined by Gauss-Newton, a
         Chebyshev profile along the cube diagonal integrated exactly against the
         density of w.X, plus a digital-net residual correction;
  * piecewise-linear integrands (collinear samples along chords)
      -> kink planes located on chords, exact least-squares model integrated in
         rational arithmetic, accepted only if it reproduces all reference points;
  * otherwise: d=2 p-adaptive Clenshaw-Curtis (pcubature port), d=3/5 tensor
    Gauss-Legendre, d=8 complete scrambled digital net with quadratic control variates.
"""
import json
import math
import sys
import time
import warnings
from fractions import Fraction

import numpy as np


T0 = time.time()


def time_left():
    return 24.0 - (time.time() - T0)


class Oracle:
    """Budget-safe batched access to the function values over stdin/stdout."""

    def __init__(self, dim, budget, fin, fout):
        self.dim = dim
        self.budget = budget
        self.used = 0
        self.fin = fin
        self.fout = fout
        self.xs = []
        self.ys = []

    def remaining(self):
        return self.budget - self.used

    def __call__(self, pts):
        pts = np.asarray(pts, dtype=float).reshape(-1, self.dim)
        n = len(pts)
        if n == 0:
            return np.zeros(0)
        if n > self.remaining():
            raise RuntimeError("budget")
        pts = np.clip(np.nan_to_num(pts, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
        out = np.empty(n)
        for s in range(0, n, 512):
            chunk = pts[s:s + 512]
            self.fout.write(json.dumps({"type": "query", "points": chunk.tolist()}) + "\n")
            self.fout.flush()
            while True:
                line = self.fin.readline()
                if not line:
                    raise EOFError("oracle closed")
                if line.strip():
                    break
            msg = json.loads(line)
            vals = np.asarray(msg.get("values", []), dtype=float).reshape(-1)
            if vals.size != len(chunk):
                raise RuntimeError("bad values")
            self.used += len(chunk)
            out[s:s + len(chunk)] = vals
        self.xs.append(pts)
        self.ys.append(out)
        return out


# ---------------------------------------------------------------------------
# 1-D Gauss-Kronrod (7/15) adaptive integration, driven externally in batches

_XGK = np.array([0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
                 0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
                 0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
                 0.207784955007898467600689403773245, 0.0])
_WGK = np.array([0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
                 0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
                 0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
                 0.204432940075298892414161999234649, 0.209482141084727828012999174891714])
_WG7 = np.array([0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
                 0.381830050505118944950369775488975, 0.417959183673469387755102040816327])
GK_X = np.concatenate([-_XGK[:7], [0.0], _XGK[6::-1]])
GK_WK = np.concatenate([_WGK[:7], [_WGK[7]], _WGK[6::-1]])
GK_WG = np.zeros(15)
GK_WG[[1, 3, 5, 9, 11, 13]] = np.concatenate([_WG7[:3], _WG7[2::-1]])
GK_WG[7] = _WG7[3]


def gk_nodes(a, b):
    return 0.5 * (a + b) + 0.5 * (b - a) * GK_X


class Adapt1D:
    def __init__(self):
        self.iv = []  # [a, b, K, err]

    def add(self, a, b, vals):
        h = 0.5 * (b - a)
        K = h * float(np.dot(GK_WK, vals))
        G = h * float(np.dot(GK_WG, vals))
        self.iv.append([a, b, K, abs(K - G), np.asarray(vals, float).copy()])

    def worst(self):
        return max(range(len(self.iv)), key=lambda i: self.iv[i][3])

    def total(self):
        return math.fsum(v[2] for v in self.iv), math.fsum(v[3] for v in self.iv)


# ---------------------------------------------------------------------------
# rank-1 (separable) integrands


def separability_test(orc, d, rng, npairs=3):
    """Cross-ratio test f(a)f(b) == f(a_S b_~S) f(b_S a_~S); returns (ok, pts, vals).
    Also records whether the cross-sum (additivity) test passed."""
    separability_test.additive = False
    pts = []
    for _ in range(npairs):
        a = rng.uniform(0.02, 0.98, d)
        b = rng.uniform(0.02, 0.98, d)
        pts.append(a)
        pts.append(b)
        for _k in range(2 if d > 2 else 1):
            m = rng.random(d) < 0.5
            if m.all() or not m.any():
                m[rng.integers(d)] = not m[0] if d > 1 else m[0]
                if m.all() or not m.any():
                    m = np.zeros(d, bool)
                    m[0] = True
            pts.append(np.where(m, a, b))
            pts.append(np.where(m, b, a))
    pts = np.array(pts)
    vals = orc(pts)
    ok = True
    i = 0
    per = 2 + 2 * (2 if d > 2 else 1)
    vmax = float(np.max(np.abs(vals))) if len(vals) else 0.0
    if not (math.isfinite(vmax) and vmax > 0.0):
        return False, pts, vals
    nv = vals / vmax  # normalised to avoid overflow in the products
    additive = True
    for _ in range(npairs):
        fa, fb = nv[i], nv[i + 1]
        for k in range((per - 2) // 2):
            ssum = nv[i + 2 + 2 * k] + nv[i + 3 + 2 * k]
            sc = max(abs(fa) + abs(fb), abs(nv[i + 2 + 2 * k]) + abs(nv[i + 3 + 2 * k]), 1e-300)
            if abs((fa + fb) - ssum) > 1e-12 * sc:
                additive = False
        lhs = fa * fb
        for k in range((per - 2) // 2):
            rhs = nv[i + 2 + 2 * k] * nv[i + 3 + 2 * k]
            scale = max(abs(lhs), abs(rhs))
            if not (scale > 1e-250) or abs(lhs - rhs) > 1e-10 * scale:
                ok = False
        i += per
    separability_test.additive = additive and not ok
    return ok, pts, vals


def _gk_bary():
    x = GK_X
    w = np.array([1.0 / np.prod([x[j] - x[k] for k in range(15) if k != j]) for j in range(15)])
    return w / np.max(np.abs(w))


GK_BARY = _gk_bary()


def slice_value(prob, t):
    for iv in prob.iv:
        if iv[0] <= t <= iv[1]:
            a, b, vals = iv[0], iv[1], iv[4]
            z = (2.0 * t - (a + b)) / (b - a)
            return float(bary_eval(GK_X, vals, GK_BARY, np.array([z]))[0])
    return float('nan')


def rank1_integral(orc, d, c, fc, rng, reserve=0, additive=False):
    """Product of adaptive 1-D integrals along slices through the anchor c.

    Returns None when the separable model fails validation on fresh points."""
    probs = [Adapt1D() for _ in range(d)]
    pending = []  # (dim, a, b)
    for i in range(d):
        pending.append((i, 0.0, 0.5))
        pending.append((i, 0.5, 1.0))
    start_rem = orc.remaining()
    validated = False
    nval = 8
    while True:
        if pending:
            pts = []
            for (i, a, b) in pending:
                t = gk_nodes(a, b)
                P = np.repeat(c[None, :], 15, axis=0)
                P[:, i] = t
                pts.append(P)
            vals = orc(np.vstack(pts))
            for k, (i, a, b) in enumerate(pending):
                probs[i].add(a, b, vals[15 * k:15 * (k + 1)])
            pending = []
        tot = [p.total() for p in probs]
        if additive:
            scale_a = max(sum(abs(I) for I, e in tot), abs(fc), 1e-300)
            rel = [e / scale_a for I, e in tot]
        else:
            rel = [e / max(abs(I), 1e-300) for I, e in tot]
        relsum = sum(rel)
        converged = relsum < 1e-14
        checkpoint = (start_rem - orc.remaining()) >= 0.45 * start_rem
        if not validated and (converged or checkpoint or orc.remaining() - reserve < 30 + nval):
            if orc.remaining() < nval:
                return None
            V = rng.uniform(0.0, 1.0, (nval, d))
            fv = orc(V)
            tol = min(0.1, max(1e-5, 1e3 * relsum))
            good = True
            if additive:
                fscale_a = max(abs(fc), float(np.max(np.abs(fv))), 1e-300)
                for k in range(nval):
                    m = -(d - 1) * fc
                    for i in range(d):
                        m += slice_value(probs[i], V[k, i])
                    if not math.isfinite(m) or abs(m - fv[k]) > tol * fscale_a:
                        good = False
                        break
                if not good:
                    return None
                validated = True
            lfc = math.log(abs(fc)) if fc != 0.0 else 0.0
            ltiny = lfc + math.log(1e-13)
            for k in range(0 if additive else nval):
                # compare model and value in log space (products can overflow)
                sgn = math.copysign(1.0, fc) ** (d - 1)
                lm = -(d - 1) * lfc
                zero_model = False
                for i in range(d):
                    sv = slice_value(probs[i], V[k, i])
                    if not math.isfinite(sv):
                        lm = float('nan')
                        break
                    if sv == 0.0:
                        zero_model = True
                        break
                    sgn *= math.copysign(1.0, sv)
                    lm += math.log(abs(sv))
                fk = float(fv[k])
                if not math.isfinite(fk):
                    good = False
                    break
                if zero_model or fk == 0.0:
                    # accept only if both model and value are negligible
                    lf = math.log(abs(fk)) if fk != 0.0 else -float('inf')
                    lmm = -float('inf') if zero_model else lm
                    if not (lf <= ltiny and (lmm <= ltiny)):
                        good = False
                        break
                    continue
                if not math.isfinite(lm):
                    good = False
                    break
                lf = math.log(abs(fk))
                if lm <= ltiny and lf <= ltiny:
                    continue  # both negligible relative to the anchor value
                # |m - f| <= tol |f| + 1e-13 |f(c)|, evaluated in ratio form
                slack = tol + math.exp(max(-700.0, min(700.0, ltiny - lf)))
                if sgn != math.copysign(1.0, fk):
                    ratio = -math.exp(max(-700.0, min(700.0, lm - lf)))
                else:
                    ratio = math.exp(max(-700.0, min(700.0, lm - lf)))
                if abs(ratio - 1.0) > slack:
                    good = False
                    break
            if not good and not additive:
                # slices not yet resolved: confirm separability directly instead (only for
                # continuous-valued integrands; piecewise-constant ones pass cross-ratios by chance)
                if relsum < 1e-8 or orc.remaining() < 60:
                    return None
                allv = np.concatenate(orc.ys) if orc.ys else np.zeros(0)
                if len(allv) == 0 or len(np.unique(allv)) < 0.9 * len(allv):
                    return None
                ok2, _, _ = separability_test(orc, d, rng, npairs=6)
                if not ok2:
                    return None
            validated = True
        if converged:
            break
        avail = (orc.remaining() - reserve) // 30
        if avail <= 0 or time_left() < 6.0:
            break
        order = sorted(range(d), key=lambda i: -rel[i])
        mx = rel[order[0]]
        chosen = [i for i in order if rel[i] > 0.1 * mx][:avail]
        for i in chosen:
            j = probs[i].worst()
            a, b = probs[i].iv.pop(j)[:2]
            m = 0.5 * (a + b)
            pending.append((i, a, m))
            pending.append((i, m, b))
    if not validated:
        return None
    if additive:
        return math.fsum([p.total()[0] for p in probs]) - (d - 1) * fc
    logp = 0.0
    sign = 1.0
    for p in probs:
        I, _ = p.total()
        if I == 0:
            return 0.0
        sign *= math.copysign(1.0, I)
        logp += math.log(abs(I))
    sign *= math.copysign(1.0, fc) ** (d - 1)
    logp -= (d - 1) * math.log(abs(fc))
    return sign * math.exp(logp)


# ---------------------------------------------------------------------------
# ridge integrands g(w.x)


def gradient4(orc, p, h):
    d = len(p)
    pts = []
    for j in range(d):
        for s in (h, -h, 2 * h, -2 * h):
            q = p.copy()
            q[j] += s
            pts.append(q)
    v = orc(np.array(pts)).reshape(d, 4)
    return (8.0 * (v[:, 0] - v[:, 1]) - (v[:, 2] - v[:, 3])) / (12.0 * h)


def cheb_lobatto(n):
    return np.cos(np.pi * np.arange(n + 1) / n)  # from 1 to -1


def bary_weights(n):
    w = np.ones(n + 1)
    w[1::2] = -1.0
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def bary_eval(xn, fn, wb, x):
    """Barycentric interpolation at points x (array) through nodes xn."""
    x = np.asarray(x, float)
    diff = x[:, None] - xn[None, :]
    exact = diff == 0
    diff[exact] = 1.0
    tmp = wb / diff
    out = (tmp @ fn) / tmp.sum(axis=1)
    hit = exact.any(axis=1)
    if hit.any():
        out[hit] = fn[np.argmax(exact[hit], axis=1)]
    return out


def cheb_coeffs(vals):
    """Chebyshev coefficients from values at Chebyshev-Lobatto points (cos order)."""
    n = len(vals) - 1
    ext = np.concatenate([vals, vals[-2:0:-1]])
    c = np.real(np.fft.fft(ext)) / n
    c = c[:n + 1]
    c[0] *= 0.5
    c[-1] *= 0.5
    return c


def ridge_expectation(lo, hi, vals, w):
    """E[p(w.X)], X~U[0,1]^d, p = interpolant through Cheb-Lobatto values on [lo, hi].

    Coordinates are integrated out one at a time: h(s) -> (H(s + w_j) - H(s)) / w_j with H the
    Chebyshev antiderivative; O(d n^2) time, O(n) memory, exact for polynomial p."""
    C = np.polynomial.chebyshev
    n = len(vals) - 1
    t = cheb_lobatto(n)
    c = cheb_coeffs(np.asarray(vals, float))
    A, B = lo, hi
    order = sorted([j for j in range(len(w)) if w[j] != 0.0], key=lambda j: -abs(w[j]))
    if not order:
        return float(np.mean(vals))
    for idx, j in enumerate(order):
        wj = float(w[j])
        A2 = A - min(0.0, wj)
        B2 = B - max(0.0, wj)
        last = idx == len(order) - 1
        if last or B2 - A2 <= 1e-300:
            sp = np.array([0.5 * (A2 + B2)])
        else:
            sp = 0.5 * (A2 + B2) + 0.5 * (B2 - A2) * t
        width = B - A
        if width <= 1e-300:
            newv = np.full(len(sp), C.chebval(0.0, c))
        else:
            def z(u):
                return np.clip((2.0 * u - (A + B)) / width, -1.0, 1.0)
            if abs(wj) > 1e-6 * width:
                Ci = C.chebint(c, lbnd=-1.0) * (0.5 * width)
                newv = (C.chebval(z(sp + wj), Ci) - C.chebval(z(sp), Ci)) / wj
            else:
                newv = C.chebval(z(sp + 0.5 * wj), c)
        if last or B2 - A2 <= 1e-300:
            return float(newv[0])
        c = cheb_coeffs(newv)
        A, B = A2, B2
    return float(np.mean(vals))


def gradient6(orc, p, h):
    d = len(p)
    pts = []
    for j in range(d):
        for s in (h, -h, 2 * h, -2 * h, 3 * h, -3 * h):
            q = p.copy()
            q[j] += s
            pts.append(q)
    v = orc(np.array(pts)).reshape(d, 6)
    return (45.0 * (v[:, 0] - v[:, 1]) - 9.0 * (v[:, 2] - v[:, 3]) + (v[:, 4] - v[:, 5])) / (60.0 * h)


class RidgeModel:
    """f(x) ~ G(lambda(x)), lambda(x) = w.(x - xlo) / sum|w|, G a Chebyshev interpolant on [0,1]."""

    def __init__(self, w, xlo, vals):
        self.w = w.copy()
        self.xlo = xlo
        self.vals = vals  # at lam_k = (1 - cos(pi k / n)) / 2
        n = len(vals) - 1
        self.c = cheb_coeffs(vals[::-1])  # series in z = 2 lam - 1
        self.dc = np.polynomial.chebyshev.chebder(self.c)

    def z(self, X, w=None):
        w = self.w if w is None else w
        S = np.sum(np.abs(w))
        return 2.0 * ((X - self.xlo) @ w) / S - 1.0

    def __call__(self, X):
        return np.polynomial.chebyshev.chebval(np.clip(self.z(X), -1.0, 1.0), self.c)

    def refine(self, X, y, iters=4):
        d = len(self.w)
        sg = np.sign(self.w)
        for _ in range(iters):
            S = np.sum(np.abs(self.w))
            z = self.z(X)
            r = y - np.polynomial.chebyshev.chebval(np.clip(z, -1, 1), self.c)
            gz = np.polynomial.chebyshev.chebval(np.clip(z, -1, 1), self.dc)
            lam = 0.5 * (z + 1.0)
            dlam = (X - self.xlo) / S - lam[:, None] * sg[None, :] / S
            J = 2.0 * gz[:, None] * dlam
            step, *_ = np.linalg.lstsq(J, r, rcond=None)
            if not np.all(np.isfinite(step)):
                break
            nw = self.w + step
            # keep the sign pattern (the profile was sampled on that diagonal)
            nw = np.where(np.sign(nw) == sg, nw, 0.0)
            nrm = np.sum(np.abs(nw))
            if not nrm > 0:
                break
            self.w = nw / nrm * np.sum(np.abs(self.w))
            if np.max(np.abs(step)) < 1e-15 * np.max(np.abs(self.w)):
                break

    def integral(self):
        w = self.w
        lo = float(np.dot(w, self.xlo))
        hi = float(np.dot(w, 1.0 - self.xlo))
        return ridge_expectation(lo, hi, self.vals[::-1], w)


def ridge_method(orc, d, rng, x_val_pts, seed, extra=None, nval0=24, nval2=16):
    """Return estimate or None if the integrand does not look like a ridge."""
    h = 2e-4
    grads = []
    for _ in range(3):
        p = rng.uniform(0.1, 0.9, d)
        grads.append(gradient6(orc, p, h))
    norms = [np.linalg.norm(g) for g in grads]
    if not max(norms) > 0 or min(norms) < 1e-6 * max(norms):
        return None
    k0 = int(np.argmax(norms))
    w = grads[k0] / norms[k0]
    for k, g in enumerate(grads):
        if k == k0:
            continue
        cs = abs(np.dot(g, w)) / norms[k]
        if cs < 1.0 - 1e-7:
            return None
    acc = np.zeros(d)
    for g in grads:
        acc += g if np.dot(g, w) >= 0 else -g
    w = acc / np.linalg.norm(acc)
    w[np.abs(w) < 1e-14] = 0.0
    if not np.any(w != 0):
        return None
    xlo = np.where(w >= 0, 0.0, 1.0)
    xhi = 1.0 - xlo
    nval = max(nval0, 3 * d)
    if orc.remaining() < nval + nval2 + 65:
        return None
    vx = x_val_pts(nval)
    vy = orc(vx)
    n = 32
    lam = 0.5 * (1.0 - cheb_lobatto(n))
    vals = orc(xlo[None, :] + lam[:, None] * (xhi - xlo)[None, :])
    while True:
        c = cheb_coeffs(vals[::-1])
        scale = max(np.max(np.abs(c)), 1e-300)
        tail = np.max(np.abs(c[-6:]))
        if tail <= 1e-14 * scale or n >= 1024:
            break
        n2 = 2 * n
        if orc.remaining() < n2 - n + nval2:
            break
        lam2 = 0.5 * (1.0 - cheb_lobatto(n2))
        newv = orc(xlo[None, :] + lam2[1::2, None] * (xhi - xlo)[None, :])
        allv = np.empty(n2 + 1)
        allv[0::2] = vals
        allv[1::2] = newv
        vals = allv
        n = n2
    model = RidgeModel(w, xlo, vals)
    fscale = max(float(np.max(np.abs(vals))), float(np.max(np.abs(vy))), 1e-300)
    mis0 = float(np.max(np.abs(model(vx) - vy)))
    if mis0 > 1e-2 * fscale:
        return None
    model.refine(vx, vy)
    if orc.remaining() < nval2:
        return None
    v2 = x_val_pts(nval2)
    y2 = orc(v2)
    mis = float(np.max(np.abs(model(v2) - y2)))
    if not mis <= 1e-3 * fscale:
        return None
    if extra is not None and len(extra[1]):
        if not float(np.max(np.abs(model(extra[0]) - extra[1]))) <= 1e-10 * fscale:
            return None
    est = model.integral()
    if not math.isfinite(est):
        return None
    m = orc.remaining()
    if m >= 16 and mis > 1e-13 * fscale:
        m = 1 << int(math.floor(math.log2(m)))  # complete digital net
        X = qmc_ref_points(d, m, seed + 5)
        r = orc(X) - model(X)
        corr = math.fsum(r) / m
        if math.isfinite(corr):
            est += corr
    return est


# ---------------------------------------------------------------------------
# piecewise-linear (kinked ridge sum) identification


def relu_mean_exact(w, b):
    """E[(w.X - b)_+] for X ~ U[0,1]^d, exact rational inclusion-exclusion on float inputs."""
    nz = [Fraction(float(x)) for x in w if x != 0.0]
    bb = Fraction(float(b))
    if not nz:
        return float(max(Fraction(0), -bb))
    n = len(nz)
    prodv = Fraction(1)
    for x in nz:
        prodv *= x
    tot = Fraction(0)
    # subset sums via iterative expansion
    sums = [(Fraction(0), 0)]
    for x in nz:
        sums = sums + [(s + x, c + 1) for (s, c) in sums]
    for s, c in sums:
        t = s - bb
        if t > 0:
            term = t ** (n + 1)
            tot += term if (n - c) % 2 == 0 else -term
    return float(tot / (math.factorial(n + 1) * prodv))


def abs_mean_exact(w, b):
    mean = float(sum(Fraction(float(x)) for x in w) / 2 - Fraction(float(b)))
    return 2.0 * relu_mean_exact(w, b) - mean


def chord(p, q):
    lo, hi = -np.inf, np.inf
    for i in range(len(p)):
        if q[i] > 1e-14:
            lo = max(lo, -p[i] / q[i]); hi = min(hi, (1 - p[i]) / q[i])
        elif q[i] < -1e-14:
            lo = max(lo, (1 - p[i]) / q[i]); hi = min(hi, -p[i] / q[i])
    return lo, hi


class LineProbe:
    """Finds the kinks of a piecewise-linear restriction t -> f(p + t q)."""

    def __init__(self, orc, p, q, lo, hi, fscale_hint=0.0):
        self.orc, self.p, self.q, self.lo, self.hi = orc, p, q, lo, hi
        self.s = np.zeros(0)
        self.f = np.zeros(0)
        self.fscale = fscale_hint
        self.kinks = []  # (s*, mL, mR, ds_left, ds_right)

    def X(self, s):
        return np.clip(self.p[None, :] + np.asarray(s)[:, None] * self.q[None, :], 0.0, 1.0)

    def evalf(self, s):
        s = np.asarray(s, float)
        if len(self.s):
            j = np.searchsorted(self.s, s)
            keep = np.ones(len(s), bool)
            for k in range(len(s)):
                for jj in (j[k] - 1, j[k]):
                    if 0 <= jj < len(self.s) and abs(self.s[jj] - s[k]) <= 1e-14 * (self.hi - self.lo):
                        keep[k] = False
            s = s[keep]
        if len(s) == 0:
            return np.zeros(0)
        if len(s) > self.orc.remaining():
            raise RuntimeError('budget')
        v = self.orc(self.X(s))
        self.s = np.concatenate([self.s, s])
        self.f = np.concatenate([self.f, v])
        o = np.argsort(self.s, kind='stable')
        self.s, self.f = self.s[o], self.f[o]
        self.fscale = max(self.fscale, float(np.max(np.abs(v))) if len(v) else 0.0)
        return v

    def slope_tol(self, ds1, ds2, m1, m2):
        F = max(self.fscale, 1e-300)
        return 2e-12 * F * (1.0 / ds1 + 1.0 / ds2) * 8 + 1e-11 * max(abs(m1), abs(m2))

    def runs(self, s, f):
        """Return list of (i0, i1, slope) runs of >=2 cells with equal slope, nodes i0..i1."""
        ds = np.diff(s)
        m = np.diff(f) / ds
        n = len(m)
        out = []
        k = 0
        while k < n:
            j = k
            while j + 1 < n and abs(m[j + 1] - m[k]) <= self.slope_tol(ds[k], ds[j + 1], m[k], m[j + 1]):
                j += 1
            if j > k:
                # least-squares slope/intercept over nodes k..j+1 not needed; use endpoints
                out.append((k, j + 1))
            k = j + 1
        return out

    def line_of(self, s, f, i0, i1):
        # line through the two extreme nodes of a confirmed run
        m = (f[i1] - f[i0]) / (s[i1] - s[i0])
        return m, f[i0] - m * s[i0]

    def run(self, M, max_iter=7, R=5, cap=None):
        s0 = self.lo + (self.hi - self.lo) * np.linspace(0.0, 1.0, M)
        start_used = self.orc.used
        self.evalf(s0)
        kinks = []
        resolved = set()
        L = self.hi - self.lo
        for _ in range(max_iter):
            s, f = self.s, self.f
            n = len(s)
            if n < 4:
                break
            ds = np.diff(s)
            m = np.diff(f) / ds
            nc = len(m)
            eq = np.array([abs(m[k + 1] - m[k]) <= self.slope_tol(ds[k], ds[k + 1], m[k], m[k + 1])
                           for k in range(nc - 1)], dtype=bool)
            runs = []
            k = 0
            while k < nc - 1:
                if not eq[k]:
                    k += 1
                    continue
                j = k
                while j < nc - 1 and eq[j]:
                    j += 1
                runs.append((k, j + 1))  # nodes k..j+1 collinear
                k = j + 1
            new_pts = []
            if not runs:
                new_pts = list(0.5 * (s[:-1] + s[1:]))
            else:
                if runs[0][0] > 0:
                    a, b = s[0], s[runs[0][0]]
                    if b - a > 1e-9 * L:
                        new_pts += list(a + (b - a) * np.arange(1, R + 1) / (R + 1))
                if runs[-1][1] < n - 1:
                    a, b = s[runs[-1][1]], s[-1]
                    if b - a > 1e-9 * L:
                        new_pts += list(a + (b - a) * np.arange(1, R + 1) / (R + 1))
                for (a0, a1), (b0, b1) in zip(runs[:-1], runs[1:]):
                    sl, sr = s[a1], s[b0]
                    key = (sl, sr)
                    if key in resolved:
                        continue
                    mA = (f[a1] - f[a0]) / (s[a1] - s[a0])
                    cA = f[a0] - mA * s[a0]
                    mB = (f[b1] - f[b0]) / (s[b1] - s[b0])
                    cB = f[b0] - mB * s[b0]
                    ok = False
                    if b0 - a1 <= 1 and abs(mA - mB) > self.slope_tol(s[a1] - s[a0], s[b1] - s[b0], mA, mB):
                        st = (cB - cA) / (mA - mB)
                        tol_s = 1e-9 * (sr - sl) + 1e-15
                        if sl - tol_s <= st <= sr + tol_s:
                            st = min(max(st, sl), sr)
                            if b0 == a1:
                                ok = True
                            else:
                                got = self.evalf([st])
                                if len(got):
                                    F = max(self.fscale, 1e-300)
                                    ok = abs(got[0] - (mA * st + cA)) <= 1e-9 * F
                                else:
                                    ok = True
                            if ok:
                                kinks.append((st, mA, mB, s[a1] - s[a0], s[b1] - s[b0]))
                                resolved.add(key)
                    if not ok and sr - sl > 1e-9 * L:
                        new_pts += list(sl + (sr - sl) * np.arange(1, R + 1) / (R + 1))
            if not new_pts:
                break
            if len(new_pts) > self.orc.remaining() - 8:
                break
            if cap is not None and self.orc.used - start_used + len(new_pts) > cap:
                break
            if time_left() < 8.0:
                break
            self.evalf(np.array(new_pts))
        self.kinks = kinks
        return kinks

    def pl_fraction(self):
        s, f = self.s, self.f
        ds = np.diff(s)
        m = np.diff(f) / ds
        if len(m) < 3:
            return 0.0
        eq = [abs(m[k + 1] - m[k]) <= self.slope_tol(ds[k], ds[k + 1], m[k], m[k + 1]) for k in range(len(m) - 1)]
        return float(np.mean(eq))


def kink_normal(orc, probe, st, mL, mR, dsl, dsr):
    p, q = probe.p, probe.q
    d = len(p)
    xs = p + st * q
    eps = max(2e-3 * min(dsl, dsr), 2e-7)
    xL = np.clip(p + (st - eps) * q, 0, 1)
    xR = np.clip(p + (st + eps) * q, 0, 1)
    jump = mR - mL
    # stencil step well inside the distance to the kink plane along each axis
    h = max(min(2e-2 * eps, 1e-6), 1e-8)
    pts = []
    steps = []
    for base in (xL, xR):
        pts.append(base)
        hs = np.where(base + h <= 1.0, h, -h)
        steps.append(hs)
        for i in range(d):
            z = base.copy()
            z[i] += hs[i]
            pts.append(z)
    if orc.remaining() < len(pts):
        return None
    v = orc(np.array(pts))
    gL = (v[1:d + 1] - v[0]) / steps[0]
    gR = (v[d + 2:] - v[d + 1]) / steps[1]
    nvec = gR - gL
    nn = np.linalg.norm(nvec)
    if not nn > 0:
        return None
    if abs(np.dot(nvec, q) - jump) > 1e-4 * (abs(jump) + 1e-300) + 1e-6 * nn:
        return None
    # one-sided gradients must agree with the line slopes on each side
    if abs(np.dot(gL, q) - mL) > 1e-4 * (abs(mL) + abs(jump)) or abs(np.dot(gR, q) - mR) > 1e-4 * (abs(mR) + abs(jump)):
        return None
    w = nvec / nn
    if w[np.argmax(np.abs(w))] < 0:
        w = -w
    return w, float(np.dot(w, xs)), (np.array(pts), v)


def fit_pl(X, y, planes):
    d = X.shape[1]
    cols = [np.ones(len(X))] + [X[:, i] for i in range(d)] + [np.abs(X @ w - b) for (w, b) in planes]
    A = np.stack(cols, axis=1)
    sc = np.maximum(np.max(np.abs(A), axis=0), 1e-300)
    coef, *_ = np.linalg.lstsq(A / sc, y, rcond=None)
    coef = coef / sc
    return coef


def pl_predict(X, coef, planes):
    d = X.shape[1]
    out = coef[0] + X @ coef[1:d + 1]
    for k, (w, b) in enumerate(planes):
        out = out + coef[d + 1 + k] * np.abs(X @ w - b)
    return out


def pl_integral(coef, planes, d):
    tot = coef[0] + 0.5 * float(np.sum(coef[1:d + 1]))
    for k, (w, b) in enumerate(planes):
        tot += coef[d + 1 + k] * abs_mean_exact(w, b)
    return tot


def random_chord(rng, d, through=None, central=True):
    for _ in range(100):
        if through is not None:
            p = np.asarray(through, float)
        elif central:
            p = rng.uniform(0.3, 0.7, d)
        else:
            p = rng.uniform(0.0, 1.0, d)
        q = rng.normal(size=d)
        q /= np.linalg.norm(q)
        lo, hi = chord(p, q)
        if hi - lo > 0.3 or (through is not None and hi - lo > 0.05):
            return p, q, lo, hi
    return p, q, lo, hi


def refine_planes(X, y, planes, iters=3):
    """Gauss-Newton on all plane parameters of f = c0 + l.x + sum c_j |w_j.x - b_j|."""
    if not planes:
        return planes
    d = X.shape[1]
    W = np.array([w for w, b in planes])
    B = np.array([b for w, b in planes])
    J = len(planes)
    for _ in range(iters):
        T = X @ W.T - B[None, :]
        A = np.concatenate([np.ones((len(X), 1)), X, np.abs(T)], axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        c = coef[d + 1:]
        r = y - A @ coef
        S = np.sign(T)
        # derivatives wrt w_j (d each) and b_j
        cols = [A]
        for j in range(J):
            cols.append(c[j] * S[:, j:j + 1] * X)
            cols.append(-c[j] * S[:, j:j + 1])
        Jm = np.concatenate(cols, axis=1)
        step, *_ = np.linalg.lstsq(Jm, r, rcond=None)
        if not np.all(np.isfinite(step)):
            break
        k = d + 1 + J
        for j in range(J):
            W[j] += step[k:k + d]
            B[j] += step[k + d]
            k += d + 1
        nrm = np.linalg.norm(W, axis=1)
        nrm[nrm == 0] = 1.0
        W /= nrm[:, None]
        B /= nrm
    return [(W[j].copy(), float(B[j])) for j in range(J)]


def dedupe_planes(planes):
    out = []
    for w, b in planes:
        if not all(np.isfinite(w)) or not math.isfinite(b):
            continue
        if any(np.linalg.norm(w - w2) + abs(b - b2) < 1e-6 or np.linalg.norm(w + w2) + abs(b + b2) < 1e-6
               for w2, b2 in out):
            continue
        out.append((w, b))
    return out


def pl_identify(orc, d, rng, probe0, M, cap, reserve, nval=16, fixed=None):
    """Returns (estimate or None, coef, planes)."""
    planes = []
    fitX, fitY = [], []
    fscale = [probe0.fscale]
    if fixed is not None and len(fixed[1]):
        fscale[0] = max(fscale[0], float(np.max(np.abs(fixed[1]))))

    def absorb(probe):
        fscale[0] = max(fscale[0], probe.fscale)
        for (st, mL, mR, dsl, dsr) in probe.kinks:
            xs = probe.p + st * probe.q
            if any(abs(np.dot(w, xs) - b) < 1e-6 for (w, b) in planes):
                continue
            r = kink_normal(orc, probe, st, mL, mR, dsl, dsr)
            if r is None:
                continue
            w, b, (P, V) = r
            planes.append((w, b))
            fitX.append(P)
            fitY.append(V)
        fitX.append(probe.X(probe.s))
        fitY.append(probe.f)

    start = orc.used
    absorb(probe0)
    nlines = 1
    coef = None
    VX, VY = [], []
    target = None
    while True:
        if fixed is not None and len(fixed[1]):
            FX, FY = fixed
            X = np.vstack(fitX + [FX])
            y = np.concatenate(fitY + [FY])
            planes = dedupe_planes(planes)
            try:
                rp = refine_planes(X, y, planes)
                move = max([np.linalg.norm(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(rp, planes)] + [0.0])
                if move < 1e-5:
                    planes = dedupe_planes(rp)
            except Exception:
                pass
            coef = fit_pl(X, y, planes)
            res = np.abs(pl_predict(X, coef, planes) - y)
            fsc = max(fscale[0], 1e-300)
            if np.max(res) <= 1e-8 * fsc:
                return pl_integral(coef, planes, d), coef, planes
            resF = res[len(X) - len(FY):]
            target = FX[int(np.argmax(resF))]
        elif nlines >= 2 and orc.remaining() >= nval:
            X = np.vstack(fitX + VX) if VX else np.vstack(fitX)
            y = np.concatenate(fitY + VY) if VY else np.concatenate(fitY)
            planes = dedupe_planes(planes)
            try:
                rp = refine_planes(X, y, planes)
                move = max([np.linalg.norm(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(rp, planes)] + [0.0])
                if move < 1e-5:
                    planes = dedupe_planes(rp)
            except Exception:
                pass
            coef = fit_pl(X, y, planes)
            V = rng.random((nval, d))
            fv = orc(V)
            pred = pl_predict(V, coef, planes)
            fsc = max(fscale[0], float(np.max(np.abs(fv))), 1e-300)
            VX.append(V)
            VY.append(fv)
            res = np.abs(fv - pred)
            if np.max(res) <= 1e-8 * fsc:
                pl_identify.uniform = (VX, VY)
                X = np.vstack(fitX + VX)
                y = np.concatenate(fitY + VY)
                coef = fit_pl(X, y, planes)
                return pl_integral(coef, planes, d), coef, planes
            target = V[int(np.argmax(res))]
        if orc.used - start > cap or orc.remaining() < reserve + M + 50 or time_left() < 8.0:
            break
        if target is not None and nlines % 2 == 0:
            p, q, lo, hi = random_chord(rng, d, through=target)
        else:
            p, q, lo, hi = random_chord(rng, d, central=(nlines < 3))
        pr = LineProbe(orc, p, q, lo, hi, fscale[0])
        try:
            pr.run(M)
        except RuntimeError:
            break
        absorb(pr)
        nlines += 1
    X = np.vstack(fitX + VX) if VX else np.vstack(fitX)
    y = np.concatenate(fitY + VY) if VY else np.concatenate(fitY)
    planes = dedupe_planes(planes)
    coef = fit_pl(X, y, planes)
    pl_identify.uniform = (VX, VY)
    return None, coef, planes


# ---------------------------------------------------------------------------
# reference-style rules (p-adaptive Clenshaw-Curtis, extensible digital net)

_CCW = {}


def cc_weights(level):
    """Clenshaw-Curtis weights on [-1,1] at x_k = cos(k pi / N), N = 2^(level+1), k = 0..N."""
    if level in _CCW:
        return _CCW[level]
    N = 2 ** (level + 1)
    k = np.arange(N + 1)
    w = np.zeros(N + 1)
    for kk in range(N + 1):
        s = 0.0
        for j in range(1, N // 2 + 1):
            b = 1.0 if j == N // 2 else 2.0
            s += b / (4.0 * j * j - 1.0) * math.cos(2.0 * j * kk * math.pi / N)
        c = 1.0 if kk in (0, N) else 2.0
        w[kk] = c / N * (1.0 - s)
    _CCW[level] = w
    return w


def low_weights(level):
    """Weights of the next-coarser rule expressed on the level-`level` node indices."""
    N = 2 ** (level + 1)
    w = np.zeros(N + 1)
    if level == 0:
        w[1] = 2.0  # trivial midpoint rule
    else:
        w[0::2] = cc_weights(level - 1)
    return w


def node(level, k):
    N = 2 ** (level + 1)
    return 0.5 + 0.5 * np.cos(np.asarray(k) * math.pi / N)


class PCub:
    """Port of S.G. Johnson's pcubature (p-adaptive tensor Clenshaw-Curtis), resumable."""

    def __init__(self, orc, d):
        self.orc, self.d = orc, d
        self.m = [0] * d
        self.K = None
        self.X = np.zeros((0, d))
        self.Y = np.zeros(0)
        self.val = None
        self.err = float('inf')
        self.done = False

    def start(self):
        d = self.d
        idx = np.array(np.meshgrid(*([np.arange(3)] * d), indexing='ij')).reshape(d, -1).T
        if len(idx) > self.orc.remaining():
            self.done = True
            return False
        self.K = idx.copy()
        self.X = np.stack([node(0, idx[:, i]) for i in range(d)], axis=1)
        self.Y = self.orc(self.X)
        self._estimate()
        return True

    def _estimate(self):
        d = self.d
        V = 0.5 ** d
        Wf = [cc_weights(self.m[i])[self.K[:, i]] for i in range(d)]
        Wl = [low_weights(self.m[i])[self.K[:, i]] for i in range(d)]
        full = np.prod(np.stack(Wf, axis=1), axis=1)
        self.val = V * float(np.dot(full, self.Y))
        maxerr = 0.0
        mi = 0
        for i in range(d):
            wl = np.ones(len(self.Y))
            for j in range(d):
                wl = wl * (Wl[j] if j == i else Wf[j])
            e = abs(self.val - V * float(np.dot(wl, self.Y)))
            if e > maxerr:
                maxerr = e
                mi = i
        self.err = maxerr
        self.mi = mi

    def run(self, limit_total=None):
        """Refine until the budget (or limit on total grid points) stops it."""
        d = self.d
        if self.K is None and not self.start():
            return self.val
        while not self.done:
            if self.err <= 0.0:
                self.done = True
                break
            mi = self.mi
            newlev = self.m[mi] + 1
            Nn = 2 ** (newlev + 1)
            others = [np.arange(2 ** (self.m[j] + 1) + 1) if j != mi else np.arange(1, Nn, 2) for j in range(d)]
            cnt = 1
            for o in others:
                cnt *= len(o)
            if cnt > self.orc.remaining() or (limit_total is not None and len(self.Y) + cnt > limit_total):
                self.done = True
                break
            newidx = np.array(np.meshgrid(*others, indexing='ij')).reshape(d, -1).T
            self.K[:, mi] *= 2
            self.m[mi] = newlev
            Xn = np.stack([node(self.m[i], newidx[:, i]) for i in range(d)], axis=1)
            Yn = self.orc(Xn)
            self.K = np.vstack([self.K, newidx])
            self.X = np.vstack([self.X, Xn])
            self.Y = np.concatenate([self.Y, Yn])
            self._estimate()
        return self.val

    def line_pl_fraction(self):
        """Fraction of consecutive node triples along grid lines that are exactly collinear."""
        d = self.d
        F = max(float(np.max(np.abs(self.Y))), 1e-300) if len(self.Y) else 1.0
        tot = 0
        hit = 0
        coords = self.X
        for i in range(d):
            if self.m[i] < 1:
                continue
            others = [j for j in range(d) if j != i]
            keys = {}
            for n in range(len(self.Y)):
                keys.setdefault(tuple(self.K[n, others]), []).append(n)
            for idxs in keys.values():
                if len(idxs) < 3:
                    continue
                idxs = sorted(idxs, key=lambda n: coords[n, i])
                x = coords[idxs, i]
                y = self.Y[idxs]
                dx = np.diff(x)
                sl = np.diff(y) / dx
                for k in range(len(sl) - 1):
                    tol = 1e-10 * F * (1.0 / dx[k] + 1.0 / dx[k + 1]) + 1e-10 * max(abs(sl[k]), abs(sl[k + 1]))
                    tot += 1
                    if abs(sl[k + 1] - sl[k]) <= tol:
                        hit += 1
        return hit / tot if tot else 0.0


def pcubature_port(orc, d, limit=None):
    pc = PCub(orc, d)
    val = pc.run(limit)
    pcubature_port.err = pc.err
    return val, pc.X, pc.Y


def qmc_ref_points(d, n, seed):
    try:
        from qmcpy.discrete_distribution import DigitalNetB2
        P = np.asarray(DigitalNetB2(d, seed=seed)(n), dtype=float)
        if P.shape == (n, d) and np.all(np.isfinite(P)):
            return np.clip(P, 0.0, 1.0)
    except Exception:
        pass
    return sobol_points(d, n, seed)


# ---------------------------------------------------------------------------
# generic fallbacks


def sobol_points(d, n, seed):
    from scipy.stats import qmc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qmc.Sobol(d=d, scramble=True, seed=seed).random(n)


def generic(orc, d, seed, smooth=False):
    B = orc.remaining()
    if B <= 0:
        return None
    if d == 3 and smooth and B >= 512:
        n = int(math.floor(B ** (1.0 / 3.0) + 1e-9))
        z, w = np.polynomial.legendre.leggauss(n)
        z = (z + 1) / 2
        w = w / 2
        grids = np.meshgrid(z, z, z, indexing="ij")
        W = np.einsum('i,j,k->ijk', w, w, w).ravel()
        pts = np.stack([g.ravel() for g in grids], axis=1)
        vals = orc(pts)
        return math.fsum(W * vals)
    if d in (2, 5):
        val, X, Y = pcubature_port(orc, d)
        if val is not None and math.isfinite(val):
            return val
        B = orc.remaining()
        if B <= 0:
            return None
    if d == 1:
        z, w = np.polynomial.legendre.leggauss(B)
        return math.fsum(0.5 * w * orc(((z + 1) / 2)[:, None]))
    n = 1 << int(math.floor(math.log2(B)))
    pts = qmc_ref_points(d, n, seed)
    vals = orc(pts)
    plain = math.fsum(vals) / n
    nfeat = 1 + d + d * (d + 1) // 2
    if d >= 6 and n >= 8 * nfeat:
        # quadratic control variates (regression estimator) on the digital net
        try:
            Z = pts - 0.5
            cols = [np.ones(n)] + [Z[:, i] for i in range(d)]
            ints = [1.0] + [0.0] * d
            for i in range(d):
                for j in range(i, d):
                    cols.append(Z[:, i] * Z[:, j])
                    ints.append(1.0 / 12.0 if i == j else 0.0)
            A = np.stack(cols, axis=1)
            coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
            reg = float(np.dot(coef, ints))
            if math.isfinite(reg):
                return reg
        except Exception:
            pass
    return plain


def pl_path(orc, d, rng, probe, seed):
    """Kinked (piecewise-linear) integrands: reference-style rule first, then an exact
    piecewise-linear model that must reproduce every reference point."""
    if d in (2, 5):
        ref, RX, RY = pcubature_port(orc, d)
        if ref is None:
            return None
        cont = None
    else:
        avail = orc.remaining()
        n1 = 1 << int(math.floor(math.log2(max(1, avail - 300)))) if avail > 301 else 0
        n1 = min(n1, 1024)
        RX = qmc_ref_points(d, max(n1, 1), seed)[:n1]
        RY = orc(RX) if n1 else np.zeros(0)
        ref = math.fsum(RY) / n1 if n1 else None
        cont = n1
    est = None
    planes = []
    if orc.remaining() >= 60 and len(RY):
        reserve = 0 if cont is None else 0
        try:
            cap = orc.remaining() if cont is None else max(0, orc.remaining() - 40)
            est, coef, planes = pl_identify(orc, d, rng, probe, 33, cap=cap, reserve=reserve,
                                            fixed=(RX, RY))
        except Exception:
            est = None
        if est is not None:
            fsc = max(float(np.max(np.abs(RY))), 1e-300)
            if not np.max(np.abs(pl_predict(RX, coef, planes) - RY)) <= 1e-8 * fsc:
                est = None
    if est is not None and math.isfinite(est):
        return est
    if cont is not None and d >= 6 and planes and len(RY) >= 4 * (len(planes) + d + 1):
        # regression (control-variate) estimator on the net with the identified kink planes
        try:
            c2 = fit_pl(RX, RY, planes)
            reg = pl_integral(c2, planes, d)
            if math.isfinite(reg):
                return reg
        except Exception:
            pass
    if cont is not None:
        # extend the digital net only by complete doublings
        if cont and orc.remaining() >= cont:
            P = qmc_ref_points(d, 2 * cont, seed)[cont:]
            Y2 = orc(P)
            return (math.fsum(RY) + math.fsum(Y2)) / (2 * cont)
    return ref


def grid_separability(pc, rng, npairs=6):
    """Cross-ratio separability test on the initial 3^d Clenshaw-Curtis grid (no new points)."""
    d = pc.d
    vmax = float(np.max(np.abs(pc.Y))) if len(pc.Y) else 0.0
    if not (math.isfinite(vmax) and vmax > 0.0):
        return False
    table = {tuple(k): v / vmax for k, v in zip(pc.K.tolist(), pc.Y.tolist())}
    for _ in range(npairs):
        a = rng.integers(0, 3, d)
        b = rng.integers(0, 3, d)
        if np.array_equal(a, b):
            b[0] = (a[0] + 1) % 3
        m = rng.random(d) < 0.5
        if m.all() or not m.any():
            m[:] = False
            m[0] = True
        fa = table.get(tuple(a.tolist()))
        fb = table.get(tuple(b.tolist()))
        f1 = table.get(tuple(np.where(m, a, b).tolist()))
        f2 = table.get(tuple(np.where(m, b, a).tolist()))
        if None in (fa, fb, f1, f2):
            return False
        lhs, rhs = fa * fb, f1 * f2
        scale = max(abs(lhs), abs(rhs))
        if not (scale > 1e-250) or abs(lhs - rhs) > 1e-10 * scale:
            return False
    return True


def ridge_pretest(orc, d, rng, h=1e-5):
    """Two central-difference gradients; True if parallel but of different length."""
    ridge_pretest.importance = None
    gs = []
    for _ in range(2):
        p = rng.uniform(0.1, 0.9, d)
        P = np.concatenate([p + h * np.eye(d), p - h * np.eye(d)])
        v = orc(P)
        gs.append((v[:d] - v[d:]) / (2 * h))
    ridge_pretest.importance = np.abs(gs[0]) + np.abs(gs[1])
    n1, n2 = np.linalg.norm(gs[0]), np.linalg.norm(gs[1])
    if not (n1 > 0 and n2 > 0):
        return False
    cs = abs(np.dot(gs[0], gs[1])) / (n1 * n2)
    if cs < 1.0 - 1e-6:
        return False
    # identical gradients signal a piecewise-linear integrand rather than a ridge
    return abs(n1 - n2) > 1e-7 * max(n1, n2)


def pl_after_grid(orc, d, rng, pc):
    """Exact piecewise-linear model fitted with leftover points, validated on the grid."""
    if orc.remaining() < 60 or not len(pc.Y):
        return None
    try:
        p, q, lo, hi = random_chord(rng, d)
        probe = LineProbe(orc, p, q, lo, hi, float(np.max(np.abs(pc.Y))))
        probe.run(33, cap=max(20, orc.remaining() // 2))
        est, coef, planes = pl_identify(orc, d, rng, probe, 33, cap=orc.remaining(), reserve=0,
                                        fixed=(pc.X, pc.Y))
    except Exception:
        return None
    if est is None or not math.isfinite(est):
        return None
    fsc = max(float(np.max(np.abs(pc.Y))), 1e-300)
    if not np.max(np.abs(pl_predict(pc.X, coef, planes) - pc.Y)) <= 1e-8 * fsc:
        return None
    return est


def _sobol_sampler(d, seed):
    from scipy.stats import qmc
    sob = qmc.Sobol(d=d, scramble=True, seed=seed + 7)

    def vpts(nv):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sob.random(nv)
    return vpts


def tensor_gl(orc, d, importance=None):
    """Tensor Gauss-Legendre rule with as many points per axis as the budget allows
    (axes with larger gradient magnitude get the extra point)."""
    B = orc.remaining()
    base = int(math.floor(B ** (1.0 / d) + 1e-9))
    if base < 1:
        return None
    ns = [base] * d
    order = list(range(d))
    if importance is not None and len(importance) == d and np.all(np.isfinite(importance)):
        order = list(np.argsort(-np.asarray(importance)))
    for i in order:
        trial = ns.copy()
        trial[i] += 1
        if int(np.prod(trial)) <= B:
            ns = trial
    zs, ws = [], []
    for n in ns:
        z, w = np.polynomial.legendre.leggauss(n)
        zs.append((z + 1) / 2)
        ws.append(w / 2)
    G = np.meshgrid(*zs, indexing="ij")
    W = np.ones(1)
    for w in ws:
        W = np.outer(W, w).ravel()
    pts = np.stack([g.ravel() for g in G], axis=1)
    vals = orc(pts)
    return math.fsum(W * vals)


def try_separable(orc, d, rng, ok, spts, svals):
    """Product (rank-1) or additive structure detected by the cross tests -> 1-D integrals."""
    additive = (not ok) and getattr(separability_test, 'additive', False)
    if not ok and not additive:
        return None
    k = int(np.argmax(np.abs(svals)))
    fc = float(svals[k])
    if not math.isfinite(fc) or (fc == 0.0 and not additive):
        return None
    try:
        est = rank1_integral(orc, d, spts[k].copy(), fc, rng, additive=additive)
    except Exception:
        est = None
    if est is not None and math.isfinite(est):
        return est
    return None


def solve_lowdim(orc, d, seed, rng):
    """d = 2 or 5: the p-adaptive Clenshaw-Curtis rule is run to completion unless a
    structural model (separable / ridge) is detected first; leftovers try to improve on it."""
    pc = PCub(orc, d)
    if d == 5:
        ok, spts, svals = separability_test(orc, d, rng)
        est = try_separable(orc, d, rng, ok, spts, svals)
        if est is not None:
            return est
        if ridge_pretest(orc, d, rng):
            try:
                est = ridge_method(orc, d, rng, _sobol_sampler(d, seed), seed)
            except Exception:
                est = None
            if est is not None and math.isfinite(est):
                return est
        est = tensor_gl(orc, d, ridge_pretest.importance)
        if est is not None and math.isfinite(est):
            return est
        return generic(orc, d, seed)
    else:
        ok, spts, svals = separability_test(orc, d, rng)
        est = try_separable(orc, d, rng, ok, spts, svals)
        if est is not None:
            return est
    val = pc.run()
    if val is None or not math.isfinite(val):
        return generic(orc, d, seed)
    plf = pc.line_pl_fraction()
    if plf >= 0.02:
        est = pl_after_grid(orc, d, rng, pc)
        if est is not None:
            return est
        return val
    ymax = float(np.max(np.abs(pc.Y))) if len(pc.Y) else 0.0
    pscale = max(abs(val), ymax * float(np.sqrt(np.mean((pc.Y / ymax) ** 2))) if ymax > 0 else 0.0)
    if pc.err > 1e-9 * pscale and orc.remaining() >= 150:
        try:
            est = ridge_method(orc, d, rng, _sobol_sampler(d, seed), seed, extra=(pc.X, pc.Y),
                               nval0=12, nval2=8)
        except Exception:
            est = None
        if est is not None and math.isfinite(est):
            # keep the Clenshaw-Curtis value when both agree to near machine precision
            if abs(est - val) <= 1e-12 * pscale:
                return val
            return est
    return val


def solve(orc, d, seed):
    rng = np.random.default_rng(seed + 12345)
    if d in (2, 5):
        return solve_lowdim(orc, d, seed, rng)
    ok, spts, svals = separability_test(orc, d, rng)
    est = try_separable(orc, d, rng, ok, spts, svals)
    if est is not None:
        return est
    smooth = False
    if orc.remaining() >= 200 and d >= 2:
        p, q, lo, hi = random_chord(rng, d)
        probe = LineProbe(orc, p, q, lo, hi)
        probe.run(33, cap=120)
        plf = probe.pl_fraction()
        smooth = plf == 0.0
        if plf >= 0.5:
            est = pl_path(orc, d, rng, probe, seed)
            if est is not None and math.isfinite(est):
                return est
    importance = None
    if d >= 2 and orc.remaining() >= 400:
        if ridge_pretest(orc, d, rng):
            est = ridge_method(orc, d, rng, _sobol_sampler(d, seed), seed)
            if est is not None and math.isfinite(est):
                return est
        importance = ridge_pretest.importance
    if d == 3 and smooth and orc.remaining() >= 512:
        est = tensor_gl(orc, d, importance)
        if est is not None and math.isfinite(est):
            return est
    return generic(orc, d, seed, smooth=smooth)


def main():
    fin = sys.stdin
    fout = sys.stdout
    while True:
        line = fin.readline()
        if not line:
            return
        if line.strip():
            break
    init = json.loads(line)
    try:
        d = int(init.get("dimension") or len(init.get("bounds") or [[0, 1]]))
    except Exception:
        d = len(init.get("bounds") or [[0, 1]])
    try:
        budget = int(init.get("budget", 2048))
    except Exception:
        budget = 2048
    try:
        seed = int(init.get("seed", 0) or 0) % (2 ** 32)
    except Exception:
        seed = 0
    orc = Oracle(d, budget, fin, fout)
    est = None
    try:
        est = solve(orc, d, seed)
    except Exception as exc:  # fall back to whatever was collected
        sys.stderr.write("solver error: %r\n" % (exc,))
        est = None
    if est is None or not math.isfinite(est):
        if orc.ys:
            v = np.concatenate(orc.ys)
            v = v[np.isfinite(v)]
            est = float(v.mean()) if v.size else 0.0
        else:
            try:
                if orc.remaining() > 0:
                    v = orc(np.full((1, d), 0.5))
                    est = float(v[0]) if math.isfinite(v[0]) else 0.0
                else:
                    est = 0.0
            except Exception:
                est = 0.0
    fout.write(json.dumps({"type": "result", "estimate": float(est)}) + "\n")
    fout.flush()


if __name__ == "__main__":
    main()
