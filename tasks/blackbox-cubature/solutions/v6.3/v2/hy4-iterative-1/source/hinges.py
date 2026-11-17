"""Kink-ridge (ReLU / absolute value) features with exact integrals."""
from __future__ import annotations
import math
from decimal import Decimal, getcontext
import numpy as np
from scipy.optimize import least_squares

getcontext().prec = 60


def _ridge_expectations(w, u):
    """E|S-u|, E[(S-u)_+] for S=sum w_i X_i, w_i>0, X_i~U(0,1) (Decimal)."""
    w = [Decimal(float(x)) for x in w]
    d = len(w)
    V = sum(w)
    u = Decimal(float(u))
    prod = Decimal(1)
    for x in w:
        prod *= x
    fac = Decimal(math.factorial(d - 1)) * prod
    if fac == 0:
        return 0.0, 0.0

    def ep(c):
        if c >= V:
            return Decimal(0)
        tot = Decimal(0)
        for mask in range(1 << d):
            a = Decimal(0)
            for i in range(d):
                if mask >> i & 1:
                    a += w[i]
            sign = -1 if (bin(mask).count('1') & 1) else 1
            b = max(c, a)
            hi = V - a
            lo = b - a
            g = (hi ** (d + 1) - lo ** (d + 1)) / Decimal(d + 1) + \
                (a - c) * (hi ** d - lo ** d) / Decimal(d)
            tot += Decimal(sign) * g
        return tot / fac

    pos = ep(u)
    absv = V / 2 - u + 2 * ep(V - u)
    return float(absv), float(pos)


def ridge_integrals(a, b):
    """Exact integrals of |a.x-b| and (a.x-b)_+ over [0,1]^d."""
    a = np.asarray(a, float)
    w = np.abs(a)
    if np.max(w) < 1e-12:
        return abs(float(b)), max(0.0, float(-b))
    C = float(np.sum(w[a < 0]))
    return _ridge_expectations(w, float(b) + C)


def _line_basis(t, taus):
    cols = [np.ones_like(t), t, t * t]
    for tau in taus:
        cols.append(np.maximum(0.0, t - tau))
    return np.column_stack(cols)


def fit_kinks_1d(t, y, K=3):
    """Fit a line profile with a quadratic plus K free-knot linear hinges."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    n = len(t)
    if n < 10:
        return []
    # slope jumps on the interior grid
    dt = np.diff(t)
    sl = np.diff(y) / np.maximum(dt, 1e-12)
    ds = np.abs(np.diff(sl))
    tm = 0.5 * (t[1:-1] + t[2:])
    if len(ds) < 3:
        return []
    # pick up to K separated local maxima
    order = np.argsort(ds)[::-1]
    peaks = []
    for j in order:
        if ds[j] < max(1e-12, 0.03 * ds.max()):
            break
        if all(abs(tm[j] - p) > 0.02 for p in peaks):
            peaks.append(float(tm[j]))
        if len(peaks) >= K:
            break
    if not peaks:
        return []
    best = None
    for kk in range(1, len(peaks) + 1):
        taus0 = np.sort(np.asarray(peaks[:kk]))
        def resid(p):
            M = _line_basis(t, p)
            beta, *_ = np.linalg.lstsq(M, y, rcond=None)
            return y - M @ beta
        try:
            out = least_squares(resid, taus0, bounds=(-0.3, 1.3), max_nfev=80,
                                ftol=1e-14, xtol=1e-14, gtol=1e-14)
            p = np.sort(np.clip(out.x, -0.3, 1.3))
        except Exception:
            p = taus0
        M = _line_basis(t, p)
        beta, *_ = np.linalg.lstsq(M, y, rcond=None)
        r = y - M @ beta
        rms = float(np.sqrt(np.mean(r ** 2)))
        sy = float(np.std(y)) + 1e-300
        jumps = [(float(p[j]), float(beta[3 + j]), rms / sy) for j in range(len(p))]
        jumps.sort(key=lambda z: -abs(z[1]))
        if best is None or rms < best[0]:
            best = (rms, jumps, rms / sy)
    return best[1] if best else []


def hinge_feature(a, b, kind='abs'):
    """Centered hinge feature (exact integral zero) and its integral."""
    a = np.asarray(a, float)
    ia, ip = ridge_integrals(a, b)
    I = ia if kind == 'abs' else ip
    def fn(X, a=a, b=b, kind=kind, I=I):
        z = np.asarray(X, float) @ a - b
        return (np.abs(z) if kind == 'abs' else np.maximum(0.0, z)) - I
    # integral of the centered feature is 0 by construction
    return fn


def discover_hinges(lines, x0, X, y, d, K=3):
    """Build a pool of candidate ridge-hinge features from line kink data."""
    specs = [fit_kinks_1d(L['t'], L['y'], K=K) for L in lines]
    out = []
    # axis-aligned hinges (exact, cheap)
    for i, sp in enumerate(specs):
        for tau, delta, rel in sp[:K]:
            a = np.zeros(d); a[i] = 1.0
            out.append(('axis_%d' % i, a, float(tau), 'pos'))
    # oblique hinges assembled by rank, with both sign conventions
    for k in range(K):
        taus, dels = [], []
        for i in range(d):
            sp = specs[i]
            if k < len(sp):
                taus.append(sp[k][0]); dels.append(sp[k][1])
            else:
                taus.append(float(x0[i])); dels.append(0.0)
        taus = np.asarray(taus); dels = np.asarray(dels)
        P = dels * (taus - np.asarray(x0))
        absP = np.abs(P)
        good = absP > 1e-12
        if np.count_nonzero(good) < 2:
            continue
        med = float(np.median(absP[good]))
        if med < 1e-12:
            continue
        # only accept if a reasonable fraction of axes agree (consistent ridge)
        if np.mean(np.abs(absP[good] - med) < 0.6 * med) < 0.5:
            continue
        for sgn in (1.0, -1.0):
            sigma = sgn * np.sign(P)
            sigma[~good] = 1.0
            a = sigma * dels
            nrm = float(np.linalg.norm(a))
            if nrm < 1e-10:
                continue
            a = a / nrm
            b = float(a @ np.asarray(x0) + sgn * med / nrm)
            out.append(('obl_%d_%g' % (k, sgn), a, b, 'abs'))
            out.append(('obl_pos%d_%g' % (k, sgn), a, b, 'pos'))
    return out
