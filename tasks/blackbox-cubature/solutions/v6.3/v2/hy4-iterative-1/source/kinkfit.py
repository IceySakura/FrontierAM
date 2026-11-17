"""Fit sums of ridge-hinge functions with exact integrals.

f(x) ~ c + polynomial(x) + sum_k gamma_k |a_k.x - b_k|.
The hinge directions (a_k,b_k) are optimized by variable projection; the
linear coefficients are then used with exactly integrated centered features.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import least_squares
import advanced as A
from hinges import ridge_integrals, fit_kinks_1d


def _poly_cols(X, d):
    cols = []
    for i in range(d):
        cols.append(X[:, i] - 0.5)
    for i in range(d):
        for j in range(i, d):
            if i == j:
                cols.append((X[:, i] - 0.5) ** 2 - 1 / 12)
            else:
                cols.append((X[:, i] - 0.5) * (X[:, j] - 0.5))
    return cols


def _design(X, p, d, K, pcols):
    cols = [np.ones(len(X))] + pcols
    for k in range(K):
        a = p[k * (d + 1):k * (d + 1) + d]
        b = p[k * (d + 1) + d]
        cols.append(np.abs(X @ a - b))
    return np.column_stack(cols)


def _init_params(lines, x0, d, K, rng):
    specs = [fit_kinks_1d(L['t'], L['y'], K=K) for L in lines]
    inits = []
    # rank-paired assembly with a few sign conventions
    for sgn in (1.0, -1.0):
        p = []
        for k in range(K):
            taus = np.array([specs[i][k][0] if k < len(specs[i]) else float(x0[i]) for i in range(d)])
            dels = np.array([specs[i][k][1] if k < len(specs[i]) else 0.0 for i in range(d)])
            P = dels * (taus - np.asarray(x0))
            sigma = sgn * np.sign(P)
            sigma[sigma == 0] = 1.0
            a = sigma * dels
            nrm = np.linalg.norm(a)
            if nrm < 1e-8:
                a = rng.normal(size=d); nrm = np.linalg.norm(a)
            a = a / nrm
            med = float(np.median(np.abs(P)[np.abs(P) > 1e-10])) if np.any(np.abs(P) > 1e-10) else 0.0
            b = float(a @ np.asarray(x0) + sgn * med / max(nrm, 1e-8))
            p.extend(a.tolist()); p.append(b)
        inits.append(np.asarray(p))
    # random restarts
    for _ in range(3):
        p = []
        for k in range(K):
            a = rng.normal(size=d); a /= np.linalg.norm(a)
            b = float(a @ np.asarray(x0) + rng.uniform(-0.25, 0.25))
            p.extend(a.tolist()); p.append(b)
        inits.append(np.asarray(p))
    return inits


def fit_kink_model(X, y, lines, x0, d, K=3, seed=0, max_inits=5):
    """Return (list of (a,b), rms) for the best hinge model fit."""
    rng = np.random.default_rng(seed + 9182)
    nfit = min(640, len(X))
    idx = rng.choice(len(X), nfit, replace=False) if len(X) > nfit else np.arange(len(X))
    Xf, yf = X[idx], y[idx]
    pcols = _poly_cols(Xf, d)
    inits = _init_params(lines, x0, d, K, rng)[:max_inits]

    def resid(p):
        M = _design(Xf, p, d, K, pcols)
        beta, *_ = np.linalg.lstsq(M, yf, rcond=1e-9)
        return yf - M @ beta

    best = None
    sy = float(np.std(yf)) + 1e-12
    for p0 in inits:
        try:
            out = least_squares(resid, p0, max_nfev=55, ftol=1e-12, xtol=1e-12,
                                gtol=1e-12, x_scale='jac')
            p = out.x
            r = resid(p)
        except Exception:
            continue
        rms = float(np.sqrt(np.mean(r ** 2))) / sy
        if best is None or rms < best[0]:
            best = (rms, p.copy())
    if best is None:
        return []
    rms, p = best
    out = []
    for k in range(K):
        a = p[k * (d + 1):k * (d + 1) + d].copy()
        b = float(p[k * (d + 1) + d])
        nrm = np.linalg.norm(a)
        if nrm < 1e-10:
            continue
        a = a / nrm; b = b / nrm
        if np.max(np.abs(a)) < 1e-12:
            continue
        out.append((a, b))
    return out


def kink_features(specs, X, y, d):
    """Centered hinge features with exact integral zero."""
    feats = []
    for k, (a, b) in enumerate(specs):
        ia, _ = ridge_integrals(a, b)
        if not np.isfinite(ia):
            continue
        feats.append(A.Feature(f'hinge{k}',
                               lambda XX, a=a, b=b, ia=ia: np.abs(np.asarray(XX, float) @ a - b) - ia,
                               0.0, 'kink'))
    return feats


def greedy_hinges(X, y, lines, x0, d, K=3, seed=0, ninit=4, max_nfev=40):
    """Add one ridge hinge at a time, then a short joint polish."""
    rng = np.random.default_rng(seed + 551)
    nfit = min(512, len(X))
    idx = rng.choice(len(X), nfit, replace=False) if len(X) > nfit else np.arange(len(X))
    Xf, yf = X[idx], y[idx]
    pcols = _poly_cols(Xf, d)
    sy = float(np.std(yf)) + 1e-12
    inits0 = _init_params(lines, x0, d, K, rng)
    hinges = []

    def design(pl):
        cols = [np.ones(len(Xf))] + pcols
        for a, b in pl:
            cols.append(np.abs(Xf @ a - b))
        return np.column_stack(cols)

    def fit_one(target, p0):
        def resid(p):
            cols = [np.ones(len(Xf))] + pcols + [np.abs(Xf @ p[:d] - p[d])]
            M = np.column_stack(cols)
            beta, *_ = np.linalg.lstsq(M, target, rcond=1e-9)
            return target - M @ beta
        try:
            out = least_squares(resid, p0, max_nfev=max_nfev, ftol=1e-12,
                                xtol=1e-12, gtol=1e-12, x_scale='jac')
            return out.x, float(np.sqrt(np.mean(resid(out.x) ** 2)))
        except Exception:
            return p0, float('inf')

    target = yf.copy()
    for k in range(K):
        cand = []
        # initializations from line-derived params and random directions
        if k < len(inits0):
            cand.append(inits0[k][:d + 1])
        for _ in range(ninit):
            a = rng.normal(size=d); a /= np.linalg.norm(a)
            b = float(a @ np.asarray(x0) + rng.uniform(-0.3, 0.3))
            cand.append(np.concatenate([a, [b]]))
        best = None
        for p0 in cand:
            p, err = fit_one(target, p0)
            if best is None or err < best[1]:
                best = (p, err)
        p, _ = best
        a = p[:d].copy(); nrm = np.linalg.norm(a)
        if nrm > 1e-10:
            hinges.append((a / nrm, float(p[d]) / nrm))
        M = design(hinges)
        beta, *_ = np.linalg.lstsq(M, yf, rcond=1e-9)
        target = yf - M @ beta
        if np.sqrt(np.mean(target ** 2)) / sy < 1e-10:
            break
    return hinges


def scan_hinges(X, y, lines, x0, d, K=3, seed=0, ndirc=120):
    """Global direction/threshold scan followed by a joint nonlinear polish."""
    rng = np.random.default_rng(seed + 31337)
    nfit = min(768, len(X))
    idx = rng.choice(len(X), nfit, replace=False) if len(X) > nfit else np.arange(len(X))
    Xf, yf = X[idx], y[idx]
    pcols = _poly_cols(Xf, d)
    B0 = np.column_stack([np.ones(len(Xf))] + pcols)
    sy = float(np.std(yf)) + 1e-12
    # candidate directions
    dirs = []
    for i in range(d):
        e = np.zeros(d); e[i] = 1.0; dirs.append(e)
    for i in range(d):
        for j in range(i + 1, d):
            e = np.zeros(d); e[i] = 1 / math.sqrt(2); e[j] = 1 / math.sqrt(2); dirs.append(e)
            e = np.zeros(d); e[i] = 1 / math.sqrt(2); e[j] = -1 / math.sqrt(2); dirs.append(e)
    for _ in range(ndirc):
        a = rng.normal(size=d); a /= np.linalg.norm(a); dirs.append(a)
    # line-derived directions
    specs = [fit_kinks_1d(L['t'], L['y'], K=K) for L in lines]
    for k in range(K):
        dels = np.array([specs[i][k][1] if k < len(specs[i]) else 0.0 for i in range(d)])
        if np.linalg.norm(dels) > 1e-8:
            dirs.append(dels / np.linalg.norm(dels))
            dirs.append(-dels / np.linalg.norm(dels))
    D = np.asarray(dirs)
    hinges = []
    for k in range(K):
        # residual after current model
        M = np.column_stack([B0] + [np.abs(Xf @ a - b) for a, b in hinges])
        beta, *_ = np.linalg.lstsq(M, yf, rcond=1e-9)
        r = yf - M @ beta
        r = r - r.mean()
        best = None
        for a in D:
            z = Xf @ a
            bs = np.quantile(z, np.linspace(0.08, 0.92, 15))
            H = np.abs(z[:, None] - bs[None, :])
            Hc = H - H.mean(0)
            den = np.sum(Hc * Hc, 0) + 1e-12
            gain = (r @ Hc) ** 2 / den
            j = int(np.argmax(gain))
            if best is None or gain[j] > best[0]:
                best = (float(gain[j]), a.copy(), float(bs[j]))
        if best is None:
            break
        hinges.append((best[1], best[2]))
        if best[0] < 1e-12 * (r @ r + 1e-30):
            break
    # joint polish of all hinge parameters
    if hinges:
        p0 = np.concatenate([np.concatenate([a, [b]]) for a, b in hinges])
        H = len(hinges)
        def resid(p):
            cols = [np.ones(len(Xf))] + pcols
            for kk in range(H):
                a = p[kk * (d + 1):kk * (d + 1) + d]; b = p[kk * (d + 1) + d]
                cols.append(np.abs(Xf @ a - b))
            M = np.column_stack(cols)
            beta, *_ = np.linalg.lstsq(M, yf, rcond=1e-9)
            return yf - M @ beta
        try:
            out = least_squares(resid, p0, max_nfev=70, ftol=1e-13, xtol=1e-13,
                                gtol=1e-13, x_scale='jac')
            if np.linalg.norm(resid(out.x)) <= np.linalg.norm(resid(p0)):
                p0 = out.x
        except Exception:
            pass
        hinges = []
        for kk in range(H):
            a = p0[kk * (d + 1):kk * (d + 1) + d].copy()
            b = float(p0[kk * (d + 1) + d])
            nrm = np.linalg.norm(a)
            if nrm > 1e-10:
                hinges.append((a / nrm, b / nrm))
    return hinges
