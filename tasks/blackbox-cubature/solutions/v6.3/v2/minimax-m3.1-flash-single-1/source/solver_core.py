"""Core cubature engine: QMC net + surrogate portfolio with internal model selection.

The engine is budget-honest: every function value it consumes is counted by the
caller.  All estimators are computed from the *same* point set, so model
selection costs no extra evaluations.
"""
import math

import numpy as np

SQRT5 = math.sqrt(5.0)


# ------------------------------------------------------------------ designs
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
           67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131)


def _van_der_corput(n, base):
    """Radical inverse in the given base (pure numpy, fallback only)."""
    out = np.zeros(n)
    idx = np.arange(1, n + 1, dtype=np.int64)
    f = 1.0 / base
    while np.any(idx > 0):
        out += f * (idx % base)
        idx = idx // base
        f /= base
    return out


def _halton_fallback(d, n, seed):
    """Scrambled Halton point set; used only if SciPy is unavailable."""
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    X = np.empty((n, d))
    for j in range(d):
        base = _PRIMES[j % len(_PRIMES)]
        v = _van_der_corput(n, base)
        shift = rng.random()
        perm = rng.permutation(n)
        u = np.mod(v + shift, 1.0)
        X[:, j] = u[perm]
    return X


def sobol_points(d, n, seed):
    try:
        from scipy.stats import qmc
        eng = qmc.Sobol(d=d, scramble=True, seed=int(seed) & 0x7FFFFFFF)
        if n & (n - 1):
            # non power of two: draw the next power of two and truncate
            m = int(math.ceil(math.log2(n)))
            return eng.random_base2(m)[:n]
        return eng.random_base2(int(round(math.log2(n))))
    except Exception:
        return _halton_fallback(d, n, seed)


def halton_points(d, n, seed):
    from scipy.stats import qmc
    eng = qmc.Halton(d=d, scramble=True, seed=int(seed) & 0x7FFFFFFF)
    return eng.random(n)


# ------------------------------------------------------------- legendre PCE
def _legendre_table(X, p):
    n = X.shape[0]
    t = 2.0 * X - 1.0
    P = np.empty((n, X.shape[1], p + 1))
    P[:, :, 0] = 1.0
    if p >= 1:
        P[:, :, 1] = t
    for k in range(1, p):
        P[:, :, k + 1] = ((2 * k + 1) * t * P[:, :, k] - k * P[:, :, k - 1]) / (k + 1)
    return P


def _multi_indices(d, p):
    seen = set()
    out = []

    def rec(prefix, left, dims_left):
        if dims_left == 0:
            t = tuple(prefix)
            if t not in seen:
                seen.add(t)
                out.append(t)
            return
        for k in range(left + 1):
            rec(prefix + [k], left - k, dims_left - 1)

    rec([], p, d)
    zero = (0,) * d
    out.remove(zero)
    out.insert(0, zero)
    return out


def legendre_basis(X, p):
    n, d = X.shape
    P = _legendre_table(X, p)
    idx = _multi_indices(d, p)
    Phi = np.empty((n, len(idx)))
    for j, k in enumerate(idx):
        col = P[:, 0, k[0]].copy()
        for i in range(1, d):
            col *= P[:, i, k[i]]
        Phi[:, j] = col
    return Phi


def _ridge_solve(G, b, lam):
    T = G.shape[0]
    scale = max(1e-300, float(np.trace(G)) / max(T, 1))
    G = G.copy()
    G[np.diag_indices_from(G)] += lam * scale
    try:
        return np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(G, b, rcond=None)[0]


def pce_fit(Phi, y, lam=1e-10):
    G = Phi.T @ Phi
    b = Phi.T @ y
    coef = _ridge_solve(G, b, lam)
    return coef


def pce_integral(Phi, y, lam=1e-10):
    """Control-variate form: analytic integral of the fitted polynomial plus the
    QMC mean of the residual.  Less biased than the raw PCE constant for
    designs that do not integrate the basis exactly."""
    coef = pce_fit(Phi, y, lam)
    resid = y - Phi @ coef
    return float(coef[0] + np.mean(resid)), coef


# ------------------------------------------------------------ kernel quad
def _matern52(d2, h):
    r = np.sqrt(np.maximum(d2, 0.0)) / h
    return (1.0 + SQRT5 * r + 5.0 * r * r / 3.0) * np.exp(-SQRT5 * r)


def _sqdist(A, B):
    return np.maximum(np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * A @ B.T, 0.0)


def kq_weights(X_fit, X_apply, h, m=320, reg=1e-8):
    n = X_fit.shape[0]
    if m >= n:
        sel = np.arange(n)
    else:
        sel = np.unique(np.linspace(0, n - 1, m).astype(int))
    Xs = X_fit[sel]
    Kss = _matern52(_sqdist(Xs, Xs), h)
    Kss[np.diag_indices_from(Kss)] += reg
    one = np.ones(len(sel))
    try:
        alpha = np.linalg.solve(Kss, one)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(Kss, one, rcond=None)[0]
    return _matern52(_sqdist(X_apply, Xs), h) @ alpha


def median_heuristic_h(X, d, rng):
    """Median pairwise-distance based kernel bandwidth."""
    n = X.shape[0]
    sub = X[rng.choice(n, size=min(256, n), replace=False)]
    dist = np.sqrt(_sqdist(sub, sub))
    med = float(np.median(dist[dist > 0]))
    return med / math.sqrt(max(d, 1))


# --------------------------------------------------------------- selection
def _degrees(d, cap):
    """Candidate total degrees with a manageable number of terms."""
    if d == 1:
        cand = [4, 8, 12, 16, 20, 24]
    elif d == 2:
        cand = [4, 8, 12, 16, 20, 24, 28]
    elif d == 3:
        cand = [4, 6, 8, 10, 12, 14, 16]
    elif d == 4:
        cand = [3, 4, 5, 6, 7]
    elif d == 5:
        cand = [3, 4, 5, 6]
    elif d == 6:
        cand = [2, 3, 4, 5]
    elif d == 7:
        cand = [2, 3, 4]
    else:
        cand = [2, 3, 4]
    out = []
    for p in cand:
        T = math.comb(d + p, p)
        if T <= cap and T >= 2:
            out.append(p)
    return out or [min(cand)]


def _split(n, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    h = n // 2
    return perm[:h], perm[h:]


def select_and_estimate(X, y, d, seed, cap=None, parsimony=2.0):
    """Return (estimate, info).  Uses a half-split consistency criterion to
    choose between the plain QMC mean and a polynomial (PCE) control variate.

    The split discrepancy of a high-degree fit cannot see the *bias* that the
    quasi-random design injects into high-order moments, so among the degrees
    whose discrepancy is statistically indistinguishable we keep the smallest
    (most parsimonious) one.
    """
    n = X.shape[0]
    if cap is None:
        cap = max(24, n // 3)
    A, B = _split(n, seed + 991)
    scale = float(np.std(y)) + 1e-300

    mA = float(np.mean(y[A]))
    mB = float(np.mean(y[B]))
    disc_mean = abs(mA - mB)

    discs = []
    for p in _degrees(d, cap):
        try:
            PhiA = legendre_basis(X[A], p)
            if PhiA.shape[1] > len(A) // 2:
                continue
            cA = pce_fit(PhiA, y[A])
            eA = float(cA[0] + np.mean(y[A] - PhiA @ cA))
            PhiB = legendre_basis(X[B], p)
            cB = pce_fit(PhiB, y[B])
            eB = float(cB[0] + np.mean(y[B] - PhiB @ cB))
            discs.append((p, abs(eA - eB)))
        except Exception:
            continue

    best_p = None
    best_disc = float('inf')
    if discs:
        dmin = min(v for _, v in discs)
        thr = max(parsimony * dmin, 1e-15 * scale)
        for p, v in discs:            # degrees are tried in increasing order
            if v <= thr:
                best_p, best_disc = p, v
                break
        if best_p is None:
            best_p, best_disc = min(discs, key=lambda kv: (kv[1], kv[0]))

    if best_p is None or disc_mean <= best_disc:
        return float(np.mean(y)), {'method': 'mean', 'disc': disc_mean}

    Phi = legendre_basis(X, best_p)
    coef = pce_fit(Phi, y)
    est = float(coef[0] + np.mean(y - Phi @ coef))
    return est, {'method': f'pce{best_p}', 'disc': best_disc, 'scale': scale}
