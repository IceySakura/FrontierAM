#!/usr/bin/env python3
"""Long-run covariance estimation from one ordered trajectory."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
from scipy.linalg import solve


def symmetrize_psd(a, floor=0.0):
    a = (a + a.T) * 0.5
    w, v = np.linalg.eigh(a)
    return (v * np.maximum(w, floor)) @ v.T


def autocovariances(x, maxlag):
    n, d = x.shape
    y = x - x.mean(axis=0)
    maxlag = min(maxlag, n - 2)
    if maxlag < 0:
        return np.zeros((1, d, d))
    nf = next_fast_len(2 * n - 1)
    f = rfft(y, nf, axis=0)
    out = np.empty((maxlag + 1, d, d))
    for i in range(d):
        for j in range(i, d):
            v = irfft(f[:, i] * np.conj(f[:, j]), nf) / n
            out[:, i, j] = v[:maxlag + 1]
            out[0, j, i] = v[0]
            if i != j and maxlag > 0:
                out[1:, j, i] = v[nf - 1:nf - maxlag - 1:-1]
    return out


def var_fit(g, p):
    d = g.shape[1]
    if p == 0:
        return np.zeros((0, d, d)), g[0].copy()
    idx = np.arange(1, p + 1)[:, None] - np.arange(1, p + 1)[None, :]
    ga = g[np.abs(idx)]
    r = np.where(idx[:, :, None, None] >= 0,
                 ga.transpose(0, 1, 3, 2), ga)
    r = r.transpose(0, 2, 1, 3).reshape(p * d, p * d)
    r = (r + r.T) * 0.5
    rhs = g[1:p + 1].transpose(0, 2, 1).reshape(p * d, d)
    try:
        coef = solve(r, rhs, assume_a="sym")
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(r, rhs, rcond=1e-10)[0]
    a = coef.reshape(p, d, d).transpose(0, 2, 1)
    om = g[0] - np.einsum("pij,pkj->ik", a, g[1:p + 1])
    om = symmetrize_psd(om, 1e-12 * max(np.trace(g[0]) / d, 1e-300))
    return a, om


def recolor(a, om):
    d = om.shape[0]
    b = np.eye(d) - a.sum(axis=0)
    try:
        h = solve(b, om)
        h = solve(b, h.T).T
    except np.linalg.LinAlgError:
        bp = np.linalg.pinv(b)
        h = bp @ om @ bp.T
    return symmetrize_psd(h)


def whitened_lags(g, n):
    w, v = np.linalg.eigh(g[0])
    w = np.maximum(w, 1e-12 * max(w.max(), 1e-300))
    c = (v / np.sqrt(w)) @ v.T
    return np.einsum("ij,kjl,ml->kim", c, g[1:], c) * np.sqrt(n)


def qs_weights(size, bandwidth):
    z = np.arange(1, size) / max(bandwidth, 1e-8)
    m = 6 * np.pi * z / 5
    w = np.ones(size)
    w[1:] = 25 / (12 * np.pi ** 2 * z ** 2) * (np.sin(m) / m - np.cos(m))
    return w


def andrews_white_estimate(x):
    n, d = x.shape
    g = autocovariances(x, min(500, n // 3))
    w, v = np.linalg.eigh(g[0])
    w = np.maximum(w, 1e-12 * max(w.max(), 1e-300))
    c = (v / np.sqrt(w)) @ v.T
    gw = np.einsum("ij,kjl,ml->kim", c, g, c)
    rho = np.max(np.abs(np.linalg.eigvals(gw[1])))
    rho = np.clip(rho, -0.95, 0.95)
    bandwidth = 1.3221 * (4 * rho ** 2 / (1 - rho ** 2) ** 4 * n) ** 0.2
    bandwidth = max(1.0, min(bandwidth, len(gw) - 1))
    kw = qs_weights(len(gw), bandwidth)
    hw = gw[0] + np.einsum("k,kij->ij", kw[1:],
                           gw[1:] + gw[1:].transpose(0, 2, 1))
    hw = symmetrize_psd(hw)
    b = (v * np.sqrt(w)) @ v.T
    return symmetrize_psd(b @ hw @ b.T)


def discrete_estimate(x, vals, inv):
    n, d = x.shape
    k = len(vals)
    if k <= 1:
        return np.zeros((d, d))
    counts = np.zeros((k, k))
    np.add.at(counts, (inv[:-1], inv[1:]), 1.0)
    alpha = 0.5 if k <= 10 else 5.0 / k
    p = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * k)
    a = np.eye(k) - p.T
    a[-1] = 1.0
    rhs = np.zeros(k)
    rhs[-1] = 1.0
    try:
        pi = solve(a, rhs)
    except np.linalg.LinAlgError:
        pi = np.bincount(inv, minlength=k) / n
    pi = np.maximum(pi, 1e-12)
    pi /= pi.sum()
    y = vals - pi @ vals
    z = np.eye(k) - p + np.ones((k, 1)) * pi
    try:
        h = solve(z, y)
    except np.linalg.LinAlgError:
        h = np.linalg.lstsq(z, y, rcond=1e-10)[0]
    ph = p @ h
    return symmetrize_psd((h * pi[:, None]).T @ h - (ph * pi[:, None]).T @ ph)


def var_average_estimate(g, n, d, pmax):
    fits = []
    best = None
    for p in range(pmax + 1):
        a, om = var_fit(g, p)
        kt = d * (p * d + 1)
        penalty = 2 * kt
        if n > kt + 1:
            penalty += 2 * kt * (kt + 1) / (n - kt - 1)
        else:
            penalty += 1e6
        score = n * np.linalg.slogdet(om)[1] + penalty
        h = recolor(a, om)
        if np.isfinite(h).all():
            fits.append((score, h))
        if best is None or score < best[0]:
            best = (score, a, p)
    if not fits:
        return symmetrize_psd(g[0]), np.zeros((0, d, d)), 0
    scores = np.array([f[0] for f in fits])
    weights = np.exp(-0.5 * (scores - scores.min()))
    weights /= weights.sum()
    h = np.zeros((d, d))
    for w, (_, hp) in zip(weights, fits):
        if w > 1e-10:
            h += w * hp
    return symmetrize_psd(h), best[1], best[2]


def estimate(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("expected a two-dimensional trajectory")
    n, d = x.shape
    if n < 2 or d == 0:
        return np.zeros((d, d))
    if not np.isfinite(x).all():
        x = np.where(np.isfinite(x), x, 0.0)
    vals, inv = np.unique(x, axis=0, return_inverse=True)
    k = len(vals)
    if 1 < k <= min(10, n // 20):
        return discrete_estimate(x, vals, inv)

    pmax = min(30, max(1, n // (10 * d + 5)), n // 4)
    maxlag = max(min(100, n // 8), pmax)
    g = autocovariances(x, maxlag)
    if np.trace(g[0]) <= 1e-300:
        return np.zeros((d, d))

    # A short moving average has a sharp autocovariance cutoff.  Confirm the
    # cutoff with VAR(1) residuals so that a weak VAR is not truncated.
    gz = whitened_lags(g, n)
    significant = np.any(np.abs(gz) > 5.5, axis=(1, 2))
    q = int(np.max(np.where(significant)[0])) + 1 if significant.any() else 0
    if 1 <= q <= 2:
        y = x - x.mean(axis=0)
        a1, _ = var_fit(g, 1)
        u = y[1:] - y[:-1] @ a1[0].T
        gu = autocovariances(u, min(10, len(u) // 8))
        if np.trace(gu[0]) > 1e-300:
            uz = whitened_lags(gu, len(u))
            if np.any(np.abs(uz) > 4.5):
                h = g[0] + g[1:q + 1].sum(axis=0)
                return symmetrize_psd(h + h.T - g[0])

    h, a, p = var_average_estimate(g, n, d, pmax)
    if p > 0:
        y = x - x.mean(axis=0)
        u = y[p:].copy()
        for i in range(p):
            u -= y[p - i - 1:n - i - 1] @ a[i].T
        w, v = np.linalg.eigh(np.cov(u.T, bias=True))
        w = np.maximum(w, 1e-12 * max(w.max(), 1e-300))
        c = (v / np.sqrt(w)) @ v.T
        z = u @ c.T
        kurtosis_ratio = np.mean(np.sum(z * z, axis=1) ** 2) / (d * (d + 2))
        weight = 0.5 * np.clip((kurtosis_ratio - 1.5) / 1.0, 0.0, 1.0)
        if weight > 0:
            h = (1 - weight) * h + weight * andrews_white_estimate(x)
            h = symmetrize_psd(h)
    return h


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: estimator.py INPUT.npy OUTPUT.npy")
    x = np.load(sys.argv[1], allow_pickle=False)
    h = estimate(x)
    h = np.ascontiguousarray((h + h.T) * 0.5, dtype="<f8")
    np.save(sys.argv[2], h, allow_pickle=False)


if __name__ == "__main__":
    main()
