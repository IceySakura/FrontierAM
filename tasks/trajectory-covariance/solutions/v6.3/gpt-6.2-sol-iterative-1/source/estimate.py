"""Estimate the zero-frequency long-run covariance from one trajectory."""

import sys

import numpy as np

from operators import poisson_fit
from states import discrete
from linear import linear_cov


def sym(a):
    return (a + a.T) * 0.5


def psd(a):
    eig, u = np.linalg.eigh(sym(a))
    return sym((u * np.maximum(0.0, eig)) @ u.T)


def autocov(x, lag):
    n, d = x.shape
    c = np.empty((lag + 1, d, d))
    c[0] = x.T @ x / n
    for k in range(1, lag + 1):
        c[k] = x[:-k].T @ x[k:] / n
    return c


def lag_model(c, n):
    """Select a vector autoregression using its covariance prediction errors."""
    pmax, d = len(c) - 1, c.shape[-1]
    ix = np.arange(pmax)
    lag = ix[:, None] - ix
    r = c[np.abs(lag)]
    r = np.where((lag < 0)[:, :, None, None], r.swapaxes(2, 3), r)
    r = r.transpose(0, 2, 1, 3).reshape(pmax * d, pmax * d)
    z = c[1:].reshape(-1, d)
    chol = np.linalg.cholesky(sym(r) + 1e-10 * np.eye(pmax * d))
    u = np.linalg.solve(chol, z)
    blocks = u.reshape(pmax, d, d)
    losses = np.r_[c[:1], c[0] - np.cumsum(blocks.transpose(0, 2, 1) @ blocks, axis=0)]
    p = int(np.argmin(np.linalg.slogdet(losses)[1] * n
                      + np.arange(pmax + 1) * d * d * (0.8 * np.log(n))))
    q = p * d
    b = np.linalg.solve(chol[:q, :q].T, u[:q]).reshape(p, d, d)
    return b, losses[p]


def var_cov(b, e):
    a = np.eye(e.shape[0]) - b.sum(axis=0)
    h = np.linalg.solve(a.T, np.linalg.cholesky(sym(e) + 1e-13 * np.eye(len(e))))
    return sym(h @ h.T)


def flat_top(x, k):
    """Flat-top lag window with a cosine taper, evaluated by FFT."""
    n, d = x.shape
    size = 1 << (n + k - 1).bit_length()
    z = np.fft.rfft(x, n=size, axis=0)
    w = np.arange(1, k + 1) / k
    w = np.where(w < 0.5, 1.0, 0.5 + 0.5 * np.cos(np.pi * (2 * w - 1)))
    answer = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1):
            a = np.fft.irfft((z[:, i] * z[:, j].conj()).real, n=size)
            answer[i, j] = answer[j, i] = a[0] + 2.0 * (w @ a[1:k + 1])
    return answer / (n - 1 - 2 * w.sum())


def continuous(y):
    n, d = y.shape
    if d >= 5:
        repeats = np.mean(np.all(y[1:] == y[:-1], axis=1))
        fit = poisson_fit(y, repeats)
        if fit is not None:
            h, _, _ = fit
            return sym(h)
    pmax = min(16, max(2, n // (10 * d)))
    c = autocov(y, pmax)
    strength = np.linalg.norm(c[1:], axis=(1, 2)) * np.sqrt(n)
    # Short-range dependence can be estimated directly without fitting an
    # autoregression to the noise floor.
    significant = np.flatnonzero(strength > d + 2.5) + 1
    if len(significant) == 0 or significant[-1] <= 2:
        q = int(significant[-1]) if len(significant) else 0
        return psd(c[0] + 2 * sym(c[1:q + 1].sum(axis=0))) * (n / (n - 1 - 2 * q))

    b, e = lag_model(c, n)
    h = var_cov(b, e)
    if d < 5 and len(b):
        try:
            h = linear_cov(y, len(b))
        except np.linalg.LinAlgError:
            pass
    timescale = max(1.0, np.linalg.eigvalsh(h)[-1])
    k = int(np.clip(3.0 * timescale, 8, max(8, n // 24)))
    if d >= 5:
        # Use the smoother estimate unless low-frequency data show departures
        # from the selected autoregression.
        l, u = np.linalg.eigh(h)
        s = (u * np.sqrt(np.maximum(l, 1e-8))) @ u.T
        si = (u / np.sqrt(np.maximum(l, 1e-8))) @ u.T
        delta = sym(si @ flat_top(y, k) @ si) - np.eye(d)
        noise = np.sqrt(2.3 * k / n * d * (d + 1))
        ratio = np.linalg.norm(delta) / max(0.05, noise)
        weight = np.clip(1.0 - (0.85 / max(ratio, 0.03)) ** 2, 0.10, 0.82)
        lam, v = np.linalg.eigh(delta)
        h = s @ ((v * np.maximum(0.12, 1 + weight * lam)) @ v.T) @ s
    return sym(h)


def covariance(x):
    n, d = x.shape
    if n < 4 or not d:
        return np.zeros((d, d))
    try:
        h = discrete(x)
    except (np.linalg.LinAlgError, FloatingPointError, RuntimeError):
        h = None
    if h is not None and np.isfinite(h).all():
        return h

    # Work in whitened coordinates so that all lag decisions are insensitive
    # to the input units and, on full rank data, to linear mixtures.
    z = x - x[0]
    z -= z.mean(axis=0)
    scale = np.max(np.abs(z), axis=0)
    scale[scale == 0] = 1
    z /= scale
    e, u = np.linalg.eigh(sym(z.T @ z / n))
    if e[0] < 1e-7 * e[-1]:
        # Avoid squaring the condition number when coordinates are nearly
        # collinear. Compute scores using identical arithmetic for equal rows.
        _, sing, vu = np.linalg.svd(z, full_matrices=False)
        s = sing / np.sqrt(n)
        u = vu.T
        keep = s > max(1e-30, 64 * np.finfo(float).eps * s[0])
    else:
        s = np.sqrt(np.maximum(e, 0))
        keep = e > max(1e-30, e[-1] * 2e-15)
    if not np.any(keep):
        return np.zeros((d, d))
    s = s[keep]
    v = u[:, keep]
    y = np.ascontiguousarray((z @ v) / s)
    unwhite = s[:, None] * v.T * scale[None, :]
    try:
        h = continuous(y)
    except (np.linalg.LinAlgError, FloatingPointError):
        h = None
    if h is None or not np.isfinite(h).all():
        h = psd(flat_top(y, min(max(6, int(n ** 0.35)), n // 4)))
    return sym(unwhite.T @ h @ unwhite)


def estimate(x):
    peak = np.max(np.abs(x), initial=0)
    change_scale = 0 < peak < 1e-140 or peak > 1e140
    h = covariance(x / peak if change_scale else x)
    scale = np.max(np.abs(h), initial=0)
    if scale == 0:
        return h
    if not np.isfinite(h).all():
        return np.zeros_like(h)
    small = sym(h / scale)
    if np.linalg.eigvalsh(small)[0] < 0:
        h = psd(small) * scale
    if change_scale:
        exponent = min(np.log(scale) + 2*np.log(peak), np.log(np.finfo(float).max) - 1)
        h = (h / scale) * np.exp(exponent)
    return h


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: run.sh INPUT.npy OUTPUT.npy")
    x = np.load(sys.argv[1], allow_pickle=False)
    if not isinstance(x, np.ndarray) or x.ndim != 2 or not np.isfinite(x).all():
        raise SystemExit("expected a finite two-dimensional NPY array")
    h = np.ascontiguousarray(estimate(x), dtype="<f8")
    with open(sys.argv[2], "wb") as f:
        np.save(f, h, allow_pickle=False)


if __name__ == "__main__":
    main()
