"""Long-run covariance estimation from a single MCMC trajectory.

Target
------
Given one trajectory x_1..x_M in R^d, estimate the d x d long-run covariance

    Sigma = lim_{n->inf} n * Cov( mean_{t<=n} x_t ),      H ~= Sigma,

so that H/M estimates the covariance of the sample mean.

Method
------
Sigma = sum_{k=-inf}^{inf} Gamma(k).  Rather than truncating that sum with a
kernel (which needs a bandwidth and pays a bias/variance price), the trajectory
is modelled as a vector autoregression whose autocovariance sequence is summed
analytically to infinity:

    y_t = sum_{i=1..p} A_i y_{t-i} + e_t,   Cov(e_t) = Om
    H(p) = B^-1 Om B^-T,   B = I - sum_i A_i

A_i and Om come from the empirical autocovariances G[0..K] through the
multivariate Yule-Walker equations (G[k] = E[y_t y_{t+k}^T], all lags at once
from a single zero-padded FFT).  An empirical autocovariance sequence is itself
positive definite, so the Yule-Walker fit is always stationary: B is invertible
and H(p) is symmetric positive semi-definite by construction.  The order p is
chosen by minimising  M*log|Om_p| + PENALTY * d^2 * p  over p = 0..pmax.

The data are whitened by the lag-0 covariance purely for conditioning; the
estimate is equivariant under any invertible affine map of the coordinates,
matching the invariance of the scoring loss.

Dependencies: Python 3 + NumPy only (no SciPy, no compiled code, no network).
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

PENALTY = 1.8        # information-criterion penalty per VAR coefficient
KFRAC = 0.30         # autocovariance lags kept, as a fraction of M
KMAX = 1200
PMAX_FRAC = 0.05     # pmax ~ PMAX_FRAC * M / d
PMAX_MIN = 4
PMAX_MAX = 48
TAU_CAP_FRAC = 0.2   # whitened H eigenvalues capped at TAU_CAP_FRAC * M


def next_fast_len(n):
    """Smallest 5-smooth integer >= n (an FFT length NumPy handles fast)."""
    if n <= 6:
        return max(int(n), 1)
    best = 1 << (int(n) - 1).bit_length()
    p5 = 1
    while p5 < best:
        p3 = 1
        while p5 * p3 < best:
            base = p5 * p3
            p2 = 1 << (-(-int(n) // base) - 1).bit_length()
            v = p2 * base
            if v < best:
                best = v
            p3 *= 3
        p5 *= 5
    return best


def acovf(Y, K):
    """G[k] = (1/M) sum_t y_t y_{t+k}^T for k = 0..K, via one zero-padded FFT."""
    M, d = Y.shape
    n = next_fast_len(2 * M)
    F = np.fft.rfft(Y, n=n, axis=0)
    S = (F[:, :, None] * np.conj(F[:, None, :])).reshape(F.shape[0], d * d)
    A = np.fft.irfft(S, n=n, axis=0)[:K + 1].reshape(K + 1, d, d)
    return A / M


def inv_sqrt(C, floor=1e-11):
    """(C^{-1/2}, C^{1/2}) for a symmetric PSD matrix C."""
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    wm = max(float(w.max()), 1e-300)
    w = np.maximum(w, floor * wm)
    return (V / np.sqrt(w)) @ V.T, (V * np.sqrt(w)) @ V.T


def psd_project(H):
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    if w[0] < 0.0:
        H = (V * np.maximum(w, 0.0)) @ V.T
    return 0.5 * (H + H.T)


def yw(G, p):
    """Yule-Walker VAR(p) coefficients A_1..A_p from autocovariances G[0..].

    Right-multiplying y_t = sum_i A_i y_{t-i} + e_t by y_{t-j}^T and taking
    expectations gives, for j = 1..p,   sum_i A_i M_ij = G[j]^T, where
    M_ij = E[y_{t-i} y_{t-j}^T] = G[j-i] (j >= i) or G[i-j]^T (j < i).
    M is symmetric, so stacking Z_i = A_i^T solves  M Z = r  with r_j = G[j].
    """
    d = G.shape[1]
    R = np.empty((p * d, p * d))
    r = np.empty((p * d, d))
    for j in range(p):
        for i in range(p):
            k = j - i
            R[j * d:(j + 1) * d, i * d:(i + 1) * d] = G[k] if k >= 0 else G[-k].T
        r[j * d:(j + 1) * d] = G[j + 1]
    R[np.diag_indices_from(R)] += 1e-10 * np.trace(G[0]) / d + 1e-300
    Z = np.linalg.solve(R, r).reshape(p, d, d)
    return np.swapaxes(Z, 1, 2).copy()


def var_fit_lrc(G, p):
    """Return (H_p, log|Om_p|) for a VAR(p) Yule-Walker fit, or None."""
    d = G.shape[1]
    if p == 0:
        Om = 0.5 * (G[0] + G[0].T)
        H = Om.copy()
    else:
        A = yw(G, p)
        Om = G[0].copy()
        for i in range(p):
            Om -= A[i] @ G[i + 1]
        Om = 0.5 * (Om + Om.T)
        Bi = np.linalg.inv(np.eye(d) - A.sum(axis=0))
        X = Bi @ Om
        H = 0.5 * (X @ Bi.T + Bi @ X.T)
    w = np.linalg.eigvalsh(Om)
    if not np.isfinite(H).all() or w.min() <= 1e-10 * max(w.max(), 1e-300):
        return None
    return H, float(np.log(np.maximum(w, 1e-300)).sum())


def estimate(X):
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    M, d = X.shape
    Y = X - X.mean(axis=0)

    K = int(min(max(20, KFRAC * M), KMAX, max(2, M - 2)))
    G0 = acovf(Y, K)

    C0 = 0.5 * (G0[0] + G0[0].T)
    P0, P0i = inv_sqrt(C0)
    Gw = np.einsum('ij,kjl,lm->kim', P0, G0, P0, optimize=True)

    pmax = max(1, min(PMAX_MAX, max(PMAX_MIN, int(PMAX_FRAC * M / d)), K - 1))

    best_crit, best_H = np.inf, np.ascontiguousarray(Gw[0])
    for p in range(0, pmax + 1):
        try:
            fit = var_fit_lrc(Gw, p)
        except np.linalg.LinAlgError:
            continue
        if fit is None:
            continue
        H, ld = fit
        crit = M * ld + PENALTY * (d * d * p)
        if crit < best_crit:
            best_crit, best_H = crit, H

    # Safety net.  In the whitened frame H is a matrix of integrated
    # autocorrelation times; no direction can be estimated beyond a fraction of
    # the trajectory length, so clip runaway eigenvalues instead of letting a
    # near-singular B inflate the answer without bound.
    w, V = np.linalg.eigh(0.5 * (best_H + best_H.T))
    cap = TAU_CAP_FRAC * M
    if w.max() > cap:
        best_H = (V * np.clip(w, 0.0, cap)) @ V.T

    return np.ascontiguousarray(psd_project(P0i @ best_H @ P0i))


def safe_fallback(X):
    """Never fail: the best simple PSD matrix obtainable from the input."""
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim == 1:
        X = X[:, None]
    M, d = X.shape
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Y = X - X.mean(axis=0)
    C = (Y.T @ Y) / max(M, 1)
    if not np.isfinite(C).all():
        return np.zeros((d, d), dtype="<f8")
    return np.ascontiguousarray(psd_project(C), dtype="<f8")


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    X = np.load(inp, allow_pickle=False)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    d = X.shape[1] if X.ndim == 2 else 1
    try:
        H = estimate(X)
        if H.shape != (d, d) or not np.isfinite(H).all():
            raise ValueError("bad estimate")
    except Exception:
        try:
            H = safe_fallback(X)
        except Exception:
            H = np.zeros((d, d), dtype="<f8")
    np.save(outp, np.ascontiguousarray(psd_project(H), dtype="<f8"))


if __name__ == "__main__":
    main()
