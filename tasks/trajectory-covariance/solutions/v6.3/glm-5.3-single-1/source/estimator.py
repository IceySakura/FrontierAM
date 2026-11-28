#!/usr/bin/env python3
"""Estimate the long-run covariance H of a single MCMC trajectory.

    Sigma = lim_{n->inf} n Cov(bar X_n) = Gamma_0 + sum_{k>=1} (Gamma_k + Gamma_k^T)

Method (fully data driven, no per-model tuning):
  1. Compute empirical autocovariances Gamma_k for k = 0..L by FFT.
  2. Whiten by the estimated Gamma_0 so every decision below is scale free
     (the input may have been passed through an arbitrary invertible affine
     map; all statistics used are invariant under such maps and the result is
     mapped back exactly).
  3. In the whitened space estimate the noise level of the autocovariance
     trace sequence (robust second-difference estimator), fit the decay rate
     of the autocovariance sequence over the lags whose signal is above that
     noise, and from the fitted autocorrelation time tau pick the lag window
     K that minimises the model risk  bias^2 + variance  of the truncated
     autocovariance sum.
  4. Sum the whitened autocovariances up to K with a mild taper, map back and
     project onto the PSD cone.
"""
import os
import sys

import numpy as np

CFG = dict(
    Lfrac=0.5,      # maximum lag as a fraction of M
    cK=0.5,         # window multiplier
    clog=8.0,       # window logarithm argument
    taper=0.5,      # fraction of the window that is tapered
    guard=2.0,      # significance threshold (in noise sigmas)
    floor=1e-10,    # relative eigenvalue floor when whitening
    tlag=4.0,       # per-lag significance threshold
    tcum=3.0,       # cumulative significance threshold
    beta=2.0,       # cap on how far past the last significant lag we sum
)


def autocov(Y, L):
    """Gamma[k][i, j] ~ E[Y_{t,i} Y_{t+k,j}] with 1/M normalisation."""
    M, d = Y.shape
    n = 1 << (int(2 * M - 1).bit_length())
    F = np.fft.rfft(Y, n=n, axis=0)
    G = np.empty((L + 1, d, d))
    for j in range(d):
        z = np.fft.irfft(F * np.conj(F[:, j:j + 1]), n=n, axis=0)
        G[:, :, j] = z[:L + 1, :]
    return G / M


def psd_proj(H):
    H = (H + H.T) / 2
    if not np.isfinite(H).all():
        return np.zeros_like(H)
    lam, V = np.linalg.eigh(H)
    if lam[0] < 0.0:
        H = (V * np.maximum(lam, 0.0)) @ V.T
        H = (H + H.T) / 2
    return H


def estimate(X, cfg=CFG):
    M, d = X.shape
    Y = np.ascontiguousarray(X - X.mean(0), dtype=np.float64)
    L = int(min(M - 1, max(8, cfg['Lfrac'] * M)))
    G = autocov(Y, L)
    Ls = G.copy()
    Ls[1:] = G[1:] + np.transpose(G[1:], (0, 2, 1))
    Ls[0] = (G[0] + G[0].T) / 2

    # ---- whiten by the lag-zero covariance
    C0 = Ls[0]
    lam, V = np.linalg.eigh(C0)
    top = max(lam.max(), 0.0)
    if top <= 0.0:
        return np.zeros((d, d))
    lamw = np.maximum(lam, top * cfg['floor'])
    T = (V / np.sqrt(lamw)) @ V.T          # C0^{-1/2}
    Tinv = (V * np.sqrt(lamw)) @ V.T       # C0^{1/2}
    Lw = np.einsum('ab,kbc,cd->kad', T, Ls, T)

    a = np.einsum('kii->k', Lw)            # whitened trace sequence, a[0] ~ d

    # ---- noise level of a_k from second differences (robust to slow decay)
    k0 = max(4, int(0.4 * L))
    sig = 0.0
    seg = a[k0:]
    if len(seg) > 12:
        d2 = seg[2:] - 2 * seg[1:-1] + seg[:-2]
        sig = float(np.sqrt(np.median(d2 ** 2) / (6.0 * 0.4549)))

    # ---- decay rate over the lags whose (smoothed) signal exceeds the noise
    sm = a.copy()
    if len(sm) > 5:
        sm[1:-1] = 0.25 * a[:-2] + 0.5 * a[1:-1] + 0.25 * a[2:]
    # J = end of the initial run of lags whose signal stays clear of the noise
    # band (a single excursion in the far tail must not extend it).
    thr = cfg['guard'] * sig
    thr2 = cfg['tlag'] * sig
    J = 1
    miss = 0
    for k in range(1, min(L, 4 * M // 5) + 1):
        if a[k] > thr2:
            J = k
            miss = 0
        else:
            miss += 1
            if miss >= 2 and k > 4:
                break
    J = int(min(L - 1, max(1, J)))
    num = float(np.dot(a[1:J], a[2:J + 1]))
    den = float(np.dot(a[1:J], a[1:J]))
    rho = num / den if den > 0 else 0.0
    rho = float(min(max(rho, 0.0), 1.0 - 1e-4))
    # significance of the accumulated autocovariance over the early lags: a
    # single lag (or the cumulative sum) must stand clear of the noise band
    nst = int(min(L, 60))
    m1 = float(np.max(np.abs(a[1:nst + 1]))) if nst >= 1 else 0.0
    csum = np.cumsum(a[1:nst + 1])
    tn = float(np.max(np.abs(csum))) / (sig * np.sqrt(np.arange(1, nst + 1))).max() if nst >= 1 else 0.0
    insignificant = sig > 0 and m1 < cfg['tlag'] * sig and tn < cfg['tcum']

    # self-consistent window: the cumulative trace at lag K gives the average
    # integrated autocorrelation, which sets the window again (handles
    # processes whose autocovariance is a mixture of fast and slow modes).
    cum = np.cumsum(a)
    # never look beyond a bounded multiple of the last lag whose (smoothed)
    # signal is above the noise: beyond that the cumulative only accumulates
    # noise and the fixed point below would drift upwards on pure noise.
    Kcap = max(2, int(cfg['beta'] * J))
    K = 0
    for _ in range(40):
        tau = max(1.0, cum[min(K, L)] / d)
        tau = min(tau, max(1.0, 0.25 * M))
        knew = int(min(Kcap, cfg['cK'] * tau * np.log(max(2.0, cfg['clog'] * M / tau))))
        if abs(knew - K) <= 1:
            K = knew
            break
        K = knew
    if insignificant:
        K = 0
    K = int(min(L, max(0, K)))
    if K <= 0:
        return psd_proj(Tinv @ Lw[0] @ Tinv)

    w = np.ones(K)
    tp = cfg['taper']
    if tp > 0:
        k0t = int((1.0 - tp) * K)
        if K - k0t > 1:
            r = np.arange(k0t, K)
            w[k0t:] = 0.5 * (1.0 + np.cos(np.pi * (r - k0t + 1.0) / (K - k0t)))
    S = Lw[0] + np.einsum('k,kab->ab', w, Lw[1:K + 1])
    return psd_proj(Tinv @ S @ Tinv)


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    X = np.load(inp, allow_pickle=False)
    if X.ndim != 2:
        raise SystemExit('input must be 2-D')
    X = np.ascontiguousarray(X, dtype=np.float64)
    H = estimate(X)
    if not np.isfinite(H).all():
        H = psd_proj(np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0))
    np.save(outp, np.ascontiguousarray(H, dtype='<f8'))


if __name__ == '__main__':
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    main()
