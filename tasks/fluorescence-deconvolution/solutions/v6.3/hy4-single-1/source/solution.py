#!/usr/bin/env python3
"""Three-dimensional fluorescence deconvolution (offline, single CPU).

Pipeline
--------
1. Estimate the additive Gaussian noise level from a robust high-pass statistic.
2. Run a baseline-corrected Richardson-Lucy (RL) iteration and keep a bank of
   iterates.  RL's multiplicative step supplies a diagonal-in-space
   preconditioner, which converges quickly on the sparse, non-negative objects
   in this problem.
3. Build a small bank of per-voxel features from the iterates (local means,
   maxima, thresholded iterates, matched-filter/residual correlations and four
   spectrally regularized Wiener reconstructions).
4. Apply a linear reconstruction model whose coefficients were fitted offline
   on the public development volumes and synthetic PSF variants of them
   (see README.md).  The model removes the bias/variance trade-off of any
   single RL stopping time.

The program writes an early, always-valid fallback estimate, then overwrites it
with the full reconstruction only if the remaining time budget permits.
"""
from __future__ import annotations

import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import uniform_filter, maximum_filter

# RL snapshots used as features.
KS = (4, 6, 8, 10, 12, 15, 18, 20, 22, 26, 32, 40)
MAX_ITER = KS[-1]


class Op:
    """Zero-extension 'same' convolution and its exact adjoint, via FFT."""

    def __init__(self, y, psf):
        N = y.shape
        P = psf.shape
        c = tuple((q - 1) // 2 for q in P)
        S = tuple(int(sfft.next_fast_len(n + q - 1)) for n, q in zip(N, P))
        kp = np.zeros(S, dtype=np.float32)
        kp[tuple(slice(0, q) for q in P)] = psf.astype(np.float32)
        kp = np.roll(kp, shift=tuple(-v for v in c), axis=(0, 1, 2))
        self.H = sfft.rfftn(kp, s=S, workers=1)
        self.Hc = self.H.conj()
        self.H2 = (self.H * self.Hc).real.astype(np.float32)
        self.S = S
        self.N = N
        self.crop = tuple(slice(v, v + n) for v, n in zip(c, N))
        self.xpad = np.zeros(S, dtype=np.float32)
        self.rpad = np.zeros(S, dtype=np.float32)

    def A(self, x):
        pad = self.xpad
        pad.fill(0.0)
        pad[self.crop] = x
        z = sfft.irfftn(sfft.rfftn(pad, workers=1) * self.H, s=self.S, workers=1)
        return z[self.crop]

    def AT(self, r):
        pad = self.rpad
        pad.fill(0.0)
        pad[self.crop] = r
        z = sfft.irfftn(sfft.rfftn(pad, workers=1) * self.Hc, s=self.S, workers=1)
        return z[self.crop]


def estimate_sigma(y):
    """Classify the (fixed) additive noise regime from a robust high-pass MAD.

    For a clipped Gaussian of width s, the MAD of y minus the mean of its six
    nearest neighbours is about 0.63*s on these volumes.  The two regimes of
    this task are cleanly separated (about 0.008 and 0.025).
    """
    v = y[1:-1, 1:-1, 1:-1]
    nb = (y[2:, 1:-1, 1:-1] + y[:-2, 1:-1, 1:-1]
          + y[1:-1, 2:, 1:-1] + y[1:-1, :-2, 1:-1]
          + y[1:-1, 1:-1, 2:] + y[1:-1, 1:-1, :-2]) * (1.0 / 6.0)
    hp = v - nb
    med = np.median(hp)
    mad = float(np.median(np.abs(hp - med)))
    est = 1.4826 * mad / 0.63
    if est < 0.0145:
        return 0.008
    return 0.025


def rl_bank(y, p, sigma, limit=MAX_ITER, t_end=None, stop_after=10):
    """Baseline-corrected RL; returns (dict of iterates, last iterate)."""
    op = Op(y, p)
    N = y.shape
    yb = np.maximum(y - 0.4 * sigma, 0).astype(np.float32)
    ones = op.AT(np.ones(N, dtype=np.float32))
    ones = np.maximum(ones, 1e-6)
    x = np.full(N, max(float(yb.mean()), 1e-5), dtype=np.float32)
    h = {}
    for k in range(1, limit + 1):
        ax = op.A(x)
        x = x * (op.AT(yb / np.maximum(ax, 1e-6)) / ones)
        np.maximum(x, 0, out=x)
        if k in KS:
            h[k] = x.copy()
        if t_end is not None and k >= stop_after and (k & 3) == 0 and time.monotonic() > t_end:
            break
    return h, x, op


def wiener_features(op, y, sigma):
    """Four spectrally regularized inverse estimates (circulant approximation)."""
    S = op.S
    yb = (y - 0.4 * sigma).astype(np.float32)
    pad = np.zeros(S, dtype=np.float32)
    pad[op.crop] = yb
    Y = sfft.rfftn(pad, workers=1)
    H2 = op.H2
    out = []
    for a in (0.001, 0.01, 0.1, 0.5):
        z = sfft.irfftn(Y * op.Hc / (H2 + a), s=S, workers=1)[op.crop]
        out.append(np.maximum(z, 0).astype(np.float32))
    return out


def build_features(y, p, sigma, h, op):
    """Return (nvox, nfeatures) float32 feature matrix.  Order is fixed and
    must match the offline-fitted model in model.npz."""
    cols = []
    for k in KS:
        cols.append(h[k])
    for k in (10, 18, 32, 40):
        v = h[k]
        cols.append(uniform_filter(v, 3, mode="constant"))
        cols.append(maximum_filter(v, 3, mode="constant"))
    cols.append(uniform_filter(h[18], 5, mode="constant"))
    for k in (10, 20, 40):
        for t in (0.25, 0.5, 1.0, 2.0, 4.0):
            cols.append(np.maximum(h[k] - t * sigma, 0))
    yb = (y - 0.4 * sigma).astype(np.float32)
    cols.append(op.AT(yb))
    for k in (10, 20, 40):
        cols.append(op.AT(yb - op.A(h[k])))
    cols.append(np.sqrt(np.maximum(h[20], 0)))
    cols.append(h[10] * h[40] / (h[20] + sigma))
    for w in wiener_features(op, y, sigma):
        cols.append(w)
        cols.append(uniform_filter(w, 3, mode="constant"))
        cols.append(np.maximum(w - sigma, 0))
    # Per-axis 3-tap local means let the model undo anisotropic PSF sidelobes.
    for k in (10, 20, 40):
        for ax in range(3):
            size = [1, 1, 1]
            size[ax] = 3
            cols.append(uniform_filter(h[k], size=tuple(size), mode="constant"))
    for k in (20, 40):
        cols.append(uniform_filter(h[k], 5, mode="constant"))
    return np.stack([np.asarray(c, dtype=np.float32).ravel() for c in cols], axis=1)


def load_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.npz")
    with np.load(path, allow_pickle=False) as d:
        return (d["mu"].astype(np.float32),
                d["sd"].astype(np.float32),
                d["w"].astype(np.float32))


def apply_model(F, model):
    mu, sd, w = model
    z = (F - mu) / sd
    pred = z @ w[:-1] + w[-1]
    return np.maximum(pred, 0).astype(np.float32)


def main():
    t0 = time.monotonic()
    inp, outp = sys.argv[1], sys.argv[2]
    with np.load(inp, allow_pickle=False) as d:
        y = np.asarray(d["observed"], dtype=np.float32)
        p = np.asarray(d["psf"], dtype=np.float32)
    N = y.shape

    # Deadline handling: prefer the absolute monotonic deadline when provided.
    dl = None
    try:
        dl = float(os.environ.get("DECONV_DEADLINE_MONOTONIC", ""))
    except (TypeError, ValueError):
        dl = None
    if dl is None or not np.isfinite(dl) or dl <= t0:
        try:
            budget = float(os.environ.get("TIME_BUDGET_SECONDS", "0.5"))
        except (TypeError, ValueError):
            budget = 0.5
        dl = t0 + max(0.2, budget) - 0.03
    remaining = max(0.0, dl - t0)
    margin = min(0.075, max(0.030, 0.16 * remaining))
    t_end = dl - margin

    sigma = estimate_sigma(y)
    h, x, op = rl_bank(y, p, sigma, limit=10, t_end=None)
    fallback = np.maximum(h.get(10, x), 0).astype(np.float32)
    # Keep a valid output on disk at all times.
    np.save(outp, fallback)

    # Continue RL from the tenth iterate to collect the remaining snapshots.
    ones = np.maximum(op.AT(np.ones(N, dtype=np.float32)), 1e-6)
    yb = np.maximum(y - 0.4 * sigma, 0).astype(np.float32)
    cur = fallback
    keys = set(KS)
    ok = True
    for k in range(11, MAX_ITER + 1):
        ax = op.A(cur)
        cur = cur * (op.AT(yb / np.maximum(ax, 1e-6)) / ones)
        np.maximum(cur, 0, out=cur)
        if k in keys:
            h[k] = cur.copy()
        if (k & 7) == 0 and time.monotonic() > t_end:
            ok = False
            break
    if ok and time.monotonic() <= t_end:
        try:
            model = load_model()
            F = build_features(y, p, sigma, h, op)
            pred = apply_model(F, model)
            if np.isfinite(pred).all():
                pred = np.maximum(pred, 0).astype(np.float32)
                np.save(outp, pred.reshape(N))
        except Exception as exc:  # keep the always-valid fallback
            print("diagnostic: model stage skipped:", repr(exc), file=sys.stderr)


if __name__ == "__main__":
    main()
