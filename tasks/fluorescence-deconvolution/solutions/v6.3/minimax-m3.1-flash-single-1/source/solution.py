#!/usr/bin/env python3
"""3-D fluorescence deconvolution under a hard CPU budget.

Method
------
The forward model is a 'same' convolution with zero extension outside the
grid.  The observation noise is signal dependent (Poisson-like) plus an
additive Gaussian component, so the maximum-likelihood estimator is the
Poisson (Richardson-Lucy) deconvolution.  Plain RL is an unregularised
maximum-likelihood method: its iterates improve monotonically for a while and
then start to fit the noise, so the number of iterations *is* the regulariser.

The stopping point is chosen per case by K-fold cross validation directly on
the observed volume: a random subset of voxels is held out, the multiplicative
update is run on the remaining voxels only (masked likelihood), and the
held-out prediction error is recorded for every iteration.  The iteration with
the smallest held-out error is the point where the estimate still predicts
unseen data best, i.e. where the implicit smoothing of RL matches the noise
level of the case.  A final RL run on all voxels with that many iterations
produces the output.

Implementation notes
--------------------
* The PSF of the supplied cases is separable; when that is detected the
  convolution and its adjoint are applied as three small matrix products
  (fast path).  A general zero-padded FFT operator is used otherwise.
* Everything runs single threaded in float32/float64 without SciPy.
* The wall-clock budget is read from DECONV_DEADLINE_MONOTONIC and
  TIME_BUDGET_SECONDS; the solver adapts the number of iterations and always
  leaves enough time to write the output file and exit normally.
"""
from __future__ import annotations

import os
import sys
import time

_T0 = time.monotonic()

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

# ---------------------------------------------------------------- budget ----
# Conservative total wall-clock cap for the whole process (interpreter start
# included).  The formal limit is at least 0.5 s; we stay well below it.
TOTAL_CAP = 0.35          # seconds of wall clock since process start
MARGIN = 0.045            # seconds reserved for output writing and exit


def _deadline() -> float:
    """Absolute monotonic time at which the solver must stop computing."""
    now = time.monotonic()
    cand = []
    env = os.environ.get("DECONV_DEADLINE_MONOTONIC")
    if env:
        try:
            cand.append(float(env) - MARGIN)
        except ValueError:
            pass
    env = os.environ.get("TIME_BUDGET_SECONDS")
    if env:
        try:
            cand.append(_T0 + float(env) - MARGIN)
        except ValueError:
            pass
    cand.append(_T0 + TOTAL_CAP)
    end = min(cand)
    if end < now + 0.01:            # already late: keep a minimal budget
        end = now + 0.01
    return end


# ------------------------------------------------------------- operators ----
def _sep_factors(psf):
    """Return (gz, gy, gx) if psf is separable, else None."""
    p = np.asarray(psf, dtype=np.float64)
    n = p.shape[0]
    M = p.reshape(n, -1)
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    if s[0] <= 0 or (len(s) > 1 and s[1] / s[0] > 1e-4):
        return None
    f = vt[0]
    g = u[:, 0] * s[0]
    if g[int(np.argmax(np.abs(g)))] < 0:
        g = -g
        f = -f
    if np.abs(g[:, None] * f[None, :] - M).max() > 1e-5 * max(p.max(), 1e-12):
        return None
    F = f.reshape(n, n)
    u2, s2, v2 = np.linalg.svd(F, full_matrices=False)
    if s2[0] <= 0 or (len(s2) > 1 and s2[1] / s2[0] > 1e-4):
        return None
    gy = u2[:, 0] * s2[0]
    gx = v2[0, :]
    if gy[int(np.argmax(np.abs(gy)))] < 0:
        gy = -gy
        gx = -gx
    gz = g
    for v in (gz, gy, gx):
        v /= v.sum()
    rec = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
    if np.abs(rec - p).max() > 1e-5 * max(p.max(), 1e-12):
        return None
    return gz, gy, gx


def _same_matrix(k, n, c):
    """M[i, m] = k[(i + c) - m] (zero outside the support of k)."""
    idx = np.arange(n)[:, None] + c - np.arange(n)[None, :]
    M = np.zeros((n, n), dtype=np.float64)
    ok = (idx >= 0) & (idx < len(k))
    M[ok] = k[idx[ok]]
    return M


class SepOp:
    """Separable 'same' convolution operator with zero extension."""

    def __init__(self, gz, gy, gx, shape):
        self.M = [_same_matrix(g, int(shape[ax]), (len(g) - 1) // 2)
                  for ax, g in enumerate((gz, gy, gx))]

    @staticmethod
    def _apply(x, mats):
        for ax in range(3):
            y = np.moveaxis(x, ax, 0)
            sh = y.shape
            y = (mats[ax] @ y.reshape(sh[0], -1)).reshape(sh)
            x = np.moveaxis(y, 0, ax)
        return x

    def fwd(self, x):
        return self._apply(x, self.M)

    def adj(self, y):
        return self._apply(y, [M.T for M in self.M])


class FFTOp:
    """General 'same' convolution operator via zero padded FFT (exact)."""

    def __init__(self, psf, shape):
        self.shape = tuple(int(s) for s in shape)
        self.c = (psf.shape[0] - 1) // 2
        self.N = tuple(self.shape[i] + psf.shape[i] - 1 for i in range(3))
        full = np.zeros(self.N, dtype=np.float64)
        full[0:psf.shape[0], 0:psf.shape[1], 0:psf.shape[2]] = psf
        self.H = np.fft.rfftn(full)
        self.Hc = np.conj(self.H)

    def fwd(self, x):
        full = np.zeros(self.N, dtype=np.float64)
        full[0:self.shape[0], 0:self.shape[1], 0:self.shape[2]] = x
        y = np.fft.irfftn(np.fft.rfftn(full) * self.H, s=self.N)
        c = self.c
        return y[c:c + self.shape[0], c:c + self.shape[1], c:c + self.shape[2]]

    def adj(self, y):
        full = np.zeros(self.N, dtype=np.float64)
        c = self.c
        full[c:c + self.shape[0], c:c + self.shape[1], c:c + self.shape[2]] = y
        r = np.fft.irfftn(np.fft.rfftn(full) * self.Hc, s=self.N)
        return r[0:self.shape[0], 0:self.shape[1], 0:self.shape[2]]


def make_op(psf, shape):
    s = _sep_factors(psf)
    if s is not None:
        return SepOp(s[0], s[1], s[2], shape)
    return FFTOp(psf, shape)


# -------------------------------------------------------------- estimator ---
def rl_iterate(y, op, iters, mask=None, a1=None, deadline=None, floor=0.0):
    """Richardson-Lucy multiplicative update.

    ``mask`` (optional) selects the voxels that enter the likelihood; the
    update is restricted to them, which is what makes cross validation
    possible.  ``a1`` is the column sum of the (masked) blur operator.
    """
    if a1 is None:
        a1 = op.adj(np.ones_like(y) if mask is None else mask)
    x = np.full_like(y, max(float(y.mean()), 1e-9))
    for it in range(iters):
        if deadline is not None and (it & 3) == 3 and time.monotonic() > deadline:
            break
        mu = op.fwd(x)
        np.maximum(mu, floor, out=mu)
        v = y / mu
        if mask is not None:
            v *= mask
        x *= op.adj(v) / a1
    return x


def rl_cv(y, op, iters, hold_frac, deadline, seed=1):
    """Cross validated iteration count for the RL trajectory.

    Returns (k_star, cv_curve).
    """
    rng = np.random.default_rng(seed)
    hold = rng.random(y.shape) < hold_frac
    mask = (~hold).astype(y.dtype)
    a1 = op.adj(mask)
    x = np.full_like(y, max(float(y.mean()), 1e-9))
    errs = []
    for it in range(iters):
        if (it & 3) == 3 and time.monotonic() > deadline:
            break
        mu = op.fwd(x)
        np.maximum(mu, 1e-12, out=mu)
        v = y / mu
        v *= mask
        x *= op.adj(v) / a1
        mu2 = op.fwd(x)
        e = (y[hold] - mu2[hold]) ** 2
        errs.append(float(e.mean()))
    if not errs:
        return 1, np.zeros(1)
    k = int(np.argmin(errs)) + 1
    return k, np.asarray(errs)


# ------------------------------------------------------------------ main ----
def main(argv):
    if len(argv) < 3:
        sys.stderr.write("usage: solution.py INPUT OUTPUT\n")
        return 2
    in_path, out_path = argv[1], argv[2]
    deadline = _deadline()

    with np.load(in_path, allow_pickle=False) as data:
        observed = np.asarray(data["observed"], dtype=np.float64)
        psf = np.asarray(data["psf"], dtype=np.float64)
    observed = np.nan_to_num(observed, nan=0.0, posinf=0.0, neginf=0.0)
    observed = np.maximum(observed, 0.0)

    op = make_op(psf, observed.shape)
    separable = isinstance(op, SepOp)

    # ---- cross validated stopping point -------------------------------
    # The number of iterations of the multiplicative update is the implicit
    # regulariser; it is selected by cross validation on the observed data.
    hold_frac = 0.1
    max_iter = 64 if separable else 20
    k, _ = rl_cv(observed, op, max_iter, hold_frac, deadline - 0.02)

    # ---- final reconstruction on all voxels ----------------------------
    a1 = op.adj(np.ones_like(observed))
    x = rl_iterate(observed, op, k, a1=a1, deadline=deadline,
                   floor=max(float(observed.max()) * 1e-6, 1e-12))
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(x, 0.0, out=x)

    out = np.asarray(x, dtype=np.float32)
    with open(out_path, "wb") as fh:
        np.save(fh, out, allow_pickle=False)
        fh.flush()
        os.fsync(fh.fileno())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
