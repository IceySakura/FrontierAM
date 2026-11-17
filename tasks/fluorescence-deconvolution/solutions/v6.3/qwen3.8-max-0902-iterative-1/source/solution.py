#!/usr/bin/env python3
"""3D fluorescence deconvolution under a fixed CPU budget.

Single file, numpy only.  scipy is deliberately NOT imported: `import scipy`
alone costs ~0.4 s here, i.e. most of the per-case wall-clock budget.

Algorithm (see README.md):
  1. exact forward operator A ('same' 3D linear convolution, zero extension),
     built as a Kronecker product of three banded Toeplitz matrices whenever
     the PSF is rank-1 separable (6x faster than a padded rFFT), with an exact
     padded-rFFT fallback otherwise;
  2. data-only estimate of the pre-clipping Gaussian noise sd sigma;
  3. correction of the zero-clipping bias  E[max(0,n)] = sigma/sqrt(2*pi);
  4. Richardson-Lucy (Poisson ML) with a deadline-guarded iteration count set
     from the observable PSF blur width and the estimated noise level;
  5. firm-threshold shrinkage between 0.8*sigma and 4.8*sigma plus a small
     global gain, with both thresholds scaled locally by (max(A^T1)/A^T1)^2 so
     boundary voxels -- informed by fewer observations, hence noisier -- are
     thresholded harder.
"""
from __future__ import annotations

import math
import os
import sys
import time

# The task specifies single-threaded numerical libraries.  These must be set
# before numpy (and therefore BLAS) initialises, so that a wider affinity mask
# on the grading core cannot cause thread oversubscription.  setdefault: an
# explicit value from the environment is left alone.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402  (must follow the thread-pool setup above)

INV_SQ2PI = 0.3989422804014327

# Tuned on public-dev/ (12 fixed examples).  Leave-one-out refitting over those
# twelve selects the same values (or adjacent grid points) in all twelve folds,
# and the surrounding plateau is broad.
#
# Shrinkage: firm (Donoho-Johnstone) threshold, locally scaled by w.  Below
# T1*sigma*w the estimate is noise and is zeroed; from there a linear ramp runs
# to T2*sigma*w; above it the value passes through unchanged.  The final GAIN
# undoes the shrinkage bias.
#
# w = (max(A^T 1) / (A^T 1)) ** ALPHA is a boundary-aware noise scaling.  A^T 1
# measures how much of the PSF still lands inside the grid, so it drops near the
# faces, edges and corners: those voxels are informed by fewer observations, the
# RL update divides by A^T 1, and the reconstruction is correspondingly noisier
# there.  ALPHA=2 (a variance-like scaling) is what the data selects; it is
# worth ~7% mean nMSE over a spatially uniform threshold (0.0928 vs 0.1001).
T1 = 0.8
T2 = 4.8
GAIN = 1.06
ALPHA = 2.0
W_CAP = 4.0          # caps w at W_CAP**ALPHA for pathological PSFs
# iteration-count rule
N_COEF = 2.0
N_P = 2.25
N_Q = 0.30
SIG_REF = 0.015
N_MIN = 5
N_MAX = 45


def boundary_weight(op):
    # (max(A^T 1) / (A^T 1)) ** ALPHA, capped -- see the comment above.
    nrm = np.asarray(op.nrm, dtype=np.float32)
    hi = float(nrm.max())
    if not (hi > 0.0):
        return np.ones(op.N, dtype=np.float32)
    w = hi / np.maximum(nrm, 1e-12)
    np.minimum(w, W_CAP, out=w)
    if ALPHA != 1.0:
        w **= ALPHA
    return w


def shrink(x, sigma, w):
    """GAIN * firm-threshold(x; T1*sigma*w, T2*sigma*w).

    Written branchlessly as min((x-t1)+ * T2/(T2-T1), x): the ramp has slope
    T2/(T2-T1), so it crosses the identity exactly at x = t1*T2/T1, making this
    precisely zero / ramp / identity with one temporary.
    """
    z = np.maximum(x - (T1 * sigma) * w, 0.0)
    z *= T2 / (T2 - T1)
    np.minimum(z, x, out=z)
    z *= GAIN
    return z


# --------------------------------------------------------------------------
# forward operator
# --------------------------------------------------------------------------
def _marginals(h):
    """Rank-1 factors of a separable PSF, each normalised to sum 1."""
    mz = h.sum(axis=(1, 2))
    my = h.sum(axis=(0, 2))
    mx = h.sum(axis=(0, 1))
    out = []
    for m in (mz, my, mx):
        s = m.sum()
        if not (s > 0.0):
            return None
        out.append(m / s)
    return out


def _toeplitz(k, n):
    """Banded Toeplitz matrix of 1D convolution by odd kernel k, zero extension."""
    p = (len(k) - 1) // 2
    idx = np.arange(n)[:, None] - np.arange(n)[None, :] + p
    ok = (idx >= 0) & (idx < len(k))
    return np.where(ok, k[np.clip(idx, 0, len(k) - 1)], 0.0)


class KronOp:
    """Separable PSF: A = Mz (x) My (x) Mx as three small Toeplitz matvecs."""

    __slots__ = ("N", "M", "Mt", "dtype", "nrm", "kind")

    def __init__(self, shape, h, dtype=np.float32):
        self.N = tuple(int(v) for v in shape)
        self.dtype = np.dtype(dtype)
        self.kind = "kron"
        h64 = np.asarray(h, dtype=np.float64)
        ks = _marginals(h64)
        if ks is None:
            raise ValueError("psf is not rank-1 separable")
        recon = np.einsum("i,j,k->ijk", ks[0], ks[1], ks[2])
        if np.abs(recon - h64).max() > 1e-6 * h64.max() + 1e-15:
            raise ValueError("psf is not rank-1 separable")
        self.M = [_toeplitz(ks[a], self.N[a]).astype(self.dtype) for a in range(3)]
        self.Mt = [np.ascontiguousarray(m.T) for m in self.M]
        self.nrm = self.adj(np.ones(self.N, dtype=self.dtype))

    def fwd(self, x):
        r = (self.M[0] @ x.reshape(self.N[0], -1)).reshape(self.N)
        r = np.matmul(self.M[1], r)
        return np.matmul(r, self.Mt[2])

    def adj(self, x):
        r = np.matmul(x, self.M[2])
        r = np.matmul(self.Mt[1], r)
        return (self.Mt[0] @ r.reshape(self.N[0], -1)).reshape(self.N)


class FftOp:
    """Exact fallback for a non-separable PSF (zero-padded real FFT)."""

    __slots__ = ("N", "S", "c", "H", "Hc", "dtype", "nrm", "kind")

    def __init__(self, shape, h, dtype=np.float32):
        self.N = tuple(int(v) for v in shape)
        self.dtype = np.dtype(dtype)
        self.kind = "fft"
        N = np.asarray(self.N)
        P = np.asarray(h.shape)
        S = N + P - 1
        self.S = tuple(int(v) for v in S)
        c = (P - 1) // 2
        self.c = tuple(int(v) for v in c)
        hp = np.zeros(self.S, dtype=self.dtype)
        hp[tuple(slice(0, int(p)) for p in P)] = h.astype(self.dtype)
        self.H = np.fft.rfftn(hp)
        self.Hc = np.conj(self.H)
        self.nrm = self.adj(np.ones(self.N, dtype=self.dtype))

    def fwd(self, x):
        buf = np.zeros(self.S, dtype=x.dtype)
        buf[tuple(slice(0, n) for n in self.N)] = x
        y = np.fft.irfftn(np.fft.rfftn(buf) * self.H, s=self.S)
        return np.ascontiguousarray(
            y[tuple(slice(ci, ci + n) for ci, n in zip(self.c, self.N))])

    def adj(self, y):
        buf = np.zeros(self.S, dtype=y.dtype)
        buf[tuple(slice(ci, ci + n) for ci, n in zip(self.c, self.N))] = y
        x = np.fft.irfftn(np.fft.rfftn(buf) * self.Hc, s=self.S)
        return np.ascontiguousarray(x[tuple(slice(0, n) for n in self.N)])


def make_op(shape, h, dtype=np.float32):
    try:
        return KronOp(shape, h, dtype)
    except ValueError:
        return FftOp(shape, h, dtype)


# --------------------------------------------------------------------------
# noise estimation
# --------------------------------------------------------------------------
# inverse standard-normal cdf (Acklam's algorithm, |rel. err| < 1.15e-9)
_PA = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
       1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_PB = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
       6.680131188771972e+01, -1.328068155288572e+01)
_PC = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
       -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_PD = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
       3.754408661907416e+00)


def _ppf(p):
    p = float(p)
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_PC[0]*q+_PC[1])*q+_PC[2])*q+_PC[3])*q+_PC[4])*q+_PC[5]) / \
               ((((_PD[0]*q+_PD[1])*q+_PD[2])*q+_PD[3])*q+1.0)
    if p > 1.0 - 0.02425:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((_PC[0]*q+_PC[1])*q+_PC[2])*q+_PC[3])*q+_PC[4])*q+_PC[5]) / \
                ((((_PD[0]*q+_PD[1])*q+_PD[2])*q+_PD[3])*q+1.0)
    q = p - 0.5
    r = q * q
    return (((((_PA[0]*r+_PA[1])*r+_PA[2])*r+_PA[3])*r+_PA[4])*r+_PA[5])*q / \
           (((((_PB[0]*r+_PB[1])*r+_PB[2])*r+_PB[3])*r+_PB[4])*r+1.0)


_QP = (0.02, 0.06, 0.12, 0.25, 0.35)
# w = n_positive/n_background.  The grid stops where the largest quantile
# argument 0.5 + p*w would leave the domain of the half-normal model, so every
# grid point is fitted against the *same* set of quantiles -- otherwise
# under-determined fits (fewer quantiles) win spuriously.
_WMAX = (0.999 - 0.5) / max(_QP)
_WGRID = np.arange(0.5, _WMAX + 1e-9, 0.02)
_ZTAB = np.asarray([[_ppf(0.5 + p * w) for p in _QP] for w in _WGRID])
_ZT = np.ascontiguousarray(_ZTAB.T)
_ZD = (_ZTAB * _ZTAB).sum(axis=1)


def estimate_sigma(y):
    """Pre-clipping Gaussian sd, from the low tail of the positive part of y.

    observed = max(0, blurred + noise), so the smallest strictly positive
    voxels are background voxels whose value is |N(0,sigma)|.  Their p-th
    quantile is sigma*ppf(0.5 + p*w) where w = n_positive/n_background >= 0.5
    accounts for signal voxels leaking into the tail.  (sigma, w) are fitted to
    five low quantiles by relative weighted least squares; sigma is taken from
    the w with the smallest relative residual.
    """
    ys = np.sort(np.asarray(y, dtype=np.float32).ravel())
    if ys.size == 0:
        return 1e-12
    pos = ys[np.searchsorted(ys, 0.0, side="right"):]
    k = int(pos.size)
    fallback = float(np.std(ys)) * 0.5 + 1e-12
    if k < 200:
        return fallback
    q = np.asarray([pos[min(int(p * k), k - 1)] for p in _QP], dtype=np.float64)
    if not (q[0] > 0.0):
        return fallback
    s = (q @ _ZT) / _ZD
    r = s[:, None] * _ZTAB - q
    e = (r * r).sum(axis=1) / float((q * q).sum())
    out = float(s[int(np.argmin(e))])
    return out if out > 0.0 else fallback


# --------------------------------------------------------------------------
# reconstruction
# --------------------------------------------------------------------------
def rl(y, op, niter, bg, t_end=None):
    """Richardson-Lucy (Poisson ML) on max(y-bg, 0); keeps x >= 0."""
    yy = np.maximum(y - bg, 0.0)
    tot = float(yy.sum())
    if not (tot > 0.0):
        return np.zeros(op.N, dtype=yy.dtype)
    eps = 1e-8 * (tot / yy.size) + 1e-15
    nrm = np.maximum(op.nrm, 1e-9)
    x = np.full(op.N, tot / max(float(op.nrm.sum()), 1e-12), dtype=yy.dtype)
    for k in range(niter):
        Ax = op.fwd(x)
        np.maximum(Ax, eps, out=Ax)
        np.divide(yy, Ax, out=Ax)
        x *= op.adj(Ax)
        x /= nrm
        if t_end is not None and (k & 3) == 3 and time.monotonic() > t_end:
            break
    return x


def psf_width(h):
    """RMS width of the PSF: sqrt of the total second moment over the 3 axes.

    Directly observable from the input PSF, and the quantity that sets how many
    Richardson-Lucy iterations are needed to undo the blur.
    """
    tot = float(h.sum())
    if not (tot > 0.0):
        return 2.0
    v = 0.0
    for a in range(3):
        n = h.shape[a]
        idx = (np.arange(n, dtype=np.float64) - (n - 1) // 2) ** 2
        m = h.sum(axis=tuple(j for j in range(3) if j != a)).astype(np.float64)
        v += float((m * idx).sum()) / tot
    return math.sqrt(max(v, 1e-12))


def n_iter(width, sigma):
    """Iteration count: more blur needs more iterations, more noise fewer.

    Fitted on public-dev/; captures 0.110 of the 0.108 reachable by choosing
    the best iteration count per case with the clean volume in hand, versus
    0.131 for any single fixed count.
    """
    n = N_COEF * (width ** N_P) * ((SIG_REF / max(sigma, 1e-12)) ** N_Q)
    if not math.isfinite(n):
        return 12
    return int(min(max(round(n), N_MIN), N_MAX))


def solve(y, h, t_end):
    op = make_op(y.shape, h, np.float32)
    sigma = estimate_sigma(y)
    # keep a pathological sigma estimate from producing an absurd shift
    scale = float(np.mean(y)) + 1e-12
    sigma = min(max(sigma, 1e-6 * scale), 4.0 * scale)
    bg = INV_SQ2PI * sigma
    # never subtract more than the data can hold (only bites on degenerate input)
    bg = min(bg, 0.5 * float(np.max(y)) if y.size else 0.0)
    x = rl(y, op, n_iter(psf_width(h), sigma), bg, t_end)
    thr = shrink(x, sigma, boundary_weight(op))
    # degenerate-input guard: never let the shrinkage erase everything
    if float(thr.sum()) > 0.0 or float(x.sum()) <= 0.0:
        return thr
    return x


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def _deadline(t_start):
    """Monotonic instant by which `solve` must have returned."""
    dl = None
    raw = os.environ.get("DECONV_DEADLINE_MONOTONIC")
    if raw:
        try:
            dl = float(raw)
        except ValueError:
            dl = None
    if dl is None:
        try:
            b = float(os.environ.get("TIME_BUDGET_SECONDS"))
        except (TypeError, ValueError):
            b = 0.5
        dl = t_start + max(b - 0.12, 0.05)
    # margin for np.save + interpreter teardown inside the hard wall-clock limit
    return max(min(dl - 0.09, t_start + 0.33), t_start + 0.02)


def main() -> None:
    argv = sys.argv
    if len(argv) < 3:
        sys.stderr.write("usage: solution.py INPUT.npz OUTPUT.npy\n")
        sys.exit(2)
    out_path = argv[2]
    t_end = _deadline(time.monotonic())
    y = None
    try:
        with np.load(argv[1], allow_pickle=False) as data:
            y = np.ascontiguousarray(np.asarray(data["observed"]), dtype=np.float32)
            h = np.ascontiguousarray(np.asarray(data["psf"]), dtype=np.float64)
        shape = tuple(y.shape)
        x = solve(y, h, t_end)
        x = np.asarray(x, dtype=np.float32).reshape(shape)
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(x, 0.0, out=x)
    except Exception:
        x = np.zeros_like(y) if y is not None else np.zeros((1, 1, 1), np.float32)
    np.save(out_path, x)


if __name__ == "__main__":
    main()
