#!/usr/bin/env python3
"""3D fluorescence deconvolution under a fixed CPU budget.

Single-file, single-threaded, numpy-only solver.

Method
------
The observation is modelled as

    y = clip( Poisson(g * (A t)) / g + N(0, s^2), 0 )

with ``A`` the ``same`` 3D convolution with the supplied PSF (zero extension
outside the grid).  Neither ``g`` nor ``s`` is given, so the additive noise
scale ``s`` is estimated from the clipped background of ``y`` itself.

The estimate is produced by a flux-normalised Richardson-Lucy iteration with
an additive-noise offset in the denominator

    x <- x * A^T( y / (A x + beta) ) / (A^T 1)^gamma ,   beta = c_beta * s

started from a lightly smoothed copy of the observation and run for a number of
iterations set by the estimated noise level, followed by a mild global power
compression that undoes the systematic flux loss of the early-stopped
iteration.  ``A`` is applied separably when the PSF is (numerically) rank-1
separable - which is the case for the Gaussian PSFs of this task - and through
zero-padded real FFTs otherwise.

Every step is guarded: the whole solve is wrapped so that any unexpected
input still produces a valid non-negative float32 volume, and a hard internal
deadline stops the iteration with the current (already useful) iterate and
leaves time to write the output and exit.
"""

import os
import sys
import time

# Keep every numerical library single threaded before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_START = time.monotonic()

# Total wall clock this process is willing to spend (imports included).  The
# task supplies an absolute monotonic deadline; we also cap ourselves by a
# conservative absolute budget so that an unexpected/absent deadline can never
# push us into the hard limit.
_SELF_CAP = 0.26            # seconds from _START for the whole computation
_WRITE_RESERVE = 0.04       # seconds reserved for writing the output + exit


# --------------------------------------------------------------------------
# convolution operator
# --------------------------------------------------------------------------
class Convolver:
    """'same' 3D convolution with zero extension, plus its adjoint."""

    __slots__ = ("shape", "pad", "sep", "k", "S", "P", "buf", "dtype")

    def __init__(self, psf, shape, dtype=np.float32):
        self.shape = tuple(int(v) for v in shape)
        self.dtype = dtype
        psf = np.asarray(psf, dtype=np.float64)
        ps = psf.shape
        self.pad = tuple(int(k) // 2 for k in ps)
        s = psf.sum()
        if not np.isfinite(s) or s <= 0:
            psf = np.zeros_like(psf)
            psf[tuple(self.pad)] = 1.0
            s = 1.0
        psf = psf / s
        sep, kern = _separable(psf)
        self.sep = sep
        if sep:
            self.k = [kk.astype(dtype) for kk in kern]
        else:
            self.S = tuple(a + b - 1 for a, b in zip(self.shape, ps))
            self.P = np.fft.rfftn(psf.astype(dtype), self.S)
        self.buf = None

    # -- public API -------------------------------------------------------
    def fwd(self, x):
        return self._apply(x, False)

    def bwd(self, x):
        return self._apply(x, True)

    def adjoint_ones(self):
        return self.bwd(np.ones(self.shape, dtype=self.dtype))

    # -- internals --------------------------------------------------------
    def _padded(self):
        p = self.pad
        if self.buf is None:
            self.buf = np.zeros(tuple(a + 2 * b for a, b in zip(self.shape, p)),
                                dtype=self.dtype)
        return self.buf

    def _apply(self, x, adjoint):
        if self.sep:
            return self._apply_sep(x, adjoint)
        return self._apply_fft(x, adjoint)

    def _apply_sep(self, x, adjoint):
        Z, Y, X = self.shape
        pz, py, px = self.k
        if adjoint:
            pz, py, px = pz[::-1], py[::-1], px[::-1]
        p0, p1, p2 = self.pad
        buf = self._padded()
        buf[:] = 0
        buf[p0:p0 + Z, p1:p1 + Y, p2:p2 + X] = x
        t1 = np.zeros((Z, Y, X), dtype=self.dtype)
        for i in range(pz.shape[0]):
            t1 += pz[i] * buf[i:i + Z, p1:p1 + Y, p2:p2 + X]
        buf[:] = 0
        buf[p0:p0 + Z, p1:p1 + Y, p2:p2 + X] = t1
        t2 = np.zeros((Z, Y, X), dtype=self.dtype)
        for i in range(py.shape[0]):
            t2 += py[i] * buf[p0:p0 + Z, i:i + Y, p2:p2 + X]
        buf[:] = 0
        buf[p0:p0 + Z, p1:p1 + Y, p2:p2 + X] = t2
        out = np.zeros((Z, Y, X), dtype=self.dtype)
        for i in range(px.shape[0]):
            out += px[i] * buf[p0:p0 + Z, p1:p1 + Y, i:i + X]
        return out

    def _apply_fft(self, x, adjoint):
        Z, Y, X = self.shape
        S = self.S
        F = np.fft.rfftn(x.astype(self.dtype, copy=False), S)
        F *= np.conj(self.P) if adjoint else self.P
        full = np.fft.irfftn(F, S)
        p0, p1, p2 = self.pad
        return full[p0:p0 + Z, p1:p1 + Y, p2:p2 + X]


def _separable(psf, tol=2e-3):
    """Rank-1 separable decomposition of a non-negative PSF (exact for the
    Gaussian PSFs used here).  Returns (ok, (kz, ky, kx))."""
    pz = psf.sum(axis=(1, 2))
    py = psf.sum(axis=(0, 2))
    px = psf.sum(axis=(0, 1))
    tot = pz.sum()
    if tot <= 0:
        return False, None
    pz = pz / tot
    py = py / tot
    px = px / tot
    if min(pz.min(), py.min(), px.min()) < 0:
        return False, None
    sep = pz[:, None, None] * py[None, :, None] * px[None, None, :]
    denom = np.linalg.norm(psf)
    if denom <= 0 or np.linalg.norm(psf - sep) / denom > tol:
        return False, None
    return True, (pz, py, px)


# --------------------------------------------------------------------------
# noise estimation
# --------------------------------------------------------------------------
def estimate_sigma(y):
    """Additive Gaussian sigma from the clipped (half-normal) background.

    Voxels whose true signal is ~0 are observed as clip(N(0, s), 0), i.e. a
    half-normal with a point mass at zero.  Hence, with f0 = P(y == 0), the
    mass in (0, u] equals 0.5 * f0 exactly when erf(u / (s*sqrt2)) = 0.5,
    i.e. u = 0.6745 * s.
    """
    n = y.size
    f0 = float(np.count_nonzero(y == 0)) / n
    if f0 < 0.02:
        # No clipping information: fall back on a robust high-pass estimate.
        d = y[1::2, 1::2, 1::2] - y[::2, ::2, ::2]
        s = float(np.median(np.abs(d))) / 0.6745 / np.sqrt(2.0)
        return max(s, 1e-6)
    q = 1.5 * f0
    if q > 0.85:
        q = 0.85
    u = float(np.quantile(y, q))
    s = u / 0.6745
    if not np.isfinite(s) or s <= 0:
        s = 1e-6
    return s


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
# Tuned on the 12 public examples plus 48 re-noised/augmented variants built
# from them (see README).
_BETA_C = 0.62          # additive-noise offset, in units of the estimated sigma
_POW = 0.9              # global compression exponent
_CLIP = 1.0             # the generator normalises every clean volume to max 1
_GAMMA = 0.84           # partial boundary-flux preconditioner (see below)
_N_REF = 18.0           # iterations at the reference noise level
_SIG_REF = 0.008        # reference (low-regime) additive noise level
_N_EXP = 0.22           # how strongly the iteration count follows the noise


def _iterations(sigma):
    n = _N_REF * (_SIG_REF / max(sigma, 1e-7)) ** _N_EXP
    if not np.isfinite(n):
        n = _N_REF
    return int(min(max(round(n), 8), 30))


_INIT_SMOOTH = 2.0      # Gaussian width of the RL starting image


def _gaussian_kernel(sigma, n):
    z = np.arange(n, dtype=np.float64) - n // 2
    g = np.exp(-z * z / (2.0 * sigma * sigma))
    return g / g.sum()


def _smooth_init(y, sigma=_INIT_SMOOTH):
    """Smoothed observation as the RL starting point (slightly better than a
    constant start at a fixed iteration count)."""
    n = 2 * int(np.ceil(2.5 * sigma)) + 1
    g = _gaussian_kernel(sigma, n).astype(np.float32)
    kern = g[:, None, None] * g[None, :, None] * g[None, None, :]
    cv = Convolver(kern, y.shape, np.float32)
    return cv.fwd(y)


def solve(y, psf, deadline):
    y = np.asarray(y, dtype=np.float32)
    shape = y.shape
    fallback = np.maximum(y, 0.0)

    if y.size == 0 or not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        fallback = y.copy()
    total = float(y.sum())
    if total <= 0:
        return np.zeros(shape, dtype=np.float32)

    cv = Convolver(psf, shape, np.float32)
    sigma = estimate_sigma(y)

    beta = np.float32(_BETA_C * sigma)
    iters = _iterations(sigma)

    try:
        if time.monotonic() + 0.03 < deadline:
            x = _smooth_init(y)
        else:
            x = np.full(shape, np.float32(total / y.size), dtype=np.float32)
    except BaseException:
        x = np.full(shape, np.float32(total / y.size), dtype=np.float32)
    np.maximum(x, np.float32(1e-4 * total / y.size), out=x)
    # A^T 1 < 1 within one PSF radius of the grid boundary because the forward
    # model extends the volume with zeros.  Dividing the multiplicative update
    # by (A^T 1)^gamma preconditions those slow-converging voxels; gamma < 1
    # because the full correction (gamma = 1) measurably overshoots the truth
    # on the outermost shell.
    dn = cv.adjoint_ones()
    np.maximum(dn, np.float32(1e-6), out=dn)
    if _GAMMA != 1.0:
        np.power(dn, np.float32(_GAMMA), out=dn)

    eps = np.float32(1e-10)
    for k in range(iters):
        if k and time.monotonic() > deadline:
            break
        ax = cv.fwd(x)
        ax += beta
        np.maximum(ax, eps, out=ax)
        np.divide(y, ax, out=ax)
        num = cv.bwd(ax)
        num /= dn
        x *= num

    # the early-stopped iteration systematically loses flux; a mild global
    # compression recovers most of it.  Truth never exceeds 1, so clip first.
    np.minimum(x, np.float32(_CLIP), out=x)
    ref = float(x.max())
    if ref > 0 and np.isfinite(ref):
        np.power(x, np.float32(_POW), out=x)
        x *= np.float32(ref ** (1.0 - _POW))

    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(x, 0.0, out=x)
    if float(x.sum()) <= 0:
        return fallback.astype(np.float32, copy=False)
    return x


def write_output(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    tmp = str(path) + ".tmp.npy"
    try:
        # atomic replace: a valid file exists from the moment of the first write
        with open(tmp, "wb") as fh:
            np.save(fh, arr, allow_pickle=False)
            fh.flush()
        os.replace(tmp, str(path))
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except BaseException:
            pass
        with open(str(path), "wb") as fh:
            np.save(fh, arr, allow_pickle=False)
            fh.flush()


def main(argv):
    if len(argv) < 3:
        sys.stderr.write("usage: solution.py INPUT OUTPUT\n")
        return 2
    inp, outp = argv[1], argv[2]

    budget = 0.5
    try:
        budget = float(os.environ.get("TIME_BUDGET_SECONDS", "") or budget)
    except ValueError:
        pass
    limit = _START + min(budget, 0.5)
    env_dl = os.environ.get("DECONV_DEADLINE_MONOTONIC")
    if env_dl:
        try:
            limit = min(limit, float(env_dl))
        except ValueError:
            pass
    # never plan past the self-imposed cap
    limit = min(limit, _START + _SELF_CAP + _WRITE_RESERVE)
    deadline = limit - _WRITE_RESERVE
    if deadline < _START + 0.01:
        deadline = _START + 0.01

    est = None
    orig_shape = None
    try:
        with np.load(inp, allow_pickle=False) as data:
            y = np.asarray(data["observed"])
            psf = np.asarray(data["psf"])
        orig_shape = y.shape
        if y.ndim == 3:
            est = solve(y, psf, deadline)
        else:                       # unexpected layout: pass the data through
            est = np.maximum(np.nan_to_num(y.astype(np.float32)), 0.0)
    except BaseException:
        est = None

    if est is None:
        # last-resort valid output
        try:
            with np.load(inp, allow_pickle=False) as data:
                est = np.maximum(np.nan_to_num(
                    np.asarray(data["observed"], dtype=np.float32)), 0.0)
        except BaseException:
            est = np.zeros(orig_shape if orig_shape else (32, 32, 32),
                           dtype=np.float32)

    est = np.nan_to_num(np.asarray(est, dtype=np.float32),
                        nan=0.0, posinf=0.0, neginf=0.0)
    if orig_shape is not None and est.shape != orig_shape:
        try:
            est = est.reshape(orig_shape)
        except ValueError:
            est = np.zeros(orig_shape, dtype=np.float32)
    est = np.maximum(est, 0.0)
    write_output(outp, est)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
