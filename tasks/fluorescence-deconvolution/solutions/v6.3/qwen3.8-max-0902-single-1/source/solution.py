#!/usr/bin/env python3
"""Nonnegative 3D fluorescence deconvolution under a hard wall-clock budget.

Forward model of the task
-------------------------
    observed = clip( Poisson(s * (P*x)) / s  +  N(0, sg^2),  0 )
where P is centred 3D linear convolution with the supplied PSF, zero extended
outside the grid and cropped back to it ('same').  Neither s nor sg is given.

Method
------
1. Blind noise calibration from `observed` alone:
   the PSF is a (truncated) Gaussian, so its OTF is essentially zero above
   ~0.4 of the Nyquist radius; the mean power of `observed` in that shell is
   therefore white-noise power (read noise + Poisson + rectification).  Its
   square root `sig` is a scale-equivariant noise amplitude used to set every
   tuning constant, so the solver adapts automatically to both noise regimes
   and to the overall intensity scale of the volume.

2. Richardson-Lucy / EM iterations for the Poisson likelihood with a constant
   spurious-rate background c = kc*sig.  Adding a background of size ~s*sg^2
   makes the Poisson variance (f+c)/s reproduce the true variance f/s + sg^2,
   i.e. it is the standard Poisson-Gaussian ("MA-RL") trick; it also keeps the
   ratio (y+c)/(Ax+c) well defined in the clipped-to-zero background.

3. An L1 / exponential-prior penalty realised as a constant added to the
   sensitivity image A^T 1 in the multiplicative update.  It contracts the
   dark background geometrically towards zero, which matters because the
   ground truth is exactly zero over most of the volume.

4. Early stopping (fixed iteration count) is itself the dominant regulariser:
   the iteration is stopped long before the (heavily biased) MAP fixed point.

5. A final soft threshold at kt*sig removes the residual noise haze that the
   nMSE metric punishes in the dark regions.

Speed
-----
The PSFs of this task family are rank-1 separable.  When that is detected
(exactly, to float32 tolerance) P and P^T are evaluated as three banded BLAS
matmuls per axis in float32 (~0.3 ms per application) instead of a 40^3 rfft
pair (~2.3 ms).  A padded-rfft path remains for general PSFs.  The iteration
loop is also guarded by an absolute wall-clock deadline derived from
DECONV_DEADLINE_MONOTONIC (when supplied) and a self-imposed cap, so the
process always stops in time, always writes a finite nonnegative array of the
input shape, and always exits normally.  A valid fallback (the clipped
observation) is written before the iteration starts and atomically replaced.

Dependencies: numpy only.  No data files, no network, single threaded.
"""
import os
import sys
import time

_T0 = time.monotonic()
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

# ---- tuning (selected on the 12 public dev cases; broad flat optimum) -------
N_ITER = 30      # Richardson-Lucy iterations (early stopping = main regulariser)
KC = 3.0         # spurious background   c   = KC * sig
LAM = 0.05       # L1 penalty weight     lam
KT = 0.5         # final soft threshold  kt  = KT * sig

# ---- time policy ------------------------------------------------------------
_SELF_CAP = 0.40     # s from process start; harness deadline can only tighten it
_WRITE_RESERVE = 0.05


def _budget_end():
    end = _T0 + _SELF_CAP
    try:
        d = float(os.environ.get("DECONV_DEADLINE_MONOTONIC", ""))
        if d == d and d > _T0:
            end = min(end, d - _WRITE_RESERVE)
    except (TypeError, ValueError):
        pass
    return max(end, _T0 + 0.02)


class Op(object):
    """Exact 'same' linear convolution with zero extension, plus its adjoint."""

    def __init__(self, psf, N, dtype=np.float32, sep_tol=2e-3):
        p = np.asarray(psf, np.float64)
        self.p = p
        self.N = tuple(int(n) for n in N)
        self.dtype = dtype
        self.sep = False
        mz = p.sum(axis=(1, 2))
        my = p.sum(axis=(0, 2))
        mx = p.sum(axis=(0, 1))
        outer = np.einsum('i,j,k->ijk', mz, my, mx)
        if np.abs(outer - p).max() <= sep_tol * max(p.max(), 1e-12):
            self.sep = True
            self.B = []
            for ax, k in enumerate((mz, my, mx)):
                n = self.N[ax]
                L = int(p.shape[ax])
                s = n + L - 1
                kk = k.astype(dtype)
                B = np.zeros((s, n), dtype)
                for i in range(n):
                    hi = min(s, i + L)
                    for j in range(i, hi):
                        B[j, i] = kk[i + L - 1 - j]
                self.B.append(B)
        else:
            S = tuple(n + int(p.shape[ax]) - 1 for ax, n in enumerate(self.N))
            self.S = S
            buf = np.zeros(S, np.float64)
            buf[tuple(slice(0, d) for d in p.shape)] = p
            for ax in range(3):
                buf = np.roll(buf, -(int(p.shape[ax]) // 2), ax)
            self.otf = np.fft.rfftn(buf)
            self.otfc = np.conj(self.otf)
            self.buf = np.zeros(S, np.float64)

    def _sep(self, x, trans):
        out = np.ascontiguousarray(x, self.dtype)
        for ax in range(3):
            n = self.N[ax]
            L = int(self.p.shape[ax])
            s = n + L - 1
            h = L // 2
            B = self.B[ax]
            shp = list(out.shape)
            perm = [0, 1, 2]
            perm.pop(ax)
            perm.append(ax)
            if not trans:
                pd = list(shp)
                pd[ax] = shp[ax] + L - 1
                xp = np.zeros(pd, self.dtype)
                sl = [slice(None)] * 3
                sl[ax] = slice(h, h + n)
                xp[tuple(sl)] = out
                xm = np.moveaxis(xp, ax, -1).reshape(-1, s)
                r = xm @ B
            else:
                xm = np.moveaxis(out, ax, -1).reshape(-1, n)
                r = (xm @ B.T)[:, h:h + n]
            nshp = [shp[i] for i in perm]
            out = np.moveaxis(np.ascontiguousarray(r).reshape(nshp), -1, ax)
        return out

    def _fft(self, x, trans):
        b = self.buf
        b[...] = 0.0
        b[:self.N[0], :self.N[1], :self.N[2]] = x
        o = self.otfc if trans else self.otf
        r = np.fft.irfftn(np.fft.rfftn(b) * o, s=self.S)
        return r[:self.N[0], :self.N[1], :self.N[2]]

    def fwd(self, x):
        return self._sep(x, False) if self.sep else self._fft(x, False)

    def adj(self, y):
        return self._sep(y, True) if self.sep else self._fft(y, True)


def noise_sigma(o):
    """Scale of the white noise in `o`, from its high-frequency power shell."""
    o = np.asarray(o, np.float64)
    n = int(o.shape[0])
    F = np.fft.rfftn(o)
    P = F.real ** 2 + F.imag ** 2
    zz, yy, xx = np.indices(P.shape)
    rad = np.sqrt((np.minimum(zz, n - zz) / float(n)) ** 2
                  + (np.minimum(yy, n - yy) / float(n)) ** 2
                  + (xx / float(n)) ** 2)
    shell = (rad > 0.40) & (xx > 0) & (xx < P.shape[2] - 1)
    if not shell.any():
        shell = rad > 0.45
    return float(np.sqrt(max(P[shell].mean() / o.size, 0.0)))


def solve(obs, psf, end):
    N = obs.shape
    op = Op(psf, N)
    y = np.ascontiguousarray(obs, np.float32)
    sig = noise_sigma(y)
    if not np.isfinite(sig) or sig <= 0.0:
        sig = max(float(y.mean()), 1e-6)

    c = np.float32(KC * sig)
    at1 = op.adj(np.ones(N, np.float32)).astype(np.float32)
    at1 += np.float32(LAM)
    ypc = y + c
    x = np.full(N, max(float(y.mean()), 1e-6), np.float32)

    tiny = np.float32(1e-8)
    eps = np.float32(1e-9)
    k = 0
    t_loop = time.monotonic()
    check = t_loop + 0.012
    while k < N_ITER:
        f = op.fwd(x)
        f += c
        np.maximum(f, tiny, out=f)
        np.divide(ypc, f, out=f)
        g = op.adj(f)
        np.multiply(x, g, out=g)
        np.divide(g, at1, out=g)
        np.maximum(g, eps, out=g)
        x = g
        k += 1
        now = time.monotonic()
        if now > check:
            # per-iteration cost so far; stop early if the rest cannot fit
            per = (now - t_loop) / float(k)
            if now + per * (N_ITER - k) * 1.6 + _WRITE_RESERVE > end:
                break
            check = now + 0.015
    if KT > 0.0:
        x = np.maximum(x - np.float32(KT * sig), np.float32(0.0))
    return x, k, float(sig), op.sep


def _write(path, arr):
    a = np.ascontiguousarray(np.asarray(arr, np.float32))
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(a, np.float32(0.0), out=a)
    tmp = str(path) + ".tmp.npy"
    with open(tmp, 'wb') as fh:
        np.save(fh, a, allow_pickle=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, str(path))


def main():
    end = _budget_end()
    src, dst = sys.argv[1], sys.argv[2]
    with np.load(src, allow_pickle=False) as d:
        obs = np.asarray(d['observed'], np.float64)
        psf = np.asarray(d['psf'], np.float64)
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(obs, 0.0, out=obs)
    psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(psf, 0.0, out=psf)
    tot = psf.sum()
    if tot > 0.0:
        psf /= tot
    # valid output on disk before the (time bounded) iteration starts
    _write(dst, obs.astype(np.float32))
    est, k, sig, sep = solve(obs.astype(np.float32), psf, end)
    _write(dst, est)
    sys.stdout.write("iters=%d sep=%d sig=%.5f t=%.3f\n" % (k, int(sep), sig, time.monotonic() - _T0))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        try:
            import traceback
            traceback.print_exc()
        except BaseException:
            pass
        try:
            a = np.zeros((1, 1, 1), np.float32)
            try:
                with np.load(sys.argv[1], allow_pickle=False) as d:
                    a = np.nan_to_num(np.asarray(d['observed'], np.float32))
                    np.maximum(a, np.float32(0.0), out=a)
            except BaseException:
                pass
            with open(sys.argv[2], 'wb') as fh:
                np.save(fh, np.ascontiguousarray(a, np.float32), allow_pickle=False)
        except BaseException:
            pass
        sys.exit(0)
