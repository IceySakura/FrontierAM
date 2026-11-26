#!/usr/bin/env python3
"""Three-dimensional fluorescence deconvolution under a fixed CPU budget.

Usage:  python3 solution.py INPUT.npz OUTPUT.npy

The input .npz holds ``observed`` (Z,Y,X) float32 and ``psf`` (odd sided,
sums to one).  The forward model is a zero-extension 'same' convolution with
the PSF, followed by Poisson + Gaussian noise and clipping at zero.

Algorithm
---------
1.  Noise identification.  A cheap non-negative Tikhonov reconstruction
    (FISTA) gives a first model of the blurred image.  The residual variance
    is regressed against the modelled intensity, which yields the Poisson
    gain ``g`` of the model  var(y|b) = g*b + sigma^2, and the read noise
    level ``sigma`` is obtained from the clipped background samples of the
    observation.  Both parameters are therefore derived from the current
    observation alone; nothing is assumed about the noise level.

2.  Reconstruction.  The hidden volume is the minimiser of

        min_{x >= 0}  1/2 || H x - y ||_2^2 + lam * TV(x) + mu * ||x||_1

    with H the (zero extension, 'same') convolution operator of the supplied
    PSF.  The total variation is handled with the standard IRLS
    re-weighting (each sub-problem is quadratic), the L1 term by its exact
    proximal operator (a soft threshold, which on the non-negative orthant
    also enforces the non-negativity), and the whole problem is solved with
    FISTA, i.e. accelerated projected gradient.  The data term is
    deliberately *unweighted*: with the noise levels of this task a
    Poisson-Gaussian weighting over-weights the (signal free) background,
    which makes the fit follow the read noise there.

3.  Adaptive regularisation.  The weight ``lam`` of the TV term and the
    weight ``mu`` of the L1 term are derived from the identified noise level,
    from the width of the PSF and from the strength of the observed signal,
    so that the same algorithm works across structure types, PSF widths and
    noise regimes (see tv_weight()).

All convolutions use rFFTs on the zero padded grid (size n + k - 1 per
axis), so the operator and its adjoint are exact and one FISTA iteration
costs two FFTs.  The number of iterations is derived from the wall clock
budget that is still available, and a valid output file is always written
before the deadline.
"""

import os
import sys
import time

_T0 = time.monotonic()

import numpy as np


# --------------------------------------------------------------------------
# forward operator
# --------------------------------------------------------------------------
class Operator:
    """H x = zero-extension 'same' convolution with the PSF, plus its adjoint.

    The linear convolution of the zero padded signal with the (flipped)
    kernel has length n + k - 1 in every dimension; an rFFT of exactly that
    size reproduces the linear convolution without wrap-around, so H and
    H^T are computed exactly (the crop offsets implement the zero
    extension of the forward and adjoint operators).
    """

    def __init__(self, psf, shape):
        self.shape = tuple(int(v) for v in shape)
        Z, Y, X = self.shape
        kz, ky, kx = psf.shape
        self.c = (kz // 2, ky // 2, kx // 2)
        self.N = (Z + kz - 1, Y + ky - 1, X + kx - 1)
        Kf = np.zeros(self.N, np.float32)
        Kf[:kz, :ky, :kx] = psf[::-1, ::-1, ::-1]
        K = np.zeros(self.N, np.float32)
        K[:kz, :ky, :kx] = psf
        self.Kfhat = rfftn(Kf)
        self.Khat = rfftn(K)
        self.buf = np.zeros(self.N, np.float32)
        self.sl3 = (slice(0, Z), slice(0, Y), slice(0, X))
        self.ds = []
        for ax in range(3):
            s1 = [slice(None)] * 3
            s2 = [slice(None)] * 3
            s1[ax] = slice(0, -1)
            s2[ax] = slice(1, None)
            self.ds.append((tuple(s1), tuple(s2)))
        self.t_iter = 0.0013      # two rFFTs of a 40^3 grid

    def fwd(self, x):
        b = self.buf
        b[self.sl3] = x
        C = irfftn(rfftn(b) * self.Kfhat, s=self.N)
        cz, cy, cx = self.c
        Z, Y, X = self.shape
        return C[cz:cz + Z, cy:cy + Y, cx:cx + X]

    def adj(self, y):
        b = self.buf
        b[self.sl3] = y
        C = irfftn(rfftn(b) * self.Khat, s=self.N)
        cz, cy, cx = self.c
        Z, Y, X = self.shape
        return C[cz:cz + Z, cy:cy + Y, cx:cx + X]


class SeparableOperator:
    """Fast path for PSFs that factorise into three 1-D kernels.

    A PSF that is (numerically) the product of three one dimensional
    kernels - which is the case for the Gaussian shaped point spread
    functions of this task - makes the convolution separable, so that it can
    done with three 1-D passes instead of two 3-D FFTs.  That is slightly
    slower per iteration but it avoids importing ``scipy.fft`` (which costs
    about 0.13 s of process start-up), and it keeps the whole program inside
    the wall-clock budget with a much larger margin.  The kernels are
    recovered from the marginals of the PSF and the factorisation is
    verified before the fast path is taken.
    """

    def __init__(self, psf, shape):
        self.shape = tuple(int(v) for v in shape)
        self.k = []
        for a in range(3):
            m = psf.sum(axis=tuple(j for j in range(3) if j != a)).astype(np.float64)
            ssum = m.sum()
            if ssum <= 0:
                raise ValueError('degenerate psf')
            self.k.append((m / ssum).astype(np.float32))
        rec = (self.k[0][:, None, None] * self.k[1][None, :, None] *
               self.k[2][None, None, :])
        err = float(np.abs(rec - psf).max())
        if err > 1e-6 * max(float(psf.max()), 1e-12):
            raise ValueError('psf is not separable')
        self.kr = [np.ascontiguousarray(k[::-1]) for k in self.k]
        Z, Y, X = self.shape
        kz, ky, kx = psf.shape
        self.N = (Z + kz - 1, Y + ky - 1, X + kx - 1)
        self.ds = []
        for ax in range(3):
            s1 = [slice(None)] * 3
            s2 = [slice(None)] * 3
            s1[ax] = slice(0, -1)
            s2[ax] = slice(1, None)
            self.ds.append((tuple(s1), tuple(s2)))
        self.t_iter = 0.0021      # six 1-D passes of a 32^3 grid

    def _conv1(self, a, k, ax):
        """Zero-extension 'same' convolution of a along axis ax."""
        n = a.shape[ax]
        r = (len(k) - 1) // 2
        pad = [(0, 0)] * 3
        pad[ax] = (r, r)
        ap = np.pad(a, pad)
        out = np.zeros_like(a)
        for i in range(len(k)):
            sl = [slice(None)] * 3
            sl[ax] = slice(i, i + n)
            out += k[i] * ap[tuple(sl)]
        return out

    def fwd(self, x):
        t = self._conv1(x, self.k[0], 0)
        t = self._conv1(t, self.k[1], 1)
        return self._conv1(t, self.k[2], 2)

    def adj(self, y):
        t = self._conv1(y, self.kr[0], 0)
        t = self._conv1(t, self.kr[1], 1)
        return self._conv1(t, self.kr[2], 2)


def grad_lap(x):
    """Gradient of 0.5 * sum_axes sum_i (x_{i+1} - x_i)^2."""
    out = np.zeros_like(x)
    for ax in range(3):
        s1 = [slice(None)] * 3
        s2 = [slice(None)] * 3
        s1[ax] = slice(0, -1)
        s2[ax] = slice(1, None)
        d = x[tuple(s2)] - x[tuple(s1)]
        out[tuple(s1)] -= d
        out[tuple(s2)] += d
    return out


def psf_width(psf):
    """Geometric mean of the per-axis second moments of the (centred) PSF.

    It measures how wide the instrument blur is; the amount of regularisation
    needed to invert it depends strongly on it.
    """
    c = [(k - 1) // 2 for k in psf.shape]
    var = []
    for a in range(3):
        m = psf.sum(axis=tuple(j for j in range(3) if j != a)).astype(np.float64)
        ssum = m.sum()
        if ssum <= 0:
            return 1.0
        idx = np.arange(len(m), dtype=np.float64) - c[a]
        var.append(float((idx * idx * m).sum() / ssum))
    var = [v if v > 1e-6 else 1e-6 for v in var]
    return float(np.prod(var) ** (1.0 / 3.0))


def tv_weight(sigma, width, strength):
    """Regularisation weight of the TV term.

    The balance between the quadratic data term and the TV prior depends on

    * the noise level ``sigma``,
    * how wide the PSF is (a wider blur is inverted further and its structure
      content lives at lower spatial frequencies), and
    * how strong the signal is (``strength`` = 99th percentile of the
      observation in units of sigma).  Sparse volumes carry little total
      variation, so the TV prior penalises them relatively harder and needs a
      smaller weight, while a bright volume can afford a stronger prior.

    The exponents were fitted on the public development examples (leave-one-
    out cross validated) and the value is clipped to a safe band and shrunk
    towards the default weight, so the rule degrades gracefully outside the
    fitted range.
    """
    lam = (max(sigma, 1e-6) ** 1.567) * (max(width, 0.2) ** -1.598) * \
          (max(strength, 0.0) + 0.5) ** 0.871
    lam = min(max(lam, 0.0006), 0.014)
    # shrink towards the (fixed) default weight
    lam = lam ** 0.75 * 0.0025 ** 0.25
    return float(min(max(lam, 0.0006), 0.014))


# --------------------------------------------------------------------------
# solvers
# --------------------------------------------------------------------------
def solve_plain(op, y, lam, iters, deadline):
    """Non-negative Tikhonov (least squares) reconstruction, FISTA."""
    x = np.maximum(y, 0).astype(np.float32)
    L = 1.05 + lam * 12.0
    step = 1.0 / L
    xk = x.copy()
    z = x.copy()
    t = 1.0
    for k in range(iters):
        g = op.adj(op.fwd(z) - y)
        g += lam * grad_lap(z)
        xn = z - step * g
        np.maximum(xn, 0, out=xn)
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = xn + ((t - 1.0) / tn) * (xn - xk)
        xk = xn
        t = tn
        if deadline is not None and (k & 1) == 1 and time.monotonic() > deadline:
            break
    return xk


def solve_tv(op, y, lam, outer, iters, deadline, eps_rel=0.003, l1=0.0):
    """Non-negative TV + L1 regularised least squares.

    FISTA (accelerated projected gradient) on

        min_{x >= 0}  1/2||Hx - y||^2 + lam*TV(x) + l1*||x||_1 ,

    the TV term being handled by IRLS re-weighting (quadratic sub-problem)
    and the L1 term by its exact proximal operator (a soft threshold, which
    on the non-negative orthant also enforces the non-negativity).
    """
    shape = op.shape
    x = np.maximum(y, 0).astype(np.float32)
    tvw = []
    for ax in range(3):
        sh = [1, 1, 1]
        sh[ax] = shape[ax] - 1
        tvw.append(np.ones(sh, np.float32))
    ds = op.ds

    for o in range(outer):
        Lprior = 0.0
        for ax in range(3):
            Lprior += 4.0 * float(tvw[ax].max())
        L = 1.05 + lam * max(Lprior, 1e-6)
        if not np.isfinite(L) or L <= 0:
            L = 1.0
        step = 1.0 / L
        xk = x.copy()
        z = x.copy()
        t = 1.0
        for k in range(iters):
            g = op.adj(op.fwd(z) - y)
            for ax in range(3):
                s1, s2 = ds[ax]
                d = z[s2] - z[s1]
                d *= (lam * tvw[ax])
                g[s1] -= d
                g[s2] += d
            xn = z - step * g
            if l1:
                xn -= step * l1
            np.maximum(xn, 0, out=xn)
            tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = xn + ((t - 1.0) / tn) * (xn - xk)
            xk = xn
            t = tn
            if deadline is not None and (k & 1) == 1 and time.monotonic() > deadline:
                break
        x = xk
        if o < outer - 1:
            mx = max(float(x.max()), 1e-12)
            for ax in range(3):
                s1, s2 = ds[ax]
                d = np.abs(x[s2] - x[s1])
                tvw[ax] = (1.0 / np.maximum(d, eps_rel * mx)).astype(np.float32)
    return x


# --------------------------------------------------------------------------
# noise identification
# --------------------------------------------------------------------------
def estimate_noise(op, y, x0):
    """Return (poisson gain g, gaussian noise std sigma) from the data."""
    y = y.astype(np.float64)
    b = op.fwd(x0).astype(np.float64)
    r = y - b
    # --- read noise: the background samples are clipped gaussian noise
    thr = np.quantile(b, 0.5)
    m = b <= thr
    sel = y[m]
    pos = sel[sel > 0]
    if pos.size >= 32:
        sigma = float(np.sqrt(np.mean(pos * pos)))
    else:
        sigma = 0.0
    # --- poisson gain: regression of residual variance on modelled level
    nb = 10
    qs = np.quantile(b, np.linspace(0.0, 1.0, nb + 1))
    bs, vs, ns = [], [], []
    for k in range(nb):
        lo = qs[k]
        hi = qs[k + 1]
        msk = (b >= lo) & (b < hi) if k < nb - 1 else (b >= lo)
        if msk.sum() < 64:
            continue
        bs.append(float(b[msk].mean()))
        vs.append(float(r[msk].var()))
        ns.append(float(msk.sum()))
    g = 0.0
    if len(bs) >= 3:
        bs = np.asarray(bs)
        vs = np.asarray(vs)
        ns = np.asarray(ns)
        wts = np.sqrt(ns)
        A = np.stack([bs, np.ones_like(bs)], 1) * wts[:, None]
        bvec = vs * wts
        try:
            sol = np.linalg.lstsq(A, bvec, rcond=None)[0]
            g = float(sol[0])
        except Exception:
            g = 0.0
    if not np.isfinite(g) or g < 0.0:
        g = 0.0
    if g <= 0.0:
        hi_m = b > np.quantile(b, 0.9)
        if hi_m.sum() > 32:
            g = float(np.var(r[hi_m]) / max(b[hi_m].mean(), 1e-12))
    if not np.isfinite(sigma) or sigma <= 0.0:
        q = r[m] if m.any() else r.ravel()
        sigma = float(np.median(np.abs(q - np.median(q)))) / 0.6745
    return g, max(sigma, 1e-6)


# --------------------------------------------------------------------------
def main(argv):
    if len(argv) < 3:
        sys.stderr.write('usage: solution.py INPUT.npz OUTPUT.npy\n')
        return 2
    inp, outp = argv[1], argv[2]

    # ---- wall clock budget ------------------------------------------------
    dl = os.environ.get('DECONV_DEADLINE_MONOTONIC')
    deadline = float(dl) if dl else float('inf')
    if not np.isfinite(deadline):
        deadline = _T0 + 0.42
    else:
        deadline = min(deadline, _T0 + 0.42)
    deadline -= 0.03          # reserve time for writing the result

    with np.load(inp, allow_pickle=False) as z:
        obs = np.ascontiguousarray(z['observed'], dtype=np.float32)
        psf = np.ascontiguousarray(z['psf'], dtype=np.float32)
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(obs, 0, out=obs)
    psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(psf, 0, out=psf)
    s = float(psf.sum())
    if s > 0:
        psf /= np.float32(s)

    # Prefer the separable fast path (no scipy.fft import, which costs
    # ~0.13 s of start-up); fall back to the general FFT operator.
    global rfftn, irfftn
    try:
        op = SeparableOperator(psf, obs.shape)
    except Exception:
        from scipy.fft import rfftn, irfftn
        op = Operator(psf, obs.shape)
    width = psf_width(psf)
    # rough cost of one FISTA iteration (two convolutions), scaled with the
    # size of the padded grid
    grid = float(op.N[0]) * op.N[1] * op.N[2] / (40.0 ** 3)
    t_iter = op.t_iter * grid

    # ---- stage 1: quick reconstruction used for the noise identification --
    avail = deadline - time.monotonic()
    n1 = int(np.clip(avail * 0.12 / t_iter, 5, 12))
    x0 = solve_plain(op, obs, 0.02, n1, deadline - 0.4 * max(avail, 0.0))
    g, sigma = estimate_noise(op, obs, x0)

    # ---- stage 2: main reconstruction -------------------------------------
    avail = deadline - time.monotonic()
    if avail > 0.02:
        outer = 2
        iters = int(np.clip((avail - 0.02) / (outer * t_iter), 6, 40))
        strength = float(np.quantile(obs, 0.99)) / max(sigma, 1e-12)
        lam = tv_weight(sigma, width, strength)
        x = solve_tv(op, obs, lam, outer, iters, deadline, l1=0.7 * sigma)
    else:
        x = x0
        x = np.maximum(x - 0.55 * sigma, 0.0)

    # The TV + L1 regularisation shrinks the recovered amplitudes (the L1
    # term subtracts a constant mu from every voxel and the TV term smooths
    # the peaks).  A small gain, proportional to the read noise, compensates
    # that shrinkage; its value was fitted on the public development
    # examples, where the optimum is flat (a gain of 1.04 to 1.08 for the
    # two noise regimes).
    x = x * (1.0 + 4.5 * sigma)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    with open(outp, 'wb') as fh:
        np.save(fh, x, allow_pickle=False)
        fh.flush()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
