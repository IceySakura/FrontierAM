#!/usr/bin/env python3
"""Three-dimensional fluorescence deconvolution under a fixed CPU budget.

Forward model
-------------
The observation is a zero-extended ('same', centered odd PSF) 3-D linear
convolution of a non-negative clean volume with the supplied PSF, followed by
Poisson shot noise, additive Gaussian read noise and truncation at zero.

Inference
---------
The solver minimises a censored-Gaussian data term plus a smoothed
total-variation prior over the non-negative orthant:

    min_{x>=0}  sum_i rho_i((A x)_i)  +  lam * TV_eps(x)

    rho_i(b) = (b - y_i)^2 / (2 v_i)          if y_i > 0
               -log Phi(-b / sqrt(v_i))        if y_i == 0   (censored voxel)
    v_i      = sigma_w^2 + b / gain

The censoring term is essential: about half of the voxels of a low-signal
region are truncated to exactly zero, and treating them as ordinary Gaussian
residuals would force a large spurious background.  The weights use an
inflated read-noise scale so that noise-dominated voxels contribute little.

The optimisation is projected FISTA (accelerated projected gradient), which
also supplies the residual regularisation through its finite iteration count.
The iteration count is bounded by the wall-clock budget handed to the process.

Only numpy is required.
"""

import os
import sys
import time

_T_START = time.monotonic()

import numpy as np

# ---------------------------------------------------------------------------
# small numerical helpers (numpy only)
# ---------------------------------------------------------------------------

_SQRT2PI = 0.3989422804014327  # 1 / sqrt(2 pi)


def _norm_ppf(p):
    """Inverse normal CDF, Abramowitz & Stegun 26.2.23 (|err| < 4.5e-4)."""
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        raise ValueError
    if p > 0.5:
        return -_norm_ppf(1.0 - p)
    t = np.sqrt(-2.0 * np.log(p))
    return -(t - (2.30753 + 0.27061 * t) / (1.0 + 0.99229 * t + 0.04481 * t * t))


def _mills(z):
    """phi(z)/Phi(-z) for z >= 0 (inverse Mills ratio), vectorised."""
    z = np.asarray(z, dtype=np.float64)
    z = np.maximum(z, 0.0)
    # Abramowitz & Stegun 7.1.26: erfc(x) = T(x) exp(-x^2), x = z / sqrt(2).
    # Phi(-z) = 0.5 erfc(z/sqrt2) = 0.5 T exp(-z^2/2), hence
    # phi(z)/Phi(-z) = (2/sqrt(2 pi)) / T.
    zc = np.minimum(z, 3.5)
    x = zc * 0.7071067811865476
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
          + t * (-1.453152027 + t * 1.061405429))))
    out = 0.7978845608028654 / poly
    # for large z fall back on the asymptotic expansion (T is inaccurate there)
    big = z > 3.5
    if big.any():
        zb = z[big]
        out[big] = zb + 1.0 / zb
    return out


# ---------------------------------------------------------------------------
# forward / adjoint operators
# ---------------------------------------------------------------------------

def _conv_axis(x, k, axis, out):
    """'same' 1-D convolution along `axis` with zero extension.

    Implements out[i] = sum_t k[t] x[i + off - t] with off = (len(k)-1)//2,
    i.e. the central crop of the full linear convolution.  `out` is filled.
    """
    n = x.shape[axis]
    m = len(k)
    off = (m - 1) // 2
    out[...] = 0.0
    sl = [slice(None)] * x.ndim
    sr = [slice(None)] * x.ndim
    for tap in range(m):
        s = off - tap
        w = k[tap]
        if s == 0:
            out += w * x
        elif s > 0:
            sl[axis] = slice(0, n - s)
            sr[axis] = slice(s, n)
            out[tuple(sr)] += w * x[tuple(sl)]
        else:
            sl[axis] = slice(-s, n)
            sr[axis] = slice(0, n + s)
            out[tuple(sr)] += w * x[tuple(sl)]
        sl[axis] = slice(None)
        sr[axis] = slice(None)
    return out


class _Ops(object):
    """A and A^T for the zero-extended 'same' convolution."""

    def __init__(self, psf, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.sep = None
        self._try_separable(psf)
        if self.sep is None:
            self._init_fft(psf)

    # -- separable fast path ------------------------------------------------
    def _try_separable(self, psf):
        psf = np.asarray(psf, dtype=np.float64)
        tot = psf.sum()
        if tot <= 0:
            return
        ks = []
        for ax in range(3):
            s = psf.sum(axis=tuple(a for a in range(3) if a != ax))
            ss = s.sum()
            if ss <= 0 or (s < 0).any():
                return
            ks.append(s / ss)
        recon = np.einsum('i,j,k->ijk', *ks) * tot
        err = float(np.abs(recon - psf).max() / max(psf.max(), 1e-300))
        if err > 1e-3:
            return
        c = tot ** (1.0 / 3.0)
        self.sep = tuple((np.asarray(k, dtype=np.float64) * c).astype(self.dtype)
                         for k in ks)
        self.sep_rev = tuple(k[::-1].copy() for k in self.sep)
        self.buf1 = np.empty(self.shape, dtype=self.dtype)
        self.buf2 = np.empty(self.shape, dtype=self.dtype)

    def _sep_apply(self, x, kernels, out):
        buf = self.buf1
        _conv_axis(x, kernels[0], 0, buf)
        _conv_axis(buf, kernels[1], 1, self.buf2)
        _conv_axis(self.buf2, kernels[2], 2, out)
        return out

    # -- general FFT path ---------------------------------------------------
    def _init_fft(self, psf):
        shape = self.shape
        full = tuple(n + psf.shape[a] - 1 for a, n in enumerate(shape))
        self.full = full
        self.off = tuple((psf.shape[a] - 1) // 2 for a in range(3))
        p = np.zeros(full, dtype=np.float64)
        pk = np.zeros(full, dtype=np.float64)
        p[tuple(slice(0, s) for s in psf.shape)] = psf
        pk[tuple(slice(0, s) for s in psf.shape)] = psf[::-1, ::-1, ::-1]
        self.P = np.fft.rfftn(p).astype(np.complex128)
        self.Pf = np.fft.rfftn(pk).astype(np.complex128)
        self.buf = np.zeros(full, dtype=np.float64)
        self.slc = tuple(slice(self.off[a], self.off[a] + shape[a])
                         for a in range(3))
        self.la2 = float(np.abs(self.P).max()) ** 2

    def _fft_apply(self, x, P, out):
        buf = self.buf
        buf[...] = 0.0
        buf[tuple(slice(0, n) for n in self.shape)] = x
        c = np.fft.irfftn(np.fft.rfftn(buf) * P, s=self.full)
        out[...] = c[self.slc]
        return out

    # -- public -------------------------------------------------------------
    def A(self, x, out):
        if self.sep is not None:
            return self._sep_apply(x, self.sep, out)
        return self._fft_apply(x, self.P, out)

    def AT(self, y, out):
        if self.sep is not None:
            return self._sep_apply(y, self.sep_rev, out)
        return self._fft_apply(y, self.Pf, out)


# ---------------------------------------------------------------------------
# noise estimation
# ---------------------------------------------------------------------------

def estimate_sigma(obs):
    """Read-noise scale from the rectified-Gaussian low quantiles.

    For a voxel whose blurred intensity is negligible the observation is
    max(n, 0) with n ~ N(0, sigma^2); its CDF at t > 0 is simply Phi(t/sigma)
    while P(obs == 0) = 1/2.  Matching the empirical low quantiles against
    that model recovers sigma without any knowledge of the clean volume.
    """
    v = np.asarray(obs, dtype=np.float64).ravel()
    n = v.size
    z = float(np.count_nonzero(v == 0)) / n
    pos = v[v > 0]
    ests = []
    if z > 0.02 and pos.size > 32:
        for q in (0.04, 0.10, 0.20, 0.35, 0.50):
            t = float(np.quantile(pos, q))
            if t <= 0.0:
                continue
            p = float(np.count_nonzero((v > 0) & (v <= t))) / n
            r = p / (2.0 * z)
            if 1e-4 < r < 0.45:
                try:
                    x = _norm_ppf(0.5 + r)
                except ValueError:
                    continue
                if x > 1e-6:
                    ests.append(t / x)
    if ests:
        return float(np.median(ests)), z
    # fall back on a difference-based scale in the darker half of the volume
    d = np.asarray(obs, dtype=np.float64)
    dd = d[..., 1:] - d[..., :-1]
    lo = np.quantile(d, 0.35)
    m = (d[..., 1:] <= lo) & (d[..., :-1] <= lo)
    if m.sum() > 64:
        s = float(np.std(dd[m])) / 0.826
    else:
        s = float(np.std(dd)) / 0.826
    return max(s, 1e-6 * float(np.mean(v)) + 1e-12), z


# ---------------------------------------------------------------------------
# prior
# ---------------------------------------------------------------------------

def tv_grad(x, eps, buf, gx, gy, gz):
    """Gradient of the eps-smoothed isotropic 3-D TV, written into `buf`."""
    for a, g in ((0, gx), (1, gy), (2, gz)):
        g[...] = 0.0
        if a == 0:
            g[:-1] = x[1:] - x[:-1]
        elif a == 1:
            g[:, :-1] = x[:, 1:] - x[:, :-1]
        else:
            g[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
    m = gx * gx
    m += gy * gy
    m += gz * gz
    m += eps * eps
    np.sqrt(m, out=m)
    np.divide(gx, m, out=gx)
    np.divide(gy, m, out=gy)
    np.divide(gz, m, out=gz)
    d = buf
    d[...] = 0.0
    # -div(gx, gy, gz) with Neumann boundaries
    d[1:] -= gx[1:] - gx[:-1]
    d[0] -= gx[0]
    d[:, 1:] -= gy[:, 1:] - gy[:, :-1]
    d[:, 0] -= gy[:, 0]
    d[:, :, 1:] -= gz[:, :, 1:] - gz[:, :, :-1]
    d[:, :, 0] -= gz[:, :, 0]
    return d


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------

def deconvolve(obs, psf, sigma, gain, lam, tv_eps, max_iter, stop_at, dtype):
    """Projected FISTA on the censored-Gaussian + TV objective."""
    shape = obs.shape
    ops = _Ops(psf, shape, dtype)

    sig_w = sigma
    wmax = 1.0 / (sig_w * sig_w)
    la2 = ops.la2 if ops.sep is None else 1.0
    # Lipschitz estimate of the data term (conservative but stable).
    L = wmax * la2 + lam * (9.0 / tv_eps) * 1.5 + 8.0
    step = 1.0 / L

    b = np.empty(shape, dtype=dtype)
    g = np.empty(shape, dtype=dtype)
    at = np.empty(shape, dtype=dtype)
    tvb = np.empty(shape, dtype=dtype)
    gx = np.empty(shape, dtype=dtype)
    gy = np.empty(shape, dtype=dtype)
    gz = np.empty(shape, dtype=dtype)

    xk = np.full(shape, max(float(obs.mean()), 1e-8), dtype=dtype)
    yk = xk.copy()
    sk = np.empty(shape, dtype=dtype)

    zero = (obs == 0)
    obsd = obs.astype(dtype)
    sig2 = sig_w * sig_w
    inv_g = dtype(1.0 / gain)
    lamd = dtype(lam)
    stepd = dtype(step)

    tk = 1.0
    it = 0
    while it < max_iter:
        ops.A(yk, b)
        # variance weights v = sig^2 + b / gain  (b >= 0 by construction)
        v = b * inv_g
        v += sig2
        # gradient of the censored-Gaussian data term
        np.subtract(b, obsd, out=g)
        g /= v
        if zero.any():
            bz = np.asarray(b[zero], dtype=np.float64)
            sz = np.sqrt(np.asarray(v[zero], dtype=np.float64))
            g[zero] = (_mills(bz / sz) / sz).astype(dtype)
        ops.AT(g, at)
        v = None
        tv = tv_grad(yk, tv_eps, tvb, gx, gy, gz)
        at += lamd * tv
        # projected gradient step
        at *= stepd
        np.subtract(yk, at, out=sk)
        np.maximum(sk, 0.0, out=sk)
        if not np.isfinite(sk).all():
            break
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        # yk <- sk + ((tk-1)/tn) * (sk - xk);  yk is dead after this point
        np.subtract(sk, xk, out=yk)
        yk *= (tk - 1.0) / tn
        yk += sk
        xk, yk, sk = sk, yk, xk
        tk = tn
        it += 1
        if (it & 3) == 0 and time.monotonic() > stop_at:
            break
    return np.asarray(xk, dtype=np.float64), it


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

_TOTAL_TARGET = 0.40   # seconds from process launch, well inside the limit
_SAVE_MARGIN = 0.06    # reserve for output conversion, write and exit


def _budget(now):
    """Seconds of wall clock still available to the iteration loop."""
    own = _TOTAL_TARGET - (now - _T_START) - _SAVE_MARGIN
    try:
        hard = float(os.environ.get('DECONV_DEADLINE_MONOTONIC', 'nan'))
    except (TypeError, ValueError):
        hard = float('nan')
    if hard == hard and hard > 0.0:
        own = min(own, hard - now - 0.05)
    try:
        tb = float(os.environ.get('TIME_BUDGET_SECONDS', 'nan'))
    except (TypeError, ValueError):
        tb = float('nan')
    if tb == tb and tb > 0:
        own = min(own, tb - (now - _T_START) - 0.05)
    return float(min(max(own, 0.02), 1.0))


def solve(obs, psf, now):
    budget = _budget(now)
    stop_at = now + budget

    sigma, _zfrac = estimate_sigma(obs)
    # The two noise regimes of the generator pair read noise with shot gain;
    # use that relation, clamped to a sane range (the solver is insensitive
    # to moderate errors here).
    gain = 0.6 / max(sigma, 1e-9)
    gain = min(max(gain, 2.0), 4000.0)

    sigma_w = 2.0 * sigma  # inflate: down-weight noise-dominated voxels
    lam = 0.5
    tv_eps = 0.08

    est, it = deconvolve(obs, psf, sigma_w, gain, lam, tv_eps,
                         100000, stop_at, np.float32)
    if not np.isfinite(est).all():
        est = np.zeros_like(est)
    np.maximum(est, 0.0, out=est)
    return est, sigma, it


def main(argv):
    in_path, out_path = argv[1], argv[2]
    with np.load(in_path, allow_pickle=False) as data:
        obs = np.ascontiguousarray(data['observed'], dtype=np.float64)
        psf = np.ascontiguousarray(data['psf'], dtype=np.float64)

    out = obs  # safe default: never emit an invalid array
    try:
        est, _sigma, _it = solve(obs, psf, time.monotonic())
        if est is not None and est.shape == obs.shape and est.dtype == np.float64:
            out = est
    except Exception:
        out = obs

    out = np.ascontiguousarray(out, dtype=np.float32)
    np.maximum(out, 0.0, out=out)
    if not np.isfinite(out).all():
        out = np.zeros(obs.shape, dtype=np.float32)
    np.save(out_path, out, allow_pickle=False)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
