"""Noise-aware 3-D fluorescence reconstruction."""

import os
import sys
import time

START = time.monotonic()
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np


def stop_time():
    stop = START + 0.79
    try:
        stop = min(stop, float(os.environ["DECONV_DEADLINE_MONOTONIC"]) - 0.065)
    except (ValueError, KeyError):
        pass
    return stop


class Blur:
    """Centered, zero-exterior convolution and its exact adjoint."""

    def __init__(self, psf, shape):
        p, _, _ = kernel_info(psf)
        self.dims = tuple(fft_size(n + k - 1) for n, k in zip(shape, p.shape))
        self.core = tuple(slice(n) for n in shape)
        pad = np.zeros(self.dims, dtype=np.float32)
        pad[tuple(slice(k) for k in p.shape)] = p
        pad = np.roll(pad, tuple(-(k // 2) for k in p.shape), (0, 1, 2))
        self.h = np.fft.rfftn(pad)
        self.ht = self.h.conj()

    def __call__(self, x, adjoint=False):
        f = np.fft.rfftn(x, s=self.dims) * (self.ht if adjoint else self.h)
        return np.fft.irfftn(f, s=self.dims)[self.core].astype(np.float32)


def fft_size(n):
    while True:
        m = n
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def kernel_info(psf):
    p = psf / psf.sum(dtype=np.float64)
    m = tuple(p.sum(axis=tuple(j for j in range(3) if j != i)) for i in range(3))
    product = m[0][:, None, None] * m[1][None, :, None] * m[2][None, None, :]
    return p, m, np.max(np.abs(p - product)) < 1e-6 * np.max(p)


def smooth_mirror(x, sigma):
    r = int(4 * sigma + 0.5)
    shifts = np.arange(-r, r + 1)
    w = np.exp((-0.5 / (sigma * sigma)) * shifts**2)
    w = (w / w.sum()).astype(np.float32)
    for axis, n in enumerate(x.shape):
        if n == 1:
            continue
        loc = (np.arange(n)[None, :] + shifts[:, None]) % (2 * n - 2)
        loc = np.minimum(loc, 2 * n - 2 - loc)
        z = np.take(x, loc[0], axis=axis) * w[0]
        for idx, weight in zip(loc[1:], w[1:]):
            z += weight * np.take(x, idx, axis=axis)
        x = z
    return x


def noise_model(o):
    # Select quiet voxels by a neighborhood statistic that excludes the center.
    smooth = smooth_mirror(o, 1.5)
    context = smooth - np.float32(0.018814) * o
    threshold = np.partition(context.ravel(), o.size // 3)[o.size // 3]
    quiet = o[(context <= threshold) & (o > 0)]
    if quiet.size >= 10:
        sigma = np.sqrt(np.mean(quiet * quiet, dtype=np.float64))
    else:
        sigma = 0.02 * float(o.max())
    y = np.where(o > 0, o, np.float32(-0.798 * sigma))
    b = smooth_mirror(y, 1.15)
    r = (y - b) ** 2
    # The wide, bright bins determine the shot variance without extrapolating
    # from the heavily censored background pixels.
    mask = b > 5 * sigma
    if np.count_nonzero(mask) > 120:
        scale = np.mean(r[mask] - 0.82 * sigma**2) / max(float(np.mean(b[mask])), 1e-12)
        shot = float(np.clip(scale, 0.55 * sigma, 2.3 * sigma))
    else:
        shot = 1.35 * sigma
    return y, b, float(sigma), shot


def reconstruct(obs, psf, deadline, noise, alpha=0.009, count=100):
    blur = Blur(psf, obs.shape)
    y, rough, sigma, shot = noise
    b = np.maximum(rough, 0)
    # A conservative variance floor limits the influence of an inaccurate
    # pilot estimate in the dim halo of a feature.
    floor = max(0.78 * sigma * sigma, 0.027 * shot, 1e-12)
    w = floor / (floor + shot * b)
    norm = blur(np.ones_like(y))
    curvature = blur(w * norm, True)
    rhs = blur(w * y, True)
    u = np.maximum(rhs / np.maximum(curvature, 1e-6), 0)
    x = u
    inv = 0.98 / np.maximum(curvature + 12.0 * alpha, 1e-5)
    t = 1.0
    for _ in range(count):
        if time.monotonic() + 0.007 >= deadline:
            break
        grad = blur(w * blur(u), True) - rhs
        for axis in range(3):
            g = grad.swapaxes(0, axis)
            f = u.swapaxes(0, axis)
            step = alpha * (f[1:] - f[:-1])
            g[1:] += step
            g[:-1] -= step
        v = np.maximum(u - inv * grad, 0)
        tn = (1 + (1 + 4 * t * t)**0.5) * 0.5
        u = v + ((t - 1) / tn) * (v - x)
        x, t = v, tn
    return x


def native_reconstruct(y, rough, marginals, sigma, shot, deadline):
    import ctypes as ct

    try:
        lib = ct.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "deconv_native.so"))
    except OSError:
        return None
    ptr = ct.POINTER(ct.c_float)
    recover = lib.recover
    recover.argtypes = [ptr, ptr, ptr, ct.c_float, ct.c_float, ct.c_double, ptr, ct.c_int]
    recover.restype = ct.c_int
    hh = np.asarray(marginals, dtype=np.float32, order="C")
    rough = np.ascontiguousarray(rough)
    x = np.empty_like(y)
    ok = recover(y.ctypes.data_as(ptr), rough.ctypes.data_as(ptr), hh.ctypes.data_as(ptr),
                 sigma, shot, deadline, x.ctypes.data_as(ptr), 0)
    return x if ok and np.isfinite(x).all() else None


def solve(obs, psf, deadline):
    peak = float(obs.max())
    if peak == 0:
        return obs
    unit = peak if peak < 1e-5 or peak > 1e5 else 1.0
    if unit != 1.0:
        obs = np.asarray(obs.astype(np.float64) / unit, dtype=np.float32)
    y, rough, sigma, empirical_shot = noise_model(obs)
    sigma = max(sigma, 1e-6 * float(obs.max()), 1e-12)
    _, marginals, separable = kernel_info(psf)
    x = None
    if separable and obs.shape == (32, 32, 32) and psf.shape == (9, 9, 9):
        shot = 0.22 * empirical_shot + 0.78 * 1.55 * sigma
        x = native_reconstruct(y, rough, marginals, sigma, shot, deadline)
    if x is None:
        if time.monotonic() + 0.025 < deadline:
            x = reconstruct(obs, psf, deadline, (y, rough, sigma, empirical_shot))
        else:
            x = np.maximum(1.65 * rough, 0)
        if not np.isfinite(x).all():
            x = np.maximum(1.65 * rough, 0)
    if unit != 1.0:
        x = x.astype(np.float64) * unit
    return x


def main():
    with np.load(sys.argv[1], allow_pickle=False) as inp:
        obs = np.ascontiguousarray(inp["observed"], dtype=np.float32)
        psf = np.ascontiguousarray(inp["psf"], dtype=np.float32)
    x = solve(obs, psf, stop_time())
    with open(sys.argv[2], "wb") as handle:
        np.save(handle, x, allow_pickle=False)


if __name__ == "__main__":
    main()
