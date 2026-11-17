#!/usr/bin/env python3
"""Single-case 3D fluorescence deconvolution with an MLP ensemble.

Pipeline: classify the clipped-Gaussian noise regime; run a separable
nonnegative Richardson-Lucy iteration with exact zero-extension boundaries;
build multi-scale local and coarse features; average five small offline-trained
MLPs (with separate weights for the two noise regimes) that map those features
to a clean intensity.  A physics-only fallback is used if the weights cannot
be loaded.  NumPy is the only runtime dependency.
"""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import math
import sys
import time

import numpy as np

START = time.monotonic()
INV_SQRT_2PI = 0.3989422804014327
HERE = os.path.dirname(os.path.abspath(__file__))
SNAPS = (4, 8, 15, 20, 25, 30)
D2 = 42
D3 = 96
D_ALL = 106
PATCH_OFFSETS = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
MAP1 = [0,1,2,4,5,8,9,10,11,13,14,15,16,19,21,22,24,25,28,29,31,34,35,36,38,39,40,41]
MAP_V4 = list(range(D2)) + list(range(D3, D_ALL))


# ---------------------------------------------------------------- operators
def conv_axis(x, k, axis):
    r = (k.size - 1) // 2
    n = x.shape[axis]
    pad = [(0, 0), (0, 0), (0, 0)]
    pad[axis] = (r, r)
    xp = np.pad(x, pad, mode='constant')
    out = np.zeros_like(x)
    sl = [slice(None), slice(None), slice(None)]
    for j in range(k.size):
        w = k[j]
        if w:
            sl[axis] = slice(j, j + n)
            out += w * xp[tuple(sl)]
    return out


def kernel_gauss(sigma, radius=None):
    if sigma <= 1e-6:
        return np.ones(1, np.float32)
    if radius is None:
        radius = int(math.ceil(2.5 * sigma))
    r = max(1, min(6, int(radius)))
    j = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (j / sigma) ** 2)
    k /= k.sum()
    return k.astype(np.float32)


def conv3(x, ks):
    for ax in range(3):
        x = conv_axis(x, ks[ax], ax)
    return x


def max_axis(x, r, axis):
    n = x.shape[axis]
    pad = [(0, 0), (0, 0), (0, 0)]
    pad[axis] = (r, r)
    xp = np.pad(x, pad, mode='constant')
    out = x.copy()
    sl = [slice(None), slice(None), slice(None)]
    for j in range(2 * r + 1):
        sl[axis] = slice(j, j + n)
        np.maximum(out, xp[tuple(sl)], out=out)
    return out


def mean_axis(x, r, axis):
    n = x.shape[axis]
    pad = [(0, 0), (0, 0), (0, 0)]
    pad[axis] = (r, r)
    xp = np.pad(x, pad, mode='constant')
    out = np.zeros_like(x)
    sl = [slice(None), slice(None), slice(None)]
    for j in range(2 * r + 1):
        sl[axis] = slice(j, j + n)
        out += xp[tuple(sl)]
    return out / (2 * r + 1)


def filt3(x, size, op):
    for ax in range(3):
        if op == 'max':
            x = max_axis(x, (size - 1) // 2, ax)
        else:
            x = mean_axis(x, (size - 1) // 2, ax)
    return x


def smooth(x, k):
    for ax in range(3):
        x = conv_axis(x, k, ax)
    return x


def shift3(a, dz, dy, dx):
    out = np.zeros_like(a)
    sz, sy, sx = a.shape
    z0 = max(0, -dz); z1 = min(sz, sz - dz)
    y0 = max(0, -dy); y1 = min(sy, sy - dy)
    x0 = max(0, -dx); x1 = min(sx, sx - dx)
    out[z0:z1, y0:y1, x0:x1] = a[z0 + dz:z1 + dz, y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return out


def pool(a, b, op):
    n = a.shape[0] // b
    v = a.reshape(n, b, n, b, n, b)
    if op == 'mean':
        v = v.mean(axis=(1, 3, 5))
    else:
        v = v.max(axis=(1, 3, 5))
    return np.repeat(np.repeat(np.repeat(v, b, axis=0), b, axis=1), b, axis=2)


def psf_marginals(p):
    return (np.ascontiguousarray(p.sum(axis=(1, 2)), np.float32),
            np.ascontiguousarray(p.sum(axis=(0, 2)), np.float32),
            np.ascontiguousarray(p.sum(axis=(0, 1)), np.float32))


def marginal_sigma(k):
    c = (np.arange(k.size) - (k.size - 1) / 2.0) ** 2
    return float(np.sqrt((c * k).sum() / max(k.sum(), 1e-12)))


def estimate_noise(y):
    s = 0.012
    div = 0.6219509777741192
    for _ in range(5):
        m = (y > 0) & (y < 1.5 * s)
        if m.sum() < 256:
            break
        v = float(y[m].mean()) / div
        if not np.isfinite(v) or v <= 0:
            break
        s = v
    sigma = 0.008 if s < 0.015 else 0.025
    return sigma, s


def rl_snapshots(y, ks, sigma, eps=1e-6):
    yc = np.maximum(y - sigma * INV_SQRT_2PI, 0).astype(np.float32)
    sens = np.maximum(conv3(np.ones_like(yc), ks), 1e-3)
    sk = kernel_gauss(0.3, radius=1)
    sks = (sk, sk, sk)
    x = np.full_like(yc, float(max(yc.mean(), 1e-5)))
    out = {}
    for i in range(1, max(SNAPS) + 1):
        ax = np.maximum(conv3(x, ks), eps)
        upd = conv3(yc / ax, ks)
        x *= upd
        x /= sens
        np.maximum(x, 0, out=x)
        x = conv3(x, sks)
        if i in SNAPS:
            out[i] = x.copy()
    return out, yc, sens


def build_features(y, p, sigma, s_est):
    ks = psf_marginals(p)
    sigmas = np.array([marginal_sigma(k) for k in ks], np.float32)
    snaps, yc, sens = rl_snapshots(y, ks, sigma)
    x = snaps[20] if sigma < 0.015 else snaps[15]
    k04 = kernel_gauss(0.4); k08 = kernel_gauss(0.8)
    k16 = kernel_gauss(1.6); k30 = kernel_gauss(3.0)
    f = []

    def add(a):
        f.append(np.asarray(a, np.float32).ravel())

    # 42 base multi-scale features
    add(yc)
    add(smooth(yc, k08)); add(smooth(yc, k16)); add(smooth(yc, k30))
    add(filt3(yc, 3, 'max')); add(filt3(yc, 3, 'mean'))
    add(filt3(yc, 5, 'max')); add(filt3(yc, 5, 'mean'))
    add(x)
    add(smooth(x, k04)); add(smooth(x, k08)); add(smooth(x, k16)); add(smooth(x, k30))
    add(filt3(x, 3, 'max')); add(filt3(x, 3, 'mean'))
    add(filt3(x, 5, 'max')); add(filt3(x, 5, 'mean'))
    add(filt3(x, 7, 'max')); add(filt3(x, 7, 'mean'))
    m3 = filt3(x, 3, 'mean'); m3sq = filt3(x * x, 3, 'mean')
    add(np.sqrt(np.maximum(m3sq - m3 * m3, 0)))
    m5 = filt3(x, 5, 'mean'); m5sq = filt3(x * x, 5, 'mean')
    add(np.sqrt(np.maximum(m5sq - m5 * m5, 0)))
    add(x - smooth(x, k08)); add(x - smooth(x, k16)); add(x - smooth(x, k30))
    add(np.maximum(x - 0.4 * sigma, 0)); add(np.maximum(x - 1.0 * sigma, 0))
    add(x / (filt3(x, 5, 'max') + 0.02))
    for i in SNAPS:
        add(snaps[i])
    add(snaps[15] - snaps[8])
    add(sens)
    add(yc / np.maximum(sens, 0.2))
    add(np.full_like(x, sigma)); add(np.full_like(x, s_est))
    for s in sigmas:
        add(np.full_like(x, s))
    sh = y.shape
    zz, yy, xx = np.meshgrid(np.arange(sh[0]) / max(sh[0] - 1, 1),
                             np.arange(sh[1]) / max(sh[1] - 1, 1),
                             np.arange(sh[2]) / max(sh[2] - 1, 1), indexing='ij')
    edge = np.minimum.reduce([zz, yy, xx, 1 - zz, 1 - yy, 1 - xx]) * 2
    add(edge)
    F2 = np.stack(f, axis=1)
    # 54 raw 3x3x3 neighborhood features
    patch = []
    for off in PATCH_OFFSETS:
        patch.append(np.asarray(shift3(x, *off), np.float32).ravel())
    for off in PATCH_OFFSETS:
        patch.append(np.asarray(shift3(yc, *off), np.float32).ravel())
    patch = np.stack(patch, axis=1)
    # 10 coarse pooled features
    c2m = pool(x, 2, 'mean'); c2M = pool(x, 2, 'max')
    c4m = pool(x, 4, 'mean'); c4M = pool(x, 4, 'max')
    y2m = pool(yc, 2, 'mean'); y4m = pool(yc, 4, 'mean')
    c2s = np.sqrt(np.maximum(pool(x * x, 2, 'mean') - c2m * c2m, 0))
    c4s = np.sqrt(np.maximum(pool(x * x, 4, 'mean') - c4m * c4m, 0))
    coarse = [c2m, c2M, c4m, c4M, y2m, y4m, x - c2m, x - c4m, c2s, c4s]
    coarse = np.stack([np.asarray(e, np.float32).ravel() for e in coarse], axis=1)
    F = np.concatenate([F2, patch, coarse], axis=1)
    return F, x, sens


def mlp_raw(F, W, prefix, tag=''):
    mu = W[prefix + tag + 'mu']; sd = W[prefix + tag + 'sd']
    X = (F - mu) / sd
    H = np.maximum(X @ W[prefix + tag + 'W1'] + W[prefix + tag + 'b1'], 0)
    H2 = np.maximum(H @ W[prefix + tag + 'W2'] + W[prefix + tag + 'b2'], 0)
    return np.maximum((H2 @ W[prefix + tag + 'W3'] + W[prefix + tag + 'b3']).ravel(), 0)


def ensemble_predict(F, W, sigma):
    tag = 'low_' if sigma < 0.015 else 'high_'
    # Five complementary estimators, weighted for a balance of development
    # accuracy and held-out synthetic generalization.
    p1 = mlp_raw(F[:, MAP1], W, 'v1_')
    p4 = mlp_raw(F[:, MAP_V4], W, 'v4_', tag)
    p5 = mlp_raw(F[:, MAP_V4], W, 'v5_', tag)
    p7 = mlp_raw(F[:, MAP_V4], W, 'm32_', tag)
    p8 = mlp_raw(F[:, MAP_V4], W, 's41_', tag)
    return (3.0 * p1 + p4 + p5 + p7 + 3.0 * p8) / 9.0


def fallback(x, sigma):
    sk = kernel_gauss(0.3, radius=1)
    x = conv3(x, (sk, sk, sk))
    x = np.maximum(x - 0.4 * sigma, 0)
    np.minimum(x, 1.2, out=x)
    return x


def deadline():
    try:
        d = float(os.environ.get('DECONV_DEADLINE_MONOTONIC', '0'))
    except ValueError:
        d = 0.0
    if d > START and d < START + 10.0:
        return d - 0.12
    try:
        budget = float(os.environ.get('TIME_BUDGET_SECONDS', '0.5'))
    except ValueError:
        budget = 0.5
    return START + max(0.08, min(budget, 0.5) - 0.22)


def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: solution.py INPUT OUTPUT')
    inp, outp = sys.argv[1], sys.argv[2]
    with np.load(inp, allow_pickle=False) as data:
        y = np.asarray(data['observed'], dtype=np.float32)
        p = np.asarray(data['psf'], dtype=np.float32)

    sigma, s_est = estimate_noise(y)
    ks = psf_marginals(p)
    recon = ks[0][:, None, None] * ks[1][None, :, None] * ks[2][None, None, :]
    separable = np.linalg.norm(recon - p) <= 2e-3 * max(np.linalg.norm(p), 1e-20)
    # Coarse pooling requires side lengths divisible by 4. Formal cases are
    # 32^3; guard any other shape with a simple, always-valid estimate.
    shape_ok = all(int(s) % 4 == 0 for s in y.shape)
    if not separable or not shape_ok:
        x = np.maximum(y - sigma * INV_SQRT_2PI, 0)
        with open(outp, 'wb') as fh:
            np.save(fh, x.astype(np.float32), allow_pickle=False)
        return

    F, x_rl, _sens = build_features(y, p, sigma, s_est)
    x = fallback(x_rl, sigma)
    try:
        with np.load(os.path.join(HERE, 'ensemble_weights.npz'), allow_pickle=False) as W:
            w = {k: W[k] for k in W.files}
        if w['s41_' + ('low_' if sigma < 0.015 else 'high_') + 'W1'].shape[0] == len(MAP_V4):
            x = ensemble_predict(F, w, sigma)
    except Exception:
        pass

    x = np.asarray(x, dtype=np.float32).reshape(y.shape)
    np.maximum(x, 0, out=x)
    np.minimum(x, 1.0, out=x)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        np.maximum(x, 0, out=x)

    with open(outp, 'wb') as fh:
        np.save(fh, x, allow_pickle=False)


if __name__ == '__main__':
    main()
