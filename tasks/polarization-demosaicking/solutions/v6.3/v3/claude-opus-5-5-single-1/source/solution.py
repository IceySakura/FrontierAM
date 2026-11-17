"""Polarization demosaicking: gated LMMSE filters learned offline (see README.md)."""
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _swv

_K = 7          # filter support (full-res pixels)
_MODE = "B"     # gate features: "A" -> [g, h]; "B" -> [g, h, g^2, h^2, g*h]
_R1 = 8         # window radius (sub-image px) for local cross-angle correlation gate
_R2 = 4         # window radius (sub-image px) for Stokes-deviation gate
_C0 = 5.0       # saturation constant of the Stokes-deviation gate
# output channel (0,45,90,135 order) -> its mosaic phase (row%2, col%2)
_NATIVE = {0: (1, 1), 1: (0, 1), 2: (0, 0), 3: (1, 0)}
_PH = [(0, 0), (0, 1), (1, 0), (1, 1)]
_W = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npy"))


def _box(a, r):
    p = np.pad(a, ((r + 1, r), (r + 1, r)), mode="reflect")
    c = p.cumsum(0)
    c = c[2 * r + 1:] - c[:-2 * r - 1]
    c = c.cumsum(1)
    c = c[:, 2 * r + 1:] - c[:, :-2 * r - 1]
    return c / ((2 * r + 1) ** 2)


def _bp(a):
    b1 = _box(a, 1)
    return b1 - _box(b1, 3)


def _gates(x):
    S = [x[pr::2, pc::2] for c, (pr, pc) in _NATIVE.items()]
    Hs = [_bp(s) for s in S]
    V = [_box(h * h, _R1) for h in Hs]
    acc = 0
    for a in range(4):
        for b in range(a + 1, 4):
            acc = acc + _box(Hs[a] * Hs[b], _R1) / np.sqrt(V[a] * V[b] + 1e-2)
    g = acc / 6
    dev = S[0] + S[2] - S[1] - S[3]
    d = np.sqrt(_box(dev * dev, _R2))
    h = d / (d + _C0)
    if _MODE == "A":
        return [g, h]
    return [g, h, g * g, h * h, g * h]  # on the half-resolution grid


def reconstruct(raw):
    raw = np.asarray(raw)
    H, Wd = raw.shape
    x = raw.astype(np.float64)
    gates = [a.reshape(-1, 1).astype(np.float32) for a in _gates(x)]
    ng = len(gates)
    R = _K // 2
    P = R + (R % 2)  # even pad keeps the mosaic phase under reflection
    p = (np.pad(x, P, mode="reflect") - 128.0).astype(np.float32)
    v = _swv(p, (_K, _K))
    kk = _K * _K
    out = np.empty((4, H, Wd), dtype=np.float32)
    for i, (pr, pc) in enumerate(_PH):
        W = _W[i]
        sub = v[P - R + pr:P - R + pr + H:2, P - R + pc:P - R + pc + Wd:2]
        n0, n1 = sub.shape[:2]
        X = sub.reshape(n0 * n1, kk)
        M = np.concatenate([W[j * kk:(j + 1) * kk] for j in range(1 + ng)], axis=1).astype(np.float32)
        Z = X @ M  # (N, 4 * (1 + ng))
        b = W[(1 + ng) * kk:].astype(np.float32)
        Y = Z[:, 0:4] + (b[0] + 128.0)
        for j, a in enumerate(gates):
            Y += a * (Z[:, 4 * (j + 1):4 * (j + 2)] + b[1 + j])
        out[:, pr::2, pc::2] = Y.T.reshape(4, n0, n1)
    for c, (pr, pc) in _NATIVE.items():
        out[c, pr::2, pc::2] = raw[pr::2, pc::2]
    np.clip(out, 0.0, 255.0, out=out)
    return out
