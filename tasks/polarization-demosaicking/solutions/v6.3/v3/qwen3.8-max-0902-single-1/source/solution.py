"""Polarization CFA demosaicking (4-angle DoFP mosaic -> 4 full intensity planes).

Mosaic layout (row, col both 0-based):
    even row / even col -> 90 deg,  even row / odd col -> 45 deg
    odd  row / even col -> 135 deg, odd  row / odd col -> 0 deg

Method
------
Every pixel observes exactly one of the four angles, so `raw` is already a
(full-rate, but angle-multiplexed) measurement of the scene.  Rather than
interpolating each angle plane independently from its own stride-2 lattice
(which throws away the detail carried by the other three lattices), we predict
each missing angle directly as a local linear correction to the observed value:

    I_theta(p) ~= raw(p) + sum_k w[theta, parity(p), k] * (raw(p+k) - raw(p)) + b

The weights are ordinary least-squares (ridge-stabilised) regressions fitted
offline on the public development scenes and shipped as constants, i.e. a
per-parity LMMSE predictor.  Using differences from the centre pixel makes the
model invariant to a global intensity offset; splitting the regression by the
2x2 mosaic phase lets each of the four sampling geometries have its own optimal
filter.  At a pixel where angle theta is itself observed the correction is
defined to be zero, so the reconstruction reproduces `raw` exactly there.
Predictions are finally clipped to the valid 0..255 output range.
"""
import base64
import numpy as np

_R = 2                      # window radius -> (2R+1)^2 - 1 taps
_PAD = _R + 2
_KS = [(dy, dx) for dy in range(-_R, _R + 1)
               for dx in range(-_R, _R + 1) if not (dy == 0 and dx == 0)]
_N = len(_KS) + 1
# observed channel index per mosaic phase p = 2*(row&1) + (col&1)
_OBS = (2, 1, 3, 0)         # ee->90, eo->45, oe->135, oo->0
_W = np.frombuffer(base64.b64decode(
    "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"), dtype=np.float32).reshape(4, 4, _N).copy()


def reconstruct(raw):
    r = np.ascontiguousarray(raw, dtype=np.float32)
    H, W = r.shape
    P = np.pad(r, _PAD, mode="reflect")

    out = np.empty((4, H, W), np.float32)
    for c in range(4):
        out[c] = r

    for p in range(4):
        pr, pc = p >> 1, p & 1
        hn = len(range(pr, H, 2))
        wn = len(range(pc, W, 2))
        base = P[_PAD + pr:_PAD + pr + 2 * hn:2,
                 _PAD + pc:_PAD + pc + 2 * wn:2]
        F = np.empty((_N - 1, hn, wn), np.float32)
        for i, (dy, dx) in enumerate(_KS):
            F[i] = P[_PAD + pr + dy:_PAD + pr + dy + 2 * hn:2,
                     _PAD + pc + dx:_PAD + pc + dx + 2 * wn:2]
        F -= base
        Wm = _W[p]
        idx = [c for c in range(4) if c != _OBS[p]]
        V = np.tensordot(Wm[idx][:, :_N - 1], F, axes=(1, 0))
        V += Wm[idx][:, _N - 1].reshape(-1, 1, 1)
        for j, c in enumerate(idx):
            out[c][pr::2, pc::2] = base + V[j]

    np.clip(out, 0.0, 255.0, out=out)
    return out
