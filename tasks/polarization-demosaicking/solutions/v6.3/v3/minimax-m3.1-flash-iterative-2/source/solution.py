"""Polarization demosaicking: reconstruct the four angle images
(0 deg, 45 deg, 90 deg, 135 deg) from a single mosaic measurement.

Mosaic layout (row / column indices start at 0):

                even col      odd col
    even row      90 deg       45 deg
    odd row      135 deg        0 deg

Method (see README.md for the details):
    Every angle image is modelled as  I_k = B + D_k  where B is the
    intensity field shared by the four angles and D_k is a smooth, angle
    specific offset (the polarization signal).  B is estimated from all
    four interleaved sub-lattices - so the image structure that the four
    angles share is recovered at full resolution - and D_k from the
    residual of the k-th sub-lattice.
"""
import numpy as np

PAR = ((0, 0), (0, 1), (1, 0), (1, 1))   # (row parity, col parity) per class
CH = (2, 1, 3, 0)                        # channel index per class


def _up2(S, cubic=True):
    """Upsample a half resolution lattice by two (phase 0,0).

    The lattice positions are copied exactly, the half-pixel positions in
    between are filled with a cubic (Catmull-Rom) or bilinear kernel.
    """
    S = np.ascontiguousarray(S, dtype=np.float32)
    h, w = S.shape
    Sp = np.empty((h + 1, w + 1), np.float32)
    Sp[:h, :w] = S
    Sp[:h, w] = S[:, -1]
    Sp[h, :w] = S[-1]
    Sp[h, w] = S[-1, -1]
    T = np.empty((h + 1, 2 * w), np.float32)
    T[:h, 0::2] = Sp[:h, :w]
    if cubic:
        P = np.pad(S, ((0, 0), (2, 2)), mode='edge')
        T[:h, 1::2] = (-0.0625 * P[:, 1:-3] + 0.5625 * P[:, 2:-2]
                       + 0.5625 * P[:, 3:-1] - 0.0625 * P[:, 4:])
    else:
        T[:h, 1::2] = 0.5 * (Sp[:h, :w] + Sp[:h, 1:w + 1])
    T[h] = T[h - 1]
    F = np.empty((2 * h, 2 * w), np.float32)
    F[0::2] = T[:h]
    if cubic:
        Q = np.pad(T[:h], ((2, 2), (0, 0)), mode='edge')
        F[1::2] = (-0.0625 * Q[1:-3] + 0.5625 * Q[2:-2]
                   + 0.5625 * Q[3:-1] - 0.0625 * Q[4:])
    else:
        F[1::2] = 0.5 * (T[:h] + T[1:h + 1])
    return F


def _phase(F, ro, co):
    """Shift an upsampled field so that the lattice sits at offset (ro, co)."""
    if ro:
        F = np.concatenate((F[:1], F[:-1]), axis=0)
    if co:
        F = np.concatenate((F[:, :1], F[:, :-1]), axis=1)
    return F


def reconstruct(raw):
    v = np.asarray(raw)
    if v.dtype != np.float32:
        v = v.astype(np.float32)
    H, W = v.shape
    # the four sub-lattices must have the same size -> work on an even crop
    ph, pw = H & 1, W & 1
    if ph or pw:
        v = np.pad(v, ((0, ph), (0, pw)), mode='edge')
        H, W = v.shape
    lat = [v[ro::2, co::2] for (ro, co) in PAR]

    # ---- shared intensity field B ------------------------------------
    base = np.zeros((H, W), np.float32)
    for k in range(4):
        base += _phase(_up2(lat[k], cubic=True), *PAR[k])
    base *= 0.25

    # ---- per angle offset fields D_k ---------------------------------
    out = np.empty((4, H, W), np.float32)
    for k, (ro, co) in enumerate(PAR):
        res = lat[k] - base[ro::2, co::2]
        out[CH[k]] = base + _phase(_up2(res, cubic=False), ro, co)
        # measured pixels are reproduced exactly
        out[CH[k]][ro::2, co::2] = lat[k]

    if ph or pw:
        out = out[:, :H - ph, :W - pw]
    return out
