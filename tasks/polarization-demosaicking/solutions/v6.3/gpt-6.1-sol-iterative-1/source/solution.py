"""Single-frame polarization demosaicking with a compact residual model."""

import os
import sys
import numpy as np
from threadpoolctl import threadpool_limits

_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
if _DIRECTORY not in sys.path:
    sys.path.insert(0, _DIRECTORY)
from polar_net import ResidualModel, mean_filter as _mean_filter


_PHASES = ((1, 1), (0, 1), (0, 0), (1, 0))
threadpool_limits(limits=1, user_api="blas")


_MODEL = ResidualModel(os.path.join(_DIRECTORY, "weights.npz"))


def _neural_detail(raw, guide):
    correction = _MODEL.predict(raw, guide)
    r = np.pad(raw.astype(np.float32), 2, mode="reflect")
    x = np.stack([r[y::2, z::2] for y in range(2) for z in range(2)], -1)
    center = x[1:-1, 1:-1]
    lx = x[1:-1, :-2] + x[1:-1, 2:] - 2 * center
    ly = x[:-2, 1:-1] + x[2:, 1:-1] - 2 * center
    # Retain the analytic estimate on flat patches and at the image rim.
    gate = np.clip((np.mean(np.abs(lx) + np.abs(ly), axis=-1) * 0.5 - 0.25) * 2,
                   0, 1)
    gate[:2] = 0
    gate[-2:] = 0
    gate[:, :2] = 0
    gate[:, -2:] = 0
    correction -= correction.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    correction *= 0.9 * gate[..., None]
    return correction


def _interpolate(a, phase, shape):
    y, x = phase
    h, w = shape
    t = np.empty((a.shape[0], w), dtype=np.float32)
    t[:, x::2] = a
    if x:
        t[:, 0] = a[:, 0]
        t[:, 2::2] = (a[:, :-1] + a[:, 1:]) * 0.5
    else:
        t[:, 1:-1:2] = (a[:, :-1] + a[:, 1:]) * 0.5
        t[:, -1] = a[:, -1]
    out = np.empty(shape, dtype=np.float32)
    out[y::2] = t
    if y:
        out[0] = t[0]
        out[2::2] = (t[:-1] + t[1:]) * 0.5
    else:
        out[1:-1:2] = (t[:-1] + t[1:]) * 0.5
        out[-1] = t[-1]
    return out


def reconstruct(raw):
    """Return intensities in the order 0, 45, 90, 135 degrees."""
    raw = np.asarray(raw)
    out = np.stack([_interpolate(raw[y::2, x::2].astype(np.float32),
                                 (y, x), raw.shape)
                    for y, x in _PHASES])
    if all(np.ptp(raw[y::2, x::2]) == 0 for y, x in _PHASES):
        return out
    # Polarization differences vary more smoothly than common image detail.
    guide = out.mean(axis=0)
    for c in range(4):
        out[c] = _mean_filter(out[c] - guide, 5)
    detail = raw.astype(np.float32)
    for c, (y, x) in enumerate(_PHASES):
        detail[y::2, x::2] -= out[c, y::2, x::2]
    out += detail
    if min(raw.shape) < 128:
        for c, (y, x) in enumerate(_PHASES):
            out[c, y::2, x::2] = raw[y::2, x::2]
        np.clip(out, 0, 255, out=out)
        return out
    correction = _neural_detail(raw, detail)
    for c, (cy, cx) in enumerate(_PHASES):
        for y in range(2):
            for x in range(2):
                if y != cy or x != cx:
                    out[c, y::2, x::2] += correction[:, :, c * 4 + y * 2 + x]
        out[c, cy::2, cx::2] = raw[cy::2, cx::2]
    np.clip(out, 0, 255, out=out)
    out *= 16
    np.rint(out, out=out)
    out *= 1 / 16
    return out
