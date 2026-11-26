"""Single-threaded polarization demosaicking using shared residual filters."""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import ctypes
from pathlib import Path

import numpy as np


# The harness may have imported NumPy before importing this module.
for _library in (Path(np.__file__).resolve().parent.parent / "numpy.libs").glob("*openblas*"):
    _blas = ctypes.CDLL(str(_library))
    for _name in ("openblas_set_num_threads64_", "openblas_set_num_threads"):
        if hasattr(_blas, _name):
            _setter = getattr(_blas, _name)
            _setter.argtypes = [ctypes.c_int]
            _setter.restype = None
            _setter(1)
            break

_PHASES = ((1, 1), (0, 1), (0, 0), (1, 0))
_MISSING = tuple((c, p) for c, (y, x) in enumerate(_PHASES)
                 for p in range(4) if p != 2*y+x)
_WEIGHTS = []
with np.load(Path(__file__).with_name("weights.npz"), allow_pickle=False) as _data:
    _i = 0
    while "w%d" % _i in _data:
        _WEIGHTS.append((_data["w%d" % _i], _data["b%d" % _i]))
        _i += 1
_WEIGHTS[0][0][:] *= np.float32(1/64)
_WEIGHTS[-1][0][:] *= np.float32(64)
_WEIGHTS[-1][1][:] *= np.float32(64)


def _residual(x):
    _, h, w = x.shape
    channels = max(max(weight.shape[:2]) for weight, _ in _WEIGHTS)
    radius = max(weight.shape[-1]//2 for weight, _ in _WEIGHTS)
    features = max(int(np.prod(weight.shape[1:])) for weight, _ in _WEIGHTS)
    padded = np.empty((channels, h+2*radius, w+2*radius), np.float32)
    feature = np.empty((channels, h, w), np.float32)
    patches = np.empty((features, 8*w), np.float32)
    f = x
    for i, (weight, bias) in enumerate(_WEIGHTS):
        co, ci, k, _ = weight.shape
        r = k//2
        xp = padded[:ci, :h+2*r, :w+2*r]
        xp[:, r:r+h, r:r+w] = f
        if r:
            xp[:, :r, r:r+w] = f[:, :1]
            xp[:, r+h:, r:r+w] = f[:, -1:]
            xp[:, :, :r] = xp[:, :, r:r+1]
            xp[:, :, r+w:] = xp[:, :, r+w-1:r+w]
        # Once the input is in the padded buffer, its feature storage is reusable.
        f = feature[:co]
        matrix = weight.reshape(co, -1)
        for y in range(0, h, 8):
            n = min(8, h-y)
            window = np.lib.stride_tricks.sliding_window_view(
                xp[:, y:y+n+2*r], (k, k), axis=(1, 2))
            patch = patches[:ci*k*k, :n*w]
            np.copyto(patch.reshape(ci, k, k, n, w), window.transpose(0, 3, 4, 1, 2))
            np.matmul(matrix, patch, out=f[:, y:y+n].reshape(co, n*w))
        f += bias[:, None, None]
        if i+1 != len(_WEIGHTS):
            np.maximum(f, 0, out=f)
    return f


def _reconstruct(raw):
    h, w = raw.shape
    packed = np.stack([raw[y::2, x::2] for y in range(2) for x in range(2)]).astype(np.float32)
    f = _residual(packed)

    out = np.empty((4, h, w), np.float32)
    padded = np.pad(packed, ((0, 0), (1, 1), (1, 1)), mode="edge")
    hh, ww = h//2, w//2
    for c, (y, x) in enumerate(_PHASES):
        out[c, y::2, x::2] = raw[y::2, x::2]
    for i, (c, p) in enumerate(_MISSING):
        y, x = _PHASES[c]
        dy, dx = p//2-y, p%2-x
        ys = (0,) if not dy else (0, dy)
        xs = (0,) if not dx else (0, dx)
        base = np.zeros((hh, ww), np.float32)
        for yy in ys:
            for xx in xs:
                base += padded[2*y+x, 1+yy:1+yy+hh, 1+xx:1+xx+ww]
        base *= np.float32(1/(len(ys)*len(xs)))
        base += f[i]
        np.clip(base, 0, 255, out=base)
        out[c, p//2::2, p%2::2] = base
    return out


def reconstruct(raw):
    """Return float32 planes in the order 0, 45, 90, 135 degrees."""
    # Even reflection padding keeps each measurement on its original mosaic phase.
    padded = np.pad(raw, 16, mode="reflect")
    return np.ascontiguousarray(_reconstruct(padded)[:, 16:-16, 16:-16])
