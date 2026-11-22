"""Single-threaded polarization demosaicking with a compact residual model."""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import ctypes
from pathlib import Path
import numpy as np
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_polar_prior", Path(__file__).with_name("polar_prior.py"))
_prior = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prior)

# The evaluator may have imported NumPy before importing this module.
for _path in (Path(np.__file__).resolve().parent.parent / "numpy.libs").glob("*openblas*"):
    _blas = ctypes.CDLL(str(_path))
    for _name in ("openblas_set_num_threads64_", "openblas_set_num_threads"):
        if hasattr(_blas, _name):
            _setter = getattr(_blas, _name)
            _setter.argtypes = [ctypes.c_int]
            _setter.restype = None
            _setter(1)
            break

with np.load(Path(__file__).with_name("model.npz"), allow_pickle=False) as _data:
    _HP = int(_data["hp"])
    _SCALE = float(_data["scale"])
    _WEIGHTS = [(_data[f"w{i}"], _data[f"b{i}"])
                for i in range(sum(name.startswith("w") for name in _data.files))]
_TILE = 4


def _box_mean(x, size):
    h, w, _ = x.shape
    r = size // 2
    p = np.pad(x, ((r, r), (0, 0), (0, 0)), mode="reflect")
    y = p[:h].copy()
    for i in range(1, size):
        y += p[i:i+h]
    p = np.pad(y, ((0, 0), (r, r), (0, 0)), mode="reflect")
    y = p[:, :w].copy()
    for i in range(1, size):
        y += p[:, i:i+w]
    y *= np.float32(1 / (size * size))
    return y


def _conv(x, weight, bias, activation=False):
    h, w, c = x.shape
    kh, kw, _, co = weight.shape
    if kh == 1 and kw == 1:
        out = x.reshape(-1, c) @ weight.reshape(c, co)
        out += bias
        if activation:
            for start in range(0, len(out), 4096):
                block = out[start:start+4096]
                np.maximum(block, block*np.float32(0.1), out=block)
        return out.reshape(h, w, co)
    p = np.pad(x, ((kh//2, kh//2), (kw//2, kw//2), (0, 0)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(p, (kh, kw), axis=(0, 1))
    windows = windows.transpose(0, 1, 3, 4, 2)
    kernel = weight.reshape(-1, co)
    out = np.empty((h, w, co), np.float32)
    for y in range(0, h, _TILE):
        batch = windows[y:y+_TILE].copy().reshape(-1, kh*kw*c)
        block = batch @ kernel
        block += bias
        if activation:
            np.maximum(block, block*np.float32(0.1), out=block)
        out[y:y+_TILE] = block.reshape(-1, w, co)
    return out


def _base_missing(q):
    h, w, _ = q.shape
    pad = np.pad(q, ((1, 1), (1, 1), (0, 0)), mode="edge")
    values = []
    for c in range(4):
        for p in range(4):
            if c == p:
                continue
            yc, xc = c//2, c%2
            yp, xp = p//2, p%2
            value = q[:, :, c]
            if xc != xp:
                dx = 1-2*xc
                value = (value+pad[1:1+h, 1+dx:1+w+dx, c])*0.5
            if yc != yp:
                dy = 1-2*yc
                other = pad[1+dy:1+h+dy, 1:1+w, c]
                if xc != xp:
                    other = (other+pad[1+dy:1+h+dy, 1+dx:1+w+dx, c])*0.5
                value = (value+other)*0.5
            values.append(value)
    return np.stack(values, axis=-1)


def reconstruct(raw):
    h, w = raw.shape
    # Packed sample order is 90, 45, 135, 0 degrees.
    q = raw.reshape(h//2, 2, w//2, 2).transpose(0, 2, 1, 3).reshape(h//2, w//2, 4)
    q = q.astype(np.float32)
    if np.all(q == q[0, 0]):
        out = np.empty((4, h, w), np.float32)
        for c, k in enumerate((2, 1, 3, 0)):
            out[k].fill(q[0, 0, c])
        return out
    observed = q
    scales, constant = _prior.contrast_scales(q)
    reference = 0
    # A weak angle is an unreliable registration reference.
    if scales is not None and scales[0] < .5*scales.max():
        reference = int(np.argmax(scales))
    shifts = _prior.estimate_shifts(q, reference)
    if scales is not None:
        q = q/scales
    kernel = _prior.make_kernel(shifts, observed=True)
    guide = _conv(q, kernel, np.zeros(16, np.float32))
    baseline = _base_missing(q)
    innovation = guide[:, :, :12] - baseline
    x = q / _SCALE
    if _HP:
        x -= _box_mean(x, _HP)
    x = np.concatenate([x, innovation/_SCALE, (q-guide[:, :, 12:])/_SCALE], axis=-1)
    del guide
    if len(_WEIGHTS) == 7:
        for i in range(3):
            x = _conv(x, *_WEIGHTS[i], activation=True)
        extra = _conv(x, *_WEIGHTS[3], activation=True)
        extra = _conv(extra, *_WEIGHTS[4])
        x += extra
        x = _conv(x, *_WEIGHTS[5], activation=True)
        x = _conv(x, *_WEIGHTS[6])
    else:
        for i, (weight, bias) in enumerate(_WEIGHTS):
            x = _conv(x, weight, bias, activation=i != len(_WEIGHTS)-1)
    x *= _SCALE
    innovation *= np.float32(0.5)
    x += innovation
    del innovation
    out = np.empty((4, h, w), np.float32)
    index = 0
    for c, k in enumerate((2, 1, 3, 0)):
        if constant[c]:
            out[k].fill(observed[0, 0, c])
            index += 3
            continue
        for phase in range(4):
            if c == phase:
                out[k, phase//2::2, phase%2::2] = observed[:, :, c]
            else:
                value = baseline[:, :, index] + x[:, :, index]
                if scales is not None:
                    value *= scales[c]
                out[k, phase//2::2, phase%2::2] = value
                index += 1
    np.clip(out, 0, 255, out=out)
    return out
