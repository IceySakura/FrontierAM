"""Fast local polarization demosaicking with fixed, generic learned filters."""
import ctypes
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

# NumPy may already have been imported by the caller.
for _path in (Path(np.__file__).resolve().parent.parent / "numpy.libs").glob(
    "*openblas*.so"
):
    _lib = ctypes.CDLL(str(_path))
    _setter = getattr(_lib, "openblas_set_num_threads64_", None)
    if _setter is not None:
        _setter.argtypes = [ctypes.c_int]
        _setter(1)

_PHASES = [(1, 1), (0, 1), (0, 0), (1, 0)]
_KNOWN = [3, 1, 0, 2]
_UNKNOWN = [k * 4 + p for k in range(4) for p in range(4) if p != _KNOWN[k]]
with np.load(Path(__file__).with_name("weights.npz"), allow_pickle=False) as _data:
    _WEIGHTS = [
        (np.ascontiguousarray(_data[f"w{i}"]), np.ascontiguousarray(_data[f"b{i}"]))
        for i in range(len(_data.files) // 2)
    ]
if _WEIGHTS[0][0].shape[1] == 11:
    _SHIFT_WEIGHT = _WEIGHTS[0][0][:, 5:].sum(axis=(2, 3))
    _FIRST_WEIGHT = np.ascontiguousarray(_WEIGHTS[0][0][:, :5])
else:
    _SHIFT_WEIGHT = None


def _translations(images):
    _, h, w = images.shape
    phases = np.asarray(_PHASES, dtype=np.float64)
    window = np.hanning(h)[:, None] * np.hanning(w)[None]
    centered = images - images.mean(axis=(1, 2), keepdims=True)
    fft = np.fft.rfft2(centered * window)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None]
    radius = np.sqrt(fy * fy + fx * fx)
    selected = (radius >= 0.025) & (radius <= 0.2)
    design = np.stack(
        [np.broadcast_to(fy, radius.shape)[selected],
         np.broadcast_to(fx, radius.shape)[selected]], axis=1
    ) * (2 * np.pi)
    result = np.zeros((4, 2), dtype=np.float64)
    ref = fft[2][selected].conj()
    for k in [0, 1, 3]:
        spectrum = fft[k][selected] * ref
        phase = np.angle(spectrum)
        weight = np.sqrt(np.abs(spectrum))
        mass = weight.mean()
        if mass <= 1e-12:
            continue
        weight /= mass
        position = np.zeros(2)
        for _ in range(5):
            pred = design @ position
            unwrapped = (phase - pred + np.pi) % (2 * np.pi) - np.pi + pred
            robust = weight / np.maximum(0.1, np.abs(unwrapped - pred))
            normal = design.T @ (design * robust[:, None])
            rhs = design.T @ (unwrapped * robust)
            position = np.linalg.solve(normal + np.eye(2) * 1e-6, rhs)
        confidence = np.sum(weight * np.cos(phase - design @ position)) / weight.sum()
        if confidence >= 0.35:
            result[k] = position * 2 - (phases[k] - phases[2])
    return np.clip(result, -8, 8)


def _bilinear(raw):
    _, h, w = raw.shape
    pad = np.pad(raw, ((0, 0), (1, 1), (1, 1)), mode="edge")
    out = np.empty((16, h, w), dtype=np.float32)
    for k, (r, c) in enumerate(_PHASES):
        src = _KNOWN[k]
        cur = pad[src, 1:h + 1, 1:w + 1]
        for p, (rt, ct) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            dst = out[k * 4 + p]
            dx = (1 if c == 0 else -1) if ct != c else 0
            dy = (1 if r == 0 else -1) if rt != r else 0
            if not dx and not dy:
                dst[:] = cur
            elif not dx:
                np.add(cur, pad[src, 1 + dy:h + 1 + dy, 1:w + 1], out=dst)
                dst *= 0.5
            elif not dy:
                np.add(cur, pad[src, 1:h + 1, 1 + dx:w + 1 + dx], out=dst)
                dst *= 0.5
            else:
                np.add(cur, pad[src, 1 + dy:h + 1 + dy, 1:w + 1], out=dst)
                dst += pad[src, 1:h + 1, 1 + dx:w + 1 + dx]
                dst += pad[src, 1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
                dst *= 0.25
    return out


def _inputs(raw):
    mean = raw.mean(axis=0)
    pad = np.pad(mean, ((1, 1), (1, 1)), mode="reflect")
    h, w = mean.shape
    mean[:] = 0
    for r in range(3):
        for c in range(3):
            mean += pad[r:r + h, c:c + w]
    mean *= 1 / 9
    result = np.empty((5, h, w), dtype=np.float32)
    result[:4] = (raw - mean) * (1 / 32)
    result[4] = mean * (1 / 128)
    return result


def _conv(x, weight, bias, relu):
    cout, cin, kh, kw = weight.shape
    _, h, w = x.shape
    py, px = kh // 2, kw // 2
    pad = np.pad(x, ((0, 0), (py, py), (px, px)), mode="reflect")
    matrix = weight.reshape(cout, -1)
    out = np.empty((cout, h, w), dtype=np.float32)
    strip = 8
    col = np.empty((cin, kh, kw, min(strip, h), w), dtype=np.float32)
    for r in range(0, h, strip):
        rows = min(strip, h - r)
        if rows != col.shape[3]:
            col = np.empty((cin, kh, kw, rows, w), dtype=np.float32)
        for j in range(kh):
            for i in range(kw):
                col[:, j, i] = pad[:, r + j:r + j + rows, i:i + w]
        y = matrix @ col.reshape(cin * kh * kw, rows * w)
        y += bias[:, None]
        if relu:
            np.maximum(y, 0, out=y)
        out[:, r:r + rows] = y.reshape(cout, rows, w)
    return out


def reconstruct(raw):
    """Return float32 planes in the order 0, 45, 90, 135 degrees."""
    raw = np.asarray(raw)
    h, w = raw.shape
    packed = np.stack(
        [raw[r::2, c::2] for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    ).astype(np.float32)
    if np.all(np.ptp(packed, axis=(1, 2)) == 0):
        out = np.empty((4, h, w), dtype=np.float32)
        out[:] = packed[_KNOWN, 0, 0, None, None]
        return out
    base = _bilinear(packed)
    x = _inputs(packed)
    if _SHIFT_WEIGHT is not None:
        offsets = _translations(packed[_KNOWN])
        relative = (offsets[[0, 1, 3]] - offsets[2:3]).reshape(6).astype(np.float32) * 0.5
        first_bias = _WEIGHTS[0][1] + _SHIFT_WEIGHT @ relative
    for i, (weight, bias) in enumerate(_WEIGHTS):
        if i == 0 and _SHIFT_WEIGHT is not None:
            weight, bias = _FIRST_WEIGHT, first_bias
        x = _conv(x, weight, bias, i != len(_WEIGHTS) - 1)
    base[_UNKNOWN] += x * 32
    np.clip(base, 0, 255, out=base)
    base = base.reshape(4, 4, h // 2, w // 2)
    out = np.empty((4, h, w), dtype=np.float32)
    for p, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        out[:, r::2, c::2] = base[:, p]
    return out
