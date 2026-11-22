"""Single-threaded residual demosaicking for the 90/45/135/0 mosaic."""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import ctypes
import numpy as np


# Also limit BLAS when the caller imported NumPy before importing this module.
try:
    _blas = ctypes.CDLL(np.core._multiarray_umath.__file__)
    _set_threads = _blas.openblas_set_num_threads64_
    _set_threads.argtypes = [ctypes.c_int]
    _set_threads.restype = None
    _set_threads(1)
except (AttributeError, OSError):
    pass

_PHASES = ((1, 1), (0, 1), (0, 0), (1, 0))
_PLANES = (3, 1, 0, 2)
_MISSING = tuple(i for i in range(16) if i not in (3, 5, 8, 14))
_TILE = 4

with np.load(os.path.join(os.path.dirname(__file__), "weights.npz"),
             allow_pickle=False) as _archive:
    _LAYERS = sum(name.startswith("w") for name in _archive.files)
    _WEIGHTS = tuple(np.ascontiguousarray(_archive["w%d" % i], dtype=np.float32)
                     for i in range(_LAYERS))
    _BIASES = tuple(np.ascontiguousarray(_archive["b%d" % i], dtype=np.float32)
                    for i in range(_LAYERS))
_RADIUS = sum(weight.shape[2] // 2 for weight in _WEIGHTS)


def _convolve(x, weight, bias):
    channels_out, channels_in, kh, kw = weight.shape
    _, height, width = x.shape
    matrix = weight.reshape(channels_out, -1)
    if kh == 1:
        result = matrix @ x.reshape(channels_in, -1)
        result += bias[:, None]
        return result.reshape(channels_out, height, width)
    windows = np.lib.stride_tricks.sliding_window_view(
        x, (kh, kw), axis=(1, 2))
    height, width = windows.shape[1:3]
    result = np.empty((channels_out, height, width), dtype=np.float32)
    for top in range(0, height, _TILE):
        bottom = min(top + _TILE, height)
        columns = windows[:, top:bottom].transpose(0, 3, 4, 1, 2).reshape(
            channels_in * kh * kw, -1)
        block = matrix @ columns
        block += bias[:, None]
        result[:, top:bottom] = block.reshape(channels_out, bottom - top, width)
    return result


def reconstruct(raw):
    """Return float32 planes in angle order 0, 45, 90, 135 degrees."""
    height, width = raw.shape
    # Even padding preserves the mosaic phase. Valid convolutions consume it.
    extended = np.pad(raw, 2 * _RADIUS, mode="reflect")
    samples = np.stack((extended[0::2, 0::2], extended[0::2, 1::2],
                        extended[1::2, 0::2], extended[1::2, 1::2])).astype(np.float32)
    samples *= 1.0 / 32.0
    residual = samples
    for i, (weight, bias) in enumerate(zip(_WEIGHTS, _BIASES)):
        residual = _convolve(residual, weight, bias)
        if i < _LAYERS - 1:
            np.maximum(residual, 0, out=residual)

    # Add the independent-channel bilinear prediction on the staggered grid.
    samples = samples[:, _RADIUS:-_RADIUS, _RADIUS:-_RADIUS]
    half_h, half_w = samples.shape[1:]
    padded = np.pad(samples, ((0, 0), (1, 1), (1, 1)), mode="edge")
    output = np.empty((4, height, width), dtype=np.float32)
    for j, index in enumerate(_MISSING):
        channel, position = divmod(index, 4)
        sample_y, sample_x = _PHASES[channel]
        out_y, out_x = divmod(position, 2)
        yy = (0,) if out_y == sample_y else (0, out_y - sample_y)
        xx = (0,) if out_x == sample_x else (0, out_x - sample_x)
        baseline = np.zeros((half_h, half_w), dtype=np.float32)
        for dy in yy:
            for dx in xx:
                baseline += padded[_PLANES[channel],
                                   1 + dy:1 + dy + half_h,
                                   1 + dx:1 + dx + half_w]
        baseline *= 1.0 / (len(yy) * len(xx))
        baseline += residual[j]
        baseline *= 32.0
        output[channel, out_y::2, out_x::2] = baseline
    for channel, (y, x) in enumerate(_PHASES):
        output[channel, y::2, x::2] = raw[y::2, x::2]
    np.clip(output, 0, 255, out=output)
    return output
