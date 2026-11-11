"""Fast, joint reconstruction of a four-angle polarization mosaic."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import correlate1d, uniform_filter

try:
    from threadpoolctl import threadpool_limits
    _thread_limit = threadpool_limits(limits=1, user_api="blas")
except ImportError:
    _thread_limit = None


_PHASES = ((1, 1), (0, 1), (0, 0), (1, 0))

with np.load(Path(__file__).with_name("weights.npz"), allow_pickle=False) as _data:
    _layers = []
    for _i in range(sum(k.startswith("w") and k[1:].isdigit() for k in _data.files)):
        _weight = _data["w" + str(_i)].transpose(2, 3, 1, 0).copy()
        _bias = _data["b" + str(_i)].copy()
        _layers.append((_weight, _bias))
    _guided_input = "guided" in _data.files
    _hybrid = int(_data["hybrid"].item()) if "hybrid" in _data.files else -1
    _eps = float(_data["eps"].item()) if "eps" in _data.files else 1.0
    _post_blend = float(_data["post_blend"].item()) if "post_blend" in _data.files else 0.5

_x = np.arange(-6, 7, dtype=np.float64)
_gaussian = np.exp(-0.5 * (_x / 1.5) ** 2)
_gaussian /= _gaussian.sum()
_filter = np.convolve(_gaussian, [0.5, 1.0, 0.5])
_offsets = np.arange(-4, 5)
_kernels = {}
for _source in range(2):
    for _target in range(2):
        _index = _target - _source - 2 * _offsets + 7
        _kernels[_source, _target] = np.where(
            (_index >= 0) & (_index < 15),
            _filter[np.clip(_index, 0, 14)],
            0,
        ).astype(np.float32)


def _initial(raw):
    height, width = raw.shape
    smooth = np.empty((4, height, width), dtype=np.float32)
    for channel, (row, col) in enumerate(_PHASES):
        source = raw[row::2, col::2].astype(np.float32)
        h, w = source.shape
        yi = np.arange(-4, h + 4)
        xi = np.arange(-4, w + 4)
        yi = np.where(yi < 0, -yi - row, yi)
        yi = np.where(yi >= h, 2 * h - 1 - row - yi, yi)
        xi = np.where(xi < 0, -xi - col, xi)
        xi = np.where(xi >= w, 2 * w - 1 - col - xi, xi)
        padded = source[yi[:, None], xi[None, :]]
        for target_col in range(2):
            horizontal = correlate1d(
                padded, _kernels[col, target_col], axis=1, mode="constant"
            )[:, 4:4 + w]
            for target_row in range(2):
                vertical = correlate1d(
                    horizontal, _kernels[row, target_row], axis=0, mode="constant"
                )
                smooth[channel, target_row::2, target_col::2] = vertical[4:4 + h]
    observed = np.empty((height, width), dtype=np.float32)
    for channel, (row, col) in enumerate(_PHASES):
        observed[row::2, col::2] = smooth[channel, row::2, col::2]
    smooth -= observed
    smooth += raw
    return smooth


def _conv(x, weight, bias):
    kh, kw, channels, outputs = weight.shape
    height, width, _ = x.shape
    padded = np.pad(
        x, ((kh // 2, kh // 2), (kw // 2, kw // 2), (0, 0)), mode="reflect"
    )
    windows = sliding_window_view(padded, (kh, kw), axis=(0, 1))
    result = np.empty((height, width, outputs), dtype=np.float32)
    matrix = weight.reshape(-1, outputs)
    for row in range(0, height, 8):
        count = min(8, height - row)
        patch = windows[row:row + count].transpose(0, 1, 3, 4, 2)
        patch = patch.reshape(count * width, kh * kw * channels)
        dest = result[row:row + count].reshape(count * width, outputs)
        np.matmul(patch, matrix, out=dest)
        dest += bias
    return result


def _network(raw, initial):
    x = np.stack([raw[r::2, c::2] for r, c in _PHASES], axis=-1)
    x = x.astype(np.float32) / 64.0
    if _guided_input:
        guide = initial.mean(axis=0) / 64.0
        g = np.stack([guide[r::2, c::2] for r, c in _PHASES], axis=-1)
        d = x - g
        mean = g.mean(axis=-1)
        mu = uniform_filter(mean, 3, mode="mirror")
        square = uniform_filter((g * g).mean(axis=-1), 3, mode="mirror")
        std = np.sqrt(np.maximum(square - mu * mu, 0) + 1e-4)
        high = g - uniform_filter(mean, 5, mode="mirror")[:, :, None]
        x = np.concatenate((high, d, std[:, :, None]), axis=-1)
    else:
        x -= uniform_filter(x.mean(axis=-1), 5, mode="mirror")[:, :, None]
    for index, (weight, bias) in enumerate(_layers):
        x = _conv(x, weight, bias)
        if index + 1 != len(_layers):
            np.maximum(x, x * 0.1, out=x)
    return x


def _upsample(x, shape, row, col):
    out = np.empty(shape, dtype=np.float32)
    out[row::2, col::2] = x
    across = (x[:, :-1] + x[:, 1:]) * 0.5
    if col == 0:
        out[row::2, 1:-1:2] = across
        out[row::2, -1] = x[:, -1]
    else:
        out[row::2, 2::2] = across
        out[row::2, 0] = x[:, 0]
    down = (out[row:-2:2] + out[row + 2::2]) * 0.5
    if row == 0:
        out[1:-1:2] = down
        out[-1] = out[-2]
    else:
        out[2::2] = down
        out[0] = out[1]
    return out


def _regress(raw, guide):
    out = np.empty((4, *raw.shape), dtype=np.float32)
    for channel, (row, col) in enumerate(_PHASES):
        g = guide[row::2, col::2].copy()
        y = raw[row::2, col::2].astype(np.float32)
        mg = uniform_filter(g, 3, mode="mirror")
        my = uniform_filter(y, 3, mode="mirror")
        variance = uniform_filter(g * g, 3, mode="mirror") - mg * mg
        covariance = uniform_filter(g * y, 3, mode="mirror") - mg * my
        a = covariance / (np.maximum(variance, 0) + _eps)
        b = my - a * mg
        a = uniform_filter(a, 3, mode="mirror")
        b = uniform_filter(b, 3, mode="mirror")
        aa = _upsample(a, raw.shape, row, col)
        bb = _upsample(b, raw.shape, row, col)
        out[channel] = aa * guide + bb
        out[channel, row::2, col::2] = y
    return out


def _add_residual(out, residual):
    index = 0
    for observed, (row, col) in enumerate(_PHASES):
        for channel in range(4):
            if channel != observed:
                out[channel, row::2, col::2] += residual[:, :, index] * 64.0
                index += 1


def reconstruct(raw):
    """Return float32 intensities in the order 0, 45, 90, 135 degrees."""
    raw = np.asarray(raw)
    height, width = raw.shape
    if min(height, width) < 16 or height % 2 or width % 2:
        from scipy.ndimage import map_coordinates
        yy, xx = np.indices(raw.shape, dtype=np.float32)
        out = np.empty((4, height, width), dtype=np.float32)
        for channel, (row, col) in enumerate(_PHASES):
            source = raw[row::2, col::2].astype(np.float32)
            if source.size:
                out[channel] = map_coordinates(
                    source, [(yy - row) / 2, (xx - col) / 2], order=1, mode="nearest"
                )
            else:
                out[channel] = raw
        return out
    out = _initial(raw)
    residual = _network(raw, out)
    if _hybrid >= 0:
        guide = out.mean(axis=0)
        for index, (row, col) in enumerate(_PHASES):
            guide[row::2, col::2] += residual[:, :, index] * 64.0
        regression = _regress(raw, guide)
        if _hybrid:
            _add_residual(out, residual[:, :, 4:])
            regression += out
            regression *= 0.5
        out = regression
    else:
        _add_residual(out, residual)
        if _post_blend:
            regression = _regress(raw, out.mean(axis=0))
            out *= 1.0 - _post_blend
            out += _post_blend * regression
    np.clip(out, 0.0, 255.0, out=out)
    for channel, (row, col) in enumerate(_PHASES):
        out[channel, row::2, col::2] = raw[row::2, col::2]
    return out
