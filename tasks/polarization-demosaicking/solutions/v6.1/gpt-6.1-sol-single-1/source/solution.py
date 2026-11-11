"""Four-angle polarization demosaicking using local affine residual models."""

import glob
import importlib.util
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np


# Import just the installed filtering extension, not SciPy's large Python API.
_scipy = importlib.util.find_spec("scipy")
_native_path = glob.glob(
    os.path.join(next(iter(_scipy.submodule_search_locations)), "ndimage", "_nd_image*.so")
)[0]
_spec = importlib.util.spec_from_file_location("_nd_image", _native_path)
_ni = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ni)

_PHASES = ((1, 1), (0, 1), (0, 0), (1, 0))
_LOWPASS = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16
_KERNELS = {}


def _correlate(a, kernel, axis):
    out = np.empty_like(a)
    _ni.correlate1d(a, kernel, axis, out, 3, 0.0, 0)
    return out


def _uniform(a, size, axis):
    out = np.empty_like(a)
    _ni.uniform_filter1d(a, size, axis, out, 3, 0.0, 0)
    return out


def _box(a, radius):
    out = np.empty_like(a)
    size = 2 * radius + 1
    _ni.uniform_filter1d(a, size, 0, out, 3, 0.0, 0)
    _ni.uniform_filter1d(out, size, 1, out, 3, 0.0, 0)
    return out


def _moments(a, radius):
    if radius not in _KERNELS:
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        variance = np.mean(x * x)
        curvature_variance = np.mean((x * x - variance) ** 2)
        size = 2 * radius + 1
        first = x / (np.sqrt(variance) * size)
        second = (x * x - variance) / (np.sqrt(curvature_variance) * size)
        center = -variance / np.sqrt(curvature_variance)
        _KERNELS[radius] = (first, second, center, second * center, variance)
    first, second, center, _, _ = _KERNELS[radius]
    size = 2 * radius + 1
    horizontal = _uniform(a, size, 1)
    vertical = _uniform(a, size, 0)
    moments = (
        _uniform(horizontal, size, 0),
        _correlate(vertical, first, 1),
        _correlate(horizontal, first, 0),
        _correlate(vertical, second, 1),
        _correlate(horizontal, second, 0),
    )
    return moments, center


def _upsample(a, shape, row, col):
    out = np.empty(shape, dtype=np.float32)
    out[row::2, col::2] = a
    if col == 0:
        out[row::2, 1:-1:2] = (a[:, :-1] + a[:, 1:]) * 0.5
        out[row::2, -1] = a[:, -1]
    else:
        out[row::2, 2::2] = (a[:, :-1] + a[:, 1:]) * 0.5
        out[row::2, 0] = a[:, 0]
    if row == 0:
        out[1:-1:2] = (out[0:-2:2] + out[2::2]) * 0.5
        out[-1] = out[-2]
    else:
        out[2::2] = (out[1:-2:2] + out[3::2]) * 0.5
        out[0] = out[1]
    return out


def _polynomial_smooth(a, radius):
    first, _, _, curvature, variance = _KERNELS[radius]
    size = 2 * radius + 1
    if min(a.shape) < size:
        horizontal = _uniform(a, size, 1)
        vertical = _uniform(a, size, 0)
        return (
            _uniform(horizontal, size, 0)
            + _correlate(vertical, curvature, 1)
            + _correlate(horizontal, curvature, 0)
        )

    # Clamp window centers, then evaluate the polynomial at the actual pixel.
    # This avoids inventing reflected coefficient samples at the image edges.
    def clamp(out, axis):
        if axis == 0:
            out[:radius] = out[radius]
            out[-radius:] = out[-radius - 1]
        else:
            out[:, :radius] = out[:, radius:radius + 1]
            out[:, -radius:] = out[:, -radius - 1:-radius]
        return out

    horizontal = clamp(_uniform(a, size, 1), 1)
    vertical = clamp(_uniform(a, size, 0), 0)
    mean = clamp(_uniform(horizontal, size, 0), 0)
    curvature_x = clamp(_correlate(vertical, curvature, 1), 1)
    curvature_y = clamp(_correlate(horizontal, curvature, 0), 0)
    out = mean + curvature_x + curvature_y
    edges = (
        (slice(0, radius), slice(0, size), np.arange(-radius, 0, dtype=np.float64)),
        (slice(-radius, None), slice(-size, None), np.arange(1, radius + 1, dtype=np.float64)),
    )
    for region, window, t in edges:
        slope_x = np.sum(vertical[:, window] * first, axis=1)
        slope_y = np.sum(horizontal[window] * first[:, None], axis=0)
        out[:, region] += (
            slope_x[:, None] * (t / np.sqrt(variance))
            - curvature_x[:, region] * (t * t / variance)
        )
        out[region] += (
            slope_y[None] * (t / np.sqrt(variance))[:, None]
            - curvature_y[region] * (t * t / variance)[:, None]
        )
    for rows, yw, ty in edges:
        for cols, xw, tx in edges:
            mixed = np.sum(a[yw, xw] * first[:, None] * first[None])
            out[rows, cols] += mixed * ty[:, None] * tx[None] / variance
    return out


def _affine_pass(raw, guide, radius=3, epsilon=4.0):
    gains = []
    offsets = []
    for row, col in _PHASES:
        measured = raw[row::2, col::2].astype(np.float32)
        sampled_guide = guide[row::2, col::2]
        gm, center = _moments(sampled_guide, radius)
        rm, _ = _moments(measured, radius)
        variance = _box(sampled_guide * sampled_guide, radius)
        covariance = _box(sampled_guide * measured, radius)
        for g, r in zip(gm, rm):
            variance -= g * g
            covariance -= g * r
        # Detrending separates slowly varying polarization from shared texture.
        prior = 0.5 + 0.5 * rm[0] / np.maximum(gm[0], 1)
        gain = prior + (covariance - prior * variance) / (np.maximum(variance, 0) + epsilon)
        gain = _box(np.clip(gain, 0.05, 4), 1)
        smooth_g = gm[0] + center * (gm[3] + gm[4])
        smooth_r = rm[0] + center * (rm[3] + rm[4])
        offset = smooth_r - gain * smooth_g
        averaged_gain = _polynomial_smooth(gain, radius)
        offset = _polynomial_smooth(offset, radius)
        gain = np.clip(averaged_gain, 0.05, 4)
        offset += (averaged_gain - gain) * smooth_g
        gains.append(_upsample(gain, raw.shape, row, col))
        offsets.append(_upsample(offset, raw.shape, row, col))

    intensity = np.empty(raw.shape, dtype=np.float32)
    for gain, offset, (row, col) in zip(gains, offsets, _PHASES):
        sample = np.s_[row::2, col::2]
        denominator = gain[sample]
        intensity[sample] = (
            denominator * (raw[sample] - offset[sample]) + 0.01 * guide[sample]
        ) / (denominator * denominator + 0.01)

    out = np.empty((4,) + raw.shape, dtype=np.float32)
    for channel, (gain, offset, (row, col)) in enumerate(zip(gains, offsets, _PHASES)):
        out[channel] = gain * intensity + offset
        out[channel, row::2, col::2] = raw[row::2, col::2]
    np.clip(out, 0, 255, out=out)
    return out


def reconstruct(raw):
    """Return aligned float32 planes in the order 0, 45, 90, 135 degrees."""
    raw = np.asarray(raw)
    height, width = raw.shape
    if min(height, width) < 2:
        return np.repeat(raw.astype(np.float32)[None], 4, axis=0)
    if height % 2 or width % 2:
        padded = np.pad(raw, ((0, height % 2), (0, width % 2)), mode="reflect")
        return reconstruct(padded)[:, :height, :width].copy()
    guide = _correlate(_correlate(raw.astype(np.float32), _LOWPASS, 0), _LOWPASS, 1)
    return _affine_pass(raw, guide)
