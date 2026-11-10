"""Independent full-frame polarization scoring and channelwise bilinear baseline."""

import numpy as np


# Sensor row, column parity for output channels 0°, 45°, 90°, 135°.
OFFSETS = ((0, 0), (0, 1), (1, 1), (1, 0))


def _image(values, ndim):
    array = np.asarray(values)
    if array.ndim != ndim or (ndim == 3 and array.shape[0] != 4):
        raise ValueError("expected (4,H,W) channels or (H,W) raw image")
    if min(array.shape[-2:]) < 2:
        raise ValueError("each sensor axis needs at least two pixels")
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError("image must contain finite real intensities")
    return array


def mosaic(ref):
    """Take the measured channel at each pixel. Preserve the reference dtype."""
    ref = _image(ref, 3)
    raw = np.empty(ref.shape[1:], dtype=ref.dtype)
    for channel, (row, col) in enumerate(OFFSETS):
        raw[row::2, col::2] = ref[channel, row::2, col::2]
    return raw


def _neighbors(length, offset):
    """Indices of the bracketing samples on one period-two axis, clamped at ends."""
    coords = np.arange(length) - offset
    last = (length - 1 - offset) // 2
    return np.clip(coords // 2, 0, last), np.clip((coords + 1) // 2, 0, last)


def bilinear(raw):
    """Interpolate each channel's grid, extending its nearest samples at edges."""
    raw = _image(raw, 2).astype(np.float64, copy=False)
    height, width = raw.shape
    output = np.empty((4, height, width), dtype=np.float64)
    for channel, (row, col) in enumerate(OFFSETS):
        samples = raw[row::2, col::2]
        left, right = _neighbors(width, col)
        top, bottom = _neighbors(height, row)
        rows = 0.5 * samples[:, left] + 0.5 * samples[:, right]
        output[channel] = 0.5 * rows[top] + 0.5 * rows[bottom]
    return output


def rmse_per_channel(pred, ref):
    """Return four full-image RMSEs in intensity units, in channel order."""
    pred, ref = _image(pred, 3), _image(ref, 3)
    if pred.shape != ref.shape:
        raise ValueError("prediction and reference dimensions differ")
    # Scaling before squaring keeps even very large finite prediction errors scoreable.
    delta = np.subtract(pred, ref, dtype=np.float64)
    scale = np.max(np.abs(delta), axis=(1, 2))
    delta /= np.where(scale == 0, 1.0, scale)[:, None, None]
    return scale * np.sqrt(np.mean(np.square(delta), axis=(1, 2)))


def score(pred, ref):
    """Single-scene mean of the four RMSEs, lower is better."""
    return float((rmse_per_channel(pred, ref) / 4).sum())
