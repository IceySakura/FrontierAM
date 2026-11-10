"""Per-angle bilinear interpolation for a uint8 polarization mosaic."""
import numpy as np


def _interpolate_axis(plane, offset):
    """Fill alternate rows, extending the nearest measured row at either end."""
    known = plane[offset::2]
    last = offset + 2 * (len(known) - 1)
    between = plane[offset + 1:last:2]
    np.add(known[:-1], known[1:], out=between)
    between *= 0.5
    if offset:
        plane[0] = known[0]
    if last < len(plane) - 1:
        plane[-1] = known[-1]


def reconstruct(raw):
    """Return float64 planes in 0°, 45°, 90°, 135° order."""
    raw = np.asarray(raw)
    if raw.dtype != np.uint8 or raw.ndim != 2 or min(raw.shape) < 2:
        raise ValueError("expected an H×W uint8 mosaic with both dimensions at least two")
    result = np.empty((4, *raw.shape), dtype=np.float64)
    for plane, (row, col) in zip(result, ((1, 1), (0, 1), (0, 0), (1, 0))):
        measured_rows = plane[row::2]
        measured_rows[:, col::2] = raw[row::2, col::2]
        _interpolate_axis(measured_rows.T, col)
        _interpolate_axis(plane, row)
    return result
