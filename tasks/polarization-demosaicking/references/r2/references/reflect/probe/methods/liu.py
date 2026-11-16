"""CPU port of Liu et al.'s LEPD and LEIC for the [0,45;135,90] mosaic."""

import numpy as np
from scipy.ndimage import convolve
from scipy.special import expit


def recon(raw, method):
    """Return float64 planes (0, 45, 90, 135) from 0..255 intensity units, k=1."""
    method = method.upper()
    if method not in ("LEPD", "LEIC"):
        raise ValueError("expected LEPD or LEIC")
    raw = np.asarray(raw, dtype=np.float64)
    height, width = raw.shape
    if min(height, width) < 5:
        raise ValueError("LEPD requires a 5×5 neighborhood")

    def at(image, row=0, col=0):
        return image[2 + row:height - 2 + row, 2 + col:width - 2 + col]

    def diagonal_estimate(col):
        near, far = at(raw, -1, col), at(raw, 1, -col)
        second = (2 * at(raw) - at(raw, -2, 2 * col) - at(raw, 2, -2 * col)) / 8
        variation = abs(near - far) / (2 ** 1.5) + (2 ** 1.5) * abs(second)
        return (near + far) / 2 + second, variation

    d, vd = diagonal_estimate(1)
    a, va = diagonal_estimate(-1)
    weight = expit(va - vd)
    diagonal = np.zeros_like(raw)
    at(diagonal)[:] = weight * d + (1 - weight) * a
    residual = np.zeros_like(raw)
    at(residual)[:] = at(raw) - at(diagonal)

    def axis_estimate(row, col):
        contrast = abs(at(residual, row, col) - at(residual, -row, -col)) / 2
        curvature = (2 * at(residual) - at(residual, 2 * row, 2 * col)
                     - at(residual, -2 * row, -2 * col)) / 4
        correction = (2 * at(raw) - at(raw, 2 * row, 2 * col)
                      - at(raw, -2 * row, -2 * col)) / 4
        same = (at(raw, -row, -col) + at(raw, row, col)) / 2 + correction
        orthogonal = (at(diagonal, -row, -col) + at(diagonal, row, col)) / 2 + correction
        return contrast + 2 * abs(curvature), same, orthogonal

    vh, h_same, h_orth = axis_estimate(0, 1)
    vv, v_same, v_orth = axis_estimate(1, 0)
    wh = expit(vv - vh)
    horizontal = wh * h_same + (1 - wh) * v_orth
    vertical = wh * h_orth + (1 - wh) * v_same
    phase = ((0, 1), (3, 2))
    planes = np.zeros((4, height, width), dtype=np.float64)
    for row in range(2):
        for col in range(2):
            for channel, field in (
                (phase[row][col], at(raw)), (phase[1-row][1-col], at(diagonal)),
                (phase[row][1-col], horizontal), (phase[1-row][col], vertical),
            ):
                at(planes[channel])[row::2, col::2] = field[row::2, col::2]
    if method == "LEPD":
        return planes

    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 4
    result = np.empty_like(planes)
    w_orth = 1 / (1 + 2 * np.sqrt(2))
    w_other = np.sqrt(2) / (1 + 2 * np.sqrt(2))
    for row in range(2):
        for col in range(2):
            channel = phase[row][col]

            def corrected(neighbor):
                diff = np.zeros_like(raw)
                diff[row::2, col::2] = raw[row::2, col::2] - planes[neighbor, row::2, col::2]
                return planes[neighbor] + convolve(diff, kernel, mode="constant", cval=0.0)

            result[channel] = (w_other * (corrected((channel - 1) % 4) + corrected((channel + 1) % 4))
                               + w_orth * corrected((channel + 2) % 4))
    return result
