"""Marginalization over positive orders in a smooth error expansion."""

import math
import numpy as np

CORRECTION_SCALE = 0.5
LOW_DATA_CORRECTION_SCALE = 0.35
FIT_TOLERANCE = 1e-9
CORRECTION_GAP = 1.5


def soft_uniform(v, lower, upper, tail):
    distance = np.maximum(np.maximum(lower - v, v - upper), 0)
    return -0.5 * (distance / tail)**2


def weighted_median(values, logweights):
    good = np.isfinite(values) & np.isfinite(logweights)
    values, logweights = values[good], logweights[good]
    if values.size == 0:
        raise ValueError("empty extrapolation posterior")
    order = np.argsort(values)
    values, logweights = values[order], logweights[order]
    weights = np.exp(logweights - np.max(logweights))
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def smooth_estimate(x, y, grid=600, posterior=False):
    p = np.linspace(0.4, 4.0, grid)
    d = np.linspace(0.1, 3.5, grid)
    pp, dd = np.meshgrid(p, d, indexing="ij")
    pp, dd = pp.ravel(), dd.ravel()
    qq = pp + dd
    powers = np.column_stack([pp, qq, qq + CORRECTION_GAP])
    logs = np.log(x)
    mat = np.exp(logs[1:][None, :, None] * powers[:, None, :])
    first = np.exp(logs[0] * powers)
    # expm1 preserves the observations' differences on clustered grids.
    mat *= -np.expm1((logs[0] - logs[1:])[None, :, None] * powers[:, None, :])
    target = np.broadcast_to(y[1:] - y[0], mat.shape[:2])
    coeff = np.linalg.solve(mat, target)
    limit = y[0] - np.sum(first * coeff, axis=1)
    _, logdet = np.linalg.slogdet(mat)

    density = coefficient_density(pp, dd, coeff, correction_scale=LOW_DATA_CORRECTION_SCALE)
    weights = density - logdet
    mode = int(np.nanargmax(weights))
    if not np.isfinite(weights[mode]) or density[mode] < -500:
        raise ValueError("smooth expansion is unsupported")
    if posterior:
        volume = math.log(p[1] - p[0]) + math.log(d[1] - d[0])
        return limit, weights + volume
    return weighted_median(limit, weights)


def power_density(p, d, coeff):
    density = soft_uniform(p, 0.55, 2.8, 0.2)
    density += soft_uniform(d, 0.4, 2.4, 0.3)
    density -= 0.5 * (coeff[:, 0] / 0.95)**2
    density += soft_uniform(coeff[:, 1], -1.2, -0.1, 0.1)
    density = np.where((p >= 0.4) & (p <= 4) & (d >= 0.08) & (d <= 4),
                       density, -np.inf)
    return density


def coefficient_density(p, d, coeff, correction_scale=CORRECTION_SCALE):
    density = power_density(p, d, coeff)
    density += correction_density(coeff[:, 2], correction_scale)
    return density


def correction_density(c, scale=CORRECTION_SCALE):
    with np.errstate(over="ignore"):
        return -0.5 * (c / scale)**2 - math.log(math.sqrt(2 * math.pi) * scale)


def quantile(values, weights, fraction):
    good = np.isfinite(values) & np.isfinite(weights)
    values, weights = values[good], weights[good]
    if values.size == 0:
        raise ValueError("empty extrapolation posterior")
    ix = np.argsort(values)
    values, weights = values[ix], weights[ix]
    cumulative = np.cumsum(np.exp(weights - np.max(weights)))
    return float(values[np.searchsorted(cumulative, fraction * cumulative[-1])])
