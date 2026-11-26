"""Two-power order-gap marginalization for four observations."""

import math
import numpy as np

from bayesian import power_density, quantile, weighted_median


def interpolation_roots(x, y, dgrid, pgrid):
    logmax = np.log(x[-1])
    logx = np.log(x / x[-1])
    scale = np.ptp(y)
    target = (y[1:] - y[0]) / scale

    def design(p, d):
        powers = np.column_stack([p, p + d])
        allx = np.exp(logx[None, :, None] * powers[:, None, :])
        return powers, allx[:, 1:] - allx[:, :1], allx

    def determinant(p, d):
        _, mat, _ = design(p, d)
        target_col = np.broadcast_to(target, mat.shape[:2])[:, :, None]
        return np.linalg.det(np.concatenate([mat, target_col], axis=2))

    dd, pp = np.meshgrid(dgrid, pgrid, indexing="ij")
    vals = determinant(pp.ravel(), dd.ravel()).reshape(pp.shape)
    rows, cols = np.where(vals[:, :-1] * vals[:, 1:] < 0)
    if rows.size == 0:
        raise ValueError("no two-power roots")
    d = dgrid[rows]
    lo, hi = pgrid[cols].copy(), pgrid[cols + 1].copy()
    flo = vals[rows, cols].copy()
    for _ in range(35):
        middle = (lo + hi) * 0.5
        fm = determinant(middle, d)
        same = fm * flo > 0
        lo = np.where(same, middle, lo)
        flo = np.where(same, fm, flo)
        hi = np.where(same, hi, middle)
    p = (lo + hi) * 0.5
    powers, mat, allx = design(p, d)
    qmat, rmat = np.linalg.qr(mat)
    coef = np.linalg.solve(rmat, np.einsum("bik,i->bk", qmat, target))
    physical = coef * scale * np.exp(-logmax * powers)
    derivative = allx * logx[None, :, None]
    derivmat = derivative[:, 1:] - derivative[:, :1]
    vp = np.sum(derivmat * coef[:, None, :], axis=2)
    jac = np.concatenate([mat, vp[:, :, None]], axis=2)
    _, logdet = np.linalg.slogdet(jac)
    logdet += powers.sum(axis=1) * logmax
    limits = y[0] - scale * np.sum(coef * allx[:, 0], axis=1)
    return p, d, limits, physical, logdet


def gap_estimate(x, y, grid=1200, posterior=False):
    dgrid = np.linspace(0.15, 3.8, grid)
    pgrid = np.linspace(0.4, 4, 220)
    p, d, limits, coeff, logdet = interpolation_roots(x, y, dgrid, pgrid)
    den = power_density(p, d, coeff)
    weights = den - logdet
    low = quantile(d, weights, 0.0001)
    high = quantile(d, weights, 0.9999)
    dd = dgrid[1] - dgrid[0]
    dgrid = np.linspace(max(0.15, low - 2 * dd), min(3.8, high + 2 * dd), grid)
    p, d, limits, coeff, logdet = interpolation_roots(x, y, dgrid, pgrid)
    den = power_density(p, d, coeff)
    weights = den - logdet
    mode = int(np.nanargmax(weights))
    if not np.isfinite(weights[mode]) or den[mode] < -500:
        raise ValueError("two-power expansion is unsupported")
    if posterior:
        volume = math.log(dgrid[1] - dgrid[0]) - math.log(np.ptp(y))
        return limits, weights + volume
    return weighted_median(limits, weights)
