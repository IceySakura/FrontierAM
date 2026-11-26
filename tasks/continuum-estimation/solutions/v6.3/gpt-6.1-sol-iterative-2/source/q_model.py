"""One-dimensional marginalization of the second convergence order."""

import math
import numpy as np

from bayesian import CORRECTION_GAP, coefficient_density, quantile, weighted_median


def positive_roots(x, y, qgrid):
    logmax = math.log(x[-1])
    logx = np.log(x / x[-1])
    scale = np.ptp(y)
    target = (y[1:] - y[0]) / scale

    def design(p, q):
        powers = np.column_stack([p, q, q + CORRECTION_GAP])
        allx = np.exp(logx[None, :, None] * powers[:, None, :])
        mat = allx[:, 1:] * -np.expm1(
            (logx[0] - logx[1:])[None, :, None] * powers[:, None, :])
        return powers, mat, allx

    def determinant(p, q):
        _, mat, _ = design(p, q)
        col = np.broadcast_to(target, mat.shape[:2])[:, :, None]
        return np.linalg.det(np.concatenate([mat, col], axis=2))

    lo = np.maximum(0.4, qgrid - 4)
    hi = np.minimum(4, qgrid - 0.08)
    domain = hi > lo
    q, lo, hi = qgrid[domain], lo[domain], hi[domain]
    flo, fhi = determinant(lo, q), determinant(hi, q)
    # The other determinant zeros are p=0, q, and q+1.5. There is at most
    # one additional positive-order root below q.
    valid = ((np.signbit(flo) != np.signbit(fhi)) | (flo == 0) | (fhi == 0))
    q, lo, hi, flo, fhi = q[valid], lo[valid], hi[valid], flo[valid], fhi[valid]
    if q.size == 0:
        raise ValueError("no admissible positive-order roots")
    hi = np.where(flo == 0, lo, hi)
    lo = np.where(fhi == 0, hi, lo)
    for _ in range(42):
        middle = (lo + hi) * 0.5
        fm = determinant(middle, q)
        same = np.signbit(fm) == np.signbit(flo)
        same &= fm != 0
        lo = np.where(same, middle, lo)
        flo = np.where(same, fm, flo)
        hi = np.where(same, hi, middle)
    p = (lo + hi) * 0.5
    powers, mat, allx = design(p, q)
    qmat, rmat = np.linalg.qr(mat)
    coef = np.linalg.solve(rmat, np.einsum("bik,i->bk", qmat, target))
    physical = coef * scale * np.exp(-logmax * powers)
    dleading = allx[:, 1:, 0] * logx[None, 1:]
    dleading -= allx[:, :1, 0] * logx[0]
    vp = coef[:, None, 0] * dleading
    jac = np.concatenate([mat, vp[:, :, None]], axis=2)
    _, logdet = np.linalg.slogdet(jac)
    logdet += np.sum(powers, axis=1) * logmax
    limit = y[0] - scale * np.sum(coef * allx[:, 0], axis=1)
    return p, q, limit, physical, logdet


def q_estimate(x, y, grid=5000, posterior=False, model_weight=1.0):
    def density(p, q, coeff):
        return coefficient_density(p, q - p, coeff) + math.log(model_weight)

    qgrid = np.linspace(0.48, 8, grid)
    p, q, limits, coeff, logdet = positive_roots(x, y, qgrid)
    den = density(p, q, coeff)
    weights = den - logdet
    low = quantile(q, weights, 0.00005)
    high = quantile(q, weights, 0.99995)
    dq = qgrid[1] - qgrid[0]
    qgrid = np.linspace(max(0.48, low - 2 * dq), min(8, high + 2 * dq), grid)
    p, q, limits, coeff, logdet = positive_roots(x, y, qgrid)
    den = density(p, q, coeff)
    weights = den - logdet
    mode = int(np.nanargmax(weights))
    if not np.isfinite(weights[mode]) or den[mode] < -500:
        raise ValueError("smooth expansion is unsupported")
    if posterior:
        volume = math.log(qgrid[1] - qgrid[0]) - math.log(np.ptp(y))
        return limits, weights + volume
    return weighted_median(limits, weights)
