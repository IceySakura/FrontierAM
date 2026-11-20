"""Estimate a continuum limit from four or five positive-level observations."""

import json
import math
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss


# A small third term after the second convergence order.
THIRD_GAP = 1.5
P_DOMAIN = (0.24, 3.55)
D_DOMAIN = (0.15, 2.85)
REGULAR_PRIORS = (
    (0.565, 2.75, 0.057, 0.10),  # p
    (0.425, 2.11, 0.049, 0.12),  # q - p
    (0.237, 1.52, 0.043, 0.064),  # |a|
    (-0.90, -0.085, 0.062, 0.041),  # b
    (-0.40, -0.036, 0.033, 0.031),  # c
    (-3.0, 3.0, 0.29, 0.29),  # T
)
BROAD_PRIORS = (
    (0.44, 2.94, 0.11, 0.21),
    (0.32, 2.25, 0.088, 0.25),
    (0.12, 1.66, 0.093, 0.20),
    (-0.99, -0.027, 0.15, 0.10),
    (-0.45, 0.012, 0.074, 0.060),
    (-3.18, 3.18, 0.55, 0.55),
)


class Levels:
    def __init__(self, x, y):
        order = np.argsort(x)
        self.x = np.asarray(x, dtype=float)[order]
        self.y = np.asarray(y, dtype=float)[order]
        self.log_max = math.log(self.x[-1])
        self.log_x = np.log(self.x)
        self.du = np.diff(self.log_x)

        # Relative subtraction resolves gaps even when log(x) has rounded away
        # several bits of the difference.
        relative = (self.x[1:] - self.x[:-1]) / self.x[1:]
        near = (relative > 0) & (relative < 0.5)
        self.du[near] = -np.log1p(-relative[near])
        self.u = np.r_[-np.cumsum(self.du[::-1])[::-1], 0.0]
        self.dy = np.diff(self.y)
        self.z = self.dy / self.du

    def column(self, order):
        """Adjacent secants in log x of (x / max(x)) ** order."""
        p = np.asarray(order)[..., None]
        return np.exp(p * self.u[1:]) * (-np.expm1(-p * self.du) / self.du)

    def column_derivative(self, order, column):
        p = np.asarray(order)[..., None]
        return self.u[1:] * column + np.exp(p * self.u[:-1])


def row_dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def c_quadrature(degree):
    node, weight = leggauss(degree)
    edges = np.array([-0.95, -0.55, -0.39, -0.035, 0.03, 0.40, 0.95])
    half_width = 0.5 * np.diff(edges)[:, None]
    center = 0.5 * (edges[:-1] + edges[1:])[:, None]
    return (center + half_width * node).ravel(), (half_width * weight).ravel()


def exact_fits(levels, resolution=None, third_gap=THIRD_GAP):
    """Quadrature points on the set of interpolating models.

    Columns of the result are T, p, q-p, a, b, c, and log measure. The
    measure includes the inverse Jacobian from conditioning on the data.
    """
    n = len(levels.x)
    pmin, pmax = P_DOMAIN
    dmin, dmax = D_DOMAIN
    qlo, qhi = pmin + dmin, pmax + dmax
    if resolution is None:
        resolution = 4096 if n == 4 else 6144
    dq = (qhi - qlo) / resolution
    qgrid = qlo + dq * (0.5 + np.arange(resolution))
    bgrid = levels.column(qgrid)
    cgrid = levels.column(qgrid + third_gap)

    if n == 4:
        if resolution >= 8192:
            degree = 20
        elif resolution >= 4096:
            degree = 16
        else:
            degree = 12
        cs, cw = c_quadrature(degree)
        qi = np.repeat(np.arange(len(qgrid)), len(cs))
        c = np.tile(cs, len(qgrid))
        log_dv = np.tile(np.log(cw * dq), len(qgrid))
        target = levels.z - (
            c * np.exp((qgrid[qi] + third_gap) * levels.log_max)
        )[:, None] * cgrid[qi]
        basis = bgrid[..., None]
    else:
        qi = np.arange(len(qgrid))
        log_dv = np.full(len(qgrid), math.log(dq))
        target = np.broadcast_to(levels.z, (len(qgrid), n - 1))
        basis = np.stack((bgrid, cgrid), axis=-1)

    orth, matrix = np.linalg.qr(basis, mode="complete")
    k = n - 3
    w0 = orth[qi, :, k]
    w1 = orth[qi, :, k + 1]
    eta0 = row_dot(target, w0)
    eta1 = row_dot(target, w1)
    eta_norm = np.hypot(eta0, eta1)
    h = (eta1[:, None] * w0 - eta0[:, None] * w1) / eta_norm[:, None]
    q = qgrid[qi]
    lo = np.maximum(pmin, q - dmax)
    hi = np.minimum(pmax, q - dmin)
    flo = row_dot(levels.column(lo), h)
    fhi = row_dot(levels.column(hi), h)
    keep = np.isfinite(flo) & np.isfinite(fhi) & (np.signbit(flo) != np.signbit(fhi))
    if not np.any(keep):
        return np.empty((0, 7))

    h, lo, hi = h[keep], lo[keep], hi[keep]
    qi, q = qi[keep], q[keep]
    slo = np.signbit(flo[keep])

    # The eliminated equation is an exponential polynomial in p. It
    # already has zeros at 0 and q (and q+THIRD_GAP for five points).
    # A system on n levels has at most n-1 zeros, so the bracketed
    # interior root is unique.
    for _ in range(34):
        mid = (lo + hi) * 0.5
        same = np.signbit(row_dot(levels.column(mid), h)) == slo
        lo = np.where(same, mid, lo)
        hi = np.where(same, hi, mid)
    p = (lo + hi) * 0.5
    vp = levels.column(p)
    d_vp = levels.column_derivative(p, vp)
    w0, w1 = w0[keep], w1[keep]
    proj0 = row_dot(w0, vp)
    proj1 = row_dot(w1, vp)
    norm = np.hypot(proj0, proj1)
    amplitude = (eta0[keep] * proj0 + eta1[keep] * proj1) / (norm * norm)
    lower_terms = target[keep] - amplitude[:, None] * vp

    bb = row_dot(orth[qi, :, 0], lower_terms)
    log_det = np.log(np.abs(amplitude * row_dot(h, d_vp) * norm))
    log_det += np.log(np.abs(matrix[qi, 0, 0]))
    log_det += (p + q) * levels.log_max
    if n == 4:
        bb = bb / matrix[qi, 0, 0]
        c = c[keep]
        cc = c * np.exp((q + third_gap) * levels.log_max)
    else:
        cc = row_dot(orth[qi, :, 1], lower_terms) / matrix[qi, 1, 1]
        bb = (bb - cc * matrix[qi, 0, 1]) / matrix[qi, 0, 0]
        c = cc * np.exp(-(q + third_gap) * levels.log_max)
        log_det += np.log(np.abs(matrix[qi, 1, 1]))
        log_det += (q + third_gap) * levels.log_max

    t = levels.y[0] - (
        amplitude * np.exp(p * levels.u[0])
        + bb * np.exp(q * levels.u[0])
        + cc * np.exp((q + third_gap) * levels.u[0])
    )
    a = amplitude * np.exp(-p * levels.log_max)
    b = bb * np.exp(-q * levels.log_max)
    rows = np.column_stack((t, p, q - p, a, b, c, log_dv[keep] - log_det))
    return rows[np.isfinite(rows).all(axis=1)]


def soft_box(v, lo, hi, below, above):
    dl = np.maximum(lo - v, 0.0) / below
    dh = np.maximum(v - hi, 0.0) / above
    return -0.5 * (dl * dl + dh * dh)


def box_density(v, lo, hi, below, above):
    return soft_box(v, lo, hi, below, above) - math.log(
        hi - lo + math.sqrt(math.pi / 2.0) * (below + above)
    )


def prior_component(fits, boxes, short_gap, symmetric):
    t, p, d, a, b, c, _ = fits.T
    p_box, d_box, a_box, b_box, c_box, t_box = boxes
    gap = box_density(d, *d_box)
    if short_gap:
        gap = np.logaddexp(
            math.log1p(-short_gap) + gap,
            math.log(short_gap) + box_density(d, 0.42, 0.73, 0.051, 0.11),
        )
    negative = box_density(b, *b_box) + box_density(c, *c_box)
    diffuse = box_density(b, -1.03, 1.03, 0.17, 0.17)
    diffuse += box_density(c, -0.44, 0.44, 0.09, 0.09)
    trailing = np.logaddexp(
        math.log1p(-symmetric) + negative, math.log(symmetric) + diffuse
    )
    return (
        box_density(p, *p_box) + gap + box_density(np.abs(a), *a_box)
        + trailing + box_density(t, *t_box)
    )


def log_weights(fits):
    regular = prior_component(fits, REGULAR_PRIORS, 0.22, 0.046)
    broad = prior_component(fits, BROAD_PRIORS, 0.06, 0.24)
    prior = np.logaddexp(math.log(0.84) + regular, math.log(0.16) + broad)
    # Normalized absolute loss corresponds to reweighting the posterior
    # by the reciprocal of its denominator, then taking the median.
    return fits[:, 6] + prior - np.log(np.maximum(1, np.abs(fits[:, 0])))


def quantile(values, log_w, prob=0.5):
    keep = np.isfinite(log_w) & np.isfinite(values)
    if not np.any(keep):
        return np.nan
    val, log_w = values[keep], log_w[keep]
    idx = np.argsort(val)
    w = np.exp(log_w[idx] - np.max(log_w))
    partial = np.cumsum(w)
    return np.interp(np.asarray(prob) * partial[-1], partial - 0.5 * w, val[idx])


def single_power(levels):
    """Return an unambiguous single-power extrapolation, if present."""
    dy = levels.dy
    nonzero = dy[dy != 0]
    if (
        len(nonzero) < 2 or not np.all(np.signbit(nonzero) == np.signbit(nonzero[0]))
        or levels.u[0] > -0.084
    ):
        return np.nan
    ulp = np.finfo(float).eps * np.max(np.abs(levels.y))
    if np.max(np.abs(dy)) < 256 * ulp:
        return np.nan
    good = np.flatnonzero(np.abs(dy) > 24 * ulp)
    if len(good) < 2:
        return np.nan
    centers = 0.5 * (levels.u[1:] + levels.u[:-1])
    i, j = max(
        ((i, j) for i in good for j in good if j > i),
        key=lambda ij: min(abs(dy[ij[0]]), abs(dy[ij[1]]))
        * (centers[ij[1]] - centers[ij[0]]),
    )
    lr = math.log(abs(dy[j])) - math.log(abs(dy[i]))
    plo, phi = 0.055, 30.0

    def ratio(p):
        return (
            p * (levels.u[j + 1] - levels.u[i + 1])
            + math.log(-math.expm1(-p * levels.du[j]))
            - math.log(-math.expm1(-p * levels.du[i]))
        )

    if not (ratio(plo) <= lr <= ratio(phi)):
        return np.nan
    for _ in range(56):
        mid = 0.5 * (plo + phi)
        if ratio(mid) < lr:
            plo = mid
        else:
            phi = mid
    p = 0.5 * (plo + phi)
    increments = levels.column(p) * levels.du
    pivot = np.argmax(increments)
    amp = dy[pivot] / increments[pivot]
    err = np.max(np.abs(amp * increments - dy))
    if not err <= 9 * ulp:
        return np.nan
    return levels.y[0] - amp * math.exp(p * levels.u[0])


def conservative(levels):
    """Bounded local extrapolation when the smooth model is unsupported."""
    y = levels.y
    unit = float(np.max(np.abs(y)))
    if unit == 0.0:
        return 0.0
    v = y / unit
    a = v[1] - v[0]
    b = v[2] - v[1]
    spread = float(np.max(v) - np.min(v))
    # A nearby reversal or an extreme local slope makes the finest
    # observation safer than extending the interpolating curve.
    if a * b <= 0.0 or spread == 0.0:
        return float(y[0])
    order = 1.45
    e = math.expm1(order * min(400.0, float(levels.du[0])))
    if e <= 0.0:
        return float(y[0])
    correction = min(abs(a / e), 0.9 * spread)
    return float(np.clip(v[0] - math.copysign(correction, a), -2.0, 2.0) * unit)


def regularized_fit(levels):
    """Fit plausible smooth shapes with relaxed, scale-free amplitudes."""
    unit = float(np.max(np.abs(levels.y)))
    y = levels.y / unit
    delta = np.diff(y)
    nonzero = delta[delta != 0]
    reversals = np.count_nonzero(np.signbit(nonzero[1:]) != np.signbit(nonzero[:-1]))
    span = -levels.u[0]
    spread = np.ptp(y)
    if spread == 0.0 or span < 0.008:
        return float(levels.y[0])
    if reversals >= 2:
        # Oscillation: average only the closest pair, downweighting the
        # coarser level according to its separation.
        fraction = 1.0 / (1.0 + math.exp(2.8 * min(30.0, levels.du[0])))
        return float((1.0 - fraction) * levels.y[0] + fraction * levels.y[1])

    p, d = np.meshgrid(np.linspace(0.25, 5.0, 81), np.linspace(0.30, 3.3, 37))
    p, d = p.ravel(), d.ravel()
    powers = np.column_stack((p, p + d, p + d + THIRD_GAP))
    differences = np.expm1(powers[:, None, :] * levels.u[None, :, None])
    level_mean = np.mean(differences, axis=1)
    scales = np.array([1.48, 0.59, 0.27]) / np.maximum(0.25, -differences[:, 0, :])
    n = len(y)
    contrast = np.linalg.qr(np.ones((n, 1)), mode="complete")[0][:, 1:]
    x = np.einsum("ij,kih->kjh", contrast, differences) * scales[:, None, :]
    z = contrast.T @ ((y - y[0]) / spread)
    cov = x @ np.swapaxes(x, 1, 2)
    candidates, weights = [], []
    prior = soft_box(p, 0.56, 2.75, 0.28, 0.90) + soft_box(d, 0.44, 2.2, 0.20, 0.55)
    bound = min(9.0, 3.0 / min(1.0, span))
    leveled_mean = (np.mean(y) - y[0]) / spread
    for noise, mass in ((0.0012, 0.49), (0.009, 0.34), (0.046, 0.17)):
        mat = cov + (noise * noise) * np.eye(n - 1)
        sign, log_det = np.linalg.slogdet(mat)
        sol = np.linalg.solve(mat, np.broadcast_to(z, (len(p), n - 1)))
        amplitude = np.einsum("kij,ki->kj", x, sol) * scales
        shift = leveled_mean - row_dot(1.0 + level_mean, amplitude)
        lw = prior + math.log(mass) - 0.5 * (log_det + sol @ z)
        lw -= 0.5 * (shift / (0.62 * bound)) ** 4
        lw[sign != 1] = -np.inf
        candidates.append(shift)
        weights.append(lw)
    correction = quantile(np.concatenate(candidates), np.concatenate(weights))
    return float((y[0] + np.clip(correction, -bound, bound) * spread) * unit)


def supported(fits):
    # Prevent a poor prior match from being interpreted as a strong signal
    # merely because the interpolation constraints are exact.
    a, b, c = np.abs(fits[:, 3:6]).T
    return np.any((a < 2.50) & (b < 2.30) & (c < 1.15))


def solve(levels):
    safe = float(levels.y[0])
    if np.all(levels.y == safe) or np.min(levels.du) <= 0.0:
        return safe
    if np.all(np.isfinite(levels.z)):
        single = single_power(levels)
        if np.isfinite(single):
            return float(single)
        if (
            -6.0 <= levels.log_max <= 0.18
            and levels.u[0] < -0.047
            and np.max(np.abs(levels.y)) < 7.0
        ):
            fits = exact_fits(levels)
            if supported(fits):
                weights = log_weights(fits)
                mass = np.exp(weights - np.max(weights))
                effective_count = np.sum(mass) ** 2 / np.sum(mass * mass)
                n = len(levels.x)
                if effective_count < (105 if n == 4 else 28):
                    fine = exact_fits(levels, 8192 if n == 4 else 24576)
                    if supported(fine):
                        fits, weights = fine, log_weights(fine)
                best = quantile(fits[:, 0], weights)
                if np.isfinite(best):
                    return float(best)
    result = regularized_fit(levels)
    return result if math.isfinite(result) else conservative(levels)


def estimate(x, y):
    with np.errstate(all="ignore"):
        levels = Levels(x, y)
        try:
            answer = solve(levels)
        except (np.linalg.LinAlgError, ArithmeticError, ValueError):
            answer = conservative(levels)
        return float(answer) if math.isfinite(answer) else float(levels.y[0])


def main():
    input_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/inputs/input.json")
    output_path = Path(sys.argv[2] if len(sys.argv) > 2 else "/workspace/result/output.json")
    with input_path.open(encoding="utf-8") as infile:
        obj = json.load(infile)
    value = estimate(obj["x"], obj["y"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump({"estimate": float(value)}, outfile, allow_nan=False)
        outfile.write("\n")


if __name__ == "__main__":
    main()
