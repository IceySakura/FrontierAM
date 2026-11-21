#!/usr/bin/env python3
"""Small-sample continuum extrapolation by marginalizing power-series fits."""

import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.optimize import brentq, minimize_scalar


P_MIN = 0.05
P_MAX = 3.5
D_MIN = 0.1
D_MAX = 4.0
EXTRA_ORDER = 1.5


def _weighted_median(values, log_weights):
    valid = np.isfinite(values) & np.isfinite(log_weights)
    values = values[valid]
    log_weights = log_weights[valid]
    if not len(values):
        raise ArithmeticError("no finite posterior mass")
    log_weights = log_weights - np.max(log_weights)
    keep = log_weights > -45.0
    values = values[keep]
    weights = np.exp(log_weights[keep])
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    positions = (cumulative - 0.5 * weights) / cumulative[-1]
    return float(np.interp(0.5, positions, values))


def _log_prior(rows):
    p, q, limit, a, b, c, log_jacobian, gap = rows.T
    core = (
        (p <= 2.8) & (gap >= 0.4) & (gap <= 2.2)
        & (np.abs(a) <= 1.5) & (np.abs(b) <= 0.9) & (np.abs(c) <= 0.4)
    )
    negative = (b <= -0.1) & (c <= -0.05)
    core_weight = core * (negative.astype(float) + 0.02)
    wide = (
        (p >= 0.4) & (p <= 3.2) & (gap >= 0.2) & (gap <= 3.0)
    )
    cost = (a / 1.0)**2 + (b / 0.6)**2 + (c / 0.25)**2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        core_log = np.log(core_weight) - np.maximum(0.6 - p, 0.0) / 0.08
        wide_log = np.where(wide, math.log(0.005) - 0.5 * cost, -np.inf)
        return np.logaddexp(core_log, wide_log)


def _posterior(rows):
    if not len(rows):
        raise ArithmeticError("empty interpolation family")
    finite = np.all(np.isfinite(rows), axis=1)
    rows = rows[finite]
    if not len(rows):
        raise ArithmeticError("ill-conditioned interpolation")
    lp = _log_prior(rows)
    if np.max(lp) < -45:
        raise ArithmeticError("observations outside the regular model")
    # The loss is absolute error divided by max(1, abs(the unknown limit)).
    log_weights = rows[:, 6] + lp - np.log(np.maximum(1.0, np.abs(rows[:, 2])))
    return _weighted_median(rows[:, 2], log_weights)


def _single_power(x, y):
    lx = np.log(x / x[-1])
    centered_y = y - np.mean(y)
    span = max(float(np.ptp(y)), np.finfo(float).tiny)

    def fit(p):
        xp = np.exp(p * lx)
        centered_x = xp - np.mean(xp)
        norm = centered_x @ centered_x
        if norm <= 1e-300:
            return np.inf, float(y[0])
        a = (centered_x @ centered_y) / norm
        residual = centered_y - a * centered_x
        limit = np.mean(y) - a * np.mean(xp)
        return float((residual / span) @ (residual / span)), float(limit)

    def log_expm1(value):
        if value > 40:
            return value + math.log1p(-math.exp(-value))
        return math.log(math.expm1(value))

    # Three well-separated observations give an accurate order for a pure
    # power, without the stopping error of scalar minimization.
    candidate = None
    for indices in [(0, len(x) // 2, len(x)-1), (len(x)-3, len(x)-2, len(x)-1)]:
        i, j, k = indices
        dy1, dy2 = y[j] - y[i], y[k] - y[j]
        if dy1 == 0 or dy2 == 0 or np.sign(dy1) != np.sign(dy2):
            continue
        h1, h2 = lx[j] - lx[i], lx[k] - lx[j]
        if h1 <= 0 or h2 <= 0:
            continue
        target = math.log(abs(dy2)) - math.log(abs(dy1))

        def equation(p):
            return p * h1 + log_expm1(p * h2) - log_expm1(p * h1) - target

        lo, hi = 0.002, 20.0
        if equation(lo) * equation(hi) > 0:
            continue
        p = brentq(equation, lo, hi, xtol=1e-13, maxiter=100)
        residual, limit = fit(p)
        if candidate is None or residual < candidate[1]:
            candidate = (limit, residual, p)
    if candidate is not None and candidate[1] < 1e-20:
        return candidate

    solution = minimize_scalar(
        lambda p: fit(p)[0], method="bounded", bounds=(0.04, 12.0),
        options={"xatol": 2e-11, "maxiter": 120},
    )
    residual, limit = fit(solution.x)
    if candidate is not None and candidate[1] < residual:
        return candidate
    return limit, residual, float(solution.x)


def _five_point_family(x, y, count=4096, q_values=None):
    """Use q as the free coordinate on the exact interpolation curve."""
    if q_values is None:
        q_values = np.linspace(P_MIN + D_MIN, P_MAX + D_MAX, count)
    log_x = np.log(x)
    log_u = log_x - log_x[-1]
    span = float(np.max(np.abs(y - y[0])))
    target = (y - y[0]) / span
    rows = []
    for q in q_values:
        lower = max(P_MIN, q - D_MAX)
        upper = min(P_MAX, q - D_MIN)
        if upper <= lower:
            continue
        uq = np.exp(q * log_u)
        ur = np.exp((q + EXTRA_ORDER) * log_u)
        fixed = np.column_stack((np.ones(5), uq, ur, target))
        _, singular, vh = np.linalg.svd(fixed.T, full_matrices=True)
        if singular[-1] < 2e-15:
            continue
        null = vh[-1]

        def residual(p):
            return float(null @ np.exp(p * log_u))

        fl, fh = residual(lower), residual(upper)
        if not np.isfinite(fl + fh) or fl * fh > 0:
            continue
        try:
            p = brentq(residual, lower, upper, xtol=4e-12, maxiter=70)
        except (ValueError, RuntimeError):
            continue
        up = np.exp(p * log_u)
        design = np.column_stack((np.ones(5), up, uq, ur))
        coefficients = np.linalg.lstsq(design, target, rcond=1e-14)[0] * span
        if not np.all(np.isfinite(coefficients)):
            continue
        jacobian = np.column_stack(
            (design, coefficients[1] * up * log_u)
        )
        _, log_det = np.linalg.slogdet(jacobian)
        # Undo column rescaling; the prior is on physical power coefficients.
        log_det += (p + 2 * q + EXTRA_ORDER) * log_x[-1]
        physical = coefficients[1:] * np.exp(
            -np.array([p, q, q + EXTRA_ORDER]) * log_x[-1]
        )
        rows.append((
            p, q, y[0] + coefficients[0], *physical, -log_det, q - p
        ))
    return np.asarray(rows, dtype=float).reshape(-1, 8)


def _five_point_estimate(x, y):
    q_values = np.linspace(P_MIN + D_MIN, P_MAX + D_MAX, 4096)
    coarse = _five_point_family(x, y, q_values=q_values)
    if not len(coarse):
        raise ArithmeticError("empty interpolation curve")
    lp = _log_prior(coarse)
    lw = coarse[:, 6] + lp - np.log(np.maximum(1.0, np.abs(coarse[:, 2])))
    finite = np.isfinite(lw)
    if not np.any(finite) or np.max(lp[finite]) < -120:
        raise ArithmeticError("no plausible interpolation curve")

    # Allocate additional nodes according to the initial posterior mass.
    # Cell widths below compensate for the resulting nonuniform quadrature.
    density = np.zeros(len(q_values))
    indices = np.searchsorted(q_values, coarse[finite, 1])
    density[indices] = np.exp(lw[finite] - np.max(lw[finite]))
    mass = (density[:-1] + density[1:]) * (0.5 * np.diff(q_values))
    cumulative = np.concatenate(([0.0], np.cumsum(mass)))
    quantiles = (np.arange(6144) + 0.5) * (cumulative[-1] / 6144)
    new_q = np.interp(quantiles, cumulative, q_values)
    new_q = np.setdiff1d(new_q, q_values)
    fine = _five_point_family(x, y, q_values=new_q)
    all_nodes = np.union1d(q_values, new_q)
    edges = np.concatenate((
        all_nodes[:1], (all_nodes[1:] + all_nodes[:-1]) * 0.5, all_nodes[-1:]
    ))
    widths = np.diff(edges)
    rows = np.concatenate((coarse, fine))
    rows[:, 6] += np.log(widths[np.searchsorted(all_nodes, rows[:, 1])])
    return _posterior(rows)


def _coefficient_quadrature():
    # Separate intervals at prior-density discontinuities.
    intervals = [
        (-1.8, -0.4, 24),
        (-0.4, -0.05, 56),
        (-0.05, 0.0, 12),
        (0.0, 0.4, 40),
        (0.4, 1.8, 24),
    ]
    values, weights = [], []
    for lo, hi, count in intervals:
        nodes, node_weights = np.polynomial.legendre.leggauss(count)
        values.extend((lo + hi) * 0.5 + nodes * ((hi - lo) * 0.5))
        weights.extend(node_weights * ((hi - lo) * 0.5))
    return np.asarray(values), np.asarray(weights)


def _four_point_family(x, y, count=3072):
    """Integrate over q and c, solving exactly for p, a, b, and the limit."""
    q_values = np.linspace(P_MIN + D_MIN, P_MAX + D_MAX, count)
    c_values, c_weights = _coefficient_quadrature()
    log_x = np.log(x)
    log_u = log_x - log_x[-1]
    delta_y = y[1:] - y[0]
    output = []

    for first in range(0, len(q_values), 64):
        qs = q_values[first:first+64]
        q, c = np.meshgrid(qs, c_values, indexing="ij")
        q, c = q.ravel(), c.ravel()
        log_quad = np.tile(np.log(c_weights), len(qs))
        uq = np.exp(q[:, None] * log_u)
        dq = uq[:, 1:] - uq[:, :1]
        vq = dq[:, :2] / dq[:, -1:]
        xr = np.exp((q[:, None] + EXTRA_ORDER) * log_x)
        dr = xr[:, 1:] - xr[:, :1]
        e = (
            delta_y[:2] - vq * delta_y[-1]
            - c[:, None] * (dr[:, :2] - vq * dr[:, -1:])
        )

        def residual(p):
            up = np.exp(p[:, None] * log_u)
            dp = up[:, 1:] - up[:, :1]
            h = dp[:, :2] / dp[:, -1:] - vq
            return (e[:, 0] * h[:, 1] - e[:, 1] * h[:, 0]) / (p - q)

        lower = np.maximum(P_MIN, q - D_MAX)
        upper = np.minimum(P_MAX, q - D_MIN)
        fl, fh = residual(lower), residual(upper)
        valid = (
            (lower < upper) & (fl * fh <= 0)
            & np.isfinite(fl) & np.isfinite(fh)
        )
        if not np.any(valid):
            continue
        q, c, uq, dq, vq, dr, e, log_quad, lower, upper, fl = (
            array[valid] for array in
            (q, c, uq, dq, vq, dr, e, log_quad, lower, upper, fl)
        )
        for _ in range(42):
            middle = (lower + upper) * 0.5
            fm = residual(middle)
            left = fl * fm <= 0
            upper = np.where(left, middle, upper)
            lower = np.where(left, lower, middle)
            fl = np.where(left, fl, fm)
        p = (lower + upper) * 0.5
        up = np.exp(p[:, None] * log_u)
        dp = up[:, 1:] - up[:, :1]
        h = dp[:, :2] - vq * dp[:, -1:]
        a_scaled = np.sum(e * h, axis=1) / np.sum(h * h, axis=1)
        b_scaled = (
            delta_y[-1] - a_scaled * dp[:, -1] - c * dr[:, -1]
        ) / dq[:, -1]
        limit = (
            y[0] - a_scaled * up[:, 0] - b_scaled * uq[:, 0]
            - c * np.exp((q + EXTRA_ORDER) * log_x[0])
        )
        jacobian = np.stack(
            (np.ones_like(up), up, uq, a_scaled[:, None] * up * log_u),
            axis=-1,
        )
        _, log_det = np.linalg.slogdet(jacobian)
        log_det += (p + q) * log_x[-1]
        a = a_scaled * np.exp(-p * log_x[-1])
        b = b_scaled * np.exp(-q * log_x[-1])
        output.append(np.column_stack((
            p, q, limit, a, b, c, -log_det + log_quad, q - p
        )))
    if not output:
        return np.empty((0, 8))
    return np.concatenate(output)


def _fallback(x, y, power_result=None):
    span = float(np.ptp(y))
    if span == 0:
        return float(y[0])
    differences = np.diff(y)
    monotone = np.all(differences >= 0) or np.all(differences <= 0)
    if monotone:
        if power_result is None:
            power_result = _single_power(x, y)
        value, residual, order = power_result
        if math.isfinite(value) and order > 0.08 and residual < 0.02:
            return float(np.clip(value, y[0] - 5 * span, y[0] + 5 * span))
    return float(0.7 * y[0] + 0.3 * y[1])


def estimate(x, y):
    order = np.argsort(x)
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    if len(x) < 2 or np.all(y == y[0]):
        return float(y[0])

    # Keep extreme but valid JSON numbers away from overflow.
    amplitude_unit = max(1.0, float(np.max(np.abs(y))) / 1e8)
    y = y / amplitude_unit
    log_x = np.log(x)
    if x[-1] > 2.0 or x[-1] < 1e-3:
        x = np.exp(np.maximum(log_x - log_x[-1], -700.0)) * 0.5
    if np.any(np.diff(x) <= 0) or np.log(x[-1] / x[0]) < 1e-7:
        return float(y[0] * amplitude_unit)

    power_result = None
    try:
        power_result = _single_power(x, y)
        limit, residual, order = power_result
        if residual < 1e-19 and 0.0021 < order < 19.99:
            result = limit
        elif len(x) == 5:
            result = _five_point_estimate(x, y)
        elif len(x) == 4:
            result = _posterior(_four_point_family(x, y))
        else:
            result = _fallback(x, y, power_result)
    except (ArithmeticError, ValueError, np.linalg.LinAlgError, FloatingPointError):
        result = _fallback(x, y, power_result)
    if not math.isfinite(result):
        result = float(y[0])
    result *= amplitude_unit
    if not math.isfinite(result):
        result = float(y[0] * amplitude_unit)
    return float(result)


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/inputs/input.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/workspace/result/output.json"
    with open(input_path, "r", encoding="utf-8") as stream:
        observations = json.load(stream)
    x, y = observations["x"], observations["y"]
    fallback = float(y[min(range(len(x)), key=lambda i: x[i])])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value = estimate(x, y)
        if not math.isfinite(value):
            value = fallback
    except Exception:
        value = fallback
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump({"estimate": float(value)}, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


if __name__ == "__main__":
    main()
