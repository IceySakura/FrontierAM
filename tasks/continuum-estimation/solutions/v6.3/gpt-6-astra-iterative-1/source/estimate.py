#!/usr/bin/env python3
"""Continuum extrapolation by integration over positive-order power models."""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from quadrature import PowerModel


def weighted_median(values, log_weights):
    good = np.isfinite(values) & np.isfinite(log_weights)
    if not np.any(good):
        raise ValueError("No finite posterior samples")
    values = values[good]
    log_weights = log_weights[good] - np.log(np.maximum(1.0, np.abs(values)))
    weights = np.exp(log_weights - np.max(log_weights))
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    centers = cumulative - 0.5 * weights[order]
    return float(np.interp(0.5 * cumulative[-1], centers, values[order]))


def order_log_prior(values, low, high, total_width):
    width = 0.04
    penalty = 0.5 * (
        (np.maximum(low-values, 0.0) / width)**2
        + (np.maximum(values-high, 0.0) / width)**2
    )
    body = np.log(0.97 / (high-low + width*np.sqrt(2.0*np.pi)))
    broad = np.log(0.03 / total_width)
    return np.logaddexp(body-penalty, broad) - np.logaddexp(body, broad)


def exact_power_limit(x, y):
    """Recognize a single power only when every level verifies the fit."""
    middle = len(x) // 2
    d1, d2 = y[middle] - y[0], y[-1] - y[middle]
    if d1 == 0 or d2 == 0 or np.signbit(d1) != np.signbit(d2):
        return None
    lx = np.log(x)
    left, right = lx[middle] - lx[0], lx[-1] - lx[middle]
    log_ratio = np.log(abs(d2)) - np.log(abs(d1))

    def constraint(p):
        pr = p * right
        numerator = pr + np.log1p(-np.exp(-pr)) if pr > 30 else np.log(np.expm1(pr))
        denominator = np.log(-np.expm1(-p * left))
        return numerator - denominator - log_ratio

    try:
        p = brentq(constraint, 1e-4, 16.0, xtol=1e-13)
        span = np.ptp(y)
        normalized = (y - y[0]) / span
        matrix = np.column_stack((np.ones(len(x)), np.exp(p * (lx - lx[-1]))))
        coeff = np.linalg.lstsq(matrix, normalized, rcond=None)[0]
        if np.max(np.abs(normalized - matrix @ coeff)) < 2e-11:
            return float(y[0] + span * coeff[0])
    except (ValueError, np.linalg.LinAlgError):
        pass
    return None


def clustered_limit(x, y):
    """Use local derivatives when the extrapolation matrix loses rank."""
    width = x[-1] - x[0]
    scale = np.ptp(y)
    u = (x - x[0]) / width
    v = (y - y[0]) / scale
    coeff = np.linalg.lstsq(np.column_stack((np.ones(len(x)), u, u*u)), v, rcond=None)[0]
    slope = coeff[1] * scale / width
    if slope == 0 or not math.isfinite(slope):
        return float(y[0])
    curvature = 2.0 * coeff[2] * scale / width**2
    p = 1.0 + x[0] * curvature / slope
    noise = 16*np.finfo(float).eps*max(1.0, float(np.max(np.abs(y))))
    order_noise = noise * x[0] / (abs(slope) * width**2)
    trust = 1.0 / (1.0 + (order_noise / 0.3)**2)
    p = np.clip(trust*p + (1.0-trust)*1.5, 0.4, 5.0)
    return float(y[0] - x[0]*slope/p)


def fallback(x, y):
    """A local power fit, bounded when the positive-order assumption fails."""
    if len(x) < 3:
        return float(y[0])
    dy = np.diff(y)
    turns = np.count_nonzero(dy[1:] * dy[:-1] < 0)
    if turns >= 2:
        return float(y[0] / 2.0 + y[1] / 2.0)
    xx, yy = x[:3] / x[2], y[:3]
    span = np.ptp(yy)
    if span <= 1e-14 * max(1.0, float(np.max(np.abs(yy)))):
        return float(y[0])
    yy = (yy - y[0]) / span

    def fit(p, full=False):
        a = np.column_stack((np.ones(3), xx**p))
        coeff = np.linalg.lstsq(a, yy, rcond=None)[0]
        residual = yy - a @ coeff
        return coeff if full else float(residual @ residual)

    result = minimize_scalar(fit, bounds=(0.35, 6.0), method="bounded")
    candidate = y[0] + span * fit(result.x, True)[0]
    max_correction = 3.0 * np.ptp(y) / min(1.0, np.log(x[-1] / x[0]))
    return float(np.clip(candidate, y[0] - max_correction, y[0] + max_correction))


def estimate(x, y):
    order = np.argsort(x)
    x, y = x[order], y[order]
    nearest = float(y[0])
    if np.ptp(y) <= 5e-15 * max(1.0, float(np.max(np.abs(y)))):
        return nearest
    if x[0] < 1e-40 or x[-1] > 1e40 or np.max(np.abs(y)) > 1e100:
        return nearest
    simple = exact_power_limit(x, y)
    if simple is not None:
        return simple
    if x[0] / x[-1] > 0.9999:
        return clustered_limit(x, y)
    if np.min(np.diff(np.log(x))) < 1e-7:
        keep = np.r_[True, np.diff(np.log(x)) >= 1e-7]
        x, y = x[keep], y[keep]
        if len(x) < 4:
            return fallback(x, y)
    try:
        model = PowerModel(x, y)
        if len(x) == 4:
            pp, dd, abc, z, lw = model.four_samples(900, 200)
        else:
            pp, dd, abc, z, lw = model.five_samples(1500)
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        return fallback(x, y)
    magnitude = np.abs(abc)
    energy = 0.5 * np.sum((
        np.maximum(magnitude - np.array((1.5, 1.0, 0.4)), 0.0)
        / np.array((0.04, 0.03, 0.0125))
    )**2, axis=1)
    energy += 0.5 * (np.maximum(0.2 - magnitude[:, 0], 0.0) / 0.04)**2
    log_order = (
        order_log_prior(pp, 0.6, 2.8, 3.35)
        + order_log_prior(dd, 0.4, 2.2, 2.9)
    )
    if np.nanmin(energy - log_order) > 50.0:
        return fallback(x, y)
    both_negative = np.all(abc[:, 1:] < 0, axis=1)
    sign_penalty = np.where(both_negative, 0.0, np.log(37.0))
    result = weighted_median(z, lw - energy + log_order - sign_penalty)
    if not math.isfinite(result):
        return fallback(x, y)
    return result


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/inputs/input.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/workspace/result/output.json"
    with open(input_path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    result = float(y[np.argmin(x)])
    try:
        with np.errstate(all="ignore"):
            candidate = estimate(x, y)
        if math.isfinite(candidate):
            result = float(candidate)
    except (ArithmeticError, ValueError, RuntimeError, np.linalg.LinAlgError):
        pass
    result = max(-1e150, min(1e150, result))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump({"estimate": result}, stream, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
