"""Continuum extrapolation from a short, nonuniform refinement sequence."""

import json
import math
import os
import sys

import numpy as np
from scipy.optimize import brentq, least_squares, minimize_scalar

from bayesian import FIT_TOLERANCE
from selection import selected_estimate


def linear_fit(logx, y, powers):
    a = np.column_stack([np.ones(y.size)] +
                        [np.exp(p * logx) for p in powers])
    c = np.linalg.lstsq(a, y, rcond=1e-13)[0]
    return c, y - a @ c


def single_fit(logx, y):
    middle = y.size // 2
    d1, d2 = y[middle] - y[0], y[-1] - y[middle]
    t1, t2 = logx[middle] - logx[0], logx[-1] - logx[middle]
    if d1 != 0 and d2 != 0 and np.signbit(d1) == np.signbit(d2) and t1 > 0 and t2 > 0:
        target = math.log(abs(d2)) - math.log(abs(d1))

        def equation(p):
            a = p * t2
            log_numerator = a if a > 50 else math.log(math.expm1(a))
            return log_numerator - math.log(-math.expm1(-p * t1)) - target

        try:
            if equation(1e-6) <= 0 <= equation(16):
                p = brentq(equation, 1e-6, 16, xtol=1e-13, rtol=1e-14)
                c, r = linear_fit(logx, y, [p])
                if np.linalg.norm(r) < 1e-9:
                    return float(c[0]), float(p), float(np.linalg.norm(r))
        except (ArithmeticError, ValueError, np.linalg.LinAlgError):
            pass

    def objective(p):
        return np.linalg.norm(linear_fit(logx, y, [p])[1])**2
    opt = minimize_scalar(objective, bounds=(0.005, 8), method="bounded",
                          options={"xatol": 1e-10})
    c, r = linear_fit(logx, y, [opt.x])
    return float(c[0]), float(opt.x), float(np.linalg.norm(r))


def fixed_gap(logx, y, gap):
    def objective(p):
        return np.linalg.norm(linear_fit(logx, y, [p, p + gap])[1])**2
    grid = np.linspace(0.08, 5, 80)
    values = np.array([objective(p) for p in grid])
    solutions = []
    for i in range(1, grid.size - 1):
        if values[i] <= min(values[i - 1], values[i + 1]):
            opt = minimize_scalar(objective, bounds=(grid[i - 1], grid[i + 1]),
                                  method="bounded", options={"xatol": 1e-11})
            c, r = linear_fit(logx, y, [opt.x, opt.x + gap])
            solutions.append((float(c[0]), float(opt.x), float(np.linalg.norm(r)), c,
                              float(opt.x + gap)))
    if not solutions:
        i = int(np.argmin(values))
        c, r = linear_fit(logx, y, [grid[i], grid[i] + gap])
        solutions.append((float(c[0]), float(grid[i]), float(np.linalg.norm(r)), c,
                          float(grid[i] + gap)))
    return solutions


def free_two(logx, y):
    def residual(v):
        return 1000 * linear_fit(logx, y, [v[0], v[0] + v[1]])[1]
    solutions = []
    for p in [0.4, 0.9, 1.6, 2.5, 3.8]:
        for gap in [0.35, 0.9, 1.8, 3]:
            opt = least_squares(residual, [p, gap], bounds=([0.08, 0.05], [5, 6]),
                                ftol=1e-13, xtol=1e-13, gtol=1e-13,
                                max_nfev=300)
            c, r = linear_fit(logx, y, [opt.x[0], opt.x.sum()])
            solutions.append((float(c[0]), float(opt.x[0]), float(np.linalg.norm(r)), c,
                              float(opt.x.sum())))
    accurate = [s for s in solutions if s[2] < FIT_TOLERANCE]
    if accurate:
        return min(accurate, key=lambda s: np.linalg.norm(s[3][1:]))
    return min(solutions, key=lambda s: s[2])


def baseline_estimate(x, y, details=False, order_floor=0.0):
    order = np.argsort(x)
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    if np.all(y == y[0]):
        return (float(y[0]), None) if details else float(y[0])
    magnitude = max(1.0, float(np.max(np.abs(y))))
    if magnitude < 1e150:
        magnitude = 1.0
    yn = y / magnitude
    offset = float(yn[0])
    scale = float(np.ptp(yn))
    if not scale > 0:
        return (float(y[0]), None) if details else float(y[0])
    z = (yn - offset) / scale
    logx = np.log(x) - math.log(float(x[-1]))
    one = single_fit(logx, z)
    if one[2] < 1e-10:
        single_result = (offset + scale * one[0]) * magnitude
        if math.isfinite(single_result):
            if details:
                c, _ = linear_fit(logx, z, [one[1]])
                return float(single_result), dict(
                    p=one[1], q=one[1] + 1.5, residual=one[2],
                    coeff=np.array([c[1] * magnitude * scale, 0.0]),
                    scale=magnitude * scale, valid=True, single=True)
            return float(single_result)
    if x.size == 5:
        chosen = free_two(logx, z)
    else:
        gap = 1.5
        sols = fixed_gap(logx, z, gap)
        near = [s for s in sols if s[2] <= min(a[2] for a in sols) + 1e-6]
        chosen = max(near, key=lambda s: s[1])
        if chosen[1] < order_floor:
            c, r = linear_fit(logx, z, [order_floor, order_floor + gap])
            chosen = (float(c[0]), order_floor, float(np.linalg.norm(r)), c,
                      order_floor + gap)
    limit = chosen[0]
    if not math.isfinite(limit) or abs(limit) > 20:
        limit = one[0] if math.isfinite(one[0]) and abs(one[0]) < 20 else 0.0
    result = (offset + scale * limit) * magnitude
    result = float(result) if math.isfinite(result) else float(y[0])
    if details:
        info = dict(p=chosen[1], q=chosen[4], residual=chosen[2],
                    coeff=chosen[3][1:] * magnitude * scale,
                    scale=magnitude * scale, valid=(limit == chosen[0]))
        return result, info
    return result


def estimate(x, y):
    order = np.argsort(x)
    xs = np.asarray(x, dtype=float)[order]
    ys = np.asarray(y, dtype=float)[order]
    if np.all(ys == ys[0]):
        return float(ys[0])
    base, fit = baseline_estimate(xs, ys, details=True, order_floor=0.5)
    if fit is not None and fit.get("single"):
        return base
    if xs.size == 4:
        if (fit is not None and fit["residual"] > 1e-4 and 1e-8 < xs[0]
                and xs[-1] < 2 and np.max(np.abs(ys)) < 1e100):
            try:
                return selected_estimate(xs, ys, base, fit)
            except (ArithmeticError, ValueError, np.linalg.LinAlgError):
                pass
        return base
    if xs.size == 5 and 1e-8 < xs[0] and xs[-1] < 2 and np.max(np.abs(ys)) < 1e100:
        try:
            smooth = selected_estimate(xs, ys, base, fit, probability_three=0.5)
            return smooth
        except (ArithmeticError, ValueError, np.linalg.LinAlgError):
            pass
    return base


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else "/workspace/inputs/input.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "/workspace/result/output.json"
    with open(inp, encoding="utf-8") as f:
        data = json.load(f)
    fallback = float(data["y"][int(np.argmin(data["x"]))])
    try:
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            result = estimate(data["x"], data["y"])
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        result = fallback
    if not math.isfinite(result):
        result = fallback
    if abs(result) > 1e200:
        result = math.copysign(1e200, result)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"estimate": result}, f, allow_nan=False, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
