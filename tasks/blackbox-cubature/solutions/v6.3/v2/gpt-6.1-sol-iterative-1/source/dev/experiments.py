#!/usr/bin/env python3
"""Independent analytic stress functions for development, not case lookup."""
import argparse
import itertools
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import erf, iv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solver import integrate


class LocalOracle:
    def __init__(self, d, func, budget=2048):
        self.dimension = d
        self.budget = budget
        self.total_budget = budget
        self.used = 0
        self.func = func
        self.history_x = []
        self.history_y = []
        self.smooth = False
        self.line_data = []
        self.fallback_y = []

    def evaluate(self, x):
        self.used += len(x)
        if self.used > self.budget or np.any(x < 0) or np.any(x > 1):
            raise ValueError("invalid local query")
        y = np.asarray(self.func(x), dtype=float)
        if y.shape != (len(x),) or not np.all(np.isfinite(y)):
            raise ValueError("invalid local value")
        self.history_x.append(np.asarray(x).copy())
        self.history_y.append(y.copy())
        return y


def wave_case(rng, d, k=1, high=1):
    frequencies = rng.uniform(-12 * high, 12 * high, size=(k, d))
    phases = rng.uniform(-math.pi, math.pi, size=k)
    amplitudes = rng.uniform(0.2, 1.0, size=k)
    base = 0.7 if k > 1 else 0.0
    value = lambda x: base + np.cos(x @ frequencies.T + phases) @ amplitudes
    complex_integrals = (np.exp(1j * (phases + frequencies.sum(axis=1) / 2))
                         * np.prod(np.sinc(frequencies / (2 * math.pi)), axis=1))
    exact = base + float(np.dot(amplitudes, complex_integrals.real))
    return value, exact


def gaussian_case(rng, d, k=1, sharp=False):
    centers = rng.uniform(0.05, 0.95, size=(k, d))
    rates = rng.uniform(3, 18 if sharp else 8, size=(k, d))
    amplitudes = rng.uniform(0.3, 1.0, size=k)
    value = lambda x: np.exp(-np.sum(
        ((x[:, None, :] - centers) * rates) ** 2, axis=2)) @ amplitudes
    factors = math.sqrt(math.pi) / (2 * rates) * (
        erf(rates * (1 - centers)) + erf(rates * centers))
    exact = float(np.dot(amplitudes, np.prod(factors, axis=1)))
    return value, exact


def product_case(rng, d, exponent=False):
    centers = rng.uniform(0.02, 0.98, size=d)
    slopes = rng.uniform(2, 12, size=d)
    if exponent:
        value = lambda x: np.exp(-np.abs(x - centers) @ slopes)
        factors = (2 - np.exp(-slopes * centers)
                   - np.exp(-slopes * (1 - centers))) / slopes
    else:
        value = lambda x: np.prod(1 + slopes * np.abs(x - centers), axis=1)
        factors = 1 + slopes * (centers ** 2 + (1 - centers) ** 2) / 2
    return value, float(np.prod(factors))


def positive_part_integral(w, t):
    # Long double protects the inclusion-exclusion formula in moderate dimension.
    w = np.asarray(w, dtype=np.longdouble)
    t = np.longdouble(t) - np.sum(w[w < 0])
    w = np.abs(w)
    w = w[w > 1e-12]
    d = len(w)
    total = np.longdouble(0)
    for mask in itertools.product((0, 1), repeat=d):
        s = np.dot(w, mask) - t
        if s > 0:
            total += (-1 if sum(mask) % 2 else 1) * s ** (d + 1)
    return float((-1) ** d * total / (math.factorial(d + 1) * np.prod(w)))


def hinge_case(rng, d, k=3, parallel=False):
    if parallel:
        directions = rng.uniform(0.8, 1.0, size=(1, d)) + rng.normal(0, 0.035, (k, d))
    else:
        directions = rng.uniform(-1, 1, size=(k, d))
    thresholds = directions.sum(axis=1) / 2 + rng.uniform(-0.3, 0.3, size=k)
    amplitudes = rng.uniform(0.2, 1.2, size=k)
    value = lambda x: 1 + np.abs(x @ directions.T - thresholds) @ amplitudes
    pieces = [2 * positive_part_integral(w, t) - (w.sum() / 2 - t)
              for w, t in zip(directions, thresholds)]
    return value, 1 + float(np.dot(amplitudes, pieces))


def periodic_profile_case(rng, d, high=False, rational=False):
    frequencies = rng.uniform(-12, 12, size=d) * (10 if high else 1)
    phase = rng.uniform(-math.pi, math.pi)
    if rational:
        value = lambda x: 1 / (1.4 + np.cos((x - 0.5) @ frequencies + phase))
        root = math.sqrt(1.4 ** 2 - 1)
        ratio = -1 / (1.4 + root)
        coefficients = np.r_[1 / root, 2 * ratio ** np.arange(1, 61) / root]
    else:
        value = lambda x: np.exp(0.8 * np.cos((x - 0.5) @ frequencies + phase))
        coefficients = iv(np.arange(61), 0.8)
        coefficients[1:] *= 2
    harmonics = np.arange(len(coefficients))
    factors = np.prod(np.sinc(harmonics[:, None] * frequencies / (2 * math.pi)),
                      axis=1)
    exact = float(np.dot(coefficients, np.cos(harmonics * phase) * factors))
    return value, exact


def corner_relu_case(rng, d):
    w = rng.uniform(0.8, 1.2, d)
    threshold = np.sum(w) * 0.99
    value = lambda x: 1 + np.maximum(x @ w - threshold, 0)
    width = np.sum(w) - threshold
    exact = 1 + width ** (d + 1) / (math.factorial(d + 1) * np.prod(w))
    return value, float(exact)


def exponential_ridge_case(rng, d, k=3):
    from projected import exponential_absolute_integral
    directions = rng.uniform(-3, 3, size=(k, d))
    biases = rng.uniform(-0.5, 0.5, size=k)
    amplitudes = rng.uniform(0.2, 1.2, size=k)
    value = lambda x: 0.4 + np.exp(-np.abs(
        (x - 0.5) @ directions.T - biases)) @ amplitudes
    pieces = [exponential_absolute_integral(w, b)
              for w, b in zip(directions, biases)]
    return value, 0.4 + float(np.dot(amplitudes, pieces))


def offset_power_case(rng, d, k=3):
    from offset_power import offset_power_integral
    directions = rng.uniform(-1, 1, size=(k, d))
    biases = rng.uniform(-0.4, 0.4, size=k)
    offsets = rng.uniform(0.01, 0.12, size=k)
    exponents = rng.uniform(-1.2, -0.25, size=k)
    amplitudes = rng.uniform(0.2, 1.2, size=k)
    weights = 2 * directions / np.sum(np.abs(directions), axis=1)[:, None]
    value = lambda x: 0.4 + np.exp(
        np.log(offsets + np.abs((x - 0.5) @ weights.T - biases))
        * exponents) @ amplitudes
    pieces = [offset_power_integral(v, b, a, p) for v, b, a, p in zip(
        directions, biases, offsets, exponents)]
    return value, 0.4 + float(np.dot(amplitudes, pieces))


def make_case(name, rng, d):
    suffix = re.search(r"(\d+)$", name)
    count = int(suffix.group(1)) if suffix else 1
    if name.startswith("wave"):
        high = 80 if "veryhigh" in name else (10 if "high" in name else 1)
        return wave_case(rng, d, count, high)
    if name.startswith("gauss"):
        return gaussian_case(rng, d, count, "sharp" in name)
    if name == "absolute_product":
        return product_case(rng, d)
    if name == "absolute_exp":
        return product_case(rng, d, True)
    if name.startswith("hinge"):
        return hinge_case(rng, d, count, "parallel" in name)
    if name.startswith("expridge"):
        return exponential_ridge_case(rng, d, count)
    if name.startswith("powridge"):
        return offset_power_case(rng, d, count)
    if name.startswith("periodic"):
        return periodic_profile_case(rng, d, "high" in name, "rational" in name)
    if name == "corner_relu":
        return corner_relu_case(rng, d)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", nargs="+", type=int, default=[2, 3, 5, 8])
    parser.add_argument("--families", nargs="+", default=[
        "wave1", "wave3", "gauss1", "gauss3", "gausssharp3",
        "absolute_product", "absolute_exp", "hinge1", "hinge3"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2170)
    args = parser.parse_args()
    for name in args.families:
        for d in args.dimensions:
            errors = []
            durations = []
            for r in range(args.repeats):
                rng = np.random.default_rng(args.seed + 991 * d + 3109 * r)
                function, exact = make_case(name, rng, d)
                oracle = LocalOracle(d, function)
                started = time.monotonic()
                estimate = integrate(oracle, args.seed + r, started + 22)
                duration = time.monotonic() - started
                error = abs(estimate - exact) / max(abs(exact), 1e-300)
                errors.append(error)
                durations.append(duration)
                print(f"{name:17s} d={d} r={r} error={error:.4g} "
                      f"seconds={duration:.3f} queries={oracle.used}", flush=True)
            if args.repeats > 1:
                print(f"  geometric error={10 ** np.mean(np.log10(np.maximum(errors, 1e-15))):.4g} "
                      f"median seconds={np.median(durations):.3f}", flush=True)


if __name__ == "__main__":
    main()
