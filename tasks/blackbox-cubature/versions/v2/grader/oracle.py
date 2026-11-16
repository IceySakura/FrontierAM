"""Oracle for blackbox-cubature v2: function values and exact integrals of the hidden families.

fourier_ridge and product_peak are the author's formulas from task/private_hidden/grade_hidden.py (v4 package).
kinked_ridge_sum replaces polynomial_ridge (results/OPEN_ITEMS.md): f(x) = 1 + sum_j c_j max(0, v_j.(x - u_j))^{p_j};
each term integrates by the truncated-power formula

    int_[0,1]^d max(0, b + w.x)^p dx = sum_S (-1)^(d-|S|) max(0, b + sum_{i in S} w_i)^(p+d) / (prod_{k=1..d}(p+k) prod_i w_i),

evaluated in exact rational arithmetic because it cancels heavily in floating point.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np

_EXACT: dict[tuple, float] = {}


def i0_approx(x: float) -> float:
    x2 = (float(x) / 2.0) ** 2
    term = total = 1.0
    for k in range(1, 40):
        term *= x2 / (k * k)
        total += term
    return total


def truncated_power_integral(w, b, p: int) -> Fraction:
    """Integral over [0,1]^d of max(0, b + w.x)^p (every w_i nonzero)."""
    w = [Fraction(x) for x in w]
    b = Fraction(b)
    total = Fraction(0)
    for mask in itertools.product((0, 1), repeat=len(w)):
        s = b + sum(wi for wi, m in zip(w, mask) if m)
        if s > 0:
            total += (-1) ** (len(w) - sum(mask)) * s ** (p + len(w))
    denominator = Fraction(math.prod(range(p + 1, p + len(w) + 1)))
    for wi in w:
        denominator *= wi
    return total / denominator


def kinked_ridge_exact(case: dict) -> float:
    key = (case['case_id'], case['seed'])
    if key not in _EXACT:
        total = Fraction(1)
        for ridge in case['ridges']:
            b = -sum(Fraction(vi) * Fraction(ui) for vi, ui in zip(ridge['v'], ridge['u']))
            total += Fraction(ridge['c']) * truncated_power_integral(ridge['v'], b, ridge['p'])
        _EXACT[key] = float(total)
    return _EXACT[key]


def value_and_exact(case: dict, x) -> tuple[float, float]:
    family = case['family']
    x = np.asarray(x, dtype=float)
    if family == 'kinked_ridge_sum':
        value = 1.0
        for ridge in case['ridges']:
            t = float(np.dot(ridge['v'], x - np.asarray(ridge['u'], dtype=float)))
            if t > 0:
                value += ridge['c'] * t ** ridge['p']
        return value, kinked_ridge_exact(case)
    if family == 'fourier_ridge':
        k = np.asarray(case['k'], int)
        theta = 2.0 * math.pi * float(np.dot(k, x)) + float(case['phase'])
        return math.exp(float(case['concentration']) * math.cos(theta)), i0_approx(float(case['concentration']))
    if family == 'product_peak':
        a, u = np.asarray(case['a'], float), np.asarray(case['u'], float)
        value = float(np.prod(1.0 / (a ** -2 + (x - u) ** 2)))
        exact = float(np.prod(a * (np.arctan(a * u) + np.arctan(a * (1.0 - u)))))
        return value, exact
    raise ValueError(family)
