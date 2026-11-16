"""Prototype family for blackbox-cubature: a sum of 2-3 kinked ridges with exact integrals (proposal for the author).

f(x) = 1 + sum_j c_j * max(0, v_j . (x - u_j)) ** p_j, p_j in {2, 3}. Each term is a truncated power of a linear form,
so no polynomial fit is exact, and the kinks lie on hyperplanes in different directions. The exact integral of each
term is the truncated-power formula, evaluated in exact rational arithmetic (it cancels heavily in floating point).
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np

_EXACT: dict[str, float] = {}


def truncated_power_integral(w, b, p) -> Fraction:
    """Integral over [0,1]^d of max(0, b + w.x)^p, exactly (all w_i nonzero)."""
    w = [Fraction(x) for x in w]
    b = Fraction(b)
    d = len(w)
    total = Fraction(0)
    for mask in itertools.product((0, 1), repeat=d):
        s = b + sum(wi for wi, m in zip(w, mask) if m)
        if s > 0:
            total += (-1) ** (d - sum(mask)) * s ** (p + d)
    denom = Fraction(1)
    for k in range(1, d + 1):
        denom *= p + k
    for wi in w:
        denom *= wi
    return total / denom


def exact(case) -> float:
    key = case['case_id']
    if key not in _EXACT:
        total = Fraction(1)
        for r in case['ridges']:
            v, u = r['v'], r['u']
            b = -sum(Fraction(vi) * Fraction(ui) for vi, ui in zip(v, u))
            total += Fraction(r['c']) * truncated_power_integral(v, b, r['p'])
        _EXACT[key] = float(total)
    return _EXACT[key]


def value(case, x) -> float:
    x = np.asarray(x, dtype=float)
    out = 1.0
    for r in case['ridges']:
        t = float(np.dot(r['v'], x - np.asarray(r['u'])))
        if t > 0:
            out += r['c'] * t ** r['p']
    return out


def value_and_exact(case, x):
    return value(case, x), exact(case)


def make_case(dimension: int, seed: int, split: str = 'M') -> dict:
    rng = np.random.default_rng(seed)
    ridges = []
    for _ in range(int(rng.integers(2, 4))):
        v = rng.normal(0.0, 1.0, size=dimension)
        v /= np.linalg.norm(v)
        ridges.append({'v': v.tolist(), 'u': rng.uniform(0.25, 0.75, size=dimension).tolist(),
                       'c': float(rng.uniform(0.5, 3.0)), 'p': int(rng.integers(2, 4))})
    return {'split': split, 'suite': 'M1', 'case_id': f'M1-d{dimension}-s{seed}', 'family': 'kinked_ridge_sum',
            'dimension': dimension, 'seed': seed, 'budget': 2048, 'ridges': ridges}
