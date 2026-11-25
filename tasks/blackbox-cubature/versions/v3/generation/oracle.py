"""Evaluation oracle for blackbox-cubature v3: f(z) = g(Q z) with g the canonical function of the instance's family
and Q a Haar orthogonal matrix fixed by the instance's rotation seed (so E f(Z) = E g(Z))."""
from __future__ import annotations

import numpy as np

import options
import paths
import synthetic
from gauss import rotation


def canonical(spec: dict):
    if spec['family'] in synthetic.FAMILIES:
        return lambda y: synthetic.value(spec, y)
    if spec['family'] in paths.FAMILIES:
        return lambda y: paths.payoff(spec, y)
    return lambda y: options.payoff(spec, y)


def make(spec: dict):
    q = rotation(spec['d'], spec['rotation_seed'])
    g = canonical(spec)

    def f(z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        if z.shape[1] != spec['d']:
            raise ValueError(f'expected points of dimension {spec["d"]}, got {z.shape[1]}')
        return g(z @ q.T)

    return f
