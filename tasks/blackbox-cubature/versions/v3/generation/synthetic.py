"""Synthetic hard families for blackbox-cubature v3: the extreme cases of the preintegration theory.

Canonical coordinates y = Q z (Q a Haar rotation fixed by the instance). Every instance is f(y) = B(y) + N(y).

B is a smooth, genuinely d-dimensional background with mean beta0 (no low-dimensional structure to exploit):
    B(y) = beta0 + beta1 [prod_i (1 + c_i cos(om_i (V y)_i + ph_i)) - prod_i (1 + c_i exp(-om_i^2/2) cos ph_i)],
with V an independent Haar rotation and c_i = c0 / i.

N lives on the first k coordinates through isotropic quadratic forms on disjoint index sets S,
    Q_S(y) = lam_S sum_{j in S} (y_j - m_j)^2   (lam_S times a non-central chi-square variable, |S| degrees of freedom),
so that no direction satisfies the monotonicity that preintegration theory needs (Gilbert, Kuo, Sloan 2022):
    H1   alpha (Q_+ - Q_- - c)^+                 hyperboloid kink, monotone only along null directions of Q_+ - Q_-
    H2   alpha max(c_1 - Q_1, c_2 - Q_2, 0)       two ellipsoidal tents with the interior kink c_1 - Q_1 = c_2 - Q_2 > 0
    H3   alpha exp(w.y - |w|^2/2) 1{Q < c}        closed curved jump (ball), probability 0.15 to 0.6
    H3t  (a / p) 1{Q < c}                         the same ball as a rare event, probability p in [1e-3, 1e-2]
k runs over {k_min, d/2, d}: the non-smooth set is low-, half- or full-dimensional.

References: H3/H3t in closed form, H1/H2 as one-dimensional integrals over the density of one form. Route A uses
SciPy's non-central chi-square functions; route B uses an independent Poisson mixture of central chi-square
distributions and integrates over the other form's density.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, special, stats

from gauss import random_unit, rotation

FAMILIES = ('H1', 'H2', 'H3', 'H3t')


def k_levels(family: str, d: int) -> list[int]:
    kmin = 2 if family in ('H3', 'H3t') else 4
    return sorted({k for k in (kmin, d // 2, d) if k >= kmin})


# ---------------------------------------------------------------- sampling

def _form(rng, start: int, stop: int, radius: float, scale: float) -> dict:
    k = stop - start
    return {'idx': [start, stop], 'lam': scale / k, 'm': (random_unit(rng, k) * radius).tolist()}


def _nc(form: dict, shift=None) -> float:
    m = np.asarray(form['m'])
    if shift is not None:
        m = m - shift[form['idx'][0]:form['idx'][1]]
    return float(m @ m)


def _rv(form: dict, rng, n: int) -> np.ndarray:
    k = form['idx'][1] - form['idx'][0]
    return form['lam'] * stats.ncx2.rvs(k, _nc(form), size=n, random_state=rng)


def sample(family: str, d: int, k: int, rng: np.random.Generator) -> dict:
    spec = {'family': family, 'd': int(d), 'k': int(k),
            'beta0': float(rng.uniform(1.0, 2.0)), 'beta1': float(rng.uniform(0.2, 0.5)),
            'c0': float(rng.uniform(0.3, 0.6)), 'om': rng.uniform(0.5, 1.5, d).tolist(),
            'ph': rng.uniform(0.0, 2 * math.pi, d).tolist(), 'bg_seed': int(rng.integers(2 ** 31))}
    draws = np.random.default_rng(int(rng.integers(2 ** 31)))
    if family == 'H1':
        h = k // 2
        fp, fm = _form(rng, 0, h, rng.uniform(0, 1), rng.uniform(0.7, 1.4)), \
            _form(rng, h, k, rng.uniform(0, 1), rng.uniform(0.7, 1.4))
        diff = _rv(fp, draws, 40000) - _rv(fm, draws, 40000)
        spec.update(forms=[fp, fm], alpha=float(rng.uniform(0.5, 1.5)),
                    c=float(np.quantile(diff, 1.0 - rng.uniform(0.2, 0.5))))
    elif family == 'H2':
        h = k // 2
        f1, f2 = _form(rng, 0, h, rng.uniform(0, 1), rng.uniform(0.7, 1.4)), \
            _form(rng, h, k, rng.uniform(0, 1), rng.uniform(0.7, 1.4))
        c1 = f1['lam'] * special.chndtrix(rng.uniform(0.3, 0.7), h, _nc(f1))
        c2 = f2['lam'] * special.chndtrix(rng.uniform(0.3, 0.7), k - h, _nc(f2))
        spec.update(forms=[f1, f2], alpha=float(rng.uniform(0.5, 1.5)), c1=float(c1), c2=float(c2))
    elif family in ('H3', 'H3t'):
        tail = family == 'H3t'
        f = _form(rng, 0, k, rng.uniform(1.5, 3.0) if tail else rng.uniform(0.0, 1.5), 1.0)
        p = 10 ** rng.uniform(-3, -2) if tail else rng.uniform(0.15, 0.6)
        spec.update(forms=[f], c=float(f['lam'] * special.chndtrix(p, k, _nc(f))), p=float(p))
        if tail:
            spec.update(alpha=float(rng.uniform(0.5, 1.5) / p), w=[0.0] * d)
        else:
            spec.update(alpha=float(rng.uniform(0.5, 1.5)), w=(random_unit(rng, d) * rng.uniform(0.1, 0.5)).tolist())
    else:
        raise ValueError(family)
    return spec


# ---------------------------------------------------------------- evaluation

def background(spec: dict, y: np.ndarray) -> np.ndarray:
    d = spec['d']
    c = spec['c0'] / np.arange(1, d + 1)
    om, ph = np.asarray(spec['om']), np.asarray(spec['ph'])
    t = y @ rotation(d, spec['bg_seed']).T
    mean = float(np.prod(1.0 + c * np.exp(-0.5 * om ** 2) * np.cos(ph)))
    return spec['beta0'] + spec['beta1'] * (np.prod(1.0 + c * np.cos(om * t + ph), axis=1) - mean)


def form_value(form: dict, y: np.ndarray) -> np.ndarray:
    a, b = form['idx']
    return form['lam'] * np.sum(np.square(y[:, a:b] - np.asarray(form['m'])), axis=1)


def nonsmooth(spec: dict, y: np.ndarray) -> np.ndarray:
    fam, forms = spec['family'], spec['forms']
    if fam == 'H1':
        return spec['alpha'] * np.maximum(form_value(forms[0], y) - form_value(forms[1], y) - spec['c'], 0.0)
    if fam == 'H2':
        return spec['alpha'] * np.maximum(np.maximum(spec['c1'] - form_value(forms[0], y),
                                                     spec['c2'] - form_value(forms[1], y)), 0.0)
    w = np.asarray(spec['w'])
    return spec['alpha'] * np.exp(y @ w - 0.5 * w @ w) * (form_value(forms[0], y) < spec['c'])


def value(spec: dict, y: np.ndarray) -> np.ndarray:
    return background(spec, y) + nonsmooth(spec, y)


# ---------------------------------------------------------------- distribution functions (two independent routes)

class _Scipy:
    name = 'SciPy non-central chi-square'

    @staticmethod
    def cdf(x, k, nc):
        return float(special.chndtr(x, k, nc)) if x > 0 else 0.0

    @staticmethod
    def sf(x, k, nc):
        return float(stats.ncx2.sf(x, k, nc)) if x > 0 else 1.0

    @staticmethod
    def pdf(x, k, nc):
        return float(stats.ncx2.pdf(x, k, nc)) if x > 0 else 0.0


class _Mixture:
    """Poisson(nc/2) mixture of central chi-square distributions with k + 2j degrees of freedom."""
    name = 'Poisson mixture of central chi-square'

    @staticmethod
    def _weights(nc):
        lam = 0.5 * nc
        if lam == 0.0:
            return np.zeros(1), np.ones(1)
        j = np.arange(int(lam + 14 * math.sqrt(lam) + 80))
        return j, np.exp(-lam + j * math.log(lam) - special.gammaln(j + 1))

    @classmethod
    def cdf(cls, x, k, nc):
        if x <= 0:
            return 0.0
        j, w = cls._weights(nc)
        return float(w @ special.gammainc(0.5 * k + j, 0.5 * x))

    @classmethod
    def sf(cls, x, k, nc):
        if x <= 0:
            return 1.0
        j, w = cls._weights(nc)
        return float(w @ special.gammaincc(0.5 * k + j, 0.5 * x))

    @classmethod
    def pdf(cls, x, k, nc):
        if x <= 0:
            return 0.0
        j, w = cls._weights(nc)
        nu = 0.5 * k + j
        return float(w @ np.exp((nu - 1) * math.log(0.5 * x) - 0.5 * x - special.gammaln(nu)) * 0.5)


def _tail_moment(lib, a, k, nc):
    """E[X 1{X > a}] for X ~ chi2_k(nc): k P(chi2_{k+2}(nc) > a) + nc P(chi2_{k+4}(nc) > a)."""
    return k * lib.sf(a, k + 2, nc) + nc * lib.sf(a, k + 4, nc)


def _excess(lib, form, s, shift=None):
    """E[(Q - s)^+]."""
    k, lam, nc = form['idx'][1] - form['idx'][0], form['lam'], _nc(form, shift)
    if s <= 0:
        return lam * (k + nc) - s
    a = s / lam
    return lam * (_tail_moment(lib, a, k, nc) - a * lib.sf(a, k, nc))


def _shortfall(lib, form, b):
    """E[(b - Q)^+]."""
    if b <= 0:
        return 0.0
    k, lam, nc = form['idx'][1] - form['idx'][0], form['lam'], _nc(form)
    a = b / lam
    return b * lib.cdf(a, k, nc) - lam * ((k + nc) - _tail_moment(lib, a, k, nc))


def _density_integral(lib, form, g, breaks):
    """Integral over t >= 0 of g(t) times the density of Q (split at breaks and at the bulk of the density)."""
    k, lam, nc = form['idx'][1] - form['idx'][0], form['lam'], _nc(form)
    mean, sd = k + nc, math.sqrt(2 * (k + 2 * nc))
    hi = lam * (mean + 45 * sd + 60)
    pts = sorted({0.0, hi} | {b for b in breaks if 0 < b < hi} | {lam * max(mean - 2 * sd, 0.0), lam * mean,
                                                                    lam * (mean + 4 * sd)})
    total = 0.0
    for lo, up in zip(pts[:-1], pts[1:]):
        if up > lo:
            total += integrate.quad(lambda t: g(t) * lib.pdf(t / lam, k, nc) / lam, lo, up,
                                    epsabs=1e-15, epsrel=1e-13, limit=400)[0]
    return total


def _nonsmooth_mean(spec, lib, swap: bool) -> float:
    fam, forms = spec['family'], spec['forms']
    if fam == 'H1':
        fp, fm, c = forms[0], forms[1], spec['c']
        if not swap:
            return spec['alpha'] * _density_integral(lib, fm, lambda t: _excess(lib, fp, c + t), [-c])
        return spec['alpha'] * _density_integral(lib, fp, lambda s: _shortfall(lib, fm, s - c), [c])
    if fam == 'H2':
        (fa, ca), (fb, cb) = ((forms[0], spec['c1']), (forms[1], spec['c2']))
        if swap:
            (fa, ca), (fb, cb) = (fb, cb), (fa, ca)

        def g(t):
            m = max(cb - t, 0.0)
            return m + _shortfall(lib, fa, ca - m)

        return spec['alpha'] * _density_integral(lib, fb, g, [cb, cb - ca])
    form = forms[0]
    k = form['idx'][1] - form['idx'][0]
    return spec['alpha'] * lib.cdf(spec['c'] / form['lam'], k, _nc(form, np.asarray(spec['w'])))


def check_distributions(spec: dict, n: int = 25, seed: int = 0) -> float:
    """Max relative difference between the two routes' cdf, survival function and density of every form, at points
    spread over its bulk and tails."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for form in spec['forms']:
        k, nc = form['idx'][1] - form['idx'][0], _nc(form)
        mean, sd = k + nc, math.sqrt(2 * (k + 2 * nc))
        for x in np.maximum(mean + sd * rng.uniform(-3.0, 8.0, n), 1e-3):
            for fn in ('cdf', 'sf', 'pdf'):
                a, b = getattr(_Scipy, fn)(x, k, nc), getattr(_Mixture, fn)(x, k, nc)
                worst = max(worst, abs(a - b) / max(abs(b), 1e-300) if abs(b) > 1e-280 else abs(a - b))
    return worst


def reference(spec: dict) -> dict:
    """Route A: SciPy's non-central chi-square functions."""
    val = spec['beta0'] + _nonsmooth_mean(spec, _Scipy, swap=False)
    how = 'closed form' if spec['family'] in ('H3', 'H3t') else '1D integral over the density of the second form'
    return {'value': val, 'abs_err': 1e-13 * max(1.0, abs(val)), 'method': f'A: {_Scipy.name}, {how}'}


def reference_swapped(spec: dict) -> dict:
    """Route B: Poisson mixture of central chi-square distributions, integrating over the other form's density."""
    val = spec['beta0'] + _nonsmooth_mean(spec, _Mixture, swap=True)
    how = 'closed form' if spec['family'] in ('H3', 'H3t') else '1D integral over the density of the first form'
    return {'value': val, 'abs_err': 1e-13 * max(1.0, abs(val)), 'method': f'B: {_Mixture.name}, {how}'}
