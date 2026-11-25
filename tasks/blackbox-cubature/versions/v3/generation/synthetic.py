"""Synthetic hard families for blackbox-cubature v3: the extreme cases of the preintegration theory.

Canonical coordinates y = Q z (Q a Haar rotation fixed by the instance). Every instance is f(y) = B(y) + N(y).

B is a smooth, genuinely d-dimensional background with mean beta0 (no low-dimensional structure to exploit):
    B(y) = beta0 + beta1 [prod_i (1 + c_i cos(om_i (V y)_i + ph_i)) - prod_i (1 + c_i exp(-om_i^2/2) cos ph_i)],
with V an independent Haar rotation and c_i = c0 / i.

N lives on the first k coordinates through quadratic forms on disjoint index sets S,
    Q_S(y) = sum_{j in S} lam_j (y_j - m_j)^2,
isotropic (one lam per form: a scaled non-central chi-square) in the prototype and v3, anisotropic in v3.1 (`aniso`:
log-uniform weights with ratio up to aniso^2, so recovering a form from chords needs O(|S|^2) of them),
so that no direction satisfies the monotonicity that preintegration theory needs (Gilbert, Kuo, Sloan 2022):
    H1   alpha (Q_+ - Q_- - c)^+                 hyperboloid kink, monotone only along null directions of Q_+ - Q_-
    H2   alpha max(c_1 - Q_1, c_2 - Q_2, 0)       two ellipsoidal tents with the interior kink c_1 - Q_1 = c_2 - Q_2 > 0
    H3   alpha exp(w.y - |w|^2/2) 1{Q < c}        closed curved jump (ball), probability 0.15 to 0.6
    H3t  (a / p) 1{Q < c}                         the same ball as a rare event, probability p in [1e-3, 1e-2]
k runs over {k_min, d/2, d}: the non-smooth set is low-, half- or full-dimensional.

References: H3/H3t in closed form, H1/H2 as one-dimensional integrals over the density of one form. Every form's law
is a generalized chi-square in Ruben's mixture form (gchisq.py): route A with beta = min lam, route B with
beta = 0.9 min lam and the other form's density; Imhof's inversion is the pointwise check (check_distributions). The
v3 sets were built with SciPy's non-central chi-square (route A) and a Poisson mixture (route B), which agree with
these routes for isotropic forms.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, optimize, special, stats

import gchisq
from gauss import random_unit, rotation

FAMILIES = ('H1', 'H2', 'H3', 'H3t')


def k_levels(family: str, d: int) -> list[int]:
    kmin = 2 if family in ('H3', 'H3t') else 4
    return sorted({k for k in (kmin, d // 2, d) if k >= kmin})


# ---------------------------------------------------------------- sampling

def _form(rng, start: int, stop: int, radius: float, scale: float, aniso: float | None = None) -> dict:
    k = stop - start
    form = {'idx': [start, stop], 'lam': scale / k, 'm': (random_unit(rng, k) * radius).tolist()}
    if aniso:
        rho = np.exp(rng.uniform(-math.log(aniso), math.log(aniso), k))
        form['lam'] = (scale / k * rho / rho.mean()).tolist()
    return form


def _lam(form: dict) -> np.ndarray:
    k = form['idx'][1] - form['idx'][0]
    return np.broadcast_to(np.asarray(form['lam'], float), (k,)).copy()


def _delta2(form: dict, shift=None) -> np.ndarray:
    m = np.asarray(form['m'])
    if shift is not None:
        m = m - shift[form['idx'][0]:form['idx'][1]]
    return m * m


def _nc(form: dict, shift=None) -> float:
    return float(_delta2(form, shift).sum())


def _rv(form: dict, rng, n: int) -> np.ndarray:
    k = form['idx'][1] - form['idx'][0]
    if np.ndim(form['lam']) == 0:
        return form['lam'] * stats.ncx2.rvs(k, _nc(form), size=n, random_state=rng)
    z = rng.standard_normal((n, k)) - np.asarray(form['m'])
    return (z * z) @ _lam(form)


def _quantile(form: dict, p: float, shift=None) -> float:
    """p-quantile of Q_S (isotropic: SciPy's inverse; anisotropic: Ruben's cdf inverted by bracketing)."""
    k = form['idx'][1] - form['idx'][0]
    if np.ndim(form['lam']) == 0:
        return float(form['lam'] * special.chndtrix(p, k, _nc(form, shift)))
    law = gchisq.Ruben(_lam(form), _delta2(form, shift))
    hi = law.mean + 10 * math.sqrt(law.var)
    return float(optimize.brentq(lambda x: law.cdf(x) - p, 1e-300, hi, xtol=1e-14 * hi, rtol=1e-15))


def sample(family: str, d: int, k: int, rng: np.random.Generator, aniso: float | None = None) -> dict:
    spec = {'family': family, 'd': int(d), 'k': int(k), **({'aniso': float(aniso)} if aniso else {}),
            'beta0': float(rng.uniform(1.0, 2.0)), 'beta1': float(rng.uniform(0.2, 0.5)),
            'c0': float(rng.uniform(0.3, 0.6)), 'om': rng.uniform(0.5, 1.5, d).tolist(),
            'ph': rng.uniform(0.0, 2 * math.pi, d).tolist(), 'bg_seed': int(rng.integers(2 ** 31))}
    draws = np.random.default_rng(int(rng.integers(2 ** 31)))
    if family == 'H1':
        h = k // 2
        fp, fm = _form(rng, 0, h, rng.uniform(0, 1), rng.uniform(0.7, 1.4), aniso), \
            _form(rng, h, k, rng.uniform(0, 1), rng.uniform(0.7, 1.4), aniso)
        diff = _rv(fp, draws, 40000) - _rv(fm, draws, 40000)
        spec.update(forms=[fp, fm], alpha=float(rng.uniform(0.5, 1.5)),
                    c=float(np.quantile(diff, 1.0 - rng.uniform(0.2, 0.5))))
    elif family == 'H2':
        h = k // 2
        f1, f2 = _form(rng, 0, h, rng.uniform(0, 1), rng.uniform(0.7, 1.4), aniso), \
            _form(rng, h, k, rng.uniform(0, 1), rng.uniform(0.7, 1.4), aniso)
        c1, c2 = _quantile(f1, rng.uniform(0.3, 0.7)), _quantile(f2, rng.uniform(0.3, 0.7))
        spec.update(forms=[f1, f2], alpha=float(rng.uniform(0.5, 1.5)), c1=float(c1), c2=float(c2))
    elif family in ('H3', 'H3t'):
        tail = family == 'H3t'
        f = _form(rng, 0, k, rng.uniform(1.5, 3.0) if tail else rng.uniform(0.0, 1.5), 1.0, aniso)
        p = 10 ** rng.uniform(-3, -2) if tail else rng.uniform(0.15, 0.6)
        spec.update(forms=[f], c=_quantile(f, p), p=float(p))
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
    return np.square(y[:, a:b] - np.asarray(form['m'])) @ _lam(form)


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

class _Ruben:
    """Every form's law as Ruben's mixture (gchisq.Ruben); mode 'min' (route A) or 'scaled' (route B)."""

    def __init__(self, mode: str):
        self.mode = mode
        self.name = "Ruben's mixture of central chi-square, beta = " + ('min lam' if mode == 'min' else '0.9 min lam')
        self._laws = {}

    def law(self, form: dict, shift=None) -> gchisq.Ruben:
        key = (id(form), None if shift is None else tuple(np.asarray(shift)[form['idx'][0]:form['idx'][1]]))
        if key not in self._laws:
            self._laws[key] = gchisq.Ruben(_lam(form), _delta2(form, shift), self.mode)
        return self._laws[key]


def _density_integral(law: gchisq.Ruben, g, breaks, lam_max: float) -> float:
    """Integral over t >= 0 of g(t) times the density of Q (split at breaks and at the bulk of the density)."""
    mean, sd = law.mean, math.sqrt(law.var)
    hi = mean + 45 * sd + 60 * lam_max
    pts = sorted({0.0, hi} | {b for b in breaks if 0 < b < hi} | {max(mean - 2 * sd, 0.0), mean, mean + 4 * sd})
    total = 0.0
    for lo, up in zip(pts[:-1], pts[1:]):
        if up > lo:
            total += integrate.quad(lambda t: g(t) * law.pdf(t), lo, up, epsabs=1e-15, epsrel=1e-13, limit=400)[0]
    return total


def _nonsmooth_mean(spec, lib, swap: bool) -> float:
    fam, forms = spec['family'], spec['forms']
    if fam == 'H1':
        fp, fm, c = forms[0], forms[1], spec['c']
        lp, lm = lib.law(fp), lib.law(fm)
        if not swap:
            return spec['alpha'] * _density_integral(lm, lambda t: lp.excess(c + t), [-c], float(_lam(fm).max()))
        return spec['alpha'] * _density_integral(lp, lambda s: lm.shortfall(s - c), [c], float(_lam(fp).max()))
    if fam == 'H2':
        (fa, ca), (fb, cb) = ((forms[0], spec['c1']), (forms[1], spec['c2']))
        if swap:
            (fa, ca), (fb, cb) = (fb, cb), (fa, ca)
        la, lb = lib.law(fa), lib.law(fb)

        def g(t):
            m = max(cb - t, 0.0)
            return m + la.shortfall(ca - m)

        return spec['alpha'] * _density_integral(lb, g, [cb, cb - ca], float(_lam(fb).max()))
    return spec['alpha'] * lib.law(forms[0], np.asarray(spec['w'])).cdf(spec['c'])


def check_distributions(spec: dict, n: int = 6, seed: int = 0) -> float:
    """Max difference (relative above 1e-4, absolute below) of every form's survival function and density between
    route A (Ruben, beta = min lam) and an independent method, at points spread over its bulk and tails: direct conditioning for two-term forms,
    Imhof's inversion of the characteristic function for six terms or more (its slowly decaying oscillatory tail is
    not resolved to 1e-12 below that). Four-term forms rely on the same series code, checked at every other size."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for form in spec['forms']:
        lam, d2 = _lam(form), _delta2(form)
        if len(lam) == 4:
            continue
        law = gchisq.Ruben(lam, d2, 'min')
        for x in np.maximum(law.mean + math.sqrt(law.var) * rng.uniform(-2.0, 6.0, n), 0.05 * law.mean):
            other = gchisq.direct2(lam, d2, x) if len(lam) == 2 else (gchisq.imhof_sf(lam, d2, x),
                                                                       gchisq.imhof_pdf(lam, d2, x))
            pairs = ((law.sf(x), other[0]), (law.pdf(x), other[1]))
            for a, b in pairs:  # relative above 1e-4, absolute below (far-tail values carry no weight)
                worst = max(worst, abs(a - b) / max(abs(a), 1e-4))
    return worst


def reference(spec: dict) -> dict:
    """Route A: Ruben's mixture with beta = min lam."""
    val = spec['beta0'] + _nonsmooth_mean(spec, _Ruben('min'), swap=False)
    how = 'closed form' if spec['family'] in ('H3', 'H3t') else '1D integral over the density of the second form'
    return {'value': val, 'abs_err': 1e-13 * max(1.0, abs(val)), 'method': f"A: {_Ruben('min').name}, {how}"}


def reference_swapped(spec: dict) -> dict:
    """Route B: Ruben's mixture with beta = 0.9 min lam, integrating over the other form's density."""
    val = spec['beta0'] + _nonsmooth_mean(spec, _Ruben('scaled'), swap=True)
    how = 'closed form' if spec['family'] in ('H3', 'H3t') else '1D integral over the density of the first form'
    return {'value': val, 'abs_err': 1e-13 * max(1.0, abs(val)), 'method': f"B: {_Ruben('scaled').name}, {how}"}
