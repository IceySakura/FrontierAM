"""Option-payoff families for blackbox-cubature v3: discounted payoffs under correlated geometric Brownian motion.

Canonical coordinates y ~ N(0, I_d). European payoffs use d = L (one Gaussian per asset, correlated through the
Cholesky factor C of the correlation matrix); path payoffs use d = L m (m equal time steps, increments sqrt(dt) C eps_l).

Reference values use at least two independent routes per family (verify.py compares them):
- closed forms (geometric basket, Margrabe exchange, Stulz call on min / max of two, digital on max of two);
- conditioning: with assets ordered so that the last asset alone depends on the last Gaussian, the expectation over that
  Gaussian is a lognormal call / put formula, and the remaining 1 or 2 Gaussians are integrated by adaptive quadrature
  split at the kinks (different asset orders give different, independent quadratures);
- preintegration plus RQMC for sums of lognormals (arithmetic baskets, Asian, Asian spread): along a direction in which
  every positive term increases and every negative term decreases, the conditional expectation has a closed form at
  the single root, and the remaining d - 1 dimensions, ordered by an active-subspace estimate, are integrated by
  replicated scrambled Sobol' points. Two different directions give two independent routes; plain RQMC of the payoff
  (with a geometric control variate where one exists) is a coarse third check. check_representation and
  check_preint test the two identities this rests on.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, optimize, special

from gauss import bvn_cdf, cdf, pdf, rqmc_cv_estimate, rqmc_estimate

FAMILIES = ('geo_basket_call', 'basket_call', 'exchange', 'spread', 'rainbow_max_call', 'rainbow_min_call',
            'digital_max', 'asian_call', 'basket_asian_call', 'asian_spread')
DIGITAL_CASH = 10.0


# ---------------------------------------------------------------- market and sampling

def _market(rng: np.random.Generator, L: int) -> dict:
    f = rng.normal(0.0, 1.0, (L, 2))
    cov = f @ f.T + np.diag(rng.uniform(0.3, 1.0, L))
    dd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(dd, dd)
    return {'L': L, 's0': (100.0 * rng.uniform(0.9, 1.1, L)).tolist(), 'sigma': rng.uniform(0.15, 0.45, L).tolist(),
            'r': float(rng.uniform(0.0, 0.05)), 'T': float(rng.uniform(0.5, 2.0)), 'corr': corr.tolist()}


def sample(family: str, rng: np.random.Generator, L: int = 2, m: int = 1) -> dict:
    spec = {'family': family, **_market(rng, L), 'm': int(m)}
    spec['d'] = L * m
    level = float(np.mean(spec['s0']))
    if family in ('geo_basket_call', 'basket_call', 'basket_asian_call'):
        w = rng.uniform(0.5, 1.5, L)
        spec['w'] = (w / w.sum()).tolist()
        spec['K'] = float(level * rng.uniform(0.85, 1.15))
    elif family == 'asian_call':
        spec['K'] = float(level * rng.uniform(0.85, 1.15))
    elif family in ('spread', 'asian_spread'):
        spec['K'] = float(rng.uniform(1.0, 10.0))
    elif family == 'exchange':
        spec['K'] = 0.0
    elif family == 'rainbow_max_call':
        spec['K'] = float(level * rng.uniform(0.95, 1.3))
    elif family == 'rainbow_min_call':
        spec['K'] = float(level * rng.uniform(0.7, 1.05))
    elif family == 'digital_max':
        spec['K'] = float(level * rng.uniform(1.0, 1.3))
    else:
        raise ValueError(family)
    return spec


def sample_hard(family: str, rng: np.random.Generator, L: int = 1, m: int = 32) -> dict:
    """asian_call / basket_asian_call around the literature settings (He 2019; Liu and Owen 2023; Chen, Xu, Du, Wang
    2025: S0 = 100, sigma 0.1 to 0.4, r 0.05 to 0.1, K/S0 0.8 to 1.2), pushed towards higher volatility and deeper
    out-of-the-money strikes."""
    spec = sample(family, rng, L=L, m=m)
    spec['s0'] = [100.0] if L == 1 else (100.0 * rng.uniform(0.9, 1.1, L)).tolist()
    spec['sigma'] = rng.uniform(0.3, 0.6, L).tolist() if L == 1 else rng.uniform(0.1, 0.5, L).tolist()
    spec['r'], spec['T'] = float(rng.uniform(0.0, 0.1)), float(rng.uniform(0.5, 1.5))
    spec['K'] = float(np.mean(spec['s0']) * rng.uniform(0.8, 1.4))
    return spec


def _chol(spec: dict, order=None) -> np.ndarray:
    corr = np.asarray(spec['corr'])
    if order is not None:
        corr = corr[np.ix_(order, order)]
    return np.linalg.cholesky(corr)


def _drift(spec: dict, t) -> np.ndarray:
    s0, sig = np.asarray(spec['s0']), np.asarray(spec['sigma'])
    return np.log(s0) + (spec['r'] - 0.5 * sig ** 2) * np.asarray(t)[..., None]


# ---------------------------------------------------------------- canonical evaluation

def log_prices(spec: dict, y: np.ndarray) -> np.ndarray:
    """European: (n, L) log S_i(T). Path: (n, m, L) log S_i(t_j)."""
    L, m, sig, T = spec['L'], spec['m'], np.asarray(spec['sigma']), spec['T']
    c = _chol(spec)
    if m == 1:
        return _drift(spec, T) + sig * math.sqrt(T) * (y @ c.T)
    dt = T / m
    eps = y.reshape(len(y), m, L)
    w = np.cumsum(math.sqrt(dt) * (eps @ c.T), axis=1)
    return _drift(spec, dt * np.arange(1, m + 1)) + sig * w


def argument(spec: dict, y: np.ndarray) -> np.ndarray:
    """Undiscounted quantity inside (.)^+ for the call-type payoffs (not defined for digital_max)."""
    fam, K = spec['family'], spec['K']
    x = log_prices(spec, y)
    if fam == 'geo_basket_call':
        return np.exp(x @ np.asarray(spec['w'])) - K
    if fam == 'basket_call':
        return np.exp(x) @ np.asarray(spec['w']) - K
    if fam in ('exchange', 'spread'):
        s = np.exp(x)
        return s[:, 0] - s[:, 1] - K
    if fam == 'rainbow_max_call':
        return np.exp(x).max(axis=1) - K
    if fam == 'rainbow_min_call':
        return np.exp(x).min(axis=1) - K
    if fam == 'asian_call':
        return np.exp(x[:, :, 0]).mean(axis=1) - K
    if fam == 'basket_asian_call':
        return np.exp(x).mean(axis=1) @ np.asarray(spec['w']) - K
    if fam == 'asian_spread':
        a = np.exp(x).mean(axis=1)
        return a[:, 0] - a[:, 1] - K
    raise ValueError(fam)


def payoff(spec: dict, y: np.ndarray) -> np.ndarray:
    disc = math.exp(-spec['r'] * spec['T'])
    if spec['family'] == 'digital_max':
        return disc * DIGITAL_CASH * (np.exp(log_prices(spec, y)).max(axis=1) > spec['K'])
    return disc * np.maximum(argument(spec, y), 0.0)


# ---------------------------------------------------------------- lognormal closed forms

def ln_call(mu, a, x):
    """E[(exp(mu + a U) - x)^+], U ~ N(0,1), a > 0; x may be <= 0 (then the payoff is linear)."""
    mu, x = np.asarray(mu, float), np.asarray(x, float)
    mean = np.exp(mu + 0.5 * a * a)
    pos = x > 0
    out = mean - x
    xs = np.where(pos, x, 1.0)
    d2 = (mu - np.log(xs)) / a
    out = np.where(pos, mean * cdf(d2 + a) - xs * cdf(d2), out)
    return out


def ln_put(mu, a, x):
    """E[(x - exp(mu + a U))^+]; zero for x <= 0."""
    mu, x = np.asarray(mu, float), np.asarray(x, float)
    pos = x > 0
    xs = np.where(pos, x, 1.0)
    d2 = (mu - np.log(xs)) / a
    val = xs * cdf(-d2) - np.exp(mu + 0.5 * a * a) * cdf(-d2 - a)
    return np.where(pos, val, 0.0)


def ln_prob_above(mu, a, x):
    """P(exp(mu + a U) > x) for x > 0."""
    return cdf((np.asarray(mu, float) - np.log(x)) / a)


# ---------------------------------------------------------------- closed forms

def geo_basket_closed(spec: dict) -> float:
    sig, T, w = np.asarray(spec['sigma']), spec['T'], np.asarray(spec['w'])
    c = _chol(spec)
    mu = float(w @ _drift(spec, T))
    a = float(np.linalg.norm((w * sig * math.sqrt(T)) @ c))
    return math.exp(-spec['r'] * spec['T']) * float(ln_call(mu, a, spec['K']))


def geo_basket_quad(spec: dict) -> float:
    sig, T, w = np.asarray(spec['sigma']), spec['T'], np.asarray(spec['w'])
    mu = float(w @ _drift(spec, T))
    a = float(np.linalg.norm((w * sig * math.sqrt(T)) @ _chol(spec)))
    u0 = (math.log(spec['K']) - mu) / a
    sq = math.sqrt(2.0 * math.pi)
    val, _ = integrate.quad(lambda u: (math.exp(mu + a * u - 0.5 * u * u) - spec['K'] * math.exp(-0.5 * u * u)) / sq,
                            u0, max(u0, a) + 40.0, epsabs=1e-14, epsrel=1e-14, limit=400)
    return math.exp(-spec['r'] * spec['T']) * val


def margrabe(spec: dict) -> float:
    s1, s2 = spec['s0']
    sig1, sig2 = spec['sigma']
    rho = spec['corr'][0][1]
    v = math.sqrt(sig1 ** 2 + sig2 ** 2 - 2 * rho * sig1 * sig2) * math.sqrt(spec['T'])
    d1 = (math.log(s1 / s2) + 0.5 * v * v) / v
    return float(s1 * cdf(d1) - s2 * cdf(d1 - v))


def stulz_min_call(spec: dict) -> float:
    """Call on the minimum of two assets (Stulz 1982, no dividends)."""
    s1, s2 = spec['s0']
    sig1, sig2 = spec['sigma']
    rho, r, T, K = spec['corr'][0][1], spec['r'], spec['T'], spec['K']
    sq = math.sqrt(T)
    sig = math.sqrt(sig1 ** 2 + sig2 ** 2 - 2 * rho * sig1 * sig2)
    g1 = (math.log(s1 / K) + (r - 0.5 * sig1 ** 2) * T) / (sig1 * sq)
    g2 = (math.log(s2 / K) + (r - 0.5 * sig2 ** 2) * T) / (sig2 * sq)
    rho1 = (rho * sig2 - sig1) / sig
    rho2 = (rho * sig1 - sig2) / sig
    t1 = bvn_cdf(g1 + sig1 * sq, (math.log(s2 / s1) - 0.5 * sig ** 2 * T) / (sig * sq), rho1)
    t2 = bvn_cdf(g2 + sig2 * sq, (math.log(s1 / s2) - 0.5 * sig ** 2 * T) / (sig * sq), rho2)
    t3 = bvn_cdf(g1, g2, rho)
    return float(s1 * t1 + s2 * t2 - K * math.exp(-r * T) * t3)


def bs_call(s0: float, sig: float, r: float, T: float, K: float) -> float:
    a = sig * math.sqrt(T)
    mu = math.log(s0) + (r - 0.5 * sig ** 2) * T
    return math.exp(-r * T) * float(ln_call(mu, a, K))


def stulz_max_call(spec: dict) -> float:
    """(max - K)^+ + (min - K)^+ = (S1 - K)^+ + (S2 - K)^+ for K >= 0."""
    r, T, K = spec['r'], spec['T'], spec['K']
    c1 = bs_call(spec['s0'][0], spec['sigma'][0], r, T, K)
    c2 = bs_call(spec['s0'][1], spec['sigma'][1], r, T, K)
    return c1 + c2 - stulz_min_call(spec)


def digital_max_closed(spec: dict) -> float:
    r, T, K = spec['r'], spec['T'], spec['K']
    g = [(math.log(spec['s0'][i] / K) + (r - 0.5 * spec['sigma'][i] ** 2) * T) / (spec['sigma'][i] * math.sqrt(T))
         for i in range(2)]
    return DIGITAL_CASH * math.exp(-r * T) * (1.0 - bvn_cdf(-g[0], -g[1], spec['corr'][0][1]))


# ---------------------------------------------------------------- conditioning (European, L <= 3)

def _conditional_setup(spec: dict, order):
    order = list(order)
    sig = np.asarray(spec['sigma'])[order]
    s0 = np.asarray(spec['s0'])[order]
    T, r = spec['T'], spec['r']
    c = _chol(spec, order)
    base = np.log(s0) + (r - 0.5 * sig ** 2) * T
    load = sig[:, None] * math.sqrt(T) * c
    return order, base, load


def _inner(spec: dict, order, known: np.ndarray, mu_last: float, a_last: float) -> float:
    """Expectation over the last Gaussian given the other assets' prices `known` (in `order`, last excluded)."""
    fam, K = spec['family'], spec['K']
    last = order[-1]
    if fam == 'basket_call':
        w = np.asarray(spec['w'])
        rest = sum(w[order[i]] * known[i] for i in range(len(known)))
        return w[last] * float(ln_call(mu_last, a_last, (K - rest) / w[last]))
    if fam in ('exchange', 'spread'):
        if last == 1:
            return float(ln_put(mu_last, a_last, known[0] - K))
        return float(ln_call(mu_last, a_last, known[0] + K))
    if fam == 'rainbow_max_call':
        mx = max(known)
        return max(mx - K, 0.0) + float(ln_call(mu_last, a_last, max(mx, K)))
    if fam == 'rainbow_min_call':
        mn = min(known)
        return 0.0 if mn <= K else float(ln_call(mu_last, a_last, K) - ln_call(mu_last, a_last, mn))
    if fam == 'digital_max':
        return DIGITAL_CASH if max(known) > K else DIGITAL_CASH * float(ln_prob_above(mu_last, a_last, K))
    raise ValueError(fam)


def _thresholds(spec: dict, order, known: np.ndarray, j: int) -> list[float]:
    """Prices of asset order[j] at which the conditional integrand has a kink or jump, given assets before it."""
    fam, K = spec['family'], spec['K']
    out = []
    if fam == 'basket_call':
        w = np.asarray(spec['w'])
        rest = K - sum(w[order[i]] * known[i] for i in range(j))
        if rest > 0:
            out.append(rest / w[order[j]])
    elif fam in ('exchange', 'spread'):
        if j == 0 and order[-1] == 1 and K > 0:
            out.append(K)
    else:
        out.append(K)
        out += [float(v) for v in known[:j]]
    return [v for v in out if v is not None and v > 0]


def conditional_quad(spec: dict, order) -> float:
    order, base, load = _conditional_setup(spec, order)
    L = len(order)
    disc = math.exp(-spec['r'] * spec['T'])
    opts = dict(epsabs=1e-13, epsrel=1e-13, limit=400)

    def breaks(j, prev_y):
        mu = base[j] + sum(load[j, i] * prev_y[i] for i in range(j))
        known = np.exp([base[i] + sum(load[i, k] * prev_y[k] for k in range(i + 1)) for i in range(j)])
        pts = [(math.log(t) - mu) / load[j, j] for t in _thresholds(spec, order, known, j)]
        return sorted(p for p in pts if -12 < p < 12)

    def integrand(prev_y):
        known = np.exp([base[i] + sum(load[i, k] * prev_y[k] for k in range(i + 1)) for i in range(L - 1)])
        mu_last = base[L - 1] + sum(load[L - 1, i] * prev_y[i] for i in range(L - 1))
        return _inner(spec, order, known, mu_last, load[L - 1, L - 1])

    def level(j, prev_y):
        if j == L - 1:
            return integrand(prev_y)
        cuts = [-12.0] + breaks(j, prev_y) + [12.0]
        total = 0.0
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if hi - lo > 1e-14:
                total += integrate.quad(lambda t: level(j + 1, prev_y + [t]) * float(pdf(t)), lo, hi, **opts)[0]
        return total

    return disc * level(0, [])


# ---------------------------------------------------------------- sums of lognormals: preintegration + RQMC

def lognormal_terms(spec: dict):
    """Payoff as (sum_k c_k exp(mu_k + b_k . y) - sum_k' e_k' exp(nu_k' + f_k' . y) - K)^+ (second sum may be empty)."""
    fam, L, m, T = spec['family'], spec['L'], spec['m'], spec['T']
    sig = np.asarray(spec['sigma'])
    c = _chol(spec)
    d = spec['d']
    if m == 1:
        times = np.array([T])
    else:
        times = T / m * np.arange(1, m + 1)
    terms = []
    for j, t in enumerate(times):
        for i in range(L):
            b = np.zeros(d)
            if m == 1:
                b[:L] = sig[i] * math.sqrt(T) * c[i]
            else:
                dt = T / m
                for l in range(j + 1):
                    b[l * L:(l + 1) * L] = sig[i] * math.sqrt(dt) * c[i]
            mu = math.log(spec['s0'][i]) + (spec['r'] - 0.5 * sig[i] ** 2) * t
            terms.append((i, mu, b))
    w = np.asarray(spec.get('w', [1.0] * L))
    if fam in ('basket_call', 'asian_call', 'basket_asian_call'):
        pos = [(w[i] / len(times), mu, b) for i, mu, b in terms]
        neg = []
    elif fam == 'asian_spread':
        pos = [(1.0 / len(times), mu, b) for i, mu, b in terms if i == 0]
        neg = [(1.0 / len(times), mu, b) for i, mu, b in terms if i == 1]
    else:
        raise ValueError(fam)
    return pos, neg


def _drivers(pos, neg) -> tuple[np.ndarray, np.ndarray]:
    bp = np.array([b for _, _, b in pos])
    bn = np.array([b for _, _, b in neg]) if neg else np.zeros((0, bp.shape[1]))
    return bp, bn


def monotone(theta: np.ndarray, pos, neg) -> bool:
    """Every positive term strictly increases and every negative term strictly decreases along theta."""
    bp, bn = _drivers(pos, neg)
    tol = 1e-8 * np.abs(np.vstack([bp, bn])).max()
    return bool((bp @ theta).min() > tol and (len(bn) == 0 or (bn @ theta).max() < -tol))


def _closest_monotone(target: np.ndarray, pos, neg, margin: float = 0.05) -> np.ndarray:
    """Unit direction closest to `target` whose positive slopes are >= margin s and negative slopes <= -margin s
    (s = largest |slope| along target): bounded least squares in the slopes, using that the stacked drivers B are
    square and invertible for every family here (one driver per asset and date)."""
    bp, bn = _drivers(pos, neg)
    inv = np.linalg.inv(np.vstack([bp, bn]))
    t = target / np.linalg.norm(target)
    s = np.abs(np.vstack([bp, bn]) @ t).max()
    lb = np.concatenate([np.full(len(bp), margin * s), np.full(len(bn), -np.inf)])
    ub = np.concatenate([np.full(len(bp), np.inf), np.full(len(bn), -margin * s)])
    th = inv @ optimize.lsq_linear(inv, t, bounds=(lb, ub), lsq_solver='exact', tol=1e-12).x
    return th / np.linalg.norm(th)


def directions(spec: dict, pos, neg) -> list[tuple[str, np.ndarray]]:
    """Two different unit directions along which the payoff argument is strictly increasing.

    The first is the gradient of the argument at y = 0 (the direction of largest variation, as in Liu-Owen), replaced
    by the closest monotone direction when the gradient is not monotone (negative correlations, spreads). The second
    tilts the first by a fixed random orthogonal vector (cosine about 0.9) and is made monotone the same way, so the
    two routes condition on different variables at comparable precision.
    """
    bp, bn = _drivers(pos, neg)
    size_p = np.array([cw * math.exp(mu) for cw, mu, _ in pos])
    size_n = np.array([cw * math.exp(mu) for cw, mu, _ in neg])
    grad = size_p @ bp - (size_n @ bn if len(bn) else 0.0)
    grad = grad / np.linalg.norm(grad)
    first = ('gradient', grad) if monotone(grad, pos, neg) else \
        ('closest monotone to gradient', _closest_monotone(grad, pos, neg))
    rng = np.random.default_rng(20261124)
    for _ in range(10):
        xi = rng.standard_normal(len(grad))
        xi -= (xi @ first[1]) * first[1]
        tilt = first[1] + 0.5 * xi / np.linalg.norm(xi)
        tilt /= np.linalg.norm(tilt)
        second = ('tilted gradient', tilt) if monotone(tilt, pos, neg) else \
            ('closest monotone to tilted gradient', _closest_monotone(tilt, pos, neg))
        if abs(first[1] @ second[1]) < 0.995:
            break
    for name, th in (first, second):
        if not monotone(th, pos, neg):
            raise RuntimeError(f'direction {name} is not monotone')
    return [first, second]


def _root(alpha: np.ndarray, beta: np.ndarray, alpha_n: np.ndarray, gamma: np.ndarray, K: float) -> np.ndarray:
    """Per row, the u solving sum_k exp(alpha_k + beta_k u) = K + sum_j exp(alpha_n_j + gamma_j u), with beta > 0,
    gamma < 0, K > 0, so that h(u) = log(lhs) - log(rhs) is strictly increasing. Safeguarded Newton on h."""
    n, cap = len(alpha), 60.0

    def h(u):
        a = alpha + np.outer(u, beta)
        amax = a.max(axis=1)
        e = np.exp(a - amax[:, None])
        s = e.sum(axis=1)
        val, der = np.log(s) + amax, (e * beta).sum(axis=1) / s
        if gamma.size:
            en = np.exp(alpha_n + np.outer(u, gamma))
            x = K + en.sum(axis=1)
            val, der = val - np.log(x), der - (en * gamma).sum(axis=1) / x
        else:
            val = val - math.log(K)
        return val, der

    # at lo every positive term is <= K / n <= rhs; at hi one positive term reaches rhs(lo) >= rhs(hi)
    lo = np.maximum(((math.log(K / len(beta)) - alpha) / beta).min(axis=1), -cap)
    rhs_lo = K + (np.exp(alpha_n + np.outer(lo, gamma)).sum(axis=1) if gamma.size else np.zeros(n))
    hi = np.minimum(np.maximum(((np.log(rhs_lo)[:, None] - alpha) / beta).max(axis=1), lo), cap)
    h_lo, _ = h(lo)
    h_hi, _ = h(hi)
    below, above = h_lo >= 0, h_hi <= 0
    u = 0.5 * (lo + hi)
    for _ in range(200):
        g, dg = h(u)
        lo = np.where(g < 0, u, lo)
        hi = np.where(g > 0, u, hi)
        step = u - g / dg
        u_new = np.where((step <= lo) | (step >= hi) | ~np.isfinite(step), 0.5 * (lo + hi), step)
        done = np.max(np.abs(u_new - u)) < 1e-13
        u = u_new
        if done:
            break
    else:
        raise RuntimeError('preintegration root did not converge')
    u = np.where(below, -cap, u)
    return np.where(above & ~below, cap, u)


def preintegrated(spec: dict, theta: np.ndarray, basis: np.ndarray, pos, neg):
    """v -> E over u ~ N(0, 1) of the payoff at y = theta u + basis v (rows of v are points in R^{d-1}).

    The argument is strictly increasing in u, so the payoff is positive exactly for u > u*, and each term integrates to
    c exp(alpha + beta^2/2) Phi(beta - u*) in closed form."""
    K, disc = spec['K'], math.exp(-spec['r'] * spec['T'])
    bp, bn = _drivers(pos, neg)
    cp = np.array([math.log(cw) + mu for cw, mu, _ in pos])
    cn = np.array([math.log(cw) + mu for cw, mu, _ in neg])
    load_p, load_n = basis.T @ bp.T, basis.T @ bn.T
    beta, gamma = bp @ theta, bn @ theta

    def value(v):
        alpha, alpha_n = cp + v @ load_p, cn + v @ load_n
        u = _root(alpha, beta, alpha_n, gamma, K)[:, None]
        val = (np.exp(alpha + 0.5 * beta ** 2) * special.ndtr(beta - u)).sum(axis=1) - K * special.ndtr(-u[:, 0])
        if gamma.size:
            val = val - (np.exp(alpha_n + 0.5 * gamma ** 2) * special.ndtr(gamma - u)).sum(axis=1)
        return disc * val

    return value


def _complement(theta: np.ndarray) -> np.ndarray:
    """Orthonormal basis (columns) of the complement of theta: the Householder reflection taking e_1 to -+theta."""
    u = theta.copy()
    u[0] += 1.0 if theta[0] >= 0 else -1.0
    return (np.eye(len(theta)) - 2.0 * np.outer(u, u) / (u @ u))[:, 1:]


def _ordered_complement(spec: dict, theta: np.ndarray, pos, neg, n: int = 1 << 14, seed: int = 0) -> np.ndarray:
    """Orthonormal basis of the complement of theta ordered for QMC: eigenvectors of the projected E[grad f grad f^T]
    (the payoff's active-subspace matrix, estimated from n Gaussian points), largest eigenvalue first, so the leading
    Sobol' coordinates carry most of the variance."""
    bp, bn = _drivers(pos, neg)
    big = np.vstack([bp, bn])
    const = np.array([math.log(cw) + mu for cw, mu, _ in pos + neg])
    sign = np.concatenate([np.ones(len(pos)), -np.ones(len(neg))])
    y = np.random.default_rng(seed).standard_normal((n, spec['d']))
    terms = np.exp(const + y @ big.T) * sign
    grad = (terms * (terms.sum(axis=1) > spec['K'])[:, None]) @ big
    q = _complement(theta)
    _, vec = np.linalg.eigh(q.T @ (grad.T @ grad) @ q)
    return q @ vec[:, ::-1]


def preint_rqmc(spec: dict, theta: np.ndarray, m: int = 20, replicates: int = 16, seed: int = 1) -> tuple[float, float]:
    pos, neg = lognormal_terms(spec)
    f = preintegrated(spec, theta, _ordered_complement(spec, theta, pos, neg), pos, neg)
    return rqmc_estimate(f, spec['d'] - 1, m, replicates, seed)


def preint_gh(spec: dict, theta: np.ndarray, n: int = 64) -> float:
    """Tensor Gauss-Hermite rule with n nodes per dimension on the preintegrated function; for d - 1 <= 3, where the
    analytic preintegrated function makes the rule converge exponentially in n."""
    pos, neg = lognormal_terms(spec)
    k = spec['d'] - 1
    x, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / math.sqrt(2.0 * math.pi)
    v = np.stack([g.ravel() for g in np.meshgrid(*([x] * k), indexing='ij')], axis=1)
    weights = np.ones(1)
    for _ in range(k):
        weights = np.outer(weights, w).ravel()
    f = preintegrated(spec, theta, _complement(theta), pos, neg)
    return float(np.concatenate([f(v[i:i + 65536]) for i in range(0, len(v), 65536)]) @ weights)


def check_representation(spec: dict, n: int = 4000, seed: int = 0) -> float:
    """Max relative difference between `argument` and its sum-of-lognormals form at random Gaussian points."""
    pos, neg = lognormal_terms(spec)
    y = np.random.default_rng(seed).standard_normal((n, spec['d']))
    lhs = argument(spec, y)
    rhs = sum(cw * np.exp(mu + y @ b) for cw, mu, b in pos) - spec['K']
    if neg:
        rhs = rhs - sum(cw * np.exp(mu + y @ b) for cw, mu, b in neg)
    return float(np.max(np.abs(lhs - rhs) / (np.abs(lhs) + spec['K'])))


def check_preint(spec: dict, theta: np.ndarray, n: int = 12, seed: int = 0) -> float:
    """Max absolute difference between preint_value and adaptive quadrature of the payoff along the line, with the
    kink located by brentq on `argument` (independent of the term representation and of the Newton root)."""
    pos, neg = lognormal_terms(spec)
    basis = _complement(theta)
    v = np.random.default_rng(seed).standard_normal((n, spec['d'] - 1))
    closed = preintegrated(spec, theta, basis, pos, neg)(v)
    worst = 0.0
    for i in range(n):
        point = basis @ v[i]
        arg = lambda t: float(argument(spec, (point + t * theta)[None, :])[0])
        f = lambda t: float(payoff(spec, (point + t * theta)[None, :])[0] * pdf(t))
        lo, hi = -12.0, 12.0
        if arg(lo) < 0 < arg(hi):
            kink = optimize.brentq(arg, lo, hi, xtol=1e-14, maxiter=200)
            val = integrate.quad(f, kink, hi, epsabs=1e-13, epsrel=1e-13, limit=400)[0]
        else:
            val = integrate.quad(f, lo, hi, epsabs=1e-13, epsrel=1e-13, limit=400)[0]
        worst = max(worst, abs(val - closed[i]))
    return worst


def _geometric_terms(spec: dict):
    """(mu, b, mean): the geometric-average counterpart exp(mu + b . y) of a sum-of-lognormals payoff without negative
    terms, and the closed-form mean of its discounted call; None if the payoff has negative terms."""
    pos, neg = lognormal_terms(spec)
    if neg:
        return None
    wts = np.array([cw for cw, _, _ in pos])
    wts = wts / wts.sum()
    mu = float(sum(wk * mu for wk, (_, mu, _) in zip(wts, pos)))
    bvec = sum(wk * b for wk, (_, _, b) in zip(wts, pos))
    disc = math.exp(-spec['r'] * spec['T'])
    return mu, bvec, disc * float(ln_call(mu, float(np.linalg.norm(bvec)), spec['K']))


def geometric_control(spec: dict):
    """Geometric-average counterpart (closed form mean) of a sum-of-lognormals payoff without negative terms."""
    geo = _geometric_terms(spec)
    if geo is None:
        return None
    mu, bvec, mean = geo
    disc = math.exp(-spec['r'] * spec['T'])
    return (lambda y: disc * np.maximum(np.exp(mu + y @ bvec) - spec['K'], 0.0)), mean


def preint_cv_rqmc(spec: dict, theta: np.ndarray, m: int = 20, replicates: int = 16, seed: int = 1) -> tuple[float, float]:
    """preint_rqmc with the geometric-average control preintegrated along the same direction: on y = theta u + basis v
    the control exp(mu + b . basis v + (b . theta) u) is lognormal in u, so its conditional call value is ln_call."""
    pos, neg = lognormal_terms(spec)
    basis = _ordered_complement(spec, theta, pos, neg)
    f = preintegrated(spec, theta, basis, pos, neg)
    geo = _geometric_terms(spec)
    if geo is None:
        return rqmc_estimate(f, spec['d'] - 1, m, replicates, seed)
    mu, bvec, mean = geo
    a, load = abs(float(bvec @ theta)), basis.T @ bvec
    disc = math.exp(-spec['r'] * spec['T'])
    return rqmc_cv_estimate(lambda v: (f(v), disc * ln_call(mu + v @ load, a, spec['K'])), spec['d'] - 1, m,
                            replicates, seed, mean)


def plain_rqmc_cv(spec: dict, m: int = 20, replicates: int = 16, seed: int = 2) -> tuple[float, float]:
    ctrl = geometric_control(spec)
    if ctrl is None:
        return rqmc_estimate(lambda y: payoff(spec, y), spec['d'], m, replicates, seed)
    g, gmean = ctrl
    return rqmc_estimate(lambda y: payoff(spec, y) - g(y) + gmean, spec['d'], m, replicates, seed)


# ---------------------------------------------------------------- reference dispatch

def _route(method: str, value: float, tol: float = 0.0, se: float | None = None, derived: bool = False) -> dict:
    return {'method': method, 'value': float(value), 'tol': float(tol), 'se': None if se is None else float(se),
            'derived': derived}


def references(spec: dict, rqmc_m: int = 20, check_m: int = 18, replicates: int = 16) -> list[dict]:
    """All reference routes for an instance; the first entry is the primary reference.

    Deterministic routes carry a claimed absolute tolerance `tol`; RQMC routes carry the standard error `se` over
    independent scrambles (every route has its own seeds). A `derived` route combines other routes and is not an
    independent check."""
    fam, L = spec['family'], spec['L']
    out = []
    if fam == 'geo_basket_call':
        out.append(_route('closed form (lognormal)', geo_basket_closed(spec), 1e-13))
        out.append(_route('1D quadrature of the lognormal payoff', geo_basket_quad(spec), 1e-12))
    elif fam in ('exchange', 'spread'):
        if fam == 'exchange':
            out.append(_route('Margrabe closed form', margrabe(spec), 1e-13))
        for p in ([0, 1], [1, 0]):
            out.append(_route(f'conditioning on asset {p[0] + 1}, 1D quadrature', conditional_quad(spec, p), 1e-11))
    elif fam in ('rainbow_max_call', 'rainbow_min_call', 'digital_max'):
        if L == 2:
            closed = {'rainbow_max_call': stulz_max_call, 'rainbow_min_call': stulz_min_call,
                      'digital_max': digital_max_closed}[fam](spec)
            out.append(_route('closed form (bivariate normal CDF)', closed, 1e-13))
        for p in ([[0, 1], [1, 0]] if L == 2 else [[0, 1, 2], [2, 1, 0]]):
            out.append(_route(f'conditioning, order {p}, nested quadrature', conditional_quad(spec, p), 1e-11))
    elif fam in ('basket_call', 'asian_call', 'basket_asian_call', 'asian_spread'):
        pos, neg = lognormal_terms(spec)
        dirs = directions(spec, pos, neg)
        if fam == 'basket_call' and L <= 3:
            for p in ([[0, 1], [1, 0]] if L == 2 else [[0, 1, 2], [2, 0, 1]]):
                out.append(_route(f'conditioning, order {p}, nested quadrature', conditional_quad(spec, p), 1e-11))
        if spec['d'] - 1 <= 3:
            for name, theta in dirs:
                fine, coarse = preint_gh(spec, theta, 64), preint_gh(spec, theta, 48)
                out.append(_route(f'preintegration along {name} direction + tensor Gauss-Hermite 64^{spec["d"] - 1}',
                                  fine, max(abs(fine - coarse), 1e-13 * max(1.0, abs(fine)))))
        else:
            routes = []
            cv = ' + geometric control variate' if geometric_control(spec) is not None else ''
            for k, (name, theta) in enumerate(dirs):
                val, se = preint_cv_rqmc(spec, theta, m=rqmc_m, replicates=replicates, seed=11 + k)
                routes.append(_route(f'preintegration along {name} direction{cv} + RQMC 2^{rqmc_m} x {replicates}',
                                     val, se=max(se, 1e-300)))
            w = np.array([r['se'] ** -2 for r in routes])
            comb = float(sum(wi * r['value'] for wi, r in zip(w, routes)) / w.sum())
            out.append(_route('inverse-variance mean of the two preintegration routes', comb,
                              se=float(w.sum() ** -0.5), derived=True))
            out += routes
        val, se = plain_rqmc_cv(spec, m=check_m, replicates=replicates, seed=21)
        cv = ' + geometric control variate' if geometric_control(spec) is not None else ''
        out.append(_route(f'plain RQMC of the payoff 2^{check_m} x {replicates}{cv}', val, se=se))
    else:
        raise ValueError(fam)
    return out
