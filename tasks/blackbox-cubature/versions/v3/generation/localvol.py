"""Single-asset local-volatility option families for blackbox-cubature v3.1 (DUAL_PLAN section 9.9).

Canonical coordinates y ~ N(0, I_m), one per monitoring step (d = m). Log-Euler scheme with a random smooth local
volatility sigma(t, x), x the log-price:

    x_{j+1} = x_j + (r - sigma_j^2 / 2) dt + sigma_j sqrt(dt) y_j,   sigma_j = sigma(t_j, x_j),

which keeps e^{-rt} S_t an exact martingale. sigma(t, x) = lo + (hi - lo) * logistic(g(t, x)) with
g = c0 + sum_{k<=6} (a_k cos(k pi u) + b_k sin(k pi u)) + tau (t / T - 1/2), u = (x - x0) / W: about fourteen random
parameters per instance, so the payoff is no longer a sum of exponentials of linear forms and its kink and jump
surfaces are curved in y.

  lv_call       e^{-rT} (S_T - K)^+
  lv_digital    e^{-rT} 10 1{S_T > K}
  lv_do_call    e^{-rT} (S_T - K)^+ 1{min_j S_j > B_low}
  lv_dko_call   e^{-rT} (S_T - K)^+ 1{B_low < S_j < B_high for all j}

Reference: backward Nystrom recursion of the Gaussian transition kernel (mean and variance depend on the source
state and the step) on Gauss-Legendre panels, with log K and the barriers as panel edges; two resolutions bound the
error.
"""
from __future__ import annotations

import math

import numpy as np

FAMILIES = ('lv_call', 'lv_digital', 'lv_do_call', 'lv_dko_call')
CASH = 10.0
MODES = 6


def sample(family: str, rng: np.random.Generator, m: int = 32) -> dict:
    s0 = 100.0
    T = float(rng.uniform(0.5, 1.5))
    lo, hi = float(rng.uniform(0.08, 0.15)), float(rng.uniform(0.55, 0.9))
    level = float(rng.uniform(0.2, 0.45))
    spec = {'family': family, 'L': 1, 'm': int(m), 'd': int(m), 's0': s0, 'r': float(rng.uniform(0.0, 0.08)), 'T': T,
            'vol': {'lo': lo, 'hi': hi, 'c0': math.log((level - lo) / (hi - level)),
                    'a': (rng.standard_normal(MODES) * 0.8 / np.arange(1, MODES + 1)).tolist(),
                    'b': (rng.standard_normal(MODES) * 0.8 / np.arange(1, MODES + 1)).tolist(),
                    'tau': float(rng.uniform(-1.0, 1.0)), 'W': float(4.0 * level * math.sqrt(T))}}
    if family == 'lv_call':
        spec['K'] = float(s0 * rng.uniform(0.9, 1.35))
    elif family == 'lv_digital':
        spec['K'] = float(s0 * rng.uniform(1.0, 1.5))
    elif family == 'lv_do_call':
        spec.update(K=float(s0 * rng.uniform(0.9, 1.2)), B_low=float(s0 * rng.uniform(0.80, 0.97)))
    elif family == 'lv_dko_call':
        spec.update(K=float(s0 * rng.uniform(0.9, 1.15)), B_low=float(s0 * rng.uniform(0.75, 0.95)),
                    B_high=float(s0 * rng.uniform(1.2, 1.6)))
    else:
        raise ValueError(family)
    return spec


def sigma(spec: dict, t, x) -> np.ndarray:
    v = spec['vol']
    u = (np.asarray(x, float) - math.log(spec['s0'])) / v['W']
    k = np.arange(1, MODES + 1)
    phase = np.multiply.outer(u, k) * math.pi
    g = v['c0'] + np.cos(phase) @ np.asarray(v['a']) + np.sin(phase) @ np.asarray(v['b']) \
        + v['tau'] * (np.asarray(t, float) / spec['T'] - 0.5)
    return v['lo'] + (v['hi'] - v['lo']) / (1.0 + np.exp(-g))


def log_paths(spec: dict, y: np.ndarray) -> np.ndarray:
    """(n, m) log-prices on the monitoring dates."""
    m, dt, r = spec['m'], spec['T'] / spec['m'], spec['r']
    x = np.full(len(y), math.log(spec['s0']))
    out = np.empty((len(y), m))
    for j in range(m):
        s = sigma(spec, j * dt, x)
        x = x + (r - 0.5 * s * s) * dt + s * math.sqrt(dt) * y[:, j]
        out[:, j] = x
    return out


def payoff(spec: dict, y: np.ndarray) -> np.ndarray:
    fam, K = spec['family'], spec['K']
    x = log_paths(spec, np.atleast_2d(y))
    sT = np.exp(np.minimum(x[:, -1], 700.0))
    if fam == 'lv_digital':
        val = CASH * (sT > K)
    else:
        val = np.maximum(sT - K, 0.0)
        if fam in ('lv_do_call', 'lv_dko_call'):
            alive = x.min(axis=1) > math.log(spec['B_low'])
            if fam == 'lv_dko_call':
                alive &= x.max(axis=1) < math.log(spec['B_high'])
            val = val * alive
    return math.exp(-spec['r'] * spec['T']) * val


TAIL = 9.0   # domain half-width in units of sigma_max sqrt(T): the exponential martingale bound leaves < e^{-40}
BAND = 12.0  # kernel support in one-step standard deviations


def quad(spec: dict, width: float = 0.5, nodes: int = 20) -> float:
    """Backward recursion V_j(x) = 1{alive} int V_{j+1}(x') p_j(x' | x) dx', V_m = payoff, price = e^{-rT} V_0(x0)
    with p_j(. | x) = N(x + (r - s^2/2) dt, s^2 dt), s = sigma(t_j, x) sqrt(dt). Gauss-Legendre panels of width
    `width` x the smallest one-step standard deviation on the domain, log K and the barriers as edges; each source
    node only reaches targets within BAND standard deviations (banded kernel)."""
    fam, m, r, T, K = spec['family'], spec['m'], spec['r'], spec['T'], spec['K']
    dt = T / m
    x0, lk = math.log(spec['s0']), math.log(K)
    reach = abs(r) * T + TAIL * spec['vol']['hi'] * math.sqrt(T)
    lo = max(math.log(spec['B_low']), x0 - reach) if 'B_low' in spec else x0 - reach
    hi = min(math.log(spec['B_high']), x0 + reach) if 'B_high' in spec else x0 + reach
    fine = np.linspace(lo, hi, 4001)
    smin = float(min(sigma(spec, t, fine).min() for t in np.linspace(0.0, T, 9))) * math.sqrt(dt)
    cuts = sorted({lo, hi} | ({lk} if lo < lk < hi else set()))
    xi, wi = np.polynomial.legendre.leggauss(nodes)
    xs, ws = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        n_pan = max(1, int(math.ceil((b - a) / (width * smin))))
        edges = np.linspace(a, b, n_pan + 1)
        xs.append((0.5 * (edges[:-1] + edges[1:])[:, None] + 0.5 * np.diff(edges)[:, None] * xi[None, :]).ravel())
        ws.append((0.5 * np.diff(edges)[:, None] * wi[None, :]).ravel())
    x, w = np.concatenate(xs), np.concatenate(ws)
    val = CASH * (x > lk).astype(float) if fam == 'lv_digital' else np.maximum(np.exp(x) - K, 0.0)

    def step(j: int, src: np.ndarray, val: np.ndarray) -> np.ndarray:
        s = sigma(spec, j * dt, src) * math.sqrt(dt)
        mean = src + r * dt - 0.5 * s * s
        start = np.searchsorted(x, mean - BAND * s)
        stop = np.searchsorted(x, mean + BAND * s)
        out = np.zeros(len(src))
        chunk = 512
        for c in range(0, len(src), chunk):
            sl = slice(c, c + chunk)
            a, b = int(start[sl].min()), int(stop[sl].max())
            if b <= a:
                continue
            z = (x[None, a:b] - mean[sl, None]) / s[sl, None]
            out[sl] = (np.exp(-0.5 * z * z) * (w[a:b] * val[a:b])[None, :]).sum(axis=1) / (s[sl] * math.sqrt(2 * math.pi))
        return out

    for j in range(m - 1, 0, -1):
        val = step(j, x, val)
    return math.exp(-r * T) * float(step(0, np.array([x0]), val)[0])


def references(spec: dict) -> list[dict]:
    fine, coarse = quad(spec, 0.5, 20), quad(spec, 1.0, 16)
    tol = max(abs(fine - coarse), 1e-13 * abs(fine))
    return [{'method': 'backward Nystrom recursion of the local-volatility transition kernel (QUAD), panels 0.5 s x 20',
             'value': fine, 'tol': tol, 'se': None, 'derived': False},
            {'method': 'same recursion, panels 1.0 s x 16 (resolution check)', 'value': coarse, 'tol': tol, 'se': None,
             'derived': True}]


def check_constant_vol(spec: dict) -> float:
    """Relative difference between quad() with the volatility frozen at its value at (0, x0) and an independent value
    of the same contract under geometric Brownian motion: Black-Scholes for the call and the digital, the dense GBM
    recursion of paths.barrier_quad for the barriers. Exercises this instance's grid, cuts, truncation and dates."""
    from scipy.special import ndtr

    import paths

    v = float(sigma(spec, 0.0, math.log(spec['s0'])))
    vol = spec['vol']
    frozen = {**spec, 'vol': {**vol, 'a': [0.0] * MODES, 'b': [0.0] * MODES, 'tau': 0.0,
                              'c0': math.log((v - vol['lo']) / (vol['hi'] - v))}}
    q = quad(frozen)
    s0, r, T, K = spec['s0'], spec['r'], spec['T'], spec['K']
    if spec['family'] in ('lv_call', 'lv_digital'):
        d2 = (math.log(s0 / K) + (r - 0.5 * v * v) * T) / (v * math.sqrt(T))
        exact = (s0 * ndtr(d2 + v * math.sqrt(T)) - K * math.exp(-r * T) * ndtr(d2) if spec['family'] == 'lv_call'
                 else CASH * math.exp(-r * T) * ndtr(d2))
    else:
        gbm = {'family': 'barrier_do_call' if spec['family'] == 'lv_do_call' else 'barrier_dko_call', 'L': 1,
               'm': spec['m'], 'd': spec['m'], 's0': [s0], 'sigma': [v], 'r': r, 'T': T, 'K': K,
               'B_low': spec['B_low'], **({'B_high': spec['B_high']} if 'B_high' in spec else {})}
        exact = paths.barrier_quad(gbm, 0.5, 20)
    return abs(q - exact) / max(abs(exact), 1e-300)
