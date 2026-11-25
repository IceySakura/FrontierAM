"""Path-dependent option families from the QMC literature for blackbox-cubature v3, in their hard regime.

Canonical coordinates y ~ N(0, I_d). GBM families use options.log_prices (d = L m, m equal steps, assets correlated
through the Cholesky factor of corr). heston_asian uses y = (z_v, z_s), d = 2m: full-truncation Euler for the
variance and log-Euler for the asset.

  digital_asian       e^{-rT} 10 1{A > K}, A the arithmetic average             Xiao-Wang 2018
  rainbow_asian_max   e^{-rT} (max(A_1, A_2) - K)^+                             Liu-Owen 2023
  rainbow_asian_min   e^{-rT} (min(A_1, A_2) - K)^+                             Liu-Owen 2023
  barrier_do_call     e^{-rT} (S_T - K)^+ 1{min_j S_j > L}                       Xiao-Wang 2018
  barrier_dko_call    e^{-rT} (S_T - K)^+ 1{L < S_j < H for all j}               double knock-out
  barrier_asian_do    e^{-rT} (A - K)^+ 1{min_j S_j > L}                         Xiao-Wang 2018
  heston_asian        e^{-rT} (A - K)^+ under Heston                             Chen-Xu-Du-Wang 2025

Reference routes (two per family, see references()):
  - uniform scaling: moving one Gaussian along a fixed direction multiplies every price on the path by the same factor
    e^{a u}, so the expectation over it is a closed form (call, digital, or barrier-truncated call); the other d - 1
    coordinates use replicated scrambled Sobol' points in a PCA-ordered basis;
  - barrier_do_call, barrier_dko_call: recursive Gauss-Legendre quadrature of the Gaussian transition kernel between
    monitoring dates (QUAD), exponentially convergent in the panel width;
  - rainbow options: conditioning on the first increments of both assets gives Stulz's formula;
  - digital_asian: preintegration along the gradient direction of the average (root of a sum of lognormals);
  - barrier_asian_do: conditioning on the last increment instead of the first;
  - heston_asian: conditioning on the second asset shock (a two-term exponential in it) instead of the first.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import special

import options
from gauss import cdf, rqmc_estimate

FAMILIES = ('digital_asian', 'rainbow_asian_max', 'rainbow_asian_min', 'barrier_do_call', 'barrier_dko_call',
            'barrier_asian_do', 'heston_asian')
CASH = 10.0


# ---------------------------------------------------------------- sampling

def sample(family: str, rng: np.random.Generator, L: int = 1, m: int = 32) -> dict:
    if family == 'heston_asian':
        s0 = 100.0
        spec = {'family': family, 'L': 1, 'm': int(m), 'd': 2 * int(m), 's0': s0,
                'r': float(rng.uniform(0.0, 0.08)), 'T': float(rng.uniform(0.5, 1.5)),
                'v0': float(rng.uniform(0.04, 0.25)), 'theta': float(rng.uniform(0.04, 0.25)),
                'kappa': float(rng.uniform(0.5, 2.0)), 'xi': float(rng.uniform(0.2, 0.8)),
                'rho': float(rng.uniform(-0.9, 0.5)), 'K': float(s0 * rng.uniform(0.85, 1.3))}
        return spec
    if family in ('rainbow_asian_max', 'rainbow_asian_min'):
        spec = {'family': family, **options._market(rng, 2), 'm': int(m)}
        spec['sigma'] = rng.uniform(0.1, 0.5, 2).tolist()
        level = float(np.mean(spec['s0']))
        spec['K'] = float(level * (rng.uniform(1.05, 1.4) if family.endswith('max') else rng.uniform(0.7, 1.05)))
    else:
        spec = {'family': family, **options._market(rng, 1), 'm': int(m)}
        spec['s0'], spec['sigma'] = [100.0], [float(rng.uniform(0.2, 0.6))]
        spec['T'] = float(rng.uniform(0.5, 1.5))
        s0 = 100.0
        if family == 'digital_asian':
            spec['K'] = float(s0 * rng.uniform(1.0, 1.5))
        elif family == 'barrier_do_call':
            spec.update(K=float(s0 * rng.uniform(0.9, 1.2)), B_low=float(s0 * rng.uniform(0.80, 0.97)))
        elif family == 'barrier_dko_call':
            spec.update(K=float(s0 * rng.uniform(0.9, 1.15)), B_low=float(s0 * rng.uniform(0.75, 0.95)),
                        B_high=float(s0 * rng.uniform(1.2, 1.6)))
        elif family == 'barrier_asian_do':
            spec.update(K=float(s0 * rng.uniform(0.85, 1.2)), B_low=float(s0 * rng.uniform(0.80, 0.97)))
        else:
            raise ValueError(family)
    spec['d'] = spec['L'] * spec['m']
    return spec


# ---------------------------------------------------------------- canonical evaluation

def heston_paths(spec: dict, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prices (n, m) and the truncated variances V_j^+ (n, m) used in step j."""
    m, dt = spec['m'], spec['T'] / spec['m']
    rho, rhohat = spec['rho'], math.sqrt(1.0 - spec['rho'] ** 2)
    v = np.full(len(y), spec['v0'])
    logs = np.full(len(y), math.log(spec['s0']))
    out, used = np.empty((len(y), m)), np.empty((len(y), m))
    for j in range(m):
        vp = np.maximum(v, 0.0)
        sq = np.sqrt(vp * dt)
        logs = logs + (spec['r'] - 0.5 * vp) * dt + sq * (rho * y[:, j] + rhohat * y[:, m + j])
        v = v + spec['kappa'] * (spec['theta'] - vp) * dt + spec['xi'] * sq * y[:, j]
        out[:, j], used[:, j] = np.exp(logs), vp
    return out, used


def prices(spec: dict, y: np.ndarray) -> np.ndarray:
    """(n, m, L) prices on the monitoring dates."""
    if spec['family'] != 'heston_asian':
        return np.exp(options.log_prices(spec, y))
    return heston_paths(spec, y)[0][:, :, None]


def _payoff_from_prices(spec: dict, s: np.ndarray) -> np.ndarray:
    fam, K = spec['family'], spec['K']
    if fam == 'digital_asian':
        val = CASH * (s[:, :, 0].mean(axis=1) > K)
    elif fam in ('rainbow_asian_max', 'rainbow_asian_min'):
        a = s.mean(axis=1)
        val = np.maximum((a.max(axis=1) if fam.endswith('max') else a.min(axis=1)) - K, 0.0)
    elif fam == 'heston_asian':
        val = np.maximum(s[:, :, 0].mean(axis=1) - K, 0.0)
    else:
        p = s[:, :, 0]
        alive = p.min(axis=1) > spec['B_low']
        if fam == 'barrier_dko_call':
            alive &= p.max(axis=1) < spec['B_high']
        under = p.mean(axis=1) if fam == 'barrier_asian_do' else p[:, -1]
        val = np.maximum(under - K, 0.0) * alive
    return val


def payoff(spec: dict, y: np.ndarray) -> np.ndarray:
    return math.exp(-spec['r'] * spec['T']) * _payoff_from_prices(spec, prices(spec, y))


# ---------------------------------------------------------------- route 1: uniform scaling along one direction

def scaling_direction(spec: dict) -> tuple[np.ndarray, float]:
    """Unit theta and a > 0 such that y -> y + u theta multiplies every price by e^{a u}."""
    d, m, L = spec['d'], spec['m'], spec['L']
    theta = np.zeros(d)
    if spec['family'] == 'heston_asian':
        theta[m] = 1.0
        return theta, math.sqrt(1.0 - spec['rho'] ** 2) * math.sqrt(spec['v0'] * spec['T'] / m)
    dt = spec['T'] / m
    c = options._chol(spec)
    delta = np.linalg.solve(c, 1.0 / np.asarray(spec['sigma']))
    theta[:L] = delta / np.linalg.norm(delta)
    return theta, math.sqrt(dt) / float(np.linalg.norm(delta))


def _linear_map(spec: dict) -> np.ndarray:
    """Derivative of the log-price path (rows) with respect to y (columns); for Heston with the variance frozen at v0."""
    d, m, L = spec['d'], spec['m'], spec['L']
    dt = spec['T'] / m
    if spec['family'] == 'heston_asian':
        out = np.zeros((m, d))
        sq = math.sqrt(spec['v0'] * dt)
        for j in range(m):
            out[j, :j + 1] = spec['rho'] * sq
            out[j, m:m + j + 1] = math.sqrt(1.0 - spec['rho'] ** 2) * sq
        return out
    c = options._chol(spec)
    sig = np.asarray(spec['sigma'])
    out = np.zeros((m * L, d))
    for j in range(m):
        for l in range(L):
            for i in range(j + 1):
                out[j * L + l, i * L:(i + 1) * L] = sig[l] * math.sqrt(dt) * c[l]
    return out


def pca_complement(spec: dict, theta: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the complement of theta ordered by how strongly it moves the log-price path."""
    q = options._complement(theta)
    _, _, vt = np.linalg.svd(_linear_map(spec) @ q, full_matrices=False)
    basis = q @ vt.T
    if basis.shape[1] < q.shape[1]:
        extra = q - basis @ (basis.T @ q)
        u, s, _ = np.linalg.svd(extra, full_matrices=False)
        basis = np.hstack([basis, u[:, :q.shape[1] - basis.shape[1]]])
    return basis


def scaled_value(spec: dict, theta: np.ndarray, a: float, basis: np.ndarray):
    """v -> E over u of the payoff at y = basis v + u theta, using the uniform scaling e^{a u}."""
    fam, K, disc = spec['family'], spec['K'], math.exp(-spec['r'] * spec['T'])
    ea = math.exp(0.5 * a * a)

    def value(v):
        s = prices(spec, v @ basis.T)
        if fam == 'digital_asian':
            ustar = np.log(K / s[:, :, 0].mean(axis=1)) / a
            return disc * CASH * cdf(-ustar)
        if fam in ('rainbow_asian_max', 'rainbow_asian_min', 'heston_asian'):
            if fam == 'heston_asian':
                base = s[:, :, 0].mean(axis=1)
            else:
                avg = s.mean(axis=1)
                base = avg.max(axis=1) if fam.endswith('max') else avg.min(axis=1)
            ustar = np.log(K / base) / a
            return disc * (base * ea * cdf(a - ustar) - K * cdf(-ustar))
        p = s[:, :, 0]
        ulow = np.log(spec['B_low'] / p.min(axis=1)) / a
        under = p.mean(axis=1) if fam == 'barrier_asian_do' else p[:, -1]
        ustar = np.maximum(ulow, np.log(K / under) / a)
        if fam == 'barrier_dko_call':
            uhigh = np.log(spec['B_high'] / p.max(axis=1)) / a
            upper = np.maximum(uhigh, ustar)
            return disc * (under * ea * (cdf(upper - a) - cdf(ustar - a)) - K * (cdf(upper) - cdf(ustar)))
        return disc * (under * ea * cdf(a - ustar) - K * cdf(-ustar))

    return value


def scaled_rqmc(spec: dict, m: int = 20, replicates: int = 16, seed: int = 31) -> tuple[float, float]:
    theta, a = scaling_direction(spec)
    f = scaled_value(spec, theta, a, pca_complement(spec, theta))
    return rqmc_estimate(f, spec['d'] - 1, m, replicates, seed)


# ---------------------------------------------------------------- route 1b: a monotone direction carrying more variance

def monotone_direction(spec: dict) -> np.ndarray:
    """Unit direction with weights m, m-1, ..., 1 on the asset increments (the gradient of the average at zero noise);
    every price on the path increases along it, at its own rate."""
    d, m = spec['d'], spec['m']
    theta = np.zeros(d)
    w = np.arange(m, 0, -1, dtype=float)
    start = m if spec['family'] == 'heston_asian' else 0
    theta[start:start + m] = w / np.linalg.norm(w)
    return theta


def _exponents(spec: dict, y0: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """(n, m): d log S_j / du along theta at the points y0 (theta moves only the asset increments)."""
    m, dt = spec['m'], spec['T'] / spec['m']
    if spec['family'] == 'heston_asian':
        vp = heston_paths(spec, y0)[1]
        return np.cumsum(math.sqrt(1.0 - spec['rho'] ** 2) * np.sqrt(vp * dt) * theta[m:2 * m], axis=1)
    return np.tile(spec['sigma'][0] * math.sqrt(dt) * np.cumsum(theta[:m]), (len(y0), 1))


def _sum_exp_root(c: np.ndarray, a: np.ndarray, K: float) -> np.ndarray:
    """Per row, u with sum_j c_j e^{a_j u} = K for c > 0 and a >= 0 (not all zero): safeguarded Newton on the log;
    rows whose root lies outside [-60, 60] get the nearer end."""
    cap, lc = 60.0, np.log(c)

    def h(u):
        e = lc + a * u[:, None]
        mx = e.max(axis=1)
        w = np.exp(e - mx[:, None])
        s = w.sum(axis=1)
        return np.log(s) + mx - math.log(K), (w * a).sum(axis=1) / s

    lo, hi = np.full(len(c), -cap), np.full(len(c), cap)
    below, above = h(lo)[0] >= 0, h(hi)[0] <= 0
    u = np.zeros(len(c))
    for _ in range(200):
        g, dg = h(u)
        lo, hi = np.where(g < 0, u, lo), np.where(g > 0, u, hi)
        step = u - g / np.where(dg > 0, dg, 1.0)
        u_new = np.where((step <= lo) | (step >= hi) | (dg <= 0), 0.5 * (lo + hi), step)
        done = np.max(np.abs(u_new - u)) < 1e-13
        u = u_new
        if done:
            break
    return np.where(below, -cap, np.where(above, cap, u))


def monotone_value(spec: dict, theta: np.ndarray, basis: np.ndarray):
    """v -> E over u of the payoff at y = basis v + u theta, where S_j(u) = S_j e^{a_j u} with a_j >= 0: barrier
    conditions and the average become one threshold each, and every term integrates in closed form."""
    fam, K, m = spec['family'], spec['K'], spec['m']
    disc = math.exp(-spec['r'] * spec['T'])

    def value(v):
        y0 = v @ basis.T
        s = prices(spec, y0)[:, :, 0]
        a = _exponents(spec, y0, theta)
        if fam in ('barrier_do_call', 'barrier_dko_call'):
            c, e = s[:, -1:], a[:, -1:]
            ustar = np.log(K / s[:, -1]) / a[:, -1]
        else:
            c, e = s / m, a
            ustar = _sum_exp_root(c, e, K)
        if fam.startswith('barrier'):
            ustar = np.maximum(ustar, np.max(np.log(spec['B_low'] / s) / a, axis=1))
        if fam == 'digital_asian':
            return disc * CASH * cdf(-ustar)
        upper = np.full(len(v), np.inf)
        if fam == 'barrier_dko_call':
            upper = np.maximum(np.min(np.log(spec['B_high'] / s) / a, axis=1), ustar)
        mass = cdf(upper[:, None] - e) - cdf(ustar[:, None] - e)
        return disc * ((c * np.exp(0.5 * e * e) * mass).sum(axis=1) - K * (cdf(upper) - cdf(ustar)))

    return value


def monotone_rqmc(spec: dict, m: int = 20, replicates: int = 16, seed: int = 53) -> tuple[float, float]:
    theta = monotone_direction(spec)
    return rqmc_estimate(monotone_value(spec, theta, pca_complement(spec, theta)), spec['d'] - 1, m, replicates, seed)


# ---------------------------------------------------------------- route 2

def bvn_cdf_vec(h: np.ndarray, k: np.ndarray, rho: float) -> np.ndarray:
    """P(X <= h, Y <= k) for a standard bivariate normal with correlation rho, vectorized (Owen's T)."""
    eps = 1e-13
    h = np.where(np.abs(h) > eps, h, np.where(h >= 0, eps, -eps))
    k = np.where(np.abs(k) > eps, k, np.where(k >= 0, eps, -eps))
    s = math.sqrt(1.0 - rho * rho)
    beta = np.where(h * k > 0, 0.0, 0.5)
    return (0.5 * special.ndtr(h) + 0.5 * special.ndtr(k) - special.owens_t(h, (k - rho * h) / (h * s))
            - special.owens_t(k, (h - rho * k) / (k * s)) - beta)


def stulz_rqmc(spec: dict, m: int = 20, replicates: int = 16, seed: int = 37) -> tuple[float, float]:
    """Rainbow options: condition on the first increments of both assets. Given the rest, A_l = Atilde_l e^{g_l} with
    (g_1, g_2) bivariate normal (variances sigma_l^2 dt, correlation rho_12): Stulz's formula with zero rate, maturity
    dt and spots Atilde_l e^{sigma_l^2 dt / 2}. The other coordinates are PCA-ordered by their effect on the paths."""
    L, mm, d = spec['L'], spec['m'], spec['d']
    dt, K = spec['T'] / mm, spec['K']
    sig = np.asarray(spec['sigma'])
    rho = float(spec['corr'][0][1])
    tau_s = sig * math.sqrt(dt)
    disc = math.exp(-spec['r'] * spec['T'])
    sv = math.sqrt(tau_s[0] ** 2 + tau_s[1] ** 2 - 2 * rho * tau_s[0] * tau_s[1])
    rho1, rho2 = (rho * tau_s[1] - tau_s[0]) / sv, (rho * tau_s[0] - tau_s[1]) / sv
    rest = np.linalg.svd(_linear_map(spec)[:, L:], full_matrices=False)[2].T

    def value(v):
        y = np.hstack([np.zeros((len(v), L)), v @ rest.T])
        avg = prices(spec, y).mean(axis=1)
        s1, s2 = avg[:, 0] * math.exp(0.5 * tau_s[0] ** 2), avg[:, 1] * math.exp(0.5 * tau_s[1] ** 2)
        g1 = (np.log(s1 / K) - 0.5 * tau_s[0] ** 2) / tau_s[0]
        g2 = (np.log(s2 / K) - 0.5 * tau_s[1] ** 2) / tau_s[1]
        t1 = bvn_cdf_vec(g1 + tau_s[0], (np.log(s2 / s1) - 0.5 * sv * sv) / sv, rho1)
        t2 = bvn_cdf_vec(g2 + tau_s[1], (np.log(s1 / s2) - 0.5 * sv * sv) / sv, rho2)
        cmin = s1 * t1 + s2 * t2 - K * bvn_cdf_vec(g1, g2, rho)
        if spec['family'].endswith('min'):
            return disc * cmin
        c1 = s1 * cdf(g1 + tau_s[0]) - K * cdf(g1)
        c2 = s2 * cdf(g2 + tau_s[1]) - K * cdf(g2)
        return disc * (c1 + c2 - cmin)

    return rqmc_estimate(value, d - L, m, replicates, seed)


def last_step_rqmc(spec: dict, m: int = 20, replicates: int = 16, seed: int = 41) -> tuple[float, float]:
    """barrier_asian_do: condition on the last increment; only S_m moves, S_m = Stilde_m e^{a u}, a = sigma sqrt(dt)."""
    mm, K = spec['m'], spec['K']
    a = spec['sigma'][0] * math.sqrt(spec['T'] / mm)
    disc = math.exp(-spec['r'] * spec['T'])
    ea = math.exp(0.5 * a * a)

    def value(v):
        p = prices(spec, np.hstack([v, np.zeros((len(v), 1))]))[:, :, 0]
        alive = p[:, :-1].min(axis=1) > spec['B_low']
        c0, c1 = p[:, :-1].sum(axis=1) / mm, p[:, -1] / mm
        ulow = np.log(spec['B_low'] / p[:, -1]) / a
        gap = K - c0
        ukey = np.where(gap > 0, np.log(np.maximum(gap, 1e-300) / c1) / a, -np.inf)
        ustar = np.maximum(ulow, ukey)
        val = (c0 - K) * cdf(-ustar) + c1 * ea * cdf(a - ustar)
        return disc * np.where(alive, val, 0.0)

    return rqmc_estimate(value, spec['d'] - 1, m, replicates, seed)


def second_shock_rqmc(spec: dict, m: int = 20, replicates: int = 16, seed: int = 43) -> tuple[float, float]:
    """heston_asian: condition on z_s[1]. Prices from date 2 on scale by e^{b u}, b = rhohat sqrt(V_1^+ dt), so
    A(u) = c0 + c1 e^{b u} with c0 = S_1 / m."""
    mm, K = spec['m'], spec['K']
    dt = spec['T'] / mm
    rhohat = math.sqrt(1.0 - spec['rho'] ** 2)
    disc = math.exp(-spec['r'] * spec['T'])
    d = spec['d']
    keep = [i for i in range(d) if i != mm + 1]

    def value(v):
        y = np.zeros((len(v), d))
        y[:, keep] = v
        s = prices(spec, y)[:, :, 0]
        v1 = spec['v0'] + spec['kappa'] * (spec['theta'] - spec['v0']) * dt + spec['xi'] * math.sqrt(
            spec['v0'] * dt) * y[:, 0]
        b = rhohat * np.sqrt(np.maximum(v1, 0.0) * dt)
        c0, c1 = s[:, 0] / mm, s[:, 1:].sum(axis=1) / mm
        gap = K - c0
        bb = np.maximum(b, 1e-300)
        ustar = np.where(gap > 0, np.log(np.maximum(gap, 1e-300) / c1) / bb, -np.inf)
        val = (c0 - K) * cdf(-ustar) + c1 * np.exp(0.5 * b * b) * cdf(b - ustar)
        flat = np.maximum(c0 + c1 - K, 0.0)
        return disc * np.where(b > 1e-12, val, flat)

    return rqmc_estimate(value, d - 1, m, replicates, seed)


def digital_gradient_rqmc(spec: dict, m: int = 20, replicates: int = 16, seed: int = 47) -> tuple[float, float]:
    """digital_asian: preintegrate along the gradient direction of the average (a sum of lognormals): P(A > K) along
    the line is Phi(-u*) with u* the root of A(u) = K (options._root)."""
    tmp = dict(spec, family='asian_call')
    pos, neg = options.lognormal_terms(tmp)
    theta = options.directions(tmp, pos, neg)[0][1]
    basis = options._ordered_complement(tmp, theta, pos, neg)
    bp, _ = options._drivers(pos, neg)
    cp = np.array([math.log(cw) + mu for cw, mu, _ in pos])
    load, beta = basis.T @ bp.T, bp @ theta
    disc = math.exp(-spec['r'] * spec['T'])
    empty = np.zeros((0,))

    def value(v):
        alpha = cp + v @ load
        u = options._root(alpha, beta, np.zeros((len(v), 0)), empty, spec['K'])
        return disc * CASH * cdf(-u)

    return rqmc_estimate(value, spec['d'] - 1, m, replicates, seed)


def barrier_quad(spec: dict, width: float = 1.0, nodes: int = 16) -> float:
    """QUAD: price = e^{-rT} int V_1(x) p(x | x0) dx with V_j = 1{alive} int V_{j+1}(x') p(x' | x) dx' and Gaussian
    log-price transitions; Gauss-Legendre panels of width `width` x sigma sqrt(dt) on the alive interval, with log K as
    a panel edge."""
    m, sig, r, T, K = spec['m'], spec['sigma'][0], spec['r'], spec['T'], spec['K']
    dt = T / m
    s, mu = sig * math.sqrt(dt), (r - 0.5 * sig * sig) * dt
    x0 = math.log(spec['s0'][0])
    lo = math.log(spec['B_low'])
    hi = math.log(spec['B_high']) if 'B_high' in spec else max(x0, math.log(K)) + abs(r - 0.5 * sig * sig) * T + \
        14.0 * sig * math.sqrt(T)
    cuts = sorted({lo, hi} | ({math.log(K)} if lo < math.log(K) < hi else set()))
    xi, wi = np.polynomial.legendre.leggauss(nodes)
    xs, ws = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        n_pan = max(1, int(math.ceil((b - a) / (width * s))))
        edges = np.linspace(a, b, n_pan + 1)
        for e0, e1 in zip(edges[:-1], edges[1:]):
            xs.append(0.5 * (e0 + e1) + 0.5 * (e1 - e0) * xi)
            ws.append(0.5 * (e1 - e0) * wi)
    x, w = np.concatenate(xs), np.concatenate(ws)
    kern = lambda xa, xb: np.exp(-0.5 * ((xb - xa - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    val = np.maximum(np.exp(x) - K, 0.0)
    mat = kern(x[:, None], x[None, :]) * w[None, :]
    for _ in range(m - 1):
        val = mat @ val
    return math.exp(-r * T) * float((kern(x0, x) * w) @ val)


def check_line(spec: dict, n: int = 6, seed: int = 0) -> float:
    """Max absolute difference between the closed-form line expectation of route 1 and adaptive quadrature of the
    payoff along the same lines, split where the payoff switches between zero and non-zero (found by a fine scan of
    the payoff and bisection, independent of the thresholds the closed form uses)."""
    from scipy import integrate
    from gauss import pdf
    if spec['family'].startswith('rainbow') or spec['family'] == 'digital_asian':
        theta, a = scaling_direction(spec)
        basis = pca_complement(spec, theta)
        f = scaled_value(spec, theta, a, basis)
    else:
        theta = monotone_direction(spec)
        basis = pca_complement(spec, theta)
        f = monotone_value(spec, theta, basis)
    v = np.random.default_rng(seed).standard_normal((n, spec['d'] - 1))
    closed = f(v)
    grid = np.linspace(-12.0, 12.0, 100001)
    worst = 0.0
    for j in range(n):
        base = basis @ v[j]
        pay = lambda u: payoff(spec, base[None, :] + np.atleast_1d(u)[:, None] * theta)
        pos = pay(grid) > 0
        cuts = []
        for i in np.nonzero(pos[:-1] != pos[1:])[0]:
            lo, hi = grid[i], grid[i + 1]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if (pay(mid)[0] > 0) == pos[i] else (lo, mid)
            cuts.append(0.5 * (lo + hi))
        edges = [-12.0] + cuts + [12.0]
        q = sum(integrate.quad(lambda u: float(pay(u)[0] * pdf(u)), lo, hi, epsabs=1e-15, epsrel=1e-13, limit=500)[0]
                for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo)
        worst = max(worst, abs(q - closed[j]))
    return worst


# ---------------------------------------------------------------- reference dispatch

ROUTES = {
    'digital_asian': ((scaled_rqmc, 'uniform-scaling preintegration'),
                      (digital_gradient_rqmc, 'preintegration along the gradient direction')),
    'rainbow_asian_max': ((scaled_rqmc, 'uniform-scaling preintegration (parallel to the kink A_1 = A_2)'),
                          (stulz_rqmc, 'conditioning on the first increments (Stulz)')),
    'rainbow_asian_min': ((scaled_rqmc, 'uniform-scaling preintegration (parallel to the kink A_1 = A_2)'),
                          (stulz_rqmc, 'conditioning on the first increments (Stulz)')),
    'barrier_do_call': ((monotone_rqmc, 'monotone-direction preintegration'),),
    'barrier_dko_call': ((monotone_rqmc, 'monotone-direction preintegration'),),
    'barrier_asian_do': ((monotone_rqmc, 'monotone-direction preintegration'),
                         (scaled_rqmc, 'uniform-scaling preintegration'),
                         (last_step_rqmc, 'conditioning on the last increment')),
    'heston_asian': ((monotone_rqmc, 'monotone-direction preintegration'),
                     (scaled_rqmc, 'uniform-scaling preintegration'),
                     (second_shock_rqmc, 'conditioning on the second asset shock')),
}


def references(spec: dict, rqmc_m: int = 20, replicates: int = 16) -> list[dict]:
    route = options._route
    fam = spec['family']
    out = []
    if fam in ('barrier_do_call', 'barrier_dko_call'):
        fine, coarse = barrier_quad(spec, 0.5, 20), barrier_quad(spec, 1.0, 16)
        out.append(route('recursive Gauss-Legendre quadrature of the transition kernel (QUAD)', fine,
                         max(abs(fine - coarse), 1e-13 * abs(fine))))
    rq = []
    for k, (fn, name) in enumerate(ROUTES[fam]):
        val, se = fn(spec, rqmc_m, replicates, seed=31 + 6 * k)
        rq.append(route(f'{name} + RQMC 2^{rqmc_m} x {replicates}', val, se=max(se, 1e-300)))
    if not out:
        w = np.array([r['se'] ** -2 for r in rq])
        comb = float(sum(wi * r['value'] for wi, r in zip(w, rq)) / w.sum())
        out.append(route(f'inverse-variance mean of the {len(rq)} RQMC routes', comb, se=float(w.sum() ** -0.5),
                         derived=True))
    return out + rq
