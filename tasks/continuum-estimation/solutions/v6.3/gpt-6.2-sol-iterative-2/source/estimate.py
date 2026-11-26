"""Condition positive-order error models on a short sequence of exact observations."""

import json
import math
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss


TAIL_GAP = 1.5
P_MIN, P_MAX = 0.41, 3.45
D_MIN, D_MAX = 0.18, 3.05
C_LIMIT = 1.04
NODES, WEIGHTS = leggauss(48)


def dot(a, b):
    return np.sum(a * b, axis=-1)


def norm(a):
    return np.sqrt(dot(a, a))


class Observations:
    def __init__(self, x, y):
        with np.errstate(over="ignore"):
            relative = np.diff(x) / x[:-1]
        self.step = np.where(relative < 0.5, np.log1p(relative), np.diff(np.log(x)))
        self.right = np.r_[-np.cumsum(self.step[:0:-1])[::-1], 0.0]
        self.y = y
        self.diff = np.diff(y) / self.step
        self.log_max = math.log(x[-1])

    def column(self, order):
        # First divided differences of (x / max(x))**order in log(x),
        # normalized by order to keep their sizes comparable.
        s = np.asarray(order)[..., None]
        v = s * self.step
        return np.exp(s * self.right) * (-np.expm1(-v) / v)

    def derivative(self, p, column):
        s = p[:, None]
        v = s * self.step
        return column * (self.right + self.step * np.exp(-v) / (-np.expm1(-v)) - 1.0 / s)

    def intercept_and_coefficients(self, p, q, a, b, c):
        r = q + TAIL_GAP
        A, B, C = a / p, b / q, c / r
        t = self.y[-1] - A - B - C
        a = A * np.exp(-p * self.log_max)
        b = B * np.exp(-q * self.log_max)
        c = C * np.exp(-r * self.log_max)
        log_scale = np.log(p * q * r) + (p + q + r) * self.log_max
        return t, a, b, c, log_scale


def exponent_grid(size):
    step = (P_MAX + D_MAX - P_MIN - D_MIN) / size
    q = P_MIN + D_MIN + step * (np.arange(size) + 0.5)
    p_lo = np.maximum(P_MIN, q - D_MAX)
    p_hi = np.minimum(P_MAX, q - D_MIN)
    return q, p_lo, p_hi


def four_point_models(obs, size):
    q, lo, hi = exponent_grid(size)
    r = q + TAIL_GAP
    u = obs.column(q)
    v = obs.column(r)
    n = np.cross(u, v)
    h = dot(n, n)
    c_dual = np.cross(n, u) / h[:, None]
    b_dual = np.cross(v, n) / h[:, None]
    a_y = dot(n, obs.diff)
    b_y = dot(b_dual, obs.diff)
    c_y = dot(c_dual, obs.diff)
    c_to_physical = np.exp(-r * obs.log_max) / r

    def c_at(p, k):
        f = obs.column(p)
        a = a_y[k] / dot(n[k], f)
        return (c_y[k] - a * dot(c_dual[k], f)) * c_to_physical[k]

    k = np.arange(len(q))
    clo, chi = c_at(lo, k), c_at(hi, k)
    k = k[(np.minimum(clo, chi) < C_LIMIT) & (np.maximum(clo, chi) > -C_LIMIT)]
    if not len(k):
        return np.empty((0, 7))
    # For fixed q, the last coefficient is a monotone determinant ratio
    # in p. Restrict integration to its support before applying quadrature.
    p_left, p_right = lo[k].copy(), hi[k].copy()
    increasing = clo[k] < chi[k]
    sign = np.where(increasing, 1.0, -1.0)
    lower = sign * clo[k] < -C_LIMIT
    upper = sign * chi[k] > C_LIMIT
    if np.any(lower) or np.any(upper):
        kl, ku = k[lower], k[upper]
        indices = np.r_[kl, ku]
        direction = np.r_[sign[lower], sign[upper]]
        bound = np.r_[np.full(len(kl), -C_LIMIT), np.full(len(ku), C_LIMIT)]
        left, right = lo[indices].copy(), hi[indices].copy()
        for _ in range(33):
            mid = 0.5 * (left + right)
            below = direction * c_at(mid, indices) < bound
            left = np.where(below, mid, left)
            right = np.where(below, right, mid)
        roots = 0.5 * (left + right)
        p_left[lower], p_right[upper] = roots[: len(kl)], roots[len(kl) :]
    half = 0.5 * (p_right - p_left)
    keep = half > 0.0
    k, half = k[keep], half[keep]
    p = (0.5 * (p_left[keep] + p_right[keep]))[:, None] + half[:, None] * NODES
    f = obs.column(p)
    det = dot(n[k, None, :], f)
    a = a_y[k, None] / det
    b = b_y[k, None] - a * dot(b_dual[k, None, :], f)
    c = c_y[k, None] - a * dot(c_dual[k, None, :], f)
    q = np.broadcast_to(q[k, None], p.shape)
    t, a, b, c, log_scale = obs.intercept_and_coefficients(p, q, a, b, c)
    log_mass = np.log(half[:, None] * WEIGHTS) - np.log(np.abs(det)) - log_scale
    return np.stack((t, a, b, c, p, q - p, log_mass), axis=-1).reshape(-1, 7)


def cross4(a, b, c):
    """The vector of signed maximal minors of a three-by-four matrix."""
    def minor(i, j, k):
        return (
            a[:, i] * (b[:, j] * c[:, k] - b[:, k] * c[:, j])
            + a[:, j] * (b[:, k] * c[:, i] - b[:, i] * c[:, k])
            + a[:, k] * (b[:, i] * c[:, j] - b[:, j] * c[:, i])
        )
    return np.column_stack((minor(1, 2, 3), -minor(0, 2, 3), minor(0, 1, 3), -minor(0, 1, 2)))


def five_point_models(obs, size):
    q, lo, hi = exponent_grid(size)
    u, v = obs.column(q), obs.column(q + TAIL_GAP)
    u_size = norm(u)
    U = u / u_size[:, None]
    v_on_u = dot(v, U)
    V = v - v_on_u[:, None] * U
    v_size = norm(V)
    V /= v_size[:, None]
    y_on_u, y_on_v = dot(U, obs.diff), dot(V, obs.diff)
    E = obs.diff - y_on_u[:, None] * U - y_on_v[:, None] * V
    # Remove projection error from the complement before finding its direction.
    E -= dot(E, U)[:, None] * U + dot(E, V)[:, None] * V
    e_size = norm(E)
    E /= e_size[:, None]
    normal = cross4(U, V, E)
    f_lo, f_hi = dot(normal, obs.column(lo)), dot(normal, obs.column(hi))
    k = np.flatnonzero(np.signbit(f_lo) != np.signbit(f_hi))
    if not len(k):
        return np.empty((0, 7))
    lo, hi, normal = lo[k], hi[k], normal[k]
    left_sign = np.signbit(f_lo[k])
    # The unnormalized determinant has roots at 0, q, and q+TAIL_GAP.
    # With five exponential terms, it has at most four real roots.
    for _ in range(38):
        mid = 0.5 * (lo + hi)
        left = np.signbit(dot(normal, obs.column(mid))) == left_sign
        lo = np.where(left, mid, lo)
        hi = np.where(left, hi, mid)
    p = 0.5 * (lo + hi)
    f = obs.column(p)
    a = e_size[k] / dot(f, E[k])
    c = (y_on_v[k] - a * dot(V[k], f)) / v_size[k]
    b = (y_on_u[k] - a * dot(U[k], f) - c * v_on_u[k]) / u_size[k]
    df = obs.derivative(p, f)
    jac = np.abs(dot(normal, df)) * u_size[k] * v_size[k] * e_size[k]
    t, a, b, c, log_scale = obs.intercept_and_coefficients(p, q[k], a, b, c)
    log_mass = -np.log(jac) - log_scale
    return np.column_stack((t, a, b, c, p, q[k] - p, log_mass))


def window(x, lo, hi, left, right=None):
    """Uniform-like prior with smooth edges and no hard coefficient cutoffs."""
    if right is None:
        right = left
    return -np.logaddexp(0.0, (lo - x) / left) - np.logaddexp(0.0, (x - hi) / right)


def prior_log_density(models):
    t, a, b, c, p, gap, _ = models.T
    p_prior = window(p, 0.628, 2.75, 0.042, 0.14)
    gap_prior = window(gap, 0.41, 2.3, 0.075, 0.19)
    gap_prior += np.log1p(1.8 * np.exp(-0.5 * ((gap - 0.58) / 0.23) ** 2))
    a_prior = window(np.abs(a), 0.18, 1.65, 0.095, 0.18)
    b_prior = window(np.abs(b), 0.0, 0.97, 0.003, 0.15)
    c_prior = window(np.abs(c), 0.0, 0.405, 0.003, 0.075)
    b_prior += np.log(0.16 + 0.68 / (1.0 + np.exp(np.clip(b / 0.022, -700.0, 700.0))))
    c_prior += np.log(0.11 + 0.78 / (1.0 + np.exp(np.clip(c / 0.015, -700.0, 700.0))))
    t_prior = -np.log1p((t / 4.7) ** 8)
    return p_prior + gap_prior + a_prior + b_prior + c_prior + t_prior


def median_limit(models):
    if not len(models):
        return math.nan, 0.0
    prior = prior_log_density(models)
    if not np.any(prior > -28.0):
        return math.nan, 0.0
    logs = prior + models[:, -1] - np.log(np.maximum(1.0, np.abs(models[:, 0])))
    valid = np.isfinite(logs)
    if not np.any(valid):
        return math.nan, 0.0
    t, logs = models[valid, 0], logs[valid]
    w = np.exp(logs - np.max(logs))
    ess = np.sum(w) ** 2 / dot(w, w)
    order = np.argsort(t)
    w = w[order]
    cumulative = np.cumsum(w) - 0.5 * w
    median = np.interp(0.5 * np.sum(w), cumulative, t[order])
    return float(median), float(ess)


def power_order(g0, g1, ratio, lo, hi):
    """Invert the ratio of two consecutive log-level secants."""
    for _ in range(48):
        p = 0.5 * (lo + hi)
        slope_ratio = p * g1 + math.log(-math.expm1(-p * g1) / g1)
        slope_ratio -= math.log(-math.expm1(-p * g0) / g0)
        if slope_ratio < ratio:
            lo = p
        else:
            hi = p
    return 0.5 * (lo + hi)


def single_power_limit(obs):
    """Recognize a resolved single power when it agrees at every level."""
    y = obs.y
    delta = float(y[-1] - y[0])
    scale = float(np.max(np.abs(y)))
    if abs(delta) < 3e-9 * scale or delta == 0.0:
        return None
    if not np.all(np.sign(obs.diff) == np.sign(delta)):
        return None
    j = 1 + int(np.argmin(np.abs((y[1:-1] - y[0]) / delta - 0.5)))
    g0, g1 = float(np.sum(obs.step[:j])), float(np.sum(obs.step[j:]))
    left = float((y[j] - y[0]) / g0)
    right = float((y[-1] - y[j]) / g1)
    if left == 0.0 or right == 0.0:
        return None
    ratio = math.log(abs(right)) - math.log(abs(left))
    p = power_order(g0, g1, ratio, 0.001, 20.0)
    if p * (g0 + g1) < 0.075:
        return None
    amplitude = delta / (-math.expm1(-p * (g0 + g1)))
    predicted = amplitude * np.expm1(p * obs.right)
    error = np.max(np.abs(predicted - (y[1:] - y[-1])))
    if error <= max(12.0 * np.finfo(float).eps * scale, 2e-12 * abs(delta)):
        return float(y[-1] - amplitude)
    return None


def local_fallback(obs):
    """Damped lowest-level Richardson step, or the finest value at a turn."""
    y = obs.y
    d = obs.diff[:2]
    if d[0] == 0 or d[1] == 0 or np.signbit(d[0]) != np.signbit(d[1]):
        return float(y[0])
    s = np.sign(obs.diff[obs.diff != 0])
    if np.count_nonzero(s[1:] != s[:-1]) > 1:
        return float(y[0])
    ratio = math.log(abs(d[1])) - math.log(abs(d[0]))
    p = power_order(*obs.step[:2], ratio, 0.65, 5.5)
    reach = p * obs.step[0]
    correction = 0.0 if reach > 700 else (y[1] - y[0]) / math.expm1(reach)
    cap = 2.2 * float(np.max(y) - np.min(y))
    return float(y[0] - np.clip(correction, -cap, cap))


def estimate(x, y):
    order = np.argsort(x)
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    if np.max(np.abs(y)) > 1e300:
        return 0.0
    if np.all(y == y[0]):
        return float(y[0])
    with np.errstate(all="ignore"):
        obs = Observations(x, y)
        if np.any(obs.step <= 0.0) or not np.all(np.isfinite(obs.diff)):
            return float(y[0])
        single = single_power_limit(obs)
        if single is not None:
            return single
        fallback = local_fallback(obs)
        if not (0.002 < x[-1] < 1.6 and 0.025 < np.sum(obs.step) < 35.0):
            return fallback
        if np.max(np.abs(y)) > 1e4:
            return fallback
        if len(y) == 4:
            models = four_point_models(obs, 1050)
            point, ess = median_limit(models)
            if ess < 32:
                point, ess = median_limit(four_point_models(obs, 4250))
        else:
            models = five_point_models(obs, 2100)
            point, ess = median_limit(models)
            if ess < 25:
                point, ess = median_limit(five_point_models(obs, 21000))
    if not math.isfinite(point) or abs(point - y[0]) > 4.5:
        return fallback
    return point


def main():
    inp = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/inputs/input.json")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "/workspace/result/output.json")
    values = json.loads(inp.read_text())
    result = estimate(values["x"], values["y"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"estimate": result}, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
