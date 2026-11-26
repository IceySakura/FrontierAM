"""Continuum inference from a small set of positive discretization levels."""

import json
import math
import os
import sys

import numpy as np
from scipy.special import logsumexp, roots_legendre


def quadrature(lo, hi, count):
    nodes, weights = roots_legendre(count)
    return lo + (hi - lo) * (nodes + 1.0) / 2.0, weights * (hi - lo) / 2.0


def samples(x, y, nq=2400, nc=160, delta=1.5):
    """Return rows (limit, p, q-p, a, b, c, log quadrature weight).

    Integrate q for five observations, or (q, c) for four. Projection away
    from the fixed basis leaves a scalar equation for the leading order p.
    """
    n = len(x)
    center = float(y[0])
    scale = max(float(np.ptp(y)), 1e-100)
    z = (y - center) / scale
    logx = np.log(x)
    qs, qw = quadrature(0.4, 7.0, nq)
    xq = np.exp(qs[:, None] * logx)
    xr = np.exp((qs[:, None] + delta) * logx)
    if n == 5:
        B = np.stack([np.ones_like(xq), xq, xr], axis=-1)
        Q, _ = np.linalg.qr(B, mode="complete")
        u, v = Q[:, :, -2], Q[:, :, -1]
        w = u * (v @ z)[:, None] - v * (u @ z)[:, None]
        q = qs
        logweight = np.log(qw)
        residual_y = np.broadcast_to(z, w.shape)
    else:
        cs, cw = quadrature(-0.65, 0.65, nc)
        B = np.stack([np.ones_like(xq), xq], axis=-1)
        Q, _ = np.linalg.qr(B, mode="complete")
        u, v = Q[:, :, -2], Q[:, :, -1]
        residual_y = z[None, None, :] - cs[None, :, None] * xr[:, None, :] / scale
        udoty = np.einsum("qi,qci->qc", u, residual_y)
        vdoty = np.einsum("qi,qci->qc", v, residual_y)
        w = (
            u[:, None, :] * vdoty[:, :, None]
            - v[:, None, :] * udoty[:, :, None]
        ).reshape(-1, n)
        residual_y = residual_y.reshape(-1, n)
        q = np.repeat(qs, nc)
        c = np.tile(cs, nq)
        logweight = (np.log(qw[:, None]) + np.log(cw[None, :])).ravel()

    lo = np.full(len(q), 0.20)
    hi = np.minimum(3.8, q - 0.15)

    def f(p):
        return np.sum(w * np.exp(p[:, None] * logx), axis=1)

    flo, fhi = f(lo), f(hi)
    keep = (hi > lo) & (flo * fhi < 0.0)
    if not np.any(keep):
        return np.empty((0, 7))
    w, q, logweight = w[keep], q[keep], logweight[keep]
    lo, hi, flo = lo[keep], hi[keep], flo[keep]
    residual_y = residual_y[keep]
    if n == 4:
        c = c[keep]
    for _ in range(47):
        mid = (lo + hi) / 2.0
        fm = f(mid)
        left = (flo * fm > 0.0)
        lo = np.where(left, mid, lo)
        flo = np.where(left, fm, flo)
        hi = np.where(left, hi, mid)
    p = (lo + hi) / 2.0
    xp = np.exp(p[:, None] * logx)
    xq = np.exp(q[:, None] * logx)
    xr = np.exp((q[:, None] + delta) * logx)
    if n == 5:
        A = np.stack([np.ones_like(xp), xp, xq, xr], axis=-1)
    else:
        A = np.stack([np.ones_like(xp), xp, xq], axis=-1)
    # Reduced QR avoids squaring the condition number of the power basis.
    Q, R = np.linalg.qr(A, mode="reduced")
    coeff = np.linalg.solve(R, np.einsum("qij,qi->qj", Q, residual_y))
    estimate = center + scale * coeff[:, 0]
    a = scale * coeff[:, 1]
    b = scale * coeff[:, 2]
    if n == 5:
        c = scale * coeff[:, 3]
    # Change of variables for the coefficients and the eliminated order p.
    J = np.concatenate([A, (a[:, None] * xp * logx)[:, :, None]], axis=-1)
    sign, ld = np.linalg.slogdet(J)
    logweight -= ld
    result = np.column_stack([estimate, p, q - p, a, b, c, logweight])
    return result[np.all(np.isfinite(result), axis=1) & (sign != 0)]


def logprior(s):
    _, p, gap, a, b, c, _ = s.T
    # Smooth tails retain support outside the principal asymptotic regime.
    out = -0.5 * (a / 1.2) ** 2
    out -= 0.5 * (np.maximum(0.45 - p, 0.0) / 0.12) ** 2
    out -= 0.5 * (np.maximum(p - 2.9, 0.0) / 0.25) ** 2
    out -= 0.5 * (np.maximum(0.35 - gap, 0.0) / 0.1) ** 2
    out -= 0.5 * (np.maximum(gap - 2.4, 0.0) / 0.3) ** 2
    out -= 0.5 * (np.maximum(abs(a) - 1.5, 0.0) / 0.15) ** 2
    out -= 0.5 * (np.maximum(abs(b) - 0.9, 0.0) / 0.12) ** 2
    out -= 0.5 * (np.maximum(abs(c) - 0.4, 0.0) / 0.05) ** 2
    out += np.log1p(0.8 * np.tanh(b / 0.1) * np.tanh(c / 0.05))
    return out


def posterior(s):
    if len(s) == 0:
        return None
    lp = logprior(s)
    if np.max(lp) < -80.0:
        return None
    lw = s[:, -1] + lp
    lw -= logsumexp(lw)
    order = np.argsort(s[:, 0])
    z = s[order, 0]
    # The inverse scale factor matches the normalized absolute-error loss.
    weight = np.exp(lw[order]) / np.maximum(1.0, np.abs(z))
    cdf = (np.cumsum(weight) - 0.5 * weight) / np.sum(weight)
    return float(np.interp(0.5, cdf, z))


def fallback(x, y):
    """A bounded local power-law fit for incompatible or degenerate inputs."""
    from scipy.optimize import minimize_scalar

    x = x / x[-1]
    scale = max(float(np.ptp(y)), 1e-100)
    z = (y - y[0]) / scale

    def fit(p):
        A = np.column_stack([np.ones(len(x)), x**p, x**(p + 1)])
        c = np.linalg.lstsq(A, z, rcond=1e-12)[0]
        r = A @ c - z
        return float(r @ r), float(y[0] + scale * c[0])

    grid = np.linspace(0.35, 5.0, 80)
    scores = [fit(p)[0] for p in grid]
    j = int(np.argmin(scores))
    opt = minimize_scalar(
        lambda p: fit(p)[0],
        bounds=(grid[max(0, j - 1)], grid[min(len(grid) - 1, j + 1)]),
        method="bounded",
    )
    value = fit(opt.x)[1]
    bound = max(2 * float(np.ptp(y)), 1e-12 * max(1.0, abs(float(y[0]))))
    return float(np.clip(value, y[0] - bound, y[0] + bound))


def exact_power(x, y):
    """Recognize an overdetermined constant-plus-single-power model."""
    from scipy.optimize import brentq

    z = (y - y[0]) / np.ptp(y)
    logx = np.log(x) - np.log(x[-1])
    if logx[0] >= 0.0:
        return None
    # Endpoints fix the offset and amplitude, leaving only the exponent.
    k = 1 + int(np.argmax(np.minimum(-logx[1:-1], logx[1:-1] - logx[0])))
    target = z[k] / z[-1] if abs(z[-1]) > 1e-14 else float("nan")
    if not (0.0 < target < 1.0):
        return None

    def fraction(p):
        numerator = np.exp(p * logx[k]) * -np.expm1(p * (logx[0] - logx[k]))
        return numerator / -np.expm1(p * logx[0])

    if not (fraction(0.0001) > target > fraction(12.0)):
        return None
    p = brentq(lambda v: fraction(v) - target, 0.0001, 12.0, xtol=2e-14)
    xp = np.exp(p * logx)
    denominator = -np.expm1(p * logx[0])
    curve = xp * -np.expm1(p * (logx[0] - logx)) / denominator * z[-1]
    if np.max(abs(curve - z)) < 2e-11:
        return float(y[0] - (y[-1] - y[0]) * xp[0] / denominator)
    return None


def estimate(x, y):
    order = np.argsort(x)
    x, y = np.asarray(x, dtype=float)[order], np.asarray(y, dtype=float)[order]
    span = float(np.ptp(y))
    if not math.isfinite(span):
        return float(y[0])
    if span <= 3e-15 * max(1.0, float(np.max(abs(y)))):
        return float(y[0])
    if x[-1] > 2.0 or x[-1] < 0.01:
        x = x / x[-1] * 0.5
    if x[0] <= 0.0:
        return float(y[0])
    exact = exact_power(x, y)
    if exact is not None:
        return exact
    s = samples(x, y)
    result = posterior(s)
    if result is None or not math.isfinite(result):
        return fallback(x, y)
    return result


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else "/workspace/inputs/input.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "/workspace/result/output.json"
    with open(inp, "r", encoding="utf-8") as f:
        data = json.load(f)
    fallback_value = float(data["y"][int(np.argmin(data["x"]))])
    try:
        with np.errstate(all="ignore"):
            value = float(estimate(data["x"], data["y"]))
        if not math.isfinite(value):
            value = fallback_value
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        value = fallback_value
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"estimate": value}, f, allow_nan=False, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
