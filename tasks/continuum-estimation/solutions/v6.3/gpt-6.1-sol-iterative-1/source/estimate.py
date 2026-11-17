#!/usr/bin/env python3
"""Continuum extrapolation by integrating regularized power-law fits."""

import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Prior:
    p_lo: float = 0.6
    p_hi: float = 2.8
    d_lo: float = 0.4
    d_hi: float = 2.2
    a_lo: float = -1.6
    a_hi: float = 1.6
    b_lo: float = -0.9
    b_hi: float = -0.1
    c_lo: float = -0.4
    c_hi: float = -0.05
    gap: float = 1.5


class PowerFit:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.lx = np.log(x)
        self.dl = np.diff(self.lx)
        self.dy = np.diff(y) / self.dl
        if np.any(self.dl <= 0) or not np.all(np.isfinite(self.dy)):
            raise ValueError("unresolved level differences")

    def matrix(self, p, d, gap, derivative=False):
        p, d = np.broadcast_arrays(p, d)
        e = np.stack([p, p + d, p + d + gap], axis=-1)
        left = np.exp(e[..., None, :] * self.lx[:-1, None])
        a = left * np.expm1(e[..., None, :] * self.dl[:, None]) / self.dl[:, None]
        b0 = np.exp(e * self.lx[0])
        if derivative:
            ap = a * self.lx[:-1, None] + np.exp(e[..., None, :] * self.lx[1:, None])
            return a, ap, b0
        return a, b0

    def residual(self, p, d, gap, c=None):
        a, _ = self.matrix(p, d, gap)
        if len(self.x) == 4:
            f0 = np.linalg.det(np.stack(
                [a[..., 0], a[..., 1], np.broadcast_to(self.dy, a.shape[:-1])],
                axis=-1,
            ))
            return f0 - c * np.linalg.det(a)
        aug = np.concatenate(
            [a, np.broadcast_to(self.dy[..., None], a.shape[:-1] + (1,))],
            axis=-1,
        )
        return np.linalg.det(aug)

    def integrate(self, prior, nd=320, nc=72, np_scan=96):
        ds = prior.d_lo + (np.arange(nd) + 0.5) * (prior.d_hi - prior.d_lo) / nd
        ps = np.linspace(prior.p_lo, prior.p_hi, np_scan)
        a, _ = self.matrix(ps[None, :], ds[:, None], prior.gap)
        if len(self.x) == 4:
            cs = prior.c_lo + (np.arange(nc) + 0.5) * (prior.c_hi - prior.c_lo) / nc
            f0 = np.linalg.det(np.stack(
                [a[..., 0], a[..., 1], np.broadcast_to(self.dy, a.shape[:-1])],
                axis=-1,
            ))
            f1 = np.linalg.det(a)
            fs = f0[:, None, :] - cs[None, :, None] * f1[:, None, :]
            crosses = (
                (np.signbit(fs[..., 1:]) != np.signbit(fs[..., :-1]))
                & np.isfinite(fs[..., 1:]) & np.isfinite(fs[..., :-1])
            )
            di, ci, pi = np.where(crosses)
            d = ds[di]
            c = cs[ci]
            flo = fs[di, ci, pi]
        else:
            aug = np.concatenate(
                [a, np.broadcast_to(self.dy[..., None], a.shape[:-1] + (1,))],
                axis=-1,
            )
            fs = np.linalg.det(aug)
            crosses = (
                (np.signbit(fs[:, 1:]) != np.signbit(fs[:, :-1]))
                & np.isfinite(fs[:, 1:]) & np.isfinite(fs[:, :-1])
            )
            di, pi = np.where(crosses)
            d = ds[di]
            c = None
            flo = fs[di, pi]
        if not len(pi) or len(pi) > 200000:
            return None
        lo, hi = ps[pi].copy(), ps[pi + 1].copy()
        # Batch root finding resolves thin coefficient-feasible regions without
        # requiring a finely spaced convergence-order grid.
        for _ in range(38):
            mid = (lo + hi) * 0.5
            fm = self.residual(mid, d, prior.gap, c)
            same = np.signbit(fm) == np.signbit(flo)
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
            flo = np.where(same, fm, flo)
        p = (lo + hi) * 0.5
        a, ap, b0 = self.matrix(p, d, prior.gap, derivative=True)
        if len(self.x) == 4:
            coefs = np.linalg.solve(a, np.broadcast_to(self.dy, a.shape[:-1]))
            jac = np.stack(
                [a[..., 0], a[..., 1], np.sum(ap * coefs[:, None, :], axis=-1)],
                axis=-1,
            )
        else:
            choices = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            dets = np.stack([np.abs(np.linalg.det(a[:, rows])) for rows in choices], axis=-1)
            pick = np.argmax(dets, axis=-1)
            coefs = np.empty((len(p), 3))
            for j, rows in enumerate(choices):
                ids = np.flatnonzero(pick == j)
                if len(ids):
                    coefs[ids] = np.linalg.solve(
                        a[ids][:, rows], np.broadcast_to(self.dy[rows], (len(ids), 3))
                    )
            jac = np.concatenate(
                [a, np.sum(ap * coefs[:, None, :], axis=-1)[..., None]], axis=-1
            )
        _, logdet = np.linalg.slogdet(jac)
        limits = self.y[0] - np.sum(coefs * b0, axis=-1)
        lows = np.array([prior.a_lo, prior.b_lo, prior.c_lo])
        highs = np.array([prior.a_hi, prior.b_hi, prior.c_hi])
        tol = 2e-7
        ok = (
            np.isfinite(limits) & np.isfinite(logdet)
            & np.all(np.isfinite(coefs), axis=-1)
            & np.all(coefs >= lows - tol, axis=-1)
            & np.all(coefs <= highs + tol, axis=-1)
        )
        if not np.any(ok):
            return None
        z = limits[ok]
        logw = -logdet[ok]
        logw -= np.max(logw)
        w = np.exp(logw) / np.maximum(1.0, np.abs(z))
        order = np.argsort(z)
        z, w = z[order], w[order]
        cum = np.cumsum(w)
        estimate = z[np.searchsorted(cum, 0.5 * cum[-1])]
        return float(estimate)


def exact_power(x, y, scale):
    """Recognize an exactly supported single-power trend before regularizing."""
    lx = np.log(x)
    if lx[-1] - lx[0] < 0.15 or not math.isfinite(scale):
        return None
    i = len(x) // 2
    dy1, dy2 = y[i] - y[0], y[-1] - y[i]
    if dy1 == 0 or dy2 == 0 or np.signbit(dy1) != np.signbit(dy2):
        return None
    if not math.isfinite(dy1) or not math.isfinite(dy2):
        return None
    dl1, dl2 = lx[i] - lx[0], lx[-1] - lx[i]
    log_ratio = math.log(abs(dy1)) - math.log(abs(dy2))

    def log_expm1(v):
        if v < 50:
            return math.log(math.expm1(v))
        return v + math.log1p(-math.exp(-v))

    def residual(p):
        return log_ratio + p * dl1 - log_expm1(p * dl1) + log_expm1(p * dl2)

    lo, hi = 1e-4, 16.0
    if residual(lo) > 0 or residual(hi) < 0:
        return None
    for _ in range(52):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0:
            lo = mid
        else:
            hi = mid
    p = 0.5 * (lo + hi)
    power = np.exp(p * (lx - lx[-1]))
    amplitude = (y[-1] - y[0]) / (1 - power[0])
    error = amplitude * (power - power[0]) - (y - y[0])
    tolerance = max(3e-12 * scale, 2e-14 * np.max(np.abs(y)))
    limit = y[0] - amplitude * power[0]
    if np.max(np.abs(error)) <= tolerance and math.isfinite(limit):
        return float(limit)
    return None


def fallback(x, y):
    if len(x) < 3 or np.ptp(y) <= 1e-14 * max(1.0, np.max(np.abs(y))):
        return float(y[0])
    from scipy.optimize import minimize_scalar

    scale = np.ptp(y)
    if not math.isfinite(scale) or scale <= 0:
        return float(y[0])
    center = y[0]
    yy = (y - center) / scale
    if not np.all(np.isfinite(yy)):
        return float(y[0])
    xx = np.log(x) - np.log(x[-1])

    def fit(p, count, terms):
        exponents = [0, p] if terms == 1 else [0, p, p + 1]
        a = np.exp(xx[:count, None] * np.array(exponents)[None, :])
        coef = np.linalg.lstsq(a, yy[:count], rcond=1e-12)[0]
        err = np.linalg.norm(a @ coef - yy[:count])
        return err, center + scale * coef[0]

    candidates = []
    for count, terms in [(3, 1), (len(x), 1), (len(x), 2)]:
        grid = np.linspace(0.2, 6.0, 80)
        errs = np.array([fit(p, count, terms)[0] for p in grid])
        local = [
            i for i in range(1, len(grid) - 1)
            if errs[i] <= errs[i - 1] and errs[i] <= errs[i + 1]
        ]
        local.append(int(np.argmin(errs)))
        best = None
        for i in local:
            lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
            if hi == lo:
                continue
            r = minimize_scalar(
                lambda p: fit(p, count, terms)[0],
                bounds=(lo, hi), method="bounded", options={"xatol": 1e-10},
            )
            err, estimate = fit(r.x, count, terms)
            if math.isfinite(estimate) and (best is None or err < best[0]):
                best = (err, estimate, r.x)
        if best is not None:
            candidates.append(best)
    if not candidates:
        return float(y[0])
    if candidates[-1][0] < 1e-3:
        estimate = candidates[-1][1]
    else:
        estimate = candidates[0][1]
    return float(np.clip(estimate, y[0] - 4 * scale, y[0] + 4 * scale))


def estimate_limit(x, y):
    order = np.argsort(x)
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    nearest = float(y[0])
    with np.errstate(over="ignore", invalid="ignore"):
        spread = np.ptp(y)
    if spread <= 2e-14 * max(1.0, np.max(np.abs(y))):
        return nearest
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        signs = np.sign(np.diff(y))
        signs = signs[signs != 0]
        if len(signs) > 2 and np.count_nonzero(signs[1:] != signs[:-1]) >= 2:
            return float(0.5 * y[0] + 0.5 * y[1])
        try:
            exact = exact_power(x, y, spread)
            if exact is not None:
                return exact
        except (ArithmeticError, ValueError):
            pass
        try:
            model = PowerFit(x, y)
            result = model.integrate(Prior(), nd=16000 if len(x) == 5 else 320)
            if result is not None and math.isfinite(result):
                return result
            wide = Prior(
                p_lo=0.2, p_hi=4.0, d_lo=0.2, d_hi=3.0,
                a_lo=-4.0, a_hi=4.0, b_lo=-2.0, b_hi=2.0,
                c_lo=-1.0, c_hi=1.0,
            )
            result = model.integrate(wide, nd=800 if len(x) == 5 else 180, nc=56)
            if result is not None and math.isfinite(result):
                return result
        except (ArithmeticError, ValueError, np.linalg.LinAlgError):
            pass
        try:
            result = fallback(x, y)
            if math.isfinite(result):
                return result
        except (ArithmeticError, ValueError, np.linalg.LinAlgError):
            pass
    return nearest


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/inputs/input.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/workspace/result/output.json"
    with open(input_path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    value = estimate_limit(data["x"], data["y"])
    if not math.isfinite(value):
        value = float(data["y"][int(np.argmin(data["x"]))])
    value = max(-1e290, min(1e290, value))
    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump({"estimate": float(value)}, stream, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
