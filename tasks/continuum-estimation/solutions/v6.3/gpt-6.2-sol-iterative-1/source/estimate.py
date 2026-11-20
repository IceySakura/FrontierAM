"""Continuum extrapolation by integrating over positive-power expansions."""

import json
import math
from pathlib import Path
import sys

import numpy as np


P_BOUNDS = (0.20, 4.10)
G_BOUNDS = (0.12, 3.10)
THIRD_GAP = 1.5
COEFF_PANELS = (-2.0, -0.47, -0.085, 0.085, 0.47, 2.0)


def box(v, left, right, shoulder):
    """A normalized uniform density with Gaussian shoulders."""
    distance = np.maximum(np.maximum(left - v, v - right), 0.0)
    return np.exp(-0.5 * (distance / shoulder) ** 2) / (
        right - left + math.sqrt(2 * math.pi) * shoulder
    )


def heavy(v, width):
    """A normalized, heavy-tailed location prior."""
    return (2.0 / (math.pi * width)) / (1 + (v / width) ** 2) ** 2


def broad_shape_prior(p, g, a, b, c):
    pp = 0.96 * box(p, 0.55, 2.82, 0.095) + 0.04 * box(p, *P_BOUNDS, 0.20)
    pg = 0.95 * box(g, 0.36, 2.05, 0.12) + 0.05 * box(g, *G_BOUNDS, 0.18)
    pa = 0.96 * 0.5 * box(abs(a), 0.20, 1.55, 0.085) + 0.04 * heavy(a, 1.25)
    pb = (
        0.925 * box(b, -0.92, -0.09, 0.045)
        + 0.055 * box(-b, -0.92, -0.09, 0.045)
        + 0.020 * heavy(b, 0.95)
    )
    pc = (
        0.915 * box(c, -0.40, -0.045, 0.027)
        + 0.070 * box(-c, -0.40, -0.045, 0.027)
        + 0.015 * heavy(c, 0.45)
    )
    return pp * pg * pa * pb * pc


def sign_densities(v, left, right, shoulder, tail, width):
    h = tail * heavy(v, width)
    return (1 - tail) * box(-v, left, right, shoulder) + h, (
        (1 - tail) * box(v, left, right, shoulder) + h
    )


def prior(p, g, a, b, c, t):
    pp = 0.98 * box(p, 0.58, 2.77, 0.044) + 0.02 * box(p, *P_BOUNDS, 0.18)
    pg = 0.98 * box(g, 0.36, 2.06, 0.064) + 0.02 * box(g, *G_BOUNDS, 0.18)
    pa = 0.49 * box(abs(a), 0.25, 1.53, 0.041) + 0.02 * heavy(a, 1.25)
    bm, bp = sign_densities(b, 0.10, 0.90, 0.026, 0.018, 0.95)
    cm, cp = sign_densities(c, 0.047, 0.389, 0.017, 0.014, 0.45)
    # Prefer the subleading sign pattern suggested by the development examples,
    # but keep positive and mixed signs available for cancellation and rescaling.
    bc = bm * (0.965 * cm + 0.014 * cp) + bp * (0.011 * cm + 0.010 * cp)
    broad = broad_shape_prior(p, g, a, b, c)
    pt = 0.92 * box(t, -3.0, 3.0, 0.42) + 0.08 * heavy(t, 5.0)
    return (0.70 * pp * pg * pa * bc + 0.30 * broad) * pt


def gauss_panels(cuts, count):
    z, w = np.polynomial.legendre.leggauss(count)
    cuts = np.asarray(cuts)
    center = 0.5 * (cuts[1:] + cuts[:-1])
    half = 0.5 * np.diff(cuts)
    return (center[:, None] + half[:, None] * z).ravel(), (half[:, None] * w).ravel()


class PowerFit:
    """Eliminate the linear coefficients using logarithmic divided differences."""

    def __init__(self, s, y):
        self.s = s
        self.gaps = [s[k:] - s[:-k] for k in range(1, len(s))]
        # Preserve the differences between nearly equal observed values.
        v = np.asarray(y, dtype=np.longdouble)
        self.base = float(v[0])
        self.obs = [0.0]
        for gap in self.gaps:
            v = (v[1:] - v[:-1]) / gap
            self.obs.append(float(v[0]))
        self.n = len(s)

    def basis(self, exponents, derivative=False):
        k = np.asarray(exponents)[..., None]
        e = np.exp(k * self.s)
        d = e[..., :-1] * (np.expm1(k * self.gaps[0]) / self.gaps[0])
        out = [e[..., 0], d[..., 0]]
        if derivative:
            dd = self.s[:-1] * d + e[..., 1:]
            dout = [self.s[0] * e[..., 0], dd[..., 0]]
        for gap in self.gaps[1:]:
            d = (d[..., 1:] - d[..., :-1]) / gap
            out.append(d[..., 0])
            if derivative:
                dd = (dd[..., 1:] - dd[..., :-1]) / gap
                dout.append(dd[..., 0])
        e1 = out[1]
        scaled = [v / e1 for v in out]
        if derivative:
            return e1, scaled, [v / e1 for v in dout]
        return e1, scaled, None

    def equations(self, p, g, c_fixed=None, full=False):
        e, a, da = self.basis(p, full)
        f, b, db = self.basis(p + g, full)
        h, c, dc = self.basis(p + g + THIRD_GAP, full)
        y = self.obs
        u = b[2] - a[2]
        v = c[2] - a[2]
        m = (b[3] - a[3]) / u
        w = c[3] - a[3] - m * v

        def L3(z):
            return z[3] - a[3] * z[1] - m * (z[2] - a[2] * z[1])

        r = L3(y)
        if self.n == 4:
            # For a fixed third coefficient the residual is affine in it.
            if c_fixed is None and not full:
                return r, w * h
            cc = c_fixed * h
            linear_jac = e * f * u
            l = L3
            res = r - w * cc
        else:
            cc = r / w
            linear_jac = e * f * h * u * w
            j = (b[4] - a[4]) / u
            k = (c[4] - a[4] - j * v) / w

            def L4(z):
                return z[4] - a[4] * z[1] - j * (z[2] - a[2] * z[1]) - k * L3(z)

            l = L4
            res = L4(y)
        if not full:
            return res
        bb = (y[2] - a[2] * y[1] - cc * v) / u
        aa = y[1] - bb - cc
        t = self.base - (aa * a[0] + bb * b[0] + cc * c[0])
        dg = -(bb * l(db) + cc * l(dc))
        dp = dg - aa * l(da)
        return t, aa / e, bb / f, cc / h, abs(linear_jac), dp, dg

    def two_contrasts(self):
        """Lower-rank model when close levels don't resolve a third contrast."""
        p, wp = gauss_panels(P_BOUNDS, 112)
        g, wg = gauss_panels(G_BOUNDS, 84)
        cf, wc = gauss_panels(COEFF_PANELS, 8)
        p, wp = p[None, :, None], wp[None, :, None]
        g, wg = g[None, None, :], wg[None, None, :]
        cf, wc = cf[:, None, None], wc[:, None, None]
        e, a, _ = self.basis(p)
        f, b, _ = self.basis(p + g)
        h, c, _ = self.basis(p + g + THIRD_GAP)
        u = b[2] - a[2]
        cc = cf * h
        bb = (self.obs[2] - a[2] * self.obs[1] - (c[2] - a[2]) * cc) / u
        aa = self.obs[1] - bb - cc
        t = self.base - (aa * a[0] + bb * b[0] + cc * c[0])
        w = wp * wg * wc / abs(e * f * u)
        return np.array(np.broadcast_arrays(p, g, aa/e, bb/f, cf, t, w), dtype=float).reshape(7, -1)

    def integrate(self, resolution=1.0):
        """Trace the exact-fit surface in two overlapping root coordinates."""
        if self.n == 3:
            return self.two_contrasts()
        results = []
        count = int((229 if self.n == 4 else 751) * resolution)
        count = max(30, count)
        coarse = max(45, int(110 * min(1.5, resolution)))
        if self.n == 4:
            # Resolve the prior around both signs and retain a broad tail.
            coeffs, cw = gauss_panels(COEFF_PANELS, max(5, int(9 * resolution)))
        else:
            coeffs = [None]
            cw = [1.0]

        for solve_p in (True, False):
            bounds = P_BOUNDS if solve_p else G_BOUNDS
            other = G_BOUNDS if solve_p else P_BOUNDS
            root_grid = np.linspace(*bounds, coarse)
            fixed_grid, step = np.linspace(*other, count, retstep=True)
            if solve_p:
                p, g = root_grid, fixed_grid[:, None]
            else:
                p, g = fixed_grid[:, None], root_grid
            raw = self.equations(p, g)
            if self.n == 4:
                intercept, slope = raw
                values = intercept - coeffs[:, None, None] * slope
            else:
                values = raw[None, :, :]
            signs = values > 0
            ci, fi, ri = np.nonzero(signs[..., :-1] != signs[..., 1:])
            if not len(ri):
                continue
            # Very dense sign changes signal numerical, not model, structure.
            # Bound temporary storage and work even on ill-conditioned data.
            if len(ri) > 105000:
                continue
            lo, hi = root_grid[ri], root_grid[ri + 1]
            flo, fhi = values[ci, fi, ri], values[ci, fi, ri + 1]
            fixed = fixed_grid[fi]
            cv = coeffs[ci] if self.n == 4 else None

            def evaluate(root, full=False):
                if solve_p:
                    return self.equations(root, fixed, cv, full)
                return self.equations(fixed, root, cv, full)

            for _ in range(22):
                mid = 0.5 * (lo + hi)
                fm = evaluate(mid)
                same = (fm > 0) == (flo > 0)
                lo = np.where(same, mid, lo)
                flo = np.where(same, fm, flo)
                hi = np.where(same, hi, mid)
                fhi = np.where(same, fhi, fm)
            root = lo + (hi - lo) * (abs(flo) / (abs(flo) + abs(fhi)))
            t, a, b, c, jac, dp, dg = evaluate(root, full=True)
            norm = np.hypot(dp, dg)
            # Partition unity: derivative squared divided by gradient squared.
            # This keeps either chart integrable across its own turning points.
            component = abs(dp if solve_p else dg)
            weights = (component / norm) / (norm * jac)
            weights *= np.asarray(cw)[ci] * step
            weights *= np.where((fi == 0) | (fi == count - 1), 0.5, 1.0)
            p, g = (root, fixed) if solve_p else (fixed, root)
            points = np.array([p, g, a, b, c, t, weights], dtype=float)
            valid = np.all(np.isfinite(points), axis=0) & (points[6] > 0)
            results.append(points[:, valid])
        return np.concatenate(results, axis=1) if results else np.empty((7, 0))


def quantile(t, weights, q=0.5):
    order = np.argsort(t)
    t = t[order]
    w = weights[order]
    w = w / np.max(w)
    total = np.cumsum(w)
    return np.interp(q * total[-1], total - 0.5 * w, t)


def posterior(candidates, weighting=prior):
    p, g, a, b, c, t, geometric_weights = candidates
    weights = geometric_weights * weighting(p, g, a, b, c, t)
    # For the relative absolute loss, reweight before taking the median.
    weights /= np.maximum(1.0, abs(t))
    valid = (weights > 0) & np.isfinite(weights)
    if not np.any(valid):
        return None
    return float(quantile(t[valid], weights[valid]))


def compatible(candidates):
    """Is there a fit without extremely large, mutually canceling terms?"""
    _, _, a, b, c, _, _ = candidates
    return np.any((abs(a) < 5.0) & (abs(b) < 3.0) & (abs(c) < 1.4))


def rounding_proxy(s):
    """Absolute condition number of the highest divided difference."""
    diffs = s[:, None] - s
    np.fill_diagonal(diffs, 1.0)
    return np.sum(1.0 / abs(np.prod(diffs, axis=1)))


def resolved_grid(s, y):
    """Retain the widest span and omit levels whose contrast is roundoff-led."""
    noise = 4.0 * np.finfo(float).eps * np.max(abs(y))
    slope_scale = np.ptp(y) / (s[-1] - s[0])
    while len(s) > 3 and noise * rounding_proxy(s) > 0.00022 * slope_scale:
        subsets = [np.delete(np.arange(len(s)), i) for i in range(1, len(s) - 1)]
        best = min(subsets, key=lambda ids: rounding_proxy(s[ids]))
        s, y = s[best], y[best]
    if len(s) == 3 and noise * rounding_proxy(s) > 0.0025 * slope_scale:
        return None
    return s, y


def exact_single_power(s, y):
    span = float(s[-1] - s[0])
    delta = float(y[-1] - y[0])
    if span < 0.1 or abs(delta) < 1.0e-11:
        return None
    k = len(s) // 2
    mid = float(s[k] - s[-1])
    left = float(s[k] - s[0])
    ratio = float(y[k] - y[0]) / delta
    if mid >= 0 or left <= 0 or not 0 < ratio < 1:
        return None

    def fraction(p):
        return math.exp(p * mid) * (-math.expm1(-p * left)) / (-math.expm1(-p * span))

    lo, hi = 0.025, 20.0
    if not fraction(hi) <= ratio <= fraction(lo):
        return None
    for _ in range(50):
        p = 0.5 * (lo + hi)
        if fraction(p) > ratio:
            lo = p
        else:
            hi = p
    power = np.exp(p * (s - s[-1]))
    amplitude = delta / (-math.expm1(-p * span))
    fitted = y[-1] + amplitude * (power - 1.0)
    if np.max(abs(fitted - y)) <= 1.0e-9 * abs(delta) + 8.0e-15:
        return float(y[-1] - amplitude)
    return None


def fallback(s, y):
    """A bounded single-power fit, or a damped-error average for reversals."""
    scale = float(np.max(abs(y)))
    if scale == 0 or scale > 1.0e300:
        return 0.0
    yn = y / scale
    change = np.ptp(yn)
    if change == 0:
        return float(y[0])
    diff = np.diff(yn)
    signs = np.sign(diff[abs(diff) > change * 0.015])
    turns = np.count_nonzero(signs[1:] != signs[:-1])
    if turns >= 2:
        # An oscillating, converging error is typically smaller on fine levels.
        w = np.exp(-3.2 * (s - s[0]))
        return float((np.dot(w, yn) / w.sum()) * scale)
    if turns == 1:
        return float(y[0])
    orders = np.linspace(0.45, 5.0, 201)
    powers = np.exp(orders[:, None] * (s - s[-1]))
    centered = powers - powers.mean(axis=1, keepdims=True)
    denominator = np.sum(centered**2, axis=1)
    if np.min(denominator) == 0.0:
        return float(y[0])
    ys = (yn - yn[0]) / change
    amplitude = np.sum(centered * ys, axis=1) / denominator
    limit = np.mean(ys) - amplitude * np.mean(powers, axis=1)
    err = np.sum((ys - limit[:, None] - amplitude[:, None] * powers) ** 2, axis=1)
    best = np.argmin(err + 0.00025 * ((orders - 1.7) / 1.3) ** 2)
    adjust = float(np.clip(limit[best], -2.0, 2.0)) * change
    return float((yn[0] + adjust) * scale)


def estimate(x, y):
    sort = np.argsort(x)
    x = np.asarray(x, dtype=float)[sort]
    s = np.log(x)
    y = np.asarray(y, dtype=float)[sort]
    if np.all(y == y[0]):
        return float(y[0])
    if np.min(np.diff(s)) < 1.0e-4 or s[-1] - s[0] < 0.16:
        s = np.log(x.astype(np.longdouble))
    scale = float(np.max(abs(y)))
    exact = exact_single_power(s, y / scale)
    if exact is not None and math.isfinite(exact * scale):
        return exact * scale
    if (
        s[0] < -11.0 or s[-1] > 1.2 or s[-1] - s[0] > 9.0
        or np.min(np.diff(s)) <= 0 or scale > 100.0
    ):
        return fallback(s, y)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        grid = resolved_grid(s, y)
        z = None
        if grid is not None:
            levels, values = grid
            resolution = 1.85 if len(levels) == 4 else 1.2
            candidates = PowerFit(levels, values).integrate(resolution)
            if compatible(candidates):
                z = posterior(candidates)
    if z is None or not math.isfinite(z):
        return fallback(s, y)
    return z


def main():
    inp = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/inputs/input.json")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "/workspace/result/output.json")
    with inp.open() as f:
        data = json.load(f)
    try:
        z = estimate(data["x"], data["y"])
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        i = min(range(len(data["x"])), key=data["x"].__getitem__)
        z = float(data["y"][i])
    if not math.isfinite(z):
        z = 0.0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({"estimate": z}, f, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
