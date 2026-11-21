#!/usr/bin/env python3
"""Budgeted black-box integration with validated low-complexity surrogates."""
import itertools
import decimal
import json
import math
import signal
import sys
import time
import warnings

import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy import integrate, linalg, optimize, special
from scipy.spatial.distance import cdist
from scipy.stats import qmc

warnings.filterwarnings("ignore")
START = time.monotonic()
TRACE = False


class StopWork(Exception):
    pass


def check_time(margin=0.0):
    if time.monotonic() - START > 25.0 - margin:
        raise StopWork()


def trace(*args):
    if TRACE:
        print(*args, file=sys.stderr, flush=True)


class Oracle:
    def __init__(self, init):
        self.d = int(init["dimension"])
        self.budget = int(init["budget"])
        self.seed = int(init.get("seed", 0)) % (2**32)
        self.used = 0
        self.xs = []
        self.ys = []
        self.fallback = None

    @property
    def remaining(self):
        return self.budget - self.used

    def __call__(self, x):
        x = np.asarray(x, dtype=float).reshape(-1, self.d)
        if len(x) > self.remaining:
            raise StopWork()
        if not np.all(np.isfinite(x)):
            raise ValueError("nonfinite query")
        x = np.clip(x, 0.0, 1.0)
        out = []
        for start in range(0, len(x), 512):
            xx = x[start:start + 512]
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
            try:
                print(json.dumps({"type": "query", "points": xx.tolist()},
                                 separators=(",", ":")), flush=True)
                self.used += len(xx)
                line = sys.stdin.readline()
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            if not line:
                raise StopWork()
            reply = json.loads(line)
            yy = np.asarray(reply["values"], dtype=float)
            if yy.shape != (len(xx),) or not np.all(np.isfinite(yy)):
                raise ValueError("invalid oracle values")
            self.xs.append(xx.copy())
            self.ys.append(yy)
            out.append(yy)
            if self.fallback is None and len(yy):
                self.fallback = stable_mean(yy)
        return np.concatenate(out) if out else np.empty(0)

    def data(self):
        return np.vstack(self.xs), np.concatenate(self.ys)


def rms(y):
    y = np.asarray(y)
    scale = float(np.max(np.abs(y))) if y.size else 0.0
    if not scale:
        return 0.0
    if not np.isfinite(scale):
        return float("inf")
    return float(scale*np.sqrt(np.mean((y/scale)**2)))


def stable_mean(y):
    y = np.asarray(y)
    scale = float(np.max(np.abs(y))) if y.size else 0.0
    return float(np.mean(y/scale)*scale) if scale else 0.0


def cube_vertices(d, seed):
    if d <= 8:
        return np.array(list(itertools.product((0.0, 1.0), repeat=d)))
    return (qmc.Sobol(d, scramble=True, seed=seed).random(128) > .5).astype(float)


def accurate(pred, y, tol=2e-10):
    scale = max(rms(y), float(np.max(np.abs(y))) * 0.03, 1e-290)
    return (np.all(np.isfinite(pred)) and
            rms(np.asarray(pred) - y) < tol * scale and
            np.max(np.abs(np.asarray(pred) - y)) < tol * scale * 8)


def poly_design(x, degree=2):
    z = np.asarray(x) - 0.5
    d = z.shape[1]
    columns = [np.ones(len(z))]
    means = [1.0]
    for k in range(1, degree + 1):
        for inds in itertools.combinations_with_replacement(range(d), k):
            columns.append(np.prod(z[:, inds], axis=1))
            counts = np.bincount(inds, minlength=d)
            means.append(float(np.prod([
                0.0 if n % 2 else 0.5**n / (n + 1) for n in counts
            ])))
    return np.column_stack(columns), np.asarray(means)


def exp_quad_integral(c, b, a):
    """Integrate exp(c + sum(b*z+a*z*z)), z=x-1/2."""
    total = float(c)
    for bb, aa in zip(b, a):
        if aa < -1e-9:
            root = math.sqrt(-aa)
            mu = -bb / (2 * aa)
            lo = root * (-0.5 - mu)
            hi = root * (0.5 - mu)
            if lo > 0:
                diff = special.erfc(lo) - special.erfc(hi)
            elif hi < 0:
                diff = special.erfc(-hi) - special.erfc(-lo)
            else:
                diff = special.erf(hi) - special.erf(lo)
            if diff > 1e-270:
                total += -bb * bb / (4 * aa) + math.log(math.sqrt(math.pi) *
                                                                    diff / (2 * root))
                continue
        candidates = [-0.5, 0.5]
        if aa < 0 and abs(bb / (2 * aa)) < 0.5:
            candidates.append(-bb / (2 * aa))
        shift = max(bb * z + aa * z * z for z in candidates)
        val = integrate.quad(lambda z: math.exp(bb*z + aa*z*z - shift),
                             -0.5, 0.5, epsabs=2e-13, epsrel=2e-13)[0]
        total += shift + math.log(max(val, 1e-300))
    return float(np.exp(total))


def simple_global(x, y):
    if not np.any(y):
        return None
    scale = max(float(np.max(np.abs(y))), 1e-290)
    a, means = poly_design(x, 2)
    ntrain = max(a.shape[1] + 8, int(0.72 * len(x)))
    if ntrain < len(x):
        c = linalg.lstsq(a[:ntrain], y[:ntrain] / scale,
                        lapack_driver="gelsy")[0]
        if accurate(a @ c, y / scale, 4e-12):
            c = linalg.lstsq(a, y / scale, lapack_driver="gelsy")[0]
            return float(c @ means * scale)
    sign = 1.0 if np.all(y >= 0) else (-1.0 if np.all(y <= 0) else 0.0)
    if sign:
        z = x - 0.5
        a = np.column_stack((np.ones(len(x)), z, z*z))
        good = np.abs(y) > 1e-290
        if np.sum(good) < a.shape[1]+8:
            return None
        logs = np.log(np.abs(y[good]))
        c = linalg.lstsq(a[good], logs, lapack_driver="gelsy")[0]
        if (np.max(np.abs(a[good] @ c - logs)) < 2e-9 and
                (np.all(good) or np.max(a[~good] @ c) < -660)):
            d = x.shape[1]
            val = sign * exp_quad_integral(c[0], c[1:d+1], c[d+1:])
            if np.isfinite(val):
                return val
    for degree in (3, 4):
        count = math.comb(x.shape[1]+degree, degree)
        if count > 260 or count*1.7+8 >= len(x):
            continue
        a, means = poly_design(x, degree)
        ntrain = max(count+8, int(.72*len(x)))
        c = linalg.lstsq(a[:ntrain], y[:ntrain]/scale, lapack_driver="gelsy")[0]
        if accurate(a @ c, y/scale, 3e-12):
            c = linalg.lstsq(a, y/scale, lapack_driver="gelsy")[0]
            return float(c @ means*scale)
    return None


def _line_fit(x, y, tol=3e-11):
    """Small, independently overdetermined one-dimensional model library."""
    x = np.asarray(x)
    y = np.asarray(y)
    order = np.argsort(x)
    x, y = x[order], y[order]
    distinct = np.r_[True, np.diff(x) > 2e-14]
    x, y = x[distinct], y[distinct]
    scale = max(float(np.max(np.abs(y))), 1e-290)
    v = y / scale
    z = 2*x - 1
    for degree in (0, 1, 2, 3, 4, 6, 8, 12):
        if len(x) < 2*degree + 6:
            continue
        coef = cheb.chebfit(z, v, degree)
        if accurate(cheb.chebval(z, coef), v, tol):
            ci = cheb.chebint(coef)
            val = 0.5*(cheb.chebval(1, ci) - cheb.chebval(-1, ci))*scale
            return float(val), lambda t, c=coef, s=scale: s*cheb.chebval(2*np.asarray(t)-1, c)

    transforms = [(v, "raw")]
    if np.all(v > 0) or np.all(v < 0):
        transforms.append((np.log(np.abs(v)), "log"))
        if np.min(np.abs(v)) > 1e-120:
            transforms.append((1/v, "inverse"))
    for target, kind in transforms[1:]:
        for degree in (1, 2, 3, 4, 6):
            if len(x) < 2*degree+7:
                continue
            coef = cheb.chebfit(z, target, degree)
            fitted = cheb.chebval(z, coef)
            if kind == "log":
                pred = np.sign(v[0])*np.exp(fitted)
            else:
                pred = 1/fitted
            if not accurate(pred, v, tol):
                continue
            if kind == "log":
                fun = lambda t, c=coef, s=scale*np.sign(v[0]): s*np.exp(cheb.chebval(2*np.asarray(t)-1, c))
            else:
                roots = cheb.chebroots(coef)
                if any(abs(r.imag) < 1e-9 and -1 <= r.real <= 1 for r in roots):
                    continue
                fun = lambda t, c=coef, s=scale: s/cheb.chebval(2*np.asarray(t)-1, c)
            val = integrate.quad(fun, 0, 1, epsabs=scale*1e-13,
                                 epsrel=2e-13, limit=150)[0]
            if np.isfinite(val):
                return float(val), fun

    # Two smooth pieces joined at a knot, including exponential cusps.
    for target, kind in transforms[:2]:
        if kind == "inverse":
            continue
        for k in range(3, len(x)-3):
            left = np.polyfit(x[:k], target[:k], 1)
            right = np.polyfit(x[k:], target[k:], 1)
            den = left[0] - right[0]
            if abs(den) < 1e-12:
                continue
            t = (right[1]-left[1])/den
            if not x[k-1]-1e-12 <= t <= x[k]+1e-12:
                continue
            fitted = np.where(x <= t, np.polyval(left, x), np.polyval(right, x))
            pred = np.sign(v[0])*np.exp(fitted) if kind == "log" else fitted
            if not accurate(pred, v, tol):
                continue
            if kind == "raw":
                val = (left[0]*t*t/2 + left[1]*t +
                       right[0]*(1-t*t)/2 + right[1]*(1-t))*scale
                fun = lambda u, l=left, r=right, t=t, s=scale: s*np.where(np.asarray(u) <= t, np.polyval(l, u), np.polyval(r, u))
            else:
                def exp_segment(co, lo, hi):
                    b, a = co
                    if abs(b*(hi-lo)) < 1e-7:
                        return math.exp(a+b*lo)*(hi-lo)*float(np.expm1(b*(hi-lo))/(b*(hi-lo))) if b else math.exp(a)*(hi-lo)
                    return (math.exp(a+b*hi)-math.exp(a+b*lo))/b
                val = (exp_segment(left, 0, t)+exp_segment(right, t, 1))*scale*np.sign(v[0])
                fun = lambda u, l=left, r=right, t=t, s=scale*np.sign(v[0]): s*np.exp(np.where(np.asarray(u) <= t, np.polyval(l, u), np.polyval(r, u)))
            return float(val), fun

    # Algebraic cusps are common nonsmooth low-complexity factors.
    if len(x) >= 16 and time.monotonic() - START < 18:
        slopes = np.diff(v)/np.maximum(np.diff(x), 1e-12)
        bend = np.abs(np.diff(slopes))
        knot = x[1:-1][np.argmax(bend)] if len(bend) else 0.5
        def residual(p):
            a, b, c, t, power = p
            return a+b*x+c*np.abs(x-t)**power-v
        for power in (0.5, 1.0, 1.5):
            design = np.column_stack((np.ones(len(x)), x, np.abs(x-knot)**power))
            abc = linalg.lstsq(design, v, lapack_driver="gelsy")[0]
            p = np.r_[abc, knot, power]
            try:
                fit = optimize.least_squares(residual, p,
                         bounds=([-np.inf]*3+[0, 0.05], [np.inf]*3+[1, 6]),
                         max_nfev=100, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                if accurate(residual(fit.x)+v, v, tol):
                    a, b, c, t, power = fit.x
                    val = (a+b/2+c*(t**(power+1)+(1-t)**(power+1))/(power+1))*scale
                    fun = lambda u, a=a, b=b, c=c, t=t, p=power, s=scale: s*(a+b*np.asarray(u)+c*np.abs(np.asarray(u)-t)**p)
                    return float(val), fun
            except (ValueError, FloatingPointError):
                pass
        if np.all(v > 0):
            logs = np.log(v)
            knot = x[np.argmax(v)]
            for power in (-0.5, 0.5):
                if power > 0:
                    knot = x[np.argmin(v)]
                def residual(p):
                    a, power, lb, t = p
                    return a+power*np.log(np.exp(lb)+np.abs(x-t))-logs
                b0 = 0.03
                a0 = np.mean(logs-power*np.log(b0+np.abs(x-knot)))
                fit = optimize.least_squares(residual, [np.clip(a0, -999, 999), power, math.log(b0), knot],
                        bounds=([-1000, -8, -22, 0], [1000, 8, 5, 1]),
                        max_nfev=160, ftol=1e-13, xtol=1e-13, gtol=1e-13)
                if np.max(np.abs(residual(fit.x))) < tol*3:
                    a, p, lb, t = fit.x
                    b = math.exp(lb)
                    if abs(p+1) < 1e-7:
                        integ = math.log1p(t/b)+math.log1p((1-t)/b)
                    else:
                        integ = ((b+t)**(p+1)+(b+1-t)**(p+1)-2*b**(p+1))/(p+1)
                    val = scale*math.exp(a)*integ
                    fun = lambda u, a=a, b=b, t=t, p=p, s=scale: s*math.exp(a)*(b+np.abs(np.asarray(u)-t))**p
                    return float(val), fun
    return None


GKX = np.array([0.9914553711208126, 0.9491079123427585,
                0.8648644233597691, 0.7415311855993945,
                0.5860872354676911, 0.4058451513773972,
                0.2077849550078985, 0.0])
GKW = np.array([0.02293532201052922, 0.06309209262997855,
                0.1047900103222502, 0.1406532597155259,
                0.1690047266392679, 0.1903505780647854,
                0.2044329400752989, 0.2094821410847278])
GW = np.array([0.1294849661688697, 0.2797053914892767,
               0.3818300505051189, 0.4179591836734694])
NODES = np.r_[-GKX[:-1], 0.0, GKX[-2::-1]]
KW = np.r_[GKW[:-1], GKW[-1], GKW[-2::-1]]
GAUSS_W = np.zeros(15)
GAUSS_W[[1, 3, 5, 7, 9, 11, 13]] = np.r_[GW[:3], GW[3], GW[2::-1]]


def integrate_line(query, xs, ys, allowance):
    fit = _line_fit(xs, ys)
    if fit is not None:
        return fit[0]
    used = 0
    order = np.argsort(xs)
    sx = np.asarray(xs)[order]
    sy = np.asarray(ys)[order]
    scale = max(np.max(np.abs(sy)), 1e-290)

    # Locate jumps between constant pieces without spending quadrature nodes
    # on both sides of every discontinuity.
    levels = []
    labels = []
    for value in sy:
        matches = [j for j, a in enumerate(levels) if abs(value-a) < scale*2e-12]
        if matches:
            labels.append(matches[0])
        else:
            labels.append(len(levels))
            levels.append(value)
    if 1 < len(levels) <= 8 and len(levels) < len(sy)//3:
        changes = np.flatnonzero(np.diff(labels))
        if len(changes) and allowance >= 12*len(changes):
            lo = sx[changes].copy()
            hi = sx[changes+1].copy()
            lv = sy[changes].copy()
            nsteps = min(36, (allowance-4)//len(changes))
            for _ in range(nsteps):
                mid = (lo+hi)*0.5
                ym = query(mid)
                used += len(mid)
                same = np.abs(ym-lv) < scale*2e-12
                lo = np.where(same, mid, lo)
                hi = np.where(same, hi, mid)
            edges = np.r_[0, (lo+hi)*0.5, 1]
            vals = np.r_[sy[0], sy[changes+1]]
            return float(np.dot(np.diff(edges), vals))

    # A zero-valued exterior is handled as a free boundary, then the smooth
    # interior is fitted again after rescaling its support.
    nonzero = np.abs(sy) > 0
    if np.any(nonzero) and np.any(~nonzero):
        first, last = np.flatnonzero(nonzero)[[0, -1]]
        if np.all(nonzero[first:last+1]):
            bounds = []
            sides = []
            if first > 0:
                bounds.append([sx[first-1], sx[first]])
                sides.append(False)
            if last < len(sx)-1:
                bounds.append([sx[last], sx[last+1]])
                sides.append(True)
            if bounds and allowance >= 64*len(bounds)+17:
                ab = np.asarray(bounds)
                for _ in range(45):
                    mid = ab.mean(axis=1)
                    ym = query(mid)
                    used += len(mid)
                    for j, inside_left in enumerate(sides):
                        same = bool(ym[j] != 0) == inside_left
                        ab[j, 0 if same else 1] = mid[j]
                lo = ab[0].mean() if first > 0 else 0.0
                hi = ab[-1].mean() if last < len(sx)-1 else 1.0
                tx = (np.arange(24)+0.5)/24
                yy = query(lo+(hi-lo)*tx)
                used += len(tx)
                fit = _line_fit(tx, yy)
                if fit:
                    return (hi-lo)*fit[0]

    def rule(a, b):
        yy = query((a+b)/2+(b-a)/2*NODES)
        value = float(KW @ yy)*(b-a)/2
        err = abs(float((KW-GAUSS_W) @ yy))*(b-a)/2
        return [a, b, value, err]

    intervals = [rule(0, 1)]
    used += 15
    while used+30 <= allowance:
        check_time()
        total = sum(r[2] for r in intervals)
        error = sum(r[3] for r in intervals)
        if error < 2e-12*max(abs(total), scale*0.01):
            break
        j = int(np.argmax([r[3] for r in intervals]))
        a, b = intervals.pop(j)[:2]
        mid = (a+b)/2
        intervals.extend((rule(a, mid), rule(mid, b)))
        used += 30
    return sum(r[2] for r in intervals)


def separable(oracle, x, y):
    d = oracle.d
    ia = int(np.argmax(np.abs(y)))
    anchor, fa = x[ia].copy(), float(y[ia])
    if fa == 0:
        return None
    # Spread the validation locations across the observed range.
    ids = np.unique(np.r_[np.linspace(0, len(x)-1, min(16, len(x)), dtype=int),
                         np.argmin(y), np.argmax(y), len(x)-3, len(x)-2])
    targets = x[ids]
    points = np.repeat(anchor[None, :], len(ids)*d, axis=0)
    for k in range(d):
        points[k*len(ids):(k+1)*len(ids), k] = targets[:, k]
    lines = oracle(points).reshape(d, len(ids))
    pred = fa*np.prod(lines/fa, axis=0)
    product_ok = accurate(pred, y[ids], 3e-11)
    if product_ok and fa > 0 and np.all(lines >= 0) and np.all(y[ids] >= 0):
        positive = (y[ids] > 1e-270) & np.all(lines > 1e-290, axis=0)
        if np.any(positive):
            logpred = np.sum(np.log(lines[:, positive]), axis=0)-(d-1)*math.log(fa)
            if np.max(np.abs(logpred-np.log(y[ids][positive]))) > 2e-7:
                product_ok = False
        if np.any((y[ids] > 1e-270) & ~np.all(lines > 0, axis=0)):
            product_ok = False
    shift = 0.0
    mode = None
    if product_ok:
        mode = "product"
    elif accurate(np.sum(lines, axis=0)-(d-1)*fa, y[ids], 3e-11):
        mode = "sum"
    elif d >= 2 and oracle.remaining > 512:
        pair = np.repeat(anchor[None, :], len(ids), axis=0)
        pair[:, :2] = targets[:, :2]
        yp = oracle(pair)
        den = yp+fa-lines[0]-lines[1]
        valid = np.abs(den) > max(np.max(np.abs(y)), 1e-290)*1e-7
        if np.any(valid):
            shifts = (yp*fa-lines[0]*lines[1])[valid]/den[valid]
            shift = float(np.median(shifts))
            if abs(fa-shift) > max(abs(fa), 1e-290)*1e-12:
                pred = shift+(fa-shift)*np.prod((lines-shift)/(fa-shift), axis=0)
                if accurate(pred, y[ids], 3e-11):
                    shifted_ok = True
                    active = (np.abs(y[ids]-shift) > max(1e-270, abs(shift)*1e-10))
                    active &= np.all(np.abs(lines-shift) > max(1e-290, abs(shift)*1e-10), axis=0)
                    if np.any(active):
                        lp = np.sum(np.log(np.abs(lines[:, active]-shift)), axis=0)
                        lp -= (d-1)*math.log(abs(fa-shift))
                        if np.max(np.abs(lp-np.log(np.abs(y[ids][active]-shift)))) > 2e-6:
                            shifted_ok = False
                    if shifted_ok:
                        mode = "product"
    if mode is None:
        return None
    trace("separable", mode, "shift", shift)
    nodes = (1-np.cos(np.linspace(0, np.pi, 25)))/2
    pts = np.repeat(anchor[None, :], d*len(nodes), axis=0)
    for k in range(d):
        pts[k*len(nodes):(k+1)*len(nodes), k] = nodes
    sampled = oracle(pts).reshape(d, len(nodes))
    integrals = []
    for k in range(d):
        check_time()
        def line_query(t, k=k):
            xx = np.repeat(anchor[None, :], len(np.atleast_1d(t)), axis=0)
            xx[:, k] = t
            yy = oracle(xx)
            return ((yy-shift)/(fa-shift) if mode == "product" else yy-fa)
        xx = np.r_[nodes, targets[:, k], anchor[k]]
        yy = np.r_[sampled[k], lines[k], fa]
        yy = (yy-shift)/(fa-shift) if mode == "product" else yy-fa
        allowance = max(15, (oracle.remaining-16)//(d-k))
        value = integrate_line(line_query, xx, yy, allowance)
        integrals.append(value)
    result = shift+(fa-shift)*float(np.prod(integrals)) if mode == "product" else fa+sum(integrals)
    return result if np.isfinite(result) else None


def differential_samples(oracle, x, y, count=18, h=0.0002):
    ids = np.linspace(0, len(x)-1, count, dtype=int)
    centers, values = x[ids], y[ids]
    centers = np.clip(centers, 2*h, 1-2*h)
    changed = np.any(centers != x[ids], axis=1)
    if np.any(changed):
        values = values.copy()
        values[changed] = oracle(centers[changed])
    pts = np.repeat(centers[:, None, :], 2*oracle.d, axis=1)
    for j in range(oracle.d):
        pts[:, 2*j, j] -= h
        pts[:, 2*j+1, j] += h
    ys = oracle(pts.reshape(-1, oracle.d)).reshape(count, oracle.d, 2)
    gradient = (ys[:, :, 1]-ys[:, :, 0])/(2*h)
    hessian_diag = (ys[:, :, 1]+ys[:, :, 0]-2*values[:, None])/h**2
    return centers, values, gradient, hessian_diag


def extend_piecewise(oracle, dif, count):
    """Seek unobserved affine regions using supporting-plane residuals."""
    x, y = oracle.data()
    scale = max(np.max(np.abs(y)), 1e-290)
    good = np.max(np.abs(dif[3]), axis=1) < scale*.01
    g = dif[2][good]
    intercepts = dif[1][good]-np.sum(g*dif[0][good], axis=1)
    if not len(g):
        return dif
    predictions = x @ g.T+intercepts
    convex = np.max(np.max(predictions, axis=1)-y) < 2e-7*scale
    concave = np.max(y-np.min(predictions, axis=1)) < 2e-7*scale
    if not convex and not concave:
        return dif
    for _ in range(count):
        check_time(4)
        predictions = x @ g.T+intercepts
        gaps = y-np.max(predictions, axis=1) if convex else np.min(predictions, axis=1)-y
        j = int(np.argmax(gaps))
        if gaps[j] < 2e-10*scale or oracle.remaining < 2*oracle.d+160:
            break
        more = differential_samples(oracle, x[j:j+1], y[j:j+1], 1, h=1e-5)
        dif = tuple(np.concatenate((a, b), axis=0) for a, b in zip(dif, more))
        if np.max(np.abs(more[3])) < .03*scale:
            newg = more[2]
            newc = more[1]-np.sum(newg*more[0], axis=1)
            g = np.vstack((g, newg))
            intercepts = np.r_[intercepts, newc]
        else:
            x = np.delete(x, j, axis=0)
            y = np.delete(y, j)
    return dif


def cosine_fit(x, y, centers, values, gradient, hess):
    scale = max(np.max(np.abs(y)), 1e-290)
    yy = y/scale
    vv = values/scale
    gg = gradient/scale
    hh = hess/scale
    if np.std(vv) < 1e-7:
        return None
    coeff = linalg.lstsq(np.column_stack((vv, np.ones(len(vv)))), hh,
                        lapack_driver="gelsy")[0]
    slopes = coeff[0]
    if np.any(slopes > 1e-3) or np.max(-slopes) < 1e-4:
        return None
    _, s, vt = linalg.svd(gg, full_matrices=False)
    if len(s) > 1 and s[1] > 2e-3*s[0]:
        return None
    direction = vt[0]
    frequency = 10000*np.arcsin(np.minimum(.0001*np.sqrt(np.maximum(-slopes, 0)), 1))
    w0 = frequency*np.sign(direction)
    phase = x @ w0
    abc = linalg.lstsq(np.column_stack((np.ones(len(x)), np.cos(phase), np.sin(phase))),
                       yy, lapack_driver="gelsy")[0]
    p0 = np.r_[w0, abc]
    d = x.shape[1]
    def residual(p):
        phase = x @ p[:d]
        return p[d]+p[d+1]*np.cos(phase)+p[d+2]*np.sin(phase)-yy
    def jac(p):
        phase = x @ p[:d]
        co, si = np.cos(phase), np.sin(phase)
        return np.column_stack((x*(-p[d+1]*si+p[d+2]*co)[:, None],
                                np.ones(len(x)), co, si))
    fit = optimize.least_squares(residual, p0, jac=jac, max_nfev=100,
                                 ftol=1e-13, xtol=1e-13, gtol=1e-13)
    if not accurate(residual(fit.x)+yy, yy, 2e-11):
        return None
    noise = rms(residual(fit.x))
    if abs(fit.x[d]) < max(5e-16, 5*noise/math.sqrt(len(x))):
        base = fit.x.copy()
        base[d] = 0
        free = np.r_[np.arange(d), d+1, d+2]
        def bare_res(p):
            full = base.copy()
            full[free] = p
            return residual(full)
        def bare_jac(p):
            full = base.copy()
            full[free] = p
            return jac(full)[:, free]
        bare = optimize.least_squares(bare_res, base[free], jac=bare_jac,
                    max_nfev=40, ftol=1e-14, xtol=1e-14, gtol=1e-14)
        if rms(bare.fun) < 2*noise+5e-16:
            base[free] = bare.x
            fit.x = base
    w = fit.x[:d]
    c, a, b = fit.x[d:]
    integral = np.exp(0.5j*np.sum(w))*np.prod(np.sinc(w/(2*np.pi)))
    return float((c+np.real((a-1j*b)*integral))*scale)


def periodic_fit(x, y, centers, values, gradient, hess):
    """Sparse Fourier regression initialized by an observed active direction."""
    started = time.monotonic()
    scale = max(np.max(np.abs(y)), 1e-290)
    yy = y/scale
    gg = (gradient-np.mean(gradient, axis=0))/scale
    _, singular, vt = linalg.svd(gg, full_matrices=False)
    if singular[0] < 1e-7 or (len(singular) > 1 and singular[1] > .008*singular[0]):
        return None
    v = vt[0]
    d = x.shape[1]
    projection = x @ v
    spread = max(np.ptp(projection), 0.2)
    hnorm = rms(hess/scale)
    omega_guess = math.sqrt(hnorm/max(np.std(yy), 1e-5))*d**.25
    maxfreq = min(1400, max(100, omega_guess*6))
    step = math.pi/(5*spread)
    grid = np.arange(step, maxfreq, step)
    grid = grid[:5000]
    co = np.cos(projection[:, None]*grid)
    si = np.sin(projection[:, None]*grid)
    base = np.column_stack((np.ones(len(x)), x))
    c = linalg.lstsq(base, yy, lapack_driver="gelsy")[0]
    params = c.copy()
    waves = []
    freqs = []
    basis = base
    resid = yy-base @ c
    for rank in range(1, 7):
        if time.monotonic()-started > 3 or time.monotonic()-START > 20:
            return None
        Q, _ = linalg.qr(basis, mode="economic")
        C = co-Q @ (Q.T @ co)
        S = si-Q @ (Q.T @ si)
        cc, ss, cs = np.sum(C*C, axis=0), np.sum(S*S, axis=0), np.sum(C*S, axis=0)
        bc, bs = resid @ C, resid @ S
        gains = (ss*bc*bc+cc*bs*bs-2*cs*bc*bs)/np.maximum(cc*ss-cs*cs, 1e-15)
        for w in freqs:
            gains[np.abs(grid-w) < step*0.6] = -1
        j = int(np.argmax(gains))
        def freq_loss(f):
            A = np.column_stack((basis, np.cos(projection*f), np.sin(projection*f)))
            coef = linalg.lstsq(A, yy, lapack_driver="gelsy")[0]
            return float(np.sum((A @ coef-yy)**2))
        opt = optimize.minimize_scalar(freq_loss,
                    bounds=(max(step*.1, grid[j]-step), grid[j]+step),
                    method="bounded", options={"xatol": 1e-9})
        omega = opt.x
        freqs.append(omega)
        waves.append(omega*v)
        W = np.asarray(waves)
        phase = x @ W.T
        basis = np.column_stack((base, np.cos(phase), np.sin(phase)))
        coeff = linalg.lstsq(basis, yy, lapack_driver="gelsy")[0]
        params = np.r_[coeff[:d+1], W.ravel(), coeff[d+1:]]

        def residual(p):
            W = p[d+1:d+1+rank*d].reshape(rank, d)
            a = p[d+1+rank*d:d+1+rank*(d+1)]
            b = p[d+1+rank*(d+1):]
            ph = x @ W.T
            return base @ p[:d+1]+np.cos(ph) @ a+np.sin(ph) @ b-yy

        def jac(p):
            W = p[d+1:d+1+rank*d].reshape(rank, d)
            a = p[d+1+rank*d:d+1+rank*(d+1)]
            b = p[d+1+rank*(d+1):]
            ph = x @ W.T
            cph, sph = np.cos(ph), np.sin(ph)
            dw = ((-sph*a+cph*b)[:, :, None]*x[:, None, :]).reshape(len(x), -1)
            return np.column_stack((base, dw, cph, sph))

        fitted = optimize.least_squares(residual, params, jac=jac,
                max_nfev=70, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        params = fitted.x
        resid = -residual(params)
        W = params[d+1:d+1+rank*d].reshape(rank, d)
        waves = list(W)
        freqs = [abs(float(w @ v)) for w in W]
        phase = x @ W.T
        basis = np.column_stack((base, np.cos(phase), np.sin(phase)))
        if accurate(yy-resid, yy, 2e-11):
            a = params[d+1+rank*d:d+1+rank*(d+1)]
            b = params[d+1+rank*(d+1):]
            factors = np.exp(.5j*np.sum(W, axis=1))*np.prod(np.sinc(W/(2*np.pi)), axis=1)
            val = params[0]+np.sum(params[1:d+1])/2+np.real((a-1j*b) @ factors)
            def predict(xnew, W=W.copy(), a=a.copy(), b=b.copy(), c=params[:d+1].copy()):
                ph = xnew @ W.T
                return scale*(c[0]+xnew @ c[1:]+np.cos(ph) @ a+np.sin(ph) @ b)
            trace("Fourier terms", rank)
            return float(val*scale), predict
    return None


def multivariate_fourier(oracle, x, y, gradient):
    """Recover a short exponential sum from uniformly spaced axis fibers."""
    d = oracle.d
    if oracle.remaining < 49*d+128 or d > 12:
        return None
    anchor = x[0].copy()
    order = np.argsort(-np.std(gradient, axis=0))
    terms = []
    amplitudes = []
    scale = max(np.max(np.abs(y)), 1e-290)

    def fiber_fit(t, values):
        values = values/scale
        seq = values
        size = len(seq)//2
        H = linalg.hankel(seq[:size], seq[size-1:])
        U, singular, _ = linalg.svd(H, full_matrices=False)
        if singular[0] < 1e-11:
            return []
        largest_rank = int(np.sum(singular > singular[0]*8e-16))
        best_fiber = None
        best_loss = np.inf
        for rank in range(2, min(15, largest_rank+1)+1):
            Ur = U[:, :rank]
            shift = linalg.lstsq(Ur[:-1], Ur[1:], lapack_driver="gelsy")[0]
            roots = linalg.eigvals(shift)
            nonreal = roots[np.imag(roots) > 1e-5]
            if not len(nonreal) or len(nonreal) > 7:
                continue
            if np.max(np.abs(np.abs(nonreal)-1)) > .025:
                continue
            frequencies = np.sort(np.angle(nonreal)/(t[1]-t[0]))
            r = len(frequencies)
            base = np.column_stack((np.ones(len(t)), t))
            phase = t[:, None]*frequencies
            design = np.column_stack((base, np.cos(phase), np.sin(phase)))
            co = linalg.lstsq(design, values, lapack_driver="gelsy")[0]
            p = np.r_[co[:2], frequencies, co[2:]]
            def res(p):
                ph = t[:, None]*p[2:2+r]
                return base @ p[:2]+np.cos(ph) @ p[2+r:2+2*r]+np.sin(ph) @ p[2+2*r:]-values
            def jac(p):
                ph = t[:, None]*p[2:2+r]
                cc, ss = np.cos(ph), np.sin(ph)
                dw = t[:, None]*(-ss*p[2+r:2+2*r]+cc*p[2+2*r:])
                return np.column_stack((base, dw, cc, ss))
            fitted = optimize.least_squares(res, p, jac=jac, max_nfev=100,
                                            ftol=2e-13, xtol=2e-13, gtol=2e-13)
            loss = rms(res(fitted.x))
            if loss < best_loss:
                best_loss = loss
                best_fiber = (fitted.x.copy(), r)
            if not accurate(res(fitted.x)+values, values, 3e-11):
                continue
            p = fitted.x
            return [(abs(p[2+k]), p[2+r+k], p[2+2*r+k]*np.sign(p[2+k]))
                    for k in range(r) if abs(p[2+k]) > .1]
        if best_fiber is not None and best_loss < 3e-6*max(rms(values), 1e-8):
            p, r = best_fiber
            return [(abs(p[2+k]), p[2+r+k], p[2+2*r+k]*np.sign(p[2+k]))
                    for k in range(r) if abs(p[2+k]) > .1]
        return None

    for j in order:
        check_time(5)
        ts = np.linspace(0, 1, 49)
        pts = np.repeat(anchor[None, :], len(ts), axis=0)
        pts[:, j] = ts
        vals = oracle(pts)
        found = fiber_fit(ts-anchor[j], vals)
        if found is None:
            trace("Fourier fiber rejected", int(j))
            return None
        used = set()
        for frequency, a, b in found:
            key = np.array([a, abs(b)])
            errors = [np.linalg.norm(key-np.array([aa, abs(bb)])) /
                      max(np.linalg.norm(key), np.hypot(aa, bb), 1e-8)
                      if k not in used else np.inf
                      for k, (aa, bb) in enumerate(amplitudes)]
            if errors and min(errors) < (.06 if frequency < 3 else .015):
                k = int(np.argmin(errors))
                terms[k][j] = frequency*(1 if b*amplitudes[k][1] >= 0 else -1)
            else:
                if len(terms) >= 7:
                    return None
                k = len(terms)
                terms.append(np.zeros(d))
                terms[k][j] = frequency
                amplitudes.append((a, b))
            used.add(k)
    if not terms:
        return None
    xx, yy = oracle.data()
    xx = xx-anchor
    yy = yy/scale
    W = np.array(terms)
    rank = len(W)
    base = np.column_stack((np.ones(len(xx)), xx))
    phase = xx @ W.T
    design = np.column_stack((base, np.cos(phase), np.sin(phase)))
    coef = linalg.lstsq(design, yy, lapack_driver="gelsy")[0]
    params = np.r_[coef[:d+1], W.ravel(), coef[d+1:]]
    def residual(p):
        W = p[d+1:d+1+rank*d].reshape(rank, d)
        ph = xx @ W.T
        a, b = p[d+1+rank*d:d+1+rank*(d+1)], p[d+1+rank*(d+1):]
        return base @ p[:d+1]+np.cos(ph) @ a+np.sin(ph) @ b-yy
    def jac(p):
        W = p[d+1:d+1+rank*d].reshape(rank, d)
        ph = xx @ W.T
        a, b = p[d+1+rank*d:d+1+rank*(d+1)], p[d+1+rank*(d+1):]
        cc, ss = np.cos(ph), np.sin(ph)
        dw = ((-ss*a+cc*b)[:, :, None]*xx[:, None, :]).reshape(len(xx), -1)
        return np.column_stack((base, dw, cc, ss))
    fitted = optimize.least_squares(residual, params, jac=jac, max_nfev=100,
                                    ftol=2e-13, xtol=2e-13, gtol=2e-13)
    p = fitted.x
    if not accurate(residual(p)+yy, yy, 2e-10):
        trace("Fourier global residual", rms(residual(p)), "terms", rank)
        return None
    W = p[d+1:d+1+rank*d].reshape(rank, d)
    a, b = p[d+1+rank*d:d+1+rank*(d+1)], p[d+1+rank*(d+1):]
    factors = np.exp(1j*(W @ (.5-anchor)))*np.prod(np.sinc(W/(2*np.pi)), axis=1)
    value = p[0]+p[1:d+1] @ (.5-anchor)+np.real((a-1j*b) @ factors)
    def predict(xnew):
        z = xnew-anchor
        ph = z @ W.T
        return scale*(p[0]+z @ p[1:d+1]+np.cos(ph) @ a+np.sin(ph) @ b)
    trace("multivariate Fourier terms", rank)
    return float(value*scale), predict


def abs_ridge_integral(w, t):
    w = np.asarray(w, dtype=np.longdouble)
    t = np.longdouble(t) - np.sum(w[w < 0])
    w = np.abs(w)
    threshold = np.max(w)*1e-9 if len(w) else 0
    tiny = w < threshold
    t -= np.sum(w[tiny])/2
    w = w[~tiny]
    total = np.sum(w)
    if t <= 0:
        return float(total/2-t)
    if t >= total:
        return float(t-total/2)
    t = min(t, total-t)
    if len(w) >= 5 or (len(w) and np.min(w) < np.max(w)*.015):
        with decimal.localcontext() as context:
            context.prec = 60
            dw = [decimal.Decimal(str(a)) for a in w]
            dt = decimal.Decimal(str(t))
            dk = [decimal.Decimal(0)]
            ds = [1]
            for a in dw:
                dk += [b+a for b in dk]
                ds += [-s for s in ds]
            power = len(dw)+1
            accum = sum((s*(dt-k)**power for s, k in zip(ds, dk) if dt > k),
                        decimal.Decimal(0))
            denominator = decimal.Decimal(math.factorial(power))
            for a in dw:
                denominator *= a
            lower = accum/denominator
            return float(sum(dw)/2-dt+2*lower)
    knots = np.array([0], dtype=np.longdouble)
    signs = np.array([1], dtype=np.longdouble)
    for wi in w:
        knots = np.r_[knots, knots+wi]
        signs = np.r_[signs, -signs]
    n = len(w)
    term = np.sum(signs*np.maximum(t-knots, 0)**(n+1), dtype=np.longdouble)
    lower = term/(np.longdouble(math.factorial(n+1))*np.prod(w))
    return float(total/2-t+2*lower)


def hinge_fit(x, y, centers, values, gradient, hess):
    scale = max(np.max(np.abs(y)), 1e-290)
    if np.median(np.max(np.abs(hess), axis=1))/scale > 1e-3:
        return None
    good = np.max(np.abs(hess), axis=1)/scale < 0.01
    if np.sum(good) < 5:
        return None
    centers, values, gradient = centers[good], values[good]/scale, gradient[good]/scale
    yy = y/scale
    planes = np.column_stack((gradient, values-np.sum(gradient*centers, axis=1)))
    unique = []
    for p in planes:
        if not unique or min(np.linalg.norm(p-q) for q in unique) > 2e-7:
            unique.append(p)
    if len(unique) < 2:
        return None
    candidates, counts = [], []
    for a, b in itertools.combinations(unique, 2):
        dif = a-b
        norm = np.linalg.norm(dif[:-1])
        if norm < 1e-9:
            continue
        dif = dif/norm
        if dif[np.argmax(np.abs(dif[:-1]))] < 0:
            dif = -dif
        w, t = dif[:-1], -dif[-1]
        if not np.sum(np.minimum(w, 0)) < t < np.sum(np.maximum(w, 0)):
            continue
        distances = np.array([np.linalg.norm(dif-c) for c in candidates])
        if not len(distances) or np.min(distances) > 2e-6:
            candidates.append(dif)
            counts.append(1)
        else:
            counts[int(np.argmin(distances))] += 1
    if not candidates:
        return None
    if len(candidates) > 140:
        ids = np.argsort(-np.asarray(counts), kind="stable")[:140]
        candidates = [candidates[j] for j in ids]
    cs = np.array(candidates)
    d = x.shape[1]
    w, t = cs[:, :d], -cs[:, d]
    base = np.column_stack((np.ones(len(x)), x))
    features = np.abs(x @ w.T-t)
    # Derivatives make the sparse decomposition much better determined.
    weight = 0.12
    grad_base = np.tile(np.column_stack((np.zeros(d), np.eye(d))), (len(centers), 1))
    activation = np.sign(centers @ w.T-t)
    grad_features = (activation[:, None, :]*w.T[None, :, :]).reshape(-1, len(w))
    b = np.r_[yy, weight*gradient.ravel()]
    B = np.vstack((base, weight*grad_base))
    F = np.vstack((features, weight*grad_features))
    Q, _ = linalg.qr(B, mode="economic")
    FF = F-Q @ (Q.T @ F)
    norms = np.linalg.norm(FF, axis=0)
    res = b-Q @ (Q.T @ b)
    selected, coef = [], None
    residual_target = res.copy()
    if len(candidates) < 220:
        all_coef = linalg.lstsq(np.column_stack((B, F)), b, cond=1e-10,
                               lapack_driver="gelsy")[0]
        active = np.flatnonzero(np.abs(all_coef[d+1:]) > 2e-6)
        if len(active) <= 14:
            A = np.column_stack((B, F[:, active]))
            cc = linalg.lstsq(A, b, lapack_driver="gelsy")[0]
            if rms(A @ cc-b) < 1e-7:
                selected, coef = active.tolist(), cc
    if coef is None and len(candidates) < 600:
        for sign in (1, -1):
            try:
                nn, _ = optimize.nnls(FF, sign*residual_target, maxiter=3*len(candidates))
                active = np.flatnonzero(nn > 1e-7)
                if len(active) <= 16:
                    A = np.column_stack((B, F[:, active]))
                    cc = linalg.lstsq(A, b, lapack_driver="gelsy")[0]
                    if rms(A @ cc-b) < 2e-7:
                        selected, coef = active.tolist(), cc
                        break
            except (RuntimeError, ValueError):
                pass
    if coef is None:
        for _ in range(min(14, len(candidates), max(1, (len(x)-d-1)//(d+1)))):
            scores = np.abs(FF.T @ res)/np.maximum(norms, 1e-14)
            scores[selected] = -1
            j = int(np.argmax(scores))
            selected.append(j)
            A = np.column_stack((B, F[:, selected]))
            coef = linalg.lstsq(A, b, lapack_driver="gelsy")[0]
            res = b-A @ coef
            if rms(res) < 2e-8:
                break
    if coef is None:
        return None
    # Refine hyperplanes directly against values, avoiding finite-difference
    # error in the final integral.
    c = coef[:d+1]
    weights = coef[d+1:]
    keep = np.abs(weights) > 1e-8
    if not np.any(keep):
        return None
    ws = w[np.asarray(selected)[keep]]*np.abs(weights[keep])[:, None]
    ts = t[np.asarray(selected)[keep]]*np.abs(weights[keep])
    signs = np.sign(weights[keep])
    params = np.r_[c, np.column_stack((ws, ts)).ravel()]
    rank = len(signs)
    def residual(p):
        terms = p[d+1:].reshape(rank, d+1)
        return base @ p[:d+1]+np.abs(x @ terms[:, :d].T-terms[:, d]) @ signs-yy
    def jac(p):
        terms = p[d+1:].reshape(rank, d+1)
        act = np.sign(x @ terms[:, :d].T-terms[:, d])*signs
        parts = (act[:, :, None]*np.column_stack((x, -np.ones(len(x))))[:, None, :]).reshape(len(x), -1)
        return np.column_stack((base, parts))
    fit = optimize.least_squares(residual, params, jac=jac, max_nfev=70,
                                 ftol=2e-13, xtol=2e-13, gtol=2e-13)
    if not accurate(residual(fit.x)+yy, yy, 2e-10):
        return None
    terms = fit.x[d+1:].reshape(rank, d+1)
    # Identical observed activation patterns do not identify two separate
    # nearby knots. Merge them into the smaller, stable representation.
    keep = np.linalg.norm(terms[:, :d], axis=1) > 1e-10
    terms, signs = terms[keep].copy(), signs[keep].copy()
    merged = True
    while merged:
        merged = False
        for i in range(len(terms)):
            for j in range(i+1, len(terms)):
                if signs[i] != signs[j]:
                    continue
                ni = np.linalg.norm(terms[i, :d])
                nj = np.linalg.norm(terms[j, :d])
                orient = 1 if terms[i, :d] @ terms[j, :d] >= 0 else -1
                if abs(terms[i, :d] @ terms[j, :d]) < .92*ni*nj:
                    continue
                ti, tj = terms[i], terms[j]*orient
                vi, vj = x @ ti[:d]-ti[d], x @ tj[:d]-tj[d]
                compatible = np.all((vi >= -1e-10*ni) | (vj <= 1e-10*nj))
                compatible &= np.all((vi <= 1e-10*ni) | (vj >= -1e-10*nj))
                if compatible:
                    terms[i] += tj
                    terms = np.delete(terms, j, axis=0)
                    signs = np.delete(signs, j)
                    merged = True
                    break
            if merged:
                break
    if len(terms) != rank:
        rank = len(terms)
        params = np.r_[fit.x[:d+1], terms.ravel()]
        fit = optimize.least_squares(residual, params, jac=jac, max_nfev=30,
                                     ftol=2e-13, xtol=2e-13, gtol=2e-13)
        if not accurate(residual(fit.x)+yy, yy, 2e-10):
            return None
        terms = fit.x[d+1:].reshape(rank, d+1)
    val = fit.x[0]+np.sum(fit.x[1:d+1])/2
    for term, sign in zip(terms, signs):
        val += sign*abs_ridge_integral(term[:d], term[d])
    def predict(xnew):
        return scale*(fit.x[0]+xnew @ fit.x[1:d+1] +
                      np.abs(xnew @ terms[:, :d].T-terms[:, d]) @ signs)
    return float(val*scale), predict


def gaussian_mixture_fit(x, y, time_limit=8.0, differential=None, oracle=None):
    if np.any(y < 0):
        return None
    nonzero = y > 1e-280
    x, y = x[nonzero], y[nonzero]
    if len(y) < 60:
        return None
    started = time.monotonic()
    d = x.shape[1]
    scale = max(np.max(y), 1e-290)
    logs = np.log(y/scale)
    if np.ptp(logs) < 1e-3:
        return None
    z = x-0.5
    B = np.column_stack((np.ones(len(x)), z, z*z))
    width = B.shape[1]
    rng = np.random.default_rng(8913+d)

    def run(init, max_nfev=90):
        rank = len(init)
        def residual(flat):
            check_time(2)
            if time.monotonic()-started > time_limit:
                raise StopWork()
            p = flat.reshape(rank, width)
            return special.logsumexp(B @ p.T, axis=1)-logs
        def jac(flat):
            p = flat.reshape(rank, width)
            q = B @ p.T
            prob = np.exp(q-special.logsumexp(q, axis=1)[:, None])
            return (prob[:, :, None]*B[:, None, :]).reshape(len(x), -1)
        lower = np.tile(np.r_[-15000, np.full(d, -30000), np.full(d, -30000)], rank)
        upper = np.tile(np.r_[15000, np.full(d, 30000), np.zeros(d)], rank)
        init = np.clip(init.ravel(), lower+1e-7, upper-1e-7)
        fit = optimize.least_squares(residual, init, jac=jac, bounds=(lower, upper),
                  max_nfev=max_nfev, ftol=2e-13, xtol=2e-13, gtol=2e-13)
        err = residual(fit.x)
        return fit.x.reshape(rank, width), err

    def make_model(fitted):
        value = sum(exp_quad_integral(p[0]+math.log(scale), p[1:d+1], p[1+d:])
                    for p in fitted)
        def predict(xnew, fitted=fitted.copy()):
            zz = xnew-0.5
            bb = np.column_stack((np.ones(len(xnew)), zz, zz*zz))
            return scale*np.exp(special.logsumexp(bb @ fitted.T, axis=1))
        return float(value), predict

    global_q = linalg.lstsq(B, logs, lapack_driver="gelsy")[0]
    global_q[1+d:] = np.minimum(global_q[1+d:], -0.5)
    best = None
    besterr = np.inf
    try:
        # Logarithmic finite differences identify locally dominant Gaussian
        # terms even when all observed values are deep in their tails.
        if differential is not None:
            cx, cy, cg, ch = differential
            h = .0002
            yp = cy[:, None]+h*cg+.5*h*h*ch
            ym = cy[:, None]-h*cg+.5*h*h*ch
            pool = []
            for i in range(len(cx)):
                if cy[i] <= 1e-280 or np.any(yp[i] <= 1e-280) or np.any(ym[i] <= 1e-280):
                    continue
                lv = math.log(cy[i]/scale)
                lp, lm = np.log(yp[i]/scale), np.log(ym[i]/scale)
                a = (lp+lm-2*lv)/(2*h*h)
                if np.any(a >= -1e-5):
                    continue
                zc = cx[i]-.5
                b = (lp-lm)/(2*h)-2*a*zc
                c = lv-b @ zc-a @ (zc*zc)
                p = np.r_[c, b, a]
                error = B @ p-logs
                if np.max(error) > .03:
                    continue
                score = int(np.sum(np.abs(error) < .003))
                if score >= max(5, len(x)//100):
                    pool.append((score, p))
            pool.sort(key=lambda entry: -entry[0])
            init = []
            for score, p in pool:
                if not init or min(rms(B @ (p-q)) for q in init) > .03:
                    init.append(p)
                if len(init) >= 7:
                    break
            if init and oracle is not None:
                for _ in range(5):
                    predicted = special.logsumexp(B @ np.asarray(init).T, axis=1)
                    gaps = logs-predicted
                    j = int(np.argmax(gaps))
                    if gaps[j] < .01 or len(init) >= 8 or oracle.remaining < 300:
                        break
                    site = differential_samples(oracle, x[j:j+1], y[j:j+1], 1)
                    sx, sy, sg, sh = site
                    sp = sy[:, None]+h*sg+.5*h*h*sh
                    sm = sy[:, None]-h*sg+.5*h*h*sh
                    if sy[0] <= 1e-280 or np.any(sp <= 1e-280) or np.any(sm <= 1e-280):
                        break
                    lv = math.log(sy[0]/scale)
                    lp, lm = np.log(sp[0]/scale), np.log(sm[0]/scale)
                    a = (lp+lm-2*lv)/(2*h*h)
                    zc = sx[0]-.5
                    b = (lp-lm)/(2*h)-2*a*zc
                    c = lv-b @ zc-a @ (zc*zc)
                    p = np.r_[c, b, a]
                    if np.any(a >= -1e-5) or np.max(B @ p-logs) > .1:
                        break
                    if min(rms(B @ (p-q)) for q in init) < .03:
                        break
                    init.append(p)
            if init:
                trace("local Gaussian starts", len(init), [a[0] for a in pool])
                fitted, err = run(np.array(init), max_nfev=65)
                best, besterr = fitted, rms(err)
                if np.max(np.abs(err)) < 3e-8:
                    trace("local Gaussian terms", len(fitted), "residual", besterr)
                    return make_model(fitted)
        for rank in (2, 3, 4, 5):
            if time.monotonic()-started > time_limit*0.8:
                break
            for attempt in range(2 if rank < 4 else 1):
                # Spatial partitions supply distinct local quadratic models.
                centers = [int(np.argmax(logs))]
                for _ in range(1, rank):
                    distances = np.min(cdist(x, x[centers])**2, axis=1)
                    weight = distances*np.exp((logs-np.max(logs))/max(np.ptp(logs)*0.22, 1))
                    if attempt:
                        weight *= rng.uniform(0.4, 1.6, len(x))
                    centers.append(int(np.argmax(weight)))
                labels = np.argmin(cdist(x, x[centers]), axis=1)
                init = []
                for k in range(rank):
                    ids = labels == k
                    if np.sum(ids) < width+3:
                        ids = np.argsort(np.linalg.norm(x-x[centers[k]], axis=1))[:max(width+5, len(x)//rank)]
                    q = linalg.lstsq(B[ids], logs[ids], lapack_driver="gelsy")[0]
                    q[1+d:] = np.minimum(q[1+d:], -0.5)
                    init.append(q)
                if best is not None and len(best) == rank-1 and attempt == 0:
                    pred = special.logsumexp(B @ best.T, axis=1)
                    center = x[np.argmax(logs-pred)]
                    q = global_q.copy()
                    curvature = np.maximum(-np.mean(best[:, 1+d:], axis=0), 2)
                    q[1+d:] = -curvature
                    q[1:d+1] = 2*curvature*(center-0.5)
                    q[0] = np.max(logs)+np.sum(q[1+d:]*(center-0.5)**2)
                    init = np.vstack((best, q))
                fitted, err = run(np.asarray(init))
                loss = rms(err)
                if loss < besterr:
                    best, besterr = fitted, loss
                if np.max(np.abs(err)) < 3e-8:
                    trace("gaussian mixture", rank, "residual", loss)
                    return make_model(fitted)
    except (StopWork, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        trace("Gaussian fit stopped", type(exc).__name__, str(exc), "best", besterr)
    if best is not None and besterr < 2e-5:
        trace("approximate Gaussian model", besterr)
        return make_model(best)
    trace("Gaussian fit residual", besterr)
    return None


def gp_control(xt, yt, xv, yv):
    """Analytically integrated kernel surrogate plus an independent residual."""
    check_time(2)
    # Nearly coincident derivative probes add cost but little interpolation
    # information, so use just one point per small grid cell.
    _, ids = np.unique(np.round(xt, 3), axis=0, return_index=True)
    ids = np.sort(ids)
    xt, yt = xt[ids], yt[ids]
    if len(xt) > 850:
        ids = np.linspace(0, len(xt)-1, 850, dtype=int)
        xt, yt = xt[ids], yt[ids]
    scale = max(np.max(np.abs(yt)), np.max(np.abs(yv)), 1e-290)
    yt, yv = yt/scale, yv/scale
    d = xt.shape[1]
    Bt, means = poly_design(xt, 2)
    Bv, _ = poly_design(xv, 2)
    # Length scales are selected using held-out training points, never using
    # the residual sample that supplies the final correction.
    cut = max(50, int(len(xt)*0.75))
    xa, xb = xt[:cut], xt[cut:]
    ya, yb = yt[:cut], yt[cut:]
    if len(yb) < 10:
        return float(np.mean(yv)*scale)
    trend = linalg.lstsq(Bt[:cut], ya, lapack_driver="gelsy")[0]
    ra, rb = ya-Bt[:cut] @ trend, yb-Bt[cut:] @ trend

    def kernel(xa, xb, ell, kind):
        if kind == "rbf":
            return np.exp(-0.5*cdist(xa/ell, xb/ell, "sqeuclidean"))
        mat = np.ones((len(xa), len(xb)))
        for j in range(d):
            rr = np.abs(xa[:, None, j]-xb[None, :, j])*math.sqrt(5)/ell
            mat *= (1+rr+rr*rr/3)*np.exp(-rr)
        return mat

    def kernel_mean(xa, ell, kind):
        if kind == "rbf":
            return np.prod(ell*math.sqrt(math.pi/2) *
                           (special.erf(xa/(math.sqrt(2)*ell))+
                            special.erf((1-xa)/(math.sqrt(2)*ell))), axis=1)
        a = math.sqrt(5)/ell
        z1, z2 = a*xa, a*(1-xa)
        q = (16-(z1*z1+5*z1+8)*np.exp(-z1)-
                (z2*z2+5*z2+8)*np.exp(-z2))/(3*a)
        return np.prod(q, axis=1)

    best = None
    for kind, ell, noise in [
            ("rbf", 0.12, 1e-10), ("rbf", 0.24, 1e-10),
            ("rbf", 0.48, 1e-9), ("rbf", 0.85, 1e-8),
            ("matern", 0.24, 1e-9), ("matern", 0.6, 1e-9)]:
        if time.monotonic()-START > 23:
            break
        try:
            K = kernel(xa, xa, ell, kind)
            K.flat[::len(xa)+1] += noise
            factor = linalg.cho_factor(K, check_finite=False)
            alpha = linalg.cho_solve(factor, ra, check_finite=False)
            pred = kernel(xb, xa, ell, kind) @ alpha
            loss = rms(pred-rb)
            if best is None or loss < best[0]:
                best = (loss, kind, ell, noise)
        except linalg.LinAlgError:
            continue
    if best is None:
        return float(np.mean(yv)*scale)
    _, kind, ell, noise = best
    K = kernel(xt, xt, ell, kind)
    K.flat[::len(xt)+1] += noise
    factor = linalg.cho_factor(K, check_finite=False)
    # Generalized least squares preserves polynomial reproduction.
    KI = linalg.cho_solve(factor, np.column_stack((yt, Bt)), check_finite=False)
    coef = linalg.lstsq(Bt.T @ KI[:, 1:], Bt.T @ KI[:, 0],
                       lapack_driver="gelsy")[0]
    alpha = KI[:, 0]-KI[:, 1:] @ coef
    integral = means @ coef+kernel_mean(xt, ell, kind) @ alpha
    pred = Bv @ coef+kernel(xv, xt, ell, kind) @ alpha
    result = float((integral+np.mean(yv-pred))*scale)
    trace("gp", kind, ell, "cv", best[0], "residual", rms(yv-pred))
    if not np.isfinite(result) or rms(yv-pred) > 3*max(np.std(yv), 1e-12):
        return float(np.mean(yv)*scale)
    return result


def finish_model(oracle, model):
    value, predict = model
    if oracle.remaining < 16:
        return value
    xx = qmc.Sobol(oracle.d, scramble=True,
                  seed=(oracle.seed+49979687) % 2**32).random(oracle.remaining)
    yy = oracle(xx)
    prediction = predict(xx)
    correction = stable_mean(yy-prediction)
    result = float(value+correction)
    oracle.fallback = stable_mean(yy)
    if np.isfinite(result):
        return result
    return oracle.fallback


def solve(oracle):
    d = oracle.d
    rng = qmc.Sobol(d, scramble=True, seed=oracle.seed)
    n0 = min(128, oracle.budget)
    x = rng.random(n0)
    y = oracle(x)
    oracle.fallback = stable_mean(y)
    if oracle.budget <= 128:
        return oracle.fallback
    if d > 12:
        more = oracle(rng.random(oracle.remaining))
        return stable_mean(np.r_[y, more])
    probes = np.array([np.zeros(d), np.ones(d), np.full(d, 0.5)])
    py = oracle(probes)
    x, y = np.vstack((x, probes)), np.r_[y, py]
    result = simple_global(x, y)
    if result is not None:
        trace("global")
        return result
    result = separable(oracle, x, y)
    if result is not None:
        return result
    check_time()
    dif = differential_samples(oracle, x, y)
    initial_dif = dif
    xx, yy = oracle.data()
    positive = y[y > 1e-280]
    peaklike = (np.all(y >= 0) and len(positive) > len(y)*.5 and
                np.ptp(np.log(positive)) > 35)
    result = None if peaklike else cosine_fit(xx, yy, *dif)
    if result is not None:
        trace("cosine")
        return result
    result = None if peaklike else periodic_fit(xx, yy, *dif)
    if result is not None:
        vx = rng.random(min(128, oracle.remaining))
        vy = oracle(vx)
        if accurate(result[1](vx), vy, 5e-11):
            return finish_model(oracle, result)
    if not peaklike and np.median(np.max(np.abs(dif[3]), axis=1)) > max(np.max(np.abs(y)), 1e-290)*1e-4:
        result = multivariate_fourier(oracle, x, y, dif[2])
        if result is not None:
            vx = rng.random(min(128, oracle.remaining))
            vy = oracle(vx)
            if accurate(result[1](vx), vy, 5e-10):
                return finish_model(oracle, result)
    result = None if peaklike else hinge_fit(xx, yy, *dif)
    if result is not None:
        vertices = cube_vertices(d, oracle.seed)
        vx = np.vstack((rng.random(128), vertices))
        vy = oracle(vx)
        if accurate(result[1](vx), vy, 5e-11):
            trace("hinges validated")
            return finish_model(oracle, result)
        trace("hinge validation failed")
    else:
        vx, vy = None, None

    # Extra derivative sites expose activation regions of small hinge terms.
    scale = max(np.max(np.abs(y)), 1e-290)
    if not peaklike and np.median(np.max(np.abs(dif[3]), axis=1))/scale < 1e-3:
        if vx is None:
            vertices = cube_vertices(d, oracle.seed)
            vx = np.vstack((rng.random(128), vertices))
            vy = oracle(vx)
        extra_count = min(max(18, 96//d), (oracle.remaining-256)//(2*d))
        if extra_count >= 8:
            old_count = len(dif[0])
            dif = extend_piecewise(oracle, dif, extra_count)
            if len(dif[0]) < old_count+4:
                more = differential_samples(oracle, vx, vy,
                        min(extra_count, max(0, (oracle.remaining-256)//(2*d))))
                dif = tuple(np.concatenate((a, b), axis=0) for a, b in zip(dif, more))
            xx, yy = oracle.data()
            result = hinge_fit(xx, yy, *dif)
            if result is not None:
                tx = rng.random(min(128, oracle.remaining))
                ty = oracle(tx)
                if accurate(result[1](tx), ty, 5e-11):
                    trace("hinges extended validated")
                    return finish_model(oracle, result)

    # Half of the budget is held back for an independent digital-net residual.
    reserve = min(1024, max(128, oracle.budget//2))
    ntrain = max(0, oracle.remaining-reserve)
    if ntrain:
        newx = rng.random(ntrain)
        newy = oracle(newx)
        oracle.fallback = stable_mean(np.r_[y, newy])
    xt, yt = oracle.data()
    result = simple_global(xt, yt)
    if result is not None:
        return result
    result = None if peaklike else hinge_fit(xt, yt, *dif)
    if result is not None:
        vx = rng.random(min(128, oracle.remaining))
        vy = oracle(vx)
        if accurate(result[1](vx), vy, 5e-11):
            trace("hinges refined validated")
            return finish_model(oracle, result)
    if time.monotonic()-START < 15:
        result = gaussian_mixture_fit(xt, yt, min(10, 20-(time.monotonic()-START)),
                                      differential=initial_dif, oracle=oracle)
        if result is not None:
            vx = rng.random(min(128, oracle.remaining))
            vy = oracle(vx)
            if accurate(result[1](vx), vy, 3e-6):
                return finish_model(oracle, result)
    nv = oracle.remaining
    if nv <= 0:
        return oracle.fallback
    xv = qmc.Sobol(d, scramble=True, seed=(oracle.seed+104729) % 2**32).random(nv)
    yv = oracle(xv)
    oracle.fallback = stable_mean(yv)
    if nv == 0 or time.monotonic()-START > 24:
        return oracle.fallback
    return gp_control(xt, yt, xv, yv)


def main():
    global START
    START = time.monotonic()
    init = json.loads(sys.stdin.readline())
    oracle = Oracle(init)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(StopWork()))
    signal.setitimer(signal.ITIMER_REAL, 26)
    try:
        result = solve(oracle)
    except Exception as exc:
        trace("fallback", type(exc).__name__, str(exc))
        result = oracle.fallback
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    if result is None or not np.isfinite(result):
        if oracle.ys:
            result = stable_mean(oracle.ys[0])
        else:
            raise RuntimeError("no valid oracle observations")
    print(json.dumps({"type": "result", "estimate": float(result)},
                     separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
