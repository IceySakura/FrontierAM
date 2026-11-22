#!/usr/bin/env python3
"""Budgeted black-box integration with checked surrogate models."""
import sys
import time

START = time.monotonic()
import json
import math
import itertools
import warnings
import signal

import numpy as np
from scipy import linalg, optimize, special
from scipy.interpolate import CubicSpline, BarycentricInterpolator
from scipy.stats import qmc
from numpy.polynomial.legendre import leggauss, legvander
import ridges
import projection
import ridgepoly
import tensor
import spectral
from sampling import average as qmc_average
from sampling import mean as stable_mean, standard_deviation as stable_std
from sampling import root_mean_square as stable_rms
from sampling import query_uniform, uniform_values

warnings.filterwarnings("ignore", category=RuntimeWarning)
ACTIVE_ORACLE = None


class ComputationDeadline(BaseException):
    pass


def deadline_handler(_signum, _frame):
    if ACTIVE_ORACLE is not None:
        if not ACTIVE_ORACLE.computing:
            return
        if ACTIVE_ORACLE.in_query:
            ACTIVE_ORACLE.expired = True
            return
    raise ComputationDeadline()


class Oracle:
    def __init__(self, init):
        self.d = int(init["dimension"])
        self.budget = min(2048, int(init["budget"]))
        self.seed = int(init.get("seed", 0)) % (2**32)
        self.used = 0
        self.computing = True
        self.in_query = False
        self.expired = False
        self.xs = []
        self.ys = []
        bounds = np.asarray(init.get("bounds", [[0, 1]] * self.d), float)
        self.lo = bounds[:, 0]
        self.width = bounds[:, 1] - self.lo

    @property
    def remaining(self):
        return self.budget - self.used

    def ask(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, self.d)
        if len(points) > self.remaining:
            raise ValueError("query budget exhausted")
        points = np.clip(points, 0.0, 1.0)
        result = []
        for k in range(0, len(points), 512):
            p = points[k:k + 512]
            physical = self.lo + self.width * p
            self.in_query = True
            try:
                print(json.dumps({"type": "query", "points": physical.tolist()},
                                 separators=(",", ":"), allow_nan=False), flush=True)
                answer = json.loads(sys.stdin.readline())
                y = np.asarray(answer["values"], dtype=float)
                if len(y) != len(p) or not np.all(np.isfinite(y)):
                    raise ValueError("invalid oracle response")
                self.used += len(p)
                self.xs.append(p.copy())
                self.ys.append(y.copy())
                result.append(y)
            finally:
                self.in_query = False
            if self.expired:
                raise ComputationDeadline()
        return np.concatenate(result) if result else np.empty(0)

    def finish(self, value):
        value = float(value) * float(np.prod(self.width))
        if not np.isfinite(value):
            value = qmc_average(uniform_values(self, self.ys[0]))*float(np.prod(self.width))
        print(json.dumps({"type": "result", "estimate": value},
                         separators=(",", ":"), allow_nan=False), flush=True)


def remaining_time():
    return 26.0 - (time.monotonic() - START)


def rms(y):
    return stable_rms(y)


def total_indices(d, degree):
    result = []
    def rec(prefix, left, n):
        if n == 1:
            result.append(tuple(prefix + [left]))
        else:
            for j in range(left + 1):
                rec(prefix + [j], left - j, n - 1)
    for k in range(degree + 1):
        rec([], k, d)
    return result


def polynomial_matrix(x, indices):
    degree = max(max(i) for i in indices)
    vv = [legvander(2 * x[:, j] - 1, degree) for j in range(x.shape[1])]
    a = np.ones((len(x), len(indices)))
    for j in range(x.shape[1]):
        a *= vv[j][:, [i[j] for i in indices]]
    return a


def exact_polynomial(x, y):
    n, d = x.shape
    scale = max(np.max(np.abs(y)), 1e-300)
    yy = y / scale
    offset = stable_mean(yy)
    yy = yy-offset
    for degree in range(1, 7):
        indices = total_indices(d, degree)
        if len(indices) > n*3//4-16:
            break
        a = polynomial_matrix(x, indices)
        nt = n * 3 // 4
        coef = linalg.lstsq(a[:nt], yy[:nt], cond=1e-12,
                            lapack_driver="gelsy")[0]
        residual = a @ coef - yy
        if np.max(np.abs(residual)) < 3e-14:
            coef = linalg.lstsq(a, yy, cond=1e-13, lapack_driver="gelsy")[0]
            return float((coef[0]+offset) * scale)
    return None


def exponential_quadratic(x, y):
    # An exponential of an additive quadratic is integrated as 1-D factors.
    mask = np.abs(y) > 0
    if np.count_nonzero(mask) < max(32, 4 * (2 * x.shape[1] + 1)):
        return None
    sign = np.sign(y[mask][0])
    if np.any(np.sign(y[mask]) != sign):
        return None
    xx = x[mask] - 0.5
    scale = float(np.max(np.abs(y)))
    ratios = np.abs(y[mask])/scale
    yy = np.log(np.maximum(ratios, np.nextafter(0.0, 1.0)))
    tiny = ratios == 0
    yy[tiny] = np.log(np.abs(y[mask][tiny]))-math.log(scale)
    a = np.column_stack((np.ones(len(xx)), xx, xx**2))
    nt = len(xx) * 3 // 4
    coef = linalg.lstsq(a[:nt], yy[:nt], lapack_driver="gelsy")[0]
    if np.max(np.abs(a @ coef - yy)) > 2e-14*(1+np.max(np.abs(yy))):
        return None
    if np.count_nonzero(mask) != len(y):
        xp = x[~mask] - 0.5
        predicted = coef[0] + xp @ coef[1:1+x.shape[1]] + xp**2 @ coef[1+x.shape[1]:]
        if np.any(predicted+math.log(scale) > -680):
            return None
    coef = linalg.lstsq(a, yy, lapack_driver="gelsy")[0]
    d = x.shape[1]
    ans = coef[0]
    z, w = leggauss(96)
    for b, c in zip(coef[1:1+d], coef[1+d:]):
        if c < -1e-7:
            precision = -c
            center = b / (2 * precision)
            lo = math.sqrt(2 * precision) * (-0.5 - center)
            hi = math.sqrt(2 * precision) * (0.5 - center)
            if lo > 0:
                loga = special.log_ndtr(-lo)
                logb = special.log_ndtr(-hi)
            else:
                loga = special.log_ndtr(hi)
                logb = special.log_ndtr(lo)
            diff = logb - loga
            if diff < -1e-12:
                logprob = loga + math.log(-math.expm1(diff))
                ans += b*b/(4*precision) + 0.5*math.log(math.pi/precision) + logprob
                continue
        vals = b * z / 2 + c * z*z / 4
        ans += special.logsumexp(vals, b=w/2)
    if ans < 700:
        return float(sign * scale * math.exp(ans))
    if ans+math.log(scale) < 700:
        return float(sign*math.exp(ans+math.log(scale)))
    return None


def smooth_tensor_rule(oracle, x, y):
    d = oracle.d
    if d < 2 or oracle.remaining < 1200:
        return None
    eligible = np.flatnonzero(np.min(np.minimum(x, 1-x), axis=1) > 0.025)
    if len(eligible) < 7:
        return None
    ids = eligible[:7]
    rng = np.random.default_rng(oracle.seed ^ 0x6C759E21)
    directions = rng.normal(size=(7, d))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    h = 0.01
    shifts = np.array([-2, -1, 1, 2])*h
    points = x[ids, None, :]+shifts[None, :, None]*directions[:, None, :]
    values = oracle.ask(points.reshape(-1, d)).reshape(7, 4)
    scale = max(np.max(np.abs(y)), 1e-300)
    values = values/scale
    fourth = values[:, 0]-4*values[:, 1]+6*(y[ids]/scale)-4*values[:, 2]+values[:, 3]
    if not np.all(np.isfinite(fourth)):
        return None
    if np.count_nonzero(np.abs(fourth) > 5e-13) < 5:
        return None
    if np.max(np.abs(fourth)) > 25*max(np.median(np.abs(fourth)), 1e-15):
        return None
    frequency = (np.max(np.abs(fourth))/max(stable_std(y/scale), 1e-12))**0.25/h
    oracle.analytic = True
    oracle.characteristic_frequency = frequency
    if d not in (2, 3):
        return None
    if frequency > (50 if d == 2 else 12):
        return None
    n = max(1, int(oracle.remaining**(1/d)))
    while (n+1)**d <= oracle.remaining:
        n += 1
    nodes, weights = leggauss(n)
    nodes = (nodes+1)/2
    weights = weights.astype(np.longdouble)
    weights /= np.sum(weights)
    mesh = np.meshgrid(*([nodes]*d), indexing="ij")
    wm = np.meshgrid(*([weights]*d), indexing="ij")
    points = np.column_stack([m.ravel() for m in mesh])
    ww = np.prod(np.asarray(wm), axis=0).ravel()
    values = oracle.ask(points).astype(np.longdouble)
    return float(np.dot(values, ww))


# The nested 7/15 Gauss-Kronrod rule permits batched local error estimates.
GK_X = np.array([-0.9914553711208126, -0.9491079123427585,
                 -0.8648644233597691, -0.7415311855993945,
                 -0.5860872354676911, -0.4058451513773972,
                 -0.2077849550078985, 0.0, 0.2077849550078985,
                 0.4058451513773972, 0.5860872354676911,
                 0.7415311855993945, 0.8648644233597691,
                 0.9491079123427585, 0.9914553711208126])
GK_W = np.array([0.0229353220105292, 0.0630920926299786,
                 0.1047900103222502, 0.1406532597155259,
                 0.1690047266392679, 0.1903505780647854,
                 0.2044329400752989, 0.2094821410847278,
                 0.2044329400752989, 0.1903505780647854,
                 0.1690047266392679, 0.1406532597155259,
                 0.1047900103222502, 0.0630920926299786,
                 0.0229353220105292])
G_W = np.zeros(15)
G_W[[1, 3, 5, 7, 9, 11, 13]] = [0.1294849661688697,
    0.2797053914892767, 0.3818300505051189, 0.4179591836734694,
    0.3818300505051189, 0.2797053914892767, 0.1294849661688697]


def check_product(oracle, x, y):
    d = oracle.d
    pivot = int(np.argmax(np.abs(y)))
    f0 = float(y[pivot])
    if f0 == 0 or abs(f0) < 1e-280 or oracle.remaining < d * 12 + 30*d:
        return None
    anchor = x[pivot]
    ids = np.linspace(0, len(x) - 1, 12, dtype=int)
    tests = x[ids]
    pts = np.tile(anchor, (12*d, 1))
    for j in range(d):
        pts[j*12:(j+1)*12, j] = tests[:, j]
    slices = oracle.ask(pts).reshape(d, 12) / f0
    oracle.product_slices = slices
    with np.errstate(over="ignore", invalid="ignore"):
        pred = np.prod(slices.astype(np.longdouble), axis=0) * f0
    err = np.max(np.abs(pred-y[ids]))
    mode = "product"
    if not np.isfinite(err) or err > 2e-13*max(np.max(np.abs(y[ids])), abs(f0)*1e-5):
        additive = f0*(np.sum(slices, axis=0)-d+1)
        error = np.max(np.abs(additive-y[ids]))
        if np.isfinite(error) and error < 2e-13*max(np.max(np.abs(y[ids])), abs(f0)*1e-5):
            mode = "sum"
        else:
            return None
    strongest = np.argmax(np.abs(slices-1), axis=1)
    coordinates = np.array([tests[strongest[j], j] for j in range(d)])
    factors = slices[np.arange(d), strongest]
    masks = []
    for j in range(d-1):
        mask = np.zeros(d, dtype=bool)
        mask[j:j+2] = True
        masks.append(mask)
    masks.append(np.ones(d, dtype=bool))
    pp = np.asarray([np.where(mask, coordinates, anchor) for mask in masks])
    actual = oracle.ask(pp)
    expected = []
    for mask in masks:
        fs = np.where(mask, factors, 1).astype(np.longdouble)
        expected.append(f0*(np.prod(fs) if mode == "product" else np.sum(fs)-d+1))
    expected = np.asarray(expected)
    if np.max(np.abs(expected-actual)) > 3e-13*max(np.max(np.abs(actual)), abs(f0)):
        return None
    return anchor, f0, mode


def integrate_product(oracle, anchor, f0, mode="product"):
    d = oracle.d
    discrete = getattr(oracle, "product_slices", None)
    if discrete is not None and all(len(np.unique(v)) <= 4 for v in discrete):
        value = integrate_discrete_product(oracle, anchor, f0, mode)
        if value is not None:
            return value
    intervals = [[] for _ in range(d)]

    def evaluate(regions):
        pp = []
        for j, a, b in regions:
            p = np.tile(anchor, (15, 1))
            p[:, j] = (a+b)/2 + (b-a)/2*GK_X
            pp.append(p)
        yy = oracle.ask(np.vstack(pp)).reshape(-1, 15) / f0
        result = []
        for (j, a, b), v in zip(regions, yy):
            factor = (b-a)/2
            value = factor * np.dot(GK_W, v)
            error = factor * abs(np.dot(GK_W-G_W, v))
            result.append((j, a, b, value, error))
        return result

    for reg in evaluate([(j, 0.0, 1.0) for j in range(d)]):
        intervals[reg[0]].append(reg)
    while oracle.remaining >= 30 and remaining_time() > 2:
        sums = np.array([sum(r[3] for r in ints) for ints in intervals])
        totals = np.array([sum(r[4] for r in ints) for ints in intervals])
        scores = totals / np.maximum(np.abs(sums), 1e-300)
        if np.sum(scores) < 2e-14:
            break
        # Refine several independent intervals in one protocol round trip.
        choices = []
        for j in range(d):
            for k, r in enumerate(intervals[j]):
                choices.append((r[4]/max(abs(sums[j]), 1e-300), j, k))
        choices.sort(reverse=True)
        active = sum(c[0] >= choices[0][0]*0.1 for c in choices)
        take = min(8, oracle.remaining//30, active)
        chosen = choices[:take]
        children = []
        for _, j, k in chosen:
            _, a, b, _, _ = intervals[j][k]
            mid = (a+b)/2
            children.extend([(j, a, mid), (j, mid, b)])
        for j in range(d):
            for k in sorted([k for _, jj, k in chosen if jj == j], reverse=True):
                del intervals[j][k]
        for r in evaluate(children):
            intervals[r[0]].append(r)
    sums = [sum(r[3] for r in ints) for ints in intervals]
    if mode == "sum":
        value = np.longdouble(f0)*(np.sum(np.asarray(sums, dtype=np.longdouble))-d+1)
    else:
        value = np.longdouble(f0) * np.prod(np.asarray(sums, dtype=np.longdouble))
    return float(value)


def integrate_discrete_product(oracle, anchor, f0, mode):
    d = oracle.d
    n = 129 if d <= 3 else 65
    if oracle.remaining < d*n+40:
        return None
    ts = np.linspace(0, 1, n)
    pp = np.tile(anchor, (n*d, 1))
    for j in range(d):
        pp[j*n:(j+1)*n, j] = ts
    vals = oracle.ask(pp).reshape(d, n)/f0
    if any(len(np.unique(v)) > 8 for v in vals):
        return None
    fixed = np.zeros(d, dtype=np.longdouble)
    regions = []
    for j in range(d):
        for a, b, fa, fb in zip(ts[:-1], ts[1:], vals[j, :-1], vals[j, 1:]):
            if fa == fb:
                fixed[j] += (b-a)*fa
            else:
                regions.append((j, a, b, fa, fb))
    while regions and oracle.remaining and remaining_time() > 2:
        regions.sort(key=lambda r: (r[2]-r[1])*abs(r[4]-r[3]), reverse=True)
        take = min(len(regions), oracle.remaining, 256)
        chosen = regions[:take]
        regions = regions[take:]
        pp = np.tile(anchor, (take, 1))
        for k, (j, a, b, _, _) in enumerate(chosen):
            pp[k, j] = (a+b)/2
        middle = oracle.ask(pp)/f0
        for (j, a, b, fa, fb), fm in zip(chosen, middle):
            mid = (a+b)/2
            if mid == a or mid == b or b-a < 2e-15:
                fixed[j] += (b-a)*(fa+fb)/2
                continue
            if fm == fa:
                fixed[j] += (mid-a)*fa
            else:
                regions.append((j, a, mid, fa, fm))
            if fm == fb:
                fixed[j] += (b-mid)*fb
            else:
                regions.append((j, mid, b, fm, fb))
    for j, a, b, fa, fb in regions:
        fixed[j] += (b-a)*(fa+fb)/2
    if mode == "sum":
        return float(np.longdouble(f0)*(sum(fixed)-d+1))
    return float(np.longdouble(f0)*np.prod(fixed))


def sample_gradients(oracle, x, n=24, order=2):
    count = 4 if order == 4 else 2
    n = min(n, oracle.remaining // (count*oracle.d))
    xx = np.clip(x[:n], 0.0001, 0.9999)
    h = 2**-17 if order == 4 else 2**-20
    pp = np.repeat(xx, count*oracle.d, axis=0)
    for j in range(oracle.d):
        pp[j*count::count*oracle.d, j] += h
        pp[j*count+1::count*oracle.d, j] -= h
        if order == 4:
            pp[j*count+2::count*oracle.d, j] += 2*h
            pp[j*count+3::count*oracle.d, j] -= 2*h
    yy = oracle.ask(pp).reshape(n, oracle.d, count)
    gradients = yy[:, :, 0]-yy[:, :, 1]
    if order == 4:
        gradients = (8*gradients-(yy[:, :, 2]-yy[:, :, 3]))/(12*h)
    else:
        gradients /= 2*h
    if not np.all(np.isfinite(gradients)):
        raise FloatingPointError("unrepresentable finite difference")
    return xx, gradients


def ridge_density(t, coefficients):
    weights = np.abs(np.asarray(coefficients, dtype=np.longdouble))
    weights = weights[weights > np.max(weights)*1e-14]
    weights /= np.sum(weights)
    weights = np.sort(weights)[::-1]
    tt = np.asarray(t, dtype=np.longdouble)
    tt = np.minimum(tt, 1-tt)
    if len(weights) == 1:
        return np.ones(len(tt))
    largest, rest = weights[0], weights[1:]
    n = len(rest)
    totals = np.r_[np.cumsum(rest[::-1])[::-1], np.longdouble(0)]
    moments = np.zeros((n+1, n+1), dtype=np.longdouble)
    moments[:, 0] = 1
    for i in range(n-1, -1, -1):
        for degree in range(2, n+1, 2):
            for k in range(0, degree+1, 2):
                moments[i, degree] += (math.comb(degree, k)*moments[i+1, degree-k]*
                                      (rest[i]/2)**k/(k+1))

    # Integrating out the largest remaining uniform avoids cancellation
    # between nearly identical subset sums when the widths are anisotropic.
    def truncated_moment(i, power, points):
        out = np.zeros(len(points), dtype=np.longdouble)
        positive = points >= totals[i]
        if np.any(positive):
            center = points[positive]-totals[i]/2
            value = np.zeros(len(center), dtype=np.longdouble)
            for k in range(0, power+1, 2):
                value += math.comb(power, k)*center**(power-k)*moments[i, k]
            out[positive] = value
        active = (points > 0) & ~positive
        if i < n and np.any(active):
            v = points[active]
            out[active] = (truncated_moment(i+1, power+1, v)-
                           truncated_moment(i+1, power+1, v-rest[i]))/(rest[i]*(power+1))
        return out

    def cdf(points):
        upper = points > totals[0]/2
        mirrored = np.minimum(points, totals[0]-points)
        value = truncated_moment(0, 0, mirrored)
        return np.where(upper, 1-value, value)

    out = (cdf(tt)-cdf(tt-largest))/largest
    return np.asarray(np.maximum(out, 0), dtype=float)


def smooth_ridge_quadrature(oracle, x, y, direction):
    d = oracle.d
    if oracle.remaining < len(x)+250:
        return None
    scale = max(np.max(np.abs(y)), 1e-300)
    w = direction/np.sum(np.abs(direction))
    orientation = np.where(w < 0, -1.0, 1.0)
    tx = 0.5+(x-0.5)@w
    projected_x = 0.5+(tx[:, None]-0.5)*orientation
    values = oracle.ask(projected_x)/scale
    residual = y/scale-values
    uniform_residual = np.delete(residual, getattr(oracle, "corner_indices", np.empty(0, dtype=int)))
    correction = qmc_average(uniform_residual)
    variance = max(np.std(y/scale), 1e-15)
    best_error = float(np.std(residual))
    for degree in (1, 2, 3):
        indices = total_indices(d, degree)
        if len(indices) > len(x)//2:
            break
        a = polynomial_matrix(x, indices)
        nt = len(x)*3//4
        c = linalg.lstsq(a[:nt], residual[:nt], cond=1e-12, lapack_driver="gelsy")[0]
        error = rms(a[nt:]@c-residual[nt:])
        if error < best_error:
            best_error = error
            c = linalg.lstsq(a, residual, cond=1e-12, lapack_driver="gelsy")[0]
            correction = float(c[0])
    if best_error > variance*2e-4:
        return None
    if d <= 6:
        breaks = np.zeros(1)
        for weight in np.abs(w):
            breaks = np.r_[breaks, breaks+weight]
        breaks = np.unique(np.r_[0, np.clip(breaks, 0, 1), 1])
    else:
        breaks = np.array([0.0, 1.0])
    lengths = np.diff(breaks)
    if 6*len(lengths) > oracle.remaining:
        breaks, lengths = np.array([0.0, 1.0]), np.ones(1)
    available = oracle.remaining-6*len(lengths)
    allocations = 6+np.floor(available*lengths).astype(int)
    left = oracle.remaining-int(np.sum(allocations))
    if left:
        order = np.argsort(available*lengths-np.floor(available*lengths))[::-1]
        allocations[order[:left]] += 1
    nodes, weights, rules = [], [], []
    for a, b, count in zip(breaks[:-1], breaks[1:], allocations):
        t, q = special.roots_legendre(int(count))
        nodes.append((a+b)/2+(b-a)/2*t)
        weights.append((b-a)/2*q)
        rules.append((t, q))
    t = np.concatenate(nodes)
    base_weights = np.concatenate(weights)
    q = base_weights*ridge_density(t, w)
    points = 0.5+(t[:, None]-0.5)*orientation
    actual = oracle.ask(points)/scale
    offset = float(np.mean(y/scale))
    if d > 1 and remaining_time() > 3:
        try:
            interpolants = []
            cursor = 0
            for xx, (roots, qw) in zip(nodes, rules):
                count = len(xx)
                yy = actual[cursor:cursor+count]-offset
                cursor += count
                unique, indices = np.unique(xx, return_index=True)
                if len(unique) == count:
                    bary_weights = (-1.0)**np.arange(count)*np.sqrt((1-roots*roots)*qw)
                    interpolant = BarycentricInterpolator(xx, yy, wi=bary_weights)
                    interpolants.append((interpolant, interpolant.derivative))
                elif len(unique) > 1:
                    interpolant = CubicSpline(unique, yy[indices])
                    interpolants.append((interpolant, interpolant.derivative()))
                else:
                    value = yy[0]
                    interpolants.append((lambda xx, c=value: np.full(len(xx), c),
                                          lambda xx: np.zeros(len(xx))))

            def evaluate_model(xx, derivative=False):
                intervals = np.clip(np.searchsorted(breaks, xx, side="right")-1, 0, len(interpolants)-1)
                result = np.empty(len(xx))
                for k in np.unique(intervals):
                    mask = intervals == k
                    result[mask] = interpolants[k][int(derivative)](xx[mask])
                return result

            fixed = int(np.argmax(np.abs(w)))
            ids = np.array([j for j in range(d) if j != fixed])
            z = x-0.5
            design = z-(tx-0.5)[:, None]*orientation
            beta = linalg.lstsq(design[:, ids], residual, lapack_driver="gelsy")[0]
            initial = np.r_[w[ids]/abs(w[fixed]), beta]
            lower = np.r_[np.where(orientation[ids] > 0, 0, -np.inf), np.full(d-1, -np.inf)]
            upper = np.r_[np.where(orientation[ids] > 0, np.inf, 0), np.full(d-1, np.inf)]

            def parameters(p):
                v = np.zeros(d)
                v[fixed] = orientation[fixed]
                v[ids] = p[:d-1]
                norm = float(np.dot(v, orientation))
                derivative = (np.eye(d)[:, ids]*norm-v[:, None]*orientation[ids])/norm**2
                return v/norm, derivative, p[d-1:]

            nt = len(x)*3//4
            def fun(p):
                if remaining_time() < 1.5:
                    raise TimeoutError()
                ww, _, bb = parameters(p)
                tt = 0.5+z[:nt]@ww
                aa = z[:nt]-(tt-0.5)[:, None]*orientation
                return (offset+evaluate_model(tt)+aa[:, ids]@bb-y[:nt]/scale)/variance

            def jac(p):
                ww, dw, bb = parameters(p)
                tt = 0.5+z[:nt]@ww
                aa = z[:nt]-(tt-0.5)[:, None]*orientation
                slope = evaluate_model(tt, True)-orientation[ids]@bb
                return np.column_stack(((z[:nt]@dw)*slope[:, None], aa[:, ids]))/variance

            fit = optimize.least_squares(fun, initial, jac=jac, bounds=(lower, upper),
                                        max_nfev=60, ftol=1e-13, xtol=1e-13, gtol=1e-13,
                                        x_scale="jac")
            refined, _, beta = parameters(fit.x)
            tt = 0.5+z@refined
            design = z-(tt-0.5)[:, None]*orientation
            predicted = offset+evaluate_model(tt)+design[:, ids]@beta
            if np.max(np.abs(predicted-y/scale)) < max(2e-11*variance, 5e-14):
                q = base_weights*ridge_density(t, refined)
                correction = 0.0
        except (ValueError, linalg.LinAlgError, FloatingPointError, TimeoutError):
            pass
    return float(scale*(offset+np.dot(q.astype(np.longdouble),
                                     (actual-offset).astype(np.longdouble))+correction))


def integrate_ridge(oracle, x, y, direction):
    if oracle.remaining < 350:
        return None
    d = oracle.d
    scale = max(np.max(np.abs(y)), 1e-300)
    centered = y/scale-np.mean(y/scale)
    spread = max(rms(centered), 1e-30)
    if getattr(oracle, "analytic", False) or np.max(np.abs(centered)) < 6*spread:
        answer = smooth_ridge_quadrature(oracle, x, y, direction)
        if answer is not None:
            return answer
    direction = direction/np.sum(np.abs(direction))
    orientation = np.where(direction < 0, -1.0, 1.0)
    n = 129 if oracle.remaining >= 650 else 65
    ts = np.linspace(0, 1, n)
    pp = 0.5+(ts[:, None]-0.5)*orientation
    values = oracle.ask(pp)/scale
    interpolator = CubicSpline(ts, values)
    projected = 0.5+(x-0.5)@direction
    residual = y/scale-interpolator(projected)
    linear = linalg.lstsq(x-0.5, residual, lapack_driver="gelsy")[0]
    residual -= (x-0.5)@linear
    deviation = max(float(np.std(y/scale)), 1e-12)
    if rms(residual) > deviation*0.003 or np.max(np.abs(residual)) > deviation*0.025:
        return None
    offset = float(values[n//2])
    weights = np.abs(direction)
    if d <= 4:
        breaks = np.zeros(1)
        for w in weights:
            breaks = np.r_[breaks, breaks+w]
        breaks = np.unique(np.clip(breaks, 0, 1))
    else:
        breaks = np.linspace(0, 1, 9)
    if (len(breaks)-1)*15 > oracle.remaining:
        breaks = np.linspace(0, 1, min(8, oracle.remaining//15)+1)
    def evaluate(regions):
        t = np.concatenate([(a+b)/2+(b-a)/2*GK_X for a, b in regions])
        p = 0.5+(t[:, None]-0.5)*orientation
        v = (oracle.ask(p)/scale-offset)*ridge_density(t, direction)
        v = v.reshape(-1, 15)
        out = []
        for (a, b), vv in zip(regions, v):
            value = (b-a)/2*np.dot(GK_W, vv)
            error = (b-a)/2*abs(np.dot(GK_W-G_W, vv))
            out.append((a, b, value, error))
        return out
    regions = evaluate(list(zip(breaks[:-1], breaks[1:])))
    while oracle.remaining >= 30 and remaining_time() > 3:
        error = sum(r[3] for r in regions)
        total = offset+sum(r[2] for r in regions)
        if error < max(2e-12*abs(total), 2e-14*deviation):
            break
        regions.sort(key=lambda r: r[3], reverse=True)
        n = min(4, oracle.remaining//30,
                sum(r[3] >= regions[0][3]*0.2 for r in regions))
        children = []
        for a, b, _, _ in regions[:n]:
            children.extend([(a, (a+b)/2), ((a+b)/2, b)])
        regions = regions[n:]+evaluate(children)
    return float(scale*(offset+sum(r[2] for r in regions)))


def fit_sinusoid(x, y, direction):
    z = x - 0.5
    offset = stable_mean(y)
    scale = max(stable_std(y), np.max(np.abs(y))*1e-14, 1e-300)
    yy = y/scale-offset/scale
    t = z @ direction
    frequencies = np.arange(0.25, 160.01, 0.5)
    phase = t[:, None] * frequencies[None, :]
    co, si = np.cos(phase), np.sin(phase)
    centered = yy - yy.mean()
    strength = ((co.T @ centered)**2 / np.maximum(np.sum(co*co, axis=0), 1e-15)
                + (si.T @ centered)**2 / np.maximum(np.sum(si*si, axis=0), 1e-15))
    order = np.argsort(strength)[::-1]
    choices = []
    for k in order:
        if all(abs(frequencies[k] - f) > 2 for f in choices):
            choices.append(frequencies[k])
        if len(choices) == 5:
            break
    best = None
    for omega in choices:
        if remaining_time() < 5:
            break
        w = direction * omega
        a = np.column_stack((np.ones(len(z)), np.cos(z @ w), np.sin(z @ w)))
        coeff = linalg.lstsq(a, yy, lapack_driver="gelsy")[0]
        theta = np.r_[coeff, w]
        def fun(p):
            u = z @ p[3:]
            return p[0] + p[1]*np.cos(u) + p[2]*np.sin(u) - yy
        def jac(p):
            u = z @ p[3:]
            co, si = np.cos(u), np.sin(u)
            return np.column_stack((np.ones(len(z)), co, si,
                                    z * (-p[1]*si+p[2]*co)[:, None]))
        fit = optimize.least_squares(fun, theta, jac=jac, method="lm",
                                    max_nfev=100, ftol=1e-14, xtol=1e-14, gtol=1e-14)
        error = rms(fit.fun)
        if best is None or error < best[0]:
            best = (error, fit.x)
        if error < 1e-11:
            break
    tolerance = max(3e-13, 3e-14*max(np.max(np.abs(y)), scale)/scale)
    if best is not None and best[0] < tolerance:
        p = best[1]
        return float(offset+scale * (p[0] + p[1]*np.prod(np.sinc(p[3:]/(2*np.pi)))))
    return None


def hinge_integral(w, offset):
    """Expectation of an affine absolute-value basis function."""
    positive = projection.positive_moment(w, offset, 1.0)
    if offset == 0:
        return 2*positive
    return positive+projection.positive_moment(w, -offset, 1.0)


def fit_hinges(x, y, gx, gradients):
    scale = max(np.max(np.abs(y)), 1e-300)
    yy = y/scale
    gg = gradients/scale
    gscale = max(rms(gg), 1e-14)
    clusters = []
    for row in gg:
        if not any(np.linalg.norm(row-other) < 2e-6*gscale for other in clusters):
            clusters.append(row)
    if len(clusters) < 2 or len(clusters) > len(gg)*0.8:
        return None
    dirs, counts = [], []
    for a, b in itertools.combinations(clusters, 2):
        delta = a-b
        norm = np.linalg.norm(delta)
        if norm < gscale*1e-6:
            continue
        v = delta/norm
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        matches = [k for k, u in enumerate(dirs) if abs(np.dot(v, u)) > 1-1e-8]
        if matches:
            counts[matches[0]] += 1
        else:
            dirs.append(v)
            counts.append(1)
    order = np.argsort(counts)[::-1][:min(40, len(dirs))]
    candidates = [dirs[k] for k in order]
    z, zg = x-0.5, gx-0.5
    residual = gg-gg.mean(axis=0)
    selected, knots = [], []
    for iteration in range(min(8, len(candidates))):
        best = None
        for v in candidates:
            if any(abs(np.dot(v, u)) > 1-1e-8 for u in selected):
                continue
            t = zg @ v
            ts = np.unique(t)
            for knot in (ts[1:]+ts[:-1])/2:
                signs = np.sign(t-knot)
                sc = signs-signs.mean()
                amplitude = np.sum((residual@v)*sc)/max(np.sum(sc*sc), 1e-20)
                improve = amplitude**2*np.sum(sc*sc)
                if best is None or improve > best[0]:
                    best = (improve, v, knot)
        if best is None:
            break
        _, v, knot = best
        selected.append(v)
        knots.append(knot)
        aa = np.zeros((gg.size, gg.shape[1]+len(selected)))
        for j in range(gg.shape[1]):
            aa[j::gg.shape[1], j] = 1
        for j, (u, k) in enumerate(zip(selected, knots)):
            aa[:, gg.shape[1]+j] = (np.sign(zg@u-k)[:, None]*u).ravel()
        coefficients = linalg.lstsq(aa, gg.ravel(), lapack_driver="gelsy")[0]
        residual = (gg.ravel()-aa@coefficients).reshape(gg.shape)
        if rms(residual) < gscale*1e-6:
            break
    if not selected or rms(residual) > gscale*1e-4:
        return None
    d, r = x.shape[1], len(selected)
    directions = np.asarray(selected)
    knots = np.asarray(knots)
    design = np.column_stack((np.ones(len(x)), z, np.abs(z@directions.T-knots)))
    cc = linalg.lstsq(design, yy, lapack_driver="gelsy")[0]
    signs = np.sign(cc[1+d:])
    signs[signs == 0] = 1
    weights = directions*np.abs(cc[1+d:, None])
    biases = knots*np.abs(cc[1+d:])
    p0 = np.r_[cc[:d+1], np.column_stack((weights, biases)).ravel()]
    def fun(p):
        params = p[d+1:].reshape(r, d+1)
        activation = z @ params[:, :d].T-params[:, d]
        return p[0]+z@p[1:d+1]+np.abs(activation)@signs-yy
    def jac(p):
        params = p[d+1:].reshape(r, d+1)
        activation = z @ params[:, :d].T-params[:, d]
        der = np.sign(activation)*signs
        out = np.empty((len(x), len(p)))
        out[:, :d+1] = np.column_stack((np.ones(len(x)), z))
        for k in range(r):
            out[:, d+1+k*(d+1):d+1+(k+1)*(d+1)] = np.column_stack((z, -np.ones(len(x))))*der[:, k, None]
        return out
    fit = optimize.least_squares(fun, p0, jac=jac, max_nfev=120, method="lm",
                                ftol=1e-12, xtol=1e-12, gtol=1e-12)
    if np.max(np.abs(fit.fun)) > 2e-9:
        return None
    p = fit.x
    params = p[d+1:].reshape(r, d+1)
    ans = p[0]
    for s, par in zip(signs, params):
        ans += s*hinge_integral(par[:d], par[d])
    return float(ans*scale)


def kernel_matrix(x, z, lengths, kind):
    if kind.startswith("radial"):
        r2 = np.zeros((len(x), len(z)))
        for j, ell in enumerate(lengths):
            r2 += ((x[:, j, None]-z[None, :, j])/ell)**2
        r = np.sqrt(r2)
        if kind == "radial5":
            r *= math.sqrt(5)
            return (1+r+r*r/3)*np.exp(-r)
        if kind == "radial3":
            r *= math.sqrt(3)
            return (1+r)*np.exp(-r)
        return np.exp(-r)
    result = np.ones((len(x), len(z)))
    for j, ell in enumerate(lengths):
        t = np.abs(x[:, j, None] - z[None, :, j]) / ell
        if kind == "gaussian":
            result *= np.exp(-0.5*t*t)
        elif kind == "matern5":
            t *= math.sqrt(5)
            result *= (1+t+t*t/3)*np.exp(-t)
        elif kind == "matern3":
            t *= math.sqrt(3)
            result *= (1+t)*np.exp(-t)
        else:
            result *= np.exp(-t)
    return result


def kernel_integral(x, lengths, kind):
    if kind.startswith("radial"):
        nu = 2.5 if kind == "radial5" else 1.5 if kind == "radial3" else 0.5
        nodes, weights = special.roots_genlaguerre(160, nu-1)
        result = np.zeros(len(x))
        weights /= special.gamma(nu)
        for t, weight in zip(nodes, weights):
            if weight < 1e-18:
                continue
            result += weight*kernel_integral(x, lengths*math.sqrt(t/nu), "gaussian")
        return result
    result = np.ones(len(x))
    for j, ell in enumerate(lengths):
        if kind == "gaussian":
            val = ell*math.sqrt(math.pi/2)*(special.erf(x[:, j]/(math.sqrt(2)*ell))+
                                           special.erf((1-x[:, j])/(math.sqrt(2)*ell)))
        else:
            c = (math.sqrt(5) if kind == "matern5" else math.sqrt(3) if kind == "matern3" else 1)/ell
            a, b = c*x[:, j], c*(1-x[:, j])
            if kind == "matern5":
                val = (16/3 - (a*a/3+5*a/3+8/3)*np.exp(-a)
                       - (b*b/3+5*b/3+8/3)*np.exp(-b))/c
            elif kind == "matern3":
                val = (4-(a+2)*np.exp(-a)-(b+2)*np.exp(-b))/c
            else:
                val = (2-np.exp(-a)-np.exp(-b))/c
        result *= val
    return result


def gp_fit(x, y, lengths, kind, jitter=1e-10):
    k = kernel_matrix(x, x, lengths, kind)
    k.flat[::len(x)+1] += jitter
    cf = linalg.cho_factor(k, lower=True, overwrite_a=True, check_finite=False)
    design = np.column_stack((np.ones(len(x)), x-0.5))
    solved = linalg.cho_solve(cf, np.column_stack((y, design)), check_finite=False)
    beta = linalg.lstsq(design.T@solved[:, 1:], design.T@solved[:, 0],
                       cond=1e-12, lapack_driver="gelsy")[0]
    alpha = solved[:, 0]-solved[:, 1:]@beta
    return alpha, beta


def fallback_cubature(x, y, sample_values=None):
    n, d = x.shape
    sample_values = y if sample_values is None else sample_values
    scale = max(np.max(np.abs(y)), 1e-300)
    yy = y/scale
    nt = min(1152, n*3//4)
    valid = np.arange(nt, min(n, nt+256))
    if len(valid) < 16:
        return qmc_average(sample_values)
    xt, yt = x[:nt], yy[:nt]
    xv, yv = x[valid], yy[valid]
    candidates = []
    for degree in range(1, 13):
        indices = total_indices(d, degree)
        if len(indices) > nt//2 or len(indices) > 600:
            break
        a = polynomial_matrix(xt, indices)
        c = linalg.lstsq(a, yt, cond=1e-10, lapack_driver="gelsy")[0]
        residual = polynomial_matrix(xv, indices)@c-yv
        candidates.append((rms(residual), "poly", indices))
    # Marginal regression supplies conservative anisotropy estimates.
    indices = total_indices(d, 2)
    a = polynomial_matrix(xt, indices)
    c = linalg.lstsq(a, yt, lapack_driver="gelsy")[0]
    importance = np.zeros(d)
    for coeff, idx in zip(c, indices):
        for j in range(d):
            if idx[j]:
                importance[j] += coeff*coeff
    imp = np.sqrt(importance + max(np.mean(importance), 1e-8)*0.15)
    anisotropy = np.clip(np.sqrt(np.mean(imp)/np.maximum(imp, 1e-10)), 0.4, 3)
    for kind in ("gaussian", "matern5", "matern3", "radial5", "radial3", "radial1"):
        for ell in (0.12, 0.3, 0.7, 1.5):
            if remaining_time() < 7:
                break
            lengths = anisotropy * ell
            try:
                alpha, beta = gp_fit(xt, yt, lengths, kind)
                pred = beta[0]+(xv-0.5)@beta[1:]+kernel_matrix(xv, xt, lengths, kind)@alpha
                error = rms(pred-yv)
                if np.isfinite(error):
                    candidates.append((error, kind, lengths))
            except (ValueError, linalg.LinAlgError):
                pass
    if not candidates:
        return qmc_average(sample_values)
    candidates.sort(key=lambda c: c[0])
    baseline_error = rms(yv-yt.mean())
    best = candidates[0]
    if best[0] > baseline_error*0.95:
        return qmc_average(sample_values)
    try:
        if best[1] == "poly":
            a = polynomial_matrix(x, best[2])
            coeff = linalg.lstsq(a, yy, cond=1e-10, lapack_driver="gelsy")[0]
            estimate = coeff[0]*scale
        else:
            alpha, beta = gp_fit(x, yy, best[2], best[1])
            estimate = (beta[0]+kernel_integral(x, best[2], best[1])@alpha)*scale
        if np.isfinite(estimate):
            return float(estimate)
    except (ValueError, linalg.LinAlgError):
        pass
    return qmc_average(sample_values)


def solve(oracle):
    d, budget = oracle.d, oracle.budget
    if d > 10:
        count = min(budget, 1024, max(1, 1000000//d))
        try:
            sampler = qmc.Sobol(d, scramble=True, seed=oracle.seed)
            high_dimensional = sampler.random_base2(int(math.floor(math.log2(count))))
        except ValueError:
            high_dimensional = np.random.default_rng(oracle.seed).random((count, d))
        return qmc_average(query_uniform(oracle, high_dimensional))
    sobol = qmc.Sobol(d, scramble=True, seed=oracle.seed)
    nn = 2**math.ceil(math.log2(max(budget, 2)))
    points = sobol.random_base2(int(math.log2(nn)))
    n0 = min(256, budget)
    x = points[:n0]
    y = query_uniform(oracle, x)
    oracle.corner_indices = np.empty(0, dtype=int)
    if budget < 128 or np.max(np.abs(y)) == 0:
        query_uniform(oracle, points[n0:n0+oracle.remaining])
        return qmc_average(uniform_values(oracle, y))
    ans = exact_polynomial(x, y)
    if ans is not None:
        return ans
    ans = exponential_quadratic(x, y)
    if ans is not None:
        return ans
    product = check_product(oracle, x, y)
    if product is not None:
        return integrate_product(oracle, *product)
    ans = smooth_tensor_rule(oracle, x, y)
    if ans is not None:
        return ans
    if getattr(oracle, "analytic", False):
        if oracle.remaining >= 8*d+600:
            try:
                _, derivatives = sample_gradients(oracle, x, n=2, order=4)
                _, singular, vectors = linalg.svd(derivatives, full_matrices=False)
            except (FloatingPointError, ValueError, linalg.LinAlgError):
                vectors = None
            if vectors is not None and singular[0] > 0 and (len(singular) == 1 or singular[1] < singular[0]*1e-8):
                ans = fit_sinusoid(x, y, vectors[0])
                if ans is not None:
                    return ans
                ans = ridgepoly.integrate(oracle, x, y, vectors[0], ridge_density,
                                          min(START+15, time.monotonic()+5))
                if ans is not None:
                    return ans
                ans = integrate_ridge(oracle, x, y, vectors[0])
                if ans is not None:
                    return ans
        ans = tensor.integrate(oracle, x, y, points[n0:], START+25, max_rank=3)
        if ans is not None:
            return ans
        ans = spectral.integrate(oracle, x, y, points[n0:], START+20)
        if ans is None and oracle.remaining >= 65*(d+1):
            if hasattr(oracle, "extra_global"):
                for xx, yy in oracle.extra_global:
                    x = np.vstack((x, xx))
                    y = np.r_[y, yy]
                    n0 += len(xx)
                del oracle.extra_global
            ans = ridges.fit_fourier(oracle, x, y, START+22, points[n0:])
        if ans is not None:
            return ans
        if hasattr(oracle, "extra_global"):
            for xx, yy in oracle.extra_global:
                x = np.vstack((x, xx))
                y = np.r_[y, yy]
                n0 += len(xx)
    ans = tensor.integrate(oracle, x, y, points[n0:], START+25)
    if ans is not None:
        return ans
    if oracle.remaining > 2*d*24+512 and remaining_time() > 15:
        try:
            gx, gradients = sample_gradients(oracle, x, 24)
            ans = fit_hinges(x, y, gx, gradients)
            if ans is not None:
                return ans
            centered = gradients-gradients.mean(axis=0)
            _, singular, vectors = linalg.svd(centered, full_matrices=False)
            if len(singular) == 1 or singular[1] < singular[0]*1e-7:
                ans = fit_sinusoid(x, y, vectors[0])
                if ans is not None:
                    return ans
                ans = ridgepoly.integrate(oracle, x, y, vectors[0], ridge_density,
                                          min(START+20, time.monotonic()+6))
                if ans is not None:
                    return ans
                ans = integrate_ridge(oracle, x, y, vectors[0])
                if ans is not None:
                    return ans
            model = projection.power_lines(oracle, x, y, min(START+18, time.monotonic()+5))
            if model is not None:
                error, integral, predict, maximum = model
                extra = points[n0:n0+oracle.remaining]
                residual = query_uniform(oracle, extra)-predict(extra)
                return float(integral+qmc_average(residual, n0)) if len(residual) else integral
            clusters = []
            gs = max(rms(gradients), 1e-200)
            for row in gradients:
                if not any(np.linalg.norm(row-other) < 2e-6*gs for other in clusters):
                    clusters.append(row)
            if remaining_time() > 12:
                model = projection.fit(oracle, x, y, min(START+19, time.monotonic()+6), (gx, gradients))
                if model is not None:
                    error, integral, predict, maximum = model
                    if predict is None:
                        return integral
                    if error < stable_std(y)*0.0001:
                        extra = points[n0:n0+oracle.remaining]
                        residual = query_uniform(oracle, extra)-predict(extra)
                        return float(integral+qmc_average(residual, n0))
                if hasattr(oracle, "projections") and oracle.remaining >= 256:
                    extra = points[n0:n0+oracle.remaining]
                    y = np.r_[y, query_uniform(oracle, extra)]
                    x = np.vstack((x, extra))
                    answer = projection.spline_cubature(x, y, oracle.projections,
                                                       ridge_density, START+25)
                    if answer is not None:
                        return answer
                    return fallback_cubature(x, y, uniform_values(oracle, y))
            if oracle.remaining > 500:
                ans = ridges.fit_piecewise(oracle, x, y, hinge_integral, START+25)
                if ans is not None:
                    return ans
            if oracle.remaining > 250:
                ans = ridges.fit_fourier(oracle, x, y, START+25)
                if ans is not None:
                    return ans
        except (ValueError, linalg.LinAlgError, FloatingPointError, TimeoutError):
            pass
    nleft = oracle.remaining
    if nleft:
        extra = points[n0:n0+nleft]
        y = np.r_[y, query_uniform(oracle, extra)]
        x = np.vstack((x, extra))
    return fallback_cubature(x, y, uniform_values(oracle, y))


def main():
    global ACTIVE_ORACLE
    init = json.loads(sys.stdin.readline())
    if init.get("type") != "init":
        raise ValueError("missing init")
    oracle = Oracle(init)
    ACTIVE_ORACLE = oracle
    signal.signal(signal.SIGALRM, deadline_handler)
    signal.setitimer(signal.ITIMER_REAL, max(0.1, 26.5-(time.monotonic()-START)))
    try:
        try:
            estimate = solve(oracle)
        except Exception as exc:
            print(type(exc).__name__ + ": " + str(exc), file=sys.stderr, flush=True)
            if not oracle.ys:
                raise
            estimate = qmc_average(uniform_values(oracle, oracle.ys[0]))
        finally:
            oracle.computing = False
            signal.setitimer(signal.ITIMER_REAL, 0)
    except ComputationDeadline:
        oracle.computing = False
        signal.setitimer(signal.ITIMER_REAL, 0)
        print("ComputationDeadline:", file=sys.stderr, flush=True)
        if not oracle.ys:
            raise
        estimate = qmc_average(uniform_values(oracle, oracle.ys[0]))
    oracle.finish(estimate)


if __name__ == "__main__":
    main()
