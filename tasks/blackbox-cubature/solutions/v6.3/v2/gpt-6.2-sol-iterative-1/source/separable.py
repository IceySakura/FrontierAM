"""Tensor rank-one and additive interpolation with adaptive 1-D integrals."""

import math
from dataclasses import dataclass

import numpy as np
from scipy import interpolate, optimize, special
from numpy.polynomial import Chebyshev

from core import Model, agrees, gauss, least


@dataclass
class Curve:
    predict: object
    integrate: object
    probes: tuple = ()


def grid_points(anchor, t):
    m, d = t.shape
    out = np.broadcast_to(anchor, (m, d, d)).copy()
    j = np.arange(d)
    out[:, j, j] = t
    return out.reshape(m*d, d)


def across(s, anchor, t):
    return (s.ask(grid_points(anchor, t))/s.scale).reshape(t.shape)


def factor_predict(mode, offset, base, v):
    if mode == "additive":
        return base+np.sum(v-base, axis=1)
    return offset+(base-offset)*np.prod((v-offset)/(base-offset), axis=1)


def cross_agrees(mode, offset, base, v, y):
    test = factor_predict(mode, offset, base, v)
    if mode == "product" and offset == 0.:
        return agrees(test, y, 4.e-9, 1.e-265)
    return agrees(test, y, 4.e-10, 6.e-13)


def choose_factor(s, anchor, base, x, y, v, levels):
    if levels is not None and len(levels) == 2:
        off = (levels[0]+levels[-1])*.5
        if abs(base-off) > 1.e-200 and cross_agrees("product", off, base, v, y):
            return "product", off
    if abs(base) > 1.e-280 and cross_agrees("product", 0., base, v, y):
        return "product", 0.
    if cross_agrees("additive", 0., base, v, y):
        return "additive", 0.
    if levels is not None:
        off = (levels[0]+levels[-1])*.5
        if abs(base-off) > 1.e-200 and cross_agrees("product", off, base, v, y):
            return "product", off
    if s.left < 12 or s.d < 2 or len(x) < 4:
        return None

    # In a two-coordinate rectangle, the mixed difference is the
    # product of the two main differences divided by the shifted anchor.
    vi = v-base
    which = np.argsort(np.abs(vi), axis=1)[:, -2:]
    strength = np.prod(np.take_along_axis(vi, which, axis=1), axis=1)
    ii = np.argsort(-np.abs(strength))[:5]
    if np.max(np.abs(strength)) < 1.e-8:
        return None
    pts = np.tile(anchor, (len(ii), 1))
    jj = which[ii]
    np.put_along_axis(pts, jj, np.take_along_axis(x[ii], jj, axis=1), axis=1)
    mixed = s.ask(pts)/s.scale - base - np.take_along_axis(vi[ii], jj, axis=1).sum(axis=1)
    good = np.abs(mixed) > 1.e-9
    ratios = strength[ii][good]/mixed[good]
    if not len(ratios):
        return None
    for ratio in ratios:
        if abs(ratio) < 1.e-8 or abs(ratio) > 1.e5:
            continue
        off = base-ratio
        if cross_agrees("product", off, base, v, y):
            return "product", off
    return None


def integral_on(predict, lo, hi, knots=(), n=96):
    z, w = gauss(n)
    edges = [lo] + [k for k in knots if lo < k < hi] + [hi]
    if hi <= lo:
        return 0.
    return math.fsum(float((b-a)*(w @ predict(a+(b-a)*z)))
                     for a, b in zip(edges, edges[1:]))


def piecewise_linear(x, y):
    """Fit a broken line by intersecting its two least-squares lines."""
    n = len(x)
    for k in range(2, n-1):
        c0, c1 = np.polyfit(x[:k], y[:k], 1)
        d0, d1 = np.polyfit(x[k:], y[k:], 1)
        if abs(c0-d0) < 1.e-9:
            continue
        knot = (d1-c1)/(c0-d0)
        if x[k-1]-1.e-12 <= knot <= x[k]+1.e-12:
            p = lambda z, a=c0, b=c1, c=d0, t=knot: a*z+b+(c-a)*np.maximum(z-t, 0.)
            if np.max(np.abs(p(x)-y)) <= 8.e-12*max(1., np.max(np.abs(y))):
                return p, knot
    return None


def fit_power(x, y, cusp):
    if len(x) < 8 or np.ptp(y) < 1.e-6:
        return None
    # A translated absolute or quadratic distance, with a fitted power.
    p0 = x[np.argmax(y)]
    if p0 < .005 or p0 > .995:
        p0 = x[np.argmin(y)]
    spread = max(float(np.ptp(x)), .001)
    mx = float(np.median(y))
    lower = [-2., -26., -300., mx-2000.]
    upper = [3., 3., 300., mx+2000.]

    def calc(t, jac=False):
        c, l, p, b = t
        w = np.exp(l)
        delta = x-c
        if cusp:
            v = np.abs(delta)+w
            dc, dl = -np.sign(delta), w
        else:
            v = delta*delta+w*w
            dc, dl = -2.*delta, 2.*w*w
        lv = np.log(v)
        if jac:
            return np.column_stack((p*dc/v, p*dl/v, lv, np.ones(len(x))))
        return b+p*lv-y

    for cc in (p0, min(.99, max(.01, p0-.016)), min(.99, max(.01, p0+.017))):
        for width in ((.025, .15) if cusp else (.07, .24)):
            l = np.log(width*spread)
            a = np.abs(x-cc)
            v = np.log(a+np.exp(l) if cusp else a*a+np.exp(2.*l))
            p, b = np.polyfit(v, y, 1)
            start = [cc, l, np.clip(p, -299., 299.), np.clip(b, mx-1900., mx+1900.)]
            opt = optimize.least_squares(calc, start, jac=lambda z: calc(z, True),
                                         bounds=(lower, upper), max_nfev=65,
                                         ftol=5.e-12, xtol=5.e-13, gtol=5.e-13)
            if np.max(np.abs(opt.fun)) <= 1.5e-10:
                cc, l, p, b = opt.x
                width = np.exp(l)
                if cusp:
                    pred = lambda z: b+p*np.log(np.abs(z-cc)+width)
                else:
                    pred = lambda z: b+p*np.log((z-cc)**2+width**2)
                return pred, cc, width
    return None


def fit_curve(x, y, allow_power=True):
    ym = np.max(np.abs(y), initial=0.)
    if ym == 0.:
        return Curve(lambda z: np.zeros_like(z), lambda a, b: 0.)
    u = y/ym
    for degree in (0, 1, 2, 3, 5, 8, 12, 17):
        if len(x) < 1.65*(degree+1):
            break
        p = Chebyshev.fit(x, u, degree, domain=(0, 1))
        if np.max(np.abs(p(x)-u)) < 1.2e-12:
            pi = p.integ()
            return Curve(lambda z: ym*p(z), lambda a, b: ym*(pi(b)-pi(a)))
    broken = piecewise_linear(x, u)
    if broken is not None:
        p, k = broken
        pred = lambda z: ym*p(z)
        return Curve(pred, lambda a, b: integral_on(pred, a, b, [k], 4))

    sign = 1. if np.max(u) > -np.min(u) else -1.
    v = sign*u
    if np.any(v < 0.) or np.count_nonzero(v > 1.e-280) < 5:
        return None
    good = v > 1.e-280
    logv = np.log(v[good])
    xx = x[good]
    for degree in (1, 2, 3, 4, 6, 9):
        if len(logv) < 1.8*(degree+1):
            break
        lp = Chebyshev.fit(xx, logv, degree, domain=(0, 1))
        if np.max(np.abs(lp(xx)-logv)) < 3.e-11 and np.all(lp(x[~good]) < -600.):
            pred = lambda z: sign*ym*np.exp(np.minimum(lp(z), 650.))
            return Curve(pred, lambda a, b: integral_on(pred, a, b))
    broken = piecewise_linear(xx, logv)
    if broken is not None:
        p, k = broken
        if np.all(p(x[~good]) < -620.):
            pred = lambda z: ym*sign*np.exp(np.minimum(p(z), 650.))
            return Curve(pred, lambda a, b: integral_on(pred, a, b, [k]))
    if not allow_power or not np.all(good):
        return None
    for cusp in (False, True):
        fitted = fit_power(x, logv, cusp)
        if fitted is None:
            continue
        lp, c, w = fitted
        pred = lambda z: ym*sign*np.exp(np.minimum(lp(z), 650.))
        offsets = [w, 4*w]
        while offsets[-1] < .35:
            offsets.append(offsets[-1]*4.)
        knots = sorted([c]+[c+sgn*z for z in offsets for sgn in (-1., 1.)])
        probes = (c, c-.8*w, c+.8*w, c-3*w, c+3*w) if w < .032 else ()
        return Curve(pred, lambda a, b: integral_on(pred, a, b, knots, 80), probes)
    return None


def find_levels(y, max_levels=4):
    y = np.sort(y)
    if len(y) == 0:
        return None
    tol = 8.e-14*np.max(np.abs(y), initial=0.)
    # Values that accidentally look alike only in the far tail don't
    # make a plateau unless every observed level has that property.
    i = np.r_[True, np.diff(y) > tol]
    levels = y[i]
    if len(levels) > max_levels:
        return None
    return levels


def support_splits(s, a, samples):
    split = []
    kinds = []
    for j, (x, y) in enumerate(samples):
        levels = find_levels(y)
        if levels is not None:
            code = np.searchsorted((levels[:-1]+levels[1:])*.5, y)
            kinds.append(("step", levels, int(code[0])))
        elif np.any(y == 0.) and np.any(y != 0.):
            # Do not treat numerical underflow of an exponential as a jump.
            nz = np.abs(y[y != 0.])
            if np.min(nz) < 1.e-200*np.max(nz):
                kinds.append(("smooth", None, 0))
                continue
            code = (y != 0.).astype(int)
            kinds.append(("support", None, int(code[0])))
        else:
            kinds.append(("smooth", None, 0))
            continue
        changes = np.flatnonzero(np.diff(code))
        split.extend((j, x[k], x[k+1], int(code[k]), int(code[k+1])) for k in changes)
    if len(split) > min(16, max(0, (s.left-74)//46)):
        return None
    if not split:
        return kinds, [[] for _ in samples]
    jj, lo, hi, c0, c1 = np.array(split).T
    jj = jj.astype(int)
    p = np.tile(a, (len(jj), 1))
    ii = np.arange(len(jj))
    mids = [(levels[:-1]+levels[1:])*.5 if levels is not None else None
            for kind, levels, _ in kinds]
    for k in range(47):
        mm = (lo+hi)*.5
        p[ii, jj] = mm
        y = s.ask(p)/s.scale
        code = np.array([np.searchsorted(mids[j], v) if mids[j] is not None
                         else int(v != 0.) for j, v in zip(jj, y)])
        go = code == c0
        lo = np.where(go, mm, lo)
        hi = np.where(go, hi, mm)
    edges = [[] for _ in samples]
    for j, z, c in zip(jj, (lo+hi)*.5, c1):
        edges[j].append((float(z), int(c)))
    return kinds, edges


_gx = np.array([.9914553711208126, .9491079123427585, .8648644233597691,
                .7415311855993944, .5860872354676911, .4058451513773972,
                .2077849550078985, 0.])
_kw = np.array([.022935322010529225, .06309209262997855, .10479001032225018,
                .14065325971552592, .16900472663926790, .19035057806478541,
                .20443294007529889, .20948214108472783])
_gw = np.array([0., .12948496616886969, 0., .27970539148927667, 0.,
                .38183005050511894, 0., .41795918367346939])
GK_X = (1.+np.r_[-_gx, _gx[-2::-1]])*.5
GK_W = np.r_[_kw, _kw[-2::-1]]*.5
G7_W = np.r_[_gw, _gw[-2::-1]]*.5


def axis_quadrature(s, anchor, init, usable):
    """Prioritize the slice intervals with the largest relative errors."""
    todo = [(j, a, b) for j, limits in usable.items() for a, b in limits]
    cells = []
    if 15*len(todo) > s.left:
        return None
    while todo:
        j, a, b = np.asarray(todo).T
        jj = j.astype(int)
        p = np.tile(anchor, (15*len(todo), 1))
        z = a[:, None]+(b-a)[:, None]*GK_X
        p[np.arange(len(p)), np.repeat(jj, 15)] = z.ravel()
        y = (s.ask(p)/s.scale).reshape(-1, 15)
        wide = b-a
        got = (y @ GK_W)*wide
        err = np.abs(y@(GK_W-G7_W))*wide
        for i in range(len(todo)):
            k = jj[i]
            near = (init[k][0] > a[i]) & (init[k][0] < b[i])
            if np.any(near):
                q = np.max(np.abs(init[k][1][near]))
                err[i] = max(err[i], (q-2*np.max(np.abs(y[i])))*wide[i]*.13)
            cells.append((k, a[i], b[i], got[i], err[i]))
        todo = []
        total = np.zeros(s.d)
        for j, a, b, val, err in cells:
            total[j] += val
        rank = sorted(range(len(cells)), reverse=True,
                      key=lambda i: cells[i][4]/max(abs(total[cells[i][0]]), 1.e-16))
        todo = []
        take = []
        cap = min(8, s.left//30)
        if cap <= 0 or s.time_left() < 1.5:
            return total
        for i in rank[:cap]:
            j, a, b, got, er = cells[i]
            if er <= 1.e-11*max(abs(total[j]), 1.e-16) or b-a < 1.e-14:
                continue
            m = (a+b)*.5
            todo.extend([(j, a, m), (j, m, b)])
            take.append(i)
        for i in sorted(take, reverse=True):
            cells.pop(i)
    return total


def try_separable(s, n, small=False):
    d = s.d
    if s.left < (37*d+88 if not small else 20*d+20):
        return None
    x0, y0 = s.data()
    y0 = y0[:n]
    peak = int(np.argmax(np.abs(y0)))
    a = x0[peak].copy()
    base = y0[peak]
    if base == 0.:
        return None

    levels = find_levels(y0, 3)
    # Include representative interior points and points with appreciable values.
    top = np.argsort(-np.abs(y0))
    ii = np.unique(np.r_[np.arange(min(5, n)), top[[1, n//9, n//3, n//2]]])
    x = x0[ii].copy()
    y = y0[ii]
    v = across(s, a, x)
    choice = choose_factor(s, a, base, x, y, v, levels)
    if choice is None:
        return None
    mode, off = choice
    # A near-anchor check is useful for narrow peaks in a sum of products.
    size = min(.37, .14+np.max(np.std(x, axis=0))*.42)
    z = np.clip(a+size*(s.design(5)*2.-1.), .00007, .99993)
    vz = across(s, a, z)
    yz = s.ask(z)/s.scale
    if not cross_agrees(mode, off, base, vz, yz):
        return None
    x, v = np.vstack((x, z)), np.vstack((v, vz))
    t = (1-np.cos(np.linspace(0., np.pi, 25 if not small else 13)))*.5
    yt = across(s, a, np.broadcast_to(t[:, None], (len(t), d)))
    samples = []
    for j in range(d):
        tx = np.r_[t, x[:, j], a[j]]
        yy = np.r_[yt[:, j], v[:, j], base]
        tx, keep = np.unique(tx, return_index=True)
        samples.append((tx, yy[keep]))
    result = support_splits(s, a, samples)
    if result is None:
        return None
    kinds, edges = result
    integrals = np.zeros(d)
    fitted = {}
    needed = {}
    for j, ((tx, ty), (kind, ll, code), cuts) in enumerate(zip(samples, kinds, edges)):
        xs = [0.] + [c[0] for c in cuts] + [1.]
        cs = [code] + [c[1] for c in cuts]
        if kind == "step":
            vv = ll[cs]
            predict = lambda z, v=vv, k=np.array(xs[1:-1]): v[np.searchsorted(k, z)]
            fitted[j] = (predict, list(zip(xs, xs[1:])))
            integrals[j] = np.diff(xs)@vv
            continue
        pp = []
        intervals = []
        for left, right, c in zip(xs, xs[1:], cs):
            good = (tx >= left) & (tx <= right)
            if kind == "support" and c == 0:
                continue
            intervals.append((left, right))
            if right-left < 1.e-16:
                continue
            if np.count_nonzero(good) < 7 or s.time_left() < 4. or (small and not np.any(good)):
                break
            f = fit_curve(tx[good], ty[good], allow_power=not small)
            if f is None:
                break
            if f.probes and s.left > 14*d+80:
                tpeak = np.array([q for q in f.probes if left+1.e-15 < q < right-1.e-15])
                if len(tpeak):
                    zpeak = np.tile(a, (len(tpeak), 1))
                    zpeak[:, j] = tpeak
                    ypeak = s.ask(zpeak)/s.scale
                    if not agrees(f.predict(tpeak), ypeak, 4.e-10, 8.e-14):
                        xx = np.r_[tx[good], tpeak]
                        yy = np.r_[ty[good], ypeak]
                        ii = np.argsort(xx)
                        f = fit_curve(xx[ii], yy[ii], allow_power=True)
                        if f is None:
                            break
            pp.append((left, right, f))
        else:
            def pred(z, pp=pp):
                out = np.zeros_like(z)
                for l, r, f in pp:
                    mask = (z >= l) & (z <= r)
                    out[mask] = f.predict(z[mask])
                return out
            fitted[j] = (pred, intervals)
            integrals[j] = sum(f.integrate(l, r) for l, r, f in pp)
            continue
        needed[j] = [(l, r) for l, r, c in zip(xs, xs[1:], cs) if c or kind != "support"]

    jjs = list(fitted)
    t = s.design(7)
    if 7*len(jjs) <= s.left and jjs:
        p = np.tile(a, (len(jjs), len(t), 1))
        for i, j in enumerate(jjs):
            intervals = fitted[j][1]
            if len(intervals) == 1:
                l, r = intervals[0]
                t[:, j] = l+(r-l)*t[:, j]
            p[i, :, j] = t[:, j]
        yy = (s.ask(p.reshape(-1, d))/s.scale).reshape(-1, len(t))
        for j, row in zip(jjs, yy):
            f, intervals = fitted[j]
            if not agrees(f(t[:, j]), row, 2.e-9, 5.e-13):
                needed[j] = intervals or [(0., 1.)]
                del fitted[j]

    if needed:
        g = axis_quadrature(s, a, samples, needed)
        if g is None:
            return None
        for j in needed:
            integrals[j] = g[j]
    else:
        p = np.column_stack([fitted[j][0](x0[:n, j]) for j in range(d)])
        if not agrees(factor_predict(mode, off, base, p), y0, 3.e-8, 2.e-12):
            return None
    result = factor_predict(mode, off, base, integrals[None, :])[0]
    if not math.isfinite(result):
        return None
    s.route = mode + (" axes" if needed else " fitted axes")
    return float(result)
