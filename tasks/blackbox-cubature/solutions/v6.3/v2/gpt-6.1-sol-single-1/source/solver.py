#!/usr/bin/env python3
"""Budgeted, data-adaptive cubature over the unit hypercube."""
import time

STARTED = time.monotonic()

import json
import math
import sys
import warnings
from decimal import Decimal, localcontext

import numpy as np
from scipy import linalg, optimize, special, integrate
from scipy.interpolate import PchipInterpolator
from scipy.stats import qmc

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def time_left():
    return 26.0 - (time.monotonic() - STARTED)


def stable_mean(values):
    values = np.asarray(values,dtype=float)
    scale = float(np.max(np.abs(values)))
    if scale==0:
        return 0.0
    return float(np.mean(values/scale)*scale)


class Oracle:
    def __init__(self, init):
        self.d = int(init["dimension"])
        self.budget = int(init["budget"])
        self.seed = int(init.get("seed", 0)) % (2**32)
        self.lo = np.asarray([b[0] for b in init["bounds"]], dtype=float)
        self.hi = np.asarray([b[1] for b in init["bounds"]], dtype=float)
        self.volume = float(np.prod(self.hi - self.lo))
        self.used = 0
        self.cache = {}
        self.x = []
        self.y = []

    @property
    def remaining(self):
        return self.budget - self.used

    def __call__(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, self.d)
        points = np.clip(points, 0.0, 1.0)
        result = np.empty(len(points))
        fresh = []
        locations = []
        pending = {}
        for i, p in enumerate(points):
            key = tuple(p)
            if key in self.cache:
                result[i] = self.cache[key]
            elif key in pending:
                pending[key].append(i)
            else:
                pending[key] = [i]
                fresh.append(p)
                locations.append(key)
        if len(fresh) > self.remaining:
            raise ValueError("internal query budget exceeded")
        for start in range(0, len(fresh), 512):
            unit = np.asarray(fresh[start:start + 512])
            actual = self.lo + unit * (self.hi - self.lo)
            request = {"type": "query", "points": actual.tolist()}
            sys.stdout.write(json.dumps(request, separators=(",", ":"), allow_nan=False) + "\n")
            sys.stdout.flush()
            response = json.loads(sys.stdin.readline())
            vals = np.asarray(response["values"], dtype=float)
            if len(vals) != len(unit) or not np.all(np.isfinite(vals)):
                raise ValueError("invalid oracle response")
            self.used += len(unit)
            for p, y, key in zip(unit, vals, locations[start:start + 512]):
                self.cache[key] = float(y)
                self.x.append(p)
                self.y.append(float(y))
                for i in pending[key]:
                    result[i] = y
        return result


class Model:
    def __init__(self, name, predict, integral):
        self.name = name
        self.predict = predict
        self.integral = float(integral)
        self.error = math.inf


def quality(model, x, y):
    try:
        pred = np.asarray(model.predict(x))
        if pred.shape != y.shape or not np.all(np.isfinite(pred)):
            return math.inf
        norm = float(np.max(np.abs(y)))
        if norm==0:
            norm = 1.0
        normalized = y/norm
        scale = max(float(np.std(normalized)),1e-8)
        model.error = float(np.sqrt(np.mean((pred/norm-normalized)**2))/scale)
        if not math.isfinite(model.error):
            model.error = math.inf
        return model.error
    except (ValueError, OverflowError, FloatingPointError):
        return math.inf


def quadratic_matrix(x, full=False):
    z = x - 0.5
    cols = [np.ones(len(x))]
    cols.extend(z[:, j] for j in range(x.shape[1]))
    cols.extend(z[:, j]**2 for j in range(x.shape[1]))
    if full:
        cols.extend(z[:, i] * z[:, j] for i in range(x.shape[1]) for j in range(i))
    return np.column_stack(cols)


def exp_quadratic_integral(coef, d):
    logval = float(coef[0])
    for j in range(d):
        a, b = float(coef[1 + j]), float(coef[1 + d + j])
        peak = float(np.clip(-a / (2 * b), -0.5, 0.5)) if b < -1e-14 else (-0.5 if a < 0 else 0.5)
        maximum = a * peak + b * peak**2
        val = integrate.quad(
            lambda t: math.exp(min(0.0, a*t + b*t*t - maximum)),
            -0.5, 0.5, epsabs=2e-14, epsrel=2e-13,
            points=[peak] if -0.5 < peak < 0.5 else None, limit=150,
        )[0]
        if val <= 0:
            return 0.0
        logval += maximum + math.log(val)
    return math.exp(logval) if logval > -745 else 0.0


def affine_power_integral(minimum,widths,power):
    widths = np.abs(np.asarray(widths))
    widths = widths[widths!=0]
    if not len(widths):
        return minimum**(-power)
    with localcontext() as ctx:
        ctx.prec = 75
        base = Decimal.from_float(float(minimum))
        a = [Decimal.from_float(float(w)) for w in widths]
        p = Decimal.from_float(float(power))
        d = len(a)
        terms = [(base,(-1)**d)]
        for width in a:
            terms += [(v+width,-s) for v,s in terms]
        if p == int(power) and 1<=int(power)<=d:
            degree = d-int(power)
            denominator = Decimal(math.factorial(int(power)-1)*math.factorial(degree))
            multiplier = Decimal((-1)**(int(power)-1))/denominator

            def primitive(z):
                return multiplier*z**degree*z.ln()
        else:
            denominator = Decimal(1)
            for j in range(1,d+1):
                denominator *= Decimal(j)-p
            exponent = Decimal(d)-p

            def primitive(z):
                return z**exponent/denominator
        total = sum(sign*primitive(z) for z,sign in terms)
        for width in a:
            total /= width
        return float(total)


def fit_affine_power(x,y):
    d = x.shape[1]
    scale = float(np.max(y))
    target = y/scale
    logtarget = np.log(target)
    matrix = np.column_stack([np.ones(len(x)),x])
    best = None
    for power in [.3,.5,1,2,3,5,8,12,20,40,80]:
        transformed = np.exp(np.minimum(300,-logtarget/power))
        coef = linalg.lstsq(matrix,transformed,lapack_driver="gelsy")[0]
        minimum = coef[0]+np.sum(np.minimum(coef[1:],0))
        if minimum<=0 or np.any(np.abs(coef[1:])>1e5):
            continue
        pred = np.exp(np.clip(-power*np.log(matrix@coef),-745,700))
        error = float(np.mean((pred-target)**2))
        if best is None or error<best[0]:
            best = (error,minimum,coef[1:],power)
    if best is None:
        return None
    _,minimum,a,power = best
    p0 = np.r_[math.log(minimum),a,math.log(power)]

    def values(p):
        a = p[1:1+d]
        minimum,power = math.exp(p[0]),math.exp(p[-1])
        shifted = x-(a<0)[None,:]
        den = minimum+shifted@a
        pred = np.exp(np.clip(-power*np.log(den),-745,700))
        return pred,den,shifted,minimum,power

    def residual(p):
        return values(p)[0]-target

    def jac(p):
        pred,den,shifted,minimum,power = values(p)
        factor = -power*pred/den
        return np.column_stack([factor*minimum,factor[:,None]*shifted,-power*pred*np.log(den)])

    try:
        got = optimize.least_squares(
            residual,p0,jac=jac,
            bounds=([-40]+[-1e5]*d+[math.log(.05)],[40]+[1e5]*d+[math.log(100)]),
            max_nfev=80,ftol=2e-12,xtol=2e-12,gtol=2e-12,
        )
        if np.max(np.abs(residual(got.x)))>1e-7:
            return None
        minimum,power = math.exp(got.x[0]),math.exp(got.x[-1])
        a = got.x[1:1+d]
        integral = scale*affine_power_integral(minimum,a,power)

        def pred(points,a=a,m=minimum,p=power,s=scale):
            den = m+(points-(a<0)[None,:])@a
            return s*np.exp(np.clip(-p*np.log(den),-745,700))

        return Model("affine-power",pred,integral)
    except (ValueError,OverflowError,ArithmeticError,linalg.LinAlgError):
        return None


def cheap_models(x, y):
    d = x.shape[1]
    models = []
    a = quadratic_matrix(x, full=True)
    if len(x) >= a.shape[1] + 4:
        coef = linalg.lstsq(a, y, lapack_driver="gelsy")[0]
        models.append(Model(
            "quadratic", lambda p, c=coef: quadratic_matrix(p, True) @ c,
            coef[0] + np.sum(coef[1+d:1+2*d]) / 12.0,
        ))
    if np.all(y > 0):
        a = quadratic_matrix(x, full=False)
        coef = linalg.lstsq(a, np.log(y), lapack_driver="gelsy")[0]
        try:
            val = exp_quadratic_integral(coef, d)
            models.append(Model(
                "exponential-quadratic",
                lambda p, c=coef: np.exp(np.clip(quadratic_matrix(p) @ c, -745, 700)),
                val,
            ))
        except (OverflowError, ValueError):
            pass
        if min((quality(model,x,y) for model in models),default=1)>1e-7:
            power = fit_affine_power(x,y)
            if power:
                models.append(power)
    if np.all(y < 0):
        for model in cheap_models(x, -y):
            if model.name in ("exponential-quadratic","affine-power"):
                models.append(Model(
                    "negative-exponential-quadratic",
                    lambda p, m=model: -m.predict(p), -model.integral,
                ))
    return models


def piecewise_lines(t, y):
    scale = max(float(np.ptp(y)), float(np.max(np.abs(y))) * 1e-10, 1e-300)
    h = np.diff(t)
    slopes = np.diff(y) / h
    changes = np.diff(slopes)
    active = np.flatnonzero(np.abs(changes) > scale * 2e-7)
    if len(active) == 0:
        coef = np.polyfit(t, y, 1)
        if np.max(np.abs(np.polyval(coef, t)-y)) < scale * 1e-8:
            return [], [(float(coef[0]), float(coef[1]))]
        return None
    groups = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1)
    if any(len(g) > 2 for g in groups) or len(active) > 0.65 * len(changes):
        return None
    boundaries = []
    lines = []
    previous = 0
    for group in groups:
        first, last = int(group[0]), int(group[-1])
        stop = first + 2
        if stop - previous < 2:
            return None
        slope, intercept = np.polyfit(t[previous:stop], y[previous:stop], 1)
        lines.append((float(slope), float(intercept)))
        previous = last + 1
        boundaries.append((first, last))
    if len(t) - previous < 2:
        return None
    slope, intercept = np.polyfit(t[previous:], y[previous:], 1)
    lines.append((float(slope), float(intercept)))
    knots = []
    for (left, right), (first, last) in zip(zip(lines[:-1], lines[1:]), boundaries):
        if abs(left[0] - right[0]) < scale * 1e-8:
            return None
        knot = (right[1] - left[1]) / (left[0] - right[0])
        if not t[first] - 1e-10 <= knot <= t[last+2] + 1e-10:
            return None
        knots.append(float(knot))
    pred = np.empty_like(y)
    regions = np.searchsorted(knots, t)
    for j, (slope, intercept) in enumerate(lines):
        mask = regions == j
        pred[mask] = slope * t[mask] + intercept
    if np.max(np.abs(pred-y)) > scale * 5e-8:
        return None
    return knots, lines


def exp_line_integral(a, b, lo, hi):
    if hi <= lo:
        return 0.0
    if abs(a * (hi-lo)) < 1e-7:
        return math.exp(a*lo+b) * (hi-lo) * (1 + a*(hi-lo)/2 + (a*(hi-lo))**2/6)
    if a > 0:
        return math.exp(a*hi+b) * (-math.expm1(a*(lo-hi))) / a
    return math.exp(a*lo+b) * math.expm1(a*(hi-lo)) / a


def trig_fit(t, y):
    if len(t) < 12 or not np.allclose(np.diff(t), t[1]-t[0], atol=1e-12):
        return None
    scale = max(float(np.std(y)), 1e-300)
    dy = np.diff(y)
    length = len(dy)//2
    h0 = linalg.hankel(dy[:length], dy[length-1:-1])
    h1 = linalg.hankel(dy[1:length+1], dy[length:])
    u, s, vt = linalg.svd(h0, full_matrices=False)
    if s[0] < 1e-13 * max(float(np.max(np.abs(y))), 1e-300):
        return None
    rank = int(np.sum(s > s[0]*2e-11))
    rank = min(rank, 16)
    if rank < 2:
        rank = 2
    ranks = sorted(set([rank, max(2, rank-1), min(16, rank+1)]))
    best = None
    z = t - 0.5
    for r in ranks:
        if r>len(s) or s[r-1]<=s[0]*1e-14:
            continue
        eigen = linalg.eigvals((u[:, :r].T @ h1 @ vt[:r].T) / s[:r][None, :])
        frequencies = sorted(float(np.angle(e)/(t[1]-t[0])) for e in eigen if np.angle(e) > 1e-5)
        frequencies = [w for w in frequencies if w < 0.98*np.pi/(t[1]-t[0])]
        if not frequencies or len(frequencies) > 8:
            continue
        w = np.asarray(frequencies)

        def matrix(freq):
            angles = z[:, None] * freq[None, :]
            return np.column_stack([np.ones(len(z)), np.cos(angles), np.sin(angles)])

        coef = linalg.lstsq(matrix(w), y, lapack_driver="gelsy")[0]
        k = len(w)
        p0 = np.r_[w, coef]

        def residual(p):
            return (matrix(p[:k]) @ p[k:] - y)/scale

        def jac(p):
            angles = z[:, None] * p[:k]
            a, b = p[k+1:k+1+k], p[k+1+k:]
            dw = z[:, None] * (-np.sin(angles)*a + np.cos(angles)*b)
            return np.column_stack([dw, matrix(p[:k])])/scale

        try:
            got = optimize.least_squares(
                residual, p0, jac=jac, method="lm", max_nfev=100,
                ftol=1e-12, xtol=1e-12, gtol=1e-12,
            )
            error = float(np.max(np.abs(residual(got.x))))
            if best is None or error < best[0]:
                best = (error, got.x[:k], got.x[k:])
        except (ValueError, linalg.LinAlgError):
            continue
    if best is None or best[0] > 1e-6:
        return None
    error, w, c = best
    order = np.argsort(np.abs(w))
    signs = np.sign(w[order])
    w = np.abs(w[order])
    a = c[1:1+len(w)][order]
    b = c[1+len(w):][order] * signs
    return w, a, b, float(c[0]), error


def trig_fit_trend(t,y):
    """Resolve oscillations when a very slow component resembles a trend."""
    z = t-.5
    dy = np.diff(y,n=4)
    length = len(dy)//2
    if length < 5:
        return None
    h0 = linalg.hankel(dy[:length],dy[length-1:-1])
    h1 = linalg.hankel(dy[1:length+1],dy[length:])
    u,s,vt = linalg.svd(h0,full_matrices=False)
    if s[0] <= 1e-300:
        return None
    r = min(len(s),16,max(2,int(np.sum(s>s[0]*3e-9))))
    if s[r-1]<=s[0]*1e-14:
        return None
    eigen = linalg.eigvals((u[:,:r].T@h1@vt[:r].T)/s[:r][None,:])
    w = np.asarray(sorted(float(np.angle(e)/(t[1]-t[0])) for e in eigen if np.angle(e)>1e-4))
    if len(w)==0 or len(w)>8:
        return None
    k = len(w)
    poly = np.column_stack([np.ones(len(t)),z,z*z,z*z*z])
    scale = max(float(np.std(y)),1e-100)

    def design(freq):
        angles = z[:,None]*freq
        return np.column_stack([poly,np.cos(angles),np.sin(angles)])

    coef = linalg.lstsq(design(w),y,lapack_driver="gelsy")[0]

    def residual(p):
        return (design(p[:k])@p[k:]-y)/scale

    def jac(p):
        angles = z[:,None]*p[:k]
        a,b = p[k+4:k+4+k],p[k+4+k:]
        df = z[:,None]*(-np.sin(angles)*a+np.cos(angles)*b)
        return np.column_stack([df,design(p[:k])])/scale

    got = optimize.least_squares(residual,np.r_[w,coef],jac=jac,method="lm",max_nfev=130,
                                 ftol=1e-11,xtol=1e-11,gtol=1e-11)
    error = float(np.max(np.abs(residual(got.x))))
    if error>2e-5:
        return None
    w,coef = got.x[:k],got.x[k:]
    a,b = coef[4:4+k],coef[4+k:]
    order = np.argsort(np.abs(w))
    b = b[order]*np.sign(w[order])
    w,a = np.abs(w[order]),a[order]
    data = (w,a,b,float(coef[0]),error)
    model = Model(
        "axis-trigonometric-trend",
        lambda p,w=w,a=a,b=b,c=coef[:4]: np.polynomial.polynomial.polyval(np.asarray(p)-.5,c)+np.cos((np.asarray(p)-.5)[...,None]*w)@a+np.sin((np.asarray(p)-.5)[...,None]*w)@b,
        coef[0]+coef[2]/12+np.sum(a*np.sinc(w/(2*np.pi))),
    )
    model.data = data
    model.error = error
    return model


def fit_power_axis(t,y):
    scale = max(float(np.ptp(y)),1e-300)
    origin = float(np.mean(y))
    target = (y-origin)/scale
    change = np.abs(np.diff(target,n=2))
    if np.max(change,initial=0) < 1.15*np.median(change):
        return None
    knot = float(t[1:-1][int(np.argmax(change))])
    best = None
    for exponent in [.3,.5,.9,1.3,1.7,2.5]:
        base = np.abs(t-knot)**exponent
        c = linalg.lstsq(np.column_stack([np.ones(len(t)),t,base]),target,lapack_driver="gelsy")[0]
        p0 = np.r_[c,knot,exponent]

        def residual(p):
            return p[0]+p[1]*t+p[2]*np.abs(t-p[3])**p[4]-target

        def jac(p):
            delta = t-p[3]
            radius = np.maximum(np.abs(delta),1e-14)
            power = radius**p[4]
            return np.column_stack([
                np.ones(len(t)),t,power,
                -p[2]*p[4]*np.sign(delta)*radius**(p[4]-1),
                p[2]*power*np.log(radius),
            ])

        try:
            got = optimize.least_squares(
                residual,p0,jac=jac,
                bounds=([-100,-100,-100,-.2,.05],[100,100,100,1.2,5]),
                max_nfev=100,ftol=2e-12,xtol=2e-12,gtol=2e-12,
            )
            error = float(np.max(np.abs(residual(got.x))))
            if best is None or error < best[0]:
                best = (error,got.x)
            if error < 1e-11:
                break
        except (ValueError,OverflowError):
            pass
    if best is None or best[0]>1e-6:
        return None
    error,p = best
    b,exponent = p[3],p[4]
    moment = (np.sign(1-b)*abs(1-b)**(exponent+1)+np.sign(b)*abs(b)**(exponent+1))/(exponent+1)
    model = Model(
        "axis-affine-power",
        lambda x,p=p,o=origin,s=scale: o+s*(p[0]+p[1]*np.asarray(x)+p[2]*np.abs(np.asarray(x)-p[3])**p[4]),
        origin+scale*(p[0]+p[1]/2+p[2]*moment),
    )
    model.data = p
    model.error = error
    return model


def fit_shifted_power_axis(t,y):
    if not (np.all(y>0) or np.all(y<0)):
        return None
    sign = 1 if y[0]>0 else -1
    logy = np.log(sign*y)
    best = None
    for exponent in [-.5,.5,1.5]:
        center = float(t[np.argmax(sign*y) if exponent<0 else np.argmin(sign*y)])
        width = .025
        logamp = float(np.mean(logy-exponent*np.log(width+np.abs(t-center))))
        p0 = np.asarray([logamp,math.log(width),center,exponent])

        def residual(p):
            width = math.exp(p[1])
            return p[0]+p[3]*np.log(width+np.abs(t-p[2]))-logy

        def jac(p):
            width = math.exp(p[1])
            delta = t-p[2]
            radius = width+np.abs(delta)
            return np.column_stack([np.ones(len(t)),p[3]*width/radius,
                                    -p[3]*np.sign(delta)/radius,np.log(radius)])

        try:
            got = optimize.least_squares(
                residual,p0,jac=jac,
                bounds=([-740,-25,-.5,-8],[700,3,1.5,8]),max_nfev=130,
                ftol=2e-12,xtol=2e-12,gtol=2e-12,
            )
            error = float(np.max(np.abs(residual(got.x))))
            if best is None or error<best[0]:
                best = (error,got.x)
            if error<1e-11:
                break
        except (ValueError,OverflowError):
            pass
    if best is None or best[0]>1e-7:
        return None
    _,p = best
    width,center,exponent = math.exp(p[1]),p[2],p[3]

    def segment(lo,hi):
        power = exponent+1
        if abs(power)<1e-9:
            return math.log(hi/lo)
        return lo**power*math.expm1(power*math.log(hi/lo))/power

    if 0<=center<=1:
        moment = segment(width,width+center)+segment(width,width+1-center)
    elif center<0:
        moment = segment(width-center,width+1-center)
    else:
        moment = segment(width+center-1,width+center)
    model = Model(
        "axis-shifted-power",
        lambda x,p=p,s=sign: s*np.exp(np.clip(p[0]+p[3]*np.log(math.exp(p[1])+np.abs(np.asarray(x)-p[2])),-745,700)),
        sign*math.exp(p[0])*moment,
    )
    model.data = p
    scale = max(float(np.ptp(y)),float(np.max(np.abs(y)))*1e-10,1e-300)
    model.error = float(np.max(np.abs(model.predict(t)-y))/scale)
    return model


def fit_axis(t, y):
    """Small univariate approximation portfolio, selected from its samples."""
    scale = max(float(np.ptp(y)), float(np.max(np.abs(y))) * 1e-10, 1e-300)
    candidates = []

    def add(name, pred, val, data=None):
        try:
            error = float(np.max(np.abs(np.asarray(pred(t)) - y))/scale)
            if math.isfinite(val) and math.isfinite(error):
                model = Model(name, pred, val)
                model.error = error
                model.data = data
                candidates.append(model)
        except (ValueError, OverflowError, FloatingPointError):
            pass

    z = t - 0.5
    for degree in [1, 2, 4, 8, 12]:
        coef = np.polynomial.legendre.legfit(2*z, y, degree)
        add("axis-polynomial", lambda p, c=coef: np.polynomial.legendre.legval(2*np.asarray(p)-1, c), float(coef[0]))
        if candidates and candidates[-1].error < 1e-10:
            break
    lines = piecewise_lines(t, y)
    if lines:
        knots, pieces = lines

        def pred(p, k=np.asarray(knots), c=np.asarray(pieces)):
            p = np.asarray(p)
            region = np.searchsorted(k, p)
            return c[region, 0]*p + c[region, 1]

        edges = [0.0] + knots + [1.0]
        val = sum(a*(hi*hi-lo*lo)/2 + b*(hi-lo) for (a,b),lo,hi in zip(pieces, edges[:-1], edges[1:]))
        add("axis-piecewise-linear", pred, val, lines)
    if np.all(y > 0) or np.all(y < 0):
        sign = 1 if y[0] > 0 else -1
        logy = np.log(sign*y)
        coef = np.polynomial.polynomial.polyfit(z, logy, 2)
        try:
            val = exp_quadratic_integral(np.asarray([coef[0], coef[1], coef[2]]), 1)
            add("axis-exp-quadratic",
                lambda p, c=coef, s=sign: s*np.exp(np.clip(np.polynomial.polynomial.polyval(np.asarray(p)-0.5, c), -745, 700)),
                sign*val)
        except (OverflowError, ValueError):
            pass
        loglines = piecewise_lines(t, logy)
        if loglines:
            knots, pieces = loglines

            def logpred(p, k=np.asarray(knots), c=np.asarray(pieces), s=sign):
                p = np.asarray(p)
                region = np.searchsorted(k, p)
                return s*np.exp(np.clip(c[region, 0]*p + c[region, 1], -745, 700))

            edges = [0.0] + knots + [1.0]
            try:
                val = sign*sum(exp_line_integral(a,b,lo,hi) for (a,b),lo,hi in zip(pieces, edges[:-1], edges[1:]))
                add("axis-exp-piecewise-linear", logpred, val, loglines)
            except OverflowError:
                pass
        if np.min(np.abs(y)) > np.max(np.abs(y))*1e-100:
            norm = float(np.max(np.abs(y)))
            coef = np.polynomial.polynomial.polyfit(z, norm/y, 2)
            den = np.polynomial.polynomial.polyval(z, coef)
            if np.all(den * y > 0):
                peak = float(np.clip(-coef[1]/(2*coef[2]), -0.5, 0.5)) if coef[2] > 0 else 0.0
                try:
                    val = integrate.quad(
                        lambda p: norm/np.polynomial.polynomial.polyval(p-0.5, coef), 0, 1,
                        epsabs=abs(norm)*1e-13, epsrel=3e-13, limit=200, points=[peak+0.5],
                    )[0]
                    add("axis-reciprocal-quadratic",
                        lambda p, c=coef, n=norm: n/np.polynomial.polynomial.polyval(np.asarray(p)-0.5, c),
                        val)
                except (ValueError, OverflowError):
                    pass
    trig = trig_fit(t, y)
    if trig:
        w, a, b, c, err = trig
        add("axis-trigonometric",
            lambda p, w=w,a=a,b=b,c=c: c+np.cos((np.asarray(p)-0.5)[...,None]*w)@a + np.sin((np.asarray(p)-0.5)[...,None]*w)@b,
            c+float(np.sum(a*np.sinc(w/(2*np.pi)))), trig)
    else:
        trend = trig_fit_trend(t,y)
        if trend:
            candidates.append(trend)
    if not candidates or min(m.error for m in candidates)>1e-8:
        power = fit_power_axis(t,y)
        if power:
            candidates.append(power)
        if not power or power.error>1e-8:
            shifted = fit_shifted_power_axis(t,y)
            if shifted:
                candidates.append(shifted)
    spline = PchipInterpolator(t, y)
    model = Model("axis-spline", spline, spline.integrate(0.0, 1.0))
    model.error = 1e-4
    model.data = None
    candidates.append(model)
    candidates.sort(key=lambda m: m.error)
    return candidates


def axis_candidates(oracle, x, y):
    d = oracle.d
    anchor = np.full(d, 0.5)
    f0 = float(oracle([anchor])[0])
    if len(np.unique(y))==2:
        anchor = x[int(np.argmax(np.abs(y-np.median(y))))].copy()
        f0 = float(oracle([anchor])[0])
    elif f0==0 or abs(f0) < np.max(np.abs(y))*1e-100:
        anchor = x[int(np.argmax(np.abs(y)))].copy()
        f0 = float(oracle([anchor])[0])
        if f0==0:
            probes = np.asarray([np.zeros(d),np.ones(d)])
            values = oracle(probes)
            if np.max(np.abs(values))>0:
                anchor = probes[int(np.argmax(np.abs(values)))].copy()
                f0 = float(values[int(np.argmax(np.abs(values)))])
    nodes = np.linspace(0.0, 1.0, 33)
    if oracle.remaining < d*32:
        return [], None, None, None
    points = np.tile(anchor, (d*len(nodes), 1))
    for j in range(d):
        points[j*len(nodes):(j+1)*len(nodes),j] = nodes
    vals = oracle(points).reshape(d, len(nodes))
    fits = [fit_axis(nodes, row) for row in vals]
    refine_step_axes(oracle,anchor,nodes,vals,fits)
    refine_truncated_axes(oracle,anchor,nodes,vals,fits)
    candidates = []
    models = [f[0] for f in fits]
    integral = sum(m.integral for m in models) - (d-1)*f0
    candidates.append(Model(
        "additive-axis", lambda p, ms=models, f=f0: sum(m.predict(p[:,j]) for j,m in enumerate(ms))-(d-1)*f,
        integral,
    ))
    if f0 != 0 and abs(f0) > np.max(np.abs(y))*1e-100:
        def product(p, ms=models, f=f0):
            result = np.full(len(p), f)
            for j, model in enumerate(ms):
                result *= model.predict(p[:,j])/f
            return result
        val = f0
        for model in models:
            val *= model.integral/f0
        if math.isfinite(val):
            candidates.append(Model("product-axis", product, val))
    for baseline in set([float(np.min(y)),float(np.max(y)),float((np.min(y)+np.max(y))/2)]):
        if abs(f0-baseline)<max(float(np.ptp(y))*1e-12,1e-300):
            continue

        def offset_product(p,ms=models,f=f0,b=baseline):
            result = np.full(len(p),f-b)
            for j,model in enumerate(ms):
                result *= (model.predict(p[:,j])-b)/(f-b)
            return b+result

        val = f0-baseline
        for model in models:
            val *= (model.integral-baseline)/(f0-baseline)
        val += baseline
        if math.isfinite(val):
            candidates.append(Model("offset-product-axis",offset_product,val))
    if np.allclose(anchor, 0.5):
        trig = trigonometric_ridges(fits, x, y)
        if trig:
            candidates.append(trig)
    for model in candidates:
        if model.name in ("additive-axis","product-axis","offset-product-axis"):
            model.trusted = max(m.error for m in models)<1e-8
        quality(model, x, y)
    return candidates, (nodes, vals), anchor, fits


def refine_step_axes(oracle,anchor,nodes,vals,fits):
    eligible = []
    states = []
    valid_axes = set()
    for j,row in enumerate(vals):
        scale = max(float(np.ptp(row)),float(np.max(np.abs(row)))*1e-12,1e-300)
        changes = np.flatnonzero(np.abs(np.diff(row))>scale*1e-9)
        if not len(changes) or len(changes)>len(row)*.4:
            continue
        segments = np.r_[0,changes+1,len(row)]
        levels = []
        valid = True
        for start,end in zip(segments[:-1],segments[1:]):
            segment = row[start:end]
            if np.ptp(segment)>scale*1e-10:
                valid = False
                break
            levels.append(float(np.mean(segment)))
        if not valid:
            continue
        valid_axes.add(j)
        eligible.append((j,levels,changes))
        for change in changes:
            states.append([j,int(change),float(nodes[change]),float(nodes[change+1]),float(row[change]),float(row[change+1])])
    if not states or oracle.remaining<36*len(states)+256:
        return
    for iteration in range(36):
        points = []
        for j,index,lo,hi,left,right in states:
            p = anchor.copy()
            p[j] = (lo+hi)/2
            points.append(p)
        values = oracle(points)
        for state,v in zip(states,values):
            midpoint = (state[2]+state[3])/2
            left_match = abs(v-state[4])<=1e-10*max(abs(v),abs(state[4]),1e-300)
            right_match = abs(v-state[5])<=1e-10*max(abs(v),abs(state[5]),1e-300)
            if not left_match and not right_match:
                valid_axes.discard(state[0])
            if left_match:
                state[2] = midpoint
            else:
                state[3] = midpoint
    boundaries = {(j,index):(lo+hi)/2 for j,index,lo,hi,left,right in states}
    for j,levels,changes in eligible:
        if j not in valid_axes:
            continue
        knots = np.asarray([boundaries[(j,int(index))] for index in changes])
        levels = np.asarray(levels)
        edges = np.r_[0,knots,1]
        model = Model(
            "axis-steps",lambda p,k=knots,c=levels: c[np.searchsorted(k,np.asarray(p))],
            np.sum(levels*np.diff(edges)),
        )
        model.error = 0.0
        model.data = (knots,levels)
        fits[j].insert(0,model)


def trigonometric_ridges(fits, x, y):
    trig = []
    for family in fits:
        options = [m for m in family if m.name in ("axis-trigonometric","axis-trigonometric-trend") and m.error < 3e-5]
        if not options:
            trig.append(None)
        else:
            trig.append(options[0].data)
    counts = [len(v[0]) if v else 0 for v in trig]
    k = max(counts)
    if k == 0 or k > 8:
        return None
    d = x.shape[1]
    references = [i for i,v in enumerate(trig) if v and len(v[0]) == k]
    references.sort(key=lambda i: float(np.min(np.diff(np.r_[0,trig[i][0]]))),reverse=True)
    best = None
    for reference in references[:3]:
        if time_left() < 6:
            break
        w0,a0,b0,c0,e0 = trig[reference]
        directions = np.full((k,d),0.2)
        directions[:,reference] = w0
        for i,data in enumerate(trig):
            if i == reference or data is None:
                continue
            w,a,b,c,err = data
            cost = (a0[:,None]-a[None,:])**2+(np.abs(b0[:,None])-np.abs(b[None,:]))**2
            rows,cols = optimize.linear_sum_assignment(cost)
            directions[rows,i] = w[cols]*np.where(b[cols]*b0[rows]>=0,1.0,-1.0)
        model = refine_trigonometric(directions,x,y,max_nfev=240)
        if model:
            quality(model,x,y)
            if best is None or model.error < best.error:
                best = model
            if model.error < 1e-9:
                return model
    return best


def refine_truncated_axes(oracle,anchor,nodes,vals,fits):
    eligible = []
    states = []
    for j,row in enumerate(vals):
        nonzero = np.flatnonzero(row != 0)
        if len(nonzero)<4 or len(nonzero)==len(row) or np.any(np.diff(nonzero)!=1):
            continue
        sign = np.sign(row[nonzero[0]])
        if np.any(sign*row[nonzero]<=0):
            continue
        logrow = np.log(sign*row[nonzero])
        coef = np.polynomial.polynomial.polyfit(nodes[nonzero]-.5,logrow,2)
        residual = np.polynomial.polynomial.polyval(nodes[nonzero]-.5,coef)-logrow
        knots = []
        if np.max(np.abs(residual))<=1e-8:
            logpred = lambda p,c=coef: np.polynomial.polynomial.polyval(np.asarray(p)-.5,c)
        else:
            lines = piecewise_lines(nodes[nonzero],logrow)
            tfit = nodes[nonzero].copy()
            for _ in range(12):
                if lines:
                    break
                slopes = np.diff(logrow)/np.diff(tfit)
                changes = np.diff(slopes)
                scale = max(float(np.ptp(logrow)),1e-100)
                active = np.flatnonzero(np.abs(changes)>scale*2e-7)
                if not len(active) or len(active)>.6*len(changes):
                    break
                intervals = np.unique(np.r_[active,active+1])
                mids = (tfit[intervals]+tfit[intervals+1])/2
                if oracle.remaining<len(mids)+72*oracle.d+300:
                    break
                points = np.tile(anchor,(len(mids),1))
                points[:,j] = mids
                values = oracle(points)
                if np.any(sign*values<=0):
                    break
                order = np.argsort(np.r_[tfit,mids])
                tfit = np.r_[tfit,mids][order]
                logrow = np.r_[logrow,np.log(sign*values)][order]
                lines = piecewise_lines(tfit,logrow)
            if not lines:
                continue
            knots,pieces = lines

            def logpred(p,k=np.asarray(knots),c=np.asarray(pieces)):
                p = np.asarray(p)
                region = np.searchsorted(k,p)
                return c[region,0]*p+c[region,1]
        lower,upper = 0.0,1.0
        if nonzero[0]>0:
            states.append([j,"lower",float(nodes[nonzero[0]-1]),float(nodes[nonzero[0]])])
        if nonzero[-1]<len(row)-1:
            states.append([j,"upper",float(nodes[nonzero[-1]]),float(nodes[nonzero[-1]+1])])
        eligible.append([j,logpred,knots,sign,lower,upper])
    if not states or oracle.remaining < 36*len(states)+256:
        return
    for iteration in range(36):
        points = []
        for j,which,lo,hi in states:
            p = anchor.copy()
            p[j] = (lo+hi)/2
            points.append(p)
        values = oracle(points)
        for state,v in zip(states,values):
            inside = v != 0
            if (state[1]=="lower" and inside) or (state[1]=="upper" and not inside):
                state[3] = (state[2]+state[3])/2
            else:
                state[2] = (state[2]+state[3])/2
    bounds = {}
    for j,which,lo,hi in states:
        bounds[(j,which)] = (lo+hi)/2
    for j,logpred,knots,sign,lo,hi in eligible:
        lo,hi = bounds.get((j,"lower"),lo),bounds.get((j,"upper"),hi)

        def pred(p,lp=logpred,s=sign,lo=lo,hi=hi):
            p = np.asarray(p)
            return np.where((p>=lo)&(p<=hi),s*np.exp(np.clip(lp(p),-745,700)),0.0)

        val = integrate.quad(lambda p: float(pred(p)),lo,hi,epsabs=1e-13,epsrel=3e-13,
                             points=[k for k in knots if lo<k<hi],limit=100)[0]
        model = Model("axis-truncated-exponential",pred,val)
        model.error = 0.0
        model.data = (lo,hi,knots)
        fits[j].insert(0,model)


def refine_trigonometric(directions, x, y, max_nfev=200):
    k,d = directions.shape
    z = x-0.5
    angles = z@directions.T
    design = np.column_stack([np.ones(len(x)), np.cos(angles), np.sin(angles)])
    coef = linalg.lstsq(design, y, lapack_driver="gelsy")[0]
    p0 = np.r_[directions.ravel(),coef]
    scale = max(float(np.std(y)), 1e-100)

    def residual(p):
        if time_left() < 3:
            raise TimeoutError("trigonometric approximation reserve")
        freq = p[:k*d].reshape(k,d)
        angles = z@freq.T
        c = p[k*d:]
        return (c[0] + np.cos(angles)@c[1:1+k] + np.sin(angles)@c[1+k:] - y)/scale

    def jac(p):
        angles = z@p[:k*d].reshape(k,d).T
        c = p[k*d:]
        cos, sin = np.cos(angles), np.sin(angles)
        dtheta = -sin*c[1:1+k]+cos*c[1+k:]
        df = (dtheta[:,:,None]*z[:,None,:]).reshape(len(x),k*d)
        return np.column_stack([df,np.ones(len(x)),cos,sin])/scale

    if len(y) <= len(p0):
        return None
    got = optimize.least_squares(
        residual,p0,jac=jac,method="lm",max_nfev=max_nfev,
        ftol=1e-12,xtol=1e-12,gtol=1e-12,
    )
    freq = got.x[:k*d].reshape(k,d)
    coef = got.x[k*d:]
    integral = coef[0] + np.sum(coef[1:1+k]*np.prod(np.sinc(freq/(2*np.pi)),axis=1))
    return Model(
        "trigonometric-ridges",
        lambda p, w=freq,c=coef: c[0]+np.cos((p-0.5)@w.T)@c[1:1+k]+np.sin((p-0.5)@w.T)@c[1+k:],
        integral,
    )


def oblique_trigonometric(oracle,fits,x,y):
    if fits is None or oracle.remaining<64*oracle.d+256 or time_left()<10:
        return None
    d = oracle.d
    lines = []
    credible = 0
    identities = []
    periodic = 0
    for i,family in enumerate(fits):
        options = [m for m in family if m.name in ("axis-trigonometric","axis-trigonometric-trend") and m.error<3e-5]
        if not options:
            continue
        model = options[0]
        w,a,b,c,err = model.data
        spline = next(m for m in family if m.name=="axis-spline")
        spread = float(np.std(spline.predict(np.linspace(0,1,33))))
        if np.sum(np.hypot(a,b))<30*max(spread,1e-100):
            credible += 1
            energy = np.hypot(a,b)
            mask = energy>np.max(energy)*.02
            identities.append((a[mask],np.abs(b[mask])))
            if len(w) and np.all(np.abs(w/(2*np.pi)-np.round(w/(2*np.pi)))<1e-5):
                periodic += 1
        direction = np.zeros(d)
        direction[i] = 1
        lines.append((direction,model.data))
    if credible<min(2,d):
        return None
    shared = False
    for j,(a,b) in enumerate(identities):
        for aa,bb in identities[:j]:
            if not len(a) or not len(aa):
                continue
            cost = (a[:,None]-aa[None,:])**2+(b[:,None]-bb[None,:])**2
            row,col = optimize.linear_sum_assignment(cost)
            strength = max(float(np.linalg.norm(np.r_[a,b,aa,bb])),1e-100)
            matches = np.sum(cost[row,col]<(strength*.003)**2)
            if matches>=min(2,len(a),len(aa)):
                shared = True
    if not shared and periodic<min(2,d):
        return None
    rng = np.random.default_rng(oracle.seed ^ 0x6F41AB3)
    rotation = linalg.qr(rng.normal(size=(d,d)),mode="economic")[0].T
    rotation /= np.max(np.abs(rotation),axis=1)[:,None]
    maximum = np.full(d,50.0)
    for i,family in enumerate(fits):
        options = [m for m in family if m.name in ("axis-trigonometric","axis-trigonometric-trend")]
        if options:
            maximum[i] = max(1.0,float(np.max(options[0].data[0])))
    nodes = np.linspace(0,1,65)
    best = None

    def reconstruct():
        if not lines:
            return None
        k = max(len(data[0]) for direction,data in lines)
        refs = [j for j,(v,data) in enumerate(lines) if len(data[0])==k]
        refs.sort(key=lambda j: float(np.min(np.diff(np.r_[0,lines[j][1][0]]))),reverse=True)
        answer = None
        for reference in refs[:3]:
            w0,a0,b0,c0,e0 = lines[reference][1]
            strength = max(float(np.linalg.norm(np.r_[a0,b0])),1e-100)
            vectors,frequencies = [],[]
            for direction,data in lines:
                w,a,b,c,err = data
                if len(w)!=k:
                    continue
                cost = (a0[:,None]-a[None,:])**2+(np.abs(b0[:,None])-np.abs(b[None,:]))**2
                row,col = optimize.linear_sum_assignment(cost)
                if np.sqrt(np.sum(cost[row,col]))>strength*.045:
                    continue
                projection = w[col]*np.where(b[col]*b0>=0,1.,-1.)
                vectors.append(direction)
                frequencies.append(projection)
            if len(vectors)<d:
                continue
            v = np.asarray(vectors)
            if np.linalg.matrix_rank(v,tol=1e-5)<d:
                continue
            directions = linalg.lstsq(v,np.asarray(frequencies),lapack_driver="gelsy")[0].T
            candidate = refine_trigonometric(directions,x,y,max_nfev=240)
            if candidate:
                quality(candidate,x,y)
                if answer is None or candidate.error<answer.error:
                    answer = candidate
                if candidate.error<1e-9:
                    return candidate
        return answer

    for i in range(d+5):
        if oracle.remaining<320 or time_left()<6:
            break
        if i<d:
            direction = rotation[i].copy()
        else:
            direction = rng.uniform(-1,1,d)
            direction /= np.max(np.abs(direction))
        scale = min(1.0,160/max(float(np.abs(direction)@maximum),1.0))
        direction *= scale
        points = .5+(nodes[:,None]-.5)*direction[None,:]
        values = oracle(points)
        data = trig_fit(nodes,values)
        if data is None:
            trend = trig_fit_trend(nodes,values)
            if trend:
                data = trend.data
        if data:
            lines.append((direction,data))
        if i>=d-1:
            model = reconstruct()
            if model:
                if best is None or model.error<best.error:
                    best = model
                if model.error<1e-9:
                    return model
    return best


def refine_hinges(directions, offsets, x, y):
    directions = np.asarray(directions, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    k,d = directions.shape
    if len(y) < (k+1)*(d+1)+4:
        return None
    affine = np.column_stack([np.ones(len(x)), x-0.5])
    feature = np.maximum(x@directions.T+offsets,0)
    coef = linalg.lstsq(np.column_stack([affine,feature]), y, lapack_driver="gelsy")[0]
    signs = np.where(coef[d+1:] >= 0, 1.0, -1.0)
    directions = directions*np.abs(coef[d+1:,None])
    offsets = offsets*np.abs(coef[d+1:])
    linear = coef[:d+1]
    previous = None
    for iteration in range(8):
        active = x@directions.T+offsets > 0
        design = np.column_stack([affine] + [signs[j]*active[:,j,None]*np.column_stack([x,np.ones(len(x))]) for j in range(k)])
        solution = linalg.lstsq(design, y, lapack_driver="gelsy", cond=1e-12)[0]
        linear = solution[:d+1]
        shapes = solution[d+1:].reshape(k,d+1)
        directions = shapes[:,:d]
        offsets = shapes[:,d]
        if previous is not None and np.array_equal(active,previous):
            break
        previous = active
    integral = float(linear[0])
    for a,b,s in zip(directions,offsets,signs):
        integral += s*hinge_integral(a,b)
    return Model(
        "affine-hinges",
        lambda p,a=directions,b=offsets,s=signs,c=linear: c[0]+(p-0.5)@c[1:]+np.maximum(p@a.T+b,0)@s,
        integral,
    )


def reconstruct_hinges(oracle, axis, anchor, x, y):
    if axis is None or oracle.remaining < 128:
        return None
    nodes, vals = axis
    d = oracle.d
    line_data = []

    def plausible(row,t):
        slope = np.diff(row)/np.diff(t)
        change = np.diff(slope)
        scale = max(float(np.ptp(row)),float(np.max(np.abs(row)))*1e-10,1e-300)
        return np.mean(np.abs(change)<scale*2e-7)>.40 and np.any(np.abs(change)>scale*1e-6)

    for j, row in enumerate(vals):
        pieces = piecewise_lines(nodes,row)
        if (pieces and pieces[0]) or (not pieces and plausible(row,nodes)):
            a,b = anchor.copy(),anchor.copy()
            a[j],b[j] = 0,1
            line_data.append((a,b,nodes,row,pieces))
    if not line_data:
        return None
    if oracle.remaining >= 33+max(128,4*d):
        diagonal = np.outer(nodes,np.ones(d))
        diagonal_y = oracle(diagonal)
        pieces = piecewise_lines(nodes,diagonal_y)
        if (pieces and pieces[0]) or (not pieces and plausible(diagonal_y,nodes)):
            line_data.insert(0,(np.zeros(d),np.ones(d),nodes,diagonal_y,pieces))
    if oracle.remaining>=33+4*d+300:
        lo = np.arange(d)%2
        hi = 1-lo
        points = lo[None,:]+nodes[:,None]*(hi-lo)[None,:]
        values = oracle(points)
        pieces = piecewise_lines(nodes,values)
        if (pieces and pieces[0]) or (not pieces and plausible(values,nodes)):
            line_data.insert(1,(lo,hi,nodes,values,pieces))
    directions, offsets = [], []
    model = None
    for lo,hi,t,row,description in line_data:
        for level in range(3):
            if description is not None or oracle.remaining < len(t)+4*d+300:
                break
            mids = (t[:-1]+t[1:])/2
            fresh = oracle(lo[None,:]+mids[:,None]*(hi-lo)[None,:])
            order = np.argsort(np.r_[t,mids])
            t = np.r_[t,mids][order]
            row = np.r_[row,fresh][order]
            description = piecewise_lines(t,row)
        if description is None:
            continue
        knots,pieces = description
        edges = [0.0]+knots+[1.0]
        for j,knot in enumerate(knots):
            if len(directions) >= 10 or oracle.remaining < 4*d+256 or time_left() < 8:
                break
            p = lo+(hi-lo)*knot
            if any(abs(float(a@p+b))<2e-7 for a,b in zip(directions,offsets)):
                continue
            gap = min(knot-edges[j],edges[j+2]-knot)
            delta = min(0.015,gap*0.25)
            if delta < 1e-6:
                continue
            pm = lo+(hi-lo)*(knot-delta)
            pp = lo+(hi-lo)*(knot+delta)
            step = min(0.001,delta*0.08)
            queries = []
            for base in [pm,pp]:
                for i in range(d):
                    left,right = base.copy(),base.copy()
                    left[i] = max(0.0,left[i]-step)
                    right[i] = min(1.0,right[i]+step)
                    queries.extend([left,right])
            result = oracle(queries).reshape(2,d,2)
            widths = np.asarray([[queries[base*2*d+2*i+1][i]-queries[base*2*d+2*i][i] for i in range(d)] for base in range(2)])
            gradient = (result[:,:,1]-result[:,:,0])/widths
            direction = gradient[1]-gradient[0]
            norm = float(np.linalg.norm(direction))
            if norm < max(float(np.std(y)),1e-100)*1e-8:
                continue
            direction /= norm
            offset = -float(direction@p)
            duplicate = False
            for old,oldb in zip(directions,offsets):
                dot = float(old@direction)
                if abs(dot) > 1-1e-7 and abs(oldb-np.sign(dot)*offset)<1e-5:
                    duplicate = True
                    break
            if duplicate:
                continue
            directions.append(direction)
            offsets.append(offset)
            model = refine_hinges(directions,offsets,np.asarray(oracle.x),np.asarray(oracle.y))
            if model and quality(model,x,y) < 2e-8:
                return model
    return model


def gaussian_model(parameters, d, origin, scale):
    constant = float(parameters[0])
    shapes = parameters[1:].reshape(-1,2*d+1)
    amp = shapes[:,0]
    centers = shapes[:,1:1+d]
    precision = np.exp(shapes[:,1+d:])
    k = len(shapes)
    val = origin + scale*constant
    for w,c,q in zip(amp,centers,precision):
        coef = np.r_[math.log(max(w,1e-300))-np.sum(q*(0.5-c)**2), -2*q*(0.5-c), -q]
        val += scale*exp_quadratic_integral(coef,d)

    def predict(p,a=amp,c=centers,q=precision,b=constant,o=origin,s=scale):
        delta = p[:,None,:]-c[None,:,:]
        return o+s*(b+np.exp(-np.sum(q[None,:,:]*delta*delta,axis=2))@a)

    model = Model("gaussian-expansion-"+str(k),predict,val)
    model.parameters = parameters.copy()
    model.centers = centers.copy()
    model.widths = np.sqrt(precision)
    model.origin = origin
    model.scale = scale
    return model


def fit_gaussians(x, y, max_components=6, warm=None):
    """Greedy nonlinear refinement of a positive Gaussian expansion."""
    if np.any(y < 0) or time_left() < 8:
        return None
    d = x.shape[1]
    origin = float(np.min(y))
    scale = max(float(np.max(y)-origin),1e-300)
    target = (y-origin)/scale
    weights = 1/np.sqrt(0.04+target)
    best = None
    parameters = np.asarray([0.0])
    predicted = np.zeros(len(y))
    lower = [-2.0]
    upper = [2.0]
    initial_k = 0
    if warm is not None:
        parameters = warm.parameters.copy()
        parameters[0] = (warm.origin+warm.scale*parameters[0]-origin)/scale
        shapes = parameters[1:].reshape(-1,2*d+1)
        shapes[:,0] *= warm.scale/scale
        initial_k = len(shapes)
        for _ in range(initial_k):
            lower.extend([0.0]+[-.5]*d+[math.log(.05)]*d)
            upper.extend([1e5]+[1.5]*d+[math.log(4e4)]*d)
        predicted = (warm.predict(x)-origin)/scale
    for k in range(max(1,initial_k),max_components+1):
        if time_left() < 7:
            break
        if k > initial_k:
            residual_y = target-predicted
            peak = int(np.argmax(residual_y))
            center = x[peak].copy()
            amp = max(float(residual_y[peak]),0.01)
            logq = np.full(d,math.log(25.0))
            mask = (residual_y > amp*.025) & (np.sum((x-center)**2,axis=1) < max(0.45,d*0.09))
            if np.sum(mask) >= 2*d+4:
                design = quadratic_matrix(x[mask])
                coef = linalg.lstsq(design,np.log(residual_y[mask]),lapack_driver="gelsy")[0]
                q = -coef[1+d:]
                if np.all(q > .05) and np.all(q < 4e4):
                    proposed = .5+coef[1:1+d]/(2*q)
                    if np.all(proposed > -.3) and np.all(proposed < 1.3):
                        center = proposed
                        logq = np.log(q)
                        amp = min(100.0,math.exp(min(20.0,coef[0]+np.sum(q*(.5-center)**2))))
            new = np.r_[amp,center,logq]
            parameters = np.r_[parameters,new]
            lower.extend([0.0]+[-.5]*d+[math.log(.05)]*d)
            upper.extend([1e5]+[1.5]*d+[math.log(4e4)]*d)
        parameters = np.maximum(np.minimum(parameters,np.asarray(upper)-1e-12),np.asarray(lower)+1e-12)

        def values_and_jac(p, need_jac=False):
            shapes = p[1:].reshape(k,2*d+1)
            a = shapes[:,0]
            c = shapes[:,1:1+d]
            q = np.exp(shapes[:,1+d:])
            delta = x[:,None,:]-c[None,:,:]
            delta2q = delta*delta*q[None,:,:]
            atoms = np.exp(-np.sum(delta2q,axis=2))
            pred = p[0]+atoms@a
            if not need_jac:
                return pred
            block = np.concatenate([
                atoms[:,:,None],
                (2*atoms*a)[:,:,None]*delta*q[None,:,:],
                -(atoms*a)[:,:,None]*delta2q,
            ],axis=2).reshape(len(x),-1)
            return np.column_stack([np.ones(len(x)),block])

        def residual(p):
            if time_left() < 4:
                raise TimeoutError("approximation time reserve")
            return (values_and_jac(p)-target)*weights

        def jac(p):
            return values_and_jac(p,True)*weights[:,None]

        try:
            got = optimize.least_squares(
                residual,parameters,jac=jac,bounds=(lower,upper),max_nfev=130,
                ftol=3e-11,xtol=3e-11,gtol=3e-11,
            )
            parameters = got.x
            predicted = values_and_jac(parameters)
            model = gaussian_model(parameters,d,origin,scale)
            error = quality(model,x,y)
            if best is None or error < best.error:
                best = model
            if error < 1e-9:
                return model
        except (ValueError,OverflowError,linalg.LinAlgError,TimeoutError):
            break
    return best


def explore_gaussians(oracle, model):
    d = oracle.d
    shapes = model.parameters[1:].reshape(-1,2*d+1)
    order = np.argsort(shapes[:,0])[::-1]
    points = []
    for j in order:
        if shapes[j,0] < np.max(shapes[:,0])*1e-5:
            continue
        center = np.clip(model.centers[j],0,1)
        points.append(center)
        for i in range(d):
            for shift in [-1.25,-0.4,0.4,1.25]:
                p = center.copy()
                p[i] = np.clip(p[i]+shift/model.widths[j,i],0,1)
                points.append(p)
        if len(points)+4*d+1 > oracle.remaining-256:
            break
    if points and len(points) <= oracle.remaining-256:
        oracle(points)
        return True
    return False


def improve_gaussian_seed(model,x,y):
    d = x.shape[1]
    parameters = model.parameters.copy()
    shapes = parameters[1:].reshape(-1,2*d+1)
    if len(shapes)>2 and model.error>1e-8:
        keep = shapes[:,0] >= np.max(shapes[:,0])*.002
        if np.sum(keep)>=2 and not np.all(keep):
            shapes = shapes[keep].copy()
            parameters = np.r_[parameters[0],shapes.ravel()]
    target = (y-model.origin)/model.scale
    for j in np.argsort(shapes[:,0])[::-1]:
        q = np.exp(shapes[j,1+d:])
        centers = shapes[:,1:1+d]
        precision = np.exp(shapes[:,1+d:])
        delta = x[:,None,:]-centers[None,:,:]
        atoms = np.exp(-np.sum(precision[None,:,:]*delta*delta,axis=2))
        residual = target-parameters[0]-atoms@shapes[:,0]+atoms[:,j]*shapes[j,0]
        radius = np.sum(q*(x-shapes[j,1:1+d])**2,axis=1)
        mask = (residual>max(float(np.max(residual)),1e-300)*.015)&(radius<12)
        if np.sum(mask)<2*d+3:
            continue
        design = quadratic_matrix(x[mask])
        coef,_,rank,_ = linalg.lstsq(design,np.log(residual[mask]),lapack_driver="gelsy")
        if rank<2*d+1 or np.sqrt(np.mean((design@coef-np.log(residual[mask]))**2))>.25:
            continue
        qnew = -coef[1+d:]
        if np.any(qnew<.05) or np.any(qnew>4e4):
            continue
        center = .5+coef[1:1+d]/(2*qnew)
        if np.any(center<-.4) or np.any(center>1.4):
            continue
        logamp = coef[0]+np.sum(qnew*(.5-center)**2)
        if logamp>math.log(1e5):
            continue
        shapes[j] = np.r_[math.exp(logamp),center,np.log(qnew)]
    return gaussian_model(np.r_[parameters[0],shapes.ravel()],d,model.origin,model.scale)


def fit_log_gaussians(x,y,warm):
    if time_left()<6 or np.any(y<0):
        return None
    positive = y>0
    if np.sum(positive)<len(warm.parameters)+8:
        return None
    x,y = x[positive],y[positive]
    d = x.shape[1]
    target = np.log(y)
    scale = float(np.max(y))
    old = warm.parameters[1:].reshape(-1,2*d+1)
    k = len(old)
    minimum_log = float(np.min(target))
    baseline = max(0.0,warm.origin+warm.scale*warm.parameters[0])
    if baseline>float(np.min(y))*.9:
        logbase = minimum_log-1e-8
    else:
        logbase = max(-744.0,minimum_log-35)
    shapes = old.copy()
    shapes[:,0] = np.log(np.maximum(old[:,0]*warm.scale,1e-300))
    p0 = np.r_[logbase,shapes.ravel()]
    upper_amp = min(700.0,float(np.max(target))+100)
    lower = np.r_[-745,np.tile(np.r_[-745,[-.5]*d,[math.log(.05)]*d],k)]
    upper = np.r_[minimum_log,np.tile(np.r_[upper_amp,[1.5]*d,[math.log(4e4)]*d],k)]
    p0 = np.maximum(np.minimum(p0,upper-1e-12),lower+1e-12)

    def values(p,need_jac=False):
        shapes = p[1:].reshape(k,2*d+1)
        delta = x[:,None,:]-shapes[None,:,1:1+d]
        precision = np.exp(shapes[:,1+d:])
        squared = delta*delta*precision[None,:,:]
        logs = shapes[None,:,0]-np.sum(squared,axis=2)
        all_logs = np.column_stack([np.full(len(x),p[0]),logs])
        pred = special.logsumexp(all_logs,axis=1)
        if not need_jac:
            return pred
        posterior = np.exp(all_logs-pred[:,None])
        block = np.concatenate([
            posterior[:,1:,None],
            2*posterior[:,1:,None]*delta*precision[None,:,:],
            -posterior[:,1:,None]*squared,
        ],axis=2).reshape(len(x),-1)
        return np.column_stack([posterior[:,0],block])

    def residual(p):
        if time_left()<3:
            raise TimeoutError("logarithmic fit time reserve")
        return values(p)-target

    try:
        got = optimize.least_squares(
            residual,p0,jac=lambda p:values(p,True),bounds=(lower,upper),
            max_nfev=140,ftol=2e-11,xtol=2e-11,gtol=2e-11,
        )
        shapes = got.x[1:].reshape(k,2*d+1)
        shapes[:,0] = np.exp(shapes[:,0]-math.log(scale))
        params = np.r_[math.exp(got.x[0]-math.log(scale)),shapes.ravel()]
        model = gaussian_model(params,d,0.0,scale)
        quality(model,x,y)
        return model
    except (ValueError,OverflowError,TimeoutError,linalg.LinAlgError):
        return None


def polynomial_indices(d, degree, product_limit=None):
    result = []

    def recurse(prefix,remaining,product):
        if len(prefix)==d:
            result.append(tuple(prefix))
            return
        for k in range(remaining+1):
            if product_limit is not None and product*(k+1)>product_limit:
                break
            recurse(prefix+[k],remaining-k,product*(k+1))

    recurse([],degree,1)
    result.sort(key=lambda a:(sum(a),np.prod(np.asarray(a)+1),a))
    return result


def polynomial_matrix(x,indices):
    maximum = max(max(a) for a in indices)
    z = 2*x-1
    values = np.empty((len(x),x.shape[1],maximum+1))
    values[:,:,0] = 1.0
    if maximum:
        values[:,:,1] = z
    for k in range(2,maximum+1):
        values[:,:,k] = ((2*k-1)*z*values[:,:,k-1]-(k-1)*values[:,:,k-2])/k
    values *= np.sqrt(2*np.arange(maximum+1)+1)[None,None,:]
    matrix = np.ones((len(x),len(indices)))
    for column,index in enumerate(indices):
        for j,k in enumerate(index):
            if k:
                matrix[:,column] *= values[:,j,k]
    return matrix


def polynomial_control(oracle,qx,qy,base):
    if len(qy)<512 or time_left()<4:
        return None
    d = oracle.d
    n = len(qy)
    ntrain = 2**int(math.floor(math.log2(n*.62)))
    ntrain = max(256,min(ntrain,n-128))
    test_x,test_y = qx[ntrain:],qy[ntrain:]
    all_x = np.asarray(oracle.x)
    all_y = np.asarray(oracle.y)
    validation_keys = {tuple(p) for p in test_x}
    mask = np.asarray([tuple(p) not in validation_keys for p in all_x])
    train_x,train_y = all_x[mask],all_y[mask]
    if base is not None:
        train_y = train_y-base.predict(train_x)
        test_y = test_y-base.predict(test_x)
        base_integral = base.integral
    else:
        base_integral = 0.0
    original_rms = max(float(np.sqrt(np.mean(test_y**2))),1e-300)
    if d==1:
        levels = [4,8,12,20,32,48]
    elif d==2:
        levels = [3,5,8,11,15,19,24]
    elif d==3:
        levels = [2,3,5,7,9,11,13]
    elif d<=5:
        levels = [2,3,4,5,6,7]
    else:
        levels = [1,2,3,4,5]
    maxfeatures = min(650,int(len(train_y)*.64))
    configurations = [(levels[-1],None)]
    if d>=3:
        configurations.append((10,10 if d<=5 else 8))
    best = None
    for degree,product_limit in configurations:
        if time_left()<3:
            break
        indices = polynomial_indices(d,degree,product_limit)
        indices = indices[:maxfeatures]
        if len(indices)<2:
            continue
        train_matrix = polynomial_matrix(train_x,indices)
        test_matrix = polynomial_matrix(test_x,indices)
        q,r = linalg.qr(train_matrix,mode="economic",overwrite_a=True,check_finite=False)
        projected = q.T@train_y
        sizes = sorted(set([sum(sum(index)<=level for index in indices) for level in levels]+[len(indices)]))
        for size in sizes:
            if size<2 or size>len(projected):
                continue
            if np.min(np.abs(np.diag(r[:size,:size])))<1e-7:
                continue
            coef = linalg.solve_triangular(r[:size,:size],projected[:size],check_finite=False)
            residual = test_y-test_matrix[:,:size]@coef
            rms = float(np.sqrt(np.mean(residual**2)))
            if best is None or rms<best[0]:
                estimate = float(base_integral+coef[0]+np.mean(residual))
                best = (rms,estimate)
    if best and best[0] < original_rms*.45:
        return best[1]
    return None


def hinge_integral(direction, offset, power=1):
    """Integral of the positive part of an affine form to an integer power."""
    a = np.asarray(direction, dtype=float)
    c = float(offset + np.sum(a[a < 0]))
    a = np.abs(a)
    a = a[a > 1e-15 * max(1.0, float(np.max(a, initial=0)))]
    if len(a) == 0:
        return max(c, 0.0)**power
    if c >= 0:
        moments = np.zeros(power + 1)
        moments[0] = 1
        for width in a:
            new = np.zeros_like(moments)
            for k in range(power + 1):
                for j in range(k + 1):
                    new[k] += math.comb(k, j) * moments[k-j] * width**j / (j+1)
            moments = new
        return float(sum(math.comb(power, j) * c**(power-j) * moments[j] for j in range(power+1)))
    if c + np.sum(a) <= 0:
        return 0.0
    if power == 1 and c + np.sum(a)/2 > 0:
        return c + np.sum(a)/2 + hinge_integral(-direction, -offset, 1)
    with localcontext() as ctx:
        ctx.prec = 65
        dc = Decimal.from_float(c)
        widths = [Decimal.from_float(float(v)) for v in a]
        terms = [(dc, 1)]
        for w in widths:
            terms += [(s+w, -sign) for s, sign in terms]
        total = Decimal(0)
        exponent = len(a) + power
        for s, sign in terms:
            if s > 0:
                total += sign * s**exponent
        den = Decimal(math.factorial(exponent))
        for w in widths:
            den *= w
        return float(((-1)**len(a)) * total * math.factorial(power) / den)


def solve(oracle):
    d, budget = oracle.d, oracle.budget
    m = int(math.ceil(math.log2(max(budget, 2))))
    net = qmc.Sobol(d, scramble=True, seed=oracle.seed).random_base2(m)[:budget]
    n0 = min(256, budget)
    y0 = oracle(net[:n0])
    if budget<32:
        return stable_mean(y0)
    best = None
    train = np.arange(n0) % 4 != 0
    if n0 >= 32 and np.ptp(y0)>max(float(np.max(np.abs(y0)))*1e-14,1e-300):
        models = cheap_models(net[:n0][train], y0[train])
        for model in models:
            quality(model, net[:n0][~train], y0[~train])
        models.sort(key=lambda model: model.error)
        if models and models[0].error < 1e-8:
            best = models[0]
            ncheck = min(256, oracle.remaining)
            if ncheck:
                check_x = net[n0:n0+ncheck]
                check_y = oracle(check_x)
                if quality(best, check_x, check_y) < 1e-8:
                    return best.integral
    models = []
    axis = anchor = fits = None
    if oracle.remaining > d*32+128 and time_left() > 10:
        models,axis,anchor,fits = axis_candidates(oracle,net[:n0],y0)
        models.sort(key=lambda model: model.error)
        exact = [model for model in models if model.error<2e-8 and getattr(model,"trusted",True)]
        if exact:
            best = exact[0]
        elif time_left() > 8:
            oblique = oblique_trigonometric(oracle,fits,net[:n0],y0)
            if oblique:
                quality(oblique,net[:n0],y0)
                models.append(oblique)
                if oblique.error<2e-8:
                    best = oblique
            hinge = reconstruct_hinges(oracle,axis,anchor,net[:n0],y0)
            if hinge:
                quality(hinge,net[:n0],y0)
                models.append(hinge)
                if hinge.error < 2e-8:
                    best = hinge
    if best is None or best.error >= 2e-8:
        if np.all(y0 >= 0) and time_left() > 12:
            all_x = np.asarray(oracle.x)
            all_y = np.asarray(oracle.y)
            fitmask = np.ones(len(all_x),dtype=bool)
            fitmask[:n0] = train
            gaussian = fit_gaussians(all_x[fitmask],all_y[fitmask])
            if gaussian:
                quality(gaussian,net[:n0],y0)
                for round_index in range(2):
                    if round_index and gaussian.error < 1e-10:
                        break
                    if time_left() < 10 or oracle.remaining < 4*d+300:
                        break
                    if not explore_gaussians(oracle,gaussian):
                        break
                    all_x = np.asarray(oracle.x)
                    all_y = np.asarray(oracle.y)
                    fitmask = np.ones(len(all_x),dtype=bool)
                    fitmask[:n0] = train
                    seed = improve_gaussian_seed(gaussian,all_x[fitmask],all_y[fitmask])
                    refined = fit_gaussians(all_x[fitmask],all_y[fitmask],warm=seed)
                    if refined is None:
                        break
                    quality(refined,net[:n0],y0)
                    gaussian = refined
                    if gaussian.error < 1e-10:
                        break
                models.append(gaussian)
                structured_error = quality(gaussian,np.asarray(oracle.x),np.asarray(oracle.y))
                sample_error = quality(gaussian,net[:n0],y0)
                gaussian.error = max(structured_error,sample_error)
                if gaussian.error < 2e-8:
                    best = gaussian
    if best is not None and best.error < 2e-8:
        ncheck = min(256,oracle.remaining)
        if ncheck:
            check_x = net[n0:n0+ncheck]
            check_y = oracle(check_x)
            if quality(best,check_x,check_y) < 2e-8:
                return best.integral
    count = oracle.remaining
    if count:
        existing = 0
        while existing < len(net) and tuple(net[existing]) in oracle.cache:
            existing += 1
        end = min(len(net),existing+count)
        oracle(net[existing:end])
    available = [p for p in net if tuple(p) in oracle.cache]
    qx = np.asarray(available)
    qy = np.asarray([oracle.cache[tuple(p)] for p in available])
    for model in models:
        quality(model,qx,qy)
    gaussian_options = [model for model in models if model.name.startswith("gaussian-expansion")]
    if gaussian_options and np.all(qy>=0) and time_left()>7:
        warm = min(gaussian_options,key=lambda model:model.error)
        if warm.error>1e-10:
            all_x,all_y = np.asarray(oracle.x),np.asarray(oracle.y)
            if time_left()>12:
                seed = improve_gaussian_seed(warm,all_x,all_y)
                refined = fit_gaussians(all_x,all_y,warm=seed)
                if refined:
                    quality(refined,qx,qy)
                    models.append(refined)
                    if refined.error<warm.error:
                        warm = refined
            if time_left()>6 and warm.error>1e-10:
                logarithmic = fit_log_gaussians(all_x,all_y,warm)
                if logarithmic:
                    quality(logarithmic,qx,qy)
                    models.append(logarithmic)
    models = sorted((model for model in models if math.isfinite(model.integral)),key=lambda model:model.error)
    base = models[0] if models and models[0].error<.65 else None
    control = polynomial_control(oracle,qx,qy,base)
    if control is not None:
        return control
    if models and models[0].error < 0.5:
        model = models[0]
        return float(model.integral+stable_mean(qy-model.predict(qx)))
    return stable_mean(qy)


def qmc_fallback(oracle):
    m = int(math.ceil(math.log2(max(oracle.budget,2))))
    net = qmc.Sobol(oracle.d,scramble=True,seed=oracle.seed).random_base2(m)[:oracle.budget]
    missing = [p for p in net if tuple(p) not in oracle.cache][:oracle.remaining]
    if missing:
        oracle(missing)
    vals = [oracle.cache[tuple(p)] for p in net if tuple(p) in oracle.cache]
    if vals:
        return stable_mean(vals)
    return stable_mean(oracle.y)


def main():
    init = json.loads(sys.stdin.readline())
    if init.get("type") != "init":
        raise ValueError("missing initialization")
    oracle = Oracle(init)
    try:
        estimate = solve(oracle)
    except Exception as exc:
        sys.stderr.write("numerical fallback: " + repr(exc) + "\n")
        if not oracle.y:
            raise
        estimate = qmc_fallback(oracle)
    if not math.isfinite(estimate):
        estimate = qmc_fallback(oracle)
    estimate *= oracle.volume
    if not math.isfinite(estimate):
        raise ValueError("non-finite final integral")
    sys.stdout.write(json.dumps({"type": "result", "estimate": estimate}, allow_nan=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
