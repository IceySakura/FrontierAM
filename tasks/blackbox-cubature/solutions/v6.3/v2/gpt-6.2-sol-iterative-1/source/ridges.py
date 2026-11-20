"""Ridge surrogates learned from oblique restrictions of the integrand."""

import itertools
import math
from dataclasses import dataclass
from decimal import Decimal, localcontext

import numpy as np
from scipy import linalg, optimize
from numpy.polynomial import Polynomial

from core import Model, agrees, legendre_features, least


@dataclass
class Line:
    start: np.ndarray
    v: np.ndarray
    t: np.ndarray
    y: np.ndarray

    def points(self, t):
        return self.start+np.asarray(t)[..., None]*self.v


def central_line(s, n=65, attempt=0):
    d = s.d
    if attempt == 0:
        # Use unequal slopes so distinct frequencies don't trivially alias.
        v = s.rng.uniform(.48, 1., d)*s.rng.choice([-1., 1.], d)
        v[0] = 1.
        mid = np.full(d, .5)
    else:
        v = s.rng.uniform(-1., 1., d)
        v[np.argmax(np.abs(v))] = 1.
        mid = s.rng.uniform(.35, .65, d)
    v *= .8 if attempt == 0 else .64
    t = np.linspace(0., 1., n)
    a = mid-v*.5
    return Line(a, v, t, s.ask(a+t[:, None]*v)/s.scale)


def connecting_line(s, x, q, n=49):
    v = q-x
    ln = np.max(np.abs(v))
    v = v/max(ln, 1.e-14)
    nonzero = np.abs(v) > 1.e-12
    limit_a, limit_b = (2.e-7-x[nonzero])/v[nonzero], (1.-2.e-7-x[nonzero])/v[nonzero]
    l = np.max(np.minimum(limit_a, limit_b))
    r = np.min(np.maximum(limit_a, limit_b))
    if not r > l or ln < 1.e-10:
        return central_line(s, n, attempt=1)
    v = v*(r-l)
    a = x+(l/(r-l))*v
    t = np.linspace(0., 1., n)
    return Line(a, v, t, s.ask(a+t[:, None]*v)/s.scale)


def restrict_line(s, line, a, b, n=33):
    start = line.points(a)
    v = (b-a)*line.v
    t = np.linspace(0., 1., n)
    return Line(start, v, t, s.ask(start+t[:, None]*v)/s.scale)


def wave_basis(phase, drift):
    return np.column_stack((drift, np.cos(phase), np.sin(phase)))


def spectral_line(line, extra=None):
    """Extract real frequencies of a low-rank exponential sequence."""
    n = len(line.y)
    y0 = np.column_stack((line.y, extra)) if extra is not None else line.y[:, None]
    spread = max(np.max(np.ptp(y0, axis=0)), 1.e-9)
    t = line.t-.5
    for df in (1, 2, 3):
        y = np.diff(y0, n=df, axis=0)
        if np.max(np.abs(y), initial=0.) < 1.e-12:
            continue
        a = np.lib.stride_tricks.sliding_window_view(y, min(80, len(y)//2), axis=0)
        A = a.reshape(-1, a.shape[-1]).T
        u, z, _ = linalg.svd(A, full_matrices=False, check_finite=False)
        tried = set()
        ranks = []
        for eps in (1.e-12, 2.e-11, 7.e-10, 2.e-8, 2.e-7):
            r = int(np.count_nonzero(z > eps*z[0]))
            ranks.extend((r,) if r % 2 == 0 else (r-1, r+1))
        for rank in ranks:
            if rank in tried or not 2 <= rank <= (22 if extra is not None else 18) or rank >= len(z):
                continue
            tried.add(rank)
            U = u[:, :rank]
            roots = np.linalg.eigvals(least(U[:-1], U[1:]))
            ang = np.angle(roots)
            if np.max(np.abs(np.abs(roots)-1.)) > .02 or np.any(np.abs(ang) < .004):
                continue
            w = np.sort(ang[ang > 0.]*(n-1))
            if len(w)*2 != rank:
                continue
            drift = np.column_stack([t**k for k in range(df)])
            b = wave_basis(t[:, None]*w, drift)
            c = least(b, y0)
            if np.max(np.abs(b@c-y0)) > 7.e-8*spread:
                def res(w):
                    B = wave_basis(t[:, None]*w, drift)
                    return (B@least(B, y0)-y0).ravel()
                opt = optimize.least_squares(res, w, method="lm", max_nfev=40,
                                             ftol=1.e-12, xtol=1.e-12, gtol=1.e-12)
                if np.max(np.abs(opt.fun)) > 1.e-8*spread:
                    continue
                w = opt.x
            return w, df-1
    return None


def fit_wave(x, y, frequencies, order, optimize_freq=True, max_nfev=45, exact=False):
    t = x-.5
    drift = legendre_features(x, order)
    n, d = x.shape
    m = len(frequencies)
    p = drift.shape[1]
    b = wave_basis(t@frequencies.T, drift)
    c = least(b, y)

    def eval_(z, need_jac=False):
        w = z[:m*d].reshape(m, d)
        coef = z[m*d:]
        arg = t@w.T
        cs = np.cos(arg)
        sn = np.sin(arg)
        b = np.column_stack((drift, cs, sn))
        if need_jac:
            ph = sn*(-coef[p:p+m]) + cs*coef[p+m:p+2*m]
            return np.column_stack(((ph[:, :, None]*t[:, None, :]).reshape(n, m*d), b))
        return b@coef-y

    z0 = np.r_[frequencies.ravel(), c]
    tol = 3.e-15 if exact else 4.e-13
    if optimize_freq and np.max(np.abs(b@c-y)) > tol and n >= len(z0):
        if np.max(np.abs(c), initial=0.) > 1.e3:
            return None
        f = optimize.least_squares(eval_, z0, jac=lambda z: eval_(z, True), method="lm",
                                   max_nfev=max_nfev, ftol=3.e-12, xtol=5.e-13, gtol=3.e-13)
        z0 = f.x
    if not agrees(eval_(z0)+y, y, 2.e-9, 1.e-10):
        return None
    w = z0[:m*d].reshape(m, d)
    c = z0[m*d:]
    if np.max(np.abs(c)) > 1.e3:
        return None
    a = w.astype(np.longdouble)*.5
    moments = np.prod(np.divide(np.sin(a), a, out=np.ones_like(a), where=(a != 0.)), axis=1)
    integral = np.longdouble(c[0])+c[p:p+m].astype(np.longdouble)@moments
    return Model(lambda x: wave_basis((x-.5)@w.T, legendre_features(x, order))@c,
                 float(integral), "plane waves")


def try_waves(s, line, spectrum):
    w, order = spectrum
    t = line.t-.5
    m = len(w)
    ba = wave_basis(t[:, None]*w, np.column_stack([t**k for k in range(order+1)]))
    _, _, piv = linalg.qr(ba.T, pivoting=True, mode="economic", check_finite=False)
    i = np.unique(np.r_[piv[:ba.shape[1]], np.arange(0, len(t), 8)])
    b = ba[i]
    # The complex amplitudes in shifted traces measure the coordinate
    # frequencies by phase difference. A short shift unwraps the longer one.
    h = np.array([.0067, .072])
    if s.left < len(i)*2*s.d+96 or np.linalg.cond(b) > 4.e7:
        return None
    base = least(b, line.y[i])
    a = base[order+1:order+1+m]-1j*base[order+1+m:]
    if np.min(np.abs(a)) < 1.e-10*np.max(np.abs(a)):
        return None
    d = s.d
    p = line.points(line.t[i])
    out = p[None, None, :, :] + h[:, None, None, None]*np.eye(d)[None, :, None, :]
    if np.max(out) > 1.0 or np.min(out) < 0.:
        return None
    y = (s.ask(out.reshape(-1, d))/s.scale).reshape(-1, len(p)).T
    c = least(b, y)
    if np.max(np.abs(b @ c-y)) > 5.e-7*max(1., np.max(np.abs(y))):
        return None
    z = (c[order+1:order+1+m]-1j*c[order+1+m:])/a[:, None]
    if np.max(np.abs(np.abs(z)-1.)) > .015:
        return None
    z = np.angle(z).reshape(m, 2, d)
    w0 = z[:, 0, :]/h[0]
    step = np.rint((w0*h[1]-z[:, 1, :])/(2.*np.pi))
    w1 = (z[:, 1, :]+2.*np.pi*step)/h[1]
    alias = 2.*np.pi*(len(line.t)-1)
    check = (w1@line.v-w+.5*alias) % alias-.5*alias
    if np.max(np.abs(check)) > .25:
        w1 = unwrap_projection(w1, line.v, w, h)
    if w1 is None:
        return None
    x, y = s.data()
    return fit_wave(x, y, w1, order)


def try_shifted_waves(s, line):
    if s.left < 575 or s.time_left() < 4.:
        return None
    dif = np.diff(line.y)
    nz = dif[np.abs(dif) > 1.e-9*max(1.e-4, np.max(np.abs(line.y)))]
    if len(nz) < 10 or np.count_nonzero(nz[1:]*nz[:-1] < 0.) < 3:
        return None
    x = line.points(line.t)
    room = np.minimum(x.min(axis=0), 1.-x.max(axis=0))
    if np.min(room) < .025:
        return None
    h = (2.*s.design(3)-1.)*np.minimum(.091, .86*room)
    yy = (s.ask((x[None]+h[:, None]).reshape(-1, s.d))/s.scale).reshape(len(h), -1).T
    fit = spectral_line(line, yy)
    if fit is not None:
        return try_waves(s, line, fit)
    return None


def unwrap_projection(w, v, projected, h):
    # Use a projection constraint to resolve modest aliasing. The fit to
    # full-dimensional samples is still required afterward.
    k = w.copy()
    alias = 2.*np.pi/h[0]
    for i in range(len(k)):
        e = projected[i]-k[i]@v
        if abs(e) < .25:
            continue
        shift = v*alias
        j = np.argmin(np.minimum(np.abs(e-shift), np.abs(e+shift)))
        m = (h[0]*alias/h[1]) # spacing of the long-shift candidates
        restore = np.rint(alias/m)*m
        delta = math.copysign(restore, e*v[j])
        if abs(e-delta*v[j]) > 5.:
            return None
        k[i, j] += delta
    return k


def polyfit_local(t, y, l, r, order, origin, scale):
    z = (t[l:r]-origin)/scale
    return Polynomial(np.polynomial.polynomial.polyfit(z, y[l:r], order))


@dataclass
class Knot:
    a: float
    b: float
    t: float
    left: tuple
    right: tuple
    order: int
    change: Polynomial


def root_near(g, a, b, guess):
    if not len(g.coef) or np.max(np.abs(g.coef)) < 1.e-13:
        return None
    c = g.coef.copy()
    while len(c) > 1 and abs(c[-1]) < 3.e-9*max(np.max(np.abs(c)), 1.e-13):
        c = c[:-1]
    g = Polynomial(c)
    if len(c) == 1:
        return None
    r = g.roots()
    near = [z.real for z in r if abs(z.imag) < .007 and a <= z.real <= b]
    if len(near) >= 2 and max(near)-min(near) < .006:
        ans = float(np.mean(near))
    elif near:
        ans = min(near, key=lambda z: abs(z-guess))
    else:
        return None
    # For repeated roots, the derivative root is much more accurate
    # than an individual root of the jump polynomial.
    cand = g.deriv().roots() if len(c) > 2 else []
    for z in cand:
        if abs(z.imag) < 1.e-8 and abs(z.real-ans) < .015:
            if abs(g(z.real)) < 5.e-9*np.max(np.abs(c)):
                return float(z.real)
    return float(ans)


def line_knots(line, y):
    t = line.t
    size = len(t)
    base_noise = 5.e-12*max(np.max(np.abs(y)), np.max(np.abs(line.y)), 1.e-3)
    unresolved = 0
    for p in (1, 2, 3):
        jump = np.diff(y, n=p+1)
        a = np.flatnonzero(np.abs(jump) > base_noise*(1.8**p))
        if not len(a):
            return p, []
        if len(a) > min(len(jump)*.5, 22):
            continue
        if not unresolved:
            unresolved = p
        groups = np.split(a, np.flatnonzero(np.diff(a) > p+1)+1)
        fit = []
        for k, g in enumerate(groups):
            ll = int(0 if not k else groups[k-1][-1]+p+1)
            lr = int(g[0]+p+1)
            rl = int(g[-1]+1)
            rr = int(size if k+1 == len(groups) else groups[k+1][0]+1)
            # Keep only the uncontaminated points on either side.
            lr = min(lr, rl)
            rl = max(rl, lr)
            ll, rr = max(ll, lr-7), min(rr, rl+7)
            if lr-ll < p+1 or rr-rl < p+1:
                continue
            left = t[lr-1]
            right = t[rl]
            tm = (left+right)*.5
            unit = max(.04, right-left)
            gl = polyfit_local(t, y, ll, lr, p, tm, unit)
            gr = polyfit_local(t, y, rl, rr, p, tm, unit)
            tol = base_noise*8.
            if (np.max(np.abs(gl((t[ll:lr]-tm)/unit)-y[ll:lr])) > tol or
                    np.max(np.abs(gr((t[rl:rr]-tm)/unit)-y[rl:rr])) > tol):
                continue
            g = gr-gl
            dt = t[1]-t[0]
            knot = root_near(g, (left-tm-.004*dt)/unit, (right-tm+.004*dt)/unit, 0.)
            if knot is None:
                continue
            fit.append(Knot(tm, unit, tm+unit*knot, (t[ll], t[lr-1]), (t[rl], t[rr-1]), p, g))
        if fit:
            return p, fit
    return unresolved, []


@dataclass
class Plane:
    n: np.ndarray
    bias: float
    order: int = 1
    coeffs: object = None


def make_plane(n, line, item):
    bias = float((line.points(item.t)-.5)@n)
    v = float(line.v@n)
    if abs(v) < 2.e-10:
        return None
    # Translate the jump polynomial to the crossing and convert from t
    # to the signed distance coordinate of the plane.
    center = (item.t-item.a)/item.b
    c = item.change(Polynomial([center, 1./(item.b*v)])).coef
    c = np.pad(c, (0, item.order+1-len(c)))[1:]*np.sign(v)
    return Plane(n, bias, item.order, c)


def linear_plane(s, line, item, prev):
    # One-sided gradients are well-conditioned even when the shifted
    # crossing is outside the range spanned by the fitting points.
    a = .47*item.left[1]+.53*item.left[0]
    b = .47*item.right[0]+.53*item.right[1]
    a = max(a, (item.t+item.left[1])*.5-.035)
    b = min(b, (item.t+item.right[0])*.5+.035)
    t = np.array([a, b])
    x = line.points(t)
    d = s.d
    h0 = min(.0032, .14*min(item.t-a, b-item.t))
    expect = item.change.coef[-1]/item.b
    if abs(expect) < 7.e-10 or h0 <= 0.:
        return None
    hsign = np.where(x < .5, 1., -1.)
    for h in (h0, .19*h0, .037*h0):
        if s.left < 4*d+80:
            return None
        step = (hsign*h)[:, :, None]*np.eye(d)
        pts = np.column_stack((x[:, None, :], x[:, None, :]+step,
                               x[:, None, :]+.41*step)).reshape(-1, d)
        y = s.ask(pts)/s.scale
        if prev is not None:
            y -= prev.predict(pts)
        z = y.reshape(2, 1+2*d)
        g0 = (z[:, 1:d+1]-z[:, :1])/(h*hsign)
        g = (z[:, d+1:]-z[:, :1])/(.41*h*hsign)
        jump = g[1]-g[0]
        scale = max(np.max(np.abs(g)), abs(expect), 1.e-5)
        if (np.max(np.abs(g-g0)) > 4.e-7*scale or
                abs(jump@line.v-expect) > 2.e-7*scale or
                np.max(np.abs(jump)) <= 9.e-10):
            continue
        return make_plane(jump/jump[np.argmax(np.abs(jump))], line, item)
    return None


def plane_of_knot(s, line, item, prev=None):
    p = item.order
    if p == 1:
        plane = linear_plane(s, line, item, prev)
        if plane is not None:
            return plane
    l0, l1 = item.left
    r0, r1 = item.right
    if min(l1-l0, r1-r0) < 1.e-8:
        return None
    l1 = min(l1, .63*l1+.37*l0)
    r0 = max(r0, .63*r0+.37*r1)
    lt = np.linspace(l0, l1, p+1)
    rt = np.linspace(r0, r1, p+1)
    t = np.r_[lt, rt]
    x = line.points(t)
    d = s.d
    g = np.zeros(d)
    chosen = np.arange(d)
    h0 = min(.003, .19*min(item.t-l1, r0-item.t))
    if h0 < 2.e-5:
        return None
    room0 = np.min(x, axis=0)
    room1 = 1.-np.max(x, axis=0)
    safe = np.where(room1 >= room0, np.minimum(h0, .82*room1), -np.minimum(h0, .82*room0))
    if np.min(np.abs(safe)) < 1.e-7:
        return None
    for factor in (1., .16, .024):
        if s.left < len(chosen)*len(t)+74:
            return None
        h = safe*factor
        mat = (x[None]+(np.eye(d)*h[:, None])[chosen, None, :])
        sensor = s.ask(mat.reshape(-1, d))/s.scale
        if prev is not None:
            sensor -= prev.predict(mat.reshape(-1, d))
        sensor = sensor.reshape(-1, len(t))
        bad = []
        z = (t-item.a)/item.b
        for j, values in zip(chosen, sensor):
            a = polyfit_local(z, values, 0, p+1, p, 0., 1.)
            b = polyfit_local(z, values, p+1, 2*(p+1), p, 0., 1.)
            k = (item.t-item.a)/item.b
            t0 = root_near(b-a, (l1-item.a)/item.b, (r0-item.a)/item.b, k)
            if t0 is None or abs((b-a).coef[-1]-item.change.coef[-1]) > max(
                    1.e-10, .0002*abs(item.change.coef[-1])):
                bad.append(j)
                continue
            g[j] = -((t0-k)*item.b)/h[j]
        if not bad:
            if abs(g@line.v-1.) < .02:
                break
            chosen = np.arange(d)
        else:
            chosen = np.array(bad)
    else:
        return None
    j = np.argmax(np.abs(g))
    if abs(g[j]) < 1.e-12:
        return None
    return make_plane(g/g[j], line, item)


def hinge_mean(n, bias, degree):
    # Repeated integration of a truncated power. Do the alternating
    # differences at extended precision, dropping numerically zero axes.
    a = n[np.abs(n) > 2.e-14]
    with localcontext() as ctx:
        ctx.prec = max(65, 26+15*len(a))
        aa = [Decimal.from_float(abs(float(t))) for t in a]
        b = -Decimal.from_float(float(bias))-sum(aa)/2
        denominator = math.prod(aa)*math.prod(range(degree+1, degree+len(a)+1))
        total = Decimal(0)
        for mask in itertools.product((0, 1), repeat=len(a)):
            r = b+sum(t for t, use in zip(aa, mask) if use)
            if r > 0:
                total += (1 if (len(a)-sum(mask)) % 2 == 0 else -1)*r**(degree+len(a))
        return np.longdouble(str(total/denominator))


def plane_close(a, b, tol=3.e-6):
    return min(max(np.max(np.abs(a.n-b.n)), abs(a.bias-b.bias)),
               max(np.max(np.abs(a.n+b.n)), abs(a.bias+b.bias))) < tol


def fit_hinges(x, y, planes, order, nonlinear=True):
    d = x.shape[1]
    t = x-.5
    poly = legendre_features(x, order)
    terms = []
    for pl in planes:
        terms.extend((pl.n, pl.bias, k) for k in range(1, pl.order+1))
    n, b, p = tuple(np.array(a) for a in zip(*terms))
    base = np.maximum(t@n.T-b, 0.)
    f = np.column_stack((poly, base**p))
    c = least(f, y)
    err = f@c-y
    good = agrees(err+y, y, 4.e-12, 8.e-13)
    err_max = np.max(np.abs(err))
    if (nonlinear and err_max > 5.e-15*max(np.max(np.abs(y)), 1.e-4)
            and err_max < 5.e-3*max(np.ptp(y), 1.e-8)):
        m = len(planes)
        k0 = poly.shape[1]
        jj = np.argmax(np.abs([u.n for u in planes]), axis=1)
        selectors = np.arange(d)[None, :] != jj[:, None]
        normals = np.array([u.n for u in planes])
        owners = np.repeat(np.arange(m), [u.order for u in planes])
        z = np.r_[normals[selectors], [u.bias for u in planes], c]
        ss = m*(d-1)
        def func(z, jac=False, ret=False):
            q = normals.copy()
            q[selectors] = z[:ss]
            bias = z[ss:ss+m]
            c = z[ss+m:]
            aa = np.maximum(t@q.T-bias, 0.)
            Phi = np.column_stack((poly, aa[:, owners]**p))
            if ret:
                return q[owners], bias[owners], c
            if not jac:
                return Phi@c-y
            rates = np.zeros_like(aa)
            for i, (j, k) in enumerate(zip(owners, p)):
                rates[:, j] += (k*c[k0+i])*(aa[:, j] > 0.)*aa[:, j]**(k-1)
            df = (rates[:, :, None]*t[:, None, :])[:, selectors]
            return np.column_stack((df, -rates, Phi))
        if len(z) <= len(x) and len(z) < 250:
            opt = optimize.least_squares(func, z, jac=lambda z: func(z, True), method="lm",
                                         max_nfev=28, ftol=2.e-14, xtol=2.e-14, gtol=4.e-15)
            if np.max(np.abs(opt.fun)) <= err_max*1.2:
                n, b, c = func(opt.x, ret=True)
                good = agrees(opt.fun+y, y, 4.e-12, 9.e-13)
    def pred(z):
        return np.column_stack((legendre_features(z, order),
                                np.maximum((z-.5)@n.T-b, 0.)**p))@c
    def integral():
        bits = [np.longdouble(c[0])]
        bits.extend(np.longdouble(k)*hinge_mean(a, b0, int(p0))
                    for k, a, b0, p0 in zip(c[-len(p):], n, b, p))
        return np.sum(bits, dtype=np.longdouble)
    model = Model(pred, float(integral()) if good else float("nan"), "hinge ridges")
    model.integral = integral
    return model, good


def vertex_points(s, limit=256):
    if (1 << s.d) <= limit:
        return np.array(list(itertools.product((2.e-7, 1.-2.e-7), repeat=s.d)))
    return np.where(s.design(limit) > .5, 1.-2.e-7, 2.e-7)


def interface_checks(s, planes, limit):
    """Check several distinct intersections of each recovered interface."""
    if not planes or limit < 4:
        return np.empty((0, s.d))
    all_points = []
    for plane in planes:
        n = plane.n if plane.bias >= 0. else -plane.n
        a = np.abs(n)
        dist = .5*np.sum(a)-abs(plane.bias)
        if dist < 2.e-10:
            continue
        corner = (n >= 0.).astype(float)
        inward = 1.-2.*corner
        # Try different edges incident to the corner on the smaller side.
        for j in np.flatnonzero(a > 1.14*dist):
            hit = dist/a[j]
            q = np.tile(corner, (2, 1))
            q[:, j] += inward[j]*hit*np.array([.63, 1.17])
            all_points.extend(q)
        # Also displace along interior rays so broad planes get checked.
        rays = .25+.75*s.design(6)
        for ray in rays:
            dt = dist/(ray@a)
            if 0. < dt < 1.18:
                all_points.append(corner+inward*ray*min(.98, .88*dt))
                all_points.append(corner+inward*ray*min(.999, 1.06*dt))
    if len(all_points) > limit:
        take = s.rng.choice(len(all_points), size=limit, replace=False)
        all_points = np.asarray(all_points)[take]
    return np.clip(np.asarray(all_points).reshape(-1, s.d), 2.e-7, 1.-2.e-7)


def local_hint(x, y, planes, order):
    t = x-.5
    poly = legendre_features(x, order)
    try:
        use = [(a.n, a.bias, a.coeffs) for a in planes if a.coeffs is not None]
        def hinges(x):
            t = x-.5
            z = np.zeros(len(x))
            for n, b, c in use:
                u = np.maximum(t@n-b, 0.)
                z += Polynomial(np.r_[0., c])(u)
            return z
        c = least(poly, y-hinges(x))
        return Model(lambda z: hinges(z)+legendre_features(z, order)@c, float("nan"), "partial local")
    except (ArithmeticError, ValueError):
        return None


def zoom_intervals(line, res, order, knots):
    p = max(1, order)
    q = np.abs(np.diff(res, n=p+1))
    noise = 3.e-11*max(.001, np.max(np.abs(line.y)))
    a = np.flatnonzero(q > noise)
    if not len(a) or len(a) >= .7*len(q):
        return []
    groups = np.split(a, np.flatnonzero(np.diff(a) > p+1)+1)
    t = line.t
    out = []
    for g in groups:
        j, k = g[0], g[-1]
        lo, hi = t[max(0, j-4)], t[min(len(t)-1, k+p+5)]
        found = sum(lo < a.t < hi for a in knots)
        if found and k-j+1 <= (p+1)*found:
            continue
        if hi-lo >= .65:
            continue
        out.append((float(np.sum(q[g])), float(lo), float(hi)))
    return sorted(out, reverse=True)


def plane_on_line(line, p, t):
    a = line.v@p.n
    if abs(a) < 1.e-9:
        return False
    root = (p.bias-(line.start-.5)@p.n)/a
    return abs(root-t) < 4.e-7


@dataclass
class Affine:
    x: np.ndarray
    y: float
    g: np.ndarray

    def predict(self, x):
        return (x-self.x)@self.g+self.y


def local_affines(s, centers, hint):
    d = s.d
    z = np.asarray(centers)
    result = [None]*len(z)
    todo = np.arange(len(z))
    for h in (.0021, .00023, .000025):
        if len(todo)*(1+2*d)+100 > s.left:
            break
        x = z[todo]
        step = (h*np.where(x < .5, 1., -1.))[:, :, None]*np.eye(d)
        pts = np.concatenate((x[:, None], x[:, None]+step,
                              x[:, None]+.37*step), axis=1).reshape(-1, d)
        u = s.ask(pts)/s.scale
        scale = np.max(np.abs(u), initial=1.e-4)
        if hint is not None:
            u -= hint.predict(pts)
        u = u.reshape(len(x), 1+2*d)
        dx = np.einsum("ijj->ij", step)
        g0 = (u[:, 1:d+1]-u[:, :1])/dx
        g = (u[:, d+1:]-u[:, :1])/(.37*dx)
        err = np.max(np.abs(g0-g), axis=1)
        ok = err < (2.e-8*scale+4.e-8*np.max(np.abs(g), axis=1))
        for j, point, vals, valid, grad in zip(todo, x, u, ok, g):
            if valid:
                result[j] = Affine(point, float(vals[0]), grad)
        todo = todo[~ok]
        if not len(todo):
            break
    return result


def affine_discovery(s, a, b, hint, depth=3):
    """Search between two local affine pieces for a single interface."""
    l, r = local_affines(s, [a, b], hint)
    if l is None or r is None:
        return []
    pairs = [(l, r, 0)]
    found = []
    while pairs and s.left > 142 and len(found) < 4:
        l, r, used = pairs.pop()
        delta = r.x-l.x
        diff = r.g-l.g
        if np.max(np.abs(diff)) < 3.e-9 or np.max(np.abs(delta)) < 3.e-7:
            continue
        a = l.y-float(r.predict(l.x))
        b = float(l.predict(r.x))-r.y
        t = a/(a-b) if abs(a-b) > 1.e-13 else -1.
        root = np.clip(t, .000001, .999999)
        if 0. < t < 1.:
            tt = np.array([.39*t, t, .67+.33*t])
            z = l.x+tt[:, None]*delta
            y = s.ask(z)/s.scale
            if hint is not None:
                y -= hint.predict(z)
            pred = np.r_[l.predict(z[:2]), r.predict(z[2:])]
            if agrees(pred, y, 2.e-10, 1.6e-11):
                j = np.argmax(np.abs(diff))
                n = diff/diff[j]
                bias = float((z[1]-.5)@n)
                coeff = diff[j]*np.sign(n@delta)
                found.append(Plane(n, bias, 1, np.array([coeff])))
                continue
        if used >= depth:
            continue
        t = root if .14 < root < .86 else .5
        # Avoid measuring the gradient exactly at an interface.
        t += .028*(.5 if used % 2 else -.5)
        m = local_affines(s, [l.x+t*delta], hint)[0]
        if m is None:
            continue
        intervals = [(l, m, used+1), (m, r, used+1)]
        intervals.sort(key=lambda row: np.linalg.norm(row[0].g-row[1].g))
        pairs.extend(intervals)
    return found


def try_hinges(s, line, detected):
    order, knots = detected
    if order == 0:
        return None
    planes = []
    prev = None
    last = None
    sign_check_done = False
    vertices_done = False
    hint = None
    attempted = []
    checked_planes = []
    zooms = 0
    affine_attempts = 0
    for iteration in range(17):
        if s.left < 132 or s.time_left() < 4.:
            break
        p, knots = detected
        if p:
            order = max(p, order)
        known = [a for a in knots if any(plane_on_line(line, u, a.t) for u in planes)]
        new = []
        for item in knots:
            if any(item is q for q in known):
                continue
            plane = plane_of_knot(s, line, item, prev=prev)
            if plane is None:
                continue
            if any(plane_close(plane, a) for a in planes):
                known.append(item)
                continue
            planes.append(plane)
            known.append(item)
            new.append(plane)
        if planes:
            x, y = s.data()
            model, good = fit_hinges(x, y, planes, order)
            if good:
                need_check = [pl for pl in planes if not any(plane_close(pl, cp) for cp in checked_planes)]
                if (not sign_check_done or need_check) and s.left > 62:
                    n = min(256, s.left//3)
                    v = vertex_points(s, n) if not vertices_done else np.empty((0, s.d))
                    z = np.vstack((s.design(48 if not sign_check_done else 18), v,
                                   interface_checks(s, need_check, min(160, (s.left-len(v))//3))))
                    q = s.ask(z)/s.scale
                    sign_check_done = True
                    vertices_done = vertices_done or len(v) > 0
                    checked_planes.extend(need_check)
                    if agrees(model.predict(z), q, 5.e-11, 2.e-12):
                        x, y = s.data()
                        polished, ok = fit_hinges(x, y, planes, order)
                        if ok:
                            return polished
                        return model
                else:
                    return model
            last = model
            hint = local_hint(x, y, planes, min(order, 3))
            if hint is not None and np.linalg.norm(y-hint.predict(x)) > 2.*np.linalg.norm(y-last.predict(x)):
                hint = last
        if (not planes and iteration >= 2) or len(planes) > 12:
            if p == 0 or zooms > 2 or len(planes) > 12:
                break
        # If the line's differences indicate unresolved nearby crossings,
        # narrow that part of the line before making new random probes.
        lv = line.y-(prev.predict(line.points(line.t)) if prev else 0.)
        regions = zoom_intervals(line, lv, p if p else order, known)
        if regions and s.left >= 238 and zooms < 6 and (not new or iteration % 2 == 1):
            _, lo, hi = regions[0]
            line = restrict_line(s, line, lo, hi)
            zooms += 1
            detected = line_knots(line, line.y-(prev.predict(line.points(line.t)) if prev else 0.))
            continue
        if (planes and not vertices_done and s.left > (1 << min(s.d, 8))+350
                and (not new or iteration >= 2)):
            z = vertex_points(s)
            s.ask(z)
            vertices_done = True
        x, y = s.data()
        if last is None:
            z = legendre_features(x, min(order, 2))
            res = y-z@least(z, y)
        else:
            surrogate = hint if hint is not None else last
            res = y-surrogate.predict(x)
        rank = np.argsort(-np.abs(res))
        top = rank[0]
        if attempted:
            risk = np.abs(res).copy()
            for a in attempted:
                risk *= 1.-.85*np.exp(-np.sum((x-a)**2, axis=1)/.024)
            top = int(np.argmax(risk))
        if iteration % 3 == 0 or abs(res[top]) > 1.e-5*np.ptp(y):
            target = s.rng.uniform(.15, .85, s.d)
            if iteration % 3 == 1:
                target = x[rank[-1]]
            if iteration % 3 == 2:
                cand = np.argsort(np.sum((x-x[top])**2, axis=1))
                short = cand[np.abs(res[cand]) < .02*abs(res[top])+1.e-12]
                if len(short):
                    target = x[short[0]]
            attempted.append(x[top].copy())
            if (order == 1 and not new and iteration >= 1 and affine_attempts < 4
                    and s.left > 270):
                affine_attempts += 1
                found = affine_discovery(s, x[top], target, hint, depth=2)
                extra = []
                for a in found:
                    if not any(plane_close(a, b) for b in planes):
                        planes.append(a)
                        extra.append(a)
                if extra:
                    detected = (1, [])
                    continue
            line = connecting_line(s, x[top], target, 49)
        else:
            line = central_line(s, 49, attempt=1)
        # Remove previously identified hinges using their local amplitudes;
        # if the subtraction is inaccurate, the unmodified line is useful.
        prev = hint
        yloc = line.y-(prev.predict(line.points(line.t)) if prev else 0.)
        detected = line_knots(line, yloc)
        if not detected[0] or not detected[1]:
            prev = None
            detected = line_knots(line, line.y)
    if last is not None and s.left >= 96 and s.time_left() > 2.:
        x, y = s.data()
        residual = last.predict(x)-y
        if (np.linalg.norm(residual) < .3*np.linalg.norm(y-y.mean()) and
                np.max(np.abs(residual)) < .35*max(1.e-8, np.ptp(y))):
            last.mean = float(last.integral())
            if np.isfinite(last.mean) and np.min(y)-1.e-8 <= last.mean <= np.max(y)+1.e-8:
                s.hint = last
    return None
