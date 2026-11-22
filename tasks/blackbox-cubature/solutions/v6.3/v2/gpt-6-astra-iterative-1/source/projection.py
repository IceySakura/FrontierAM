"""Projection-pursuit control variates with analytically integrated splines."""
import math
import time
import itertools
from decimal import Decimal, localcontext
import numpy as np
from scipy import linalg, optimize, special
from scipy.interpolate import BSpline
from numpy.polynomial.legendre import leggauss
from ridges import background, hessian_at, fourier_refine


def positive_moment(w, bias, power):
    weights = np.sort(np.abs(np.asarray(w, dtype=np.longdouble)))
    weights = weights[weights > np.max(weights, initial=0)*1e-8]
    if not len(weights):
        return float(max(-bias, 0)**power)
    t = np.sum(weights)/2-np.longdouble(bias)
    if t <= 0:
        return 0.0
    if -bias >= np.sum(weights)/2:
        c = -float(bias)
        variance = float(np.sum(weights*weights)/12)
        if power == 1:
            return c
        if power == 2:
            return c*c+variance
        if power == 3:
            return c*c*c+3*c*variance
    sums = np.zeros(1, dtype=np.longdouble)
    signs = np.ones(1, dtype=np.longdouble)
    for a in weights:
        sums = np.r_[sums, sums+a]
        signs = np.r_[signs, -signs]
    denominator = np.prod(weights)
    for k in range(1, len(weights)+1):
        denominator *= np.longdouble(power+k)
    terms = np.maximum(t-sums, 0)**np.longdouble(power+len(weights))
    numerator = np.sum(signs*terms)
    answer = numerator/denominator
    if np.sum(terms) > max(abs(numerator), np.longdouble("1e-4900"))*1e4:
        with localcontext() as context:
            context.prec = 80
            ww = [Decimal.from_float(float(v)) for v in weights]
            tt = sum(ww)/2-Decimal.from_float(float(bias))
            pp = Decimal.from_float(float(power))
            subsets = [(Decimal(0), 1)]
            for v in ww:
                subsets += [(s+v, -sign) for s, sign in subsets]
            total = sum((tt-s)**(pp+len(ww))*sign for s, sign in subsets if tt > s)
            divisor = Decimal(1)
            for v in ww:
                divisor *= v
            for k in range(1, len(ww)+1):
                divisor *= pp+k
            answer = float(total/divisor)
    return float(answer)


def directions_from_hessians(hessians):
    d = hessians[0].shape[0]
    _, s, vh = linalg.svd(np.concatenate(hessians, axis=0), full_matrices=False)
    if s[0] < 1e-8:
        return []
    rank = int(np.count_nonzero(s > s[0]*2e-6))
    if rank == 0:
        return []
    u = vh[:rank].T
    answers = []
    rng = np.random.default_rng(1931)
    for trial in range(4):
        ca, cb = rng.normal(size=(2, len(hessians)))
        a = sum(c*h for c, h in zip(ca, hessians))
        b = sum(c*h for c, h in zip(cb, hessians))
        try:
            eig, vv = linalg.eig(u.T@a@u, u.T@b@u)
            if np.max(np.abs(vv.imag)) > 1e-3 or not np.all(np.isfinite(eig)):
                continue
            if np.linalg.cond(vv) > 1e5:
                continue
            w = (u@linalg.inv(vv.real).T).T
            w /= np.linalg.norm(w, axis=1)[:, None]
            score = 0.0
            pinv = linalg.pinv(w.T)
            for h in hessians:
                projected = pinv@h@pinv.T
                diagonal = np.diag(np.diag(projected))
                score += np.linalg.norm(projected-diagonal)/max(np.linalg.norm(projected), 1e-20)
            answers.append((score, w))
        except (ValueError, linalg.LinAlgError):
            continue
    answers.sort(key=lambda item: item[0])
    return answers


def fit_projection(x, y, directions, kind, initial_power, deadline, gradient_data=None,
                   initial_biases=None):
    n, d = x.shape
    r = len(directions)
    nt = n*3//4
    z = x-0.5
    def make_background(zz):
        bg = background(zz)
        if kind == "relu" and initial_power == 3:
            cubic = np.column_stack([zz[:, i]*zz[:, j]*zz[:, k]
                                     for i, j, k in itertools.combinations_with_replacement(range(d), 3)])
            bg = np.column_stack((bg, cubic))
        return bg
    bg = make_background(z)
    nb = bg.shape[1]
    if nb+r >= nt:
        return None
    trainz, trainy, trainbg = z[:nt], y[:nt], bg[:nt]
    variable = kind == "absolute"
    biases = np.median(z@directions.T, axis=0)
    if initial_biases is not None:
        biases = np.asarray(initial_biases)
    if gradient_data is not None:
        gx, gradients = gradient_data
        latent = gradients@linalg.pinv(directions)
        for k, direction in enumerate(directions):
            t = (gx-0.5)@direction
            candidates = np.linspace(np.min(t), np.max(t), 65)
            arg = t[:, None]-candidates
            if variable:
                features = np.sign(arg)*np.abs(arg)**(initial_power-1)
            else:
                features = np.maximum(arg, 0)**(initial_power-1)*(arg > 0)
            features -= features.mean(axis=0)
            target = latent[:, k]-latent[:, k].mean()
            scores = (features.T@target)**2/np.maximum(np.sum(features*features, axis=0), 1e-20)
            biases[k] = candidates[np.argmax(scores)]
    p0 = np.r_[np.column_stack((directions, biases)).ravel(),
               np.log((initial_power-0.3)/(4.0-initial_power)) if variable else []]
    cache = [None]

    def components(params, zz, bb):
        pp = params[:r*(d+1)].reshape(r, d+1)
        power = 0.3+3.7*special.expit(params[-1]) if variable else initial_power
        arg = zz@pp[:, :d].T-pp[:, d]
        if variable:
            base = np.maximum(np.abs(arg), 1e-100)
            features = base**power
            derivative = power*base**(power-1)*np.sign(arg)
        else:
            base = np.maximum(arg, 0)
            features = base**power
            derivative = power*base**(power-1)*(arg > 0)
        return np.column_stack((bb, features)), derivative, base, power

    def calc(params):
        if len(cache) == 1 or np.any(cache[0] != params):
            a, der, base, power = components(params, trainz, trainbg)
            coeff = linalg.lstsq(a, trainy, cond=1e-11, lapack_driver="gelsy")[0]
            cache[:] = [params.copy(), a, coeff, der, base, power]
        return cache[1:]

    def fun(params):
        if time.monotonic() > deadline:
            raise TimeoutError("projection fitting deadline")
        a, coeff, _, _, _ = calc(params)
        return a@coeff-trainy

    def jac(params):
        a, coeff, der, base, power = calc(params)
        q = linalg.qr(a, mode="economic", check_finite=False)[0]
        out = np.empty((nt, len(params)))
        zz = np.column_stack((trainz, -np.ones(nt)))
        for k in range(r):
            out[:, k*(d+1):(k+1)*(d+1)] = zz*(der[:, k]*coeff[nb+k])[:, None]
        if variable:
            ds = special.expit(params[-1])
            out[:, -1] = (base**power*np.log(base))@coeff[nb:]*3.7*ds*(1-ds)
        out -= q@(q.T@out)
        return out

    try:
        fit = optimize.least_squares(fun, p0, jac=jac, method="lm",
                                    max_nfev=180, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        aa, _, _, power = components(fit.x, z, bg)
        coeff = calc(fit.x)[1]
        errors = aa@coeff-y
        validation = float(np.sqrt(np.mean(errors[nt:]**2)))
        if validation > np.std(y)*0.08:
            return None
        # Retain hold-out validation but use all the samples for the final coefficients.
        coeff = linalg.lstsq(aa, y, cond=1e-12, lapack_driver="gelsy")[0]
        params = fit.x.copy()
        pp = params[:r*(d+1)].reshape(r, d+1)
        answer = float(coeff[0])
        for c, row in zip(coeff[nb:], pp):
            moment = positive_moment(row[:d], row[d], power)
            if variable:
                moment += positive_moment(row[:d], -row[d], power)
            answer += c*moment
        if not np.isfinite(answer):
            return None
        def predict(xx):
            zz = xx-0.5
            return components(params, zz, make_background(zz))[0]@coeff
        return validation, answer, predict, float(np.max(np.abs(aa@coeff-y)))
    except (ValueError, linalg.LinAlgError, FloatingPointError, TimeoutError):
        return None


def fourier_directions(x, y, directions, deadline):
    z = x-0.5
    design = background(z)
    omega = np.arange(0.5, 128.01, 0.5)
    selected, modes = [], []
    unused = list(range(len(directions)))
    for iteration in range(len(directions)):
        q = linalg.qr(design, mode="economic", check_finite=False)[0]
        residual = y-q@(q.T@y)
        best = None
        for j in unused:
            t = z@directions[j]
            co = np.cos(t[:, None]*omega)
            si = np.sin(t[:, None]*omega)
            co -= q@(q.T@co)
            si -= q@(q.T@si)
            cc, ss, cs = np.sum(co*co, axis=0), np.sum(si*si, axis=0), np.sum(co*si, axis=0)
            cy, sy = co.T@residual, si.T@residual
            det = cc*ss-cs*cs
            score = np.where(det > 1e-12, (ss*cy*cy+cc*sy*sy-2*cs*cy*sy)/np.maximum(det, 1e-30), 0)
            k = int(np.argmax(score))
            if best is None or score[k] > best[0]:
                best = (score[k], j, omega[k])
        if best is None:
            return None
        _, j, frequency = best
        w = directions[j]*frequency
        selected.append(w)
        modes.append(j)
        unused.remove(j)
        design = np.column_stack((design, np.cos(z@w), np.sin(z@w)))
    coeff = linalg.lstsq(design, y, lapack_driver="gelsy")[0]
    nb = background(z).shape[1]
    amplitudes = coeff[nb::2]-1j*coeff[nb+1::2]
    return fourier_refine(x, y, np.asarray(selected), amplitudes, deadline)


def overcomplete(x, y, hessians, deadline):
    d = x.shape[1]
    if d > 3:
        return None
    eigenvalues = [linalg.eigvalsh(h) for h in hessians]
    magnitude = max(np.max(np.abs(v)) for v in eigenvalues)
    if not (all(v[0] > -magnitude*1e-5 for v in eigenvalues) or
            all(v[-1] < magnitude*1e-5 for v in eigenvalues)):
        return None
    covariance = sum(h@h for h in hessians)
    eig, vec = linalg.eigh(covariance)
    transform = (vec*np.maximum(eig, np.max(eig)*0.005)**0.25)@vec.T
    rng = np.random.default_rng(6947)
    favorable = np.argsort(y)[:max(8, len(y)//8)]
    for r in (d+1, d+2):
        for attempt in range(8):
            if time.monotonic() > deadline-0.3:
                return None
            directions = rng.normal(size=(r, d))@transform
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            centers = x[rng.choice(favorable, size=r)]
            biases = np.sum((centers-0.5)*directions, axis=1)
            power = 1.5 if attempt%3 else 3.0
            model = fit_projection(x, y, directions, "absolute", power, deadline,
                                   initial_biases=biases)
            if model is not None and model[3] < 3e-12:
                return model
    return None


def fit(oracle, x, y, deadline, gradient_data=None):
    d = oracle.d
    cost = 2*d*d+1
    if oracle.remaining < 3*cost+320:
        return None
    scale = max(np.max(np.abs(y)), 1e-300)
    hessians = []
    for i in (5, 11, 19):
        point = np.clip(x[i], 0.02, 0.98)
        hessians.append(hessian_at(oracle, point, scale))
    guesses = directions_from_hessians(hessians)
    if not guesses or guesses[0][0] > 0.25:
        model = overcomplete(x, y/scale, hessians, deadline)
        if model is None:
            return None
        error, answer, prediction, maximum = model
        return error*scale, answer*scale, lambda p: prediction(p)*scale, maximum*scale
    if guesses[0][0] < 0.03:
        oracle.projections = guesses[0][1]
    best = None
    if gradient_data is not None:
        gx, gradients = gradient_data
        gradient_data = gx, gradients/scale
    for _, directions in guesses[:1]:
        periodic = fourier_directions(x, y/scale, directions, deadline)
        if periodic is not None:
            return 0.0, periodic*scale, None, 0.0
        for kind, power in (("relu", 1.0), ("relu", 2.0),
                            ("absolute", 1.5), ("relu", 3.0),
                            ("absolute", 3.0)):
            if time.monotonic() > deadline-1:
                break
            model = fit_projection(x, y/scale, directions, kind, power, deadline, gradient_data)
            if model is not None and (best is None or model[0] < best[0]):
                best = model
                if model[3] < 2e-10:
                    break
    if best is None or best[3] > 3e-12:
        alternative = overcomplete(x, y/scale, hessians, deadline)
        if alternative is not None:
            best = alternative
    if best is None:
        return None
    error, answer, prediction, maximum = best
    return (error*scale, answer*scale, lambda p: prediction(p)*scale, maximum*scale)


def power_lines(oracle, x, y, deadline):
    d = oracle.d
    scale = max(np.max(np.abs(y)), 1e-300)
    rng = np.random.default_rng(oracle.seed ^ 0x3712DE4F)
    directions, biases = [], []
    line_xs, line_ys = [], []
    degree = None
    for attempt in range(3):
        nline = 65 if attempt == 0 else 129
        if oracle.remaining < nline+8*d or time.monotonic() > deadline-2:
            return None
        v = np.full(d, 0.998) if attempt == 0 else rng.uniform(0.65, 0.998, d)*rng.choice([-1, 1], d)
        ts = np.linspace(0, 1, nline)
        points = 0.5+(ts[:, None]-0.5)*v
        values = oracle.ask(points)/scale
        line_xs.append(points)
        line_ys.append(values)
        nonzero2 = np.count_nonzero(np.abs(np.diff(values, n=2)) > 5e-12)
        if nonzero2 < (nline-1)/4 and degree is None:
            return None
        differences = None
        for power in (2, 3):
            diff = np.diff(values, n=power+1)
            ids = np.flatnonzero(np.abs(diff) > 5e-12)
            if 0 < len(ids) < len(diff)*0.45:
                degree = power
                differences = ids
                break
        if differences is None:
            if attempt == 0:
                return None
            continue
        if attempt == 0:
            if oracle.remaining < 192+8*d:
                return None
            nline = 257
            ts = np.linspace(0, 1, nline)
            full_points = 0.5+(ts[:, None]-0.5)*v
            new = np.arange(nline)%4 != 0
            full_values = np.empty(nline)
            full_values[::4] = values
            full_values[new] = oracle.ask(full_points[new])/scale
            values, points = full_values, full_points
            line_xs[-1], line_ys[-1] = points, values
            differences = np.flatnonzero(np.abs(np.diff(values, n=degree+1)) > 5e-12)
            if not len(differences):
                return None
        groups = []
        for _, group in itertools.groupby(enumerate(differences), lambda kv: kv[1]-kv[0]):
            ids = [int(v) for _, v in group]
            a = max(ids[0]+1, 1)
            b = min(ids[-1]+degree, nline-2)
            if groups and a <= groups[-1][1]:
                groups[-1] = (groups[-1][0], b)
            else:
                groups.append((a, b))
        for a, b in groups:
            eps = 1/(3*(nline-1))
            needed = 8*d if degree == 2 else 12*d
            if oracle.remaining < needed:
                break
            centers = np.array([ts[a], ts[b]])
            stencil = np.array([-eps, eps]) if degree == 2 else np.array([-eps, 0, eps])
            locations = (centers[:, None]+stencil).ravel()
            pp = 0.5+(locations[:, None]-0.5)*v
            from ridges import gradient_at
            gradients = gradient_at(oracle, pp, scale).reshape(2, len(stencil), d)
            if degree == 2:
                g0 = np.mean(gradients, axis=1)
                g1 = (gradients[:, 1]-gradients[:, 0])/(2*eps)
                c1 = g1
                c0 = g0-centers[:, None]*g1
                lead = c1[1]-c1[0]
                lower = c0[1]-c0[0]
                norm2 = np.dot(lead, lead)
                if norm2 < 1e-14:
                    continue
                knot = -np.dot(lower, lead)/norm2
                mismatch = np.linalg.norm(lower+knot*lead)/math.sqrt(norm2)
            else:
                g0 = gradients[:, 1]
                g1 = (gradients[:, 2]-gradients[:, 0])/(2*eps)
                c2 = (gradients[:, 2]+gradients[:, 0]-2*g0)/(2*eps*eps)
                c1 = g1-2*centers[:, None]*c2
                c0 = g0-centers[:, None]*g1+centers[:, None]**2*c2
                lead = c2[1]-c2[0]
                lower = c1[1]-c1[0]
                norm2 = np.dot(lead, lead)
                if norm2 < 1e-12:
                    continue
                knot = -np.dot(lower, lead)/(2*norm2)
                mismatch = (np.linalg.norm(lower+2*knot*lead)+
                            np.linalg.norm(c0[1]-c0[0]-knot*knot*lead))/math.sqrt(norm2)
            if not ts[a]-0.005 <= knot <= ts[b]+0.005 or mismatch > 0.002:
                continue
            direction = lead/math.sqrt(norm2)
            if direction@v < 0:
                direction = -direction
            bias = (knot-0.5)*(direction@v)
            if any(abs(direction@other) > 1-1e-6 and abs(bias-old) < 1e-3
                   for other, old in zip(directions, biases)):
                continue
            directions.append(direction)
            biases.append(bias)
        if directions:
            nt = len(x)*3//4
            xx = np.vstack((x[:nt], *line_xs, x[nt:]))
            yy = np.r_[y[:nt]/scale, *line_ys, y[nt:]/scale]
            model = fit_projection(xx, yy, np.asarray(directions), "relu", float(degree),
                                   deadline, initial_biases=biases)
            if model is not None and model[0] < np.std(yy)*1e-6:
                error, answer, predict, maximum = model
                return error*scale, answer*scale, lambda p: predict(p)*scale, maximum*scale
    return None


def spline_cubature(x, y, directions, density, deadline):
    n, d = x.shape
    weights = directions/np.sum(np.abs(directions), axis=1)[:, None]
    projected = 0.5+(x-0.5)@weights.T
    scale = max(np.max(np.abs(y)), 1e-300)
    yy = y/scale
    nt = n*3//4
    polynomial = background(x-0.5)
    npoly = polynomial.shape[1]
    best = None
    glx, glw = leggauss(7)
    for nk in (12, 24, 48, 80, 112):
        if time.monotonic() > deadline-2:
            break
        if (nk+4)*len(weights)+npoly > nt*0.65:
            break
        matrices, means = [], []
        for j, w in enumerate(weights):
            inner = np.quantile(projected[:nt, j], np.linspace(0.005, 0.995, nk))
            inner = np.unique(inner)
            knots = np.r_[np.zeros(4), inner, np.ones(4)]
            matrix = BSpline.design_matrix(np.clip(projected[:, j], 0, 1), knots, 3).toarray()
            breaks = np.zeros(1)
            for v in np.abs(w):
                breaks = np.r_[breaks, breaks+v]
            breaks = np.unique(np.r_[0, np.clip(breaks, 0, 1), inner, 1])
            left, right = breaks[:-1], breaks[1:]
            qt = ((left+right)[:, None]/2+(right-left)[:, None]/2*glx).ravel()
            qw = ((right-left)[:, None]/2*glw).ravel()*density(qt, w)
            mean = BSpline.design_matrix(qt, knots, 3).T@qw
            matrices.append(matrix-mean)
            means.append(mean)
        a = np.column_stack([polynomial]+matrices)
        gram = a[:nt].T@a[:nt]
        rhs = a[:nt].T@yy[:nt]
        for penalty in (1e-8, 1e-5, 1e-3):
            gg = gram.copy()
            regularizer = np.full(len(gg), penalty)
            regularizer[:npoly] = 1e-12
            gg.flat[::len(gg)+1] += regularizer
            try:
                factor = linalg.cho_factor(gg, lower=True, check_finite=False)
                coeff = linalg.cho_solve(factor, rhs, check_finite=False)
                err = float(np.sqrt(np.mean((a[nt:]@coeff-yy[nt:])**2)))
                if best is None or err < best[0]:
                    best = (err, a, regularizer)
            except (ValueError, linalg.LinAlgError):
                continue
    if best is None or best[0] > np.std(yy)*0.002:
        return None
    error, a, regularizer = best
    gg = a.T@a
    gg.flat[::len(gg)+1] += regularizer
    try:
        coeff = linalg.cho_solve(linalg.cho_factor(gg, lower=True, check_finite=False),
                                a.T@yy, check_finite=False)
        answer = float(coeff[0]*scale)
        return answer if np.isfinite(answer) else None
    except (ValueError, linalg.LinAlgError):
        return None
