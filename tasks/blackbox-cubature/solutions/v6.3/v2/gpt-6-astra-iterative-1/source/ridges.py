"""Low-rank periodic and piecewise-affine ridge approximation."""
import itertools
import math
import time
import numpy as np
from scipy import linalg, optimize
from sampling import mean as stable_mean, standard_deviation as stable_std
from sampling import query_uniform


def background(z):
    d = z.shape[1]
    columns = [np.ones(len(z))]
    columns.extend(z[:, j] for j in range(d))
    columns.extend(z[:, j]**2-1/12 for j in range(d))
    columns.extend(z[:, i]*z[:, j] for i in range(d) for j in range(i))
    return np.column_stack(columns)


def fourier_refine(x, y, frequencies, amplitudes, deadline):
    n, d = x.shape
    offset = stable_mean(y)
    variation = max(stable_std(y), np.max(np.abs(y))*1e-14, 1e-300)
    tolerance = max(3e-13, 3e-14*np.max(np.abs(y))/variation)
    y = y/variation-offset/variation
    amplitudes = amplitudes/variation
    r = len(frequencies)
    z = x-0.5
    bg = background(z)
    nb = bg.shape[1]
    nt = max(n*3//4, 2*(nb+r*(d+2)))
    nt = min(nt, n-32)
    if nt < nb+r*(d+2):
        return None
    p0 = np.r_[np.mean(y), np.zeros(nb-1),
               np.column_stack((amplitudes.real, -amplitudes.imag, frequencies)).ravel()]

    def residual(p, zz, yy):
        if time.monotonic() > deadline:
            raise TimeoutError("periodic fitting deadline")
        pp = p[nb:].reshape(r, d+2)
        arg = zz @ pp[:, 2:].T
        return background(zz)@p[:nb]+np.cos(arg)@pp[:, 0]+np.sin(arg)@pp[:, 1]-yy

    def jacobian(p, zz, yy):
        pp = p[nb:].reshape(r, d+2)
        arg = zz @ pp[:, 2:].T
        co, si = np.cos(arg), np.sin(arg)
        dd = -si*pp[:, 0]+co*pp[:, 1]
        out = np.empty((len(zz), len(p)))
        out[:, :nb] = background(zz)
        for k in range(r):
            out[:, nb+k*(d+2):nb+(k+1)*(d+2)] = np.column_stack((co[:, k], si[:, k], zz*dd[:, k, None]))
        return out
    try:
        fit = optimize.least_squares(residual, p0, jac=jacobian, args=(z[:nt], y[:nt]),
                                    method="lm", max_nfev=150, ftol=1e-14,
                                    xtol=1e-14, gtol=1e-14)
        err = residual(fit.x, z, y)
        if np.max(np.abs(err)) > tolerance:
            return None
        fit = optimize.least_squares(residual, fit.x, jac=jacobian, args=(z, y),
                                    method="lm", max_nfev=40, ftol=1e-14,
                                    xtol=1e-14, gtol=1e-14)
        pp = fit.x[nb:].reshape(r, d+2)
        integral = fit.x[0] + np.sum(pp[:, 0]*np.prod(np.sinc(pp[:, 2:]/(2*np.pi)), axis=1))
        return float(offset+variation*integral)
    except (ValueError, linalg.LinAlgError, FloatingPointError, TimeoutError):
        return None


def line_spectrum(values, t, threshold):
    if not np.all(np.isfinite(values)):
        return [], 0
    n, channels = values.shape
    lag = min(n//2, 64)
    h0 = np.concatenate([linalg.hankel(values[:n-lag, j], values[n-lag-1:n-1, j])
                         for j in range(channels)], axis=0)
    h1 = np.concatenate([linalg.hankel(values[1:n-lag+1, j], values[n-lag:n, j])
                         for j in range(channels)], axis=0)
    u, s, vh = linalg.svd(h0, full_matrices=False, check_finite=False)
    rank = min(int(np.count_nonzero(s > s[0]*threshold)), 60, h0.shape[1])
    if rank < 2:
        return [], rank
    compressed = (u[:, :rank].T @ h1 @ vh[:rank].T) / s[:rank, None]
    roots = linalg.eigvals(compressed, check_finite=False)
    omega = []
    for root in roots:
        if root.imag > 1e-6 and abs(abs(root)-1) < 0.08:
            freq = np.angle(root)/(t[1]-t[0])
            if freq > 0.001 and all(abs(freq-f) > 0.0001 for f in omega):
                omega.append(freq)
    return sorted(omega), rank


def refine_line_spectrum(omega, t, values):
    tc = t-0.5
    cache = [None, None, None]
    def calc(om):
        if cache[0] is None or np.any(cache[0] != om):
            arg = tc[:, None]*om
            a = np.column_stack((np.ones(len(t)), tc, tc*tc, np.cos(arg), np.sin(arg)))
            coeff = linalg.lstsq(a, values, cond=1e-13, lapack_driver="gelsy")[0]
            cache[:] = [om.copy(), a, coeff]
        return cache[1], cache[2]
    def fun(om):
        a, coeff = calc(om)
        return (a@coeff-values).ravel()
    def jac(om):
        a, coeff = calc(om)
        q = linalg.qr(a, mode="economic", check_finite=False)[0]
        out = []
        r = len(om)
        for k in range(r):
            der = tc[:, None]*(-np.sin(tc*om[k])[:, None]*coeff[3+k]+
                              np.cos(tc*om[k])[:, None]*coeff[3+r+k])
            der -= q@(q.T@der)
            out.append(der.ravel())
        return np.column_stack(out)
    fit = optimize.least_squares(fun, np.asarray(omega), jac=jac, method="lm",
                                max_nfev=45, ftol=1e-13, xtol=1e-13, gtol=1e-13)
    a, coeff = calc(fit.x)
    return fit.x, coeff, np.max(np.abs(a@coeff-values))


def fit_fourier(oracle, x, y, deadline, extra_points=None):
    d = oracle.d
    offset = stable_mean(y)
    scale = max(stable_std(y), np.max(np.abs(y))*1e-14, 1e-300)
    yy = y/scale-offset/scale
    rng = np.random.default_rng(oracle.seed ^ 0x61A7B341)
    delta = min(0.04, 1.6/max(getattr(oracle, "characteristic_frequency", 0), 1))
    previous = None
    complex_line = False
    extra_cursor = 0
    for attempt in range(2):
        if time.monotonic() > deadline-5:
            return None
        extending = attempt == 1 and complex_line
        if not extending:
            direction = rng.uniform(0.55, 0.9, d)*rng.choice([-1, 1], d)
        n = 129 if extending else 65
        t = np.linspace(0, 1, n)
        chosen_t = t[1::2] if extending else t
        count = len(chosen_t)
        if count*(d+1) > oracle.remaining:
            break
        base = 0.5+(chosen_t[:, None]-0.5)*direction
        values = np.empty((n, d+1))
        if extending:
            values[::2] = previous
        new_base = oracle.ask(base)/scale-offset/scale
        if extending:
            values[1::2, 0] = new_base
        else:
            values[:, 0] = new_base
        fourth = np.abs(np.diff(values[:, 0], n=4))
        if np.max(fourth, initial=0) > max(np.median(fourth)*300, 1e-6):
            return None
        points = np.tile(base, (d, 1))
        for j in range(d):
            points[j*count:(j+1)*count, j] += delta
        shifted = oracle.ask(points).reshape(d, count).T/scale-offset/scale
        if extending:
            values[1::2, 1:] = shifted
        else:
            values[:, 1:] = shifted
        previous = values.copy()
        enhanced = np.column_stack((values[:, 0], (values[:, 1:]-values[:, 0, None])/delta))
        enhanced /= np.maximum(np.std(enhanced, axis=0), 0.1)
        tried = []
        complex_line = False
        for threshold in (1e-7, 1e-10, 1e-12, 2e-14):
            if time.monotonic() > deadline-4:
                return None
            omega, rank = line_spectrum(enhanced, t, threshold)
            complex_line |= rank >= 28
            if not omega or len(omega) > (28 if extending else 14) or len(omega) in tried:
                continue
            tried.append(len(omega))
            try:
                omega, _, line_error = refine_line_spectrum(omega, t, enhanced)
                if line_error > 1e-4:
                    continue
                r = len(omega)
                tc = t-0.5
                a = np.column_stack((np.ones(n), tc, tc*tc, np.cos(tc[:, None]*omega),
                                     np.sin(tc[:, None]*omega)))
                coeff = linalg.lstsq(a, values, cond=1e-13, lapack_driver="gelsy")[0]
                amplitudes = coeff[3:3+r]-1j*coeff[3+r:3+2*r]
                base_amplitudes = amplitudes[:, 0]
                valid = np.abs(base_amplitudes) > 1e-12
                if not np.any(valid):
                    continue
                ratios = amplitudes[valid, 1:] / base_amplitudes[valid, None]
                frequencies = np.angle(ratios) / delta
                period = 2*np.pi/(t[1]-t[0])
                mismatch = np.abs((frequencies@direction-omega[valid]+period/2)%period-period/2)
                if np.max(mismatch) > 1.0:
                    continue
                parameters = background(x[:1]-0.5).shape[1]+len(frequencies)*(d+2)
                if extra_points is not None and parameters > 0.6*len(x):
                    target = min(512, 2*parameters+64)
                    need = min(target-len(x), len(extra_points)-extra_cursor, oracle.remaining)
                    if need > 0:
                        pp = extra_points[extra_cursor:extra_cursor+need]
                        raw = query_uniform(oracle, pp)
                        x = np.vstack((x, pp))
                        yy = np.r_[yy, raw/scale-offset/scale]
                        extra_cursor += need
                        if not hasattr(oracle, "extra_global"):
                            oracle.extra_global = []
                        oracle.extra_global.append((pp, raw))
                answer = fourier_refine(x, yy, frequencies, base_amplitudes[valid], deadline)
                if answer is not None:
                    return float(offset+answer*scale)
            except (ValueError, linalg.LinAlgError, FloatingPointError):
                continue
    return None


def gradient_at(oracle, points, scale):
    d = oracle.d
    points = np.asarray(points).reshape(-1, d)
    h = 2**-18
    pp = np.repeat(points, 2*d, axis=0)
    for j in range(d):
        pp[2*j::2*d, j] += h
        pp[2*j+1::2*d, j] -= h
    yy = oracle.ask(pp).reshape(len(points), d, 2)/scale
    return (yy[:, :, 0]-yy[:, :, 1])/(2*h)


def refine_hinges(x, y, weights, biases, signs, integral):
    n, d = x.shape
    r = len(weights)
    weights = np.asarray(weights)
    biases = np.asarray(biases)
    signs = np.asarray(signs)
    z = x-0.5
    design = background(z)
    nb = design.shape[1]
    partial = np.abs(z@weights.T-biases)@signs
    beta = linalg.lstsq(design, y-partial, lapack_driver="gelsy")[0]
    params = np.r_[beta, np.column_stack((weights, biases)).ravel()]
    def fun(p, zz, yy):
        pp = p[nb:].reshape(r, d+1)
        return background(zz)@p[:nb]+np.abs(zz@pp[:, :d].T-pp[:, d])@signs-yy
    def jac(p, zz, yy):
        pp = p[nb:].reshape(r, d+1)
        dd = np.sign(zz@pp[:, :d].T-pp[:, d])*signs
        a = np.empty((len(zz), len(p)))
        a[:, :nb] = background(zz)
        for k in range(r):
            a[:, nb+k*(d+1):nb+(k+1)*(d+1)] = np.column_stack((zz, -np.ones(len(zz))))*dd[:, k, None]
        return a
    if n <= len(params):
        return None
    fit = optimize.least_squares(fun, params, jac=jac, args=(z, y), method="lm",
                                max_nfev=70, ftol=1e-13, xtol=1e-13, gtol=1e-13)
    residual = fun(fit.x, z, y)
    if np.max(np.abs(residual)) > 3e-9:
        return None
    pp = fit.x[nb:].reshape(r, d+1)
    answer = fit.x[0]+sum(s*integral(p[:d], p[d]) for s, p in zip(signs, pp))
    return float(answer)


def hessian_at(oracle, x, scale):
    d = oracle.d
    h = 0.0003
    eye = np.eye(d)*h
    points = [x]
    for j in range(d):
        points.extend([x+eye[j], x-eye[j]])
    pairs = list(itertools.combinations(range(d), 2))
    for i, j in pairs:
        points.extend([x+eye[i]+eye[j], x+eye[i]-eye[j],
                       x-eye[i]+eye[j], x-eye[i]-eye[j]])
    y = oracle.ask(points)/scale
    hess = np.zeros((d, d))
    for j in range(d):
        hess[j, j] = (y[1+2*j]+y[2+2*j]-2*y[0])/(h*h)
    k = 1+2*d
    for i, j in pairs:
        hess[i, j] = hess[j, i] = (y[k]-y[k+1]-y[k+2]+y[k+3])/(4*h*h)
        k += 4
    return hess


def fit_piecewise(oracle, x, y, integral, deadline):
    d = oracle.d
    scale = max(np.max(np.abs(y)), 1e-300)
    yy = y/scale
    weights, biases, signs = [], [], []
    rng = np.random.default_rng(oracle.seed ^ 0x4F217C95)
    hessian = np.zeros((d, d))
    have_hessian = False

    def partial(p):
        q = 0.5*np.sum(((p-0.5)@hessian)*(p-0.5), axis=1)
        if not weights:
            return q
        return q+np.abs((p-0.5)@np.asarray(weights).T-np.asarray(biases))@np.asarray(signs)

    def partial_gradient(p):
        q = (p-0.5)@hessian
        if not weights:
            return q
        w = np.asarray(weights)
        return q+(np.sign((p-0.5)@w.T-np.asarray(biases))*np.asarray(signs))@w

    for attempt in range(7):
        if oracle.remaining < 65+4*d or time.monotonic() > deadline-4:
            break
        if attempt == 0:
            direction = np.full(d, 0.998)
            center = np.full(d, 0.5)
        else:
            direction = rng.choice([-1.0, 1.0], d)*rng.uniform(0.65, 0.998, d)
            center = 0.5+rng.uniform(-1, 1, d)*(0.499-np.abs(direction)/2)
        ts = np.linspace(0, 1, 65)
        points = center+(ts[:, None]-0.5)*direction
        vals = oracle.ask(points)/scale-partial(points)
        slopes = np.diff(vals)/np.diff(ts)
        second = np.diff(slopes)/(ts[1]-ts[0])
        curvature = np.median(second)
        irregular = np.abs(second-curvature) > max(2e-4, abs(curvature)*1e-4)
        if np.count_nonzero(irregular) > len(second)*0.75:
            return None
        if not have_hessian and abs(curvature) > 1e-5:
            if oracle.remaining < 2*d*d+1:
                return None
            hessian = hessian_at(oracle, points[5], scale)
            have_hessian = True
            vals -= 0.5*np.sum(((points-0.5)@hessian)*(points-0.5), axis=1)
            slopes = np.diff(vals)/np.diff(ts)
        differences = np.flatnonzero(np.abs(np.diff(slopes)) > 2e-6)
        groups = []
        for _, group in itertools.groupby(enumerate(differences), lambda kv: kv[1]-kv[0]):
            indices = [int(v) for _, v in group]
            groups.append((indices[0], indices[-1]+2))
        if not groups:
            continue
        edges = sorted(set(v for group in groups for v in group))
        if len(edges)*2*d > oracle.remaining:
            break
        ge = gradient_at(oracle, points[edges], scale)-partial_gradient(points[edges])
        gradient_map = dict(zip(edges, ge))
        regions = [(ts[a], ts[b], vals[a], vals[b], gradient_map[a], gradient_map[b], 0)
                   for a, b in groups]
        recovered = []
        while regions and oracle.remaining >= 1 and time.monotonic() < deadline-3:
            a, b, fa, fb, ga, gb, depth = regions.pop()
            sl, sr = ga@direction, gb@direction
            jump = sr-sl
            if abs(jump) > 1e-9:
                knot = (fb-fa+sl*a-sr*b)/(sl-sr)
            else:
                knot = (a+b)/2
            inside = a+1e-9 < knot < b-1e-9
            if not inside:
                knot = (a+b)/2
            p = center+(knot-0.5)*direction
            fk = float(oracle.ask(p[None, :])[0]/scale-partial(p[None, :])[0])
            error = abs(fk-fa-sl*(knot-a))
            if inside and error < 2e-8 and abs(jump) > 1e-9:
                w = (gb-ga)/2
                bias = w@(p-0.5)
                sign = float(np.sign(jump))
                recovered.append((w, bias, sign))
                continue
            if depth >= 13 or oracle.remaining < 2*d+1:
                continue
            if min(knot-a, b-knot) < (b-a)*0.1:
                knot = (a+b)/2
                p = center+(knot-0.5)*direction
                fk = float(oracle.ask(p[None, :])[0]/scale-partial(p[None, :])[0])
            gm = gradient_at(oracle, p[None, :], scale)[0]-partial_gradient(p[None, :])[0]
            if np.linalg.norm(gm-ga) > 1e-7 or abs(fk-fa-sl*(knot-a)) > 1e-8:
                regions.append((a, knot, fa, fk, ga, gm, depth+1))
            if np.linalg.norm(gm-gb) > 1e-7 or abs(fb-fk-sr*(b-knot)) > 1e-8:
                regions.append((knot, b, fk, fb, gm, gb, depth+1))
        for w, bias, sign in recovered:
            norm = np.linalg.norm(w)
            if norm < 1e-9:
                continue
            duplicate = False
            for v, bb in zip(weights, biases):
                ratio = np.dot(w, v)/np.dot(v, v)
                if np.linalg.norm(w-ratio*v) < 1e-6*norm and abs(bias-ratio*bb) < 1e-6*norm:
                    duplicate = True
                    break
            if not duplicate:
                weights.append(w)
                biases.append(bias)
                signs.append(sign)
        if weights:
            try:
                answer = refine_hinges(x, yy, weights, biases, signs, integral)
                if answer is not None:
                    return answer*scale
            except (ValueError, linalg.LinAlgError, FloatingPointError, TimeoutError):
                pass
    return None
