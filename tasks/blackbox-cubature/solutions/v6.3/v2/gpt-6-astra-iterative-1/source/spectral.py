"""Multivariate exponential interpolation from two-dimensional sections."""
import time
import itertools
import numpy as np
from scipy import linalg, optimize
from ridges import fourier_refine, background
from sampling import mean as stable_mean, standard_deviation as stable_std
from sampling import query_uniform


def pencil(values, n, threshold):
    if not np.all(np.isfinite(values)):
        return None
    m = n//2
    left = np.array([(i, j) for i in range(m) for j in range(m)])
    right = np.array([(i, j) for i in range(n-m) for j in range(n-m)])
    ii = left[:, None, 0]+right[None, :, 0]
    jj = left[:, None, 1]+right[None, :, 1]
    h0 = np.concatenate([values[ii, jj, k] for k in range(values.shape[2])])
    hx = np.concatenate([values[ii+1, jj, k] for k in range(values.shape[2])])
    hy = np.concatenate([values[ii, jj+1, k] for k in range(values.shape[2])])
    u, s, vh = linalg.svd(h0, full_matrices=False, check_finite=False)
    rank = min(int(np.count_nonzero(s > s[0]*threshold)), len(s))
    if rank < 3:
        return None
    a = (u[:, :rank].T@hx@vh[:rank].T)/s[:rank, None]
    b = (u[:, :rank].T@hy@vh[:rank].T)/s[:rank, None]
    _, vectors = linalg.eig(a+(0.37+0.53j)*b)
    if np.linalg.cond(vectors) > 1e10:
        return None
    zx = np.diag(linalg.solve(vectors, a@vectors))
    zy = np.diag(linalg.solve(vectors, b@vectors))
    roots = []
    for xx, yy in zip(zx, zy):
        if max(abs(abs(xx)-1), abs(abs(yy)-1)) > 0.06:
            continue
        w = np.array([np.angle(xx), np.angle(yy)])*(n-1)
        if np.linalg.norm(w) < 0.01 or w[0] < -0.001 or (abs(w[0]) < 0.001 and w[1] < 0):
            continue
        if not any(np.linalg.norm(w-v) < 0.003 for v in roots):
            roots.append(w)
    return np.asarray(roots) if roots else None


def basis(coords, frequencies):
    t, u = coords.T
    phase = coords@frequencies.T
    return np.column_stack((np.ones(len(t)), t, u, t*t, t*u, u*u,
                            np.cos(phase), np.sin(phase)))


def refine(coords, data, frequencies, deadline):
    r = len(frequencies)
    cache = [None]
    def calc(p):
        if len(cache) == 1 or np.any(cache[0] != p):
            w = p.reshape(r, 2)
            a = basis(coords, w)
            c = linalg.lstsq(a, data, cond=1e-13, lapack_driver="gelsy")[0]
            cache[:] = [p.copy(), w, a, c]
        return cache[1:]
    def fun(p):
        if time.monotonic() > deadline:
            raise TimeoutError()
        _, a, c = calc(p)
        return (a@c-data).ravel()
    def jac(p):
        w, a, c = calc(p)
        q = linalg.qr(a, mode="economic", check_finite=False)[0]
        phase = coords@w.T
        result = []
        for k in range(r):
            der = -np.sin(phase[:, k, None])*c[6+k]+np.cos(phase[:, k, None])*c[6+r+k]
            for j in range(2):
                v = coords[:, j, None]*der
                v -= q@(q.T@v)
                result.append(v.ravel())
        return np.column_stack(result)
    fit = optimize.least_squares(fun, frequencies.ravel(), jac=jac, method="lm",
                                max_nfev=60, ftol=1e-13, xtol=1e-13, gtol=1e-13)
    w, a, c = calc(fit.x)
    return w, c, float(np.max(np.abs(a@c-data)))


def integrate(oracle, x, y, extra_points, deadline):
    d = oracle.d
    n = 14
    size = n*n
    if d < 2 or oracle.remaining < size*(d-1)+64:
        return None
    offset = stable_mean(y)
    scale = max(stable_std(y), np.max(np.abs(y))*1e-14, 1e-300)
    yy = y/scale-offset/scale
    rng = np.random.default_rng(oracle.seed ^ 0x57491BD3)
    angles = rng.uniform(0, 2*np.pi, d)
    a, b = np.cos(angles), np.sin(angles)
    radius = rng.uniform(0.8, 0.9, d)/(np.abs(a)+np.abs(b))
    a, b = a*radius, b*radius
    plane = np.column_stack((a, b))
    missing = max(itertools.combinations(range(d), 2),
                  key=lambda ij: abs(np.linalg.det(plane[list(ij)])))
    observed = [j for j in range(d) if j not in missing]
    inverse = linalg.inv(plane[list(missing)])
    delta = min(0.04, 1.6/max(getattr(oracle, "characteristic_frequency", 0), 1))
    grid = np.linspace(-0.5, 0.5, n)
    coords = np.array([(t, u) for t in grid for u in grid])
    base = 0.5+coords[:, 0, None]*a+coords[:, 1, None]*b
    initial_axes = observed[:2]
    channels = [np.zeros(d)]+[np.eye(d)[j]*delta for j in initial_axes]
    points = np.vstack([base+shift for shift in channels])
    values = oracle.ask(points).reshape(len(channels), size).T/scale-offset/scale
    enhanced = np.column_stack((values[:, 0], (values[:, 1:]-values[:, 0, None])/delta))
    enhanced /= np.maximum(np.std(enhanced, axis=0), 0.1)
    candidates = []
    counts = set()
    for threshold in (1e-8, 1e-11, 2e-14):
        if time.monotonic() > deadline-4:
            break
        try:
            w = pencil(enhanced.reshape(n, n, -1), n, threshold)
            if w is None or len(w) > 24 or len(w) in counts:
                continue
            counts.add(len(w))
            w, c, error = refine(coords, enhanced, w, deadline-3)
            if error < 2e-8:
                candidates.append((error, w))
        except (ValueError, linalg.LinAlgError, TimeoutError):
            continue
    if not candidates:
        return None
    if len(observed) > 2:
        points = np.vstack([base+np.eye(d)[j]*delta for j in observed[2:]])
        tail = oracle.ask(points).reshape(len(observed)-2, size).T/scale-offset/scale
        values = np.column_stack((values, tail))
    enhanced = np.column_stack((values[:, 0], (values[:, 1:]-values[:, 0, None])/delta))
    enhanced /= np.maximum(np.std(enhanced, axis=0), 0.1)
    cursor = 0
    candidates.sort(key=lambda c: c[0])
    for _, w in candidates:
        if time.monotonic() > deadline-2:
            return None
        try:
            w, _, error = refine(coords, enhanced, w, deadline-1)
            if error > 2e-7:
                continue
            r = len(w)
            c = linalg.lstsq(basis(coords, w), values, cond=1e-13, lapack_driver="gelsy")[0]
            amplitudes = c[6:6+r]-1j*c[6+r:6+2*r]
            good = np.abs(amplitudes[:, 0]) > 1e-11
            if not np.any(good):
                continue
            amplitudes, w = amplitudes[good], w[good]
            ratios = amplitudes[:, 1:]/amplitudes[:, :1]
            if ratios.size and np.max(np.abs(np.abs(ratios)-1)) > 0.1:
                continue
            known = np.angle(ratios)/delta
            frequencies = np.zeros((len(w), d))
            frequencies[:, observed] = known
            period = 2*np.pi*(n-1)
            residual = w-known@plane[observed]
            aliases = np.array(list(itertools.product(range(-4, 5), repeat=2)))*period
            options = (residual[:, None, :]+aliases)@inverse
            best = np.argmin(np.sum(options*options, axis=2), axis=1)
            frequencies[:, list(missing)] = options[np.arange(len(w)), best]
            parameters = background(x[:1]-0.5).shape[1]+len(w)*(d+2)
            if parameters > 0.6*len(x):
                target = min(512, 2*parameters+64)
                needed = min(target-len(x), len(extra_points)-cursor, oracle.remaining)
                if needed > 0:
                    pp = extra_points[cursor:cursor+needed]
                    raw = query_uniform(oracle, pp)
                    x = np.vstack((x, pp))
                    yy = np.r_[yy, raw/scale-offset/scale]
                    cursor += needed
                    if not hasattr(oracle, "extra_global"):
                        oracle.extra_global = []
                    oracle.extra_global.append((pp, raw))
            result = fourier_refine(x, yy, frequencies, amplitudes[:, 0], deadline)
            if result is not None:
                return float(offset+scale*result)
        except (ValueError, linalg.LinAlgError, TimeoutError):
            continue
    return None
