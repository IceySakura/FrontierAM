"""Positive log-quadratic mixtures and defensive importance sampling."""

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

from core import Model, least, log_quadratic_features, log_quad_moments


def unpack(c, d):
    return c[:, 0], c[:, 1:d+1], np.exp(c[:, d+1:])


def logs(x, c):
    amp, mid, p = unpack(c, x.shape[1])
    q = x[:, None, :]-mid
    return amp-np.sum(p*q*q, axis=2)


def log_predict(x, c):
    if not len(c):
        return np.full(len(x), -np.inf)
    return special.logsumexp(logs(x, c), axis=1)


def log_means(c, d):
    amp, mid, p = unpack(c, d)
    q = .5-mid
    m = []
    for a, q0, p0 in zip(amp, q, p):
        m.append(a-np.sum(p0*q0*q0)+np.sum(log_quad_moments(-2.*p0*q0, -p0)))
    return np.array(m)


@dataclass
class PeakModel(Model):
    c: np.ndarray
    quality: float


def make_model(c, d, quality=1.):
    integral = np.exp(min(float(special.logsumexp(log_means(c, d))), 700.))
    return PeakModel(lambda x: np.exp(np.minimum(log_predict(x, c), 700.)),
                     float(integral), "Gaussian mixture", c.copy(), float(quality))


def difference_log(y, p):
    z = np.full(len(y), -np.inf)
    mask = y > p+1.e-12
    z[mask] = y[mask] + np.log(-np.expm1(p[mask]-y[mask]))
    return z


def local_gaussian(s, anchor, prior):
    d = s.d
    eye = np.eye(d)
    gap = .034
    xx = np.broadcast_to(anchor, (4, d, d)).copy()
    j = np.arange(d)
    l = np.maximum(anchor-gap*2, 1.e-8)
    r = np.minimum(anchor+gap*2, 1.-1.e-8)
    for i, q in enumerate((.0, .23, .73, 1.)):
        xx[i, j, j] = l+q*(r-l)
    xy = np.clip(anchor+(s.design(7)-.5)*.064, 1.e-8, 1.-1.e-8)
    pts = np.vstack((xx.reshape(-1, d), anchor, xy))
    vals = s.ask(pts)/s.scale
    if np.any(vals <= 0.):
        return []
    ell = np.log(vals)
    prev = log_predict(pts, prior)
    responses = (ell, difference_log(ell, prev)) if len(prior) else (ell,)
    result = []
    centered = (pts-anchor)/gap
    basis = np.column_stack((np.ones(len(pts)), centered, centered*centered))
    for y in responses:
        ok = np.isfinite(y)
        if np.count_nonzero(ok) < 3*d+2:
            continue
        c = least(basis[ok], y[ok])
        if np.max(np.abs(basis[ok]@c-y[ok])) > .94:
            continue
        p = -c[d+1:]/(gap*gap)
        if np.count_nonzero(p < -.4) > max(1, d//2):
            continue
        p = np.clip(p, .06, 48000.)
        # If the local fit would put a mode far outside the box,
        # retain its slope with sufficient precision.
        slope = c[1:d+1]/gap
        p = np.maximum(p, np.abs(slope)/(2.*np.where(slope >= 0.,
                       1.23-anchor, anchor+.23)))
        off = slope/(2.*p)
        mid = anchor+off
        amp = c[0]+np.sum(p*off*off)
        if amp > np.max(ell)+350. or np.any(p > 48000.):
            continue
        result.append(np.r_[amp, mid, np.log(p)])
    return result


def fit_mixture(x, y, candidate, limit=110, robust=True):
    n, d = x.shape
    shape = candidate.shape
    pmin, pmax = np.log(.005), np.log(55000.)
    lo = np.broadcast_to(np.r_[min(-650., y.min()-20.), np.full(d, -.24),
                               np.full(d, pmin)], shape).ravel()
    hi = np.broadcast_to(np.r_[max(80., np.max(candidate[:, 0])+5.), np.full(d, 1.24),
                               np.full(d, pmax)], shape).ravel()
    start = np.clip(candidate.ravel(), lo+1.e-10, hi-1.e-10)

    def residual(z, jac=False):
        c = z.reshape(shape)
        amp, mid, p = unpack(c, d)
        q = x[:, None, :]-mid
        E = p*q*q
        F = amp-np.sum(E, axis=2)
        l = special.logsumexp(F, axis=1)
        if not jac:
            return l-y
        r = np.exp(F-l[:, None])
        return np.concatenate((r[..., None], 2.*p*q*r[..., None], -E*r[..., None]), axis=2).reshape(n, -1)

    opts = {"max_nfev": limit, "xtol": 7.e-12, "ftol": 5.e-12, "gtol": 5.e-12,
            "jac": lambda z: residual(z, True), "x_scale": "jac"}
    z = optimize.least_squares(residual, start, bounds=(lo, hi),
                               loss="soft_l1" if robust else "linear", f_scale=.55, **opts)
    if robust and np.max(np.abs(z.fun)) < 1.2 and np.max(np.abs(z.fun)) > 2.e-11:
        z = optimize.least_squares(residual, z.x, bounds=(lo, hi), **opts)
    return z.x.reshape(shape)


def peak_points(s, c, n, expand=1.):
    d = s.d
    _, mid, p = unpack(c, d)
    sig = expand/np.sqrt(2.*p)
    l, r = special.ndtr(-mid/sig), special.ndtr((1.-mid)/sig)
    u = s.design(n*len(c)).reshape(len(c), n, d)
    return np.clip(mid[:, None, :]+sig[:, None, :]*special.ndtri(np.clip(
        l[:, None, :]+u*(r-l)[:, None, :], 1.e-14, 1.-1.e-14)), 1.e-8, 1.-1.e-8)


def log_data(s):
    x, y = s.data()
    use = y > 1.e-295
    return x[use], np.log(y[use])


def quality(err):
    return float(np.mean(2.*np.hypot(err, .55)-1.1))


def fit_peaks(s):
    x, y = s.data()
    d = s.d
    if (not np.all(y >= 0.) or np.count_nonzero(y > 0.) < (3*d+10)
            or s.left < 420):
        return None, None
    _, ly = log_data(s)
    if np.percentile(ly, 90)-np.percentile(ly, 10) < 1.3:
        return None, None
    F = log_quadratic_features(x[y > 1.e-295])
    init = least(F, ly)
    # An exploratory local quadratic requires evidence of a peak,
    # either globally or in a densely sampled neighborhood.
    rms = float(np.sqrt(np.mean((F@init-ly)**2)))
    if np.max(ly) < -3.:
        return None, None
    c = np.empty((0, 2*d+1))
    probed = []
    # Keep a representative for variance reduction only after checking
    # that the localized proposal is real and significant.
    best = None
    misses = 0
    for iteration in range(10):
        if (s.left < 328+5*d or s.time_left() < 4.5 or len(c) >= 6 or misses >= 4):
            break
        x, ly = log_data(s)
        pr = log_predict(x, c)
        if not len(c):
            stuff = ly.copy()
        else:
            diff = difference_log(ly, pr)
            lratio = np.maximum(ly-pr, 0.)
            power = (.8, .55, .3, .88)[(iteration+misses) % 4]
            stuff = (1.-power)*ly+power*diff + .08*np.minimum(lratio, 16.)
            if not np.any(np.isfinite(stuff)):
                break
        for point in probed:
            stuff -= 25.*np.exp(-np.sum(((point-x)/.13)**2, axis=1))
        which = int(np.argmax(stuff))
        a = x[which]
        if probed and min(np.linalg.norm(a-p) for p in probed) < .012:
            misses += 1
            continue
        probed.append(a.copy())
        candidates = local_gaussian(s, a, c)
        if not candidates:
            misses += 1
            continue
        x, ly = log_data(s)
        comparison = log_predict(x, c)
        q0 = quality(comparison-ly) if len(c) else float("inf")
        potential = []
        for p in candidates:
            Cp = np.vstack((c, p))
            pred = log_predict(x, Cp)
            potential.append((quality(pred-ly), Cp))
        potential.sort(key=lambda item: item[0])
        qb = float("inf")
        chosen = None
        for q_init, p in potential:
            if s.time_left() < 4.2:
                break
            pp = fit_mixture(x, ly, p, limit=84 if len(p) < 4 else 61)
            pred = log_predict(x, pp)
            qt = quality(pred-ly)
            if qt < qb:
                qb, chosen = qt, pp
            if qt < 1.e-8 or (len(p) >= 2 and qt < q0/5):
                break
        if chosen is None or qb >= q0*.995:
            misses += 1
            continue
        c = chosen
        misses = 0
        pred = log_predict(x, c)
        maxerr = np.max(np.abs(pred-ly))
        if maxerr < 5.e-8 and s.left > 128+16*len(c):
            p = np.vstack((s.design(88), peak_points(s, c, 12, expand=1.4).reshape(-1, d)))
            q = s.ask(p)/s.scale
            good = q > 1.e-295
            pred = log_predict(p, c)
            if np.max(np.abs(pred[good]-np.log(q[good])), initial=0.) < 5.e-8 and np.all(pred[~good] < -645.):
                zz, yy = log_data(s)
                c = fit_mixture(zz, yy, c, limit=24, robust=False)
                return make_model(c, d, 0.), best
        if s.left < 342+14*len(c):
            continue
        if maxerr < 12. or len(c) >= 2 or rms < 2.:
            # Check predictions near each extrapolated maximum and tails.
            scout = np.vstack((np.clip(c[:, 1:d+1], 1.e-8, 1.-1.e-8),
                               peak_points(s, c, 8, expand=1.55).reshape(-1, d)))
            q = s.ask(scout)/s.scale
            qpred = np.exp(np.minimum(log_predict(scout, c), 700.))
            if np.all(np.isfinite(qpred)):
                qc = float(np.linalg.norm(q-qpred)/max(np.linalg.norm(q), 1.e-20))
                xx, yy = log_data(s)
                if qc < 1.15 and (best is None or qc < best.quality*.85 or (qc < .5 and len(c) > len(best.c))):
                    best = make_model(c, d, qc)
                if qc < 5.:
                    c = fit_mixture(xx, yy, c, limit=56)
                else:
                    # A grossly over-extrapolated component is a poor
                    # starting point, but its observations remain useful.
                    c = fit_mixture(xx, yy, c, limit=43)
    return None, best


def integrate_peak_residual(s, model):
    """Deterministically stratify a defensive mixture of proposals."""
    d = s.d
    n = s.left
    if n < 256 or not np.isfinite(model.mean) or model.mean <= 0.:
        return None
    c = model.c
    _, mid, p = unpack(c, d)
    sig = 1./np.sqrt(2.*p)
    l, r = special.ndtr(-mid/sig), special.ndtr((1.-mid)/sig)
    if np.any(r <= l):
        return None
    m = len(c)
    means = log_means(c, d)
    prop = np.r_[.23, .77*np.exp(means-special.logsumexp(means))]
    prop[1:] = np.maximum(prop[1:], .035)
    wanted = n*prop/prop.sum()
    ns = np.floor(wanted).astype(int)
    ns[np.argsort(-(wanted-ns))[:n-ns.sum()]] += 1
    pi = ns/n
    pieces = []
    correction = []
    for i, ni in enumerate(ns):
        if not ni:
            continue
        z = s.design(int(ni))
        if i:
            j = i-1
            z = mid[j]+sig[j]*special.ndtri(np.clip(l[j]+z*(r[j]-l[j]), 1.e-15, 1.-1.e-15))
            z = np.clip(z, 0., 1.)
        pieces.append(z)
    x = np.vstack(pieces)
    raw = s.ask(x)/s.scale
    density = pi[0]+np.exp(np.minimum(logs(x, c)-means, 690.))@pi[1:]
    fx = model.predict(x)
    correct = (raw-fx)/density
    result = model.mean+np.mean(correct)
    if np.isfinite(result) and result > 0.:
        return float(result)
    return float(np.mean(raw/density))
