"""Numerical fallback: adaptive rules and regression-assisted QMC."""

from functools import lru_cache
import math

import numpy as np
from scipy import linalg, special
from scipy.integrate._rules import GenzMalikCubature

from core import Model, legendre_features, least


def error_score(err):
    return float(np.sqrt(np.mean(np.square(err))))


def orthogonal_features(x, order, extension):
    base = legendre_features(x, order)
    if not extension:
        return base
    z = 2.*x-1.
    p, q = np.ones_like(z), z
    out = [base]
    for k in range(2, extension+1):
        p, q = q, ((2*k-1)*z*q-(k-1)*p)/k
        if k > order:
            out.append(q)
    if extension >= 9:
        knots = np.array([.125, .25, .375, .5, .625, .75, .875])
        hinge = np.maximum(x[:, :, None]-knots, 0.)-.5*(1.-knots)**2
        out.append(hinge.reshape(len(x), -1))
    return np.column_stack(out)


def regress(x, y, xv, yv):
    d = x.shape[1]
    score = error_score(yv-y.mean())
    best = None
    for order, ext in [(1, 0), (2, 0), (2, 7), (3, 10), (4, 12), (6, 12)]:
        if math.comb(order+d, d) > min(.48*len(x), 380):
            continue
        f = orthogonal_features(x, order, ext)
        if f.shape[1] > .66*len(x):
            continue
        g = orthogonal_features(xv, order, ext)
        if ext >= 9:
            penalties = np.r_[np.full(f.shape[1]-7*d, .00015), np.full(7*d, .006)]
        else:
            penalties = np.full(f.shape[1], .000075)
        penalties[0] = 1.e-13
        # Normalize penalization relative to sample size.
        chol = linalg.cho_factor(f.T@f + np.diag(penalties*len(f)), check_finite=False)
        c = linalg.cho_solve(chol, f.T@y, check_finite=False)
        err = error_score(yv-g@c)
        if err < score*.99:
            score = err
            best = ((order, ext), err)
    return best


def kernel_dist(x, z, family):
    delta = x[:, None, :]-z[None, :, :]
    if family == 0:
        return np.sum(delta**2, axis=2)
    return np.abs(delta)


def kernel_from_dist(delta, ell, family):
    if family == 0:
        return np.exp(-.5*delta/(ell*ell))
    q = (math.sqrt(3.)/ell)*delta
    if family == 1:
        return np.prod((1.+q)*np.exp(-q), axis=2)
    # Sum of low-order interactions of independent 1-D kernels.
    f = (1.+q)*np.exp(-q)
    s1 = np.sum(f, axis=2)
    s2 = np.sum(f*f, axis=2)
    d = delta.shape[2]
    if d == 1:
        return s1
    return (.45*s1/d + .55*(s1*s1-s2)/(d*(d-1)))


def kernel_moments(x, ell, family):
    if family == 0:
        return np.prod(ell*math.sqrt(np.pi/2.)*(
            special.erf(x/(ell*math.sqrt(2.)))+special.erf((1.-x)/(ell*math.sqrt(2.)))), axis=1)
    a = math.sqrt(3.)/ell
    h = (4.-(a*x+2.)*np.exp(-a*x)-(a*(1.-x)+2.)*np.exp(-a*(1.-x)))/a
    d = x.shape[1]
    if family == 1:
        return np.prod(h, axis=1)
    s1, s2 = h.sum(axis=1), np.sum(h*h, axis=1)
    return s1 if d == 1 else (.45*s1/d+.55*(s1*s1-s2)/(d*(d-1)))


def kernel_fit(x, y, g, noise):
    K = g.copy()
    K.flat[::len(K)+1] += noise
    chol = linalg.cho_factor(K, check_finite=False, overwrite_a=True)
    t = linalg.cho_solve(chol, np.column_stack((y, np.ones(len(x)))), check_finite=False)
    b = t[:, 0].sum()/t[:, 1].sum()
    return b, t[:, 0]-b*t[:, 1]


def gp_select(s, x, y, xv, yv):
    d = s.d
    best = None
    ref = error_score(yv-y.mean())
    for family in (0, 1, 2):
        if s.time_left() < 3.:
            break
        dist = kernel_dist(x, x, family)
        dv = kernel_dist(xv, x, family)
        length = ((.084, .15, .27, .48) if d <= 3 else (.15, .28, .50, .84))
        if family == 2:
            length = (.08, .17, .34, .64)
        for ell in length:
            gr = kernel_from_dist(dist, ell, family)
            row = kernel_from_dist(dv, ell, family)
            for noise in (5.e-10, 1.e-4, .025):
                b, c = kernel_fit(x, y, gr, noise)
                err = error_score(yv-(b+row@c))
                if err < ref*.99:
                    ref = err
                    best = ((family, ell, noise), err)
    return best


def qmc_regression(s):
    if s.left <= 192 or s.time_left() < 2.5:
        s.uniform(s.left)
        return s.quick(), "Sobol"
    ntrain = min(768, max(256, (s.nu+s.left)//2))
    if ntrain > s.nu:
        s.uniform(ntrain-s.nu)
    x = s.ux[:s.nu]
    y = s.uy[:s.nu]/s.scale
    perm = s.rng.permutation(len(x))
    n = min(len(x)//4, 192)
    it, iv = perm[n:], perm[:n]
    yt, yv = y[it], y[iv]
    ref = error_score(yv-yt.mean())
    if ref == 0.:
        s.uniform(s.left)
        return s.quick(), "Sobol"
    poly = regress(x[it], yt, x[iv], yv)
    gp = gp_select(s, x[it], yt, x[iv], yv)
    ep = poly[1] if poly else float("inf")
    eg = gp[1] if gp else float("inf")
    if min(ep, eg) > .87*ref or s.time_left() < 2.:
        s.uniform(s.left)
        return s.quick(), "Sobol"
    if ep <= eg*1.012:
        order, ext = poly[0]
        f = orthogonal_features(x, order, ext)
        if ext >= 9:
            p = np.r_[np.full(f.shape[1]-7*s.d, .00015), np.full(7*s.d, .006)]
        else:
            p = np.full(f.shape[1], .000075)
        p[0] = 1.e-13
        # Incorporate a few non-collocated pilot measurements, lightly.
        b = linalg.solve(f.T@f + np.diag(p*len(f)), f.T@y,
                         assume_a="pos", check_finite=False)
        model = Model(lambda z: orthogonal_features(z, order, ext)@b,
                      float(b[0]), "regression QMC")
    else:
        family, ell, noise = gp[0]
        gr = kernel_from_dist(kernel_dist(x, x, family), ell, family)
        b, c = kernel_fit(x, y, gr, noise)
        moment = float(b+kernel_moments(x, ell, family)@c)
        def predict(z):
            return np.concatenate([b+kernel_from_dist(kernel_dist(q, x, family), ell, family)@c
                                   for q in np.array_split(z, (len(z)+127)//128)])
        model = Model(predict, moment, "kernel QMC")
    if not np.isfinite(model.mean):
        s.uniform(s.left)
        return s.quick(), "Sobol"
    # An independently scrambled net for the residual avoids using the
    # same noise both to choose the regression and to correct its mean.
    z = s.design(s.left)
    q = s.ask(z)/s.scale
    residual = q-model.predict(z)
    result = model.mean+float(np.mean(residual))
    if not np.isfinite(result):
        return s.quick(), "Sobol"
    return result*s.scale, model.name


@lru_cache(maxsize=8)
def gm_rule(d):
    g = GenzMalikCubature(d)
    x, w = g.nodes_and_weights
    _, wg = g.lower_nodes_and_weights
    w = np.asarray(w)/2**d
    we = w.copy()
    we[:len(wg)] -= wg/2**d
    return x, w, we


def adaptive_cubature(s, model=None):
    """Allocate embedded cubature rules to the most uncertain boxes."""
    d = s.d
    nodes, w, e = gm_rule(d)
    n = len(w)
    todo = [(np.zeros(d), np.ones(d))]
    cells = []
    total = 0.
    n0 = min(s.nu, 256)
    xx = s.ux[:n0]
    yy = s.uy[:n0]/s.scale
    if model is not None:
        yy = yy-model.predict(xx)
    while todo:
        lower, upper = np.asarray(todo).transpose(1, 0, 2)
        half = (upper-lower)*.5
        z = (lower+half)[:, None, :]+half[:, None, :]*nodes[None]
        y = s.ask(z.reshape(-1, d))/s.scale
        if model is not None:
            y -= model.predict(z.reshape(-1, d))
        y = y.reshape(-1, n)
        volume = np.prod(2.*half, axis=1)
        val = (y@w)*volume
        err = np.abs(y@e)*volume
        # Estimate the dominant direction from the two axial differences.
        center = y[:, [0]]
        ax = y[:, 1:1+2*d].reshape(-1, 2, d).sum(axis=1)-2.*center
        bx = y[:, 1+2*d:1+4*d].reshape(-1, 2, d).sum(axis=1)-2.*center
        fluct = np.abs(ax-bx/7.)
        for i, (a, b) in enumerate(todo):
            take = np.all((xx > a) & (xx < b), axis=1)
            inside = yy[take]
            if len(inside):
                mag = np.max(np.abs(inside))
                err[i] = max(err[i], .16*volume[i]*(mag-1.8*np.max(np.abs(y[i]))))
            dim = int(np.argmax(fluct[i] + .0001*np.max(fluct[i])*half[i]))
            cells.append((float(err[i]), dim, a, b, float(val[i])))
        todo = []
        cells.sort(key=lambda row: row[0], reverse=True)
        k = min(3 if len(cells) >= 10 else 1, len(cells), s.left//(2*n))
        if k == 0 or s.time_left() < 1.35:
            break
        if len(cells) > 5 and sum(c[0] for c in cells) < 3.e-12*max(
                math.fsum(abs(c[-1]) for c in cells), 1.e-14):
            break
        for er, j, a, b, vl in cells[:k]:
            am, bm = a.copy(), b.copy()
            am[j] = bm[j] = (a[j]+b[j])*.5
            todo.extend([(a, bm), (am, b)])
        del cells[:k]
    val = math.fsum(c[-1] for c in cells)+(model.mean if model else 0.)
    return val, math.fsum(c[0] for c in cells)
