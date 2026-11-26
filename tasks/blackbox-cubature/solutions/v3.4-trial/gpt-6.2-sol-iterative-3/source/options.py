"""Conditional sampling and importance sampling for payoffs with an atom at zero."""

import math
import numpy as np
from scipy import linalg, special

import exercise
import barriers
import line_option
from numerics import (local_gradient, logistic, powers, quadratic_features, quantile, ridge, rotation)


def _gaussian_pdf(x):
    return np.exp(-.5*x*x) / math.sqrt(2.*math.pi)


def _root_probability(coef, offset, d, keep):
    i,j = np.triu_indices(keep,1)
    A = np.diag(coef[d+len(i):]*2.**-.5)
    A[i,j] = A[j,i] = .5*coef[d:d+len(i)]
    ls, q = linalg.eigh(-A,check_finite=False)
    b = -q.T@coef[:d]
    cc = np.trace(A)-offset
    nodes, wt = special.roots_legendre(110)
    r = 5.7*(1.+nodes)
    t = (1j*r)[:,None]
    fac = 1.-2.*t*ls
    cg = -.5*r*r+1j*r*cc+.5*np.sum((t*b)**2/fac-np.log(fac),axis=1)
    return float(.5 + 5.7/math.pi*(wt @ (np.exp(cg).imag/r)))


def digital(w, x0, y0):
    d = w.d
    price = float(np.max(y0))
    cls = y0 > 0.
    logit = logistic(x0, cls, .012)
    direction = logit[1:]
    if np.linalg.norm(direction) < 1.e-6:
        return np.mean(w.query(w.normal(w.remaining)))
    q = rotation(np.eye(d), direction)
    xtr = w.normal(192, d-1)
    ztr = xtr @ q[:, 1:].T
    ep = np.linalg.norm(direction)
    root0 = np.clip(-(logit[0] + ztr @ direction)/ep, -1.7, 1.7)
    s = np.linalg.norm(direction) / max(np.mean(cls)*(1.-np.mean(cls)), .1)
    width = np.clip(.30 + .18 / max(s, .2), .30, .65)
    lo = root0 - 2.8*width
    hi = root0 + 2.8*width
    # The training roots are only a predictor. Invalid brackets need not give
    # correct roots; they are detected before fitting the regression.
    zz = np.concatenate([ztr + lo[:, None]*q[:, 0], ztr + hi[:, None]*q[:, 0]])
    ev = w.query(zz) > 0.
    valid = (~ev[:len(lo)]) & ev[len(lo):]
    mid = root0
    for _ in range(7):
        val = w.query(ztr + mid[:, None]*q[:, 0]) > 0.
        hi = np.where(val, mid, hi)
        lo = np.where(val, lo, mid)
        mid = (lo + hi) * .5
    if np.sum(valid) < len(valid)*.8:
        return _binary_fallback(w, x0, cls, q, price, logit)
    xtr, rt = xtr[valid], mid[valid]
    c1, m1 = ridge(xtr, rt, .055)
    rs = rt - xtr @ c1 - m1
    cov = (xtr.T * rs) @ xtr / len(rt)
    ar = rotation(cov @ cov + .6*np.outer(c1, c1))
    xr = xtr @ ar
    keep = min(7, d-1)
    feats = quadratic_features(xr, width=keep)
    pen = np.r_[np.full(d-1, .065), np.full(feats.shape[1]-d+1, .22)]
    c, m = ridge(feats, rt, pen)
    r1 = rt - feats @ c - m
    sig = max(.065, np.sqrt(np.mean(r1*r1)) * 1.35 + np.std(rs) * .23)
    scale = max(.025, np.mean(_gaussian_pdf(rt)) * sig)
    outer = q[:, 1:] @ ar

    def predict(xx):
        return special.ndtr(quadratic_features(xx, width=keep) @ c + m)

    # Smooth conditional probability control, integrated without oracle calls.
    # A small polynomial calibration is exact in expectation.
    pcoef, pint = ridge(quadratic_features(xr, width=keep),
                       (1.-special.ndtr(rt)) - (1.-predict(xr)), .20)
    pred_mean = _root_probability(c,m,d-1,keep) + pint
    total = 0.
    ns = 0
    n_inner = 6
    for n in powers(w.remaining // n_inner):
        u = w.unit(n, d + 1)
        x = quantile(u[:, :d-1])
        z = x @ outer.T
        pm = predict(x)
        lo = np.zeros(n)
        hi = np.ones(n)
        midp = np.clip(pm,1.e-9,1.-1.e-9)
        for step in range(n_inner-1):
            val = w.query(z + quantile(midp)[:, None] * q[:, 0]) > 0.
            hi = np.where(val, midp, hi)
            lo = np.where(val, lo, midp)
            midpoint = (lo + hi)*.5
            if step < 3:
                delta = scale*[1.35, 3.2, 7.5][step]
                left = np.maximum(lo + 1.e-8, pm - delta)
                right = np.minimum(hi - 1.e-8, pm + delta)
                midp = np.where(hi == 1., right, np.where(lo == 0., left, midpoint))
                midp = np.where((midp <= lo) | (midp >= hi), midpoint, midp)
            else:
                midp = midpoint
        ell = np.maximum(hi-lo, 1.e-15)
        p_focus = .91
        r = np.where(u[:, d-1] < p_focus, lo + ell*u[:, -1], u[:, -1])
        wgt = (1.-p_focus) + p_focus/ell*((r > lo) & (r < hi))
        y = w.query(z + quantile(r)[:, None] * q[:, 0]) > 0.
        residual = (1. - midp) + (y.astype(float) - (r > midp))/wgt
        ctrl = 1. - pm + quadratic_features(x, width=keep) @ pcoef + pint
        total += float(np.sum(residual - ctrl))
        ns += n
    ans = pred_mean + total/max(ns, 1)
    return price * float(np.clip(ans, 0., 1.))


def _binary_fallback(w, x0, y0, q, price, logit):
    t = q[:, 0]
    thresh = -logit[0] / max(np.linalg.norm(logit[1:]), 1.e-8)
    xx = w.normal(w.remaining) @ q.T
    y = w.query(xx) > 0.
    resid = y.astype(float) - (xx @ t > thresh)
    return price * float(np.clip(special.ndtr(-thresh) + resid.mean(), 0., 1.))


def _positive_gradients(w, x, y, count):
    iy = np.flatnonzero(y > 0.)
    if not len(iy):
        return np.empty((0, w.d)), np.empty(0), np.empty((0,w.d))
    jy = iy[np.argsort(y[iy])]
    idx = jy[np.clip((np.linspace(.13,.93,min(len(jy), count))*len(jy)).astype(int), 0, len(jy)-1)]
    g = local_gradient(w, x[idx], y[idx], .0015)
    gscale = np.linalg.norm(g, axis=1)
    gm = np.median(gscale)
    good = (gscale > 1.e-10) & (gscale < 4.*max(gm, 1.e-8))
    return g[good], y[idx][good], x[idx][good]


def _pdf_ratio(z, mu, scales):
    return np.exp(np.clip(.5*np.sum(z*z - ((z-mu)/scales)**2, axis=1)
                          - np.sum(np.log(scales)), -180., 180.))


def _weighted_payoff_data(w, x0, y0):
    d = w.d
    p = np.mean(y0>0.)
    X, Y, S = [x0], [y0], [1.]
    if p < .18:
        for sc in [.52,.72,.88]:
            z = w.normal(384) * sc
            X.append(z)
            Y.append(w.query(z))
            S.append(sc)
    if not np.any(np.concatenate(Y)>0.):
        for sc in [.34,.24,1.30]:
            z = w.normal(384)*sc
            yy = w.query(z)
            X.append(z)
            Y.append(yy)
            S.append(sc)
            if np.count_nonzero(yy)>12:
                break
    return np.concatenate(X), np.concatenate(Y), np.repeat(S, [len(y) for y in Y])


def _option_features(x, active, threshold):
    # Linear/quadratic terms within the high-payoff region, smoothly tapered
    # towards an approximate exercise point. A few t-splines model the tail.
    t = x[:, 0]
    other = x[:, 1:active]
    spread = np.maximum(t-threshold,0.)
    safe = np.clip(t, -4.5, 4.5)
    g = special.expit(5. * (t-threshold))
    base = [t, (t*t-1.)*.70710678, spread, spread*safe, spread*safe*safe,
            np.maximum(t-.6, 0.), np.maximum(t-1.5, 0.), np.maximum(t-2.6, 0.),
            np.maximum(-t-.8, 0.), g]
    a, b = np.triu_indices(other.shape[1], 1)
    return np.column_stack(base + [x[:,1:], g[:,None]*other, spread[:,None]*other,
                         g[:,None]*(other*other-1.), g[:,None]*other[:,a]*other[:,b]])


def _option_mean(width, threshold):
    def tails(t):
        p0, p1 = special.ndtr(-t), _gaussian_pdf(t)
        return np.array([p0,p1,p0+t*p1,(t*t+2.)*p1])
    def hinge(t):
        return _gaussian_pdf(t)-t*special.ndtr(-t)
    h = tails(threshold)
    t = 4.5
    r = tails(t)
    pos = h[1:]-threshold*h[:3]
    htail = r[1:]-threshold*r[:3]
    s = pos - htail + htail[0]*np.array([1.,t,t*t])
    nodes, wt = special.roots_hermitenorm(256)
    g = wt @ special.expit(5.*(nodes-threshold)) / math.sqrt(2.*math.pi)
    res = np.zeros(width)
    res[:10] = [0.,0.,*s,hinge(.6),hinge(1.5),hinge(2.6),hinge(.8),g]
    return res


def continuous(w, x0, y0):
    d = w.d
    x, y, s = _weighted_payoff_data(w, x0, y0)
    rate = np.mean(y0>0.)
    if not np.any(y>0.):
        return float(np.mean(w.query(w.normal(w.remaining))))
    g, yg, xg = _positive_gradients(w, x, y, 18 if d <= 32 else 14)
    raw_grad = g
    # Averages of pathwise gradients identify the terminal-payoff direction;
    # moments of active draws add directions for the exercise/knock-out set.
    valpos = y > 0.
    if len(g):
        g = g / np.maximum(np.linalg.norm(g,axis=1)[:,None],1.e-100)
        gbar = g.mean(axis=0)
        cc = g.T @ g / len(g) - np.outer(gbar,gbar)
    else:
        gbar = np.mean(x[valpos],axis=0)
        cc = np.zeros((d,d))
    gbar /= max(np.linalg.norm(gbar),1.e-8)
    mdl = logistic(x0, y0>0., .065) if rate>.035 else None
    if rate >= .18 and mdl is not None:
        lp = mdl[1:] / max(np.linalg.norm(mdl[1:]), 1.e-8)
        cc += .10*np.outer(lp, lp)
    pos = x[valpos]
    scales = s[valpos]
    pos0 = pos / scales[:, None]
    cov_supp = np.eye(d) - pos0.T @ pos0 / len(pos0)
    vals, vecs = linalg.eigh(cov_supp, check_finite=False)
    vals = np.maximum(vals - .25, 0.)**2
    if rate > .18:
        cc += .013 * (vecs * vals) @ vecs.T
    else:
        cc += .07 * (vecs * vals) @ vecs.T
    q = rotation(cc, gbar)
    xx = x @ q
    xs0 = x0 @ q
    h = np.zeros(d)
    sc = np.ones(d)
    pr_m = 0.
    if rate < .22:
        prop,counts = np.unique(s,return_counts=True)
        logs = .5*np.sum(pos*pos,axis=1)[:, None]*(prop**-2.-1.)
        with np.errstate(divide="ignore"):
            logq = special.logsumexp((-logs-d*np.log(prop))[..., :] + np.log(counts/max(sum(counts),1)),
                                       axis=1)
        wg = y[valpos]**.65 * np.exp(-logq)
        wg = np.minimum(wg, np.quantile(wg,.85)*2.7+1.e-12)
        wg /= max(wg.sum(), 1.e-100)
        xv = xx[valpos]
        m = wg @ xv
        var = wg @ ((xv-m)**2)
        neff = 1./max(np.sum(wg*wg),1.e-12)
        a = np.clip(neff/(neff+24.), .20, .84)
        h = np.clip(m, -1.3, 1.3) * a
        h[7:] *= .30
        h = h * min(1., 1.8/max(np.linalg.norm(h),1.e-10))
        sc = np.sqrt(np.clip(1. + a*(var-1.), .26, 1.12))
        if rate < .05:
            ambient = np.clip(np.median(np.mean(xv**2.,axis=1)), .45, 1.06)
            sc[7:] = np.sqrt(.55 * ambient + .45)
        else:
            sc[7:] = 1.
        pr_m = float(np.clip(.88 - 2.4*rate, .30, .86))

    threshold = logit_thres = 0.
    if mdl is not None:
        logit_thres = -mdl[0] / max(gbar @ mdl[1:], .1)
    threshold = np.clip(logit_thres, -.8, .9)
    act = min(7,d)
    use_ctrl = rate >= .12
    if rate >= .165:
        try_ans, _, _ = exercise.trial(w,q,x0,y0,threshold,(xg@q,yg,raw_grad@q))
        if try_ans is not None:
            return max(0., try_ans)
    if .085 <= rate < .50:
        try_ans = line_option.experiment(w,q,x0,y0,mdl)
        if try_ans is not None:
            return max(0.,try_ans)
    if .075 <= rate < .23 and w.remaining > 6000:
        return max(0.,barriers.estimate(w,x,y,s,q,gbar,h,sc,rate))
    if use_ctrl:
        ntrain = min(1792 if rate<.235 else 1024, w.remaining//6)
        zplus = w.normal(ntrain)
        yplus = w.query(zplus @ q.T)
        a = np.row_stack([xs0,zplus])
        b = np.r_[y0,yplus]
        feat = _option_features(a,act,threshold)
        coef, mu = ridge(feat, b, .012)
        ctrlmean = mu + _option_mean(len(coef),threshold) @ coef

    if pr_m > 0.:
        u = w.unit(w.remaining, d+1)
        v = quantile(u[:, :d])
        z = np.where((u[:, d] < pr_m)[:, None], h + v*sc, v)
        yy = w.query(z @ q.T)
        weight = 1./((1.-pr_m) + pr_m*_pdf_ratio(z,h,sc))
    else:
        z = w.normal(w.remaining)
        tilt = exercise.useful_tilt(a, b-(feat@coef+mu)) if use_ctrl else 0.
        z[:,0] += tilt
        yy = w.query(z @ q.T)
        weight = np.exp(-tilt*z[:,0]+.5*tilt*tilt) if tilt else 1.
    ans = np.mean(yy * weight)
    if use_ctrl:
        ans += ctrlmean - np.mean(weight * (_option_features(z,act,threshold) @ coef + mu))
    return max(0., float(ans))
