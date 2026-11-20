"""Estimate the zero-frequency covariance of a single multivariate trajectory."""

import sys

import numpy as np


def sym(a):
    return (a + a.T) * 0.5


def sqnorm(a):
    return float(np.sum(a * a))


def psd(a, floor=0.0, ceil=np.inf):
    v, q = np.linalg.eigh(sym(a))
    return (q * np.clip(v, floor, ceil)) @ q.T


def invroot(a, rel=1e-11):
    v, q = np.linalg.eigh(sym(a))
    return (q / np.sqrt(np.maximum(v, max(1e-100, v[-1] * rel)))) @ q.T


def whiten(x):
    x = x - x[0]
    x -= x.mean(axis=0)
    scale = np.abs(x).max(axis=0)
    scale[scale == 0] = 1.0
    u, s, v = np.linalg.svd(x / scale, full_matrices=False)
    keep = s > s[0] * 2e-14
    sn = np.sqrt(len(x))
    return np.ascontiguousarray(u[:, keep] * sn), (s[keep, None] / sn) * v[keep] * scale


def finite_chain(x, diagnostics=None, regularization=0.90, risk=0.56, pool=0.85):
    n, d = x.shape
    changed = np.r_[True, np.any(x[1:] != x[:-1], axis=1)]
    jumps = x[changed]
    n_jump = len(jumps)
    if n_jump <= d or len(np.unique(jumps[:256], axis=0)) > 222:
        return None, n_jump
    states, inverse = np.unique(jumps, axis=0, return_inverse=True)
    k = len(states)
    if k > 384 or k > n_jump * 0.65:
        return None, n_jump

    idx = inverse[np.cumsum(changed) - 1]
    raw = np.bincount(idx[:-1] * k + idx[1:], minlength=k * k).reshape(k, k).astype(float)
    from discrete import finite_covariance
    states = states - states[0]
    return finite_covariance(states, raw, regularization, risk, diagnostics, pool=pool), n_jump


def lagcov(x, maxlag):
    n = len(x)
    return np.stack([x[:n - j].T @ x[j:] / (n - j) for j in range(maxlag + 1)])


def lagged_design(x, p):
    n = len(x)
    design = np.concatenate([x[p - j:n - j] for j in range(p + 1)], axis=1)
    design -= design.mean(axis=0)
    return design


def prediction_from_gram(gram, d, p):
    j = d * (p + 1)
    xx = gram[d:j, d:j]
    xy = gram[d:j, :d]
    a = np.linalg.solve(xx, xy)
    return a, sym(gram[:d, :d] - xy.T @ a)


def choose_ar(x, maxlag):
    n, d = x.shape
    design = lagged_design(x, maxlag)
    gram = design.T @ design / len(design)
    scores = [0.0] * (maxlag + 1)
    scores[0] = np.linalg.slogdet(gram[:d, :d])[1]
    for p in range(1, maxlag + 1):
        _, cov = prediction_from_gram(gram, d, p)
        scores[p] = np.linalg.slogdet(cov)[1] + (2.35 * d * d * p) / (n - maxlag)
    return int(np.argmin(scores))


def skew_weight(gram, n, d):
    b0 = sym((gram[:d, :d] + gram[d:, d:]) * 0.5)
    w = invroot(b0)
    b = w @ gram[d:, :d] @ w
    bs = sym(b)
    ident = np.eye(d)
    logdet_un = np.linalg.slogdet(ident - b.T @ b)[1]
    logdet_rv = np.linalg.slogdet(ident - bs @ bs)[1]
    stat = max(0.0, (logdet_rv - logdet_un) * n)
    df = d * (d - 1) / 2
    return max(0.0, 1.0 - (0.8 * df + 1.8) / max(stat, 1e-12)), stat


def estimate_ar(x, p, allow_reversible=True):
    n, d = x.shape
    ident = np.eye(d)
    if p == 0:
        return ident.copy(), x, ident.copy(), 0.0, np.zeros((0, d))
    design = lagged_design(x, p)
    gram = design.T @ design / len(design)
    a, cov = prediction_from_gram(gram, d, p)
    sk = 1.0
    if p == 1 and allow_reversible:
        sk, _ = skew_weight(gram, n - p, d)
        if sk < 1:
            aa = sym(gram[d:, :d])
            vv = sym((gram[d:, d:] + gram[:d, :d]) * 0.5)
            a = (1 - sk) * np.linalg.solve(vv, aa) + sk * a
    e = design[:, :d] - design[:, d:] @ a
    cov = e.T @ e / (len(e) - d * p - 1)
    diff = ident - a.reshape(p, d, d).sum(axis=0)
    transfer = np.linalg.solve(diff, ident)
    h = transfer.T @ cov @ transfer
    return sym(h), e, transfer, sk, a


def moving_average_order(x, p, gammas):
    # A finite-range model is admitted only when several lags after the
    # cutoff look like noise at the scale of the lag-product variance.
    n, d = x.shape
    df = d * d
    norm = np.array([sqnorm(a) for a in gammas])
    near_zero = (p >= 3 and np.linalg.eigvalsh(2*sym(gammas[1]))[0] < -.88)
    for q in range(4):
        if p <= q:
            continue
        inflation = 1 + 2 * norm[1:q + 1].sum() / d
        test = n * norm[q + 1:q + 6] / (df * inflation)
        if np.mean(test) < 1.37 and np.max(test) < 1.95:
            return q
        if q == 1 and near_zero and test[0] < 1.83 and np.mean(test) < 1.59 and np.max(test) < 2.75:
            return q
    return None


def batch_estimate(e, b, prefix=None):
    n = len(e)
    if prefix is None:
        prefix = np.vstack([np.zeros(e.shape[1]), np.cumsum(e, axis=0)])
    avg = prefix[b:] - prefix[:-b]
    avg -= avg.mean(axis=0)
    return avg.T @ avg / (b * len(avg) * (1 - b / n))


def innovation_shape(e):
    n, d = e.shape
    ze, _ = whiten(e)
    if ze.shape[1] < d:
        return 4.0
    r = np.sum(ze*ze, axis=1)
    kurt = np.mean(r*r)/(d*(d+2)) - 1.0
    mean_radial = (r @ ze / n) / np.sqrt(2*d+4)
    z = 0.5*n*sqnorm(mean_radial)
    # The radial fourth moment and skew vector are useful for deciding
    # whether a white Gaussian innovations model is a plausible starting point.
    return max(abs(kurt)*np.sqrt(n*d*(d+2)/8), np.sqrt(z / max(1,d)))


def weak_ar(x, gammas, ar_order, a, diagnostics=None):
    n, d = x.shape
    eye = np.eye(d)
    g = sym(gammas[1])
    if ar_order > 1 or np.linalg.norm(g, 2) > 0.25 or np.linalg.norm(gammas[1], 2) > .30:
        return None
    signal = n*sqnorm(g)
    df = d*(d+1)/2
    offset = 0.82 if ar_order == 1 else 1.03
    amount = max(0., 1.0 - max(1.0,(offset*df-0.4))/max(signal,1.e-12))
    a = amount * g + ((a-a.T)*.5 if ar_order==1 else 0.)
    e = x[1:] - x[:-1] @ a
    e -= e.mean(axis=0)
    r = e.T @ e / (len(e) - amount*d - 1)
    t = np.linalg.solve(eye-a,eye)
    if diagnostics is not None:
        diagnostics["weak_lag_weight"] = amount
    return sym(t.T @ r @ t), e, t


def risk_scaling(h, n, factor, info=None, pool=0.0):
    val, vec = np.linalg.eigh(sym(h))
    val = np.maximum(0.0,val)
    d = len(val)
    if info is None:
        noise = ((1-.48*pool)*(d+2)*val + (1-.55*pool)*val.sum())/n
        h = (vec * (val/(1 + factor*np.minimum(2.0, noise)))) @ vec.T
    else:
        root = vec * np.sqrt(val)
        k = sym(root.T @ info @ root) / n
        noise = (d+2)*k+np.trace(k)*np.eye(d)
        u, q = np.linalg.eigh(noise)
        h = root @ ((q / (1 + factor*np.clip(u,0.,2.0))) @ q.T) @ root.T
    return sym(h)


def adaptive_whitened(x, repeat, diagnostics=None, arma=True, mh=True, use_poles=True,
                      coord=True, project=True, tuned=True, radial=True, anisotropic=True):
    n, d = x.shape
    eye = np.eye(d)
    pmax = min(8, max(1, n // (15 * d)), n-2)
    p = choose_ar(x, pmax)
    gamma = lagcov(x, min(9, n-1))
    q = moving_average_order(x, p, gamma) if n >= 30 and p > 0 and repeat < 0.02 else None
    if q == 1 and arma:
        try:
            from ma1 import estimate_ma1
            result = estimate_ma1(x, gamma[1])
            if diagnostics is not None:
                diagnostics.update(model="MA", order=1)
            return psd(result)
        except (ImportError, np.linalg.LinAlgError, FloatingPointError):
            pass
    dx = np.diff(x, axis=0)
    if repeat < .018 and p>=1:
        from clustered import noisy_states
        try:
            discrete = noisy_states(x,dx,diagnostics)
        except (np.linalg.LinAlgError, ValueError):
            discrete = None
        if discrete is not None:
            return psd(discrete)
    coordinate_model = None
    if coord and repeat < .018 and n >= 750 and p >= 1:
        from coordinate import coordinate_covariance
        try:
            coordinate_model = coordinate_covariance(x, dx, diagnostics)
        except (np.linalg.LinAlgError, ValueError):
            coordinate_model = None
        if coordinate_model is not None and coordinate_model[0] == "periodic":
            _, period, (reduced, model) = coordinate_model
            z, f = whiten(reduced)
            if z.shape[1] == d:
                hh = period*f.T @ adaptive_whitened(z,.0,arma=arma,coord=False) @ f
                if model is not None:
                    b, strength, _ = model
                    w = .87*strength
                    hh = (1-w)*hh+w*risk_scaling(b,n,.53)
                return psd(hh)
            coordinate_model = None
    h, e, transfer, sk, a = estimate_ar(x, p)
    if diagnostics is not None:
        diagnostics.update(model="AR", order=p, skew=sk)
    model_fit = False
    model_lag = p
    non_gauss = (innovation_shape(e) if repeat < .02 and p >= 1 and
                 (p >= 2 or np.linalg.norm(gamma[1],2)>.23) else 0.)
    if not model_fit and p >= 2 and repeat < .02 and q != 0 and arma and (
            not tuned or q is not None or non_gauss < 7.0 or (p>=4 and non_gauss<14.0)):
        try:
            from arma import try_arma
            candidate = try_arma(x, p, q)
        except (OSError, ImportError, np.linalg.LinAlgError, FloatingPointError):
            candidate = None
        if candidate is not None:
            ap, aq, h, e, transfer = candidate
            model_fit = True
            model_lag = ap
            if diagnostics is not None:
                diagnostics.update(model="ARMA", order=(ap, aq))
    if q is not None and not model_fit:
        s = eye + 2*sym(gamma[1:q+1].sum(axis=0))
        if np.linalg.eigvalsh(s)[0] > .15:
            if diagnostics is not None:
                diagnostics.update(model="MA", order=q)
            return s
    if p <= 1 and repeat < .015:
        weak = weak_ar(x, gamma, p, a, diagnostics)
        if weak is not None:
            h, e, transfer = weak

    direct = dx.T @ dx / (2*len(dx))
    vals, vecs = np.linalg.eigh(sym(direct))
    span = 1 / max(1e-4, vals[0])
    fast_hold = p >= 2 and .024 < repeat < .73 and span < 4.2
    amt = 0.0
    pool = 0.0
    wants_nonlinear = repeat > .024 or (p <= 3 and not model_fit and np.linalg.norm(gamma[1],2)>.23
                                         and non_gauss>1.9)
    if wants_nonlinear and p >= 1 and not model_fit:
        des = lagged_design(x,1)
        _, stat0 = skew_weight(des.T @ des / len(des), n-1, d)
        df = max(1,d*(d-1)/2)
        reversible = stat0 < (4.5 if repeat > .024 else 2.05)*df + 5.
        if reversible and .00015 < vals[0] and vals[-1] < (1.35 if p>1 else 1.84):
            from reversible import covariance_rates, semigroup
            pool_amount = .93 if repeat > .024 else (.22 if span > 6 else .03)
            cand = semigroup(x, power=(0.66 if tuned and not fast_hold else 0.80) if repeat > .024 else .78,
                             extra=repeat>.024, diagnostics=diagnostics,
                             directional=(.24 if fast_hold else .64) if tuned else 0.0)
            if cand is not None:
                hr, amt = cand
                c = eye.copy()
                if repeat > .09:
                    from holding import weighted_covariance
                    c = weighted_covariance(x, .77 if repeat > .52 else .55, diagnostics)
                cv, cq = np.linalg.eigh(c)
                cp = (cq * np.sqrt(cv)) @ cq.T
                ci = (cq / np.sqrt(cv)) @ cq.T
                dr = sym(ci @ direct @ ci)
                dv, du = np.linalg.eigh(dr)
                pool_ratio, pool = covariance_rates(n, dv, pool_amount)
                c = cp @ ((du * pool_ratio) @ du.T) @ cp
                hlin = 2*c @ np.linalg.solve(direct,c) - c
                rawlin = 2*np.linalg.solve(direct,eye) - eye
                base_g = min(1.0, 1.0 - .31*pool)
                hr = hlin + base_g*(hr-rawlin)
                if p == 1:
                    b = .93 if repeat > .024 else .64
                else:
                    b = (.59 if repeat > .024 else .48)*min(1.0, amt/.22)
                if b > 0:
                    h = sym((1-b)*h + b*hr)
                    if diagnostics is not None:
                        diagnostics.update(reversible_weight=b, pooled_rates=pool)
    elif not model_fit and p==1 and sk < .32 and 4.0 < span and vals[-1] < 1.25:
        from reversible import covariance_rates
        rr, pool = covariance_rates(n, vals, .38)
        h += (vecs * (2*(rr**2-1)/vals - (rr-1))) @ vecs.T
        if diagnostics is not None:
            diagnostics["pooled_rates"] = pool

    if project and repeat < .02 and non_gauss > 4.5 and p >= 1:
        des = lagged_design(x,1)
        _, stat0 = skew_weight(des.T @ des / len(des), n-1, d)
        confident = stat0 > 2.1*(d*(d-1)/2)+6
        from projection import projected
        result = projected(x,diagnostics) if confident and vals[-1] < 1.45 else None
        if result is not None:
            hh, baseline, signal = result
            b = min(0.75, 3.5*signal)
            if p == 1:
                h += b*(hh-baseline)
            else:
                h = (1-b*.70)*h+b*.70*hh

    if use_poles and (p >= 2 or repeat > .025):
        from spectral import pole_adjust
        if not model_fit or q is None:
            extra = (1.6 if p <= 4 and non_gauss < 5.0 else .72) if repeat < .02 else (0.12 if fast_hold else 1.2)
            dgp = {} if tuned else diagnostics
            dep = pole_adjust(e, span=span, cautious=(repeat < .02), diagnostics=dgp,
                              guard=extra if tuned else 0.0)
            adjustment = transfer.T @ dep @ transfer
            level = .72 if repeat < .025 else (.42 if amt > .16 else .63)
            if tuned:
                if diagnostics is not None:
                    diagnostics.update(dgp)
                signal = dgp.get("pole_evidence", 0.0)
                if repeat > .024 and not fast_hold:
                    level *= min(1.0, max(0.0, signal-3)/9.0)
                elif p <= 4 and non_gauss < 4.5:
                    level *= .6
            # Conserve the parametric estimate unless departure is well supported.
            h += level * adjustment

    if model_lag > 0:
        info = None
        if not model_fit and p > 1:
            past = lagged_design(x,p)[:,d:]
            j = np.tile(eye, (p,1))
            info = sym(j.T @ np.linalg.solve(past.T @ past / len(past), j))
        h = risk_scaling(h, n, .74 if not model_fit else .57, info, pool)
    if coordinate_model is not None:
        _, weight, b = coordinate_model
        alpha = .87*weight
        h = (1-alpha)*h + alpha*risk_scaling(b,n,.54)
    if repeat > .26 and p >= 1:
        from holding import regenerative_covariance
        hr, rb = regenerative_covariance(x, diagnostics)
        if hr is not None:
            h = sym((1-rb)*h + rb*hr)
    if mh and repeat > .068 and repeat < .965 and p >= 1 and vals[-1] < .9:
        from metropolis import common_step_covariance, metropolis_covariance
        try:
            hh, evidence = common_step_covariance(x, direct, diagnostics, use_radial=radial)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            hh, evidence = None, 0.0
        if hh is not None:
            f = .83*evidence
            h = sym((1-f)*h + f*hh)
        if anisotropic and (hh is None or evidence < .22) and .08 < repeat < .88 and span >= 6.0:
            try:
                h2, evi = metropolis_covariance(x, diagnostics)
                if h2 is not None:
                    j = .59*evi
                    h = sym((1-j)*h + j*h2)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                pass
    return psd(h)


def adaptive(x, diagnostics=None, **kwargs):
    n, d = x.shape
    if n < 5 or not np.isfinite(x).all():
        return np.zeros((d,d))
    h, n_jump = finite_chain(x, diagnostics)
    if h is not None:
        return h
    z, f = whiten(x)
    if not z.shape[1]:
        return np.zeros((d,d))
    h = adaptive_whitened(z, 1-n_jump/n, diagnostics, **kwargs)
    return sym(f.T @ h @ f)


estimate = adaptive


def fallback(x):
    z, f = whiten(x)
    if not z.shape[1]:
        return np.zeros((x.shape[1],x.shape[1]))
    b = max(2, min(len(z) // 8, int(len(z) ** 0.5)))
    return sym(f.T @ batch_estimate(z, b) @ f)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: run.sh INPUT OUTPUT")
    x = np.load(sys.argv[1], allow_pickle=False)
    if x.ndim != 2:
        raise ValueError("the input must be an M by d array")
    if len(x) < 5 or not np.isfinite(x).all():
        result = np.zeros((x.shape[1],x.shape[1]))
    else:
        _, shift = np.frexp(np.max(np.abs(x)))
        shift = int(shift) if abs(shift) > 330 else 0
        if shift:
            x = np.ldexp(x,-shift)
        try:
            result = adaptive(x)
            if not np.isfinite(result).all():
                result = fallback(x)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            result = fallback(x)
        if shift:
            _, expo = np.frexp(np.max(np.abs(result)))
            result = np.ldexp(result,min(2*shift,1023-int(expo)))
    with open(sys.argv[2], "wb") as stream:
        np.save(stream, np.ascontiguousarray(result, dtype="<f8"), allow_pickle=False)


if __name__ == "__main__":
    main()
