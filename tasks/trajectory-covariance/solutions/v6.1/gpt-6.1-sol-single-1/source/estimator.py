#!/usr/bin/env python3
"""Estimate the long-run covariance of a single vector-valued trajectory."""

import math
import sys
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, solve
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter
from scipy.special import chdtri


def sym(a):
    return (a + a.T) * 0.5


def positive(a, floor=0.0):
    values, vectors = np.linalg.eigh(sym(a))
    values = np.maximum(values, floor)
    return (vectors * values) @ vectors.T


def roots(a):
    values, vectors = np.linalg.eigh(sym(a))
    values = np.maximum(values, max(1e-300, values[-1] * 1e-14))
    root = (vectors * np.sqrt(values)) @ vectors.T
    inverse = (vectors / np.sqrt(values)) @ vectors.T
    return root, inverse


def whiten(x):
    # Subtract a nearby origin first, so a large translation does not spoil
    # summation of the mean. SVD avoids squaring the condition number.
    z = x - x[0]
    scale = np.max(np.abs(z), axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    z = z / scale
    z -= z.mean(axis=0)
    u, singular, vt = np.linalg.svd(z, full_matrices=False)
    keep = singular > max(1e-300, singular[0] * 2e-14)
    y = np.ascontiguousarray(u[:, keep] * math.sqrt(len(x)))
    back = (singular[keep, None] * vt[keep]) * scale / math.sqrt(len(x))
    return y, back


def batch_fallback(y):
    n, d = y.shape
    if n < 2:
        return np.zeros((d, d))
    b = min(n, max(1, round(math.sqrt(n))))
    cs = np.vstack((np.zeros((1, d)), np.cumsum(y, axis=0)))
    means = (cs[b:] - cs[:-b]) / math.sqrt(b)
    return means.T @ means / len(means)


def lag_covariances(y, p):
    n = len(y)
    return np.array([y[k:].T @ y[:n-k] / n for k in range(p + 1)])


def regression_statistics(y, p, cov=None):
    """Exact centered lagged OLS moments, without a large lagged design."""
    n, d = y.shape
    if cov is None:
        cov = lag_covariances(y, p)
    g = np.empty((p*d, p*d))
    cross = np.empty((d, p*d))
    for i in range(p):
        cross[:, i*d:(i+1)*d] = (
            n*cov[i+1] - y[i+1:p].T @ y[:p-i-1]
        )
        for j in range(i, p):
            lag = j-i
            block = (
                n*cov[lag]
                - y[lag:p-1-i].T @ y[:p-1-j]
                - y[n-1-i:].T @ y[n-1-j:n-j+i]
            )
            g[i*d:(i+1)*d, j*d:(j+1)*d] = block
            g[j*d:(j+1)*d, i*d:(i+1)*d] = block.T
    sy = y[p:].sum(axis=0)
    sx = np.concatenate([y[p-k:n-k].sum(axis=0) for k in range(1, p+1)])
    g -= np.outer(sx, sx) / (n-p)
    cross -= np.outer(sy, sx) / (n-p)
    v = n*cov[0] - y[:p].T @ y[:p] - np.outer(sy, sy)/(n-p)
    return sym(g)/(n-p), cross/(n-p), sym(v)/(n-p)


def pos_solve(a, b):
    return solve(a, b, assume_a="pos", check_finite=False)


def order_weights(y):
    n, d = y.shape
    maxp = min(64, max(8, int(5*math.log(max(n, 2)))), n//(20*d))
    if maxp < 1:
        return np.array([1.0])
    cov = lag_covariances(y, maxp)
    g, cross, v = regression_statistics(y, maxp, cov)
    nfit = n-maxp
    scores = []
    for p in range(maxp+1):
        q = p*d
        if p:
            a = pos_solve(g[:q, :q], cross[:, :q].T).T
            omega = sym(v - a @ cross[:, :q].T)
        else:
            omega = v
        sign, logdet = np.linalg.slogdet(omega)
        if sign <= 0:
            scores.append(np.inf)
            continue
        # Small-sample multivariate regression correction. All orders use
        # exactly the same response rows when their scores are compared.
        regressors = q+1
        correction = (
            nfit*d*(2*regressors+d+1)/(nfit-regressors-d-1)
        )
        scores.append(nfit*logdet + correction)
    scores = np.asarray(scores)
    weights = np.exp(-0.5*np.minimum(scores-np.min(scores), 100.0))
    weights[weights < 1e-4] = 0.0
    return weights / weights.sum()


def fit_moments(g, cross, v, n, d):
    """Fit a VAR and compute its covariance and Gaussian delta-method risk."""
    p = g.shape[0]//d
    a = pos_solve(g, cross.T).T
    omega = sym(v - a @ cross.T)
    ar = a
    pooled = 0.0

    if p == 1 and d > 1:
        def objective(rho):
            om = sym(v - rho*(cross+cross.T) + rho*rho*g)
            sign, logdet = np.linalg.slogdet(om)
            return logdet if sign > 0 else 1e100

        rho = minimize_scalar(
            objective, method="bounded", bounds=(-0.999999, 0.999999),
            options={"xatol": 2e-9},
        ).x
        difference = a - rho*np.eye(d)
        distance = n*np.trace(pos_solve(omega, difference @ g @ difference.T))
        pooled = min(1.0, 1.5*max(1, d*d-3)/max(distance, 1e-30))
        ar = a + pooled*(rho*np.eye(d)-a)

    residual = sym(v - ar @ cross.T - cross @ ar.T + ar @ g @ ar.T)
    transfer = np.linalg.inv(np.eye(d)-ar.reshape(d, p, d).sum(axis=1))
    h = sym(transfer @ residual @ transfer.T)
    root, inverse = roots(h)

    ones = np.tile(np.eye(d), (p, 1))
    q = root @ (ones.T @ pos_solve(g, ones)) @ root / n
    left = sym(inverse @ transfer @ omega @ transfer.T @ inverse)
    risk = (1-pooled)**2 * (
        left @ q + q @ left + np.trace(q)*left + np.trace(left)*q
    )
    risk += (left @ left + np.trace(left)*left) / n
    if pooled:
        information = n*np.trace(pos_solve(omega, g))
        derivative = inverse @ transfer @ root
        derivative += derivative.T
        risk += (2*pooled-pooled*pooled)/information * (derivative @ derivative)
    return h, root, inverse, sym(risk), ar, transfer


def centered_moments(gg, cross, vv, sx, sy, n):
    g = (gg-np.outer(sx, sx)/n)/n
    c = (cross-np.outer(sy, sx)/n)/n
    v = (vv-np.outer(sy, sy)/n)/n
    return sym(g), c, sym(v)


def kurtosis_weight(z):
    n, d = z.shape
    c = sym(z.T @ z / n)
    _, inverse = roots(c)
    zz = z @ inverse
    kurtosis = np.mean(np.sum(zz*zz, axis=1)**2)/(d*(d+2))
    return float(np.clip((kurtosis-1.05)/0.5, 0.0, 1.0))


def residual_correction(z, inverse):
    """Conservative flat-top check on a prewhitened influence series."""
    zz = np.ascontiguousarray(z @ inverse)
    n, d = zz.shape
    energy = np.sum(zz*zz, axis=1)
    best = None
    best_stat = 3.0
    for bandwidth in (4, 8, 16, 32, 64, 128, 256):
        if bandwidth > min(256, n//24):
            break
        lags = np.arange(1, bandwidth+1)
        weights = np.minimum(1.0, 2*(1-lags/(bandwidth+1)))
        past = lfilter(np.r_[0.0, weights], [1.0], zz, axis=0)
        dot = np.sum(zz*past, axis=1)
        past_energy = np.sum(past*past, axis=1)
        noise = 2*np.sum(energy*past_energy + dot*dot)/(n*n)
        center_factor = 1-(1+2*np.sum(weights*(1-lags/n)))/n
        delta = (np.eye(d)+sym(zz.T @ past)*2/n)/center_factor-np.eye(d)
        noise /= center_factor*center_factor
        stat = np.sum(delta*delta)/max(noise, 1e-30)
        if stat > best_stat:
            best_stat = stat
            best = (delta, noise, past, dot, past_energy, center_factor)
    if best is None:
        return np.eye(d), np.zeros((d, d))
    delta, noise, past, dot, past_energy, center_factor = best
    fraction = 1-3.0/best_stat
    noise_matrix = (
        zz.T @ (past_energy[:, None]*zz)
        + past.T @ (energy[:, None]*past)
        + zz.T @ (dot[:, None]*past)
        + past.T @ (dot[:, None]*zz)
    ) / ((n*center_factor)**2)
    correction = positive(np.eye(d)+fraction*delta, floor=0.03)
    return correction, sym(fraction*fraction*noise_matrix)


def shrink_covariance(root, risk, correction=None):
    values, vectors = np.linalg.eigh(sym(risk))
    values = np.maximum(values, 0.0)
    gain = (vectors / np.sqrt(1+values)) @ vectors.T
    if correction is None:
        return sym(root @ gain @ gain @ root)
    return sym(root @ gain @ correction @ gain @ root)


def fit_continuous_order(y, p, check_residual=False):
    n, d = y.shape
    blocks = min(20, max(8, n//100))
    if p == 0:
        yy = y
        xx = np.empty((n, 0))
    else:
        yy = y[p:]
        xx = np.column_stack([y[p-j-1:n-j-1] for j in range(p)])
    nfit = len(yy)
    gg, cross, vv = xx.T @ xx, yy.T @ xx, yy.T @ yy
    sx, sy = xx.sum(axis=0), yy.sum(axis=0)
    if p:
        g, c, v = centered_moments(gg, cross, vv, sx, sy, nfit)
        h, root, inverse, gaussian_risk, ar, transfer = fit_moments(
            g, c, v, nfit, d
        )
        residual = yy - xx @ ar.T
        residual -= residual.mean(axis=0)
        influence = residual @ transfer.T
    else:
        h = sym(vv/nfit)
        root, inverse = roots(h)
        gaussian_risk = (d+1)*np.eye(d)/nfit
        residual = yy
        influence = yy

    non_gaussian = kurtosis_weight(residual)
    empirical_risk = gaussian_risk
    if non_gaussian > 0.0:
        jackknife = []
        for indices in np.array_split(np.arange(nfit), blocks):
            xb, yb = xx[indices], yy[indices]
            nt = nfit-len(indices)
            stx, sty = sx-xb.sum(axis=0), sy-yb.sum(axis=0)
            vj = vv-yb.T @ yb
            if p:
                gj, cj, vj = centered_moments(
                    gg-xb.T @ xb, cross-yb.T @ xb, vj, stx, sty, nt
                )
                try:
                    hj = fit_moments(gj, cj, vj, nt, d)[0]
                except (np.linalg.LinAlgError, ValueError):
                    continue
            else:
                hj = sym((vj-np.outer(sty, sty)/nt)/nt)
            if np.isfinite(hj).all():
                jackknife.append(hj)
        if len(jackknife) >= blocks//2:
            jackknife = np.asarray(jackknife)
            centered = inverse @ (jackknife-jackknife.mean(axis=0)) @ inverse
            empirical_risk = (
                (len(jackknife)-1)/len(jackknife)
                * np.einsum("kij,kjl->il", centered, centered)
            )
            empirical_risk += positive(0.25*gaussian_risk-empirical_risk)
    risk = (
        (1-non_gaussian)*gaussian_risk + non_gaussian*empirical_risk
    ) * (1+0.5*non_gaussian)
    if check_residual:
        correction, extra_risk = residual_correction(influence, inverse)
        return shrink_covariance(root, risk+extra_risk, correction)
    return shrink_covariance(root, risk)


def continuous_estimate(y):
    weights = order_weights(y)
    primary = int(np.argmax(weights))
    result = np.zeros((y.shape[1], y.shape[1]))
    used = 0.0
    for p in np.flatnonzero(weights):
        try:
            h = fit_continuous_order(y, int(p), check_residual=(p == primary))
        except (np.linalg.LinAlgError, ValueError):
            continue
        if np.isfinite(h).all():
            result += weights[p]*h
            used += weights[p]
    result = result/used if used > 0 else batch_fallback(y)
    try:
        from moving_average import improve_estimate
        return improve_estimate(y, result, primary, regression_statistics)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return result


def finite_labels(x):
    _, first, ids, counts = np.unique(
        x, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    k = len(counts)
    if k < 2 or k > min(512, len(x)//10):
        return None
    runs = 1+np.count_nonzero(ids[1:] != ids[:-1])
    if (runs-k)/max(1, runs) < 0.5:
        return None
    return first, ids


def deviance_corrections():
    corrections = np.zeros(33)
    for m in range(1, 33):
        expectation = 0.0
        for j in range(m+1):
            deviance = 0.0
            if j:
                deviance += j*math.log(2*j/m)
            if j < m:
                deviance += (m-j)*math.log(2*(m-j)/m)
            expectation += math.comb(m, j)*2.0**(-m)*2*deviance
        corrections[m] = max(0.0, expectation-1)
    return corrections


def reversible_counts(c):
    average = (c+c.T)*0.5
    nonzero = c > 0
    deviance = 2*np.sum(c[nonzero]*np.log(c[nonzero]/average[nonzero]))
    masses = np.triu(c+c.T, 1)
    masses = masses[masses > 0]
    df = max(1, len(masses)-len(c)+1)
    table = deviance_corrections()
    small = masses <= 32
    correction = table[masses[small].astype(int)].sum()
    correction += np.sum(0.5/masses[~small])
    return deviance <= chdtri(df, 0.01)+correction


def markov_covariance(counts, values, reversible, jackknife=False):
    if reversible:
        counts = sym(counts)
    visits = counts.sum(axis=1)
    if reversible:
        keep = visits > 0
        counts = counts[np.ix_(keep, keep)]
        values = values[keep]
        visits = counts.sum(axis=1)
    k = len(visits)
    pi = visits/visits.sum()
    p = counts/np.maximum(visits, 1e-300)[:, None]
    if not reversible:
        zero = visits == 0
        if zero.any() and not jackknife:
            raise np.linalg.LinAlgError("unobserved outgoing transition row")
        p[zero] = pi
        if jackknife:
            p = (1-1e-7)*p+1e-7*pi[None, :]
        pi = solve(
            np.eye(k)-p.T+np.ones((k, k))/k,
            np.ones(k)/k, check_finite=False,
        )
        if not jackknife and (not np.isfinite(pi).all() or pi.min() < 1e-12):
            raise np.linalg.LinAlgError("degenerate empirical stationary law")
        pi = np.maximum(pi, 0.0)
        pi /= pi.sum()
    f = values-pi @ values
    poisson = solve(
        np.eye(k)-p+pi[None, :], f, check_finite=False
    )
    product = f.T @ (pi[:, None]*poisson)
    return sym(product+product.T-f.T @ (pi[:, None]*f))


def finite_estimate(y, first, ids):
    n, d = y.shape
    k = len(first)
    values = y[first]
    counts = np.bincount(
        ids[:-1]*k+ids[1:], minlength=k*k
    ).reshape(k, k).astype(float)
    reversible = reversible_counts(counts)
    transitions = n-1
    try:
        h = positive(markov_covariance(counts, values, reversible))
    except np.linalg.LinAlgError:
        counts[ids[-1], ids[0]] += 1
        transitions = n
        h = positive(markov_covariance(counts, values, reversible))
    root, inverse = roots(h)
    floor = (d+4)*h/n
    blocks = 20 if k <= 128 else 8
    jackknife = []
    for indices in np.array_split(np.arange(transitions), blocks):
        removed = np.bincount(
            ids[indices]*k+ids[(indices+1) % n], minlength=k*k
        ).reshape(k, k)
        try:
            hj = markov_covariance(
                counts-removed, values, reversible, jackknife=True
            )
        except (np.linalg.LinAlgError, ValueError):
            continue
        if np.isfinite(hj).all():
            # A disconnected leave-block fit is not an infinite amount of
            # information about the full-chain estimate.
            relative = inverse @ hj @ inverse
            ev, u = np.linalg.eigh(sym(relative))
            relative = (u*np.clip(ev, 0.0, 20.0)) @ u.T
            jackknife.append(relative)
    risk = floor
    if len(jackknife) >= blocks//2:
        jackknife = np.asarray(jackknife)
        centered = jackknife-jackknife.mean(axis=0)
        empirical = (len(jackknife)-1)/len(jackknife) * np.einsum(
            "kij,kjl->il", centered, centered
        )
        risk = floor+0.5*positive(empirical-floor)
    return shrink_covariance(root, risk)


def estimate(x):
    y, back = whiten(x)
    if y.shape[1] == 0:
        return np.zeros((x.shape[1], x.shape[1]))
    labels = finite_labels(x)
    h = None
    if labels is not None:
        try:
            h = finite_estimate(y, *labels)
        except (np.linalg.LinAlgError, ValueError):
            pass
    if h is None or not np.isfinite(h).all():
        try:
            h = continuous_estimate(y)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            h = batch_fallback(y)
    h = positive(h)
    answer = sym(back.T @ h @ back)
    if not np.isfinite(answer).all():
        answer = sym(back.T @ batch_fallback(y) @ back)
    return np.ascontiguousarray(answer, dtype="<f8")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: run.sh INPUT.npy OUTPUT.npy")
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    x = np.load(sys.argv[1], allow_pickle=False)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("expected a nonempty finite trajectory matrix")
    result = estimate(np.asarray(x, dtype=np.float64))
    with open(sys.argv[2], "wb") as output:
        np.save(output, result, allow_pickle=False)


if __name__ == "__main__":
    main()
