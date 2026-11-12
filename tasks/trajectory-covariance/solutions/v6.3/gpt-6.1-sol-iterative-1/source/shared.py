"""Fit and softly select common autoregressive dynamics across coordinates."""

import numpy as np
from scipy import linalg
from regularize import isotropic_shrink


def residual_covariance(c00, cross, gram, coef):
    p = len(coef)
    linear = np.einsum("p,apb->ab", coef, cross[:, :p, :])
    quadratic = np.einsum("p,q,paqb->ab", coef, coef, gram[:p, :, :p, :])
    q = c00-linear-linear.T+quadratic
    return (q+q.T)*.5


def fit_common(info, d):
    if "common_fit" in info:
        return info["common_fit"]
    joint = info["joint"]
    count = info["count"]
    order = info["start"]
    c00 = joint[:d, :d]
    cross = joint[:d, d:].reshape(d, order, d)
    gram = joint[d:, d:].reshape(order, d, order, d)
    # Share temporal coefficients, but retain a full innovation covariance.
    r = np.einsum("iaja->ij", gram).copy()
    r.flat[::order+1] += 1e-10
    b = np.einsum("aia->i", cross)
    factor = linalg.cholesky(r, lower=True, check_finite=False)
    best_score = count*np.linalg.slogdet(c00)[1]
    best = (np.empty(0), c00)
    for p in range(1, order+1):
        coef = linalg.cho_solve(
            (factor[:p, :p], True), b[:p], check_finite=False
        )
        q = residual_covariance(c00, cross, gram, coef)
        sign, logdet = np.linalg.slogdet(q)
        if sign <= 0:
            continue
        score = count*logdet+p*np.log(count)
        if score < best_score:
            best_score = score
            best = (coef, q)
    coef, q = best
    p = len(coef)
    if p:
        for _ in range(8):
            metric = linalg.inv(q, check_finite=False)
            rp = np.einsum("ba,iajb->ij", metric, gram[:p, :, :p, :])
            bp = np.einsum("ba,aib->i", metric, cross[:, :p, :])
            new = linalg.solve(rp, bp, assume_a="pos", check_finite=False)
            q = residual_covariance(c00, cross, gram, new)
            if np.max(np.abs(new-coef)) < 1e-8:
                coef = new
                break
            coef = new
        best_score = count*np.linalg.slogdet(q)[1]+p*np.log(count)
        companion = np.zeros((p, p))
        companion[0] = coef
        if p > 1:
            companion[1:, :-1] = np.eye(p-1)
        if np.max(np.abs(linalg.eigvals(companion, check_finite=False))) >= 1:
            result = (None, np.inf, p)
            info["common_fit"] = result
            return result
    gap = 1-coef.sum()
    result = q/(gap*gap)
    result *= count/max(1.0, count-1-p/d)
    fit = ((result+result.T)*.5, best_score, p)
    info["common_fit"] = fit
    return fit


def pooled_dynamics(h, info, n):
    shaped, retained = isotropic_shrink(h, n)
    base_strength = 4.0*len(h)
    if retained >= 1:
        return h, 1.0, {"selected": False, "strength": base_strength}
    common, score, order = fit_common(info, len(h))
    if common is None:
        return shaped, retained, {
            "selected": False, "strength": base_strength*(.25+.75*retained)
        }
    evidence = np.exp(-np.logaddexp(0.0, .5*(score-info["score"])))
    weight = evidence
    result = evidence*common+(1-evidence)*shaped
    strength = evidence
    strength += (1-evidence)*base_strength*(.25+.75*retained)
    return (result+result.T)*.5, 1-weight, {
        "selected": True, "weight": weight, "order": order,
        "score_difference": score-info["score"], "strength": strength,
    }
