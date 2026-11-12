"""Likelihood-checked pooling of statistically indistinguishable dynamic modes."""

import numpy as np
from scipy import linalg
from shared import fit_common


def mode_partition(rates, n):
    d = len(rates)
    costs = {}
    for begin in range(d):
        for end in range(begin+1, d+1):
            size = end-begin
            if size == 1:
                costs[begin, end] = (0.0, 1.0, 0.0)
            elif size >= 3:
                segment = rates[begin:end]
                center = segment.mean()
                variance = max(1-center*center, 1e-8)/n
                statistic = np.sum((segment-center)**2)/variance
                degrees = size*(size+1)/2-1
                retained = max(0.0, 1-(degrees-2)/max(statistic, 1e-30))
                costs[begin, end] = (statistic-2*degrees+2, retained, degrees)
    best = [(0.0, [])]+[(np.inf, []) for _ in range(d)]
    for end in range(1, d+1):
        for begin in range(end):
            if (begin, end) not in costs:
                continue
            cost = best[begin][0]+costs[begin, end][0]
            if cost < best[end][0]:
                best[end] = (cost, best[begin][1]+[(begin, end)])
    return best[d][1], costs


def fit_groups(info, rotation, groups):
    d = len(rotation)
    count = info["count"]
    joint = info["joint"]
    c00 = rotation.T@joint[:d, :d]@rotation
    cross = rotation.T@joint[:d, d:2*d]@rotation
    gram = rotation.T@joint[d:2*d, d:2*d]@rotation
    masks = np.zeros((len(groups), d))
    for i, (begin, end) in enumerate(groups):
        masks[i, begin:end] = 1
    rates = (masks@np.diag(cross))/(masks@np.diag(gram))

    def noise(rates):
        a = masks.T@rates
        q = c00-cross*a[None, :]-cross.T*a[:, None]
        q += gram*a[:, None]*a[None, :]
        return (q+q.T)*.5, a

    q, diagonal = noise(rates)
    for _ in range(10):
        metric = linalg.inv(q, check_finite=False)
        r = masks@(metric*gram)@masks.T
        b = masks@np.sum(metric*cross.T, axis=1)
        new = linalg.solve(r, b, assume_a="pos", check_finite=False)
        q, diagonal = noise(new)
        if np.max(np.abs(new-rates)) < 1e-8:
            rates = new
            break
        rates = new
    if np.max(np.abs(rates)) >= 1:
        return None, np.inf
    # The fitted rotations also consume degrees of freedom.
    orientation = d*(d-1)/2
    for begin, end in groups:
        size = end-begin
        orientation -= size*(size-1)/2
    score = count*np.linalg.slogdet(q)[1]
    score += (len(groups)+orientation)*np.log(count)
    gap = 1-diagonal
    h = q/(gap[:, None]*gap[None, :])
    h *= count/(count-1-len(groups)/d)
    result = rotation@h@rotation.T
    return (result+result.T)*.5, score


def grouped_dynamics(original, info, n, fallback, fallback_strength):
    d = len(original)
    if info["order"] != 1:
        return fallback, fallback_strength, {"selected": False}
    w, rotation = linalg.eigh(original, check_finite=False)
    rates = (w-1)/(w+1)
    groups, costs = mode_partition(rates, n)
    pooled = [(a, b) for a, b in groups if b-a >= 3]
    if len(groups) <= 1 or not pooled:
        return fallback, fallback_strength, {"selected": False}
    model, score = fit_groups(info, rotation, groups)
    if model is None:
        return fallback, fallback_strength, {"selected": False}
    _, common_score, _ = fit_common(info, d)
    if score >= common_score-4.0:
        return fallback, fallback_strength, {"selected": False}
    count = info["count"]
    params = info["order"]*d*d
    full_bic = info["score"]-2*params*count/max(count-params-d-1, 1)
    full_bic += params*np.log(count)
    comparison = min(common_score, full_bic)
    evidence = np.exp(-np.logaddexp(0.0, .5*(score-comparison)))
    degrees = sum(costs[g][2] for g in pooled)
    retained = sum(costs[g][1]*costs[g][2] for g in pooled)/degrees
    weight = (1-retained)*evidence
    result = retained*original+(1-retained)*model
    result = evidence*result+(1-evidence)*fallback
    all_degrees = d*(d+1)/2-1
    remaining = 1-degrees/all_degrees*(1-retained*retained)
    strength = 1+(4*d-1)*remaining
    strength = evidence*strength+(1-evidence)*fallback_strength
    return (result+result.T)*.5, strength, {
        "selected": True, "groups": groups, "weight": weight,
        "score_difference": score-info["score"],
    }
