"""Structured shrinkage targets for a vector autoregression."""

import numpy as np


def target_for(a, precision, covariance, kind, groups=None):
    d = len(a)
    if kind == "joint":
        rho = np.trace(precision@a@covariance)/np.trace(precision@covariance)
        return rho*np.eye(d), d*d-1, np.array([rho])
    if kind == "polar":
        u, _, vt = np.linalg.svd(a)
        rotation = u@vt
        rho = (
            np.trace(rotation.T@precision@a@covariance)
            / np.trace(rotation.T@precision@rotation@covariance)
        )
        return rho*rotation, d*(d+1)//2-1, np.array([rho])
    eig, vec = np.linalg.eig(a)
    order = np.argsort(eig.real)
    vec = vec[:, order]
    if np.linalg.cond(vec) > 1e5:
        raise np.linalg.LinAlgError("Poorly separated eigenspaces")
    inverse = np.linalg.inv(vec)
    projections = np.array([vec[:, lo:hi]@inverse[lo:hi] for lo, hi in groups])
    if np.max(np.abs(projections.imag)) > 1e-7:
        raise np.linalg.LinAlgError("Split a conjugate pair")
    projections = projections.real
    gram = np.empty((len(groups), len(groups)))
    rhs = np.empty(len(groups))
    for i, pi in enumerate(projections):
        rhs[i] = np.trace(pi.T@precision@a@covariance)
        for j, pj in enumerate(projections):
            gram[i, j] = np.trace(pi.T@precision@pj@covariance)
    values = np.linalg.solve(gram, rhs)
    target = np.einsum("i,ijk->jk", values, projections)
    removed = sum((hi-lo)**2-1 for lo, hi in groups)
    return target, removed, values


def select_target(a, innovation, covariance, n):
    d = len(a)
    precision = np.linalg.inv(innovation)
    target, removed, _ = target_for(a, precision, covariance, "joint")
    difference = a-target
    statistic = n*np.trace(difference.T@precision@difference@covariance)
    baseline = min(statistic+2, 2*d*d)
    selected = (target, removed, "joint", None)
    best_cost = baseline-4
    candidates = [("polar", None)]
    for mask in range(1, (1 << (d-1))-1):
        boundaries = [0] + [i+1 for i in range(d-1) if mask & (1 << i)] + [d]
        candidates.append(("cluster", tuple(zip(boundaries[:-1], boundaries[1:]))))
    noise = np.sqrt(max(np.linalg.eigvalsh(innovation)[-1], 0.0)/n)
    for kind, groups in candidates:
        try:
            candidate, count, values = target_for(a, precision, covariance, kind, groups)
        except np.linalg.LinAlgError:
            continue
        if kind == "cluster" and np.min(np.diff(np.sort(values))) < 8*noise:
            continue
        diff = a-candidate
        test = n*np.trace(diff.T@precision@diff@covariance)
        cost = test+2*(d*d-count)
        if cost < best_cost:
            best_cost = cost
            selected = candidate, count, kind, groups
    return selected


def covariance_from_moments(cxx, cyy, cross, n, model):
    d = len(cxx)
    a = np.linalg.solve(cxx, cross.T).T
    omega = (cyy-a@cross.T)
    omega = (omega+omega.T)*0.5
    precision = np.linalg.inv(omega)
    target, removed, _ = target_for(a, precision, cxx, model["kind"], model["groups"])
    deviation = a-target
    statistic = n*np.trace(deviation.T@precision@deviation@cxx)
    factor = max(0., 1-model["strength"]*max(removed-2, 0)/max(statistic, 1e-20))
    fitted = target+factor*deviation
    difference = fitted-a
    residual = omega+difference@cxx@difference.T
    inverse = np.linalg.inv(np.eye(d)-fitted)
    h = inverse@residual@inverse.T
    return (h+h.T)*0.5*n/max(n-d-1, 1)


def numeric_influence(y, h, model):
    n, d = y.shape
    c = y.T@y/n
    cross = y[1:].T@y[:-1]/n
    eig, vec = np.linalg.eigh(h)
    invroot = (vec/np.sqrt(np.maximum(eig, 1e-15)))@vec.T
    a = np.linalg.solve(c, cross.T).T
    omega = c-a@cross.T
    step = 1e-5*max(np.linalg.eigvalsh((omega+omega.T)*0.5)[0], 1e-3)
    jacobian = []
    scores = []
    past, current = y[:-1], y[1:]
    for which, observation in ((0, past), (1, current)):
        for i in range(d):
            for j in range(i, d):
                perturbation = np.zeros((d, d))
                perturbation[i, j] = perturbation[j, i] = step
                arguments = [c.copy(), c.copy(), cross]
                arguments[which] += perturbation
                plus = covariance_from_moments(*arguments, n, model)
                arguments[which] -= 2*perturbation
                minus = covariance_from_moments(*arguments, n, model)
                jacobian.append((invroot@((plus-minus)/(2*step))@invroot).ravel())
                scores.append(observation[:, i]*observation[:, j])
    for i in range(d):
        for j in range(d):
            perturbation = np.zeros((d, d))
            perturbation[i, j] = step
            plus = covariance_from_moments(c, c, cross+perturbation, n, model)
            minus = covariance_from_moments(c, c, cross-perturbation, n, model)
            jacobian.append((invroot@((plus-minus)/(2*step))@invroot).ravel())
            scores.append(current[:, i]*past[:, j])
    scores = np.array(scores).T
    scores -= scores.mean(axis=0)
    return (scores@np.array(jacobian)).reshape(n-1, d, d)
