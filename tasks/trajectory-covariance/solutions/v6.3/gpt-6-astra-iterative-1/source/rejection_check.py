"""Efficient score test for a shared Gaussian target/proposal shape."""

import time

import numpy as np


class CheckTimeout(Exception):
    pass


def shape_weight(y, fit, deadline, probability_function):
    covariance, kappa, mean = fit
    n, d = y.shape
    if d == 1:
        return 1.
    root = np.linalg.cholesky(kappa*covariance)
    z = np.linalg.solve(root, (y-mean).T).T
    held = np.all(np.diff(y, axis=0) == 0, axis=1)
    first, last = z[:-1][~held], z[1:][~held]
    rejected = z[:-1][held]
    rows, columns = np.triu_indices(d)
    size = len(rows)
    multiplier = np.where(rows == columns, 1., np.sqrt(2))
    nuisance = d+1+size
    total = nuisance+size
    gram = np.zeros((total, total))
    gradient = np.zeros(total)
    identity = np.eye(d)

    def accumulate(mean_score, step_score, proposal_score, shape_score):
        scores = np.column_stack((
            mean_score, step_score,
            proposal_score[:, rows, columns]*multiplier,
            shape_score[:, rows, columns]*multiplier,
        ))
        gram[:] += scores.T@scores
        gradient[:] += scores.sum(axis=0)

    for start in range(0, len(first), 4096):
        if time.perf_counter() > deadline:
            raise CheckTimeout()
        a, b = first[start:start+4096], last[start:start+4096]
        change = b-a
        energy = .5*kappa*(np.sum(b*b, axis=1)-np.sum(a*a, axis=1))
        uphill = (energy > 0).astype(float)
        extra = -.5*kappa*uphill[:, None, None]*(
            b[:, :, None]*b[:, None, :]-a[:, :, None]*a[:, None, :]
        )
        proposal = .5*(change[:, :, None]*change[:, None, :]-identity)-extra
        accumulate(kappa*uphill[:, None]*change, -np.maximum(energy, 0.), proposal, extra)
    for start in range(0, len(rejected), 4096):
        if time.perf_counter() > deadline:
            raise CheckTimeout()
        x = rejected[start:start+4096]
        squared = np.sum(x*x, axis=1)
        radius = kappa*squared
        probability, dz, dk, b0 = probability_function(kappa, radius, d, True)
        constant = kappa*b0/probability
        step = (kappa*dk+dz*radius)/probability
        outer = x[:, :, None]*x[:, None, :]
        extra = constant[:, None, None]*identity+(
            (step-d*constant)/np.maximum(squared, 1e-14)
        )[:, None, None]*outer
        proposal = (-kappa*dz/probability)[:, None, None]*outer
        mean_score = (-2*kappa*dz/probability)[:, None]*x
        accumulate(mean_score, step, proposal, extra)

    scaling = np.sqrt(np.maximum(np.diag(gram), 1e-15))
    gram /= scaling[:, None]*scaling[None, :]
    gradient /= scaling
    # Remove fitted nuisance-score directions before testing covariance shape.
    information = gram[:nuisance, :nuisance]
    cross = gram[:nuisance, nuisance:]
    projection = np.linalg.solve(information+np.eye(nuisance)*1e-10, cross)
    efficient = gram[nuisance:, nuisance:]-cross.T@projection
    score = gradient[nuisance:]-projection.T@gradient[:nuisance]
    eig, vec = np.linalg.eigh((efficient+efficient.T)*.5)
    active = eig > max(eig[-1]*1e-7, 1e-14)
    degrees = int(active.sum())
    if degrees != size-1:
        return None
    statistic = np.sum((vec[:, active].T@score)**2/eig[active])
    if not np.isfinite(statistic) or statistic > 2*degrees:
        return 0.
    return min(1., max(degrees-2, 1)/max(statistic, 1e-12))
