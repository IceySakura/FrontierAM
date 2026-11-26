"""Posterior model selection for two- and three-power extrapolation."""

import math
import numpy as np

from bayesian import CORRECTION_GAP, FIT_TOLERANCE, power_density, smooth_estimate, weighted_median
from gap_model import gap_estimate
from q_model import q_estimate


def two_log_evidence(x, fit):
    if fit is None or not fit["valid"] or fit["residual"] > FIT_TOLERANCE:
        return -np.inf
    powers = np.array([fit["p"], fit["q"]])
    logx = np.log(x / x[-1])
    allx = np.exp(logx[:, None] * powers[None, :])
    mat = allx[1:] - allx[:1]
    derivative = allx * logx[:, None]
    dmat = derivative[1:] - derivative[:1]
    jac = np.column_stack([mat, dmat * fit["coeff"][None, :]])
    _, logdet = np.linalg.slogdet(jac)
    if not np.isfinite(logdet):
        return 1000.0
    logdet += powers.sum() * math.log(x[-1])
    physical = fit["coeff"] * np.exp(-powers * math.log(x[-1]))
    prior = power_density(np.array([powers[0]]), np.array([powers[1] - powers[0]]),
                          physical[None, :])[0]
    return float(prior - logdet)


def selected_estimate(x, y, base, fit=None, probability_three=0.5):
    if len(x) == 5:
        if (fit is not None and fit["valid"] and fit["residual"] < 1e-12
                and abs(fit["q"] - fit["p"] - CORRECTION_GAP) < 1e-8):
            return base
        values3, weights3 = q_estimate(
            x, y, posterior=True, model_weight=probability_three)
        values2 = np.array([base])
        weights2 = np.array([two_log_evidence(x, fit)])
        weights2 += math.log(1 - probability_three)
    else:
        values3, weights3 = smooth_estimate(x, y, posterior=True)
        values2, weights2 = gap_estimate(x, y, posterior=True)
        weights3 += math.log(probability_three)
        weights2 += math.log(1 - probability_three)
    values = np.r_[values2, values3]
    weights = np.r_[weights2, weights3]
    # Posterior weights for the normalized absolute-loss objective.
    weights -= np.log(np.maximum(1.0, np.abs(values)))
    return weighted_median(values, weights)
