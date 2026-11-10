"""VARHAC long-run covariance (Den Haan & Levin 1997), multivariate form.

Fit centred VAR(p) models by least squares for p = 0..pmax on a common estimation sample, choose p by an information
criterion (AIC or BIC on log det of the residual covariance), refit the chosen order on the full sample and return
(I - sum A_j)^{-1} Sigma_u (I - sum A_j)^{-T}. pmax = floor(M^(1/3)) as in Den Haan & Levin, capped so that every
equation keeps at least five observations per regressor.
"""

import sys

import numpy as np


def fit(x: np.ndarray, p: int, start: int):
    y = x[start:]
    if p == 0:
        return np.zeros((0, x.shape[1])), y
    design = np.hstack([x[start - k:len(x) - k] for k in range(1, p + 1)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef, y - design @ coef


def varhac(x: np.ndarray, criterion: str) -> np.ndarray:
    m, d = x.shape
    x = x - x.mean(axis=0)
    pmax = max(0, min(int(m ** (1 / 3)), (m // 5) // d - 1))
    best, best_p = np.inf, 0
    for p in range(pmax + 1):
        _, resid = fit(x, p, pmax)
        n = len(resid)
        sign, logdet = np.linalg.slogdet(resid.T @ resid / n)
        if sign <= 0:
            continue
        penalty = (2.0 if criterion == 'aic' else np.log(n)) * p * d * d / n
        if logdet + penalty < best:
            best, best_p = logdet + penalty, p
    coef, resid = fit(x, best_p, best_p)
    sigma = resid.T @ resid / len(resid)
    transfer = np.eye(d) - sum(coef[k * d:(k + 1) * d].T for k in range(best_p)) if best_p else np.eye(d)
    inverse = np.linalg.inv(transfer)
    estimate = inverse @ sigma @ inverse.T
    return (estimate + estimate.T) / 2


def main(criterion: str) -> None:
    x = np.load(sys.argv[1], allow_pickle=False).astype(float)
    np.save(sys.argv[2], np.ascontiguousarray(varhac(x, criterion), dtype='<f8'), allow_pickle=False)
