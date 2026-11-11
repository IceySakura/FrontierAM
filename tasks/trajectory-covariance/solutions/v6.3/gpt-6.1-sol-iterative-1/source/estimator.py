#!/usr/bin/env python3
"""Regularized long-run covariance estimation from one vector trajectory."""

import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit

from pooled import pooled_estimate


def finite_state_estimate(x):
    n, d = x.shape
    limit = min(512, n // 8)
    if len(np.unique(x[: min(n, 4096)], axis=0)) > limit:
        return None
    states, inverse = np.unique(x, axis=0, return_inverse=True)
    k = len(states)
    changes = inverse[1:] != inverse[:-1]
    if k > limit or changes.sum() < 3 * k:
        return None
    counts = np.zeros((k, k))
    np.add.at(counts, (inverse[:-1], inverse[1:]), 1.0)
    # Closing the observed path makes the empirical stationary measure exact.
    counts[inverse[-1], inverse[0]] += 1.0
    totals = counts.sum(axis=1)
    probability = totals / totals.sum()
    symmetric = (counts + counts.T) * 0.5
    mask = counts > 0
    deviance = 2.0 * np.sum(counts[mask] * np.log(counts[mask] / symmetric[mask]))
    edges = np.count_nonzero(np.triu(symmetric, 1))
    degrees = max(0, edges - k + 1)
    circulation = max(0.0, 1.0 - degrees / max(deviance, 1e-12))
    transition = (symmetric + circulation * (counts - symmetric)) / totals[:, None]
    centered = states - probability @ states
    poisson = np.linalg.solve(
        np.eye(k) - transition + probability[None, :], centered
    )
    cross = centered.T @ (probability[:, None] * poisson)
    marginal = centered.T @ (probability[:, None] * centered)
    return cross + cross.T - marginal


def vector_estimate(x):
    n, d = x.shape
    count = n - 1
    before = x[:-1] - x[:-1].mean(axis=0)
    after = x[1:] - x[1:].mean(axis=0)
    gram = before.T @ before
    marginal = after.T @ after / count
    result = marginal
    cap_coefficient = 2.0 * d + 2.0
    try:
        factor = cho_factor(gram, lower=True, check_finite=False)
        a = cho_solve(factor, before.T @ after, check_finite=False).T
        residual = after - before @ a.T
        noise = residual.T @ residual / count
        if d > 1:
            deviation = a - np.trace(a) / d * np.eye(d)
            influence = before @ cho_solve(factor, np.eye(d), check_finite=False)
            risk = np.sum(
                np.sum(residual * residual, axis=1)
                * np.sum(influence * influence, axis=1)
                - np.sum(residual * influence, axis=1) ** 2 / d
            )
            risk = max(0.0, risk)
            magnitude = max(np.sum(deviation * deviation), 1e-30)
            strength = min(1.0, 2.0 * risk / magnitude)
            cap_coefficient = 2.0 + 2.0 * d * (1.0 - strength)
            retention = max(0.0, 1.0 - strength * risk / magnitude)
            update = (1.0 - retention) * deviation
            noise += update @ (gram / count) @ update.T
            a -= update
        inverse = np.linalg.solve(np.eye(d) - a, np.eye(d))
        spectral = inverse @ noise @ inverse.T
        sign0, logdet0 = np.linalg.slogdet(marginal)
        sign1, logdet1 = np.linalg.slogdet(noise)
        if sign0 > 0 and sign1 > 0:
            penalty = 2.0 * d * d * count / max(1, count - 2 * d - 1)
            weight = expit(0.5 * (count * (logdet0 - logdet1) - penalty))
            result = (1 - weight) * marginal + weight * spectral
    except np.linalg.LinAlgError:
        pass
    vals, vecs = np.linalg.eigh((result + result.T) * 0.5)
    vals = np.maximum(vals, 0.0)
    vals /= 1.0 + cap_coefficient * vals / n
    return (vecs * np.minimum(vals, n / 3.0)) @ vecs.T


def estimate(x):
    n, d = x.shape
    x = np.asarray(x, dtype=np.float64)
    x = x - x[:1]
    x = x - x.mean(axis=0)
    scales = np.max(np.abs(x), axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    normalized = x / scales
    _, singular, vt = np.linalg.svd(normalized, full_matrices=False)
    keep = singular > max(singular[0] * 1e-12, 1e-30)
    if not np.any(keep):
        return np.zeros((d, d), dtype="<f8")
    vectors = vt[keep].T
    standard = singular[keep] / np.sqrt(n)
    root = scales[:, None] * (vectors * standard)
    x = normalized @ (vectors / standard)
    result = vector_estimate(x)
    try:
        pooled, weight = pooled_estimate(x)
        result = (1 - weight) * result + weight * pooled
    except np.linalg.LinAlgError:
        pass
    try:
        discrete = finite_state_estimate(x)
    except np.linalg.LinAlgError:
        discrete = None
    if discrete is not None:
        vals, vecs = np.linalg.eigh((discrete + discrete.T) * 0.5)
        vals = np.maximum(vals, 0.0)
        vals /= 1.0 + 32.0 * vals / n
        result = 0.5 * (result + (vecs * vals) @ vecs.T)
    result = root @ result @ root.T
    return np.ascontiguousarray((result + result.T) * 0.5, dtype="<f8")


def main():
    x = np.load(sys.argv[1], allow_pickle=False)
    try:
        result = estimate(x)
    except (np.linalg.LinAlgError, FloatingPointError):
        result = np.zeros((x.shape[1], x.shape[1]), dtype="<f8")
    if not np.isfinite(result).all():
        result = np.zeros((x.shape[1], x.shape[1]), dtype="<f8")
    with open(sys.argv[2], "wb") as handle:
        np.save(handle, result, allow_pickle=False)


if __name__ == "__main__":
    main()
