#!/usr/bin/env python3
"""Affine-equivariant estimation of multivariate long-run covariance."""

import os

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import sys

import numpy as np
from scipy import linalg
from regularize import reversible
from shared import pooled_dynamics
from grouped import grouped_dynamics


def positive(a, floor=1e-10):
    a = (a + a.T) * 0.5
    w, v = linalg.eigh(a, check_finite=False)
    return (v * np.maximum(w, floor)) @ v.T


def finite_state(x, support=None):
    """Use a fitted finite Markov chain when the observations have finite support."""
    _, labels, counts = np.unique(
        x if support is None else support,
        axis=0, return_inverse=True, return_counts=True
    )
    n, d = x.shape
    k = len(counts)
    if k > 2048:
        return None
    if k == 1:
        return np.zeros((d, d))
    jumps = 1 + np.count_nonzero(labels[1:] != labels[:-1])
    if jumps < k * 1.15:
        return None
    states = np.column_stack([
        np.bincount(labels, weights=x[:, j], minlength=k)/counts
        for j in range(d)
    ])
    transitions = np.bincount(
        labels[:-1] * k + labels[1:], minlength=k * k
    ).reshape(k, k).astype(float)
    pi = counts.astype(float) / n
    # A small empirical-Bayes regularizer also joins unvisited transition edges.
    transitions += 1.0 * pi[None, :]
    p = transitions / transitions.sum(axis=1, keepdims=True)
    pi = linalg.solve(
        np.eye(k)-p.T+np.ones((k, k)), np.ones(k), check_finite=False
    )
    pi = np.maximum(pi, 0.0)
    pi /= pi.sum()
    f = states-pi@states
    g = linalg.solve(
        np.eye(k)-p+np.ones((k, 1))*pi[None, :],
        f, check_finite=False,
    )
    cross = (f.T*pi)@g
    h = cross+cross.T-(f.T*pi)@f
    return positive(h)


def autocovariances(x, order):
    n, d = x.shape
    covs = np.empty((order + 1, d, d))
    covs[0] = x.T @ x / n
    for lag in range(1, order + 1):
        covs[lag] = x[lag:].T @ x[:-lag] / n
    return covs


def var_estimate(x, max_order=32, return_info=False):
    """Model-averaged conditional autoregressive spectral estimate."""
    n, d = x.shape
    order = min(max_order, max(1, n // (20 * d)))
    covs = autocovariances(x, order) * n
    # All orders share target times. Boundary corrections make the lag-sum
    # blocks exact conditional-regression sufficient statistics.
    joint = np.empty(((order + 1) * d, (order + 1) * d))
    means = []
    prefix = np.vstack((np.zeros((1, d)), np.cumsum(x, axis=0)))
    for i in range(order + 1):
        means.append(prefix[n-i] - prefix[order-i])
        for j in range(i, order + 1):
            block = covs[j-i].copy()
            block -= x[j-i:order-i].T @ x[:order-j]
            block -= x[n-i:].T @ x[n-j:n-(j-i)]
            joint[i*d:(i+1)*d, j*d:(j+1)*d] = block
            if i != j:
                joint[j*d:(j+1)*d, i*d:(i+1)*d] = block.T
    count = n - order
    means = np.concatenate(means)
    joint -= np.outer(means, means) / count
    joint /= count
    gram = joint[d:, d:].copy()
    gram.flat[::len(gram)+1] += 1e-10
    factor = linalg.cholesky(gram, lower=True, check_finite=False)
    c00 = joint[:d, :d]
    models = [c00]
    fits = [None]
    scores = [count * np.linalg.slogdet(c00)[1]]
    for p in range(1, order + 1):
        end = p * d
        rhs = joint[:d, d:d+end]
        coef = linalg.cho_solve(
            (factor[:end, :end], True), rhs.T, check_finite=False
        ).T
        noise = positive(c00 - coef @ rhs.T)
        a = np.eye(d) - coef.reshape(d, p, d).sum(axis=1)
        left = linalg.solve(a, noise, check_finite=False)
        h = linalg.solve(a, left.T, check_finite=False).T
        params = p * d * d
        score = count * np.linalg.slogdet(noise)[1]
        score += 2.0 * params * count / max(count - params - d - 1, 1)
        scores.append(score)
        models.append(h)
        fits.append((coef, a))
    scores = np.asarray(scores)
    weights = np.exp(-0.5 * np.minimum(scores - scores.min(), 1400))
    weights /= weights.sum()
    result = sum(w * h for w, h in zip(weights, models))
    result = positive(result)
    if not return_info:
        return result
    best = int(np.argmin(scores))
    info = {
        "order": best,
        "start": order,
        "target_mean": means[:d] / count,
        "h": models[best],
        "joint": joint,
        "score": scores[best],
        "count": count,
    }
    if best:
        end = best * d
        info["coef"], info["a"] = fits[best]
        info["predictor_mean"] = means[d:d+end].reshape(best, d) / count
        info["inverse_sum"] = linalg.cho_solve(
            (factor[:end, :end], True), np.tile(np.eye(d), (best, 1)),
            check_finite=False,
        )
    return result, info


def estimate(x):
    n, d = x.shape
    centered = x-x[0]
    scales = np.max(np.abs(centered), axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    centered /= scales
    centered -= centered.mean(axis=0)
    u, singular, vt = linalg.svd(
        centered, full_matrices=False, check_finite=False
    )
    keep = singular > singular[0]*1e-13
    if not keep.any():
        return np.zeros((d, d))
    transform = scales[:, None]*vt[keep].T*(singular[keep]/np.sqrt(n))
    z = np.ascontiguousarray(u[:, keep]*np.sqrt(n))
    z -= z.mean(axis=0)
    h = finite_state(z, support=centered)
    strength = 4.0 * z.shape[1]
    if h is None:
        h, info = var_estimate(z, return_info=True)
        if reversible(z):
            unpooled = h
            h, _, pooling_info = pooled_dynamics(h, info, n)
            strength = pooling_info["strength"]
            h, strength, _ = grouped_dynamics(
                unpooled, info, n, h, strength
            )
    # Long time scales are noisy. Limit inverse-operator amplification.
    w, v = linalg.eigh(h, check_finite=False)
    w = w / (1.0 + strength * np.maximum(w - 1.0, 0.0) / n)
    h = (v * w) @ v.T
    result = transform @ h @ transform.T
    return np.ascontiguousarray((result + result.T) * 0.5, dtype="<f8")


def main():
    x = np.load(sys.argv[1], allow_pickle=False)
    result = estimate(x)
    with open(sys.argv[2], "wb") as stream:
        np.save(stream, result, allow_pickle=False)


if __name__ == "__main__":
    main()
