#!/usr/bin/env python3
"""Affine-equivariant long-run covariance estimation from one trajectory."""

import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from dynamics import select_target, numeric_influence
from categorical import second_order


def symmetric(a):
    return (a + a.T) * 0.5


def positive(a):
    values, vectors = np.linalg.eigh(symmetric(a))
    return symmetric((vectors * np.maximum(values, 0.0)) @ vectors.T)


def finite_state_estimate(x):
    n, d = x.shape
    probe = np.unique(x[: min(n, 1024)], axis=0)
    if len(probe) > 256:
        return None
    states, first, labels, count = np.unique(
        x, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    k = len(states)
    runs = 1 + np.count_nonzero(labels[1:] != labels[:-1])
    if k > min(256, n // 20) or runs < 5 * k:
        return None
    if k == 1:
        return np.zeros((d, d))
    permutation = np.argsort(first)
    inverse_permutation = np.empty(k, dtype=int)
    inverse_permutation[permutation] = np.arange(k)
    states, labels, count = states[permutation], inverse_permutation[labels], count[permutation]
    counts = np.bincount(labels[:-1] * k + labels[1:], minlength=k * k)
    counts = counts.reshape(k, k).astype(float)
    history_prior = None
    memory = second_order(states, labels, counts)
    if memory is not None:
        states, labels, count, counts, history_prior = memory
        n, k = len(labels), len(states)
    occupation = count / n
    identity = np.eye(k)
    reversible = False
    if history_prior is None:
        balanced = (counts+counts.T)*.5
        balanced_total = balanced.sum(axis=1)
        row_total = counts.sum(axis=1)
        probability = balanced/balanced_total[:, None]
        observed = counts > 0
        full_probability = counts/np.maximum(row_total[:, None], 1)
        statistic = 2*np.sum(counts[observed]*np.log(
            full_probability[observed]/probability[observed]
        ))
        edges = np.count_nonzero(np.triu(balanced, 1))
        degrees = max(edges-k+1, 0)
        reversible = degrees == 0 or statistic < degrees+3*np.sqrt(2*degrees)
    if history_prior is not None:
        transition = counts+.1*history_prior
        transition /= transition.sum(axis=1)[:, None]
        stationary = np.linalg.solve(
            identity-transition.T+np.ones((k, k)), np.ones(k)
        )
        stationary = np.maximum(stationary, 0.)
        stationary /= stationary.sum()
    elif reversible:
        transition = probability
        stationary = balanced_total/balanced_total.sum()
    else:
        transition = counts + 0.1 * occupation[None, :]
        transition /= transition.sum(axis=1)[:, None]
        stationary = np.linalg.solve(
            identity - transition.T + np.ones((k, k)), np.ones(k)
        )
        stationary = np.maximum(stationary, 0.0)
        stationary /= stationary.sum()
    f = states - stationary @ states
    g = np.linalg.solve(identity - transition + stationary[None, :], f)
    fg = f.T @ (stationary[:, None] * g)
    covariance = f.T @ (stationary[:, None] * f)
    h = positive(fg + fg.T - covariance)
    variance = finite_state_uncertainty(transition, stationary, f, g, h, n)
    return shrink_covariance(h, variance)


def finite_state_uncertainty(transition, stationary, f, g, h, n):
    k, d = f.shape
    eig, vec = np.linalg.eigh(h)
    invroot = (vec / np.sqrt(np.maximum(eig, 1e-14))) @ vec.T
    f = f @ invroot
    g = g @ invroot
    operator = np.eye(k) - transition + stationary[None, :]
    terms = (
        f[:, :, None] * g[:, None, :]
        + g[:, :, None] * f[:, None, :]
        - f[:, :, None] * f[:, None, :]
    )
    # Differentiate the Poisson equation with respect to transition probabilities.
    potential = np.linalg.solve(operator, terms.reshape(k, -1)).reshape(k, d, d)
    reverse = np.linalg.solve(operator.T, stationary[:, None]*f)
    reverse /= stationary[:, None]
    expected_potential = np.einsum("ij,jkl->ikl", transition, potential)
    expected_g = transition @ g
    variance = np.zeros((d, d))
    for i in range(k):
        difference = g - expected_g[i]
        influence = (
            potential - expected_potential[i]
            + reverse[i, :, None]*difference[:, None, :]
            + difference[:, :, None]*reverse[i, None, :]
        )
        influence *= np.sqrt(stationary[i]*transition[i])[:, None, None]
        flattened = influence.reshape(-1, d)
        variance += flattened.T @ flattened
    return symmetric(variance / n)


def lag_covariances(y, order):
    n, d = y.shape
    result = np.empty((order + 1, d, d))
    result[0] = y.T @ y / n
    for lag in range(1, order + 1):
        result[lag] = y[lag:].T @ y[:-lag] / n
    return result


def fit_var(y, return_pooling=False):
    n, d = y.shape
    max_order = min(48, max(8, int(8 * (n / 100.0) ** 0.25)))
    max_order = min(max_order, n // (12 * d))
    cov = lag_covariances(y, max_order)
    toeplitz = np.empty((max_order * d, max_order * d))
    for i in range(max_order):
        for j in range(max_order):
            toeplitz[i*d:(i+1)*d, j*d:(j+1)*d] = (
                cov[j-i] if j >= i else cov[i-j].T
            )
    rhs = cov[1:].transpose(0, 2, 1).reshape(max_order * d, d)
    best_score = np.inf
    best_h = cov[0]
    best_order = 0
    best_coefficients = np.zeros((0, d))
    best_innovation = cov[0]
    pooling = None
    for order in range(max_order + 1):
        if order:
            size = order * d
            factor = cho_factor(toeplitz[:size, :size], check_finite=False)
            coefficients = cho_solve(factor, rhs[:size], check_finite=False)
            innovation = symmetric(cov[0] - rhs[:size].T @ coefficients)
        else:
            innovation = cov[0]
        sign, determinant = np.linalg.slogdet(innovation)
        if sign <= 0:
            continue
        parameters = order * d * d
        score = n * determinant + 2 * parameters * n / max(n - order*d - 1, 1)
        if score < best_score:
            best_score = score
            best_order = order
            best_innovation = innovation
            if order:
                best_coefficients = coefficients.copy()
                transfer = np.eye(d) - coefficients.reshape(order, d, d).sum(0).T
                inverse = np.linalg.inv(transfer)
                best_h = inverse @ innovation @ inverse.T
            else:
                best_coefficients = np.zeros((0, d))
                best_h = innovation.copy()
    if best_order == 1:
        dynamic = best_coefficients.T
        precision = np.linalg.inv(best_innovation)
        rho = np.trace(precision @ dynamic) / np.trace(precision)
        target = rho*np.eye(d)
        removed = d*d-1
        kind, groups = "joint", None
        target, removed, kind, groups = select_target(
            dynamic, best_innovation, cov[0], n
        )
        deviation = dynamic - target
        size = n*np.trace(deviation.T @ precision @ deviation)
        strength = 1.0
        factor = max(0.0, 1.0 - strength*max(removed-2, 0)/max(size, 1e-20))
        original_dynamic = dynamic
        dynamic = target + factor*deviation
        best_coefficients = dynamic.T
        difference = dynamic-original_dynamic
        best_innovation = positive(
            best_innovation + difference @ cov[0] @ difference.T
        )
        pooling = {
            "factor": factor,
            "deviation": deviation,
            "original": original_dynamic,
            "precision": precision,
            "statistic": size/n,
            "kind": kind,
            "groups": groups,
            "strength": strength,
        }
        inverse = np.linalg.inv(np.eye(d) - dynamic)
        best_h = inverse @ best_innovation @ inverse.T
    best_h *= n / max(n - best_order * d - 1, 1)
    if return_pooling:
        return positive(best_h), best_coefficients, best_innovation, toeplitz, pooling
    return positive(best_h), best_coefficients, best_innovation, toeplitz


def uncertainty(
    y, h, coefficients, innovation, toeplitz, empirical=True, pooling=None
):
    n, d = y.shape
    order = coefficients.shape[0] // d
    eig, vec = np.linalg.eigh(h)
    root = (vec * np.sqrt(np.maximum(eig, 1e-15))) @ vec.T
    invroot = (vec / np.sqrt(np.maximum(eig, 1e-15))) @ vec.T
    if order:
        transfer = np.eye(d) - coefficients.reshape(order, d, d).sum(0).T
        j = np.tile(np.eye(d), (order, 1))
        past_inverse = np.linalg.solve(toeplitz[:order*d, :order*d], j)
        k = root @ j.T @ past_inverse @ root
    else:
        transfer = np.eye(d)
        k = np.zeros((d, d))
    theoretical = ((d + 1 + np.trace(k))*np.eye(d) + (d + 2)*k) / n
    if not empirical:
        return theoretical
    residual = y[order:].copy()
    v = np.zeros_like(residual)
    if order:
        for lag in range(order):
            past = y[order-lag-1:n-lag-1]
            residual -= past @ coefficients[lag*d:(lag+1)*d]
            v += past @ past_inverse[lag*d:(lag+1)*d] @ root
    z = np.linalg.solve(transfer, residual.T).T @ invroot
    if pooling is not None and pooling.get("kind", "joint") != "joint":
        try:
            influence = numeric_influence(y, h, pooling)
            if not np.isfinite(influence).all():
                raise np.linalg.LinAlgError("Nonfinite influence")
        except np.linalg.LinAlgError:
            influence = (
                z[:, :, None]*z[:, None, :]
                + z[:, :, None]*v[:, None, :]
                + v[:, :, None]*z[:, None, :]
            )
    elif pooling is None:
        influence = (
            z[:, :, None]*z[:, None, :]
            + z[:, :, None]*v[:, None, :]
            + v[:, :, None]*z[:, None, :]
        )
    else:
        # Include the derivative of the estimated pooling weight, not just the VAR.
        factor = pooling["factor"]
        deviation = pooling["deviation"]
        precision = pooling["precision"]
        original = pooling["original"]
        past = y[:-1]
        original_residual = y[1:] - past @ original.T
        inverse = np.linalg.inv(transfer)
        covariance = toeplitz[:d, :d]
        cross = (original-coefficients.T) @ covariance
        right = h - cross.T @ inverse.T
        v = past @ np.linalg.solve(covariance, right) @ invroot
        original_z = original_residual @ inverse.T @ invroot
        influence = z[:, :, None]*z[:, None, :]
        influence += factor*(
            original_z[:, :, None]*v[:, None, :]
            + v[:, :, None]*original_z[:, None, :]
        )
        shifted = past @ deviation.T
        rho_score = (
            np.sum((past @ precision)*(original_residual+shifted), axis=1)
            - np.sum((original_residual @ (precision@deviation@precision))
                     * original_residual, axis=1)
        ) / np.trace(precision)
        rho_effect = invroot @ inverse @ right @ invroot
        influence += (
            (1-factor)*rho_score[:, None, None]
            * (rho_effect+rho_effect.T)[None, :, :]
        )
        if factor > 0:
            score = (
                2*np.sum((original_residual @ precision)*shifted, axis=1)
                + np.sum((shifted @ precision)*shifted, axis=1)
                - np.sum((original_residual @ precision @ deviation)**2, axis=1)
            )
            coefficient_score = (1-factor)*score/max(pooling["statistic"], 1e-20)
            effect = invroot @ inverse @ deviation @ right @ invroot
            influence += coefficient_score[:, None, None]*(effect+effect.T)[None, :, :]
    influence -= influence.mean(axis=0)
    bandwidth = min(max(8, int(2*eig[-1])), n // 20)
    cumulative = np.concatenate(
        (np.zeros((1, d, d)), np.cumsum(influence, axis=0)), axis=0
    )
    results = []
    # This flat-top combination reduces the bias of the influence batch covariance.
    for b in (bandwidth, 2*bandwidth):
        batches = (cumulative[b:] - cumulative[:-b]).reshape(-1, d)
        results.append(batches.T @ batches / (b*(len(influence)-b+1)*n))
    observed = positive(2*results[1] - results[0])
    return observed


def shrink_covariance(h, variance, amount=1.0):
    eig, vec = np.linalg.eigh(h)
    root = (vec * np.sqrt(np.maximum(eig, 0.0))) @ vec.T
    return positive(root @ np.linalg.solve(
        np.eye(len(h)) + amount*variance, root
    ))


def var_estimate(y):
    h, coefficients, innovation, toeplitz, pooling = fit_var(y, True)
    variance = uncertainty(y, h, coefficients, innovation, toeplitz, True, pooling)
    h = shrink_covariance(h, variance)
    if pooling is not None and pooling["kind"] == "joint" and pooling["factor"] < .5:
        from rejection import estimate_rejections
        h = positive(estimate_rejections(y, h, pooling))
    return h


def estimate(x):
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if n < 3:
        return np.zeros((d, d))
    center = x-x[0]
    center -= center.mean(axis=0)
    scale = np.max(np.abs(center), axis=0)
    scale[scale == 0] = 1
    standardized = center/scale
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    active = singular > max(float(singular[0])*1e-14, np.finfo(float).tiny)
    if not np.any(active):
        return np.zeros((d, d))
    vectors = vt[active].T
    root = scale[:, None]*vectors*(singular[active]/np.sqrt(n))
    y = standardized @ (vectors*(np.sqrt(n)/singular[active]))
    discrete = finite_state_estimate(y)
    if discrete is not None:
        h = discrete
    else:
        h = var_estimate(y)
    return positive(root @ h @ root.T)


def main():
    x = np.load(sys.argv[1], allow_pickle=False)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite two-dimensional trajectory.")
    try:
        h = estimate(x)
        if not np.isfinite(h).all():
            raise np.linalg.LinAlgError("Nonfinite estimate")
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        n, d = x.shape
        width = max(1, int(n**(1/3)))
        count = n//width
        if count < 2:
            h = np.zeros((d, d))
        else:
            centered = x-x[0]
            batches = centered[:count*width].reshape(count, width, d).mean(axis=1)
            batches -= batches.mean(axis=0)
            h = positive(width*(batches.T@batches)/(count-1))
    with open(sys.argv[2], "wb") as output:
        np.save(output, np.ascontiguousarray(h, dtype="<f8"), allow_pickle=False)


if __name__ == "__main__":
    main()
