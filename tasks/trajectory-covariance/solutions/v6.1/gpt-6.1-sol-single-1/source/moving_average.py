"""Likelihood alternatives for short-memory Gaussian vector processes."""

import math
import time

import numpy as np
from scipy.linalg import schur, solve, solve_discrete_are, solve_discrete_lyapunov
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import lfilter


class FitBudget(Exception):
    pass


def symmetric(a):
    return (a+a.T)*0.5


def eligible(y):
    n, d = y.shape
    if n < 200 or d > 10:
        return False
    kurtosis = np.mean(np.sum(y*y, axis=1)**2)/(d*(d+2))
    if abs(kurtosis-1) > 0.25:
        return False
    lag = y[1:].T @ y[:-1]/n
    inflation = 1+2*np.sum(lag*lag)/d
    energy = sum(np.sum((y[k:].T @ y[:-k]/n)**2) for k in range(2, 7))
    threshold = inflation*(5*d*d+5*math.sqrt(10*d*d))
    return n*energy < threshold


def spectrum_bins(y):
    n, d = y.shape
    count = (n-1)//2
    fourier = np.fft.rfft(y, axis=0)[1:count+1]/math.sqrt(n)
    edges = np.unique(np.rint(
        count*(1-np.cos(np.linspace(0, np.pi, 385)))/2
    ).astype(int))
    counts = np.diff(edges)
    frequency = 2*np.pi*(1+(edges[:-1]+edges[1:]-1)/2)/n
    periodogram = fourier[:, :, None]*fourier[:, None, :].conj()
    average = np.add.reduceat(periodogram, edges[:-1], axis=0)
    average /= counts[:, None, None]
    return frequency, counts, average


def unpack(parameters, d):
    b = parameters[:d*d].reshape(d, d)
    lower = np.zeros((d, d))
    lower[np.tril_indices(d)] = parameters[d*d:]
    diagonal = np.diag_indices(d)
    lower[diagonal] = np.exp(lower[diagonal])
    return b, lower, lower @ lower.T


def whittle_objective(parameters, phase, weights, periodogram, d):
    b, lower, omega = unpack(parameters, d)
    lag = b @ omega
    spectral = (
        omega+b @ omega @ b.T
        + phase[:, None, None]*lag
        + phase.conj()[:, None, None]*lag.T
    )
    inverse = np.linalg.inv(spectral)
    objective = np.dot(
        weights,
        np.linalg.slogdet(spectral)[1]
        + np.einsum("kij,kji->k", inverse, periodogram).real,
    )
    score = inverse-inverse @ periodogram @ inverse
    gradient_b = 2*np.einsum(
        "k,kij->ij", weights,
        score @ (b+phase.conj()[:, None, None]*np.eye(d)) @ omega,
    ).real
    gradient_omega = np.einsum(
        "k,kij->ij", weights,
        score+b.T @ score @ b
        + phase[:, None, None]*(score @ b)
        + phase.conj()[:, None, None]*(b.T @ score),
    ).real
    gradient_lower = 2*symmetric(gradient_omega) @ lower
    gradient_lower[np.diag_indices(d)] *= np.diag(lower)
    gradient = np.r_[gradient_b.ravel(), gradient_lower[np.tril_indices(d)]]
    return float(objective), gradient


def matrix_filter(b, data):
    """Apply (I + B L)^(-1), using stable first/second-order Schur sections."""
    single = data.ndim == 2
    if single:
        data = data[:, :, None]
    n, d, channels = data.shape
    triangular, orthogonal = schur(-b, output="real", check_finite=False)
    inputs = np.einsum("ij,tjr->tir", orthogonal.T, data, optimize=True)
    output = np.empty_like(inputs)
    end = d
    while end:
        start = end-1
        if start and abs(triangular[start, start-1]) > 1e-14:
            start -= 1
        values = inputs[:, start:end].copy()
        if end < d:
            values[1:] += np.einsum(
                "ij,tjr->tir", triangular[start:end, end:],
                output[:-1, end:], optimize=True,
            )
        if end-start == 1:
            output[:, start] = lfilter(
                [1.0], [1.0, -triangular[start, start]], values[:, 0], axis=0
            )
        else:
            a, bb, c, dd = triangular[start:end, start:end].ravel()
            denominator = [1.0, -a-dd, a*dd-bb*c]
            output[:, start] = (
                lfilter([1.0, -dd], denominator, values[:, 0], axis=0)
                + lfilter([0.0, bb], denominator, values[:, 1], axis=0)
            )
            output[:, start+1] = (
                lfilter([0.0, c], denominator, values[:, 0], axis=0)
                + lfilter([1.0, -a], denominator, values[:, 1], axis=0)
            )
        end = start
    output = np.einsum("ij,tjr->tir", orthogonal, output, optimize=True)
    return output[:, :, 0] if single else output


def exact_objective(parameters, y):
    """Exact stationary Gaussian likelihood with the mean profiled out."""
    n, d = y.shape
    original_b, lower, original_omega = unpack(parameters, d)
    b, omega = original_b, original_omega
    radius = np.max(np.abs(np.linalg.eigvals(b)))
    if radius > 1+1e-8:
        lag = b @ omega
        omega = solve_discrete_are(
            np.zeros((d, d)), np.eye(d), omega+b @ omega @ b.T,
            np.zeros((d, d)), s=lag,
        )
        b = solve(omega, lag.T, assume_a="pos", check_finite=False).T
        radius = np.max(np.abs(np.linalg.eigvals(b)))
    inverse = np.linalg.inv(omega)

    # Only the initial transient needs matrix-valued filters. Its remaining
    # tail is bounded before replacing the constant-input response by a limit.
    if radius < 1e-10:
        length = min(n, 2*d+4)
    elif radius >= 1-1e-8:
        length = n
    else:
        bound = max(1.0, np.linalg.norm(b, 2))**d
        target = 1e-13*(1-radius)/bound
        length = min(n, max(16, math.ceil(math.log(target)/math.log(radius))+2*d))
        while length < n:
            if np.linalg.norm(np.linalg.matrix_power(-b, length)) <= target:
                break
            length = min(n, length*2)

    innovations = matrix_filter(b, y)
    impulse = np.zeros((length, d, d))
    impulse[0] = np.eye(d)
    powers = matrix_filter(b, impulse)
    initial = -b @ powers
    mean_response = np.cumsum(powers, axis=0)
    design = np.concatenate((initial, -mean_response), axis=2)
    normal = np.einsum("tki,tkj->ij", design, inverse @ design)
    normal[:d, :d] += inverse
    weighted = innovations @ inverse
    rhs = np.einsum("tki,tk->i", design, weighted[:length])
    if length < n:
        tail = mean_response[-1]
        normal[d:, d:] += (n-length)*(tail.T @ inverse @ tail)
        rhs[d:] -= tail.T @ weighted[length:].sum(axis=0)
    fitted = solve(symmetric(normal), rhs, assume_a="pos", check_finite=False)
    quadratic = np.sum(innovations*weighted)-rhs @ fitted
    logdet = (
        (n+1)*np.linalg.slogdet(omega)[1]
        + np.linalg.slogdet(normal[:d, :d])[1]
    )
    objective = (quadratic+logdet)/n

    latent = -fitted
    residual = innovations.copy()
    residual[:length] += initial @ latent[:d]-mean_response @ latent[d:]
    if length < n:
        residual[length:] -= tail @ latent[d:]
    alpha = matrix_filter(b.T, (residual @ inverse)[::-1])[::-1]
    weights = n-np.arange(length)
    weighted_powers = inverse @ powers
    diagonal = np.einsum(
        "t,tki,tkj->ij", weights, powers, weighted_powers
    )
    valid = min(length, n-1)
    subdiagonal = np.einsum(
        "t,tki,tkj->ij", weights[:valid]-1, powers[:valid], weighted_powers[:valid]
    ) @ (-b)
    woodbury = matrix_filter(b.T, (inverse @ initial)[::-1])[::-1]
    corrected = woodbury @ np.linalg.inv(normal[:d, :d])
    diagonal -= np.einsum("tik,tjk->ij", corrected, woodbury)
    subdiagonal -= np.einsum(
        "tik,tjk->ij", corrected[1:], woodbury[:-1]
    )
    g0 = diagonal-alpha.T @ alpha
    g1 = 2*(subdiagonal-alpha[1:].T @ alpha[:-1])
    gradient_b = 2*g0 @ original_b @ original_omega+g1 @ original_omega
    gradient_omega = (
        g0+original_b.T @ g0 @ original_b+symmetric(original_b.T @ g1)
    )
    gradient_lower = 2*symmetric(gradient_omega) @ lower
    gradient_lower[np.diag_indices(d)] *= np.diag(lower)
    gradient = np.r_[gradient_b.ravel(), gradient_lower[np.tril_indices(d)]]/n
    if not np.isfinite(objective) or not np.isfinite(gradient).all():
        raise ValueError("nonfinite likelihood")
    return float(objective), gradient


def scalar_fit(y):
    n, d = y.shape

    def objective(theta, details=False):
        innovations = lfilter([1.0], [1.0, theta], y, axis=0)
        initial = np.power(-theta, np.arange(1, n+1))
        response = lfilter([1.0], [1.0, theta], np.ones(n))
        precision = 1+initial @ initial
        cross = innovations.T @ initial
        overlap = initial @ response
        mean = innovations.T @ response-cross*overlap/precision
        information = response @ response-overlap*overlap/precision
        scatter = (
            innovations.T @ innovations-np.outer(cross, cross)/precision
            - np.outer(mean, mean)/information
        )
        omega = symmetric(scatter)/n
        sign, logdet = np.linalg.slogdet(omega)
        value = logdet+d*math.log(precision)/n if sign > 0 else 1e100
        return (value, omega) if details else value

    fit = minimize_scalar(
        objective, method="bounded", bounds=(-1.0, 1.0),
        options={"xatol": 1e-10},
    )
    theta = min((fit.x, -1.0, 1.0), key=objective)
    value, omega = objective(theta, details=True)
    h = (1+theta)**2*omega
    variance = max(1-theta*theta, 1/n)/(d*n)
    risk = (d+1)/n+4*variance/max((1+theta)**2, 1e-100)
    return h/(1+risk), n*(value+d)+2


def fisher_shrink(parameters, frequency, counts, d):
    b, lower, omega = unpack(parameters, d)
    phase = np.exp(-1j*frequency)
    transfer = np.eye(d)+phase[:, None, None]*b
    spectral = transfer @ omega @ transfer.conj().transpose(0, 2, 1)
    inverse = np.linalg.inv(spectral)
    derivatives = []
    h_derivatives = []
    total = np.eye(d)+b
    for i in range(d):
        for j in range(d):
            delta = np.zeros((d, d))
            delta[i, j] = 1.0
            dtransfer = phase[:, None, None]*delta
            derivatives.append(
                dtransfer @ omega @ transfer.conj().transpose(0, 2, 1)
                + transfer @ omega @ dtransfer.conj().transpose(0, 2, 1)
            )
            h_derivatives.append(
                delta @ omega @ total.T+total @ omega @ delta.T
            )
    for i, j in zip(*np.tril_indices(d)):
        delta = np.zeros((d, d))
        delta[i, j] = lower[i, j] if i == j else 1.0
        domega = delta @ lower.T+lower @ delta.T
        derivatives.append(
            transfer @ domega @ transfer.conj().transpose(0, 2, 1)
        )
        h_derivatives.append(total @ domega @ total.T)
    derivatives = np.asarray(derivatives)
    left = inverse[None] @ derivatives
    information = np.einsum("k,akij,bkji->ab", counts, left, left).real
    ev, u = np.linalg.eigh(symmetric(information))
    covariance = (u/np.maximum(ev, ev[-1]*1e-11)) @ u.T
    h = symmetric(total @ omega @ total.T)
    eh, uh = np.linalg.eigh(h)
    eh = np.maximum(eh, max(1e-300, eh[-1]*1e-14))
    root = (uh*np.sqrt(eh)) @ uh.T
    inverse_root = (uh/np.sqrt(eh)) @ uh.T
    dh = inverse_root @ np.asarray(h_derivatives) @ inverse_root
    risk = np.einsum("ab,aik,bkj->ij", covariance, dh, dh)
    er, ur = np.linalg.eigh(symmetric(risk))
    gain = (ur/np.sqrt(1+np.maximum(er, 0.0))) @ ur.T
    return symmetric(root @ gain @ gain @ root)


def ar_score(y, p, statistics):
    n, d = y.shape
    if not p:
        return n*(np.linalg.slogdet(y.T @ y/n)[1]+d)
    g, c, v = statistics(y, p)
    a = solve(g, c.T, assume_a="pos", check_finite=False).T
    omega = symmetric(v-a @ c.T)
    intercept = y[p:].mean(axis=0)
    for j in range(p):
        intercept -= a[:, j*d:(j+1)*d] @ y[p-j-1:n-j-1].mean(axis=0)
    mean = np.linalg.solve(np.eye(d)-a.reshape(d, p, d).sum(axis=1), intercept)
    companion = np.zeros((p*d, p*d))
    companion[:d] = a
    if p > 1:
        companion[d:, :-d] = np.eye((p-1)*d)
    if np.max(abs(np.linalg.eigvals(companion))) >= 1:
        raise ValueError("unstable AR comparison")
    noise = np.zeros_like(companion)
    noise[:d, :d] = omega
    initial_covariance = solve_discrete_lyapunov(companion, noise)
    initial = (y[:p][::-1]-mean).ravel()
    initial_cost = (
        np.linalg.slogdet(initial_covariance)[1]
        + initial @ solve(initial_covariance, initial, assume_a="pos",
                          check_finite=False)
    )
    return (n-p)*(np.linalg.slogdet(omega)[1]+d)+initial_cost+2*p*d*d


def improve_estimate(y, baseline, p, statistics):
    if not eligible(y):
        return baseline
    n, d = y.shape
    deadline = time.monotonic()+6.0
    matrices = [baseline]
    scores = [ar_score(y, p, statistics)]
    pooled, pooled_score = scalar_fit(y)
    matrices.append(pooled)
    scores.append(pooled_score)

    frequency, counts, periodogram = spectrum_bins(y)
    phase = (
        np.exp(-1j*frequency)*(1-1/n)*np.sinc(counts/n)/np.sinc(1/n)
    )
    weights = counts/counts.sum()
    lag = y[1:].T @ y[:-1]/n
    zero = y.T @ y/n
    for attempt in range(20):
        try:
            omega = solve_discrete_are(
                np.zeros((d, d)), np.eye(d), zero, np.zeros((d, d)), s=lag
            )
            lower = np.linalg.cholesky(omega)
            break
        except np.linalg.LinAlgError:
            lag *= 0.94
    else:
        omega = zero
        lag *= 0
        lower = np.linalg.cholesky(omega)
    b = solve(omega, lag.T, assume_a="pos", check_finite=False).T
    encoded = lower.copy()
    encoded[np.diag_indices(d)] = np.log(np.diag(lower))
    parameters = np.r_[b.ravel(), encoded[np.tril_indices(d)]]
    best = [np.inf, parameters]

    def warmup(candidate):
        if time.monotonic() > deadline:
            raise FitBudget()
        try:
            value, gradient = whittle_objective(
                candidate, phase, weights, periodogram, d
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 1e30, np.zeros_like(candidate)
        if value < best[0] and np.isfinite(value):
            best[:] = [value, candidate.copy()]
        return value, gradient

    try:
        minimize(
            warmup, parameters, jac=True, method="L-BFGS-B",
            options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-7, "maxls": 20},
        )
    except FitBudget:
        pass
    initial = best[1]
    best = [np.inf, initial]

    def refine(candidate):
        if time.monotonic() > deadline:
            raise FitBudget()
        try:
            value, gradient = exact_objective(candidate, y)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 1e30, np.zeros_like(candidate)
        if value < best[0] and np.isfinite(value):
            best[:] = [value, candidate.copy()]
        return value, gradient

    bounds = [(None, None)]*(d*d)
    bounds += [
        (-15.0, 2.0) if i == j else (-5.0, 5.0)
        for i, j in zip(*np.tril_indices(d))
    ]
    try:
        minimize(
            refine, initial, jac=True, bounds=bounds, method="L-BFGS-B",
            options={"maxiter": 60, "ftol": 2e-12, "gtol": 2e-7, "maxls": 15},
        )
    except FitBudget:
        pass
    if np.isfinite(best[0]):
        full_score = n*best[0]+2*d*d
        if full_score < min(scores)+18:
            matrices.append(fisher_shrink(best[1], frequency, counts, d))
            scores.append(full_score)
    scores = np.asarray(scores)
    weights = np.exp(-0.5*np.minimum(scores-scores.min(), 100.0))
    weights /= weights.sum()
    return symmetric(sum(w*h for w, h in zip(weights, matrices)))
