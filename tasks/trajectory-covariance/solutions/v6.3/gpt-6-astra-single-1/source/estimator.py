"""Single-trajectory long-run covariance estimation."""

import os

for _name in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import sys
import warnings

import numpy as np
from scipy import fft, optimize, special


def symmetric(a):
    return (a + a.T) * 0.5


def positive(a, floor=0.0, ceiling=None):
    values, vectors = np.linalg.eigh(symmetric(a))
    values = np.maximum(values, floor)
    if ceiling is not None:
        values = np.minimum(values, ceiling)
    return symmetric((vectors * values) @ vectors.T)


def standardize(x):
    centered = x - x[0]
    centered -= centered.mean(axis=0)
    scales = np.sqrt(np.mean(centered * centered, axis=0))
    scales[scales == 0] = 1.0
    z = centered / scales
    left, singular, right = np.linalg.svd(z, full_matrices=False)
    keep = singular > max(1e-14, singular[0] * 1e-12)
    if not np.any(keep):
        return np.empty((len(x), 0)), np.empty((x.shape[1], 0))
    # Avoid squaring the condition number of an unknown coordinate transform.
    root = scales[:, None] * right[keep].T * (singular[keep] / np.sqrt(len(x)))
    y = left[:, keep] * np.sqrt(len(x))
    return y, root


def clock_probabilities(exits, visits):
    mean = exits.sum() / visits.sum()
    if mean < 1e-10 or mean > 1 - 1e-6:
        return exits / visits

    def objective(theta):
        m = special.expit(theta[0])
        concentration = np.exp(theta[1])
        a, b = m * concentration, (1 - m) * concentration
        value = -np.sum(
            special.betaln(exits + a, visits - exits + b)
            - special.betaln(a, b)
        )
        common = special.digamma(concentration) - special.digamma(visits + concentration)
        da = np.sum(special.digamma(exits + a) - special.digamma(a) + common)
        db = np.sum(special.digamma(visits - exits + b) - special.digamma(b) + common)
        gradient = -np.array([
            (da - db) * concentration * m * (1 - m),
            da * a + db * b,
        ])
        return value, gradient

    fit = optimize.minimize(
        objective, [special.logit(mean), np.log(100.0)], jac=True,
        method="L-BFGS-B", bounds=[(-15, 15), (-5, 14)],
        options={"maxiter":70, "ftol":1e-10},
    )
    m, concentration = special.expit(fit.x[0]), np.exp(fit.x[1])
    return (exits + m * concentration) / (visits + concentration)


def finite_state_estimate(x, y):
    n, d = y.shape
    limit = 512
    if len(np.unique(x[:min(n, 2048)], axis=0)) > limit:
        return None
    _, indices, labels, visits = np.unique(
        x, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    count = len(visits)
    changes = np.count_nonzero(labels[1:] != labels[:-1])
    # Rejected proposals alone do not establish a finite state alphabet.
    if count < 2 or count > limit or changes < 4 * count:
        return None
    next_labels = np.roll(labels, -1)
    transitions = np.bincount(
        labels * count + next_labels, minlength=count * count
    ).reshape(count, count).astype(float)
    exits = visits - np.diag(transitions)
    if np.any(exits <= 0):
        return None
    probabilities = clock_probabilities(exits, visits)
    np.fill_diagonal(transitions, 0.0)
    transitions *= probabilities[:, None] / exits[:, None]
    np.fill_diagonal(transitions, 1.0 - probabilities)
    stationary = exits / probabilities
    stationary /= stationary.sum()
    functions = y[indices]
    functions = functions - stationary @ functions
    poisson = np.linalg.solve(
        np.eye(count) - transitions + stationary[None, :], functions
    )
    conditional = transitions @ poisson
    rows, columns = np.nonzero(transitions)
    residuals = poisson[columns] - conditional[rows]
    weights = stationary[rows] * transitions[rows, columns]
    h = residuals.T @ (weights[:, None] * residuals)
    values, vectors = np.linalg.eigh(symmetric(h))
    values = np.maximum(values, 0)
    values /= 1 + (d + 3) * values / n
    return symmetric((vectors * values) @ vectors.T)


def autocovariances(y, order):
    n, d = y.shape
    size = fft.next_fast_len(2 * n)
    transformed = fft.rfft(y, n=size, axis=0)
    covariances = np.empty((order + 1, d, d))
    for i in range(d):
        for j in range(d):
            covariances[:, i, j] = fft.irfft(
                transformed[:, i] * transformed[:, j].conj(), n=size
            )[:order + 1] / n
    return covariances


def select_order(y, penalty):
    n, d = y.shape
    maximum = min(60, n // (12 * d))
    cov = autocovariances(y, maximum)
    forward = np.empty((0, d, d))
    backward = forward.copy()
    vf = cov[0].copy()
    vb = vf.copy()
    best_score = np.linalg.slogdet(vf)[1]
    best_order = 0
    # Block Levinson recursion avoids a large regression at every candidate order.
    for order in range(1, maximum + 1):
        delta = cov[order].copy()
        if order > 1:
            delta -= np.einsum("kij,kjl->il", forward, cov[order - 1:0:-1])
        kf = np.linalg.solve(vb, delta.T).T
        kb = np.linalg.solve(vf, delta).T
        af = np.concatenate((
            forward - np.einsum("ij,kjl->kil", kf, backward[::-1]), kf[None]
        ))
        ab = np.concatenate((
            backward - np.einsum("ij,kjl->kil", kb, forward[::-1]), kb[None]
        ))
        next_vf = symmetric(vf - kf @ vb @ kf.T)
        vb = symmetric(vb - kb @ vf @ kb.T)
        vf = next_vf
        forward, backward = af, ab
        sign, logdet = np.linalg.slogdet(vf)
        if sign <= 0:
            break
        score = logdet + penalty * order * d * d / (n - order * d - 1)
        if score < best_score:
            best_score, best_order = score, order
    return best_order


def var_fit(y, order, blocks=16):
    n, d = y.shape
    design = np.column_stack(
        [np.ones(n - order)]
        + [y[order - lag - 1:n - lag - 1] for lag in range(order)]
    )
    target = y[order:]
    joined = np.column_stack((design, target))
    bounds = np.linspace(0, len(target), blocks + 1, dtype=int)
    scatters = np.array([
        joined[a:b].T @ joined[a:b] for a, b in zip(bounds[:-1], bounds[1:])
    ])
    total = scatters.sum(axis=0)
    columns = design.shape[1]

    def fitted(scatter, observations):
        gram = scatter[:columns, :columns].copy()
        gram.flat[::columns + 1] += observations * 1e-11
        coefficients = np.linalg.solve(gram, scatter[:columns, columns:])
        noise = symmetric(
            scatter[columns:, columns:]
            - scatter[:columns, columns:].T @ coefficients
        ) / max(1, observations - columns)
        noise = positive(noise, 1e-12)
        ar = coefficients[1:].reshape(order, d, d).transpose(0, 2, 1)
        transfer = np.linalg.inv(np.eye(d) - ar.sum(axis=0))
        h = symmetric(transfer @ noise @ transfer.T)
        return h, ar, noise, transfer, coefficients

    h, ar, noise, transfer, coefficients = fitted(total, len(target))
    replicates = np.array([
        fitted(total - scatter, len(target) - (b - a))[0]
        for scatter, a, b in zip(scatters, bounds[:-1], bounds[1:])
    ])
    residuals = target - design @ coefficients
    return h, ar, noise, transfer, residuals, replicates


def uncertainty_shrink(h, replicates, strength):
    h = positive(h, 1e-12)
    factor = np.linalg.cholesky(h)
    inverse = np.linalg.inv(factor)
    delta = np.einsum(
        "ij,kjl,lm->kim", inverse,
        replicates - replicates.mean(axis=0), inverse.T
    )
    count = len(replicates)
    uncertainty = (count - 1) / count * np.einsum("kij,kjl->il", delta, delta)
    return symmetric(
        factor @ np.linalg.solve(np.eye(len(h)) + strength * uncertainty, factor.T)
    )


def spectrum_bins(y):
    n, d = y.shape
    transformed = fft.rfft(y, axis=0)[1:(n + 1) // 2]
    periodogram = transformed[:, :, None] * transformed[:, None, :].conj() / n
    size = len(transformed)
    starts = np.unique(np.r_[
        np.arange(min(24, size)),
        np.geomspace(25, max(25, size), 100).astype(int) - 1,
    ])
    starts = starts[starts < size]
    starts = np.unique(np.r_[starts, size - 1 - starts])
    counts = np.diff(np.r_[starts, size]).astype(float)
    spectra = np.add.reduceat(periodogram, starts, axis=0) / counts[:, None, None]
    frequency = 2 * np.pi * (np.arange(size) + 1) / n
    cosine = np.add.reduceat(np.cos(frequency), starts) / counts
    omega = np.add.reduceat(frequency, starts) / counts
    return spectra, counts, cosine, omega, starts, frequency


def mixture_fit(y, bins=None, maximum=3):
    n, d = y.shape
    if bins is None:
        bins = spectrum_bins(y)
    pc, counts, cosine, omega, _, _ = bins
    periodogram = pc.real
    average = np.trace(periodogram, axis1=1, axis2=2) / d
    total = counts.sum()
    cap = np.arctanh(np.exp(-24 / n))
    lag = np.trace(y[1:].T @ y[:-1]) / ((n - 1) * d)

    def kernel(theta):
        poles = np.tanh(theta)
        denominator = 1 + poles[:, None] ** 2 - 2 * poles[:, None] * cosine
        numerator = 1 - poles ** 2
        g = numerator[:, None] / denominator
        derivative = (
            -2 * poles[:, None] * denominator
            - numerator[:, None] * (2 * poles[:, None] - 2 * cosine)
        ) / denominator ** 2
        derivative *= numerator[:, None]
        return poles, g, derivative

    def trace_fit(number, initials):
        best = None
        for initial in initials:
            start = np.r_[
                np.full(number, -np.log(number)),
                np.clip(np.arctanh(initial), -cap, cap),
            ]

            def objective(parameters):
                amplitudes = np.exp(parameters[:number])
                _, kernels, derivatives = kernel(parameters[number:])
                spectrum = amplitudes @ kernels
                score = counts * (1 - average / spectrum) / spectrum
                value = np.dot(counts, np.log(spectrum) + average / spectrum) / total
                gradient = np.r_[
                    amplitudes * (kernels @ score),
                    amplitudes * (derivatives @ score),
                ] / total
                return value, gradient

            fit = optimize.minimize(
                objective, start, jac=True, method="L-BFGS-B",
                bounds=[(-15, 3)] * number + [(-cap, cap)] * number,
                options={"maxiter":150, "ftol":1e-11, "gtol":1e-7},
            )
            if best is None or fit.fun < best.fun:
                best = fit
        amplitudes = np.exp(best.x[:number])
        poles = np.tanh(best.x[number:])
        score = best.fun * total * d + 3.5 * number * np.log(n)
        return score, amplitudes, poles

    fits = [trace_fit(1, [[np.clip(lag, -.95, .95)]])]
    if maximum >= 2:
        fits.append(trace_fit(2, [
            [np.clip(lag, -.8, .8), .97], [-.8, max(.2, min(.95, lag))]
        ]))
    if maximum >= 3:
        poles = fits[1][2]
        fits.append(trace_fit(3, [np.r_[poles, .995], np.r_[poles, -.7]]))
    _, amplitudes, poles = min(fits, key=lambda item: item[0])
    number = len(amplitudes)
    poles, kernels, _ = kernel(np.arctanh(poles))
    zero = (1 + poles) / (1 - poles)
    start = np.sqrt(amplitudes)[:, None, None] * np.eye(d)[None]

    def objective(parameters):
        factors = parameters.reshape(number, d, d)
        weights = factors @ factors.transpose(0, 2, 1)
        spectra = np.einsum("jk,jab->kab", kernels, weights) + np.eye(d) * 1e-10
        inverse = np.linalg.inv(spectra)
        logdet = np.linalg.slogdet(spectra)[1]
        score = inverse - inverse @ periodogram @ inverse
        value = np.dot(
            counts, logdet + np.einsum("kij,kji->k", inverse, periodogram)
        )
        gradient = np.einsum("jk,k,kab->jab", kernels, counts, score)
        h = np.einsum("j,jab->ab", zero, weights)
        value += 2 * np.linalg.slogdet(np.eye(d) + h)[1]
        gradient += 2 * zero[:, None, None] * np.linalg.inv(np.eye(d) + h)
        derivatives = 2 * gradient @ factors
        return value / total, derivatives.ravel() / total

    fit = optimize.minimize(
        objective, start.ravel(), jac=True, method="L-BFGS-B",
        options={"maxiter":180, "ftol":1e-10, "gtol":1e-6, "maxcor":12},
    )
    factors = fit.x.reshape(number, d, d)
    weights = factors @ factors.transpose(0, 2, 1)
    h = symmetric(np.einsum("j,jab->ab", zero, weights))
    nll = fit.fun * total - 2 * np.linalg.slogdet(np.eye(d) + h)[1]
    parameters = number * (d * (d + 1) / 2 + 1)
    return h, nll, parameters


def var_likelihood(ar, noise, bins):
    pc, counts, _, omega, _, _ = bins
    order, d, _ = ar.shape
    transfers = np.broadcast_to(
        np.eye(d, dtype=complex), (len(omega), d, d)
    ).copy()
    if order:
        transfers -= np.einsum(
            "kp,pij->kij",
            np.exp(-1j * omega[:, None] * np.arange(1, order + 1)), ar,
        )
    inverse = transfers.conj().transpose(0, 2, 1) @ np.linalg.inv(noise) @ transfers
    logdet = np.linalg.slogdet(noise)[1] - 2 * np.linalg.slogdet(transfers)[1]
    return np.dot(counts, logdet + np.einsum("kij,kji->k", inverse, pc).real)


def residual_correction(residuals):
    y, root = standardize(residuals)
    n, d = y.shape
    if d == 0:
        return np.zeros((residuals.shape[1], residuals.shape[1]))
    pc, counts, _, _, starts, frequency = spectrum_bins(y)
    periodogram = pc.real
    cosine = np.cos(frequency)
    times = np.geomspace(1.2, max(1.21, n / 24), 26)
    best_score = 0.0
    best = np.zeros((d, d))
    average = np.trace(periodogram, axis1=1, axis2=2) / d
    for time in times:
        pole = np.exp(-1 / time)
        kernel = (1 - pole * pole) / ((1 - pole) ** 2 + 2 * pole * (1 - cosine))
        w = np.add.reduceat(kernel - 1, starts) / counts

        def scalar_fit(observed):
            def objective(amplitude):
                fitted = 1 + amplitude * w
                return np.dot(counts, np.log(fitted) + observed / fitted - observed)
            fit = optimize.minimize_scalar(
                objective, bounds=(0, .98), method="bounded",
                options={"xatol":2e-6},
            )
            return (fit.x, -fit.fun) if fit.fun < 0 else (0.0, 0.0)

        amplitude, _ = scalar_fit(average)
        weighted = np.einsum(
            "k,kij->ij", counts * w / (1 + amplitude * w), periodogram
        )
        _, directions = np.linalg.eigh(symmetric(weighted))
        projected = np.einsum("ji,kjl,li->ki", directions, periodogram, directions)
        fitted = np.array([scalar_fit(projected[:, j]) for j in range(d)])
        order = np.argsort(fitted[:, 1])[::-1]
        for rank in range(1, d + 1):
            selected = order[:rank]
            improvement = fitted[selected, 1].sum()
            penalty = rank * (2 * d - rank + 1) / 2 + 1 + np.log(len(times))
            score = improvement - penalty
            if score > best_score:
                amplitudes = np.zeros(d)
                amplitudes[selected] = fitted[selected, 0] * (1 - penalty / improvement)
                best = (
                    directions * (amplitudes * 2 * pole / (1 - pole))
                ) @ directions.T
                best_score = score
    return symmetric(root @ best @ root.T)


def continuous_estimate(y):
    n, d = y.shape
    kurtosis = np.mean(np.sum(y * y, axis=1) ** 2) / (d * (d + 2))
    strength = float(np.clip(kurtosis, 1, 2))
    order = select_order(y, 2 * strength)
    h, ar, noise, transfer, residuals, replicates = var_fit(y, order)
    bins = spectrum_bins(y)
    candidate, nll, parameters = mixture_fit(y, bins)
    var_parameters = order * d * d + d * (d + 1) / 2
    gain = var_likelihood(ar, noise, bins) - nll + .75 * (var_parameters - parameters)
    if gain > 8:
        return candidate
    corrected = uncertainty_shrink(h, replicates, strength)
    bonus = residual_correction(residuals)
    corrected += transfer @ bonus @ transfer.T
    return symmetric(corrected)


def fallback(y):
    n, d = y.shape
    batch = max(1, int(np.sqrt(n)))
    cumulative = np.concatenate((np.zeros((1, d)), np.cumsum(y, axis=0)))
    sums = (cumulative[batch:] - cumulative[:-batch]) / np.sqrt(batch)
    return symmetric(sums.T @ sums / max(1, len(sums)))


def estimate(x):
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if n < 3 or not np.isfinite(x).all():
        return np.zeros((d, d), dtype="<f8")
    y, root = standardize(x)
    if y.shape[1] == 0:
        return np.zeros((d, d), dtype="<f8")
    try:
        h = finite_state_estimate(x, y)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        h = None
    if h is None:
        try:
            h = continuous_estimate(y)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, OverflowError):
            h = fallback(y)
    if not np.isfinite(h).all():
        h = fallback(y)
    h = positive(h, 0.0, max(1.0, n / 2))
    result = symmetric(root @ h @ root.T)
    return np.ascontiguousarray(result, dtype="<f8")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: run.sh INPUT.npy OUTPUT.npy")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    x = np.load(sys.argv[1], allow_pickle=False)
    result = estimate(x)
    with open(sys.argv[2], "wb") as output:
        np.save(output, result, allow_pickle=False)


if __name__ == "__main__":
    main()
