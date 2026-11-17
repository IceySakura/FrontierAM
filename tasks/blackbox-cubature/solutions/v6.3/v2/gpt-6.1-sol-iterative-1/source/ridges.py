"""Integrable harmonic and piecewise-affine ridge approximations."""
import itertools
import math
import time
from decimal import Decimal, localcontext

import numpy as np
from scipy.optimize import least_squares


class Harmonic:
    def __init__(self, frequencies, cosine, sine, constant):
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.cosine = np.asarray(cosine, dtype=float)
        self.sine = np.asarray(sine, dtype=float)
        self.constant = float(constant)

    def predict(self, x):
        phase = (x - 0.5) @ self.frequencies.T
        return (self.constant + np.cos(phase) @ self.cosine
                + np.sin(phase) @ self.sine)

    def integral(self):
        weights = np.prod(np.sinc(self.frequencies / (2 * math.pi)), axis=1)
        return float(self.constant + weights @ self.cosine)

    def refine(self, x, y, deadline):
        k, d = self.frequencies.shape
        p = np.r_[self.constant, self.cosine, self.sine,
                  self.frequencies.ravel()]
        scale = max(float(np.std(y)), np.finfo(float).tiny)
        u = x - 0.5

        def values_jacobian(p):
            a = p[1:1 + k]
            b = p[1 + k:1 + 2 * k]
            w = p[1 + 2 * k:].reshape(k, d)
            phase = u @ w.T
            co = np.cos(phase)
            si = np.sin(phase)
            prediction = p[0] + co @ a + si @ b
            gradient = (-si * a + co * b)[:, :, None] * u[:, None, :]
            jac = np.column_stack([np.ones(len(x)), co, si,
                                   gradient.reshape(len(x), k * d)])
            return (prediction - y) / scale, jac / scale

        def residual(p):
            if time.monotonic() > deadline:
                raise TimeoutError("harmonic fit time limit")
            return values_jacobian(p)[0]

        fit = least_squares(residual, p, jac=lambda p: values_jacobian(p)[1],
                            max_nfev=100, ftol=1e-12, xtol=1e-12,
                            gtol=1e-12)
        p = fit.x
        self.constant = float(p[0])
        self.cosine = p[1:1 + k]
        self.sine = p[1 + k:1 + 2 * k]
        self.frequencies = p[1 + 2 * k:].reshape(k, d)
        residual = self.predict(x) - y
        coefficient_floor = max(float(np.sqrt(np.mean(residual * residual))) * 8,
                                float(np.max(np.abs(y))) * np.finfo(float).eps * 16)
        if abs(self.constant) < coefficient_floor:
            self.constant = 0.0
        amplitude = float(np.sum(np.hypot(self.cosine, self.sine)))
        frequency_scale = float(np.max(np.sum(np.abs(self.frequencies), axis=1)))
        self.roundoff = max(float(np.max(np.abs(y))), amplitude) * (
            32 * np.finfo(float).eps * max(1.0, frequency_scale))
        return self


def harmonic_design(t, frequencies):
    phase = np.outer(t, frequencies)
    return np.column_stack([np.ones(len(t)), np.cos(phase), np.sin(phase)])


def line_frequencies(t, y, max_waves=16):
    """A matrix pencil gives starting frequencies without a grid search."""
    n = len(y)
    length = n // 2
    windows = np.lib.stride_tricks.sliding_window_view(y, length)
    left = windows[:-1].T
    right = windows[1:].T
    u, singular, vh = np.linalg.svd(left, full_matrices=False)
    if singular[0] <= np.finfo(float).tiny:
        return None
    rank = int(np.sum(singular > singular[0] * 2e-12))
    if rank < 2 or rank > 2 * max_waves + 2:
        return None
    reduced = (u[:, :rank].T @ right @ vh[:rank].T) / singular[:rank]
    roots = np.linalg.eigvals(reduced)
    dt = float(t[1] - t[0])
    frequencies = sorted(float(np.angle(z) / dt) for z in roots
                         if np.imag(z) > 1e-7 and 0.9 < abs(z) < 1.1)
    frequencies = np.asarray(frequencies)
    frequencies = frequencies[frequencies > 1e-4]
    if len(frequencies) == 0 or len(frequencies) > max_waves:
        return None
    if len(frequencies) > 1:
        frequencies = frequencies[np.r_[True, np.diff(frequencies) > 1e-3]]
    scale = max(float(np.std(y)), float(np.max(np.abs(y))) * 1e-10)

    def residual(w):
        design = harmonic_design(t, w)
        coef = np.linalg.lstsq(design, y, rcond=1e-13)[0]
        return (design @ coef - y) / scale

    result = least_squares(residual, frequencies,
                           bounds=(0.0, math.pi / dt), max_nfev=100,
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    frequencies = result.x
    design = harmonic_design(t, frequencies)
    coefficients = np.linalg.lstsq(design, y, rcond=1e-13)[0]
    error = np.max(np.abs(design @ coefficients - y)) / scale
    if not np.isfinite(error) or error > 1e-6:
        return None
    return frequencies, coefficients, error


def fit_harmonic(oracle, x, y, rng, deadline, tries=2):
    d = x.shape[1]
    candidates = []
    scale = max(float(np.std(y)), float(np.max(np.abs(y))) * 1e-10)
    for _ in range(tries):
        if time.monotonic() > deadline - 3:
            break
        t = np.linspace(-0.4, 0.4, 65)
        direction = rng.choice([-1.0, 1.0], d) * rng.uniform(0.7, 1.0, d)
        base = 0.5 + np.outer(t, direction)
        if oracle.budget - oracle.used < len(base) + 33 * d + 256:
            break
        base_y = oracle.evaluate(base)
        start = line_frequencies(t, base_y)
        oracle.line_data.append((t, direction, base_y, start is None))
        if start is None:
            continue
        omega, coef, _ = start
        amplitudes = coef[1:1 + len(omega)] - 1j * coef[1 + len(omega):]
        keep = np.abs(amplitudes) > scale * 1e-12
        omega = omega[keep]
        if len(omega) == 0:
            continue
        coef = np.linalg.lstsq(harmonic_design(t, omega), base_y, rcond=1e-13)[0]
        k = len(omega)
        if len(x) + len(base) + 33 * d < 2 * (1 + k * (d + 2)):
            continue
        t_shift = t[::2]
        shift = 0.0005
        lines = np.tile(0.5 + np.outer(t_shift, direction), (d, 1))
        for j in range(d):
            lines[j * len(t_shift):(j + 1) * len(t_shift), j] += shift
        line_y = oracle.evaluate(lines).reshape(d, len(t_shift))
        design = harmonic_design(t_shift, omega)
        all_coef = np.linalg.lstsq(design, line_y.T, rcond=1e-13)[0].T
        amplitudes = coef[1:1 + k] - 1j * coef[1 + k:]
        shifted = all_coef[:, 1:1 + k] - 1j * all_coef[:, 1 + k:]
        if np.any(np.abs(amplitudes) < scale * 1e-14):
            continue
        ratios = shifted / amplitudes
        frequencies = (np.angle(ratios) / shift).T
        model = Harmonic(frequencies, coef[1:1 + k], coef[1 + k:], coef[0])
        fit_x = np.vstack([x, base, lines])
        fit_y = np.r_[y, base_y, line_y.ravel()]
        try:
            model.refine(fit_x, fit_y, deadline - 1)
        except (ValueError, np.linalg.LinAlgError, TimeoutError):
            continue
        error = np.sqrt(np.mean((model.predict(x) - y) ** 2)) / scale
        candidates.append(model)
        if error < 1e-8:
            break
    return candidates


def absolute_ridge_integral(v, bias):
    """Exact box integral of |v.(x-1/2)-bias| by truncated powers."""
    w = np.abs(np.asarray(v, dtype=float))
    b = abs(float(bias))
    w = w[w > max(float(np.max(w, initial=0.0)) * 1e-14, 1e-16)]
    if len(w) == 0:
        return b
    if b >= np.sum(w) / 2:
        return b
    with localcontext() as context:
        context.prec = 65
        weights = [Decimal.from_float(float(a)) for a in w]
        total_width = sum(weights, Decimal(0))
        threshold = total_width / 2 - Decimal.from_float(b)
        numerator = Decimal(0)
        n = len(weights)
        for bits in itertools.product((0, 1), repeat=n):
            t = threshold - sum((weight for bit, weight in zip(bits, weights)
                                 if bit), Decimal(0))
            if t > 0:
                term = t ** (n + 1)
                numerator += -term if sum(bits) % 2 else term
        denominator = Decimal(math.factorial(n + 1))
        for weight in weights:
            denominator *= weight
        return float(Decimal.from_float(b) + 2 * numerator / denominator)


class PiecewiseAffine:
    def __init__(self, vectors, biases, signs=None):
        self.vectors = np.asarray(vectors, dtype=float).copy()
        self.biases = np.asarray(biases, dtype=float).copy()
        self.signs = (np.ones(len(self.biases)) if signs is None
                      else np.asarray(signs, dtype=float).copy())
        self.constant = 0.0
        self.linear = np.zeros(self.vectors.shape[1])

    def predict(self, x):
        u = x - 0.5
        z = u @ self.vectors.T - self.biases
        return self.constant + u @ self.linear + np.abs(z) @ self.signs

    def integral(self):
        result = self.constant
        for v, b, sign in zip(self.vectors, self.biases, self.signs):
            result += sign * absolute_ridge_integral(v, b)
        return float(result)

    def refine(self, x, y, deadline, max_nfev=150,
               gradient_x=None, gradient_y=None):
        k, d = self.vectors.shape
        u = x - 0.5
        features = np.column_stack([np.ones(len(x)), u,
                                   np.abs(u @ self.vectors.T - self.biases)])
        coef = np.linalg.lstsq(features, y, rcond=1e-12)[0]
        self.constant = coef[0]
        self.linear = coef[1:1 + d]
        amplitudes = coef[1 + d:]
        self.signs = np.where(np.abs(amplitudes) > 1e-12, np.sign(amplitudes), self.signs)
        amplitudes = np.maximum(np.abs(amplitudes), 1e-9)
        self.vectors *= amplitudes[:, None]
        self.biases *= amplitudes
        p = np.r_[self.constant, self.linear,
                  self.vectors.ravel(), self.biases]
        scale = max(float(np.std(y)), float(np.max(np.abs(y))) * 1e-10)

        use_gradients = gradient_x is not None and len(gradient_x) > 0
        if use_gradients:
            gu = gradient_x - 0.5
            gradient_scale = max(float(np.std(gradient_y)),
                                 float(np.max(np.abs(gradient_y))) * 1e-10)

        def values_jacobian(p):
            vectors = p[1 + d:1 + d + k * d].reshape(k, d)
            biases = p[1 + d + k * d:]
            z = u @ vectors.T - biases
            sign_z = np.sign(z) * self.signs
            prediction = p[0] + u @ p[1:1 + d] + np.abs(z) @ self.signs
            gradient = sign_z[:, :, None] * u[:, None, :]
            jac = np.column_stack([np.ones(len(x)), u,
                                   gradient.reshape(len(x), k * d), -sign_z])
            residual = (prediction - y) / scale
            jac = jac / scale
            if use_gradients:
                gz = gu @ vectors.T - biases
                signs = np.sign(gz) * self.signs
                prediction_g = p[1:1 + d] + signs @ vectors
                jac_g = np.zeros((len(gu), d, len(p)))
                jac_g[:, :, 1:1 + d] = np.eye(d)
                for j in range(k):
                    cols = slice(1 + d + j * d, 1 + d + (j + 1) * d)
                    jac_g[:, :, cols] = signs[:, j, None, None] * np.eye(d)
                weight = 0.3 / gradient_scale
                residual = np.r_[residual, ((prediction_g - gradient_y) * weight).ravel()]
                jac = np.vstack([jac, jac_g.reshape(-1, len(p)) * weight])
            return residual, jac

        def residual(p):
            if time.monotonic() > deadline:
                raise TimeoutError("piecewise fit time limit")
            return values_jacobian(p)[0]

        result = least_squares(residual, p, jac=lambda p: values_jacobian(p)[1],
                               max_nfev=max_nfev, ftol=1e-12, xtol=1e-12,
                               gtol=1e-12)
        if use_gradients and time.monotonic() < deadline - 1:
            use_gradients = False
            result = least_squares(residual, result.x,
                                   jac=lambda p: values_jacobian(p)[1],
                                   max_nfev=max_nfev, ftol=1e-12, xtol=1e-12,
                                   gtol=1e-12)
        p = result.x
        self.constant = float(p[0])
        self.linear = p[1:1 + d]
        self.vectors = p[1 + d:1 + d + k * d].reshape(k, d)
        self.biases = p[1 + d + k * d:]
        return self


def cluster_gradients(gradients, tolerance):
    representatives = []
    ids = []
    for g in gradients:
        found = None
        for i, representative in enumerate(representatives):
            if np.linalg.norm(g - representative) < tolerance:
                found = i
                break
        if found is None:
            found = len(representatives)
            representatives.append(g)
        ids.append(found)
    return np.asarray(representatives), np.asarray(ids)


def candidate_ridges(x, gradients):
    norms = np.linalg.norm(gradients, axis=1)
    tolerance = max(float(np.max(norms)) * 2e-7, 1e-9)
    representatives, ids = cluster_gradients(gradients, tolerance)
    if len(representatives) < 2:
        return []
    directions = []
    for i in range(len(representatives)):
        for j in range(i):
            difference = representatives[i] - representatives[j]
            size = float(np.linalg.norm(difference))
            if size < tolerance:
                continue
            direction = difference / size
            if direction[np.argmax(np.abs(direction))] < 0:
                direction = -direction
            found = None
            for entry in directions:
                if np.dot(entry["direction"], direction) > 1 - 1e-8:
                    found = entry
                    break
            if found is None:
                found = {"direction": direction, "pairs": [], "sizes": []}
                directions.append(found)
            found["pairs"].append((i, j))
            found["sizes"].append(size)
    result = []
    u = x - 0.5
    for entry in directions:
        direction = entry["direction"]
        low = -np.inf
        high = np.inf
        orientations = []
        for i, j in entry["pairs"]:
            zi = u[ids == i] @ direction
            zj = u[ids == j] @ direction
            gi = representatives[i] @ direction
            gj = representatives[j] @ direction
            if gi < gj:
                zi, zj = zj, zi
            orientations.append(float(np.median(zi) - np.median(zj)))
        sign = 1.0 if np.median(orientations) >= 0 else -1.0
        for i, j in entry["pairs"]:
            zi = u[ids == i] @ direction
            zj = u[ids == j] @ direction
            gi = representatives[i] @ direction
            gj = representatives[j] @ direction
            if sign * gi < sign * gj:
                zi, zj = zj, zi
            low = max(low, float(np.max(zj)))
            high = min(high, float(np.min(zi)))
        if low > high + 1e-5:
            continue
        bias = (low + high) / 2
        size = float(np.median(entry["sizes"])) / 2
        result.append((len(entry["pairs"]), direction * size, bias * size, sign))
    result.sort(key=lambda row: row[0], reverse=True)
    return result


def fit_piecewise(oracle, x, y, existing_x, existing_g, deadline):
    d = x.shape[1]
    count = min(48, len(x))
    corner_indices = np.flatnonzero(np.all((x < 0.002) | (x > 0.998), axis=1))
    interior_indices = np.flatnonzero(np.all((x > 0.002) & (x < 0.998), axis=1))
    if len(corner_indices) > 0:
        nc = min(32, len(corner_indices))
        corners = corner_indices[np.linspace(0, len(corner_indices) - 1, nc, dtype=int)]
        interior = interior_indices[np.linspace(0, len(interior_indices) - 1,
                                                  count - nc, dtype=int)]
        indices = np.r_[interior, corners]
        anchors = x[indices]
        base_y = y[indices]
    else:
        interior = interior_indices[np.linspace(0, len(interior_indices) - 1,
                                                  min(20, count), dtype=int)]
        nc = count - len(interior)
        rng = np.random.default_rng(7619 + d)
        corners = rng.integers(0, 2, (nc, d)).astype(float)
        corners[0] = 0
        if nc > 1:
            corners[1] = 1
        corner_y = oracle.evaluate(corners)
        anchors = np.vstack([x[interior], corners])
        base_y = np.r_[y[interior], corner_y]
        x = np.vstack([x, corners])
        y = np.r_[y, corner_y]
    step = 2e-6
    probes = np.repeat(anchors, d, axis=0)
    signs = np.where(anchors <= 0.5, 1.0, -1.0)
    for j in range(d):
        probes[j::d, j] += step * signs[:, j]
    if oracle.budget - oracle.used < len(probes) + 384:
        return []
    probe_y = oracle.evaluate(probes).reshape(count, d)
    gradients = (probe_y - base_y[:, None]) / step * signs
    gradient_x = np.vstack([existing_x, anchors])
    gradients = np.vstack([existing_g, gradients])
    entries = candidate_ridges(gradient_x, gradients)
    if not entries:
        return []
    scale = max(float(np.std(y)), float(np.max(np.abs(y))) * 1e-10)
    models = []
    maximum = min(24, len(entries), max(1, (len(x) - d - 1) // (2 * (d + 1))))
    vectors = np.asarray([entry[1] for entry in entries])
    biases = np.asarray([entry[2] for entry in entries])
    basis = np.abs((x - 0.5) @ vectors.T - biases)
    linear = np.column_stack([np.ones(len(x)), x - 0.5])
    q, _ = np.linalg.qr(linear, mode="reduced")
    orthogonal = basis - q @ (q.T @ basis)
    norms = np.sum(orthogonal * orthogonal, axis=0)
    residual = y - q @ (q.T @ y)
    selected = []
    checkpoints = {1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, maximum}
    for count_ridges in range(1, maximum + 1):
        scores = (orthogonal.T @ residual) ** 2 / np.maximum(norms, 1e-30)
        if selected:
            scores[selected] = -1
        index = int(np.argmax(scores))
        if norms[index] < 1e-25:
            break
        selected.append(index)
        design = np.column_stack([linear, basis[:, selected]])
        coefficients = np.linalg.lstsq(design, y, rcond=1e-12)[0]
        residual = y - design @ coefficients
        if count_ridges not in checkpoints:
            continue
        if time.monotonic() > deadline - 3:
            break
        subset = [entries[index] for index in selected]
        model = PiecewiseAffine([entry[1] for entry in subset],
                                [entry[2] for entry in subset],
                                [entry[3] for entry in subset])
        try:
            model.refine(x, y, deadline - 1, max_nfev=80,
                         gradient_x=gradient_x, gradient_y=gradients)
        except (ValueError, np.linalg.LinAlgError, TimeoutError):
            continue
        error = np.sqrt(np.mean((model.predict(x) - y) ** 2)) / scale
        models.append(model)
        if error < 1e-9:
            break
        for j, index in enumerate(selected):
            entries[index] = (entries[index][0], model.vectors[j].copy(),
                              float(model.biases[j]), float(model.signs[j]))
            basis[:, index] = np.abs((x - 0.5) @ model.vectors[j] - model.biases[j])
        residual = y - model.predict(x)
    return models


def structure_models(oracle, x, y, rng, deadline):
    d = x.shape[1]
    safe = np.flatnonzero(np.all((x > 0.002) & (x < 0.998), axis=1))
    if len(safe) < 4:
        return []
    indices = safe[np.linspace(0, len(safe) - 1, 4, dtype=int)]
    anchors = x[indices]
    base_y = y[indices]
    step = 0.0005 * (math.sqrt(5.0) - 1.0) / 2.0
    probes = np.repeat(anchors, 2 * d, axis=0)
    for j in range(d):
        probes[2 * j::2 * d, j] += step
        probes[2 * j + 1::2 * d, j] -= step
    if oracle.budget - oracle.used < len(probes) + 512:
        return []
    probes = np.clip(probes, 0, 1)
    values = oracle.evaluate(probes).reshape(4, d, 2)
    second = values[:, :, 0] + values[:, :, 1] - 2 * base_y[:, None]
    scale = max(float(np.std(y)), float(np.max(np.abs(y))) * 1e-10)
    curvature = np.median(np.max(np.abs(second), axis=1)) / (scale * step * step)
    gradients = (values[:, :, 0] - values[:, :, 1]) / (2 * step)
    if curvature < 1e-4:
        if np.max(np.linalg.norm(gradients, axis=1)) < scale * 1e-8 and len(np.unique(y)) < len(y) // 4:
            index = int(np.argmax(np.abs(y - np.mean(y))))
            a = x[index]
            signs = np.where(a <= 0.5, 1.0, -1.0)
            probes = np.repeat(a[None, :], 2 * d, axis=0)
            for j in range(d):
                probes[2 * j, j] += step * signs[j]
                probes[2 * j + 1, j] += 2 * step * signs[j]
            values = oracle.evaluate(probes).reshape(d, 2)
            gradient = (values[:, 0] - y[index]) / (step * signs)
            second = values[:, 1] - 2 * values[:, 0] + y[index]
            if np.linalg.norm(gradient) < scale * 1e-8:
                return []
            smooth_corner = np.max(np.abs(second)) > scale * step * step * 1e-4
            flat_after = np.max(np.abs(values[:, 1] - values[:, 0])) < scale * 1e-12
            if smooth_corner and not flat_after:
                return []
        from creases import fit_creases
        models = fit_creases(oracle, rng, deadline, fast=True)
        errors = [np.sqrt(np.mean((model.predict(x) - y) ** 2)) / scale for model in models]
        if ((not errors or min(errors) > 1e-8)
                and oracle.budget - oracle.used > 48 * d + 448
                and time.monotonic() < deadline - 5):
            models.extend(fit_piecewise(oracle, x, y, anchors, gradients, deadline))
        return models
    oracle.smooth = True
    models = fit_harmonic(oracle, x, y, rng, deadline, tries=1)
    errors = [np.sqrt(np.mean((model.predict(x) - y) ** 2)) / scale for model in models]
    if (not errors or min(errors) > 1e-8) and time.monotonic() < deadline - 5:
        from hinge_power import fit_hinge_powers
        models.extend(fit_hinge_powers(oracle, gradients, deadline))
        errors = [np.sqrt(np.mean((model.predict(x) - y) ** 2)) / scale for model in models]
    if (not errors or min(errors) > 1e-8) and time.monotonic() < deadline - 5:
        from rank_one import fit_active_quadrature
        active = fit_active_quadrature(oracle, x, y, rng, deadline)
        if active is not None:
            models.append(active)
            errors.append(np.sqrt(np.mean((active.predict(x) - y) ** 2)) / scale)
    if (not errors or min(errors) > 1e-8) and any(line[3] for line in oracle.line_data):
        from creases import fit_creases
        models.extend(fit_creases(oracle, rng, deadline))
        errors = [np.sqrt(np.mean((model.predict(x) - y) ** 2)) / scale for model in models]
    if (not errors or min(errors) > 1e-8) and time.monotonic() < deadline - 5:
        from projected import fit_projected
        models.extend(fit_projected(oracle, x, y, anchors, gradients, deadline))
    return models
