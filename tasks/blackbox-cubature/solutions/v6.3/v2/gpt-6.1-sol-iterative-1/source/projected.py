"""General additive ridge controls, including smooth and kinked profiles."""
import itertools
import math
import time
from decimal import Decimal, localcontext

import numpy as np
from scipy.interpolate import BSpline, PPoly
from scipy.optimize import least_squares


def truncated_moment(weights, threshold, power, precision=65):
    """Integral of (threshold-sum(weights*x))_+**power."""
    with localcontext() as context:
        context.prec = precision
        w = [Decimal.from_float(float(v)) for v in weights]
        t = Decimal.from_float(float(threshold))
        if t <= 0:
            return Decimal(0)
        d = len(w)
        numerator = Decimal(0)
        for bits in itertools.product((0, 1), repeat=d):
            z = t - sum((a for a, bit in zip(w, bits) if bit), Decimal(0))
            if z > 0:
                term = z ** (d + power)
                numerator += -term if sum(bits) % 2 else term
        denominator = Decimal(math.factorial(d + power))
        for a in w:
            denominator *= a
        return numerator * math.factorial(power) / denominator


def spline_box_integral(knots, coef, degree, direction):
    w = 2 * np.abs(direction) / np.sum(np.abs(direction))
    w = w[w > 1e-13]
    total = float(np.sum(w))
    ppoly = PPoly.from_spline((knots, coef, degree))
    breaks = ppoly.x
    first = int(np.flatnonzero(np.diff(breaks) > 0)[0])
    initial = ppoly.c[:, first]
    with localcontext() as context:
        context.prec = 65
        s = Decimal.from_float(total)
        sum_squares = sum((Decimal.from_float(float(a)) ** 2 for a in w),
                          Decimal(0))
        moments = [Decimal(1), s / 2,
                   s * s / 4 + sum_squares / 12,
                   s ** 3 / 8 + s * sum_squares / 8]
        result = sum((Decimal.from_float(float(a)) * moments[degree - i]
                      for i, a in enumerate(initial)), Decimal(0))
        previous = float(initial[0])
        for i in range(first + 1, len(breaks) - 1):
            if breaks[i + 1] <= breaks[i]:
                continue
            coefficient = float(ppoly.c[0, i])
            jump = coefficient - previous
            previous = coefficient
            if abs(jump) < 1e-18:
                continue
            threshold = total - (float(breaks[i]) + 1)
            expectation = truncated_moment(w, threshold, degree)
            result += Decimal.from_float(jump) * expectation
        return float(result)


class SplineRidges:
    def __init__(self, directions, intervals=18):
        self.directions = np.asarray(directions, dtype=float)
        self.degree = 3
        self.knots = np.r_[np.full(4, -1.0),
                           np.linspace(-1, 1, intervals + 1)[1:-1],
                           np.full(4, 1.0)]
        self.ncoef = len(self.knots) - self.degree - 1

    def design(self, x):
        u = x - 0.5
        widths = np.sum(np.abs(self.directions), axis=1)
        projections = 2 * (u @ self.directions.T) / widths
        blocks = [np.ones((len(x), 1)), u]
        for j in range(len(self.directions)):
            z = np.clip(projections[:, j], -1, 1)
            blocks.append(BSpline.design_matrix(z, self.knots, 3).toarray())
        return np.column_stack(blocks)

    def fit(self, x, y):
        design = self.design(x)
        d = x.shape[1]
        penalty = np.zeros((len(self.directions) * (self.ncoef - 2),
                            design.shape[1]))
        differences = np.diff(np.eye(self.ncoef), n=2, axis=0)
        for j in range(len(self.directions)):
            rows = slice(j * (self.ncoef - 2), (j + 1) * (self.ncoef - 2))
            cols = slice(1 + d + j * self.ncoef, 1 + d + (j + 1) * self.ncoef)
            penalty[rows, cols] = differences
        penalty *= 0.002 * math.sqrt(len(x))
        self.coefficients = np.linalg.lstsq(
            np.vstack([design, penalty]), np.r_[y, np.zeros(len(penalty))],
            rcond=1e-10)[0]
        self.constant = float(self.coefficients[0])
        self.linear = self.coefficients[1:1 + d]
        self.curves = self.coefficients[1 + d:].reshape(-1, self.ncoef)
        return self

    def predict(self, x):
        return self.design(x) @ self.coefficients

    def integral(self):
        result = self.constant
        for direction, curve in zip(self.directions, self.curves):
            result += spline_box_integral(self.knots, curve, 3, direction)
        return float(result)


def exponential_absolute_integral(v, bias, growth=-1):
    w = np.abs(v)
    w = w[w > max(np.max(w, initial=0) * 1e-13, 1e-15)]
    if len(w) == 0:
        return float(np.exp(growth * abs(bias)))
    with localcontext() as context:
        context.prec = 70
        weights = [Decimal.from_float(float(a)) for a in w]
        center = sum(weights, Decimal(0)) / 2 + Decimal.from_float(float(bias))
        n = len(weights)
        denominator = Decimal(1)
        for a in weights:
            denominator *= a

        def antiderivative(z):
            if growth == -1:
                if z < 0:
                    return z.exp()
                result = Decimal((-1) ** n) * (-z).exp()
                for m in range(n):
                    factor = 1 - (-1) ** (n + m)
                    if factor:
                        result += factor * z ** m / math.factorial(m)
                return result
            if z < 0:
                return Decimal((-1) ** n) * (-z).exp()
            result = z.exp()
            for m in range(n):
                factor = (-1) ** (n + m) - 1
                if factor:
                    result += factor * z ** m / math.factorial(m)
            return result

        total = Decimal(0)
        for bits in itertools.product((0, 1), repeat=n):
            z = sum((a for a, bit in zip(weights, bits) if bit),
                    Decimal(0)) - center
            term = antiderivative(z)
            total += -term if (sum(bits) + n) % 2 else term
        return float(total / denominator)


class ExponentialAbsolute:
    def __init__(self, vectors, biases, amplitudes, constant=0, linear=None,
                 growth=None):
        self.vectors = np.asarray(vectors, dtype=float)
        self.biases = np.asarray(biases, dtype=float)
        self.amplitudes = np.asarray(amplitudes, dtype=float)
        self.constant = float(constant)
        self.linear = (np.zeros(self.vectors.shape[1]) if linear is None
                       else np.asarray(linear, dtype=float))
        self.growth = (np.full(len(self.biases), -1.0) if growth is None
                       else np.asarray(growth, dtype=float))

    def predict(self, x):
        u = x - 0.5
        z = u @ self.vectors.T - self.biases
        return (self.constant + u @ self.linear
                + np.exp(np.clip(np.abs(z) * self.growth, -745, 700))
                @ self.amplitudes)

    def integral(self):
        result = self.constant
        for v, b, a, growth in zip(self.vectors, self.biases,
                                   self.amplitudes, self.growth):
            result += a * exponential_absolute_integral(v, b, int(growth))
        return float(result)

    def refine(self, x, y, deadline):
        k, d = self.vectors.shape
        u = x - 0.5
        p = np.r_[self.constant, self.linear, self.amplitudes,
                  self.vectors.ravel(), self.biases]
        scale = max(float(np.std(y)), float(np.max(np.abs(y))) * 1e-10)

        def residual_jacobian(p):
            if time.monotonic() > deadline:
                raise TimeoutError("ridge profile time limit")
            a = p[1 + d:1 + d + k]
            v = p[1 + d + k:1 + d + k + k * d].reshape(k, d)
            b = p[1 + d + k + k * d:]
            z = u @ v.T - b
            basis = np.exp(np.clip(np.abs(z) * self.growth, -745, 200))
            derivative = basis * self.growth * np.sign(z) * a
            prediction = p[0] + u @ p[1:1 + d] + basis @ a
            jac = np.column_stack([
                np.ones(len(x)), u, basis,
                (derivative[:, :, None] * u[:, None, :]).reshape(len(x), k * d),
                -derivative])
            return (prediction - y) / scale, jac / scale

        result = least_squares(lambda p: residual_jacobian(p)[0], p,
                               jac=lambda p: residual_jacobian(p)[1],
                               max_nfev=120, ftol=1e-12, xtol=1e-12,
                               gtol=1e-12)
        p = result.x
        self.constant = p[0]
        self.linear = p[1:1 + d]
        self.amplitudes = p[1 + d:1 + d + k]
        self.vectors = p[1 + d + k:1 + d + k + k * d].reshape(k, d)
        self.biases = p[1 + d + k + k * d:]
        return self


def profile_exponential(z, y):
    scale = max(float(np.std(y)), np.finfo(float).tiny)
    linear = np.linalg.lstsq(np.column_stack([np.ones(len(z)), z]), y,
                             rcond=1e-12)[0]
    residual = y - linear[0] - linear[1] * z
    center = float(z[np.argmax(np.abs(residual))])
    best = None
    for rate in (1.0, 4.0, 12.0, -1.0):
        initial = [linear[0], linear[1], residual[np.argmax(np.abs(residual))],
                   rate, center]
        try:
            result = least_squares(
                lambda p: (p[0] + p[1] * z
                           + p[2] * np.exp(np.clip(-p[3] * np.abs(z - p[4]),
                                                  -100, 100)) - y) / scale,
                initial, bounds=([-np.inf, -np.inf, -np.inf, -20, -1.5],
                                 [np.inf, np.inf, np.inf, 100, 1.5]),
                max_nfev=100, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            error = np.linalg.norm(result.fun)
            if best is None or error < best[0]:
                best = error, result.x
        except ValueError:
            pass
    return None if best is None else best[1]


def profile_harmonic(z, y):
    scale = max(float(np.std(y)), np.finfo(float).tiny)
    best = None
    for frequency in np.linspace(0.1, 60, 150):
        design = np.column_stack([np.ones(len(z)), z, np.cos(frequency * z),
                                   np.sin(frequency * z)])
        coef = np.linalg.lstsq(design, y, rcond=1e-12)[0]
        error = np.mean((design @ coef - y) ** 2)
        if best is None or error < best[0]:
            best = error, frequency

    def residual(p):
        frequency = p[0]
        design = np.column_stack([np.ones(len(z)), z, np.cos(frequency * z),
                                   np.sin(frequency * z)])
        coef = np.linalg.lstsq(design, y, rcond=1e-12)[0]
        return (design @ coef - y) / scale

    result = least_squares(residual, [best[1]], bounds=(0.001, 100),
                           max_nfev=70, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    frequency = result.x[0]
    design = np.column_stack([np.ones(len(z)), z, np.cos(frequency * z),
                               np.sin(frequency * z)])
    coef = np.linalg.lstsq(design, y, rcond=1e-12)[0]
    return frequency, coef


def directions_from_hessians(oracle, x, y, old_x, old_g):
    d = x.shape[1]
    safe = np.flatnonzero(np.all((x > 0.02) & (x < 0.98), axis=1))
    if len(safe) < 12:
        return []
    chosen = safe[np.linspace(0, len(safe) - 1, 8, dtype=int)]
    anchors = x[chosen]
    step = 0.0005
    probes = np.repeat(anchors, 2 * d, axis=0)
    for j in range(d):
        probes[2 * j::2 * d, j] += step
        probes[2 * j + 1::2 * d, j] -= step
    if oracle.budget - oracle.used < len(probes) + 512:
        return []
    values = oracle.evaluate(probes).reshape(8, d, 2)
    gradients = np.vstack([old_g, (values[:, :, 0] - values[:, :, 1]) / (2 * step)])
    _, singular, vh = np.linalg.svd(gradients - np.mean(gradients, axis=0),
                                    full_matrices=False)
    if singular[0] < 1e-12:
        return []
    rank = min(d, int(np.sum(singular > singular[0] * 2e-5)))
    if rank < 1:
        return []
    subspace = vh[:rank]
    if rank == 1:
        return [subspace]
    count = 3
    h = 0.002
    hess_points = []
    hess_anchors = safe[np.linspace(0, len(safe) - 1, count, dtype=int)]
    for index in hess_anchors:
        a = x[index]
        for i in range(rank):
            hess_points.extend([a + h * subspace[i], a - h * subspace[i]])
        for i in range(rank):
            for j in range(i):
                for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    hess_points.append(a + h * (si * subspace[i] + sj * subspace[j]))
    if oracle.budget - oracle.used < len(hess_points) + 384:
        return []
    values = oracle.evaluate(np.asarray(hess_points))
    hessians = []
    offset = 0
    for index in hess_anchors:
        matrix = np.zeros((rank, rank))
        for i in range(rank):
            matrix[i, i] = (values[offset] + values[offset + 1] - 2 * y[index]) / h ** 2
            offset += 2
        for i in range(rank):
            for j in range(i):
                v = values[offset:offset + 4]
                matrix[i, j] = matrix[j, i] = (v[0] - v[1] - v[2] + v[3]) / (4 * h ** 2)
                offset += 4
        hessians.append(matrix)
    candidates = []
    for a in range(count):
        for b in range(a):
            try:
                base = hessians[b]
                if np.linalg.cond(base) > 1e10:
                    continue
                eigenvalues, vectors = np.linalg.eig(hessians[a] @ np.linalg.inv(base))
                if np.max(np.abs(vectors.imag)) > 1e-4:
                    continue
                vectors = vectors.real
                inverse = np.linalg.inv(vectors)
                error = 0.0
                for matrix in hessians:
                    diagonalized = inverse @ matrix @ inverse.T
                    offdiag = diagonalized - np.diag(np.diag(diagonalized))
                    error += np.linalg.norm(offdiag) / max(np.linalg.norm(diagonalized), 1e-12)
                directions = vectors.T @ subspace
                directions /= np.linalg.norm(directions, axis=1)[:, None]
                candidates.append((error, directions))
            except np.linalg.LinAlgError:
                continue
    candidates.sort(key=lambda row: row[0])
    return [entry[1] for entry in candidates[:2]]


def fit_projected(oracle, x, y, old_x, old_g, deadline):
    from ridges import Harmonic
    from offset_power import OffsetPower, profile_offset_power
    direction_sets = directions_from_hessians(oracle, x, y, old_x, old_g)
    if not direction_sets:
        return []
    all_x = np.vstack(oracle.history_x)
    all_y = np.concatenate(oracle.history_y)
    models = []
    scale = max(float(np.std(all_y)), float(np.max(np.abs(all_y))) * 1e-10)
    for directions in direction_sets:
        if time.monotonic() > deadline - 4:
            break
        spline = SplineRidges(directions).fit(all_x, all_y)
        models.append(spline)
        weights = 2 * directions / np.sum(np.abs(directions), axis=1)[:, None]
        projections = (all_x - 0.5) @ weights.T
        profile_z = [np.linspace(*np.quantile(projections[:, j], [0.02, 0.98]), 161)
                     for j in range(len(directions))]
        profiles = [BSpline(spline.knots, coef, 3)(z)
                    for coef, z in zip(spline.curves, profile_z)]
        try:
            params = [profile_harmonic(z, profile)
                      for z, profile in zip(profile_z, profiles)]
            frequencies = np.asarray([p[0] for p in params])
            coefficients = np.asarray([p[1] for p in params])
            harmonic = Harmonic(weights * frequencies[:, None],
                                coefficients[:, 2], coefficients[:, 3],
                                spline.constant + np.sum(coefficients[:, 0]))
            harmonic.refine(all_x, all_y, deadline - 1)
            models.append(harmonic)
            error = np.sqrt(np.mean((harmonic.predict(all_x) - all_y) ** 2)) / scale
            if error < 1e-9:
                break
        except (ValueError, np.linalg.LinAlgError, TimeoutError):
            pass
        if time.monotonic() > deadline - 4:
            break
        try:
            params = [profile_offset_power(z, profile, deadline - 3)
                      for z, profile in zip(profile_z, profiles)]
            if all(p is not None for p in params):
                params = np.asarray(params)
                power_model = OffsetPower(
                    directions, params[:, 3], np.exp(params[:, 4]), params[:, 5],
                    params[:, 2], spline.constant + np.sum(params[:, 0]),
                    spline.linear + params[:, 1] @ weights)
                power_model.refine(all_x, all_y, deadline - 1)
                models.append(power_model)
                error = np.sqrt(np.mean((power_model.predict(all_x) - all_y) ** 2)) / scale
                if error < 1e-9:
                    break
        except (ValueError, np.linalg.LinAlgError, TimeoutError):
            pass
        if time.monotonic() > deadline - 4:
            break
        try:
            params = [profile_exponential(z, profile)
                      for z, profile in zip(profile_z, profiles)]
            if any(p is None for p in params):
                continue
            params = np.asarray(params)
            rates = np.abs(params[:, 3])
            exp_model = ExponentialAbsolute(
                weights * rates[:, None], rates * params[:, 4], params[:, 2],
                spline.constant + np.sum(params[:, 0]),
                spline.linear + params[:, 1] @ weights,
                -np.sign(params[:, 3]))
            exp_model.refine(all_x, all_y, deadline - 1)
            models.append(exp_model)
            error = np.sqrt(np.mean((exp_model.predict(all_x) - all_y) ** 2)) / scale
            if error < 1e-9:
                break
        except (ValueError, np.linalg.LinAlgError, TimeoutError):
            continue
    return models
