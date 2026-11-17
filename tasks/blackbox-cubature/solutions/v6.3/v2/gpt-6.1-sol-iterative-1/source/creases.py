"""Active crease localization and gradient-jump ridge reconstruction."""
import time

import numpy as np
from scipy.interpolate import BSpline

from offset_power import OffsetPower, profile_offset_power
from projected import ExponentialAbsolute, SplineRidges, profile_exponential
from ridges import PiecewiseAffine


class AffineCreaseSum:
    def __init__(self, creases):
        self.vectors = np.asarray([entry[0] * entry[4] for entry in creases])
        self.biases = np.asarray([entry[1] * entry[4] for entry in creases])
        self.signs = np.asarray([entry[2] for entry in creases])

    def predict(self, x):
        if len(self.signs) == 0:
            return np.zeros(len(x))
        return np.abs((x - 0.5) @ self.vectors.T - self.biases) @ self.signs

    def gradient(self, x):
        if len(self.signs) == 0:
            return np.zeros_like(x)
        signs = np.sign((x - 0.5) @ self.vectors.T - self.biases) * self.signs
        return signs @ self.vectors


def refine_crease(oracle, direction, left, middle, right, values, scale,
                  fast=False, subtraction=None):
    yleft, ymid, yright = values
    initial_width = (right - left) / 2
    initial_jump = abs(yleft - 2 * ymid + yright) / initial_width
    jump = initial_jump
    reserve = 128 if fast else 512
    iterations = 5 if fast else 7
    for iteration in range(iterations):
        if oracle.budget - oracle.used < 6 + 4 * len(direction) + reserve:
            return None
        t = np.linspace(left, right, 9)
        t[4] = middle
        indices = np.array([1, 2, 3, 5, 6, 7])
        x = 0.5 + np.outer(t[indices], direction)
        queried = oracle.evaluate(x)
        if subtraction is not None:
            queried -= subtraction.predict(x)
        y = np.empty(9)
        y[0], y[4], y[8] = yleft, ymid, yright
        y[indices] = queried
        second = np.abs(np.diff(y, n=2))
        peak = 1 + int(np.argmax(second))
        grid_width = (right - left) / 8
        jump = second[peak - 1] / grid_width
        if iteration >= 2 and jump < initial_jump * 0.025:
            return None
        left, middle, right = t[peak - 1], t[peak], t[peak + 1]
        yleft, ymid, yright = y[peak - 1], y[peak], y[peak + 1]
    width = (right - left) / 2
    if jump < scale * (1e-9 if fast else 1e-5) or jump < initial_jump * 0.02:
        return None
    epsilon = max(width * (2 if fast else 8), 4e-6 if fast else 2e-5)
    maximum = float(np.max(np.abs(direction)))
    margin = 0.5 / maximum - abs(middle)
    epsilon = min(epsilon, margin / 4)
    if epsilon < width:
        return None
    a = 0.5 + (middle - epsilon) * direction
    b = 0.5 + (middle + epsilon) * direction
    d = len(direction)
    if fast:
        anchors = np.asarray([a, b])
        signed_h = np.where(anchors <= 0.5, 1.0, -1.0) * min(epsilon * 0.001, 1e-7)
        probes = np.repeat(anchors, d, axis=0)
        for j in range(d):
            probes[j::d, j] += signed_h[:, j]
        if oracle.budget - oracle.used < len(probes) + 2 + reserve:
            return None
        y = oracle.evaluate(np.vstack([anchors, probes]))
        base_y = y[:2]
        gradient = (y[2:].reshape(2, d) - base_y[:, None]) / signed_h
        if subtraction is not None:
            base_y -= subtraction.predict(anchors)
            gradient -= subtraction.gradient(anchors)
        first_difference = gradient[1] - gradient[0]
        normal = first_difference / max(np.linalg.norm(first_difference), 1e-30)
        if oracle.budget - oracle.used > len(probes) + 512:
            larger = np.minimum(
                0.002, epsilon * 0.2 * abs(direction @ normal)
                / np.maximum(np.abs(normal), 1e-8))
            signed_larger = np.where(anchors <= 0.5, 1.0, -1.0) * larger
            probes = np.repeat(anchors, d, axis=0)
            for j in range(d):
                probes[j::d, j] += signed_larger[:, j]
            queried = oracle.evaluate(probes).reshape(2, d)
            if subtraction is not None:
                queried -= subtraction.predict(probes).reshape(2, d)
            improved = (queried - base_y[:, None]) / signed_larger
            delta = improved[1] - improved[0]
            if np.linalg.norm(delta - first_difference) < np.linalg.norm(first_difference) * 0.02:
                gradient = improved
        slopes = gradient @ direction
        intercepts = base_y - slopes * np.asarray([middle - epsilon, middle + epsilon])
        if abs(slopes[1] - slopes[0]) > scale * 1e-10:
            intersection = (intercepts[0] - intercepts[1]) / (slopes[1] - slopes[0])
            if middle - epsilon <= intersection <= middle + epsilon:
                middle = float(intersection)
    else:
        h = min(epsilon * 0.003, 2e-7)
        probes = np.repeat(np.asarray([a, b]), 2 * d, axis=0)
        for j in range(d):
            probes[2 * j::2 * d, j] += h
            probes[2 * j + 1::2 * d, j] -= h
        if oracle.budget - oracle.used < len(probes) + reserve:
            return None
        y = oracle.evaluate(probes).reshape(2, d, 2)
        gradient = (y[:, :, 0] - y[:, :, 1]) / (2 * h)
    difference = gradient[1] - gradient[0]
    norm = float(np.linalg.norm(difference))
    if norm < scale * (1e-9 if fast else 1e-5) or not np.isfinite(norm):
        return None
    normal = difference / norm
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    x0 = 0.5 + middle * direction
    bias = float((x0 - 0.5) @ normal)
    orientation = float(np.sign((difference @ normal) / (direction @ normal)))
    return normal, bias, orientation, jump, norm / 2


def discover_creases(oracle, rng, deadline, fast=False):
    d = oracle.dimension
    records = [] if fast else list(oracle.line_data)
    directions = [np.ones(d), rng.choice([-1.0, 1.0], d)]
    creases = []
    explained = False
    for line_index in range(min(4, len(records) + len(directions))):
        if time.monotonic() > deadline - 5:
            break
        if line_index < len(records):
            t, direction, values, _ = records[line_index]
        else:
            if oracle.budget - oracle.used < 65 + (128 if fast else 512):
                break
            direction = directions[line_index - len(records)]
            t = np.linspace(-0.4999, 0.4999, 65)
            values = oracle.evaluate(0.5 + np.outer(t, direction))
        scale = max(float(np.std(values)), float(np.max(np.abs(values))) * 1e-10)
        blocked = set()
        attempts = 48 if fast else 10
        for _ in range(attempts):
            if time.monotonic() > deadline - 4 or len(creases) >= (24 if fast else 10):
                break
            subtraction = AffineCreaseSum(creases) if fast else None
            signal = values - subtraction.predict(0.5 + np.outer(t, direction)) if fast else values
            second = np.abs(np.diff(signal, n=2))
            peaks = [i for i in range(len(second))
                     if i not in blocked and (i == 0 or second[i] >= second[i - 1])
                     and (i + 1 == len(second) or second[i] >= second[i + 1])]
            if not peaks:
                break
            index = max(peaks, key=lambda i: second[i])
            if fast and second[index] < scale * 2e-11:
                break
            blocked.add(index)
            center = index + 1
            candidate = refine_crease(
                oracle, direction, t[center - 1], t[center], t[center + 1],
                signal[center - 1:center + 2], scale, fast=fast, subtraction=subtraction)
            if candidate is None:
                continue
            normal, bias, orientation, jump, strength = candidate
            duplicate = False
            for old in creases:
                cosine = 0.999999 if fast else 0.9995
                distance = 1e-6 if fast else 0.003
                if abs(old[0] @ normal) > cosine and abs(old[1] - bias) < distance:
                    duplicate = True
                    break
            if not duplicate:
                creases.append((normal, bias, orientation, jump, strength))
                if fast:
                    blocked.clear()
                    all_x = np.vstack(oracle.history_x)
                    all_y = np.concatenate(oracle.history_y)
                    remaining_y = all_y - AffineCreaseSum(creases).predict(all_x)
                    linear = np.column_stack([np.ones(len(all_x)), all_x - 0.5])
                    coef = np.linalg.lstsq(linear, remaining_y, rcond=1e-12)[0]
                    error = np.sqrt(np.mean((linear @ coef - remaining_y) ** 2))
                    if error < scale * 1e-10:
                        explained = True
                        break
        if explained or oracle.budget - oracle.used < (220 if fast else 600):
            break
    return creases


def fit_creases(oracle, rng, deadline, fast=False):
    creases = discover_creases(oracle, rng, deadline, fast=fast)
    if not creases or time.monotonic() > deadline - 3:
        return []
    directions = np.asarray([entry[0] for entry in creases])
    biases = np.asarray([entry[1] for entry in creases])
    signs = np.asarray([entry[2] for entry in creases])
    x = np.vstack(oracle.history_x)
    y = np.concatenate(oracle.history_y)
    scale = max(float(np.std(y)), float(np.max(np.abs(y))) * 1e-10)
    models = []
    try:
        affine = PiecewiseAffine(directions, biases, signs)
        affine.refine(x, y, deadline - 1)
        models.append(affine)
        if np.sqrt(np.mean((affine.predict(x) - y) ** 2)) < scale * 1e-9:
            return models
    except (ValueError, np.linalg.LinAlgError, TimeoutError):
        pass
    if time.monotonic() > deadline - 4:
        return models
    weights = 2 * directions / np.sum(np.abs(directions), axis=1)[:, None]
    hints = 2 * biases / np.sum(np.abs(directions), axis=1)
    projection = (x - 0.5) @ weights.T
    for offset, exponent in ((0.02, -0.5), (0.005, -1.0),
                             (0.08, -0.5), (0.001, 0.5)):
        if time.monotonic() > deadline - 4:
            return models
        basis = (offset + np.abs(projection - hints)) ** exponent
        design = np.column_stack([np.ones(len(x)), x - 0.5, basis])
        coef = np.linalg.lstsq(design, y, rcond=1e-12)[0]
        k, d = directions.shape
        try:
            model = OffsetPower(
                directions.copy(), hints.copy(), np.full(k, offset),
                np.full(k, exponent), coef[1 + d:], coef[0], coef[1:1 + d])
            model.refine(x, y, deadline - 1)
            models.append(model)
            if np.sqrt(np.mean((model.predict(x) - y) ** 2)) < scale * 1e-9:
                return models
        except (ValueError, np.linalg.LinAlgError, TimeoutError):
            continue
    for rate in (3.0, 10.0):
        if time.monotonic() > deadline - 4:
            return models
        basis = np.exp(-rate * np.abs(projection - hints))
        design = np.column_stack([np.ones(len(x)), x - 0.5, basis])
        coef = np.linalg.lstsq(design, y, rcond=1e-12)[0]
        try:
            model = ExponentialAbsolute(
                weights * rate, hints * rate, coef[1 + d:], coef[0],
                coef[1:1 + d])
            model.refine(x, y, deadline - 1)
            models.append(model)
            if np.sqrt(np.mean((model.predict(x) - y) ** 2)) < scale * 1e-9:
                return models
        except (ValueError, np.linalg.LinAlgError, TimeoutError):
            continue
    if time.monotonic() > deadline - 4:
        return models
    spline = SplineRidges(directions).fit(x, y)
    models.append(spline)
    projections = (x - 0.5) @ weights.T
    profile_z = [np.linspace(*np.quantile(projections[:, j], [0.01, 0.99]), 161)
                 for j in range(len(directions))]
    profiles = [BSpline(spline.knots, coef, 3)(z)
                for coef, z in zip(spline.curves, profile_z)]
    try:
        params = [profile_offset_power(z, profile, deadline - 3, hint)
                  for z, profile, hint in zip(profile_z, profiles, hints)]
        if all(p is not None for p in params):
            params = np.asarray(params)
            model = OffsetPower(
                directions, params[:, 3], np.exp(params[:, 4]), params[:, 5],
                params[:, 2], spline.constant + np.sum(params[:, 0]),
                spline.linear + params[:, 1] @ weights)
            model.refine(x, y, deadline - 1)
            models.append(model)
            if np.sqrt(np.mean((model.predict(x) - y) ** 2)) < scale * 1e-9:
                return models
    except (ValueError, np.linalg.LinAlgError, TimeoutError):
        pass
    if time.monotonic() > deadline - 4:
        return models
    try:
        params = [profile_exponential(z, profile)
                  for z, profile in zip(profile_z, profiles)]
        if all(p is not None for p in params):
            params = np.asarray(params)
            rates = np.abs(params[:, 3])
            model = ExponentialAbsolute(
                weights * rates[:, None], rates * params[:, 4], params[:, 2],
                spline.constant + np.sum(params[:, 0]),
                spline.linear + params[:, 1] @ weights, -np.sign(params[:, 3]))
            model.refine(x, y, deadline - 1)
            models.append(model)
    except (ValueError, np.linalg.LinAlgError, TimeoutError):
        pass
    return models
