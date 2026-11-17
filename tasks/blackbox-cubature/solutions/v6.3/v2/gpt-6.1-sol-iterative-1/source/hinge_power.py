"""Integrable asymmetric ridge powers for continuous piecewise-smooth controls."""
import itertools
import math
import time
from decimal import Decimal, localcontext

import numpy as np
from scipy.optimize import least_squares


class HingePowers:
    def __init__(self, directions, biases, powers):
        self.directions = np.asarray(directions, dtype=float).copy()
        self.biases = np.asarray(biases, dtype=float).copy()
        self.powers = np.asarray(powers, dtype=float).copy()

    def basis(self, x):
        widths = np.maximum(np.sum(np.abs(self.directions), axis=1), 1e-12)
        z = 2 * ((x - 0.5) @ self.directions.T) / widths - self.biases
        return np.maximum(z, 0) ** self.powers

    def predict(self, x):
        return self.constant + (x - 0.5) @ self.linear + self.basis(x) @ self.amplitudes

    def integral(self):
        result = self.constant
        for direction, bias, power, amplitude in zip(
                self.directions, self.biases, self.powers, self.amplitudes):
            widths = 2 * np.abs(direction) / np.sum(np.abs(direction))
            widths = widths[widths > 1e-13]
            threshold = np.sum(widths) / 2 - bias
            if threshold <= 0:
                continue
            with localcontext() as context:
                context.prec = 65
                weights = [Decimal.from_float(float(w)) for w in widths]
                t = Decimal.from_float(float(threshold))
                p = Decimal.from_float(float(power))
                numerator = Decimal(0)
                for bits in itertools.product((0, 1), repeat=len(weights)):
                    z = t - sum((w for w, bit in zip(weights, bits) if bit), Decimal(0))
                    if z > 0:
                        term = z ** (p + len(weights))
                        numerator += -term if sum(bits) % 2 else term
                denominator = Decimal(1)
                for w in weights:
                    denominator *= w
                for j in range(1, len(weights) + 1):
                    denominator *= p + j
                result += amplitude * float(numerator / denominator)
        return float(result)

    def fit(self, x, y, deadline):
        k, d = self.directions.shape
        u = x - 0.5
        design = np.column_stack([np.ones(len(x)), u, self.basis(x)])
        coef = np.linalg.lstsq(design, y, rcond=1e-12)[0]
        start = 1 + d + k
        bias_start = start + k * d
        initial = np.r_[coef, self.directions.ravel(), self.biases, self.powers]
        lower = np.r_[np.full(bias_start, -np.inf), np.full(k, -2), np.full(k, 0.3)]
        upper = np.r_[np.full(bias_start, np.inf), np.full(k, 2), np.full(k, 5)]
        scale = max(float(np.std(y)), float(np.max(np.abs(y))) * 1e-10)

        def residual_jacobian(p):
            if time.monotonic() > deadline:
                raise TimeoutError("hinge power fit time limit")
            amplitude = p[1 + d:start]
            v = p[start:bias_start].reshape(k, d)
            bias = p[bias_start:bias_start + k]
            powers = p[bias_start + k:]
            widths = np.maximum(np.sum(np.abs(v), axis=1), 1e-12)
            projection = 2 * (u @ v.T) / widths
            z = projection - bias
            positive = np.maximum(z, 1e-150)
            basis = np.where(z > 0, positive ** powers, 0)
            derivative = np.where(z > 0, amplitude * powers * positive ** (powers - 1), 0)
            dz = (2 * u[:, None, :] - projection[:, :, None] * np.sign(v)) / widths[None, :, None]
            prediction = p[0] + u @ p[1:1 + d] + basis @ amplitude
            jac = np.column_stack([
                np.ones(len(x)), u, basis,
                (derivative[:, :, None] * dz).reshape(len(x), k * d),
                -derivative, amplitude * basis * np.log(positive)])
            return (prediction - y) / scale, jac / scale

        result = least_squares(lambda p: residual_jacobian(p)[0], initial,
                               jac=lambda p: residual_jacobian(p)[1], bounds=(lower, upper),
                               max_nfev=100, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        p = result.x
        self.constant = p[0]
        self.linear = p[1:1 + d]
        self.amplitudes = p[1 + d:start]
        self.directions = p[start:bias_start].reshape(k, d)
        self.biases = p[bias_start:bias_start + k]
        self.powers = p[bias_start + k:]
        return self


def fit_hinge_powers(oracle, gradients, deadline):
    if time.monotonic() > deadline - 5:
        return []
    x = np.vstack(oracle.history_x)
    y = np.concatenate(oracle.history_y)
    _, singular, vh = np.linalg.svd(gradients - np.mean(gradients, axis=0),
                                    full_matrices=False)
    if singular[0] < 1e-12:
        return []
    rank = min(3, x.shape[1], int(np.sum(singular > singular[0] * 1e-5)))
    if rank == 0:
        return []
    directions = np.vstack([vh[:rank], -vh[:rank]])
    models = []
    for power in (2.0, 1.0, 0.5):
        if time.monotonic() > deadline - 4:
            break
        try:
            model = HingePowers(directions, np.zeros(len(directions)),
                                np.full(len(directions), power)).fit(x, y, deadline - 2)
            models.append(model)
            if np.max(np.abs(model.predict(x) - y)) < max(np.std(y), 1e-15) * 1e-8:
                break
        except (ValueError, ArithmeticError, np.linalg.LinAlgError, TimeoutError):
            continue
    return models
