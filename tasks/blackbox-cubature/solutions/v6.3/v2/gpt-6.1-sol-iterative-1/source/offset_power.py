"""Ridge controls with learned offset, cusp location, and real exponent."""
import itertools
import math
import time
from decimal import Decimal, localcontext

import numpy as np
from scipy.optimize import least_squares


def offset_power_integral(direction, bias, offset, exponent):
    weights = 2 * np.abs(direction) / max(np.sum(np.abs(direction)), 1e-15)
    weights = weights[weights > 1e-13]
    if len(weights) == 0:
        return float((offset + abs(bias)) ** exponent)
    with localcontext() as context:
        context.prec = 75
        w = [Decimal.from_float(float(v)) for v in weights]
        a = Decimal.from_float(float(offset))
        p = Decimal.from_float(float(exponent))
        n = len(w)
        center = sum(w, Decimal(0)) / 2 + Decimal.from_float(float(bias))
        integer = round(float(p))
        singular = integer < 0 and abs(float(p) - integer) < 1e-12
        if singular:
            p = Decimal(integer)

        def primitive(order, z):
            if order == 0:
                return z ** p
            if singular and order >= -integer:
                q = -integer
                degree = order - q
                harmonic = sum((Decimal(1) / j for j in range(1, degree + 1)),
                               Decimal(0))
                coefficient = Decimal((-1) ** (q - 1)) / (
                    math.factorial(q - 1) * math.factorial(degree))
                return coefficient * z ** degree * (z.ln() - harmonic)
            denominator = Decimal(1)
            for j in range(1, order + 1):
                denominator *= p + j
            return z ** (p + order) / denominator

        matching = []
        for m in range(n):
            factor = (-1) ** (n + m) - 1
            matching.append(Decimal(factor) * primitive(n - m, a)
                            / math.factorial(m))

        def antiderivative(z):
            if z < 0:
                return Decimal((-1) ** n) * primitive(n, a - z)
            result = primitive(n, a + z)
            for m, coefficient in enumerate(matching):
                if coefficient:
                    result += coefficient * z ** m
            return result

        total = Decimal(0)
        for bits in itertools.product((0, 1), repeat=n):
            z = sum((v for v, bit in zip(w, bits) if bit), Decimal(0)) - center
            value = antiderivative(z)
            total += -value if (sum(bits) + n) % 2 else value
        denominator = Decimal(1)
        for v in w:
            denominator *= v
        return float(total / denominator)


class OffsetPower:
    def __init__(self, directions, biases, offsets, exponents, amplitudes,
                 constant=0.0, linear=None):
        self.directions = np.asarray(directions, dtype=float)
        self.biases = np.asarray(biases, dtype=float)
        self.offsets = np.asarray(offsets, dtype=float)
        self.exponents = np.asarray(exponents, dtype=float)
        self.amplitudes = np.asarray(amplitudes, dtype=float)
        self.constant = float(constant)
        self.linear = (np.zeros(self.directions.shape[1]) if linear is None
                       else np.asarray(linear, dtype=float))

    def predict(self, x):
        widths = np.maximum(np.sum(np.abs(self.directions), axis=1), 1e-12)
        u = x - 0.5
        z = 2 * (u @ self.directions.T) / widths - self.biases
        basis = np.exp(self.exponents * np.log(self.offsets + np.abs(z)))
        return self.constant + u @ self.linear + basis @ self.amplitudes

    def integral(self):
        result = self.constant
        for v, b, a, p, amplitude in zip(
                self.directions, self.biases, self.offsets, self.exponents,
                self.amplitudes):
            result += amplitude * offset_power_integral(v, b, a, p)
        return float(result)

    def refine(self, x, y, deadline):
        k, d = self.directions.shape
        u = x - 0.5
        p = np.r_[self.constant, self.linear, self.amplitudes,
                  self.directions.ravel(), self.biases, np.log(self.offsets),
                  self.exponents]
        vector_start = 1 + d + k
        bias_start = vector_start + k * d
        lower = np.r_[np.full(bias_start, -np.inf),
                      np.full(k, -3.0), np.full(k, -25.0), np.full(k, -4.0)]
        upper = np.r_[np.full(bias_start, np.inf),
                      np.full(k, 3.0), np.full(k, 5.0), np.full(k, 4.0)]
        scale = max(float(np.std(y)), float(np.max(np.abs(y))) * 1e-10)

        def residual_jacobian(p):
            if time.monotonic() > deadline:
                raise TimeoutError("offset profile time limit")
            amplitude = p[1 + d:vector_start]
            v = p[vector_start:bias_start].reshape(k, d)
            biases = p[bias_start:bias_start + k]
            offsets = np.exp(p[bias_start + k:bias_start + 2 * k])
            exponents = p[bias_start + 2 * k:]
            widths = np.maximum(np.sum(np.abs(v), axis=1), 1e-12)
            projection = 2 * (u @ v.T) / widths
            z = projection - biases
            denominator = offsets + np.abs(z)
            logarithm = np.log(denominator)
            basis = np.exp(exponents * logarithm)
            derivative = amplitude * exponents * basis / denominator * np.sign(z)
            dzdv = (2 * u[:, None, :] - projection[:, :, None]
                    * np.sign(v)[None, :, :]) / widths[None, :, None]
            prediction = p[0] + u @ p[1:1 + d] + basis @ amplitude
            jac = np.column_stack([
                np.ones(len(x)), u, basis,
                (derivative[:, :, None] * dzdv).reshape(len(x), k * d),
                -derivative,
                amplitude * exponents * basis / denominator * offsets,
                amplitude * basis * logarithm])
            return (prediction - y) / scale, jac / scale

        result = least_squares(lambda p: residual_jacobian(p)[0], p,
                               jac=lambda p: residual_jacobian(p)[1],
                               bounds=(lower, upper), max_nfev=150,
                               ftol=1e-12, xtol=1e-12, gtol=1e-12)
        p = result.x
        self.constant = p[0]
        self.linear = p[1:1 + d]
        self.amplitudes = p[1 + d:vector_start]
        self.directions = p[vector_start:bias_start].reshape(k, d)
        self.biases = p[bias_start:bias_start + k]
        self.offsets = np.exp(p[bias_start + k:bias_start + 2 * k])
        self.exponents = p[bias_start + 2 * k:]
        return self


def profile_offset_power(z, y, deadline, center_hint=None):
    scale = max(float(np.std(y)), np.finfo(float).tiny)
    linear = np.linalg.lstsq(np.column_stack([np.ones(len(z)), z]), y,
                             rcond=1e-12)[0]
    residual = y - linear[0] - linear[1] * z
    curvature = np.diff(y, n=2)
    middle = np.arange(len(curvature))
    valid = middle[(middle > len(curvature) * 0.1)
                   & (middle < len(curvature) * 0.9)]
    c_curvature = float(z[1 + valid[np.argmax(np.abs(curvature[valid]))]])
    if center_hint is not None:
        c_curvature = float(center_hint)
    c_residual = float(z[np.argmax(np.abs(residual))])
    best = None
    for exponent, center, offset in ((-0.5, c_curvature, 0.05),
                                     (-1.0, c_curvature, 0.2),
                                     (0.5, c_curvature, 0.01),
                                     (1.0, c_curvature, 0.05),
                                     (-1.0, c_residual, 0.1),
                                     (-0.5, 0.0, 0.1)):
        if time.monotonic() > deadline:
            break
        basis = (offset + np.abs(z - center)) ** exponent
        design = np.column_stack([np.ones(len(z)), z, basis])
        coef = np.linalg.lstsq(design, y, rcond=1e-12)[0]
        initial = np.r_[coef, center, math.log(offset), exponent]

        def residual_jacobian(p):
            u = z - p[3]
            a = math.exp(p[4])
            denominator = a + np.abs(u)
            logarithm = np.log(denominator)
            basis = np.exp(p[5] * logarithm)
            derivative = p[2] * p[5] * basis / denominator
            prediction = p[0] + p[1] * z + p[2] * basis
            jac = np.column_stack([
                np.ones(len(z)), z, basis, -derivative * np.sign(u),
                derivative * a, p[2] * basis * logarithm])
            return (prediction - y) / scale, jac / scale

        try:
            result = least_squares(
                lambda p: residual_jacobian(p)[0], initial,
                jac=lambda p: residual_jacobian(p)[1],
                bounds=([-np.inf, -np.inf, -np.inf, -2, -15, -4],
                        [np.inf, np.inf, np.inf, 2, 5, 4]),
                max_nfev=100, ftol=1e-11, xtol=1e-11, gtol=1e-11)
            error = np.linalg.norm(result.fun)
            if best is None or error < best[0]:
                best = error, result.x
        except ValueError:
            continue
    return None if best is None else best[1]
