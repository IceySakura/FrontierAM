"""Small, integrable function spaces fitted only to oracle observations."""
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
from scipy.special import dawsn, erfcx, log_ndtr


class Polynomial:
    def __init__(self, d):
        self.d = d
        self.pairs = [(i, j) for i in range(d) for j in range(i, d)]

    def features(self, x):
        u = x - 0.5
        return np.column_stack([np.ones(len(x)), u]
                               + [u[:, i] * u[:, j] for i, j in self.pairs])

    def fit(self, x, y):
        self.coef = np.linalg.lstsq(self.features(x), y, rcond=1e-12)[0]
        return self

    def predict(self, x):
        return self.features(x) @ self.coef

    def integral(self):
        result = self.coef[0]
        for k, (i, j) in enumerate(self.pairs):
            if i == j:
                result += self.coef[1 + self.d + k] / 12.0
        return float(result)


class ExponentialQuadratic:
    def fit(self, x, y):
        u = x - 0.5
        design = np.column_stack([np.ones(len(x)), u, u * u])
        self.coef = np.linalg.lstsq(design, np.log(y), rcond=1e-12)[0]
        self.d = x.shape[1]
        return self

    def predict(self, x):
        u = x - 0.5
        logy = (self.coef[0] + u @ self.coef[1:1 + self.d]
                + (u * u) @ self.coef[1 + self.d:])
        return np.exp(np.clip(logy, -745.0, 700.0))

    def integral(self):
        result = self.coef[0]
        for linear, quadratic in zip(self.coef[1:1 + self.d],
                                     self.coef[1 + self.d:]):
            result += quadratic_log_integral(float(quadratic), float(linear))
        return float(np.exp(result))


def quadratic_log_integral(a, b):
    if abs(a) < 1e-9:
        if abs(b) < 1e-8:
            return a / 12 + a * a / 360 + b * b / 24.0
        magnitude = abs(b)
        if magnitude < 0.1:
            moment = 1 / 12 + magnitude * magnitude / 360 - magnitude ** 4 / 15120
        else:
            coth = 1.0 if magnitude > 700 else 1 + 2 / np.expm1(magnitude)
            moment = 0.25 - coth / magnitude + 2 / (magnitude * magnitude)
        return (magnitude / 2 + np.log(-np.expm1(-magnitude))
                - np.log(magnitude) + a * moment)
    if a < 0:
        rate = np.sqrt(-a)
        center = -b / (2 * a)
        t_low = rate * (-0.5 - center)
        t_high = rate * (0.5 - center)
        if t_low > 0 or t_high < 0:
            near = min(abs(t_low), abs(t_high))
            far = max(abs(t_low), abs(t_high))
            difference = erfcx(near) - np.exp(-abs(b)) * erfcx(far)
            return (a / 4 + abs(b) / 2 + 0.5 * np.log(np.pi)
                    - np.log(2 * rate) + np.log(difference))
        low = np.sqrt(2) * rate * (-0.5 - center)
        high = np.sqrt(2) * rate * (0.5 - center)
        if low > 0:
            log_high = log_ndtr(-low)
            log_low = log_ndtr(-high)
        else:
            log_high = log_ndtr(high)
            log_low = log_ndtr(low)
        probability = log_high + np.log(-np.expm1(log_low - log_high))
        return -b * b / (4 * a) + 0.5 * np.log(np.pi) - np.log(rate) + probability
    rate = np.sqrt(a)
    center = -b / (2 * a)
    low = rate * (-0.5 - center)
    high = rate * (0.5 - center)
    maximum = max(low * low, high * high)
    difference = (np.exp(high * high - maximum) * dawsn(high)
                  - np.exp(low * low - maximum) * dawsn(low))
    if difference > 0:
        return a / 4 + abs(b) / 2 - np.log(rate) + np.log(difference)
    z, w = leggauss(96)
    values = a * (z / 2) ** 2 + b * z / 2
    maximum = np.max(values)
    return maximum + np.log(np.dot(np.exp(values - maximum), w / 2))


class SeparableNonlinear:
    def __init__(self, d, kind):
        self.d = d
        self.kind = kind

    def log_prediction_jacobian(self, x, p):
        d = self.d
        c = p[1:1 + d]
        u = x - c
        if self.kind == "absolute_exponential":
            b = p[1 + d:]
            v = np.abs(u)
            logy = p[0] + v @ b
            jac = np.column_stack([np.ones(len(x)), -np.sign(u) * b, v])
        elif self.kind == "inverse_quadratic":
            a = np.exp(p[1 + d:])
            denom = u * u + a * a
            logy = p[0] - np.sum(np.log(denom), axis=1)
            jac = np.column_stack([np.ones(len(x)), 2.0 * u / denom,
                                   -2.0 * a * a / denom])
        else:
            a = np.exp(p[1 + d:])
            denom = a + np.abs(u)
            logy = p[0] + np.sum(np.log(denom), axis=1)
            jac = np.column_stack([np.ones(len(x)), -np.sign(u) / denom,
                                   a / denom])
        return logy, jac

    def fit(self, x, y, center=0.5):
        d = self.d
        logy = np.log(y)
        if self.kind == "absolute_exponential":
            design = np.column_stack([np.ones(len(x)), np.abs(x - center)])
            coef = np.linalg.lstsq(design, logy, rcond=1e-12)[0]
            initial = np.r_[coef[0], np.full(d, center), coef[1:]]
            lower = np.r_[-np.inf, np.zeros(d), np.full(d, -500.0)]
            upper = np.r_[np.inf, np.ones(d), np.full(d, 500.0)]
        else:
            initial = np.r_[0.0, np.full(d, center), np.full(d, np.log(0.2))]
            pred, _ = self.log_prediction_jacobian(x, initial)
            initial[0] = np.mean(logy - pred)
            lower = np.r_[-np.inf, np.zeros(d), np.full(d, -12.0)]
            upper = np.r_[np.inf, np.ones(d), np.full(d, 5.0)]
        result = least_squares(
            lambda p: self.log_prediction_jacobian(x, p)[0] - logy, initial,
            jac=lambda p: self.log_prediction_jacobian(x, p)[1],
            bounds=(lower, upper), max_nfev=100, ftol=1e-11, xtol=1e-11,
            gtol=1e-11)
        self.p = result.x
        return self

    def predict(self, x):
        logy, _ = self.log_prediction_jacobian(x, self.p)
        return np.exp(np.clip(logy, -745.0, 700.0))

    def refine(self, x, y, deadline):
        d = self.d
        original = self.p.copy()
        origin = original[1:1 + d].copy()
        scales = np.maximum(np.exp(original[1 + d:]), 1e-5)
        initial = np.r_[original[0], np.zeros(d), original[1 + d:]]
        lower = np.r_[-np.inf, -origin / scales, np.full(d, -36.0)]
        upper = np.r_[np.inf, (1 - origin) / scales, np.full(d, 5.0)]
        target = np.log(y)

        def physical(q):
            p = q.copy()
            p[1:1 + d] = origin + scales * q[1:1 + d]
            return p

        def residual(q):
            if time.monotonic() > deadline:
                raise TimeoutError("peak calibration time limit")
            return self.log_prediction_jacobian(x, physical(q))[0] - target

        def jacobian(q):
            jac = self.log_prediction_jacobian(x, physical(q))[1]
            jac[:, 1:1 + d] *= scales
            return jac

        result = least_squares(residual, initial, jac=jacobian,
                               bounds=(lower, upper), x_scale="jac",
                               max_nfev=150, ftol=1e-13, xtol=1e-14, gtol=1e-12)
        self.p = physical(result.x)
        if np.max(np.abs(residual(result.x))) > 2e-7:
            self.p = original
            return False
        return True

    def integral(self):
        d = self.d
        c = self.p[1:1 + d]
        if self.kind == "absolute_exponential":
            b = self.p[1 + d:]
            factors = np.ones(d)
            mask = np.abs(b) > 1e-8
            factors[mask] = (np.expm1(b[mask] * c[mask])
                             + np.expm1(b[mask] * (1 - c[mask]))) / b[mask]
        elif self.kind == "inverse_quadratic":
            a = np.exp(self.p[1 + d:])
            factors = (np.arctan((1 - c) / a) + np.arctan(c / a)) / a
        else:
            a = np.exp(self.p[1 + d:])
            factors = a + (c * c + (1 - c) ** 2) / 2.0
        return float(np.exp(self.p[0] + np.sum(np.log(factors))))


def fit_models(x, y, deadline):
    d = x.shape[1]
    models = []
    if len(y) >= 2 * (1 + d + d * (d + 1) // 2):
        models.append(Polynomial(d).fit(x, y))
    if np.all(y > 0) and time.monotonic() < deadline - 2.0:
        try:
            models.append(ExponentialQuadratic().fit(x, y))
        except (ValueError, np.linalg.LinAlgError):
            pass
        for kind in ("absolute_exponential", "inverse_quadratic", "absolute_linear"):
            if time.monotonic() > deadline - 4.0:
                break
            for center in (0.5, 0.3):
                if time.monotonic() > deadline - 4.0:
                    break
                try:
                    model = SeparableNonlinear(d, kind).fit(x, y, center)
                    models.append(model)
                    err = np.max(np.abs(np.log(model.predict(x)) - np.log(y)))
                    if err < 1e-8:
                        break
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    continue
    return models
