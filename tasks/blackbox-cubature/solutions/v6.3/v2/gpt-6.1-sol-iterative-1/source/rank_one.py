"""Adaptive one-dimensional quadrature for a validated active direction."""
import heapq
import itertools
import math
import time
from decimal import Decimal, localcontext

import numpy as np
from numpy.polynomial import legendre
from scipy.integrate._rules import GaussKronrodQuadrature
from scipy.optimize import least_squares


_rule = GaussKronrodQuadrature(15)
_nodes, _weights = _rule.nodes_and_weights
_lower_nodes, _lower_weights = _rule.lower_nodes_and_weights
_lower_indices = np.argmin(np.abs(_nodes[:, None] - _lower_nodes[None, :]), axis=0)


class ProjectedDensity:
    def __init__(self, direction):
        w = np.abs(direction)
        w = 2 * w / np.sum(w)
        self.weights = w[w > 1e-13]
        self.d = len(self.weights)
        sums = [(0.0, 1.0)]
        for weight in self.weights:
            sums += [(s + weight, -sign) for s, sign in sums]
        self.subsets = np.asarray([s for s, _ in sums], dtype=np.longdouble)
        self.signs = np.asarray([sign for _, sign in sums], dtype=np.longdouble)
        self.denominator = (np.longdouble(math.factorial(max(0, self.d - 1)))
                            * np.prod(self.weights.astype(np.longdouble)))
        self.use_decimal = bool(self.denominator < 1e-9)

    def values(self, x):
        if self.d == 1:
            return np.full(len(x), 0.5)
        lower = np.maximum(0, 1 - np.abs(np.asarray(x, dtype=float)))
        if self.use_decimal:
            result = []
            with localcontext() as context:
                context.prec = 55
                weights = [Decimal.from_float(float(a)) for a in self.weights]
                subsets = [(Decimal(0), 1)]
                for weight in weights:
                    subsets += [(s + weight, -sign) for s, sign in subsets]
                denominator = Decimal(math.factorial(self.d - 1))
                for weight in weights:
                    denominator *= weight
                for value in lower:
                    t = Decimal.from_float(float(value))
                    total = Decimal(0)
                    for s, sign in subsets:
                        if t > s:
                            total += sign * (t - s) ** (self.d - 1)
                    result.append(max(0.0, float(total / denominator)))
            return np.asarray(result)
        powers = np.maximum(
            lower.astype(np.longdouble)[:, None] - self.subsets, 0) ** (self.d - 1)
        values = (powers @ self.signs) / self.denominator
        return np.maximum(np.asarray(values, dtype=float), 0)

    def breaks(self):
        if self.d <= 4:
            points = np.asarray(self.subsets, dtype=float) - 1
            return np.unique(np.clip(np.r_[-1.0, points, 1.0], -1, 1))
        return np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])


class ActiveQuadrature:
    def __init__(self, direction, regions):
        self.direction = np.asarray(direction, dtype=float).copy()
        self.regions = sorted(regions, key=lambda region: region["left"])
        self.ends = np.asarray([region["right"] for region in self.regions])
        self.linear = np.zeros(len(direction))
        self.direct_tolerance = 0.0
        self.direct_quadrature = True

    def profile(self, z, derivative=False):
        z = np.clip(np.asarray(z, dtype=float), -1, 1)
        indices = np.clip(np.searchsorted(self.ends, z), 0, len(self.regions) - 1)
        result = np.empty(len(z))
        for index in np.unique(indices):
            mask = indices == index
            region = self.regions[index]
            half = (region["right"] - region["left"]) / 2
            mid = (region["right"] + region["left"]) / 2
            t = (z[mask] - mid) / half
            if derivative:
                result[mask] = legendre.legval(t, legendre.legder(region["coef"])) / half
            else:
                result[mask] = legendre.legval(t, region["coef"])
        return result

    def predict(self, x):
        u = x - 0.5
        z = 2 * (u @ self.direction) / np.sum(np.abs(self.direction))
        return self.profile(z) + u @ self.linear

    def integral(self):
        density = ProjectedDensity(self.direction)
        total = 0.0
        mass = 0.0
        for region in self.regions:
            half = (region["right"] - region["left"]) / 2
            pdf = density.values(region["nodes"])
            total += half * np.dot(_weights, region["values"] * pdf)
            mass += half * np.dot(_weights, pdf)
        if mass <= 0 or not np.isfinite(total):
            raise ValueError("invalid projected quadrature")
        return float(total / mass)

    def refine(self, x, y, deadline):
        d = x.shape[1]
        u = x - 0.5
        signs = np.sign(self.direction)
        projector = np.eye(d) - np.outer(signs, signs) / np.dot(signs, signs)
        residual = y - self.predict(x)
        self.linear = np.linalg.lstsq(u @ projector, residual, rcond=1e-12)[0] @ projector
        initial_error = float(np.mean((self.predict(x) - y) ** 2))
        scale = max(float(np.std(y)), float(np.max(np.abs(y))) * 1e-10)
        initial = np.r_[self.direction, self.linear]

        def residual_jacobian(p):
            if time.monotonic() > deadline:
                raise TimeoutError("active direction time limit")
            direction = p[:d]
            width = max(float(np.sum(np.abs(direction))), 1e-12)
            z = 2 * (u @ direction) / width
            value = self.profile(z)
            derivative = self.profile(z, derivative=True)
            dz = (2 * u - z[:, None] * np.sign(direction)) / width
            prediction = value + (u @ projector) @ p[d:]
            jac = np.column_stack([derivative[:, None] * dz, u @ projector])
            return (prediction - y) / scale, jac / scale

        result = least_squares(lambda p: residual_jacobian(p)[0], initial,
                               jac=lambda p: residual_jacobian(p)[1],
                               max_nfev=25, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        new_error = float(np.mean(result.fun ** 2)) * scale * scale
        old_unit = 2 * self.direction / np.sum(np.abs(self.direction))
        new_unit = 2 * result.x[:d] / np.sum(np.abs(result.x[:d]))
        if new_error < initial_error * 0.8 and np.linalg.norm(new_unit - old_unit) < 0.002:
            self.direction = result.x[:d]
            self.linear = result.x[d:] @ projector
        self.direct_tolerance = max(float(np.max(np.abs(y))), scale) * 2e-9
        return self


def fit_active_quadrature(oracle, x, y, rng, deadline):
    d = x.shape[1]
    if d > 10 or oracle.budget - oracle.used < 16 * d + 768:
        return None
    safe = np.flatnonzero(np.all((x > 0.001) & (x < 0.999), axis=1))
    if len(safe) < 8:
        return None
    selected = safe[np.linspace(0, len(safe) - 1, 8, dtype=int)]
    anchors = x[selected]
    h = rng.uniform(4e-8, 8e-8, size=d)
    probes = np.repeat(anchors, 2 * d, axis=0)
    for j in range(d):
        probes[2 * j::2 * d, j] += h[j]
        probes[2 * j + 1::2 * d, j] -= h[j]
    values = oracle.evaluate(probes).reshape(8, d, 2)
    gradients = (values[:, :, 0] - values[:, :, 1]) / (2 * h)
    _, singular, vh = np.linalg.svd(gradients - np.mean(gradients, axis=0),
                                    full_matrices=False)
    if singular[0] < max(float(np.max(np.abs(y))) * 1e-8, 1e-12):
        return None
    if len(singular) > 1 and singular[1] > singular[0] * 2e-7:
        return None
    direction = vh[0]
    signs = np.sign(direction)
    signs[signs == 0] = 1
    density = ProjectedDensity(direction)
    reserve = 384
    counter = 0
    regions = {}
    heap = []

    def evaluate(intervals):
        nonlocal counter
        nodes = []
        for left, right in intervals:
            nodes.append((right + left) / 2 + (right - left) / 2 * _nodes)
        z = np.concatenate(nodes)
        points = 0.5 + np.outer(z, signs) / 2
        vals = oracle.evaluate(points).reshape(-1, 15)
        for (left, right), t, values in zip(intervals, nodes, vals):
            half = (right - left) / 2
            pdf = density.values(t)
            integrand = values * pdf
            integral = half * np.dot(_weights, integrand)
            lower = half * np.dot(_lower_weights, integrand[_lower_indices])
            mass = half * np.dot(_weights, pdf)
            mass_low = half * np.dot(_lower_weights, pdf[_lower_indices])
            coef = legendre.legfit(_nodes, values, 14)
            lower_coef = legendre.legfit(_nodes[_lower_indices], values[_lower_indices], 6)
            variation = float(np.max(np.abs(values - legendre.legval(_nodes, lower_coef))))
            error = (abs(integral - lower) + abs(mass - mass_low) * np.max(np.abs(values))
                     + variation * mass * 0.002)
            region = {"left": left, "right": right, "nodes": t, "values": values,
                      "integral": integral, "mass": mass, "error": error, "coef": coef}
            regions[counter] = region
            heapq.heappush(heap, (-error, counter))
            counter += 1

    breaks = density.breaks()
    intervals = [(left, right) for left, right in zip(breaks[:-1], breaks[1:])
                 if right - left > 1e-12]
    if oracle.budget - oracle.used < 15 * len(intervals) + reserve:
        intervals = [(-1, 0), (0, 1)]
    evaluate(intervals)
    scale = max(float(np.std(y)), float(np.max(np.abs(y))) * 1e-10)
    while oracle.budget - oracle.used >= 30 + reserve and time.monotonic() < deadline - 4:
        total_error = sum(region["error"] for region in regions.values())
        if total_error < scale * 2e-12:
            break
        _, index = heapq.heappop(heap)
        region = regions.pop(index)
        left, right = region["left"], region["right"]
        mid = (left + right) / 2
        evaluate([(left, mid), (mid, right)])
    model = ActiveQuadrature(direction, list(regions.values()))
    try:
        model.refine(x, y, deadline - 1)
    except (ValueError, np.linalg.LinAlgError, TimeoutError):
        pass
    return model
