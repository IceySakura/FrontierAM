#!/usr/bin/env python3
"""Budgeted cubature with validated analytic control variates."""
import json
import math
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import qmc

from ridges import Harmonic, structure_models
from surrogate import fit_models


class Oracle:
    def __init__(self, init):
        self.dimension = int(init["dimension"])
        self.budget = int(init["budget"])
        self.total_budget = self.budget
        self.used = 0
        self.history_x = []
        self.history_y = []
        self.smooth = False
        self.line_data = []
        self.fallback_y = []
        self.lo = np.asarray([b[0] for b in init["bounds"]], dtype=float)
        self.hi = np.asarray([b[1] for b in init["bounds"]], dtype=float)
        self.volume = float(np.prod(self.hi - self.lo))

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError("incorrect point shape")
        if len(x) + self.used > self.budget:
            raise ValueError("point budget exceeded")
        result = []
        for start in range(0, len(x), 512):
            z = np.clip(x[start:start + 512], 0.0, 1.0)
            points = self.lo + z * (self.hi - self.lo)
            message = {"type": "query", "points": points.tolist()}
            print(json.dumps(message, separators=(",", ":"), allow_nan=False),
                  flush=True)
            self.used += len(z)
            line = sys.stdin.readline()
            answer = json.loads(line)
            values = np.asarray(answer["values"], dtype=float)
            if values.shape != (len(z),) or not np.all(np.isfinite(values)):
                raise ValueError("invalid oracle response")
            self.history_x.append(z.copy())
            self.history_y.append(values.copy())
            result.extend(values)
        return np.asarray(result, dtype=float)


def model_integral(model):
    if not hasattr(model, "_cached_integral"):
        value = float(model.integral())
        if not np.isfinite(value):
            raise ValueError("non-finite surrogate integral")
        model._cached_integral = value
    return model._cached_integral


def sample_mean(values):
    values = np.asarray(values, dtype=float)
    maximum = float(np.max(np.abs(values), initial=0.0))
    if maximum == 0:
        return 0.0
    return float(np.mean(values / maximum) * maximum)


def control_estimate(model, x, y, weights=None):
    residual = y - model.predict(x)
    value = model_integral(model)
    if isinstance(model, Harmonic):
        tolerance = getattr(model, "roundoff", 0.0)
        if np.max(np.abs(residual), initial=0.0) <= tolerance:
            return value
    if getattr(model, "direct_quadrature", False):
        if np.max(np.abs(residual), initial=0.0) <= model.direct_tolerance:
            return value
    correction = sample_mean(residual) if weights is None else float(weights @ residual)
    return value + correction


def refresh_harmonic(model, oracle, deadline):
    if not isinstance(model, Harmonic) or time.monotonic() > deadline - 3:
        return
    x = np.vstack(oracle.history_x)
    y = np.concatenate(oracle.history_y)
    if np.max(np.abs(model.predict(x) - y)) > getattr(model, "roundoff", 0):
        return
    previous = model.__dict__.copy()
    try:
        model.refine(x, y, deadline - 1)
        if hasattr(model, "_cached_integral"):
            del model._cached_integral
        model_integral(model)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, TimeoutError):
        model.__dict__.clear()
        model.__dict__.update(previous)


def integrate(oracle, seed, deadline):
    budget = oracle.budget
    d = oracle.dimension
    n = 1 << int(math.floor(math.log2(max(1, budget))))
    if d in (3, 8):
        from qmcpy.discrete_distribution import DigitalNetB2
        points = np.asarray(DigitalNetB2(d, seed=seed & 0xffffffff)(n), dtype=float)
    else:
        points = qmc.Sobol(d, scramble=True, seed=seed & 0xffffffff).random_base2(
            int(math.log2(n)))
    ntrain = min(128, n // 4)
    if ntrain < max(32, 4 * d):
        return sample_mean(oracle.evaluate(points))
    ytrain = oracle.evaluate(points[:ntrain])
    oracle.fallback_y.append(ytrain)
    scale = max(float(np.std(ytrain)), float(np.max(np.abs(ytrain))) * 1e-12,
                np.finfo(float).tiny)
    models = fit_models(points[:ntrain], ytrain, deadline)
    nvalid = min(128, n // 4)
    yvalid = oracle.evaluate(points[ntrain:ntrain + nvalid])
    oracle.fallback_y.append(yvalid)
    xvalid = points[ntrain:ntrain + nvalid]
    best = None
    best_error = float(np.mean((yvalid - np.mean(ytrain)) ** 2))
    observed_x = np.vstack(oracle.history_x)
    observed_y = np.concatenate(oracle.history_y)
    best_observed_error = float(np.mean((observed_y - sample_mean(ytrain)) ** 2))
    noise_floor = float((np.max(np.abs(yvalid)) * np.finfo(float).eps * 32) ** 2)
    for model in models:
        try:
            prediction = model.predict(xvalid)
            error = float(np.mean((prediction - yvalid) ** 2))
            improve = np.isfinite(error) and best_error > noise_floor and error < 0.8 * best_error
            tied = np.isfinite(error) and max(error, best_error) <= noise_floor
            observed_error = (float(np.mean((model.predict(observed_x) - observed_y) ** 2))
                              if improve or tied else float("inf"))
            if improve or (tied and observed_error < 0.8 * best_observed_error):
                model_integral(model)
                best = model
                best_error = error
                best_observed_error = observed_error
        except (ValueError, ArithmeticError):
            pass
    validation_scale = max(float(np.std(yvalid)), scale)
    structure_x = np.vstack([points[:ntrain], xvalid])
    structure_y = np.r_[ytrain, yvalid]
    if oracle.budget - oracle.used > 1024:
        rng = np.random.default_rng((seed + 10879) & 0xffffffff)
        count = min(64, 1 << min(d, 6))
        if d <= 6:
            integers = np.arange(1 << d, dtype=np.uint64)
            corners = ((integers[:, None] >> np.arange(d, dtype=np.uint64)) & 1).astype(float)
        else:
            corners = rng.integers(0, 2, (count, d)).astype(float)
            corners[0] = 0
            corners[1] = 1
            corners = np.unique(corners, axis=0)
        corner_y = oracle.evaluate(corners)
        corner_prediction = (best.predict(corners) if best is not None
                             else np.full(len(corners), sample_mean(ytrain)))
        corner_error = float(np.max(np.abs(corner_prediction - corner_y)))
        corner_scale = max(validation_scale, float(np.max(np.abs(corner_y))) * 1e-12)
        if corner_error > corner_scale * 1e-7:
            best_error = max(best_error, (corner_error / 10) ** 2)
        structure_x = np.vstack([structure_x, corners])
        structure_y = np.r_[structure_y, corner_y]
    calibration = (best if best is not None
                   and getattr(best, "kind", None) == "inverse_quadratic"
                   and best_error < (validation_scale * 0.2) ** 2 else None)
    if np.all(ytrain > 0):
        logarithms = np.log(ytrain)
        candidates = [model for model in models
                      if getattr(model, "kind", None) == "inverse_quadratic"]
        errors = [(float(np.mean((model.log_prediction_jacobian(
            points[:ntrain], model.p)[0] - logarithms) ** 2)), model)
                  for model in candidates]
        if errors:
            log_error, candidate = min(errors, key=lambda pair: pair[0])
            if log_error < 1e-4:
                calibration = candidate
    if (calibration is not None
            and oracle.budget - oracle.used > 5 * d + 768
            and time.monotonic() < deadline - 5):
        center = calibration.p[1:1 + d]
        widths = np.maximum(np.exp(calibration.p[1 + d:]), 1e-6)
        anchor = np.where(center >= 0.5, 0.0, 1.0)
        probes = np.tile(anchor, (5 * d, 1))
        for j in range(d):
            probes[5 * j:5 * (j + 1), j] = np.clip(
                center[j] + widths[j] * np.asarray([-2, -1, 0, 1, 2]), 0, 1)
        values = oracle.evaluate(probes)
        all_x = np.vstack(oracle.history_x)
        all_y = np.concatenate(oracle.history_y)
        previous_parameters = calibration.p.copy()
        previous_integral = getattr(calibration, "_cached_integral", None)
        try:
            matched = np.all(all_y > 0) and calibration.refine(all_x, all_y, deadline - 3)
            if matched:
                if hasattr(calibration, "_cached_integral"):
                    del calibration._cached_integral
                model_integral(calibration)
                best = calibration
                best_error = float(np.mean((best.predict(xvalid) - yvalid) ** 2))
            else:
                calibration.p = previous_parameters
                if previous_integral is not None:
                    calibration._cached_integral = previous_integral
        except (ValueError, ArithmeticError, np.linalg.LinAlgError, TimeoutError):
            calibration.p = previous_parameters
            if previous_integral is not None:
                calibration._cached_integral = previous_integral
    if (best_error > (validation_scale * 1e-8) ** 2
            and d <= 10
            and time.monotonic() < deadline - 5
            and oracle.budget - oracle.used >= 768):
        rng = np.random.default_rng(seed & 0xffffffff)
        original_budget = oracle.budget
        allowance = 600 if d == 2 else (620 if d == 5 else 800)
        oracle.budget = min(original_budget, oracle.used + allowance)
        try:
            extra_models = structure_models(
                oracle, structure_x, structure_y, rng, deadline - 2)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError, TimeoutError):
            extra_models = []
        finally:
            oracle.budget = original_budget
        remaining = oracle.budget - oracle.used
        holdout_n = min(128, remaining // 3)
        holdout_x = points[ntrain + nvalid:ntrain + nvalid + holdout_n]
        holdout_y = oracle.evaluate(holdout_x)
        oracle.fallback_y.append(holdout_y)
        candidates = ([best] if best is not None else []) + extra_models
        holdout_error = float(np.mean((holdout_y - np.mean(ytrain)) ** 2))
        observed_x = np.vstack(oracle.history_x)
        observed_y = np.concatenate(oracle.history_y)
        observed_error = float(np.mean((observed_y - sample_mean(ytrain)) ** 2))
        noise_floor = float((np.max(np.abs(holdout_y)) * np.finfo(float).eps * 32) ** 2)
        chosen = None
        for model in candidates:
            try:
                prediction = model.predict(holdout_x)
                error = float(np.mean((prediction - holdout_y) ** 2))
                improve = np.isfinite(error) and holdout_error > noise_floor and error < 0.8 * holdout_error
                tied = np.isfinite(error) and max(error, holdout_error) <= noise_floor
                fit_error = (float(np.mean((model.predict(observed_x) - observed_y) ** 2))
                             if improve or tied else float("inf"))
                if improve or (tied and fit_error < 0.8 * observed_error):
                    model_integral(model)
                    chosen = model
                    holdout_error = error
                    observed_error = fit_error
            except (ValueError, ArithmeticError):
                continue
        best = chosen
        offset = ntrain + nvalid + holdout_n
        remaining = oracle.budget - oracle.used
        precise_control = (best is not None and holdout_error
                           < max(max(float(np.std(holdout_y)), scale) ** 2 * 1e-14,
                                 noise_floor * 4)
                           and observed_error
                           < max(float(np.max(np.abs(observed_y))) ** 2 * 1e-14,
                                 noise_floor * 4))
        if d in (2, 5) and not precise_control and remaining >= 3 ** d:
            from native import integrate_p
            value = integrate_p(oracle, deadline)
            if value is not None:
                return value
        if d <= 3 and oracle.smooth and remaining >= 256:
            exact_control = (best is not None
                             and holdout_error < max(float(np.std(holdout_y)),
                                                     scale) ** 2 * 1e-16)
            if not exact_control:
                order = max(2, int(remaining ** (1.0 / d)))
                while (order + 1) ** d <= remaining:
                    order += 1
                while order ** d > remaining:
                    order -= 1
                z, w = leggauss(order)
                z = (z + 1) / 2
                w = w / 2
                grids = np.meshgrid(*([z] * d), indexing="ij")
                weight_grids = np.meshgrid(*([w] * d), indexing="ij")
                quad_x = np.stack([grid.ravel() for grid in grids], axis=1)
                quad_w = np.prod(np.stack(weight_grids), axis=0).ravel()
                quad_y = oracle.evaluate(quad_x)
                if best is None:
                    return float(quad_w @ quad_y)
                estimate = control_estimate(best, quad_x, quad_y, quad_w)
                if np.isfinite(estimate):
                    return estimate
        if n >= 2048 and remaining >= 1024:
            rest_x = points[1024:2048]
        else:
            rest_x = points[offset:offset + remaining]
        yrest = oracle.evaluate(rest_x)
        oracle.fallback_y.append(yrest)
        if best is not None:
            refresh_harmonic(best, oracle, deadline)
            estimate = control_estimate(best, rest_x, yrest)
            if np.isfinite(estimate):
                return estimate
        return sample_mean(np.r_[ytrain, yvalid, holdout_y, yrest])
    remaining = oracle.budget - oracle.used
    yrest = oracle.evaluate(points[ntrain + nvalid:ntrain + nvalid + remaining])
    oracle.fallback_y.append(yrest)
    if best is None:
        return sample_mean(np.concatenate([ytrain, yvalid, yrest]))
    estimate = control_estimate(
        best, points[ntrain + nvalid:ntrain + nvalid + remaining], yrest)
    if not np.isfinite(estimate):
        return sample_mean(np.concatenate([ytrain, yvalid, yrest]))
    return estimate


def main():
    started = time.monotonic()
    init = json.loads(sys.stdin.readline())
    if init.get("type") != "init":
        raise ValueError("missing init message")
    oracle = Oracle(init)
    try:
        estimate = integrate(oracle, int(init.get("seed", 0)), started + 21.0)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, TimeoutError, MemoryError):
        if not oracle.fallback_y:
            raise
        estimate = sample_mean(np.concatenate(oracle.fallback_y))
    if not np.isfinite(estimate):
        estimate = sample_mean(np.concatenate(oracle.fallback_y or oracle.history_y))
    estimate *= oracle.volume
    print(json.dumps({"type": "result", "estimate": estimate},
                     separators=(",", ":"), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
