import time
import ctypes
import os
from math import isfinite

import numpy as np

try:
    _native_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libcg.so"))
    _step = _native_lib.cg_step
    _step.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double] + [ctypes.c_void_p] * 4
    _step.restype = ctypes.c_double
except (OSError, AttributeError):
    _step = None
if os.environ.get("NO_NATIVE"):
    _step = None


class Solver:
    """Online CG with refreshed Galerkin recycling and deadline-aware work."""

    def __init__(self, dimension, seed, options):
        self.d = dimension
        self.history = []
        self.hvp_cost = 0.002
        self.exhausted = False
        self.settle = 0
        self.phase_age = 0
        self.phase_credit = 0.
        self.base_tick = None
        self.seen_shift = False
        self.recovery = False
        self.phase_reference = 1.
        self.direction = None
        self.fast_spectrum = False

    def solve(self, hvp, b, context):
        start = time.monotonic()
        bscale = float(np.max(np.abs(b)))
        if not bscale > 0 or not np.isfinite(bscale):
            return np.zeros_like(b)
        b = b / bscale
        remaining = context["steps"] - context["step"]
        condition = context["L"] / context["mu"]
        deadline = context["deadline"] - 0.10
        new_phase = bool(self.history and context["phase_changed"])
        if self.base_tick is None:
            self.base_tick = max(.001, (deadline - start) / remaining)
        if new_phase:
            self.seen_shift = True
        if not self.history or context["phase_changed"]:
            weight = 10.
            self.settle = 1
            self.phase_age = 0
            self.phase_credit = 0.
        elif self.settle:
            weight = 2.
            self.settle -= 1
            self.phase_age += 1
        else:
            weight = 1.
            self.phase_age += 1
        target = (1e-20 if condition <= 1e4 or self.fast_spectrum else 1e-12) if remaining > 1 else 0.
        base_stop = min(deadline, start + max(0., (deadline - start) * weight / (remaining + weight - 1)))
        extra = self.phase_credit * (.75 if self.phase_age == 1 else 1.) if 1 <= self.phase_age <= 3 else 0.
        stop = min(deadline, base_stop + extra)
        # Spend more on unresolved jumps, preserving a small follow-up budget.
        if new_phase and condition > 1e6 and remaining > 1:
            reserve = .25 * self.base_tick * (remaining - 1)
            available = max(0., deadline - start - reserve)
            budget = max(stop - start, min(8. * self.base_tick, available))
            stop = min(deadline, start + budget)
        unrestricted_stop = stop
        recovery_limited = self.recovery and self.phase_age >= 3 and remaining > 1
        if recovery_limited:
            stop = min(stop, start + .25 * self.base_tick)
        x = np.zeros_like(b)
        count = 0
        oracle_time = 0.
        iterations = 0
        bnorm2 = float(b @ b)
        x_limit = 8. * np.sqrt(bnorm2) / context["mu"]
        residual_limit = 64. * max(1., condition) * bnorm2
        rr = bnorm2
        initial_rr = 1.
        energy = 0.
        upper = 1. / context["mu"]
        p = None
        basis = hb = gram = eig = rotation = r = None

        def product(v):
            nonlocal count, oracle_time
            begin = time.monotonic()
            result = hvp(v)
            elapsed = time.monotonic() - begin
            oracle_time += elapsed
            columns = 1 if v.ndim == 1 else v.shape[1]
            count += columns
            if columns == 1:
                self.hvp_cost = 0.85 * self.hvp_cost + 0.15 * elapsed
            return result

        try:
            if self.exhausted or start >= deadline - max(.01, 2 * self.hvp_cost):
                fallback = self.history[0] if self.history else x
                if not np.isfinite(fallback).all() or np.linalg.norm(fallback) > x_limit:
                    return np.zeros_like(b)
                fallback = fallback * bscale
                return fallback if np.isfinite(fallback).all() else np.zeros_like(b)
            r = b.copy()
            if self.history:
                # All old directions are tested against the current operator.
                history = self.history
                if self.direction is not None:
                    history = history + [self.direction]
                basis = np.linalg.qr(np.column_stack(history), mode="reduced")[0]
                hb = product(basis)
                gram = basis.T @ hb
                gram = 0.5 * (gram + gram.T)
                eig, rotation = np.linalg.eigh(gram)
                coefficients = rotation @ ((rotation.T @ (basis.T @ b)) /
                                            np.maximum(eig, context["mu"] * 0.5))
                x = basis @ coefficients
                r -= hb @ coefficients
                energy = float(b @ x + r @ x)
            rr = float(r @ r)
            p = r.copy()
            if _step is not None:
                xp, rp, pp = x.ctypes.data, r.ctypes.data, p.ctypes.data
            initial_rr = rr / bnorm2
            while bnorm2 * 1e-24 < rr <= residual_limit:
                if time.monotonic() + max(.0015, 1.3 * self.hvp_cost) >= stop:
                    if (recovery_limited and stop < unrestricted_stop
                            and upper * rr > 32. * self.phase_reference * max(energy, 1e-300)):
                        self.recovery = False
                        recovery_limited = False
                        stop = unrestricted_stop
                        if time.monotonic() + max(.0015, 1.3 * self.hvp_cost) < stop:
                            continue
                    break
                hp = product(p)
                php = float(p @ hp)
                if not isfinite(php) or php <= 0:
                    break
                alpha = rr / php
                energy += alpha * rr
                if _step is None:
                    x += alpha * p
                    r -= alpha * hp
                    new_rr = float(r @ r)
                else:
                    new_rr = _step(self.d, alpha, rr, xp, rp, pp, hp.ctypes.data)
                if not isfinite(new_rr) or new_rr > residual_limit:
                    rr = new_rr
                    break
                beta = new_rr / rr
                if not isfinite(beta):
                    rr = new_rr
                    break
                # Gauss-Radau upper bound on the remaining squared energy error.
                delta = upper - alpha
                upper = delta / (context["mu"] * delta + beta) if delta > 0 else 1. / context["mu"]
                if _step is None:
                    p *= beta
                    p += r
                rr = new_rr
                iterations += 1
                if target and upper * rr < target * max(energy, 1e-300):
                    if context["step"] == 0 and target == 1e-12 and iterations <= 128:
                        self.fast_spectrum = True
                        target = 1e-20
                    else:
                        break
                if (self.fast_spectrum and condition > 1e4
                        and target == 1e-20 and iterations >= 256):
                    self.fast_spectrum = False
                    target = 1e-12
                    if upper * rr < target * max(energy, 1e-300):
                        break
        except Exception as error:
            if type(error).__name__ != "BudgetExceeded":
                raise
            self.exhausted = True
        error_bound = upper * rr
        if (self.fast_spectrum and condition > 1e4 and target == 1e-20
                and rr > bnorm2 * 1e-24 and error_bound > 1e-20 * max(energy, 1e-300)):
            self.fast_spectrum = False
        estimate = x
        estimate_rr = rr
        estimate_energy = energy
        estimate_error_bound = error_bound
        # Correct the output while retaining the original CG trajectory as state.
        if (hb is not None and eig is not None and r is not None
                and np.isfinite(rr) and rr > bnorm2 * 1e-28):
            projected = basis.T @ r
            correction = rotation @ ((rotation.T @ projected) /
                                     np.maximum(eig, context["mu"] * .5))
            slope = float(projected @ correction)
            curvature = float(correction @ (gram @ correction))
            if slope > 0 and curvature > 0 and np.isfinite(slope + curvature):
                scale = slope / curvature
                gain = scale * slope
                if np.isfinite(gain) and gain > 1e-22 * max(energy, 1e-300):
                    estimate = x + scale * (basis @ correction)
                    estimate_residual = r - scale * (hb @ correction)
                    estimate_energy = energy + gain
                    estimate_rr = float(estimate_residual @ estimate_residual)
                    residual_bound = estimate_rr / context["mu"]
                    estimate_error_bound = (min(residual_bound, max(0., error_bound - gain))
                                            if error_bound >= gain else residual_bound)
        valid_x = np.isfinite(x).all() and np.linalg.norm(x) <= x_limit
        if not valid_x:
            x = np.zeros_like(b)
            p = None
            estimate = x
        if not np.isfinite(estimate).all() or np.linalg.norm(estimate) > x_limit:
            estimate = x
        self.direction = None
        if p is not None and np.isfinite(p).all() and rr > bnorm2 * 1e-26:
            norm = np.linalg.norm(p)
            if norm > 0 and np.isfinite(norm):
                self.direction = p / norm
        if self.phase_age == 0:
            self.phase_credit = max(0., base_stop - time.monotonic())
            self.phase_reference = min(1., max(1e-20, error_bound / max(energy, 1e-300)))
            self.recovery = bool(self.seen_shift and condition > 1e6
                                 and error_bound > 1e-3 * max(energy, 1e-300))
        elif self.phase_age <= 3:
            self.phase_credit = max(0., self.phase_credit - max(0., time.monotonic() - base_stop))
        else:
            self.phase_credit = 0.
        stored = x.copy()
        self.history.insert(0, stored)
        del self.history[16:]
        result = estimate * bscale
        if not np.isfinite(result).all():
            result = np.zeros_like(b)
        return result, {
            "iterations": iterations, "hvp": count, "query_seconds": oracle_time,
            "wall": time.monotonic() - start, "rr": estimate_rr / bnorm2,
            "initial_rr": initial_rr,
            "energy_bound": estimate_error_bound / max(estimate_energy, 1e-300),
        }


def make_solver(dimension, seed, options):
    """Create a deterministic solver with no state shared between sequences."""
    return Solver(dimension, seed, options)
