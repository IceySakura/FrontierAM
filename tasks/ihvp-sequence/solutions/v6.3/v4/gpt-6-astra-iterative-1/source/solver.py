"""Deadline-aware augmented conjugate gradients for online SPD systems."""

import ctypes
import json
import math
import os
import time

import numpy as np


class Solver:
    def __init__(self, dimension, seed, options):
        self.dimension = int(dimension)
        self.options = {
            "phase_boost": 4.0,
            "compiled": True,
            "keep_phase": True,
            "phase_correlation": 0.0,
            "adapt_after_phase": True,
            "after_phase_boost": 8.0,
        }
        self.options.update(options or {})
        if os.environ.get("SPD_CONFIG"):
            self.options.update(json.loads(os.environ["SPD_CONFIG"]))
        self.previous = None
        self.direction = None
        self.previous_b = None
        self.phase_start = 0
        self.phase_lengths = []
        self.query_time = 0.003
        self.kernel = None
        self.kernel_out = np.zeros(3)
        if self.options.get("compiled", True):
            try:
                self.library = ctypes.CDLL(
                    os.path.join(os.path.dirname(__file__), "_kernels.so"))
                self.kernel = self.library.cg_step
                self.kernel.argtypes = [
                    ctypes.c_size_t, ctypes.c_int,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_double, ctypes.c_void_p,
                ]
                self.kernel.restype = ctypes.c_int
            except (OSError, AttributeError):
                self.kernel = None

    def _budget(self, context, now):
        step = int(context["step"])
        remaining = int(context["steps"]) - step
        deadline = float(context["deadline"]) - 0.05
        changed = bool(context.get("phase_changed", False))
        if changed:
            if step > self.phase_start:
                self.phase_lengths.append(step - self.phase_start)
            self.phase_start = step
        boost = float(self.options.get("phase_boost", 4.0))
        denominator = float(remaining)
        # Forecast later changes only after observing a completed phase.
        if self.options.get("adapt_after_phase", False) and self.phase_lengths:
            length = max(2.0, float(np.mean(self.phase_lengths[-3:])))
            boost = max(boost, min(
                float(self.options.get("after_phase_boost", 8.0)), 0.5 * length))
            age = step - self.phase_start
            delay = max(1.0, length - age)
            expected_changes = 0.0
            if remaining - 1 >= delay:
                expected_changes = 1.0 + math.floor((remaining - 1 - delay) / length)
                expected_changes *= math.exp(
                    -max(0.0, age - 1.1 * length) / (0.25 * length))
            denominator += (boost - 1.0) * (float(changed) + expected_changes)
        budget = max(0.0, (deadline - now) / denominator)
        if changed:
            budget *= boost
        end = min(deadline, now + budget)
        limit = min(self.dimension, 10000)
        fixed = int(self.options.get("fixed_iterations", 0))
        if fixed:
            end = deadline
            limit = int(fixed * boost) if changed else fixed
        return end, limit

    def solve(self, hvp, b, context):
        start = time.monotonic()
        end, limit = self._budget(context, start)
        b = np.asarray(b, dtype=np.float64)
        initial_rr = float(b @ b)
        x = np.zeros(self.dimension, dtype=np.float64)
        r = b.copy()
        p = None
        rr = initial_rr
        iterations = calls = 0
        W = AW = None
        rank = 0
        if context.get("phase_changed", False):
            keep = bool(self.options.get("keep_phase", True))
            threshold = float(self.options.get("phase_correlation", 0.0))
            if keep and threshold > 0 and self.previous_b is not None:
                denominator = np.sqrt(initial_rr * float(self.previous_b @ self.previous_b))
                keep = denominator > 0 and abs(float(b @ self.previous_b)) >= threshold * denominator
            if not keep:
                self.previous = self.direction = None

        def query(vector):
            nonlocal calls
            tick = time.monotonic()
            product = hvp(vector)
            self.query_time = 0.85 * self.query_time + 0.15 * (time.monotonic() - tick)
            calls += 1
            return product

        try:
            if (self.previous is not None and
                    time.monotonic() + 2.0 * self.query_time < end):
                vectors = [self.previous]
                if self.direction is not None:
                    vectors.append(self.direction)
                U = np.asfortranarray(np.column_stack(vectors))
                norms = np.sqrt(np.sum(U * U, axis=0))
                U /= np.maximum(norms, 1e-100)
                AU = np.asfortranarray(np.column_stack(
                    [query(U[:, j]) for j in range(U.shape[1])]))
                # Refresh an H-orthonormal basis before reusing old directions.
                C = U.T @ AU
                eigenvalues, eigenvectors = np.linalg.eigh((C + C.T) * 0.5)
                keep = eigenvalues > max(eigenvalues[-1] * 1e-12, 1e-100)
                transform = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])
                W = np.asfortranarray(U @ transform)
                AW = np.asfortranarray(AU @ transform)
                rank = W.shape[1]
                coefficients = W.T @ r
                x += W @ coefficients
                r -= AW @ coefficients
                # Refine the Galerkin projection before forming the complement.
                for _ in range(2):
                    coefficients = W.T @ r
                    x += W @ coefficients
                    r -= AW @ coefficients
                if rank >= self.dimension:
                    limit = 0
                rr = float(r @ r)

            p = r.copy()
            if W is not None:
                p -= W @ (AW.T @ r)
            x_pointer, r_pointer, p_pointer = x.ctypes.data, r.ctypes.data, p.ctypes.data
            w_pointer = 0 if W is None else W.ctypes.data
            aw_pointer = 0 if AW is None else AW.ctypes.data
            output_pointer = self.kernel_out.ctypes.data
            while (rr > initial_rr * 1e-16 and iterations < limit and
                   time.monotonic() + 1.5 * self.query_time < end):
                hp = query(p)
                if self.kernel is not None:
                    hp = np.ascontiguousarray(hp, dtype=np.float64)
                    succeeded = self.kernel(
                        self.dimension, rank, x_pointer, r_pointer, p_pointer,
                        hp.ctypes.data, w_pointer, aw_pointer, rr, output_pointer)
                    if not succeeded:
                        break
                    rr = float(self.kernel_out[0])
                else:
                    curvature = float(p @ hp)
                    if not np.isfinite(curvature) or curvature <= 0:
                        break
                    alpha = rr / curvature
                    x += alpha * p
                    r -= alpha * hp
                    next_rr = float(r @ r)
                    p *= next_rr / rr
                    p += r
                    if W is not None:
                        p -= W @ (AW.T @ r)
                    rr = next_rr
                iterations += 1
                if not np.isfinite(rr):
                    break
        except Exception as error:
            if type(error).__name__ != "BudgetExceeded":
                raise
        if not np.isfinite(rr) or not np.isfinite(x).all():
            x.fill(0.0)
            p = None
            rr = initial_rr
        self.previous = x.copy()
        self.direction = p.copy() if p is not None and np.isfinite(p).all() else None
        self.previous_b = b.copy()
        return x, {
            "iterations": iterations,
            "hvp": calls,
            "rank": rank,
            "relative_residual": float(np.sqrt(rr / initial_rr)) if initial_rr > 0 else 0.0,
            "seconds": time.monotonic() - start,
        }


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)
