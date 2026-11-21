"""Deadline-aware augmented conjugate gradients for online SPD systems."""

import math
import time

import numpy as np


class Solver:
    def __init__(self, dimension, seed, options):
        self.dimension = int(dimension)
        self.x = None
        self.direction = None
        self.hvp_time = 0.003
        self.phase_start = 0
        self.phase_lengths = []

    def solve(self, hvp, b, context):
        started = time.monotonic()
        step = int(context["step"])
        remaining = max(1, int(context["steps"]) - step)
        changed = bool(context["phase_changed"])
        condition = max(1.0, float(context["L"]) / float(context["mu"]))
        cold_weight = min(4.0, max(1.5, 2.0 + 0.7 * math.log10(condition / 3e5)))
        if changed:
            if step > self.phase_start:
                self.phase_lengths.append(step - self.phase_start)
            self.phase_start = step
            self.direction = None

        # Reserve work for later phase starts and adapt their spacing estimate online.
        period = sum(self.phase_lengths) / len(self.phase_lengths) if self.phase_lengths else 20.0
        denominator = remaining + (cold_weight - 1.0) * (
            (remaining - 1) / period + int(changed)
        )
        weight = cold_weight if changed else 1.0
        available = max(0.0, float(context["deadline"]) - started - 0.16)
        end = started + available * min(1.0, weight / denominator)
        x = np.zeros_like(b)
        r = b.copy()
        bb = float(b @ b)
        calls = 0
        iterations = 0
        failed_query = False
        if bb <= 0.0 or not np.isfinite(bb):
            return x
        if end <= started:
            if changed:
                self.x = None
            return x if changed or self.x is None else self.x.copy()
        residual_goal = bb * min(1e-16, 1e-12 / condition)

        def query(v):
            nonlocal calls
            begin = time.monotonic()
            result = hvp(v)
            elapsed = time.monotonic() - begin
            self.hvp_time = 0.8 * self.hvp_time + 0.2 * elapsed
            calls += 1
            return result

        # Refresh both retained vectors under the current Hessian.
        vectors, products = [], []
        for v in (self.x, self.direction):
            if v is None or time.monotonic() + max(0.001, 1.6 * self.hvp_time) >= end:
                continue
            norm = float(np.linalg.norm(v))
            if not np.isfinite(norm) or norm <= 1e-150:
                continue
            u = v / norm
            try:
                au = query(u)
            except Exception:
                failed_query = True
                break
            if np.isfinite(au).all():
                vectors.append(u)
                products.append(au)

        s = y = None
        if vectors:
            raw_s = np.column_stack(vectors)
            raw_y = np.column_stack(products)
            gram = raw_s.T @ raw_y
            gram = 0.5 * (gram + gram.T)
            if np.isfinite(gram).all():
                try:
                    values, rotation = np.linalg.eigh(gram)
                    keep = values > max(1e-100, float(values[-1]) * 1e-12)
                    if np.any(keep):
                        transform = rotation[:, keep] / np.sqrt(values[keep])
                        s = np.ascontiguousarray(raw_s @ transform)
                        y = np.ascontiguousarray(raw_y @ transform)
                        coefficients = s.T @ b
                        x = s @ coefficients
                        r = b - y @ coefficients
                except np.linalg.LinAlgError:
                    pass
        rr = float(r @ r)
        initial_rr = rr
        z = r.copy() if s is None else r - s @ (y.T @ r)
        rz = float(r @ z)
        if not np.isfinite(rz) or rz <= 0.0:
            s = y = None
            z = r.copy()
            rz = rr
        p = z.copy()
        last_direction = None

        # S.T H S = I; projected CG remains H-orthogonal to this coarse space.
        limit = min(30000, max(128, 16 * self.dimension))
        while not failed_query and rr > residual_goal and iterations < limit:
            if time.monotonic() + max(0.0005, 1.5 * self.hvp_time) >= end:
                break
            try:
                ap = query(p)
            except Exception:
                break
            curvature = float(p @ ap)
            if not np.isfinite(curvature) or curvature <= 0.0:
                break
            alpha = rz / curvature
            if not np.isfinite(alpha):
                break
            x += alpha * p
            r -= alpha * ap
            rr = float(r @ r)
            last_direction = p
            iterations += 1
            if not np.isfinite(rr) or rr <= residual_goal:
                break
            z = r.copy() if s is None else r - s @ (y.T @ r)
            rz_new = float(r @ z)
            if not np.isfinite(rz_new) or rz_new <= 0.0:
                break
            p = z + (rz_new / rz) * p
            rz = rz_new

        if calls == 0 and not changed and self.x is not None:
            return self.x.copy()
        if not np.isfinite(rr) or not np.isfinite(x).all():
            x = np.zeros_like(b)
            last_direction = None
        else:
            objective_gain = float(x @ (b + r))
            if not np.isfinite(objective_gain) or objective_gain < 0.0:
                x = np.zeros_like(b)
                last_direction = None
        self.x = x.copy()
        if last_direction is not None:
            norm = float(np.linalg.norm(last_direction))
            self.direction = last_direction / norm if np.isfinite(norm) and norm > 1e-150 else None
        return x, {
            "iterations": iterations,
            "hvp": calls,
            "relative_residual_squared": rr / bb,
            "initial_residual_squared": initial_rr / bb,
            "seconds": time.monotonic() - started,
        }


def make_solver(d, seed, options):
    return Solver(d, seed, options)
