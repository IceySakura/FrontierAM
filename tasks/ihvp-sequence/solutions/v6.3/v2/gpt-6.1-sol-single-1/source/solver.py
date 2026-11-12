"""Online, deadline-aware Galerkin-recycled conjugate gradients."""

import ctypes
import os
import time

import numpy as np


try:
    _library = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "_cg.so"))
    _cg_step = _library.cg_step
    _cg_step.argtypes = (
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_void_p,
    )
    _cg_step.restype = ctypes.c_double
except (OSError, AttributeError):
    _cg_step = None


class Solver:
    def __init__(self, dimension, seed, options):
        self.dimension = int(dimension)
        self.options = options or {}
        self.history_size = min(16, max(1, int(self.options.get("history", 6))))
        self.history = []
        self.age = 0
        self.phase_start = 0
        self.phase_period = 20.0
        self.hvp_seconds = 0.001
        self.energy_tolerance = float(self.options.get("energy_tol", 1e-14))
        self.ritz_size = min(24, max(0, int(self.options.get("ritz", 8))))
        self.ritz_basis = None

    def _stop_time(self, context, start):
        step = int(context["step"])
        remaining = max(1, int(context["steps"]) - step)
        if context["phase_changed"] or step == 0:
            if step > self.phase_start:
                duration = step - self.phase_start
                self.phase_period = 0.5 * (self.phase_period + duration)
            self.phase_start = step
            self.age = 0
        else:
            self.age += 1
        condition = float(context["L"]) / float(context["mu"])
        weights = self.options.get(
            "weights", (10.0 if condition >= 1e6 else 4.5, 3.0, 2.0, 1.3)
        )
        weight = weights[self.age] if self.age < len(weights) else 1.0
        current_extra = sum(w - 1.0 for w in weights[self.age:self.age + remaining])
        future_changes = (
            max(0.0, (remaining + self.age - 1) / self.phase_period - 0.5)
            - max(0.0, self.age / self.phase_period - 0.5)
        )
        denominator = remaining + current_extra + future_changes * sum(w - 1.0 for w in weights)
        deadline = float(context["deadline"]) - 0.15
        stop = start + max(0.0, deadline - start) * weight / max(weight, denominator)
        return min(stop, deadline), deadline

    def _history_basis(self):
        vectors = [self.history[-1]]
        differences = self.history[::-1]
        for _ in range(min(5, len(differences) - 1)):
            differences = [differences[j] - differences[j + 1]
                           for j in range(len(differences) - 1)]
            vectors.append(differences[0])
        for j in range(6, len(self.history)):
            vectors.append(self.history[-j] - self.history[-j - 1])
        if self.ritz_basis is not None:
            vectors.extend(self.ritz_basis.T)
        block = np.column_stack(vectors)
        norms = np.linalg.norm(block, axis=0)
        keep = np.isfinite(norms) & (norms > 1e-100)
        if not np.any(keep):
            return None
        block = block[:, keep] / norms[keep]
        q, triangular = np.linalg.qr(block, mode="reduced")
        if np.all(np.abs(np.diag(triangular)) > 1e-8):
            return q
        # Rank selection on the small factor also handles dependent early columns.
        rotations, singular_values, _ = np.linalg.svd(triangular, full_matrices=False)
        keep = singular_values > 1e-8 * singular_values[0]
        return q @ rotations[:, keep] if np.any(keep) else None

    def _postproject(self, b, x, r, coarse, mu, upper):
        q, aq, gram, base_x, base_r = coarse
        displacement = x - base_x
        # The residual change supplies H times the entire CG correction for free.
        image = base_r - r
        original_norm = np.linalg.norm(displacement)
        for _ in range(2):
            coefficients = q.T @ displacement
            displacement -= q @ coefficients
            image -= aq @ coefficients
        length = np.linalg.norm(displacement)
        if length > 1e-9 * max(original_norm, 1e-100):
            direction = displacement / length
            adirection = image / length
            cross = q.T @ adirection
            augmented = np.empty((gram.shape[0] + 1, gram.shape[0] + 1))
            augmented[:-1, :-1] = gram
            augmented[:-1, -1] = cross
            augmented[-1, :-1] = cross
            augmented[-1, -1] = direction @ adirection
            spectrum, eigenvectors = np.linalg.eigh(augmented)
            if spectrum[0] >= 0.25 * mu and spectrum[-1] <= 2.0 * upper:
                if self.ritz_size:
                    count = min(self.ritz_size, len(spectrum))
                    selected = eigenvectors[:, :count]
                    self.ritz_basis = q @ selected[:-1] + direction[:, None] * selected[-1]
                rhs = np.append(q.T @ b, direction @ b)
                coefficients = np.linalg.solve(augmented, rhs)
                candidate = q @ coefficients[:-1] + direction * coefficients[-1]
                candidate_r = b - aq @ coefficients[:-1] - adirection * coefficients[-1]
            else:
                coefficients = np.linalg.solve(gram, q.T @ r)
                candidate = x + q @ coefficients
                candidate_r = r - aq @ coefficients
        else:
            coefficients = np.linalg.solve(gram, q.T @ r)
            candidate = x + q @ coefficients
            candidate_r = r - aq @ coefficients
        old_value = float(x @ (b + r))
        new_value = float(candidate @ (b + candidate_r))
        if np.isfinite(new_value) and new_value >= old_value - 1e-12 * max(1.0, abs(old_value)):
            return candidate, candidate_r
        return x, r

    def solve(self, hvp, b, context):
        start = time.monotonic()
        stop, deadline = self._stop_time(context, start)
        b = np.asarray(b, dtype=np.float64)
        b_squared = float(b @ b)
        zero = np.zeros(self.dimension, dtype=np.float64)
        if not b_squared > 0.0 or not np.isfinite(b_squared) or start >= deadline:
            return zero
        mu = float(context["mu"])
        upper = float(context["L"])
        x = zero.copy()
        r = b.copy()
        coarse = None
        if self.history and time.monotonic() + 2.0 * self.hvp_seconds < stop:
            try:
                q = self._history_basis()
                if q is not None and time.monotonic() + (q.shape[1] + 1) * self.hvp_seconds < deadline:
                    then = time.monotonic()
                    aq = hvp(q)
                    elapsed = time.monotonic() - then
                    self.hvp_seconds = 0.7 * self.hvp_seconds + 0.3 * elapsed / q.shape[1]
                    gram = q.T @ aq
                    gram = (gram + gram.T) * 0.5
                    coefficients = np.linalg.solve(gram, q.T @ b)
                    candidate = q @ coefficients
                    if np.isfinite(candidate).all():
                        x = np.ascontiguousarray(candidate)
                        r = np.ascontiguousarray(b - aq @ coefficients)
                        coarse = (q, aq, gram, x.copy(), r.copy())
            except Exception:
                pass

        p = r.copy()
        rr = float(r @ r)
        threshold = b_squared * 1e-18
        bound = np.array([1.0 / mu], dtype=np.float64)
        x_pointer, r_pointer, p_pointer = x.ctypes.data, r.ctypes.data, p.ctypes.data
        bound_pointer = bound.ctypes.data
        iterations = 0
        error_bound = np.inf
        while rr > threshold and time.monotonic() + 1.35 * self.hvp_seconds < stop:
            then = time.monotonic()
            try:
                ap = np.ascontiguousarray(hvp(p), dtype=np.float64)
            except Exception:
                break
            elapsed = time.monotonic() - then
            self.hvp_seconds = 0.85 * self.hvp_seconds + 0.15 * elapsed
            if _cg_step is not None:
                new_rr = _cg_step(self.dimension, x_pointer, r_pointer, p_pointer,
                                  ap.ctypes.data, rr, mu, bound_pointer)
                if new_rr < 0.0 or not np.isfinite(new_rr):
                    break
            else:
                curvature = float(p @ ap)
                if not curvature > 0.0 or not np.isfinite(curvature):
                    break
                alpha = rr / curvature
                x += alpha * p
                r -= alpha * ap
                new_rr = float(r @ r)
                if not np.isfinite(new_rr):
                    break
                beta = new_rr / rr
                p *= beta
                p += r
                delta = bound[0] - alpha
                bound[0] = delta / (mu * delta + beta) if delta > 0.0 else 1.0 / mu
            rr = new_rr
            iterations += 1
            if iterations % 32 == 0:
                objective = float(x @ (b + r))
                error_upper = float(bound[0]) * rr
                error_bound = error_upper / max(1e-100, objective + error_upper)
                if error_bound < self.energy_tolerance:
                    break

        if coarse is not None and iterations:
            try:
                x, r = self._postproject(b, x, r, coarse, mu, upper)
            except Exception:
                pass
        norm_squared = float(x @ x)
        maximum_norm = 4.0 * (float(np.sqrt(b_squared)) / mu)
        if (not np.isfinite(norm_squared)
                or np.sqrt(norm_squared) > maximum_norm
                or not np.isfinite(x).all()):
            x = coarse[3].copy() if coarse is not None else zero
        self.history.append(x.copy())
        self.history = self.history[-self.history_size:]
        if self.options.get("diagnostics", False):
            return x, {"iterations": iterations, "energy_error_bound": error_bound}
        return x


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)
