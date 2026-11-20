"""A deadline-aware, recycled CG solver for an online stream of SPD systems."""

import ctypes
from math import isfinite, sqrt
from pathlib import Path
from time import monotonic

import numpy as np

try:
    from contracts import BudgetExceeded
except ImportError:
    class BudgetExceeded(RuntimeError):
        pass


def _basis(z, hz):
    """Whiten the measured energy Gram matrix of the recycle vectors."""
    gram = z.T @ hz
    if not np.isfinite(gram).all():
        return None, None
    diagonal = np.diag(gram)
    if not (diagonal > 0.0).all():
        keep = diagonal > 0.0
        if not keep.any():
            return None, None
        z, hz = z[:, keep], hz[:, keep]
        gram = gram[keep][:, keep]
        diagonal = np.diag(gram)
    scales = 1.0 / np.sqrt(diagonal)
    g = (0.5 * (gram + gram.T) * scales) * scales[:, None]
    lam, rotation = np.linalg.eigh(g)
    keep = lam > 4.0e-11 * max(1.0, float(lam[-1]))
    if not keep.any():
        return None, None
    transform = rotation[:, keep] * scales[:, None] / np.sqrt(lam[keep])
    return np.asfortranarray(z @ transform), np.asfortranarray(hz @ transform)


def _load_kernel():
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("cg_native.so")))
        update = lib.pcg_update
    except (OSError, AttributeError):
        return None
    update.restype = ctypes.c_double
    update.argtypes = [ctypes.c_size_t, ctypes.c_size_t] + [ctypes.c_void_p] * 5 + [
        ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
    return update


class Solver:
    def __init__(self, dimension):
        self.dimension = dimension
        self.phase = None
        self.age = 0
        self.old_b = None
        self.solutions = []
        self.seeds = []
        self.relres = 1.0
        self.hvp_seconds = 0.0035
        self.phase_length = 12.0
        self.price = None
        self.kernel = _load_kernel()
        self.kernel_gain = ctypes.c_double()

    def _refresh(self, hvp, vectors, inv_l, end):
        if not vectors:
            return None, None
        count = len(vectors)
        if monotonic() + self.hvp_seconds * (count + 1) >= end:
            vectors = vectors[:1]
            count = len(vectors)
            if monotonic() + self.hvp_seconds >= end:
                return None, None
        try:
            if count <= 3:
                hz = np.column_stack([hvp(v) * inv_l for v in vectors])
                z = np.column_stack(vectors)
            else:
                z = np.column_stack(vectors)
                hz = hvp(z) * inv_l
            return _basis(z, hz)
        except (BudgetExceeded, np.linalg.LinAlgError):
            return None, None

    def _cg(self, hvp, rhs, z, hz, inv_l, start, planned, earliest, end, keep_state):
        tic = monotonic()
        if z is None:
            x0 = np.zeros_like(rhs)
            residual = rhs.copy()
            f0 = 0.0
        else:
            c = z.T @ rhs
            x0 = z @ c
            residual = rhs - hz @ c
            f0 = float(c @ c)
        y = np.zeros_like(rhs)
        p = residual.copy()
        b2 = float(rhs @ rhs)
        rr = float(p @ p)
        tol = 2.0e-24 * b2
        n = 0
        ring = []
        mid = None
        checkpoint = (2.0 * tic + end) / 3.0
        carry_count = 2 if keep_state else 0
        measure_from = tic
        smoothed_time = 0.0
        smoothed_gain = 0.0
        captured = f0
        slope = None
        interval_slope = None
        reuse_weight = 1.30 if self.age == 0 else 1.10 if self.age == 1 else 1.0
        kernel = self.kernel
        if kernel is not None:
            dim = self.dimension
            width = 0 if hz is None else hz.shape[1]
            vp = None if hz is None else hz.ctypes.data
            yp, rp, pp = y.ctypes.data, residual.ctypes.data, p.ctypes.data
        while rr > tol and isfinite(rr) and monotonic() < end:
            try:
                hp = hvp(p)
            except BudgetExceeded:
                break
            old_p = p.copy() if carry_count else None
            new_rr = -1.0
            if kernel is not None:
                hp = np.asarray(hp, dtype=np.float64, order="C")
                new_rr = kernel(dim, width, yp, rp, pp, hp.ctypes.data, vp,
                                rr, inv_l, self.kernel_gain)
                if new_rr < -1.0:
                    break
            if new_rr >= 0.0:
                gain = self.kernel_gain.value
            else:
                hp = np.asarray(hp, dtype=np.float64) * inv_l
                if hz is not None:
                    hp -= hz @ (hz.T @ p)
                curvature = float(p @ hp)
                if curvature <= 0.0 or not isfinite(curvature):
                    break
                step = rr / curvature
                if not isfinite(step):
                    break
                y += step * p
                residual -= step * hp
                new_rr = float(residual @ residual)
                p *= new_rr / rr
                p += residual
                gain = step * rr
            if not isfinite(new_rr):
                break
            captured += gain
            smoothed_gain = .93 * smoothed_gain + gain
            if carry_count:
                if len(ring) == carry_count:
                    ring.pop(0)
                ring.append(old_p)
            rr = new_rr
            n += 1
            if keep_state and n > 7 and mid is None and monotonic() >= checkpoint:
                mid = y.copy()
            if n % 8 == 0:
                now = monotonic()
                smoothed_time = .56 * smoothed_time + (now - measure_from)
                measure_from = now
                slope = smoothed_gain / max(1.e-120, smoothed_time * captured)
                if n >= 24 and now >= earliest:
                    if self.price is not None and reuse_weight * slope <= self.price:
                        interval_slope = slope
                        break
                    if interval_slope is None and now >= planned:
                        interval_slope = slope
        elapsed = monotonic() - tic
        if n > 6:
            self.hvp_seconds = 0.84 * self.hvp_seconds + 0.16 * elapsed / n
        if n >= 24 and slope is not None:
            measured = max(1.e-12, slope if interval_slope is None else interval_slope)
            if self.price is None:
                self.price = .92 * measured
            else:
                ratio = (monotonic() - start) / max(1.e-6, planned-start)
                feedback = 1.0 + .31 * min(1.0, max(-.65, ratio-1.0))
                measured = min(2.0*self.price, max(.60*self.price, measured))
                self.price = (.83*self.price + .17*measured) * feedback
        if carry_count and n and rr > tol and np.isfinite(p).all():
            if ring:
                ring.pop(0)
            ring.append(p)

        def project(a):
            return a if hz is None else a - z @ (hz.T @ a)

        dy = project(y)
        x = x0 + dy
        # Protect the return value against numerical failure in the projected solve.
        objective = float(x @ (rhs + residual))
        if not isfinite(objective) or objective < f0 * (1.0 - 1.e-7):
            return x0, [], 1.0
        if hz is not None:
            ring = [project(a) for a in ring]
        if keep_state and n and rr > tol:
            ring.append(dy)
            if mid is not None:
                ring.append(project(y - mid))
        return x, ring, sqrt(rr / b2)

    def solve(self, hvp, b, context):
        start = monotonic()
        phase = context.get("phase")
        left = int(context["steps"]) - int(context["step"])
        end_sequence = float(context["deadline"])
        available = end_sequence - start - max(.34, 2.5*self.hvp_seconds) - .0028 * left
        if available <= 0.0:
            return np.zeros_like(b)
        L = float(context["L"])
        mu = float(context["mu"])
        b_scale = float(np.max(np.abs(b)))
        if not (1.e-150 <= L <= 1.e150 and 0.0 < mu <= L and b_scale > 0.0):
            return np.zeros_like(b)
        rhs = b / b_scale
        inv_l = 1.0 / L
        if self.dimension <= 12:
            try:
                return np.linalg.solve(hvp(np.eye(self.dimension)), b)
            except (BudgetExceeded, np.linalg.LinAlgError):
                pass
        corr = 0.0
        overlap = 0.0
        if self.old_b is not None:
            b2, old2 = float(rhs @ rhs), float(self.old_b @ self.old_b)
            corr = abs(float(rhs @ self.old_b)) / sqrt(b2 * old2)
            if self.solutions:
                last = self.solutions[0]
                old_energy = float(self.old_b @ last)
                if old_energy > 0.0 and isfinite(old_energy):
                    overlap = min(1.0, abs(float(rhs @ last)) / old_energy * sqrt(old2 / b2))
        change = self.phase != phase or bool(context.get("phase_changed", False))
        restart = not change and corr < .18 and overlap < .22
        if change or restart:
            if change and self.age:
                self.phase_length = .55 * self.phase_length + .45 * self.age
            self.phase = phase
            self.age = 0
            self.solutions = self.solutions[:1] if change and overlap > .51 else []
            self.seeds.clear()
        front = 2.65 if L / mu >= 4.e6 else 1.60
        front = 1.0 + (front - 1.0) * min(1.0, self.phase_length / 7.0)
        if self.age == 0:
            weight = front
        elif self.age == 1:
            weight = 1.0 + .14 * (front - 1.0)
        else:
            weight = 1.0
        if left <= 2 or L / mu < 6000.0 or restart:
            weight = 1.0
        share = available / max(1, left)
        budget = share * weight
        planned = start + min(available, budget)
        if left == 1:
            earliest = end = planned
        elif self.price is None or self.age == 0:
            earliest = start + min(available, max(.9*share, .64*budget))
            end = start + min(available, max(budget, min(1.40*budget, 3.7*share)))
        else:
            earliest = start + min(available, .81*budget)
            end = start + min(available, 1.23*budget)
        vectors = self.solutions + self.seeds
        if max(corr, overlap) < .51 or self.relres < 1.e-9:
            vectors = self.solutions[:1]
        z, hz = self._refresh(hvp, vectors[:min(64, self.dimension)], inv_l, end)
        save = left > 1 and self.dimension > 12
        x, seeds, self.relres = self._cg(
            hvp, rhs, z, hz, inv_l, start, planned, earliest, end, save)
        scale = b_scale * inv_l
        output = scale * x
        if not np.isfinite(output).all():
            self.solutions.clear()
            self.seeds.clear()
            output = np.zeros_like(b)
        else:
            self.solutions = ([x] + self.solutions)[:2]
            self.seeds = [v for v in seeds if np.isfinite(v).all() and (v @ v) > 1.e-22]
        self.old_b = rhs
        self.age += 1
        return output


def make_solver(dimension, seed, options):
    return Solver(int(dimension))
