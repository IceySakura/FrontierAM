import time
from collections import deque

import numpy as np


class LBFGSPrecond:
    def __init__(self, pairs):
        self.s = [p[0] for p in pairs]
        self.y = [p[1] for p in pairs]
        self.rho = [p[2] for p in pairs]
        self.m = len(pairs)
        if self.m:
            s, y, rho = pairs[-1]
            self.gamma = float((s @ s) / (s @ y))
        else:
            self.gamma = 1.0

    def apply(self, r):
        q = r.copy()
        alphas = [0.0] * self.m
        for i in range(self.m - 1, -1, -1):
            a = self.rho[i] * float(self.s[i] @ q)
            alphas[i] = a
            q -= a * self.y[i]
        q *= self.gamma
        for i in range(self.m):
            b = self.rho[i] * float(self.y[i] @ q)
            q += (alphas[i] - b) * self.s[i]
        return q


class Solver:
    def __init__(self, dimension, seed, options):
        self.d = int(dimension)
        self.options = options
        self.memory = max(0, int(options.get("memory", 2)))
        self.history = deque(maxlen=self.memory)
        self.previous = None
        self.phase = None

    def solve(self, hvp, b, context):
        start = time.monotonic()
        b = np.asarray(b, dtype=np.float64)
        absolute_deadline = float(context.get("deadline", start + 29.0))
        if time.monotonic() >= absolute_deadline - 0.02:
            x = self.previous if self.previous is not None else np.zeros(self.d)
            if not np.isfinite(x).all():
                x = np.zeros(self.d)
            return x, {"skipped": True}
        phase = context.get("phase")
        changed = context.get("phase_changed", False)
        if changed:
            self.history.clear()
            self.previous = None
        x = np.zeros(self.d) if self.previous is None else self.previous.copy()
        r = b - hvp(x) if self.previous is not None else b.copy()
        count = int(self.previous is not None)
        bn = float(np.linalg.norm(b))
        rn0 = float(np.linalg.norm(r))
        precond = LBFGSPrecond(list(self.history))
        z = precond.apply(r)
        p = z.copy()
        rz = float(r @ z)
        target = 1e-9 * bn
        deadline = absolute_deadline - 0.25
        step = int(context.get("step", 0))
        steps = int(context.get("steps", 50))
        allocation = max(0.05, (deadline - start) / max(1, steps - step))
        factor = 3.0 if changed else 1.0
        local_deadline = min(deadline, start + allocation * factor)
        new_pairs = []
        iterations = 0
        for iteration in range(1000):
            if rz <= 0 or not np.isfinite(rz):
                break
            rn = float(np.linalg.norm(r))
            if rn <= target:
                break
            if iteration % 4 == 0 and time.monotonic() >= local_deadline:
                break
            q = hvp(p)
            count += 1
            pq = float(p @ q)
            if not np.isfinite(pq) or pq <= 0.0:
                break
            alpha = rz / pq
            s = alpha * p
            y = alpha * q
            sy = float(s @ y)
            if sy > 1e-30:
                new_pairs.append((s, y, 1.0 / sy))
            x += s
            r -= y
            z = precond.apply(r)
            rz_new = float(r @ z)
            if not np.isfinite(rz_new) or rz_new <= 0:
                break
            beta = rz_new / rz
            p *= beta
            p += z
            rz = rz_new
            iterations += 1
        if not np.isfinite(x).all():
            x = np.zeros(self.d)
            r = b.copy()
            new_pairs = []
        self.history.extend(new_pairs)
        self.previous = x
        self.phase = phase
        rn = float(np.linalg.norm(r))
        diagnostics = {
            "iterations": iterations,
            "hvp": count,
            "history": len(self.history),
            "initial_relative_residual": rn0 / max(bn, 1e-300),
            "relative_residual": rn / max(bn, 1e-300),
            "seconds": time.monotonic() - start,
        }
        return x, diagnostics


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)
