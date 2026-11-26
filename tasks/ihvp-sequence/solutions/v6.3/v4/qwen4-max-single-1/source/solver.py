import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import time
import numpy as np


class Solver:
    def __init__(self, dimension, seed, options):
        self.d = int(dimension)
        self.seed = int(seed)
        self.options = dict(options or {})
        self.memory = 2
        self.x = None
        self.prev_x = None
        self.pairs = []
        self.W = None
        self.C = None
        self.gamma = 1.0
        self.phase_age = 0

    def _build(self):
        pairs = [(s, y) for s, y in self.pairs if float(np.dot(s, y)) > 0]
        if not pairs:
            self.W = None
            self.gamma = 1.0
            return
        m = len(pairs)
        S = np.stack([s for s, y in pairs], axis=1)
        Y = np.stack([y for s, y in pairs], axis=1)
        SY = S.T @ Y
        R = np.triu(SY)
        D = np.diag(np.diag(SY))
        s, y = pairs[-1]
        yy = float(np.dot(y, y))
        self.gamma = float(np.dot(s, y)) / yy if yy > 0 else 1.0
        try:
            Rinv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            Rinv = np.linalg.pinv(R)
        YY = Y.T @ Y
        C = np.zeros((2 * m, 2 * m))
        C[:m, :m] = Rinv.T @ (D + self.gamma * YY) @ Rinv
        C[:m, m:] = -Rinv.T
        C[m:, :m] = -Rinv
        self.W = np.concatenate([S, self.gamma * Y], axis=1)
        self.C = C

    def _precond(self, r):
        if self.W is None:
            return r
        t = self.W.T @ r
        return self.gamma * r + self.W @ (self.C @ t)

    def solve(self, hvp, b, context):
        start = time.monotonic()
        deadline = float(context["deadline"])
        remaining = max(0.0, deadline - start)
        steps_left = max(1, int(context["steps"]) - int(context["step"]))
        if context.get("phase_changed"):
            self.phase_age = 0
        else:
            self.phase_age += 1
        if self.phase_age == 0:
            budget = min(1.5, max(0.02, (remaining - 0.25) / steps_left * 3.0))
        elif self.phase_age == 1:
            budget = min(0.8, max(0.02, (remaining - 0.25) / steps_left * 1.6))
        else:
            budget = min(0.55, max(0.02, (remaining - 0.25) / steps_left))
        limit = min(start + budget, deadline - 0.05)
        b = np.asarray(b, dtype=np.float64)
        mu = float(context.get("mu", 1.0))
        L = float(context.get("L", 1.0))
        use_pc = (L / max(mu, 1e-300)) > 5e5
        if context.get("phase_changed"):
            self.x = None
            self.prev_x = None
            self.pairs = []
            self.W = None
        x = self.x.copy() if self.x is not None else np.zeros_like(b)
        try:
            if use_pc and self.x is not None and self.prev_x is not None:
                s = self.x - self.prev_x
                y = hvp(s)
                if float(np.dot(s, y)) > 0:
                    self.pairs.append((s, y))
                    if len(self.pairs) > self.memory:
                        self.pairs.pop(0)
            if use_pc:
                self._build()
            else:
                self.pairs = []
                self.W = None
                self.gamma = 1.0
            r = b - hvp(x) if np.any(x) else b.copy()
            bn = float(np.linalg.norm(b))
            rn = float(np.linalg.norm(r))
            z = self._precond(r)
            p = z.copy()
            rz = float(np.dot(r, z))
            iterations = 0
            for iterations in range(1, 2001):
                if time.monotonic() >= limit or rn <= bn * 1e-8:
                    break
                hp = hvp(p)
                denom = float(np.dot(p, hp))
                if denom <= 0.0 or not np.isfinite(denom):
                    break
                alpha = rz / denom
                x += alpha * p
                r -= alpha * hp
                rn = float(np.linalg.norm(r))
                if rn <= bn * 1e-8:
                    break
                z = self._precond(r)
                rz_new = float(np.dot(r, z))
                if rz <= 0 or rz_new <= 0:
                    z = r.copy()
                    rz_new = float(np.dot(r, z))
                    p = z.copy()
                else:
                    p *= rz_new / rz
                    p += z
                rz = rz_new
            self.prev_x = self.x
            self.x = x
            return x, {"iterations": iterations, "residual": rn / bn,
                       "seconds": time.monotonic() - start, "pc": use_pc}
        except Exception:
            self.prev_x = self.x
            self.x = x
            return x


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)
