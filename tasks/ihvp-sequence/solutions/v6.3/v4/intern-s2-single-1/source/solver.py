"""Online SPD sequence solver.

Per-step strategy
-----------------
1. Warm start from the previous solution (zero on a phase change).
2. Subspace projection: given an orthonormal recycling basis V built from
   recent solution increments, solve the small projected system
       (V' H V) a = V' r,   r = b - H x
   and set x <- x + V a.  This is the best correction inside span(V) in the
   H-energy norm and costs one block HVP (<= 64 columns).
3. Preconditioned CG with an L-BFGS preconditioner built from recent
   (delta_x, delta_b) pairs plus the scaled-identity base 2/(mu+L).
4. Record the step's (delta_x, delta_b) pair and the new increment in V.

Time is budgeted adaptively from the measured HVP cost and the remaining
sequence deadline, so the solver stays inside the wall-clock limit regardless
of how expensive the operator is.
"""
from __future__ import annotations

import time

import numpy as np

__all__ = ['make_solver']

MAX_BLOCK = 64


def make_solver(dimension, seed, options):
    return Solver(int(dimension), int(seed), dict(options or {}))


class Solver:
    def __init__(self, d, seed, options):
        self.d = d
        self.seed = seed
        self.opt = options

        # Recycling subspace: at most MAX_BLOCK columns are projected per step
        # (block HVP contract), but we keep a larger basis and use the newest.
        self.m_rec = int(options.get('recycle', min(60, MAX_BLOCK - 4)))
        self.m_rec = min(self.m_rec, MAX_BLOCK - 4)
        self.m_max = int(options.get('recycle_max', 100))
        self.m_max = max(self.m_max, self.m_rec)

        self.max_lbfgs = int(options.get('max_history', 32))
        self.min_pcg = int(options.get('min_pcg', 4))
        self.max_pcg = int(options.get('max_pcg', 1200))
        self.tol = float(options.get('tol', 1e-7))
        self.safety = float(options.get('safety', 0.60))
        self.polish = bool(options.get('polish', True))

        self.hvp_time = 0.0          # running estimate, seconds per HVP
        self.hvp_calls = 0
        self.first = True

        # L-BFGS history: s_i = dx, y_i = db, rho_i = 1/(s_i' y_i)
        self.s = []
        self.y = []
        self.rho = []
        self.gamma = None

        # Recycling basis: first Vsize rows of V are valid and (numerically)
        # orthonormal.  Stored row major so V[:m] @ (d,m) block is one gemm.
        self.V = np.empty((self.m_max, self.d), dtype=np.float64)
        self.Vsize = 0

        self.x_prev = None
        self.b_prev = None
        self.phase = None
        self.mu = 1.0
        self.L = 1.0
        self.scale = 1.0

        # Reusable buffers (size d) to avoid per-iteration allocations.
        self._b0 = np.zeros(self.d, dtype=np.float64)
        self._b1 = np.zeros(self.d, dtype=np.float64)
        self._b2 = np.zeros(self.d, dtype=np.float64)
        self._alpha = np.zeros(self.max_lbfgs, dtype=np.float64)

    # ------------------------------------------------------------------ helpers
    def _precond(self, r):
        """M r with the L-BFGS (two-loop) preconditioner."""
        m = len(self.s)
        if m == 0:
            z = self._b0
            z[:] = r * self.scale
            return z
        q = self._b0
        q[:] = r
        for i in range(m - 1, -1, -1):
            self._alpha[i] = self.rho[i] * float(np.dot(self.s[i], q))
            q -= self._alpha[i] * self.y[i]
        z = self._b1
        z[:] = q
        z *= self.gamma
        for i in range(m):
            beta = self.rho[i] * float(np.dot(self.y[i], z))
            z += self.s[i] * (self._alpha[i] - beta)
        return z

    def _add_V(self, v):
        """Orthogonalise v against the current basis and insert it."""
        w = self._b2
        w[:] = v
        for i in range(self.Vsize):
            w -= float(np.dot(self.V[i], w)) * self.V[i]
        n = float(np.linalg.norm(w))
        if n <= 1e-14:
            return
        w /= n
        if self.Vsize < self.m_max:
            self.V[self.Vsize] = w
            self.Vsize += 1
        else:
            # Drop the oldest vector; w is orthogonal to all kept rows.
            self.V[:-1] = self.V[1:]
            self.V[-1] = w

    def _project(self, hvp, x, r):
        """Best correction of x inside span(V): x <- x + V a."""
        m = min(self.Vsize, self.m_rec)
        if m == 0:
            return r
        V = self.V[:m]
        HV = hvp(np.ascontiguousarray(V.T))
        if HV is None or not isinstance(HV, np.ndarray):
            raise BudgetExceeded
        self.hvp_calls += int(HV.shape[1]) if HV.ndim == 2 else 1
        A = V @ HV
        c = V @ r
        try:
            Lc = np.linalg.cholesky(A)
            a = np.linalg.solve(Lc.T, np.linalg.solve(Lc, c))
        except np.linalg.LinAlgError:
            a = np.linalg.lstsq(A, c, rcond=None)[0]
        x += V.T @ a
        r -= HV @ a
        return r

    def _pcg(self, hvp, x, b, r, maxiter, deadline):
        bnorm = float(np.linalg.norm(b))
        if bnorm <= 0.0:
            return x, 0, 0.0
        tol = self.tol * bnorm
        z = self._precond(r)
        p = np.array(z, copy=True)
        rz = float(np.dot(r, z))
        if rz <= 0.0:
            return x, 0, float(np.linalg.norm(r)) / bnorm
        it = 0
        for it in range(maxiter):
            if time.monotonic() >= deadline:
                break
            try:
                Hp = hvp(p)
            except Exception:
                break
            if Hp is None or not isinstance(Hp, np.ndarray):
                break
            self.hvp_calls += int(Hp.shape[1]) if Hp.ndim == 2 else 1
            pHp = float(np.dot(p, Hp))
            if pHp <= 1e-300:
                break
            alpha = rz / pHp
            x += alpha * p
            r -= alpha * Hp
            z = self._precond(r)
            rz_new = float(np.dot(r, z))
            if rz_new <= 0.0:
                break
            beta = rz_new / rz
            p = z + beta * p
            rz = rz_new
            if float(np.linalg.norm(r)) <= tol:
                break
        return x, it + 1, float(np.linalg.norm(r)) / bnorm

    # ------------------------------------------------------------------ public
    def solve(self, hvp, b, context):
        try:
            return self._solve(hvp, b, context)
        except Exception as exc:  # safety net: never crash the worker
            try:
                return np.zeros(self.d, dtype=np.float64), {'error': str(exc)}
            except Exception:
                return [0.0] * self.d

    def _solve(self, hvp, b, context):
        t = int(context['step'])
        steps = int(context['steps'])
        deadline = float(context['deadline'])
        phase = int(context['phase'])
        phase_changed = bool(context.get('phase_changed', False))

        self.mu = float(context['mu'])
        self.L = float(context['L'])
        self.scale = 2.0 / (self.mu + self.L)

        # Adaptive per-step budget from the measured HVP cost.
        remaining_steps = steps - t
        now = time.monotonic()
        remaining = max(deadline - now, 0.0)
        per_step = remaining / max(remaining_steps, 1) * self.safety
        ht = self.hvp_time if self.hvp_time > 0 else 0.0025
        max_hvp = max(self.min_pcg, min(self.max_pcg,
                                        int(per_step / max(ht, 1e-9))))

        if self.first:
            self.first = False
            self.phase = phase

        # ---- initial guess -------------------------------------------------
        if self.x_prev is None or phase_changed:
            x = np.zeros(self.d, dtype=np.float64)
            if phase_changed:
                self.s = []
                self.y = []
                self.rho = []
                self.gamma = None
                self.Vsize = 0
        else:
            x = np.array(self.x_prev, copy=True)

        # ---- time the first HVP of the sequence ----------------------------
        def timed(vector):
            t0 = time.monotonic()
            value = hvp(vector)
            dt = time.monotonic() - t0
            if value is None or not isinstance(value, np.ndarray):
                raise BudgetExceeded
            k = int(value.shape[1]) if value.ndim == 2 else 1
            self.hvp_calls += k
            if self.hvp_time <= 0.0:
                self.hvp_time = max(dt, 1e-9)
            else:
                self.hvp_time = 0.9 * self.hvp_time + 0.1 * max(dt, 1e-9)
            return value

        r = b - timed(x)

        # ---- subspace projection + PCG ------------------------------------
        r = self._project(timed, x, r)
        if time.monotonic() < deadline:
            x, it, res = self._pcg(timed, x, b, r, max_hvp, deadline)
        else:
            it, res = 0, float('nan')

        # ---- optional polish if time remains ------------------------------
        if self.polish and time.monotonic() < deadline:
            r2 = b - timed(x)
            r2 = self._project(timed, x, r2)
            if time.monotonic() < deadline:
                x, it2, res = self._pcg(timed, x, b, r2,
                                        max(self.min_pcg, max_hvp // 4),
                                        deadline)
                it += it2

        # ---- record history ------------------------------------------------
        if self.x_prev is not None and not phase_changed:
            s = x - self.x_prev
            y = b - self.b_prev
            sty = float(np.dot(s, y))
            if sty > 1e-14:
                if len(self.s) >= self.max_lbfgs:
                    self.s.pop(0)
                    self.y.pop(0)
                    self.rho.pop(0)
                self.s.append(s)
                self.y.append(y)
                self.rho.append(1.0 / sty)
                self.gamma = sty / float(np.dot(y, y))

        self._add_V(x - self.x_prev if self.x_prev is not None else x)
        self.x_prev = np.array(x, copy=True)
        self.b_prev = np.array(b, copy=True)
        self.phase = phase

        diag = {'iter': it, 'hvp_calls': self.hvp_calls, 'hvp_time_ms': self.hvp_time * 1000,
                'recycle': int(self.Vsize), 'lbfgs': len(self.s),
                'residual': res if np.isfinite(res) else None,
                'budget_hvp': max_hvp, 'step': t}
        print("DIAG step=%2d phase=%d pcg=%4d rec=%2d lbfgs=%2d res=%.3g "
              "hvp_t=%.3fms budget=%4d hvp=%5d rem=%.2f" % (
                  t, phase, it, int(self.Vsize), len(self.s),
                  res if np.isfinite(res) else -1.0, self.hvp_time * 1000,
                  max_hvp, self.hvp_calls, remaining), flush=True)
        return x, diag
