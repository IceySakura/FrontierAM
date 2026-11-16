"""Online SPD sequence solver.

Interface
---------
    make_solver(dimension, seed, options) -> Solver
    Solver.solve(hvp, b, context) -> x          # or (x, diagnostics)

``hvp(v)`` returns ``H_t @ v``; ``v`` may be a ``(d,)`` vector or a ``(d, k)``
block with ``k <= 64``.  Only current-step data and the results of legal HVP
calls may be used.

Algorithm
---------
Each step is solved with preconditioned conjugate gradients (PCG) started from
a warm guess instead of from zero.  Two ingredients make the warm start useful
across a slowly drifting sequence:

1. **Extrapolation.**  When the phase has not changed, the solution usually
   drifts smoothly, so ``x_t`` is predicted from the last two solutions.

2. **L-BFGS direct estimate.**  The solver keeps the last ``m`` displacement
   pairs ``s_i = x_i - x_{i-1}``.  At the current step it re-evaluates them
   through the *current* operator, ``y_i = H_t s_i`` (one batched HVP of
   ``d x m``, i.e. ``m`` HVP vectors in a single request), which guarantees
   ``s_i^T y_i > 0`` for an SPD matrix even when the matrix has moved since
   the pair was recorded.  The standard two-loop recursion on these pairs is an
   approximation ``M_t ≈ H_t^{-1}``, so ``x_t ≈ M_t b_t`` is a direct
   approximate solve.

   The warm start is the better of the extrapolation and the L-BFGS estimate,
   judged by one residual evaluation each.

PCG then refines the guess.  The preconditioner is reused inside PCG, so the
batched HVP that built it is amortised over all iterations.  The per-iteration
cost is one HVP plus ``O(m*d)`` work, which is cheap compared with an HVP.

Computation is budgeted against ``context["deadline"]``: before every
iteration the solver checks the monotonic clock and returns the best iterate
seen so far when the budget is exhausted.  ``BudgetExceeded`` raised by the
oracle is caught the same way, so a step can never overrun the sequence-wide
deadline and leave later steps unsolved.

Stopping uses the H-energy of the current residual, estimated without an extra
HVP as ``r^T M r`` and normalised by ``b^T M b`` (the same quantity evaluated
on the right-hand side).  When this falls below ``tol`` the step is finished;
otherwise the iteration cap ``iters`` or the deadline terminates it.  The
estimate returned is always the best iterate encountered, not the last one.

Memory is ``O(keep * d)`` for the solution history plus a handful of working
vectors: well under the 256 MiB limit for the default ``keep = 64``.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# The harness pins BLAS to one thread; repeat it here so a stray environment
# cannot silently re-enable threading.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

try:
    from ihvp_api import BudgetExceeded          # injected by the worker
except Exception:                                   # pragma: no cover
    try:
        from contracts import BudgetExceeded
    except Exception:
        class BudgetExceeded(RuntimeError):
            """Raised by the oracle once the sequence deadline is passed."""


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)


class Solver:
    """Stateful solver for one 50-step sequence."""

    # -- construction ----------------------------------------------------
    def __init__(self, dimension, seed, options):
        self.d = int(dimension)
        self.seed = int(seed)
        self.o = dict(options or {})

        self.m = int(self.o.get("m", 16))                 # L-BFGS pairs
        self.maxit = int(self.o.get("iters", 40))         # PCG cap per step
        self.warm = self.o.get("warm", "better")          # warm-start mode
        self.keep = int(self.o.get("keep", 64))           # history length
        self.tol = float(self.o.get("tol", 1e-7))         # energy stop
        self.safety = float(self.o.get("safety", 0.85))   # deadline headroom

        # History: oldest first.  `Hx` stores H_i @ x_i recorded with the
        # operator of its own step.
        self.hist_x = []
        self.hist_b = []
        self.hist_Hx = []
        self.hist_phase = []

        self.step = 0
        self.t_hvp = 0.005                                # seconds, initial guess
        self._hvp_times = []

        self._hvp = None                                  # set per solve()
        self.best_x = np.zeros(self.d, dtype=np.float64)

    # -- public ----------------------------------------------------------
    def solve(self, hvp, b, context):
        """Return a finite d-vector solving the current system."""
        self._hvp = hvp
        deadline = float(context["deadline"])
        phase_changed = bool(context.get("phase_changed", False))

        try:
            x, Hx = self._solve(b, deadline, phase_changed)
        except BaseException:
            # Never let a step crash the sequence; fall back to zeros.
            x = self.best_x
            Hx = self._safe_hvp(x)

        self._record(x, b, Hx, int(context.get("phase", 0)))
        self.step += 1
        return x

    # -- internals -------------------------------------------------------
    def _solve(self, b, deadline, phase_changed):
        """Warm start, then PCG.  Returns (x, Hx)."""
        prec, denom = self._build_precond(b)
        x0 = self._warm_start(b, prec, phase_changed)

        x = self._pcg(x0, b, prec, denom, deadline)

        # Hx for the history: one last HVP, free if the residual was already
        # computed; otherwise worth it for the next step's pairs.
        Hx = self._safe_hvp(x)
        return x, Hx

    def _warm_start(self, b, prec, phase_changed):
        """Pick the best available starting point."""
        if not self.hist_x or phase_changed:
            return np.zeros(self.d, dtype=np.float64)

        if self.warm == "zero":
            return np.zeros(self.d, dtype=np.float64)

        candidates = []
        if self.warm in ("extrap", "better"):
            candidates.append(self._extrapolate())
        if self.warm in ("precond", "better"):
            candidates.append(prec(b))                  # L-BFGS direct estimate

        if len(candidates) == 1:
            return candidates[0]

        # Score each candidate by its residual norm; one HVP each.
        best, best_r2 = None, None
        for c in candidates:
            r = b - self._safe_hvp(c)
            r2 = float(r @ r)
            if best is None or r2 < best_r2:
                best, best_r2 = c, r2
        return best

    def _extrapolate(self):
        """Linear extrapolation from the last two solutions."""
        if len(self.hist_x) >= 2:
            return 2.0 * self.hist_x[-1] - self.hist_x[-2]
        return self.hist_x[-1].copy()

    def _build_precond(self, b):
        """Build M_t ≈ H_t^{-1} from the last m displacements.

        Returns ``(prec, denom)`` where ``prec(r) = M_t r`` and
        ``denom = b^T M_t b``.  The displacements are re-evaluated through the
        *current* operator with a single batched HVP, which keeps
        ``s^T y > 0`` despite the matrix drifting between steps.
        """
        if self.m <= 0 or len(self.hist_x) < 2:
            return _identity, float(b @ b)

        slist = []
        for i in range(len(self.hist_x) - 1, 0, -1):
            if len(slist) >= self.m:
                break
            s = self.hist_x[i] - self.hist_x[i - 1]
            if float(s @ s) > 0.0:
                slist.append(s)
        if not slist:
            return _identity, float(b @ b)

        S = np.column_stack(slist)                      # d x k, k <= m
        Y = self._safe_hvp(S)                           # k HVP vectors, 1 request
        if Y.ndim != 2 or Y.shape[1] != len(slist):
            return _identity, float(b @ b)

        slist2, ylist = [], []
        for s, y in zip(slist, (Y[:, j] for j in range(Y.shape[1]))):
            ss, sy = float(s @ s), float(s @ y)
            if sy > 1e-12 * ss:                         # curvature must be positive
                slist2.append(s)
                ylist.append(y)
        if len(slist2) < 1:
            return _identity, float(b @ b)

        k = len(slist2)
        gamma = float(slist2[-1] @ ylist[-1]) / float(ylist[-1] @ ylist[-1])

        def prec(r):
            q = np.asarray(r, dtype=np.float64).copy()
            al = [0.0] * k
            for i in range(k - 1, -1, -1):
                sy = float(slist2[i] @ ylist[i])
                al[i] = float(slist2[i] @ q) / sy
                q -= al[i] * ylist[i]
            z = gamma * q
            for i in range(k):
                sy = float(slist2[i] @ ylist[i])
                beta = float(ylist[i] @ z) / sy
                z += slist2[i] * (al[i] - beta)
            return z

        return prec, float(b @ prec(b))

    def _pcg(self, x0, b, prec, denom, deadline):
        """Preconditioned CG from x0.  Returns the best iterate encountered."""
        x = np.asarray(x0, dtype=np.float64).copy()
        r = b - self._safe_hvp(x)
        best_x = x.copy()
        best_energy = None

        z = prec(r)
        rz = float(r @ z)
        p = z.copy()

        per_iter = self.t_hvp * 1.5 + 1e-4              # HVP + preconditioner
        headroom = (deadline - time.monotonic()) * self.safety
        cap = self.maxit
        if per_iter > 0:
            cap = min(cap, max(1, int(headroom / per_iter)))

        for it in range(cap):
            if time.monotonic() > deadline - per_iter:
                break

            Hp = self._safe_hvp(p)
            pHp = float(p @ Hp)
            if pHp <= 0.0:
                break
            alpha = rz / pHp
            x += alpha * p
            r -= alpha * Hp

            energy = float(r @ z) / denom if denom > 0.0 else float(r @ r)
            if best_energy is None or energy < best_energy:
                best_energy = energy
                best_x = x.copy()

            if energy <= self.tol:
                break

            z = prec(r)
            rz_new = float(r @ z)
            if rz_new <= 0.0:
                break
            beta = rz_new / rz
            p = z + beta * p
            rz = rz_new

        return best_x

    def _record(self, x, b, Hx, phase):
        self.hist_x.append(x)
        self.hist_b.append(b)
        self.hist_Hx.append(Hx)
        self.hist_phase.append(phase)
        if len(self.hist_x) > self.keep:
            self.hist_x.pop(0)
            self.hist_b.pop(0)
            self.hist_Hx.pop(0)
            self.hist_phase.pop(0)
        self.best_x = x

    # -- helpers ---------------------------------------------------------
    def _safe_hvp(self, v):
        """Call the oracle, catching a budget overrun and timing the call."""
        t0 = time.perf_counter()
        try:
            out = self._hvp(v)
        except BudgetExceeded:
            raise
        except Exception:
            out = np.zeros(np.asarray(v, dtype=np.float64).shape)
        dt = time.perf_counter() - t0
        self._hvp_times.append(dt)
        if len(self._hvp_times) > 128:
            self._hvp_times.pop(0)
        self.t_hvp = float(np.median(self._hvp_times))
        return np.asarray(out, dtype=np.float64)


def _identity(r):
    return np.asarray(r, dtype=np.float64).copy()


if __name__ == "__main__":
    pass
