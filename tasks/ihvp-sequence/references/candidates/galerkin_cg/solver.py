"""Causal solution-history augmented CG for SPD sequences.

Research-informed baseline, not a reproduction of a published implementation.
Only the current RHS, current HVP oracle, public context and past iterates are used.
"""
from __future__ import annotations

from collections import deque
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve, qr

try:
    from contracts import BudgetExceeded
except ImportError:
    class BudgetExceeded(RuntimeError):
        pass


class _LocalDeadline(RuntimeError):
    pass


class Solver:
    def __init__(self, d, seed, options):
        self.d = int(d)
        self.options = dict(options or {})
        self.history_size = int(self.options.get("history_size", 16))
        if not 0 <= self.history_size <= 64:
            raise ValueError("history_size must be in [0,64]")
        self.warm_start = bool(self.options.get("warm_start", True))
        self.projected = bool(self.options.get("projected", True))
        self.wall_budget = float(self.options.get("wall_budget", 20.0))
        if self.wall_budget <= 0:
            raise ValueError("wall_budget must be positive")
        self.created = time.monotonic()
        self.started = None
        self.history = deque(maxlen=max(1, self.history_size))
        self.previous = None
        self.vector_seconds = 0.001

    def _basis(self, deadline):
        """Span recent snapshots using a recent endpoint and normalized increments.

        Increment normalization retains directions when a large common component
        obscures small, but independent, changes in the solution history.
        Pivoted QR excludes linearly dependent directions rather than inventing
        arbitrary orthogonal completion vectors.
        """
        if self.history_size == 0 or not self.history:
            return None
        snapshots = list(self.history)
        columns = []
        candidates = [snapshots[-1]]
        candidates.extend(snapshots[j] - snapshots[j - 1]
                          for j in range(len(snapshots) - 1, 0, -1))
        for candidate in candidates:
            if time.monotonic() >= deadline:
                raise _LocalDeadline
            norm = float(np.linalg.norm(candidate))
            if np.isfinite(norm) and norm > np.finfo(float).tiny:
                columns.append(candidate / norm)
        if not columns:
            return None
        if time.monotonic() >= deadline:
            raise _LocalDeadline
        matrix = np.asfortranarray(np.column_stack(columns))
        q, r, _ = qr(matrix, mode="economic", pivoting=True,
                     overwrite_a=True, check_finite=False)
        diagonal = np.abs(np.diag(r))
        tolerance = float(self.options.get("basis_rtol", 1e-10))
        rank = int(np.sum(diagonal > tolerance * diagonal[0])) if diagonal.size else 0
        return np.asfortranarray(q[:, :rank]) if rank else None

    def solve(self, hvp, b, context):
        b = np.asarray(b, dtype=np.float64)
        if b.shape != (self.d,) or not np.isfinite(b).all():
            raise ValueError("b must be a finite d-vector")
        now = time.monotonic()
        step = int(context.get("step", 0))
        if step == 0:
            self.history.clear()
            self.previous = None
            # This budget starts at the first call. Startup/import time is still
            # charged externally and must be included in reference qualification.
            self.started = now
        if self.started is None:
            self.started = now
        sequence_deadline = float(context.get("sequence_deadline",
                                              context.get("deadline", float("inf"))))
        finish = min(sequence_deadline, self.started + self.wall_budget)
        remaining_steps = max(1, int(context.get("steps", step + 1)) - step)
        deadline = min(finish, now + max(0.0, finish - now) / remaining_steps)
        # Reserve small output/dispatch time; the actual sequence cap is external.
        deadline -= min(0.002, max(0.0, deadline - now) * 0.01)
        target = float(self.options.get("rtol", context.get("target_rtol", 1e-8)))
        maxiter = int(self.options.get("maxiter",
                      float(self.options.get("maxiter_factor", 50)) * self.d))
        refresh_every = max(1, int(self.options.get("residual_refresh", 128)))
        x = np.zeros(self.d)
        r = b.copy()
        norm_b = float(np.linalg.norm(b))
        iterations = 0
        setup_vectors = 0
        calls = 0
        vectors = 0
        coarse_rank = 0
        reason = "deadline"
        residual_is_true = True
        last_relative = 0.0 if norm_b == 0 else 1.0
        U = C = factor = None

        def check():
            if time.monotonic() >= deadline:
                raise _LocalDeadline

        def product(v):
            nonlocal calls, vectors
            check()
            begin = time.monotonic()
            value = np.asarray(hvp(v), dtype=np.float64)
            count = 1 if v.ndim == 1 else v.shape[1]
            duration = time.monotonic() - begin
            calls += 1
            vectors += count
            if count == 1:
                self.vector_seconds = 0.75 * self.vector_seconds + 0.25 * duration
            if value.shape != v.shape or not np.isfinite(value).all():
                raise ValueError("HVP returned an invalid product")
            return value

        def coarse_correct():
            nonlocal x, r
            if U is not None:
                check()
                coefficient = cho_solve(factor, U.T @ r, check_finite=False)
                x += U @ coefficient
                r -= C @ coefficient

        def precondition(residual):
            # Identity preconditioning with the solution-space projector
            # P = I-U(U^T H U)^-1(HU)^T.  On U^T r=0 this is equivalent
            # to the symmetric PSD operator P P^T on its positive subspace.
            check()
            z = residual.copy()
            if U is not None and self.projected:
                z -= U @ cho_solve(factor, C.T @ z, check_finite=False)
            return z

        try:
            check()
            if norm_b == 0:
                reason = "zero_rhs"
            else:
                if self.warm_start and self.previous is not None:
                    candidate = self.previous.copy()
                    Hcandidate = product(candidate)
                    # Select between the current-operator-validated warm point
                    # and zero by quadratic objective, without any true solution.
                    objective = 0.5 * float(candidate @ Hcandidate) - float(b @ candidate)
                    if np.isfinite(objective) and objective <= 0:
                        x = candidate
                        r = b - Hcandidate
                check()
                U = self._basis(deadline)
                if U is not None:
                    C = product(U)  # refreshed and charged for this H_t
                    setup_vectors = U.shape[1]
                    check()
                    E = U.T @ C
                    E = (E + E.T) * 0.5
                    eigenvalues, rotation = np.linalg.eigh(E)
                    cutoff = max(0.0, float(eigenvalues[-1])) * float(
                        self.options.get("coarse_rtol", 1e-12))
                    keep = eigenvalues > cutoff
                    if not np.all(keep):
                        U = np.asfortranarray(U @ rotation[:, keep])
                        C = np.asfortranarray(C @ rotation[:, keep])
                        E = np.diag(eigenvalues[keep])
                    if U.shape[1] == 0:
                        U = C = None
                    else:
                        check()
                        try:
                            factor = cho_factor(E, lower=True, check_finite=False)
                        except np.linalg.LinAlgError:
                            U = C = factor = None
                        else:
                            coarse_rank = U.shape[1]
                            coarse_correct()
                last_relative = float(np.linalg.norm(r) / norm_b)
                if last_relative <= target:
                    # Always check convergence against a paid true residual.
                    r = b - product(x)
                    coarse_correct()
                    last_relative = float(np.linalg.norm(r) / norm_b)
                if last_relative <= target:
                    reason = "converged"
                else:
                    z = precondition(r)
                    rz = float(r @ z)
                    p = z.copy()
                    reason = "maxiter"
                    for iteration in range(maxiter):
                        check()
                        if not np.isfinite(rz) or rz <= 0:
                            reason = "nonpositive_projected_norm"
                            break
                        Hp = product(p)
                        curvature = float(p @ Hp)
                        if not np.isfinite(curvature) or curvature <= 0:
                            reason = "nonpositive_curvature"
                            break
                        alpha = rz / curvature
                        x += alpha * p
                        r -= alpha * Hp
                        iterations += 1
                        residual_is_true = False
                        last_relative = float(np.linalg.norm(r) / norm_b)
                        replace = (last_relative <= target or
                                   iterations % refresh_every == 0)
                        if replace:
                            r = b - product(x)
                            residual_is_true = True
                            # A true-residual replacement can recover U components
                            # through roundoff: restore U^T r=0 before restarting.
                            if self.projected:
                                coarse_correct()
                            last_relative = float(np.linalg.norm(r) / norm_b)
                            if last_relative <= target:
                                reason = "converged"
                                break
                            z = precondition(r)
                            rz = float(r @ z)
                            p = z.copy()
                            continue
                        z = precondition(r)
                        rz_next = float(r @ z)
                        p *= rz_next / rz
                        p += z
                        rz = rz_next
        except (_LocalDeadline, BudgetExceeded):
            reason = "deadline"

        # A final paid true residual is diagnostic, not access to reference truth.
        if (norm_b > 0 and not residual_is_true and
                deadline - time.monotonic() > max(0.002, 2 * self.vector_seconds)):
            try:
                r = b - product(x)
                residual_is_true = True
                last_relative = float(np.linalg.norm(r) / norm_b)
            except (_LocalDeadline, BudgetExceeded):
                pass
        if not np.isfinite(x).all():
            raise FloatingPointError("nonfinite CG iterate")
        self.previous = x.copy()
        if self.history_size:
            self.history.append(x.copy())
        return x, {"iterations": iterations, "stopreason": reason,
                   "history_size": self.history_size, "coarse_rank": coarse_rank,
                   "projected": self.projected, "warm_start": self.warm_start,
                   "relative_residual": last_relative,
                   "residual_is_true": residual_is_true,
                   "hvp_requests": calls, "hvp_vectors": vectors,
                   "setup_hvp_vectors": setup_vectors,
                   "internal_wall_budget": self.wall_budget,
                   "step_budget_seconds": max(0.0, deadline - now)}


def make_solver(d, seed, options):
    return Solver(d, seed, options)
