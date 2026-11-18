"""Stateful, matrix-free baselines for a sequence of SPD linear systems.

These are reference implementations, not reproductions of LancBiO or RMINRES.
Only the supplied current-system HVP oracle is used. Its caller accounts for
all vector columns, including setup, refreshes, and residual verification.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

try:
    from contracts import BudgetExceeded
except ImportError:  # Lets the module's standalone numerical checks run first.
    class BudgetExceeded(RuntimeError):
        pass


Array = np.ndarray


class _Guard:
    def __init__(self, context, options):
        self.deadline = float(context.get("deadline", float("inf")))
        self.margin = float(options.get("safety_margin", 0.002))

    def check(self):
        if time.monotonic() >= self.deadline - self.margin:
            raise BudgetExceeded("solver deadline")

    def available(self):
        return time.monotonic() < self.deadline - self.margin


def _norm(x):
    return float(np.linalg.norm(x))


def _transport_from(context, d):
    """An optional, explicitly supplied old-to-new coordinate map.

    A nonorthogonal map is used only for a candidate initial vector/history;
    no congruence identity for the current Hessian is assumed.
    """
    value = context.get("transport", context.get("transport_matrix"))
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (d, d) or not np.all(np.isfinite(value)):
        raise ValueError("transport must be a finite d-by-d map")
    return value


class _Solver:
    kind = "cg"

    def __init__(self, d, seed, options):
        self.d = int(d)
        self.options = dict(options or {})
        self.rng = np.random.default_rng(seed)
        self.last_x = None
        self.rank = min(self.d, int(self.options.get("rank", 16)))
        self.recycle_basis = None
        self.lbfgs_s = []
        self.omega = None
        self.nys_preconditioner = None
        self.nys_step = None

    def _initial(self, b, context):
        warm = self.kind != "cg" and self.options.get("warm_start", True)
        if not warm or self.last_x is None:
            return np.zeros_like(b)
        x = self.last_x.copy()
        transport = _transport_from(context, self.d)
        if transport is not None:
            x = transport @ x
        return x if np.all(np.isfinite(x)) else np.zeros_like(b)

    def _prepare_history(self, context):
        transport = _transport_from(context, self.d)
        if context.get("phase_changed", False) and self.options.get("reset_on_phase", True):
            self.recycle_basis = None
            self.lbfgs_s = []
            self.nys_preconditioner = None
        elif transport is not None:
            if self.recycle_basis is not None:
                self.recycle_basis = transport @ self.recycle_basis
            self.lbfgs_s = [transport @ s for s in self.lbfgs_s]
            # A stored low-rank preconditioner is not blindly transported.
            self.nys_preconditioner = None

    def solve(self, hvp: Callable, b: Array, context: dict):
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if b.shape != (self.d,) or not np.all(np.isfinite(b)):
            raise ValueError("invalid right-hand side")
        guard = _Guard(context, self.options)
        start = time.monotonic()
        diag = dict(method=self.kind, iterations=0, stopreason="not_started",
                    setup_hvp=0, solve_hvp=0, true_residual_hvp=0,
                    residual_verified=False, setup_seconds=0.0,
                    target_rtol=float(context.get("target_rtol", 1e-8)))
        counts = {"setup": 0, "solve": 0, "verify": 0}

        def apply(v, purpose="solve"):
            guard.check()
            v = np.asarray(v, dtype=np.float64)
            answer = np.asarray(hvp(v), dtype=np.float64)
            counts[purpose] += 1 if v.ndim == 1 else v.shape[1]
            if answer.shape != v.shape or not np.all(np.isfinite(answer)):
                raise FloatingPointError("invalid HVP result")
            return answer

        self._prepare_history(context)
        x = self._initial(b, context)
        norm_b = _norm(b)
        if norm_b == 0:
            self.last_x = np.zeros_like(b)
            diag.update(stopreason="zero_rhs", relative_residual=0.0,
                        residual_verified=True, elapsed_seconds=time.monotonic()-start)
            return self.last_x.copy(), diag

        M = lambda v: v
        residual_corrector = None
        collector = None
        current_pairs = []
        Q = []
        HQ = []
        window = min(self.d, int(self.options.get("ritz_window", max(32, 4*self.rank))))
        history_ready = False
        setup_start = time.monotonic()
        last_verified = None
        r = None
        try:
            guard.check()
            if np.any(x):
                r = b - apply(x, "verify")
                # Warm-start validation is charged, including after a reset.
                if _norm(r) > norm_b or not np.all(np.isfinite(r)):
                    x = np.zeros_like(b)
                    r = b.copy()
                    diag["warm_start_rejected"] = True
            else:
                r = b.copy()  # H @ 0 is known without an oracle call.
            last_verified = _norm(r) / norm_b
            diag["initial_relative_residual"] = last_verified
            if last_verified <= diag["target_rtol"]:
                self.last_x = x.copy()
                diag.update(stopreason="converged", relative_residual=last_verified,
                            residual_verified=True, setup_complete=True,
                            setup_seconds=time.monotonic()-setup_start,
                            setup_hvp=counts["setup"], solve_hvp=counts["solve"],
                            true_residual_hvp=counts["verify"],
                            elapsed_seconds=time.monotonic()-start)
                return x, diag

            if self.kind == "nystrom_pcg":
                M = self._nystrom(apply, context, guard, diag)
            elif self.kind == "probe_diag_pcg":
                M = self._probe_diagonal(apply, context, guard, diag)
            elif self.kind == "lbfgs_pcg":
                M = self._lbfgs(apply, context, guard, diag)
                memory = int(self.options.get("memory", 12))
                def collector(p, ap):
                    scale = _norm(p)
                    if scale > 0 and np.isfinite(scale):
                        current_pairs.append((p/scale, ap/scale))
                        if len(current_pairs) > memory:
                            current_pairs.pop(0)
            elif self.kind == "recycled_cg":
                U = self.recycle_basis
                if U is not None and U.shape[1]:
                    guard.check()
                    U, _ = np.linalg.qr(U, mode="reduced")
                    HU = apply(U, "setup")
                    E = U.T @ HU
                    E = (E + E.T) * 0.5
                    guard.check()
                    ev, vec = np.linalg.eigh(E)
                    positive = ev > max(float(context.get("mu", 0))*1e-9,
                                        np.finfo(float).eps*max(float(ev[-1]), 1.0))
                    if not np.all(positive):
                        U, HU, ev = U @ vec[:, positive], HU @ vec[:, positive], ev[positive]
                        Einv = np.diag(1.0/ev)
                    else:
                        Einv = (vec / ev) @ vec.T
                    if U.shape[1]:
                        alpha = Einv @ (U.T @ r)
                        x = x + U @ alpha
                        r = r - HU @ alpha
                        # Directions are H-orthogonal to the coarse space.
                        M = lambda v, U=U, HU=HU, Ei=Einv: v - U @ (Ei @ (HU.T @ v))
                        def residual_corrector(x, r, U=U, HU=HU, Ei=Einv):
                            coefficient = Ei @ (U.T @ r)
                            return x + U @ coefficient, r - HU @ coefficient
                        Q.extend(U[:, j].copy() for j in range(U.shape[1]))
                        HQ.extend(HU[:, j].copy() for j in range(U.shape[1]))
                        diag["recycle_rank"] = U.shape[1]
                def collector(p, ap):
                    if len(Q) >= window:
                        return
                    guard.check()
                    q, hq = p.copy(), ap.copy()
                    if Q:
                        basis, images = np.column_stack(Q), np.column_stack(HQ)
                        # Two-pass Gram-Schmidt, transporting Hq identically.
                        for _ in range(2):
                            coefficient = basis.T @ q
                            q -= basis @ coefficient
                            hq -= images @ coefficient
                    scale = _norm(q)
                    if scale > 1e-10 * max(_norm(p), np.finfo(float).tiny):
                        Q.append(q/scale)
                        HQ.append(hq/scale)
                history_ready = True
            diag["setup_seconds"] = time.monotonic() - setup_start
            diag["setup_complete"] = True
            x, info = self._pcg(apply, b, x, r, M, collector, guard, context,
                                 residual_corrector=residual_corrector)
            diag.update(info)
        except BudgetExceeded:
            diag["stopreason"] = "deadline"
            # Preserve the most recent iterate committed by _pcg, if any.
            if hasattr(self, "_current_x"):
                x = self._current_x.copy()
                diag["iterations"] = getattr(self, "_current_iterations", 0)
                diag["relative_residual"] = getattr(self, "_current_residual", None)
            else:
                diag["relative_residual"] = last_verified
        finally:
            if hasattr(self, "_current_x"):
                del self._current_x

        # Ritz extraction has no additional HVP; it is still timed and guarded.
        if self.kind == "recycled_cg" and history_ready and Q and guard.available():
            try:
                guard.check()
                basis, images = np.column_stack(Q), np.column_stack(HQ)
                rayleigh = basis.T @ images
                rayleigh = (rayleigh + rayleigh.T) * 0.5
                guard.check()
                ev, vec = np.linalg.eigh(rayleigh)
                rank = min(self.rank, len(ev))
                self.recycle_basis = basis @ vec[:, :rank]
                diag["stored_recycle_rank"] = rank
            except BudgetExceeded:
                pass
        if self.kind == "lbfgs_pcg" and current_pairs:
            self.lbfgs_s = [s.copy() for s, _ in current_pairs]
        self.last_x = x.copy()
        diag.update(setup_hvp=counts["setup"], solve_hvp=counts["solve"],
                    true_residual_hvp=counts["verify"],
                    elapsed_seconds=time.monotonic()-start)
        if not diag.get("setup_complete"):
            diag["setup_seconds"] = time.monotonic()-setup_start
        return x, diag

    def _pcg(self, apply, b, x, r, M, collector, guard, context,
             residual_corrector=None):
        target = float(context.get("target_rtol", 1e-8))
        norm_b = _norm(b)
        maxiter = int(self.options.get("maxiter", self.options.get("maxiter_factor", 4)*self.d))
        if maxiter < self.d:
            raise ValueError("maxiter must be at least the system dimension")
        relative = _norm(r) / norm_b
        self._current_x = x.copy()
        self._current_iterations = 0
        self._current_residual = relative
        if relative <= target:
            # r was either verified, known for x=0, or updated by exact coarse actions.
            return x, dict(iterations=0, stopreason="converged", relative_residual=relative,
                           residual_verified=True)
        z = M(r)
        rz = float(r @ z)
        if not np.isfinite(rz) or rz <= 0:
            z, rz = r.copy(), float(r @ r)
            M = lambda v: v
        p = z.copy()
        info = dict(iterations=0, stopreason="maxiter", residual_verified=False)
        # Unnecessary periodic restarts destroy much of CG's superlinear gain.
        # Verify at convergence; a periodic replacement is an explicit option.
        refresh_every = int(self.options.get("residual_refresh", 0))
        for iteration in range(1, maxiter+1):
            guard.check()
            ap = apply(p)
            curvature = float(p @ ap)
            if not np.isfinite(curvature) or curvature <= 0:
                info.update(stopreason="nonpositive_curvature", iterations=iteration-1)
                break
            alpha = rz/curvature
            x = x + alpha*p
            r = r - alpha*ap
            relative = _norm(r)/norm_b
            self._current_x = x.copy()
            self._current_iterations = iteration
            self._current_residual = relative
            info.update(iterations=iteration)
            if collector is not None:
                collector(p, ap)
            refresh = relative <= target or (refresh_every > 0 and iteration % refresh_every == 0)
            if refresh:
                r = b - apply(x, "verify")
                relative = _norm(r)/norm_b
                self._current_residual = relative
                info["residual_verified"] = True
                if relative <= target:
                    info["stopreason"] = "converged"
                    break
                if residual_corrector is not None:
                    # Residual replacement can reintroduce a component in U.
                    # Restore U.T @ r = 0 before restarting deflated CG. HU
                    # was already evaluated and charged during current setup.
                    guard.check()
                    x, r = residual_corrector(x, r)
                    relative = _norm(r)/norm_b
                    self._current_x = x.copy()
                    self._current_residual = relative
                    if relative <= target:
                        info["stopreason"] = "converged"
                        break
                # Recompute direction after replacement of the recursive residual.
                z = M(r)
                rz = float(r @ z)
                if not np.isfinite(rz) or rz <= 0:
                    z, rz = r.copy(), float(r @ r)
                    M = lambda v: v
                p = z.copy()
                continue
            info["residual_verified"] = False
            guard.check()
            z = M(r)
            next_rz = float(r @ z)
            if not np.isfinite(next_rz) or next_rz <= 0:
                info["stopreason"] = "preconditioner_breakdown"
                break
            p = z + (next_rz/rz)*p
            rz = next_rz
        info["relative_residual"] = relative
        return x, info

    def _nystrom(self, apply, context, guard, diag):
        """FTU-style shifted Nystrom spectral preconditioner.

        A=H-mu I is PSD. If Ahat=U diag(lam) U.T, use
        P^{-1}=I+U diag((lam_min+mu)/(lam+mu)-1) U.T.
        """
        mu = float(context["mu"])
        if mu <= 0:
            raise ValueError("Nystrom PCG needs a positive lower spectral bound")
        refresh = max(1, int(self.options.get("refresh_every", 1)))
        step = int(context.get("step", 0))
        if (self.nys_preconditioner is not None and self.nys_step is not None
                and 0 <= step-self.nys_step < refresh):
            U, gain = self.nys_preconditioner
        else:
            guard.check()
            if self.omega is None:
                self.omega, _ = np.linalg.qr(self.rng.standard_normal((self.d, self.rank)), mode="reduced")
            omega = self.omega
            Y = apply(omega, "setup") - mu*omega
            small = (omega.T @ Y)
            small = (small + small.T)*0.5
            guard.check()
            ev, vec = np.linalg.eigh(small)
            tolerance = np.finfo(float).eps*max(float(np.max(np.abs(ev))), 1.0)*max(10, self.rank)
            keep = ev > tolerance
            if np.any(keep):
                F = (Y @ vec[:, keep]) / np.sqrt(ev[keep])
                guard.check()
                U, singular, _ = np.linalg.svd(F, full_matrices=False)
                lam = singular**2
                gain = (float(lam[-1])+mu)/(lam+mu)-1.0
            else:
                U, gain = np.empty((self.d, 0)), np.empty(0)
            self.nys_preconditioner = U, gain
            self.nys_step = step
        diag["nystrom_rank"] = U.shape[1]
        return lambda v, U=U, gain=gain: v + U @ (gain*(U.T @ v))

    def _probe_diagonal(self, apply, context, guard, diag):
        probes = min(self.d, int(self.options.get("probes", 16)))
        guard.check()
        omega = self.rng.choice(np.array([-1.0, 1.0]), size=(self.d, probes))
        images = apply(omega, "setup")
        estimate = np.mean(omega*images, axis=1)
        mu = float(context.get("mu", np.finfo(float).eps))
        upper = float(context.get("L", float("inf")))
        # A raw Hutchinson diagonal can be negative or almost zero despite
        # H being SPD. Shrink toward its trace estimate and bound the diagonal
        # preconditioner's condition number, rather than amplifying probe noise.
        center = max(float(np.mean(estimate)), mu, np.finfo(float).eps)
        shrinkage = float(self.options.get("diagonal_shrinkage", 0.5))
        estimate = (1-shrinkage)*estimate + shrinkage*center
        estimate = np.clip(estimate, max(mu, 0.25*center), min(upper, 4*center))
        diag["diagonal_probes"] = probes
        return lambda v, estimate=estimate: v/estimate

    def _lbfgs(self, apply, context, guard, diag):
        if not self.lbfgs_s:
            return lambda v: v
        guard.check()
        S = np.column_stack(self.lbfgs_s)
        Y = apply(S, "setup")  # Refresh all secants against the current H.
        pairs = []
        for j in range(S.shape[1]):
            s, y = S[:, j], Y[:, j]
            sy = float(s @ y)
            if sy > np.finfo(float).eps*_norm(s)*_norm(y):
                pairs.append((s, y, 1.0/sy))
        if not pairs:
            return lambda v: v
        s, y, _ = pairs[-1]
        gamma = float(s @ y)/float(y @ y)
        gamma = float(np.clip(gamma, 1.0/float(context.get("L", 1e300)),
                              1.0/float(context.get("mu", 1e-300))))
        diag["lbfgs_pairs"] = len(pairs)
        def two_loop(v):
            q = v.copy()
            alphas = []
            for s, y, rho in reversed(pairs):
                alpha = rho*float(s @ q)
                alphas.append(alpha)
                q -= alpha*y
            out = gamma*q
            for (s, y, rho), alpha in zip(pairs, reversed(alphas)):
                out += s*(alpha-rho*float(y @ out))
            return out
        return two_loop


def make_solver(name: str, d: int, seed: int = 0, options: dict | None = None):
    options = dict(options or {})
    canonical = name.lower().replace("-", "_")
    aliases = {
        "cold_cg": "cg", "cg": "cg", "warm": "warm_cg",
        "warm_cg": "warm_cg", "nystrom": "nystrom_pcg",
        "nys_pcg": "nystrom_pcg", "nystrom_pcg": "nystrom_pcg",
        "recycled": "recycled_cg", "augmented_cg": "recycled_cg",
        "recycled_cg": "recycled_cg", "ritz_cg": "recycled_cg",
        "lbfgs": "lbfgs_pcg", "lbfgs_pcg": "lbfgs_pcg",
        "probe_diag_pcg": "probe_diag_pcg", "diagonal_probe_pcg": "probe_diag_pcg",
    }
    # Convenient rank configurations for the runner's method grid.
    for prefix in ("nystrom_pcg", "nystrom", "recycled_cg", "recycled"):
        suffix = canonical.removeprefix(prefix+"_")
        if canonical.startswith(prefix+"_") and suffix.lstrip("r").isdigit():
            options["rank"] = int(suffix.lstrip("r"))
            canonical = prefix
            break
    if canonical not in aliases:
        raise ValueError(f"Unknown solver {name!r}; choose one of {sorted(aliases)}")
    solver = _Solver(d, seed, options)
    solver.kind = aliases[canonical]
    return solver
