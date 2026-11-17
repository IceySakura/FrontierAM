"""Deadline-aware continuation CG for online SPD systems."""

import sys
import time
import numpy as np


_ADAPTIVE_ALLOCATION = True


class _Deadline(Exception):
    pass


class Solver:
    def __init__(self, dimension, seed, options):
        self.dimension = dimension
        self.x = None
        self.history = []
        self.direction = None
        self.hvp_seconds = 0.002
        self.update_seconds = 0.0001
        self.phase_age = 0
        self.phase_changes = 0
        self.previous_b = None
        self.rho = 0.9
        self.adaptive = _ADAPTIVE_ALLOCATION

    def _end_time(self, b, context, begin):
        remaining = context["steps"] - context["step"]
        deadline = context.get("deadline", begin + 1.0)
        if not self.adaptive:
            return begin + max(0.0, (deadline - begin - 0.15) / remaining), deadline

        changed = context.get("phase_changed", False)
        if self.previous_b is not None and not changed:
            corr = abs(float(b @ self.previous_b)) / max(
                float(np.linalg.norm(b) * np.linalg.norm(self.previous_b)), 1.0e-100)
            self.rho = 0.5 * self.rho + 0.5 * np.clip(corr, 0.0, 1.0)
        self.previous_b = b.copy()
        if changed:
            self.phase_age = 0
            if context["step"]:
                self.phase_changes += 1

        logcond = np.log10(context["L"] / context["mu"])
        front = 1.0 + np.clip(logcond - 5.5, 0.0, 2.5)
        memory = min(remaining, 1.0 / max(1.0e-4, 1.0 - self.rho ** 2))
        front = min(9.0, 1.0 + (front - 1.0) * (memory / 5.0) ** (2.0 / 3.0))
        decay = np.clip(1.5 * (memory / 5.0) ** 0.3, 0.8, 4.0)
        attenuation = min(1.0, (remaining - 1) / 8.0)
        weight = 1.0 + (front - 1.0) * attenuation * np.exp(-self.phase_age / decay)

        hazard = (2.0 + self.phase_changes) / (50.0 + context["step"])
        a = (1.0 - hazard) * np.exp(-1.0 / decay)
        stationary = hazard / (1.0 - a)
        ages = np.arange(remaining)
        expected = stationary + (np.exp(-self.phase_age / decay) - stationary) * a ** ages
        attenuations = np.minimum(1.0, (remaining - ages - 1) / 8.0)
        normalizer = remaining + (front - 1.0) * float(attenuations @ expected)
        end = begin + max(0.0, deadline - begin - 0.15) * weight / normalizer
        self.phase_age += 1
        return min(deadline - 0.1, end), deadline

    def _clear(self):
        self.x = None
        self.history.clear()
        self.direction = None

    def solve(self, hvp, b, context):
        self.safe_x = np.zeros_like(b)
        try:
            x, diagnostics = self._solve(hvp, b, context)
        except Exception as exc:
            if not isinstance(exc, _Deadline) and type(exc).__name__ != "BudgetExceeded":
                print("solver fallback:", type(exc).__name__, str(exc),
                      file=sys.stderr, flush=True)
            x = self.safe_x
            diagnostics = {"fallback": type(exc).__name__}
            self._clear()
        norm_x = float(np.linalg.norm(x))
        max_norm = 4.0 * float(np.linalg.norm(b)) / context["mu"]
        if not np.isfinite(norm_x) or norm_x > max_norm or not np.isfinite(x).all():
            x = np.zeros_like(b)
            self._clear()
        return x, diagnostics

    def _solve(self, hvp, b, context):
        begin = time.monotonic()
        remaining = context["steps"] - context["step"]
        end, deadline = self._end_time(b, context, begin)
        queries = 0

        def product(v):
            nonlocal queries
            tic = time.monotonic()
            if tic + 1.5 * self.hvp_seconds > deadline - 0.025:
                raise _Deadline()
            result = hvp(v)
            dt = time.monotonic() - tic
            self.hvp_seconds = 0.8 * self.hvp_seconds + 0.2 * dt
            queries += 1
            return result

        changed = context.get("phase_changed", False)
        if changed:
            self.history.clear()
            self.direction = None
        x = np.zeros_like(b)
        U = None
        W = None
        if self.history or self.direction is not None:
            pieces = list(self.history)
            if self.direction is not None:
                pieces.append(self.direction)
            X = np.asfortranarray(np.column_stack(pieces))
            Q, singular, _ = np.linalg.svd(X, full_matrices=False)
            Q = np.asfortranarray(Q[:, singular > 1.0e-9 * singular[0]])
            HQ = np.column_stack([product(Q[:, j]) for j in range(Q.shape[1])])
            C = Q.T @ HQ
            eig, V = np.linalg.eigh((C + C.T) * 0.5)
            good = eig > max(eig[-1] * 1.0e-13, 0.0)
            transform = V[:, good] / np.sqrt(eig[good])
            # U.T H U = I; the refreshed coarse solve is exact in this span.
            U = np.asfortranarray(Q @ transform)
            W = np.asfortranarray(HQ @ transform)
            coeff = U.T @ b
            x = U @ coeff
            r = b - W @ coeff
        elif self.x is not None and not changed:
            x[:] = self.x
            r = b - product(x)
        else:
            r = b.copy()
        self.safe_x = x

        def project(v):
            if U is None:
                return v.copy()
            return v - U @ (W.T @ v)

        p = project(r)
        rr = float(r @ r)
        bnorm2 = float(b @ b)
        objective = float(x @ (b + r))
        reserve = self.update_seconds if remaining > 1 else 0.0
        iterations = 0
        while (time.monotonic() + max(0.0005, 1.3 * self.hvp_seconds) + reserve < end
               and rr > max(1.0e-22 * bnorm2,
                            1.0e-10 * context["mu"] * max(0.0, objective))
               and iterations < 5000):
            q = product(p)
            pq = float(p @ q)
            if not np.isfinite(pq) or pq <= 0.0:
                break
            pr = float(r @ p)
            alpha = pr / pq
            x += alpha * p
            r -= alpha * q
            objective += alpha * pr
            rr_new = float(r @ r)
            p *= rr_new / rr
            p += project(r)
            rr = rr_new
            iterations += 1
        if U is not None:
            coeff = U.T @ r
            x += U @ coeff
            r -= W @ coeff
            rr = float(r @ r)

        self.x = x.copy()
        if remaining > 1 and iterations:
            tic = time.monotonic()
            pn = float(np.linalg.norm(p))
            self.direction = (np.asfortranarray(p[:, None] / pn)
                              if pn > 0 and np.isfinite(pn) else None)
            self.update_seconds = max(0.00005, time.monotonic() - tic) * 1.05
        xn = float(np.linalg.norm(x))
        if xn > 0.0:
            self.history.append(x / xn)
            self.history = self.history[-1:]
        return x, {
            "iterations": iterations,
            "hvp": queries,
            "residual": float(np.sqrt(rr / bnorm2)),
            "seconds": time.monotonic() - begin,
        }


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)
