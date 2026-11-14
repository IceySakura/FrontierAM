"""Online solver for a sequence of correlated SPD linear systems.

Warm-started conjugate gradient with an energy-optimal scaling of the previous
solution, a sequence-wide adaptive time budget, extra budget for steps with no
useful warm start (first step / phase changes), and strict deadline and exception
guards so that every step returns a finite vector.
"""

import time
import numpy as np


class Solver:
    def __init__(self, dimension, seed, options):
        self.d = int(dimension)
        self.rng = np.random.default_rng(seed)
        self.xprev = None
        self.bprev = None
        self.hvprev = None          # H_{t-1} x_{t-1} = b_{t-1} - r_{t-1}
        self.hvp_cost = 0.004      # online estimate of seconds per HVP column
        self.t_start = None
        self.fallback_deadline = None
        cap = 1600
        if isinstance(options, dict):
            try:
                cap = int(options.get('cap', 1600))
            except (TypeError, ValueError):
                pass
        self.cap = max(50, cap)
        self.share = 2.0

    def _hvp(self, hvp, v):
        t0 = time.monotonic()
        out = hvp(v)
        dt = time.monotonic() - t0
        self.hvp_cost = 0.92 * self.hvp_cost + 0.08 * dt
        return out

    def solve(self, hvp, b, context):
        x = np.zeros(self.d)
        try:
            x = self._solve(hvp, b, context)
        except Exception:
            pass
        if not np.isfinite(x).all():
            x = np.zeros(self.d)
        return x

    def _solve(self, hvp, b, context):
        d = self.d
        now = time.monotonic()
        if self.t_start is None:
            self.t_start = now
            self.fallback_deadline = now + 27.8
        # absolute monotonic deadline, with a self-contained fallback
        dl = context.get('deadline') if isinstance(context, dict) else None
        if not isinstance(dl, (int, float)) or not np.isfinite(dl):
            dl = self.fallback_deadline
        else:
            left = dl - now
            if left < 0 or left > 300:
                dl = self.fallback_deadline
            else:
                dl = min(dl - 0.25, self.fallback_deadline)
        t = int(context.get('step', 0))
        steps = int(context.get('steps', 50))
        steps_left = max(1, steps - t)
        phase_changed = bool(context.get('phase_changed', False))
        hard = (t == 0) or phase_changed or self.xprev is None
        # With a very wide spectrum, a cold start converges much more slowly and
        # its error persists through the warm starts; spend more on those steps.
        if t == 0:
            mu = float(context.get('mu', 1.0))
            L = float(context.get('L', 1.0))
            kappa = max(1.0, L / max(mu, 1e-12))
            self.share = float(np.clip(2.0 + 2.0 * np.log10(kappa / 1e5), 2.0, 8.0))
        share = self.share if hard else 1.0
        remain = max(0.0, dl - now - 0.35)
        step_budget = remain * share / (steps_left - 1.0 + share)

        x = np.zeros(d)
        r = b.copy()
        if self.xprev is not None and not phase_changed and self.hvprev is not None:
            denom = float(self.xprev @ self.hvprev)
            c = float(b @ self.xprev) / denom if denom > 0 else 0.0
            if not np.isfinite(c):
                c = 0.0
            rho = float(b @ self.bprev) / max(float(self.bprev @ self.bprev), 1e-300)
            # the energy-optimal factor is trusted only near the AR(1) estimate
            c = min(max(c, 0.0), 1.2) if abs(c - rho) < 0.35 else min(max(rho, 0.0), 1.05)
            if c > 1e-8:
                x = c * self.xprev
                try:
                    r = b - self._hvp(hvp, x)
                except Exception:
                    x = np.zeros(d)
                    r = b.copy()

        n = int((step_budget - 0.015) / max(self.hvp_cost, 5e-4))
        n = min(self.cap, max(6, n))
        p = r.copy()
        rr = float(r @ r)
        guard = dl - 0.28
        for i in range(n):
            if time.monotonic() >= guard:
                break
            try:
                Ap = self._hvp(hvp, p)
            except Exception:
                break
            denom = float(p @ Ap)
            if not np.isfinite(denom) or denom <= 0:
                break
            a = rr / denom
            x += a * p
            r -= a * Ap
            rr2 = float(r @ r)
            if not np.isfinite(rr2) or rr2 <= 0:
                break
            p = r + (rr2 / rr) * p
            rr = rr2

        self.hvprev = b - r
        self.xprev = x.copy()
        self.bprev = b.copy()
        return x


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)
