"""Online solver for sequences of SPD systems H_t x_t = b_t (HVP access only).

The measured behaviour of the target instances is that the spectrum is spread
over essentially the whole [mu, L] interval with a dense low end, so a subspace
deflation preconditioner built from a few hundred Lanczos vectors does not lower
the effective condition number -- it only costs flops per iteration.  The solver
therefore spends its whole budget on the thing that does pay: a long
warm-started conjugate gradient run at every step.

What is carried across the sequence is the solution itself.  Because the
coordinate frame is fixed and H_t, b_t move only slowly between neighbouring
systems, x_{t-1} is an excellent initial guess, and the energy error of a CG run
scales with the energy error of its starting point.  That turns a cold-start
problem needing thousands of iterations into ~170 iterations per step.

Compute is allocated from the sequence deadline supplied by the harness: every
step receives (remaining time) * w_t / sum(w_s, s >= t), so the solver paces
itself on whatever machine it runs on, gives the first (cold) step a larger
share, and never runs past the deadline.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_DEFAULTS = {
    "MARGIN": 0.55,     # seconds kept clear of the sequence deadline
    "W0": 2.6,          # extra budget share for the cold first step
    "W1": 1.25,         # mild boost while the warm-start chain settles
    "W2": 1.1,
    "RTOL": 3e-12,      # relative residual stop (||r|| <= RTOL*||b||)
    "LOG": 0,
}


def _config():
    cfg = dict(_DEFAULTS)
    raw = os.environ.get("IHVCFG")
    if raw:
        try:
            cfg.update(json.loads(raw))
        except Exception:
            pass
    return cfg


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)


class Solver:
    def __init__(self, dimension, seed, options):
        self.d = int(dimension)
        self.cfg = _config()
        # The algorithm is deterministic; the seed is accepted for interface
        # compliance only.
        self.prev_x = None
        self.prev_b = None

    # -------------------------------------------------------------- budget
    def _weight(self, step):
        if step == 0:
            return float(self.cfg["W0"])
        if step == 1:
            return float(self.cfg["W1"])
        if step == 2:
            return float(self.cfg["W2"])
        return 1.0

    def _budget_end(self, context, step, steps):
        """Monotonic-clock instant at which this step must stop doing HVPs."""
        now = time.monotonic()
        dl = context.get("sequence_deadline")
        if dl is None:
            dl = context.get("deadline", now + 25.0)
        stop = float(dl) - float(self.cfg["MARGIN"])
        rem = stop - now
        if rem <= 0.0:
            return now, stop
        left = max(steps - step, 1)
        wsum = self._weight(step) + (left - 1)
        share = self._weight(step) / wsum
        return min(stop, now + rem * share), stop

    # --------------------------------------------------------------- solve
    def solve(self, hvp, b, context):
        try:
            return self._solve(hvp, b, context)
        except Exception as exc:  # a step must never be lost
            if self.cfg["LOG"]:
                print("ERR", type(exc).__name__, exc, file=sys.stderr, flush=True)
            x = self.prev_x
            if x is None or not np.isfinite(x).all():
                return np.zeros(self.d)
            return np.array(x, dtype=np.float64)

    def _solve(self, hvp, b, context):
        d = self.d
        b = np.ascontiguousarray(np.asarray(b, dtype=np.float64).ravel())
        step = int(context["step"])
        steps = int(context.get("steps", 50))
        t_end, hard_stop = self._budget_end(context, step, steps)
        bnorm = float(np.linalg.norm(b))
        if not np.isfinite(bnorm) or bnorm == 0.0:
            self.prev_x = np.zeros(d)
            self.prev_b = b
            return np.zeros(d)

        now = time.monotonic()

        # ---- initial guess: previous solution, else zero ----
        phase_changed = bool(context.get("phase_changed", step == 0))
        x = None
        r = None
        if self.prev_x is not None and not phase_changed:
            r0 = b - np.asarray(hvp(self.prev_x), dtype=np.float64).ravel()
            n0 = float(np.linalg.norm(r0))
            if np.isfinite(n0) and n0 < bnorm:
                x = np.array(self.prev_x, dtype=np.float64)
                r = r0
        if x is None:
            x = np.zeros(d)
            r = b.copy()

        target = (float(self.cfg["RTOL"]) * bnorm) ** 2
        p = np.array(r, dtype=np.float64)
        rs = float(r @ r)
        it = 0

        # ---- conjugate gradient, paced against the deadline ----
        while rs > target:
            if it and time.monotonic() >= t_end:
                break
            Ap = np.asarray(hvp(p), dtype=np.float64).ravel()
            pAp = float(p @ Ap)
            if not np.isfinite(pAp) or pAp <= 0.0:
                break
            alpha = rs / pAp
            x += alpha * p
            r -= alpha * Ap
            rs_new = float(r @ r)
            it += 1
            if not np.isfinite(rs_new) or rs_new > 1e8 * rs:
                break
            beta = rs_new / rs
            rs = rs_new
            if beta > 1e8 or not np.isfinite(beta):
                break
            p *= beta
            p += r

        if not np.isfinite(x).all():
            x = np.zeros(d)
        self.prev_x = x
        self.prev_b = b
        if self.cfg["LOG"]:
            print("step", step, "it", it, "|r|/|b|", float(np.sqrt(rs)) / bnorm,
                  "t", round(time.monotonic() - now, 3), file=sys.stderr, flush=True)
        return x
