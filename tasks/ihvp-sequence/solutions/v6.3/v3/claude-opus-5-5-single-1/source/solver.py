"""Online SPD sequence solver: warm-started CG with Galerkin restart and stale-Krylov recycling.

Each step: (1) energy-optimal scaling of the previous solution with the current H (1 HVP);
(2) a few corrections built from the previous step's H-conjugate CG directions (approximate inverse on the
old Krylov space, stale but cheap), each accepted only through an exact line search with the current H;
(3) plain CG until the per-step time budget is used. Never raises; always returns a finite vector.
"""
import sys
import time
import numpy as np


class Solver:
    def __init__(self, dimension, seed, options):
        self.d = int(dimension)
        opts = options if isinstance(options, dict) else {}
        self.m_store = int(opts.get('m_store', 0))
        self.n_corr = int(opts.get('n_corr', 0))
        self.margin = float(opts.get('margin', 0.9))
        self.first_mult = float(opts.get('first_mult', 3.0))
        self.tol = float(opts.get('tol', 1e-13))
        self.x_prev = None
        self.P = None       # stored CG directions (rows) of the previous step
        self.D = None       # p^T H p for the stored directions
        self.m = 0
        self.Pn = None      # buffer being filled in the current step
        self.best = None
        self.created = time.monotonic()

    def solve(self, hvp, b, context):
        d = self.d
        self.best = None
        info = {}
        try:
            b = np.asarray(b, dtype=np.float64).reshape(d)
            x = self._solve(hvp, b, context, info)
        except Exception as exc:  # never let an exception escape (budget, numerics)
            print(f"[solver] step {context.get('step')} exception {type(exc).__name__}: {exc}", file=sys.stderr)
            x = self.best
            info['exception'] = type(exc).__name__
        if x is None or np.shape(x) != (d,) or not np.all(np.isfinite(x)):
            x = np.zeros(d)
        x = np.array(x, dtype=np.float64)
        if np.any(x):
            self.x_prev = x.copy()
        return x, info

    def _solve(self, hvp, b, ctx, info):
        d = self.d
        t0 = time.monotonic()
        t = int(ctx.get('step', 0))
        T = int(ctx.get('steps', 50))
        try:
            ctx_deadline = float(ctx.get('deadline', ctx.get('sequence_deadline')))
        except (TypeError, ValueError):
            ctx_deadline = float('inf')
        if not np.isfinite(ctx_deadline):
            ctx_deadline = float('inf')
        deadline = min(ctx_deadline, self.created + 28.3) - self.margin
        left = deadline - t0
        steps_left = max(1, T - t)
        budget = max(0.0, left) / steps_left
        if (t == 0 or ctx.get('phase_changed', False)) and steps_left > 1:
            budget = min(budget * self.first_mult, 0.5 * max(0.0, left))
        t_end = t0 + budget
        bn = float(np.linalg.norm(b))
        if not np.isfinite(bn) or bn == 0.0:
            return np.zeros(d)
        if left <= 0.05:
            if self.x_prev is not None and not ctx.get('phase_changed', False):
                return self.x_prev.copy()
            return np.zeros(d)

        x = np.zeros(d)
        r = b.copy()
        self.best = x
        # (1) Galerkin on the previous solution.
        if self.x_prev is not None:
            xp = self.x_prev
            Hxp = np.asarray(hvp(xp), dtype=np.float64)
            den = float(xp @ Hxp)
            if den > 0 and np.isfinite(den):
                g = float(xp @ b) / den
                x = g * xp
                r = b - g * Hxp
                self.best = x
        r0n = float(np.linalg.norm(r)) / bn
        # (2) Stale-Krylov corrections with exact line search.
        ncorr = 0
        if self.P is not None and self.m > 0 and not ctx.get('phase_changed', False):
            P = self.P[:self.m]
            Dinv = 1.0 / self.D[:self.m]
            for _ in range(self.n_corr):
                delta = P.T @ (Dinv * (P @ r))
                Hd = np.asarray(hvp(delta), dtype=np.float64)
                den = float(delta @ Hd)
                if not (den > 0) or not np.isfinite(den):
                    break
                gam = float(delta @ r) / den
                x = x + gam * delta
                r = r - gam * Hd
                self.best = x
                ncorr += 1
        r1n = float(np.linalg.norm(r)) / bn

        # (3) CG, storing directions for the next step.
        if self.Pn is None:
            self.Pn = np.empty((self.m_store, d))
            self.Dn = np.empty(self.m_store)
        Pn, Dn = self.Pn, self.Dn
        mn = 0
        rr = float(r @ r)
        tol = self.tol * bn
        p = r.copy()
        it = 0
        while True:
            rn = np.sqrt(rr)
            if not np.isfinite(rn) or rn <= tol or time.monotonic() >= t_end:
                break
            Hp = np.asarray(hvp(p), dtype=np.float64)
            pHp = float(p @ Hp)
            if not (pHp > 0.0) or not np.isfinite(pHp):
                break
            if mn < self.m_store:
                Pn[mn] = p
                Dn[mn] = pHp
                mn += 1
            alpha = rr / pHp
            x = x + alpha * p
            self.best = x
            r -= alpha * Hp
            rr_new = float(r @ r)
            beta = rr_new / rr
            rr = rr_new
            it += 1
            p = r + beta * p
        # swap buffers: this step's directions become next step's recycled basis
        if mn > 0:
            self.Pn, self.P = self.P, Pn
            self.Dn, self.D = self.D, Dn
            self.m = mn
        info.update(iters=it, relres=float(np.sqrt(rr)) / bn, ncorr=ncorr)
        el = time.monotonic() - t0
        print(f"[solver] t={t} ph={ctx.get('phase')} chg={ctx.get('phase_changed')} r0={r0n:.2e} r1={r1n:.2e} "
              f"it={it} res={info['relres']:.2e} time={el:.3f}/{budget:.3f}", file=sys.stderr)
        return x


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)
