"""Online SPD sequence solver (single process, one thread, numpy only).

Per step:
  1. Build a starting guess from the stored solution history.  The coefficients
     are fitted in right-hand-side space (a tiny least-squares problem), and the
     multi-vector fit is only used when it predicts the current b measurably
     better than the single previous one.
  2. One HVP gives H x0; the guess is then rescaled optimally (free) which also
     neutralises the stale guess after a phase jump, and the cold start is used
     whenever it would give a smaller residual.
  3. Conjugate gradients runs until this step's share of the sequence budget is
     spent.  The share is uniform except for a decaying boost at the start of a
     phase, where the error is largest and the payoff compounds down the phase.

HVP failures (e.g. the oracle budget message) end the iteration and return the
best iterate so far; solve() never raises and always returns a finite vector.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_monotonic = time.monotonic
_env = os.environ.get

LOG = bool(int(_env('SOL_LOG', '0')))
P_HIST = int(_env('SOL_P', '3'))                 # max history vectors in the fit
AR_GATE = float(_env('SOL_GATE', '0.98'))        # use AR(p) only if res < gate*res(1)
DECAY_B = float(_env('SOL_DB', '0.9'))           # boost amplitude at a phase start
DECAY_T = float(_env('SOL_DT', '2.5'))           # boost decay length in steps
W_LO, W_HI = 0.8, 1.9
MARGIN = float(_env('SOL_MARGIN', '0.10'))       # s kept before the soft deadline
FALLBACK_BUDGET = float(_env('SOL_BUDGET', '28.0'))


def _log(*a):
    if LOG:
        print(*a, file=sys.stderr, flush=True)


class _Solver:
    def __init__(self, d, seed, options):
        self.d = int(d)
        self.seed = seed
        self.options = options or {}
        self.x_hist = []                 # oldest .. newest, length <= P_HIST
        self.b_hist = []
        self.step = 0
        self.phase = None
        self.since_phase = 0
        self.t_init = _monotonic()
        self._x = np.zeros(self.d)
        self._r = np.zeros(self.d)
        self._p = np.zeros(self.d)

    # ------------------------------------------------------------------
    def solve(self, hvp, b, context):
        d = self.d
        diag = {}
        try:
            x, diag = self._solve(hvp, b, context)
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (d,) or not np.isfinite(x).all():
                raise ValueError('bad output vector')
            x = np.ascontiguousarray(x)
        except BaseException as exc:                 # never let anything escape
            _log('solve failure', type(exc).__name__, exc)
            x = self.x_hist[-1] if self.x_hist else np.zeros(d)
            try:
                x = np.array(x, dtype=np.float64)
            except BaseException:
                x = np.zeros(d)
            if x.shape != (d,) or not np.isfinite(x).all():
                x = np.zeros(d)
            diag = {'error': repr(exc)[:200]}
        self.x_hist.append(x)
        self.b_hist.append(np.array(b, dtype=np.float64, copy=True))
        keep = max(P_HIST, 1)
        if len(self.x_hist) > keep:
            self.x_hist.pop(0)
            self.b_hist.pop(0)
        self.step += 1
        return x, diag

    # ------------------------------------------------------------------
    def _warm_start(self, b, bn, bn2):
        """x0 = sum_j c_j x_{t-j} with c fitted in b-space.  Returns False if
        no history is available."""
        x = self._x
        nh = len(self.x_hist)
        if nh == 0 or bn <= 0.0:
            x[:] = 0.0
            return False
        b1 = self.b_hist[nh - 1]
        n1 = float(b1 @ b1)
        c1 = float(b1 @ b) / n1 if n1 > 0.0 else 1.0
        # AR(1) residual norm (also the value the multi-vector fit must beat)
        res1 = bn2 - c1 * c1 * n1
        res1 = res1 if res1 > 0.0 else 0.0
        p = min(P_HIST, nh)
        used = False
        if p >= 2:
            G = np.empty((p, p))
            rhs = np.empty(p)
            for i in range(p):
                bi = self.b_hist[nh - 1 - i]
                rhs[i] = float(bi @ b)
                for j in range(i, p):
                    v = float(bi @ self.b_hist[nh - 1 - j])
                    G[i, j] = v
                    G[j, i] = v
            sc = np.sqrt(np.diag(G))
            sc[sc <= 0.0] = 1.0
            try:
                c = np.linalg.lstsq(G / np.outer(sc, sc), rhs / sc, rcond=1e-8)[0] / sc
                # residual of the fit: |b|^2 - 2 c.rhs + c.G.c
                resc = bn2 - 2.0 * float(c @ rhs) + float(c @ G @ c)
                if (np.isfinite(c).all() and np.abs(c).sum() <= 2.0
                        and resc >= 0.0 and resc < AR_GATE * AR_GATE * res1):
                    x[:] = 0.0
                    for j in range(p):
                        x += c[j] * self.x_hist[nh - 1 - j]
                    used = bool(np.isfinite(x).all())
            except np.linalg.LinAlgError:
                used = False
        if not used:
            np.copyto(x, self.x_hist[-1])
            if n1 > 0.0 and np.isfinite(c1) and abs(c1) <= 4.0:
                x *= c1
            if not np.isfinite(x).all():
                np.copyto(x, self.x_hist[-1])
        return True

    # ------------------------------------------------------------------
    def _solve(self, hvp, b, context):
        d = self.d
        step = int(context.get('step', self.step))
        steps = int(context.get('steps', 50))
        t0 = _monotonic()
        soft = float(context.get('deadline') or 0.0)
        hard = float(context.get('hard_deadline') or 0.0)
        if soft <= 0.0:
            soft = self.t_init + FALLBACK_BUDGET
        if hard <= 0.0:
            hard = soft + 1.2
        stop_all = min(soft - MARGIN, hard - 0.70)
        if stop_all <= t0:                       # already out of time
            stop_all = t0 + 0.005
        remaining_steps = max(steps - step, 1)

        ph = context.get('phase')
        if context.get('phase_changed') or self.phase is None or ph != self.phase:
            self.since_phase = 0
        else:
            self.since_phase += 1
        self.phase = ph
        wgt = 1.0 + DECAY_B * np.exp(-self.since_phase / DECAY_T) if DECAY_B > 0.0 else 1.0
        wgt = W_HI if wgt > W_HI else (W_LO if wgt < W_LO else wgt)
        left = stop_all - t0
        per = left * wgt / (wgt + remaining_steps - 1)
        stop = t0 + per

        b = np.asarray(b, dtype=np.float64)
        bn2 = float(b @ b)
        bn = np.sqrt(bn2)
        x = self._x
        r = self._r
        p = self._p
        if bn == 0.0:
            x[:] = 0.0
            return np.zeros(d), {'it': 0}

        r0_rel = 1.0
        budget = False
        if self._warm_start(b, bn, bn2) and np.isfinite(x).all():
            try:
                w = hvp(x)
            except BaseException as exc:
                # out of oracle budget: keep the (finite) starting guess as the answer
                _log('warm-start hvp failed', type(exc).__name__)
                budget = True
                w = None
                r[:] = 0.0
            if w is not None:
                np.subtract(b, w, out=r)
                rn2 = float(r @ r)
                ww = float(w @ w)
                if ww > 0.0 and np.isfinite(ww):
                    bw = float(b @ w)
                    cs = bw / ww
                    if 0.0 <= cs <= 4.0:
                        alt = bn2 - cs * bw
                        if 0.0 <= alt < rn2:
                            x *= cs
                            np.copyto(r, b)
                            r -= cs * w
                            rn2 = alt
                rn = np.sqrt(max(rn2, 0.0))
                if rn >= bn:                     # cold start is free and better
                    x[:] = 0.0
                    np.copyto(r, b)
                    rn = bn
                r0_rel = rn / bn
        else:
            x[:] = 0.0
            np.copyto(r, b)

        rs = float(r @ r)
        np.copyto(p, r)
        it = 0
        restarts = 0
        tol = 1e-26 * bn2
        if not budget:
            while rs > tol:
                if _monotonic() > stop:
                    break
                try:
                    Ap = hvp(p)
                except BaseException as exc:     # oracle budget / protocol error
                    _log('hvp failed at it', it, type(exc).__name__)
                    break
                pAp = float(p @ Ap)
                if not np.isfinite(pAp) or pAp <= 0.0:
                    restarts += 1
                    if restarts > 3:
                        break
                    try:
                        np.subtract(b, hvp(x), out=r)
                    except BaseException:
                        break
                    if not np.isfinite(r).all():
                        break
                    rs = float(r @ r)
                    np.copyto(p, r)
                    continue
                alpha = rs / pAp
                x += alpha * p
                r -= alpha * Ap
                rs_new = float(r @ r)
                if not np.isfinite(rs_new):
                    break
                p *= (rs_new / rs)
                p += r
                rs = rs_new
                it += 1

        rel = float(np.sqrt(max(rs, 0.0)) / bn)
        _log(f"step {step:2d} it {it:4d} rel {rel:.3e} r0 {r0_rel:.4f} ph{ph} "
             f"s{self.since_phase} t {(_monotonic()-t0)*1e3:6.0f}ms w {wgt:.2f}")
        if not np.isfinite(x).all():
            x[:] = 0.0
        return x.copy(), {'it': it, 'rel': rel, 'r0': r0_rel}


def make_solver(dimension, seed, options):
    return _Solver(dimension, seed, options)
