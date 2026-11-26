"""Online SPD sequence solver.

Strategy
--------
For every step the evaluator hands us the current right hand side b_t and a
matrix free operator for the current H_t.  Two observations drive the design.

1.  The scored quantity is the normalised H-energy error

        l_t = (x-x*)^T H (x-x*) / (x*)^T H x*,      r = b - H x.

    With D = (x*)^T H x* = b^T H^{-1} b > 0 the identity

        A(x) := x^T H x - 2 b^T x = (x-x*)^T H (x-x*) - D = l_t * D - D

    shows that A is the loss minus a step constant.  A is therefore an *exact*
    and free yardstick: it only needs quantities the solver already has, and
    it compares candidate vectors against each other and against the zero
    vector (whose loss is exactly 1 by definition).  We use it as a guard
    against ever returning something worse than the trivial answer.

2.  Neighbouring systems are strongly correlated while the matrices drift
    slowly, so the previous solution is an excellent starting point: the
    residual left over from it is the only thing conjugate gradients has to
    clean up, and the error that is left over lives in benign directions.

Every step therefore

  * forms a starting point by minimising the exact loss functional A over
    the small span of the most recent solutions (a reduced space projection,
    also guarded by A),
  * runs conjugate gradients from there until the step's time slice is spent,
  * returns the better of {candidate, zero vector} under the exact criterion.

Robustness
----------
The evaluation harness is treated as unknown: every context field is read
defensively, the wall clock budget is anchored on the process start time as a
fallback for the supplied deadline, only single vector HVP queries are used,
and any failure degrades to the best vector known so far instead of raising.
"""

import sys
import time

import numpy as np

_T0 = time.monotonic()          # import time ~ process start


def _num(value, default):
    """Best effort float conversion."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


class Solver:
    def __init__(self, dimension, seed, options):
        self.d = int(dimension)
        self.opt = dict(options) if isinstance(options, dict) else {}
        self.hist = []            # recent solutions, newest first
        self.phase = None         # phase the stored solutions belong to
        self.nhvp = 0

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _field(context, key, default=None):
        """Read a context field without ever raising."""
        try:
            if hasattr(context, "get"):
                value = context.get(key, default)
            else:
                value = context[key]
        except Exception:
            try:
                value = getattr(context, key)
            except Exception:
                return default
        return default if value is None else value

    def _phase_of(self, context, step):
        phase = self._field(context, "phase", 0)
        if isinstance(phase, (list, tuple, np.ndarray)):
            try:
                idx = int(step)
                if 0 <= idx < len(phase):
                    phase = phase[idx]
                else:
                    phase = phase[-1]
            except Exception:
                phase = 0
        try:
            return int(phase)
        except (TypeError, ValueError):
            return 0

    def _budget_end(self, context, now):
        """Wall clock instant at which we must stop asking for HVP products.

        The evaluator supplies a deadline on the monotonic clock.  Should that
        field be missing or inconsistent we fall back to a budget anchored on
        this process' own start, which is always meaningful.
        """
        end = None
        raw = self._field(context, "deadline", None)
        if raw is None:
            raw = self._field(context, "sequence_deadline", None)
        if raw is None:
            raw = self._field(context, "hard_deadline", None)
        value = _num(raw, None)
        if value is not None:
            left = value - now
            if 0.05 < left < 3600.0:
                end = value
        if end is None:
            end = _T0 + 28.0
        end = min(end, _T0 + 29.0)
        # keep a margin so that the last query still lands before the oracle
        # stops answering and before the harness tears the process down
        return end - 0.15

    # ------------------------------------------------------------------ CG
    def _cg(self, hvp, b, x0, r0, rtol, step_deadline, maxit):
        """Conjugate gradients.  Returns (x, r, iterations, relative residual)."""
        nb = np.linalg.norm(b)
        x = x0.copy()
        r = r0.copy()
        rr = float(r @ r)
        res = np.sqrt(rr)
        if not np.isfinite(rr):
            return x, r, 0, 1.0
        if res <= rtol * nb:
            return x, r, 0, res / nb
        p = r.copy()
        it = 0
        for it in range(1, maxit + 1):
            if time.monotonic() > step_deadline:
                break
            try:
                Ap = hvp(p)
            except BaseException:
                break                      # oracle refused: keep what we have
            self.nhvp += 1
            pAp = float(p @ Ap)
            if not np.isfinite(pAp) or pAp <= 0.0:
                break
            alpha = rr / pAp
            x += alpha * p
            r -= alpha * Ap
            rr_new = float(r @ r)
            if not np.isfinite(rr_new):
                rr = rr_new
                break
            res = np.sqrt(rr_new)
            if res <= rtol * nb:
                rr = rr_new
                break
            p = r + (rr_new / rr) * p
            rr = rr_new
        return x, r, it, res / nb

    # --------------------------------------------------------------- solve
    def solve(self, hvp, b, context):
        try:
            vec = np.asarray(b, dtype=np.float64).reshape(-1)
            if vec.shape[0] != self.d:
                vec = np.zeros(self.d)
            return self._solve(hvp, vec, context)
        except BaseException as exc:                      # never fail a step
            try:
                sys.stderr.write(f"[err] {type(exc).__name__}: {exc}\n")
            except Exception:
                pass
            return self._fallback()

    def _fallback(self):
        if self.hist:
            x = self.hist[0]
            if x.shape == (self.d,) and np.isfinite(x).all():
                return x
        return np.zeros(self.d)

    def _solve(self, hvp, b, context):
        o = self.opt
        now = time.monotonic()
        deadline = self._budget_end(context, now)
        step = int(self._field(context, "step", 0))
        steps = int(self._field(context, "steps", 50) or 50)
        phase = self._phase_of(context, step)
        phase_changed = bool(self._field(context, "phase_changed", False))

        # ---- time slicing -------------------------------------------------
        # every step gets an equal share of what is left; the first step of a
        # phase is the least accurate one and gets a larger slice.
        remaining = max(0.0, deadline - now)
        steps_left = max(1, steps - step)
        allow = remaining / steps_left
        if step == 0:
            allow *= float(o.get("first_boost", 4.0))
        elif phase_changed:
            allow *= float(o.get("phase_boost", 6.0))
        step_deadline = min(deadline, now + allow)
        maxit = int(o.get("maxit", 1 << 30))
        rtol = float(o.get("rtol", 1e-12))
        khist = int(o.get("hist", 5))

        # ---- starting point ------------------------------------------------
        x0 = np.zeros(self.d)
        r0 = b.copy()
        same_phase = (self.phase == phase)
        cross = bool(o.get("cross_phase", 1))
        use_hist = bool(o.get("predict", 1)) and (same_phase or cross) and len(self.hist) > 0
        if use_hist and time.monotonic() < deadline - 0.20:
            Xs = list(self.hist[:khist])
            if bool(o.get("inc_b", 0)):
                Xs.append(b)
            k = len(Xs)
            try:
                Ws = [hvp(xi) for xi in Xs]              # one query per column
                self.nhvp += k
                if all(w is not None and w.shape == (self.d,) and np.isfinite(w).all()
                       for w in Ws):
                    M = np.empty((k, k))
                    for i in range(k):
                        for j in range(k):
                            M[i, j] = 0.5 * (float(Xs[i] @ Ws[j]) + float(Xs[j] @ Ws[i]))
                    rhs = np.array([float(xi @ b) for xi in Xs])
                    lam = float(o.get("reg", 1e-8)) * max(float(np.trace(M)) / k, 1e-300)
                    try:
                        c = np.linalg.solve(M + lam * np.eye(k), rhs)
                    except np.linalg.LinAlgError:
                        c = None
                    if c is not None and np.isfinite(c).all():
                        cand = np.zeros(self.d)
                        for i in range(k):
                            cand += c[i] * Xs[i]
                        rc = b.copy()
                        for j in range(k):
                            rc -= c[j] * Ws[j]
                        if np.isfinite(cand).all() and np.isfinite(rc).all():
                            # exact comparison against the zero vector:
                            # loss(cand) < 1  <=>  cand'(b + rc) > 0
                            if float(cand @ (b + rc)) > 0.0:
                                x0, r0 = cand, rc
            except BaseException:
                use_hist = False
                x0 = np.zeros(self.d)
                r0 = b.copy()
        elif use_hist:
            use_hist = False

        # ---- conjugate gradients --------------------------------------------
        if time.monotonic() <= step_deadline:
            x, r, it, rel = self._cg(hvp, b, x0, r0, rtol, step_deadline, maxit)
        else:
            x, r, it, rel = x0, r0, 0, 1.0

        if x.shape != (self.d,) or not np.isfinite(x).all():
            x = x0.copy()
            r = b - x
        # exact comparison against the zero vector (its loss is 1 by definition)
        best = x if float(x @ (b + r)) > 0.0 else np.zeros(self.d)
        self.hist.insert(0, best.copy())
        if len(self.hist) > khist:
            self.hist.pop()
        self.phase = phase
        try:
            sys.stderr.write(f"[s{step}] ph={phase} it={it} rel={rel:.3g} hist={int(use_hist)} "
                             f"nh={self.nhvp}\n")
        except Exception:
            pass
        return best


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)
