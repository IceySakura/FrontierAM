"""Online SPD sequence solver.

Per step: Galerkin warm start on recent solutions, then CG that continues the
previous step's search direction instead of restarting (exact line search along
it, later directions kept H-conjugate to it), with a time-aware budget that
front-loads cold starts.
"""
import sys
import time

import numpy as np

_clock = time.monotonic


def set_clock(fn):
    global _clock
    _clock = fn


class _OutOfTime(Exception):
    pass


class _BadVector(Exception):
    pass


class Solver:
    def __init__(self, dimension, seed, options):
        self.d = int(dimension)
        self.options = dict(options or {})
        self.hist = []          # previous solutions (most recent last)
        self.max_hist = 2
        self.c_hvp = 3.0e-3     # running estimate of one HVP round trip (seconds)
        self.c_iter = 0.1e-3    # running estimate of solver-side work per CG iteration
        self.emergency = False
        self.prev_b = None
        self.rho_phase = []     # cosines between consecutive rhs within the current phase
        self.rho_est = None
        self.last_change = 0
        self.phase_lengths = []
        self.boost = 3.0
        self.p_last = None      # previous step's last search direction (unit norm)

    # ------------------------------------------------------------------ budgeting
    def _budget(self, ctx, now):
        steps = int(ctx.get("steps", 50))
        step = int(ctx.get("step", 0))
        deadline = float(ctx.get("deadline", now + 1.0))
        left = deadline - now - 0.25
        if ctx.get("phase_changed", False) or step == 0:
            if step > 0:
                self.phase_lengths.append(step - self.last_change)
            self.last_change = step
            self.p_last = None
            if step == 0:
                self.boost = 3.0
            else:
                rho = self.rho_est if self.rho_est is not None else 0.9
                horizon = min(1.0 / max(1e-3, 1.0 - rho ** 2), steps - step,
                              float(np.mean(self.phase_lengths)))
                self.boost = float(np.clip(1.0 + 0.35 * horizon, 1.0, 5.0))
        # expected weight of a future step: when phases are short, reserve budget for the
        # boosts of phase changes that are likely to recur (long phases favour front-loading)
        w_future = 1.0
        if self.phase_lengths and float(np.mean(self.phase_lengths)) < 14.0:
            plen = max(1.0, float(np.mean(self.phase_lengths)))
            rho = self.rho_est if self.rho_est is not None else 0.9
            b_future = float(np.clip(1.0 + 0.35 * min(1.0 / max(1e-3, 1.0 - rho ** 2), plen), 1.0, 5.0))
            w_future = 1.0 + (b_future - 1.0) / plen
        w_now = self.boost if step == self.last_change else 1.0
        share = w_now / (w_now + (steps - step - 1) * w_future)
        return now + max(0.0, left * share), deadline - 0.1

    def _track_rho(self, b, ctx):
        if ctx.get("phase_changed", False) and int(ctx.get("step", 0)) > 0:
            if len(self.rho_phase) >= 2:
                self.rho_est = float(np.median(self.rho_phase))
            self.rho_phase = []
        elif self.prev_b is not None:
            nb = np.linalg.norm(b) * np.linalg.norm(self.prev_b)
            if nb > 0:
                self.rho_phase.append(float(b @ self.prev_b) / nb)
                if len(self.rho_phase) >= 3:
                    self.rho_est = float(np.median(self.rho_phase))
        self.prev_b = b.copy()

    # ------------------------------------------------------------------ main entry
    def solve(self, hvp, b, ctx):
        now = _clock()
        b = np.asarray(b, dtype=np.float64)
        d = self.d
        try:
            self._track_rho(b, ctx)
        except Exception:
            pass
        if self.emergency:
            return self._fallback(b), {"mode": "emergency"}
        step_end, hard_end = self._budget(ctx, now)

        def H(v):
            if not np.isfinite(v).all():
                raise _BadVector()
            t = _clock()
            if t + self.c_hvp > hard_end:
                raise _OutOfTime()
            try:
                out = hvp(v)
            except Exception:
                self.emergency = True
                raise _OutOfTime()
            return np.asarray(out, dtype=np.float64)

        x = np.zeros(d)
        r = b.copy()
        info = {}
        best = None
        try:
            if self.hist:
                x, r = self._warm(H, b)
            best = x.copy()
            x, its = self._cg(H, b, x, r, step_end, info)
            info["its"] = its
        except _OutOfTime:
            self.emergency = True
            x = best if best is not None else self._fallback(b)
        except Exception as exc:  # never fail a step
            print(f"[solver] step {ctx.get('step')} error {type(exc).__name__}: {exc}", file=sys.stderr)
            x = best if best is not None else self._fallback(b)
        if not np.isfinite(x).all():
            x = self._fallback(b)
        self.hist.append(x.copy())
        self.hist = self.hist[-self.max_hist:]
        return x, info

    def _warm(self, H, b):
        S = np.array(self.hist[-self.max_hist:][::-1]).T   # most recent first
        Qs, R = np.linalg.qr(S)
        diag = np.abs(np.diag(R))
        keep = diag > 1e-8 * diag[0]
        Qs = Qs[:, keep]
        if Qs.shape[1] == 0:
            return np.zeros(self.d), b.copy()
        cols = [H(np.ascontiguousarray(Qs[:, j])) for j in range(Qs.shape[1])]
        HQ = np.stack(cols, axis=1)
        G = Qs.T @ HQ
        G = 0.5 * (G + G.T)
        ev, EV = np.linalg.eigh(G)
        ok = ev > 1e-12 * max(ev.max(), 1e-300)
        if not ok.any():
            return np.zeros(self.d), b.copy()
        y = EV[:, ok] @ ((EV[:, ok].T @ (Qs.T @ b)) / ev[ok])
        x0 = Qs @ y
        r0 = b - HQ @ y
        if not (np.isfinite(x0).all() and np.isfinite(r0).all()):
            return np.zeros(self.d), b.copy()
        return x0, r0

    def _fallback(self, b):
        if self.hist:
            return self.hist[-1].copy()
        return np.zeros(self.d)

    def _deflation_setup(self, H, x, r, info):
        """Exact Galerkin step along the previous step's last search direction.

        Returns (x, r, (w, Hw, g)) so CG can stay H-conjugate to w, or (x, r, None)."""
        w = self.p_last
        self.p_last = None
        if w is None:
            return x, r, None
        Hw = H(w)
        g = float(w @ Hw)
        if not (g > 0.0) or not np.isfinite(g):
            return x, r, None
        al = float(w @ r) / g
        if np.isfinite(al):
            x = x + al * w
            r = r - al * Hw
        return x, r, (w, Hw, g)

    def _cg(self, H, b, x, r, step_end, info):
        x, r, defl = self._deflation_setup(H, x, r, info)
        d = self.d
        z = np.empty(d)
        tmp = np.empty(d)
        if defl is not None:
            w_d, Hw_d, g_d = defl

            def proj(v):
                c = float(Hw_d @ v) / g_d
                np.multiply(w_d, c, out=z)
                np.subtract(v, z, out=z)
                return z
        else:
            def proj(v):
                return v
        nring = 0
        p = np.array(proj(r), dtype=np.float64, copy=True)
        if defl is not None and not (np.linalg.norm(p) > 1e-3 * np.linalg.norm(r)):
            def proj(v):
                return v
            p = r.copy()
        rr = float(r @ r)
        its = 0
        x_safe = x.copy()
        bn2 = float(b @ b)
        tol = 1e-24 * bn2
        plain_tol = 1e-12 * bn2      # below relative residual 1e-6 fall back to plain CG
        deflating = defl is not None
        while rr > tol:
            t = _clock()
            if t + self.c_hvp + self.c_iter > step_end:
                break
            try:
                Ap = H(p)
            except _OutOfTime:
                self.emergency = True
                break
            except _BadVector:
                break
            t2 = _clock()
            self.c_hvp = 0.8 * self.c_hvp + 0.2 * (t2 - t)
            pAp = float(np.dot(p, Ap))
            if not (pAp > 0.0) or not np.isfinite(pAp):
                break
            nring += 1
            a = float(np.dot(p, r)) / pAp
            if not np.isfinite(a):
                break
            np.multiply(p, a, out=tmp)
            x += tmp
            np.multiply(Ap, a, out=tmp)
            r -= tmp
            rr_new = float(np.dot(r, r))
            if not np.isfinite(rr_new):
                return x_safe, its
            if deflating and rr_new < plain_tol:
                def proj(v):
                    return v
                deflating = False
                p[:] = r
                rr = rr_new
                its += 1
                continue
            zz = proj(r)
            p *= rr_new / rr
            p += zz
            rr = rr_new
            its += 1
            if its % 32 == 0:
                if np.isfinite(x[:64]).all():
                    x_safe = x.copy()
                else:
                    return x_safe, its
            self.c_iter = 0.8 * self.c_iter + 0.2 * (_clock() - t2)
        if nring > 0 and np.isfinite(p).all():
            pn = float(np.linalg.norm(p))
            if pn > 0:
                self.p_last = p / pn
        if not np.isfinite(x).all():
            return x_safe, its
        return x, its


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)
