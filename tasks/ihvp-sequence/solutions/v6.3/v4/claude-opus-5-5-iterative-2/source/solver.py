"""Online solver for sequences of SPD systems H_t x = b_t accessed through HVPs.

Each step runs CG on the current system, using what earlier steps left behind:

* Warm start: Galerkin projection on the span of the recent solutions X and of
  a small recycled block P (the last CG directions of the previous step, made
  H-orthonormal).  Products H x_s are the ones known when x_s was computed
  (H_s x_s = b_s - r_s); the latest one is recomputed each step to measure how
  fast H drifts, and everything is refreshed after a phase change or a large
  drift (then the recycled block is dropped).
* Deflated CG: search directions are kept H-orthogonal to P, so the new Krylov
  process continues where the previous step's left off instead of
  rediscovering the same slowly converging directions.  It falls back to plain
  CG at high accuracy, on stagnation or when the block is exhausted.
* Time: the remaining time is split evenly over the remaining steps.  A step
  that starts a new phase may run longer (bounded) while its energy-loss
  decrease per iteration stays above the level observed at the end of ordinary
  steps; the loss decrease per CG iteration is alpha * r'r, known exactly.

solve() never raises: on any error it returns the best finite iterate so far.
"""
import time

import numpy as np

_now = time.monotonic

HIST = 6            # recent solutions in the warm-start space
KEEP = 8            # last CG directions recycled into the next step
MARGIN = 0.35       # seconds kept free before the sequence deadline
BOOST_CAP = 5.0     # a phase-start step may use up to this many even shares
BOOST_MIN = 1.5     # ... and runs at least this many before checking progress
BOOST_GAP = 4       # minimum distance between two long steps
BOOST_FRAC = 0.25   # at most this share of the whole budget goes to long steps
BOOST_KEEP = 0.85   # ordinary steps keep at least this share of an even split
TAU0 = 2e-4         # progress threshold before any ordinary step was observed
DRIFT_TOL = 0.05    # relative change of H x_prev that triggers a full refresh


def _psolve(G, rhs, rel):
    """Solve the symmetric system G y = rhs on its numerically positive eigenspace."""
    G = 0.5 * (G + G.T)
    lam, S = np.linalg.eigh(G)
    top = lam[-1] if lam.size else 0.0
    if not (top > 0) or not np.isfinite(top):
        return np.zeros_like(rhs)
    keep = lam > rel * top
    return S[:, keep] @ ((S[:, keep].T @ rhs) / lam[keep])


def _h_orthonormal(P, W):
    """Rows of P with products W = P H -> H-orthonormal rows (dependent ones dropped)."""
    G = P @ W.T
    lam, S = np.linalg.eigh(0.5 * (G + G.T))
    if not lam.size or not (lam[-1] > 0) or not np.isfinite(lam).all():
        return None, None
    ok = lam > 1e-8 * lam[-1]
    T = S[:, ok] / np.sqrt(lam[ok])
    return T.T @ P, T.T @ W


class Solver:
    def __init__(self, dimension, seed, options):
        self.d = int(dimension)
        self.X = []            # recent solutions
        self.HX = []           # their products, as last computed
        self.P = None          # recycled H-orthonormal directions (rows) ...
        self.W = None          # ... and their products
        self.tail = []         # end-of-step progress per iteration of ordinary steps
        self.total = None
        self.boost_used = 0.0
        self.last_boost = -10 ** 9
        self._cur = None

    def solve(self, hvp, b, context):
        self._cur = None
        try:
            return self._solve(hvp, b, context)
        except Exception:  # noqa: BLE001 - an exception would invalidate the step
            x = self._cur
            if x is None or np.shape(x) != (self.d,) or not np.isfinite(x).all():
                x = np.zeros(self.d)
            self.X, self.HX = [], []
            self.P = self.W = None
            return np.array(x, dtype=np.float64)

    def _solve(self, hvp, b, context):
        t_start = _now()
        d = self.d
        step = int(context.get("step", 0))
        steps = int(context.get("steps", 50))
        deadline = float(context.get("deadline", context.get("sequence_deadline", t_start + 0.5)))
        b = np.asarray(b, dtype=np.float64).reshape(d)
        left = max(1, steps - step)
        phase_start = step == 0 or bool(context.get("phase_changed"))

        # ---------------- time slice
        avail = deadline - MARGIN - t_start
        base = avail / left
        if self.total is None:
            self.total = max(avail, 0.0)
        boost = phase_start and step - self.last_boost >= BOOST_GAP and left > 1
        extra_left = BOOST_FRAC * self.total - self.boost_used
        if boost and extra_left > 0.5 * base:
            self.last_boost = step
            extra = min(base * (BOOST_CAP - 1), avail - BOOST_KEEP * base * (left - 1) - base, extra_left)
            t_hard = t_start + base + max(extra, 0.0)
            t_soft = min(t_start + base * BOOST_MIN, t_hard)
        else:
            boost = False
            t_soft = t_hard = t_start + base
        tau = float(np.median(self.tail[-5:])) if self.tail else TAU0

        # ---------------- warm start
        x = np.zeros(d)
        r = b.copy()
        nX = len(self.X)
        if nX:
            # HVP-free estimate from stored products: the fallback if the oracle fails
            Xa = np.array(self.X)
            self._cur = _psolve(Xa @ np.array(self.HX).T, Xa @ b, 1e-12) @ Xa
        refresh_all = phase_start
        if nX and not phase_start:
            h1 = np.asarray(hvp(self.X[-1]), dtype=np.float64)
            drift = np.linalg.norm(h1 - self.HX[-1]) / max(np.linalg.norm(h1), 1e-300)
            self.HX[-1] = h1
            if not drift <= DRIFT_TOL:
                refresh_all = True
        use_defl = self.P is not None and not refresh_all
        if use_defl:
            P, W = self.P, self.W
            k = P.shape[0]
            n = nX + k
            G = np.zeros((n, n))
            rhs = np.zeros(n)
            G[nX:, nX:] = np.eye(k)
            rhs[nX:] = P @ b
            if nX:
                Xa = np.array(self.X)
                HXa = np.array(self.HX)
                G[:nX, :nX] = Xa @ HXa.T
                rhs[:nX] = Xa @ b
                blk = HXa @ P.T
                G[:nX, nX:] = blk
                G[nX:, :nX] = blk.T
            y = _psolve(G, rhs, 1e-8)
            x = y[nX:] @ P
            if nX:
                x += y[:nX] @ Xa
            r = b - np.asarray(hvp(x), dtype=np.float64)
        elif nX:
            Xa = np.array(self.X)
            if refresh_all:
                HXa = np.asarray(hvp(np.ascontiguousarray(Xa.T)), dtype=np.float64).T
                self.HX = list(HXa.copy())
                fresh = True
            else:
                HXa = np.array(self.HX)
                fresh = nX == 1
            y = _psolve(Xa @ HXa.T, Xa @ b, 1e-12)
            x = y @ Xa
            r = b - y @ HXa if fresh else b - np.asarray(hvp(x), dtype=np.float64)

        # ---------------- (deflated) CG
        self._cur = x
        if use_defl:
            def project(v):
                return v - (W @ v) @ P
        Pn = np.empty((KEEP, d))
        Wn = np.empty((KEEP, d))
        rr = r @ r
        if use_defl:
            p = project(r)
            if p @ p < 1e-10 * rr:
                use_defl = False
                p = r.copy()
        else:
            p = r.copy()
        nb2 = b @ b
        rr_stop = 1e-30 * nb2       # relative residual 1e-15
        rr_plain = 1e-12 * nb2      # relative residual 1e-6: plain CG from here on
        it = 0
        t_it = 3.2e-3
        nrm = max(abs(b @ x), 1e-300)
        gains = []
        while True:
            now = _now()
            if now + t_it > t_hard:
                break
            if now + t_it > t_soft and len(gains) >= 16 and np.mean(gains[-16:]) / nrm < tau:
                break
            w = np.asarray(hvp(p), dtype=np.float64)
            pw = p @ w
            if not (pw > 0) or not np.isfinite(pw):
                if use_defl and np.isfinite(rr) and rr > rr_stop:
                    use_defl = False          # recycled space exhausted: plain CG
                    p = r.copy()
                    continue
                break
            if use_defl:
                rp = r @ p
                alpha = rp / pw
                gains.append(rp * alpha)
            else:
                alpha = rr / pw
                gains.append(alpha * rr)
            x += alpha * p
            r -= alpha * w
            sc = 1.0 / np.sqrt(pw)
            Pn[it % KEEP] = p * sc
            Wn[it % KEEP] = w * sc
            rr_new = r @ r
            if not np.isfinite(rr_new) or rr_new <= rr_stop:
                break
            if it >= 16 and it % 8 == 0 and max(gains[-16:]) < 1e-22 * nrm:
                break                         # converged to rounding level
            if use_defl and (rr_new < rr_plain or (it >= 16 and it % 8 == 0 and
                                                   max(gains[-8:]) < 1e-8 * max(gains))):
                use_defl = False              # plain CG can also repair the recycled part
                p = r.copy()
                rr = rr_new
                it += 1
                continue
            if use_defl:
                z = project(r)
                if z @ z < 1e-10 * rr_new:
                    use_defl = False
                    p = r.copy()
                    rr = rr_new
                    it += 1
                    continue
                p = z - ((z @ w) / pw) * p
            else:
                p = r + (rr_new / rr) * p
            rr = rr_new
            it += 1
            if it % 16 == 0:
                nrm = max(abs(b @ x), 1e-300)
            el = _now() - now
            t_it = 0.8 * t_it + 0.2 * el if it > 1 else el

        # ---------------- bookkeeping for the next steps
        nrm = max(abs(b @ x), 1e-300)
        if boost:
            self.boost_used += max(0.0, (_now() - t_start) - base)
        if len(gains) >= 16 and not phase_start:
            self.tail.append(float(np.mean(gains[-16:]) / nrm))
        m = min(it, KEEP)
        if m >= 2:
            order = np.arange(it - m, it) % KEEP
            self.P, self.W = _h_orthonormal(Pn[order], Wn[order])
        else:
            self.P = self.W = None
        if not np.isfinite(x).all():
            x = np.zeros(d)
            r = b.copy()
        self.X.append(x.copy())
        self.HX.append(b - r)
        if len(self.X) > HIST:
            self.X.pop(0)
            self.HX.pop(0)
        return x


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)
