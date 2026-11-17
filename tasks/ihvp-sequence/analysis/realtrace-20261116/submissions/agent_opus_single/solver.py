"""Online SPD sequence solver: warm-started CG with a small recycled deflation space.

Each step starts from the previous solution, refreshes H_t on the last few CG
search directions of the previous step (one block HVP together with the warm
start residual), applies a Galerkin correction on that space and runs CG whose
directions are kept H_t-orthogonal to it.  The sequence deadline is shared over
the remaining steps, with extra weight on steps where the phase changes.
"""
import sys
import time

import numpy as np

_K = 2
_PHASE_WEIGHT = 3.0
_RESERVE = 0.6          # seconds kept free before the sequence deadline
_RANK_TOL = 1e-3


class Solver:
    def __init__(self, dimension, seed, options):
        self.d = int(dimension)
        self.k = max(0, min(_K, self.d // 4))
        self.prev = None
        self.dirs = []
        self.t_start = time.monotonic()

    def _deadline(self, ctx):
        dl = ctx.get("deadline")
        if dl is None:
            dl = ctx.get("sequence_deadline", self.t_start + 27.0)
        dl = float(dl)
        hard = ctx.get("hard_deadline")
        if hard is not None:
            dl = min(dl, float(hard) - 1.0)
        return dl

    def _stop_time(self, ctx, deadline):
        t = int(ctx.get("step", 0))
        T = int(ctx.get("steps", 50))
        now = time.monotonic()
        left = max(0.0, deadline - _RESERVE - now)
        w_now = _PHASE_WEIGHT if ctx.get("phase_changed") else 1.0
        total = w_now + max(0, T - t - 1)
        return min(now + left * w_now / total, deadline - 0.3)

    def solve(self, hvp, b, context):
        d = self.d
        x = np.zeros(d) if self.prev is None else self.prev.copy()
        it = 0
        try:
            deadline = self._deadline(context)
            stop = self._stop_time(context, deadline)
            b = np.asarray(b, dtype=np.float64).reshape(d)
            bn = float(np.linalg.norm(b))
            if not np.isfinite(bn) or bn == 0.0:
                raise ValueError("bad right-hand side")
            W = AW = Gi = None
            if self.k > 0 and self.dirs and self.prev is not None:
                W, R = np.linalg.qr(np.stack(self.dirs, axis=1))
                dg = np.abs(np.diag(R))
                W = W[:, dg > _RANK_TOL * dg.max()]
                if W.shape[1] == 0:
                    W = None
            if self.prev is None:
                r = b.copy()
            elif W is not None:
                blk = np.asarray(hvp(np.column_stack([x, W])), dtype=np.float64)
                r = b - blk[:, 0]
                AW = blk[:, 1:]
                G = W.T @ AW
                G = 0.5 * (G + G.T)
                Gi = np.linalg.inv(G)
                if np.all(np.isfinite(Gi)) and np.all(np.linalg.eigvalsh(G) > 0):
                    y = Gi @ (W.T @ r)
                    x_try = x + W @ y
                    if np.all(np.isfinite(x_try)):
                        x = x_try
                        r = r - AW @ y
                    else:
                        W = None
                else:
                    W = None
            else:
                r = b - np.asarray(hvp(x), dtype=np.float64)
            p = r.copy()
            if W is not None:
                p -= W @ (Gi @ (AW.T @ r))
            rr = float(r @ r)
            newdirs = []
            while time.monotonic() < stop:
                pn = float(p @ p)
                if not (pn > 1e-28 * rr):
                    if W is None:
                        break
                    W = None
                    p = r.copy()
                    pn = rr
                Ap = np.asarray(hvp(p), dtype=np.float64)
                pAp = float(p @ Ap)
                if not (pAp > 0.0) or not np.isfinite(pAp):
                    break
                a = float(r @ p) / pAp
                x_new = x + a * p
                if not np.all(np.isfinite(x_new)):
                    break
                x = x_new
                r -= a * Ap
                rn = float(r @ r)
                newdirs.append(p / np.sqrt(pn))
                if len(newdirs) > self.k:
                    newdirs.pop(0)
                it += 1
                if not (rn > 1e-24 * bn * bn):
                    break
                p *= rn / rr
                p += r
                rr = rn
                if W is not None:
                    p -= W @ (Gi @ (AW.T @ r))
            self.dirs = newdirs
        except Exception as exc:  # never let an exception escape: return best iterate
            print(f"solver step {context.get('step')}: {type(exc).__name__}: {exc}", file=sys.stderr)
        if not np.all(np.isfinite(x)):
            x = np.zeros(d)
        self.prev = x.copy()
        return x, {"iterations": it}


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)
