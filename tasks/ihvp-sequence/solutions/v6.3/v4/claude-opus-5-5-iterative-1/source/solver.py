"""Online solver for sequences of SPD systems H_t x = b_t accessed through HVPs.

Per regular step: the previous solution, the previous step's correction
(x_end - x_start) and its last CG search direction span an augmentation space W.
H_t W is recomputed (3 HVPs), the start is the Galerkin solution on W, and CG
then runs deflated against W (search directions kept H-orthogonal to W), i.e. a
restarted CG "with memory" in the spirit of LGMRES.  Cold steps (first step /
phase change) run CG while the marginal energy-error decrease per HVP
stays above a fraction of its value at the fair-share iteration count; regular
steps use an equal share of the remaining time.
"""
import time
import numpy as np


class Solver:
    def __init__(self, d, seed, options):
        self.d = int(d)
        self.nz = 1            # previous corrections kept in the augmentation space
        self.cfrac = 0.07      # cold-step stopping threshold (relative marginal decrease)
        self.capfrac = 0.35    # a cold step may use at most this share of the remaining time
        self.kcold = 6.0       # ... and at most this multiple of the current fair share
        self.margin = 0.2      # seconds kept free before the oracle deadline
        self.x_prev = None
        self.Zc = []
        self.p_last = None
        self.t_it = 4e-3

    def augmentation(self, hvp, b):
        """Orthonormal W (rows), H W and diag(W H W^T) after Rayleigh-Ritz; None if degenerate."""
        rows = [self.x_prev] + self.Zc[:self.nz]
        if self.p_last is not None:
            rows.append(self.p_last)
        Wm = np.array(rows)
        nrm = np.sqrt(np.einsum("ij,ij->i", Wm, Wm))
        if not np.all(np.isfinite(nrm)):
            return None
        Wm = Wm[nrm > 0] / nrm[nrm > 0, None]
        if Wm.shape[0] == 0:
            return None
        G = Wm @ Wm.T
        gw, gq = np.linalg.eigh(0.5 * (G + G.T))
        if not (gw.size and gw.max() > 0 and np.isfinite(gw).all()):
            return None
        keep = gw > 1e-10 * gw.max()
        Wm = (gq[:, keep] / np.sqrt(gw[keep])).T @ Wm
        HWm = np.array([np.asarray(hvp(w), dtype=np.float64).reshape(-1) for w in Wm])
        Kw = Wm @ HWm.T
        tw, yw = np.linalg.eigh(0.5 * (Kw + Kw.T))
        if not (tw.size and tw.max() > 0 and np.isfinite(tw).all()):
            return None
        ok = tw > 1e-12 * tw.max()
        return yw[:, ok].T @ Wm, yw[:, ok].T @ HWm, tw[ok]

    def solve(self, hvp, b, ctx):
        now = time.monotonic()
        d = self.d
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        best = np.zeros(d)
        x = None
        try:
            step, T = int(ctx["step"]), int(ctx["steps"])
            end = float(ctx["deadline"]) - self.margin
            n = max(T - step, 1)
            # cold steps get the long CG budget; the augmentation is kept across phase
            # changes (its Galerkin start is exact, so it cannot hurt the start)
            cold = step == 0 or bool(ctx.get("phase_changed")) or self.x_prev is None
            remaining = max(end - now, 0.0)
            fair = remaining / n
            if cold:
                lim = min(self.capfrac * remaining, remaining - (n - 1) * 0.25 * fair, self.kcold * fair)
                t_cap = now + max(lim, fair)
            else:
                t_cap = now + fair
            t_cap = min(t_cap, end)
            nb = float(np.linalg.norm(b))
            if not (nb > 0 and np.isfinite(nb)):
                return best
            x = np.zeros(d)
            r = b.copy()
            aug = None
            if self.x_prev is not None and time.monotonic() + 4 * self.t_it < t_cap:
                aug = self.augmentation(hvp, b)
            if aug is not None:
                W, HW, thw = aug
                c = (W @ b) / thw
                x = c @ W
                r = b - c @ HW
                if not (np.all(np.isfinite(x)) and np.all(np.isfinite(r))):
                    aug = None
                    x = np.zeros(d)
                    r = b.copy()
                else:
                    best = x.copy()
            x_start = x.copy()
            phi = -0.5 * float(x @ b) - 0.5 * float(x @ r)   # 0.5 x'Hx - b'x
            p = r.copy()
            if aug is not None:
                p -= ((HW @ r) / thw) @ W
            rr = float(r @ r)
            t0 = time.monotonic()
            its = 0
            m_fair = max(int(fair / max(self.t_it, 1e-4)), 10)
            dec = []
            g_ref = None
            tol2 = (1e-12 * nb) ** 2        # stop well above the rounding floor
            near2 = (1e-8 * nb) ** 2        # near convergence: guard against stagnation blow-up
            rr_min = rr
            x_min = None
            while rr > tol2:
                if time.monotonic() + 1.5 * self.t_it > t_cap:
                    break
                Ap = np.asarray(hvp(p), dtype=np.float64).reshape(-1)
                pAp = float(p @ Ap)
                if not (pAp > 0 and np.isfinite(pAp)):
                    break
                a = rr / pAp
                x += a * p
                r -= a * Ap
                phi -= 0.5 * a * rr
                if cold:
                    dec.append(a * rr)
                rr_new = float(r @ r)
                if not np.isfinite(rr_new):
                    break
                if rr_new < rr_min:
                    rr_min = rr_new
                    if rr_new < near2:
                        x_min = x.copy()
                elif x_min is not None and rr_new > 100.0 * rr_min:
                    x = x_min
                    break
                p *= rr_new / rr
                p += r
                if aug is not None:
                    p -= ((HW @ r) / thw) @ W
                rr = rr_new
                its += 1
                if its % 10 == 0:
                    if np.all(np.isfinite(x)):
                        best = x.copy()
                    if cold and its >= m_fair:
                        g = float(np.mean(dec[-20:])) / max(-2.0 * phi, 1e-300)
                        if g_ref is None:
                            g_ref = g
                        elif g < self.cfrac * g_ref:
                            break
            if its:
                self.t_it = 0.5 * self.t_it + 0.5 * (time.monotonic() - t0) / its
            if np.all(np.isfinite(x)):
                best = x
                self.Zc.insert(0, x - x_start)
                self.Zc = self.Zc[:self.nz]
                self.x_prev = x.copy()
                self.p_last = p.copy() if (its > 0 and np.all(np.isfinite(p))) else None
        except Exception:
            # e.g. an oracle refusal at the deadline: keep the latest finite iterate
            if x is not None and np.all(np.isfinite(x)):
                best = x.copy()
        if not np.all(np.isfinite(best)):
            best = np.zeros(d)
        return best


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)
