"""Online solver for a sequence of slowly varying SPD linear systems.

For each system we
  * keep a short history of past solutions and refresh their curvature with a
    single batched HVP query,
  * take an exact Galerkin (reduced-basis) initial guess in that span,
  * run (preconditioned) CG -- with a tiny recycled secant preconditioner for
    extremely wide spectra,
  * and finally take the Galerkin-optimal combination of the CG iterate, the
    initial guess and the stored solutions (all of whose H-images are known),
    which can only reduce the H-energy error.

Computation is budgeted against the sequence wall-clock deadline, with extra
time reserved for the beginning of each phase (where the warm start is weak).
"""
from __future__ import annotations

import time
import numpy as np


class Solver:
    def __init__(self, dimension, seed, options):
        self.d = int(dimension)
        self.rng = np.random.default_rng(seed)

        # ---- tunables -----------------------------------------------------
        self.safety = float(options.get('safety', 0.93))
        self.reserve = float(options.get('reserve', 1.2))
        self.max_iters = int(options.get('max_iters', 500))
        self.K = int(options.get('K', 2))                 # stored past solutions
        self.M = int(options.get('M', 4))                 # secant directions
        self.secant_threshold = float(options.get('secant_threshold', 2e6))
        self.boost_wide = float(options.get('boost_wide', 3.0))
        self.tau_wide = float(options.get('tau_wide', 3.0))
        self.boost_narrow = float(options.get('boost_narrow', 2.2))
        self.debug = bool(options.get('debug', False))
        self.reg = float(options.get('reg', 1e-9))

        # ---- state --------------------------------------------------------
        self.deadline = None
        self.X = np.zeros((max(1, self.K), self.d))
        self.nx = 0
        self.prev = np.zeros(self.d)
        m = max(1, self.M)
        self.mS = np.zeros((m, self.d))
        self.mY = np.zeros((m, self.d))
        self.mr = np.zeros(m)
        self.npairs = 0
        self.use_secant = False
        self.phase_age = 0
        self.n_hvp = 0

    # ------------------------------------------------------------------
    def log(self, *a):
        if self.debug:
            import sys
            print('[solver]', *a, file=sys.stderr, flush=True)

    def _apply_B(self, q, gamma):
        m = self.npairs
        if m == 0:
            return gamma * q
        al = np.empty(m)
        for j in range(m - 1, -1, -1):
            a = self.mr[j] * float(self.mS[j] @ q)
            al[j] = a
            q -= a * self.mY[j]
        r = gamma * q
        for j in range(m):
            be = self.mr[j] * float(self.mY[j] @ r)
            r += (al[j] - be) * self.mS[j]
        return r

    @staticmethod
    def _small_solve(G, rhs, reg):
        ev, Z = np.linalg.eigh((G + G.T) * 0.5)
        mx = max(float(ev[-1]), 1e-30)
        keep = ev > mx * reg
        if not np.any(keep):
            return None
        Zk = Z[:, keep]
        return Zk @ ((Zk.T @ rhs) / ev[keep])

    # ------------------------------------------------------------------
    def solve(self, hvp, b, context):
        step = int(context.get('step', 0))
        steps = int(context.get('steps', 50))
        dl = context.get('deadline') or context.get('sequence_deadline')
        self.deadline = float(dl) if dl else None
        t0 = time.monotonic()
        L = float(context.get('L', 1e5))
        mu = float(context.get('mu', 1.0))
        kappa = L / max(mu, 1e-12)
        wide = kappa > self.secant_threshold
        self.use_secant = (self.M > 0) and wide
        phase_changed = bool(context.get('phase_changed', False))
        if phase_changed:
            self.phase_age = 0
        else:
            self.phase_age += 1
        if phase_changed and step > 0:
            self.npairs = 0
            self.nx = max(0, self.nx - 1)

        # ---- clock budget -------------------------------------------------
        if self.deadline:
            remaining = max(0.0, self.deadline - t0 - self.reserve)
            share = remaining * self.safety / max(1, steps - step)
            if step == 0:
                share = min(remaining * 0.12, share * 2.2)
            if wide:
                share *= 1.0 + (self.boost_wide - 1.0) * np.exp(-self.phase_age / self.tau_wide)
            elif phase_changed and step > 0:
                share *= self.boost_narrow
            step_end = min(t0 + share, self.deadline - 0.55 * self.reserve)
        else:
            step_end = t0 + 0.5

        try:
            x = self._impl(hvp, b, step, step_end, L)
        except Exception as exc:
            self.log(f'step {step} fallback: {type(exc).__name__}: {exc}')
            x = self.prev.copy()
        if not np.all(np.isfinite(x)):
            x = np.zeros(self.d)
        self.prev = x
        return x

    # ------------------------------------------------------------------
    def _impl(self, hvp, b, step, step_end, L):
        d = self.d
        bnorm2 = max(float(b @ b), 1e-30)
        n = self.nx
        V = []          # candidate vectors (H-image is b - HV[i])
        HV = []

        # ---- exact curvature for past solutions ---------------------------
        x0 = np.zeros(d)
        r0 = b.copy()
        if n > 0:
            HX = hvp(np.ascontiguousarray(self.X[:n].T))     # (d,n)
            self.n_hvp += n
            G = self.X[:n] @ HX                              # (n,n)
            rhs = self.X[:n] @ b
            try:
                a = self._small_solve(G, rhs, self.reg)
                if a is not None:
                    x0 = a @ self.X[:n]
                    r0 = b - HX @ a
                    V.append(x0.copy())
                    HV.append(r0.copy())
                    for j in range(n):
                        V.append(self.X[j].copy())
                        HV.append(b - HX[:, j])
            except Exception:
                pass
        if not V:
            V.append(x0.copy())
            HV.append(r0.copy())
        if float(r0 @ r0) > 9.0 * bnorm2:      # reject pathological guesses
            x0 = self.prev.copy()
            if np.any(x0):
                r0 = b - hvp(x0).ravel()
                self.n_hvp += 1
            else:
                r0 = b.copy()
            V[0] = x0.copy()
            HV[0] = r0.copy()

        # ---- refresh the tiny secant basis --------------------------------
        if step > 0 and self.use_secant and self.npairs:
            m = self.npairs
            Yh = hvp(np.ascontiguousarray(self.mS[:m].T))
            self.n_hvp += m
            self.mY[:m] = Yh.T
            for j in range(m):
                self.mr[j] = 1.0 / max(float(self.mS[j] @ self.mY[j]), 1e-300)
        if self.use_secant and self.npairs:
            i = self.npairs - 1
            yy = float(self.mY[i] @ self.mY[i])
            gamma = (float(self.mS[i] @ self.mY[i]) / yy) if yy > 0 else 1.0 / L
            gamma = min(max(gamma, 1e-14), 1e8)
        else:
            gamma = 1.0
        use_b = self.use_secant and self.npairs > 0

        # ---- (preconditioned) CG ------------------------------------------
        x = x0.copy()
        r = r0.copy()
        rr0 = float(r @ r)
        z = self._apply_B(r.copy(), gamma) if use_b else r.copy()
        p = z.copy()
        rz = float(r @ z)
        if not np.isfinite(rz) or rz <= 0:
            z = r.copy(); p = z.copy(); rz = rr0; use_b = False
        if self.use_secant:
            cs = np.empty((self.M, d)); cy = np.empty((self.M, d))
            cc = np.full(self.M, np.inf); nc = 0
        rrn = rr0
        it = 0
        for it in range(self.max_iters):
            now = time.monotonic()
            if now >= step_end - 0.002:
                break
            if self.deadline and now > self.deadline - 0.45 * self.reserve:
                break
            try:
                Ap = hvp(p).ravel()
            except Exception:
                break                       # deadline reached: keep best-so-far
            self.n_hvp += 1
            pAp = float(p @ Ap)
            if not np.isfinite(pAp) or pAp <= 0:
                break
            a = rz / pAp
            x += a * p
            r -= a * Ap
            if self.use_secant:
                pn = np.linalg.norm(p)
                if pn > 0 and np.isfinite(pn):
                    q = p * (1.0 / pn)
                    hq = Ap * (1.0 / pn)
                    cv = float(q @ hq)
                    if nc < self.M:
                        j = nc; nc += 1
                    else:
                        j = int(np.argmax(cc))
                        if cv >= cc[j]:
                            j = -1
                    if j >= 0:
                        cs[j] = q; cy[j] = hq; cc[j] = cv
            rrn = float(r @ r)
            it += 1
            if not np.isfinite(rrn) or rrn > 1e6 * rr0 or rrn < 1e-28 * bnorm2:
                break
            z = self._apply_B(r.copy(), gamma) if use_b else r.copy()
            rzn = float(r @ z)
            if not np.isfinite(rzn) or rzn <= 0:
                break
            p = z + (rzn / rz) * p
            rz = rzn
        if self.use_secant and nc:
            order = np.argsort(cc[:nc]); k = min(self.M, nc)
            self.mS[:k] = cs[order[:k]]
            self.mY[:k] = cy[order[:k]]
            self.mr[:k] = [1.0 / max(float(self.mS[j] @ self.mY[j]), 1e-300) for j in range(k)]
            self.npairs = k

        # ---- Galerkin-optimal post-projection -----------------------------
        V.append(x.copy())
        HV.append(r.copy())
        k = len(V)
        Gm = np.empty((k, k)); rh = np.empty(k)
        hb = [b - hv for hv in HV]
        for i in range(k):
            for j in range(k):
                Gm[i, j] = float(V[j] @ hb[i])
            rh[i] = float(V[i] @ b)
        try:
            co = self._small_solve(Gm, rh, self.reg)
            if co is not None:
                xn = np.zeros(d)
                for i in range(k):
                    xn += co[i] * V[i]
                if (np.all(np.isfinite(xn)) and
                        np.linalg.norm(xn) < 20.0 * max(np.linalg.norm(x), 1e-30)):
                    x = xn
        except Exception:
            pass

        # ---- store this solution ------------------------------------------
        if self.K > 0:
            if self.nx < self.K:
                self.X[self.nx] = x
                self.nx += 1
            else:
                self.X[:-1] = self.X[1:]
                self.X[-1] = x
        if self.debug and step % 10 == 0:
            self.log(f'step {step} it={it} r0={rr0**.5:.3g} rf={rrn**.5:.3g} '
                     f'K={self.nx} left={(self.deadline-time.monotonic()):.2f}')
        return x


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options or {})
