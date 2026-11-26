"""Online SPD sequence solver.

Sequence of 50 SPD systems  H_t x_t = b_t  (d = 16384), matrix-free (HVP only),
one process per sequence, hard wall-clock budget for the whole sequence.

Strategy
--------
1.  Spectral preconditioner learnt from the sequence itself.
    A Lanczos run on the first system (random start, full reorthogonalisation)
    yields Ritz pairs.  The bottom of the spectrum is where the normalised
    squared H-energy error lives (the loss equals a 1/lambda-weighted average
    of the per-eigen-direction relative errors), so the low Ritz pairs are used
    as an exact deflation block of a projected-inverse preconditioner

        M^{-1} v = U diag(1/lambda) U^T v + (v - U U^T v)/theta .

    The Lanczos basis is reused for every later system: consecutive matrices
    drift only ~1% per step, so the learnt operator model stays useful.

2.  Warm start: the previous solution is the best available guess for the
    current system (right hand sides change, operators do not).

3.  Preconditioned CG on the deflated system.  The iteration count per step is
    chosen from the remaining wall-clock budget so that all 50 systems are
    answered inside the sequence deadline.
"""

import json
import os
import sys
import time

import numpy as np

_LOG = os.environ.get("SPD_SOLVER_LOG")


def _log(msg):
    if not _LOG:
        return
    try:
        with open(_LOG, "a") as handle:
            handle.write("%s\n" % msg)
    except Exception:
        pass


class _Solver:
    def __init__(self, dimension, seed, options):
        self.d = int(dimension)
        try:
            seed = int(seed)
        except Exception:
            seed = 0
        self.rng = np.random.default_rng(seed & 0x7FFFFFFF)
        opts = options if isinstance(options, dict) else {}
        self.k_defl = int(opts.get("rank", 20) or 20)
        self.k_lanczos = int(opts.get("lanczos", 160) or 160)
        self.warm = bool(opts.get("warm", True))
        self.max_iter = int(opts.get("maxiter_factor", 50) or 50) * 4
        self.U = None            # (d, k) deflation basis
        self.lam = None          # (k,) Ritz values
        self.theta = 1.0         # scalar used on the orthogonal complement
        self.gram = None         # precomputed k x k operator of the preconditioner
        self.x_prev = None
        self.step = 0
        self.deadline = None
        self.steps = 50
        self.cost = 0.004        # running estimate of seconds per iteration
        self.precond_ready = False

    # ------------------------------------------------------------------ setup
    def _lanczos(self, hvp, k_steps):
        d = self.d
        q = self.rng.standard_normal(d)
        q /= np.linalg.norm(q)
        basis = np.zeros((k_steps + 1, d))
        basis[0] = q
        alpha = np.zeros(k_steps)
        beta = np.zeros(k_steps)
        for j in range(k_steps):
            w = hvp(basis[j].copy())
            if j > 0:
                w -= beta[j - 1] * basis[j - 1]
            aj = float(basis[j] @ w)
            alpha[j] = aj
            w -= aj * basis[j]
            qj = basis[: j + 1]
            for _ in range(2):
                c = qj @ w
                w -= qj.T @ c
            bj = float(np.linalg.norm(w))
            beta[j] = bj
            if bj <= 0.0:
                break
            basis[j + 1] = w / bj
        tridiag = (np.diag(alpha) + np.diag(beta[: k_steps - 1], 1)
                   + np.diag(beta[: k_steps - 1], -1))
        return basis[:k_steps].T.copy(), tridiag

    def _build_preconditioner(self, hvp, L, mu):
        """Learn the deflation block from the first system."""
        k_steps = max(self.k_lanczos, self.k_defl + 5)
        k_steps = min(k_steps, self.d - 1)
        vecs, tridiag = self._lanczos(hvp, k_steps)
        values, coeff = np.linalg.eigh(tridiag)
        order = np.argsort(values)
        values = values[order]
        coeff = coeff[:, order]
        k = min(self.k_defl, values.size)
        sel = np.arange(k)
        lam = values[sel]
        basis = vecs @ coeff[:, sel]
        norms = np.linalg.norm(basis, axis=0)
        norms[norms <= 0] = 1.0
        basis = np.ascontiguousarray(basis / norms)
        # Rayleigh quotients with the current operator (more accurate than the
        # Ritz values, and they track the slow drift of the matrix).
        lam = np.einsum("ij,ij->j", basis, np.column_stack(
            [hvp(basis[:, j].copy()) for j in range(k)]))
        lam = np.where(np.isfinite(lam) & (lam > 0), lam, values[sel])
        top = float(max(lam.max(), mu if mu else 1.0))
        theta = float(np.sqrt(max(top, 1e-12) * max(L, top)))
        self.U = basis
        self.lam = lam
        self.theta = theta
        self._refresh_gram()
        self.precond_ready = True
        _log("precond: k=%d lam=%s theta=%.6g" % (
            k, " ".join("%.5g" % v for v in lam[:8]), theta))

    def _refresh_gram(self):
        """Precompute the k x k operator of the projected preconditioner."""
        k = self.lam.size
        gram = self.U.T @ self.U
        gram = 0.5 * (gram + gram.T)
        gram.flat[:: k + 1] += 1e-10 * max(1.0, float(gram.diagonal().max()))
        inv = np.linalg.inv(gram)
        # M^{-1} v = v/theta + U (G^{-1} diag(1/lam) - I/theta) U^T v
        inner = inv * (1.0 / self.lam)[None, :]
        self.gram = inner - np.eye(k) / self.theta

    def _apply_preconditioner(self, vec):
        coeff = self.U.T @ vec
        return vec / self.theta + self.U @ (self.gram @ coeff)

    # ----------------------------------------------------------------- solve
    def _pcg(self, hvp, b, x0, max_iter, stop_at):
        d = self.d
        x = np.zeros(d) if x0 is None else x0.copy()
        if x0 is None:
            r = b.copy()
        else:
            r = b - hvp(x)
        z = self._apply_preconditioner(r)
        p = z.copy()
        rz = float(r @ z)
        if rz <= 0.0 or not np.isfinite(rz):
            return x, 0
        it = 0
        for it in range(1, max_iter + 1):
            ap = hvp(p)
            pap = float(p @ ap)
            if pap <= 0.0 or not np.isfinite(pap):
                break
            alpha = rz / pap
            x = x + alpha * p
            r = r - alpha * ap
            z = self._apply_preconditioner(r)
            rz_new = float(r @ z)
            if rz_new <= 0.0 or not np.isfinite(rz_new):
                break
            beta = rz_new / rz
            p = z + beta * p
            rz = rz_new
            if (it & 3) == 0 and time.monotonic() > stop_at:
                break
        return x, it

    def solve(self, hvp, b, context):
        """Return an approximation of H_t^{-1} b_t for the current system."""
        try:
            return self._solve(hvp, b, context)
        except Exception:
            # Never let an exception escape: a finite fallback is always valid.
            fallback = self.x_prev
            if fallback is None or not np.all(np.isfinite(fallback)):
                fallback = np.zeros(self.d)
            else:
                fallback = np.asarray(fallback, dtype=np.float64)
            self.step += 1
            return fallback

    def _solve(self, hvp, b, context):
        start = time.monotonic()
        b = np.asarray(b, dtype=np.float64)
        mu = float(context.get("mu", 1.0) or 1.0)
        L = float(context.get("L", 1.0) or 1.0)
        self.steps = int(context.get("steps", 50) or 50)
        deadline = context.get("deadline")
        if deadline is None:
            deadline = context.get("sequence_deadline")
        if deadline is None:
            deadline = start + 25.0
        self.deadline = float(deadline)

        try:
            if not self.precond_ready and self.deadline - start > 3.0:
                self._build_preconditioner(hvp, L, mu)
        except Exception:
            self.precond_ready = False

        # ---- budget split over the remaining systems -------------------------
        remaining_steps = max(1, self.steps - self.step)
        now = time.monotonic()
        reserve = 0.45
        available = max(0.0, self.deadline - now - reserve)
        per_step = available / remaining_steps
        if per_step < 0.02:
            per_step = 0.02
        max_iter = int(per_step / max(self.cost, 1e-4))
        max_iter = max(1, min(max_iter, self.max_iter))
        stop_at = min(self.deadline - 0.15, now + per_step)

        x = None
        if self.precond_ready:
            x0 = self.x_prev if (self.warm and self.x_prev is not None) else None
            try:
                x, it = self._pcg(hvp, b, x0, max_iter, stop_at)
                self.cost = 0.5 * self.cost + 0.5 * ((time.monotonic() - now) / max(it, 1))
            except Exception:
                x = None
        if x is None:
            try:
                x, it = self._cg(hvp, b, self.x_prev if (self.warm and self.x_prev is not None) else None,
                                  max_iter, stop_at)
            except Exception:
                x = np.zeros(self.d)
        if x is None or not np.all(np.isfinite(x)):
            x = self.x_prev if (self.x_prev is not None and np.all(np.isfinite(self.x_prev))) \
                else np.zeros(self.d)
        self.x_prev = x
        self.step += 1
        _log("t=%d time=%.2f" % (self.step, time.monotonic() - start))
        return x

    def _cg(self, hvp, b, x0, max_iter, stop_at):
        d = self.d
        x = np.zeros(d) if x0 is None else x0.copy()
        r = b.copy() if x0 is None else b - hvp(x)
        p = r.copy()
        rr = float(r @ r)
        bn = float(np.linalg.norm(b))
        rel = 1.0
        it = 0
        for it in range(1, max_iter + 1):
            ap = hvp(p)
            pap = float(p @ ap)
            if pap <= 0.0 or not np.isfinite(pap):
                break
            alpha = rr / pap
            x = x + alpha * p
            r = r - alpha * ap
            rr_new = float(r @ r)
            beta = rr_new / rr
            p = r + beta * p
            rr = rr_new
            rel = np.sqrt(rr) / bn if bn > 0 else 0.0
            if (it & 3) == 0 and time.monotonic() > stop_at:
                break
        return x, it


def make_solver(dimension, seed, options):
    return _Solver(dimension, seed, options)
