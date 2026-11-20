"""An online, spectrally recycled conjugate gradient solver."""

import math
import time

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.linalg.blas import dgemv


class Solver:
    def __init__(self, dimension):
        self.d = dimension
        self.solutions = []
        self.ritz = None
        self.tail = None
        self.last_rhs = None
        self.correlation = 0.90
        self.coarse_fraction = 0.60
        self.age = 0
        self.switches = 0
        self.call_time = 0.0034
        self.oracle_margin = 0.0034
        self.observed_call = False
        self.capacity = max(8, min(960, (120 << 20) // (8 * dimension)))
        self.basis = None
        self.diagonal = None
        self.subdiagonal = None

    def _coarse_space(self, query, columns, b):
        """Refresh the history and orthonormalize it in the current H metric."""
        if not columns:
            return None, None, np.zeros_like(b), b.copy()
        h = np.ascontiguousarray(np.column_stack(columns))
        if h.shape[1] == 1:
            u = h[:, 0]
            v = query(u)
            if v is not None:
                curv = u @ v
                if math.isfinite(curv) and curv > 0.0:
                    s = 1.0 / math.sqrt(curv)
                    u, w = u * s, v * s
                    c = u @ b
                    return u[:, None], w[:, None], c * u, b - c * w
        else:
            y = query(h)
            if y is not None:
                g = h.T @ y
                if np.isfinite(g).all():
                    val, vec = np.linalg.eigh((g + g.T) * 0.5)
                    ok = val > max(0.0, 2.e-11 * val[-1])
                    if np.any(ok):
                        rot = vec[:, ok] / np.sqrt(val[ok])
                        u = np.asfortranarray(h @ rot)
                        w = np.asfortranarray(y @ rot)
                        c = u.T @ b
                        return u, w, u @ c, b - w @ c
        return None, None, np.zeros_like(b), b.copy()

    def _recycle(self, count, rank, mu, u, w):
        """Take the low Ritz vectors of the projected Lanczos tridiagonal."""
        m = min(count, self.capacity)
        begin = (count - m) % self.capacity
        end = min(self.capacity, begin + m)
        coordinates = (np.arange(m) + begin) % self.capacity
        val, vec = eigh_tridiagonal(
            self.diagonal[coordinates],
            self.subdiagonal[coordinates[1:]],
            select="i", select_range=(0, min(m - 1, rank - 1)),
            lapack_driver="stebz",
        )
        ok = (val > 0.0) & (val >= 0.30 * mu)
        if not np.any(ok):
            return None
        c = vec[:, ok] / np.sqrt(val[ok])
        c[1::2] *= -1.0  # CG residuals have alternating Lanczos signs.
        z = self.basis[begin:end].T @ c[:end-begin]
        if begin + m > self.capacity:
            z += self.basis[:begin+m-self.capacity].T @ c[end-begin:]
        if u is not None:
            z -= u @ (w.T @ z)
        if not np.isfinite(z).all():
            return None
        return np.asfortranarray(z)

    def solve(self, hvp, b, context):
        clock = time.monotonic
        start = clock()
        step = context["step"]
        left = context["steps"] - step
        soft = min(context["deadline"], context.get("sequence_deadline", math.inf))
        deadline = min(soft - 0.23, context.get("hard_deadline", math.inf) - 0.55)
        available = deadline - start
        if not available > 0.0018 * left:
            return np.zeros_like(b)

        mu, upper = context["mu"], context["L"]
        b2 = b @ b
        normalization = 1.0
        if not (1.e-100 < b2 < 1.e100):
            normalization = float(np.max(np.abs(b)))
            if normalization == 0.0 or not math.isfinite(normalization):
                return np.zeros_like(b)
            b = b / normalization
            b2 = b @ b
        norm_limit = 2.01 * math.sqrt(b2) / mu
        if self.d <= 64:
            try:
                x = np.linalg.solve(hvp(np.eye(self.d)), b) * normalization
                if np.isfinite(x).all():
                    return x
            except np.linalg.LinAlgError:
                pass
            except RuntimeError as err:
                if err.__class__.__name__ != "BudgetExceeded":
                    raise
                return np.zeros_like(b)

        cost = self.call_time
        uncertainty = self.oracle_margin
        if context["phase_changed"]:
            self.age = 0
            self.switches += int(step > 0)
            self.solutions.clear()
            self.ritz = None
            self.tail = None
            self.last_rhs = None
        else:
            self.age += 1
        if self.last_rhs is not None:
            alignment = abs(float(b @ self.last_rhs)) / math.sqrt(b2)
            if self.age == 1 and self.switches == 0:
                self.correlation = min(1.0, alignment)
            else:
                self.correlation = 0.78 * self.correlation + 0.22 * min(1.0, alignment)
        else:
            alignment = self.correlation
        relative_depth = math.sqrt(upper) / math.sqrt(mu) * (cost + 0.00008) / max(0.08, available / left)
        alpha = 1.38 * math.log(max(1.0, 0.56 * relative_depth)) + 0.34 * math.log1p(relative_depth / 12.0)
        alpha = min(5.7, max(0.13, alpha))
        if self.correlation < 0.80:
            alpha *= 0.40 + 0.75 * self.correlation
        shape = 0.19
        weight = alpha * shape**self.age
        # Leave some time for later phase starts, whose locations are unknown.
        hazard = (1.18 + self.switches) / (34.0 + step)
        reserve = (weight + (left - 1) * hazard * alpha) / (1.0 - shape)
        allotment = available * (1.0 + weight) / (left + reserve)
        end = min(deadline, start + allotment)

        rank = 0
        keep = 1
        if relative_depth > 5.4:
            rank, keep = (4, 3) if relative_depth > 12.0 else (2, 2)
            if alignment < 0.52 and self.coarse_fraction < 0.24:
                rank, keep = 1, 1
        if allotment < 32.0 * cost:
            rank = min(rank, 2)

        budget_exhausted = False

        def query(v):
            nonlocal cost, uncertainty, budget_exhausted
            if budget_exhausted:
                return None
            k = 1 if v.ndim == 1 else v.shape[1]
            margin = max(0.0014, 1.15 * cost, uncertainty)
            expected = margin + cost * (k - 1)
            before = clock()
            if (before + expected + 0.0012 >= end or
                    before + 1.2 * expected + 0.0018 * (left - 1) >= deadline):
                return None
            try:
                y = hvp(v)
            except RuntimeError as err:
                if err.__class__.__name__ != "BudgetExceeded":
                    raise
                budget_exhausted = True
                return None
            if k == 1:
                elapsed = clock() - before
                if not self.observed_call:
                    cost = max(1.e-5, min(0.25, elapsed))
                    self.observed_call = True
                else:
                    cost += 0.07 * (min(elapsed, 3.0 * cost) - cost)
                uncertainty = max(0.985 * uncertainty, min(0.22, 4.0 * cost, elapsed))
            return y

        cols = self.solutions[:keep]
        retain_tail = rank >= 4 and alignment >= 0.74
        if self.tail is not None and retain_tail:
            cols.append(self.tail)
        if rank and self.ritz is not None:
            cols = cols + [self.ritz[:, j] for j in range(min(rank, self.ritz.shape[1]))]
        while len(cols) > 1 and 2.5 * cost * (len(cols) + 1) > end - clock():
            cols.pop()
        try:
            u, w, x0, r = self._coarse_space(query, cols, b)
        except np.linalg.LinAlgError:
            u = w = None
            x0, r = np.zeros_like(b), b.copy()
        e0 = float(b @ x0)
        x0norm = float(np.linalg.norm(x0))
        if (not math.isfinite(e0) or not np.isfinite(r).all() or
                not math.isfinite(x0norm) or x0norm > norm_limit):
            u = w = None
            x0, r, e0 = np.zeros_like(b), b.copy(), 0.0

        record = rank > 0 and left > 1
        if record and self.basis is None:
            self.basis = np.empty((self.capacity, self.d))
            self.diagonal = np.empty(self.capacity)
            self.subdiagonal = np.empty(self.capacity)
        # B = H - (H U)(H U)^T. Iterates are formed in the quotient and
        # projected back at the end, so each fine iteration uses a single HVP.
        rr = float(r @ r)
        p = r.copy()
        dx = np.zeros_like(b)
        count = 0
        beta = 0.0
        inverse_step = 0.0
        gain = max(e0, 0.0)
        while (count < 8192 and
               rr > max(4.e-27 * b2, 2.e-18 * mu * gain) and math.isfinite(rr)):
            ap = query(p)
            if ap is None:
                break
            if w is not None:
                if np.may_share_memory(p, ap):
                    ap = ap.copy()
                ap = dgemv(-1.0, w, dgemv(1.0, w, p, trans=1),
                           beta=1.0, y=ap, overwrite_y=1)
            curv = float(p @ ap)
            if not math.isfinite(curv) or curv <= 0.0:
                break
            inv = curv / rr
            a = rr / curv
            if not (a > 0.0 and math.isfinite(a) and math.isfinite(inv)):
                break
            if record:
                q = count % self.capacity
                np.multiply(r, 1.0 / math.sqrt(rr), out=self.basis[q])
                self.diagonal[q] = inv + beta * inverse_step
                self.subdiagonal[q] = math.sqrt(beta) * inverse_step
            dx += a * p
            r -= a * ap
            gain += a * rr
            rr2 = float(r @ r)
            count += 1
            if not math.isfinite(rr2):
                break
            beta = rr2 / rr
            if not math.isfinite(beta):
                break
            p *= beta
            p += r
            rr = rr2
            inverse_step = inv

        if w is not None:
            dx -= dgemv(1.0, u, dgemv(1.0, w, dx, trans=1))
        result = x0 + dx
        objective = float((b + r) @ result)
        xnorm = float(np.linalg.norm(result))
        if (not math.isfinite(objective) or objective < 0.995 * e0 or
                not math.isfinite(xnorm) or xnorm > norm_limit):
            result = x0
            objective = e0

        if record and count >= 4 and end + 0.02 < deadline and clock() + 0.015 < deadline:
            try:
                self.ritz = self._recycle(count, rank, mu, u, w)
            except (np.linalg.LinAlgError, ValueError):
                self.ritz = None
        else:
            self.ritz = None
        self.tail = None
        if record and retain_tail and count and rr > 1.e-19 * b2:
            # At a restart this direction is outside the previously used
            # conjugate directions, and is often still useful after a small drift.
            tail = p - u @ (w.T @ p) if w is not None else p
            scale2 = max(mu * float(tail @ tail), rr * inverse_step)
            if scale2 > 0.0 and math.isfinite(scale2):
                tail = tail / math.sqrt(scale2)
                if np.isfinite(tail).all():
                    self.tail = tail
        if objective > 0.0 and math.isfinite(objective):
            self.solutions.insert(0, result / math.sqrt(objective))
            if self.age > 0:
                self.coarse_fraction = 0.72 * self.coarse_fraction + 0.28 * min(1.0, e0 / objective)
        del self.solutions[keep:]
        self.last_rhs = b / math.sqrt(b2)
        self.call_time = cost
        self.oracle_margin = uncertainty
        out = result * normalization if normalization != 1.0 else result
        if not np.isfinite(out).all():
            return np.zeros_like(b)
        return out


def make_solver(dimension, seed, options):
    return Solver(dimension)
