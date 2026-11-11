"""Online SPD solves using a queried transform preconditioner."""

import time

import numpy as np
from scipy.fft import dct, idct
from scipy.linalg import eigh


def _forward(v):
    return dct(v, type=2, norm="ortho", axis=0, workers=1)


def _inverse(v):
    return idct(v, type=2, norm="ortho", axis=0, workers=1)


def _coordinate_signs(columns):
    n, k = columns.shape
    p = k - 1
    # A cosine-diagonal operator commutes with the Neumann tridiagonal.
    # Fit its signed edges on the leading block, then along the whole chain.
    edges = 2.0 * (
        (np.arange(1 << p, dtype=np.uint32)[:, None] >> np.arange(p)) & 1
    ) - 1.0
    pieces = np.zeros((p, p, p))
    for j in range(p):
        pieces[j, j, :] += columns[j + 1, :p]
        pieces[j, :, j] -= columns[:p, j + 1]
        if j + 1 < p:
            pieces[j, j + 1, :] += columns[j, :p]
            pieces[j, :, j + 1] -= columns[:p, j]
    boundary = np.zeros((p, p))
    boundary[0, :] += columns[0, :p]
    boundary[:, 0] -= columns[:p, 0]
    residuals = edges @ pieces.reshape(p, -1) + boundary.reshape(1, -1)
    costs = np.einsum("ij,ij->i", residuals, residuals)
    prefix = edges[np.argmin(costs)]

    target = columns[:, 1 : p + 1] * prefix[None, :]
    target[:, 1:] += columns[:, : p - 1] * prefix[None, : p - 1]
    target[:, 0] += columns[:, 0]
    target[0, :] -= columns[0, :p]
    target[-1, :] -= columns[-1, :p]
    left, right = columns[:-2, :p], columns[2:, :p]
    cross = np.einsum("ij,ij->i", left, right)
    dl = np.einsum("ij,ij->i", left, target[1:-1])
    dr = np.einsum("ij,ij->i", right, target[1:-1])
    history = np.zeros((n - 1, 2), dtype=np.uint8)
    old0 = 0.0 if prefix[0] < 0 else np.inf
    old1 = 0.0 if prefix[0] > 0 else np.inf
    # Two-state dynamic programming enforces agreement of neighboring edges.
    for j in range(1, n - 1):
        c, a, b = cross[j - 1], dl[j - 1], dr[j - 1]
        c00 = old0 + c + a + b
        c10 = old1 - c - a + b
        c01 = old0 - c + a - b
        c11 = old1 + c - a - b
        history[j, 0] = c10 < c00
        history[j, 1] = c11 < c01
        old0 = min(c00, c10)
        old1 = min(c01, c11)
        if j < p:
            if prefix[j] > 0:
                old0 = np.inf
            else:
                old1 = np.inf
    a = columns[-2, :p]
    q = target[-1]
    old0 += 0.5 * np.dot(a + q, a + q)
    old1 += 0.5 * np.dot(a - q, a - q)
    state = int(old1 < old0)
    recovered = np.empty(n - 1)
    for j in range(n - 2, -1, -1):
        recovered[j] = 2 * state - 1
        state = int(history[j, state])
    signs = np.ones(n)
    signs[1:] = np.cumprod(recovered)
    return signs


class Solver:
    def __init__(self, dimension, seed, options):
        self.d = dimension
        self.rng = np.random.default_rng(seed)
        self.options = options or {}
        self.signs = None
        self.diagonal = None
        self.basis = None
        self.previous = None
        self.mode = "new"

    def _to_basis(self, v):
        if v.ndim == 1:
            return _forward(self.signs * v)
        return _forward(self.signs[:, None] * v)

    def _from_basis(self, v):
        if v.ndim == 1:
            return self.signs * _inverse(v)
        return self.signs[:, None] * _inverse(v)

    def _initialize(self, hvp, context):
        n = self.d
        k = min(12, n)
        probes = np.zeros((n, k))
        probes[np.arange(k), np.arange(k)] = 1.0
        columns = hvp(probes)
        self.signs = _coordinate_signs(columns)
        del probes, columns
        m = 64
        omega = self.rng.integers(0, 2, size=(n, m), dtype=np.int8).astype(float)
        omega *= 2.0
        omega -= 1.0
        y = self._to_basis(hvp(self._from_basis(omega)))
        dots = np.einsum("ij,ij->i", omega, y)
        diagonal = dots / m
        rank = min(12, m // 3)
        relative_change = np.inf
        # Alternate the diagonal fit and the row space of the low-rank remainder.
        for _ in range(32):
            residual = y - diagonal[:, None] * omega
            gram = residual.T @ residual
            _, vectors = eigh(
                gram, subset_by_index=(m - rank, m - 1), check_finite=False
            )
            projected = omega @ vectors
            projected_y = y @ vectors
            denominator = m - np.einsum("ij,ij->i", projected, projected)
            updated = (
                dots - np.einsum("ij,ij->i", projected, projected_y)
            ) / denominator
            relative_change = np.max(
                np.abs(updated - diagonal) / np.maximum(np.abs(updated), context["mu"])
            )
            diagonal = updated
            if relative_change < 2e-10:
                break
            if time.monotonic() > context["deadline"] - 8.0:
                break
        residual = y - diagonal[:, None] * omega
        q, _ = np.linalg.qr(residual @ vectors, mode="reduced")
        fit_residual = residual - (residual @ vectors) @ vectors.T
        fit_error = np.linalg.norm(fit_residual) / max(np.linalg.norm(y), 1e-300)
        positive = np.min(diagonal) > context["mu"] * 0.25
        if positive and fit_error < 1e-5:
            self.diagonal = diagonal
            self.basis = q
            self.mode = "transform"
        else:
            self.mode = "cg"
        return {"fit_error": float(fit_error), "fit_change": float(relative_change)}

    def _cg(self, hvp, b, context):
        now = time.monotonic()
        remaining_steps = max(1, context["steps"] - context["step"])
        end = now + max(0.0, context["deadline"] - now - 0.2) / remaining_steps
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        rr = np.dot(r, r)
        initial = rr
        iterations = 0
        while iterations < self.d and time.monotonic() < end - 0.005:
            try:
                hp = hvp(p)
            except Exception:
                break
            curvature = np.dot(p, hp)
            if not np.isfinite(curvature) or curvature <= 0:
                break
            alpha = rr / curvature
            x += alpha * p
            r -= alpha * hp
            new_rr = np.dot(r, r)
            iterations += 1
            if new_rr < 1e-22 * initial:
                break
            p *= new_rr / rr
            p += r
            rr = new_rr
        return x, {"mode": "cg", "iterations": iterations}

    def solve(self, hvp, b, context):
        if self.d <= 64:
            try:
                matrix = hvp(np.eye(self.d))
                return np.linalg.solve(matrix, b)
            except Exception:
                return np.zeros_like(b)
        diagnostics = {}
        if time.monotonic() >= context["deadline"] - 0.1:
            return np.zeros_like(b)
        if self.mode == "new":
            try:
                diagnostics.update(self._initialize(hvp, context))
            except Exception:
                self.mode = "cg"
        if self.mode != "transform":
            return self._cg(hvp, b, context)
        now = time.monotonic()
        remaining_steps = max(1, context["steps"] - context["step"])
        end = now + max(0.0, context["deadline"] - now - 0.2) / remaining_steps
        if context["step"] > 0:
            v = self.rng.integers(0, 2, size=self.d, dtype=np.int8).astype(float)
            v *= 2.0
            v -= 1.0
            # A probe orthogonal to the shared correction exposes today's diagonal.
            v -= self.basis @ (self.basis.T @ v)
            try:
                product = self._to_basis(hvp(self._from_basis(v)))
                proposed = product / v
                if (
                    np.min(proposed) > context["mu"] * 0.25
                    and np.isfinite(proposed).all()
                ):
                    self.diagonal = proposed
            except Exception:
                pass
        bhat = self._to_basis(b)
        diagonal = self.diagonal
        scale = np.dot(bhat, bhat / diagonal)
        if self.previous is not None and not context.get("phase_changed", False):
            x = self.previous.copy()
            try:
                r = bhat - self._to_basis(hvp(self._from_basis(x)))
            except Exception:
                return self._from_basis(x)
        else:
            x = np.zeros_like(bhat)
            r = bhat.copy()
        z = r / diagonal
        p = z.copy()
        rz = np.dot(r, z)
        iterations = 0
        for _ in range(40):
            if rz <= 1e-22 * scale or time.monotonic() > end - 0.008:
                break
            try:
                hp = self._to_basis(hvp(self._from_basis(p)))
            except Exception:
                break
            curvature = np.dot(p, hp)
            if curvature <= 0 or not np.isfinite(curvature):
                break
            alpha = rz / curvature
            x += alpha * p
            r -= alpha * hp
            z = r / diagonal
            new_rz = np.dot(r, z)
            iterations += 1
            p *= new_rz / rz
            p += z
            rz = new_rz
        if not np.isfinite(x).all():
            x = np.zeros_like(bhat)
        self.previous = x.copy()
        diagnostics.update(
            mode=self.mode,
            iterations=iterations,
            preconditioned_residual=float(rz / scale),
        )
        return self._from_basis(x), diagnostics


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)
