"""Online SPD solves using operator-learned structure and recycled PCG."""

import time

import numpy as np
from scipy.fft import dct, idct


class Solver:
    def __init__(self, dimension, seed, options):
        self.n = int(dimension)
        self.rng = np.random.default_rng(seed)
        self.options = options or {}
        self.signs = None
        self.q = None
        self.probes = None
        self.probe_spectral = None
        self.last_x = None
        self.history = []
        self.structured = None
        self.inverse_ready = False
        self.hvp_seconds = 0.002

    def _forward(self, value):
        transformed = idct(value, type=2, norm="ortho", axis=0, workers=1)
        if value.ndim == 1:
            return transformed * self.signs
        return transformed * self.signs[:, None]

    def _transpose(self, value):
        if value.ndim == 1:
            value = value * self.signs
        else:
            value = value * self.signs[:, None]
        return dct(value, type=2, norm="ortho", axis=0, workers=1)

    def _query(self, hvp, value):
        begin = time.monotonic()
        answer = hvp(value)
        count = 1 if value.ndim == 1 else value.shape[1]
        elapsed = (time.monotonic() - begin) / count
        self.hvp_seconds = 0.8 * self.hvp_seconds + 0.2 * elapsed
        return np.asarray(answer, dtype=np.float64)

    def _learn_signs(self, hvp):
        # Cosine-diagonal operators commute with a path adjacency matrix.
        # Its unknown signed edges form a two-state chain.
        n = self.n
        columns = 8
        vectors = np.zeros((n, columns))
        vectors[np.arange(columns), np.arange(columns)] = 1.0
        observed = self._query(hvp, vectors)
        width = columns - 1
        rows = np.linspace(
            columns + 2, n - 3, min(768, n - columns - 4)
        ).astype(int)
        left = observed[rows - 1, :width]
        right = observed[rows + 1, :width]
        predictions = np.stack(
            [-left - right, -left + right, left - right, left + right],
            axis=1,
        )
        best = np.inf
        best_edges = None
        for bits in range(1 << width):
            edges = 2.0 * ((bits >> np.arange(width)) & 1) - 1.0
            target = np.empty((len(rows), width))
            target[:, 0] = observed[rows, 0] + edges[0] * observed[rows, 1]
            for j in range(1, width):
                target[:, j] = (
                    edges[j - 1] * observed[rows, j - 1]
                    + edges[j] * observed[rows, j + 1]
                )
            error = predictions - target[:, None, :]
            costs = np.sum(error * error, axis=2)
            score = np.mean(np.min(costs, axis=1))
            if score < best:
                best = score
                best_edges = edges

        target = np.empty((n, width))
        target[:, 0] = observed[:, 0] + best_edges[0] * observed[:, 1]
        for j in range(1, width):
            target[:, j] = (
                best_edges[j - 1] * observed[:, j - 1]
                + best_edges[j] * observed[:, j + 1]
            )
        left = observed[:-2, :width]
        right = observed[2:, :width]
        center = target[1:-1]
        cross = np.sum(left * right, axis=1)
        lt = np.sum(left * center, axis=1)
        rt = np.sum(right * center, axis=1)
        costs = np.empty((n - 2, 2, 2))
        for a, sa in enumerate((-1.0, 1.0)):
            for c, sc in enumerate((-1.0, 1.0)):
                costs[:, a, c] = 2.0 * (
                    sa * sc * cross - sa * lt - sc * rt
                )
        parents = np.empty((n - 2, 2), dtype=np.int8)
        state = np.full(2, np.inf)
        state[int(best_edges[0] > 0)] = 0.0
        for i in range(n - 2):
            c = costs[i]
            a0 = state[0] + c[0, 0]
            a1 = state[1] + c[1, 0]
            b0 = state[0] + c[0, 1]
            b1 = state[1] + c[1, 1]
            parents[i, 0] = int(a1 < a0)
            parents[i, 1] = int(b1 < b0)
            state = np.array((min(a0, a1), min(b0, b1)))
            if i + 1 < width:
                state[int(best_edges[i + 1] < 0)] = np.inf
        final_cost = np.sum(
            (
                np.array((-1.0, 1.0))[:, None] * observed[-2, :width]
                + observed[-1, :width]
                - target[-1]
            )
            ** 2,
            axis=1,
        )
        end = int(np.argmin(state + final_cost))
        recovered = np.empty(n - 1)
        recovered[-1] = 2 * end - 1
        for i in range(n - 3, -1, -1):
            end = int(parents[i, end])
            recovered[i] = 2 * end - 1
        signs = np.ones(n)
        signs[1:] = np.cumprod(recovered)
        self.signs = signs
        return float(best / max(np.mean(observed[rows, :width] ** 2), 1e-300))

    def _discover(self, hvp, mu):
        sign_error = self._learn_signs(hvp)
        if sign_error > 0.03:
            raise ValueError("No accurate signed cosine structure")
        n = self.n
        split = n // 3
        k = 24
        # Disjoint supports expose the correction without diagonal pollution.
        # Their shared untouched rows align the two low-rank sketches.
        omega = np.zeros((n, 2 * k))
        omega[:split, :k] = self.rng.standard_normal((split, k))
        omega[split : 2 * split, k:] = self.rng.standard_normal((split, k))
        y = self._transpose(self._query(hvp, self._forward(omega)))
        ya, yb = y[:, :k], y[:, k:]
        ua, sa, va = np.linalg.svd(ya[2 * split :], full_matrices=False)
        ub, sb, vb = np.linalg.svd(yb[2 * split :], full_matrices=False)
        rank = int(np.count_nonzero(sa > max(sa[0] * 1e-8, mu * 1e-10)))
        if rank < 1 or rank > k - 3:
            raise ValueError("Unsupported correction rank")
        rank_b = int(np.count_nonzero(sb > max(sb[0] * 1e-8, mu * 1e-10)))
        if rank != rank_b:
            raise ValueError("Unstable correction rank")
        basis = np.empty((n, rank))
        basis[split:] = ya[split:] @ (va[:rank].T / sa[:rank])
        alignment = ub[:, :rank].T @ ua[:, :rank]
        basis[:split] = yb[:split] @ (
            (vb[:rank].T / sb[:rank]) @ alignment
        )
        self.q = np.linalg.qr(basis, mode="reduced")[0]
        self.split = split
        tail = self.q[split:]
        self.tail_left = np.linalg.solve(tail.T @ tail, tail.T)
        head = self.q[:split]
        head_inverse = np.linalg.inv(head.T @ head)
        # Q.T @ probe[:, :rank] = I, while the probe vanishes on the tail.
        self.probe_spectral = np.zeros((n, rank + 1))
        self.probe_spectral[:split, :rank] = head @ head_inverse
        self.probe_spectral[:, -1] = self.rng.choice((-1.0, 1.0), n)
        self.probes = self._forward(self.probe_spectral)
        self.structured = True
        return sign_error, rank

    def _refresh(self, hvp, mu):
        y = self._transpose(self._query(hvp, self.probes))
        rank = self.q.shape[1]
        core = self.tail_left @ y[self.split :, :rank]
        core = (core + core.T) * 0.5
        omega = self.probe_spectral[:, -1]
        diagonal = (
            y[:, -1] - self.q @ (core @ (self.q.T @ omega))
        ) * omega
        if not np.isfinite(diagonal).all() or np.min(diagonal) < mu * 0.5:
            raise ValueError("Invalid reconstructed diagonal")
        eigenvalues, eigenvectors = np.linalg.eigh(core)
        if np.min(eigenvalues) < -max(1e-8, np.max(eigenvalues) * 1e-8):
            raise ValueError("Indefinite reconstructed correction")
        factor = (self.q @ eigenvectors) * np.sqrt(
            np.maximum(eigenvalues, 0.0)
        )
        self.inverse_diagonal = 1.0 / diagonal
        self.scaled_factor = factor * self.inverse_diagonal[:, None]
        small = np.eye(rank) + factor.T @ self.scaled_factor
        self.small_inverse = np.linalg.inv(small)
        self.factor = factor
        self.inverse_ready = True

    def _inverse_spectral(self, rhs):
        # Woodbury inversion of the currently reconstructed operator.
        first = rhs * self.inverse_diagonal
        return first - self.scaled_factor @ (
            self.small_inverse @ (self.factor.T @ first)
        )

    def _inverse(self, rhs):
        return self._forward(self._inverse_spectral(self._transpose(rhs)))

    def _cg(self, hvp, b, end, precondition=None, initial=None):
        x = np.zeros(self.n) if initial is None else initial.copy()
        iterations = 0
        if time.monotonic() + max(0.005, 3 * self.hvp_seconds) >= end:
            return x, iterations
        try:
            r = b.copy() if initial is None else b - self._query(hvp, x)
        except Exception:
            return x, iterations
        z = r.copy() if precondition is None else precondition(r)
        p = z.copy()
        rz = float(r @ z)
        bnorm = float(b @ b)
        while (
            np.isfinite(rz)
            and rz > 0.0
            and float(r @ r) > 1e-24 * bnorm
            and time.monotonic() + 2.5 * self.hvp_seconds < end
        ):
            try:
                product = self._query(hvp, p)
            except Exception:
                break
            curvature = float(p @ product)
            if not np.isfinite(curvature) or curvature <= 0.0:
                break
            alpha = rz / curvature
            x += alpha * p
            r -= alpha * product
            iterations += 1
            z = r.copy() if precondition is None else precondition(r)
            new_rz = float(r @ z)
            if not np.isfinite(new_rz) or new_rz <= 0.0:
                break
            p *= new_rz / rz
            p += z
            rz = new_rz
        return x, iterations

    def _fallback(self, hvp, b, context, end):
        initial = None if context.get("phase_changed", False) else self.last_x
        precondition = None
        if (
            self.history
            and time.monotonic() + max(0.025, 24 * self.hvp_seconds) < end
        ):
            try:
                candidates = np.column_stack(self.history)
                q, triangular = np.linalg.qr(candidates, mode="reduced")
                keep = np.abs(np.diag(triangular)) > 1e-9
                q = q[:, keep]
                if q.shape[1]:
                    hq = self._query(hvp, q)
                    coarse = q.T @ hq
                    coarse = (coarse + coarse.T) * 0.5
                    eigenvalues, eigenvectors = np.linalg.eigh(coarse)
                    if np.min(eigenvalues) <= 0:
                        raise ValueError("Invalid coarse curvature")
                    inverse = (eigenvectors / eigenvalues) @ eigenvectors.T
                    initial = q @ (inverse @ (q.T @ b))
                    gamma = 1.0 / np.sqrt(
                        float(context.get("mu", 1.0))
                        * float(context.get("L", 1.0))
                    )

                    def precondition(r):
                        coefficients = inverse @ (q.T @ r)
                        projected = gamma * (r - hq @ coefficients)
                        return projected + q @ (
                            coefficients - inverse @ (hq.T @ projected)
                        )
            except Exception:
                precondition = None
        return self._cg(
            hvp, b, end, precondition=precondition, initial=initial
        )

    def _remember(self, x):
        if self.last_x is not None:
            difference = x - self.last_x
            norm = float(np.linalg.norm(difference))
            if norm > 1e-14:
                self.history.append(difference / norm)
        norm = float(np.linalg.norm(x))
        if norm > 1e-14:
            self.history.append(x / norm)
        self.history = self.history[-12:]

    def solve(self, hvp, b, context):
        now = time.monotonic()
        deadline = float(context.get("deadline", now + 28.0))
        left_steps = max(
            1, int(context.get("steps", 50)) - int(context.get("step", 0))
        )
        end = min(
            deadline - 0.08,
            now + max(0.0, (deadline - now - 0.15) / left_steps),
        )
        mu = float(context.get("mu", 1.0))
        diagnostics = {}
        x = np.zeros(self.n)
        try:
            if now + 0.015 >= deadline:
                return self._inverse(b) if self.inverse_ready else x
            if self.n <= 64:
                matrix = self._query(hvp, np.eye(self.n))
                x = np.linalg.solve((matrix + matrix.T) * 0.5, b)
            else:
                if self.structured is None and deadline - now > 3.0:
                    error, rank = self._discover(hvp, mu)
                    diagnostics.update(sign_error=error, rank=rank)
                if self.structured:
                    refresh_time = max(
                        0.02, 2 * self.probes.shape[1] * self.hvp_seconds
                    )
                    if time.monotonic() + refresh_time < deadline - 0.12:
                        self._refresh(hvp, mu)
                        x = self._inverse(b)
                        cg_end = min(
                            deadline - 0.08,
                            max(end, time.monotonic() + 0.025),
                        )
                        x, iterations = self._cg(
                            hvp, b, cg_end, precondition=self._inverse,
                            initial=x,
                        )
                        diagnostics["iterations"] = iterations
                    elif self.inverse_ready:
                        x = self._inverse(b)
                else:
                    x, iterations = self._fallback(hvp, b, context, end)
                    diagnostics["iterations"] = iterations
        except Exception as error:
            self.structured = False
            diagnostics["fallback"] = type(error).__name__ + ": " + str(error)
            try:
                if time.monotonic() + 3 * self.hvp_seconds < end:
                    x, iterations = self._fallback(hvp, b, context, end)
                    diagnostics["iterations"] = iterations
            except Exception:
                x = np.zeros(self.n)
        if not np.isfinite(x).all():
            x = np.zeros(self.n)
        if not self.structured:
            self._remember(x)
        self.last_x = x.copy()
        return x, diagnostics


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)
