"""Recycled conjugate gradients with a sequence-wide wall-clock budget."""
import time
import numpy as np
from structured import DiagonalLowRank


class Solver:
    def __init__(self, dimension, seed, options):
        self.dimension = int(dimension)
        options = options or {}
        self.rank = max(2, min(32, int(options.get("history_rank", 3))))
        self.frontload = float(options.get("frontload", 4.0))
        self.seed = int(seed) % (1 << 64)
        self.rng = np.random.default_rng(self.seed)
        self.structured = bool(options.get("structured", True))
        self.learn_basis = bool(options.get("learn_basis", True))
        self.reuse_residual = bool(options.get("reuse_residual", True))
        self.residual_threshold = float(options.get("residual_threshold", .2))
        self.rhs_fraction = None
        self.previous_residual = None
        self.previous_b = None
        self.basis = None
        self.basis_seconds = 0.
        self.model = None
        self.diagonal = None
        self.history = []
        self.previous = None
        self.hvp_seconds = .003

    def _initialize_preconditioner(self, hvp, mu, L, end):
        self.model = None
        self.diagonal = None
        if time.monotonic() + max(.1, 64 * self.hvp_seconds) > end:
            return
        model = DiagonalLowRank(self.dimension, self.seed, 32)
        if model.fit(hvp, mu, L):
            model.columns = None
            self.model = model
            return
        if (self.dimension >= 128 and model.singular is not None and
                model.singular[-1] < .12 * model.singular[0] and
                time.monotonic() + .2 < end):
            larger = DiagonalLowRank(self.dimension, self.seed, 64)
            probes = np.zeros((self.dimension, 32))
            probes[larger.j[32:], np.arange(32)] = 1
            columns = np.column_stack((model.columns, hvp(probes)))
            if larger.fit(hvp, mu, L, columns):
                larger.columns = None
                self.model = larger
                return
        diagonal = model.columns[model.j, np.arange(model.rank)]
        if np.std(np.log(np.maximum(diagonal, mu))) < .7:
            return
        probes = self.rng.choice([-1., 1.], size=(self.dimension, 32))
        estimates = hvp(probes) * probes
        d1 = np.mean(estimates[:, :16], axis=1)
        d2 = np.mean(estimates[:, 16:], axis=1)
        correlation = np.corrcoef(d1, d2)[0, 1]
        if not np.isfinite(correlation) or correlation < .8:
            return
        diagonal = .5 * (d1 + d2)
        uncertainty = np.std(estimates, axis=1) / np.sqrt(32)
        diagonal = np.maximum(np.maximum(diagonal, 2 * uncertainty), mu)
        if np.quantile(diagonal, .9) >= 3 * np.quantile(diagonal, .1):
            self.diagonal = 1 / diagonal

    def _apply(self, r):
        if self.basis is not None:
            return self.basis.apply(r)
        if self.model is not None:
            return self.model.apply(r)
        if self.diagonal is not None:
            return self.diagonal * r
        return r

    def _initialize_basis(self, hvp, mu, deadline):
        if self.model is not None or self.diagonal is not None:
            self.basis = None
            return
        if (not self.learn_basis or self.dimension < 128 or
                self.dimension & (self.dimension - 1)):
            return
        start = time.monotonic()
        if self.basis is not None:
            try:
                end = min(deadline - .4, start + .7)
                if self.basis.refresh(hvp, mu, end):
                    self.basis_seconds += time.monotonic() - start
                    return
            except Exception:
                pass
            self.basis = None
        available = deadline - time.monotonic() - .5
        cap = min(6., 8.5 - self.basis_seconds, .6 * available)
        if cap < 3.5:
            return
        try:
            from basis import LearnedBasis
            candidate = LearnedBasis(self.dimension, self.seed)
            end = min(deadline - .4, time.monotonic() + cap)
            if candidate.fit(hvp, mu, end):
                self.basis = candidate
        except Exception:
            self.basis = None
        self.basis_seconds += time.monotonic() - start

    def solve(self, hvp, b, context):
        start = time.monotonic()
        step = int(context.get("step", 0))
        remaining = max(1, int(context.get("steps", 50)) - step)
        deadline = float(context.get("deadline", start + .55 * remaining))
        left = max(0., deadline - start - .12)
        phase_changed = context.get("phase_changed", step == 0)
        weight = self.frontload if phase_changed else 1.
        end = min(deadline - .08,
                  start + left * weight / (remaining + weight - 1))
        x = np.zeros(self.dimension)
        if left < .015:
            return self.previous.copy() if self.previous is not None else x
        if self.dimension <= 64:
            try:
                x = np.linalg.solve(hvp(np.eye(self.dimension)), b)
            except Exception:
                pass
            if not np.isfinite(x).all():
                x.fill(0)
            self.previous = x.copy()
            return x
        bn2 = float(b @ b)
        iterations = 0
        p = None
        mu = float(context.get("mu", 1.0))
        L = float(context.get("L", 1.0))
        if self.structured and phase_changed:
            try:
                self._initialize_preconditioner(hvp, mu, L, end)
            except Exception:
                self.model = None
                self.diagonal = None
            self._initialize_basis(hvp, mu, deadline)
            now = time.monotonic()
            end = max(end, min(deadline - .1, now + max(
                .05, (deadline - now - .15) / remaining)))
        identity = self.model is None and self.diagonal is None and self.basis is None
        if (self.reuse_residual and identity and phase_changed and step > 0 and
                time.monotonic() + 2 * self.hvp_seconds < end - .01):
            try:
                self.rhs_fraction = float(b @ hvp(b)) / (bn2 * L)
            except Exception:
                self.rhs_fraction = None
        recycled_residual = False
        try:
            project = None
            history = self.history
            if (self.reuse_residual and identity and not phase_changed and
                    self.rhs_fraction is not None and
                    self.rhs_fraction < self.residual_threshold and
                    self.previous_residual is not None and self.previous_b is not None):
                correlation = float(b @ self.previous_b) / max(
                    1e-100, np.sqrt(bn2 * float(self.previous_b @ self.previous_b)))
                if correlation > .5:
                    history = history + [self.previous_residual]
                    recycled_residual = True
            if history:
                u = np.column_stack(history)
                tick = time.monotonic()
                au = hvp(u)
                self.hvp_seconds = max(.0001, (time.monotonic() - tick) / len(history))
                e = u.T @ au
                vals, rot = np.linalg.eigh((e + e.T) * .5)
                keep = vals > max(1e-14, vals[-1] * 1e-12)
                inverse = (rot[:, keep] / vals[keep]) @ rot[:, keep].T
                coeff = inverse @ (u.T @ b)
                x = u @ coeff
                r = b - au @ coeff
                project = u @ inverse
            else:
                r = b.copy()
            z = self._apply(r)
            if project is not None:
                z = z - project @ (au.T @ z)
            rz = float(r @ z)
            p = z.copy()
            while float(r @ r) > 1e-20 * bn2:
                if time.monotonic() + max(.0015, self.hvp_seconds * 1.6) >= end - .002:
                    break
                if rz <= 0 or not np.isfinite(rz):
                    self.model = None
                    self.diagonal = None
                    self.basis = None
                    project = None
                    p = r.copy()
                    rz = float(r @ r)
                if not np.isfinite(float(p @ p)):
                    break
                tick = time.monotonic()
                ap = hvp(p)
                self.hvp_seconds = .97 * self.hvp_seconds + .03 * (
                    time.monotonic() - tick)
                den = float(p @ ap)
                if den <= 0 or not np.isfinite(den):
                    break
                if (self.reuse_residual and step == 0 and iterations == 0 and
                        identity and project is None):
                    self.rhs_fraction = den / (bn2 * L)
                alpha = rz / den
                x += alpha * p
                r -= alpha * ap
                z = self._apply(r)
                if project is not None:
                    z = z - project @ (au.T @ z)
                newrz = float(r @ z)
                beta = newrz / rz
                rz = newrz
                p *= beta
                p += z
                iterations += 1
        except Exception:
            # The last iterate remains usable if the shared oracle budget expires.
            pass
        norm = np.linalg.norm(x)
        if (not np.isfinite(x).all() or not np.isfinite(norm) or
                norm > 4 * np.sqrt(bn2) / mu):
            x.fill(0)
            norm = 0.
        self.previous_residual = None
        if norm > 0:
            if self.reuse_residual and "r" in locals():
                rnorm = np.linalg.norm(r)
                if np.isfinite(rnorm) and rnorm > 1e-8 * np.sqrt(bn2):
                    self.previous_residual = r / rnorm
            self.history.append(x / norm)
            if p is not None:
                norm = np.linalg.norm(p)
                if norm > 1e-100 and np.isfinite(norm):
                    self.history.append(p / norm)
            self.history = self.history[-self.rank:]
        self.previous = x.copy()
        self.previous_b = b.copy()
        return x, {"iterations": iterations, "history": len(self.history),
                   "rhs_fraction": self.rhs_fraction,
                   "residual_recycled": recycled_residual}


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)
