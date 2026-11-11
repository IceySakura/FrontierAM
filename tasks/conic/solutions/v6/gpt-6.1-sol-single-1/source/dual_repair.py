"""Alternating convex projections for accurate dual stationarity and cones."""

import time
import numpy as np
from scipy import sparse, linalg
from scipy.sparse import linalg as sla
from common import norm, smat, svec, assess


class DualRepair:
    def __init__(self, p, initial_z=None):
        self.p = p
        row_degree = p.A.getnnz(axis=1).astype(float)
        if p.n > 100000 or (p.n > 2000 and float(row_degree @ row_degree) > 5000000):
            raise ValueError("normal-equation fill limit")
        self.nn, self.soc, self.psd = [], {}, []
        for kind, d, i, j in p.blocks:
            if kind == "nonnegative":
                self.nn.extend(range(i, j))
            elif kind == "soc":
                self.soc.setdefault(d, []).append(i)
            elif kind == "psd":
                self.psd.append((d, i, j))
            elif kind == "exponential":
                raise ValueError("nonsymmetric cone")
        self.nn = np.asarray(self.nn, int)
        self.soc = {d: np.asarray(v)[:, None] + np.arange(d) for d, v in self.soc.items()}
        self.dual_scale = np.ones(p.m)
        if initial_z is not None:
            self.dual_scale[p.eq] = np.maximum(1, abs(initial_z[p.eq]))
            self.dual_scale[self.nn] = np.maximum(0.05, abs(initial_z[self.nn]))
            for idx in self.soc.values():
                self.dual_scale[idx] = np.maximum(0.05, np.max(abs(initial_z[idx]), axis=1))[:, None]
            for d, i, j in self.psd:
                self.dual_scale[i:j] = max(0.05, norm(initial_z[i:j]))
        weighted_A = p.A.multiply(self.dual_scale[:, None]).tocsc()
        column_norm = np.sqrt(np.asarray(weighted_A.power(2).sum(axis=0)).ravel())
        self.scale = 1 / np.maximum(column_norm, 1e-12)
        self.B = weighted_A.multiply(self.scale).tocsc()
        self.G = (self.B.T @ self.B).tocsc()
        if self.G.nnz > 5000000:
            raise ValueError("normal-equation fill limit")
        self.factor = sla.splu(self.G + sparse.eye(p.n, format="csc") * 1e-13)
        self.bs = p.b * self.dual_scale
        self.w = self.bs - self.B @ self.solve(self.B.T @ self.bs)
        self.ww = float(self.w @ self.w)

    def solve(self, rhs):
        v = self.factor.solve(rhs)
        for _ in range(2):
            r = rhs - self.G @ v
            if norm(r) < 1e-13 * (1 + norm(rhs)):
                break
            v += self.factor.solve(r)
        return v

    def cone(self, v):
        out = v.copy()
        out[self.nn] = np.maximum(out[self.nn], 0)
        for idx in self.soc.values():
            a = v[idx]
            r = np.linalg.norm(a[:, 1:], axis=1)
            inside, zero = r <= a[:, 0], r <= -a[:, 0]
            mid = ~(inside | zero)
            b = np.zeros_like(a)
            b[inside] = a[inside]
            b[mid, 0] = (r[mid] + a[mid, 0]) / 2
            b[mid, 1:] = b[mid, :1] * a[mid, 1:] / r[mid, None]
            out[idx] = b
        for d, i, j in self.psd:
            ev, V = linalg.eigh(smat(v[i:j], d), check_finite=False)
            out[i:j] = svec((V * np.maximum(ev, 0)) @ V.T)
        return out

    def repair(self, candidate, budget=5.0):
        p = self.p
        started = time.monotonic()
        x, s = candidate["x"].copy(), candidate["s"]
        if not p.P.nnz:
            residual = (p.b - s - p.A @ x) * self.dual_scale
            x += self.scale * self.solve(self.B.T @ residual)
        px = p.P @ x
        target = (-px - p.q) * self.scale
        bz_target = -float(x @ px + p.q @ x)
        primal = float(0.5 * x @ px + p.q @ x)
        width = 7e-9 * max(1, abs(primal))
        linear_objective = not p.P.nnz and norm(p.q) > 0
        if linear_objective:
            objective_direction = self.scale * self.solve(p.q * self.scale)
            objective_denominator = float(p.q @ objective_direction)
        def affine(v):
            z = v + self.B @ self.solve(target - self.B.T @ v)
            if linear_objective:
                return z
            bz = float(self.bs @ z)
            delta = np.clip(bz, bz_target - width, bz_target + width) - bz
            if self.ww > 1e-20 and delta:
                z += (delta / self.ww) * self.w
            return z
        z = candidate["z"] / self.dual_scale
        u, v = np.zeros(p.m), np.zeros(p.m)
        best = dict(candidate, x=x)
        besterr = assess(p, x, s, candidate["z"], check_all=True)[1]
        for it in range(350):
            if time.monotonic() - started >= budget:
                break
            y = affine(z + u)
            u += z - y
            zz = self.cone(y + v)
            v += y - zz
            z = zz
            if it % 3 == 0:
                for trial in (y, zz):
                    trial = trial * self.dual_scale
                    xx = x
                    if linear_objective and objective_denominator > 1e-30:
                        delta = -float(p.b @ trial) - float(p.q @ x)
                        xx = x + objective_direction * (delta / objective_denominator)
                    ok, err = assess(p, xx, s, trial, check_all=True)
                    if err < besterr:
                        best, besterr = dict(candidate, x=xx.copy(), z=trial.copy()), err
                    if ok:
                        return dict(candidate, x=xx, z=trial)
        return best
