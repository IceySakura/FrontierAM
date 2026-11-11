"""Log-sum-exp reduction and primal-dual polishing of geometric programs."""

import os
import time
import warnings

import numpy as np
from scipy import linalg, sparse
from scipy.optimize import minimize, nnls

from problem import blocks
from quality import acceptable, norm


class Deadline(Exception):
    pass


class GeometricProgram:
    def __init__(self, p):
        self.p = p
        nn, exp = [], []
        for kind, dim, sl in blocks(p.cones):
            if kind == "nonnegative":
                nn.extend(range(sl.start, sl.stop))
            elif kind == "exponential":
                exp.append(np.arange(sl.start, sl.stop))
            else:
                raise ValueError("Not an eliminable geometric program")
        self.nn = np.asarray(nn, np.int32)
        self.exp = np.asarray(exp, np.int32)
        if self.exp.size == 0:
            raise ValueError("No exponential terms")
        U, V, W = (p.A[self.exp[:, i]].tocsr() for i in range(3))
        if V.nnz or np.any(np.diff(W.indptr) != 1):
            raise ValueError("Exponential perspective is not constant")
        self.nu = p.b[self.exp[:, 1]]
        self.beta = -W.data
        self.wcols = W.indices
        if np.any(self.nu <= 0) or np.any(self.beta <= 0) or np.unique(self.wcols).size != self.wcols.size:
            raise ValueError("Nonstandard epigraph")
        if np.any(p.q[self.wcols]) or np.any(np.diff(p.P.indptr)[self.wcols]) or U[:, self.wcols].nnz:
            raise ValueError("Coupled exponential epigraph")
        core = np.ones(p.n, bool)
        core[self.wcols] = False
        self.core = np.flatnonzero(core)
        if self.core.size > 1200:
            raise ValueError("Log-space problem too large for dense polishing")
        G = p.A[self.nn].tocsc()
        Gw = G[:, self.wcols].tocsc()
        if np.any(np.diff(Gw.indptr) != 1) or np.any(Gw.data <= 0):
            raise ValueError("Epigraph terms are not monotone and separable")
        self.group = Gw.indices.copy()
        self.wcoef = Gw.data.copy()
        self.m = self.nn.size
        self.nonlinear = np.bincount(self.group, minlength=self.m) > 0
        self.linear = ~self.nonlinear
        self.G = G[:, self.core].tocsc()
        if self.G[self.nonlinear].nnz:
            raise ValueError("Mixed affine and exponential inequality")
        self.rhs = p.b[self.nn] + np.bincount(self.group,
            weights=self.wcoef * p.b[self.exp[:, 2]] / self.beta, minlength=self.m)
        if np.any(self.rhs[self.nonlinear] <= 0):
            raise ValueError("Nonpositive posynomial bound")
        self.A = (sparse.diags(1 / self.nu) @ (-U[:, self.core])).tocsr()
        self.offset = (p.b[self.exp[:, 0]] / self.nu
                       + np.log(self.wcoef * self.nu / self.beta))
        self.Gdense = self.G.toarray()
        rowmax = np.max(np.abs(self.Gdense), axis=1, initial=0)
        self.linear_scale = 1 / np.maximum(1.0, np.maximum(rowmax, np.abs(self.rhs)))
        self.q = p.q[self.core]
        self.P = p.P[self.core][:, self.core].toarray()
        self.last = None
        self.y = np.zeros(self.core.size)

    def evaluate(self, y):
        if self.last is not None and np.array_equal(y, self.last):
            return self.f, self.J
        values = self.A @ y + self.offset
        maximum = np.full(self.m, -np.inf)
        np.maximum.at(maximum, self.group, values)
        w = np.exp(values - maximum[self.group])
        total = np.bincount(self.group, weights=w, minlength=self.m)
        probability = w / total[self.group]
        S = sparse.csr_matrix((probability, (self.group, np.arange(self.group.size))),
                              shape=(self.m, self.group.size))
        J = (S @ self.A).toarray()
        f = np.empty(self.m)
        nl = self.nonlinear
        f[nl] = maximum[nl] + np.log(total[nl]) - np.log(self.rhs[nl])
        f[self.linear] = (self.Gdense[self.linear] @ y - self.rhs[self.linear]) * self.linear_scale[self.linear]
        J[self.linear] = self.Gdense[self.linear] * self.linear_scale[self.linear, None]
        self.f, self.J, self.probability = f, J, probability
        self.last = y.copy()
        return f, J

    def hessian(self, multipliers):
        weights = multipliers[self.group] * self.probability
        H = (self.A.T @ self.A.multiply(weights[:, None])).toarray()
        J = self.J[self.nonlinear]
        H -= J.T @ (multipliers[self.nonlinear, None] * J)
        return 0.5 * (H + H.T) + self.P

    def result(self, y, multipliers):
        p = self.p
        self.evaluate(y)
        x = np.zeros(p.n)
        x[self.core] = y
        ratios = (p.b[self.exp[:, 0]] - p.A[self.exp[:, 0]][:, self.core] @ y) / self.nu
        ev = np.exp(np.clip(ratios, -690, 690))
        w = self.nu * ev
        x[self.wcols] = (w - p.b[self.exp[:, 2]]) / self.beta
        z = np.zeros(p.m)
        raw = multipliers.copy()
        summed = np.bincount(self.group, weights=self.wcoef * w / self.beta, minlength=self.m)
        raw[self.nonlinear] /= summed[self.nonlinear]
        raw[self.linear] *= self.linear_scale[self.linear]
        z[self.nn] = raw
        zw = raw[self.group] * self.wcoef / self.beta
        z[self.exp[:, 0]] = -zw * ev
        z[self.exp[:, 1]] = zw * ev * (ratios - 1)
        z[self.exp[:, 2]] = zw
        s = p.b - p.A @ x
        s[self.exp[:, 2]] = w
        s[self.nn] = np.maximum(s[self.nn], 0.0)
        return {"status": "solved", "x": x, "s": s, "z": z}


def solve(p, deadline):
    try:
        gp = GeometricProgram(p)
    except ValueError:
        return None
    n = gp.core.size
    if n == 0:
        return None

    def obj(y):
        return float(0.5 * y @ gp.P @ y + gp.q @ y)

    def jac(y):
        return gp.P @ y + gp.q

    def callback(y):
        gp.y = y.copy()
        if time.monotonic() > deadline:
            raise Deadline()

    multipliers = None
    try:
        optimized = minimize(obj, gp.y, jac=jac, method="SLSQP",
            constraints={"type": "ineq", "fun": lambda y: -gp.evaluate(y)[0],
                         "jac": lambda y: -gp.evaluate(y)[1]},
            callback=callback, options={"ftol": 2e-12, "maxiter": 400, "disp": False})
        y = optimized.x
        multipliers = getattr(optimized, "multipliers", None)
        if os.environ.get("CONIC_DEBUG"):
            print("GP", optimized.message, optimized.nit, "base", n,
                  "terms", gp.group.size, file=__import__("sys").stderr)
    except Deadline:
        y = gp.y
    f, J = gp.evaluate(y)
    if multipliers is None or len(multipliers) != gp.m:
        active = f > -1e-5
        multipliers = np.zeros(gp.m)
        if np.any(active):
            try:
                multipliers[active] = nnls(J[active].T, -jac(y), maxiter=10 * gp.m)[0]
            except RuntimeError:
                multipliers[active] = np.maximum(0, linalg.lstsq(J[active].T, -jac(y))[0])
    multipliers = np.asarray(multipliers)
    result = gp.result(y, multipliers)
    if acceptable(p, result):
        return result
    active = (f > -1e-5) & (multipliers > 1e-8)
    active |= f > 1e-7
    for iteration in range(30):
        if time.monotonic() > deadline:
            break
        f, J = gp.evaluate(y)
        multipliers[~active] = 0
        result = gp.result(y, multipliers)
        if acceptable(p, result):
            return result
        H = gp.hessian(multipliers)
        B = J[active]
        K = np.block([[H, B.T], [B, np.zeros((B.shape[0], B.shape[0]))]])
        rhs = np.r_[-jac(y) - J.T @ multipliers, -f[active]]
        regularization = np.r_[np.full(n, 1e-10), np.full(B.shape[0], -1e-10)]
        Kr = K.copy()
        Kr.flat[::Kr.shape[0]+1] += regularization
        try:
            factor = linalg.lu_factor(Kr, check_finite=False)
            direction = linalg.lu_solve(factor, rhs, check_finite=False)
            for refinement in range(6):
                residual = rhs - K @ direction
                if norm(residual) < 1e-13 * norm(rhs) + 1e-16:
                    break
                trial = direction + linalg.lu_solve(factor, residual, check_finite=False)
                if norm(rhs - K @ trial) > 1.1 * norm(residual):
                    break
                direction = trial
        except (np.linalg.LinAlgError, ValueError):
            break
        if not np.isfinite(direction).all():
            break
        dy, dl = direction[:n], direction[n:]
        alpha = min(1.0, 10.0 / max(1.0, norm(dy)))
        y += alpha * dy
        multipliers[active] += alpha * dl
        active[multipliers < -1e-9] = False
        f, J = gp.evaluate(y)
        active |= f > 1e-8
        multipliers = np.maximum(0, multipliers)
    return None
