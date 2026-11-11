#!/usr/bin/env python3
"""Low-startup LP/QP entry point; imports sparse conic machinery on demand."""

import argparse
import glob
import importlib.util
import json
import math
import os
import sys
import sysconfig
import time

import numpy as np
sys.modules.setdefault("fast", sys.modules[__name__])


VERBOSE = bool(os.environ.get("SOLVER_VERBOSE"))
DEADLINE = time.monotonic() + 282.0


def remaining():
    return max(0., DEADLINE - time.monotonic())


def log(*v):
    if VERBOSE:
        print(*v, file=sys.stderr, flush=True)


def norm(v):
    return float(np.max(np.abs(v), initial=0.0))


def core():
    if "_core" not in sys.modules:
        base = sysconfig.get_paths()["purelib"]
        path = glob.glob(base + "/scipy/optimize/_highspy/_core*.so")[0]
        spec = importlib.util.spec_from_file_location("_core", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_core"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["_core"]


class Raw:
    def __init__(self, stream, meta):
        self.meta = meta
        self.q, self.b = stream["q"], stream["b"]
        self.n, self.m = len(self.q), len(self.b)
        self.eq = np.zeros(self.m, dtype=bool)
        self.nn = np.zeros(self.m, dtype=bool)
        self.blocks = []
        pos = 0
        for c in meta["cones"]:
            d, t = c["dim"], c["type"]
            width = d * (d + 1) // 2 if t == "psd" else d
            if c["type"] == "zero":
                self.eq[pos:pos + width] = True
            elif c["type"] == "nonnegative":
                self.nn[pos:pos + width] = True
            self.blocks.append((t, d, pos, pos + width))
            pos += width
        self.nonlinear = any(t not in ("zero", "nonnegative") for t, _, _, _ in self.blocks)
        for name in ("A", "P"):
            data, indices, ptr = (stream[name + s] for s in ("_data", "_indices", "_indptr"))
            indices, ptr = indices.astype(np.int32), ptr.astype(np.int32)
            cols = np.repeat(np.arange(self.n, dtype=np.int32), np.diff(ptr))
            if len(data) > 1 and np.any((cols[1:] == cols[:-1]) &
                                       (indices[1:] <= indices[:-1])):
                from minisparse import CSC
                matrix = CSC(self.m if name == "A" else self.n, self.n,
                             data.copy(), indices.copy(), ptr.copy()).canonical()
                data, indices, ptr, cols = matrix.data, matrix.rows, matrix.ptr, matrix.cols
            if np.any(data == 0):
                use = data != 0
                data, indices, cols = data[use], indices[use], cols[use]
                ptr = np.r_[0, np.cumsum(np.bincount(cols, minlength=self.n))].astype(np.int32)
            setattr(self, name, (data, indices, ptr, cols))

    def mul(self, name, x, transpose=False):
        data, rows, ptr, cols = getattr(self, name)
        size = self.n if name == "P" or transpose else self.m
        return np.bincount(cols if transpose else rows,
                           weights=data * x[rows if transpose else cols], minlength=size)

    def mul_scaled(self, name, x, scale, transpose=False):
        data, rows, ptr, cols = getattr(self, name)
        size = self.n if name == "P" or transpose else self.m
        return np.bincount(cols if transpose else rows,
                           weights=(data / scale) * x[rows if transpose else cols], minlength=size)

    def result(self, x, z):
        ax = self.mul("A", x)
        slack = self.b - ax
        slack[self.eq] = 0
        slack[self.nn] = np.maximum(slack[self.nn], 0)
        return {"status": "solved", "x": x, "s": slack, "z": z}

    def cone(self, v, dual=False):
        if not hasattr(self, "cone_groups"):
            grouped, psd = {}, []
            for t, d, lo, hi in self.blocks:
                if t == "psd":
                    psd.append((d, lo, hi))
                elif t != "zero":
                    grouped.setdefault((t, d), []).append(lo)
            self.cone_groups = [(t, np.array(starts)[:, None] + np.arange(d))
                                for (t, d), starts in grouped.items()], psd
        worst = 0.0
        if not dual:
            nv = norm(v[self.eq])
            worst = nv / max(1, nv)
        grouped, psd = self.cone_groups
        for t, ix in grouped:
            u = v[ix]
            u /= np.maximum(1, np.max(np.abs(u), axis=1))[:, None]
            if t == "nonnegative":
                val = max(0., -float(np.min(u)))
            elif t == "soc":
                val = max(0., float(np.max(np.linalg.norm(u[:, 1:], axis=1) - u[:, 0])))
            else:
                a, b, c = (-u[:, 1], -u[:, 0], math.e * u[:, 2]) if dual else u.T
                valarray = np.maximum.reduce((np.abs(b), a, -c, np.zeros(len(a))))
                good = (b > 0) & (c > 0)
                valarray[good] = np.minimum(valarray[good], np.maximum(
                    0, a[good] - b[good] * (np.log(c[good]) - np.log(b[good]))))
                val = float(np.max(valarray, initial=0))
            worst = max(worst, val)
        for d, lo, hi in psd:
            c = np.repeat(np.arange(d), np.arange(1, d + 1))
            r = np.arange(hi - lo) - c * (c + 1) // 2
            a = v[lo:hi] / max(1, norm(v[lo:hi]))
            a = a / np.where(r == c, 1, math.sqrt(2))
            M = np.zeros((d, d))
            M[r, c], M[c, r] = a, a
            worst = max(worst, max(0, -float(np.linalg.eigvalsh(M)[0])))
        return worst

    def valid(self, r, tol=8.8e-9):
        st = r["status"]
        if st == "solved":
            x, s, z = r["x"], r["s"], r["z"]
            ax, px, az = self.mul("A", x), self.mul("P", x), self.mul("A", z, True)
            rp = norm(ax + s - self.b) / (1 + max(norm(ax), norm(s), norm(self.b)))
            rd = norm(px + self.q + az) / (1 + max(norm(px), norm(self.q), norm(az)))
            op, od = 0.5 * x @ px + self.q @ x, -0.5 * x @ px - self.b @ z
            gap = abs(op - od) / max(1, abs(op), abs(od))
            # Every linear block has the same sign test; normalization is
            # unnecessary for vectors explicitly projected onto these cones.
            cn = (max(self.cone(s), self.cone(z, True)) if self.nonlinear else
                  (0. if np.min(z[~self.eq], initial=0.) >= 0 else self.cone(z, True)))
            vals = [rp, rd, gap, cn]
            log("fast residuals", vals)
        elif st == "primal_infeasible":
            if not norm(r["z"]) or not norm(self.b):
                return False
            z = r["z"] / norm(r["z"])
            margin = -(self.b / norm(self.b)) @ z
            if margin <= 1e-8:
                return False
            vals = [norm(self.mul_scaled("A", z, norm(self.A[0]) or 1, True)) / margin,
                    (self.cone(z, True) if self.nonlinear else
                     max(0, -np.min(z[~self.eq], initial=0.0))) / margin]
        else:
            if not norm(r["x"]) or not norm(self.q):
                return False
            x, s = r["x"] / norm(r["x"]), r["s"] / norm(r["x"])
            margin = -(self.q / norm(self.q)) @ x
            if margin <= 1e-8:
                return False
            a = norm(self.A[0]) or 1
            vals = [norm(self.mul_scaled("P", x, norm(self.P[0]) or 1)) / margin,
                    norm(self.mul_scaled("A", x, a) + s / a) / margin,
                    self.cone(s / a) / margin]
        passed = all(math.isfinite(v) and v <= tol for v in vals)
        if passed and hasattr(self, "_postcheck"):
            return self._postcheck(r)
        return passed


def engine():
    hmod = core()
    h = hmod._Highs()
    for k, v in {
        "output_flag": False, "threads": 1, "parallel": "off",
        "primal_feasibility_tolerance": 1e-9,
        "dual_feasibility_tolerance": 1e-9,
        "ipm_optimality_tolerance": 1e-11,
        "small_matrix_value": 1e-12, "large_matrix_value": 1e25,
        "time_limit": min(270., max(.01, remaining() - 1.)),
    }.items():
        h.setOptionValue(k, v)
    method = os.environ.get("SOLVER_LP_METHOD")
    if method:
        h.setOptionValue("solver", method)
    crossover = os.environ.get("SOLVER_CROSSOVER")
    if crossover:
        h.setOptionValue("run_crossover", crossover)
    return h


def pass_raw(h, n, m, q, lower, upper, data, rows, ptr,
             col_lower=None, col_upper=None, P=None):
    mod = core()
    lp = mod.HighsLp()
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = q
    lp.col_lower_ = np.full(n, -mod.kHighsInf) if col_lower is None else col_lower
    lp.col_upper_ = np.full(n, mod.kHighsInf) if col_upper is None else col_upper
    lp.row_lower_, lp.row_upper_ = lower, upper
    mat = lp.a_matrix_
    mat.format_ = mod.MatrixFormat.kColwise
    mat.num_col_, mat.num_row_ = n, m
    mat.start_, mat.index_, mat.value_ = ptr.astype(np.int32), rows.astype(np.int32), data
    if P is not None and len(P[0]):
        model = mod.HighsModel()
        model.lp_ = lp
        pd, pi, pp, pc = P
        use = pi >= pc
        model.hessian_.dim_ = n
        model.hessian_.format_ = mod.HessianFormat.kTriangular
        model.hessian_.start_ = np.r_[0, np.cumsum(
            np.bincount(pc[use], minlength=n))].astype(np.int32)
        model.hessian_.index_, model.hessian_.value_ = pi[use].astype(np.int32), pd[use]
        h.passModel(model)
    else:
        h.passModel(lp)


def phase1(p):
    mod = core()
    ad, ar, ap, ac = p.A
    ei = np.flatnonzero(p.eq)
    loc = np.full(p.m, -1, dtype=np.int32)
    loc[ei] = np.arange(len(ei))
    use = p.eq[ar]
    rows = np.r_[ar, p.m + loc[ar[use]]]
    cols = np.r_[ac, ac[use]]
    vals = np.r_[ad, -ad[use]]
    order = np.argsort(cols, kind="stable")
    rows, vals = rows[order], vals[order]
    ptr = np.r_[0, np.cumsum(np.bincount(cols, minlength=p.n))].astype(np.int32)
    m = p.m + len(ei)
    rows = np.r_[rows, np.arange(m, dtype=np.int32)]
    vals = np.r_[vals, -np.ones(m)]
    ptr = np.r_[ptr, len(rows)].astype(np.int32)
    rhs = np.r_[p.b, -p.b[ei]]
    q = np.zeros(p.n + 1)
    q[-1] = 1
    lb, ub = np.full(p.n + 1, -mod.kHighsInf), np.full(p.n + 1, mod.kHighsInf)
    lb[-1] = 0
    h = engine()
    refine_basis = norm(p.q) == 0 and not os.environ.get("SOLVER_LP_METHOD")
    if refine_basis:
        h.setOptionValue("solver", "ipm")
        h.setOptionValue("run_crossover", "off")
    pass_raw(h, p.n + 1, m, q, np.full(m, -mod.kHighsInf), rhs,
             vals, rows, ptr, lb, ub)
    for attempt in range(2 if refine_basis else 1):
        h.run()
        sol = h.getSolution()
        if len(sol.col_value) == p.n + 1 and sol.col_value[-1] > 1e-11:
            y = -np.asarray(sol.row_dual)
            z = y[:p.m].copy()
            z[ei] -= y[p.m:]
            r = {"status": "primal_infeasible", "z": z}
            if p.valid(r):
                return r
        if refine_basis and attempt == 0:
            h.setOptionValue("solver", "simplex")
            h.setOptionValue("run_crossover", "on")
    return None


def recession(p):
    if norm(p.q) == 0:
        return None
    mod = core()
    ad, ar, ap, ac = p.A
    pd, pr, pp, pc = p.P
    pscale = norm(pd) or 1
    vals = np.r_[ad, pd / pscale]
    rows, cols = np.r_[ar, p.m + pr], np.r_[ac, pc]
    order = np.argsort(cols, kind="stable")
    rows, vals = rows[order], vals[order]
    ptr = np.r_[0, np.cumsum(np.bincount(cols, minlength=p.n))].astype(np.int32)
    m = p.m + p.n
    eq = np.r_[p.eq, np.ones(p.n, dtype=bool)]
    h = engine()
    pass_raw(h, p.n, m, p.q / norm(p.q), np.where(eq, 0., -mod.kHighsInf),
             np.zeros(m), vals, rows, ptr, -np.ones(p.n), np.ones(p.n))
    h.run()
    sol = h.getSolution()
    if len(sol.col_value) != p.n:
        return None
    x = np.asarray(sol.col_value)
    r = {"status": "dual_infeasible", "x": x, "s": -p.mul("A", x)}
    return r if p.valid(r) else None


def dense_polish(p, x, z):
    if p.n > 600:
        return None
    ad, ar, ap, ac = p.A
    pd, pr, pp, pc = p.P
    slack = p.b - p.mul("A", x)
    active = p.eq | ((z > 1e-8 * max(1, norm(z))) |
                      (np.abs(slack) < 1e-9))
    ix = np.flatnonzero(active)
    if len(ix) > 2 * p.n + 20:
        return None
    pos = np.full(p.m, -1, dtype=int)
    pos[ix] = np.arange(len(ix))
    B = np.zeros((len(ix), p.n))
    use = active[ar]
    np.add.at(B, (pos[ar[use]], ac[use]), ad[use])
    P = np.zeros((p.n, p.n))
    np.add.at(P, (pr, pc), pd)
    K = np.block([[P, B.T], [B, np.zeros((len(ix), len(ix)))]])
    rhs = np.r_[-p.q, p.b[ix]]
    try:
        y = np.linalg.solve(K, rhs)
        zn = np.zeros(p.m)
        zn[ix] = y[p.n:]
        rn = p.result(y[:p.n], zn)
        if p.valid(rn):
            return rn
    except np.linalg.LinAlgError:
        pass
    return None


def diagonal_box(p):
    pd, pr, pp, pc = p.P
    if (len(pd) != p.n or np.any(pr != pc) or np.any(pd <= 0)
            or not np.array_equal(pc, np.arange(p.n)) or np.sum(p.eq) > 20):
        return None
    ad, ar, ap, ac = p.A
    degree = np.bincount(ar, minlength=p.m)
    if np.any(degree[~p.eq] != 1):
        return None
    eig = np.flatnonzero(p.eq)
    pos = np.full(p.m, -1, dtype=np.int32)
    pos[eig] = np.arange(len(eig))
    take = p.eq[ar]
    F = np.zeros((len(eig), p.n))
    np.add.at(F, (pos[ar[take]], ac[take]), ad[take])
    sc = np.maximum(np.max(np.abs(F), axis=1, initial=0), 1e-12)
    b = p.b[eig] / sc
    F /= sc[:, None]
    qr, qc, qa = ar[~take], ac[~take], ad[~take]
    bounds = p.b[qr] / qa
    lower, upper = np.full(p.n, -np.inf), np.full(p.n, np.inf)
    positive = qa > 0
    np.minimum.at(upper, qc[positive], bounds[positive])
    np.maximum.at(lower, qc[~positive], bounds[~positive])
    if np.any(lower > upper):
        return None
    diag = np.empty(p.n)
    diag[pc] = pd
    y = np.zeros(len(eig))
    for it in range(70):
        c = p.q + F.T @ y
        raw = -c / diag
        x = np.clip(raw, lower, upper)
        g = F @ x - b
        z = np.zeros(p.m)
        z[eig] = y / sc
        station = diag * x + c
        eligible = (np.abs(x[qc] - bounds) < 1e-8 * np.maximum(1, np.abs(bounds)))
        eligible &= -station[qc] / qa >= -1e-10
        cc, rr, aa = qc[eligible], qr[eligible], qa[eligible]
        _, first = np.unique(cc, return_index=True)
        z[rr[first]] = np.maximum(0, -station[cc[first]] / aa[first])
        r = p.result(x, z)
        if p.valid(r):
            return r
        if not len(eig):
            return None
        free = (raw >= lower - 1e-12) & (raw <= upper + 1e-12) & (upper > lower)
        H = (F * (free / diag)[None, :]) @ F.T
        try:
            dy = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None
        obj = float(np.sum((0.5 * diag * x + p.q) * x) + y @ g)
        slope = float(g @ dy)
        alpha = 1.0
        for _ in range(40):
            yn = y + alpha * dy
            cn = p.q + F.T @ yn
            xn = np.clip(-cn / diag, lower, upper)
            gn = F @ xn - b
            objn = float(np.sum((0.5 * diag * xn + p.q) * xn) + yn @ gn)
            if objn >= obj + 1e-4 * alpha * slope - 1e-10 * max(1, abs(obj)):
                break
            alpha *= 0.5
        y = yn
    return None


def unconstrained(p):
    if len(p.A[0]) or p.n > 650 or not len(p.P[0]):
        return None
    pd, pr, pp, pc = p.P
    h, qscale = norm(pd), norm(p.q) or 1.
    H = np.zeros((p.n, p.n))
    np.add.at(H, (pr, pc), pd / h)
    H = .5 * (H + H.T)
    eigen, U = np.linalg.eigh(H)
    q = p.q / qscale
    projected = U.T @ q
    small = eigen <= max(np.max(eigen, initial=0), 1.) * 1e-13
    if np.any(small):
        x = -(U[:, small] @ projected[small])
        ray = {"status": "dual_infeasible", "x": x, "s": np.zeros(p.m)}
        if p.valid(ray):
            return ray
    use = ~small
    xhat = -(U[:, use] @ (projected[use] / eigen[use]))
    with np.errstate(over="ignore", invalid="ignore"):
        x = (xhat.astype(np.longdouble) * np.longdouble(qscale) /
             np.longdouble(h)).astype(np.float64)
    r = p.result(x, np.zeros(p.m))
    return r if p.valid(r) else None


def direct_cases(p):
    if norm(p.q) == 0:
        r = p.result(np.zeros(p.n), np.zeros(p.m))
        if p.valid(r):
            return r
    eq = np.flatnonzero(p.eq)
    if p.n and len(eq) >= p.n and p.n <= 650 and len(eq) * p.n <= 1000000:
        ad, ar, ap, ac = p.A
        pos = np.full(p.m, -1, dtype=np.int32)
        pos[eq] = np.arange(len(eq))
        F = np.zeros((len(eq), p.n))
        use = p.eq[ar]
        np.add.at(F, (pos[ar[use]], ac[use]), ad[use])
        x = np.linalg.lstsq(F, p.b[eq], rcond=1e-13)[0]
        z = np.zeros(p.m)
        z[eq] = np.linalg.lstsq(F.T, -p.q - p.mul("P", x), rcond=1e-13)[0]
        r = p.result(x, z)
        if p.valid(r):
            return r
        z[:] = 0
        z[eq] = F @ x - p.b[eq]
        r = {"status": "primal_infeasible", "z": z}
        if p.valid(r):
            return r
    if len(p.A[0]):
        return None
    for t, d, lo, hi in p.blocks:
        b = p.b[lo:hi]
        v = np.zeros(hi - lo)
        if t == "zero" and norm(b):
            j = int(np.argmax(np.abs(b)))
            v[j] = -np.sign(b[j])
        elif t == "nonnegative" and np.min(b) < 0:
            v[int(np.argmin(b))] = 1
        elif t == "soc" and b[0] < np.linalg.norm(b[1:]):
            length = np.linalg.norm(b[1:])
            v[0] = 1
            if length:
                v[1:] = -b[1:] / length
        elif t == "exponential":
            a, y, w = b / max(1., norm(b))
            if y < 0:
                v[1] = 1
            elif w < 0:
                v[2] = 1
            else:
                theta = a / y if y > 0 else (max(10., math.log(max(w, 1e-300) / max(a, 1e-300)) + 2) if a > 0 else 0.)
                theta = float(np.clip(theta, -700., 700.))
                if theta <= 0:
                    ev = math.exp(theta)
                    v[:] = -ev, (theta - 1) * ev, 1.
                else:
                    v[:] = -1., theta - 1, math.exp(-theta)
        elif t == "psd":
            c = np.repeat(np.arange(d), np.arange(1, d + 1))
            rr = np.arange(hi - lo) - c * (c + 1) // 2
            scale = np.where(rr == c, 1., math.sqrt(2.))
            M = np.zeros((d, d))
            value = (b / max(1., norm(b))) / scale
            M[rr, c], M[c, rr] = value, value
            eigen, U = np.linalg.eigh(M)
            if eigen[0] < 0:
                v = scale * U[rr, 0] * U[c, 0]
        if norm(v):
            z = np.zeros(p.m)
            z[lo:hi] = v
            r = {"status": "primal_infeasible", "z": z}
            if p.valid(r):
                return r
    return None


def solve(p):
    mod = core()
    direct = direct_cases(p)
    if direct is not None:
        return direct
    if len(p.P[0]) == 0:
        a, bscale, qscale = norm(p.A[0]) or 1., norm(p.b) or 1., norm(p.q) or 1.
        if (a < 1e-8 or a > 1e8 or bscale < 1e-8 or bscale > 1e8) and not hasattr(p, "_global_scaled"):
            scaled = object.__new__(Raw)
            scaled.__dict__.update(p.__dict__)
            scaled._global_scaled = True
            scaled.q, scaled.b = p.q / qscale, p.b / bscale
            scaled.A = p.A[0] / a, p.A[1], p.A[2], p.A[3]
            answer = solve(scaled)
            if answer is not None:
                if answer["status"] == "solved":
                    with np.errstate(over="ignore", invalid="ignore"):
                        x = (answer["x"].astype(np.longdouble) * np.longdouble(bscale) /
                             np.longdouble(a)).astype(np.float64)
                        z = (answer["z"].astype(np.longdouble) * np.longdouble(qscale) /
                             np.longdouble(a)).astype(np.float64)
                    answer = p.result(x, z)
                elif answer["status"] == "dual_infeasible":
                    answer = dict(answer, s=-p.mul("A", answer["x"]))
                if p.valid(answer):
                    return answer
            return None
    if (os.environ.get("SOLVER_NATIVE_IPM") == "1" or
            (len(p.P[0]) == 0 and p.n > 8000 and p.n < 100000 and p.m < 1.6 * p.n)):
        from fastsoc import solve as solve_ipm
        result = solve_ipm(p, maxiter=60)
        if result is not None:
            return result
    if len(p.P[0]):
        r = unconstrained(p)
        if r is not None:
            return r
        r = diagonal_box(p)
        if r is not None:
            return r
        if p.n >= 1500 and len(p.P[0]) <= 400:
            from lowrankqp import solve as solve_lowrank
            r = solve_lowrank(p)
            if r is not None:
                return r
    h = engine()
    if len(p.P[0]):
        h.setOptionValue("time_limit", 0.12 if p.n < 2000 else 0.3)
    elif norm(p.q) == 0 and not os.environ.get("SOLVER_LP_METHOD"):
        h.setOptionValue("solver", "ipm")
        h.setOptionValue("run_crossover", "off")
    ad, ar, ap, ac = p.A
    costscale = 1. if len(p.P[0]) or os.environ.get("SOLVER_OBJECTIVE_SCALE") == "0" else 1 / max(1., norm(p.q))
    pass_raw(h, p.n, p.m, p.q * costscale, np.where(p.eq, p.b, -mod.kHighsInf),
             p.b, ad, ar, ap, P=p.P)
    h.run()
    st = h.getModelStatus()
    sol = h.getSolution()
    log("fast highs", st)
    if len(sol.col_value) == p.n and len(sol.row_dual) == p.m:
        x, z = np.asarray(sol.col_value), -np.asarray(sol.row_dual) / costscale
        if len(p.P[0]) == 0 and norm(p.q) == 0:
            z[:] = 0
        z[~p.eq] = np.maximum(z[~p.eq], 0)
        r = p.result(x, z)
        if p.valid(r):
            return r
        if st == mod.HighsModelStatus.kOptimal and len(p.P[0]):
            r = dense_polish(p, x, z)
            if r is not None:
                return r
    if st in (mod.HighsModelStatus.kInfeasible, mod.HighsModelStatus.kUnknown,
              mod.HighsModelStatus.kUnboundedOrInfeasible):
        r = phase1(p)
        if r is not None:
            return r
    if st in (mod.HighsModelStatus.kUnbounded, mod.HighsModelStatus.kUnboundedOrInfeasible):
        r = recession(p)
        if r is not None:
            return r
    if len(p.P[0]) and p.n <= 650:
        from denseqp import solve_dense
        r = solve_dense(p)
        if r is not None:
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = None
    try:
        with np.load(args.input, allow_pickle=False) as f:
            meta = json.loads(str(f["metadata_json"]))
            shape = f["P_shape"]
            if shape[0] <= 100000 and all(c["type"] in ("zero", "nonnegative")
                                         for c in meta["cones"]):
                p = Raw(f, meta)
                result = solve(p)
            elif (shape[0] <= 150000 and any(c["type"] == "exponential" for c in meta["cones"])
                  and all(c["type"] in ("zero", "nonnegative", "exponential")
                          for c in meta["cones"])):
                from fastgp import solve as solve_gp
                p = Raw(f, meta)
                result = direct_cases(p)
                if result is None:
                    result = solve_gp(p)
            elif shape[0] <= 200 and all(c["type"] == "psd" for c in meta["cones"]):
                from fastpsd import solve as solve_psd
                p = Raw(f, meta)
                result = direct_cases(p)
                if result is None:
                    result = solve_psd(p)
            elif (any(c["type"] == "soc" for c in meta["cones"]) and
                  all(c["type"] in ("zero", "nonnegative", "soc") for c in meta["cones"])):
                from fastsoc import solve as solve_soc
                p = Raw(f, meta)
                from fastpresolve import reduce, recover
                result = direct_cases(p)
                if result is not None:
                    reduced, stages = p, []
                else:
                    reduced, stages = reduce(p)
                recovered = [None]
                if stages:
                    def postcheck(r):
                        original = recover(r, stages)
                        if p.valid(original, tol=8.8e-9):
                            recovered[0] = original
                            return True
                        if original["status"] == "solved":
                            from refine import dual
                            polished = dual(p, original)
                            if polished is not None:
                                recovered[0] = polished
                                return True
                        return False
                    reduced._postcheck = postcheck
                if result is None:
                    result = solve_soc(reduced)
                    if result is not None:
                        result = recovered[0] if recovered[0] is not None else recover(result, stages)
            elif shape[0] <= 650:
                p = Raw(f, meta)
                result = direct_cases(p)
    except Exception as error:
        log("fast fallback", type(error).__name__, str(error))
    if result is None:
        if remaining() < 1.:
            result = {"status": "unresolved"}
        else:
            from solver import main as fallback
            fallback()
            return
    arrays = {k: np.asarray(result[k], dtype=np.float64)
              for k in ("x", "s", "z") if k in result}
    arrays["metadata_json"] = np.array(json.dumps(
        {"schema_version": 1, "status": result["status"]}, allow_nan=False))
    with open(args.output, "wb") as f:
        np.savez(f, **arrays)


if __name__ == "__main__":
    main()
