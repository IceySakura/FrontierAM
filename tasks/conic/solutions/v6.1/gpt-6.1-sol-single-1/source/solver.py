#!/usr/bin/env python3
"""Offline convex quadratic/conic optimization."""

import argparse
import json
import math
import os
import sys
import time
import warnings
import importlib.util
import glob

import numpy as np
import scipy
from scipy import sparse, linalg

# Loading the installed extension directly avoids importing unrelated optimizers.
if "_core" in sys.modules:
    highs = sys.modules["_core"]
else:
    _spec = importlib.util.spec_from_file_location(
        "_core", glob.glob(scipy.__path__[0] + "/optimize/_highspy/_core*.so")[0])
    highs = importlib.util.module_from_spec(_spec)
    sys.modules["_core"] = highs
    _spec.loader.exec_module(highs)
sys.modules.setdefault("solver", sys.modules[__name__])


START = time.monotonic()
VERBOSE = bool(os.environ.get("SOLVER_VERBOSE"))
INF = highs.kHighsInf
SQRT2 = math.sqrt(2.0)


def log(*args):
    if VERBOSE:
        print(*args, file=sys.stderr, flush=True)


def norm(x):
    return float(np.max(np.abs(x), initial=0.0))


class Problem:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as d:
            self.q = d["q"]
            self.b = d["b"]
            self.meta = json.loads(str(d["metadata_json"]))
            for k in ("P", "A"):
                mat = sparse.csc_matrix(
                    (d[k + "_data"], d[k + "_indices"], d[k + "_indptr"]),
                    shape=tuple(d[k + "_shape"]),
                )
                mat.sum_duplicates()
                mat.eliminate_zeros()
                mat.sort_indices()
                setattr(self, k, mat)
        self.n, self.m = self.q.size, self.b.size
        self.blocks = []
        self.kind = np.empty(self.m, dtype=np.int8)
        pos = 0
        for c in self.meta["cones"]:
            t, d = c["type"], c["dim"]
            w = d * (d + 1) // 2 if t == "psd" else d
            self.blocks.append((t, d, pos, pos + w))
            self.kind[pos:pos + w] = {
                "zero": 0, "nonnegative": 1, "soc": 2,
                "exponential": 3, "psd": 4,
            }[t]
            pos += w
        self.linear = np.flatnonzero(self.kind < 2)
        self.eq = np.flatnonzero(self.kind == 0)
        self.ineq = np.flatnonzero(self.kind == 1)
        self.nonlinear = [v for v in self.blocks if v[0] not in ("zero", "nonnegative")]
        self.Ar = self.A.tocsr()


def smat(v, d):
    c = np.repeat(np.arange(d), np.arange(1, d + 1))
    r = np.arange(len(v)) - c * (c + 1) // 2
    M = np.zeros((d, d))
    vals = v.copy()
    vals[r != c] /= SQRT2
    M[r, c] = vals
    M[c, r] = vals
    return M


def svec(M):
    d = len(M)
    c = np.repeat(np.arange(d), np.arange(1, d + 1))
    r = np.arange(len(c)) - c * (c + 1) // 2
    v = M[r, c].copy()
    v[r != c] *= SQRT2
    return v


def exp_defect(a, b, c):
    boundary = max(abs(b), a, -c, 0.0)
    if b > 0 and c > 0:
        return min(boundary, max(0.0, a - b * (math.log(c) - math.log(b))))
    return boundary


def cone_defect(p, v, dual=False):
    worst = 0.0
    if not hasattr(p, "_cone_groups"):
        groups = {}
        psds = []
        for t, d, lo, hi in p.blocks:
            if t == "psd":
                psds.append((d, lo, hi))
            elif t != "zero":
                groups.setdefault((t, d), []).append((lo, hi))
        indexed = []
        for (t, d), ranges in groups.items():
            if t == "nonnegative" and len(ranges) < 8:
                indexed.append((t, None, ranges))
            else:
                ix = np.array([lo for lo, hi in ranges])[:, None] + np.arange(d)
                indexed.append((t, ix, None))
        p._cone_groups = indexed, psds
    if not dual and len(p.eq):
        zero = norm(v[p.eq])
        worst = zero / max(1.0, zero)
    indexed, psds = p._cone_groups
    for t, ix, ranges in indexed:
        if ix is None:
            for lo, hi in ranges:
                val = max(0.0, -float(np.min(v[lo:hi]))) / max(1.0, norm(v[lo:hi]))
                worst = max(worst, val)
            continue
        u = v[ix]
        u = u / np.maximum(1.0, np.max(np.abs(u), axis=1))[:, None]
        if t == "nonnegative":
            val = max(0.0, -float(np.min(u)))
        elif t == "soc":
            val = max(0.0, float(np.max(np.linalg.norm(u[:, 1:], axis=1) - u[:, 0])))
        else:
            if dual:
                a, b, c = -u[:, 1], -u[:, 0], math.e * u[:, 2]
            else:
                a, b, c = u.T
            defect = np.maximum.reduce((np.abs(b), a, -c, np.zeros(len(a))))
            good = (b > 0) & (c > 0)
            interior = np.maximum(0.0, a[good] - b[good] * (np.log(c[good]) - np.log(b[good])))
            defect[good] = np.minimum(defect[good], interior)
            val = float(np.max(defect, initial=0.0))
        worst = max(worst, val)
    for d, lo, hi in psds:
        u = v[lo:hi] / max(1.0, norm(v[lo:hi]))
        val = max(0.0, -float(linalg.eigh(
            smat(u, d), subset_by_index=[0, 0],
            eigvals_only=True, check_finite=False)[0]))
        worst = max(worst, val)
    return worst


def residuals(p, result):
    st = result["status"]
    if st == "solved":
        x, s, z = (result[k] for k in ("x", "s", "z"))
        ax, px, az = p.A @ x, p.P @ x, p.A.T @ z
        rp = norm(ax + s - p.b) / (1 + max(norm(ax), norm(s), norm(p.b)))
        rd = norm(px + p.q + az) / (1 + max(norm(px), norm(p.q), norm(az)))
        objp, objd = 0.5 * (x @ px) + p.q @ x, -0.5 * (x @ px) - p.b @ z
        gap = abs(objp - objd) / max(1.0, abs(objp), abs(objd))
        return [rp, rd, gap, cone_defect(p, s), cone_defect(p, z, True)]
    if st == "primal_infeasible":
        z = result["z"]
        if not norm(z) or not norm(p.b):
            return [INF]
        z = z / norm(z)
        beta = -(p.b / norm(p.b)) @ z
        if beta <= 1e-8:
            return [INF]
        As = p.A.copy()
        As.data /= norm(p.A.data) or 1.
        return [norm(As.T @ z) / beta,
                cone_defect(p, z, True) / beta]
    if st == "dual_infeasible":
        x, s = result["x"], result["s"]
        if not norm(x) or not norm(p.q):
            return [INF]
        scale = norm(x)
        x, s = x / scale, s / scale
        alpha = -(p.q / norm(p.q)) @ x
        if alpha <= 1e-8:
            return [INF]
        a = norm(p.A.data) or 1.0
        Ps, As = p.P.copy(), p.A.copy()
        Ps.data /= norm(p.P.data) or 1.
        As.data /= a
        return [norm(Ps @ x) / alpha,
                norm(As @ x + s / a) / alpha, cone_defect(p, s / a) / alpha]
    return [INF]


def accepted(p, r, tol=8e-9):
    try:
        vals = residuals(p, r)
        return all(math.isfinite(v) and v <= tol for v in vals)
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        return False


def feasible_slack(p, x):
    s = p.b - p.A @ x
    s[p.eq] = 0.0
    s[p.ineq] = np.maximum(s[p.ineq], 0.0)
    return s


def polish_dual(p, result):
    from scipy.sparse.linalg import lsmr
    if len(p.eq) == 0 or result["status"] != "solved":
        return result
    r = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in result.items()}
    F = p.A[p.eq].tocsc()
    scaling = np.sqrt(np.asarray(F.multiply(F).sum(axis=1)).ravel())
    scaling = 1 / np.maximum(scaling, 1e-100)
    FT = (sparse.diags(scaling) @ F).T.tocsc()
    for _ in range(2):
        rd = p.P @ r["x"] + p.q + p.A.T @ r["z"]
        dz = lsmr(FT, -rd, atol=1e-14, btol=1e-14,
                  maxiter=min(2000, 3 * len(p.eq) + 20))[0]
        r["z"][p.eq] += scaling * dz
        if accepted(p, r, tol=9e-9):
            return r
    return result


def new_highs():
    h = highs._Highs()
    from fast import remaining
    for k, v in {
        "output_flag": False, "threads": 1, "parallel": "off",
        "primal_feasibility_tolerance": 1e-9,
        "dual_feasibility_tolerance": 1e-9,
        "ipm_optimality_tolerance": 1e-11,
        "small_matrix_value": 1e-12,
        "large_matrix_value": 1e25,
        "time_limit": min(270., max(.01, remaining() - 1.)),
    }.items():
        h.setOptionValue(k, v)
    return h


def pass_model(h, A, q, lower, upper, P=None, colbound=None):
    A = A.tocsc()
    n, m = A.shape[1], A.shape[0]
    lp = highs.HighsLp()
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = q
    lp.col_lower_ = np.full(n, -INF) if colbound is None else colbound[0]
    lp.col_upper_ = np.full(n, INF) if colbound is None else colbound[1]
    lp.row_lower_, lp.row_upper_ = lower, upper
    lp.a_matrix_.format_ = highs.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, m
    lp.a_matrix_.start_ = A.indptr.astype(np.int32, copy=False)
    lp.a_matrix_.index_ = A.indices.astype(np.int32, copy=False)
    lp.a_matrix_.value_ = A.data
    if P is None or P.nnz == 0:
        h.passModel(lp)
    else:
        model = highs.HighsModel()
        model.lp_ = lp
        L = sparse.tril(P, format="csc")
        model.hessian_.dim_ = n
        model.hessian_.format_ = highs.HessianFormat.kTriangular
        model.hessian_.start_ = L.indptr.astype(np.int32, copy=False)
        model.hessian_.index_ = L.indices.astype(np.int32, copy=False)
        model.hessian_.value_ = L.data
        h.passModel(model)


def phase1(A, b, eq):
    """A one-artificial-variable LP yields a Farkas vector on failure."""
    m, n = A.shape
    e = np.flatnonzero(eq)
    I = sparse.csc_matrix(-np.ones((m, 1)))
    C = sparse.vstack((
        sparse.hstack((A, I)),
        sparse.hstack((-A[e], sparse.csc_matrix(-np.ones((len(e), 1))))),
    ), format="csc")
    rhs = np.r_[b, -b[e]]
    h = new_highs()
    q = np.zeros(n + 1)
    q[-1] = 1.0
    lb, ub = np.full(n + 1, -INF), np.full(n + 1, INF)
    lb[-1] = 0.0
    pass_model(h, C, q, np.full(len(rhs), -INF), rhs, colbound=(lb, ub))
    h.run()
    sol = h.getSolution()
    if len(sol.col_value) != n + 1:
        return None
    if sol.col_value[-1] <= 1e-10:
        return None
    dual = -np.asarray(sol.row_dual)
    z = dual[:m].copy()
    z[e] -= dual[m:]
    return z


def recession(p, C, d, eq):
    n = p.n
    h = new_highs()
    if p.P.nnz:
        sc = norm(p.P.data) or 1.0
        C = sparse.vstack((C, p.P / sc), format="csc")
        eq = np.r_[eq, np.ones(n, dtype=bool)]
    upper = np.zeros(C.shape[0])
    lower = np.where(eq, 0.0, -INF)
    pass_model(h, C, p.q / (norm(p.q) or 1), lower, upper,
               colbound=(np.full(n, -1.0), np.ones(n)))
    h.run()
    sol = h.getSolution()
    if len(sol.col_value) == n:
        x = np.asarray(sol.col_value)
        r = {"status": "dual_infeasible", "x": x, "s": -(p.A @ x)}
        if accepted(p, r):
            return r
    return None


def polish_qp(p, x, z, max_iter=4):
    """Refine the active face in the original, unregularized KKT system."""
    from scipy.sparse.linalg import splu, lsmr
    s = p.b - p.A @ x
    for _ in range(max_iter):
        active = (p.kind == 0) | ((p.kind == 1) &
            ((z > 1e-8 * max(1.0, norm(z))) | (np.abs(s) < 1e-8)))
        ii = np.flatnonzero(active)
        B = p.A[ii]
        K = sparse.bmat([[p.P, B.T], [B, None]], format="csc")
        rhs = np.r_[-p.q, p.b[ii]]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lu = splu(K, permc_spec="COLAMD")
                v = lu.solve(rhs)
                for _ in range(2):
                    v += lu.solve(rhs - K @ v)
        except (RuntimeError, ValueError):
            if p.n + len(ii) > 20000:
                break
            v = lsmr(K, rhs, atol=1e-13, btol=1e-13,
                     maxiter=min(2000, 5 * len(rhs)))[0]
        xn, zn = v[:p.n], np.zeros(p.m)
        zn[ii] = v[p.n:]
        sn = p.b - p.A @ xn
        rn = {"status": "solved", "x": xn, "s": feasible_slack(p, xn), "z": zn}
        if accepted(p, rn):
            return rn
        if norm(xn - x) > 10 * max(1.0, norm(x)):
            break
        x, z, s = xn, zn, sn
    return None


def solve_linear(p):
    if p.P.nnz and p.n < 100000:
        from boxqp import solve_box
        rn = solve_box(p)
        if rn is not None:
            return rn
        if os.environ.get("SOLVER_QP_ENGINE") == "ipm":
            from ipm import solve_qp
            rn = solve_qp(p)
            if rn is not None:
                return rn
    if p.n > 100000 and p.P.nnz:
        from largeqp import zero_loss, solve_lasso
        rn = zero_loss(p)
        if rn is not None:
            return rn
        rn = solve_lasso(p)
        if rn is not None:
            return rn
        return {"status": "unresolved"}
    h = new_highs()
    lower = np.where(p.kind == 0, p.b, -INF)
    pass_model(h, p.A, p.q, lower, p.b, p.P)
    h.run()
    st = h.getModelStatus()
    sol = h.getSolution()
    log("highs", st, "time", time.monotonic() - START)
    if len(sol.col_value) == p.n and len(sol.row_dual) == p.m:
        x, z = np.asarray(sol.col_value), -np.asarray(sol.row_dual)
        r = {"status": "solved", "x": x, "s": feasible_slack(p, x), "z": z}
        vals = residuals(p, r)
        log("residuals", vals)
        if max(vals) < 8e-9:
            return r
        if st == highs.HighsModelStatus.kOptimal and p.P.nnz:
            rn = polish_qp(p, x, z)
            if rn is not None:
                return rn
    if st in (highs.HighsModelStatus.kInfeasible,
              highs.HighsModelStatus.kUnboundedOrInfeasible,
              highs.HighsModelStatus.kUnknown):
        z = phase1(p.A, p.b, p.kind == 0)
        if z is not None:
            r = {"status": "primal_infeasible", "z": z}
            log("phase1", residuals(p, r))
            if accepted(p, r):
                return r
    if st in (highs.HighsModelStatus.kUnbounded,
              highs.HighsModelStatus.kUnboundedOrInfeasible):
        rn = recession(p, p.A, p.b, p.kind == 0)
        if rn is not None:
            return rn
    if p.P.nnz and p.n < 100000:
        from ipm import solve_qp
        rn = solve_qp(p)
        if rn is not None:
            return rn
    return {"status": "unresolved"}


def soc_cut(v):
    v = v / max(norm(v), 1e-300)
    a = np.linalg.norm(v[1:])
    if a == 0:
        return np.r_[1.0, np.zeros(len(v) - 1)]
    return np.r_[1.0, -v[1:] / a]


def exp_cut(t):
    if t <= 0:
        e = math.exp(max(-700, t))
        return np.array([-e, (t - 1) * e, 1.0])
    return np.array([-1.0, t - 1.0, math.exp(-min(t, 700))])


class Outer:
    def __init__(self, p):
        self.p = p
        self.base = p.linear
        self.maps = []
        self.h = new_highs()
        C = p.A[self.base]
        self.C = C.tocsr()
        self.d = p.b[self.base].copy()
        self.eq = p.kind[self.base] == 0
        pass_model(self.h, C, p.q, np.where(self.eq, self.d, -INF), self.d, p.P)

    def add(self, cuts):
        p = self.p
        rr, cc, vv = [], [], []
        for i, (lo, vec) in enumerate(cuts):
            nz = np.flatnonzero(vec)
            rr.extend([i] * len(nz))
            cc.extend(lo + nz)
            vv.extend(vec[nz])
            self.maps.append((lo + nz, vec[nz].copy()))
        if not cuts:
            return
        T = sparse.csr_matrix((vv, (rr, cc)), shape=(len(cuts), p.m))
        C = (T @ p.A).tocsr()
        d = T @ p.b
        self.h.addRows(len(cuts), np.full(len(cuts), -INF), d,
                       C.nnz, C.indptr.astype(np.int32),
                       C.indices.astype(np.int32), C.data)
        self.C = sparse.vstack((self.C, C), format="csr")
        self.d = np.r_[self.d, d]
        self.eq = np.r_[self.eq, np.zeros(len(cuts), dtype=bool)]

    def recover(self, dual):
        z = np.zeros(self.p.m)
        z[self.base] = dual[:len(self.base)]
        for v, (ix, w) in zip(dual[len(self.base):], self.maps):
            if v != 0:
                z[ix] += max(v, 0.0) * w
        return z

    def initial(self):
        cuts = []
        for t, d, lo, hi in self.p.nonlinear:
            w = hi - lo
            if t == "soc":
                v = np.zeros(w)
                v[0] = 1
                cuts.append((lo, v.copy()))
                for j in range(1, w):
                    v[j] = 1
                    cuts.append((lo, v.copy()))
                    v[j] = -1
                    cuts.append((lo, v.copy()))
                    v[j] = 0
            elif t == "exponential":
                cuts.extend([(lo, np.array([0., 1., 0.])),
                             (lo, np.array([0., 0., 1.]))])
                for val in (-8., -4., -2., -1., 0., 1., 2., 4., 8.):
                    cuts.append((lo, exp_cut(val)))
            else:
                for j in range(d):
                    v = np.zeros(w)
                    v[j * (j + 3) // 2] = 1
                    cuts.append((lo, v))
                ev, U = linalg.eigh(smat(self.p.b[lo:hi], d), check_finite=False)
                for j in np.flatnonzero(ev < -1e-8)[:min(30, d)]:
                    cuts.append((lo, svec(np.outer(U[:, j], U[:, j]))))
        self.add(cuts)

    def separating(self, s, tol=2e-9):
        cuts = []
        for t, d, lo, hi in self.p.nonlinear:
            v = s[lo:hi]
            scale = max(1.0, norm(v))
            if t == "soc":
                if np.linalg.norm(v[1:]) - v[0] > tol * scale:
                    cuts.append((lo, soc_cut(v)))
            elif t == "exponential":
                a, b, c = v
                if exp_defect(*(v / scale)) > tol:
                    if b > 1e-12 * scale:
                        val = np.clip(a / b, -300, 300)
                    elif a > 0:
                        val = min(100, max(10, math.log(max(abs(c), 1) / max(a, 1e-12)) + 2))
                    else:
                        val = -30.
                    cuts.append((lo, exp_cut(float(val))))
            else:
                ev, U = linalg.eigh(smat(v / scale, d), check_finite=False)
                for j in np.flatnonzero(ev < -tol)[:min(20, d)]:
                    cuts.append((lo, svec(np.outer(U[:, j], U[:, j]))))
        return cuts

    def solve(self):
        from fast import remaining
        self.initial()
        p = self.p
        bounded = False
        for it in range(250):
            if remaining() < 1.:
                break
            self.h.run()
            st = self.h.getModelStatus()
            sol = self.h.getSolution()
            if st == highs.HighsModelStatus.kInfeasible:
                dual = phase1(self.C, self.d, self.eq)
                if dual is not None:
                    r = {"status": "primal_infeasible", "z": self.recover(dual)}
                    if accepted(p, r):
                        return r
                break
            if st in (highs.HighsModelStatus.kUnbounded,
                      highs.HighsModelStatus.kUnboundedOrInfeasible):
                rn = recession(p, self.C, self.d, self.eq)
                if rn is not None:
                    return rn
                if not bounded:
                    radius = 1e6 * max(1.0, norm(p.b))
                    self.h.changeColsBounds(p.n, np.arange(p.n, dtype=np.int32),
                                            np.full(p.n, -radius), np.full(p.n, radius))
                    bounded = True
                    continue
                break
            if len(sol.col_value) != p.n or len(sol.row_dual) != len(self.d):
                break
            x = np.asarray(sol.col_value)
            s = p.b - p.A @ x
            z = self.recover(-np.asarray(sol.row_dual))
            r = {"status": "solved", "x": x, "s": s, "z": z}
            vals = residuals(p, r)
            if it % 5 == 0:
                log("outer", it, vals, "cuts", len(self.maps), "t", time.monotonic() - START)
            if max(vals) < 7e-9:
                return r
            cuts = self.separating(s, tol=5e-10)
            if not cuts:
                break
            self.add(cuts)
        return {"status": "unresolved"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {"status": "unresolved"}
    try:
        from fast import remaining
        if remaining() < 1.:
            raise RuntimeError("solve deadline")
        p = Problem(args.input)
        log("loaded", p.n, p.m, "t", time.monotonic() - START)
        original = p
        from presolve import reduce_problem
        p, stages = reduce_problem(p)
        if not p.nonlinear:
            result = solve_linear(p)
        else:
            result = None
            if not any(v[0] == "psd" for v in p.nonlinear):
                if all(v[0] == "exponential" for v in p.nonlinear):
                    from gp import solve_gp
                    result = solve_gp(p)
                if result is None:
                    from coneipm import solve_conic
                    result = solve_conic(p)
            else:
                from psdipm import solve_psd
                result = solve_psd(p)
            if result is None:
                result = Outer(p).solve()
        for stage in reversed(stages):
            result = stage.recover(result)
        if result["status"] != "unresolved":
            vals = residuals(original, result)
            if result["status"] == "solved" and max(vals) > 8e-9:
                result = polish_dual(original, result)
                vals = residuals(original, result)
            log("original residuals", vals)
            if not all(math.isfinite(v) and v < 9.5e-9 for v in vals):
                result = {"status": "unresolved"}
    except Exception as e:
        log(type(e).__name__, str(e))
        if VERBOSE:
            import traceback
            traceback.print_exc()
    meta = {"schema_version": 1, "status": result["status"]}
    arrays = {k: np.asarray(result[k], dtype=np.float64)
              for k in ("x", "s", "z") if k in result}
    arrays["metadata_json"] = np.array(json.dumps(meta, allow_nan=False))
    with open(args.output, "wb") as f:
        np.savez(f, **arrays)
    log("done", result["status"], time.monotonic() - START)


if __name__ == "__main__":
    main()
