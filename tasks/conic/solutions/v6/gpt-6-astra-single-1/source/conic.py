"""Conic outer approximation with dual-preserving supporting hyperplanes."""

import math
import os
import time

import numpy as np
from scipy import linalg, sparse
from scipy.optimize._highspy import _core as hc

from linear import certificate, highs_model, polish as polish_linear
from problem import Problem, blocks
from quality import acceptable, cone_error, norm, smat, svec, residuals
from scaling import equilibrate, recover


class Cuts:
    def __init__(self, p):
        self.p = p
        self.rows = []
        self.cols = []
        self.vals = []
        self.equal = []

    def add(self, start, vector, equality=False):
        vec = np.asarray(vector)
        keep = np.flatnonzero(np.abs(vec) > 1e-15 * max(1.0, norm(vec)))
        row = len(self.equal)
        self.rows.extend([row] * keep.size)
        self.cols.extend(keep + start)
        self.vals.extend(vec[keep])
        self.equal.append(equality)

    def coordinate(self, index, equality=False):
        self.rows.append(len(self.equal))
        self.cols.append(index)
        self.vals.append(1.0)
        self.equal.append(equality)

    def matrix(self):
        mat = sparse.csr_matrix((self.vals, (self.rows, self.cols)),
                                shape=(len(self.equal), self.p.m))
        return mat, np.asarray(self.equal, dtype=bool)


def initial_cuts(p, warm=None):
    cuts = Cuts(p)
    for kind, dim, sl in blocks(p.cones):
        if kind in ("zero", "nonnegative"):
            for i in range(sl.start, sl.stop):
                cuts.coordinate(i, kind == "zero")
        elif kind == "soc":
            cuts.coordinate(sl.start)
            for j in range(1, dim):
                for sign in (-1.0, 1.0):
                    v = np.zeros(dim)
                    v[0], v[j] = 1.0, sign
                    cuts.add(sl.start, v)
            if warm is not None:
                v = warm["s"][sl][1:]
                length = np.linalg.norm(v)
                if length > 1e-12:
                    u = v / length
                    cuts.add(sl.start, np.r_[1.0, -u])
                    for j in range(dim - 1):
                        for delta in (-0.01, 0.01):
                            w = u.copy()
                            w[j] += delta
                            cuts.add(sl.start, np.r_[1.0, -w / np.linalg.norm(w)])
        elif kind == "exponential":
            cuts.coordinate(sl.start + 1)
            cuts.coordinate(sl.start + 2)
            for r in (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0):
                v = np.array([-1.0, r - 1.0, math.exp(-r)])
                cuts.add(sl.start, v)
        else:
            for i in range(dim):
                cuts.coordinate(sl.start + i * (i + 1) // 2 + i)
    return cuts.matrix()


def separating_cuts(p, slack, tol=1e-9):
    cuts = Cuts(p)
    worst = 0.0
    for kind, dim, sl in blocks(p.cones):
        if kind in ("zero", "nonnegative"):
            continue
        s = slack[sl]
        scale = max(1.0, norm(s))
        u = s / scale
        if kind == "soc":
            length = np.linalg.norm(u[1:])
            defect = length - u[0]
            worst = max(worst, defect)
            if defect > tol and length > 0:
                cuts.add(sl.start, np.r_[1.0, -u[1:] / length])
        elif kind == "exponential":
            a, b, c = u
            boundary = max(abs(b), a, -c, 0.0)
            if b > 0 and c > 0:
                ratio = math.log(c) - math.log(b)
                defect = min(boundary, max(0.0, a - b * ratio))
            else:
                defect = boundary
                ratio = np.clip(a / max(1e-16, b), -40.0, 40.0)
            worst = max(worst, defect)
            if defect > tol:
                ratio = float(np.clip(ratio, -700, 700))
                v = np.array([-1.0, ratio - 1.0, math.exp(-ratio)])
                cuts.add(sl.start, v / max(1.0, norm(v) / 1e8))
        else:
            eigenvalues, eigenvectors = linalg.eigh(smat(u, dim), check_finite=False)
            worst = max(worst, -eigenvalues[0])
            negative = np.flatnonzero(eigenvalues < -tol)
            for k in negative[:min(32, max(4, dim // 2))]:
                v = eigenvectors[:, k]
                cuts.add(sl.start, svec(np.outer(v, v)))
    return cuts.matrix()[0], worst


def lift_dual(chunks, multipliers, m):
    z = np.zeros(m)
    start = 0
    for mat in chunks:
        stop = start + mat.shape[0]
        z += mat.T @ multipliers[start:stop]
        start = stop
    return z


def repair_slack(p, s):
    out = s.copy()
    for kind, dim, sl in blocks(p.cones):
        v = out[sl]
        if kind == "zero":
            v[:] = 0
        elif kind == "nonnegative":
            np.maximum(v, 0.0, out=v)
        elif kind == "soc":
            length = np.linalg.norm(v[1:])
            if length > v[0] and length - v[0] < 1e-6 * max(1.0, norm(v)):
                v[0] = length
        elif kind == "exponential":
            a, b, c = v
            if b > 0 and a / b < 700:
                needed = b * math.exp(max(-740.0, a / b))
                if c < needed and needed - c < 1e-6 * max(1.0, norm(v)):
                    v[2] = max(needed * (1.0 + 1e-14), np.nextafter(0.0, 1.0))
    return out


def refine_outer(p, chunks, equality, x, dual):
    T = sparse.vstack(chunks, format="csr")
    eq = np.concatenate(equality)
    changes = np.r_[0, np.flatnonzero(eq[1:] != eq[:-1]) + 1, eq.size]
    cones = [{"type": "zero" if eq[start] else "nonnegative", "dim": int(stop-start)}
             for start, stop in zip(changes[:-1], changes[1:]) if stop > start]
    outer = Problem(p.P, p.q, (T @ p.A).tocsc(), T @ p.b, cones)
    r = {"status": "solved", "x": x, "s": outer.b - outer.A @ x, "z": dual}
    refined = polish_linear(outer, r, eq)
    result = {"status": "solved", "x": refined["x"], "s": p.b - p.A @ refined["x"],
              "z": T.T @ refined["z"]}
    if acceptable(p, result):
        return result
    result["s"] = repair_slack(p, result["s"])
    return result if acceptable(p, result) else None


def solve(p, deadline, warm=None, try_interior=False):
    original = p
    scaled = p.n > 3000
    if scaled:
        p, D, E, cost = equilibrate(p)
        if warm is not None:
            warm = dict(warm, x=warm["x"] / D, s=warm["s"] * E,
                        z=warm["z"] * cost / E)

    def finish(result):
        return recover(result, D, E, cost) if scaled else result

    def check(result):
        return acceptable(original, finish(result))

    T, eq = initial_cuts(p, warm)
    A = (T @ p.A).tocsc()
    b = T @ p.b
    lo = np.where(eq, b, -hc.kHighsInf)
    ipm = not p.P.nnz and p.n > 3000
    if ipm:
        remaining = max(0.1, deadline - time.monotonic())
        engine = highs_model(None, np.zeros(p.n), A, lo, b,
                             seconds=min(45.0, 0.3 * remaining),
                             solver="simplex", tolerance=1e-7)
        if os.environ.get("CONIC_DEBUG"):
            print("OA feasibility", engine.getModelStatus(), file=__import__("sys").stderr)
        if engine.getModelStatus() != hc.HighsModelStatus.kInfeasible:
            engine.changeColsCost(p.n, np.arange(p.n, dtype=np.int32), p.q)
            engine.setOptionValue("simplex_strategy", 4)
            engine.setOptionValue("primal_feasibility_tolerance", 1e-8)
            engine.setOptionValue("dual_feasibility_tolerance", 1e-7)
            budget = max(0.1, min(75.0, 0.45 * (deadline-time.monotonic())))
            engine.setOptionValue("time_limit", engine.getRunTime() + budget)
            engine.run()
    else:
        engine = highs_model(p.P, p.q, A, lo, b,
                             seconds=max(0.1, deadline - time.monotonic()))
    chunks = [T]
    eqs = [eq]
    bounded = False
    bound = 1e4
    started = time.monotonic()
    retried = False
    for iteration in range(300):
        if time.monotonic() >= deadline:
            break
        status = engine.getModelStatus()
        sol = engine.getSolution()
        if os.environ.get("CONIC_DEBUG"):
            print("OA status", iteration, status, "primal", sol.value_valid,
                  "dual", sol.dual_valid, file=__import__("sys").stderr)
        if status in (hc.HighsModelStatus.kUnknown, hc.HighsModelStatus.kSolveError) and not retried:
            engine.setOptionValue("solver", "simplex")
            engine.setOptionValue("primal_feasibility_tolerance", 1e-8)
            engine.setOptionValue("dual_feasibility_tolerance", 1e-8)
            engine.setOptionValue("time_limit", engine.getRunTime() + max(0.1, deadline-time.monotonic()))
            engine.run()
            retried = True
            continue
        if status == hc.HighsModelStatus.kInfeasible:
            T = sparse.vstack(chunks, format="csr")
            eq = np.concatenate(eqs)
            permutation = np.r_[np.flatnonzero(eq), np.flatnonzero(~eq)]
            T = T[permutation]
            cones = []
            if np.sum(eq):
                cones.append({"type": "zero", "dim": int(np.sum(eq))})
            if np.sum(~eq):
                cones.append({"type": "nonnegative", "dim": int(np.sum(~eq))})
            outer = Problem(p.P, p.q, (T @ p.A).tocsc(), T @ p.b, cones)
            cr = certificate(outer, "primal_infeasible", deadline)
            if cr is not None:
                result = {"status": "primal_infeasible", "z": T.T @ cr["z"]}
                if check(result):
                    return finish(result)
            break
        if status in (hc.HighsModelStatus.kUnbounded,
                      hc.HighsModelStatus.kUnboundedOrInfeasible):
            bounded = True
            engine.changeColsBounds(p.n, np.arange(p.n, dtype=np.int32),
                                    np.full(p.n, -bound), np.full(p.n, bound))
            engine.setOptionValue("time_limit", engine.getRunTime() + max(0.1, deadline - time.monotonic()))
            engine.run()
            continue
        if not sol.value_valid or not sol.dual_valid:
            break
        x = np.asarray(sol.col_value)
        z = lift_dual(chunks, -np.asarray(sol.row_dual), p.m)
        s = p.b - p.A @ x
        result = {"status": "solved", "x": x, "s": s, "z": z}
        if check(result):
            return finish(result)
        repaired = dict(result, s=repair_slack(p, s))
        if check(repaired):
            return finish(repaired)
        if iteration <= 1 and try_interior and status == hc.HighsModelStatus.kOptimal:
            from polish_soc import polish
            polished = polish(p, result, min(deadline, time.monotonic() + 20))
            if polished is not None and check(polished):
                return finish(polished)
            from soc_direct import solve as direct
            remaining = max(0.0, deadline - time.monotonic())
            interior = direct(p, time.monotonic() + min(80, 0.6 * remaining),
                              warm=result, return_best=True)
            if check(interior):
                return finish(interior)
            try_interior = False
        new, worst = separating_cuts(p, s)
        if os.environ.get("CONIC_DEBUG"):
            print("OA", iteration, status, "rows", engine.getNumRow(), "cone", worst,
                  "obj", float(p.q @ x), "sec", time.monotonic() - started,
                  file=__import__("sys").stderr)
        if not new.shape[0]:
            if p.P.nnz:
                refined = refine_outer(p, chunks, eqs, x, -np.asarray(sol.row_dual))
                if refined is not None and check(refined):
                    return finish(refined)
            if bounded and np.max(np.abs(np.asarray(sol.col_dual)), initial=0) > 1e-8:
                dx = x / max(1.0, norm(x))
                direction = {"status": "dual_infeasible", "x": dx, "s": -p.A @ dx}
                if check(direction):
                    return finish(direction)
                bound *= 100.0
                engine.changeColsBounds(p.n, np.arange(p.n, dtype=np.int32),
                                        np.full(p.n, -bound), np.full(p.n, bound))
            else:
                break
        else:
            add_A = (new @ p.A).tocsr()
            add_b = new @ p.b
            engine.addRows(new.shape[0], np.full(new.shape[0], -hc.kHighsInf),
                           add_b, add_A.nnz, add_A.indptr.astype(np.int32),
                           add_A.indices.astype(np.int32), add_A.data)
            chunks.append(new)
            eqs.append(np.zeros(new.shape[0], dtype=bool))
            if ipm:
                engine.setOptionValue("simplex_strategy", 1)
        engine.setOptionValue("time_limit", engine.getRunTime() + max(0.1, deadline - time.monotonic()))
        engine.run()
    return {"status": "unresolved"}
