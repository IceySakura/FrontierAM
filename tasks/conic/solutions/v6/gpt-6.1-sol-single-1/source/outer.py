"""Adaptive supporting-plane LP method for second-order cones."""

import time
import numpy as np
from scipy import sparse
from scipy.optimize._highspy import _core as hc
from common import Problem, assess, norm, equilibrate
from presolve import Presolve
from poly import highs, certificates


def solve_soc(original, budget=270.0, maxiter=160):
    start = time.monotonic()
    pre = Presolve(original)
    if pre.early is not None:
        return pre.early
    p, D, E, c = equilibrate(pre.problem, 4)

    def recover(x, z, status="solved"):
        r = {"status": status, "z": z * E / c}
        if status == "solved":
            r.update(x=x * D, s=(p.b - p.A @ x) / E)
        return pre.restore(r)

    eq = np.flatnonzero(p.eq)
    nn = []
    groups = {}
    for kind, d, i, j in p.blocks:
        if kind == "nonnegative":
            nn.extend(range(i, j))
        elif kind == "soc":
            groups.setdefault(d, []).append(i)
    groups = {d: np.asarray(starts)[:, None] + np.arange(d) for d, starts in groups.items()}
    scalar = np.r_[eq, np.asarray(nn, int)]
    base = sparse.coo_matrix((np.ones(len(scalar)), (np.arange(len(scalar)), scalar)),
                             shape=(len(scalar), p.m)).tocsr()
    transforms = [base]

    def cuts(idx, u):
        count, d = idx.shape
        values = np.column_stack((np.ones(count), -u)).ravel()
        return sparse.coo_matrix((values, (np.repeat(np.arange(count), d), idx.ravel())),
                                 shape=(count, p.m)).tocsr()

    for d, idx in groups.items():
        for j in range(d - 1):
            for sign in (-1, 1):
                u = np.zeros((len(idx), d - 1))
                u[:, j] = sign
                transforms.append(cuts(idx, u))
    B = sparse.vstack(transforms, format="csr")
    transforms = [B]
    A, b = (B @ p.A).tocsc(), B @ p.b
    lower = np.full(len(b), -hc.kHighsInf)
    lower[:len(eq)] = b[:len(eq)]
    h = highs(p.P, p.q, A, lower, b, method="ipm" if p.n > 3000 else "simplex",
              time_limit=max(1, budget - (time.monotonic() - start)), return_solver=True)
    best, besterr = {"status": "unresolved"}, float("inf")
    for it in range(maxiter):
        remaining = budget - (time.monotonic() - start)
        if remaining <= 0.25:
            break
        sol = h.getSolution()
        status = h.getModelStatus()
        if status in (hc.HighsModelStatus.kInfeasible, hc.HighsModelStatus.kUnboundedOrInfeasible):
            B = sparse.vstack(transforms, format="csr")
            A, b = (B @ p.A).tocsc(), B @ p.b
            cones = []
            if len(eq):
                cones.append({"type": "zero", "dim": len(eq)})
            if len(b) > len(eq):
                cones.append({"type": "nonnegative", "dim": len(b) - len(eq)})
            lp = Problem(sparse.csc_matrix((p.n, p.n)), p.q, A, b, cones)
            r = certificates(lp, time_limit=min(remaining, 60.0))
            if r["status"] == "primal_infeasible":
                out = recover(None, B.T @ r["z"], "primal_infeasible")
                if assess(original, z=out["z"], status=out["status"])[0]:
                    return out
            break
        if status != hc.HighsModelStatus.kOptimal:
            break
        x, y = np.asarray(sol.col_value), -np.asarray(sol.row_dual)
        z = np.zeros(p.m)
        off = 0
        for T in transforms:
            z += T.T @ y[off:off + T.shape[0]]
            off += T.shape[0]
        candidate = recover(x, z)
        ok, err = assess(original, candidate["x"], candidate["s"], candidate["z"])
        if err < besterr:
            best, besterr = candidate, err
        if ok:
            return candidate
        slack = p.b - p.A @ x
        new = []
        for d, idx in groups.items():
            s = slack[idx]
            r = np.linalg.norm(s[:, 1:], axis=1)
            scale = np.maximum(1, np.max(abs(s), axis=1))
            violated = r - s[:, 0] > 1e-10 * scale
            if np.any(violated):
                ss, rr = s[violated], r[violated]
                u = ss[:, 1:] / np.maximum(rr[:, None], 1e-300)
                new.append(cuts(idx[violated], u))
        if not new:
            break
        T = sparse.vstack(new, format="csr")
        C, d = (T @ p.A).tocsr(), T @ p.b
        transforms.append(T)
        h.addRows(C.shape[0], np.full(len(d), -hc.kHighsInf), d, C.nnz,
                  C.indptr, C.indices, C.data)
        h.setOptionValue("solver", "simplex")
        h.setOptionValue("time_limit", max(1, remaining + h.getRunTime()))
        h.run()
    return best
