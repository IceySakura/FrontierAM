"""Eigenvector supporting-plane fallback for small and medium SDPs."""

import time
import numpy as np
from scipy import sparse, linalg
from scipy.optimize._highspy import _core as hc
from common import Problem, norm, smat, assess, equilibrate
from presolve import Presolve, AffineReduction
from poly import highs, certificates


def solve_psd(original, budget=180.0):
    started = time.monotonic()
    pre = Presolve(original)
    if pre.early is not None:
        return pre.early
    affine = AffineReduction(pre.problem, 4)
    p, D, E, c = equilibrate(affine.problem, 4)
    if p.P.nnz or p.n > 1800 or any(k not in ("zero", "nonnegative", "psd") for k, _, _, _ in p.blocks):
        return {"status": "unresolved"}
    eq = np.flatnonzero(p.eq)
    nn, blocks = [], []
    for kind, d, i, j in p.blocks:
        if kind == "nonnegative":
            nn.extend(range(i, j))
        elif kind == "psd":
            if d > 128:
                return {"status": "unresolved"}
            r, s = np.triu_indices(d)
            order = np.lexsort((r, s))
            r, s = r[order], s[order]
            blocks.append((d, i, j, r, s))
    def restore(x, z, status="solved"):
        r = {"status": status, "z": z * E / c}
        if status == "solved":
            r["x"] = x * D
            r["s"] = (p.b - p.A @ x) / E
        return pre.restore(affine.restore(r))
    scalar = np.r_[eq, np.asarray(nn, int)]
    transforms = [sparse.coo_matrix((np.ones(len(scalar)), (np.arange(len(scalar)), scalar)),
                                   shape=(len(scalar), p.m)).tocsr()]
    for d, i, j, r, s in blocks:
        diag = np.arange(d) * (np.arange(d) + 1) // 2 + np.arange(d)
        rr, cc = [np.arange(d)], [i + diag]
        data = [np.ones(d)]
        a, b = np.triu_indices(d, 1)
        off = b * (b + 1) // 2 + a
        pos = d
        for sign in (-1, 1):
            rows = np.arange(len(a)) + pos
            rr.extend((rows, rows, rows))
            cc.extend((i + diag[a], i + diag[b], i + off))
            data.extend((np.full(len(a), 0.5), np.full(len(a), 0.5),
                         np.full(len(a), sign / np.sqrt(2))))
            pos += len(a)
        transforms.append(sparse.coo_matrix((np.concatenate(data),
            (np.concatenate(rr), np.concatenate(cc))), shape=(pos, p.m)).tocsr())
    B = sparse.vstack(transforms, format="csr")
    transforms = [B]
    A, b = (B @ p.A).tocsc(), B @ p.b
    lower = np.full(len(b), -hc.kHighsInf)
    lower[:len(eq)] = b[:len(eq)]
    radius = 1e5 * max(1, norm(p.b)) / max(1, norm(p.q))
    h = highs(None, p.q, A, lower, b, -np.full(p.n, radius), np.full(p.n, radius), method="ipm",
              time_limit=max(0.1, budget - (time.monotonic() - started)), return_solver=True)
    best, besterr = {"status": "unresolved"}, float("inf")
    for it in range(260):
        remaining = budget - (time.monotonic() - started)
        if remaining <= 0.1:
            break
        status = h.getModelStatus()
        if status == hc.HighsModelStatus.kInfeasible:
            B = sparse.vstack(transforms, format="csr")
            A, b = (B @ p.A).tocsc(), B @ p.b
            cones = []
            if len(eq):
                cones.append({"type": "zero", "dim": len(eq)})
            if len(b) > len(eq):
                cones.append({"type": "nonnegative", "dim": len(b) - len(eq)})
            lp = Problem(sparse.csc_matrix((p.n, p.n)), p.q, A, b, cones)
            r = certificates(lp, time_limit=min(remaining, 45))
            if r["status"] == "primal_infeasible":
                out = restore(None, B.T @ r["z"], "primal_infeasible")
                if assess(original, z=out["z"], status=out["status"])[0]:
                    return out
            break
        if status != hc.HighsModelStatus.kOptimal:
            break
        sol = h.getSolution()
        x, y = np.asarray(sol.col_value), -np.asarray(sol.row_dual)
        z = np.zeros(p.m)
        off = 0
        for T in transforms:
            z += T.T @ y[off:off + T.shape[0]]
            off += T.shape[0]
        candidate = restore(x, z)
        ok, err = assess(original, candidate["x"], candidate["s"], candidate["z"], check_all=True)
        if err < besterr:
            best, besterr = candidate, err
        if ok:
            return candidate
        slack = p.b - p.A @ x
        cuts = []
        for d, i, j, r, s in blocks:
            ev, V = linalg.eigh(smat(slack[i:j], d), check_finite=False)
            neg = ev < -1e-10 * max(1, norm(slack[i:j]))
            if not np.any(neg):
                continue
            V = V[:, neg]
            vals = (V[r] * V[s] * np.where(r == s, 1.0, np.sqrt(2))[:, None]).T
            count = V.shape[1]
            cuts.append(sparse.coo_matrix((vals.ravel(),
                (np.repeat(np.arange(count), j - i), np.tile(np.arange(i, j), count))),
                shape=(count, p.m)).tocsr())
        if not cuts:
            break
        T = sparse.vstack(cuts, format="csr")
        C, d = (T @ p.A).tocsr(), T @ p.b
        transforms.append(T)
        h.addRows(C.shape[0], np.full(len(d), -hc.kHighsInf), d, C.nnz,
                  C.indptr, C.indices, C.data)
        h.setOptionValue("solver", "simplex")
        h.setOptionValue("time_limit", max(0.1, remaining + h.getRunTime()))
        h.run()
    return best
