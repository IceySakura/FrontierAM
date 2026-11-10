#!/usr/bin/env python3
"""Sparse convex conic optimization in the portable NPZ interface."""

import argparse
import json
import os
import sys
import time
import importlib.util
from importlib.machinery import EXTENSION_SUFFIXES

import numpy as np
_numpy_exports = np.__all__
try:
    if os.environ.get("SOLVER_TRIM_IMPORTS", "1") == "1":
        np.__all__ = [name for name in _numpy_exports if name not in ("testing", "f2py")]
    import scipy
    from scipy import sparse
finally:
    np.__all__ = _numpy_exports


def norm(v):
    return float(np.max(np.abs(v), initial=0.0))


class Problem:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as f:
            self.q = f["q"]
            self.b = f["b"]
            meta = json.loads(str(f["metadata_json"]))
            self.cones = meta["cones"]
            for p in ("P", "A"):
                a = sparse.csc_matrix(
                    (f[p + "_data"], f[p + "_indices"], f[p + "_indptr"]),
                    shape=tuple(f[p + "_shape"]),
                )
                a.sum_duplicates()
                a.eliminate_zeros()
                setattr(self, p, a)
        self.n, self.m = self.q.size, self.b.size
        self.core_offset = 0.0
        self.blocks = []
        self.equal = np.zeros(self.m, dtype=bool)
        start = 0
        for c in self.cones:
            k, d = c["type"], c["dim"]
            w = d * (d + 1) // 2 if k == "psd" else d
            self.blocks.append((k, d, start, start + w))
            if k == "zero":
                self.equal[start:start + w] = True
            start += w


def smat(v, d):
    a = np.zeros((d, d))
    i, j = np.tril_indices(d)
    a[i, j] = v / np.where(i == j, 1.0, np.sqrt(2.0))
    return a + np.tril(a, -1).T


def svec(a):
    i, j = np.tril_indices(a.shape[0])
    return a[i, j] * np.where(i == j, 1.0, np.sqrt(2.0))


_cone_groups = {}


def cone_error(v, blocks, dual=False):
    key = id(blocks)
    if key not in _cone_groups:
        soc, ex, linear, psd = {}, [], [], []
        for k, d, a, b in blocks:
            if k == "soc":
                soc.setdefault(d, []).append(np.arange(a, b))
            elif k == "exponential":
                ex.append(np.arange(a, b))
            elif k == "psd":
                psd.append((d, a, b))
            else:
                linear.append((k, a, b))
        _cone_groups[key] = (blocks, [(d, np.array(ids)) for d, ids in soc.items()],
                             np.array(ex, dtype=int).reshape(-1, 3), linear, psd)
    _, soc, ex, linear, psd = _cone_groups[key]
    worst = 0.0
    for k, a, b in linear:
        if k == "zero" and dual:
            continue
        u = v[a:b]
        e = norm(u) if k == "zero" else max(0.0, -float(np.min(u)))
        worst = max(worst, e / max(1.0, norm(u)))
    for _, ids in soc:
        u = v[ids]
        u = u / np.maximum(1.0, np.max(np.abs(u), axis=1))[:, None]
        e = np.linalg.norm(u[:, 1:], axis=1) - u[:, 0]
        worst = max(worst, float(np.max(e, initial=0.0)))
    if ex.size:
        u = v[ex]
        u /= np.maximum(1.0, np.max(np.abs(u), axis=1))[:, None]
        x, y, z = (-u[:, 1], -u[:, 0], np.e * u[:, 2]) if dual else u.T
        e = np.maximum.reduce([np.abs(y), np.maximum(x, 0), np.maximum(-z, 0)])
        good = (y > 0) & (z > 0)
        e[good] = np.minimum(e[good], np.maximum(
            0.0, x[good] - y[good] * (np.log(z[good]) - np.log(y[good]))))
        worst = max(worst, float(np.max(e, initial=0.0)))
    if psd:
        from scipy import linalg
        for d, a, b in psd:
            u = v[a:b] / max(1.0, norm(v[a:b]))
            e = -float(linalg.eigh(smat(u, d), subset_by_index=[0, 0],
                                   eigvals_only=True, check_finite=False, driver="evr")[0])
            worst = max(worst, e)
    return worst


def quality(p, r, tol=9.95e-9):
    status = r.get("status", "unresolved")
    if status == "unresolved":
        return False
    if not all(np.isfinite(v).all() for k, v in r.items() if k != "status"):
        return False
    P, q, A, b = p.P, p.q, p.A, p.b
    if status == "solved":
        x, s, z = (r[k] for k in ("x", "s", "z"))
        ax, px, az = A @ x, P @ x, A.T @ z
        rp = norm(ax + s - b) / (1.0 + max(norm(ax), norm(s), norm(b)))
        rd = norm(px + q + az) / (1.0 + max(norm(px), norm(q), norm(az)))
        po, do = 0.5 * (x @ px) + q @ x, -0.5 * (x @ px) - b @ z
        offset = getattr(p, "core_offset", 0.0)
        gap = abs(po - do) / max(1.0, abs(po + offset), abs(do + offset))
        if os.environ.get("SOLVER_DEBUG"):
            print("quality", rp, rd, gap, file=sys.stderr)
        return (max(rp, rd, gap) <= tol and
                cone_error(s, p.blocks) <= tol and
                cone_error(z, p.blocks, True) <= tol)
    aa = norm(A.data) or 1.0
    if status == "primal_infeasible":
        z = r["z"] / max(norm(r["z"]), 1e-300)
        beta = -(b / max(norm(b), 1e-300)) @ z
        return (beta > 1e-8 and norm(A.T @ z) / aa <= tol * beta
                and cone_error(z, p.blocks, True) <= tol * beta)
    x = r["x"] / max(norm(r["x"]), 1e-300)
    s = r["s"] / max(norm(r["x"]), 1e-300)
    alpha = -(q / max(norm(q), 1e-300)) @ x
    return (alpha > 1e-8 and norm(P @ x) / (norm(P.data) or 1.0) <= tol * alpha
            and norm(A @ x + s) / aa <= tol * alpha
            and cone_error(s / aa, p.blocks) <= tol * alpha)


_highs = None


def native_highs():
    global _highs
    if _highs is None:
        module_name = "scipy.optimize._highspy._core"
        if module_name in sys.modules:
            _highs = sys.modules[module_name]
            return _highs
        path = (scipy.__path__[0] + "/optimize/_highspy/_core"
                + EXTENSION_SUFFIXES[0])
        spec = importlib.util.spec_from_file_location(module_name, path)
        _highs = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = _highs
        spec.loader.exec_module(_highs)
    return _highs


def highs_model(q, A, lo, hi, P=None, lower=None, upper=None, solver="choose",
                presolve="on", time_limit=240.0):
    h = native_highs()
    n, m = q.size, hi.size
    model = h.HighsModel()
    lp = model.lp_
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = q
    lp.col_lower_ = np.full(n, -h.kHighsInf) if lower is None else lower
    lp.col_upper_ = np.full(n, h.kHighsInf) if upper is None else upper
    lp.row_lower_, lp.row_upper_ = lo, hi
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, m
    lp.a_matrix_.format_ = h.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    if P is not None and P.nnz:
        Q = sparse.tril(P, format="csc")
        hh = model.hessian_
        hh.dim_ = n
        hh.format_ = h.HessianFormat.kTriangular
        hh.start_, hh.index_, hh.value_ = Q.indptr, Q.indices, Q.data
    opt = h._Highs()
    for key, val in (
        ("output_flag", False), ("threads", 1), ("parallel", "off"),
        ("presolve", presolve), ("solver", solver),
        ("primal_feasibility_tolerance", 1e-9),
        ("dual_feasibility_tolerance", 1e-9),
        ("ipm_optimality_tolerance", float(os.environ.get("SOLVER_IPM_TOL", "1e-8"))),
        ("ipx_dualize_strategy", int(os.environ.get("SOLVER_DUALIZE", "2"))),
        ("run_crossover", "on"),
        ("qp_regularization_value", 1e-12),
        ("qp_iteration_limit", 50000),
        ("time_limit", float(time_limit)),
    ):
        opt.setOptionValue(key, val)
    opt.passModel(model)
    opt.run()
    sol = opt.getSolution()
    if os.environ.get("SOLVER_DEBUG"):
        print("highs", opt.getModelStatus(), opt.getRunTime(), file=sys.stderr)
    return opt, np.asarray(sol.col_value), -np.asarray(sol.row_dual)


def linear_certificate(p):
    # A bounded alternative LP finds a normalized Farkas separator.
    lo = np.where(p.equal, -1.0, 0.0)
    _, z, _ = highs_model(
        p.b / max(1.0, norm(p.b)), p.A.T.tocsc(),
        np.zeros(p.n), np.zeros(p.n), lower=lo, upper=np.ones(p.m),
    )
    r = {"status": "primal_infeasible", "z": z}
    if quality(p, r):
        return r
    return {"status": "unresolved"}


def polyhedral(p):
    lower = np.where(p.equal, p.b, -np.inf)
    native_lp = "ipm" if p.n > 1000 else "choose"
    opt, x, z = highs_model(p.q, p.A, lower, p.b, p.P,
                            solver=os.environ.get("SOLVER_LP", native_lp if not p.P.nnz else "choose"))
    if x.size == p.n and z.size == p.m:
        r = {"status": "solved", "x": x, "s": p.b - p.A @ x, "z": z}
        if quality(p, r):
            return r
        if not p.P.nnz and opt.getSolution().value_valid:
            opt.crossover(opt.getSolution())
            sol = opt.getSolution()
            x, z = np.asarray(sol.col_value), -np.asarray(sol.row_dual)
            r = {"status": "solved", "x": x, "s": p.b - p.A @ x, "z": z}
            if quality(p, r):
                return r
        if p.P.nnz:
            polished = polish_qp(p, x, z)
            if polished is not None and quality(p, polished):
                return polished
    h = native_highs()
    status = opt.getModelStatus()
    if status in (h.HighsModelStatus.kInfeasible, h.HighsModelStatus.kUnboundedOrInfeasible):
        return linear_certificate(p)
    if status == h.HighsModelStatus.kUnbounded:
        M = sparse.vstack((p.A, p.P), format="csc")
        lo = np.r_[np.where(p.equal, 0.0, -np.inf), np.zeros(p.n)]
        hi = np.zeros(p.m + p.n)
        _, x, _ = highs_model(
            p.q / max(1.0, norm(p.q)), M, lo, hi,
            lower=-np.ones(p.n), upper=np.ones(p.n))
        r = {"status": "dual_infeasible", "x": x, "s": -p.A @ x}
        if quality(p, r):
            return r
    return {"status": "unresolved"}


def polish_qp(p, x, z):
    """Refine an identified active face in the original, unregularized KKT system."""
    sl = p.b - p.A @ x
    active = p.equal | ((sl < 1e-7 * (1.0 + norm(p.b))) & (z > 1e-10))
    for _ in range(6):
        ids = np.flatnonzero(active)
        G = p.A[ids, :]
        rhs = np.r_[-(p.P @ x + p.q + G.T @ z[ids]), p.b[ids] - G @ x]
        try:
            from refine import kkt_solve
            delta = kkt_solve(p.P, G, rhs[:p.n], rhs[p.n:])
        except Exception:
            if os.environ.get("SOLVER_DEBUG"):
                import traceback
                traceback.print_exc()
            return None
        if not np.isfinite(delta).all():
            return None
        xx = x + delta[:p.n]
        zz = np.zeros(p.m)
        zz[ids] = z[ids] + delta[p.n:]
        ss = p.b - p.A @ xx
        bad_z = (zz < -1e-9) & ~p.equal & active
        bad_s = (ss < -1e-8 * (1.0 + norm(p.b))) & ~active
        if os.environ.get("SOLVER_DEBUG"):
            print("qp face", ids.size, "remove", bad_z.sum(), "add", bad_s.sum(),
                  "step", norm(delta), file=sys.stderr)
        if not (np.any(bad_z) or np.any(bad_s)):
            return {"status": "solved", "x": xx, "s": ss, "z": zz}
        active[bad_z] = False
        active[bad_s] = True
        x, z = xx, zz
    return None


def solve(p):
    if p.n == 0:
        r = {"status": "solved", "x": np.empty(0), "s": p.b.copy(), "z": np.zeros(p.m)}
        if quality(p, r):
            return r
        for k, d, a, b in p.blocks:
            v = p.b[a:b]
            zz = np.zeros(p.m)
            if k == "zero" and norm(v):
                zz[a:b] = -v
            elif k == "nonnegative" and np.any(v < 0):
                zz[a:b] = np.maximum(-v, 0)
            elif k == "soc" and np.linalg.norm(v[1:]) > v[0]:
                rr = np.linalg.norm(v[1:])
                if rr <= -v[0]:
                    zz[a:b] = -v
                else:
                    zz[a] = 0.5 * (rr - v[0])
                    zz[a+1:b] = -v[1:] * (rr - v[0]) / (2 * rr)
            elif k == "psd":
                from scipy import linalg
                w, U = linalg.eigh(smat(v, d), check_finite=False)
                zz[a:b] = svec((U * np.maximum(-w, 0)) @ U.T)
            if norm(zz):
                r = {"status": "primal_infeasible", "z": zz}
                if quality(p, r):
                    return r
        return {"status": "unresolved"}
    if all(k in ("zero", "nonnegative") for k, _, _, _ in p.blocks):
        if (os.environ.get("SOLVER_REDUCED") == "1" and p.P.nnz and p.n > 3000
                and np.count_nonzero(p.equal) > 0.7 * p.n):
            from reduced import solve_reduced
            reduced_result = solve_reduced(p)
            if reduced_result is not None:
                return reduced_result
        row_nnz = np.diff(p.A.tocsr().indptr)
        diagonal_bounds = np.all(row_nnz[~p.equal] <= 1)
        use_ipm_qp = p.P.nnz and p.n >= 150 and p.P.nnz <= 10 * p.n
        if use_ipm_qp and p.n < 4000 and p.P.nnz < p.n:
            use_ipm_qp = False
        use_ipm_lp = not p.P.nnz and p.n > 6000 and (
            diagonal_bounds or p.m < 1.6 * p.n)
        if use_ipm_qp or use_ipm_lp:
            from conic import solve_conic
            r = solve_conic(p)
            if r["status"] != "unresolved":
                return r
            if use_ipm_qp and 2500 <= p.n <= 30000 and p.P.nnz < 5 * p.n:
                from fwqp import solve_fw
                frank_wolfe = solve_fw(p)
                if frank_wolfe is not None:
                    return frank_wolfe
        r = polyhedral(p)
        if r["status"] == "unresolved" and p.n < 2500:
            from conic import solve_conic
            return solve_conic(p)
        return r
    from conic import solve_conic, primal_barrier
    if any(k == "exponential" for k, _, _, _ in p.blocks):
        from geometric import solve_geometric
        r = solve_geometric(p)
        if r["status"] != "unresolved":
            return r
        r = closure_direction(p)
        if r is not None:
            return r
        r = primal_barrier(p)
        if r["status"] != "unresolved":
            return r
    return solve_conic(p)


def closure_direction(p):
    if p.P.nnz or not norm(p.q) or any(
            k not in ("zero", "nonnegative", "exponential") for k, _, _, _ in p.blocks):
        return None
    lo, hi = np.full(p.m, -np.inf), np.zeros(p.m)
    for k, _, a, b in p.blocks:
        if k == "zero":
            lo[a:b] = 0.0
        elif k == "exponential":
            lo[a], hi[a] = 0.0, np.inf
            lo[a + 1] = hi[a + 1] = 0.0
    _, x, _ = highs_model(p.q / norm(p.q), p.A, lo, hi,
                          lower=-np.ones(p.n), upper=np.ones(p.n), time_limit=20.0)
    r = {"status": "dual_infeasible", "x": x, "s": -p.A @ x}
    return r if quality(p, r) else None


def simple_result(p):
    """Check zero optima and coordinate certificates before any factorization."""
    if not norm(p.q) and not norm(p.b[p.equal]) and cone_error(p.b, p.blocks) <= 1e-10:
        return {"status": "solved", "x": np.zeros(p.n), "s": p.b.copy(), "z": np.zeros(p.m)}
    lengths = np.diff(p.A.indptr)
    pcols = np.diff(p.P.indptr)
    nonneg = np.zeros(p.m, dtype=bool)
    for k, _, a, b in p.blocks:
        if k == "nonnegative":
            nonneg[a:b] = True
    qnz = np.flatnonzero((p.q != 0) & (pcols == 0))
    if qnz.size:
        free = qnz[lengths[qnz] == 0]
        single = qnz[lengths[qnz] == 1]
        if single.size:
            pos = p.A.indptr[single]
            good = nonneg[p.A.indices[pos]] & (p.A.data[pos] * p.q[single] >= 0)
            free = np.r_[free, single[good]]
        if free.size:
            j = free[np.argmax(np.abs(p.q[free]))]
            xx = np.zeros(p.n)
            xx[j] = -np.sign(p.q[j])
            r = {"status": "dual_infeasible", "x": xx, "s": -p.A @ xx}
            if quality(p, r):
                return r
    rows = np.bincount(p.A.indices, minlength=p.m)
    bad = np.flatnonzero((rows == 0) & (
        (p.equal & (p.b != 0)) | (nonneg & (p.b < 0))))
    if bad.size:
        j = bad[np.argmax(np.abs(p.b[bad]))]
        z = np.zeros(p.m)
        z[j] = -np.sign(p.b[j])
        r = {"status": "primal_infeasible", "z": z}
        if quality(p, r):
            return r
    return None


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--input", required=True)
    args.add_argument("--output", required=True)
    args = args.parse_args()
    try:
        p = Problem(args.input)
        from presolve import presolve, restore
        from affine import affine_presolve, affine_restore
        r = simple_result(p)
        if r is None:
            base, affine_records = p, []
            affine_rounds = 0 if any(k == "soc" for k, _, _, _ in p.blocks) else 3
            if (not p.P.nnz and p.n > 10000 and p.A.nnz < 5 * p.n
                    and all(k in ("zero", "nonnegative") for k, _, _, _ in p.blocks)):
                affine_rounds = 0
            for _ in range(affine_rounds):
                step = affine_presolve(base)
                if step is None:
                    break
                base, record = step
                affine_records.append(record)
            reduced, records = presolve(base)
            if os.environ.get("SOLVER_DEBUG"):
                print("presolve", p.n, p.m, "to", reduced.n, reduced.m, file=sys.stderr)
            r = affine_restore(restore(solve(reduced), records), affine_records)
        if not quality(p, r, tol=1e-8):
            if r["status"] == "solved" and p.P.nnz and all(
                    k in ("zero", "nonnegative") for k, _, _, _ in p.blocks):
                rr = polish_qp(p, r["x"], r["z"])
                r = rr if rr is not None and quality(p, rr, tol=1e-8) else {"status": "unresolved"}
            else:
                r = {"status": "unresolved"}
        if r["status"] == "unresolved" and p.P.nnz and p.n <= 300:
            from conic import solve_conic
            r = solve_conic(p, maxiter=80, legacy=True)
            if not quality(p, r, tol=1e-8):
                r = {"status": "unresolved"}
    except Exception as e:
        if os.environ.get("SOLVER_DEBUG"):
            import traceback
            traceback.print_exc()
        r = {"status": "unresolved"}
    status = r.pop("status")
    meta = np.array(json.dumps({"schema_version": 1, "status": status}))
    with open(args.output, "wb") as f:
        np.savez(f, metadata_json=meta, **r)


if __name__ == "__main__":
    sys.modules["solver"] = sys.modules[__name__]
    main()
