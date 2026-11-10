"""Sparse equality elimination followed by inequality generation."""
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph, linalg as sla


def solve_reduced(p):
    from solver import highs_model, native_highs, quality, norm, linear_certificate
    eq = np.flatnonzero(p.equal)
    if p.n < 3000 or eq.size < 0.7 * p.n or eq.size >= p.n:
        return None
    Ae = p.A[eq].tocsr()
    cheap = (np.diff(p.P.indptr) == 0) & (p.q == 0)
    order = np.r_[np.flatnonzero(cheap), np.flatnonzero(~cheap)]
    match = csgraph.maximum_bipartite_matching(Ae[:, order].tocsr(), perm_type="column")
    r_eq = np.flatnonzero(match >= 0)
    dep = order[match[r_eq]]
    chosen = eq[r_eq]
    free_mask = np.ones(p.n, dtype=bool)
    free_mask[dep] = False
    free = np.flatnonzero(free_mask)
    nf = free.size
    keep_mask = np.ones(p.m, dtype=bool)
    keep_mask[chosen] = False
    keep = np.flatnonzero(keep_mask)
    if nf > 1000 or nf * keep.size > 35000000 or nf < 1:
        return None
    started = time.monotonic()
    try:
        D = p.A[chosen][:, dep].tocsc()
        try:
            lu = sla.splu(D)
        except RuntimeError:
            from affine import numeric_basis
            selected_rows, dep = numeric_basis(Ae, cheap)
            chosen = eq[selected_rows]
            if dep.size != chosen.size:
                return None
            free_mask[:] = True
            free_mask[dep] = False
            free = np.flatnonzero(free_mask)
            nf = free.size
            keep_mask[:] = True
            keep_mask[chosen] = False
            keep = np.flatnonzero(keep_mask)
            if nf > 1000 or nf * keep.size > 35000000 or nf < 1:
                return None
            D = p.A[chosen][:, dep].tocsc()
            lu = sla.splu(D)
        F = p.A[chosen][:, free].toarray()
        T = lu.solve(-F)
        constant = lu.solve(p.b[chosen])
        for _ in range(2):
            T += lu.solve(-F - D @ T)
            constant += lu.solve(p.b[chosen] - D @ constant)
        if not np.isfinite(T).all() or norm(D @ T + F) > 1e-7 * max(1.0, norm(F)):
            return None
        Ak = p.A[keep].tocsc()
        Ad = Ak[:, free].toarray()
        Ad += Ak[:, dep] @ T
        bound = p.b[keep] - Ak[:, dep] @ constant
        cross, Pdd = p.P[free][:, dep], p.P[dep][:, dep]
        PP = p.P[free][:, free].toarray()
        cr = cross @ T
        PP += cr + cr.T + T.T @ (Pdd @ T)
        PP = sparse.csc_matrix((PP + PP.T) * 0.5)
        x0 = np.zeros(p.n)
        x0[dep] = constant
        grad0 = p.P @ x0 + p.q
        qq = grad0[free] + T.T @ grad0[dep]
        equal = p.equal[keep]
        Acsr = Ak.tocsr()
        rowlength = np.diff(Acsr.indptr)
        seed = equal.copy()
        singles = np.flatnonzero(rowlength == 1)
        seed[singles] |= free_mask[Acsr.indices[Acsr.indptr[singles]]]
        violation0 = -bound / (1 + np.abs(bound))
        worst = np.flatnonzero(violation0 > 0)
        if worst.size:
            seed[worst[np.argsort(violation0[worst])[-40:]]] = True
        selected = np.flatnonzero(seed)
        if selected.size == 0:
            selected = np.arange(min(40, keep.size))
            seed[selected] = True
        opt = None
        if not PP.nnz:
            opt, xx, zz = highs_model(
                qq, sparse.csc_matrix(Ad[selected]), np.where(equal[selected], bound[selected], -np.inf),
                bound[selected], solver="choose", time_limit=20.0)
        h = native_highs()
        index_order = selected.copy()
        debug = os.environ.get("SOLVER_DEBUG")
        last_x = np.zeros(nf)

        def recover(status, vec, dual):
            result = {"status": status}
            if status in ("solved", "dual_infeasible"):
                x = np.zeros(p.n)
                x[free] = vec
                x[dep] = T @ vec
                if status == "solved":
                    x[dep] += constant
                s = (p.b if status == "solved" else 0) - p.A @ x
                s[p.equal] = 0.0
                result.update(x=x, s=s)
            if status in ("solved", "primal_infeasible"):
                z = np.zeros(p.m)
                z[keep[index_order]] = dual
                g = p.A.T @ z
                if status == "solved":
                    g += p.P @ x + p.q
                z[chosen] = lu.solve(-g[dep], trans="T")
                result["z"] = z
            return result

        def proxy_problem():
            AA = sparse.csc_matrix(Ad[index_order])
            bb, ee = bound[index_order], equal[index_order]
            blocks = []
            for i, e in enumerate(ee):
                kind = "zero" if e else "nonnegative"
                if blocks and blocks[-1][0] == kind:
                    k, d, a, b = blocks[-1]
                    blocks[-1] = (k, d + 1, a, b + 1)
                else:
                    blocks.append((kind, 1, i, i + 1))
            return SimpleNamespace(A=AA, P=PP, q=qq, b=bb, n=nf, m=bb.size,
                                   equal=ee, blocks=blocks)

        for iteration in range(30):
            if PP.nnz:
                from scipy.optimize import minimize
                from presolve import prune, restore
                proxy, rec = prune(proxy_problem())
                dense_p = proxy.P.toarray()
                dense_a = proxy.A.toarray()
                row_scale = np.maximum(1.0, np.maximum(
                    np.max(np.abs(dense_a), axis=1), np.abs(proxy.b)))
                aaa = dense_a / row_scale[:, None]
                bbb = proxy.b / row_scale
                objective_scale = max(1.0, norm(proxy.q), norm(dense_p))
                eqmask = proxy.equal & (np.max(np.abs(aaa), axis=1) > 1e-12)
                inmask = ~proxy.equal
                constraints = []
                if np.any(eqmask):
                    ae, be = aaa[eqmask], bbb[eqmask]
                    constraints.append({"type": "eq", "fun": lambda v: be - ae @ v,
                                        "jac": lambda v: -ae})
                ai, bi = aaa[inmask], bbb[inmask]
                constraints.append({"type": "ineq", "fun": lambda v: bi - ai @ v,
                                    "jac": lambda v: -ai})
                initial = last_x if last_x.size == proxy.n else np.zeros(proxy.n)
                ss = minimize(lambda v: (0.5 * v @ (dense_p @ v) + proxy.q @ v) / objective_scale,
                              initial, jac=lambda v: (dense_p @ v + proxy.q) / objective_scale,
                              constraints=constraints, method="SLSQP",
                              options={"ftol": 1e-12, "maxiter": 120, "disp": False})
                z0 = np.zeros(proxy.m)
                multipliers = np.asarray(ss.multipliers)
                neq = np.count_nonzero(eqmask)
                z0[eqmask] = multipliers[:neq] * objective_scale / row_scale[eqmask]
                z0[inmask] = multipliers[neq:] * objective_scale / row_scale[inmask]
                sol = {"status": "solved", "x": ss.x, "s": proxy.b - proxy.A @ ss.x, "z": z0}
                sol["s"][proxy.equal] = 0
                if not quality(proxy, sol):
                    from solver import polish_qp
                    pp = polish_qp(proxy, ss.x, z0)
                    if pp is not None and quality(proxy, pp):
                        sol = pp
                    else:
                        return None
                if rec is not None:
                    sol = restore(sol, [rec])
                if sol["status"] == "unresolved":
                    return None
                if sol["status"] != "solved":
                    rr = recover(sol["status"], sol.get("x"), sol.get("z"))
                    return rr if quality(p, rr) else None
                xx, zz = sol["x"], sol["z"]
                last_x = xx.copy()
                status = h.HighsModelStatus.kOptimal
            else:
                status = opt.getModelStatus()
            if debug:
                print("reduced", p.n, "to", nf, "rows", index_order.size,
                      "iteration", iteration, status, "time", time.monotonic() - started,
                      file=sys.stderr)
            if status == h.HighsModelStatus.kInfeasible:
                AA = sparse.csc_matrix(Ad[index_order])
                bb = bound[index_order]
                ee = equal[index_order]
                blocks, pos = [], 0
                for i in range(bb.size):
                    kind = "zero" if ee[i] else "nonnegative"
                    if blocks and blocks[-1][0] == kind:
                        k, d, a, b = blocks[-1]
                        blocks[-1] = (k, d + 1, a, b + 1)
                    else:
                        blocks.append((kind, 1, i, i + 1))
                proxy = SimpleNamespace(A=AA, P=PP, q=qq, b=bb, n=nf, m=bb.size,
                                        equal=ee, blocks=blocks)
                cert = linear_certificate(proxy)
                if cert["status"] == "primal_infeasible":
                    result = recover("primal_infeasible", None, cert["z"])
                    if quality(p, result):
                        return result
                return None
            if status != h.HighsModelStatus.kOptimal:
                return None
            if opt is not None:
                sol = opt.getSolution()
                xx = np.asarray(sol.col_value)
                zz = -np.asarray(sol.row_dual)
            values = Ad @ xx - bound
            violation = np.where(equal, np.abs(values), values) / (1 + np.abs(bound))
            bad = np.flatnonzero((violation > 2e-10) & ~seed)
            if bad.size == 0:
                result = recover("solved", xx, zz)
                if quality(p, result):
                    return result
                # Tighten the active reduced QP without refactoring the
                # eliminated equality system.
                if p.P.nnz:
                    from solver import polish_qp
                    AA = sparse.csc_matrix(Ad[index_order])
                    bb, ee = bound[index_order], equal[index_order]
                    blocks = [("zero" if e else "nonnegative", 1, i, i + 1)
                              for i, e in enumerate(ee)]
                    proxy = SimpleNamespace(A=AA, P=PP, q=qq, b=bb, n=nf, m=bb.size,
                                            equal=ee, blocks=blocks)
                    rr = polish_qp(proxy, xx, zz)
                    if rr is not None:
                        result = recover("solved", rr["x"], rr["z"])
                        if quality(p, result):
                            return result
                return None
            add = bad[np.argsort(violation[bad])[-min(80, bad.size):]]
            rows = sparse.csr_matrix(Ad[add])
            if opt is not None:
                opt.addRows(add.size, np.where(equal[add], bound[add], -np.inf), bound[add],
                            rows.nnz, rows.indptr, rows.indices, rows.data)
            index_order = np.r_[index_order, add]
            seed[add] = True
            if opt is not None:
                opt.run()
            if time.monotonic() - started > 25:
                return None
    except (RuntimeError, ValueError, ArithmeticError, AttributeError, np.linalg.LinAlgError):
        if os.environ.get("SOLVER_DEBUG"):
            import traceback
            traceback.print_exc()
    return None
