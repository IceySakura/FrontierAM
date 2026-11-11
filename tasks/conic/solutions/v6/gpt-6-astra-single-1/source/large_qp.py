"""Exact recognition and elimination of sparse regression epigraphs."""

import ctypes
import os
import time
from pathlib import Path

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as sla

from problem import blocks
from quality import acceptable, norm


def regression_form(p):
    if p.P.nnz == 0:
        return None
    counts = np.diff(p.P.indptr)
    rcols = np.flatnonzero(counts)
    if np.any(counts[rcols] != 1) or np.any(p.P.indices != rcols) or np.any(p.P.data <= 0):
        return None
    eq = np.zeros(p.m, bool)
    for kind, dim, sl in blocks(p.cones):
        if kind == "zero":
            eq[sl] = True
        elif kind != "nonnegative":
            return None
    erows, grows = np.flatnonzero(eq), np.flatnonzero(~eq)
    if erows.size != rcols.size or np.any(p.b[grows] != 0) or np.any(p.q[rcols] != 0):
        return None
    acount = np.diff(p.A.indptr)
    if np.any(acount[rcols] != 1):
        return None
    ar = p.A.indices[p.A.indptr[rcols]]
    av = p.A.data[p.A.indptr[rcols]]
    rowmap = np.full(p.m, -1, dtype=np.int32)
    rowmap[erows] = np.arange(erows.size)
    rrow = rowmap[ar]
    if np.any(rrow < 0) or np.unique(rrow).size != erows.size:
        return None
    rvar = np.empty(erows.size, dtype=np.int32)
    rcoef = np.empty(erows.size)
    weight = np.empty(erows.size)
    rvar[rrow], rcoef[rrow] = rcols, av
    weight[rrow] = p.P.data / av**2
    E = p.A[erows].tocsc()
    G = p.A[grows].tocsr()
    return erows, grows, rvar, rcoef, weight, E, G


def native():
    library = ctypes.CDLL(str(Path(__file__).resolve().parent / "numerics.so"))
    f = library.lasso_sweeps
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    f.argtypes = [ctypes.c_int32, i32, i32, i32, f64, f64, f64, f64, f64,
                  ctypes.c_int32, ctypes.c_double]
    f.restype = ctypes.c_double
    return f


def lasso(p, form, deadline):
    erows, grows, rvar, rcoef, weight, E, G = form
    if not np.all(np.diff(G.indptr) == 2):
        return None
    cols, vals = G.indices.reshape(-1, 2), G.data.reshape(-1, 2)
    hascost = p.q[cols] > 0
    if not np.all(np.sum(hascost, axis=1) == 1):
        return None
    tcols = np.where(hascost[:, 0], cols[:, 0], cols[:, 1])
    fcols = np.where(hascost[:, 0], cols[:, 1], cols[:, 0])
    tc = np.where(hascost[:, 0], vals[:, 0], vals[:, 1])
    fc = np.where(hascost[:, 0], vals[:, 1], vals[:, 0])
    if np.any(tc >= 0) or np.any(p.q[fcols] != 0) or np.any(np.diff(E.indptr)[tcols] != 0):
        return None
    features, count = np.unique(fcols, return_counts=True)
    if np.any(count != 2):
        return None
    fmap = np.full(p.n, -1, dtype=np.int32)
    fmap[features] = np.arange(features.size)
    fi = fmap[fcols]
    positive, negative = fc > 0, fc < 0
    if np.sum(positive) != features.size or np.sum(negative) != features.size:
        return None
    pos, neg = np.empty(features.size, np.int32), np.empty(features.size, np.int32)
    pos[fi[positive]], neg[fi[negative]] = np.flatnonzero(positive), np.flatnonzero(negative)
    if np.any(tcols[pos] != tcols[neg]) or not np.allclose(fc[pos], -fc[neg], rtol=1e-12, atol=0):
        return None
    if not np.allclose(tc[pos], tc[neg], rtol=1e-12, atol=0):
        return None
    epigraph = tcols[pos]
    magnitude, boundcoef = fc[pos], -tc[pos]
    lam = p.q[epigraph] * magnitude / boundcoef
    M = E[:, features].tocsc()
    used = np.flatnonzero(np.diff(M.indptr))
    M = M[:, used].tocsc()
    X = (sparse.diags(np.sqrt(weight)) @ M).tocsc()
    y = np.sqrt(weight) * p.b[erows]
    penalty = np.ascontiguousarray(lam[used])
    diagonal = np.asarray(X.power(2).sum(axis=0)).ravel()
    coef = np.zeros(used.size)
    residual = y.copy()
    gradient = X.T @ residual
    violation = np.maximum(0.0, np.abs(gradient) - penalty)
    initial = np.flatnonzero(violation > 1e-10)
    if initial.size > 1500:
        initial = initial[np.argpartition(violation[initial], -1500)[-1500:]]
    active = np.sort(initial).astype(np.int32)
    sweep = native()
    tol = 5e-10 * max(1.0, norm(p.q))
    started = time.monotonic()
    for iteration in range(2000):
        if time.monotonic() > deadline:
            return None
        sweep(active.size, active, X.indptr, X.indices, X.data, diagonal,
              penalty, coef, residual, 80, 2e-13)
        residual = y - X @ coef
        gradient = X.T @ residual
        nonzero = coef != 0
        violation = np.maximum(0.0, np.abs(gradient) - penalty)
        violation[nonzero] = np.abs(gradient[nonzero] - penalty[nonzero] * np.sign(coef[nonzero]))
        gap = float(penalty @ np.abs(coef) - coef @ gradient)
        objective = 0.5 * float(residual @ residual) + penalty @ np.abs(coef)
        if os.environ.get("CONIC_DEBUG") and iteration % 20 == 0:
            print("LASSO", iteration, "active", active.size, "kkt", norm(violation),
                  "gap", gap / max(1.0, abs(objective)), "sec", time.monotonic()-started,
                  file=__import__("sys").stderr)
        if norm(violation) < tol and abs(gap) < 4e-9 * max(1.0, abs(objective)):
            fullcoef = np.zeros(features.size)
            fullcoef[used] = coef
            x = np.zeros(p.n)
            x[features] = fullcoef
            x[epigraph] = magnitude / boundcoef * np.abs(fullcoef)
            error = p.b[erows] - M @ coef
            x[rvar] = error / rcoef
            z = np.zeros(p.m)
            z[erows] = -weight * error
            grad = np.zeros(features.size)
            grad[used] = M.T @ (weight * error)
            total = p.q[epigraph] / boundcoef
            z[grows[pos]] = 0.5 * (total + grad / magnitude)
            z[grows[neg]] = 0.5 * (total - grad / magnitude)
            result = {"status": "solved", "x": x, "s": p.b - p.A @ x, "z": z}
            if acceptable(p, result):
                return result
            tol *= 0.1
        if iteration and iteration % 10 == 0 and 0 < np.sum(nonzero) < 4000:
            use = np.flatnonzero(nonzero)
            V = X[:, use]
            Gram = (V.T @ V).tocsc()
            rhs = V.T @ y - penalty[use] * np.sign(coef[use])
            try:
                factor = sla.splu(Gram + sparse.eye(use.size, format="csc") * 1e-12)
                sol = factor.solve(rhs)
                for refinement in range(3):
                    sol += factor.solve(rhs - Gram @ sol)
                if np.all(sol * coef[use] > 0):
                    trial = coef.copy()
                    trial[use] = sol
                    rtrial = y - X @ trial
                    value = 0.5 * (rtrial @ rtrial) + penalty @ np.abs(trial)
                    if value <= objective * (1 + 1e-13):
                        coef, residual = trial, rtrial
            except RuntimeError:
                pass
        inactive = (coef == 0)
        inactive[active] = False
        new = np.flatnonzero(inactive & (violation > tol))
        limit = max(500, min(5000, active.size // 3))
        if new.size > limit:
            new = new[np.argpartition(violation[new], -limit)[-limit:]]
        active = np.union1d(active, new).astype(np.int32)
        if iteration % 10 == 9:
            keep = (coef[active] != 0) | (np.abs(gradient[active]) > penalty[active] * (1 - 1e-8))
            active = active[keep]
    return None


def huber(p, form, deadline):
    erows, grows, rvar, rcoef, weight, E, G = form
    if not np.all(np.diff(G.indptr) == 1) or np.any(G.data >= 0):
        return None
    uvar = G.indices
    if np.unique(uvar).size != grows.size or np.any(p.q[uvar] <= 0):
        return None
    count = np.diff(E.indptr)
    if np.any(count[uvar] != 1):
        return None
    ur = E.indices[E.indptr[uvar]]
    uc = E.data[E.indptr[uvar]]
    kappa = p.q[uvar] / np.abs(uc)
    positive, negative = uc > 0, uc < 0
    if np.sum(positive) != erows.size or np.sum(negative) != erows.size:
        return None
    pos, neg = np.empty(erows.size, np.int32), np.empty(erows.size, np.int32)
    pos[ur[positive]], neg[ur[negative]] = np.flatnonzero(positive), np.flatnonzero(negative)
    if not np.allclose(kappa[pos], kappa[neg], rtol=1e-12, atol=0):
        return None
    kappa = kappa[pos]
    bound = kappa / weight
    free = np.ones(p.n, bool)
    free[rvar], free[uvar] = False, False
    free &= count > 0
    features = np.flatnonzero(free)
    if np.any(p.q[features] != 0):
        return None
    M = E[:, features].tocsr()
    active_rows = np.arange(erows.size)
    current = M.tocsc()
    stages = []
    for unused in range(30):
        leaves = np.flatnonzero(np.diff(current.indptr) == 1)
        if not leaves.size:
            break
        positions = current.indptr[leaves]
        leaf_rows, values = current.indices[positions], current.data[positions]
        order = np.lexsort((leaves, -np.abs(values), leaf_rows))
        leaves, leaf_rows, values = leaves[order], leaf_rows[order], values[order]
        first = np.r_[True, leaf_rows[1:] != leaf_rows[:-1]]
        chosen_rows = leaf_rows[first]
        stages.append((active_rows[chosen_rows], leaves[first], values[first]))
        remain = np.ones(active_rows.size, bool)
        remain[chosen_rows] = False
        active_rows = active_rows[remain]
        if not active_rows.size:
            break
        current = M[active_rows].tocsc()
    if stages and active_rows.size < 10000:
        coef = np.zeros(features.size)
        if active_rows.size:
            coef = sla.lsmr(current, p.b[erows][active_rows],
                            atol=1e-13, btol=1e-13, maxiter=1500)[0]
        for rows, columns, values in reversed(stages):
            residual = p.b[erows][rows] - M[rows] @ coef
            coef[columns] += residual / values
        x = np.zeros(p.n)
        x[features] = coef
        z = np.zeros(p.m)
        z[grows] = p.q[uvar] / (-G.data)
        slack = p.b - p.A @ x
        slack[erows] = 0
        candidate = {"status": "solved", "x": x, "s": slack, "z": z}
        if acceptable(p, candidate):
            return candidate
    groups, representative = {}, []
    inverse = np.empty(erows.size, np.int32)
    for i in range(erows.size):
        lo, hi = M.indptr[i:i+2]
        key = (M.indices[lo:hi].tobytes(), M.data[lo:hi].tobytes())
        group = groups.get(key)
        if group is None:
            group = len(representative)
            groups[key] = group
            representative.append(i)
        inverse[i] = group
    nr = len(representative)
    lower, upper = np.full(nr, np.inf), np.full(nr, -np.inf)
    np.minimum.at(lower, inverse, p.b[erows])
    np.maximum.at(upper, inverse, p.b[erows])
    for iteration in range(60):
        mid = (lower + upper) * 0.5
        gradient = np.bincount(inverse, weights=weight * np.clip(p.b[erows] - mid[inverse], -bound, bound),
                               minlength=nr)
        lower = np.where(gradient > 0, mid, lower)
        upper = np.where(gradient <= 0, mid, upper)
    target = (lower + upper) * 0.5
    R = M[representative].tocsc()
    nonempty = np.diff(R.tocsr().indptr) > 0
    target[~nonempty] = 0
    Ruse, tuse = R[nonempty].tocsc(), target[nonempty]
    coef = np.zeros(features.size)
    nzcol = np.diff(Ruse.indptr)
    estimated = float(nzcol.astype(float) @ nzcol)
    try:
        if Ruse.shape[0] < 100000 and estimated < 6e7:
            Gram = (Ruse @ Ruse.T).tocsc()
            diagonal = Gram.diagonal()
            reg = np.maximum(1e-14, 1e-12 * diagonal)
            factor = sla.splu(Gram + sparse.diags(reg, format="csc"))
            dual = factor.solve(tuse)
            for iteration in range(8):
                error = tuse - Gram @ dual
                if norm(error) < 1e-10:
                    break
                dual += factor.solve(error)
            coef = Ruse.T @ dual
        else:
            coef = sla.lsmr(Ruse, tuse, atol=1e-12, btol=1e-12, maxiter=2500)[0]
    except RuntimeError:
        coef = sla.lsmr(Ruse, tuse, atol=1e-12, btol=1e-12, maxiter=2500)[0]
    for iteration in range(30):
        error = p.b[erows] - M @ coef
        quadratic = np.clip(error, -bound, bound)
        x = np.zeros(p.n)
        x[features] = coef
        x[rvar] = quadratic / rcoef
        remaining = error - quadratic
        x[uvar[pos]] = np.maximum(remaining, 0) / uc[pos]
        x[uvar[neg]] = np.minimum(remaining, 0) / uc[neg]
        z = np.zeros(p.m)
        z[erows] = -weight * quadratic
        z[grows] = (p.q[uvar] + uc * z[erows][ur]) / (-G.data)
        result = {"status": "solved", "x": x, "s": p.b - p.A @ x, "z": z}
        if acceptable(p, result):
            return result
        if time.monotonic() > deadline:
            break
        weights = weight * np.minimum(1.0, bound / np.maximum(1e-300, np.abs(error)))
        V = sparse.diags(np.sqrt(weights)) @ M
        rhs = np.sqrt(weights) * error
        update = sla.lsmr(V, rhs, atol=1e-12, btol=1e-12, maxiter=1000)[0]
        coef += update
    return None


def solve(p, deadline):
    form = regression_form(p)
    if form is not None:
        result = lasso(p, form, deadline)
        if result is not None:
            return result
        if time.monotonic() < deadline:
            result = huber(p, form, deadline)
            if result is not None:
                return result
    return {"status": "unresolved"}
