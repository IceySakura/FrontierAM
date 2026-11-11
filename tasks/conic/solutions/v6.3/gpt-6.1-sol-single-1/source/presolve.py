"""Small sparse equality reduction with exact primal/dual reconstruction."""
import os
import sys
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import splu
from accuracy import assess, norm
from cones import ConeProduct


def trivial(P, q, A, b, K):
    if norm(q) == 0 and norm(b[K.zi]) == 0:
        if not K.ni.size or np.min(b[K.ni]) >= 0:
            if K.violation(b) <= 5e-9:
                return "solved", np.zeros(q.size), b.copy(), np.zeros(b.size)
    empty = np.asarray(A.getnnz(axis=1)).ravel() == 0
    z = np.zeros(b.size)
    ids = np.flatnonzero(empty & K.zero & (b != 0))
    z[ids] = -np.sign(b[ids])
    ids = np.flatnonzero(empty & K.nonnegative & (b < 0))
    z[ids] = 1.
    for kind, dim, sl in K.blocks:
        if kind in ("zero", "nonnegative") or not np.all(empty[sl]):
            continue
        v = b[sl] / max(1., norm(b[sl]))
        if kind == "soc":
            u = -v
            r = np.linalg.norm(u[1:])
            if u[0] >= r:
                z[sl] = u
            elif u[0] > -r:
                z[sl] = .5*(u[0]+r)*np.r_[1., u[1:]/r]
        elif kind == "exponential":
            a, vv, c = v
            if vv < 0:
                z[sl.start+1] = 1.
            elif c < 0:
                z[sl.start+2] = 1.
            elif vv > 0 and c > 0:
                theta = np.log(c)-np.log(vv)
                if a > vv*theta:
                    if theta >= 0:
                        z[sl] = [-1., theta-1., np.exp(-theta)]
                    else:
                        z[sl] = [-np.exp(theta), np.exp(theta)*(theta-1.), 1.]
            elif vv == 0 and a > 0:
                theta = max(1., np.log(max(c, 1e-300))-np.log(a)+2.)
                z[sl] = [-1., theta-1., np.exp(-theta)]
            elif vv > 0 and c == 0:
                theta = np.clip(a/vv, -700., 700.)
                if theta >= 0:
                    z[sl] = [-1., theta-1., np.exp(-theta)]
                else:
                    z[sl] = [-np.exp(theta), np.exp(theta)*(theta-1.), 1.]
        else:
            block = next(p for p in K.psd if p[0].start == sl.start)
            B = K.mat(b, block)
            eig, U = linalg.eigh(B, check_finite=False, driver="evd")
            eig = np.maximum(-eig, 0.)
            z[sl] = K.vec((U*eig)@U.T, block)
    if np.any(z):
        result = ("primal_infeasible", None, None, z)
        if assess(P, q, A, b, K, result)[0] < .8:
            return result
    cols = (np.diff(A.indptr) == 0) & (np.diff(P.indptr) == 0) & (q != 0)
    if np.any(cols):
        x = np.zeros(q.size)
        x[cols] = -np.sign(q[cols])
        result = ("dual_infeasible", x, np.zeros(b.size), None)
        if assess(P, q, A, b, K, result)[0] < .8:
            return result
    return None


def solve_separable(P, q, A, b, K):
    if not np.all(K.zero | K.nonnegative):
        return None
    n, m = q.size, b.size
    pc = np.repeat(np.arange(n), np.diff(P.indptr))
    if np.any(P.indices != pc):
        return None
    R = A.tocsr()
    counts = np.diff(R.indptr)
    if np.any(counts > 1):
        return None
    rows = np.flatnonzero(counts)
    cols, val = R.indices, R.data
    if np.any(val == 0):
        return None
    rhs = b[rows] / val
    lower = np.full(n, -np.inf)
    upper = np.full(n, np.inf)
    is_lower = (val < 0) | K.zero[rows]
    is_upper = (val > 0) | K.zero[rows]
    np.maximum.at(lower, cols[is_lower], rhs[is_lower])
    np.minimum.at(upper, cols[is_upper], rhs[is_upper])
    li, ui = np.full(n, m, dtype=int), np.full(n, m, dtype=int)
    mask = is_lower & (rhs == lower[cols])
    np.minimum.at(li, cols[mask], rows[mask])
    mask = is_upper & (rhs == upper[cols])
    np.minimum.at(ui, cols[mask], rows[mask])
    bad = np.flatnonzero(lower > upper)
    rowval = np.zeros(m)
    rowval[rows] = val
    if bad.size:
        j = bad[np.argmax(lower[bad]-upper[bad])]
        z = np.zeros(m)
        z[ui[j]] = 1./rowval[ui[j]]
        z[li[j]] -= 1./rowval[li[j]]
        result = ("primal_infeasible", None, None, z)
        return result if assess(P, q, A, b, K, result)[0] <= .8 else None
    p = P.diagonal()
    positive = p > 0
    unbounded = (~positive) & (((q > 0) & ~np.isfinite(lower)) |
                              ((q < 0) & ~np.isfinite(upper)))
    if np.any(unbounded):
        x = np.zeros(n)
        x[unbounded] = -np.sign(q[unbounded])
        result = ("dual_infeasible", x, -A @ x, None)
        return result if assess(P, q, A, b, K, result)[0] <= .8 else None
    x = np.zeros(n)
    x[positive] = -q[positive]/p[positive]
    choose = (~positive) & (q > 0)
    x[choose] = lower[choose]
    choose = (~positive) & (q < 0)
    x[choose] = upper[choose]
    x = np.clip(x, lower, upper)
    gradient = p*x + q
    z = np.zeros(m)
    choose = (gradient > 0) & (li < m)
    z[li[choose]] = -gradient[choose]/rowval[li[choose]]
    choose = (gradient < 0) & (ui < m)
    z[ui[choose]] = -gradient[choose]/rowval[ui[choose]]
    result = ("solved", x, b-A@x, z)
    return result if assess(P, q, A, b, K, result)[0] <= .8 else None


def reduce_equalities(P, q, A, b, K):
    n, m = q.size, b.size
    ne = K.zi.size
    if not (2 <= ne < n and n <= 2000 and ne <= 1600 and ne*(n-ne) <= 700000):
        return None
    E = A[K.zi, :].toarray()
    # Rank revelation also removes redundant equality rows without changing K.
    _, R, rows = linalg.qr(E.T, mode="economic", pivoting=True, check_finite=False)
    diag = np.abs(np.diag(R))
    rank = int(np.count_nonzero(diag > 2e-11 * max(1., norm(E))))
    if rank == 0 or rank >= n or rank < ne*.5:
        return None
    rows = np.asarray(rows[:rank])
    EE = E[rows, :]
    pcols = np.diff(P.indptr) > 0
    cost = 1. + 3.*(q != 0) + 5.*pcols
    cn = np.max(np.abs(EE), axis=0)
    col_weight = 1 / (cost * np.maximum(cn, 1e-12))
    _, _, pivot = linalg.qr(EE * col_weight, mode="economic", pivoting=True,
                             check_finite=False)
    pivot = np.asarray(pivot[:rank])
    free = np.setdiff1d(np.arange(n), pivot, assume_unique=True)
    if rank*free.size > 1000000:
        return None
    B = sparse.csc_matrix(EE[:, pivot])
    try:
        lu = splu(B)
    except RuntimeError:
        return None
    x0 = np.zeros(n)
    x0[pivot] = lu.solve(b[K.zi[rows]])
    coeff = -lu.solve(EE[:, free])
    T = np.zeros((n, free.size))
    T[free, np.arange(free.size)] = 1.
    T[pivot, :] = coeff
    if norm(E @ x0 - b[K.zi]) > 1e-9 * max(1., norm(b[K.zi])):
        return None
    if norm(E @ T) > 1e-10 * max(1., norm(E)):
        return None
    # Avoid reductions which replace sparse constraints with dense ones.
    Ts = sparse.csc_matrix(T)
    remain = np.flatnonzero(~K.zero)
    Ar = (A[remain, :] @ Ts).tocsc()
    Ar.eliminate_zeros()
    if Ar.nnz > max(A.nnz * 3, 1000):
        return None
    Pr = (Ts.T @ P @ Ts).tocsc()
    Pr = ((Pr + Pr.T) * .5).tocsc()
    qr = np.asarray(Ts.T @ (q + P @ x0)).ravel()
    br = b[remain] - A[remain, :] @ x0
    cones = [dict(c) for c in K.spec if c["type"] != "zero"]
    Kr = ConeProduct(cones)
    Kr.exp_mode = K.exp_mode
    eqrows = K.zi[rows]
    Ep = A[eqrows, :].tocsc()

    def restore(result):
        status, xx, ss, zz = result
        if status == "unresolved":
            return result
        sx = None
        if xx is not None:
            xx = np.asarray(Ts @ xx).ravel()
            if status == "solved":
                xx += x0
            sx = np.zeros(m)
            sx[remain] = ss
        if zz is not None:
            outz = np.zeros(m)
            outz[remain] = zz
            gradient = A.T @ outz
            if status == "solved":
                gradient += P @ xx + q
            ze = lu.solve(-gradient[pivot], trans="T")
            for _ in range(2):
                residual = -gradient[pivot] - B.T @ ze
                ze += lu.solve(residual, trans="T")
            outz[eqrows] = ze
            zz = outz
        return status, xx, sx, zz

    if os.environ.get("SOLVER_DEBUG"):
        print("presolve", n, ne, "->", free.size, "nnz", A.nnz, Ar.nnz, file=sys.stderr)
    return (Pr, qr, Ar, br, Kr), restore
