"""Checked least-squares reduction of a single norm epigraph."""
import numpy as np


def norm(v):
    return float(np.max(np.abs(v), initial=0.0))


def solve(P, q, A, b, cones, la, check):
    if P.nnz or cones.exp.size or cones.psd or len(cones.eq) > 50:
        return None
    if sum(len(rows) for rows in cones.soc.values()) != 1:
        return None
    nz = np.flatnonzero(q)
    if len(nz) != 1:
        return None
    j = int(nz[0])
    rows = next(iter(cones.soc.values()))[0]
    head = A[rows[0]].tocsc()
    if head.nnz != 1:
        return None
    hj = int(np.flatnonzero(np.diff(head.indptr))[0])
    if hj != j:
        return None
    a = float(head.data[0])
    if not a:
        return None
    zhead = -q[j] / a
    if not zhead > 0:
        return None
    T = A[rows[1:]].tocsc()
    if T[:, j].nnz or (len(cones.eq) and A[cones.eq][:, j].nnz):
        return None
    free = np.flatnonzero(np.diff(T.indptr))
    if len(free) > 1000:
        return None
    if len(cones.pos) and len(free) > 250:
        return None
    X = T[:, free].tocsc()
    y = b[rows[1:]]
    bscale = max(1.0, norm(y))
    ys = y / bscale
    coeff = np.asarray(abs(X).max(axis=0).toarray()).ravel() if len(free) else np.empty(0)
    d = 1.0 / np.maximum(coeff, 1e-150)
    E = A[cones.eq][:, free].toarray() if len(cones.eq) else np.empty((0, len(free)))
    F = E * d[None, :]
    g = b[cones.eq] / bscale
    try:
        if len(free):
            if X.nnz > .12 * X.shape[0] * len(free) and X.shape[0] * len(free) <= 25000000:
                V = X.toarray() * d[None, :]
                G = V.T @ V
                h = V.T @ ys
                product = lambda v: V @ v
                transpose = lambda v: V.T @ v
            else:
                # Right scaling preserves the CSC pattern.
                V = X.copy()
                V.data *= np.repeat(d, np.diff(V.indptr))
                G = (V.T @ V).toarray()
                h = V.T @ ys
                product = lambda v: V @ v
                transpose = lambda v: V.T @ v
            ne = len(cones.eq)
            if ne:
                K = np.block([[G, F.T], [F, np.zeros((ne, ne))]])
                factor = la.lu_factor(K, check_finite=False)
                solution = la.lu_solve(factor, np.r_[h, g], check_finite=False)
                beta, eta = solution[:len(free)], solution[len(free):]
                for _ in range(3):
                    residual = np.r_[transpose(ys - product(beta)) - F.T @ eta,
                                     g - F @ beta]
                    delta = la.lu_solve(factor, residual, check_finite=False)
                    beta += delta[:len(free)]
                    eta += delta[len(free):]
            else:
                try:
                    factor = la.cho_factor(G, lower=True, check_finite=False)
                    inverse = lambda rhs: la.cho_solve(factor, rhs, check_finite=False)
                except la.LinAlgError:
                    ev, U = la.eigh(G, check_finite=False)
                    keep = ev > max(1e-150, np.max(ev) * 1e-13)
                    if not np.any(keep):
                        return None
                    U, ev = U[:, keep], ev[keep]
                    inverse = lambda rhs: U @ ((U.T @ rhs) / ev)
                beta = inverse(h)
                eta = np.empty(0)
                for _ in range(3):
                    beta += inverse(transpose(ys - product(beta)))
            residual = ys - product(beta)
        else:
            beta, eta, residual = np.empty(0), np.zeros(len(cones.eq)), ys
        scale = norm(residual)
        rn = scale * float(np.linalg.norm(residual / scale)) if scale else 0.0
        length = rn * bscale
        x = np.zeros(len(q))
        x[free] = d * beta * bscale
        x[j] = (b[rows[0]] - length) / a
        s = b - A @ x
        s[cones.eq] = 0
        s[rows[0]] = length
        s[rows[1:]] = bscale * residual
        z = np.zeros(len(b))
        z[rows[0]] = zhead
        if rn:
            z[rows[1:]] = -zhead * residual / rn
            z[cones.eq] = zhead * eta / rn
        out = {"status": "solved", "x": x, "s": s, "z": z}
        return out if check(out) else None
    except (ValueError, RuntimeError, la.LinAlgError, FloatingPointError, OverflowError):
        return None
