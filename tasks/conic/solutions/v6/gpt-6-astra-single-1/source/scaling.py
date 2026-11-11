"""Diagonal equilibration preserving each product cone."""

import numpy as np
from scipy import sparse

from problem import Problem, blocks
from quality import norm


def equilibrate(p, iterations=6):
    A = p.A.copy()
    P = p.P.copy()
    q, b = p.q.copy(), p.b.copy()
    D, E = np.ones(p.n), np.ones(p.m)
    cost = 1.0 / max(1e-6, norm(q), norm(P.data))
    P.data *= cost
    q *= cost
    cone_blocks = list(blocks(p.cones))
    for iteration in range(iterations):
        ac = np.asarray(abs(A).max(axis=0).toarray()).ravel() if p.m else np.zeros(p.n)
        pc = np.asarray(abs(P).max(axis=0).toarray()).ravel() if P.nnz else np.zeros(p.n)
        dc = 1.0 / np.sqrt(np.clip(np.maximum(ac, pc), 1e-4, 1e4))
        dc[(ac == 0) & (pc == 0)] = 1.0
        A = A @ sparse.diags(dc, format="csc")
        P = sparse.diags(dc, format="csc") @ P @ sparse.diags(dc, format="csc")
        q *= dc
        D *= dc
        ar = np.asarray(abs(A).max(axis=1).toarray()).ravel() if p.n else np.zeros(p.m)
        er = 1.0 / np.sqrt(np.clip(ar, 1e-4, 1e4))
        er[ar == 0] = 1.0
        for kind, dim, sl in cone_blocks:
            if kind not in ("zero", "nonnegative"):
                er[sl] = 1.0 / np.sqrt(max(1e-4, np.max(ar[sl])))
        A = sparse.diags(er, format="csc") @ A
        b *= er
        E *= er
    return Problem(P.tocsc(), q, A.tocsc(), b, p.cones), D, E, cost


def recover(result, D, E, cost):
    r = dict(result)
    if "x" in r:
        r["x"] = D * r["x"]
    if "s" in r:
        r["s"] = r["s"] / E
    if "z" in r:
        r["z"] = E * r["z"] / cost
    return r
