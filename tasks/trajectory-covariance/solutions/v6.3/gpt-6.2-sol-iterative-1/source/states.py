"""Transition-count estimation for trajectories with recurrent observations."""

import numpy as np

from finite_pool import pooled_model
from finite_risk import damp


def sym(a):
    return 0.5 * (a + a.T)


def directed(counts, freq, states):
    m, d = states.shape
    counts += 0.5 * np.outer(freq, freq)
    p = counts / counts.sum(axis=1, keepdims=True)
    a = np.eye(m) - p + freq
    pi = np.linalg.solve(a.T, freq)
    if pi.min() <= 0:
        return None
    f = states - pi @ states
    g = np.linalg.solve(a, f)
    pg = p @ g
    return sym(g.T @ (pi[:, None] * g) - pg.T @ (pi[:, None] * pg))


def discrete(x):
    n, d = x.shape
    loc = np.r_[0, 1 + np.flatnonzero(np.any(x[1:] != x[:-1], axis=1))]
    if len(loc) == 1:
        return np.zeros((d, d))
    probe = loc[:2048]
    distinct = len(np.unique(x[probe], axis=0))
    if len(probe) - distinct < max(3, int(0.02 * len(probe))):
        return None
    states, states_at_run = np.unique(x[loc], return_inverse=True, axis=0)
    m = len(states)
    if m > 6000 or len(loc) - m < max(3, int(0.02 * len(loc))):
        return None
    i, j = states_at_run[:-1], states_at_run[1:]
    states -= states[0]
    durations = np.diff(np.r_[loc, n])
    diag = np.bincount(states_at_run, weights=durations - 1, minlength=m)
    occ = np.bincount(states_at_run, weights=durations, minlength=m)
    occ[states_at_run[[0, -1]]] -= 0.5
    if states_at_run[0] == states_at_run[-1]:
        occ[states_at_run[0]] -= 0.5
    freq = occ / (n-1)
    f = states - freq @ states
    c = f.T @ (freq[:, None] * f)
    rhs = (freq[:, None] * f) * (n-1)
    smooth = 0.5
    pooled = None

    if m <= 320:
        counts = np.bincount(i * m + j, minlength=m*m).reshape(m, m).astype(float)
        tot = counts + counts.T
        a, b = np.triu_indices(m, k=1)
        has = tot[a, b] > 0
        u, v = a[has], b[has]
        skew = ((counts[u, v] - counts[v, u])**2 / tot[u, v]).sum()
        df = max(0, len(u) - m + 1)
        if skew > df + 3.2*np.sqrt(2*df + 1) + 3:
            counts[np.diag_indices(m)] += diag
            return directed(counts, freq, states)
        pooled = pooled_model(counts, diag, freq, states, n)
        weight = 0.5 * tot
        degree = weight.sum(axis=1)
        mat = -weight
        mat[np.diag_indices(m)] = degree + smooth * freq
        g = np.linalg.solve(mat, rhs)
    else:
        from scipy.sparse import coo_matrix, diags
        from scipy.sparse.linalg import cg, spsolve

        flow = coo_matrix((np.ones(len(i)), (i, j)), shape=(m, m)).tocsc()
        weight = (flow + flow.T) * 0.5
        degree = np.asarray(weight.sum(axis=1)).ravel()
        mat = diags(degree + smooth * freq, format="csc") - weight
        if m <= 900:
            g = spsolve(mat, rhs)
        else:
            precond = diags(1 / (degree + smooth * freq), format="csc")
            g = np.column_stack([cg(mat, r, M=precond, atol=0,
                                     rtol=1e-7, maxiter=500)[0] for r in rhs.T])

    h = sym((2.0 / (n-1)) * rhs.T @ g - c)
    if m <= 200:
        try:
            tot = weight.copy()
            tot[np.diag_indices(m)] = diag
            h = damp(h, f, g, mat, tot, n)
        except np.linalg.LinAlgError:
            pass
    if pooled is not None:
        weight, reduced = pooled
        h = (1-weight)*h + weight*reduced
    e, v = np.linalg.eigh(h)
    return sym((v * np.maximum(0.0, e)) @ v.T)
