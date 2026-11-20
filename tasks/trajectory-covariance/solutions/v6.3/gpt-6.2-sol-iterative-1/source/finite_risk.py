"""Sampling uncertainty for the reversible transition-count estimator."""

import numpy as np


def damp(h, f, g, mat, count, n):
    m, d = f.shape
    l = np.linalg.cholesky(h)
    il = np.linalg.solve(l, np.eye(d))
    f = f @ il.T
    g = g @ il.T
    i, j = np.nonzero(count)
    weights = count[i, j] / (n - 1)
    gv = g[i] - g[j]
    product = f[:, :, None] * g[:, None, :]
    dif = 2*(product + product.swapaxes(1, 2)) - f[:, :, None]*f[:, None, :]
    score = 0.5*(dif[i] + dif[j]) - gv[:, :, None]*gv[:, None, :]
    score -= np.einsum("i,ijk->jk", weights, score)
    b = np.zeros((m, d*d))
    np.add.at(b, i, weights[:, None]*score.reshape(-1, d*d))
    b -= b.mean(axis=0)
    tail = np.linalg.solve(mat, b).reshape(m, d, d)
    v = np.einsum("k,kij,kjl->il", weights, score, score, optimize=True)
    v += 2*(n-1)*np.einsum("kij,kjl->il", b.reshape(m, d, d), tail, optimize=True)
    e, u = np.linalg.eigh((v + v.T) / (2*n))
    gain = 1 / (1 + 0.64*np.maximum(0, e))
    rt = l @ u
    return (rt * gain) @ rt.T
