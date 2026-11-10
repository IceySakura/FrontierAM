"""Fully corrective conditional gradients for low-rank polyhedral QPs."""
import os
import sys
import time

import numpy as np
from scipy import sparse

from solver import highs_model, native_highs, norm, quality


def solve_fw(p):
    if not (2500 <= p.n <= 30000 and 0 < p.P.nnz < 5 * p.n):
        return None
    if np.count_nonzero(p.equal) < 0.5 * p.n:
        return None
    started = time.monotonic()
    opt, x, _ = highs_model(
        p.q, p.A, np.where(p.equal, p.b, -np.inf), p.b,
        solver="ipm", time_limit=8.0)
    h = native_highs()
    if opt.getModelStatus() != h.HighsModelStatus.kOptimal:
        return None
    opt.setOptionValue("solver", "simplex")
    cols = np.arange(p.n, dtype=np.int32)
    vertices = [x.copy()]
    products = [p.P @ x]
    costs = [p.q @ x]
    for it in range(70):
        gradient = p.P @ x + p.q
        opt.changeColsCost(p.n, cols, gradient)
        opt.run()
        if opt.getModelStatus() != h.HighsModelStatus.kOptimal:
            return None
        s = opt.getSolution()
        v, z = np.asarray(s.col_value), -np.asarray(s.row_dual)
        slack = p.b - p.A @ x
        slack[p.equal] = 0.0
        r = {"status": "solved", "x": x.copy(), "s": slack, "z": z}
        if quality(p, r):
            return r
        if os.environ.get("SOLVER_DEBUG"):
            print("corrective QP", it, "gap", gradient @ (x - v),
                  "time", time.monotonic() - started, file=sys.stderr)
        vertices.append(v.copy())
        products.append(p.P @ v)
        costs.append(p.q @ v)
        V = np.column_stack(vertices)
        PV = np.column_stack(products)
        directions = V - x[:, None]
        product_directions = PV - (p.P @ x)[:, None]
        G = directions.T @ product_directions
        G = (G + G.T) * 0.5
        ridge = 1e-12 * max(1.0, norm(G))
        G.flat[::G.shape[0] + 1] += ridge
        k = len(vertices)
        _, weights, _ = highs_model(
            gradient @ directions, sparse.csc_matrix(np.ones((1, k))),
            np.ones(1), np.ones(1), sparse.csc_matrix(G),
            lower=np.zeros(k), upper=np.full(k, np.inf), time_limit=1.0)
        weights = np.maximum(weights, 0)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            return None
        weights /= total
        x = V @ weights
        if time.monotonic() - started > 7.0:
            break
    return None
