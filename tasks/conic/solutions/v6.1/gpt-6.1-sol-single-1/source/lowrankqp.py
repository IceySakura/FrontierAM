"""Fully corrective conditional gradients for sparse low-rank diagonal QPs."""

import time
import numpy as np


def solve(p, maxiter=100):
    from fast import engine, core, pass_raw, log, norm, remaining
    pd, pr, pp, pc = p.P
    if (p.n < 1500 or len(pd) > 400 or len(pd) == 0 or
            np.any(pr != pc) or np.any(pd <= 0)):
        return None
    mod = core()
    ad, ar, ap, ac = p.A
    h = engine()
    costscale = 1 / max(1., norm(p.q))
    pass_raw(h, p.n, p.m, p.q * costscale, np.where(p.eq, p.b, -mod.kHighsInf),
             p.b, ad, ar, ap)
    h.run()
    if h.getModelStatus() != mod.HighsModelStatus.kOptimal:
        return None
    x = np.asarray(h.getSolution().col_value)
    X = x[:, None]
    costs = np.array([p.q @ x])
    begin = time.monotonic()
    for it in range(maxiter):
        px = np.zeros(p.n)
        np.add.at(px, pc, pd * x[pc])
        grad = p.q + px
        costscale = 1 / max(1., norm(grad))
        h.changeColsCost(p.n, np.arange(p.n, dtype=np.int32), grad * costscale)
        h.run()
        sol = h.getSolution()
        if h.getModelStatus() != mod.HighsModelStatus.kOptimal:
            break
        vertex = np.asarray(sol.col_value)
        z = -np.asarray(sol.row_dual) / costscale
        z[~p.eq] = np.maximum(z[~p.eq], 0)
        r = p.result(x, z)
        if p.valid(r, tol=8e-9):
            log("lowrank qp solved", it, time.monotonic() - begin)
            return r
        gap = float(grad @ (x - vertex))
        if it % 10 == 0:
            log("lowrank qp", it, "gap", gap, "vertices", X.shape[1],
                "t", time.monotonic() - begin)
        if gap <= 1e-12 or time.monotonic() - begin > 30 or remaining() < 1.:
            break
        X = np.column_stack((X, vertex))
        costs = np.r_[costs, p.q @ vertex]
        V = X[pc] * np.sqrt(pd)[:, None]
        G = V.T @ V
        cq = costs - costs[0]
        scale = 1e6 / max(1., norm(G), norm(cq))
        count = len(costs)
        mh = engine()
        gd = (G * scale).ravel(order="F")
        gi = np.tile(np.arange(count, dtype=np.int32), count)
        gc = np.repeat(np.arange(count, dtype=np.int32), count)
        gp = np.arange(0, count * count + 1, count, dtype=np.int32)
        pass_raw(mh, count, 1, cq * scale, np.ones(1), np.ones(1),
                 np.ones(count), np.zeros(count, np.int32),
                 np.arange(count + 1, dtype=np.int32),
                 col_lower=np.zeros(count), P=(gd, gi, gp, gc))
        mh.run()
        weights = np.asarray(mh.getSolution().col_value)
        if len(weights) != count or not np.isfinite(weights).all():
            break
        weights = np.maximum(weights, 0)
        if np.sum(weights) == 0:
            break
        weights /= np.sum(weights)
        active = weights > 1e-10
        ii = np.flatnonzero(active)
        if len(ii):
            K = np.block([[G[np.ix_(ii, ii)] * scale, np.ones((len(ii), 1))],
                          [np.ones((1, len(ii))), np.zeros((1, 1))]])
            try:
                solw = np.linalg.solve(K, np.r_[-cq[ii] * scale, 1.])[:-1]
                if np.min(solw) >= -1e-10:
                    weights[:] = 0
                    weights[ii] = np.maximum(solw, 0)
                    weights /= np.sum(weights)
            except np.linalg.LinAlgError:
                pass
        x = X @ weights
        active = weights > 1e-12
        X, costs = X[:, active], costs[active]
    return None
