"""Equality-multiplier refinement using matrix-free LSQR."""

import math
import numpy as np


def dual(p, result, maxiter=1200):
    from minisparse import CSC
    eq = np.flatnonzero(p.eq)
    if not len(eq) or result["status"] != "solved":
        return None
    if not hasattr(p, "_refine_F"):
        ad, ar, ap, ac = p.A
        A = CSC(p.m, p.n, ad.copy(), ar.copy(), ap.copy()).canonical()
        F = A.select_rows(eq)
        squares = np.bincount(F.rows, weights=F.data ** 2, minlength=len(eq))
        sc = 1 / np.sqrt(np.maximum(squares, 1e-100))
        p._refine_F = F.scale(row=sc), sc
    F, sc = p._refine_F
    b = -(p.mul("P", result["x"]) + p.q + p.mul("A", result["z"], True))
    beta = np.linalg.norm(b)
    if not np.isfinite(beta) or beta == 0:
        return None
    u = b / beta
    v = F.mul(u)
    alpha = np.linalg.norm(v)
    if alpha == 0 or not np.isfinite(alpha):
        return None
    v /= alpha
    w, x = v.copy(), np.zeros(len(eq))
    phibar, rhobar = beta, alpha
    trial = dict(result)
    last = beta
    stagnant = 0
    for it in range(maxiter):
        u = F.tmul(v) - alpha * u
        beta = np.linalg.norm(u)
        if beta:
            u /= beta
        v = F.mul(u) - beta * v
        alpha = np.linalg.norm(v)
        if alpha:
            v /= alpha
        rho = math.hypot(rhobar, beta)
        if rho == 0 or not math.isfinite(rho):
            break
        c, s = rhobar / rho, beta / rho
        theta, rhobar = s * alpha, -c * alpha
        phi, phibar = c * phibar, s * phibar
        x += (phi / rho) * w
        w = v - (theta / rho) * w
        if it % 20 == 19 or it == 4:
            z = result["z"].copy()
            z[eq] += sc * x
            trial["z"] = z
            if p.valid(trial, tol=8.8e-9):
                return trial
            error = np.linalg.norm(F.tmul(x) - b)
            if error > last * .999:
                stagnant += 1
            else:
                stagnant = 0
            last = error
            if stagnant > 5:
                break
    trial["z"] = result["z"].copy()
    trial["z"][eq] += sc * x
    if p.valid(trial, tol=8.8e-9):
        return trial
    return None
