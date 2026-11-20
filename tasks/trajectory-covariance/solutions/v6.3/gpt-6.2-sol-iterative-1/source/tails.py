"""Estimate departures from a reversible projected Poisson equation."""

import numpy as np


def residual_correction(y, g, base, tau):
    n, d = y.shape
    l = np.linalg.cholesky(base + 1e-12*np.eye(d))
    il = np.linalg.solve(l, np.eye(d))
    dl = (y[1:-1] + g[:-2] - g[1:-1]) @ il.T
    dr = (y[1:-1] + g[2:] - g[1:-1]) @ il.T
    # At a Markov state, backward and forward neighbours are conditionally
    # independent. These products therefore measure the Poisson residual.
    cap = min(95, max(3, int(tau)), n//24)
    horizons = [k for k in (0, 2, 5, 11, 23, 47, 95) if k <= cap]
    last = len(dl)
    ac = np.empty((horizons[-1] + 1, d, d))
    for k in range(len(ac)):
        cross = dl[:last-k].T @ dr[k:] / (last-k)
        ac[k] = (cross + cross.T)
    ml = np.mean(np.sum(dl**2, axis=1))
    mr = np.mean(np.sum(dr**2, axis=1))
    hscale = np.mean(np.sum(dl**2, axis=1) * np.sum(dr**2, axis=1))
    kurt = max(0.90, min(4.0, hscale / max(ml * mr, 1e-12)))
    df = d*(d+1)/2
    best = -1.0
    new = None
    for hh in horizons:
        w = (1 + np.cos(np.pi*np.arange(hh+1)/(hh+1))) * 0.5
        mat = np.einsum("i,ijk->jk", w, ac[:hh+1])
        e, v = np.linalg.eigh(mat)
        variance = (4*kurt/(n-2)) * np.sum(w*w)
        pe = np.maximum(e, 0)
        amplitude = pe @ pe
        score = amplitude / variance
        test = 0.50 * df + 0.90 * np.sqrt(df) + 1.2
        if score > max(best, test):
            # Include a tapered remainder only when above its sampling noise.
            re = np.maximum(0, pe - 0.60*np.sqrt(d*variance))
            conf = np.clip((score - test) / max(score, 1), 0.0, 1.0)
            new = (v * np.minimum(2.0, re*conf)) @ v.T
            best = score
    return base if new is None else base + l @ new @ l.T
