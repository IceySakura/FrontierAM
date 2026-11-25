"""Estimate gradient directions inside the positive-payoff region."""

import numpy as np

from cubature import grad_batch, normals


def positive_directions(s, zp, yp, rng):
    z, y = zp, yp
    if np.sum(yp > 0) < 13:
        z2 = normals(rng, 9, s.d)
        z, y = np.vstack([z, z2]), np.r_[y, s.query(z2)]
    idx = np.flatnonzero(y > 0)
    if not len(idx):
        return None
    idx = idx[np.argsort(y[idx])]
    idx = idx[np.linspace(0, len(idx)-1, min(16, len(idx)), dtype=int)]
    g = grad_batch(s, z[idx], step=1.5e-5)
    ng = np.linalg.norm(g, axis=1)
    good = (ng > 1e-8) & (ng < 8*max(1e-8, np.median(ng)))
    if not np.any(good):
        return None
    g = g[good] / ng[good, None]
    v = g.mean(axis=0)
    v /= max(np.linalg.norm(v), 1e-50)
    return v, g, z, y
