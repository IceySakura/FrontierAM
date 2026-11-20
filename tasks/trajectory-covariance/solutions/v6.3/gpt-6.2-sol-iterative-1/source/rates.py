"""Shrink poorly separated relaxation rates in a symmetric transition fit."""

import numpy as np


def clustered_time(r, n, noise):
    """Average nearby time scales without forcing well-separated modes together."""
    d = len(r)
    t = 1 / r
    if d < 2 or d > 10:
        return t
    costs = np.zeros((d, d+1))
    centers = np.zeros_like(costs)
    for i in range(d):
        for j in range(i+1, d+1):
            a = r[i:j]
            m = len(a)
            tr = t[i:j].mean()
            rc = a.mean()
            v = max(rc*(2-rc), 1e-8) * noise / n
            diff = a-rc
            dispersion = diff @ diff / v
            df = m * (m + 1) / 2 - 1
            # Soft model averaging over possible contiguous rate groups.
            costs[i, j] = 0.48 * (1.48*df - dispersion)
            ps = np.exp(np.clip(0.48 * (1.25 - m*(1-rc)**2 / v), -40, 10))
            if abs(rc - 1) < 0.12:
                tr = (tr + 0.60*ps) / (1 + 0.60*ps)
                costs[i, j] += np.log1p(0.60*ps)
            centers[i, j] = tr

    left = np.zeros(d+1)
    right = np.zeros(d+1)
    for j in range(1, d+1):
        left[j] = np.logaddexp.reduce(left[:j] + costs[:j, j])
    for i in range(d-1, -1, -1):
        right[i] = np.logaddexp.reduce(costs[i, i+1:] + right[i+1:])
    h = np.zeros(d)
    for i in range(d):
        for j in range(i+1, d+1):
            prob = np.exp(left[i] + costs[i, j] + right[j] - left[d])
            h[i:j] += prob * centers[i, j]
    return h
