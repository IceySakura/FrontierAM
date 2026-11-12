"""Reversibility diagnostics and dynamic covariance-shape regularization."""

import numpy as np
from scipy import linalg


def reversible(x):
    """Check a blocked time-reversal asymmetry statistic."""
    n, d = x.shape
    ij = [(i, j) for i in range(d) for j in range(i+1, d)]
    if not ij:
        return True
    scores = np.column_stack([
        x[1:, i]*x[:-1, j] - x[:-1, i]*x[1:, j] for i, j in ij
    ])
    batch = max(16, min(64, (n-1)//30))
    nb = len(scores)//batch
    means = scores[:nb*batch].reshape(nb, batch, -1).mean(axis=1)
    mean = means.mean(axis=0)
    covariance = np.atleast_2d(np.cov(means, rowvar=False))
    covariance.flat[::len(ij)+1] += 1e-9
    statistic = nb*mean@linalg.solve(
        covariance, mean, assume_a="pos", check_finite=False
    )
    q = len(ij)
    return statistic < q+4*np.sqrt(2*q)+10


def isotropic_shrink(h, n, noise_scale=1.0):
    d = len(h)
    if d < 3:
        return h, 1.0
    w, v = linalg.eigh(h, check_finite=False)
    w = np.maximum(w, 1e-12)
    # Effective rates stabilize pooling decisions for very long time scales.
    rates = (w-1.0)/(w+1.0)
    center = rates.mean()
    distance = np.sum((rates-center)**2)
    degrees = d*(d+1)/2-1
    noise = (degrees-2)*max(1-center*center, 1e-8)*noise_scale/n
    retained = max(0.0, 1.0-noise/max(distance, 1e-30))
    if retained > .9:
        return h, 1.0
    rates = center+retained*(rates-center)
    spectrum = (1+rates)/(1-rates)
    result = (v*spectrum)@v.T
    return (result+result.T)*.5, retained
