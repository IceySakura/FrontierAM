"""Conservative likelihood-based pooling of homogeneous vector dynamics."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaincc


def pooled_estimate(x):
    n, d = x.shape
    count = n - 1
    before = x[:-1] - x[:-1].mean(axis=0)
    after = x[1:] - x[1:].mean(axis=0)
    xx = before.T @ before / count
    yy = after.T @ after / count
    xy = before.T @ after / count
    unrestricted = yy - xy.T @ np.linalg.solve(xx, xy)
    logdet_full = np.linalg.slogdet(unrestricted)[1]

    def profile(rho):
        omega = yy - rho * (xy + xy.T) + rho * rho * xx
        sign, logdet = np.linalg.slogdet(omega)
        return logdet if sign > 0 else 1e100

    optimum = minimize_scalar(
        profile,
        bounds=(-1 + 0.1 / n, 1 - 0.1 / n),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 100},
    )
    rho = optimum.x
    omega = yy - rho * (xy + xy.T) + rho * rho * xx
    statistic = max(0.0, count * (optimum.fun - logdet_full))
    degrees = d * d - 1
    weight = float(gammaincc(degrees * 0.5, statistic * 0.5)) if degrees > 0 else 0.0
    h = omega / (1 - rho) ** 2
    vals, vecs = np.linalg.eigh((h + h.T) * 0.5)
    h = (vecs * np.clip(vals, 0.0, n / 3.0)) @ vecs.T
    return h, weight
