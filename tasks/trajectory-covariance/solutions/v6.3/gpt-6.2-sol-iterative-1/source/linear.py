"""Conditional autoregressive fit with uncertainty control at frequency zero."""

import numpy as np

from operators import poisson_fit, sym


def linear_cov(y, order):
    n, d = y.shape
    nr = n - order
    if order == 1:
        x = y[:-1].copy()
    else:
        # Use the same set of rows for all lags, newest predictor first.
        windows = np.lib.stride_tricks.sliding_window_view(y[:-1], order, axis=0)
        x = np.ascontiguousarray(windows[:, :, ::-1].transpose(0, 2, 1)).reshape(nr, -1)
    z = y[order:].copy()
    x -= x.mean(axis=0)
    z -= z.mean(axis=0)
    ne = nr - x.shape[1] - 1
    l = np.linalg.cholesky(sym(x.T @ x))
    q = np.linalg.solve(l, x.T @ z)
    b = np.linalg.solve(l.T, q).reshape(order, d, d)
    if order == 1 and np.max(np.abs(np.linalg.eigvals(b[0]))) >= 1:
        raise np.linalg.LinAlgError("unstable fitted autoregression")
    e = sym(z.T @ z - q.T @ q) / ne
    root = np.linalg.solve(np.eye(d) - b.sum(axis=0).T, np.linalg.cholesky(e))
    if order == 1:
        rev = poisson_fit(y, 0.0, nonlinear=False)
        if rev is not None:
            root = np.linalg.cholesky(sym(0.28 * (root @ root.T) + 0.72 * rev[0]))

    # In units of the fitted long-run covariance, uncertainty in the
    # zero-frequency filter is governed by the regression design below.
    # Apply a tempered second-moment correction to the relative matrix loss.
    design = np.linalg.solve(l, np.tile(root, (order, 1)))
    u, v = np.linalg.eigh(sym(design.T @ design))
    u = np.clip(0.56 * u, 0, 0.25)
    mean = 1 + d * u
    mat = np.diag(mean**2 + d*u**2 + 2*u + 1/ne)
    mat += np.outer(mean, u) + (u + 1/ne)[:, None]
    tuned = np.clip(np.linalg.solve(mat, mean), 0.06, 1.0)
    root = root @ v
    return sym((root * tuned) @ root.T)
