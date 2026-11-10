"""Intercept VAR(1) least squares with a residual ML covariance."""
import sys
import numpy as np

x = np.load(sys.argv[1], allow_pickle=False)
u = x[:-1] - x[:-1].mean(axis=0)
v = x[1:] - x[1:].mean(axis=0)
q, r = np.linalg.qr(u, mode="reduced")
lag = np.linalg.solve(r, q.T @ v)
innovations = v - u @ lag
transfer = np.eye(x.shape[1]) - lag
long_innovations = np.linalg.solve(transfer.T, innovations.T).T
estimate = long_innovations.T @ long_innovations / len(u)
np.save(sys.argv[2], np.ascontiguousarray(estimate, dtype="<f8"), allow_pickle=False)
