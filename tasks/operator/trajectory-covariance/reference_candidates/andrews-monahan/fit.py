"""Prewhitened quadratic-spectral HAC estimator (Andrews & Monahan 1992; bandwidth of Andrews 1991).

1. Prewhiten with a centred VAR(1); singular values of the coefficient matrix above 0.97 are set to 0.97 (the
   Andrews-Monahan eigenvalue adjustment, in its singular-value form).
2. Residual long-run covariance with the QS kernel and the AR(1) plug-in bandwidth
   S = 1.3221 (alpha(2) n)^(1/5), alpha(2) = sum 4 rho^2 s^4 / (1 - rho)^8 / sum s^4 / (1 - rho)^4 (equal weights).
3. Recolour: (I - A)^{-1} Omega (I - A)^{-T}.
"""

import sys

import numpy as np


def qs(z: np.ndarray) -> np.ndarray:
    out = np.ones_like(z)
    nz = z != 0
    a = 6 * np.pi * z[nz] / 5
    out[nz] = 25 / (12 * np.pi ** 2 * z[nz] ** 2) * (np.sin(a) / a - np.cos(a))
    return out


def autocovariances(e: np.ndarray) -> np.ndarray:
    """Gamma_j = sum_t e_{t+j} e_t^T / n for j = 0..n-1, via FFT."""
    n, d = e.shape
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(e, size, axis=0)
    cross = np.fft.irfft(f[:, :, None] * np.conj(f[:, None, :]), size, axis=0)[:n]
    return cross / n


def estimate(x: np.ndarray) -> np.ndarray:
    m, d = x.shape
    x = x - x.mean(axis=0)
    u, v = x[:-1], x[1:]
    coef, *_ = np.linalg.lstsq(u, v, rcond=None)
    left, sv, right = np.linalg.svd(coef.T)
    a = left @ np.diag(np.minimum(sv, 0.97)) @ right
    e = v - u @ a.T
    n = len(e)
    rho = np.array([np.dot(e[1:, i], e[:-1, i]) / np.dot(e[:-1, i], e[:-1, i]) for i in range(d)])
    rho = np.clip(rho, -0.97, 0.97)
    s2 = np.array([np.mean((e[1:, i] - rho[i] * e[:-1, i]) ** 2) for i in range(d)])
    alpha = np.sum(4 * rho ** 2 * s2 ** 2 / (1 - rho) ** 8) / np.sum(s2 ** 2 / (1 - rho) ** 4)
    bandwidth = 1.3221 * (alpha * n) ** 0.2
    gamma = autocovariances(e)
    weights = qs(np.arange(n) / bandwidth)
    omega = gamma[0] + np.einsum('j,jab->ab', weights[1:], gamma[1:] + np.transpose(gamma[1:], (0, 2, 1)))
    inverse = np.linalg.inv(np.eye(d) - a)
    h = inverse @ omega @ inverse.T
    return (h + h.T) / 2


if __name__ == '__main__':
    data = np.load(sys.argv[1], allow_pickle=False).astype(float)
    np.save(sys.argv[2], np.ascontiguousarray(estimate(data), dtype='<f8'), allow_pickle=False)
