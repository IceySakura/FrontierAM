#!/usr/bin/env python3
"""Long-run covariance (HAC) estimator for a single trajectory.

Usage:
    python3 lrcov.py INPUT.npy OUTPUT.npy

Reads one (M, d) float64 C-order .npy array and writes the (d, d) long-run
covariance estimate H as a finite, symmetric, positive-semidefinite float64
C-order .npy array.  H / M estimates the covariance of the sample mean and
sqrt(H[j, j] / M) the standard error of component j.

Algorithm
---------
Heteroskedasticity-and-autocorrelation-consistent estimator with

  * VAR(1) pre-whitening (Andrews & Monahan, 1992) to remove the dominant
    geometric autocorrelation that MCMC and linear-dynamics samplers exhibit;
  * the quadratic-spectral kernel (Andrews, 1991), which is MSE-optimal among
    the usual kernels for the residual series;
  * a data-dependent bandwidth from the AR(1) plug-in rule, driven by the
    spectral radius of the fitted VAR(1) coefficient matrix.  That quantity is
    invariant under invertible linear re-scalings of the coordinates, so the
    estimate is affine equivariant (a per-trajectory linear change of input
    coordinates changes the output by the same congruence);
  * a multivariate portmanteau pre-test at lag 1 that collapses the bandwidth
    to zero (i.e. to the plain sample covariance) when the trajectory carries
    no detectable serial dependence.  Without it the data-dependent bandwidth
    has a noise floor that over-smoothes independent observations.

The result is symmetrised and projected onto the positive-semidefinite cone.
"""

import sys

import numpy as np

try:
    from scipy import stats as _stats
except Exception:  # pragma: no cover
    _stats = None

DEFAULT = {
    "prewhiten": True,
    "kernel": "qs",
    "kappa": 1.0,
    "q": 0.95,
    "blend": 1.0,
}


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def _center(x):
    return x - x.mean(axis=0)


def autocov(x, maxlag):
    """Gamma[h] = (1/M) sum_{t=h+1}^{M} (x_t - xbar)(x_{t-h} - xbar)^T."""
    x = _center(x)
    M = x.shape[0]
    return [(x[h:].T @ x[:M - h]) / M for h in range(maxlag + 1)]


def fit_var1(x):
    """OLS fit of x_t = c + A x_{t-1} + e_t.

    Returns (A, Omega, spectral_radius) where Omega is the residual covariance
    (1/(M-1) normalised) and the spectral radius is max|eig(A)|.
    """
    x1, x2 = x[:-1], x[1:]
    m1 = x1.mean(axis=0)
    x1c, x2c = x1 - m1, x2 - m1
    S11 = x1c.T @ x1c
    S21 = x2c.T @ x1c
    A = np.linalg.solve(S11, S21.T).T
    e = x2c - x1c @ A.T
    Omega = (e.T @ e) / e.shape[0]
    rho = float(np.max(np.abs(np.linalg.eigvals(A)))) if A.size else 0.0
    return A, Omega, rho


def qs_weights(hmax, b):
    """Quadratic-spectral kernel weights (Andrews, 1991)."""
    h = np.arange(hmax + 1, dtype=float)
    w = np.ones(hmax + 1)
    if b > 0.0:
        x_ = h / b
        z = 6.0 * np.pi * x_ / 5.0
        m = x_ > 0.0
        w[m] = (25.0 / (12.0 * np.pi ** 2 * x_[m] ** 2)) * (
            np.sin(z[m]) / z[m] - np.cos(z[m]))
    else:
        w[1:] = 0.0
    return w


def bartlett_weights(hmax, b):
    h = np.arange(hmax + 1, dtype=float)
    if b > 0.0:
        return np.where(h <= b, 1.0 - h / (b + 1.0), 0.0)
    w = np.zeros(hmax + 1)
    w[0] = 1.0
    return w


def hac(gammas, weights):
    """H = sum_h w[h] (Gamma[h] + Gamma[h]^T), with w[0] on Gamma[0] once."""
    H = weights[0] * gammas[0]
    for h in range(1, len(gammas)):
        H = H + weights[h] * (gammas[h] + gammas[h].T)
    return H


def lag1_stat(g0, g1, M):
    """Hosking's multivariate portmanteau statistic at lag 1.

    Under white noise it is chi-squared with d^2 degrees of freedom.  Written
    as M * ||L^{-1} G1 L^{-T}||_F^2 with G0 = L L^T, which is invariant under
    invertible linear transformations of the coordinates.
    """
    d = g0.shape[0]
    L = np.linalg.cholesky(g0 + 1e-12 * np.eye(d))
    C = np.linalg.solve(L, g1)
    C = np.linalg.solve(L.T, C.T).T
    return float(M * np.sum(C * C))


def _chi2_ppf(p, df):
    if _stats is not None:
        return float(_stats.chi2.ppf(p, df))
    # Wilson-Hilferty normal approximation, used only if SciPy is absent.
    return df * (1.0 - 2.0 / (9.0 * df) + _norm_ppf(p) * np.sqrt(2.0 / (9.0 * df))) ** 3


def _norm_ppf(p):
    if _stats is not None:
        return float(_stats.norm.ppf(p))
    # Acklam's algorithm.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1.0 - 0.02425
    if p < pl:
        q = np.sqrt(-2.0 * np.log(p))
        return ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5] \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > ph:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        return -((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5] \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q \
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def andrews_bandwidth(rho, M, kernel="qs"):
    """Andrews (1991) AR(1) plug-in bandwidth."""
    rho = min(max(rho, 0.0), 0.999999)
    alpha = 4.0 * rho ** 2 / (1.0 - rho ** 2) ** 2
    if kernel == "qs":
        return 1.3221 * (alpha * M) ** 0.2
    return 1.1447 * (alpha * M) ** (1.0 / 3.0)


def relative_error(reference, prediction):
    """Whitened Frobenius error, matching the checker in samples/check.py."""
    if not np.isfinite(prediction).all():
        raise ValueError("prediction contains a nonfinite value")
    scale = max(1.0, np.abs(prediction).max())
    h = prediction / scale
    tol = 1e-10 * max(1.0 / scale, np.linalg.norm(h, "fro"))
    if np.linalg.norm(h - h.T, "fro") > tol:
        raise ValueError("prediction exceeds the symmetry tolerance")
    h = (h + h.T) / 2
    eig, vectors = np.linalg.eigh(h)
    if eig[0] < -tol:
        raise ValueError("prediction has a negative eigenvalue beyond tolerance")
    if eig[0] < 0:
        h = (vectors * np.maximum(eig, 0.0)) @ vectors.T
        accepted = h * scale
    elif np.array_equal(prediction, prediction.T):
        accepted = prediction
    else:
        accepted = h * scale
    difference = accepted - reference
    factor = np.linalg.cholesky(reference)
    left = np.linalg.solve(factor, difference)
    whitened = np.linalg.solve(factor, left.T).T
    return np.hypot.reduce(whitened.ravel())


def project_psd(H):
    """Symmetrise and project onto the positive-semidefinite cone."""
    H = 0.5 * (H + H.T)
    ev, Q = np.linalg.eigh(H)
    ev = np.maximum(ev, 0.0)
    return Q @ np.diag(ev) @ Q.T


# ---------------------------------------------------------------------------
# estimator
# ---------------------------------------------------------------------------

def estimate(x, prewhiten=True, kernel="qs", kappa=1.0, q=0.95, blend=1.0):
    """Long-run covariance estimate of the trajectory x (M, d).

    Parameters
    ----------
    prewhiten : bool
        VAR(1) pre-whitening with recolouring.
    kernel : "qs" | "bartlett"
        Kernel family.
    kappa : float
        Shrinkage of the AR(1) coefficient before turning it into a bandwidth:
        rho_eff = max(0, rho - kappa / sqrt(M)).  kappa > 0 suppresses the
        bandwidth noise floor for independent data.
    q : float
        Portmanteau pre-test quantile.  q = 1.0 disables the test.
    blend : float
        Weight on the pre-whitened estimate; 1 - blend on the direct one.
    """
    M, d = x.shape
    if M < 2:
        return project_psd(np.zeros((d, d), dtype=np.float64))
    x = _center(x)

    A, _Omega, rho = fit_var1(x)
    rho_eff = max(0.0, rho - kappa / np.sqrt(M))

    b = andrews_bandwidth(rho_eff, M, kernel)
    b = min(b, M - 1)

    if q < 1.0 and d > 1 and M > d * d + 2:
        g_short = autocov(x, 1)
        stat = lag1_stat(g_short[0], g_short[1], M)
        crit = _chi2_ppf(q, d * d)
        if stat <= crit:
            b = 0.0

    hmax = min(M - 1, max(3, int(np.ceil(b)) + 3))
    g = autocov(x, hmax)
    weights = (qs_weights if kernel == "qs" else bartlett_weights)(hmax, b)
    H_direct = hac(g, weights)

    if prewhiten and b > 0.0 and rho_eff > 0.0 and M > 2 * d + 2:
        x1, x2 = x[:-1], x[1:]
        m1 = x1.mean(axis=0)
        resid = (x2 - m1) - (x1 - m1) @ A.T
        e = np.vstack([np.zeros((1, d), dtype=x.dtype), resid])
        ge = autocov(e, hmax)
        Hu = hac(ge, weights)
        IA = np.eye(d, dtype=x.dtype) - A
        try:
            tmp = np.linalg.solve(IA, Hu)
            H_pw = np.linalg.solve(IA, tmp.T).T
        except np.linalg.LinAlgError:
            H_pw = H_direct
    else:
        H_pw = H_direct

    H = blend * H_pw + (1.0 - blend) * H_direct
    return project_psd(H)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def _as_float64_array(x):
    x = np.asarray(x)
    if x.dtype != np.float64:
        x = x.astype(np.float64)
    if not x.flags.c_contiguous:
        x = np.ascontiguousarray(x)
    return x


def main(argv):
    if len(argv) != 3:
        print("usage: lrcov.py INPUT.npy OUTPUT.npy", file=sys.stderr)
        return 2
    try:
        x = np.load(argv[1], allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        print("error: cannot read input: %s" % exc, file=sys.stderr)
        return 1
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.size == 0:
        print("error: input must be a 2-D .npy array", file=sys.stderr)
        return 1
    x = _as_float64_array(x)
    try:
        H = estimate(x, **DEFAULT)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError,
            OverflowError, ZeroDivisionError) as exc:
        print("error: estimation failed: %s" % exc, file=sys.stderr)
        return 1
    if not np.isfinite(H).all():
        print("error: estimate is not finite", file=sys.stderr)
        return 1
    H = np.ascontiguousarray(H, dtype=np.float64)
    np.save(argv[2], H)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
