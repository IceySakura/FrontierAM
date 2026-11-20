"""Small-order vector ARMA fits using Gaussian conditional innovations."""

import ctypes
from pathlib import Path
from time import monotonic

import numpy as np


def companion(coefficients):
    p, d, _ = coefficients.shape
    a = np.eye(p * d, k=-d)
    a[:d] = np.concatenate(coefficients.transpose(0, 2, 1), axis=1)
    return a


def radius(coefficients):
    if len(coefficients) == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(companion(coefficients)))))


class Filter:
    def __init__(self):
        lib = ctypes.CDLL(str(Path(__file__).resolve().with_name("libinnovations.so")))
        mat = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        i = ctypes.c_int
        lib.innovations.argtypes = i, i, i, mat, mat, mat
        lib.innovations.restype = None
        lib.innovation_adjoint.argtypes = i, i, i, mat, mat
        lib.innovation_adjoint.restype = None
        self.lib = lib

    def apply(self, x, b):
        e = np.empty_like(x)
        self.lib.innovations(*x.shape, len(b), b, x, e)
        return e

    def adjoint(self, score, b):
        self.lib.innovation_adjoint(*score.shape, len(b), b, score)


_FILTER = None


class BudgetReached(Exception):
    pass


def innovations_filter():
    global _FILTER
    if _FILTER is None:
        _FILTER = Filter()
    return _FILTER


def high_order_innovations(x, m):
    from scipy.fft import irfft, next_fast_len, rfft
    from scipy.linalg import solve, toeplitz

    n, d = x.shape
    nt = next_fast_len(2 * n - 1)
    f = rfft(x, n=nt, axis=0)
    r = np.empty((m + 1, d, d))
    for i in range(d):
        for j in range(i, d):
            value = irfft(f[:, i].conj() * f[:, j], n=nt)
            r[:, i, j] = value[:m + 1]
            r[:, j, i] = value[np.r_[0, np.arange(nt - 1, nt - m - 1, -1)]]
    r /= n - np.arange(m + 1)[:, None, None]
    mat = np.empty((m, d, m, d))
    for i in range(d):
        for j in range(d):
            mat[:, i, :, j] = toeplitz(r[:m, i, j], r[:m, j, i])
    mat = mat.reshape(m * d, m * d)
    tr = np.trace(mat) / (m*d)
    target = r[1:].reshape(m * d, d)
    coef = None
    for ridge in (0.0, 1e-5, 1e-3, 0.01):
        try:
            coef = solve(mat + np.eye(m*d)*(tr*ridge), target, assume_a="pos").reshape(m,d,d)
            break
        except np.linalg.LinAlgError:
            pass
    if coef is None:
        return x
    e = x.copy()
    for j, a in enumerate(coef, start=1):
        e[m:] -= x[m - j:n - j] @ a
    return e


def conditional_var(x, p, burn):
    n, d = x.shape
    y = x[burn:]
    if p > 0:
        xx = np.concatenate([x[burn-j:n-j] for j in range(1, p+1)], axis=1)
        xx -= xx.mean(axis=0)
        a = np.linalg.lstsq(xx.T @ xx, xx.T @ y, rcond=1e-11)[0]
        e = y - xx @ a
    else:
        e = y.copy()
    e -= e.mean(axis=0)
    return np.linalg.slogdet(e.T @ e / len(e))[1]


def fit(x, p, q, pilot, burn, budget=60, start=None, deadline=np.inf):
    from scipy.optimize import minimize

    filt = innovations_filter()
    n, d = x.shape
    edf = n - burn
    x_past = np.zeros((n, p*d))
    for j in range(1, p+1):
        x_past[j:, (j-1)*d:j*d] = x[:n-j]
    if start is None:
        predictors = [x_past[burn:]] + [pilot[burn-j:n-j] for j in range(1,q+1)]
        des = np.concatenate(predictors, axis=1)
        des -= des.mean(axis=0)
        a0 = np.linalg.lstsq(des.T @ des, des.T @ x[burn:], rcond=1e-11)[0]
        a0 = a0.reshape(p+q,d,d)
    else:
        a0 = start.copy()

    # For higher-order blocks the common power scaling keeps the
    # companion roots at the same relative positions.
    bound = min(0.997, 1.0 - 8.0 / n)
    r = radius(-a0[p:])
    if r >= bound:
        a0[p:] *= (0.94 / r)**np.arange(1, q+1)[:,None,None]
    ar = radius(a0[:p])
    if ar >= bound:
        a0[:p] *= (min(0.993, bound * 0.995) / ar)**np.arange(1, p+1)[:,None,None]

    score = np.empty_like(x)
    last = [np.inf, None, None]

    def objective(step):
        if monotonic() >= deadline:
            raise BudgetReached
        stack = np.ascontiguousarray(step.reshape(p+q,d,d))
        a, b = stack[:p], stack[p:]
        if radius(-b) >= bound or radius(a) > min(0.9999, 1 - 0.4/n):
            return 1e6, np.zeros_like(step)
        r = np.ascontiguousarray(x - x_past @ a.reshape(p*d,d)) if p else x
        e = filt.apply(r, b)
        ev = e[burn:] - e[burn:].mean(axis=0)
        c = ev.T @ ev / edf
        sign, logdet = np.linalg.slogdet(c)
        if sign <= 0 or not np.isfinite(logdet):
            return 1e6, np.zeros_like(step)
        if logdet < last[0]:
            last[:] = [logdet, stack.copy(), ev]
        score.fill(0)
        score[burn:] = ev @ np.linalg.solve(c, np.eye(d))
        filt.adjoint(score, b)
        grad = np.empty_like(stack)
        if p:
            grad[:p] = (x_past.T @ score).reshape(p,d,d)
        for j in range(1, q+1):
            grad[p+j-1] = e[:n-j].T @ score[j:]
        return logdet, (-2.0 / edf) * grad.ravel()

    try:
        minimize(objective, a0.ravel(), jac=True, method="L-BFGS-B",
                 options={"maxfun": budget, "maxiter": budget, "ftol": 2e-9, "gtol": 5e-6,
                          "maxls": 12, "maxcor": 9})
    except BudgetReached:
        pass
    score, stack, e = last
    if stack is None:
        return None
    a, b = stack[:p], stack[p:]
    ident = np.eye(d)
    transfer = (ident + b.sum(axis=0)) @ np.linalg.solve(ident - a.sum(axis=0), ident)
    cov = e.T @ e / (edf - (p+q)*d - 1)
    h = transfer.T @ cov @ transfer
    return score, h, e, transfer, stack


def try_arma(x, ar_order, ma_order):
    n, d = x.shape
    deadline = monotonic() + 2.65
    burn = min(200, max(75, int(np.sqrt(n))))
    if ma_order is not None:
        candidates = [(0, ma_order)]
        if ma_order == 0:
            return None
    else:
        candidates = [(1, 1)]
    m = min(50, max(16, int(3.3 * n**0.245)), n // (15*d))
    pilot = high_order_innovations(x, m)
    best_score = (n-burn) * conditional_var(x, ar_order, burn) + 2.5*ar_order*d*d
    best = None
    for p, q in candidates:
        fit0 = fit(x, p, q, pilot, burn, budget=72, deadline=deadline)
        if fit0 is None:
            continue
        merit = (n-burn)*fit0[0] + 2.5*(p+q)*d*d
        if merit < best_score:
            best_score = merit
            best = (p, q, *fit0[1:4])
    if best is not None and best[0] == 1 and ar_order >= 5 and monotonic() + .05 < deadline:
        p, q = 2, 1
        fit0 = fit(x, p, q, pilot, burn, budget=65, deadline=deadline)
        if fit0 is not None:
            merit = (n-burn)*fit0[0] + 2.5*(p+q)*d*d
            if merit < best_score:
                best = (p, q, *fit0[1:4])
    return best
