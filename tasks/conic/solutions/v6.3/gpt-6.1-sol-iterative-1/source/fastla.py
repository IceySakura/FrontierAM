"""Small LAPACK facade used by the conic algorithm."""
import numpy as np
from kernels import extension

lapack = extension("scipy.linalg._flapack", "linalg/_flapack*.so")
LinAlgError = np.linalg.LinAlgError


def _ok(info):
    if info:
        raise LinAlgError("LAPACK status %d" % info)


def cholesky(a, lower=False, check_finite=False):
    c, info = lapack.dpotrf(np.asarray(a, dtype=float), lower=int(lower), clean=1)
    _ok(info)
    return c


def cho_factor(a, lower=False, check_finite=False):
    return cholesky(a, lower=lower), lower


def cho_solve(factor, b, check_finite=False):
    a, lower = factor
    b = np.asarray(b, dtype=float)
    vector = b.ndim == 1
    x, info = lapack.dpotrs(a, b[:, None] if vector else b, lower=int(lower))
    _ok(info)
    return x[:, 0] if vector else x


def solve_triangular(a, b, lower=False, check_finite=False, trans=0):
    b = np.asarray(b, dtype=float)
    if a.shape[0] == 0:
        return b.copy()
    vector = b.ndim == 1
    bb = b[:, None] if vector else b
    x, info = lapack.dtrtrs(np.asarray(a, dtype=float), bb, lower=int(lower), trans=trans)
    _ok(info)
    return x[:, 0] if vector else x


def eigh(a, eigvals_only=False, subset_by_index=None, check_finite=False, **kwargs):
    a = np.asarray(a, dtype=float)
    if a.shape[0] == 0:
        return np.empty(0) if eigvals_only else (np.empty(0), np.empty((0, 0)))
    opts = {"compute_v": int(not eigvals_only), "lower": int(kwargs.get("lower", True))}
    if subset_by_index is not None:
        opts.update(range="I", il=subset_by_index[0] + 1, iu=subset_by_index[1] + 1)
    w, v, count, support, info = lapack.dsyevr(a, **opts)
    _ok(info)
    return w[:count] if eigvals_only else (w[:count], v[:, :count])


def lu_factor(a, check_finite=False):
    lu, piv, info = lapack.dgetrf(np.asarray(a, dtype=float))
    _ok(info)
    return lu, piv


def lu_solve(factor, b, check_finite=False, trans=0):
    lu, piv = factor
    b = np.asarray(b, dtype=float)
    vector = b.ndim == 1
    x, info = lapack.dgetrs(lu, piv, b[:, None] if vector else b, trans=trans)
    _ok(info)
    return x[:, 0] if vector else x


def qr(a, mode="full", pivoting=False, check_finite=False):
    a = np.asarray(a, dtype=float)
    m, n = a.shape
    k = min(m, n)
    if not k:
        nq = k if mode == "economic" else m
        Q = np.eye(m, nq)
        R = np.zeros((nq, n))
        return (Q, R, np.arange(n)) if pivoting else (Q, R)
    if pivoting:
        _, _, _, work, info = lapack.dgeqp3(a, lwork=-1)
        _ok(info)
        packed, perm, tau, _, info = lapack.dgeqp3(a, lwork=max(1, int(work[0])))
        perm = perm - 1
    else:
        _, _, work, info = lapack.dgeqrf(a, lwork=-1)
        _ok(info)
        packed, tau, _, info = lapack.dgeqrf(a, lwork=max(1, int(work[0])))
        perm = np.arange(n)
    _ok(info)
    nq = k if mode == "economic" else m
    R = np.triu(packed[:nq]).copy()
    if nq > packed.shape[1]:
        reflectors = np.zeros((m, nq), order="F")
        reflectors[:, :k] = packed[:, :k]
        R = np.vstack((R, np.zeros((nq - R.shape[0], n))))
    else:
        reflectors = np.array(packed[:, :nq], order="F")
    _, work, info = lapack.dorgqr(reflectors, tau, lwork=-1)
    _ok(info)
    Q, _, info = lapack.dorgqr(reflectors, tau, lwork=max(1, int(work[0])))
    _ok(info)
    return (Q, R, perm) if pivoting else (Q, R)
