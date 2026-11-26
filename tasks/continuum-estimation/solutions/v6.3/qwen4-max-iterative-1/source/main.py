#!/usr/bin/env python3
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares


P_RANGE = (0.55, 2.75)
D_RANGE = (0.40, 2.00)


def _gaussian_nodes(x, y, p, d, sigma):
    n = len(x)
    scale = max(1.0, float(np.max(np.abs(y))))
    q = p + d
    K = len(p)
    A = np.empty((K, n, 4), dtype=np.float64)
    A[:, :, 0] = 1.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        A[:, :, 1] = x[None, :] ** p[:, None]
        A[:, :, 2] = x[None, :] ** q[:, None]
        A[:, :, 3] = x[None, :] ** (q + 1.5)[:, None]
    st = 2.0 * scale
    prior_prec = np.diag([1.0 / st**2, 1.0, 1.0 / 0.4**2, 1.0 / 0.15**2])
    prior_mean = np.array([y[-1], 0.0, -0.5, -0.2])
    M = np.einsum("kni,knj->kij", A, A) / sigma**2 + prior_prec
    rhs = np.einsum("kni,n->ki", A, y) / sigma**2 + prior_prec @ prior_mean
    mean = np.linalg.solve(M, rhs[:, :, None])[:, :, 0]
    _, logdet = np.linalg.slogdet(M)
    yy = float(y @ y) / sigma**2 + float(prior_mean @ prior_prec @ prior_mean)
    quad = np.einsum("ki,ki->k", mean, np.einsum("kij,kj->ki", M, mean))
    return -0.5 * (yy - quad + logdet), mean[:, 0]


def _blend(w, ts):
    mean = float(w @ ts)
    ww = w / np.maximum(1.0, np.abs(ts))
    ww /= float(ww.sum())
    order = np.argsort(ts)
    cw = np.cumsum(ww[order])
    med = float(ts[order][np.searchsorted(cw, 0.5)])
    return 0.5 * mean + 0.5 * med


def _wmedian(w, ts):
    ww = w / np.maximum(1.0, np.abs(ts))
    ww /= float(ww.sum())
    order = np.argsort(ts)
    cw = np.cumsum(ww[order])
    return float(ts[order][np.searchsorted(cw, 0.5)])


def _grid_fallback(x, y):
    scale = max(1.0, float(np.max(np.abs(y))))
    p = np.linspace(*P_RANGE, 141)
    d = np.linspace(*D_RANGE, 141)
    P, D = np.meshgrid(p, d, indexing="ij")
    logw, ts = _gaussian_nodes(x, y, P.ravel(), D.ravel(), 1e-7 * scale)
    ok = np.isfinite(logw) & np.isfinite(ts)
    if not np.any(ok):
        raise ValueError("no finite grid nodes")
    logw = logw[ok]
    ts = ts[ok]
    w = np.exp(logw - np.max(logw))
    w /= float(w.sum())
    return _blend(w, ts)


def _four_point(x, y):
    # With four observations the four linear coefficients are determined by
    # the orders.  Integrate the resulting exact-fit surface under bounded
    # coefficient priors inferred from the public smooth examples.
    p = np.linspace(*P_RANGE, 181)
    d = np.linspace(*D_RANGE, 181)
    P, D = np.meshgrid(p, d, indexing="ij")
    pp = P.ravel()
    dd = D.ravel()
    qq = pp + dd
    K = len(pp)
    A = np.empty((K, 4, 4), dtype=np.float64)
    A[:, :, 0] = 1.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        A[:, :, 1] = x[None, :] ** pp[:, None]
        A[:, :, 2] = x[None, :] ** qq[:, None]
        A[:, :, 3] = x[None, :] ** (qq + 1.5)[:, None]
    c = np.linalg.solve(A, y[None, :, None])[:, :, 0]
    _, logdet = np.linalg.slogdet(A)
    ok = (
        np.isfinite(logdet)
        & np.isfinite(c).all(axis=1)
        & (c[:, 0] >= -4.5) & (c[:, 0] <= 4.5)
        & (c[:, 1] >= -1.6) & (c[:, 1] <= 1.6)
        & (c[:, 2] >= -1.0) & (c[:, 2] <= -0.05)
        & (c[:, 3] >= -0.45) & (c[:, 3] <= -0.02)
    )
    if not np.any(ok):
        raise ValueError("no bounded four-point nodes")
    logw = -logdet[ok] - 0.5 * ((dd[ok] - 0.9) / 0.7) ** 2
    ts = c[ok, 0]
    w = np.exp(logw - np.max(logw))
    w /= float(w.sum())
    return float(w @ ts)


def _five_point(x, y):
    # Five observations impose one scalar compatibility condition on the two
    # orders.  Trace its roots and integrate along that curve with the Jacobian
    # of the compatibility condition.
    n = 5
    scale = max(1.0, float(np.max(np.abs(y))))
    pg = np.linspace(*P_RANGE, 241)
    dg = np.linspace(*D_RANGE, 161)
    lx = np.log(x)
    ones = np.ones(n)

    def det_scalar(p, q):
        return np.linalg.det(np.column_stack((ones, x**p, x**q, x**(q + 1.5), y)))

    roots = []
    for p in pg:
        qs = p + dg
        A = np.empty((len(qs), 5, 5), dtype=np.float64)
        A[:, :, 0] = 1.0
        A[:, :, 1] = x**p
        A[:, :, 2] = x[None, :] ** qs[:, None]
        A[:, :, 3] = x[None, :] ** (qs + 1.5)[:, None]
        A[:, :, 4] = y
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            vals = np.linalg.det(A)
        for i in range(len(qs) - 1):
            if not (np.isfinite(vals[i]) and np.isfinite(vals[i + 1])):
                continue
            if vals[i] == 0.0:
                roots.append((p, qs[i]))
            elif vals[i] * vals[i + 1] < 0.0:
                try:
                    q = brentq(lambda q: det_scalar(p, q), qs[i], qs[i + 1], xtol=1e-12, rtol=1e-12)
                    roots.append((p, q))
                except Exception:
                    pass
        if len(roots) > 5000:
            raise ValueError("too many compatibility roots")
    if not roots:
        raise ValueError("no compatibility roots")
    keep = []
    for p, q in roots:
        if not keep or abs(p - keep[-1][0]) > 1e-8 or abs(q - keep[-1][1]) > 1e-8:
            keep.append((p, q))
    roots = np.asarray(keep)
    p = roots[:, 0]
    q = roots[:, 1]
    K = len(p)
    A = np.empty((K, n, 4), dtype=np.float64)
    A[:, :, 0] = 1.0
    A[:, :, 1] = x[None, :] ** p[:, None]
    A[:, :, 2] = x[None, :] ** q[:, None]
    A[:, :, 3] = x[None, :] ** (q + 1.5)[:, None]
    M = np.einsum("kni,knj->kij", A, A)
    rhs = np.einsum("kni,n->ki", A, y)
    c = np.linalg.solve(M, rhs[:, :, None])[:, :, 0]
    _, logdet = np.linalg.slogdet(M)
    st = 2.0 * scale
    prior_prec = np.diag([1.0 / st**2, 1.0, 1.0 / 0.4**2, 1.0 / 0.15**2])
    prior_mean = np.array([y[-1], 0.0, -0.5, -0.2])
    diff = c - prior_mean
    penalty = np.einsum("ki,ij,kj->k", diff, prior_prec, diff)
    B = np.empty((K, 5, 5), dtype=np.float64)
    B[:, :, 0] = 1.0
    B[:, :, 1] = x[None, :] ** p[:, None]
    B[:, :, 2] = x[None, :] ** q[:, None]
    B[:, :, 3] = x[None, :] ** (q + 1.5)[:, None]
    B[:, :, 4] = y
    B1 = B.copy()
    B1[:, :, 2] = B[:, :, 2] * lx[None, :]
    B2 = B.copy()
    B2[:, :, 3] = B[:, :, 3] * lx[None, :]
    dq = np.linalg.det(B1) + np.linalg.det(B2)
    logw = -0.5 * penalty - 0.5 * logdet - np.log(np.abs(dq) + 1e-300)
    logw -= 0.5 * ((q - p - 0.9) / 0.7) ** 2
    ts = c[:, 0]
    ok = np.isfinite(logw) & np.isfinite(ts)
    if not np.any(ok):
        raise ValueError("no finite root nodes")
    logw = logw[ok]
    ts = ts[ok]
    w = np.exp(logw - np.max(logw))
    w /= float(w.sum())
    return _wmedian(w, ts)


def estimate(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    scale = max(1.0, float(np.max(np.abs(y))))
    if float(np.max(y) - np.min(y)) <= 1e-12 * scale:
        return float(np.mean(y))
    dy = np.diff(y)
    tol = 1e-10 * scale
    signs = np.where(dy > tol, 1, np.where(dy < -tol, -1, 0))
    nonzero = signs[signs != 0]
    turns = int(np.count_nonzero(nonzero[1:] != nonzero[:-1])) if len(nonzero) else 0
    if turns >= 2:
        try:
            scale = max(1.0, float(np.max(np.abs(y))))
            sol = least_squares(
                lambda v: v[0] + v[1] * x - y,
                [float(np.mean(y)), 0.0],
                loss="cauchy",
                f_scale=0.005 * scale,
            )
            z = float(sol.x[0])
            if math.isfinite(z):
                return z
        except Exception:
            pass
        return fallback(x, y)
    try:
        if len(x) == 5:
            return _five_point(x, y)
        if len(x) == 4:
            return _four_point(x, y)
    except Exception:
        pass
    return _grid_fallback(x, y)


def fallback(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    c = np.polyfit(x, y, 1)
    z = float(np.polyval(c, 0.0))
    if math.isfinite(z):
        return z
    return float(y[0])


def main():
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/inputs/input.json")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/workspace/result/output.json")
    data = json.loads(inp.read_text())
    x, y = data["x"], data["y"]
    try:
        z = estimate(x, y)
    except Exception:
        z = fallback(x, y)
    if not math.isfinite(z):
        z = fallback(x, y)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"estimate": z}, allow_nan=False))


if __name__ == "__main__":
    main()
