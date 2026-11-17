#!/usr/bin/env python3
"""Offline, single-process fluorescence reconstruction."""
import os
import sys
import time

_started = time.monotonic()
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import ctypes
import numpy as np


def fft_fallback(y, h, deadline):
    shape = tuple(a+b-1 for a, b in zip(y.shape, h.shape))
    crop = tuple(slice(b//2, b//2+a) for a, b in zip(y.shape, h.shape))
    domain = tuple(slice(0, a) for a in y.shape)
    H = np.fft.rfftn(h, shape)
    work = np.zeros(shape, np.float64)
    x = y.astype(np.float64).copy()
    p = np.exp(-.5*(np.arange(-6, 7)/1.5)**2)
    p /= p.sum()
    smooth_h = p[:, None, None]*p[None, :, None]*p[None, None, :]
    sm_shape = tuple(a+12 for a in y.shape)
    sm_crop = tuple(slice(6, 6+a) for a in y.shape)
    sm = np.fft.irfftn(np.fft.rfftn(y, sm_shape)*
                       np.fft.rfftn(smooth_h, sm_shape), sm_shape)[sm_crop]
    sm -= p[6]**3*y
    inside = tuple(slice(3, -3) for _ in range(3))
    region = sm[inside]
    if region.size:
        sigma = float(np.mean(y[inside][region <= np.quantile(region, .35)])*
                      np.sqrt(2*np.pi))
    else:
        sigma = float(np.mean(y))*.1
    sigma = max(sigma, 1e-6)
    yy = y-.5*sigma
    l2 = min(.035, .0025*max(sigma/.008, .2)**1.05)
    step = 1/(1+12*l2)
    x = np.maximum(yy, 0).astype(np.float64)
    z = x.copy()
    t = 1.0
    for _ in range(100):
        if time.monotonic() > deadline-.015:
            break
        b = np.fft.irfftn(np.fft.rfftn(z, shape)*H, shape)[crop]
        work.fill(0)
        work[crop] = b-yy
        grad = np.fft.irfftn(np.fft.rfftn(work)*H.conj(), shape)[domain]
        lap = np.zeros_like(z)
        for axis in range(3):
            a = [slice(None)]*3
            b = [slice(None)]*3
            a[axis] = slice(1, None)
            b[axis] = slice(None, -1)
            diff = z[tuple(a)]-z[tuple(b)]
            lap[tuple(a)] += diff
            lap[tuple(b)] -= diff
        new = np.maximum(z-step*(grad+l2*lap+.055*sigma), 0)
        tn = .5*(1+np.sqrt(1+4*t*t))
        z = new+((t-1)/tn)*(new-x)
        x, t = new, tn
    return x.astype(np.float32)


def main():
    with np.load(sys.argv[1], allow_pickle=False) as data:
        y = np.ascontiguousarray(data["observed"], dtype=np.float32)
        h = np.ascontiguousarray(data["psf"], dtype=np.float32)
    end = _started+1.05
    try:
        supplied = float(os.environ.get("DECONV_DEADLINE_MONOTONIC", end))
        if np.isfinite(supplied):
            end = min(end, supplied-.035)
    except ValueError:
        pass
    out = y.copy()
    if np.any(y > 0) and time.monotonic() < end-.025:
        try:
            lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "core.so"))
            fp = ctypes.POINTER(ctypes.c_float)
            dp = ctypes.POINTER(ctypes.c_double)
            lib.solve.argtypes = [fp, fp]+[ctypes.c_int]*6+[ctypes.c_double, fp, dp]
            lib.solve.restype = ctypes.c_int
            diag = np.zeros(8, np.float64)
            rc = lib.solve(y.ctypes.data_as(fp), h.ctypes.data_as(fp),
                           *y.shape, *h.shape, end-.025,
                           out.ctypes.data_as(fp), diag.ctypes.data_as(dp))
            if rc != 0 and time.monotonic() < end-.025:
                out = fft_fallback(y, h, end)
            if os.environ.get("DECONV_DIAGNOSTICS"):
                print("native", rc, diag.tolist(), "elapsed", time.monotonic()-_started, flush=True)
        except (OSError, AttributeError):
            out = fft_fallback(y, h, end)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(out, 0, out=out)
    with open(sys.argv[2], "wb") as f:
        np.save(f, out, allow_pickle=False)
        f.flush()


if __name__ == "__main__":
    main()
