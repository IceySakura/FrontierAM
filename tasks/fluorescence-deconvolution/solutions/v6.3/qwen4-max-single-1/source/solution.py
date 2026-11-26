import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import time
import ctypes
import numpy as np


def noise_estimate(y):
    vals = []
    for ax in range(3):
        if y.shape[ax] >= 3:
            d = np.diff(y, n=2, axis=ax)
            med = np.median(d)
            vals.append(np.median(np.abs(d - med)) / (0.6745 * np.sqrt(6.0)))
    if not vals:
        return 0.01
    return float(np.clip(np.mean(vals), 0.002, 0.05))


def bilateral(x, radius, sigma_s, sigma_r):
    out = np.zeros_like(x)
    wsum = np.zeros_like(x)
    xp = np.pad(x, [(radius, radius)] * 3, mode="edge")
    nz, ny, nx = x.shape
    inv_s = 1.0 / (2.0 * sigma_s * sigma_s)
    inv_r = 1.0 / (2.0 * sigma_r * sigma_r)
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                shifted = xp[radius + dz:radius + dz + nz,
                             radius + dy:radius + dy + ny,
                             radius + dx:radius + dx + nx]
                w = np.exp(-(dz * dz + dy * dy + dx * dx) * inv_s
                           - (shifted - x) ** 2 * inv_r)
                out += w * shifted
                wsum += w
    return out / np.maximum(wsum, 1e-12)


def tv_prox(z, lam, iters=15):
    p = np.zeros((3,) + z.shape, np.float32)

    def D(x):
        g = np.zeros((3,) + x.shape, np.float32)
        for ax in range(3):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[ax] = slice(0, -1)
            b[ax] = slice(1, None)
            g[(ax,) + tuple(a)] = x[tuple(b)] - x[tuple(a)]
        return g

    def Dt(q):
        out = np.zeros_like(z)
        for ax in range(3):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[ax] = slice(0, -1)
            b[ax] = slice(1, None)
            out[tuple(a)] -= q[(ax,) + tuple(a)]
            out[tuple(b)] += q[(ax,) + tuple(a)]
        return out

    for _ in range(iters):
        g = D(z - Dt(p))
        p += 0.14 * g
        den = np.maximum(1.0, np.sqrt(np.sum(p * p, axis=0)) / lam)
        p /= den
    return np.maximum(z - Dt(p), 0.0)


def load_lib():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver.so")
    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return None
    ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    lib.solve_separable.argtypes = [ptr, ptr,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ptr, ptr, ptr,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_float, ctypes.c_float,
                                    ctypes.c_float]
    lib.solve_separable.restype = ctypes.c_int
    return lib


def separable_factors(p):
    fz = np.ascontiguousarray(p.sum(axis=(1, 2)), dtype=np.float32)
    fy = np.ascontiguousarray(p.sum(axis=(0, 2)), dtype=np.float32)
    fx = np.ascontiguousarray(p.sum(axis=(0, 1)), dtype=np.float32)
    approx = fz[:, None, None] * fy[None, :, None] * fx[None, None, :]
    if np.max(np.abs(approx - p)) <= 1e-5:
        return fz, fy, fx
    return None


def c_solve(lib, y, factors, iters, lam):
    fz, fy, fx = factors
    out = np.empty_like(y)
    nz, ny, nx = y.shape
    rc = lib.solve_separable(y, out, nz, ny, nx, fz, fy, fx,
                             fz.size // 2, fy.size // 2, fx.size // 2,
                             iters, lam, 0.003, 0.0)
    if rc != 0:
        raise RuntimeError("C solver failed")
    return out


def fallback(y, p, ne):
    from scipy.fft import rfftn, irfftn
    shape = tuple(n + m - 1 for n, m in zip(y.shape, p.shape))
    hp = np.zeros(shape, np.float32)
    hp[tuple(slice(0, m) for m in p.shape)] = p
    hp = np.roll(hp, tuple(-(m // 2) for m in p.shape), axis=(0, 1, 2))
    H = rfftn(hp)
    Ht = H.conj()
    crop = tuple(slice(0, n) for n in y.shape)

    def fwd(x):
        return irfftn(rfftn(x, shape) * H, shape).real[crop]

    def adj(x):
        return irfftn(rfftn(x, shape) * Ht, shape).real[crop]

    norm = adj(np.ones(y.shape, np.float32))
    x = np.maximum(y, np.mean(y) * 0.1)
    lam = 0.01 + ne
    for _ in range(50):
        ratio = y / np.maximum(fwd(x), 1e-6)
        gs = []
        for ax in range(3):
            g = np.zeros_like(x)
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[ax] = slice(0, -1)
            b[ax] = slice(1, None)
            g[tuple(a)] = x[tuple(b)] - x[tuple(a)]
            gs.append(g)
        den = np.sqrt(gs[0] * gs[0] + gs[1] * gs[1] + gs[2] * gs[2] + 9e-6)
        tv = np.zeros_like(x)
        for ax, g in enumerate(gs):
            q = g / den
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[ax] = slice(0, -1)
            b[ax] = slice(1, None)
            tv[tuple(a)] -= q[tuple(a)]
            tv[tuple(b)] += q[tuple(a)]
        denom = np.maximum(norm + lam * tv, 0.05)
        x = np.maximum(x * adj(ratio) / denom, 1e-8)
    return x


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    with np.load(inp, allow_pickle=False) as data:
        y = np.ascontiguousarray(data["observed"], dtype=np.float32)
        p = np.ascontiguousarray(data["psf"], dtype=np.float32)
    ne = noise_estimate(y)
    lib = load_lib()
    factors = separable_factors(p)
    if lib is not None and factors is not None:
        try:
            probe = c_solve(lib, y, factors, 15, 0.03)
            e = probe.astype(np.float64) ** 2
            total = float(e.sum())
            if total > 0:
                k = max(1, e.size // 100)
                top = np.partition(e.ravel(), -k)[-k:].sum() / total
            else:
                top = 0.0
            lam = 0.005 + ne + 60.0 * max(0.0, top - 0.6) * max(0.0, ne - 0.008)
            lam = float(np.clip(lam, 0.005, 0.12))
            est = c_solve(lib, y, factors, 25, lam)
        except Exception:
            est = fallback(y, p, ne)
    else:
        est = fallback(y, p, ne)
    est = bilateral(est, 3, 2.5, max(0.006, 2.0 * ne))
    est = tv_prox(est, max(0.001, 0.4 * ne), 20)
    est = np.nan_to_num(est, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(est, 0.0, out=est)
    np.save(outp, est.astype(np.float32, copy=False))


if __name__ == "__main__":
    main()
