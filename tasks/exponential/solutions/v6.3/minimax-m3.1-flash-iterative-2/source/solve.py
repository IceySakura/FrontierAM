"""Pure Python fallback solver for the FAMEXP2 protocol.

Same algorithm as the C++ implementation in src/fastsolve.cpp: both matrix
functions are expanded in Chebyshev polynomials of the normalised operator
A = (L - shift I)/scale, so that every degree costs one sparse matvec.

This file is only used when the compiled solver is unavailable.
"""

import json
import struct
import sys

import numpy as np
import scipy.sparse as sp

MAGIC = b"FAMEXP2\n"
READY = b"READY\n"


def read_exact(stream, n):
    parts = []
    left = n
    while left > 0:
        block = stream.read(left)
        if not block:
            raise EOFError("short input")
        parts.append(block)
        left -= len(block)
    return b"".join(parts)


def read_header(stream):
    if read_exact(stream, len(MAGIC)) != MAGIC:
        raise ValueError("bad magic")
    (length,) = struct.unpack("<I", read_exact(stream, 4))
    meta = json.loads(read_exact(stream, length).decode("utf-8"))
    n = int(meta["n"])
    nnz = int(meta["nnz"])
    indptr = np.frombuffer(read_exact(stream, 8 * (n + 1)), dtype="<u8")
    indices = np.frombuffer(read_exact(stream, 4 * nnz), dtype="<u4")
    data = np.frombuffer(read_exact(stream, 8 * nnz), dtype="<f8")
    meta["indptr"] = indptr
    meta["indices"] = indices
    meta["data"] = data
    return meta


def phi1(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    small = np.abs(z) < 1e-8
    out[small] = 1.0 + z[small] * (0.5 + z[small] * (1.0 / 6.0 + z[small] / 24.0))
    out[~small] = np.expm1(z[~small]) / z[~small]
    return out


def spectral_window(L, t):
    """Estimate a window [a, b] containing the spectrum of L."""
    nrm_inf = float(abs(L).sum(axis=1).max())
    if nrm_inf <= 0.0:
        return 0.0, 0.0, nrm_inf
    lo, hi = 0.0, nrm_inf
    try:
        from scipy.sparse.linalg import eigsh
        k = 1
        rng = np.random.default_rng(0)
        v = rng.standard_normal(L.shape[0])
        try:
            h = eigsh(L, k=k, which="LA", v0=v, tol=1e-8, maxiter=5000,
                      return_eigenvectors=False)
            hi = float(h[0])
        except Exception:
            pass
        try:
            h = eigsh(L, k=k, which="SA", v0=v, tol=1e-8, maxiter=5000,
                      return_eigenvectors=False)
            lo = float(h[0])
        except Exception:
            pass
    except Exception:
        pass
    width = hi - lo
    if not (width > 0.0):
        width = nrm_inf
    margin = 0.02 * width + 1e-6 * nrm_inf
    b = min(hi + margin, nrm_inf)
    delta = 1e-6 * width + 1e-12 * nrm_inf
    if lo > 0.05 * width:
        a = 0.99 * lo - delta
    else:
        a = min(0.0, lo) - delta
    a = max(a, -700.0 / t if t > 0 else a)
    a = max(a, -nrm_inf)
    if b - a > 2.0 * nrm_inf:
        a, b = -nrm_inf, nrm_inf
    return a, b, nrm_inf


def cheb_coefficients(fun, N, m_out):
    """Chebyshev coefficients of fun(u) on [-1,1] from Chebyshev-Gauss nodes."""
    from scipy.fft import dct
    k = np.arange(N)
    u = np.cos(np.pi * (k + 0.5) / N)
    vals = fun(u)
    c = dct(vals, type=2, norm=None)
    c = c / N
    out = np.empty(m_out + 1)
    out[0] = 0.5 * c[0]
    out[1:] = c[1:m_out + 1]
    return out


def build_plan(L, t, rtol, atol):
    n = L.shape[0]
    a, b, nrm_inf = spectral_window(L, t)
    m0 = 0.5 * (a + b)
    h = 0.5 * (b - a)
    th = t * h
    tm = t * m0
    W = 2.0 * th
    if not (W > 0.0):
        W = 1.0
    gscale = t * float(phi1(np.array([-t * max(0.0, a)])) [0])
    worst_y = 1.0 + gscale
    tol_trunc = 0.05 * (atol + rtol * worst_y)
    scale_min = min(1.0, gscale) if gscale > 0 else 1.0
    R = scale_min * np.sqrt(W) / (tol_trunc * np.sqrt(np.pi))
    R = max(R, 2.0)
    m = np.sqrt(W * np.log(R))
    for _ in range(5):
        rr = max(R / max(m, 1.0), 2.0)
        m = max(m, np.sqrt(W * np.log(rr)))
    m = max(m, 6.0)
    mneed = int(m + 6.0 * np.sqrt(W) + 10.0)
    N = 2 * mneed + 64
    m_out = (N - 1) // 2

    def f_of_u(u):
        lam = m0 + h * u
        # only the upper side needs clipping: exp underflows harmlessly
        return np.exp(np.minimum(-t * lam, 700.0))

    def g_of_u(u):
        lam = m0 + h * u
        z = np.minimum(-t * lam, 700.0)
        return t * phi1(z)

    f = cheb_coefficients(f_of_u, N, m_out)
    g = cheb_coefficients(g_of_u, N, m_out)
    tailF = np.concatenate([np.cumsum(np.abs(f)[::-1])[::-1], [0.0]])
    tailG = np.concatenate([np.cumsum(np.abs(g)[::-1])[::-1], [0.0]])
    return dict(n=n, m0=m0, h=h, f=f, g=g, tailF=tailF, tailG=tailG,
                mmax=m_out, mprov=min(mneed, m_out), t=t, rtol=rtol, atol=atol)


def solve_request(plan, L, b0, b1):
    n = plan["n"]
    t = plan["t"]
    if t == 0.0:
        return b0.copy()
    if plan["h"] <= 0.0:
        return b0 + t * b1
    m0, h = plan["m0"], plan["h"]
    nb0 = float(np.max(np.abs(b0)))
    nb1 = float(np.max(np.abs(b1)))
    m = max(2, min(plan["mprov"], plan["mmax"]))
    f, g = plan["f"], plan["g"]

    X = np.empty((n, 2))
    X[:, 0] = b0
    X[:, 1] = b1
    acc = np.zeros((n, 2))
    prev = np.zeros((n, 2))
    cur = X.copy()
    acc[:, 0] += f[0] * cur[:, 0]
    acc[:, 1] += g[0] * cur[:, 1]
    for j in range(1, m):
        w = L @ cur
        w -= m0 * cur
        w /= h
        if j == 1:
            nxt = w
        else:
            nxt = 2.0 * w - prev
        prev, cur = cur, nxt
        acc[:, 0] += f[j] * cur[:, 0]
        acc[:, 1] += g[j] * cur[:, 1]
    ymax = float(np.max(np.abs(acc[:, 0] + acc[:, 1])))
    tol = plan["atol"] + plan["rtol"] * ymax
    mfinal = plan["mmax"]
    for j in range(plan["mmax"] + 1):
        err = plan["tailF"][j] * nb0 + plan["tailG"][j] * nb1
        if err <= 0.25 * tol:
            mfinal = j
            break
    mfinal = max(mfinal, m)
    for j in range(m, mfinal):
        w = L @ cur
        w -= m0 * cur
        w /= h
        nxt = 2.0 * w - prev
        prev, cur = cur, nxt
        acc[:, 0] += f[j] * cur[:, 0]
        acc[:, 1] += g[j] * cur[:, 1]
    return acc[:, 0] + acc[:, 1]


def main():
    out = sys.stdout.buffer
    out.write(READY)
    out.flush()
    meta = read_header(sys.stdin.buffer)
    n = int(meta["n"])
    L = sp.csr_matrix((meta["data"], meta["indices"], meta["indptr"]),
                       shape=(n, n))
    plan = build_plan(L, float(meta["t"]), float(meta["rtol"]),
                       float(meta["atol"]))
    for _ in range(int(meta["requests"])):
        b0 = np.frombuffer(read_exact(sys.stdin.buffer, 8 * n), dtype="<f8")
        b1 = np.frombuffer(read_exact(sys.stdin.buffer, 8 * n), dtype="<f8")
        y = solve_request(plan, L, b0, b1)
        out.write(np.asarray(y, dtype="<f8").tobytes())
        out.flush()


if __name__ == "__main__":
    main()
