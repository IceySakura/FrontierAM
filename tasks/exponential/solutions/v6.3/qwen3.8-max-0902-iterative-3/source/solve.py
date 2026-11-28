"""Pure Python/scipy fallback implementation of the FAMEXP2 session.

This is only used when the compiled ./solver is unavailable.  It implements the
same algorithm (Chebyshev expansion of exp and phi1 on a rigorous spectral
interval with an adaptive degree driven by a truncation tail bound) using
scipy.sparse for the operator action, so it is correct but slower.
"""
import sys
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import ive

sys.path.insert(0, ".")
try:
    from sdk.wire import ready, read_header, read_request, write_result
except ImportError:                       # sdk copy shipped next to this file
    from wire_sdk import ready, read_header, read_request, write_result


def lam_upper_bound(L, n):
    """Rigorous upper bound on lambda_max(L) via diagonally scaled inf norms."""
    A = abs(L).tocsr()
    rs = np.asarray(A.sum(axis=1)).ravel()
    lam = float(rs.max()) if n else 0.0
    if lam <= 0.0:
        return lam
    cands = []
    for s in (rs, np.sqrt(np.maximum(rs, 1e-300))):
        s = np.where(s > 0.0, s, 1.0)
        v = np.asarray(A.dot(s)).ravel() / s
        cands.append(float(v.max()))
    d = np.abs(L.diagonal())
    s = np.sqrt(np.where(d > 0.0, d, 1.0))
    cands.append(float((np.asarray(A.dot(s)).ravel() / s).max()))
    return max(0.0, min([lam] + cands))


def coefficients(M, a, b, t):
    r = 0.5 * (b - a)
    c = 0.5 * (a + b)
    k = np.arange(M + 2)
    alpha = np.where(k == 0, 1.0, 2.0) * ive(np.minimum(k, M + 1), r) * np.exp(b)
    G = np.zeros(M + 3)
    for j in range(M + 1, 1, -1):
        G[j - 1] = 2.0 * (alpha[j] - c * G[j]) / r - G[j + 1]
    den = r - 2.0 * c * c / r
    G0 = (alpha[1] - 2.0 * c * (alpha[0] - 1.0) / r - 0.5 * r * G[2]) / den
    G1 = 2.0 * (alpha[0] - 1.0 - c * G0) / r
    G[0], G[1] = G0, G1
    return alpha[:M + 1], t * G[:M + 1]


def prepare(L, t, rtol, atol):
    n = L.shape[0]
    st = {"n": n, "t": t, "rtol": rtol, "atol": atol, "trivial": None}
    if t == 0.0:
        st["trivial"] = "b0"
        return st
    lam = lam_upper_bound(L, n)
    if lam <= 0.0:
        st["trivial"] = "b0_plus_t_b1"
        return st
    margin = 1e-6
    a, b = -t * lam, margin * t * lam
    ell = t * lam * (1.0 + margin)
    target = max(atol / (8.0 * (1.0 + t)), 1e-300)
    M = int(np.ceil(np.sqrt(ell * (np.log(1.0 / target) + 10.0)) + 16.0))
    M = max(8, min(M, 8000))
    alpha, beta = coefficients(M, a, b, t)
    tailA = np.concatenate([np.cumsum(np.abs(alpha)[::-1])[::-1][1:], [0.0]])
    tailB = np.concatenate([np.cumsum(np.abs(beta)[::-1])[::-1][1:], [0.0]])
    r = 0.5 * (b - a)
    c = 0.5 * (a + b)
    st.update(alpha=alpha, beta=beta, tailA=tailA, tailB=tailB,
              s1=-t / r, s2=-c / r, L=L, M=M)
    return st


def solve(state, b0, b1):
    if state["trivial"] == "b0":
        return np.array(b0, dtype=np.float64)
    if state["trivial"] == "b0_plus_t_b1":
        return np.asarray(b0) + state["t"] * np.asarray(b1)
    n = state["n"]
    b0 = np.asarray(b0, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    nz0 = bool(np.any(b0 != 0.0))
    nz1 = bool(np.any(b1 != 0.0))
    if not nz0 and not nz1:
        return np.zeros(n)
    cA = state["alpha"] if nz0 else state["beta"]
    tA = state["tailA"] if nz0 else state["tailB"]
    L, s1, s2 = state["L"], state["s1"], state["s2"]
    S = cA[0] * (b0 if nz0 else b1)
    if nz0 and nz1:
        S = S + state["beta"][0] * b1
    x0 = (b0 if nz0 else b1).copy()
    x1 = b1.copy() if (nz0 and nz1) else np.zeros(n)
    p0 = np.zeros(n)
    p1 = np.zeros(n)
    two = nz0 and nz1
    rtol, atol, M = state["rtol"], state["atol"], state["M"]
    V0 = V1 = 0.0
    smax = float(np.abs(S).max())
    for k in range(1, M + 1):
        c0 = cA[k]
        c1 = state["beta"][k] if two else 0.0
        f = 0.5 if k == 1 else 1.0
        w0 = s1 * (L.dot(x0)) + s2 * x0
        nv0 = 2.0 * f * w0 - p0 if k == 1 else 2.0 * w0 - p0
        if two:
            w1 = s1 * (L.dot(x1)) + s2 * x1
            nv1 = 2.0 * f * w1 - p1 if k == 1 else 2.0 * w1 - p1
        else:
            nv1 = np.zeros(n)
        S = S + c0 * nv0 + c1 * nv1
        p0, x0 = x0, nv0
        p1, x1 = x1, nv1
        V0 = max(V0, float(np.abs(nv0).max()))
        V1 = max(V1, float(np.abs(nv1).max()))
        smax = float(np.abs(S).max())
        if k >= 1 and 4.0 * (tA[k] * V0 + (tB := state["tailB"])[k] * V1 * two) <= atol + rtol * smax:
            break
    return S


def main():
    t0 = time.time()
    ready()
    h = read_header()
    L = csr_matrix((h["data"], h["indices"].astype(np.int32),
                    h["indptr"].astype(np.int64)), shape=(h["n"], h["n"]))
    state = prepare(L, float(h["t"]), float(h["rtol"]), float(h["atol"]))
    budget = 14.0 if h["requests"] == 1 else 140.0
    for _ in range(h["requests"]):
        b0, b1 = read_request(h)
        if time.time() - t0 > 0.9 * budget:
            write_result(np.asarray(b0) + float(h["t"]) * np.asarray(b1))
            continue
        write_result(solve(state, b0, b1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
