"""Krylov (Lanczos) propagator for y = exp(-tL) b0 + t phi1(-tL) b1, L symmetric PSD."""
import json
import struct
import sys

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh_tridiagonal

STDIN = sys.stdin.buffer
STDOUT = sys.stdout.buffer
SAFETY = 0.01


def read_exact(length):
    buf = bytearray(length)
    view = memoryview(buf)
    pos = 0
    while pos < length:
        got = STDIN.readinto(view[pos:])
        if not got:
            raise EOFError("unexpected end of input")
        pos += got
    return buf


def phi_exp(x):
    return np.exp(-x)


def phi_one(x):
    out = np.ones_like(x)
    mask = np.abs(x) > 1e-300
    out[mask] = -np.expm1(-x[mask]) / x[mask]
    return out


class Lanczos:
    def __init__(self, matvec, vec, n, mmax, func, scale):
        self.matvec = matvec
        self.func = func
        self.scale = scale
        self.norm = float(np.linalg.norm(vec))
        self.V = np.empty((mmax + 1, n))
        self.alpha = []
        self.beta = []
        self.mmax = mmax
        self.done = self.norm == 0.0
        self.exact = self.done
        self.m = 0
        self.result = np.zeros(n)
        self.prev = None
        self.tail = np.inf
        if not self.done:
            self.V[0] = vec / self.norm

    def step(self):
        j = self.m
        V = self.V
        w = self.matvec(V[j])
        a = float(V[j] @ w)
        w -= a * V[j]
        if j > 0:
            w -= self.beta[-1] * V[j - 1]
        basis = V[: j + 1]
        w -= basis.T @ (basis @ w)
        w -= basis.T @ (basis @ w)
        self.alpha.append(a)
        b = float(np.linalg.norm(w))
        self.m = j + 1
        anorm = max(abs(x) for x in self.alpha) + (max(self.beta) if self.beta else 0.0)
        if b <= 1e-14 * max(anorm, 1e-300) or self.m >= V.shape[0]:
            self.beta.append(b)
            if b <= 1e-14 * max(anorm, 1e-300):
                self.exact = True
            return
        self.beta.append(b)
        V[j + 1] = w / b

    def evaluate(self):
        m = self.m
        alpha = np.array(self.alpha)
        if m == 1:
            theta = alpha
            S = np.ones((1, 1))
        else:
            theta, S = eigh_tridiagonal(alpha, np.array(self.beta[: m - 1]))
        coef = S @ (self.func(self.scale * theta) * S[0])
        self.tail = self.norm * self.beta[m - 1] * float(np.max(np.abs(coef[-2:])))
        self.prev = self.result
        self.result = (self.norm * coef) @ self.V[:m]
        return self.result


def min_dim(norm, t):
    return int(np.ceil(np.sqrt(5.0 * 0.25 * norm * t * np.log(10.0)))) + 2


def propagate(matvec, n, t, b0, b1, rtol, atol, mmax, norm, depth=0):
    if t == 0.0:
        return b0.copy()
    mmin = min_dim(norm, t)
    if mmin > mmax // 2 and depth <= 40:
        half = 0.5 * t
        mid = propagate(matvec, n, half, b0, b1, rtol, atol, mmax, norm, depth + 1)
        return propagate(matvec, n, half, mid, b1, rtol, atol, mmax, norm, depth + 1)
    runs = [Lanczos(matvec, b0, n, mmax, phi_exp, t),
            Lanczos(matvec, b1, n, mmax, phi_one, t)]
    weights = [1.0, t]
    for run in runs:
        if run.done:
            run.result = np.zeros(n)
    check = 6
    stable = [0, 0]
    while True:
        active = [r for r in runs if not r.done]
        if not active:
            break
        for run in active:
            run.step()
        for i, run in enumerate(runs):
            if run.done:
                continue
            if run.exact or run.m % check == 0 or run.m >= mmax:
                run.evaluate()
        y = runs[0].result + t * runs[1].result
        target = SAFETY * (atol + rtol * float(np.max(np.abs(y))))
        share = 0.5 * target
        for i, run in enumerate(runs):
            if run.done:
                continue
            if run.exact:
                run.done = True
                continue
            if run.m % check == 0 and run.prev is not None and run.m > check and run.m >= mmin:
                diff = weights[i] * float(np.max(np.abs(run.result - run.prev)))
                tail = weights[i] * run.tail
                if diff <= share and tail <= share:
                    stable[i] += 1
                    if stable[i] >= 1:
                        run.done = True
                else:
                    stable[i] = 0
            if not run.done and run.m >= mmax:
                if depth > 40:
                    run.done = True
                else:
                    half = 0.5 * t
                    mid = propagate(matvec, n, half, b0, b1, rtol, atol, mmax, norm, depth + 1)
                    return propagate(matvec, n, half, mid, b1, rtol, atol, mmax, norm, depth + 1)
    return runs[0].result + t * runs[1].result


def main():
    STDOUT.write(b"READY\n")
    STDOUT.flush()
    if bytes(read_exact(8)) != b"FAMEXP2\n":
        raise SystemExit("bad magic")
    length = struct.unpack("<I", read_exact(4))[0]
    meta = json.loads(bytes(read_exact(length)).decode("utf-8"))
    n, nnz, requests = meta["n"], meta["nnz"], meta["requests"]
    t, rtol, atol = float(meta["t"]), float(meta["rtol"]), float(meta["atol"])
    indptr = np.frombuffer(read_exact(8 * (n + 1)), dtype="<u8").astype(np.int64)
    indices = np.frombuffer(read_exact(4 * nnz), dtype="<u4").astype(np.int32)
    data = np.frombuffer(read_exact(8 * nnz), dtype="<f8").astype(np.float64)
    L = sp.csr_matrix((data, indices, indptr), shape=(n, n))
    mmax = int(max(30, min(1000, 6e8 / (8.0 * 2 * n))))
    matvec = L.dot
    norm = float(np.max(np.asarray(abs(L).sum(axis=1)))) if nnz else 0.0
    for _ in range(requests):
        raw = np.frombuffer(read_exact(16 * n), dtype="<f8")
        b0 = np.array(raw[:n], dtype=np.float64)
        b1 = np.array(raw[n:], dtype=np.float64)
        if nnz == 0 or t == 0.0:
            y = b0 + t * b1
        else:
            y = propagate(matvec, n, t, b0, b1, rtol, atol, mmax, norm)
        y = np.ascontiguousarray(y, dtype="<f8")
        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        STDOUT.write(y.tobytes())
        STDOUT.flush()


if __name__ == "__main__":
    main()
