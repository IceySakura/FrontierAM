"""Independent numerical and sequential-protocol smoke checks."""

import json
import os
from pathlib import Path
import select
import struct
import subprocess
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.fft import fftn, ifftn


ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261127)


def read_exact(stream, length, timeout):
    deadline = time.monotonic() + timeout
    result = bytearray()
    while len(result) < length:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
            raise AssertionError("protocol timeout")
        chunk = os.read(stream.fileno(), length - len(result))
        if not chunk:
            raise AssertionError("short response")
        result.extend(chunk)
    return bytes(result)


def write_all(stream, data):
    data = memoryview(data)
    while data:
        written = os.write(stream.fileno(), data)
        if written <= 0:
            raise AssertionError("short write")
        data = data[written:]


def reference(A, t, b0, b1):
    if t == 0:
        return b0.copy()
    values, vectors = eigh(A.toarray())
    z = t * values
    exponential = np.exp(-z)
    forcing = np.ones_like(z)
    np.divide(-np.expm1(-z), z, out=forcing, where=z != 0)
    return ((b0 @ vectors) * exponential + t * (b1 @ vectors) * forcing) @ vectors.T


def session(name, A, t, b0, b1, rtol=1e-10, expected=None):
    A = csr_matrix(A, dtype=np.float64)
    A.sort_indices()
    b0, b1 = np.atleast_2d(b0), np.atleast_2d(b1)
    count, n = b0.shape
    expected = reference(A, t, b0, b1) if expected is None else np.atleast_2d(expected)
    metadata = dict(version=2, n=n, nnz=A.nnz, requests=count,
                    scenario="one_shot" if count == 1 else "reuse",
                    t=float(t), rtol=rtol, atol=rtol * 1e-3)
    encoded = json.dumps(metadata, separators=(",", ":")).encode()
    process = subprocess.Popen([str(ROOT / "run.sh")], cwd=ROOT, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    worst = 0.0
    try:
        assert read_exact(process.stdout, 6, 10) == b"READY\n", "invalid READY"
        write_all(process.stdin, b"FAMEXP2\n" + struct.pack("<I", len(encoded)) + encoded)
        for data in (np.asarray(A.indptr, dtype="<u8"), np.asarray(A.indices, dtype="<u4"),
                     np.asarray(A.data, dtype="<f8")):
            write_all(process.stdin, data.tobytes())
        for k in range(count):
            write_all(process.stdin, np.asarray(b0[k], dtype="<f8").tobytes())
            write_all(process.stdin, np.asarray(b1[k], dtype="<f8").tobytes())
            result = np.frombuffer(read_exact(process.stdout, 8 * n, 15), dtype="<f8")
            assert np.all(np.isfinite(result)), f"{name}: nonfinite output"
            threshold = rtol * 1e-3 + rtol * np.max(np.abs(expected[k]))
            scaled = float(np.max(np.abs(result - expected[k])) / threshold)
            worst = max(worst, scaled)
            assert scaled <= 1, f"{name}, request {k}: error/tolerance = {scaled}"
        process.stdin.close()
        assert process.wait(timeout=10) == 0, f"{name}: nonzero exit"
        assert process.stdout.read(1) == b"", f"{name}: extra output"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            if not stream.closed:
                stream.close()
    return count, worst


def main():
    total_sessions = total_requests = 0
    worst = 0.0

    def run(*args, **kwargs):
        nonlocal total_sessions, total_requests, worst
        count, error = session(*args, **kwargs)
        total_sessions += 1
        total_requests += count
        worst = max(worst, error)

    b0, b1 = RNG.uniform(-1, 1, (2, 16, 8))
    run("zero matrix", csr_matrix((8, 8)), 2500, b0, b1, expected=b0 + 2500 * b1)
    run("zero time", eye(8, format="csr"), 0, b0, b1, expected=b0)
    run("diagonal", diags([0, 1e-8, 2, 10]), 0.5,
        [[1, -1, 0.25, -0.5]], [[0.5, 1, -1, 0.25]])
    run("positive off-diagonal", [[1, 1], [1, 1]], 2,
        [[0.5, -0.5]], [[-0.25, 0.25]], expected=[[0, 0]])

    n = 41
    q, _ = np.linalg.qr(RNG.normal(size=(n, n)))
    matrix = (q * np.geomspace(1e-4, 1, n)) @ q.T
    matrix = csr_matrix((matrix + matrix.T) * 0.5)
    norm = np.max(np.asarray(abs(matrix).sum(axis=1)))
    for eta in (0.01, 1, 100, 100000):
        t = min(10000.0, eta)
        A = matrix * (eta / (norm * t))
        for rtol in (1e-4, 1e-7, 1e-10):
            b0, b1 = RNG.uniform(-1, 1, (2, 1, n))
            run(f"dense PSD eta={eta:g}", A, t, b0, b1, rtol)

    n = 31
    A = diags([-np.ones(n-1), 2 * np.ones(n), -np.ones(n-1)], [-1, 0, 1]).tolil()
    A[0, -1] = A[-1, 0] = -1
    signs = RNG.choice([-1.0, 1.0], n)
    A = diags(signs) @ A.tocsr() @ diags(signs)
    b0, b1 = RNG.uniform(-1, 1, (2, 16, n))
    run("signed singular reuse", A, 150, b0, b1)

    n = 48
    diagonal = np.full(n, 2.0)
    diagonal[[0, -1]] = 1
    A = diags([-np.ones(n-1), diagonal, -np.ones(n-1)], [-1, 0, 1], format="csr")
    b0 = RNG.integers(-8, 9, size=(1, n)) / 32
    b1 = (A @ b0.T).T
    run("stationary state", A, 10000, b0, b1, expected=b0)

    side = 32
    path = diags([-np.ones(side-1), 2*np.ones(side), -np.ones(side-1)],
                 [-1, 0, 1]).tolil()
    path[0, -1] = path[-1, 0] = -1
    A = kron(path, eye(side)) + kron(eye(side), path)
    n = side * side
    signs = RNG.choice([-1.0, 1.0], n)
    A = diags(signs) @ A.tocsr() @ diags(signs)
    b0, b1 = RNG.uniform(-1, 1, (2, 16, n))
    b0[3], b1[3] = signs, -signs
    t = 1.0
    A *= 12500
    lam = 100000 * (np.sin(np.pi*np.arange(side)[:, None]/side)**2
                   + np.sin(np.pi*np.arange(side)[None, :]/side)**2) / 2
    ph = np.ones_like(lam)
    np.divide(-np.expm1(-lam), lam, out=ph, where=lam != 0)
    transformed0 = fftn((b0*signs).reshape(16, side, side), axes=(1, 2))
    transformed1 = fftn((b1*signs).reshape(16, side, side), axes=(1, 2))
    expected = np.real(ifftn(np.exp(-lam)*transformed0 + ph*transformed1,
                            axes=(1, 2))).reshape(16, n) * signs
    run("periodic signed reuse", A, t, b0, b1, expected=expected)
    print(f"Passed {total_sessions} sessions, {total_requests} requests; "
          f"worst error/tolerance {worst:.6g}.")


if __name__ == "__main__":
    main()
