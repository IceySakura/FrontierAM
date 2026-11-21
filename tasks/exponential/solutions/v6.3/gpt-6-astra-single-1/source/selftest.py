"""Small, standalone protocol and numerical checks. Requires NumPy and SciPy."""
import json
import os
from pathlib import Path
import struct
import subprocess

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy import fft, sparse

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(912853)


def read_exact(stream, size):
    parts = bytearray()
    while len(parts) < size:
        block = stream.read(size-len(parts))
        if not block:
            raise RuntimeError("short response")
        parts.extend(block)
    return bytes(parts)


def check(name, matrix, t, b0, b1, reference):
    matrix = sparse.csr_matrix(matrix)
    matrix.sort_indices()
    b0, b1, reference = map(np.atleast_2d, (b0, b1, reference))
    count, n = b0.shape
    metadata = dict(version=2, n=n, nnz=matrix.nnz, requests=count,
                    scenario="reuse" if count == 16 else "one_shot",
                    t=float(t), rtol=1e-10, atol=1e-13)
    header = json.dumps(metadata).encode()
    process = subprocess.Popen([str(ROOT/"run.sh")], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert read_exact(process.stdout, 6) == b"READY\n"
        process.stdin.write(b"FAMEXP2\n"+struct.pack("<I", len(header))+header)
        for array, dtype in ((matrix.indptr, "<u8"), (matrix.indices, "<u4"),
                             (matrix.data, "<f8")):
            process.stdin.write(np.asarray(array, dtype=dtype).tobytes())
        worst = 0.
        for index in range(count):
            process.stdin.write(np.asarray(b0[index], dtype="<f8").tobytes())
            process.stdin.write(np.asarray(b1[index], dtype="<f8").tobytes())
            process.stdin.flush()
            result = np.frombuffer(read_exact(process.stdout, n*8), dtype="<f8")
            error = np.max(np.abs(result-reference[index]))
            tolerance = 1e-13+1e-10*np.max(np.abs(reference[index]))
            assert np.all(np.isfinite(result)) and error <= tolerance, (name, index, error/tolerance)
            worst = max(worst, float(error/tolerance))
        process.stdin.close()
        assert process.wait(timeout=5) == 0
        assert process.stdout.read() == b""
        print(f"{name}: passed; maximum scaled error {worst:.3g}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def phi(x):
    result = np.ones_like(x)
    np.divide(-np.expm1(-x), x, out=result, where=x != 0)
    return result


def main():
    b0, b1 = RNG.uniform(-1, 1, (2, 16, 9))
    check("zero operator / sequential reuse", np.zeros((9, 9)), 10000, b0, b1,
          b0.astype(np.longdouble)+10000*b1.astype(np.longdouble))
    check("zero time", np.eye(9)*1e8, 0, b0, b1, b0)
    diagonal = np.array([0., 1e-12, 1e-8, 1e-4, .01, .1, .5, 1., 2.])
    t = 5000.
    reference = b0*np.exp(-t*diagonal)+b1*(t*phi(t*diagonal))
    check("singular diagonal", sparse.diags(diagonal), t, b0, b1, reference)
    check("subnormal operator", np.ones((9, 9))*5e-324, 10000., b0, b1,
          b0.astype(np.longdouble)+10000*b1.astype(np.longdouble))
    n, t = 64, 10000.
    rows = np.repeat(np.arange(n), 3)
    cols = np.column_stack(((np.arange(n)-1) % n, np.arange(n),
                            (np.arange(n)+1) % n)).ravel()
    matrix = sparse.csr_matrix((np.tile([-1., 2., -1.], n), (rows, cols)), shape=(n, n))
    b0, b1 = RNG.uniform(-1, 1, (2, 16, n))
    b0[:4] = 0
    b1[4:8] = 0
    b0[8], b1[8] = 1, 0
    b0[9], b1[9] = 0, 1
    eigenvalues = 4*np.sin(np.pi*np.arange(n//2+1)/n)**2
    reference = fft.irfft(fft.rfft(b0)*np.exp(-t*eigenvalues)+
                          fft.rfft(b1)*(t*phi(t*eigenvalues)), n)
    gauge = RNG.choice([-1., 1.], n)
    check("signed singular diffusion", sparse.diags(gauge)@matrix@sparse.diags(gauge),
          t, b0*gauge, b1*gauge, reference*gauge)
    n, t = 64, 1562.5
    matrix = np.ones((n, n))
    b0, b1 = RNG.uniform(-.75, .75, (2, 16, n))
    b0[8:] -= b0[8:].mean(axis=1)[:, None]
    b0[8:] *= .9/np.max(np.abs(b0[8:]))
    null = b0[8:].astype(np.longdouble)
    null -= null.mean(axis=1)[:, None]
    b1[8:] = -null/t
    a, b = b0.astype(np.longdouble), b1.astype(np.longdouble)
    ma, mb = a.mean(axis=1)[:, None], b.mean(axis=1)[:, None]
    reference = (a-ma)+t*(b-mb)+mb/n
    check("rank-deficient / cancellation", matrix, t, b0, b1, reference)


if __name__ == "__main__":
    main()
