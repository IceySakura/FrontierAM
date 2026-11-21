#!/usr/bin/env python3
"""Offline protocol and accuracy checks with analytically known spectra."""

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


def exact_read(stream, length, deadline):
    result = bytearray()
    while len(result) < length:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
            raise TimeoutError("solver response timed out")
        block = os.read(stream.fileno(), length - len(result))
        if not block:
            raise RuntimeError("solver closed the output early")
        result.extend(block)
    return bytes(result)


def write_all(stream, data):
    view = memoryview(data)
    while view:
        count = stream.write(view)
        if not count:
            raise RuntimeError("short input write")
        view = view[count:]


def phi(x):
    result = np.ones_like(x)
    np.divide(-np.expm1(-x), x, out=result, where=x != 0)
    return result


def run_case(name, matrix, b0, b1, expected, t, rtol):
    n = matrix.shape[0]
    requests = len(b0)
    rows, columns = np.nonzero(matrix)
    indptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))].astype("<u8")
    indices = columns.astype("<u4")
    values = matrix[rows, columns].astype("<f8")
    metadata = dict(version=2, n=n, nnz=len(values), requests=requests,
                    scenario="one_shot" if requests == 1 else "reuse",
                    t=float(t), rtol=rtol, atol=rtol * 1e-3)
    payload = json.dumps(metadata, separators=(",", ":"))
    if n == 1:
        payload = payload.replace('"n":', '"\\u006e":').replace('"rtol":', '"r\\u0074ol":')
    payload = payload.encode()
    process = subprocess.Popen(["./run.sh"], cwd=Path(__file__).resolve().parent,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, bufsize=0)
    try:
        if exact_read(process.stdout, 6, time.monotonic() + 10) != b"READY\n":
            raise AssertionError("invalid READY")
        deadline = time.monotonic() + (15 if requests == 1 else 150)
        for block in [b"FAMEXP2\n", struct.pack("<I", len(payload)), payload,
                      indptr.tobytes(), indices.tobytes(), values.tobytes()]:
            write_all(process.stdin, block)
        maximum = 0.
        for j in range(requests):
            write_all(process.stdin, np.asarray(b0[j], dtype="<f8").tobytes())
            write_all(process.stdin, np.asarray(b1[j], dtype="<f8").tobytes())
            actual = np.frombuffer(exact_read(process.stdout, 8 * n, deadline), dtype="<f8")
            tolerance = metadata["atol"] + rtol * float(np.max(abs(expected[j])))
            error = float(np.max(abs(actual - expected[j]))) / tolerance
            if not np.all(np.isfinite(actual)) or error > 1:
                raise AssertionError(f"{name}, request {j}: scaled error {error:.6g}")
            maximum = max(maximum, error)
        process.stdin.close()
        if process.wait(timeout=max(.1, deadline-time.monotonic())) != 0:
            raise AssertionError(process.stderr.read().decode())
        if process.stdout.read():
            raise AssertionError("extra protocol output")
        return maximum
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for stream in [process.stdin, process.stdout, process.stderr]:
            if not stream.closed:
                stream.close()


def main():
    rng = np.random.default_rng(771320)
    fixtures = []
    for n, weight, potential, positive in [
        (1, 0., 0., False),
        (2, .5, 0., False),
        (17, .125, 0., False),
        (48, 1., .25, False),
        (19, .5, .125, True),
    ]:
        if positive:
            matrix = weight * np.ones((n, n)) + np.eye(n) * (potential + (n-2)*weight)
            eigenvalues = (potential + (2*n-2)*weight, potential + (n-2)*weight)
        else:
            matrix = -weight * np.ones((n, n)) + np.eye(n) * (potential + n*weight)
            eigenvalues = (potential, potential + n*weight)
        signs = rng.choice([-1., 1.], n)
        matrix = signs[:, None] * matrix * signs[None, :]

        def reference(b0, b1, t, ev=eigenvalues, s=signs):
            x0, x1 = b0.astype(np.longdouble) * s, b1.astype(np.longdouble) * s
            m0, m1 = np.mean(x0, axis=1, keepdims=True), np.mean(x1, axis=1, keepdims=True)
            e = np.exp(-np.longdouble(t) * np.asarray(ev, dtype=np.longdouble))
            f = np.longdouble(t) * phi(np.longdouble(t) * np.asarray(ev, dtype=np.longdouble))
            return np.asarray((e[0]*m0 + e[1]*(x0-m0) + f[0]*m1 + f[1]*(x1-m1))*s,
                              dtype=np.float64)

        fixtures.append((f"graph-{n}-{positive}", matrix, reference, signs))

    q = np.array([[1, 1, 1, 1], [1, -1, 1, -1],
                  [1, 1, -1, -1], [1, -1, -1, 1]], dtype=np.longdouble) / 2
    eigen = np.array([0, 2.5*2**-30, .625, 10.], dtype=np.longdouble)
    matrix = np.asarray((q * eigen) @ q.T, dtype=np.float64)

    def spectral_reference(b0, b1, t):
        x = np.longdouble(t) * eigen
        result = (q @ (np.exp(-x)[:, None] * (q.T @ b0.astype(np.longdouble).T) +
                       (np.longdouble(t)*phi(x))[:, None] *
                       (q.T @ b1.astype(np.longdouble).T))).T
        return np.asarray(result, dtype=np.float64)

    fixtures.append(("log-spectrum", matrix, spectral_reference, np.ones(4)))
    sessions = 0
    maximum = 0.
    for name, matrix, reference, signs in fixtures:
        n = len(matrix)
        norm = float(np.max(np.sum(abs(matrix), axis=1)))
        for eta in [0., .01, 100., 100000.]:
            t = min(10000., eta / norm) if norm else (10000. if eta else 0.)
            b0, b1 = rng.uniform(-.75, .75, (2, 16, n))
            b0[0] = 0
            b1[1] = 0
            b0[2], b1[2] = signs * .25, 0
            b0[3], b1[3] = 0, signs * .25
            b0[4] = b1[4] = 0
            if t > 0:
                b0[5] = min(.25, .25*t) * signs
                b1[5] = -b0[5] / t
            b0[6] *= 1e-16
            b1[6] *= 1e-16
            expected = reference(b0, b1, t)
            for rtol in [1e-4, 1e-7, 1e-10]:
                for requests in [1, 16]:
                    maximum = max(maximum, run_case(
                        name, matrix, b0[:requests], b1[:requests],
                        expected[:requests], t, rtol))
                    sessions += 1
    print(f"Passed {sessions} sessions; maximum scaled error {maximum:.6g}.")


if __name__ == "__main__":
    main()
