#!/usr/bin/env python3
"""Small analytic FAMEXP2 checks; Python standard library only."""
import json
import math
import signal
import struct
import subprocess
from pathlib import Path


def read_exact(stream, size):
    result = bytearray()
    while len(result) < size:
        block = stream.read(size - len(result))
        if not block:
            raise AssertionError("short protocol response")
        result.extend(block)
    return bytes(result)


def session(matrix, t, requests, reference):
    n = len(matrix)
    ptr, indices, data = [0], [], []
    for row in matrix:
        for j, value in enumerate(row):
            if value:
                indices.append(j)
                data.append(value)
        ptr.append(len(data))
    meta = dict(version=2, n=n, nnz=len(data), requests=len(requests),
                scenario="one_shot" if len(requests) == 1 else "reuse",
                t=t, rtol=1e-10, atol=1e-13)
    payload = json.dumps(meta).encode()
    process = subprocess.Popen(["./run.sh"], cwd=Path(__file__).resolve().parent,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    try:
        assert read_exact(process.stdout, 6) == b"READY\n"
        process.stdin.write(b"FAMEXP2\n" + struct.pack("<I", len(payload)) + payload)
        process.stdin.write(struct.pack(f"<{n + 1}Q", *ptr))
        process.stdin.write(struct.pack(f"<{len(indices)}I", *indices))
        process.stdin.write(struct.pack(f"<{len(data)}d", *data))
        for b0, b1 in requests:
            process.stdin.write(struct.pack(f"<{2*n}d", *b0, *b1))
            process.stdin.flush()
            actual = struct.unpack(f"<{n}d", read_exact(process.stdout, 8 * n))
            expected = reference(b0, b1)
            tolerance = 1e-13 + 1e-10 * max(map(abs, expected))
            assert all(math.isfinite(v) for v in actual)
            assert max(abs(a-b) for a, b in zip(actual, expected)) <= tolerance
        process.stdin.close()
        assert process.stdout.read() == b""
        assert process.wait(timeout=10) == 0, process.stderr.read().decode()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()


def main():
    signal.alarm(30)
    session([[0, 0], [0, 0]], 2.0, [([0.2, -0.4], [0.3, 0.1])],
            lambda b, c: [b[i] + 2*c[i] for i in range(2)])
    session([[2, -1], [-1, 2]], 0.0, [([0.2, -0.4], [0.3, 0.1])],
            lambda b, c: b)
    session([[1e8]], 0.001, [([0.7], [-0.4])], lambda b, c: [c[0] / 1e8])
    requests = [([(-1)**j * 0.3, 0.2], [0.1, (-1)**j * 0.4]) for j in range(16)]

    def positive_offdiag(b, c):
        low = (b[0] - b[1]) / 2 + 0.5 * (c[0] - c[1]) / 2
        high = math.exp(-1) * (b[0] + b[1]) / 2
        high += -math.expm1(-1) / 2 * (c[0] + c[1]) / 2
        return [high + low, high - low]

    session([[1, 1], [1, 1]], 0.5, requests, positive_offdiag)
    tiny = [[2e-300, -1e-300, -1e-300],
            [-1e-300, 2e-300, -1e-300],
            [-1e-300, -1e-300, 2e-300]]
    session(tiny, 1e-20, [([0.1, 0.2, -0.3], [0.2, -0.4, 0.1])],
            lambda b, c: [b[i] + 1e-20*c[i] for i in range(3)])
    signal.alarm(0)
    print("All analytic protocol checks passed.")


if __name__ == "__main__":
    main()
