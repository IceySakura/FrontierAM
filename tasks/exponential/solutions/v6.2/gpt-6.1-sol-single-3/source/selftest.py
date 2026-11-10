#!/usr/bin/env python3
"""Small analytic FAMEXP2 checks using only Python's standard library."""
import json
import math
import os
from pathlib import Path
import select
import struct
import subprocess


def read_exact(pipe, size):
    result = bytearray()
    while len(result) < size:
        if not select.select([pipe], [], [], 10)[0]:
            raise RuntimeError("solver response timed out")
        block = os.read(pipe.fileno(), size-len(result))
        if not block:
            raise RuntimeError("solver output ended early")
        result.extend(block)
    return result


def session(name, matrix, t, count, reference):
    n = len(matrix)
    ptr, col, data = [0], [], []
    for row in matrix:
        for j, value in enumerate(row):
            if value:
                col.append(j)
                data.append(value)
        ptr.append(len(col))
    header = dict(version=2, n=n, nnz=len(col), requests=count,
                  scenario="one_shot" if count == 1 else "reuse",
                  t=t, rtol=1e-10, atol=1e-13)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    command = os.environ.get("FAMEXP_TEST_COMMAND",
                             str(Path(__file__).resolve().parent/"run.sh"))
    p = subprocess.Popen([command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, bufsize=0)
    try:
        assert read_exact(p.stdout, 6) == b"READY\n"
        p.stdin.write(b"FAMEXP2\n"+struct.pack("<I", len(encoded))+encoded)
        p.stdin.write(struct.pack("<"+"Q"*len(ptr), *ptr))
        p.stdin.write(struct.pack("<"+"I"*len(col), *col))
        p.stdin.write(struct.pack("<"+"d"*len(data), *data))
        for request in range(count):
            b0 = [math.sin(.7*(request+1)*(j+1)) for j in range(n)]
            b1 = [math.cos(.3*(request+2)*(j+1)) for j in range(n)]
            p.stdin.write(struct.pack("<"+"d"*(2*n), *(b0+b1)))
            expected = reference(b0, b1)
            actual = struct.unpack("<"+"d"*n, read_exact(p.stdout, 8*n))
            threshold = 1e-13+1e-10*max(map(abs, expected))
            assert all(map(math.isfinite, actual)), name
            assert max(abs(a-b) for a, b in zip(actual, expected)) <= threshold, name
        p.stdin.close()
        assert p.wait(timeout=10) == 0, p.stderr.read().decode()
        assert p.stdout.read() == b"", "extra protocol output"
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        p.stdout.close()
        p.stderr.close()
        if not p.stdin.closed:
            p.stdin.close()


def diagonal_reference(diagonal, t):
    def solve(a, b):
        return [math.exp(-t*x)*ai +
                (t if x == 0 else -math.expm1(-t*x)/x)*bi
                for x, ai, bi in zip(diagonal, a, b)]
    return solve


def two_modes(t, plus, minus):
    def scalar(x):
        return math.exp(-t*x), t if x == 0 else -math.expm1(-t*x)/x
    ep, gp = scalar(plus)
    em, gm = scalar(minus)
    def solve(a, b):
        p = ep*(a[0]+a[1])+gp*(b[0]+b[1])
        m = em*(a[0]-a[1])+gm*(b[0]-b[1])
        return [(p+m)/2, (p-m)/2]
    return solve


def main():
    session("zero", [[0.]*4 for _ in range(4)], 3.5, 16,
            diagonal_reference([0.]*4, 3.5))
    session("zero time", [[2., 1.], [1., 2.]], 0., 1, lambda a, b: a)
    d = [0., 2., 30., 1e8]
    session("diagonal", [[x if i == j else 0. for j in range(4)]
                         for i, x in enumerate(d)], .001, 16,
            diagonal_reference(d, .001))
    session("positive off-diagonal", [[2., 1.], [1., 2.]], .75, 16,
            two_modes(.75, 3., 1.))
    session("singular", [[1., -1.], [-1., 1.]], 10000., 16,
            two_modes(10000., 0., 2.))
    session("signed singular", [[1., 1.], [1., 1.]], 10000., 16,
            two_modes(10000., 2., 0.))
    session("stiff scalar", [[1e8]], .001, 1, diagonal_reference([1e8], .001))
    session("near scalar pair", [[1e8, 1e-20], [1e-20, 1e8]], .001, 16,
            diagonal_reference([1e8, 1e8], .001))
    n, a = 65, 1e-312
    session("underflow scale", [[(n-1)*a if i == j else -a for j in range(n)]
                               for i in range(n)], 2.5, 16,
            diagonal_reference([0.]*n, 2.5))
    print("Passed 9 analytic sessions (114 sequential requests).")


if __name__ == "__main__":
    main()
