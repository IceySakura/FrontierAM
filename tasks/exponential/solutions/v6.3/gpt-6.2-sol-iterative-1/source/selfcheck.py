"""Small, independent FAMEXP2 correctness and protocol checks."""

import json
import math
import random
import struct
import subprocess
from pathlib import Path


def phi(x):
    return -math.expm1(-x) / x if x else 1.0


def run(matrix, t, pairs, solve_exact, rtol=1e-10):
    n = len(matrix)
    row, col, data = [0], [], []
    for entries in matrix:
        for j, value in enumerate(entries):
            if value:
                col.append(j)
                data.append(value)
        row.append(len(col))
    requests = len(pairs)
    atol = rtol * .001
    meta = dict(t=t, requests=requests, version=2, nnz=len(col),
                scenario="one_shot" if requests == 1 else "reuse",
                n=n, atol=atol, rtol=rtol)
    header = json.dumps(meta, indent=2).replace('"n"', '"\\u006e"').encode()
    cwd = Path(__file__).resolve().parent
    with subprocess.Popen(["./run.sh"], cwd=cwd, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE) as process:
        assert process.stdout.read(6) == b"READY\n"
        process.stdin.write(b"FAMEXP2\n" + struct.pack("<I", len(header)) + header)
        process.stdin.write(struct.pack(f"<{n+1}Q", *row))
        process.stdin.write(struct.pack(f"<{len(col)}I", *col))
        process.stdin.write(struct.pack(f"<{len(data)}d", *data))
        for x, q in pairs:
            process.stdin.write(struct.pack(f"<{2*n}d", *(x+q)))
            process.stdin.flush()
            raw = process.stdout.read(8*n)
            assert len(raw) == 8*n
            y = struct.unpack(f"<{n}d", raw)
            ref = solve_exact(x, q)
            assert all(map(math.isfinite, y))
            err = max(abs(a-b) for a,b in zip(y,ref))
            bound = atol + rtol * max(map(abs,ref))
            assert err <= bound, (err, bound, y, ref)
        process.stdin.close()
        assert process.stdout.read(1) == b""
        assert process.wait(timeout=3) == 0


def main():
    rng = random.Random(19)

    def pairs(n, count):
        return [(tuple(rng.uniform(-1,1) for _ in range(n)),
                 tuple(rng.uniform(-1,1) for _ in range(n))) for _ in range(count)]

    run([[0.0]*7 for _ in range(7)], 2.0, pairs(7,16),
        lambda x,q: [a+2*b for a,b in zip(x,q)])
    run([[1., -1.], [-1., 1.]], 0.0, pairs(2,1), lambda x,q: x)
    run([[1., 1.], [1., 1.]], 10000.0, pairs(2,16),
        lambda x,q: [(x[0]-x[1])/2 + 5000*(q[0]-q[1]) + (q[0]+q[1])/4,
                    (x[1]-x[0])/2 + 5000*(q[1]-q[0]) + (q[0]+q[1])/4])

    # Scaled signed rank one, with three exact null directions.
    sign = (1., -1., 1., -1.)
    matrix = [[2*si*sj for sj in sign] for si in sign]
    for t in (1e-10, .15, 10000.):
        def exact(x,q):
            a = math.fsum(si*xi for si,xi in zip(sign,x))/4
            b = math.fsum(si*qi for si,qi in zip(sign,q))/4
            return [xi-a*si+t*(qi-b*si)+si*(math.exp(-8*t)*a+t*phi(8*t)*b)
                    for si,xi,qi in zip(sign,x,q)]
        run(matrix,t,pairs(4,16),exact)
    print("FAMEXP2 protocol and analytic checks passed.")


if __name__ == "__main__":
    main()
