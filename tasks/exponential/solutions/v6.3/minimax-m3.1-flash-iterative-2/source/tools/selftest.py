"""Self test of a candidate solver binary on a small analytic case.

Usage: python3 tools/selftest.py ./bin/fastsolve
Exits with status 0 when the solver answers the analytic case correctly.
"""
import json
import struct
import subprocess
import sys

import numpy as np
import shlex


def main(argv):
    if len(argv) < 2:
        print("usage: selftest.py <command>", file=sys.stderr)
        return 2
    cmd = argv[1]
    n = 3
    diag = np.array([0.0, 1.0, 4.0])
    indptr = np.arange(n + 1, dtype=np.uint64)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    data = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    t = 0.7
    rtol, atol = 1e-10, 1e-13
    b0 = np.array([0.5, -0.25, 0.75])
    b1 = np.array([0.1, 0.2, -0.3])

    meta = dict(version=2, n=n, nnz=int(data.size), requests=1,
                scenario="one_shot", t=t, rtol=rtol, atol=atol)
    payload = json.dumps(meta, separators=(",", ":")).encode("utf-8")

    p = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, bufsize=0)
    if p.stdout.read(6) != b"READY\n":
        print("selftest: no READY", file=sys.stderr)
        return 1
    p.stdin.write(b"FAMEXP2\n" + struct.pack("<I", len(payload)) + payload)
    p.stdin.write(indptr.tobytes())
    p.stdin.write(indices.tobytes())
    p.stdin.write(data.tobytes())
    p.stdin.write(b0.tobytes())
    p.stdin.write(b1.tobytes())
    p.stdin.flush()
    raw = p.stdout.read(8 * n)
    p.stdin.close()
    rc = p.wait(timeout=120)
    if rc != 0 or len(raw) != 8 * n:
        print(f"selftest: bad exit {rc}", file=sys.stderr)
        return 1
    y = np.frombuffer(raw, dtype="<f8")

    def phi1(z):
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        small = np.abs(z) < 1e-12
        out[small] = 1.0
        out[~small] = np.expm1(z[~small]) / z[~small]
        return out

    ref = np.exp(-t * diag) * b0 + t * phi1(-t * diag) * b1
    err = float(np.max(np.abs(y - ref)))
    thr = atol + rtol * float(np.max(np.abs(ref)))
    if not (err <= thr):
        print(f"selftest: error {err:.3e} > {thr:.3e}", file=sys.stderr)
        return 1
    print("selftest: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
