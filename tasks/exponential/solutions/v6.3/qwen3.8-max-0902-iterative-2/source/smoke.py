"""Build-time numerical self test for the native solver (tiny dense-checked sessions)."""
import json
import struct
import subprocess
import sys

import numpy as np


def reference(L, t, b0, b1):
    w, V = np.linalg.eigh(L)
    z = -t * w
    e = np.exp(z)
    p = np.where(np.abs(z) < 1e-8, 1.0 + z / 2 + z * z / 6, np.expm1(z) / np.where(z == 0, 1, z))
    return V @ (e * (V.T @ b0) + t * p * (V.T @ b1))


def session(binary, L, t, rtol, b0s, b1s):
    n = L.shape[0]
    csr = L.tocsr() if hasattr(L, "tocsr") else L
    indptr = np.asarray(csr.indptr, dtype="<u8")
    indices = np.asarray(csr.indices, dtype="<u4")
    data = np.asarray(csr.data, dtype="<f8")
    req = len(b0s)
    meta = dict(version=2, n=n, nnz=int(data.size), requests=req,
                scenario="one_shot" if req == 1 else "reuse",
                t=float(t), rtol=float(rtol), atol=float(rtol) * 1e-3)
    payload = json.dumps(meta, separators=(",", ":")).encode()
    stdin = [b"FAMEXP2\n", struct.pack("<I", len(payload)), payload,
             indptr.tobytes(), indices.tobytes(), data.tobytes()]
    for k in range(req):
        stdin.append(np.asarray(b0s[k], dtype="<f8").tobytes())
        stdin.append(np.asarray(b1s[k], dtype="<f8").tobytes())
    p = subprocess.run([binary], input=b"".join(stdin), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if p.returncode != 0:
        return None, "exit=%d %s" % (p.returncode, p.stderr.decode()[-200:])
    out = p.stdout
    if not out.startswith(b"READY\n"):
        return None, "bad READY"
    body = out[6:]
    if len(body) != 8 * n * req:
        return None, "wrong byte count %d != %d" % (len(body), 8 * n * req)
    ys = [np.frombuffer(body, dtype="<f8", count=n, offset=8 * n * k) for k in range(req)]
    return ys, None


def mat(rows):
    from scipy.sparse import csr_matrix
    n = len(rows)
    A = np.asarray(rows, dtype=float)
    return csr_matrix(A)


def main():
    binary = sys.argv[1]
    rng = np.random.default_rng(7)
    fails = []

    # 1) small PSD laplacian-like operator, stiff, p10
    A = np.array([[4., -1., -1., 0., 0.],
                  [-1., 3., -1., -1., 0.],
                  [-1., -1., 5., -2., -1.],
                  [0., -1., -2., 4., -1.],
                  [0., 0., -1., -1., 2.]])
    L = mat(A)
    for t, rtol in ((0.7, 1e-10), (3000.0, 1e-10), (3000.0, 1e-4), (0.0, 1e-10)):
        b0 = rng.random(5) * 2 - 1
        b1 = rng.random(5) * 2 - 1
        ys, err = session(binary, L, t, rtol, [b0], [b1])
        if err:
            fails.append(("stiff t=%g rtol=%g" % (t, rtol), err))
            continue
        want = reference(A, t, b0, b1)
        tol = rtol * 1e-3 + rtol * np.abs(want).max()
        got = np.abs(ys[0] - want).max()
        if not (got <= tol * 3 + 1e-300):
            fails.append(("stiff t=%g rtol=%g" % (t, rtol), "err %.3g > tol %.3g" % (got, tol)))

    # 2) reuse session with 16 requests
    b0s = [rng.random(5) * 2 - 1 for _ in range(16)]
    b1s = [rng.random(5) * 2 - 1 for _ in range(16)]
    ys, err = session(binary, L, 12.5, 1e-7, b0s, b1s)
    if err:
        fails.append(("reuse16", err))
    else:
        for k in range(16):
            want = reference(A, 12.5, b0s[k], b1s[k])
            tol = 1e-10 + 1e-7 * np.abs(want).max()
            got = np.abs(ys[k] - want).max()
            if not (got <= tol * 3):
                fails.append(("reuse16 req%d" % k, "err %.3g > tol %.3g" % (got, tol)))
                break

    # 3) diagonal operator (elementwise fast path)
    D = np.diag([0.0, 1e-6, 2.5, 100.0])
    b0 = rng.random(4) * 2 - 1
    b1 = rng.random(4) * 2 - 1
    ys, err = session(binary, mat(D), 1e4, 1e-10, [b0], [b1])
    if err:
        fails.append(("diagonal", err))
    else:
        want = reference(D, 1e4, b0, b1)
        tol = 1e-13 + 1e-10 * np.abs(want).max()
        got = np.abs(ys[0] - want).max()
        if not (got <= tol * 3):
            fails.append(("diagonal", "err %.3g > tol %.3g" % (got, tol)))

    # 4) zero matrix (nnz = 0)
    from scipy.sparse import csr_matrix
    Z = csr_matrix((3, 3))
    b0 = rng.random(3) * 2 - 1
    b1 = rng.random(3) * 2 - 1
    ys, err = session(binary, Z, 5.0, 1e-4, [b0], [b1])
    if err:
        fails.append(("zero", err))
    else:
        want = b0 + 5.0 * b1
        if np.abs(ys[0] - want).max() > 1e-12:
            fails.append(("zero", "err %.3g" % np.abs(ys[0] - want).max()))

    # 5) n = 1
    ys, err = session(binary, mat(np.array([[3.0]])), 2.0, 1e-10, [np.array([0.5])], [np.array([-0.25])])
    if err:
        fails.append(("n1", err))
    else:
        want = reference(np.array([[3.0]]), 2.0, np.array([0.5]), np.array([-0.25]))
        if abs(ys[0][0] - want[0]) > 1e-12 + 1e-10 * abs(want[0]):
            fails.append(("n1", "err %.3g" % abs(ys[0][0] - want[0])))

    if fails:
        for name, msg in fails:
            print("  FAIL %s: %s" % (name, msg))
        return 1
    print("  smoke: all tiny sessions OK")
    return 0


if __name__ == "__main__":
    try:
        import scipy.sparse  # noqa: F401
    except Exception:
        sys.exit(0)
    sys.exit(main())
