#!/usr/bin/env python3
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from scipy.linalg import eigh
import struct, sys, math

P = int(os.environ.get("FMM_P", "8"))
R = int(os.environ.get("FMM_R", "120"))
NPOW = 5
EPS_MAX = 0.25

nodes = np.cos(np.pi * np.arange(P) / (P - 1))[::-1].copy()
bary = np.array([(-1.0)**(P-1-k) * (0.5 if k in (0, P-1) else 1.0) for k in range(P)])
grid = np.array(np.meshgrid(nodes, nodes, nodes, indexing="ij")).reshape(3, -1).T
n3 = P**3

offs_all = [(x,y,z) for x in range(-3,4) for y in range(-3,4) for z in range(-3,4)
            if max(abs(x),abs(y),abs(z)) >= 2]
offs_all.sort()
offs_unique = [o for o in offs_all if o > tuple(-v for v in o)]
off_index = {o:i for i,o in enumerate(offs_all)}

def kernel(off, power):
    d = grid[None,:,:] - grid[:,None,:] - 2.0*np.array(off, dtype=float)
    r2 = np.sum(d*d, axis=2)
    return r2 ** (-(power + 0.5))

print(f"generating p={P} r={R} n3={n3} offsets={len(offs_all)}", flush=True)
GH = np.zeros((n3,n3))
GV = np.zeros((n3,n3))
for oi, off in enumerate(offs_unique):
    for k in range(NPOW):
        K = kernel(off, k)
        GH += K @ K.T
        GV += K.T @ K
    if oi % 20 == 0:
        print("gram", oi, flush=True)

wH, U = eigh(GH, subset_by_index=[n3-R, n3-1])
wV, V = eigh(GV, subset_by_index=[n3-R, n3-1])
U = np.ascontiguousarray(U[:, ::-1])
V = np.ascontiguousarray(V[:, ::-1])
print("eigen done", wH[-1], wV[-1], flush=True)

D = np.zeros((NPOW, len(offs_all), R, R), dtype=np.float64)
for oi, off in enumerate(offs_all):
    for k in range(NPOW):
        K = kernel(off, k)
        D[k, oi] = U.T @ K @ V
    if oi % 40 == 0:
        print("D", oi, flush=True)

def interp_weights(t):
    d = t[:,None] - nodes[None,:]
    exact = np.abs(d) < 1e-14
    d = np.where(exact, 1.0, d)
    L = bary[None,:] / d
    L /= L.sum(axis=1, keepdims=True)
    L[exact.any(axis=1)] = 0.0
    L[exact] = 1.0
    return L

W = np.zeros((2, P, P))
for bit in range(2):
    child_coord = (0.5 if bit else -0.5) + 0.5*nodes
    W[bit] = interp_weights(child_coord)

Lmat = np.zeros((8, R, R))
for octant in range(8):
    bx = (octant >> 2) & 1
    by = (octant >> 1) & 1
    bz = octant & 1
    T = np.kron(W[bx], np.kron(W[by], W[bz]))
    Lmat[octant] = U.T @ T @ U

out = sys.argv[1] if len(sys.argv) > 1 else "fmm_data.bin"
with open(out, "wb") as f:
    f.write(struct.pack("<4siiiiid", b"FMM1", P, R, n3, len(offs_all), NPOW, EPS_MAX))
    f.write(nodes.astype("<f8").tobytes())
    f.write(bary.astype("<f8").tobytes())
    f.write(U.astype("<f8").tobytes())
    f.write(V.astype("<f8").tobytes())
    f.write(D.astype("<f8").tobytes())
    f.write(W.astype("<f8").tobytes())
    f.write(Lmat.astype("<f8").tobytes())
    f.write(np.array(offs_all, dtype=np.int8).tobytes())
print("wrote", out, os.path.getsize(out), flush=True)
