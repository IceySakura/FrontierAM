"""Exact cone reductions with recovery in the original coordinates."""

import math

import numpy as np
from scipy import sparse
from numeric import linalg

from cones import ConeSystem


def constant_certificate(kind, dim, value):
    """Return a separating dual vector, or None for a feasible constant."""
    scale = max(1., float(np.max(np.abs(value), initial=0.)))
    if kind == "zero":
        return -value if np.max(np.abs(value), initial=0.) > 1e-10 * scale else None
    if kind == "nonnegative":
        j = int(np.argmin(value))
        if value[j] >= -1e-10 * scale:
            return None
        z = np.zeros(dim)
        z[j] = 1
        return z
    if kind == "soc":
        r = np.linalg.norm(value[1:])
        if value[0] >= r - 1e-10 * scale:
            return None
        return np.r_[1., -value[1:] / r] if r else np.r_[1., np.zeros(dim - 1)]
    if kind == "exponential":
        a, b, c = value
        if b < 0:
            return np.array([0., 1., 0.])
        if c < 0:
            return np.array([0., 0., 1.])
        if b == 0:
            if a <= 0:
                return None
            v = max(1., math.log(max(c, 1e-300)) - math.log(a) + 2)
            return np.array([-1., v, math.exp(-v)])
        if c > 0:
            lr = math.log(c) - math.log(b)
            if a <= b * lr:
                return None
            ratio = b / c
            return np.array([-1., lr - 1, ratio])
        ratio = a / b
        v = min(600., max(-600., ratio - 1))
        return np.array([-1., v, math.exp(-v)])
    cs = ConeSystem([{"type": "psd", "dim": dim}])
    mat = cs.smat(value, cs.psd[0])
    vals, vecs = linalg.eigh(mat, subset_by_index=[0, 0], check_finite=False)
    if vals[0] >= -1e-10 * scale:
        return None
    v = vecs[:, 0]
    return cs.svec(np.outer(v, v), cs.psd[0])


class Reduction:
    def __init__(self, A, b, T, cones, kernels):
        self.A, self.b, self.T, self.cones = A, b, T, cones
        self.kernels = kernels
        self.weights = np.asarray(T.power(2).sum(axis=0)).ravel()

    def recover(self, result):
        status = result["status"]
        if status == "unresolved":
            return result
        out = {"status": status}
        if "x" in result:
            x = result["x"]
            s = (self.b - self.A @ x) if status == "solved" else -self.A @ x
            diff = self.T.T @ (result["s"] - self.T @ s)
            use = self.weights > 0
            s[use] += diff[use] / self.weights[use]
            out["x"], out["s"] = x, s
        if "z" in result:
            z = self.T.T @ result["z"]
            if status == "primal_infeasible":
                magnitude = float(np.max(np.abs(z), initial=0.))
                if magnitude:
                    z /= magnitude
            for kind, off, width in self.kernels:
                if kind == "soc":
                    z[off] = np.linalg.norm(z[off + 1:off + width])
                elif kind == "exponential":
                    u, w = -z[off], z[off + 2]
                    if u > 0:
                        if w <= 0:
                            row = self.A.getrow(off + 2)
                            scale = max(1., abs(float(self.b[off + 2])),
                                        float(np.max(np.abs(row.data), initial=0.)))
                            w = max(float(np.nextafter(0., 1.)), 1e-24 / scale)
                            z[off + 2] = w
                        z[off + 1] = max(0., u * (math.log(u) - math.log(w) + 1))
            out["z"] = z
        return out


def reduce(A, b, cones):
    counts = np.bincount(A.indices, minlength=len(b))
    if all(c["type"] in ("zero", "nonnegative") for c in cones) and not np.any(counts == 0):
        return None
    tr, tc, tv, newcones, kernels = [], [], [], [], []
    off = newoff = 0
    changed = False

    def push(indices, kind, dim=None, coeff=None):
        nonlocal newoff
        indices = np.asarray(indices, dtype=int)
        width = len(indices)
        tr.append(np.arange(newoff, newoff + width))
        tc.append(indices)
        tv.append(np.ones(width) if coeff is None else np.asarray(coeff, dtype=float))
        newoff += width
        dimension = width if dim is None else dim
        if kind in ("zero", "nonnegative") and newcones and newcones[-1]["type"] == kind:
            newcones[-1]["dim"] += dimension
        elif dimension:
            newcones.append({"type": kind, "dim": dimension})

    for cone in cones:
        kind, dim = cone["type"], cone["dim"]
        width = dim * (dim + 1) // 2 if kind == "psd" else dim
        rows = np.arange(off, off + width)
        value = b[off:off + width]
        if kind in ("zero", "nonnegative"):
            constant = counts[rows] == 0
            feasible = (np.abs(value) <= 1e-12) if kind == "zero" else (value >= 0)
            bad = rows[constant & ~feasible]
            if len(bad):
                weights = -np.sign(b[bad]) if kind == "zero" else np.ones(len(bad))
                margin = -float((b[bad] / max(float(np.max(np.abs(b), initial=0.)), 1e-300)) @ weights)
                if margin > 1.1e-8:
                    z = np.zeros(len(b))
                    z[bad] = weights
                    return {"status": "primal_infeasible", "z": z}
            take = rows[~constant | ~feasible]
            push(take, kind)
            changed |= len(take) != width
        elif not np.any(counts[rows]):
            cert = constant_certificate(kind, dim, value)
            if cert is not None:
                z = np.zeros(len(b))
                z[rows] = cert
                return {"status": "primal_infeasible", "z": z}
            changed = True
        elif kind == "soc":
            if counts[off] == 0 and b[off] == 0:
                push(rows[1:], "zero")
                kernels.append(("soc", off, width))
                changed = True
            else:
                keep = np.r_[True, (counts[rows[1:]] > 0) | (value[1:] != 0)]
                selected = rows[keep]
                if len(selected) == 1:
                    push(selected, "nonnegative")
                    changed = True
                elif len(selected) == 2:
                    r0, r1 = selected
                    tr.append(np.array([newoff, newoff, newoff + 1, newoff + 1]))
                    tc.append(np.array([r0, r1, r0, r1]))
                    tv.append(np.array([1., 1., 1., -1.]))
                    newoff += 2
                    newcones.append({"type": "nonnegative", "dim": 2})
                    changed = True
                else:
                    push(selected, "soc")
                    changed |= len(selected) != width
        elif kind == "exponential" and counts[off + 1] == 0 and b[off + 1] == 0:
            push([off, off + 2], "nonnegative", coeff=[-1., 1.])
            kernels.append(("exponential", off, width))
            changed = True
        elif kind == "psd":
            active = np.flatnonzero((counts[rows] > 0) | (value != 0))
            col = np.floor((np.sqrt(8. * active + 1) - 1) / 2).astype(int)
            row = active - col * (col + 1) // 2
            if np.all(row == col):
                components, labels = dim, np.arange(dim)
            else:
                from scipy.sparse.csgraph import connected_components
                graph = sparse.coo_matrix((np.ones(len(row)), (row, col)), shape=(dim, dim))
                components, labels = connected_components(graph, directed=False)
            if components == 1 and dim > 1:
                push(rows, "psd", dim)
            else:
                for comp in range(components):
                    idx = np.flatnonzero(labels == comp)
                    cc = np.repeat(idx, np.arange(1, len(idx) + 1))
                    rr = np.concatenate([idx[:j + 1] for j in range(len(idx))])
                    positions = cc * (cc + 1) // 2 + rr
                    take = off + positions
                    if not np.any(counts[take]) and not np.any(b[take]):
                        continue
                    push(take, "nonnegative" if len(idx) == 1 else "psd", len(idx))
                changed = True
        else:
            push(rows, kind, dim)
        off += width
    if not changed:
        return None
    T = sparse.csc_matrix((np.concatenate(tv) if tv else np.empty(0),
                           (np.concatenate(tr) if tr else np.empty(0, dtype=int),
                            np.concatenate(tc) if tc else np.empty(0, dtype=int))),
                          shape=(newoff, len(b)))
    return Reduction(A, b, T, newcones, kernels)
