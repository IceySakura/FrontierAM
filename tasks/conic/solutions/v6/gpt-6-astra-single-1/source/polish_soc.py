"""Active-boundary Newton polishing for second-order cone programs."""

import time
import warnings
import os
import sys

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla

from problem import blocks
from quality import acceptable, norm, residuals
from scaling import equilibrate, recover


def polish(original, result, deadline):
    p, D, E, cost = equilibrate(original, iterations=4)
    x = result["x"] / D
    z = result["z"] * cost / E
    s = p.b - p.A @ x
    eq = np.zeros(p.m, bool)
    nn = np.zeros(p.m, bool)
    groups = {}
    for kind, dim, sl in blocks(p.cones):
        if kind == "zero":
            eq[sl] = True
        elif kind == "nonnegative":
            nn[sl] = True
        elif kind == "soc":
            groups.setdefault(dim, []).append(np.arange(sl.start, sl.stop))
        else:
            return None
    groups = {d: np.asarray(v) for d, v in groups.items()}
    active_nn = nn & ((s < 1e-4 * (1 + np.abs(p.b))) & (z > 1e-8))
    active_soc = {}
    for d, idx in groups.items():
        gap = s[idx[:, 0]] - np.linalg.norm(s[idx[:, 1:]], axis=1)
        active_soc[d] = (gap < 1e-4 * (1 + np.max(np.abs(s[idx]), axis=1))) & (z[idx[:, 0]] > 1e-8)
    zero_rows = np.diff(p.A.tocsr().indptr) == 0
    eq[zero_rows] = False
    active_nn[zero_rows] = False
    for iteration in range(12):
        if time.monotonic() > deadline:
            return None
        s = p.b - p.A @ x
        linear_rows = np.flatnonzero(eq | active_nn)
        rr = [np.arange(linear_rows.size)]
        cc = [linear_rows]
        vv = [np.ones(linear_rows.size)]
        multipliers = [z[linear_rows]]
        offset = linear_rows.size
        cone_records = []
        hi_r, hi_c, hi_v = [], [], []
        for d, allidx in groups.items():
            idx = allidx[active_soc[d]]
            if not idx.size:
                continue
            sv = s[idx[:, 1:]]
            length = np.linalg.norm(sv, axis=1)
            keep = length > 1e-10
            idx, sv, length = idx[keep], sv[keep], length[keep]
            if not idx.size:
                continue
            u = sv / length[:, None]
            lam = np.maximum(0.0, z[idx[:, 0]])
            rr.append(np.repeat(np.arange(offset, offset + idx.shape[0]), d))
            cc.append(idx.ravel())
            vv.append(np.column_stack((np.ones(idx.shape[0]), -u)).ravel())
            multipliers.append(lam)
            cone_records.append((idx, offset))
            offset += idx.shape[0]
            H = (np.eye(d - 1)[None, :, :] - u[:, :, None] * u[:, None, :])
            H *= (lam / length)[:, None, None]
            hi_r.append(np.repeat(idx[:, 1:], d - 1, axis=1).ravel())
            hi_c.append(np.tile(idx[:, 1:], (1, d - 1)).ravel())
            hi_v.append(H.ravel())
        T = sparse.csr_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                              shape=(offset, p.m))
        J = (T @ p.A).tocsc()
        lam = np.concatenate(multipliers)
        z = T.T @ lam
        candidate = recover({"status": "solved", "x": x, "s": s, "z": z}, D, E, cost)
        if os.environ.get("CONIC_DEBUG"):
            print("SOC polish", iteration, "active", offset,
                  "res", residuals(original, candidate), file=sys.stderr)
        if acceptable(original, candidate):
            return candidate
        H = p.P.copy()
        if hi_v:
            Q = sparse.csc_matrix((np.concatenate(hi_v), (np.concatenate(hi_r), np.concatenate(hi_c))),
                                   shape=(p.m, p.m))
            H = H + p.A.T @ Q @ p.A
        K = sparse.bmat([[H, J.T], [J, sparse.csc_matrix((offset, offset))]], format="csc")
        rhs = np.r_[-p.P @ x - p.q - J.T @ lam, -(J @ x - T @ p.b)]
        reg = np.r_[np.full(p.n, 1e-9), np.full(offset, -1e-9)]
        try:
            large = p.n > 1000
            factor = sla.splu(K + sparse.diags(reg, format="csc"),
                              diag_pivot_thresh=0.0 if large else 0.1,
                              permc_spec="MMD_AT_PLUS_A" if large else "COLAMD",
                              options={"SymmetricMode": large})
            delta = factor.solve(rhs)
            for unused in range(8):
                defect = rhs - K @ delta
                if norm(defect) < 1e-13 * norm(rhs) + 1e-16:
                    break
                trial = delta + factor.solve(defect)
                if norm(rhs - K @ trial) > norm(defect) * 1.1:
                    break
                delta = trial
        except RuntimeError:
            return None
        if not np.isfinite(delta).all() or norm(delta[:p.n]) > 100 * (1 + norm(x)):
            return None
        dx, dl = delta[:p.n], delta[p.n:]
        alpha = min(1.0, 0.2 * (1 + norm(x)) / max(1e-300, norm(dx)))
        x = x + alpha * dx
        lam = lam + alpha * dl
        z = T.T @ lam
        negative = lam[:linear_rows.size] < -1e-9
        active_nn[linear_rows[negative & nn[linear_rows]]] = False
        s = p.b - p.A @ x
        active_nn |= nn & (s < -1e-8)
        for d, idx in groups.items():
            gap = s[idx[:, 0]] - np.linalg.norm(s[idx[:, 1:]], axis=1)
            active_soc[d] |= gap < -1e-8
            active_soc[d] &= z[idx[:, 0]] > -1e-8
    return None
