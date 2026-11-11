"""Cone bookkeeping and vectorised cone defects / separating hyperplanes.

Encoding follows the task specification: PSD uses svec (upper triangle,
column-wise, off-diagonal entries multiplied by sqrt(2)); exponential cones
use (u, v, w); SOC uses (t, v).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SQ2 = math.sqrt(2.0)


@dataclass
class ConeInfo:
    blocks: list                       # (kind, dim, start, width)
    m: int
    groups: dict = field(default_factory=dict)
    # groups[(kind,dim)] = dict(starts=int32[num,width], defect_index=int32[num], kind, dim)
    block_of_row_groups: dict = field(default_factory=dict)

    @classmethod
    def build(cls, cones):
        blocks = []
        k = 0
        for c in cones:
            kind, dim = c["type"], int(c["dim"])
            width = dim * (dim + 1) // 2 if kind == "psd" else dim
            blocks.append((kind, dim, k, width))
            k += width
        obj = cls(blocks, k)
        bykey = {}
        for bi, (kind, dim, st, w) in enumerate(blocks):
            bykey.setdefault((kind, dim), []).append((bi, st, w))
        for key, lst in bykey.items():
            kind, dim = key
            w = lst[0][2]
            starts = np.array([t[1] for t in lst], dtype=np.int64)
            idx = (starts[:, None] + np.arange(w, dtype=np.int64)[None, :])
            obj.groups[key] = dict(starts=starts, idx=idx,
                                   bis=np.array([t[0] for t in lst], dtype=np.int64),
                                   kind=kind, dim=dim, width=w)
        return obj

    def gather(self, v, key):
        g = self.groups[key]
        return np.asarray(v)[g["idx"]]

    def defects(self, v, dual=False):
        """Return (worst, per-block-defect array)."""
        v = np.asarray(v, dtype=np.float64)
        out = np.zeros(len(self.blocks), dtype=np.float64)
        worst = 0.0
        for key, g in self.groups.items():
            kind, dim, w = g["kind"], g["dim"], g["width"]
            blk = v[g["idx"]]
            scale = np.maximum(1.0, np.max(np.abs(blk), axis=1))
            u = blk / scale[:, None]
            if kind == "zero":
                d = np.zeros(len(blk)) if dual else np.max(np.abs(u), axis=1)
            elif kind == "nonnegative":
                d = np.maximum(0.0, -np.min(u, axis=1))
            elif kind == "soc":
                d = np.maximum(0.0, np.linalg.norm(u[:, 1:], axis=1) - u[:, 0])
            elif kind == "exponential":
                if dual:
                    ut = np.stack((-u[:, 1], -u[:, 0], math.e * u[:, 2]), axis=1)
                else:
                    ut = u
                d = _exp_defect(ut)
            else:  # psd
                d = np.empty(len(blk))
                for i in range(len(blk)):
                    M = smat(u[i], dim)
                    try:
                        d[i] = max(0.0, -float(np.linalg.eigvalsh(M)[0]))
                    except np.linalg.LinAlgError:
                        d[i] = 0.0
            out[g["bis"]] = d
            if len(d):
                worst = max(worst, float(np.max(d)))
        return worst, out


def _exp_defect(u):
    a, b, c = u[:, 0], u[:, 1], u[:, 2]
    boundary = np.maximum.reduce((np.abs(b), np.maximum(a, 0.0), np.maximum(-c, 0.0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        interior = np.maximum(0.0, a - b * (np.log(c) - np.log(b)))
    good = (b > 0) & (c > 0)
    return np.where(good, np.minimum(boundary, interior), boundary)


def smat(v, d):
    """svec -> dense symmetric matrix."""
    M = np.zeros((d, d), dtype=np.float64)
    k = 0
    for j in range(d):
        vals = np.array(v[k:k + j + 1], dtype=np.float64, copy=True)
        vals[:j] /= SQ2
        M[:j + 1, j] = vals
        M[j, :j + 1] = vals
        k += j + 1
    return M


def svec(M, d):
    """dense symmetric matrix -> svec."""
    out = []
    for j in range(d):
        out.extend(M[:j, j] * SQ2)
        out.append(M[j, j])
    return np.asarray(out, dtype=np.float64)


def exp_dual_cut(u, t_min=-25.0, t_max=30.0):
    """A dual-cone vector c in K_exp^* violated by s=(u,v,w), if possible.

    Boundary tangent: c(t) = (-1, t-1, exp(-t)), t = u/v.  c(s) < 0 exactly
    when w < v*exp(u/v).  For non-positive v or w, coordinate cuts are used.
    """
    a, b, c = map(float, u)
    out = []
    if b <= 0:
        out.append(np.array([0.0, 1.0, 0.0]))
    if c < 0:
        out.append(np.array([0.0, 0.0, 1.0]))
    if b > 0:
        t = float(np.clip(a / b, t_min, t_max))
        out.append(np.array([-1.0, t - 1.0, math.exp(-t)]))
    return out
