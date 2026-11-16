"""Cone products and Euclidean projections.

Primal cones (all closed, convex):
  zero        {0}^d
  nonnegative R_+^d
  soc         {(t, v) : t >= ||v||_2}, dim d >= 2, first component is the scalar
  psd         S_+^d, encoded as a d(d+1)/2 vector: upper triangle column by
              column, off-diagonal entries multiplied by sqrt(2).
  exponential {(u, v, w) : v > 0, v*exp(u/v) <= w} U {(u, 0, w) : u <= 0, w >= 0}

Every cone here is invariant under positive scaling of an entire block, which
is what lets us row-scale A one cone block at a time.  Every cone is self-dual
except the exponential.  By Moreau's identity, for any closed convex cone C

    Pi_{C*}(w) = w + Pi_C(-w)

so only the primal projections need to be implemented.
"""

import math

import numpy as np

_SQRT2 = math.sqrt(2.0)
_INV_SQRT2 = 1.0 / _SQRT2
_E = math.e


# -------------------------------------------------------------------------- #
# second-order cone
# -------------------------------------------------------------------------- #

def project_soc(block):
    """Projection of a single SOC block (t, v) onto the cone."""
    t = float(block[0])
    rest = block[1:]
    norm = float(np.linalg.norm(rest)) if rest.size else 0.0
    if t >= norm:
        return block.copy()
    if t <= -norm:
        return np.zeros_like(block)
    out = np.empty_like(block)
    out[0] = 0.5 * (t + norm)
    out[1:] = (0.5 * (1.0 + t / norm)) * rest
    return out


# -------------------------------------------------------------------------- #
# PSD cone
# -------------------------------------------------------------------------- #

def smat(u):
    """Recover the symmetric matrix from the sqrt(2)-scaled upper-triangle
    vector stored column by column."""
    d = int(round((math.sqrt(1.0 + 8.0 * u.size) - 1.0) / 2.0))
    M = np.zeros((d, d))
    pos = 0
    for col in range(d):
        k = col + 1
        vals = u[pos:pos + k].copy()
        if k > 1:
            vals[:k - 1] *= _INV_SQRT2
        M[:k, col] = vals
        M[col, :k] = vals
        pos += k
    return M


def svec(M):
    """Encode a symmetric matrix with sqrt(2) on the off-diagonals."""
    d = M.shape[0]
    out = np.empty(d * (d + 1) // 2)
    pos = 0
    for col in range(d):
        k = col + 1
        vals = M[:k, col].copy()
        if k > 1:
            vals[:k - 1] *= _SQRT2
        out[pos:pos + k] = vals
        pos += k
    return out


def project_psd(block):
    """Projection of a PSD block onto the cone of PSD matrices."""
    M = smat(block)
    try:
        eigvals, eigvecs = np.linalg.eigh(M)
    except np.linalg.LinAlgError:
        return np.zeros_like(block)
    eigvals = np.maximum(eigvals, 0.0)
    M2 = (eigvecs * eigvals) @ eigvecs.T
    return svec(M2)


# -------------------------------------------------------------------------- #
# exponential cone
# -------------------------------------------------------------------------- #

# Projection onto the curved face v*exp(u/v) = w, v > 0.  With t = u/v the normal
# to the face is (exp(t), exp(t)*(1-t), -1), so the KKT conditions reduce to a
# single scalar equation F(t) = 0:
#
#     D(t) = t + exp(2t)
#     v(t) = (a + c*exp(t)) / D(t)
#     F(t) = a + c*exp(t) - b*D(t) - (c*t - a*exp(t))*exp(t)*(1-t)
#
# F is monotone on the admissible interval, so a dense scan followed by
# bisection finds the root reliably.  Feasibility additionally requires
# v(t) > 0 and lambda(t) = (c*t - a*exp(t))/D(t) >= 0.

_EXP_T_LO = -35.0
_EXP_T_HI = 35.0
_EXP_NSCAN = 200
_EXP_BISECT = 80


def _exp_face_candidates(a, b, c):
    """Yield feasible points on the curved face closest to (a, b, c)."""
    def F(t):
        et = math.exp(t)
        e2t = et * et
        D = t + e2t
        return a + c * et - b * D - (c * t - a * et) * et * (1.0 - t)

    def v_of(t):
        et = math.exp(t)
        return (a + c * et) / (t + et * et)

    def lam_of(t):
        et = math.exp(t)
        return (c * t - a * et) / (t + et * et)

    # Scan for sign changes.
    prev_t, prev_f = None, None
    for i in range(_EXP_NSCAN + 1):
        t = _EXP_T_LO + (_EXP_T_HI - _EXP_T_LO) * i / _EXP_NSCAN
        try:
            f = F(t)
        except (OverflowError, ValueError):
            continue
        if not math.isfinite(f):
            continue
        if prev_f is not None and f * prev_f < 0.0:
            lo, hi = prev_t, t
            f_lo, f_hi = prev_f, f
            for _ in range(_EXP_BISECT):
                mid = 0.5 * (lo + hi)
                try:
                    fm = F(mid)
                except (OverflowError, ValueError):
                    break
                if not math.isfinite(fm):
                    break
                if f_lo * fm <= 0.0:
                    hi = mid
                else:
                    lo = mid
                    f_lo = fm
                if hi - lo < 1e-15 * max(1.0, abs(lo), abs(hi)):
                    break
            t = 0.5 * (lo + hi)
            et = math.exp(t)
            D = t + et * et
            v = (a + c * et) / D
            lam = (c * t - a * et) / D
            if v > 0.0 and lam >= 0.0 and math.isfinite(v):
                yield np.array([t * v, v, v * et])
        prev_t, prev_f = t, f


def _exp_v0_face(a, b, c):
    """Projection onto the degenerate face v = 0: (u, 0, w) with u <= 0, w >= 0."""
    return np.array([min(a, 0.0), 0.0, max(c, 0.0)])


def _in_exp_cone(v, tol=1e-14):
    u, vv, w = float(v[0]), float(v[1]), float(v[2])
    if vv > 0.0:
        return vv * math.exp(u / vv) <= w + tol
    if vv == 0.0:
        return u <= tol and w >= -tol
    return False


def _in_dual_exponential(v, tol=1e-14):
    """(a, b, c) in -K*, i.e. the projection onto K is the origin."""
    a, b_, c_ = float(v[0]), float(v[1]), float(v[2])
    if a > 0.0:
        return a * math.exp(b_ / a) <= -c_ + tol
    if a == 0.0:
        return b_ <= tol and c_ <= tol
    return False


def project_exponential(v):
    """Euclidean projection of a 3-vector onto the exponential cone K."""
    a, b, c = float(v[0]), float(v[1]), float(v[2])

    # Interior: already in the cone.
    if b > 0.0 and b * math.exp(a / b) <= c:
        return np.array([a, b, c])

    # In the polar: the closest point in K is the origin.
    if _in_dual_exponential(v):
        return np.zeros(3)

    # Curved face.
    best = None
    for p in _exp_face_candidates(a, b, c):
        if best is None or np.sum((p - v) ** 2) < np.sum((best - v) ** 2):
            best = p
    if best is not None:
        return best

    # Degenerate face v = 0.
    p = _exp_v0_face(a, b, c)
    if _in_exp_cone(p):
        return p

    # Origin is always feasible; return it as the last resort.
    return np.zeros(3)


# -------------------------------------------------------------------------- #
# product cone
# -------------------------------------------------------------------------- #

class Cones:
    """Product of cone blocks, with primal and dual projections."""

    def __init__(self, cones):
        """cones: list of dicts {'type': ..., 'dim': ...} in row order."""
        self.blocks = []
        self.widths = []
        self.offsets = []
        off = 0
        for c in cones:
            kind, dim = c["type"], int(c["dim"])
            width = dim * (dim + 1) // 2 if kind == "psd" else dim
            self.blocks.append((kind, dim))
            self.widths.append(width)
            self.offsets.append(off)
            off += width
        self.m = off

    def project(self, v, dual=False):
        """Project v (length m) onto the product cone (dual if dual=True)."""
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        out = np.empty_like(v)
        for i, (kind, dim) in enumerate(self.blocks):
            off = self.offsets[i]
            w = self.widths[i]
            block = v[off:off + w]
            if kind == "zero":
                if dual:
                    out[off:off + w] = block  # dual of {0} is all of R^d
                else:
                    out[off:off + w] = 0.0
            elif kind == "nonnegative":
                out[off:off + w] = np.maximum(block, 0.0)
            elif kind == "soc":
                out[off:off + w] = project_soc(block)
            elif kind == "psd":
                out[off:off + w] = project_psd(block)
            elif kind == "exponential":
                assert w == 3
                if dual:
                    out[off:off + w] = block + project_exponential(-block)  # Moreau
                else:
                    out[off:off + w] = project_exponential(block)
            else:
                raise ValueError("unknown cone type: %s" % kind)
        return out

    def defect(self, v, dual=False):
        """Maximum normalized cone defect over the blocks (checker convention)."""
        worst = 0.0
        for i, (kind, dim) in enumerate(self.blocks):
            off = self.offsets[i]
            w = self.widths[i]
            block = v[off:off + w]
            norm = float(np.max(np.abs(block))) if w else 0.0
            u = block / max(1.0, norm)
            if kind == "zero":
                d = 0.0 if dual else float(np.max(np.abs(u)))
            elif kind == "nonnegative":
                d = max(0.0, -float(np.min(u)))
            elif kind == "soc":
                d = max(0.0, float(np.linalg.norm(u[1:])) - float(u[0]))
            elif kind == "psd":
                M = smat(u)
                try:
                    ev = np.linalg.eigvalsh(M)
                except np.linalg.LinAlgError:
                    ev = np.zeros(1)
                d = max(0.0, -float(ev[0]))
            elif kind == "exponential":
                if dual:
                    tr = (-u[1], -u[0], _E * u[2])
                else:
                    tr = (u[0], u[1], u[2])
                d = _exp_defect(tr)
            else:
                raise ValueError("unknown cone type: %s" % kind)
            worst = max(worst, d)
        return worst


def _exp_defect(v):
    a, b, c = float(v[0]), float(v[1]), float(v[2])
    B = max(abs(b), max(a, 0.0), max(-c, 0.0))
    if b > 0.0 and c > 0.0:
        I = max(0.0, a - b * (math.log(c) - math.log(b)))
        return min(I, B)
    return B
