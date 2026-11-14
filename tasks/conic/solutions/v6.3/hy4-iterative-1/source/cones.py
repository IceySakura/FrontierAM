"""Barrier calculus and membership/defect checks for CVXPY-style cone products.

The encoding of a PSD block is the ``svec`` packing: the upper triangle is
stored column by column and off-diagonal entries are multiplied by ``sqrt(2)``.
With that packing ``<svec(A), svec(B)> = trace(A B)`` for symmetric matrices.

Only NumPy (and the Python standard library) is imported at module import time.
No state is mutated at import time.
"""

import math
import operator
from functools import lru_cache

import numpy as np


__all__ = [
    "ConeData",
    "smat",
    "svec",
    "exp_grad_hess",
    "exp_violation",
]


_SQRT2 = math.sqrt(2.0)
_E = math.e
_TYPES = frozenset(("zero", "nonnegative", "soc", "psd", "exponential"))
_FRACTION = 0.995


# ---------------------------------------------------------------------------
# PSD packing helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _psd_index(dim):
    """Cached column-wise upper-triangle index/scale arrays for order ``dim``."""
    dim = operator.index(dim)
    if dim <= 0:
        raise ValueError("psd dim must be a positive integer")
    rows = np.concatenate([np.arange(j + 1, dtype=np.int64) for j in range(dim)])
    cols = np.concatenate([np.full(j + 1, j, dtype=np.int64) for j in range(dim)])
    diagonal = rows == cols
    # The checker divides off-diagonal encoded entries by sqrt(2) (rather than
    # multiplying by its reciprocal); keep that exact operation here.
    to_denom = np.where(diagonal, 1.0, _SQRT2)
    from_matrix = np.where(diagonal, 1.0, _SQRT2)
    return rows, cols, to_denom, from_matrix


def _psd_width(dim):
    return dim * (dim + 1) // 2


def smat(vec, dim):
    """Unpack a ``svec`` vector into a symmetric matrix.

    ``vec`` may be a 1-D array (width,) returning (dim, dim), or a 2-D array
    (width, k) returning (k, dim, dim) -- the latter is used internally to
    vectorize the PSD Hessian over several right-hand sides.
    """
    dim = operator.index(dim)
    rows, cols, to_denom, _ = _psd_index(dim)
    width = _psd_width(dim)
    v = np.asarray(vec, dtype=np.float64)
    if v.ndim == 1:
        if v.size != width:
            raise ValueError("smat: vector length does not match dim")
        vals = v / to_denom
        out = np.zeros((dim, dim), dtype=np.float64)
        out[rows, cols] = vals
        out[cols, rows] = vals
        return out
    if v.ndim == 2:
        if v.shape[0] != width:
            raise ValueError("smat: vector length does not match dim")
        k = v.shape[1]
        vals = v / to_denom[:, None]
        out = np.zeros((k, dim, dim), dtype=np.float64)
        out[:, rows, cols] = vals.T
        out[:, cols, rows] = vals.T
        return out
    raise ValueError("smat: expected a 1-D or 2-D array")


def svec(mat, dim):
    """Pack the column-wise upper triangle of a symmetric matrix into ``svec``.

    ``mat`` may be (dim, dim) returning (width,), or (k, dim, dim) returning
    (width, k).
    """
    dim = operator.index(dim)
    rows, cols, _, from_matrix = _psd_index(dim)
    a = np.asarray(mat, dtype=np.float64)
    if a.ndim == 2:
        if a.shape != (dim, dim):
            raise ValueError("svec: unexpected matrix shape")
        return a[rows, cols] * from_matrix
    if a.ndim == 3:
        if a.shape[1:] != (dim, dim):
            raise ValueError("svec: unexpected matrix shape")
        return (a[:, rows, cols] * from_matrix[None, :]).T.copy()
    raise ValueError("svec: expected a 2-D or 3-D array")


# ---------------------------------------------------------------------------
# Checker-compatible exponential violation
# ---------------------------------------------------------------------------

def exp_violation(v):
    """``E(x,y,z)`` exactly as ``samples/io/check.py::_exp_violation``."""
    x, y, z = map(float, v)
    boundary = max(abs(y), max(x, 0.0), max(-z, 0.0))
    if y > 0.0 and z > 0.0:
        interior = max(0.0, x - y * (math.log(z) - math.log(y)))
        return min(interior, boundary)
    return boundary


# ---------------------------------------------------------------------------
# Exponential-cone gradient and Hessian (standalone, for testing)
# ---------------------------------------------------------------------------

def exp_grad_hess(block):
    """Return ``(grad F, Hess F)`` for ``F = -log(psi) - log(v) - log(w)``.

    ``psi = v*log(w/v) - u`` on the interior ``v>0, w>0, psi>0`` of K_exp.
    """
    b = np.asarray(block, dtype=np.float64).reshape(-1)
    if b.size != 3:
        raise ValueError("exp_grad_hess: expected a length-3 block")
    u, v, w = float(b[0]), float(b[1]), float(b[2])
    if not (v > 0.0 and w > 0.0):
        raise ValueError("exp_grad_hess: block is not in int K_exp (v,w>0 required)")
    log_ratio = math.log(w) - math.log(v)
    psi = v * log_ratio - u
    if not (psi > 0.0 and math.isfinite(psi)):
        raise ValueError("exp_grad_hess: block is not in int K_exp (psi>0 required)")

    dpsi = np.array([-1.0, log_ratio - 1.0, v / w], dtype=np.float64)
    Hpsi = np.array(
        [[0.0, 0.0, 0.0],
         [0.0, -1.0 / v, 1.0 / w],
         [0.0, 1.0 / w, -v / (w * w)]],
        dtype=np.float64,
    )
    inv_psi = 1.0 / psi
    grad = -dpsi * inv_psi - np.array([0.0, 1.0 / v, 1.0 / w], dtype=np.float64)
    hess = -Hpsi * inv_psi + np.outer(dpsi, dpsi) * (inv_psi * inv_psi)
    hess[1, 1] += 1.0 / (v * v)
    hess[2, 2] += 1.0 / (w * w)
    hess = 0.5 * (hess + hess.T)
    return grad, hess


# ---------------------------------------------------------------------------
# Small shared utilities
# ---------------------------------------------------------------------------

def _width(kind, dim):
    if kind == "psd":
        return _psd_width(dim)
    return dim


def _nu(kind, dim):
    if kind == "zero":
        return 0
    if kind == "nonnegative":
        return dim
    if kind == "soc":
        return 2
    if kind == "psd":
        return dim
    if kind == "exponential":
        return 3
    raise ValueError("unknown cone type")


def _margin_value(margin):
    m = float(margin)
    if not math.isfinite(m) or m < 0.0:
        raise ValueError("margin must be a finite nonnegative scalar")
    return m


def _inf_norm(a):
    return float(np.max(np.abs(a), initial=0.0))


def _block_scale(a):
    return max(1.0, _inf_norm(a))


class ConeData:
    """Barrier data for a product of zero/nonnegative/SOC/PSD/exponential cones.

    Parameters
    ----------
    cones : sequence of dict
        Each dict has ``type`` and ``dim`` keys, in constraint-row order.
    m : int
        Total number of constraint rows; must equal the summed block widths.
    """

    def __init__(self, cones, m):
        try:
            m = operator.index(m)
        except TypeError as exc:
            raise ValueError("m must be an integer") from exc
        if m < 0:
            raise ValueError("m must be nonnegative")

        blocks = []
        zero_rows = []
        cone_rows = []
        start = 0
        nu = 0
        for i, spec in enumerate(cones):
            try:
                kind = spec["type"]
                dim = spec["dim"]
            except (TypeError, KeyError) as exc:
                raise ValueError("cone %d must be a dict with type and dim" % i) from exc
            if kind not in _TYPES:
                raise ValueError("unsupported cone type %r" % (kind,))
            if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)):
                raise ValueError("cone dim must be an integer")
            dim = int(dim)
            if dim <= 0:
                raise ValueError("cone dim must be positive")
            if kind == "soc" and dim < 2:
                raise ValueError("soc dim must be at least 2")
            if kind == "exponential" and dim != 3:
                raise ValueError("exponential cone dim must be 3")
            width = _width(kind, dim)
            stop = start + width
            blocks.append({"type": kind, "dim": dim, "width": width,
                           "start": start, "stop": stop})
            if kind == "zero":
                zero_rows.extend(range(start, stop))
            else:
                cone_rows.extend(range(start, stop))
            nu += _nu(kind, dim)
            start = stop
        if start != m:
            raise ValueError("cone widths sum to %d, expected m=%d" % (start, m))

        self.blocks = blocks
        self.zero_idx = np.asarray(zero_rows, dtype=np.int64)
        self.cone_idx = np.asarray(cone_rows, dtype=np.int64)
        self.nu = int(nu)
        self.m = int(m)
        self.nblocks = len(blocks)
        self.cones = [{"type": b["type"], "dim": b["dim"]} for b in blocks]

    # -- vector handling ----------------------------------------------------

    def _as_vector(self, v, name):
        a = np.asarray(v, dtype=np.float64)
        if a.ndim != 1 or a.size != self.m:
            raise ValueError("%s must be a vector of length %d" % (name, self.m))
        return a

    # -- initial point ------------------------------------------------------

    def s0(self):
        """A strictly interior primal point (zero blocks are exactly zero)."""
        s = np.zeros(self.m, dtype=np.float64)
        for b in self.blocks:
            sl = slice(b["start"], b["stop"])
            kind, dim = b["type"], b["dim"]
            if kind == "zero":
                continue
            if kind == "nonnegative":
                s[sl] = 1.0
            elif kind == "soc":
                s[sl] = 0.0
                s[b["start"]] = 1.1
            elif kind == "psd":
                s[sl] = svec(np.eye(dim), dim)
            else:  # exponential: psi = log(3) > 0
                s[sl] = (0.0, 1.0, 3.0)
        return s

    # -- gradient -----------------------------------------------------------

    def grad(self, s):
        """Gradient of the barrier ``F`` (zero on zero-cone rows)."""
        s = self._as_vector(s, "s")
        g = np.zeros(self.m, dtype=np.float64)
        for b in self.blocks:
            kind = b["type"]
            if kind == "zero":
                continue
            sl = slice(b["start"], b["stop"])
            x = s[sl]
            if kind == "nonnegative":
                g[sl] = -1.0 / x
            elif kind == "soc":
                t, v = float(x[0]), x[1:]
                rho = t * t - float(v @ v)
                g[b["start"]] = -2.0 * t / rho
                g[b["start"] + 1:b["stop"]] = (2.0 / rho) * v
            elif kind == "psd":
                mat = smat(x, b["dim"])
                g[sl] = -svec(np.linalg.inv(mat), b["dim"])
            else:  # exponential
                g[sl] = exp_grad_hess(x)[0]
        return g

    # -- Hessian action -----------------------------------------------------

    def hess_apply(self, s, Y):
        """Apply the block-diagonal barrier Hessian to each column of ``Y``.

        ``Y`` is (m,) or (m, k); the return value has the same shape.  Zero-cone
        rows are left at zero.  No dense m x m Hessian is ever formed.
        """
        s = self._as_vector(s, "s")
        arr = np.asarray(Y, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size != self.m:
                raise ValueError("Y must have leading dimension m")
            mat = arr[:, None]
            squeeze = True
        elif arr.ndim == 2:
            if arr.shape[0] != self.m:
                raise ValueError("Y must have leading dimension m")
            mat = arr
            squeeze = False
        else:
            raise ValueError("Y must be a 1-D or 2-D array")
        out = np.zeros((self.m, mat.shape[1]), dtype=np.float64)

        for b in self.blocks:
            kind = b["type"]
            if kind == "zero":
                continue
            sl = slice(b["start"], b["stop"])
            x = s[sl]
            yb = mat[sl]
            if kind == "nonnegative":
                out[sl] = yb / (x[:, None] ** 2)
            elif kind == "soc":
                t, v = float(x[0]), x[1:]
                rho = t * t - float(v @ v)
                js = np.concatenate(([t], -v))
                ju = np.concatenate((yb[0:1, :], -yb[1:, :]), axis=0)
                dot = js @ yb
                out[sl] = (-2.0 / rho) * ju + (4.0 / rho ** 2) * (js[:, None] * dot[None, :])
            elif kind == "psd":
                dim = b["dim"]
                mat_s = smat(x, dim)
                inv_s = np.linalg.inv(mat_s)
                v_mats = smat(yb, dim)                       # (k, dim, dim)
                # Batched matmul broadcasts (dim, dim) over the k leading
                # matrices: inv_s @ V_j @ inv_s.
                prod = inv_s @ v_mats @ inv_s
                prod = 0.5 * (prod + prod.transpose(0, 2, 1))
                out[sl] = svec(prod, dim)                    # (width, k)
            else:  # exponential
                out[sl] = exp_grad_hess(x)[1] @ yb
        return out[:, 0] if squeeze else out

    # -- interior predicates ------------------------------------------------

    def interior_primal(self, s, margin=0.0):
        """True iff every block is in the strict interior of its primal cone.

        ``margin >= 0`` asks for a relative slack of ``margin * max(1, ||block||inf)``;
        the default 0 means strict membership.  Zero-cone rows must be ~0.
        """
        s = self._as_vector(s, "s")
        if not np.isfinite(s).all():
            return False
        margin = _margin_value(margin)
        global_scale = _block_scale(s)
        zero_tol = 1e-10 * global_scale
        for b in self.blocks:
            sl = slice(b["start"], b["stop"])
            x = s[sl]
            kind = b["type"]
            eps = margin * _block_scale(x)
            if kind == "zero":
                if _inf_norm(x) > zero_tol:
                    return False
            elif kind == "nonnegative":
                if float(np.min(x)) <= eps:
                    return False
            elif kind == "soc":
                t = float(x[0])
                nv = float(np.linalg.norm(x[1:]))
                if not (t > eps and t - nv > eps):
                    return False
            elif kind == "psd":
                mat = smat(x, b["dim"])
                try:
                    np.linalg.cholesky(mat)
                except np.linalg.LinAlgError:
                    return False
                if margin > 0.0 and float(np.linalg.eigvalsh(mat)[0]) <= eps:
                    return False
            else:  # exponential
                u, v, w = float(x[0]), float(x[1]), float(x[2])
                if v <= eps or w <= eps:
                    return False
                psi = v * (math.log(w) - math.log(v)) - u
                if not (math.isfinite(psi) and psi > eps):
                    return False
        return True

    def interior_dual(self, z, margin=0.0):
        """True iff every block is in the strict interior of its dual cone.

        Zero-cone rows are free.  For the exponential cone the checker's
        transform ``(a,b,c) = (-u_v, -u_u, e*u_w)`` is used with
        ``u = block / max(1, ||block||inf)``; strict dual interiority is then
        ``b>0, c>0, a < b*(log c - log b)``.  Since that inequality is
        positively homogeneous, the normalization does not change membership.
        """
        z = self._as_vector(z, "z")
        if not np.isfinite(z).all():
            return False
        margin = _margin_value(margin)
        for b in self.blocks:
            sl = slice(b["start"], b["stop"])
            x = z[sl]
            kind = b["type"]
            eps = margin * _block_scale(x)
            if kind == "zero":
                continue
            if kind == "nonnegative":
                if float(np.min(x)) <= eps:
                    return False
            elif kind == "soc":
                t = float(x[0])
                nv = float(np.linalg.norm(x[1:]))
                if not (t > eps and t - nv > eps):
                    return False
            elif kind == "psd":
                mat = smat(x, b["dim"])
                try:
                    np.linalg.cholesky(mat)
                except np.linalg.LinAlgError:
                    return False
                if margin > 0.0 and float(np.linalg.eigvalsh(mat)[0]) <= eps:
                    return False
            else:  # exponential
                # u is normalized by scale = max(1, ||x||inf); because E is
                # positively homogeneous, a slack of margin*scale in the
                # original block is exactly margin on the normalized one.
                u = x / _block_scale(x)
                a, bb, c = -float(u[1]), -float(u[0]), _E * float(u[2])
                if bb <= margin or c <= margin:
                    return False
                slack = bb * (math.log(c) - math.log(bb)) - a
                if not (math.isfinite(slack) and slack > margin):
                    return False
        return True

    # -- maximum step to the boundary ---------------------------------------

    @staticmethod
    def _soc_cap(x, dx):
        """First alpha>0 at which (x+alpha dx) leaves int SOC, or inf."""
        t, v = float(x[0]), x[1:]
        dt, dv = float(dx[0]), dx[1:]
        a = dt * dt - float(dv @ dv)
        b = 2.0 * (t * dt - float(v @ dv))
        c = t * t - float(v @ v)
        if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)) or c <= 0.0:
            return None
        if a == 0.0:
            return -c / b if b < 0.0 else math.inf
        disc = b * b - 4.0 * a * c
        if not math.isfinite(disc) or disc < 0.0:
            return math.inf
        root = math.sqrt(disc)
        # Stable quadratic formula: q = -0.5*(b + sign(b)*sqrt(disc)).
        q = -0.5 * (b + (root if b >= 0.0 else -root))
        if q == 0.0:
            return math.inf
        candidates = [r for r in (q / a, c / q) if math.isfinite(r) and r > 0.0]
        return min(candidates) if candidates else math.inf

    @staticmethod
    def _psd_cap(x, dx, dim):
        """First alpha>0 at which smat(x+alpha dx) stops being PD, or inf."""
        mat = smat(x, dim)
        dmat = smat(dx, dim)
        try:
            chol = np.linalg.cholesky(mat)
            l_inv = np.linalg.inv(chol)
            cent = l_inv @ dmat @ l_inv.T
            cent = 0.5 * (cent + cent.T)
            lam_min = float(np.linalg.eigvalsh(cent)[0])
        except np.linalg.LinAlgError:
            return None
        if not math.isfinite(lam_min):
            return None
        return -1.0 / lam_min if lam_min < 0.0 else math.inf

    @staticmethod
    def _nonneg_cap(x, dx):
        decreasing = dx < 0.0
        if not np.any(decreasing):
            return math.inf
        caps = -x[decreasing] / dx[decreasing]
        if np.any(caps < 0.0):
            return 0.0
        value = float(np.min(caps))
        return value if math.isfinite(value) else math.inf

    @staticmethod
    def _exp_ok(u, v, w, dual):
        """Scalar strict-interiority test for a primal/dual exponential block."""
        if dual:
            a, b, c = -v, -u, _E * w
            if b <= 0.0 or c <= 0.0:
                return False
            return math.isfinite(b * (math.log(c) - math.log(b)) - a) and \
                b * (math.log(c) - math.log(b)) - a > 0.0
        if v <= 0.0 or w <= 0.0:
            return False
        psi = v * (math.log(w) - math.log(v)) - u
        return math.isfinite(psi) and psi > 0.0

    @classmethod
    def _exp_cap(cls, x, dx, dual):
        """Bisection for the first alpha at which an exp block leaves its cone."""
        u, v, w = float(x[0]), float(x[1]), float(x[2])
        du, dv, dw = float(dx[0]), float(dx[1]), float(dx[2])
        if not cls._exp_ok(u, v, w, dual):
            return 0.0
        if cls._exp_ok(u + du, v + dv, w + dw, dual):
            return math.inf
        lo, hi = 0.0, 1.0
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            if cls._exp_ok(u + mid * du, v + mid * dv, w + mid * dw, dual):
                lo = mid
            else:
                hi = mid
        return lo

    def _block_cap(self, b, x, dx, dual=False):
        kind = b["type"]
        if kind == "zero":
            return math.inf
        if kind == "nonnegative":
            return self._nonneg_cap(x, dx)
        if kind == "soc":
            return self._soc_cap(x, dx)
        if kind == "psd":
            return self._psd_cap(x, dx, b["dim"])
        return self._exp_cap(x, dx, dual)

    def _max_step(self, x, dx, dual):
        x = self._as_vector(x, "x")
        dx = self._as_vector(dx, "dx")
        if not (np.isfinite(x).all() and np.isfinite(dx).all()):
            return 0.0
        interior = self.interior_dual if dual else self.interior_primal
        if not interior(x):
            return 0.0
        if not dual and self.zero_idx.size:
            tol = 1e-12 * _block_scale(dx)
            if _inf_norm(dx[self.zero_idx]) > tol:
                # A primal zero-cone row must remain exactly zero.
                return 0.0
        hi = 1.0
        for b in self.blocks:
            if b["type"] == "zero":
                continue
            sl = slice(b["start"], b["stop"])
            cap = self._block_cap(b, x[sl], dx[sl], dual)
            if cap is not None and cap < hi:
                hi = max(0.0, float(cap))
                if hi <= 0.0:
                    return 0.0
        if hi >= 1.0:
            if interior(x + dx):
                return 1.0
            hi = 1.0
        else:
            candidate = _FRACTION * hi
            if interior(x + candidate * dx):
                return candidate
        # Safety net: an analytic bound failed verification, bracket properly.
        if interior(x + hi * dx):
            if hi >= 1.0 or interior(x + dx):
                return 1.0
            hi = 1.0
        lo = 0.0
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            if interior(x + mid * dx):
                lo = mid
            else:
                hi = mid
        return float(min(1.0, max(0.0, _FRACTION * lo)))

    def max_step_primal(self, s, ds):
        """Largest interior step in [0,1], multiplied by 0.995."""
        return self._max_step(s, ds, dual=False)

    def max_step_dual(self, z, dz):
        """Largest dual-interior step in [0,1], multiplied by 0.995."""
        return self._max_step(z, dz, dual=True)

    # -- checker-compatible cone defects ------------------------------------

    def _defect(self, v, dual):
        v = self._as_vector(v, "v")
        if not np.isfinite(v).all():
            return float("inf")
        worst = 0.0
        for b in self.blocks:
            sl = slice(b["start"], b["stop"])
            block = v[sl]
            u = block / _block_scale(block)
            kind = b["type"]
            if kind == "zero":
                violation = 0.0 if dual else _inf_norm(u)
            elif kind == "nonnegative":
                violation = max(0.0, -float(np.min(u)))
            elif kind == "soc":
                violation = max(0.0, float(np.linalg.norm(u[1:])) - float(u[0]))
            elif kind == "exponential":
                transformed = (-u[1], -u[0], _E * u[2]) if dual else u
                violation = exp_violation(transformed)
            else:  # psd
                mat = smat(u, b["dim"])
                violation = max(0.0, -float(np.linalg.eigvalsh(mat)[0]))
            if violation > worst:
                worst = violation
        return float(worst)

    def defect_primal(self, s):
        """C_K(s), exactly matching ``check.py::_cones(..., dual=False)``."""
        return self._defect(s, dual=False)

    def defect_dual(self, z):
        """C_K*(z), exactly matching ``check.py::_cones(..., dual=True)``."""
        return self._defect(z, dual=True)
