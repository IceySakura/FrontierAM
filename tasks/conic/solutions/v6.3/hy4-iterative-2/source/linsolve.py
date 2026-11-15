"""Cached LU factorization via the system LAPACK, using ctypes only."""
import ctypes
import glob

import numpy as np

_LIB = None
_DGETRF = None
_DGETRS = None
_PI = ctypes.POINTER(ctypes.c_int)
_PD = ctypes.POINTER(ctypes.c_double)


def _lib():
    global _LIB, _DGETRF, _DGETRS
    if _LIB is not None:
        return _LIB
    candidates = []
    for p in (glob.glob("/usr/local/lib/python3*/site-packages/scipy.libs/libscipy_openblas*.so")
              + glob.glob("/usr/lib/python3/dist-packages/scipy.libs/libscipy_openblas*.so")):
        candidates.append((p, "scipy_dgetrf_", "scipy_dgetrs_"))
    candidates += [("liblapack.so.3", "dgetrf_", "dgetrs_"),
                   ("liblapack.so", "dgetrf_", "dgetrs_")]
    for name, sym_f, sym_s in candidates:
        try:
            lib = ctypes.CDLL(name)
            f = getattr(lib, sym_f)
            s = getattr(lib, sym_s)
            f.argtypes = [_PI, _PI, _PD, _PI, _PI, _PI]
            f.restype = None
            s.argtypes = [ctypes.c_char_p, _PI, _PI, _PD, _PI, _PI, _PD, _PI, _PI]
            s.restype = None
            _LIB, _DGETRF, _DGETRS = lib, f, s
            return lib
        except Exception:
            continue
    return None


class LU:
    """Reusable LU of a (symmetrically) equilibrated matrix."""

    def __init__(self, K, equilibrate=True):
        lib = _lib()
        if lib is None:
            raise RuntimeError("LAPACK unavailable")
        K = np.asarray(K, dtype=np.float64)
        self.n = K.shape[0]
        if equilibrate:
            diag = np.abs(np.diag(K))
            self.d = 1.0 / np.sqrt(np.maximum(diag, 1e-12))
            self.Kref = K * self.d[:, None] * self.d[None, :]
        else:
            self.d = None
            self.Kref = K
        # K is symmetric, so a C-order buffer read as column-major is K^T = K.
        self.a = np.array(self.Kref, dtype=np.float64, order="C", copy=True)
        self.ipiv = np.zeros(self.n, dtype=np.int32)
        n = ctypes.c_int(self.n)
        lda = ctypes.c_int(self.n)
        info = ctypes.c_int(0)
        _DGETRF(ctypes.byref(n), ctypes.byref(n),
                self.a.ctypes.data_as(_PD), ctypes.byref(lda),
                self.ipiv.ctypes.data_as(_PI), ctypes.byref(info))
        if info.value != 0:
            raise np.linalg.LinAlgError("singular LU, pivot %d" % info.value)

    def _solve_scaled(self, bs):
        b = np.array(bs, dtype=np.float64, order="F", copy=True)
        n = ctypes.c_int(self.n)
        nrhs = ctypes.c_int(1)
        lda = ctypes.c_int(self.n)
        info = ctypes.c_int(0)
        _DGETRS(b"N", ctypes.byref(n), ctypes.byref(nrhs),
                self.a.ctypes.data_as(_PD), ctypes.byref(lda),
                self.ipiv.ctypes.data_as(_PI),
                b.ctypes.data_as(_PD), ctypes.byref(n),
                ctypes.byref(info))
        if info.value != 0:
            raise np.linalg.LinAlgError("dgetrs failed")
        return b

    def refine(self, rhs, steps=2):
        bs = rhs * self.d if self.d is not None else rhs
        y = self._solve_scaled(bs)
        for _ in range(steps):
            y += self._solve_scaled(bs - self.Kref @ y)
        return y * self.d if self.d is not None else y


def solve(K, rhs, refine=2):
    """One-shot equilibrated solve (for callers that do not reuse the factor)."""
    try:
        return LU(K).refine(rhs, refine)
    except np.linalg.LinAlgError:
        diag = np.abs(np.diag(K))
        d = 1.0 / np.sqrt(np.maximum(diag, 1e-12))
        Ks = K * d[:, None] * d[None, :]
        return np.linalg.lstsq(Ks, rhs * d, rcond=None)[0] * d
