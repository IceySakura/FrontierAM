"""Fused sparse products and unscaled residual estimates."""
import ctypes as ct
import numpy as np
from native_ldl import _lib, _ip, _dp, ip, dp

_lib.conic_state.argtypes = [
    ct.c_int, ct.c_int, _ip, _ip, _dp, _ip, _ip, _dp,
    _dp, _dp, _dp, _dp, ct.c_double, _dp, _dp, _dp, ct.c_double,
    ct.c_double, ct.c_double, ct.c_double, ct.c_double,
    _dp, _dp, _dp, _dp, _dp]
_lib.conic_state.restype = ct.c_int


class State:
    def __init__(self, P, q, A, b, D, E, cost, norms):
        self.keep = P, q, A, b, D, E
        n, m = len(q), len(b)
        self.static = (n, m, ip(P.indptr), ip(P.indices), dp(P.data),
                       ip(A.indptr), ip(A.indices), dp(A.data),
                       dp(q), dp(b), dp(D), dp(E), cost)
        self.norms = norms
        self.Px, self.Ax = np.empty(n), np.empty(m)
        self.rp, self.rd = np.empty(m), np.empty(n)
        self.values = np.empty(12)
        self.outputs = dp(self.Px), dp(self.Ax), dp(self.rp), dp(self.rd), dp(self.values)

    def compute(self, x, s, z, tau):
        code = _lib.conic_state(*self.static, dp(x), dp(s), dp(z), tau,
                                *self.norms, *self.outputs)
        return code == 0
