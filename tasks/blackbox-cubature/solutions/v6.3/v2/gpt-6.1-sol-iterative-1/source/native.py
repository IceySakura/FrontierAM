"""Atomic point-budget wrapper around the provided p-adaptive rule."""
import ctypes
import time
from pathlib import Path

import numpy as np


_DoublePointer = ctypes.POINTER(ctypes.c_double)
_Callback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_uint, ctypes.c_size_t, _DoublePointer,
    ctypes.c_void_p, ctypes.c_uint, _DoublePointer)


def integrate_p(oracle, deadline):
    d = oracle.dimension
    available = oracle.budget - oracle.used
    if available < 3 ** d:
        return None
    library = ctypes.CDLL(str(Path(__file__).resolve().parent / "libpcubature.so"))
    fn = library.pcubature_v
    fn.argtypes = [
        ctypes.c_uint, _Callback, ctypes.c_void_p, ctypes.c_uint,
        _DoublePointer, _DoublePointer, ctypes.c_size_t, ctypes.c_double,
        ctypes.c_double, ctypes.c_int, _DoublePointer, _DoublePointer]
    fn.restype = ctypes.c_int
    errors = []
    before = oracle.used

    @_Callback
    def callback(ndim, count, xptr, _data, fdim, output):
        count = int(count)
        if count > oracle.budget - oracle.used or time.monotonic() > deadline:
            return 1
        try:
            points = np.ctypeslib.as_array(xptr, shape=(count * ndim,)).reshape(count, ndim)
            values = oracle.evaluate(points)
            target = np.ctypeslib.as_array(output, shape=(count * fdim,))
            target[:] = values
            return 0
        except Exception as error:
            errors.append(error)
            return 1

    lower = np.zeros(d)
    upper = np.ones(d)
    value = np.full(1, np.nan)
    error = np.full(1, np.nan)
    fn(1, callback, None, d,
       lower.ctypes.data_as(_DoublePointer), upper.ctypes.data_as(_DoublePointer),
       available, 0.0, 0.0, 0, value.ctypes.data_as(_DoublePointer),
       error.ctypes.data_as(_DoublePointer))
    if errors:
        raise errors[0]
    if oracle.used <= before or not np.isfinite(value[0]):
        return None
    return float(value[0])
