#!/usr/bin/env python3
"""Offline, single-core fluorescence deconvolution."""
import ast
import ctypes
import math
import os
import struct
import sys
import time
import zipfile

MODEL_OPTIONS = 64 | 512


def read_array(archive, name):
    member = name + ".npy" if name + ".npy" in archive.namelist() else name
    data = bytearray(archive.read(member))
    if data[:6] != b"\x93NUMPY":
        raise ValueError("Invalid NPY input")
    major = data[6]
    if major == 1:
        size = struct.unpack_from("<H", data, 8)[0]
        offset = 10
    elif major in (2, 3):
        size = struct.unpack_from("<I", data, 8)[0]
        offset = 12
    else:
        raise ValueError("Unsupported NPY version")
    header = ast.literal_eval(bytes(data[offset:offset + size]).decode("latin1"))
    shape = tuple(header["shape"])
    if header["descr"] not in ("<f4", ">f4", "=f4", "|f4") or len(shape) != 3:
        raise ValueError("Expected three-dimensional float32")
    return data, offset + size, shape, bool(header["fortran_order"]), header["descr"] == ">f4"


def native_order(record, lib):
    data, offset, shape, fortran, swap = record
    if fortran or swap:
        reordered = bytearray(math.prod(shape) * 4)
        convert = lib.reorder_float32
        convert.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 5
        convert.restype = None
        convert(ctypes.addressof(ctypes.c_char.from_buffer(data, offset)),
                ctypes.addressof(ctypes.c_char.from_buffer(reordered)),
                *shape, int(fortran), int(swap))
        data, offset = reordered, 0
    return data, offset, shape


def write_array(path, data, shape):
    header = repr({"descr": "<f4", "fortran_order": False, "shape": shape}).encode("ascii")
    header += b" " * ((-len(header) - 11) % 64) + b"\n"
    with open(path, "wb") as stream:
        stream.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
        stream.write(data)


def main():
    started = time.monotonic()
    with zipfile.ZipFile(sys.argv[1]) as archive:
        observed_record = read_array(archive, "observed")
        psf_record = read_array(archive, "psf")
    configured = float(os.environ.get("TIME_BUDGET_SECONDS", "0.5"))
    deadline = float(os.environ.get("DECONV_DEADLINE_MONOTONIC",
                                   str(started + max(.1, configured))))
    deadline -= .045
    own_dir = os.path.dirname(os.path.abspath(__file__))
    lib = ctypes.CDLL(os.path.join(own_dir, "solver.so"))
    if lib.fast_cpu_available():
        lib = ctypes.CDLL(os.path.join(own_dir, "solver_fast.so"))
    observed, yo, shape = native_order(observed_record, lib)
    psf, ho, hshape = native_order(psf_record, lib)
    result = bytearray(math.prod(shape) * 4)
    solve = lib.reconstruct
    solve.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_double, ctypes.c_int]
    solve.restype = ctypes.c_int
    solve(ctypes.addressof(ctypes.c_float.from_buffer(observed, yo)),
          ctypes.addressof(ctypes.c_float.from_buffer(psf, ho)),
          ctypes.addressof(ctypes.c_float.from_buffer(result)),
          *shape, *hshape, deadline, MODEL_OPTIONS)
    write_array(sys.argv[2], result, shape)


if __name__ == "__main__":
    main()
