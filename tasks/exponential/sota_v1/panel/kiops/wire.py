"""FAMEXP2 streaming interface; stdout is reserved for protocol bytes.

Import generic libraries, call ready(), then read_header(). The returned dict
contains metadata plus CSR arrays indptr/indices/data. For each of its requests,
call read_request(header), compute y, and write_result(y). Exit zero afterwards.
The parent sends the next RHS only after the preceding complete response.
L is a trusted symmetric PSD input; this SDK checks representation, not PSD.
"""

import json
import math
import numbers
import struct
import sys

import numpy as np


MAGIC = b"FAMEXP2\n"
READY = b"READY\n"
MAX_HEADER_BYTES = 4096
MAX_N = 120_000
MAX_NNZ = 1_000_000
METADATA_KEYS = frozenset({"version", "n", "nnz", "requests", "scenario", "t", "rtol", "atol"})


class ProtocolError(ValueError):
    """Malformed or incomplete wire data."""


def _input(stream):
    return sys.stdin.buffer if stream is None else stream


def _output(stream):
    return sys.stdout.buffer if stream is None else stream


def _read_exact(stream, length):
    parts = bytearray()
    while len(parts) < length:
        block = stream.read(length - len(parts))
        if not block:
            raise ProtocolError("unexpected end of input")
        if not isinstance(block, bytes):
            raise ProtocolError("binary input stream required")
        parts.extend(block)
    return bytes(parts)


def _write_all(stream, data):
    remaining = memoryview(data)
    while remaining:
        written = stream.write(remaining)
        if written is None or written <= 0:
            raise ProtocolError("output stream made no progress")
        remaining = remaining[written:]


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ProtocolError("duplicate metadata key")
        value[key] = item
    return value


def _validate_metadata(meta):
    if not isinstance(meta, dict) or set(meta) != METADATA_KEYS:
        raise ProtocolError("unexpected metadata fields")
    for name in ("version", "n", "nnz", "requests"):
        if type(meta[name]) is not int:
            raise ProtocolError(f"{name} must be an integer")
    if meta["version"] != 2:
        raise ProtocolError("unsupported protocol version")
    if not 1 <= meta["n"] <= MAX_N or not 0 <= meta["nnz"] <= MAX_NNZ:
        raise ProtocolError("matrix dimensions exceed protocol limits")
    if (meta["scenario"], meta["requests"]) not in (("one_shot", 1), ("reuse", 16)):
        raise ProtocolError("scenario/request count mismatch")
    for name in ("t", "rtol", "atol"):
        if isinstance(meta[name], bool) or not isinstance(meta[name], numbers.Real):
            raise ProtocolError(f"invalid {name}")
        try:
            finite = math.isfinite(float(meta[name]))
        except (ValueError, OverflowError):
            finite = False
        if not finite:
            raise ProtocolError(f"invalid {name}")
    if not 0 <= meta["t"] <= 1e4 or meta["rtol"] not in (1e-4, 1e-7, 1e-10):
        raise ProtocolError("unsupported propagation time or precision")
    if not math.isclose(meta["atol"], meta["rtol"] * 1e-3, rel_tol=1e-14, abs_tol=0):
        raise ProtocolError("atol must equal rtol times 1e-3")


def _validate_csr(meta, indptr, indices, data):
    n, nnz = meta["n"], meta["nnz"]
    if indptr.shape != (n + 1,) or indices.shape != (nnz,) or data.shape != (nnz,):
        raise ProtocolError("wrong CSR array lengths")
    if indptr.dtype.kind not in "iu" or indices.dtype.kind not in "iu":
        raise ProtocolError("CSR pointers and indices must be integers")
    if (indptr[0] != 0 or indptr[-1] != nnz or np.any(indptr < 0)
            or np.any(indptr[1:] < indptr[:-1])):
        raise ProtocolError("invalid CSR pointers")
    if np.any(indices < 0) or np.any(indices >= n):
        raise ProtocolError("CSR column index out of range")
    if nnz > 1:
        nonincreasing = np.flatnonzero(indices[1:] <= indices[:-1]) + 1
        if nonincreasing.size and np.any(~np.isin(nonincreasing, indptr[1:-1])):
            raise ProtocolError("CSR columns must be sorted and unique within each row")
    if data.dtype.kind not in "fiu" or not np.all(np.isfinite(data)):
        raise ProtocolError("CSR data must be finite real numbers")


def ready(*, stdout=None):
    """Signal that generic imports are complete, before receiving any input."""
    stream = _output(stdout)
    _write_all(stream, READY)
    stream.flush()


def read_header(*, stdin=None):
    """Read and validate metadata and the one shared CSR matrix."""
    stream = _input(stdin)
    if _read_exact(stream, len(MAGIC)) != MAGIC:
        raise ProtocolError("wrong wire magic")
    length = struct.unpack("<I", _read_exact(stream, 4))[0]
    if not 1 <= length <= MAX_HEADER_BYTES:
        raise ProtocolError("JSON header length exceeds protocol limit")
    try:
        meta = json.loads(_read_exact(stream, length).decode("utf-8"),
                          object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid UTF-8 JSON metadata") from exc
    _validate_metadata(meta)
    n, nnz = meta["n"], meta["nnz"]
    indptr = np.frombuffer(_read_exact(stream, 8 * (n + 1)), dtype="<u8")
    indices = np.frombuffer(_read_exact(stream, 4 * nnz), dtype="<u4")
    data = np.frombuffer(_read_exact(stream, 8 * nnz), dtype="<f8")
    _validate_csr(meta, indptr, indices, data)
    return {**meta, "indptr": indptr, "indices": indices, "data": data}


def read_request(header, *, stdin=None):
    """Read exactly the next b0 and b1, each bounded by one in absolute value."""
    stream = _input(stdin)
    n = header["n"]
    if type(n) is not int or not 1 <= n <= MAX_N:
        raise ProtocolError("invalid request dimension")
    vectors = tuple(np.frombuffer(_read_exact(stream, 8 * n), dtype="<f8") for _ in range(2))
    for vector in vectors:
        if not np.all(np.isfinite(vector)) or np.any(np.abs(vector) > 1):
            raise ProtocolError("RHS values must be finite and bounded by one")
    return vectors


def write_result(y, *, stdout=None):
    """Write one finite real vector as raw little-endian float64, and flush."""
    vector = np.asarray(y)
    if (vector.ndim != 1 or not 1 <= vector.size <= MAX_N
            or vector.dtype.kind not in "fiu" or not np.all(np.isfinite(vector))):
        raise ProtocolError("result must be a finite one-dimensional real vector")
    with np.errstate(over="ignore", invalid="ignore"):
        encoded = np.asarray(vector, dtype="<f8")
    if not np.all(np.isfinite(encoded)):
        raise ProtocolError("result cannot be represented as finite float64")
    stream = _output(stdout)
    _write_all(stream, encoded.tobytes(order="C"))
    stream.flush()
