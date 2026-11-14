#!/usr/bin/env python3
"""Deterministically generate synthetic conic-solver validation cases.

All files written by this script live below ``/workspace/submission/tests``.
Feasible cases are built from an explicit KKT tuple (x, s, z):

    b = A x + s, q = -P x - A^T z, s in K, z in K*, s^T z = 0.

Every cone block uses a complementary primal/dual pair.  Most coefficients are
dyadic (exact in binary64); PSD off-diagonals and nontrivial exponential
boundary points exercise the checker's sqrt(2) and log/exp paths.  Positive
per-block row scalings and per-variable column scalings preserve KKT exactly.

Expected results are verified with the official CLI:
    python /workspace/samples/io/check_result.py --input ... --result ...
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import fnmatch
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
from scipy import sparse

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parents[1]
IO_DIR = ROOT / "samples" / "io"
if not (IO_DIR / "model.py").exists():
    IO_DIR = Path("/workspace/samples/io")
sys.path.insert(0, str(IO_DIR))
from model import Problem, save_problem  # noqa: E402

CHECKER = IO_DIR / "check_result.py"
INPUT_DIR = TESTS / "inputs"
RESULT_DIR = TESTS / "expected"
MANIFEST = TESTS / "cases_manifest.json"
REPORT = TESTS / "generation_report.json"

MASTER_SEED = 20261113
SQRT2 = math.sqrt(2.0)
ALL_CONE_TYPES = ("zero", "nonnegative", "soc", "exponential", "psd")
MODES = ("primal_interior", "dual_interior", "boundary_both")

A_VALUES = np.array([-16., -8., -4., -2., -1., -0.5, -0.25,
                     0.25, 0.5, 1., 2., 4., 8., 16.], dtype=np.float64)
SMALL_A_VALUES = np.array([-2., -1., -0.5, 0.5, 1., 2.], dtype=np.float64)
LARGE_A_VALUES = np.array([-100000., -16., -8., -4., -2., -1., -0.5,
                           0.5, 1., 2., 4., 8., 16., 100000.],
                          dtype=np.float64)
B_VALUES = np.array([-2., -1., -0.5, 0.5, 1., 2.], dtype=np.float64)
X_VALUES = np.array([-2., -1., -0.5, 0., 0.5, 1., 2.], dtype=np.float64)
X_NONZERO = np.array([-2., -1., -0.5, 0.5, 1., 2.], dtype=np.float64)
POS_VALUES = np.array([0.25, 0.5, 1., 2., 4.], dtype=np.float64)
C0_VALUES = np.array([-0.5, -0.25, 0., 0.25, 0.5], dtype=np.float64)
Z_FREE_VALUES = np.array([-4., -2., -1., -0.5, 0.5, 1., 2., 4.],
                         dtype=np.float64)
EXP_TS = np.array([-1., 0., 1., 2., math.log(2.)], dtype=np.float64)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def inf_norm(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(np.max(np.abs(v), initial=0.0))


def max_abs_data(matrix):
    return float(np.max(np.abs(matrix.data), initial=0.0))


def clean_sparse(matrix):
    matrix = sparse.csc_matrix(matrix, dtype=np.float64)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def assert_symmetric(P):
    delta = P - P.T
    tolerance = 1e-12 * max(1.0, max_abs_data(P))
    if max_abs_data(delta) > tolerance:
        raise ValueError(f"P is not stored as the full symmetric matrix: {max_abs_data(delta)}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def C(kind, dim):
    return {"type": str(kind), "dim": int(dim)}


def cone_width(cone):
    dim = int(cone["dim"])
    return dim * (dim + 1) // 2 if cone["type"] == "psd" else dim


def total_width(cones):
    return int(sum(cone_width(cone) for cone in cones))


def cone_slices(cones):
    start = 0
    for cone in cones:
        width = cone_width(cone)
        yield start, start + width
        start += width


def present_cone_types(cones):
    return sorted({cone["type"] for cone in cones})


def empty_cone_families(cones):
    present = set(present_cone_types(cones))
    return [kind for kind in ALL_CONE_TYPES if kind not in present]


def case_rng(index):
    return np.random.default_rng(np.random.SeedSequence([MASTER_SEED, int(index)]))


def finite_arrays(**arrays):
    for name, array in arrays.items():
        values = np.asarray(array)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite values in {name}")


# ---------------------------------------------------------------------------
# Cone layouts and sparse matrix construction
# ---------------------------------------------------------------------------

def split_total(total, parts, rng):
    total, parts = int(total), int(parts)
    if total <= 0:
        return []
    parts = max(1, min(parts, total))
    if parts == 1:
        return [total]
    cuts = sorted(int(v) for v in rng.choice(np.arange(1, total), size=parts - 1,
                                             replace=False))
    sizes, previous = [], 0
    for cut in cuts + [total]:
        sizes.append(int(cut - previous))
        previous = cut
    return sizes


def lp_cones(n_positive, n_zero, rng, parts=3):
    positive = split_total(n_positive, parts, rng)
    zero = split_total(n_zero, max(1, parts - 1), rng)
    cones = []
    for i in range(max(len(positive), len(zero))):
        if i < len(zero):
            cones.append(C("zero", zero[i]))
        if i < len(positive):
            cones.append(C("nonnegative", positive[i]))
    return cones


def soc_cones(dims, extra_zero=0, extra_positive=0):
    cones = [C("soc", int(dim)) for dim in dims]
    if extra_positive > 0:
        cones.insert(min(1, len(cones)), C("nonnegative", int(extra_positive)))
    if extra_zero > 0:
        cones.append(C("zero", int(extra_zero)))
    return cones


def psd_cones(dims, extra_zero=0, extra_positive=0):
    cones = [C("psd", int(dim)) for dim in dims]
    if extra_positive > 0:
        cones.insert(min(1, len(cones)), C("nonnegative", int(extra_positive)))
    if extra_zero > 0:
        cones.append(C("zero", int(extra_zero)))
    return cones


def exp_cones(count, extra_zero=0, extra_positive=0):
    cones = [C("exponential", 3) for _ in range(int(count))]
    if extra_positive > 0:
        cones.insert(min(1, len(cones)), C("nonnegative", int(extra_positive)))
    if extra_zero > 0:
        cones.append(C("zero", int(extra_zero)))
    return cones


def mixed_cones(rng, target_m=None):
    base = [
        C("zero", 3),
        C("nonnegative", 5),
        C("soc", 6),
        C("psd", 4),
        C("exponential", 3),
        C("soc", 2),
        C("nonnegative", 4),
        C("psd", 2),
        C("zero", 2),
        C("exponential", 3),
    ]
    if target_m is None:
        return base
    remaining = int(target_m) - total_width(base)
    if remaining < 0:
        raise ValueError("target_m is smaller than the all-five-cone base layout")
    extra = [C("nonnegative", dim) for dim in split_total(remaining, 4, rng)]
    # Interleave the large nonnegative pieces so cone order is not grouped.
    cones = []
    for i in range(max(len(base), len(extra))):
        if i < len(base):
            cones.append(base[i])
        if i < len(extra):
            cones.append(extra[i])
    return cones


def large_mixed_cones(rng, target_m):
    base = [
        C("zero", 50),
        C("soc", 40),
        C("psd", 10),
        C("exponential", 3),
        C("nonnegative", 100),
        C("soc", 20),
        C("psd", 8),
        C("exponential", 3),
        C("soc", 10),
        C("psd", 5),
        C("exponential", 3),
        C("nonnegative", 100),
        C("zero", 50),
    ]
    for dim in [2, 3, 4, 5, 6, 7, 8]:
        base.append(C("soc", dim))
    for dim in [2, 3, 4]:
        base.append(C("psd", dim))
    for _ in range(2):
        base.append(C("exponential", 3))
    remaining = int(target_m) - total_width(base)
    if remaining <= 0:
        raise ValueError("large mixed cone target too small")
    base.extend(C("nonnegative", dim) for dim in split_total(remaining, 5, rng))
    order = rng.permutation(len(base))
    return [base[int(i)] for i in order]


def random_sparse_A(rng, m, n, nnz_per_col=3, values=A_VALUES,
                    forbidden_rows=None):
    m, n = int(m), int(n)
    if m == 0 or n == 0:
        return sparse.csc_matrix((m, n), dtype=np.float64)
    forbidden = np.zeros(m, dtype=bool)
    if forbidden_rows is not None:
        rows = np.asarray(list(forbidden_rows), dtype=np.int64)
        if rows.size:
            forbidden[rows] = True
    allowed = np.flatnonzero(~forbidden)
    values = np.asarray(values, dtype=np.float64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices, data = [], []
    if allowed.size:
        nnz_per_col = max(1, int(nnz_per_col))
        for column in range(n):
            count = min(int(rng.integers(1, nnz_per_col + 1)), allowed.size)
            rows = rng.choice(allowed, size=count, replace=False)
            entries = rng.choice(values, size=count)
            indices.extend(int(row) for row in rows)
            data.extend(float(value) for value in entries)
            indptr[column + 1] = len(data)
    A = sparse.csc_matrix((np.asarray(data, dtype=np.float64),
                           np.asarray(indices, dtype=np.int64), indptr),
                          shape=(m, n))
    return clean_sparse(A)


def random_row_vector(rng, n, count, values=A_VALUES):
    n = int(n)
    columns = rng.choice(n, size=min(int(count), n), replace=False)
    coefficients = rng.choice(np.asarray(values, dtype=np.float64),
                              size=len(columns))
    return {int(col): float(val) for col, val in zip(columns, coefficients)
            if val != 0.0}


def add_sparse_rows(A, m, n, row_vectors):
    rows, columns, data = [], [], []
    for row, entries in row_vectors.items():
        for column, value in entries.items():
            if value != 0.0:
                rows.append(int(row))
                columns.append(int(column))
                data.append(float(value))
    additions = sparse.coo_matrix((np.asarray(data, dtype=np.float64),
                                   (np.asarray(rows, dtype=np.int64),
                                    np.asarray(columns, dtype=np.int64))),
                                  shape=(m, n)).tocsc()
    return clean_sparse(A + additions)


def random_sparse_A_for_direction(rng, m, n, x, s, values=A_VALUES,
                                  nnz_per_col=3):
    """Random CSC A with exactly A @ x == -s.

    One pivot column carries -s/x_pivot.  Random entries are placed only in
    columns where x is exactly zero, so the equality survives A/a scaling.
    """
    m, n = int(m), int(n)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    pivots = np.flatnonzero(np.abs(x) == 1.0)
    if pivots.size == 0:
        raise ValueError("direction has no unit pivot")
    pivot = int(pivots[0])
    zero_columns = set(int(j) for j in np.flatnonzero(x == 0.0))
    values = np.asarray(values, dtype=np.float64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices, data = [], []

    if m:
        for row, value in enumerate(s):
            if value != 0.0:
                indices.append(row)
                data.append(float(-value / x[pivot]))
        indptr[pivot + 1] = len(data)

    for column in range(n):
        if column == pivot:
            continue
        if column in zero_columns and m:
            count = min(int(rng.integers(1, max(1, int(nnz_per_col)) + 1)), m)
            rows = rng.choice(m, size=count, replace=False)
            entries = rng.choice(values, size=count)
            indices.extend(int(row) for row in rows)
            data.extend(float(value) for value in entries)
        indptr[column + 1] = len(data)

    A = sparse.csc_matrix((np.asarray(data, dtype=np.float64),
                           np.asarray(indices, dtype=np.int64), indptr),
                          shape=(m, n))
    return clean_sparse(A)


# ---------------------------------------------------------------------------
# PSD matrices and cone vectors
# ---------------------------------------------------------------------------

def make_psd_P(rng, n, mode="zero", x_null=None, p_scale=1.0):
    """Return a sparse PSD P stored as a full symmetric matrix."""
    n = int(n)
    p_scale = float(p_scale)
    if mode == "zero":
        P = sparse.csc_matrix((n, n), dtype=np.float64)
    elif mode in ("diagonal", "diagonal_null"):
        diagonal = rng.choice(POS_VALUES, size=n) * p_scale
        if mode == "diagonal_null":
            if x_null is None:
                raise ValueError("diagonal_null requires x_null")
            diagonal[np.asarray(x_null) != 0.0] = 0.0
        P = sparse.diags(diagonal, format="csc")
    elif mode in ("btb", "btb_null"):
        row_count = max(1, min(n, n // 2 + 1))
        rows, columns, data = [], [], []
        if mode == "btb_null":
            if x_null is None:
                raise ValueError("btb_null requires x_null")
            x = np.asarray(x_null, dtype=np.float64).reshape(-1)
            for row in range(row_count):
                if n >= 2:
                    pair = rng.choice(n, size=2, replace=False)
                    j, k = int(pair[0]), int(pair[1])
                    # B[row, :] @ x = x[k]*x[j] - x[j]*x[k] = 0 exactly.
                    rows.extend([row, row])
                    columns.extend([j, k])
                    data.extend([float(x[k]), float(-x[j])])
        else:
            for row in range(row_count):
                count = min(int(rng.integers(1, min(3, n) + 1)), n)
                cols = rng.choice(n, size=count, replace=False)
                vals = rng.choice(B_VALUES, size=count)
                rows.extend([row] * count)
                columns.extend(int(col) for col in cols)
                data.extend(float(val) for val in vals)
        B = sparse.coo_matrix((np.asarray(data, dtype=np.float64),
                               (np.asarray(rows, dtype=np.int64),
                                np.asarray(columns, dtype=np.int64))),
                              shape=(row_count, n)).tocsr()
        B.sum_duplicates()
        P = (B.T @ B).tocsc()
        P = 0.5 * (P + P.T) * p_scale
    else:
        raise ValueError(f"unknown P mode {mode}")

    P = clean_sparse(P)
    if P.nnz:
        P = clean_sparse(0.5 * (P + P.T))
    assert_symmetric(P)
    if x_null is not None and inf_norm(P @ np.asarray(x_null)) > 1e-10:
        # Exact diagonal nullspace is a safe fallback for certificate cases.
        if mode != "diagonal_null":
            return make_psd_P(rng, n, "diagonal_null", x_null=x_null,
                              p_scale=p_scale)
        raise ValueError("P does not annihilate the requested null vector")
    return P


def psd_vec(matrix, dim):
    dim = int(dim)
    matrix = 0.5 * (np.asarray(matrix, dtype=np.float64) +
                    np.asarray(matrix, dtype=np.float64).T)
    width = dim * (dim + 1) // 2
    vector = np.zeros(width, dtype=np.float64)
    position = 0
    for column in range(dim):
        for row in range(column):
            vector[position] = SQRT2 * matrix[row, column]
            position += 1
        vector[position] = matrix[column, column]
        position += 1
    if position != width:
        raise AssertionError("PSD encoding width mismatch")
    return vector


def orthogonal_U(rng, dim, groups=None, rotations=True):
    dim = int(dim)
    permutation = rng.permutation(dim)
    signs = rng.choice(np.array([-1., 1.], dtype=np.float64), size=dim)
    U = np.zeros((dim, dim), dtype=np.float64)
    for column in range(dim):
        U[int(permutation[column]), column] = signs[column]
    if not rotations:
        return U
    if groups is None:
        groups = [list(range(dim))]
    for group in groups:
        for offset in range(0, len(group) - 1, 2):
            i, j = int(group[offset]), int(group[offset + 1])
            if rng.integers(0, 2) == 0:
                cosine, sine = 0.6, 0.8
            else:
                cosine, sine = 0.8, 0.6
            if rng.integers(0, 2) == 0:
                sine = -sine
            R = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
            U[:, [i, j]] = U[:, [i, j]] @ R
    return U


def random_x(rng, n, nonzero=False, forced_zeros=()):
    values = X_NONZERO if nonzero else X_VALUES
    x = rng.choice(values, size=int(n)).astype(np.float64)
    for index in forced_zeros:
        x[int(index) % int(n)] = 0.0
    if inf_norm(x) == 0.0:
        x[0] = 1.0
    return x


def direction(rng, n, kind="e0"):
    n = int(n)
    x = np.zeros(n, dtype=np.float64)
    if kind == "e0":
        x[0] = 1.0
    elif kind == "negative_e0":
        x[0] = -1.0
    elif kind == "mixed":
        x = rng.choice(np.array([-1., 0., 1.], dtype=np.float64), size=n)
        x[0] = 1.0
    else:
        raise ValueError(f"unknown direction {kind}")
    x /= inf_norm(x)
    return x


def nonnegative_pair(rng, dim, mode):
    dim = int(dim)
    s = np.zeros(dim, dtype=np.float64)
    z = np.zeros(dim, dtype=np.float64)
    values = rng.choice(POS_VALUES, size=dim)
    if mode == "primal_interior":
        s = values
    elif mode == "dual_interior":
        z = values
    else:
        flip = int(rng.integers(0, 2))
        for i in range(dim):
            if (i + flip) % 2 == 0:
                s[i] = values[i]
            else:
                z[i] = values[i]
    return s, z


def zero_pair(rng, dim, mode):
    dim = int(dim)
    s = np.zeros(dim, dtype=np.float64)
    if mode == "primal_interior":
        z = np.zeros(dim, dtype=np.float64)
    else:
        z = rng.choice(Z_FREE_VALUES, size=dim).astype(np.float64)
    return s, z


def soc_pair(rng, dim, mode):
    dim = int(dim)
    if dim < 2:
        raise ValueError("SOC dimension must be at least 2")
    scale = float(rng.choice(POS_VALUES))
    if dim == 2:
        t, v = 1.0, np.array([1.0])
        t_interior = 2.0
    else:
        t = 5.0
        v = np.zeros(dim - 1, dtype=np.float64)
        v[0], v[1] = 3.0, 4.0
        t_interior = 8.0
    s = np.zeros(dim, dtype=np.float64)
    z = np.zeros(dim, dtype=np.float64)
    if mode == "primal_interior":
        s[0] = scale * t_interior
        s[1:] = scale * v
    elif mode == "dual_interior":
        z[0] = scale * t_interior
        z[1:] = scale * v
    else:
        lam = float(rng.choice(POS_VALUES))
        s[0] = scale * t
        s[1:] = scale * v
        z[0] = scale * lam * t
        z[1:] = -scale * lam * v
    return s, z


def psd_pair(rng, dim, mode, rotations=True):
    dim = int(dim)
    width = dim * (dim + 1) // 2
    zero = np.zeros(width, dtype=np.float64)
    if mode == "primal_interior":
        U = orthogonal_U(rng, dim, rotations=rotations)
        eigenvalues = rng.choice(POS_VALUES, size=dim)
        S = (U * eigenvalues) @ U.T
        return psd_vec(S, dim), zero
    if mode == "dual_interior":
        U = orthogonal_U(rng, dim, rotations=rotations)
        eigenvalues = rng.choice(POS_VALUES, size=dim)
        Z = (U * eigenvalues) @ U.T
        return zero, psd_vec(Z, dim)
    rank = max(1, min(dim - 1, dim // 2 + int(rng.integers(0, 2))))
    U = orthogonal_U(rng, dim,
                     groups=[list(range(rank)), list(range(rank, dim))],
                     rotations=rotations)
    U1, U2 = U[:, :rank], U[:, rank:]
    eig1 = rng.choice(POS_VALUES, size=rank)
    eig2 = rng.choice(POS_VALUES, size=dim - rank)
    S = (U1 * eig1) @ U1.T
    Z = (U2 * eig2) @ U2.T
    return psd_vec(S, dim), psd_vec(Z, dim)


def exponential_pair(rng, mode, exact_dyadic=False, exp_t=None):
    scale = float(rng.choice(POS_VALUES))
    lam = float(rng.choice(POS_VALUES))
    zero = np.zeros(3, dtype=np.float64)
    if mode == "primal_interior":
        # (0, v, 4v) is strictly interior: 4v > v*exp(0).
        return np.array([0., scale, 4. * scale], dtype=np.float64), zero
    if mode == "dual_interior":
        # z_u < 0 and z_w=1 > -z_u*exp(z_v/z_u-1)=exp(-1).
        return zero, lam * np.array([-1., 0., 1.], dtype=np.float64)
    if exact_dyadic or exp_t is None:
        t = 0.0 if exact_dyadic else float(rng.choice(EXP_TS))
    else:
        t = float(exp_t)
    v = scale
    w = v * math.exp(t)
    s = np.array([v * t, v, w], dtype=np.float64)
    z = lam * np.array([-1., math.log(w / v) - 1., v / w],
                       dtype=np.float64)
    return s, z


def choose_mode(spec, index, rng):
    if spec.get("force_z_zero"):
        return "primal_interior"
    pattern = spec.get("mode_pattern", "cycle")
    pattern = {"all_primal": "primal_interior",
               "all_dual": "dual_interior",
               "all_boundary": "boundary_both"}.get(pattern, pattern)
    if pattern == "cycle":
        return MODES[(int(index) + int(spec.get("mode_offset", 0))) % len(MODES)]
    if pattern == "random":
        return MODES[int(rng.integers(0, len(MODES)))]
    if pattern in MODES:
        return pattern
    raise ValueError(f"unknown mode pattern {pattern}")


def make_cone_pairs(rng, cones, spec):
    m = total_width(cones)
    s = np.zeros(m, dtype=np.float64)
    z = np.zeros(m, dtype=np.float64)
    modes = []
    exact_dyadic = bool(spec.get("exact_dyadic", False))
    psd_rotations = bool(spec.get("psd_rotations", True)) and not exact_dyadic
    for index, (cone, (start, stop)) in enumerate(zip(cones, cone_slices(cones))):
        kind, dim = cone["type"], int(cone["dim"])
        mode = choose_mode(spec, index, rng)
        modes.append(mode)
        if kind == "zero":
            sb, zb = zero_pair(rng, dim, mode)
        elif kind == "nonnegative":
            sb, zb = nonnegative_pair(rng, dim, mode)
        elif kind == "soc":
            sb, zb = soc_pair(rng, dim, mode)
        elif kind == "psd":
            sb, zb = psd_pair(rng, dim, mode, rotations=psd_rotations)
        elif kind == "exponential":
            sb, zb = exponential_pair(rng, mode, exact_dyadic=exact_dyadic,
                                      exp_t=spec.get("exp_t"))
        else:
            raise ValueError(f"unknown cone {kind}")
        if sb.size != stop - start or zb.size != stop - start:
            raise AssertionError("cone block width mismatch")
        s[start:stop], z[start:stop] = sb, zb
    return s, z, modes


def make_interior_slack(rng, cones):
    """Strictly feasible cone vector b=s0 for unbounded-case constructions."""
    m = total_width(cones)
    s = np.zeros(m, dtype=np.float64)
    for cone, (start, stop) in zip(cones, cone_slices(cones)):
        kind, dim = cone["type"], int(cone["dim"])
        if kind == "zero":
            block = np.zeros(stop - start, dtype=np.float64)
        elif kind == "nonnegative":
            block = rng.choice(POS_VALUES, size=dim)
        elif kind == "soc":
            block = np.zeros(dim, dtype=np.float64)
            block[0] = 10.0
        elif kind == "exponential":
            v = float(rng.choice(POS_VALUES))
            block = np.array([0., v, 4. * v], dtype=np.float64)
        elif kind == "psd":
            scale = float(rng.choice(POS_VALUES))
            block = psd_vec(scale * np.eye(dim), dim)
        else:
            raise ValueError(f"unknown cone {kind}")
        s[start:stop] = block
    return s


# ---------------------------------------------------------------------------
# Scaling, diagnostics, and result writing
# ---------------------------------------------------------------------------

def sample_exponents(rng, size, bounds, endpoint="max"):
    size = int(size)
    lo, hi = int(bounds[0]), int(bounds[1])
    if size <= 0:
        return np.zeros(0, dtype=np.int64)
    exps = rng.integers(lo, hi + 1, size=size, dtype=np.int64)
    if size >= 2:
        exps[0] = lo
        exps[-1] = hi
    else:
        exps[0] = hi if endpoint == "max" else lo
    rng.shuffle(exps)
    return exps


def apply_scaling(rng, spec, cones, P, q, A, b, x, s, z):
    m, n = len(b), len(q)
    row_exps = np.zeros(len(cones), dtype=np.int64)
    col_exps = np.zeros(n, dtype=np.int64)
    row_scales = np.ones(m, dtype=np.float64)
    col_scales = np.ones(n, dtype=np.float64)
    row_info = {"min": 1.0, "max": 1.0, "exp_min": 0, "exp_max": 0}
    col_info = {"min": 1.0, "max": 1.0, "exp_min": 0, "exp_max": 0}

    if spec.get("row_scale") is not None:
        row_exps = sample_exponents(rng, len(cones), spec["row_scale"],
                                    spec.get("single_row_endpoint", "max"))
        for exponent, (start, stop) in zip(row_exps, cone_slices(cones)):
            row_scales[start:stop] = 2.0 ** int(exponent)
        if m:
            R = sparse.diags(row_scales, format="csc")
            A = clean_sparse(R @ A)
            b = row_scales * b
            s = row_scales * s
            z = z / row_scales
        if row_scales.size:
            row_info = {"min": float(np.min(row_scales)),
                        "max": float(np.max(row_scales)),
                        "exp_min": int(np.min(row_exps)),
                        "exp_max": int(np.max(row_exps))}

    if spec.get("col_scale") is not None:
        col_exps = sample_exponents(rng, n, spec["col_scale"],
                                    spec.get("single_col_endpoint", "max"))
        col_scales = 2.0 ** col_exps
        D = sparse.diags(col_scales, format="csc")
        P = clean_sparse(D @ P @ D)
        if P.nnz:
            P = clean_sparse(0.5 * (P + P.T))
        A = clean_sparse(A @ D)
        q = col_scales * q
        x = x / col_scales
        col_info = {"min": float(np.min(col_scales)),
                    "max": float(np.max(col_scales)),
                    "exp_min": int(np.min(col_exps)),
                    "exp_max": int(np.max(col_exps))}

    assert_symmetric(P)
    return P, q, A, b, x, s, z, {"row": row_info, "col": col_info}


def kkt_diagnostics(P, q, A, b, x, s, z):
    Ax, Px, Atz = A @ x, P @ x, A.T @ z
    primal = inf_norm(Ax + s - b) / (1.0 + max(inf_norm(Ax), inf_norm(s),
                                              inf_norm(b)))
    dual = inf_norm(Px + q + Atz) / (1.0 + max(inf_norm(Px), inf_norm(q),
                                              inf_norm(Atz)))
    xPx = float(x @ Px)
    p0 = 0.5 * xPx + float(q @ x)
    d0 = -0.5 * xPx - float(b @ z)
    gap = abs(p0 - d0)
    return {"primal": float(primal), "dual": float(dual),
            "complementarity_absolute": abs(float(s @ z)),
            "gap_absolute": float(gap),
            "gap_relative": float(gap / max(abs(p0), abs(d0)) if max(abs(p0), abs(d0)) else 0.0),
            "objective_primal_without_offset": float(p0),
            "objective_dual_without_offset": float(d0)}


def save_result(path, status, *, x=None, s=None, z=None,
                objective_primal=None, objective_dual=None):
    metadata = {"schema_version": 1, "status": str(status)}
    arrays = {}
    if status == "solved":
        for name, value, required in (("x", x, True), ("s", s, True), ("z", z, True)):
            if value is None:
                if required:
                    raise ValueError(f"missing {name}")
            else:
                arrays[name] = np.asarray(value, dtype=np.float64).reshape(-1)
        if objective_primal is not None:
            metadata["objective_primal"] = float(objective_primal)
        if objective_dual is not None:
            metadata["objective_dual"] = float(objective_dual)
    elif status == "primal_infeasible":
        arrays["z"] = np.asarray(z, dtype=np.float64).reshape(-1)
    elif status == "dual_infeasible":
        arrays["x"] = np.asarray(x, dtype=np.float64).reshape(-1)
        arrays["s"] = np.asarray(s, dtype=np.float64).reshape(-1)
    else:
        raise ValueError(f"unsupported expected status {status}")
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True,
                                                  ensure_ascii=False,
                                                  allow_nan=False))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return path


def run_checker_cli(input_path, result_path):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, str(CHECKER), "--input", str(input_path),
               "--result", str(result_path)]
    try:
        process = subprocess.run(command, capture_output=True, text=True,
                                 env=env, timeout=180)
    except subprocess.TimeoutExpired as error:
        return {"outcome": "checker_timeout", "error": str(error),
                "check": {"status": None, "residuals": {}}}
    try:
        report = json.loads(process.stdout)
    except Exception as error:
        return {"outcome": "checker_error",
                "error": f"{type(error).__name__}: {error}",
                "stdout": process.stdout[-2000:],
                "stderr": process.stderr[-2000:],
                "returncode": process.returncode,
                "check": {"status": None, "residuals": {}}}
    if process.returncode != 0 and report.get("outcome") == "verified":
        report["outcome"] = "checker_exit_error"
        report["returncode"] = process.returncode
    return report


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------

def base_case_record(spec, index, cones, P, A, b, q, c0, scaling):
    n, m = len(q), len(b)
    return {
        "id": spec["id"],
        "family": spec["family"],
        "tags": list(spec.get("tags", [])),
        "expected_status": spec["expected_status"],
        "n": int(n),
        "m": int(m),
        "nnz_P": int(P.nnz),
        "nnz_A": int(A.nnz),
        "cone_types": present_cone_types(cones),
        "empty_families": empty_cone_families(cones),
        "cone_count": len(cones),
        "max_abs_A": max_abs_data(A),
        "max_abs_P": max_abs_data(P),
        "max_abs_q": inf_norm(q),
        "max_abs_b": inf_norm(b),
        "objective_offset": float(c0),
        "seed": [MASTER_SEED, int(index)],
        "scaling": json_safe(scaling),
        "notes": spec.get("notes", ""),
    }


def make_feasible_case(spec, index):
    rng = case_rng(index)
    cones = spec["cones"]
    n, m = int(spec["n"]), total_width(cones)
    if n < 2:
        raise ValueError(f"{spec['id']}: n must be at least 2")
    if m and m < 2:
        raise ValueError(f"{spec['id']}: m must be zero or at least 2")

    if spec.get("q_zero") and spec.get("p_mode") == "diagonal_null":
        x = random_x(rng, n, nonzero=False, forced_zeros=(0,))
    elif spec.get("q_zero") and spec.get("p_mode") == "btb_null":
        x = random_x(rng, n, nonzero=True)
    else:
        x = random_x(rng, n)
    p_mode = spec.get("p_mode", "zero")
    null_vector = x if p_mode in ("btb_null", "diagonal_null") else None
    P = make_psd_P(rng, n, p_mode, x_null=null_vector,
                   p_scale=float(spec.get("p_base_scale", 1.0)))
    s, z, modes = make_cone_pairs(rng, cones, spec)
    A = random_sparse_A(rng, m, n,
                        nnz_per_col=int(spec.get("nnz_per_col", 3)),
                        values=np.asarray(spec.get("a_values", A_VALUES),
                                          dtype=np.float64))
    b = A @ x + s
    q = -(P @ x) - (A.T @ z)
    q_zero_residual = 0.0
    if spec.get("q_zero"):
        q_zero_residual = inf_norm(q)
        if q_zero_residual > 1e-8:
            raise ValueError(f"{spec['id']}: q is not zero ({q_zero_residual})")
        # P x and A^T z vanish by construction; store exact zeros.
        q = np.zeros(n, dtype=np.float64)
    c0 = float(rng.choice(C0_VALUES))

    P, q, A, b, x, s, z, scaling = apply_scaling(rng, spec, cones,
                                                  P, q, A, b, x, s, z)
    if spec.get("q_zero"):
        q = np.zeros(n, dtype=np.float64)
    finite_arrays(q=q, b=b, x=x, s=s, z=z, P=P.data, A=A.data)
    assert_symmetric(P)

    diagnostics = kkt_diagnostics(P, q, A, b, x, s, z)
    if diagnostics["primal"] > 1e-9 or diagnostics["dual"] > 1e-9:
        raise ValueError(f"{spec['id']}: KKT residuals too large: {diagnostics}")
    modes_counter = Counter(modes)
    metadata = {
        "family": spec["family"],
        "tags": list(spec.get("tags", [])),
        "case_seed": [MASTER_SEED, int(index)],
        "construction": "exact KKT tuple; b=Ax+s; q=-Px-A^Tz",
        "complementarity_modes": dict(modes_counter),
        "scaling": json_safe(scaling),
    }
    problem = Problem(P, q, A, b, cones, name=spec["id"],
                      objective_offset=c0, metadata=json_safe(metadata))
    input_path = INPUT_DIR / f"{spec['id']}.input.npz"
    result_path = RESULT_DIR / f"{spec['id']}.expected.npz"
    save_problem(problem, input_path)
    save_result(result_path, "solved", x=x, s=s, z=z,
                objective_primal=diagnostics["objective_primal_without_offset"] + c0,
                objective_dual=diagnostics["objective_dual_without_offset"] + c0)

    record = base_case_record(spec, index, cones, P, A, b, q, c0, scaling)
    record.update({
        "input": str(input_path.relative_to(TESTS)),
        "result": str(result_path.relative_to(TESTS)),
        "kkt": json_safe(diagnostics),
        "cone_mode_counts": {key: int(value) for key, value in modes_counter.items()},
        "q_zero_residual_before_override": float(q_zero_residual),
        "p_mode": p_mode,
    })
    return record


def make_primal_certificate_case(spec, index):
    rng = case_rng(index)
    cones = spec["cones"]
    n, m = int(spec["n"]), total_width(cones)
    witness = spec["witness"]
    first = cones[0]
    values = np.asarray(spec.get("a_values", SMALL_A_VALUES), dtype=np.float64)
    nnz_per_col = int(spec.get("nnz_per_col", 3))
    P = make_psd_P(rng, n, spec.get("p_mode", "zero"),
                   p_scale=float(spec.get("p_base_scale", 1.0)))
    q = rng.choice(X_VALUES, size=n).astype(np.float64)
    b = np.zeros(m, dtype=np.float64)
    z = np.zeros(m, dtype=np.float64)
    forbidden_rows = []
    row_vectors = {}

    if witness == "zero_cancel":
        if first["type"] != "zero" or first["dim"] < 2:
            raise ValueError("zero_cancel needs a zero cone of dim >= 2 first")
        a_vec = random_row_vector(rng, n, spec.get("witness_nnz", 5), values)
        row_vectors[0] = dict(a_vec)
        row_vectors[1] = dict(a_vec)
        z[0], z[1] = -1.0, 1.0
        b[0], b[1] = 1.0, -1.0
        forbidden_rows = [0, 1]
    elif witness == "nonneg_cancel":
        if first["type"] != "nonnegative" or first["dim"] < 2:
            raise ValueError("nonneg_cancel needs a nonnegative cone of dim >= 2 first")
        a_vec = random_row_vector(rng, n, spec.get("witness_nnz", 5), values)
        row_vectors[0] = {col: -val for col, val in a_vec.items()}
        row_vectors[1] = dict(a_vec)
        z[0], z[1] = 1.0, 1.0
        b[0], b[1] = -1.0, -1.0
        forbidden_rows = [0, 1]
    elif witness == "soc_first":
        if first["type"] != "soc":
            raise ValueError("soc_first needs an SOC cone first")
        z[0] = 1.0
        b[0] = -1.0
        forbidden_rows = [0]
    elif witness == "psd_diag":
        if first["type"] != "psd":
            raise ValueError("psd_diag needs a PSD cone first")
        dim = int(first["dim"])
        diagonal_rows = [col * (col + 1) // 2 + col for col in range(dim)]
        z[diagonal_rows] = 1.0
        b[diagonal_rows] = -1.0
        forbidden_rows = diagonal_rows
    elif witness == "exp_cancel":
        if first["type"] != "exponential":
            raise ValueError("exp_cancel needs an exponential cone first")
        a_vec = random_row_vector(rng, n, spec.get("witness_nnz", 5), values)
        row_vectors[0] = dict(a_vec)
        row_vectors[2] = dict(a_vec)
        z[0], z[2] = -1.0, 1.0
        b[0], b[2] = 1.0, -1.0
        forbidden_rows = [0, 1, 2]
    else:
        raise ValueError(f"unknown primal witness {witness}")

    A = random_sparse_A(rng, m, n, nnz_per_col=nnz_per_col, values=values,
                        forbidden_rows=forbidden_rows)
    A = add_sparse_rows(A, m, n, row_vectors)
    c0 = float(rng.choice(C0_VALUES))
    finite_arrays(q=q, b=b, z=z, P=P.data, A=A.data)
    assert_symmetric(P)

    a_scale = max_abs_data(A) or 1.0
    z_scale = inf_norm(z) or 1.0
    b_scale = inf_norm(b) or 1.0
    zhat = z / z_scale
    margin = -float((b / b_scale) @ zhat)
    linear = inf_norm((A.T @ zhat) / a_scale)
    if margin <= 1e-3 or linear > 1e-10:
        raise ValueError(f"{spec['id']}: weak primal certificate margin={margin}, linear={linear}")

    metadata = {"family": spec["family"], "tags": list(spec.get("tags", [])),
                "case_seed": [MASTER_SEED, int(index)],
                "construction": f"exact Farkas certificate ({witness})",
                "witness": witness}
    problem = Problem(P, q, A, b, cones, name=spec["id"],
                      objective_offset=c0, metadata=json_safe(metadata))
    input_path = INPUT_DIR / f"{spec['id']}.input.npz"
    result_path = RESULT_DIR / f"{spec['id']}.expected.npz"
    save_problem(problem, input_path)
    save_result(result_path, "primal_infeasible", z=z)

    record = base_case_record(spec, index, cones, P, A, b, q, c0,
                              {"row": {"min": 1., "max": 1., "exp_min": 0, "exp_max": 0},
                               "col": {"min": 1., "max": 1., "exp_min": 0, "exp_max": 0}})
    record.update({
        "input": str(input_path.relative_to(TESTS)),
        "result": str(result_path.relative_to(TESTS)),
        "witness": witness,
        "certificate": {"margin_normalized": margin,
                        "linear_before_checker_scaling": linear,
                        "A_scale": a_scale},
    })
    return record


def make_dual_certificate_case(spec, index):
    rng = case_rng(index)
    cones = spec["cones"]
    n, m = int(spec["n"]), total_width(cones)
    x = direction(rng, n, spec.get("direction", "e0"))
    s = make_interior_slack(rng, cones)
    b = s.copy()  # x0=0 is feasible; s + tau*s_d stays in K for tau >= 0.
    values = np.asarray(spec.get("a_values", SMALL_A_VALUES), dtype=np.float64)
    A = random_sparse_A_for_direction(rng, m, n, x, s,
                                      values=values,
                                      nnz_per_col=int(spec.get("nnz_per_col", 3)))
    p_mode = spec.get("p_mode", "diagonal_null")
    P = make_psd_P(rng, n, p_mode, x_null=x,
                   p_scale=float(spec.get("p_base_scale", 1.0)))
    q_scale = float(spec.get("q_scale", 1.0))
    q = -q_scale * x
    c0 = float(rng.choice(C0_VALUES))
    finite_arrays(q=q, b=b, x=x, s=s, P=P.data, A=A.data)
    assert_symmetric(P)

    a_scale = max_abs_data(A) or 1.0
    h_scale = max_abs_data(P) or 1.0
    x_scale, q_norm = inf_norm(x), inf_norm(q)
    xhat = x / x_scale
    shat = s / x_scale
    alpha = -float((q / q_norm) @ xhat)
    quadratic = inf_norm((P @ xhat) / h_scale)
    linear = inf_norm((A @ xhat) / a_scale + shat / a_scale)
    if alpha <= 1e-3 or quadratic > 1e-10 or linear > 1e-10:
        raise ValueError(f"{spec['id']}: weak dual certificate alpha={alpha}, "
                         f"quadratic={quadratic}, linear={linear}")

    metadata = {"family": spec["family"], "tags": list(spec.get("tags", [])),
                "case_seed": [MASTER_SEED, int(index)],
                "construction": "primal feasible at x0=0; recession direction "
                                "x with q^T x<0, Px=0, Ax+s=0, s in K",
                "unbounded": True}
    problem = Problem(P, q, A, b, cones, name=spec["id"],
                      objective_offset=c0, metadata=json_safe(metadata))
    input_path = INPUT_DIR / f"{spec['id']}.input.npz"
    result_path = RESULT_DIR / f"{spec['id']}.expected.npz"
    save_problem(problem, input_path)
    save_result(result_path, "dual_infeasible", x=x, s=s)

    record = base_case_record(spec, index, cones, P, A, b, q, c0,
                              {"row": {"min": 1., "max": 1., "exp_min": 0, "exp_max": 0},
                               "col": {"min": 1., "max": 1., "exp_min": 0, "exp_max": 0}})
    record.update({
        "input": str(input_path.relative_to(TESTS)),
        "result": str(result_path.relative_to(TESTS)),
        "unbounded": True,
        "direction_nonzeros": int(np.count_nonzero(x)),
        "q_scale": q_scale,
        "certificate": {"margin_normalized": alpha,
                        "quadratic_before_checker_scaling": quadratic,
                        "linear_before_checker_scaling": linear,
                        "A_scale": a_scale, "P_scale": h_scale},
    })
    return record


# ---------------------------------------------------------------------------
# Suite specification
# ---------------------------------------------------------------------------

def build_specs():
    specs = []
    counters = Counter()
    layout_rng = np.random.default_rng(MASTER_SEED + 7919)

    def add(prefix, family, expected_status, n, cones, tags=(), **kwargs):
        counters[prefix] += 1
        spec = {"id": f"{prefix}_{counters[prefix]:02d}",
                "family": family, "expected_status": expected_status,
                "n": int(n), "cones": cones, "tags": list(tags)}
        spec.update(kwargs)
        specs.append(spec)
        return spec

    # Pure LP, P=0, zero/nonnegative cones, n,m from 2 to 3000.
    lp_sizes = [(2, 2), (3, 6), (8, 12), (32, 48), (120, 180),
                (400, 500), (1200, 1500), (3000, 3000)]
    for i, (n, m) in enumerate(lp_sizes):
        pos, zero = m // 2, m - m // 2
        add("lp", "lp", "solved", n,
            lp_cones(pos, zero, layout_rng, parts=3 if m < 200 else 5),
            tags=["feasible", "pure_lp", "P0", "zero_and_nonnegative"],
            p_mode="zero", nnz_per_col=3, mode_pattern="cycle",
            mode_offset=i % 3, exact_dyadic=True,
            notes="P=0 LP with complementary zero/nonnegative blocks")
    add("lp_only_nonnegative", "lp", "solved", 40,
        [C("nonnegative", 20), C("nonnegative", 20)],
        tags=["feasible", "empty_families", "no_zero_cone"],
        p_mode="zero", mode_pattern="all_boundary", exact_dyadic=True)
    add("lp_only_zero", "lp", "solved", 40,
        [C("zero", 20), C("zero", 20)],
        tags=["feasible", "empty_families", "zero_cone_only"],
        p_mode="zero", mode_pattern="all_dual", exact_dyadic=True)
    add("lp_boundary", "lp", "solved", 200,
        lp_cones(120, 130, layout_rng, parts=4),
        tags=["feasible", "boundary_heavy"], p_mode="zero",
        mode_pattern="all_boundary", exact_dyadic=True)

    # Quadratic programs with diagonal and B^T B PSD matrices.
    qp_sizes = [(2, 4), (6, 10), (25, 40), (100, 150),
                (300, 400), (800, 1000), (1500, 1800)]
    for i, (n, m) in enumerate(qp_sizes):
        pos, zero = m // 2, m - m // 2
        add("qp", "qp", "solved", n, lp_cones(pos, zero, layout_rng, parts=3),
            tags=["feasible", "QP", "P_nonzero"],
            p_mode="diagonal" if i % 2 == 0 else "btb",
            mode_pattern="cycle", mode_offset=i % 3,
            exact_dyadic=(i % 3 == 0),
            notes="full symmetric sparse PSD P, KKT by construction")
    add("qp_equalities", "qp", "solved", 75,
        [C("zero", 30), C("zero", 25), C("zero", 20)],
        tags=["feasible", "QP", "zero_cone_blocks"], p_mode="btb",
        mode_pattern="all_dual", exact_dyadic=True)
    add("qp_allfive", "qp", "solved", 120, mixed_cones(layout_rng, 180),
        tags=["feasible", "QP", "all_five_cones"], p_mode="btb",
        mode_pattern="cycle", mode_offset=1, psd_rotations=True,
        notes="B^T B objective with all five cone families")

    # SOCP: many small blocks, large dimensions up to 40, and larger sizes.
    soc_sets = [
        [2, 3, 4, 5],
        [2] * 20 + [3] * 10 + [4] * 5,
        [5, 6, 7, 8, 9, 10],
        [16, 24, 32, 40],
        [40, 40, 32, 20],
        [2 + int(layout_rng.integers(0, 5)) for _ in range(150)],
        [2, 3, 4, 5, 6, 7, 8, 10, 20, 40],
    ]
    for i, dims in enumerate(soc_sets):
        n = [6, 40, 60, 100, 200, 600, 1000][i]
        add("socp", "socp", "solved", n,
            soc_cones(dims, extra_zero=4 if i % 2 == 0 else 0,
                      extra_positive=5 if i % 3 == 0 else 0),
            tags=["feasible", "SOCP", "many_small_blocks" if i in (1, 5) else "soc_blocks"],
            p_mode="zero" if i % 2 == 0 else "diagonal",
            mode_pattern="cycle", mode_offset=i % 3, exact_dyadic=(i % 2 == 0),
            notes=f"SOC dimensions {min(dims)}..{max(dims)}")
    add("socp_large_sparse", "socp", "solved", 1500,
        soc_cones([2 + int(layout_rng.integers(0, 4)) for _ in range(250)],
                  extra_zero=100, extra_positive=100),
        tags=["feasible", "SOCP", "large_sparse"], p_mode="zero",
        mode_pattern="cycle", exact_dyadic=True)
    add("socp_all_boundary", "socp", "solved", 80,
        soc_cones([2, 3, 5, 8, 13, 21, 40], extra_zero=3),
        tags=["feasible", "SOCP", "boundary_heavy"], p_mode="zero",
        mode_pattern="all_boundary", exact_dyadic=True)
    add("socp_all_dual_interior", "socp", "solved", 50,
        soc_cones([2, 4, 8, 16, 32]),
        tags=["feasible", "SOCP", "s_zero_z_interior"], p_mode="zero",
        mode_pattern="all_dual", exact_dyadic=True)

    # SDP: dimensions 2..30, multiple blocks, off-diagonal sqrt(2) encoding.
    psd_sets = [[2], [2, 3, 4], [5, 6], [2, 3, 5, 8, 13],
                [10, 10], [15, 20], [30], [2, 3, 5, 8, 13, 21, 30],
                [30, 20, 10, 5], [4, 7, 11, 17]]
    for i, dims in enumerate(psd_sets):
        n = [4, 12, 25, 50, 80, 120, 200, 300, 400, 500][i]
        add("sdp", "sdp", "solved", n,
            psd_cones(dims, extra_zero=3 if i % 2 else 0,
                      extra_positive=4 if i % 3 == 0 else 0),
            tags=["feasible", "SDP", "multiple_psd_blocks" if len(dims) > 1 else "single_psd_block"],
            p_mode="zero" if i % 2 == 0 else "diagonal",
            mode_pattern="cycle", mode_offset=i % 3,
            psd_rotations=(i != 0), exact_dyadic=(i == 0),
            notes=f"PSD orders {dims}; full upper-triangle-column-wise encoding")
    add("sdp_boundary_ranges", "sdp", "solved", 100,
        psd_cones([2, 3, 5, 8, 13, 21, 30]),
        tags=["feasible", "SDP", "orthogonal_ranges"], p_mode="zero",
        mode_pattern="all_boundary", psd_rotations=True)
    add("sdp_dyadic_boundary", "sdp", "solved", 60,
        psd_cones([2, 4, 8, 16, 30]),
        tags=["feasible", "SDP", "exact_dyadic"], p_mode="zero",
        mode_pattern="all_boundary", exact_dyadic=True)

    # Exponential: 1..15 blocks, all complementarity modes and boundary t values.
    for k in range(1, 16):
        n = 3 * k + 5
        add("exp", "exp", "solved", n,
            exp_cones(k, extra_zero=2 if k % 2 else 0,
                      extra_positive=3 if k % 3 == 0 else 0),
            tags=["feasible", "exponential", f"{k}_exp_blocks"],
            p_mode="zero" if k % 2 else "diagonal",
            mode_pattern="cycle", mode_offset=k % 3,
            exact_dyadic=(k % 5 == 0), psd_rotations=True,
            notes="interior, dual-interior and exact boundary exponential pairs")
    add("exp_boundary_t0", "exp", "solved", 30,
        exp_cones(5, extra_zero=3, extra_positive=4),
        tags=["feasible", "exponential", "boundary_t0", "exact_dyadic"],
        p_mode="zero", mode_pattern="all_boundary", exact_dyadic=True, exp_t=0.0)
    add("exp_boundary_tlog2", "exp", "solved", 30,
        exp_cones(5, extra_zero=3, extra_positive=4),
        tags=["feasible", "exponential", "boundary_transcendental"],
        p_mode="zero", mode_pattern="all_boundary", exact_dyadic=False,
        exp_t=math.log(2.0))
    add("exp_interior_modes", "exp", "solved", 45,
        exp_cones(10, extra_zero=5, extra_positive=5),
        tags=["feasible", "exponential", "interior_modes"],
        p_mode="zero", mode_pattern="random", exact_dyadic=True)

    # Mixed all-five-cone feasible cases, including a 3000-by-3000 sparse case.
    mixed_sizes = [(10, None), (40, 120), (150, 250), (500, 700),
                   (1200, 1500), (3000, 3000)]
    for i, (n, target_m) in enumerate(mixed_sizes):
        cones = (mixed_cones(layout_rng, target_m) if target_m and target_m < 2000
                 else large_mixed_cones(layout_rng, target_m)
                 if target_m else mixed_cones(layout_rng))
        add("mixed", "mixed", "solved", n, cones,
            tags=["feasible", "all_five_cones", "mixed_blocks",
                  "large_sparse" if n >= 1200 else "small_mixed"],
            p_mode="zero" if i % 3 == 0 else "diagonal" if i % 3 == 1 else "btb",
            mode_pattern="cycle", mode_offset=i % 3,
            exact_dyadic=(i >= 3), psd_rotations=(i < 3),
            notes="interleaved zero/nonnegative/SOC/PSD/exponential cone blocks")
    add("mixed_all_boundary", "mixed", "solved", 80, mixed_cones(layout_rng, 150),
        tags=["feasible", "all_five_cones", "boundary_heavy"],
        p_mode="zero", mode_pattern="all_boundary", exact_dyadic=True)
    add("mixed_all_dual", "mixed", "solved", 80, mixed_cones(layout_rng, 150),
        tags=["feasible", "all_five_cones", "s_zero_z_interior"],
        p_mode="zero", mode_pattern="all_dual", exact_dyadic=True)

    # Ill-scaled cases: row and column scales from ~1e-4 to ~1e4.
    add("ill_scaled_row_lp", "ill_scaled", "solved", 200,
        lp_cones(100, 100, layout_rng, parts=4),
        tags=["feasible", "ill_scaled", "row_scale_1e-4_to_1e4"],
        p_mode="zero", row_scale=(-14, 14), col_scale=(0, 0),
        a_values=SMALL_A_VALUES, mode_pattern="cycle", exact_dyadic=True)
    add("ill_scaled_col_lp", "ill_scaled", "solved", 200,
        lp_cones(100, 100, layout_rng, parts=4),
        tags=["feasible", "ill_scaled", "column_scale_1e-4_to_1e4"],
        p_mode="zero", row_scale=(0, 0), col_scale=(-14, 14),
        a_values=SMALL_A_VALUES, mode_pattern="cycle", exact_dyadic=True)
    add("ill_scaled_row_qp", "ill_scaled", "solved", 120,
        [C("zero", 30), C("nonnegative", 50), C("soc", 10), C("exponential", 3),
         C("nonnegative", 40), C("soc", 5)],
        tags=["feasible", "ill_scaled", "QP", "row_scaled"],
        p_mode="diagonal", row_scale=(-14, 14), col_scale=(0, 0),
        a_values=SMALL_A_VALUES, exact_dyadic=True)
    add("ill_scaled_col_qp", "ill_scaled", "solved", 120,
        mixed_cones(layout_rng, 180),
        tags=["feasible", "ill_scaled", "QP", "column_scaled", "all_five_cones"],
        p_mode="diagonal", p_base_scale=2.0 ** -8,
        row_scale=(0, 0), col_scale=(-8, 8),
        a_values=SMALL_A_VALUES, exact_dyadic=True)
    add("ill_scaled_mixed", "ill_scaled", "solved", 300,
        mixed_cones(layout_rng, 400),
        tags=["feasible", "ill_scaled", "all_five_cones", "row_and_column_scaled"],
        p_mode="diagonal", p_base_scale=2.0 ** -8,
        row_scale=(-8, 8), col_scale=(-8, 8),
        a_values=SMALL_A_VALUES, exact_dyadic=True,
        notes="A coefficients reach about 1e5 with dyadic reciprocal row/column scales")
    add("ill_scaled_large_coeffs", "ill_scaled", "solved", 100,
        lp_cones(60, 60, layout_rng, parts=3),
        tags=["feasible", "ill_scaled", "large_coefficients"],
        p_mode="zero", row_scale=(0, 14), col_scale=(0, 0),
        a_values=np.array([-8., -4., -2., -1., 1., 2., 4., 8.]),
        exact_dyadic=True)
    add("ill_scaled_allfive", "ill_scaled", "solved", 600,
        mixed_cones(layout_rng, 800),
        tags=["feasible", "ill_scaled", "all_five_cones", "large_sparse"],
        p_mode="zero", row_scale=(-10, 10), col_scale=(-6, 6),
        a_values=SMALL_A_VALUES, exact_dyadic=True)
    add("ill_scaled_soc_psd", "ill_scaled", "solved", 80,
        [C("soc", 40), C("psd", 10), C("nonnegative", 20), C("zero", 10)],
        tags=["feasible", "ill_scaled", "SOCP", "SDP"],
        p_mode="zero", row_scale=(-10, 10), col_scale=(-6, 6),
        a_values=SMALL_A_VALUES, exact_dyadic=True)

    # Degenerate feasible cases.
    add("degenerate_qzero_pzero_allfive", "degenerate", "solved", 30,
        mixed_cones(layout_rng, 90),
        tags=["feasible", "q_zero", "P_zero", "z_zero", "all_five_cones"],
        p_mode="zero", q_zero=True, force_z_zero=True,
        mode_pattern="all_primal", exact_dyadic=True)
    add("degenerate_qzero_diagonalP", "degenerate", "solved", 40,
        mixed_cones(layout_rng, 120),
        tags=["feasible", "q_zero", "P_nonzero", "nullspace", "all_five_cones"],
        p_mode="diagonal_null", q_zero=True, force_z_zero=True,
        mode_pattern="all_primal", exact_dyadic=True)
    add("degenerate_qzero_btbP", "degenerate", "solved", 50,
        [C("zero", 10), C("nonnegative", 20), C("soc", 8)],
        tags=["feasible", "q_zero", "P_BtB_nullspace", "empty_families"],
        p_mode="btb_null", q_zero=True, force_z_zero=True,
        mode_pattern="all_primal", exact_dyadic=True)
    add("degenerate_pzero_all_dual", "degenerate", "solved", 30,
        mixed_cones(layout_rng, 100),
        tags=["feasible", "P_zero", "s_zero_z_interior", "all_five_cones"],
        p_mode="zero", mode_pattern="all_dual", exact_dyadic=True)
    add("degenerate_zero_cones", "degenerate", "solved", 20,
        [C("zero", 8), C("zero", 12), C("zero", 5)],
        tags=["feasible", "zero_cone_blocks", "empty_families"],
        p_mode="zero", mode_pattern="all_dual", exact_dyadic=True)
    add("degenerate_empty_cones", "degenerate", "solved", 5, [],
        tags=["feasible", "empty_cones", "m_zero", "unconstrained_QP"],
        p_mode="diagonal", mode_pattern="all_primal", exact_dyadic=True,
        notes="cones=[] and A has shape (0,n); s,z are empty arrays")
    add("degenerate_empty_cones_qzero", "degenerate", "solved", 8, [],
        tags=["feasible", "empty_cones", "m_zero", "q_zero", "P_zero"],
        p_mode="zero", q_zero=True, force_z_zero=True,
        mode_pattern="all_primal", exact_dyadic=True)
    add("degenerate_all_boundary_allfive", "degenerate", "solved", 50,
        mixed_cones(layout_rng, 140),
        tags=["feasible", "all_five_cones", "boundary_heavy", "P_zero"],
        p_mode="zero", mode_pattern="all_boundary", exact_dyadic=True)
    add("degenerate_constant_objective", "degenerate", "solved", 20,
        [C("nonnegative", 10), C("zero", 5)],
        tags=["feasible", "q_zero", "P_zero", "constant_objective"],
        p_mode="zero", q_zero=True, force_z_zero=True,
        mode_pattern="all_primal", exact_dyadic=True)

    # Primal infeasibility certificates with O(1) normalized margins.
    add("primal_infeasible_zero", "primal_infeasible", "primal_infeasible", 5,
        [C("zero", 2), C("nonnegative", 5)],
        tags=["certificate", "zero_cone", "nontrivial_cancellation"],
        witness="zero_cancel", p_mode="zero", a_values=SMALL_A_VALUES,
        notes="two contradictory equality rows with z=(-1,1)")
    add("primal_infeasible_nonneg", "primal_infeasible", "primal_infeasible", 4,
        [C("nonnegative", 2), C("nonnegative", 5)],
        tags=["certificate", "nonnegative", "nontrivial_cancellation"],
        witness="nonneg_cancel", p_mode="zero", a_values=SMALL_A_VALUES,
        notes="a^T x >= 1 and a^T x <= -1")
    add("primal_infeasible_soc", "primal_infeasible", "primal_infeasible", 8,
        [C("soc", 12), C("nonnegative", 10), C("zero", 4)],
        tags=["certificate", "soc"], witness="soc_first", p_mode="diagonal",
        notes="negative first SOC coordinate with z=e0")
    add("primal_infeasible_soc40", "primal_infeasible", "primal_infeasible", 20,
        [C("soc", 40), C("nonnegative", 20)],
        tags=["certificate", "soc", "large_soc"], witness="soc_first",
        p_mode="zero")
    add("primal_infeasible_psd", "primal_infeasible", "primal_infeasible", 10,
        [C("psd", 8), C("nonnegative", 10), C("zero", 3)],
        tags=["certificate", "psd"], witness="psd_diag", p_mode="btb",
        notes="z=vec(I), b=-z, and all diagonal rows of A vanish")
    add("primal_infeasible_psd30", "primal_infeasible", "primal_infeasible", 25,
        [C("psd", 30), C("nonnegative", 20)],
        tags=["certificate", "psd", "large_psd"], witness="psd_diag",
        p_mode="zero")
    add("primal_infeasible_exp", "primal_infeasible", "primal_infeasible", 8,
        [C("exponential", 3), C("nonnegative", 6), C("zero", 3)],
        tags=["certificate", "exponential"], witness="exp_cancel",
        p_mode="diagonal", a_values=SMALL_A_VALUES,
        notes="z=(-1,0,1) in K_exp^*, A_u=A_w, and b^T z=-2")
    add("primal_infeasible_allfive", "primal_infeasible", "primal_infeasible", 20,
        [C("exponential", 3), C("zero", 4), C("nonnegative", 6),
         C("soc", 7), C("psd", 5), C("exponential", 3)],
        tags=["certificate", "all_five_cones"], witness="exp_cancel",
        p_mode="btb", a_values=SMALL_A_VALUES,
        notes="all five cone types; exact exponential-cone Farkas witness")
    add("primal_infeasible_scaled", "primal_infeasible", "primal_infeasible", 12,
        [C("nonnegative", 2), C("nonnegative", 10), C("zero", 4)],
        tags=["certificate", "strict_A_scaling", "coefficients_1e5"],
        witness="nonneg_cancel", p_mode="zero", a_values=LARGE_A_VALUES,
        notes="max|A_ij|=1e5; A/a scaling is harmless because rows cancel exactly")
    add("primal_infeasible_large", "primal_infeasible", "primal_infeasible", 1500,
        [C("nonnegative", 2), C("nonnegative", 1498)],
        tags=["certificate", "large_sparse", "nonnegative"],
        witness="nonneg_cancel", p_mode="zero", a_values=SMALL_A_VALUES,
        witness_nnz=10,
        notes="1500-row Farkas certificate with sparse A")

    # Dual infeasibility / primal-unbounded cases: feasible at x0=0.
    add("unbounded_lp", "unbounded", "dual_infeasible", 2,
        [C("nonnegative", 2)],
        tags=["certificate", "unbounded", "LP"], direction="e0",
        p_mode="zero", notes="min -x0 with x0>=0; b=s0 is strictly feasible")
    add("unbounded_no_cones", "unbounded", "dual_infeasible", 8, [],
        tags=["certificate", "unbounded", "empty_cones", "m_zero"],
        direction="e0", p_mode="diagonal_null",
        notes="unconstrained linear objective with P x=0")
    add("unbounded_nonzeroP", "unbounded", "dual_infeasible", 60,
        [C("zero", 10), C("nonnegative", 20), C("soc", 8)],
        tags=["certificate", "unbounded", "P_nonzero", "nullspace"],
        direction="e0", p_mode="btb_null", q_scale=16.0)
    add("unbounded_soc", "unbounded", "dual_infeasible", 12,
        [C("soc", 20), C("soc", 5), C("zero", 4)],
        tags=["certificate", "unbounded", "soc"], direction="e0",
        p_mode="zero")
    add("unbounded_psd", "unbounded", "dual_infeasible", 10,
        [C("psd", 10), C("psd", 4), C("nonnegative", 5)],
        tags=["certificate", "unbounded", "psd"], direction="e0",
        p_mode="diagonal_null")
    add("unbounded_exp", "unbounded", "dual_infeasible", 8,
        [C("exponential", 3), C("exponential", 3), C("nonnegative", 4)],
        tags=["certificate", "unbounded", "exponential"], direction="e0",
        p_mode="zero")
    add("unbounded_allfive", "unbounded", "dual_infeasible", 40,
        [C("zero", 5), C("nonnegative", 8), C("soc", 6), C("psd", 4),
         C("exponential", 3), C("soc", 3)],
        tags=["certificate", "unbounded", "all_five_cones"],
        direction="e0", p_mode="btb_null", q_scale=1024.0)
    add("unbounded_mixed_direction", "unbounded", "dual_infeasible", 100,
        mixed_cones(layout_rng, 160),
        tags=["certificate", "unbounded", "all_five_cones", "dense_direction"],
        direction="mixed", p_mode="btb_null", q_scale=4.0,
        notes="recession direction has many +/-1 components; B x=0 by paired rows")
    add("unbounded_scaled_A", "unbounded", "dual_infeasible", 30,
        [C("nonnegative", 10), C("soc", 8), C("psd", 3), C("zero", 4)],
        tags=["certificate", "unbounded", "strict_A_scaling", "coefficients_1e5"],
        direction="negative_e0", p_mode="zero", a_values=LARGE_A_VALUES,
        q_scale=16384.0)
    add("unbounded_large", "unbounded", "dual_infeasible", 3000,
        lp_cones(1500, 1500, layout_rng, parts=6),
        tags=["certificate", "unbounded", "large_sparse"],
        direction="e0", p_mode="zero", a_values=SMALL_A_VALUES,
        notes="3000 variables and 3000 cone rows with sparse A")

    return specs


# ---------------------------------------------------------------------------
# Verification and report writing
# ---------------------------------------------------------------------------

def verify_record(record, expected_status):
    input_path = TESTS / record["input"]
    result_path = TESTS / record["result"]
    report = run_checker_cli(input_path, result_path)
    check = report.get("check", {}) or {}
    residuals = check.get("residuals", {}) or {}
    entry = {
        "id": record["id"],
        "family": record["family"],
        "expected_status": expected_status,
        "outcome": report.get("outcome"),
        "status": check.get("status"),
        "reason": check.get("reason"),
        "passed": bool(check.get("passed", False)),
        "residuals": json_safe(residuals),
        "input": record["input"],
        "result": record["result"],
        "input_bytes": input_path.stat().st_size if input_path.exists() else 0,
        "result_bytes": result_path.stat().st_size if result_path.exists() else 0,
        "input_sha256": sha256_file(input_path) if input_path.exists() else None,
        "result_sha256": sha256_file(result_path) if result_path.exists() else None,
    }
    if report.get("error"):
        entry["error"] = report["error"]
    if report.get("detail"):
        entry["detail"] = report["detail"]
    verified = (report.get("outcome") == "verified" and entry["passed"]
                and entry["status"] == expected_status)
    entry["verified"] = bool(verified)
    return entry


def summarize(entries, manifest_cases):
    families = Counter(case["family"] for case in manifest_cases)
    statuses = Counter(case["expected_status"] for case in manifest_cases)
    cone_types = Counter(kind for case in manifest_cases
                         for kind in case["cone_types"])
    mode_counts = Counter()
    for case in manifest_cases:
        for mode, count in case.get("cone_mode_counts", {}).items():
            mode_counts[mode] += count
    total_case_bytes = sum(case.get("input_bytes", 0) + case.get("result_bytes", 0)
                           for case in manifest_cases)
    failed = [entry for entry in entries if not entry["verified"]]
    solved_residuals = [entry["residuals"] for entry in entries
                        if entry["expected_status"] == "solved" and entry["verified"]]
    worst = {}
    for key in ("primal", "dual", "gap_absolute", "gap_relative",
                "complementarity_absolute", "primal_cone", "dual_cone"):
        values = [float(res[key]) for res in solved_residuals
                  if key in res and np.isfinite(float(res[key]))]
        worst[key] = max(values) if values else 0.0
    return {
        "seed": MASTER_SEED,
        "case_count": len(manifest_cases),
        "verified_count": len(entries) - len(failed),
        "failed_count": len(failed),
        "failed_ids": [entry["id"] for entry in failed],
        "families": dict(sorted(families.items())),
        "statuses": dict(sorted(statuses.items())),
        "cone_types": dict(sorted(cone_types.items())),
        "complementarity_modes": dict(sorted(mode_counts.items())),
        "empty_cone_family_cases": sum(1 for case in manifest_cases
                                       if case["empty_families"]),
        "all_five_cone_cases": sum(1 for case in manifest_cases
                                   if len(case["cone_types"]) == 5),
        "total_case_bytes": int(total_case_bytes),
        "total_case_MiB": round(total_case_bytes / (1024.0 ** 2), 6),
        "worst_solved_residuals": json_safe(worst),
        "checker": f"{sys.executable} {CHECKER} --input INPUT --result RESULT",
        "cases": entries,
    }


def write_manifest(cases):
    manifest = {
        "schema_version": 1,
        "purpose": "synthetic_solver_validation",
        "protocol": "candidate-v1",
        "seed": MASTER_SEED,
        "generator": "gen_cases.py",
        "checker": str(CHECKER),
        "path_base": "directory containing this manifest",
        "case_count": len(cases),
        "total_case_bytes": sum(case.get("input_bytes", 0) + case.get("result_bytes", 0)
                                for case in cases),
        "cases": json_safe(cases),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True,
                                   ensure_ascii=False, allow_nan=False) + "\n")


def matches_only(case, patterns):
    if not patterns:
        return True
    haystacks = [case["id"], case["family"], case["expected_status"]]
    haystacks.extend(case.get("tags", []))
    haystacks.extend(case.get("cone_types", []))
    for pattern in patterns:
        lowered = pattern.lower()
        for value in haystacks:
            text = str(value)
            if lowered in text.lower() or fnmatch.fnmatchcase(text, pattern):
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                        help="re-run official check_result.py on saved cases")
    parser.add_argument("--skip-verify", action="store_true",
                        help="generate files without running the official checker")
    parser.add_argument("--only", action="append", default=[],
                        help="comma-separated substring filter (verify-only)")
    args = parser.parse_args()
    if not CHECKER.exists():
        raise SystemExit(f"checker not found: {CHECKER}")

    if args.verify_only:
        manifest = json.loads(MANIFEST.read_text())
        cases = manifest["cases"]
        patterns = [piece.strip() for value in args.only
                    for piece in value.split(",") if piece.strip()]
        cases = [case for case in cases if matches_only(case, patterns)]
    else:
        for directory in (INPUT_DIR, RESULT_DIR):
            if directory.exists():
                for path in directory.glob("*.npz"):
                    path.unlink()
            else:
                directory.mkdir(parents=True)
        specs = build_specs()
        cases = []
        for index, spec in enumerate(specs):
            status = spec["expected_status"]
            if status == "solved":
                record = make_feasible_case(spec, index)
            elif status == "primal_infeasible":
                record = make_primal_certificate_case(spec, index)
            elif status == "dual_infeasible":
                record = make_dual_certificate_case(spec, index)
            else:
                raise ValueError(f"unknown status {status}")
            if not args.skip_verify:
                entry = verify_record(record, status)
                record["verification"] = {key: entry[key] for key in
                                          ("outcome", "status", "passed", "reason",
                                           "residuals", "verified")}
                print(f"{record['id']:<34} outcome={entry['outcome']:<18} "
                      f"status={entry['status']}", flush=True)
                if not entry["verified"]:
                    print(f"  FAILED: {entry.get('reason')} "
                          f"{entry.get('error', '')} {entry.get('residuals', {})}",
                          flush=True)
            record["input_bytes"] = (TESTS / record["input"]).stat().st_size
            record["result_bytes"] = (TESTS / record["result"]).stat().st_size
            record["input_sha256"] = sha256_file(TESTS / record["input"])
            record["result_sha256"] = sha256_file(TESTS / record["result"])
            cases.append(record)
        write_manifest(cases)

    entries = []
    for case in cases:
        if args.skip_verify and not args.verify_only:
            continue
        entry = verify_record(case, case["expected_status"])
        entries.append(entry)
        if args.verify_only:
            print(f"{entry['id']:<34} outcome={entry['outcome']:<18} "
                  f"verified={entry['verified']}", flush=True)

    if not args.skip_verify or args.verify_only:
        report = summarize(entries, cases)
        REPORT.write_text(json.dumps(report, indent=2, sort_keys=True,
                                     ensure_ascii=False, allow_nan=False) + "\n")
        print(f"\ncases={report['case_count']} verified={report['verified_count']} "
              f"failed={report['failed_count']} "
              f"data={report['total_case_MiB']:.3f} MiB", flush=True)
        if report["failed_count"]:
            print("failed ids: " + ", ".join(report["failed_ids"]), flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
