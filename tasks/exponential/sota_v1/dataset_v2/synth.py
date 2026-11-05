#!/usr/bin/env python3
"""Deterministic PSD synthetic mother families for exponential Dataset v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent


def _laplacian(n: int, u: np.ndarray, v: np.ndarray, weight: np.ndarray,
               potential: np.ndarray | float = 0.0) -> sp.csr_matrix:
    rows = np.concatenate((u, v, u, v))
    cols = np.concatenate((u, v, v, u))
    values = np.concatenate((weight, weight, -weight, -weight))
    matrix = sp.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
    if np.isscalar(potential):
        if potential:
            matrix = matrix + sp.eye(n, format="csr") * float(potential)
    else:
        matrix = matrix + sp.diags(np.asarray(potential), format="csr")
    return _canonical(matrix)


def _canonical(matrix: sp.spmatrix) -> sp.csr_matrix:
    result = matrix.tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def diffusion_2d(nx: int, ny: int, seed: int, contrast: float = 1e4,
                 nine_point: bool = True) -> sp.csr_matrix:
    """Variable-coefficient anisotropic 5/9-point diffusion plus potential."""
    rng = np.random.default_rng(seed)
    n = nx * ny
    log_c = rng.uniform(0, np.log(contrast), size=(ny, nx))
    coefficient = np.exp(log_c)
    anisotropy = np.exp(rng.uniform(-1.5, 1.5, size=(ny, nx)))
    potential = np.exp(rng.uniform(-8, -4, size=n))
    edges_u: list[np.ndarray] = []
    edges_v: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    grid = np.arange(n).reshape(ny, nx)
    for dy, dx, factor in ((0, 1, 1.0), (1, 0, 1.0),
                           (1, 1, 0.2), (1, -1, 0.2)):
        if not nine_point and dx != 0 and dy != 0:
            continue
        ys = slice(0, ny - dy)
        yt = slice(dy, ny)
        if dx >= 0:
            xs, xt = slice(0, nx - dx), slice(dx, nx)
        else:
            xs, xt = slice(-dx, nx), slice(0, nx + dx)
        u, v = grid[ys, xs].ravel(), grid[yt, xt].ravel()
        c0, c1 = coefficient[ys, xs].ravel(), coefficient[yt, xt].ravel()
        harmonic = 2 * c0 * c1 / (c0 + c1)
        direction = np.sqrt(anisotropy[ys, xs].ravel() * anisotropy[yt, xt].ravel())
        edges_u.append(u)
        edges_v.append(v)
        weights.append(factor * harmonic * direction)
    return _laplacian(n, np.concatenate(edges_u), np.concatenate(edges_v),
                      np.concatenate(weights), potential)


def diffusion_3d(nx: int, ny: int, nz: int, seed: int,
                 contrast: float = 1e3) -> sp.csr_matrix:
    """7-point anisotropic diffusion with mixed coefficient blocks."""
    rng = np.random.default_rng(seed)
    n = nx * ny * nz
    grid = np.arange(n).reshape(nz, ny, nx)
    block = np.indices((nz, ny, nx)).sum(axis=0) % 4
    base = np.power(contrast, block / 3.0) * np.exp(rng.normal(0, 0.15, size=block.shape))
    direction_scale = np.exp(rng.uniform(-1.2, 1.2, size=3))
    us, vs, ws = [], [], []
    for axis, scale in enumerate(direction_scale):
        first = [slice(None)] * 3
        second = [slice(None)] * 3
        first[axis] = slice(0, grid.shape[axis] - 1)
        second[axis] = slice(1, grid.shape[axis])
        u, v = grid[tuple(first)].ravel(), grid[tuple(second)].ravel()
        c0, c1 = base[tuple(first)].ravel(), base[tuple(second)].ravel()
        us.append(u)
        vs.append(v)
        ws.append(scale * 2 * c0 * c1 / (c0 + c1))
    potential = np.exp(rng.uniform(-9, -5, size=n))
    return _laplacian(n, np.concatenate(us), np.concatenate(vs), np.concatenate(ws), potential)


def geometric_graph(n: int, seed: int, target_degree: float = 8.0) -> sp.csr_matrix:
    """Random geometric graph Laplacian with log-uniform edge weights."""
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2))
    radius = np.sqrt(target_degree / (np.pi * n))
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    # Attach only isolated vertices to their nearest neighbor. A full chain would
    # inflate average degree by two and can violate the one-million-nnz cap.
    degree = np.bincount(pairs.ravel(), minlength=n) if len(pairs) else np.zeros(n, dtype=int)
    isolated = np.flatnonzero(degree == 0)
    if isolated.size:
        tree = cKDTree(points)
        _, nearest = tree.query(points[isolated], k=2)
        attachments = np.column_stack((isolated, nearest[:, 1]))
        pairs = np.vstack((pairs, attachments))
    pairs.sort(axis=1)
    pairs = np.unique(pairs, axis=0)
    weight = np.exp(rng.uniform(np.log(1e-2), np.log(1e2), size=len(pairs)))
    return _laplacian(n, pairs[:, 0], pairs[:, 1], weight)


def heavy_tailed_graph(n: int, seed: int, edges_per_vertex: int = 4) -> sp.csr_matrix:
    """Preferential-attachment/configuration-model Laplacian."""
    if edges_per_vertex < 2 or n <= edges_per_vertex:
        raise ValueError("need n > edges_per_vertex >= 2")
    rng = np.random.default_rng(seed)
    m = edges_per_vertex
    u: list[int] = []
    v: list[int] = []
    repeated: list[int] = []
    for i in range(m + 1):
        for j in range(i):
            u.append(i)
            v.append(j)
            repeated.extend((i, j))
    for vertex in range(m + 1, n):
        targets: set[int] = set()
        while len(targets) < m:
            targets.add(repeated[int(rng.integers(len(repeated)))])
        for target in sorted(targets):
            u.append(vertex)
            v.append(target)
            repeated.extend((vertex, target))
    weights = np.exp(rng.uniform(np.log(0.1), np.log(10), size=len(u)))
    return _laplacian(n, np.asarray(u), np.asarray(v), weights)


def signed_incidence(n: int, seed: int, average_degree: float = 10.0) -> sp.csr_matrix:
    """B.T W B + diag(v), allowing positive as well as negative off-diagonals."""
    rng = np.random.default_rng(seed)
    edges = int(round(n * average_degree / 2))
    u = rng.integers(0, n, size=edges)
    v = rng.integers(0, n - 1, size=edges)
    v += v >= u
    row = np.repeat(np.arange(edges), 2)
    col = np.column_stack((u, v)).ravel()
    signs = rng.choice(np.array([-1.0, 1.0]), size=(edges, 2)).ravel()
    incidence = sp.csr_matrix((signs, (row, col)), shape=(edges, n))
    weight = np.exp(rng.uniform(np.log(1e-2), np.log(1e2), size=edges))
    matrix = incidence.T @ sp.diags(weight) @ incidence
    matrix = matrix + sp.diags(np.exp(rng.uniform(-9, -5, size=n)))
    return _canonical(matrix)


def log_spectrum_local(n: int, seed: int, condition: float = 1e10) -> sp.csr_matrix:
    """Log spectrum conjugated by disjoint sparse orthogonal 2x2 rotations."""
    rng = np.random.default_rng(seed)
    diagonal = np.geomspace(1.0 / condition, 1.0, n)
    permutation = rng.permutation(n)
    diagonal = diagonal[permutation]
    rows, cols, values = [], [], []
    for i in range(0, n - 1, 2):
        a, b = diagonal[i], diagonal[i + 1]
        angle = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(angle), np.sin(angle)
        block = ((c * c * a + s * s * b, c * s * (a - b)),
                 (c * s * (a - b), s * s * a + c * c * b))
        rows.extend((i, i, i + 1, i + 1))
        cols.extend((i, i + 1, i, i + 1))
        values.extend((block[0][0], block[0][1], block[1][0], block[1][1]))
    if n % 2:
        rows.append(n - 1)
        cols.append(n - 1)
        values.append(diagonal[-1])
    return _canonical(sp.coo_matrix((values, (rows, cols)), shape=(n, n)))


def save_mother(path: Path, matrix: sp.csr_matrix, metadata: dict) -> None:
    matrix = _canonical(matrix)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        data=np.asarray(matrix.data, dtype="<f8"),
        indices=np.asarray(matrix.indices, dtype="<u4"),
        indptr=np.asarray(matrix.indptr, dtype="<u8"),
        shape=np.asarray(matrix.shape, dtype="<u8"),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def public_specs(smoke: bool) -> list[dict]:
    if smoke:
        return [
            dict(name="syn_diffusion2d", family="mesh_diffusion", fn="diffusion_2d",
                 args=dict(nx=32, ny=32, nine_point=True)),
            dict(name="syn_diffusion2d_b", family="mesh_diffusion", fn="diffusion_2d",
                 args=dict(nx=36, ny=28, nine_point=False)),
            dict(name="syn_diffusion3d", family="mesh_diffusion", fn="diffusion_3d",
                 args=dict(nx=10, ny=10, nz=10)),
            dict(name="syn_geometric", family="geometric_graph", fn="geometric_graph",
                 args=dict(n=1500, target_degree=8)),
            dict(name="syn_geometric_b", family="geometric_graph", fn="geometric_graph",
                 args=dict(n=1600, target_degree=10)),
            dict(name="syn_heavytail", family="network_operator", fn="heavy_tailed_graph",
                 args=dict(n=1500, edges_per_vertex=4)),
            dict(name="syn_signed", family="signed_incidence", fn="signed_incidence",
                 args=dict(n=1200, average_degree=10)),
            dict(name="syn_logspectrum", family="diagonal_local", fn="log_spectrum_local",
                 args=dict(n=2000, condition=1e8)),
        ]
    return [
        dict(name="syn_diffusion2d", family="mesh_diffusion", fn="diffusion_2d",
             args=dict(nx=260, ny=260, nine_point=True)),
        dict(name="syn_diffusion2d_b", family="mesh_diffusion", fn="diffusion_2d",
             args=dict(nx=280, ny=240, nine_point=False)),
        dict(name="syn_diffusion3d", family="mesh_diffusion", fn="diffusion_3d",
             args=dict(nx=42, ny=42, nz=42)),
        dict(name="syn_geometric", family="geometric_graph", fn="geometric_graph",
             args=dict(n=80_000, target_degree=8)),
        dict(name="syn_geometric_b", family="geometric_graph", fn="geometric_graph",
             args=dict(n=75_000, target_degree=10)),
        dict(name="syn_heavytail", family="network_operator", fn="heavy_tailed_graph",
             args=dict(n=80_000, edges_per_vertex=4)),
        dict(name="syn_signed", family="signed_incidence", fn="signed_incidence",
             args=dict(n=60_000, average_degree=10)),
        dict(name="syn_logspectrum", family="diagonal_local", fn="log_spectrum_local",
             args=dict(n=100_000, condition=1e10)),
    ]


def private_specs() -> list[dict]:
    return [
        dict(name="syn_diffusion2d", family="mesh_diffusion", fn="diffusion_2d",
             args=dict(nx=300, ny=300, nine_point=True)),
        dict(name="syn_diffusion3d", family="mesh_diffusion", fn="diffusion_3d",
             args=dict(nx=46, ny=46, nz=46)),
        dict(name="syn_geometric", family="geometric_graph", fn="geometric_graph",
             args=dict(n=100_000, target_degree=8)),
        dict(name="syn_heavytail", family="network_operator", fn="heavy_tailed_graph",
             args=dict(n=100_000, edges_per_vertex=4)),
        dict(name="syn_signed", family="signed_incidence", fn="signed_incidence",
             args=dict(n=80_000, average_degree=10)),
        dict(name="syn_logspectrum", family="diagonal_local", fn="log_spectrum_local",
             args=dict(n=120_000, condition=1e12)),
    ]


FUNCTIONS = {
    "diffusion_2d": diffusion_2d,
    "diffusion_3d": diffusion_3d,
    "geometric_graph": geometric_graph,
    "heavy_tailed_graph": heavy_tailed_graph,
    "signed_incidence": signed_incidence,
    "log_spectrum_local": log_spectrum_local,
}


def generate(split: str, smoke: bool) -> list[dict]:
    seed_path = ROOT / f"{split}_seeds.json"
    seeds = json.loads(seed_path.read_text())
    specs = public_specs(smoke) if split == "public" else private_specs()
    output = []
    for spec in specs:
        seed = int(seeds["mothers"][spec["name"]])
        matrix = FUNCTIONS[spec["fn"]](seed=seed, **spec["args"])
        if matrix.shape[0] > 120_000 or matrix.nnz > 1_000_000:
            raise ValueError(f"{spec['name']} exceeds protocol: {matrix.shape}, nnz={matrix.nnz}")
        metadata = {**spec, "seed": seed, "split": split, "smoke": smoke,
                    "n": matrix.shape[0], "nnz": matrix.nnz}
        path = ROOT / "mothers" / (f"{split}_smoke" if smoke else split) / f"{spec['name']}.npz"
        save_mother(path, matrix, metadata)
        output.append(metadata | {"file": str(path.relative_to(ROOT))})
        print(spec["name"], matrix.shape[0], matrix.nnz)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("public", "private"), default="public")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.split == "private" and args.smoke:
        parser.error("private smoke generation is intentionally disabled")
    generated = generate(args.split, args.smoke)
    path = ROOT / f"synthetic_{args.split}{'_smoke' if args.smoke else ''}_manifest.json"
    path.write_text(json.dumps({"schema": 1, "split": args.split, "mothers": generated},
                               indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
