#!/usr/bin/env python3
"""Fetch, convert, and validate the frozen Dataset-v2 standard operators."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp
import scipy.sparse.linalg as spla


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
MOTHERS = ROOT / "mothers"
URLS = (
    "https://suitesparse-collection-website.herokuapp.com/MM/{group}/{name}.tar.gz",
    "https://sparse.tamu.edu/MM/{group}/{name}.tar.gz",
    # The University of Florida mirror serves the same collection archives; reachable when the two above are not.
    "https://www.cise.ufl.edu/research/sparse/MM/{group}/{name}.tar.gz",
)

# Split assignment and replacements are declared before any fetch or numerical test.
# A failed public matrix is replaced by the next public candidate; private thermal1
# may replace an unavailable/unreliable GHS_psdef candidate as required by DESIGN 2.2.
MATRICES = (
    dict(group="HB", name="bcsstk17", split="public", kind="spd",
         replacement=None, timed=False),
    dict(group="HB", name="bcsstk18", split="public", kind="spd",
         replacement=None, timed=False),
    dict(group="DIMACS10", name="fe_4elt2", split="public", kind="adjacency",
         replacement="DIMACS10/wing_nodal", timed=False),
    dict(group="DIMACS10", name="wing_nodal", split="public", kind="adjacency",
         replacement="DIMACS10/fe_4elt2", timed=False),
    dict(group="Schmid", name="thermal1", split="private", kind="spd",
         replacement="a separately frozen thermal operator", timed=True),
    dict(group="SNAP", name="email-Enron", split="private", kind="adjacency",
         replacement="SNAP/cit-HepPh", timed=True),
    dict(group="SNAP", name="cit-HepPh", split="private", kind="adjacency",
         replacement="SNAP/email-Enron", timed=True),
    dict(group="DIMACS10", name="road_usa", split="private", kind="adjacency",
         replacement="SNAP/roadNet-CA", timed=True, bfs_n=120_000),
    dict(group="DIMACS10", name="fe_ocean", split="private", kind="adjacency",
         replacement="a frozen independent unstructured mesh", timed=True, bfs_n=100_000),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(spec: dict, timeout: int = 60) -> tuple[Path | None, dict]:
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / f"{spec['group']}__{spec['name']}.tar.gz"
    attempts = []
    if target.exists():
        return target, {"status": "cached", "url": None, "attempts": attempts}
    for template in URLS:
        url = template.format(**spec)
        partial = target.with_suffix(target.suffix + ".part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "FrontierAM-dataset-v2/1"})
            with urllib.request.urlopen(request, timeout=timeout) as source, partial.open("wb") as sink:
                while block := source.read(1 << 20):
                    sink.write(block)
            partial.replace(target)
            return target, {"status": "downloaded", "url": url, "attempts": attempts}
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            partial.unlink(missing_ok=True)
            attempts.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
    return None, {"status": "unavailable", "url": None, "attempts": attempts,
                  "replacement_rule": spec.get("replacement")}


def extract_matrix(archive: Path) -> tuple[sp.csr_matrix, str]:
    with tarfile.open(archive, "r:gz") as bundle:
        members = sorted(
            (member for member in bundle.getmembers()
             if member.isfile() and member.name.lower().endswith((".mtx", ".rua", ".rsa"))),
            key=lambda member: (not member.name.lower().endswith(".mtx"), member.name),
        )
        if not members:
            raise ValueError("archive contains no Matrix Market or Harwell-Boeing matrix")
        member = members[0]
        stream = bundle.extractfile(member)
        if stream is None:
            raise ValueError(f"cannot extract {member.name}")
        if member.name.lower().endswith(".mtx"):
            matrix = scipy.io.mmread(stream)
        else:
            raise ValueError("Harwell-Boeing-only archive is unsupported; expected collection .mtx")
    return sp.csr_matrix(matrix, dtype=np.float64), member.name


def canonical_bfs_vertices(adjacency: sp.csr_matrix, limit: int) -> np.ndarray:
    """DESIGN 2.2 rule: first unused canonical vertex, BFS, then induced graph."""
    n = adjacency.shape[0]
    if n <= limit:
        return np.arange(n, dtype=np.int64)
    graph = adjacency.copy()
    graph.setdiag(0)
    graph.eliminate_zeros()
    seen = np.zeros(n, dtype=bool)
    selected: list[int] = []
    for root in range(n):
        if seen[root]:
            continue
        seen[root] = True
        queue = deque([root])
        while queue and len(selected) < limit:
            vertex = queue.popleft()
            selected.append(vertex)
            begin, end = graph.indptr[vertex:vertex + 2]
            for neighbor in graph.indices[begin:end]:
                neighbor = int(neighbor)
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
        if len(selected) == limit:
            break
    # Preserve BFS discovery ordering: this is the frozen derived-operator ordering.
    return np.asarray(selected, dtype=np.int64)


def adjacency_laplacian(matrix: sp.csr_matrix) -> sp.csr_matrix:
    matrix = matrix.copy()
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    adjacency = (matrix + matrix.T) * 0.5
    adjacency.data = np.abs(adjacency.data)
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return (sp.diags(degree, format="csr") - adjacency).tocsr()


def validate_psd(matrix: sp.csr_matrix) -> dict:
    delta = matrix - matrix.T
    symmetry_residual = (float(np.max(np.abs(delta.data))) if delta.nnz else 0.0)
    diagonal = matrix.diagonal()
    row_offdiag = np.asarray(abs(matrix).sum(axis=1)).ravel() - np.abs(diagonal)
    diagonal_dominant = bool(np.all(diagonal >= row_offdiag - 1e-10 * np.maximum(1, row_offdiag)))
    smallest = None
    lanczos_error = None
    try:
        smallest = float(spla.eigsh(matrix, k=min(3, max(1, matrix.shape[0] - 1)),
                                    which="SA", return_eigenvectors=False,
                                    tol=1e-6, maxiter=400)[0])
    except (spla.ArpackError, ValueError, RuntimeError) as exc:
        lanczos_error = f"{type(exc).__name__}: {exc}"
    scale = max(1.0, float(np.max(np.asarray(abs(matrix).sum(axis=1)))))
    if smallest is None:
        # Plain Lanczos stalls on ill-conditioned FEM matrices (bcsstk*: condition ~1e10). Shift-invert just below 0
        # converges to the eigenvalues nearest the shift, i.e. the smallest ones, including any negative ones.
        try:
            smallest = float(np.min(spla.eigsh(matrix.tocsc(), k=3, sigma=-1e-6 * scale, which="LM",
                                               return_eigenvectors=False)))
            lanczos_error = f"{lanczos_error}; shift-invert fallback used"
        except (spla.ArpackError, ValueError, RuntimeError) as exc:
            lanczos_error = f"{lanczos_error}; shift-invert: {type(exc).__name__}: {exc}"
    psd = bool(np.min(diagonal, initial=0) >= -1e-10 * scale and
               (diagonal_dominant or (smallest is not None and smallest >= -1e-8 * scale)))
    return {
        "symmetry_residual_max": symmetry_residual,
        "diagonal_dominant": diagonal_dominant,
        "lanczos_smallest_estimate": smallest,
        "lanczos_error": lanczos_error,
        "psd_pass": psd,
    }


def save_mother(matrix: sp.csr_matrix, spec: dict, provenance: dict) -> Path:
    matrix = matrix.tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix.eliminate_zeros()
    path = MOTHERS / spec["split"] / f"standard_{spec['group']}_{spec['name']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        data=np.asarray(matrix.data, dtype="<f8"),
        indices=np.asarray(matrix.indices, dtype="<u4"),
        indptr=np.asarray(matrix.indptr, dtype="<u8"),
        shape=np.asarray(matrix.shape, dtype="<u8"),
        metadata=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    return path


def process(spec: dict, fetch_only: bool) -> dict:
    archive, fetch = download(spec)
    record = {**spec, **fetch}
    if archive is None:
        return record
    record.update(archive=str(archive.relative_to(ROOT)), archive_sha256=sha256_file(archive),
                  archive_bytes=archive.stat().st_size)
    if fetch_only:
        return record
    try:
        matrix, member = extract_matrix(archive)
        original_shape, original_nnz = list(matrix.shape), int(matrix.nnz)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"non-square matrix {matrix.shape}")
        vertex_hash = None
        bfs_limit = spec.get("bfs_n")
        if bfs_limit and matrix.shape[0] > bfs_limit:
            graph = (abs(matrix) + abs(matrix.T)).tocsr()
            vertices = canonical_bfs_vertices(graph, bfs_limit)
            vertex_hash = hashlib.sha256(vertices.astype("<u8").tobytes()).hexdigest()
            matrix = matrix[vertices][:, vertices].tocsr()
        if spec["kind"] == "adjacency":
            matrix = adjacency_laplacian(matrix)
        else:
            matrix = ((matrix + matrix.T) * 0.5).tocsr()
        checks = validate_psd(matrix)
        record.update(collection_member=member, original_shape=original_shape,
                      original_nnz=original_nnz, derived_n=int(matrix.shape[0]),
                      derived_nnz=int(matrix.nnz), selected_vertices_sha256=vertex_hash, **checks)
        protocol_ok = matrix.shape[0] <= 120_000 and matrix.nnz <= 1_000_000
        record["protocol_limits_pass"] = protocol_ok
        if not checks["psd_pass"] or checks["symmetry_residual_max"] > 1e-10 or not protocol_ok:
            raise ValueError("derived matrix failed PSD, symmetry, or protocol checks")
        output = save_mother(matrix, spec, record)
        record["mother_file"] = str(output.relative_to(ROOT))
        record["status"] = "validated"
    except (OSError, ValueError, tarfile.TarError) as exc:
        record["status"] = "rejected"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        record["replacement_rule"] = spec.get("replacement")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("public", "private", "all"), default="public")
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()
    selected = [item for item in MATRICES if args.split in ("all", item["split"])]
    records = [process(item, args.fetch_only) for item in selected]
    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "collection": "SuiteSparse Matrix Collection, live archive endpoints",
        "conversion": "SPD symmetric average; adjacency -> D-|A+A^T|/2",
        "bfs_rule": "canonical first-unused root; FIFO BFS; stop at limit; induced edges; discovery ordering",
        "matrices": records,
    }
    output = ROOT / f"suitesparse_{args.split}_provenance.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(output)
    for row in records:
        print(row["group"], row["name"], row["status"], row.get("archive_bytes"),
              row.get("derived_n"), row.get("derived_nnz"))
    return 1 if any(row["status"] in ("unavailable", "rejected") for row in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
