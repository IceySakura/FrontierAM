#!/usr/bin/env python3
"""Build six FAMEXP2 sessions per mother, manifests, controls, and truth plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parent
PRECISIONS = {"p4": (1e-4, 1e-7), "p7": (1e-7, 1e-10), "p10": (1e-10, 1e-13)}
ETA_STRATA = (1e-2, 1.0, 10.0, 1e2, 1e3, 1e4, 3e4, 1e5)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def load_mother(path: Path) -> tuple[sp.csr_matrix, dict]:
    with np.load(path, allow_pickle=False) as archive:
        shape = tuple(int(value) for value in archive["shape"])
        matrix = sp.csr_matrix((archive["data"].astype(np.float64),
                                archive["indices"].astype(np.int64),
                                archive["indptr"].astype(np.int64)), shape=shape)
        metadata = json.loads(str(archive["metadata"].item())) if "metadata" in archive else {}
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix, metadata


def infinity_norm(matrix: sp.csr_matrix) -> float:
    return float(np.max(np.asarray(abs(matrix).sum(axis=1)).ravel(), initial=0.0))


def eta_bucket(eta: float) -> str:
    if eta <= 10:
        return "low_eta_le_10"
    if eta <= 1e3:
        return "mid_eta_10_to_1e3"
    return "high_eta_gt_1e3"


def spectral_filter(matrix: sp.csr_matrix, vector: np.ndarray, high: bool) -> np.ndarray:
    norm = infinity_norm(matrix)
    if norm == 0:
        return vector
    if high:
        result = matrix @ vector / norm
    else:
        result = vector.copy()
        for _ in range(4):
            result -= 0.8 * (matrix @ result) / norm
    maximum = np.max(np.abs(result), initial=0.0)
    return result / max(1.0, maximum)


def category_vector(category: str, matrix: sp.csr_matrix,
                    rng: np.random.Generator) -> np.ndarray:
    n = matrix.shape[0]
    if category == "rademacher":
        return rng.choice(np.array([-1.0, 1.0]), size=n)
    if category == "impulse":
        vector = np.zeros(n)
        count = min(max(1, n // 10_000), 8)
        positions = rng.choice(n, size=count, replace=False)
        vector[positions] = rng.choice(np.array([-1.0, 1.0]), size=count)
        return vector
    base = rng.uniform(-1, 1, size=n)
    if category == "smooth":
        return spectral_filter(matrix, base, high=False)
    if category == "high_frequency":
        return spectral_filter(matrix, base, high=True)
    if category == "near_zero":
        return base * 1e-14
    raise ValueError(category)


def make_rhs(matrix: sp.csr_matrix, requests: int, seed: int,
             precision: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    if requests == 16:
        categories = (["rademacher"] * 6 + ["impulse"] * 4 + ["smooth"] * 3
                      + ["high_frequency"] * 2 + ["near_zero"])
    else:
        categories = [{"p4": "rademacher", "p7": "impulse",
                       "p10": "high_frequency"}[precision]]
    b0 = np.empty((requests, matrix.shape[0]), dtype=np.float64)
    b1 = np.empty_like(b0)
    for index, category in enumerate(categories):
        # Independent draws ensure one-shot and reuse request zero never coincide.
        b0[index] = category_vector(category, matrix, rng)
        b1_category = categories[(index + 5) % len(categories)] if requests == 16 else "smooth"
        b1[index] = category_vector(b1_category, matrix, rng)
        # Scored requests deliberately exercise both degenerate coefficients.
        if requests == 16 and index in (0, 5, 10):
            b0[index] = 0
        if requests == 16 and index in (3, 7, 14):
            b1[index] = 0
        if requests == 1 and precision == "p4":
            b0[index] = 0
        if requests == 1 and precision == "p7":
            b1[index] = 0
    return b0, b1, list(categories)


def write_case(path: Path, matrix: sp.csr_matrix, b0: np.ndarray, b1: np.ndarray,
               y_ref: np.ndarray, t: float, rtol: float, atol: float) -> None:
    matrix = matrix.tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    np.savez_compressed(
        path,
        data=np.asarray(matrix.data, dtype="<f8"),
        indices=np.asarray(matrix.indices, dtype="<u4"),
        indptr=np.asarray(matrix.indptr, dtype="<u8"),
        shape=np.asarray(matrix.shape, dtype="<u8"),
        b0=np.asarray(b0, dtype="<f8"),
        b1=np.asarray(b1, dtype="<f8"),
        y_ref=np.asarray(y_ref, dtype="<f8"),
        t=np.asarray(t, dtype="<f8"),
        rtol=np.asarray(rtol, dtype="<f8"),
        atol=np.asarray(atol, dtype="<f8"),
    )


def family_from(path: Path, metadata: dict) -> str:
    if metadata.get("family"):
        return str(metadata["family"])
    name = path.stem.lower()
    if "bcsstk" in name or "thermal" in name:
        return "standard_fem"
    return "standard_graph"


def choose_mothers(split: str, smoke: bool) -> list[Path]:
    directory = ROOT / "mothers" / split
    paths = sorted(directory.glob("*.npz"))
    if split == "public":
        standard = [path for path in paths if path.name.startswith("standard_")]
        synthetic = [path for path in paths if path.name.startswith("syn_")]
        if smoke:
            # Eight independent mothers while retaining all six synthetic families.
            preferred_standard = [
                path for token in ("bcsstk17", "fe_4elt2")
                for path in standard if token in path.name
            ]
            if len(preferred_standard) >= 2:
                names = {"syn_diffusion3d", "syn_geometric", "syn_heavytail",
                         "syn_signed", "syn_logspectrum", "syn_diffusion2d"}
                chosen = preferred_standard[:2] + [
                    path for path in synthetic if path.stem in names
                ]
            else:
                # Network-unavailable smoke fallback: independent frozen synthetic
                # mothers only. Provenance still records every failed standard fetch.
                chosen = synthetic
            if len(chosen) != 8:
                raise ValueError(f"public smoke needs exactly 8 mothers, found {len(chosen)}")
            return chosen
    return paths


def mother_seed(seeds: dict, name: str, purpose: str) -> int:
    text = f"{seeds['rhs_master']}:{name}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def build_scored(split: str, smoke: bool, output: Path) -> tuple[list[dict], list[dict]]:
    seeds = json.loads((ROOT / f"{split}_seeds.json").read_text())
    mothers = choose_mothers(split, smoke)
    if not mothers:
        raise ValueError(f"no {split} mothers; run fetch_suitesparse.py and synth.py first")
    cases, costs = [], []
    for mother_index, mother_path in enumerate(mothers):
        matrix, metadata = load_mother(mother_path)
        mother = mother_path.stem
        family = family_from(mother_path, metadata)
        target_eta = ETA_STRATA[mother_index % len(ETA_STRATA)]
        norm = infinity_norm(matrix)
        if norm <= 0:
            raise ValueError(f"scored mother {mother} has zero norm")
        # Positive scaling is fixed once per mother. It is only needed when t would
        # exceed the protocol limit, and preserves PSD and sparsity.
        scale = max(1.0, target_eta / (1e4 * norm))
        scaled = (matrix * scale).tocsr()
        scaled_norm = norm * scale
        t = target_eta / scaled_norm
        if t > 1e4 * (1 + 1e-12):
            raise AssertionError("time scaling failed")
        for precision, (rtol, atol) in PRECISIONS.items():
            for scenario, requests in (("one_shot", 1), ("reuse", 16)):
                suffix = "k1" if requests == 1 else "k16"
                case_id = f"{mother}_{precision}_{suffix}"
                seed = mother_seed(seeds, mother, f"{precision}:{scenario}")
                b0, b1, categories = make_rhs(scaled, requests, seed, precision)
                # Finite zeros intentionally reserve the exact format while truth is pending.
                y_ref = np.zeros_like(b0)
                path = output / f"{case_id}.npz"
                write_case(path, scaled, b0, b1, y_ref, t, rtol, atol)
                tolerance = atol + rtol
                degree = max(8, math.ceil(2 * math.sqrt(target_eta * math.log(2 / tolerance))))
                flops = 2.0 * scaled.nnz * degree * requests
                estimate_seconds = flops / 5e7
                case = {
                    "id": case_id, "file": path.name, "family": family, "mother": mother,
                    "scenario": scenario, "precision": precision, "role": "scored",
                    "n": scaled.shape[0], "nnz": scaled.nnz, "request_count": requests,
                    "t": t, "eta_bucket": eta_bucket(target_eta),
                    "eta_target": target_eta, "operator_scale": scale,
                    "rhs_categories": categories, "truth_status": "placeholder_zero_unqualified",
                    "sha256": file_sha256(path),
                }
                cases.append(case)
                costs.append({
                    "id": case_id, "n": scaled.shape[0], "nnz": scaled.nnz,
                    "eta": target_eta, "precision": precision, "requests": requests,
                    "chebyshev_degree_estimate": degree,
                    "sparse_flop_estimate": flops,
                    "primary_cpu_minutes_estimate": estimate_seconds / 60,
                })
    return cases, costs


def dense_truth(matrix: np.ndarray, b0: np.ndarray, b1: np.ndarray, t: float) -> np.ndarray:
    eigenvalues, vectors = scipy.linalg.eigh(matrix)
    exponential = np.exp(-t * eigenvalues)
    phi = np.full_like(eigenvalues, t)
    nonzero = np.abs(eigenvalues) > 1e-14
    phi[nonzero] = -np.expm1(-t * eigenvalues[nonzero]) / eigenvalues[nonzero]
    return ((vectors * exponential) @ (vectors.T @ b0.T)
            + (vectors * phi) @ (vectors.T @ b1.T)).T


def build_controls(output: Path) -> list[dict]:
    specs = [
        ("control_zero", np.zeros((3, 3)), 2.0, 1),
        ("control_time_zero", np.array([[2.0, -1.0], [-1.0, 2.0]]), 0.0, 1),
        ("control_diagonal", np.diag([0.0, 4.0]), 0.5, 1),
        ("control_positive_offdiag", np.array([[2.0, 1.0], [1.0, 2.0]]), 0.5, 1),
        ("control_reuse", np.array([[1.0, -1.0], [-1.0, 1.0]]), 0.5, 16),
        ("control_scalar_stiff", np.array([[1e8]]), 1e-3, 1),
        ("control_b0_zero", np.diag([0.0, 1.0, 10.0]), 3.0, 1),
        ("control_b1_zero", np.diag([0.0, 1.0, 10.0]), 3.0, 1),
    ]
    cases = []
    for index, (case_id, dense, t, requests) in enumerate(specs):
        matrix = sp.csr_matrix(dense)
        rng = np.random.default_rng(9000 + index)
        b0 = rng.uniform(-1, 1, size=(requests, dense.shape[0]))
        b1 = rng.uniform(-1, 1, size=(requests, dense.shape[0]))
        if case_id in ("control_zero", "control_b0_zero"):
            b0[:] = 0
        if case_id == "control_b1_zero":
            b1[:] = 0
        y_ref = dense_truth(dense, b0, b1, t)
        path = output / f"{case_id}.npz"
        write_case(path, matrix, b0, b1, y_ref, t, 1e-10, 1e-13)
        cases.append({
            "id": case_id, "file": path.name, "family": "analytic", "mother": case_id,
            "scenario": "reuse" if requests == 16 else "one_shot", "precision": "p10",
            "role": "control", "n": dense.shape[0], "nnz": matrix.nnz,
            "request_count": requests, "t": t,
            "eta_bucket": eta_bucket(t * infinity_norm(matrix)),
            "truth_status": "analytic_dense_eigh_qualified", "sha256": file_sha256(path),
        })
    return cases


def write_truth_plan(costs: list[dict], path: Path) -> None:
    total_minutes = sum(row["primary_cpu_minutes_estimate"] for row in costs)
    hardest = max(costs, key=lambda row: row["primary_cpu_minutes_estimate"])
    text = f"""# Dataset-v2 truth computation plan

Scored files currently contain finite zero placeholders in `y_ref`; they are
explicitly **unqualified** and must not be used for correctness or scoring.
Analytic controls already contain qualified dense-eigendecomposition truth.

## Primary computation

Use 256-bit MPFR/Arb coefficients for both `exp(-t x)` and
`(1-exp(-t x))/x` on `[0, lambda_upper]`, where `lambda_upper` is the smaller
of the infinity norm and a safety-inflated verified Lanczos upper bound.
Choose the Chebyshev tail below `1e-3` of the p10 acceptance threshold and use
one compensated binary64/long-double Clenshaw recurrence for both functions.
Reuse sessions use block SpMM over all 16 right-hand sides.

## Independent audit

Audit every case with restarted high-accuracy Lanczos/KIOPS at `1e-3*rtol`.
For `eta >= 1e4`, add rational Krylov or scaling-and-Taylor augmented action.
Audit controls and sampled principal subproblems with arbitrary precision dense
eigendecomposition. Require infinity-norm disagreement no greater than
`0.02*(atol + rtol*||y||inf)`; otherwise classify the case as evaluator error.

## Cost model

The generator estimates degree as
`ceil(2*sqrt(eta*log(2/(atol+rtol))))`, sparse work as
`2*nnz*degree*requests`, and CPU time at a conservative 50 million sparse
floating operations/second. This is a planning estimate, not a timing result.

* Cases estimated: {len(costs)}
* Primary total: {total_minutes:.1f} CPU-minutes ({total_minutes / 60:.2f} CPU-hours)
* Hardest case: `{hardest['id']}`, degree {hardest['chebyshev_degree_estimate']},
  {hardest['primary_cpu_minutes_estimate']:.1f} CPU-minutes
* Independent audit budget: 1–2x primary for ordinary cases and 2–3x for
  `eta >= 1e4`.

Machine-readable per-case estimates are in `truth_costs.json`.
"""
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("public", "private"), default="public")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.split == "private" and args.smoke:
        parser.error("private smoke generation is intentionally disabled")
    output = ROOT / "cases" / (f"{args.split}_smoke" if args.smoke else args.split)
    output.mkdir(parents=True, exist_ok=True)
    scored, costs = build_scored(args.split, args.smoke, output)
    controls = build_controls(output)
    manifest = {
        "version": 2,
        "scope": f"{args.split} {'reduced-size smoke' if args.smoke else 'dataset-v2'}",
        "truth_qualified": False,
        "truth_note": "scored y_ref arrays are placeholders; controls are qualified",
        "cases": scored + controls,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    (output / "truth_costs.json").write_text(json.dumps(costs, indent=2) + "\n")
    write_truth_plan(costs, ROOT / "truth_plan.md")
    print(f"{manifest_path}: {len(scored)} scored + {len(controls)} controls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
