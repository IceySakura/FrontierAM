#!/usr/bin/env python3
"""Materialize two public sample recipes or a newly frozen local private cohort.

Generated teachers and any private seed plan belong under ignored var/ (or
outside this checkout). No existing feedback/private seed plan is embedded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
import common
import model

HERE = Path(__file__).resolve().parent
VERSION = HERE.parent
REPO = HERE.parents[4]
SAMPLE_RECIPES = VERSION / "data" / "samples" / "recipes.json"
ARRAY_KEYS = ("spectrum", "V", "permutation", "sign_in", "sign_out", "control", "phase",
              "diagonal_features", "rank_features", "diagonal_weights", "lowrank_weights",
              "forcing_basis", "forcing_initial", "validation_latent",
              "validation_target_eigen", "b", "z_star")


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_digest(arrays):
    digest = hashlib.sha256()
    for key in ARRAY_KEYS:
        array = arrays[key]
        digest.update(key.encode() + b"\0")
        digest.update(array.dtype.str.encode() + b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode() + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def require_local_output(path):
    path = Path(path).resolve()
    if path.is_relative_to(REPO):
        check = subprocess.run(["git", "check-ignore", "-q", "--", str(path)],
                               cwd=REPO, check=False)
        if check.returncode != 0:
            raise ValueError("Generated data must be outside tracked paths; use var/ or an external directory")
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to overwrite existing data receipt: " + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def frozen_private_plan(path):
    if path.exists():
        plan = json.loads(path.read_text(encoding="utf-8"))
        if plan.get("schema") != "frontieram/ihvp-sequence-local-private-plan-1":
            raise ValueError("Unsupported local private plan")
        return plan
    used = {case["seed"] for case in json.loads(SAMPLE_RECIPES.read_text())["samples"]}
    cases = []
    for family in common.FAMILIES:
        for profile, condition in (("easy", 1e3), ("medium", 1e5), ("hard", 1e7)):
            seed = secrets.randbits(64)
            while seed in used:
                seed = secrets.randbits(64)
            used.add(seed)
            cases.append({"id": f"case_{len(cases)+1:02d}", "family": family,
                          "profile": profile, "dimension": 16384, "steps": 50,
                          "condition_target": condition, "seed": seed, "repeat": 1})
    plan = {"schema": "frontieram/ihvp-sequence-local-private-plan-1", "cases": cases,
            "seed_source": "secrets.randbits(64), sampled once and frozen locally",
            "private": True, "selection_policy": "Fixed 5 families x 3 condition scales; no solver outcomes used"}
    write_json(path, plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--samples", action="store_true")
    mode.add_argument("--private", action="store_true", help="New local cohort; does not reproduce the supplied private v1 split")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--private-plan", type=Path)
    parser.add_argument("--strict-source-match", action="store_true", help="Require exact sample array hashes; normally keep diagnostic across BLAS/FFT implementations")
    args = parser.parse_args()
    output = require_local_output(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or (output / "cases").exists():
        raise FileExistsError("Use a new output directory")
    if args.samples:
        config = json.loads(SAMPLE_RECIPES.read_text(encoding="utf-8"))
        plan = config["samples"]
        split = "samples"
    else:
        private_path = require_local_output(args.private_plan or output / "private-plan.json")
        plan = frozen_private_plan(private_path)["cases"]
        split = "private_local_generated"
    entries = []
    if args.samples:
        source = VERSION / "data" / "samples" / "smoke.json"
        shutil.copy2(source, output / "smoke.json")
        entries.append({"id": "smoke", "kind": "dense_smoke", "path": "smoke.json",
                        "dimension": 4, "steps": 3, "mu": 1.0, "L": 7.0,
                        "sha256": digest_file(source), "bytes": source.stat().st_size,
                        "scored": False})
    for configuration in plan:
        arrays, metadata = model.generate_case(configuration)
        fingerprint = logical_digest(arrays)
        expected = configuration.get("logical_arrays_sha256_on_source_environment")
        exact_match = fingerprint == expected if expected else None
        if args.strict_source_match and expected and not exact_match:
            raise ArithmeticError("Generated sample differs bitwise from the source-environment fixture")
        case_id = configuration["id"]
        directory = output / "cases" / case_id
        directory.mkdir(parents=True)
        destination = directory / "teacher.npz"
        np.savez_compressed(destination, **arrays)
        entries.append({"id": case_id, "kind": "signed_dct", "path": f"cases/{case_id}/teacher.npz",
                        "family": metadata["family"], "profile": metadata["profile"],
                        "dimension": metadata["dimension"], "steps": metadata["steps"],
                        "mu": metadata["mu"], "L": metadata["L"],
                        "sha256": digest_file(destination), "bytes": destination.stat().st_size,
                        "logical_arrays_sha256": fingerprint,
                        "source_environment_exact_array_match": exact_match,
                        "validation": metadata["validation"],
                        "source_public_feedback_id": configuration.get("source_public_feedback_id")})
        # Case identifiers and aggregate status only; never print private seeds.
        print(json.dumps({"id": case_id, "status": "PASS", "dimension": metadata["dimension"],
                          "steps": metadata["steps"]}), flush=True)
    manifest = {"schema": "frontieram/ihvp-sequence-cases-1", "split": split,
                "cases": entries, "sequence_count": len(entries),
                "full_size_sequence_count": len(plan), "generator_sha256": digest_file(__file__),
                "model_sha256": digest_file(HERE / "model.py"),
                "common_sha256": digest_file(HERE / "common.py"),
                "bitwise_portability": "Samples carry source-environment hashes as diagnostics; different BLAS/FFT implementations may differ in rounding. Analytic references and residual checks are recomputed.",
                "target_platform_validation": "Not asserted by generation; real amd64/Docker validation is separate."}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": "GENERATED", "split": split, "cases": len(entries),
                      "full_size_sequences": len(plan)}), flush=True)


if __name__ == "__main__":
    main()
