#!/usr/bin/env python3
"""Curator-only generation of independent hard feedback/private supplements.

Public feedback first:
  python -B generate.py --split public --out var/raw/ihvp-hard-v4/public

Private generation additionally requires a locally frozen design receipt. Plans,
seeds and all teacher bodies are written only under ignored storage, never to
tracked task paths. No candidate or baseline benchmark is run here.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import time
import numpy as np
from model import HardCase, SCHEMA

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
LEVELS = (("level_1", 1e5), ("level_2", 1e6), ("level_3", 1e7))
FAMILIES = (
    {"name": "log_uniform_rotating", "rho": .90,
     "diagonal_amplitude": .12, "operator_noise": .025, "rotation_noise": .012,
     "mean_fraction": 0.0},
    {"name": "low_end_weighted", "rho": .85,
     "diagonal_amplitude": .16, "operator_noise": .04, "rotation_noise": .015,
     "mean_fraction": 0.0},
    {"name": "many_bands_coupled", "rho": .95,
     "diagonal_amplitude": .10, "operator_noise": .03, "rotation_noise": .010,
     "mean_fraction": 0.0},
    {"name": "fullrank_mean_reversion", "rho": .94,
     "diagonal_amplitude": .25, "operator_noise": .075, "rotation_noise": .020,
     "mean_fraction": .35},
    {"name": "announced_phase_changes", "rho": .97,
     "diagonal_amplitude": .18, "operator_noise": .035, "rotation_noise": .015,
     "mean_fraction": 0.0},
)
ARCHITECTURES = ("routed_only", "routed_dct2", "routed_dct3")
DESIGN = {"schema": "frontieram/ihvp-routed-design-4", "public_count": 15, "private_count": 30,
          "architectures": ARCHITECTURES, "routing_depth_ranges": [[10,16],[4,7],[3,6]],
          "routing": "Independent uniform perfect matching per layer and case; no fixed bit pairing",
          "mixture": "Each family has one of each architecture across its three levels, independently permuted per split/replicate",
          "phase_change_rule": "Uniform integer cuts in [14,22] and [30,39] for phase-change family", "dimension": 16384, "steps": 50,
          "levels": list(LEVELS), "families": FAMILIES, "block_eigenvalue_bounds": [.8, 1.2],
          "temporal_rotation_layers": 2, "spectral_pair_bucket_count": 32,
          "state_path": "Full-dimensional stationary AR innovations; exogenous and not an optimizer trajectory",
          "selection": "Five predeclared families x three levels; independent frozen seed per cell; no per-case selection by solver outcome"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for value in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(value)
    return h.hexdigest()


def design_hash():
    return hashlib.sha256(json.dumps(DESIGN, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def local_only(path):
    path = Path(path).resolve()
    if path.is_relative_to(REPO):
        result = subprocess.run(["git", "check-ignore", "-q", "--", str(path)], cwd=REPO, check=False)
        if result.returncode:
            raise ValueError("Plans/teachers must be under ignored var/ or outside the checkout")
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _unit_normal(rng, n):
    return rng.standard_normal(n) / np.sqrt(n)


def _spectrum(rng, family, n, condition):
    position = np.linspace(0.0, 1.0, n)
    if family == "low_end_weighted":
        position = position ** 1.8
    elif family == "many_bands_coupled":
        band_count = min(64, n)
        band = np.minimum(band_count - 1, np.arange(n) * band_count // n)
        position = (band + rng.uniform(.06, .94, n)) / band_count
        position.sort()
        position = (position - position[0]) / (position[-1] - position[0])
    return np.exp(position * np.log(condition))


def _spectral_pairing(rng, spectrum):
    n = len(spectrum)
    bins = min(32, n // 4)
    chunks = np.array_split(np.argsort(spectrum), bins)
    if any(len(chunk) % 2 for chunk in chunks):
        raise ValueError("Power-of-two dimensions with even spectral buckets required")
    return np.concatenate([rng.permutation(chunk) for chunk in chunks]).astype(np.int32)


def generate_arrays(configuration, dimension=16384, steps=50):
    start = time.perf_counter()
    if dimension < 32 or dimension & (dimension - 1):
        raise ValueError("Use a power-of-two dimension >=32")
    settings = dict(configuration["settings"])
    rng = np.random.default_rng(configuration["seed"])
    n, half = dimension, dimension // 2
    family, architecture = settings["name"], settings["architecture"]
    base_diagonal = _spectrum(rng, family, n, configuration["condition_target"])
    stages = {"routed_only": 0, "routed_dct2": 2, "routed_dct3": 3}[architecture]
    low, high = {"routed_only": (10,16), "routed_dct2": (4,7), "routed_dct3": (3,6)}[architecture]
    depth = int(rng.integers(low, high+1))
    base_angles = rng.uniform(-np.pi, np.pi, (depth, half))
    base_routing = np.stack([rng.permutation(n) for _ in range(depth)]).astype(np.int32)
    base_permutations = np.array([rng.permutation(n) for _ in range(stages)], dtype=np.int32).reshape(stages,n)
    base_signs = rng.choice([-1,1], (stages,n)).astype(np.int8)
    post_signs = rng.choice([-1,1], n).astype(np.int8)
    temporal_permutations = np.stack([_spectral_pairing(rng, base_diagonal) for _ in range(2)])
    block_permutation = rng.permutation(n).astype(np.int32)
    diagonal = np.empty((steps, n))
    coefficients = np.empty((steps, half, 3))
    temporal_angles = np.empty((steps, 2, half))
    whitened = np.empty((steps, n))
    phases = np.zeros(steps, dtype=np.int16)
    if family == "announced_phase_changes":
        first_cut, second_cut = int(rng.integers(14,23)), int(rng.integers(30,40))
        phases[first_cut:second_cut], phases[second_cut:] = 1, 2
    diagonal_state = rng.standard_normal(n) * .2
    gain_base = rng.standard_normal((2, half))
    gain_state = np.zeros_like(gain_base)
    block_angle_base = rng.uniform(-np.pi, np.pi, half)
    block_angle_state = np.zeros(half)
    temporal_base = rng.uniform(-np.pi, np.pi, (2, half))
    temporal_state = np.zeros_like(temporal_base)
    moving = _unit_normal(rng, n)
    mean = _unit_normal(rng, n)
    mean /= np.linalg.norm(mean)
    rho, mean_fraction = settings["rho"], settings["mean_fraction"]
    for t in range(steps):
        phase_change = t > 0 and phases[t] != phases[t - 1]
        if phase_change:
            moving = _unit_normal(rng, n)
            diagonal_state += rng.standard_normal(n) * .7
            gain_state += rng.standard_normal((2, half)) * .8
            temporal_state += rng.standard_normal((2, half)) * .35
        elif t:
            moving = rho * moving + np.sqrt(1 - rho * rho) * _unit_normal(rng, n)
        whitened[t] = mean_fraction * mean + np.sqrt(1 - mean_fraction ** 2) * moving
        diagonal_state = .98 * diagonal_state + settings["operator_noise"] * rng.standard_normal(n)
        gain_state = .985 * gain_state + settings["operator_noise"] * rng.standard_normal((2, half))
        block_angle_state = .98 * block_angle_state + .015 * rng.standard_normal(half)
        temporal_state = .985 * temporal_state + settings["rotation_noise"] * rng.standard_normal((2, half))
        diagonal[t] = base_diagonal * np.exp(settings["diagonal_amplitude"] * np.tanh(diagonal_state))
        eigenvalues = .8 + .4 / (1 + np.exp(-(gain_base + gain_state)))
        theta = block_angle_base + block_angle_state
        cosine, sine = np.cos(theta), np.sin(theta)
        first, second = eigenvalues
        coefficients[t, :, 0] = first * cosine ** 2 + second * sine ** 2
        coefficients[t, :, 1] = (first - second) * cosine * sine
        coefficients[t, :, 2] = first * sine ** 2 + second * cosine ** 2
        temporal_angles[t] = temporal_base + temporal_state
    amplitude = settings["diagonal_amplitude"]
    metadata = {"schema": SCHEMA, "id": configuration["id"], "split": configuration["split"],
                "family": family, "profile": configuration["profile"], "architecture": architecture, "routing_depth": depth,
                "dimension": n, "steps": steps, "condition_target": configuration["condition_target"],
                "seed": int(configuration["seed"]), "settings": settings,
                "design_sha256": design_hash(), "mu": .64 * float(np.exp(-amplitude)),
                "L": 1.44 * configuration["condition_target"] * float(np.exp(amplitude)),
                "H_definition": "H_t=Q_t sqrt(D_t) B_t^2 sqrt(D_t) Q_t.T",
                "reference_definition": "b_t=Q_t sqrt(D_t) B_t y_t; x_star=Q_t D_t^(-1/2) B_t^(-1)y_t",
                "reference_energy": "x_star.T H_t x_star=norm(y_t)^2",
                "Q_definition": "Q_t=Q_base R_1(t) R_0(t); base architecture as specified; temporal pairings are spectral-local",
                "block_definition": "Permutation-consecutive 2x2 SPD blocks, coefficient order [a,b,c] for [[a,b],[b,c]], eigenvalues in[.8,1.2]",
                "bounds_proof": "D_min>=exp(-amplitude),D_max<=kappa*exp(amplitude),singular(B)in[.8,1.2],Q orthogonal",
                "phase_boundaries_one_based": [int(i)+1 for i in np.flatnonzero(np.r_[True, phases[1:] != phases[:-1]])],
                "coordinate_transport": "Identity ambient coordinates; no factor or whitening information is exposed to candidates",
                "innovation_expected_energy_fraction": (1 - mean_fraction ** 2) * (1 - rho ** 2),
                "trajectory_kind": "Exogenous high-dimensional stochastic operator/RHS path; not a genuine outer-optimization trajectory",
                "hardness_claim": "No provable unlearnability or per-case solver superiority is claimed; qualification is measured separately on public data"}
    arrays = {"base_angles": base_angles, "base_routing": base_routing, "post_signs": post_signs, "base_permutations": base_permutations,
              "base_signs": base_signs, "temporal_permutations": temporal_permutations,
              "temporal_angles": temporal_angles, "block_permutation": block_permutation,
              "block_coefficients": coefficients, "diagonal": diagonal,
              "whitened_rhs": whitened, "phase": phases,
              "b": np.zeros((steps, n)), "z_star": np.zeros((steps, n)),
              "metadata_json": np.array(json.dumps(metadata, ensure_ascii=False, allow_nan=False))}
    case = HardCase(arrays, validate_teacher=False)
    maximum_residual, maximum_energy_identity_error, maximum_self_loss = 0.0, 0.0, 0.0
    previous_losses, rhs_cosines = [], []
    for t in range(steps):
        case._prepare(t)
        y = whitened[t]
        arrays["b"][t] = case.apply_q(t, case._sqrt_diagonal * case.apply_b(t, y))
        arrays["z_star"][t] = case.apply_q(t, case.apply_b(t, y, inverse=True) / case._sqrt_diagonal)
        residual = np.linalg.norm(case.hvp(t, arrays["z_star"][t]) - arrays["b"][t]) / np.linalg.norm(arrays["b"][t])
        energy_error = abs(float(arrays["z_star"][t] @ arrays["b"][t]) / float(y @ y) - 1)
        self_loss = case.score(t, arrays["z_star"][t])["energy_loss"]
        maximum_residual = max(maximum_residual, float(residual))
        maximum_energy_identity_error = max(maximum_energy_identity_error, float(energy_error))
        maximum_self_loss = max(maximum_self_loss, float(self_loss))
        if t:
            previous_losses.append(case.score(t, arrays["z_star"][t-1])["energy_loss"])
            rhs_cosines.append(float(arrays["b"][t] @ arrays["b"][t-1]) /
                               float(np.linalg.norm(arrays["b"][t]) * np.linalg.norm(arrays["b"][t-1])))
    if maximum_residual >= 1e-7 or maximum_energy_identity_error >= 1e-8 or maximum_self_loss >= 1e-14:
        raise ArithmeticError("Teacher reference verification failed")
    metadata["reference_validation"] = {"status": "PASS", "all_steps": steps,
        "maximum_relative_residual": maximum_residual,
        "maximum_relative_energy_identity_error": maximum_energy_identity_error,
        "maximum_reference_self_energy_loss": maximum_self_loss,
        "previous_exact_solution_current_energy_loss": previous_losses,
        "mean_previous_exact_solution_current_energy_loss": float(np.mean(previous_losses)) if previous_losses else None,
        "mean_adjacent_rhs_cosine": float(np.mean(rhs_cosines)) if rhs_cosines else None}
    metadata["generation_and_reference_kernel_seconds"] = time.perf_counter() - start
    arrays["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    return arrays, metadata


def frozen_plan(output, split, dimension, steps, freeze_file=None):
    # Revalidate the receipt even when resuming an already-created private plan.
    if split == "private":
        if freeze_file is None:
            raise ValueError("Private generation requires an explicit --design-freeze receipt after public qualification")
        frozen = json.loads(local_only(freeze_file).read_text())
        if frozen.get("status") != "FROZEN" or frozen.get("design_sha256") != design_hash():
            raise ValueError("Private design is not frozen to the current public-qualified parameters")
        if not frozen.get("public_qualification_sha256") or not frozen.get("benchmark_sha256"):
            raise ValueError("Private freeze must bind public qualification and frozen benchmark methods")
        for field, source in (("model_sha256", HERE / "model.py"),
                              ("generator_sha256", Path(__file__))):
            if frozen.get(field) != digest(source):
                raise ValueError(f"Private freeze receipt {field} does not match the current source")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "plan.json"
    if path.exists():
        plan = json.loads(path.read_text())
        if (plan["design_sha256"] != design_hash() or plan["split"] != split
                or plan["dimension"] != dimension or plan["steps"] != steps):
            raise ValueError("Existing plan belongs to a different design; retain it and choose a new ignored output")
        return plan
    cases, used = [], set()
    # Predeclared balanced mixture, independent of solver outcomes. The 30
    # private cases are two fresh replicates, never a retained public/legacy set.
    for replicate in range(1 if split == "public" else 2):
        for family in FAMILIES:
            architectures = list(ARCHITECTURES)
            secrets.SystemRandom().shuffle(architectures)
            for (profile, condition), architecture in zip(LEVELS, architectures):
                seed = secrets.randbits(64)
                while seed in used:
                    seed = secrets.randbits(64)
                used.add(seed)
                cases.append({"id": f"v4_{split}_{len(cases)+1:02d}", "split": split,
                              "settings": {**family, "architecture": architecture},
                              "replicate": replicate, "profile": profile,
                              "condition_target": condition, "seed": seed})
    plan = {"schema": "frontieram/ihvp-hard-local-plan-1", "split": split,
            "design": DESIGN, "design_sha256": design_hash(), "dimension": dimension,
            "steps": steps, "cases": cases, "private_seed_plan": True,
            "selection_policy": "Seeded once before any solver measurements; no case rejection by algorithm outcome"}
    write_json(path, plan)
    return plan


def dense_self_test():
    settings = []
    for index, architecture in enumerate(ARCHITECTURES):
        family = {**FAMILIES[index], "architecture": architecture}
        configuration = {"id": "audit", "split": "selftest", "settings": family,
                         "profile": "audit", "condition_target": 1e6, "seed": 77191 + index}
        arrays, metadata = generate_arrays(configuration, dimension=64, steps=4)
        case = HardCase(arrays)
        worst = 0.0
        for t in (0, 3):
            q = case.apply_q(t, np.eye(64))
            block = np.zeros((64, 64))
            permutation = arrays["block_permutation"]
            for j, (a, b, c) in enumerate(arrays["block_coefficients"][t]):
                i, k = permutation[2*j:2*j+2]
                block[i, i], block[i, k], block[k, i], block[k, k] = a, b, b, c
            sqrt_d = np.sqrt(arrays["diagonal"][t])
            h = q @ (sqrt_d[:, None] * (block @ block) * sqrt_d[None, :]) @ q.T
            h = (h + h.T) / 2
            chol = np.linalg.cholesky(h)
            reference = np.linalg.solve(chol.T, np.linalg.solve(chol, arrays["b"][t]))
            error = reference - arrays["z_star"][t]
            energy = float(error @ h @ error) / float(arrays["whitened_rhs"][t] @ arrays["whitened_rhs"][t])
            worst = max(worst, np.sqrt(max(0, energy)))
            np.testing.assert_allclose(q.T @ q, np.eye(64), atol=1e-12, rtol=0)
            np.testing.assert_allclose(case.hvp(t, np.eye(64)), h, atol=1e-8, rtol=1e-11)
            if worst >= 1e-7:
                raise ArithmeticError("Dense Cholesky reference cross-check failed")
        settings.append({"family": family["name"], "architecture": family["architecture"],
                         "maximum_dense_relative_energy_error": worst})
    print(json.dumps({"status": "PASS", "dimension": 64, "cases": settings}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("public", "private"), default="public")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dimension", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--design-freeze", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        dense_self_test()
        return
    output = local_only(args.out or REPO / "var" / "raw" / "ihvp-hard-v4" / args.split)
    plan = frozen_plan(output, args.split, args.dimension, args.steps, args.design_freeze)
    if (output / "manifest.json").exists():
        raise FileExistsError("The complete cohort already exists")
    records = []
    for configuration in plan["cases"]:
        path = output / "cases" / configuration["id"] / "teacher.npz"
        if path.exists():
            raise FileExistsError("Retain existing partial cohort and choose a new output directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays, metadata = generate_arrays(configuration, args.dimension, args.steps)
        np.savez_compressed(path, **arrays)
        record = {"id": configuration["id"], "kind": "hard_factored_v4",
                  "path": f"cases/{configuration['id']}/teacher.npz", "sha256": digest(path),
                  "bytes": path.stat().st_size, "family": metadata["family"],
                  "profile": metadata["profile"], "dimension": args.dimension, "steps": args.steps,
                  "mu": metadata["mu"], "L": metadata["L"], "architecture": metadata["architecture"],
                  "reference_validation": metadata["reference_validation"]}
        records.append(record)
        print(json.dumps({"id": record["id"], "status": "PASS", "family": record["family"],
                          "profile": record["profile"], "bytes": record["bytes"],
                          "reference_residual": metadata["reference_validation"]["maximum_relative_residual"]}), flush=True)
    manifest = {"schema": "frontieram/ihvp-sequence-cases-1", "split": args.split,
                "cases": records, "sequence_count": len(records), "design_sha256": design_hash(),
                "plan_sha256": digest(output / "plan.json"), "model_sha256": digest(HERE / "model.py"),
                "generator_sha256": digest(__file__), "status": "reference_validated_not_benchmark_qualified"}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": "REFERENCE_VALIDATED", "cases": len(records),
                      "split": args.split, "design_sha256": design_hash()}), flush=True)


if __name__ == "__main__":
    main()
