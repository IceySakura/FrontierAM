#!/usr/bin/env python3
"""Package the local 15+15 hard supplement and its curator evidence.

This never generates data, launches a solver, or writes private material to Git.
It requires completed independent audits and a completed public three-method
benchmark. The resulting ZIP contains private answers and seed plans: keep it
local and never mount its data/curator trees in a candidate workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TASK = REPO / "tasks/ihvp-sequence"
VERSION = TASK / "versions/v3"
sys.path.insert(0, str(HERE))
import prepare_ihvp_hard_v3 as prepare


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_file(root, path):
    path = prepare.nonsymlink_path(root, path)
    require(path.is_file(), "required regular file is missing")
    return path


def select_audit(path, split, manifest_sha, records):
    report = read_json(path)
    require(report.get("status") == "PASS" and not report.get("errors"),
            "independent mathematical report is not PASS")
    splits = [value for value in report.get("splits", [])
              if value.get("split") == split and value.get("manifest_sha256") == manifest_sha]
    require(len(splits) == 1, "independent report does not bind this split manifest")
    audited = splits[0].get("cases", [])
    expected = {record["source_case_id"]: record["sha256"] for record, _ in records}
    actual = {record.get("case"): record.get("data_sha256") for record in audited}
    require(len(audited) == 15 and actual == expected,
            "independent report does not cover all 15 matching teacher bodies")
    require(all(int(record.get("dimension", 0)) == 16384 and
                int(record.get("steps", 0)) == 50 for record in audited),
            "independent report dimensions are incorrect")
    require(report.get("model_sha256") == prepare.sha256(VERSION / "curator/model.py"),
            "independent audit model source is unavailable or changed")
    require(report.get("audit_script_sha256") ==
            prepare.sha256(REPO / "scripts/analyze/ihvp_hard_diagnostics.py"),
            "independent audit script source is unavailable or changed")
    return report


def validate_benchmark(directory, public_sha, records):
    directory = prepare.source_root(directory)
    aggregate = read_json(safe_file(directory, directory / "aggregate.json"))
    header = read_json(safe_file(directory, directory / "run_manifest.json"))
    require(header.get("manifest_sha256") == public_sha,
            "public benchmark manifest does not match the packaged public teachers")
    require(header.get("selected_case_count") == 15 and header.get("full_case_count") == 15,
            "public benchmark is a partial or different-sized run")
    require(aggregate.get("completed_cases") == 15 and
            aggregate.get("selected_cases") == 15 and
            aggregate.get("full_manifest_cases") == 15 and
            aggregate.get("complete_selected_manifest") is True and
            aggregate.get("complete_full_manifest") is True,
            "public benchmark is incomplete")
    require(aggregate.get("run_fingerprint") == header.get("run_fingerprint"),
            "public benchmark aggregate fingerprint differs from its header")
    require(header.get("harness_sha256") ==
            prepare.sha256(REPO / "scripts/references/ihvp_hard_benchmark.py"),
            "public benchmark harness differs from the archived source")
    runtime = header.get("runtime_identity", {})
    require(header.get("runtime_fingerprint") ==
            hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest(),
            "public benchmark runtime identity fingerprint is inconsistent")
    runtime_sources = runtime.get("source_sha256", {})
    require(bool(runtime_sources), "public benchmark runtime source fingerprints are missing")
    for relative, expected_sha in runtime_sources.items():
        require(prepare.sha256(safe_file(REPO, REPO / relative)) == expected_sha,
                "public benchmark runtime source differs from the archived current source")
    controls = {key: value for key, value in header.items()
                if key not in ("run_fingerprint", "runtime_identity", "created_utc", "scope")}
    require(header.get("run_fingerprint") == hashlib.sha256(json.dumps(
        controls, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "public benchmark control fingerprint is inconsistent")
    methods = header.get("methods", {})
    require(len(methods) >= 3 and set(methods) == set(aggregate.get("methods", {})),
            "at least three completed public methods are required")
    expected = {record["source_case_id"]: record["sha256"] for record, _ in records}
    snapshots = []
    for method, configuration in methods.items():
        summary = aggregate["methods"][method]
        require(summary.get("status") == "ok" and summary.get("sequences") == 15 and
                summary.get("valid_sequences") == 15 and
                summary.get("resource_compliant_sequences") == 15,
                "a public baseline has invalid or missing sequence results")
        require(math.isfinite(float(summary.get("mean_energy_loss", float("nan")))) and
                float(summary["mean_energy_loss"]) >= 0 and
                float(summary.get("max_wall_attempt_seconds", float("inf"))) <= 30 and
                float(summary.get("max_peak_rss_mib", float("inf"))) <= 256,
                "public baseline resource/loss aggregate is invalid")
        seen = {}
        losses, walls, peaks = [], [], []
        for index in range(1, 16):
            case_path = directory / method / f"case_{index:03d}" / "result.json"
            row = read_json(safe_file(directory, case_path))
            require(row.get("run_fingerprint") == header["run_fingerprint"] and
                    row.get("runtime_fingerprint") == header.get("runtime_fingerprint") and
                    row.get("method") == method and
                    row.get("source_options_fingerprint") == configuration.get("source_options_fingerprint"),
                    "public per-case provenance does not match its run")
            require(row.get("status") == "ok" and row.get("valid") is True and
                    row.get("resource_compliant") is True and
                    math.isfinite(float(row.get("mean_energy_loss", float("nan")))) and
                    0 <= float(row["mean_energy_loss"]) and
                    float(row.get("wall_seconds", float("inf"))) <= 30 and
                    float(row.get("peak_rss_mib", float("inf"))) <= 256,
                    "a public per-case result is invalid or exceeds resources")
            steps = row.get("records", [])
            require(row.get("dimension") == 16384 and row.get("steps") == 50 and
                    row.get("completed_steps") == 50 and len(steps) == 50 and
                    not row.get("solver_errors") and
                    all(value.get("step") == tick and value.get("completed") is True and
                        value.get("error") is None for tick, value in enumerate(steps)),
                    "a public per-case record does not contain 50 completed outputs")
            step_losses = [float(value.get("energy_loss", float("nan"))) for value in steps]
            require(all(math.isfinite(value) and value >= 0 for value in step_losses) and
                    math.isclose(math.fsum(step_losses) / 50, float(row["mean_energy_loss"]),
                                 rel_tol=1e-10, abs_tol=1e-12),
                    "public step losses disagree with their case aggregate")
            require(0 < float(row["wall_seconds"]) and 0 < float(row["peak_rss_mib"]),
                    "public case time/RSS must be positive")
            losses.append(float(row["mean_energy_loss"]))
            walls.append(float(row["wall_seconds"]))
            peaks.append(float(row["peak_rss_mib"]))
            require(row.get("case_id") not in seen, "duplicate public benchmark case")
            seen[row.get("case_id")] = row.get("teacher_sha256")
        require(seen == expected, "public benchmark case identities differ from the data")
        recomputed = {"mean_energy_loss": math.fsum(losses) / 15,
                      "mean_energy_loss_finite_diagnostic": math.fsum(losses) / 15,
                      "mean_wall_attempt_seconds": math.fsum(walls) / 15,
                      "max_wall_attempt_seconds": max(walls),
                      "mean_peak_rss_mib": math.fsum(peaks) / 15,
                      "max_peak_rss_mib": max(peaks)}
        for field, expected_value in recomputed.items():
            require(field in summary and math.isclose(float(summary[field]), expected_value,
                                                     rel_tol=1e-10, abs_tol=1e-12),
                    "public aggregate disagrees with its independently combined case records")
        source_root = REPO / configuration["source"]
        for relative, expected_sha in configuration.get("source_sha256", {}).items():
            original = safe_file(REPO, REPO / relative)
            require(prepare.sha256(original) == expected_sha,
                    "reference source changed after the public run")
            source_relative = original.relative_to(source_root)
            snapshot = safe_file(directory, directory / "source_snapshots" / method / source_relative)
            require(prepare.sha256(snapshot) == expected_sha,
                    "actual public worker source snapshot is missing or changed")
            snapshots.append(snapshot)
    return header, aggregate, snapshots


def source_files():
    roots = [VERSION, TASK / "references/candidates/galerkin_cg",
             TASK / "references/r1",
             TASK / "versions/v1/grader/calibration_references/cold_cg"]
    paths = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            # Seed plans and teacher arrays are collected only from the local
            # curator input trees below, never from a model-visible source tree.
            if path.suffix.lower() not in (".py", ".md", ".json", ".lock", ".txt") and not path.name.startswith("Dockerfile"):
                continue
            if path.suffix == ".json" and any(token in path.name.lower() for token in ("plan", "freeze")):
                continue
            paths.append(safe_file(REPO, path))
    for relative in ("tasks/ihvp-sequence/analysis/HARD_V3.md",
                     "scripts/seal/prepare_ihvp_hard_v3.py",
                     "scripts/seal/package_ihvp_hard_v3.py",
                     "scripts/references/ihvp_hard_benchmark.py",
                     "scripts/analyze/ihvp_hard_diagnostics.py"):
        paths.append(safe_file(REPO, REPO / relative))
    return sorted(set(paths))


def package(args):
    additions = prepare.locate_manifests(args.addition_root)
    validation = prepare.source_root(args.validation_root)
    benchmark = prepare.source_root(args.public_benchmark or validation / "public-benchmark")
    require(benchmark.is_relative_to(validation),
            "public benchmark evidence must be inside the validation root")
    freeze_path = safe_file(validation, args.freeze_receipt or validation / "design_freeze.json")
    records = {split: prepare.load_source(path, "hard", split)
               for split, path in additions.items()}
    hashes = {split: {record["sha256"] for record, _ in entries}
              for split, entries in records.items()}
    require(all(len(values) == 15 for values in hashes.values()) and
            not hashes["public"] & hashes["private"],
            "hard supplement must have 15 distinct teachers in each disjoint split")
    manifests = {split: read_json(path) for split, path in additions.items()}
    manifest_shas = {split: prepare.sha256(path) for split, path in additions.items()}
    plans = {}
    design = None
    historical = args.historical_generator or validation / "source-before-private-gate/generate.py"
    generators = [VERSION / "curator/generate.py"]
    if historical.exists():
        generators.append(safe_file(validation, historical))
    generator_by_sha = {prepare.sha256(path): path for path in generators}
    selected_generators = {}
    for split, manifest in manifests.items():
        plan_path = safe_file(additions[split].parent, additions[split].parent / "plan.json")
        plan = read_json(plan_path)
        require(plan.get("split") == split and plan.get("dimension") == 16384 and
                plan.get("steps") == 50 and len(plan.get("cases", [])) == 15 and
                prepare.sha256(plan_path) == manifest.get("plan_sha256"),
                "seed plan does not match its generated split manifest")
        require({entry.get("id") for entry in plan["cases"]} ==
                {record["source_case_id"] for record, _ in records[split]} and
                all(entry.get("split") == split for entry in plan["cases"]),
                "seed plan case identities/roles do not match its manifest")
        require(plan.get("design_sha256") == manifest.get("design_sha256"),
                "seed plan and data design fingerprints differ")
        design = design or plan["design_sha256"]
        require(design == plan["design_sha256"], "public/private design fingerprints differ")
        require(manifest.get("model_sha256") == prepare.sha256(VERSION / "curator/model.py"),
                "teacher model source differs from the preserved data")
        generator_sha = manifest.get("generator_sha256")
        require(generator_sha in generator_by_sha,
                "exact generator source for a split is unavailable")
        if split == "private":
            require(generator_sha == prepare.sha256(VERSION / "curator/generate.py"),
                    "private data were not produced by the current gated generator")
        selected_generators[split] = generator_by_sha[generator_sha]
        plans[split] = (plan, plan_path)
    seeds = {split: {entry["seed"] for entry in plan[0]["cases"]}
             for split, plan in plans.items()}
    require(all(len(values) == 15 for values in seeds.values()) and
            not seeds["public"] & seeds["private"],
            "public/private seed plans overlap")
    freeze = read_json(freeze_path)
    require(freeze.get("status") == "FROZEN" and freeze.get("design_sha256") == design,
            "private generation design receipt is not frozen to the packaged design")
    for field, path in (("model_sha256", VERSION / "curator/model.py"),
                        ("generator_sha256", VERSION / "curator/generate.py")):
        require(field not in freeze or freeze[field] == prepare.sha256(path),
                "freeze receipt source fingerprint differs from the gated source")
    reports = {split: safe_file(validation, validation / f"{split}-structure-independent.json")
               for split in ("public", "private")}
    for split, report in reports.items():
        select_audit(report, split, manifest_shas[split], records[split])
    header, aggregate, worker_snapshots = validate_benchmark(benchmark, manifest_shas["public"], records["public"])
    for key in ("public_run_fingerprint", "public_benchmark_run_fingerprint"):
        require(key not in freeze or freeze[key] == header["run_fingerprint"],
                "freeze receipt refers to a different public qualification run")

    out = prepare.destination_root(args.out)
    companion = out.with_suffix(out.suffix + ".sha256")
    require(out.suffix.lower() == ".zip", "--out must end in .zip")
    require(not out.exists() and not companion.exists(), "refusing to overwrite an existing ZIP or checksum")
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = {}

    def add(path, name):
        path = Path(path)
        require(name not in entries and not Path(name).is_absolute() and ".." not in Path(name).parts,
                "duplicate or unsafe archive path")
        entries[name] = (path, prepare.sha256(path), path.stat().st_size)

    for split, values in records.items():
        add(additions[split], f"data/{split}/manifest.json")
        original_entries = {value["id"]: value for value in manifests[split]["cases"]}
        for record, teacher in values:
            add(teacher, f"data/{split}/" + original_entries[record["source_case_id"]]["path"])
        add(plans[split][1], f"curator/plans/{split}/plan.json")
        add(selected_generators[split], f"curator/generation-sources/{split}/generate.py")
        add(VERSION / "curator/model.py", f"curator/generation-sources/{split}/model.py")
    add(freeze_path, "curator/design_freeze.json")
    for path in source_files():
        add(path, "curator/source/" + path.relative_to(REPO).as_posix())
    for path in worker_snapshots:
        add(path, "curator/public-worker-snapshots/" + path.relative_to(benchmark / "source_snapshots").as_posix())
    # JSON diagnostics preserve per-case audits and raw benchmark records. Large
    # prediction NPYs and all other teacher/body formats are deliberately omitted.
    for path in sorted(validation.rglob("*.json")):
        if "source_snapshots" in path.parts or path.name == "plan.json":
            continue
        path = safe_file(validation, path)
        add(path, "validation/" + path.relative_to(validation).as_posix())

    readme = f"""# IHVP hard v3: local 15+15 addition

This private curator archive contains only the new 15 public-feedback and 15
private-final sequences. Existing v1 teachers are not duplicated. Treat the entire
archive as private: it contains answers, operator factors and actual seed plans.
Keep archive/extraction permissions private (for example, use `umask 077`). Never
mount `data/` or `curator/` in a candidate/model workspace and never Git-stage them.

Source branch: {args.branch_url}
The source snapshot is provenance, not a claim that a Docker/amd64 release or a
universal SOTA baseline was qualified. Independent mathematics and a complete
public baseline run are included; final platform publication gates remain separate.

From the repository branch, import these additions together with the original pack:

```sh
python scripts/seal/prepare_ihvp_hard_v3.py \\
  --addition-root <unziproot>/data --legacy-bundle <original-ihvp-sequence-v1-root>
```

The importer preserves original teacher bytes and emits 30 public plus 30 private
cases under ignored local storage. Data SHA256s, exact plans, the freeze receipt,
current sources, the exact public pre-gate generator source when applicable, and
independent/benchmark JSON audits are included. Benchmark prediction arrays are
omitted. `FILE_SHA256.json` covers every other archive entry; the adjacent ZIP
checksum file covers the ZIP itself. Public and private generation may use gate-
equivalent generator versions; their exact source SHA256s are preserved separately.
""".encode()
    file_manifest = {name: {"sha256": sha, "bytes": size}
                     for name, (_, sha, size) in entries.items()}
    file_manifest["README.md"] = {"sha256": hashlib.sha256(readme).hexdigest(), "bytes": len(readme)}
    metadata = {"schema": "frontieram/ihvp-hard-v3-private-archive-1",
                "public_sequences": 15, "private_sequences": 15,
                "design_sha256": design, "data_manifest_sha256": manifest_shas,
                "public_run_fingerprint": header["run_fingerprint"],
                "independent_reports": "PASS, bound to all 30 exact teacher SHA256s",
                "public_methods": list(header["methods"]),
                "qualification": "data/evidence bundle only; platform/formal release remains separate",
                "files": dict(sorted(file_manifest.items()))}
    manifest_bytes = (json.dumps(metadata, ensure_ascii=False, sort_keys=True,
                                 indent=2, allow_nan=False) + "\n").encode()

    def info(name):
        value = zipfile.ZipInfo(name)
        value.compress_type = zipfile.ZIP_STORED
        value.create_system = 3
        value.external_attr = 0o100600 << 16
        return value

    descriptor = os.open(out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    companion_created = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for name, (path, expected_sha, expected_size) in sorted(entries.items()):
                    actual = hashlib.sha256()
                    size = 0
                    with path.open("rb") as incoming, archive.open(info(name), "w", force_zip64=True) as outgoing:
                        for block in iter(lambda: incoming.read(1024**2), b""):
                            actual.update(block)
                            size += len(block)
                            outgoing.write(block)
                    require(actual.hexdigest() == expected_sha and size == expected_size,
                            "a source file changed while the ZIP was written")
                archive.writestr(info("README.md"), readme)
                archive.writestr(info("FILE_SHA256.json"), manifest_bytes)
        checksum = prepare.sha256(out)
        companion_descriptor = os.open(companion, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        companion_created = True
        with os.fdopen(companion_descriptor, "w") as stream:
            stream.write(f"{checksum}  {out.name}\n")
        out.chmod(0o600)
        companion.chmod(0o600)
    except BaseException:
        # Only this invocation's newly created output can be removed.
        if out.exists():
            out.unlink()
        if companion_created and companion.exists():
            companion.unlink()
        raise
    return {"status": "PACKAGED", "public_sequences": 15, "private_sequences": 15,
            "files": len(entries) + 2, "bytes": out.stat().st_size, "zip_sha256": checksum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addition-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path)
    parser.add_argument("--public-benchmark", type=Path)
    parser.add_argument("--historical-generator", type=Path)
    parser.add_argument("--branch-url", default="https://github.com/IceySakura/FrontierAM/tree/feat/ihvp-hard-sequences-v3")
    args = parser.parse_args()
    try:
        result = package(args)
    except Exception as error:
        # No private path, case ID, plan seed or body is sent to stdout/stderr.
        print(f"packaging failed ({type(error).__name__}); local prerequisites are incomplete", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
