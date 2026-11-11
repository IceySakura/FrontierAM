#!/usr/bin/env python3
"""Prepare IHVP Sequence v1 samples or locally received feedback/private data.

No private body, answer or seed is written to tracked task paths. This script
does not launch model attempts, baseline timing, Docker or target-platform checks.

Examples, from the repository root:
  python scripts/seal/prepare_ihvp_sequence.py --samples-only
  python scripts/seal/prepare_ihvp_sequence.py --source-bundle /private/ihvp-sequence-v1
  python scripts/seal/prepare_ihvp_sequence.py --image-context var/dist/ihvp-sequence-v1 \
      --wheelhouse /path/to/pre-staged-linux-amd64-wheels
"""
import argparse
import errno
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
VERSION = ROOT / "tasks" / "ihvp-sequence" / "versions" / "v1"
DEFAULT_DATA_ROOT = ROOT / "var" / "raw" / "ihvp-sequence-v1" / "data"
DEFAULT_SAMPLE_ROOT = ROOT / "var" / "outputs" / "ihvp-sequence-v1" / "samples"
SAMPLE_ASSETS = {"README.md", "recipes.json", "check.py", "support/api.py",
                 "support/worker.py", "support/model.py", "support/common.py"}


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_destination(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT):
        check = subprocess.run(["git", "check-ignore", "-q", "--", str(path)],
                               cwd=ROOT, check=False)
        if check.returncode != 0:
            raise ValueError("Data must be under git-ignored var/ or outside this checkout")
    return path


def write_once(path, payload):
    path = Path(path)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError("Existing data manifest differs; choose another ignored data root")
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(text, encoding="utf-8")


def write_derived(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_materialized_samples(output):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recipes = json.loads((VERSION / "data" / "samples" / "recipes.json").read_text())["samples"]
    expected_ids = {entry["id"] for entry in recipes}
    cases = manifest.get("cases", [])
    full = [entry for entry in cases if entry.get("kind") == "signed_dct"]
    smoke = [entry for entry in cases if entry.get("kind") == "dense_smoke"]
    if (len(cases) != 3 or len(full) != 2 or len(smoke) != 1
            or {entry["id"] for entry in full} != expected_ids):
        raise ValueError("Samples must contain exactly one smoke and two declared full-size sequences")
    recipe_by_id = {entry["id"]: entry for entry in recipes}
    for entry in cases:
        path = (output / entry["path"]).resolve()
        if not path.is_relative_to(output) or path.is_symlink() or not path.is_file():
            raise ValueError("Invalid materialized sample path")
        if digest_file(path) != entry["sha256"]:
            raise ValueError("Materialized sample bytes do not match their manifest")
        if entry in full:
            recipe = recipe_by_id[entry["id"]]
            if (entry.get("dimension") != 16384 or entry.get("steps") != 50
                    or entry.get("source_public_feedback_id") != recipe["source_public_feedback_id"]):
                raise ValueError("Full-size sample does not match the declared feedback-pool subset")
    tracked_smoke = VERSION / "data" / "samples" / "smoke.json"
    if digest_file(tracked_smoke) != smoke[0]["sha256"]:
        raise ValueError("Tracked smoke fixture changed; retain the old output and prepare a fresh sample directory")
    return manifest


def prepare_samples(output):
    output = local_destination(output)
    if not (output / "manifest.json").exists():
        subprocess.run([sys.executable, "-B", str(VERSION / "generation" / "generate.py"),
                        "--samples", "--out", str(output)], cwd=ROOT, check=True)
    manifest = verify_materialized_samples(output)
    source = VERSION / "data" / "samples"
    available = {path.relative_to(source).as_posix(): path
                 for path in source.rglob("*") if path.is_file()
                 and "__pycache__" not in path.parts and path.suffix != ".pyc"
                 and path.name != ".DS_Store"}
    missing = SAMPLE_ASSETS - set(available)
    if missing:
        raise FileNotFoundError("Standalone sample assets are incomplete: " + ", ".join(sorted(missing)))
    copied = []
    for relative, original in sorted(available.items()):
        if relative == "manifest.json" or relative.startswith("cases/"):
            raise ValueError("Tracked sample assets must not replace generated case bodies or manifest")
        if original.is_symlink():
            raise ValueError("Sample assets must be regular files")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or digest_file(destination) != digest_file(original):
            shutil.copy2(original, destination)
        copied.append({"path": relative, "sha256": digest_file(destination),
                       "bytes": destination.stat().st_size})
    # Generated case references remain unchanged; only standalone checker/docs
    # are refreshed. No full-size sequence is generated when the hashes match.
    verify_materialized_samples(output)
    write_derived(output / "sample_assets_manifest.json", {
        "schema": "frontieram/ihvp-sequence-sample-assets-1", "assets": copied,
        "data_manifest_sha256": digest_file(output / "manifest.json"),
        "case_count": len(manifest["cases"]), "full_size_sequence_count": 2,
        "standalone_layout": "check.py, support/, manifest.json and all case bodies are present locally",
    })
    subprocess.run([sys.executable, "-B", str(output / "check.py"),
                    "--verify-package", "--manifest", str(output / "manifest.json")],
                   cwd=output, check=True)
    return manifest


def prepare_image_context(output, samples, wheelhouse):
    output = local_destination(output)
    samples = local_destination(samples)
    verify_materialized_samples(samples)
    if output == samples or output.is_relative_to(samples) or samples.is_relative_to(output):
        raise ValueError("Image context and reusable samples must be separate directories")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new ignored image-context directory")
    wheelhouse = Path(wheelhouse).resolve()
    if not wheelhouse.is_dir():
        raise FileNotFoundError("Provide a pre-staged Linux amd64 wheelhouse directory")
    wheels = sorted(path for path in wheelhouse.glob("*.whl") if path.is_file())
    if not wheels:
        raise FileNotFoundError("No pre-staged wheels were found; this script does not download packages")
    requirements_path = VERSION / "requirements.lock"
    requirements = []
    for line in requirements_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if not match:
            raise ValueError("Expected exact pinned requirements for the offline context")
        requirements.append((match.group(1).lower().replace("-", "_"), match.group(2)))
    for distribution, version in requirements:
        matches = [path for path in wheels
                   if path.name.lower().startswith(distribution + "-" + version.lower() + "-")]
        if not matches:
            raise ValueError("Wheelhouse is missing a pinned package: " + distribution + "==" + version)
        if not any(path.stem.endswith("-any") or ("linux" in path.stem and "x86_64" in path.stem)
                   for path in matches):
            raise ValueError("Pinned wheel candidates do not name Linux x86_64 or pure-Python platforms")
    # Prepare a context only; no docker command, target installation or network
    # access is performed. Wheel ABI compatibility still needs the real build.
    output.mkdir(parents=True, exist_ok=True)
    for name in ("Dockerfile", "requirements.lock", "statement.md"):
        shutil.copy2(VERSION / name, output / name)
    shutil.copytree(samples, output / "data" / "samples",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    target_wheels = output / "wheelhouse"
    target_wheels.mkdir()
    wheel_records = []
    for source in wheels:
        destination = target_wheels / source.name
        try:
            destination.hardlink_to(source)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copy2(source, destination)
        wheel_records.append({"path": "wheelhouse/" + source.name,
                              "sha256": digest_file(destination),
                              "bytes": destination.stat().st_size})
    context_files = [{"path": path.relative_to(output).as_posix(),
                      "sha256": digest_file(path), "bytes": path.stat().st_size}
                     for path in sorted(output.rglob("*")) if path.is_file()]
    write_once(output / "context_manifest.json", {
        "schema": "frontieram/ihvp-sequence-offline-image-context-1",
        "status": "context_prepared_not_built", "cases": 3,
        "full_size_sequences": 2, "files": context_files, "wheels": wheel_records,
        "requirements_sha256": digest_file(requirements_path),
        "data_manifest_sha256": digest_file(samples / "manifest.json"),
        "validation": "Sample hashes/standalone verifier and pinned wheel names checked; no target Python ABI or package installation tested",
        "docker_build": "not_run", "amd64_runtime_acceptance": "pending",
    })
    return output


def read_source_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cases = manifest.get("cases", [])
    if len(cases) != 15:
        raise ValueError("Expected the supplied v1 split with 15 full-size sequences")
    ids = []
    for entry in cases:
        case_id = entry["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", case_id):
            raise ValueError("Unsafe case identifier")
        ids.append(case_id)
        if entry.get("dimension") != 16384 or entry.get("steps") != 50:
            raise ValueError("Supplied v1 cases must have dimension=16384 and steps=50")
        body = (path.parent / entry["path"]).resolve()
        if not body.is_relative_to(path.parent) or not body.is_file():
            raise ValueError("Teacher path escapes split or is unavailable")
        if digest_file(body) != entry["sha256"]:
            raise ValueError("Teacher bytes differ from supplied manifest")
    if len(set(ids)) != 15:
        raise ValueError("Duplicate case identifier")
    return manifest


def materialize_split(source_manifest_path, destination, role, mode):
    source_manifest_path = Path(source_manifest_path).resolve()
    source = read_source_manifest(source_manifest_path)
    destination = local_destination(destination)
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    entries = []
    for entry in source["cases"]:
        case_id = entry["id"]
        original = (source_manifest_path.parent / entry["path"]).resolve()
        target = destination / "cases" / case_id / "teacher.npz"
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if target.exists():
            if target.is_symlink() or digest_file(target) != entry["sha256"]:
                raise FileExistsError("Existing teacher file is not the requested frozen case")
        elif mode == "copy":
            shutil.copy2(original, target)
        else:
            try:
                target.hardlink_to(original)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copy2(original, target)
        # Do not publish private generation seeds, full paths or per-case answers
        # in command output. The teacher itself remains host-only ignored data.
        entries.append({"id": case_id, "kind": "signed_dct",
                        "path": f"cases/{case_id}/teacher.npz",
                        "sha256": entry["sha256"], "bytes": target.stat().st_size,
                        "dimension": 16384, "steps": 50,
                        "mu": entry["mu"], "L": entry["L"],
                        "family": entry["family"], "profile": entry["profile"]})
    manifest = {"schema": "frontieram/ihvp-sequence-cases-1", "split": role,
                "sequence_count": 15, "cases": entries,
                "source_manifest_sha256": digest_file(source_manifest_path),
                "data_identity": "Original supplied teacher bytes preserved; no regeneration or solver-based case selection",
                "target_platform_validation": "Pending real amd64/Docker acceptance and maintainer calibration"}
    write_once(destination / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--samples-only", action="store_true")
    parser.add_argument("--samples-out", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--image-context", type=Path,
                        help="Prepare an ignored offline Docker context; does not build an image")
    parser.add_argument("--wheelhouse", type=Path,
                        help="Existing Linux amd64 wheels for every pinned requirement; no download")
    args = parser.parse_args()
    if args.wheelhouse is not None and args.image_context is None:
        parser.error("--wheelhouse requires --image-context")
    if args.samples_only or args.image_context is not None:
        if args.source_bundle is not None:
            parser.error("Prepare samples/image context separately from source-bundle import")
        if args.image_context is not None and args.wheelhouse is None:
            parser.error("--image-context requires --wheelhouse")
        output = local_destination(args.samples_out)
        manifest = prepare_samples(output)
        if args.image_context is not None:
            prepare_image_context(args.image_context, output, args.wheelhouse)
            print(json.dumps({"status": "CONTEXT_PREPARED_NOT_BUILT", "cases": len(manifest["cases"]),
                              "full_size_sequences": 2, "target_runtime_validation": "PENDING"}), flush=True)
        else:
            print(json.dumps({"status": "SAMPLES_READY", "cases": len(manifest["cases"]),
                              "full_size_sequences": 2}), flush=True)
        return
    if args.source_bundle is None:
        parser.error("Provide --source-bundle or --samples-only")
    bundle = args.source_bundle.resolve()
    public_path = bundle / "public" / "public-dev" / "manifest.json"
    private_path = bundle / "curator" / "data" / "test15" / "manifest.json"
    public_source = read_source_manifest(public_path)
    private_source = read_source_manifest(private_path)
    if {case["sha256"] for case in public_source["cases"]} & {case["sha256"] for case in private_source["cases"]}:
        raise ValueError("Public and private teacher sets overlap")
    # Only the selected samples have public recipes. They must be an actual
    # subset of the received feedback pool, not independently invented examples.
    samples = json.loads((VERSION / "data" / "samples" / "recipes.json").read_text())["samples"]
    public_by_id = {case["id"]: case for case in public_source["cases"]}
    for sample in samples:
        original = public_by_id.get(sample["source_public_feedback_id"])
        if original is None or original["sha256"] != sample["source_teacher_sha256"]:
            raise ValueError("Sample recipe does not identify the supplied public feedback fixture")
    output = local_destination(args.data_root)
    public = materialize_split(public_path, output / "public", "public_feedback", args.mode)
    private = materialize_split(private_path, output / "private", "private_final", args.mode)
    seal = {"schema": "frontieram/ihvp-sequence-local-data-seal-1",
            "status": "data_staged_target_validation_pending", "public_cases": 15, "private_cases": 15,
            "public_manifest_sha256": digest_file(output / "public" / "manifest.json"),
            "private_manifest_sha256": digest_file(output / "private" / "manifest.json"),
            "samples_recipe_sha256": digest_file(VERSION / "data" / "samples" / "recipes.json"),
            "public_private_teacher_overlap": 0,
            "known_model_competition_results": "None are asserted by this import",
            "source_staging": "Local ignored storage only; no private data written to tracked task paths",
            "amd64_docker_acceptance": "pending", "reference_calibration_on_target": "pending"}
    write_once(output / "seal.json", seal)
    print(json.dumps({"status": "DATA_STAGED", "public_cases": len(public["cases"]),
                      "private_cases": len(private["cases"]),
                      "target_platform_validation": "PENDING"}), flush=True)


if __name__ == "__main__":
    main()
