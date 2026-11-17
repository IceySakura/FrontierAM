"""Seal QP assets, calibrate on this host and prepare directly runnable Harbor tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from arena.qp_runtime import VERSION, reference_source, run_verifier


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_data(source: Path, target: Path) -> None:
    """Copy only manifest inputs, never construction answers or historical calibration."""
    manifest = json.loads((source / "manifest.json").read_text())
    rows = manifest["cases"]
    if len(rows) != 64 or len({r["id"] for r in rows}) != 64:
        raise ValueError(f"expected 64 unique cases in {source}")
    target.mkdir(parents=True)
    target.chmod(0o700)
    for row in rows:
        path = (source / row["path"]).resolve()
        if not path.is_relative_to(source.resolve()) or sha(path) != row["sha256"]:
            raise ValueError("input path/checksum mismatch")
        destination = target / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    shutil.copy2(source / "manifest.json", target / "manifest.json")


def package(destination: Path, data: Path, image: str) -> None:
    destination.mkdir(parents=True)
    environment = destination / "environment"
    environment.mkdir()
    (environment / "README.md").write_text(f"Prebuilt task image: {image}\nNo runtime build is required.\n")
    (destination / "instruction.md").write_text((VERSION / "statement.md").read_text())
    config = (VERSION / "task.toml").read_text().replace("frontieram-pdhcg-qp:amd64-v1", image)
    (destination / "task.toml").write_text(config)
    tests = destination / "tests"
    shutil.copytree(VERSION / "grader", tests, ignore=shutil.ignore_patterns("__pycache__", "tests"))
    shutil.copytree(data, tests / "data")
    reference_source(tests / "reference")
    tests.chmod(0o700)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-data", type=Path, default=ROOT / "var/raw/pdhcg-qp-v1/private")
    parser.add_argument("--image", default="frontieram-pdhcg-qp:amd64-v1")
    parser.add_argument("--build-image", action="store_true")
    parser.add_argument("--calibrate", action="store_true", help="measure each split in the verifier container")
    parser.add_argument("--cpuset", help="physical core used to calibrate the reference")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime-ops/pdhcg-qp-v1")
    args = parser.parse_args(argv)
    sealed = args.output.resolve()
    if sealed.exists():
        parser.error(f"{sealed} already exists; use a new --output to preserve the existing seal")
    if not (args.private_data / "manifest.json").is_file():
        parser.error("private manifest is missing; supply the evaluator-side input directory")
    if args.calibrate and not args.cpuset:
        parser.error("--calibrate requires --cpuset to bind the reference to an evaluation core")
    public_manifest = json.loads((VERSION / "data/public/manifest.json").read_text())
    private_manifest = json.loads((args.private_data / "manifest.json").read_text())
    if {r["sha256"] for r in public_manifest["cases"]} & {r["sha256"] for r in private_manifest["cases"]}:
        parser.error("public and private inputs must be disjoint")
    image_id = args.image
    if args.build_image:
        subprocess.run(["docker", "build", "--platform", "linux/amd64", "-t", args.image,
                        "-f", str(VERSION / "Dockerfile"), str(VERSION)], check=True)
        image_id = subprocess.check_output(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True).strip()
    sealed.mkdir(parents=True)
    for split, source in (("public", VERSION / "data/public"), ("private", args.private_data)):
        copy_data(source, sealed / "data" / split)
    # Use the exact public helper implementation in both the reference and dev image.
    reference = reference_source(sealed / "reference")
    with tarfile.open(sealed / "public.tar", "w") as archive:
        archive.add(VERSION / "statement.md", arcname="problem.md")
        archive.add(VERSION / "data/samples", arcname="public-dev", filter=lambda i: None if "__pycache__" in i.name else i)
        for name in ("__init__.py", "model.py", "check.py"):
            archive.add(VERSION / "grader/qpbench" / name, arcname=f"public-dev/io/qpbench/{name}")
    for split in ("public", "private"):
        data = sealed / "data" / split
        if args.calibrate:
            run_verifier(reference, data, sealed / "calibration" / split,
                         image=image_id, cpuset=args.cpuset, calibrate=True)
        package(sealed / "harbor" / f"pdhcg-qp-{split}", data, args.image)
    lock = {"name": "pdhcg-qp-v1", "spec_sha256": sha(VERSION / "task.json"),
            "public_tar_sha256": sha(sealed / "public.tar"), "image": {"id": image_id, "tag": args.image},
            "calibrated": args.calibrate, "calibration_cpuset": args.cpuset,
            "data": {s: sha(sealed / "data" / s / "manifest.json") for s in ("public", "private")}}
    (sealed / "lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"seal": str(sealed), "calibrated": args.calibrate,
                      "harbor_tasks": str(sealed / "harbor")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
