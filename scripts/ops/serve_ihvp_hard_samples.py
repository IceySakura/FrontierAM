#!/usr/bin/env python3
"""Host-only v3 sample oracle launcher; diagnostics, not formal enforcement.

Run from the complete curator repository. Only its dedicated socket directory
may be mounted in a candidate development environment. Teacher arrays, seed
plans, the full data root and this host source tree must stay outside it.

This reuses the existing sample relay. It does not enforce the formal 256 MiB
worker limit or provide an adversarial security boundary. The v3 formal driver
remains a draft; starting this manual service does not activate it.
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import json
from pathlib import Path
import shutil
import signal
import stat
import sys
import uuid

REPO = Path(__file__).resolve().parents[2]
VERSION = REPO / "tasks/ihvp-sequence/versions/v3"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/seal"))
import prepare_ihvp_hard_v3 as prepare
from arena.operator_drivers import ihvp_oracle


def require(condition, message):
    if not condition:
        raise ValueError(message)


def manifest(root, relative):
    path = prepare.nonsymlink_path(root, root / relative)
    require(path.is_file(), "required public/sample manifest is missing")
    return path, json.loads(path.read_text())


def sample_cases(data_root):
    """Read public/sample data only; never open private manifests or teachers."""
    root = prepare.source_root(data_root)
    public_path, public = manifest(root, "public/manifest.json")
    sample_path, samples = manifest(root, "samples/manifest.json")
    require(public.get("version") == "v3" and public.get("split") == "public",
            "data root must be the merged v3 public/sample layout")
    require(samples.get("version") == "v3" and samples.get("split") == "samples",
            "sample manifest must belong to v3")
    public_entries = public.get("cases", [])
    require(len(public_entries) == 30 and
            sum(entry.get("group") == "legacy" for entry in public_entries) == 15 and
            sum(entry.get("group") == "hard" for entry in public_entries) == 15,
            "merged public manifest must contain 15 legacy plus 15 hard cases")
    public_by_id = {entry["id"]: entry for entry in public_entries}
    require(len(public_by_id) == 30, "merged public case identifiers are duplicated")
    hard = [entry for entry in public_entries if entry.get("group") == "hard"]
    expected_sources = {"sample_01": hard[0]["id"], "sample_02": hard[-1]["id"]}
    entries = samples.get("cases", [])
    require(len(entries) == 2 and {entry.get("id") for entry in entries} == set(expected_sources),
            "exactly sample_01 and sample_02 are required")

    # Override before the first _runtime call, in this fresh launcher process.
    ihvp_oracle.GRADER = VERSION / "grader"
    api, Case = ihvp_oracle._runtime()
    require(Path(api.__file__).resolve() == (VERSION / "grader/runtime/api.py").resolve(),
            "start the launcher as a fresh process; another runtime is already imported")
    cases = {}
    for entry in entries:
        name = entry["id"]
        source_id = entry.get("source_public_feedback_id")
        require(source_id == expected_sources[name],
                "sample selection is not the first/last added public feedback case")
        source = public_by_id[source_id]
        require(source.get("group") == "hard" and source.get("kind") == "hard_factored_v3" and
                entry.get("kind") == "hard_factored_v3",
                "full samples must be hard public feedback teachers")
        require(entry.get("sha256") == source.get("sha256") == entry.get("source_teacher_sha256"),
                "sample and public teacher fingerprints disagree")
        for record in (source, entry):
            require((int(record.get("dimension", 0)), int(record.get("steps", 0))) == (16384, 50),
                    "full samples must declare d=16384,T=50")
        relative = Path(entry["path"])
        source_relative = Path(source["path"])
        require(not relative.is_absolute() and ".." not in relative.parts and
                not source_relative.is_absolute() and ".." not in source_relative.parts,
                "teacher paths must be contained relative paths")
        path = prepare.nonsymlink_path(root, sample_path.parent / relative)
        source_path = prepare.nonsymlink_path(root, public_path.parent / source_relative)
        for teacher in (path, source_path):
            require(teacher.is_file() and prepare.sha256(teacher) == source["sha256"],
                    "sample/public teacher body is missing or changed")
        case = Case(path)
        require(case.meta.get("schema") == prepare.HARD_SCHEMA and case.meta.get("split") == "public",
                "sample teacher must have public metadata; private teachers are forbidden")
        require((int(case.dimension), int(case.steps)) == (16384, 50),
                "sample teacher dimensions differ from the protocol")
        cases[name] = path
    smoke = VERSION / "samples/smoke.json"
    small = Case(smoke)
    require(small.meta.get("kind") == "dense_smoke" and
            (int(small.dimension), int(small.steps)) == (4, 3),
            "the tracked v3 smoke fixture is invalid")
    cases["smoke"] = smoke
    return root, cases


def serve(args):
    data_root, cases = sample_cases(args.data_root)
    requested = args.socket_dir.expanduser().absolute()
    require(not requested.exists() and not requested.is_symlink(),
            "socket directory already exists; choose a fresh short path")
    require(requested.parent.is_dir(), "socket directory parent must already exist")
    directory = requested.resolve()
    require(not directory.is_relative_to(data_root) and not data_root.is_relative_to(directory),
            "socket directory and teacher data root must be separate")
    require(max(len(os.fsencode(directory / f"{name}.sock")) for name in cases) < 100,
            "Unix socket path is too long; use a fresh directory under /tmp or /run")
    service = ihvp_oracle.SampleService(directory, cases)
    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, lambda *_: service.stopped.set())
    token = uuid.uuid4().hex
    created = False
    identity = None
    marker = directory / ".ihvp-hard-samples-owner"
    try:
        directory.mkdir(mode=0o755)  # exclusive creation; no existing directory is reused
        created = True
        directory.chmod(0o755)
        details = directory.stat(follow_symlinks=False)
        identity = (details.st_dev, details.st_ino)
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(token)
        service.start()
        print(json.dumps({"status": "ready", "samples": ["sample_01", "sample_02", "smoke"],
                          "scope": "manual sample diagnostics only; no formal resource/security acceptance",
                          "socket_dir": str(directory)}), flush=True)
        service.stopped.wait()
    finally:
        service.stop()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        # Never remove a pre-existing/replaced/symlink directory or someone
        # else's files. Only this invocation's exact disposable directory is owned.
        if created and directory.exists() and not directory.is_symlink():
            details = directory.stat(follow_symlinks=False)
            if (stat.S_ISDIR(details.st_mode) and (details.st_dev, details.st_ino) == identity
                    and marker.is_file() and not marker.is_symlink() and marker.read_text() == token):
                shutil.rmtree(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="host-only merged v3 data root containing public/ and samples/")
    parser.add_argument("--socket-dir", type=Path, required=True,
                        help="fresh short socket-only directory; must not already exist")
    args = parser.parse_args()
    try:
        serve(args)
    except Exception as error:
        print(f"sample service failed ({type(error).__name__}): {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
