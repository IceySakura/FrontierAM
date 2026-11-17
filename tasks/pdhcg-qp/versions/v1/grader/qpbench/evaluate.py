"""Trusted evaluation and calibration; only the current input reaches the worker."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile

from .check import check_solution
from .model import load_output, load_problem
from .process import FILE_LIMIT, WORKER_UID, run
from .scoring import aggregate_cases


def read_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    cases = manifest["cases"]
    ids = [row["id"] for row in cases]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("manifest must contain unique, nonempty cases")
    for row in cases:
        source = (path.parent / row["path"]).resolve()
        if not source.is_relative_to(path.parent.resolve()):
            raise ValueError("input escapes the data directory")
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"input checksum mismatch: {row['id']}")
    return manifest


def copy_submission(source: Path, destination: Path, uid: int) -> None:
    """Reject links and special files before copying the artifact into a writable tree."""
    if not source.is_dir() or source.is_symlink():
        raise ValueError("submission must be a directory")
    paths = list(source.rglob("*"))
    if len(paths) > 10000:
        raise ValueError("too many submission files")
    total = 0
    for path in paths:
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError("submission contains a link or special file")
        total += path.stat().st_size if path.is_file() else 0
    if total > 64 * 1024 * 1024:
        raise ValueError("submission exceeds 64 MiB")
    shutil.copytree(source, destination)
    for path in [destination, *destination.rglob("*")]:
        os.chown(path, uid, uid)


def checked_output(path: Path, problem) -> dict:
    # Open without following links or blocking on FIFOs, then take an immutable snapshot.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > FILE_LIMIT:
            raise ValueError("result must be a regular file of at most 4 MiB")
        payload = stream.read(FILE_LIMIT + 1)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if sum(item.file_size for item in archive.infolist()) > FILE_LIMIT:
            raise ValueError("expanded result exceeds 4 MiB")
    result = load_output(io.BytesIO(payload), expected_n=problem.n, expected_m=problem.m)
    return {**check_solution(problem, result), "status": result["metadata"].get("status")}


def evaluate(source: Path, manifest_path: Path, limits: dict[str, float], *, uid=WORKER_UID) -> dict:
    manifest = read_manifest(manifest_path)
    ids = [row["id"] for row in manifest["cases"]]
    if set(limits) != set(ids) or any(not 1 <= t <= 60 for t in limits.values()):
        raise ValueError("calibration must cover exactly this manifest, with limits in [1,60]")
    rows = {row["id"]: {"case_id": row["id"], "group": row["family"], "repeats": []}
            for row in manifest["cases"]}
    with tempfile.TemporaryDirectory(prefix="qp-verifier-") as directory:
        root = Path(directory)
        # The parent is traversable, while trusted data and logs remain elsewhere, root-only.
        root.chmod(0o755)
        candidate = root / "submission"
        try:
            copy_submission(source, candidate, uid)
        except ValueError as exc:
            return {"valid": True, "p": 0.0, "g": None, "cases": [], "groups": {},
                    "build": {"outcome": "algorithm_failure", "reason": str(exc)}}
        entry = candidate / "run.sh"
        if not entry.is_file() or not os.access(entry, os.X_OK):
            build = {"outcome": "algorithm_failure", "reason": "missing_executable_run_sh"}
        elif (candidate / "build.sh").exists():
            build = run(["./build.sh"], candidate, 900, uid=uid)
        else:
            build = {"outcome": "completed", "reason": "no_build"}
        if build["outcome"] != "completed":
            return {"valid": True, "p": 0.0, "g": None, "cases": [], "groups": {}, "build": build}
        # Freeze build artifacts once, then give each solve a fresh writable copy.
        # Otherwise the second/third process could reuse a cached answer and make
        # the median measure disk reads rather than solving the problem.
        built = root / "built"
        candidate.rename(built)
        os.chown(built, 0, 0)
        built.chmod(0o700)
        for repeat in (1, 2, 3):
            for row in manifest["cases"]:
                # Root owns the input directory; the worker cannot replace its input.
                # Never expose manifest IDs, family labels or repeat numbers in paths.
                current = root / "current"
                current.mkdir(mode=0o755)
                working = current / "submission"
                try:
                    copy_submission(built, working, uid)
                except ValueError as exc:
                    return {"valid": True, "p": 0.0, "g": None, "cases": [], "groups": {},
                            "build": {"outcome": "algorithm_failure", "reason": f"invalid_build_artifacts:{exc}"}}
                input_path = current / "input.npz"
                shutil.copyfile(manifest_path.parent / row["path"], input_path)
                input_path.chmod(0o444)
                problem = load_problem(input_path)
                output_dir = current / "result"
                output_dir.mkdir()
                os.chown(output_dir, uid, uid)
                output_path = output_dir / "output.npz"
                process = run([str(working / "run.sh"), "--input", str(input_path), "--output", str(output_path)],
                              working, limits[row["id"]], uid=uid)
                record = {"repeat": repeat, "outcome": "algorithm_failure", "status": None,
                          "elapsed_seconds": process["elapsed_seconds"], "process": process}
                if process["outcome"] == "completed":
                    try:
                        check = checked_output(output_path, problem)
                        record.update(checker=check, status=check["status"], reason=check["reason"])
                        if check["passed"]:
                            record["outcome"] = "accepted"
                    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
                        record["reason"] = f"result_error:{type(exc).__name__}"
                else:
                    record["reason"] = process["reason"]
                rows[row["id"]]["repeats"].append(record)
                shutil.rmtree(current)
        summary = aggregate_cases(list(rows.values()), expected_case_ids=ids, time_limits=limits)
        return {**summary, "build": build, "raw_cases": list(rows.values())}
