#!/usr/bin/env python3
"""Seal legacy 15+15 and added 15+15 IHVP teachers, without generation.

Only ignored var/ or an external destination is allowed. Source teacher bytes
are preserved. This tool does not run solvers, inspect model submissions, or
qualify a reference. Private inputs and their manifests remain local.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO / "var/raw/ihvp-sequence-v3/data"
LEGACY_SCHEMA = "coupled_quadratic_bilevel_ihvp_signed_dct_v1"
HARD_SCHEMA = "frontieram/ihvp-hard-factored-spd-1"
SCHEMAS = {"legacy": (LEGACY_SCHEMA, "signed_dct"),
           "hard": (HARD_SCHEMA, "hard_factored_v3")}
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,80}")
SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def nonsymlink_path(root, path):
    """Check lexical containment first, then all components inside this root."""
    root = Path(root).absolute()
    path = Path(path).absolute()
    if not path.is_relative_to(root):
        raise ValueError("path escapes its declared root")
    if root.is_symlink():
        raise ValueError("root must not be a symlink")
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlinks are not allowed inside a data root")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("resolved path escapes its declared root")
    return path.resolve()


def source_root(path):
    original = Path(path).expanduser().absolute()
    if original.is_symlink() or not original.is_dir():
        raise ValueError("source root must be a nonsymlink directory")
    return original.resolve()


def destination_root(path):
    original = Path(path).expanduser().absolute()
    if original.is_relative_to(REPO):
        nonsymlink_path(REPO, original)
    elif original.is_symlink():
        raise ValueError("destination root must not be a symlink")
    resolved = original.resolve()
    if resolved.is_relative_to(REPO) and not resolved.is_relative_to(REPO / "var"):
        raise ValueError("repository data destinations must be under ignored var/")
    if resolved in (Path("/"), Path.home().resolve()):
        raise ValueError("choose a dedicated data destination")
    return resolved


def secure_directory(root, path):
    path = nonsymlink_path(root, path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    # This changes only owned output directories, never source teachers or their
    # hard links. Root-only traversal protects all staged teacher factors.
    path.chmod(0o700)
    return path


def write_once(root, path, value):
    path = nonsymlink_path(root, path)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False) + "\n"
    if path.exists():
        if not path.is_file() or path.read_text() != text:
            raise FileExistsError("existing output differs; choose a fresh local data root")
    else:
        secure_directory(root, path.parent)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    return sha256(path)


def locate_manifests(root, original_bundle=False):
    root = source_root(root)
    layouts = ([{"public": "public/public-dev/manifest.json",
                 "private": "curator/data/test15/manifest.json"}]
               if original_bundle else
               [{"public": "public/manifest.json", "private": "private/manifest.json"},
                {"public": "data/public/manifest.json", "private": "data/private/manifest.json"}])
    candidates = []
    for layout in layouts:
        paths = {split: nonsymlink_path(root, root / relative)
                 for split, relative in layout.items()}
        if all(path.is_file() for path in paths.values()):
            candidates.append(paths)
    if len(candidates) != 1:
        raise ValueError("expected exactly one complete public/private manifest layout")
    return candidates[0]


def load_source(manifest_path, group, split):
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    legacy_roles = {"public": {"dev", "public", "public_feedback"},
                    "private": {"test", "private", "private_final"}}
    accepted_roles = {split} if group == "hard" else legacy_roles[split]
    if manifest.get("split") not in accepted_roles:
        raise ValueError("source manifest role does not match its requested split")
    if "role" in manifest and manifest["role"] not in legacy_roles[split]:
        raise ValueError("source manifest role field conflicts with its split")
    entries = manifest.get("cases", [])
    if len(entries) != 15:
        raise ValueError("each source split must contain exactly 15 sequences")
    expected_schema, expected_kind = SCHEMAS[group]
    ids = set()
    output_ids = set()
    records = []
    for entry in entries:
        identifier = str(entry["id"])
        if IDENTIFIER.fullmatch(identifier) is None or identifier in ids:
            raise ValueError("unsafe or duplicated source case identifier")
        ids.add(identifier)
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("teacher paths must be contained relative paths")
        path = nonsymlink_path(root, root / relative)
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError("teacher file is missing or empty")
        expected_sha = str(entry["sha256"])
        if SHA256.fullmatch(expected_sha) is None or sha256(path) != expected_sha:
            raise ValueError("source teacher SHA does not match its manifest")
        size = path.stat().st_size
        if "bytes" in entry and int(entry["bytes"]) != size:
            raise ValueError("source teacher byte size does not match its manifest")
        if (int(entry.get("dimension", 0)), int(entry.get("steps", 0))) != (16384, 50):
            raise ValueError("source manifest must declare d=16384,T=50")
        if entry.get("kind", expected_kind) != expected_kind:
            raise ValueError("source kind differs from its required teacher schema")
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("schema") != expected_schema:
                raise ValueError("source teacher schema differs from source group")
            if ((group == "hard" and metadata.get("split") != split) or
                    (group == "legacy" and "split" in metadata and
                     metadata["split"] not in legacy_roles[split])):
                raise ValueError("teacher metadata role does not match its requested split")
            if (int(metadata["dimension"]), int(metadata["steps"])) != (16384, 50):
                raise ValueError("source teacher dimensions are not d=16384,T=50")
            for key in ("b", "z_star"):
                values = archive[key]
                if values.shape != (50, 16384) or not np.isfinite(values).all():
                    raise ValueError("source RHS/reference dimensions or values are invalid")
        mu, bound = float(metadata["mu"]), float(metadata["L"])
        if not np.isfinite(mu) or not np.isfinite(bound) or not 0 < mu <= bound:
            raise ValueError("source teacher has invalid spectral bounds")
        if any(key in entry and not np.isclose(float(entry[key]), value, rtol=1e-12, atol=0)
               for key, value in (("mu", mu), ("L", bound))):
            raise ValueError("manifest and teacher spectral bounds differ")
        family = str(entry.get("family", metadata.get("family", "unspecified")))
        profile = str(entry.get("profile", metadata.get("profile", "unspecified")))
        output_id = identifier if identifier.startswith(group + "_") else group + "_" + identifier
        if output_id in output_ids:
            raise ValueError("source IDs collide after canonical prefixing")
        output_ids.add(output_id)
        record = {"id": output_id, "source_case_id": identifier,
                  "origin": "ihvp-sequence-v1" if group == "legacy" else "ihvp-hard-v3",
                  "group": group, "kind": expected_kind, "teacher_schema": expected_schema,
                  "family": family, "profile": profile,
                  "dimension": 16384, "steps": 50, "mu": mu, "L": bound,
                  "sha256": expected_sha, "bytes": size,
                  "path": f"cases/{output_id}/teacher.npz"}
        for key in ("architecture", "reference_validation"):
            if key in entry:
                record[key] = entry[key]
        records.append((record, path))
    return records


def transfer(root, source, destination, expected_sha, expected_bytes, mode):
    destination = nonsymlink_path(root, destination)
    secure_directory(root, destination.parent)
    if destination.exists():
        if not destination.is_file():
            raise ValueError("existing teacher destination is not a regular file")
    elif mode == "copy":
        # Exclusive creation avoids replacing any prior local body.
        with Path(source).open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024**2)
    else:
        try:
            destination.hardlink_to(source)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            with Path(source).open("rb") as incoming, destination.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024**2)
    if destination.stat().st_size != expected_bytes or sha256(destination) != expected_sha:
        raise ValueError("staged teacher differs from the frozen source bytes")


def prepare(args):
    additions = locate_manifests(args.addition_root)
    legacy = locate_manifests(args.legacy_bundle, original_bundle=True) if args.legacy_bundle else locate_manifests(args.legacy_data_root)
    sources = {"legacy": legacy, "hard": additions}
    loaded = {split: [] for split in ("public", "private")}
    source_digests = {}
    for group in ("legacy", "hard"):
        for split in ("public", "private"):
            manifest = sources[group][split]
            loaded[split].extend(load_source(manifest, group, split))
            source_digests[group + "_" + split] = sha256(manifest)
    # Check before writing any teacher body; additions must be actual new cases.
    sets = {split: [record["sha256"] for record, _ in values]
            for split, values in loaded.items()}
    if any(len(values) != 30 or len(set(values)) != 30 for values in sets.values()):
        raise ValueError("each combined split must have 30 distinct teachers")
    if set(sets["public"]) & set(sets["private"]):
        raise ValueError("public/private teachers overlap")
    output = destination_root(args.out)
    for paths in sources.values():
        for manifest in paths.values():
            source = manifest.parent
            if output.is_relative_to(source) or source.is_relative_to(output):
                raise ValueError("destination and source split directories must be separate")
    secure_directory(output, output)
    manifest_digests = {}
    for split in ("public", "private"):
        split_root = secure_directory(output, output / split)
        records = []
        for record, source in loaded[split]:
            transfer(output, source, split_root / record["path"], record["sha256"], record["bytes"], args.mode)
            records.append(record)
        manifest = {"schema": "frontieram/ihvp-sequence-cases-1", "version": "v3",
                    "split": split, "role": "public_feedback" if split == "public" else "private_final",
                    "sequence_count": 30, "group_counts": {"legacy": 15, "hard": 15},
                    "source_manifest_sha256": {group: source_digests[group + "_" + split]
                                               for group in ("legacy", "hard")},
                    "teacher_access": "host evaluator only; never a candidate input or mount",
                    "cases": records}
        manifest_digests[split] = write_once(output, split_root / "manifest.json", manifest)

    # Samples deliberately expose only two selected feedback cases to the host
    # sample evaluator; no private case or frozen seed plan is copied here.
    hard_public = [(record, source) for record, source in loaded["public"] if record["group"] == "hard"]
    sample_root = secure_directory(output, output / "samples")
    sample_records = []
    for index, (original, source) in enumerate((hard_public[0], hard_public[-1]), 1):
        record = dict(original)
        record.update(id=f"sample_{index:02d}", source_public_feedback_id=original["id"],
                      source_teacher_sha256=original["sha256"],
                      path=f"cases/sample_{index:02d}/teacher.npz")
        transfer(output, source, sample_root / record["path"], record["sha256"], record["bytes"], args.mode)
        sample_records.append(record)
    manifest_digests["samples"] = write_once(output, sample_root / "manifest.json", {
        "schema": "frontieram/ihvp-sequence-cases-1", "version": "v3", "split": "samples",
        "sequence_count": 2, "selection": "first and last added public-feedback cases in frozen manifest order",
        "teacher_access": "host sample evaluator only; never a candidate input or mount",
        "cases": sample_records})
    seal = {"schema": "frontieram/ihvp-hard-v3-local-seal-1", "version": "v3",
            "status": "data_sealed_reference_qualification_pending",
            "public_sequences": 30, "private_sequences": 30, "sample_sequences": 2,
            "group_counts_per_split": {"legacy": 15, "hard": 15},
            "source_manifest_sha256": source_digests, "data_manifest_sha256": manifest_digests,
            "data_fingerprint": hashlib.sha256(json.dumps(manifest_digests, sort_keys=True,
                                                           separators=(",", ":")).encode()).hexdigest(),
            "public_private_teacher_overlap": 0,
            "preservation": "original teacher bytes preserved; no regeneration or solver run",
            "private_plan": "not copied; seed plans remain in their original local storage",
            "reference_qualification": "pending; no formal score asserted"}
    write_once(output, output / "seal.json", seal)
    return seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addition-root", type=Path, required=True)
    legacy = parser.add_mutually_exclusive_group(required=True)
    legacy.add_argument("--legacy-data-root", type=Path)
    legacy.add_argument("--legacy-bundle", type=Path)
    parser.add_argument("--out", "--data-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    try:
        prepare(args)
    except Exception as error:
        # Console output never includes a private ID, seed, source path or body.
        print(f"data seal failed ({type(error).__name__}); inspect local inputs", file=sys.stderr)
        return 2
    print(json.dumps({"status": "DATA_SEALED", "public_sequences": 30,
                      "private_sequences": 30, "sample_sequences": 2,
                      "reference_qualification": "PENDING"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
