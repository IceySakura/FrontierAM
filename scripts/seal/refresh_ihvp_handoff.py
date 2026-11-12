#!/usr/bin/env python3
"""Add the maintainer guide and exact learner view to an existing v3 ZIP.

This is a DELIVERY revision only. Every original archive member is preserved
byte for byte; the original README and file manifest get provenance names.
No data generation, solver run, reference change or formal activation occurs.
The completed child ZIP is checked against its file manifest and every original
member is then compared byte for byte with its preserved child counterpart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
VERSION = ROOT / "tasks/ihvp-sequence/versions/v3"
ALLOWLIST = {
    "problem.md": "statement.md",
    "samples/README.md": "samples/README.md",
    "samples/run_sample.py": "samples/run_sample.py",
    "samples/smoke.json": "samples/smoke.json",
    "samples/support/api.py": "samples/support/api.py",
    "samples/support/worker.py": "samples/support/worker.py",
}
RENAMES = {"README.md": "provenance/README.before-guide.md",
           "FILE_SHA256.json": "provenance/FILE_SHA256.before-guide.json"}
SOURCE_EXTRAS = ("scripts/seal/build_ihvp_model_visible.py",
                 "scripts/ops/serve_ihvp_hard_samples.py",
                 "arena/operator_drivers/ihvp_oracle.py", "arena/__init__.py",
                 "scripts/seal/refresh_ihvp_handoff.py")
DRIVER_INITIALIZER = "arena/operator_drivers/__init__.py"
CHUNK = 1024 * 1024


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def safe_name(name):
    path = PurePosixPath(name)
    require(name and not name.endswith("/") and not path.is_absolute()
            and ".." not in path.parts and "\\" not in name and "\x00" not in name
            and path.as_posix() == name, "unsafe or noncanonical ZIP member name")


def inventory(archive):
    entries = archive.infolist()
    names = [entry.filename for entry in entries]
    require(len(names) == len(set(names)), "duplicate ZIP members")
    for entry in entries:
        safe_name(entry.filename)
        require(not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16), "ZIP member must be a regular file")
        require(not entry.flag_bits & 1, "encrypted ZIP members are unsupported")
    return {entry.filename: entry for entry in entries}


def validate_source_checksum(path):
    companion = path.with_suffix(path.suffix + ".sha256")
    require(companion.is_file() and not companion.is_symlink(), "source ZIP checksum file is missing")
    lines = companion.read_text().splitlines()
    require(len(lines) == 1, "expected exactly one source ZIP checksum line")
    match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", lines[0])
    require(match is not None and match.group(2) == path.name, "source checksum filename differs")
    actual = digest_file(path)
    require(actual == match.group(1).lower(), "source ZIP checksum mismatch")
    return actual


def validate_model_zip(path):
    require(path.stat().st_size <= 2 * CHUNK, "learner ZIP is larger than the six-file interface")
    data = path.read_bytes()
    sources, records = {}, {}
    with zipfile.ZipFile(path) as archive:
        entries = inventory(archive)
        require(set(entries) == set(ALLOWLIST), "learner ZIP must contain exactly the six allowed files")
        require(not archive.comment, "learner ZIP must not contain an archive comment")
        for name, relative in ALLOWLIST.items():
            source = VERSION / relative
            require(source.is_file() and not source.is_symlink()
                    and source.resolve().is_relative_to(VERSION.resolve()), "learner source is not a contained file")
            entry = entries[name]
            require(entry.file_size <= CHUNK and not entry.comment and not entry.extra,
                    "unexpected learner member payload/metadata")
            content = archive.read(entry)
            source_content = source.read_bytes()
            require(content == source_content, "learner member differs from the current whitelist source")
            records[name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            sources[name] = "tasks/ihvp-sequence/versions/v3/" + relative
    return data, hashlib.sha256(data).hexdigest(), records, sources


def source_metadata(archive, entries):
    require("README.md" in entries and "FILE_SHA256.json" in entries, "source is not a flat-root curator ZIP")
    raw = archive.read("FILE_SHA256.json")
    metadata = json.loads(raw)
    files = metadata.get("files")
    require(isinstance(files, dict) and set(files) == set(entries) - {"FILE_SHA256.json"},
            "source file manifest does not cover every other member exactly")
    for name, record in files.items():
        safe_name(name)
        require(isinstance(record, dict) and re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
                and type(record.get("bytes")) is int and record["bytes"] >= 0,
                "invalid source manifest file record")
        require(entries[name].file_size == record["bytes"], "source entry size differs from its file manifest")
    require(metadata.get("public_sequences") == metadata.get("private_sequences") == 15,
            "source must be the new 15+15 supplement, not the combined cohort")
    scientific = {}
    for split in ("public", "private"):
        manifest_name = f"data/{split}/manifest.json"
        require(manifest_name in files, "source scientific manifest is missing")
        manifest = json.loads(archive.read(manifest_name))
        require(manifest.get("split") == split and len(manifest.get("cases", [])) == 15,
                "source scientific split is not 15 sequences with its declared role")
        names = []
        ids = set()
        for case in manifest["cases"]:
            relative = str(case["path"])
            safe_name(relative)
            name = f"data/{split}/" + relative
            require(case["id"] not in ids and name not in names, "duplicate scientific case")
            ids.add(case["id"]); names.append(name)
            require(name in files and relative.endswith(".npz")
                    and files[name]["sha256"] == case["sha256"], "scientific body/manifest fingerprint differs")
            if "bytes" in case:
                require(files[name]["bytes"] == case["bytes"], "scientific body size differs")
            require(case.get("dimension") == 16384 and case.get("steps") == 50, "scientific dimensions changed")
        scientific[split] = names
    all_npz = {name for name in entries if name.endswith(".npz")}
    require(all_npz == set(scientific["public"] + scientific["private"]), "source contains unexpected scientific bodies")
    require(not {files[name]["sha256"] for name in scientific["public"]}
            & {files[name]["sha256"] for name in scientific["private"]}, "source public/private bodies overlap")
    return metadata, raw, scientific


def zip_info(name):
    entry = zipfile.ZipInfo(name)
    entry.create_system = 3
    entry.external_attr = 0o100600 << 16
    entry.compress_type = zipfile.ZIP_STORED
    return entry


def compare_streams(left, right):
    while True:
        first, second = left.read(CHUNK), right.read(CHUNK)
        require(first == second, "an original entry changed in the refreshed archive")
        if not first:
            return


def verify_child(source, child_path, metadata, original_names):
    with zipfile.ZipFile(child_path) as child:
        entries = inventory(child)
        require(set(entries) == set(metadata["files"]) | {"FILE_SHA256.json"}, "child inventory differs from its manifest")
        require(json.loads(child.read("FILE_SHA256.json")) == metadata, "child file manifest changed")
        for name, record in metadata["files"].items():
            digest, size = hashlib.sha256(), 0
            with child.open(name) as stream:
                for block in iter(lambda: stream.read(CHUNK), b""):
                    digest.update(block); size += len(block)
            require(digest.hexdigest() == record["sha256"] and size == record["bytes"], "child member checksum mismatch")
        for old_name in original_names:
            with source.open(old_name) as left, child.open(RENAMES.get(old_name, old_name)) as right:
                compare_streams(left, right)


def refresh(args):
    source_path = args.source_zip.expanduser().resolve()
    model_path = args.model_visible.expanduser().resolve()
    out = args.out.expanduser().resolve()
    companion = out.with_suffix(out.suffix + ".sha256")
    require(source_path.is_file() and model_path.is_file() and source_path != model_path, "source/model ZIP is missing")
    require(out.suffix.lower() == ".zip" and out not in (source_path, model_path), "output must be a new ZIP path")
    require(not out.is_relative_to(ROOT) or out.is_relative_to(ROOT / "var"), "output must be ignored var/ or outside checkout")
    require(not any(p.exists() or p.is_symlink() for p in (out, companion)), "refusing to overwrite an existing delivery")
    source_sha = validate_source_checksum(source_path)
    source_stat = source_path.stat()
    model_bytes, model_sha, visible, visible_sources = validate_model_zip(model_path)
    guide_path = VERSION / "USAGE_GUIDE.zh.md"
    require(guide_path.is_file() and not guide_path.is_symlink(), "maintainer guide is missing")
    extras = {"USAGE_GUIDE.zh.md": guide_path.read_bytes(), "model-visible.zip": model_bytes}
    extra_sources = {"USAGE_GUIDE.zh.md": "tasks/ihvp-sequence/versions/v3/USAGE_GUIDE.zh.md",
                     "model-visible.zip": "validated exact six-file archive"}
    for relative in SOURCE_EXTRAS:
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT), "delivery helper is missing")
        name = "curator/source/" + relative
        extras[name] = path.read_bytes(); extra_sources[name] = relative
    initializer = ROOT / DRIVER_INITIALIZER
    initializer_name = "curator/source/" + DRIVER_INITIALIZER
    require(not initializer.is_symlink(), "driver initializer must not be a symlink")
    extras[initializer_name] = initializer.read_bytes() if initializer.is_file() else b""
    extra_sources[initializer_name] = DRIVER_INITIALIZER if initializer.is_file() else "generated empty delivery-only package initializer"
    report_metadata = None
    if args.validation_report is not None:
        report_path = args.validation_report.expanduser().resolve()
        report_bytes = report_path.read_bytes()
        report_metadata = json.loads(report_bytes)
        require(report_metadata.get("status") == "PASS", "optional delivery validation is not PASS")
        require(report_metadata.get("model_visible_zip_sha256") == model_sha,
                "delivery validation does not bind this exact learner ZIP")
        require(set(report_metadata.get("visible_files", [])) == set(ALLOWLIST), "delivery validation whitelist differs")
        extras["validation/delivery_guide_validation.json"] = report_bytes
        extra_sources["validation/delivery_guide_validation.json"] = "maintainer functional validation report"
    visibility = {"schema": "frontieram/ihvp-v3-model-visibility-handoff-1", "maintainer_only": True,
        "delivery_revision": "usage-guide-1", "scientific_data_version": "v3",
        "scientific_data_changed": False, "frozen_algorithms_changed": False,
        "parent_archive_sha256": source_sha, "model_visible_zip_sha256": model_sha,
        "model_visible_file_count": 6, "model_visible_files": visible,
        "model_visible_source_paths": visible_sources,
        "only_teacher_exception": "public d=4,T=3 smoke.json; full-scale teachers are host-only",
        "allowed_runtime_material": ["own submission and temporary builds", "preinstalled general numerical libraries",
            "current RHS/context and current HVP replies", "dedicated sample sockets and sample results"],
        "never_model_visible": ["parent/curator ZIP", "USAGE_GUIDE.zh.md", "MODEL_VISIBILITY.json", "FILE_SHA256.json",
            "whole repository or .git", "curator/source", "generators and teacher factors", "references and grader",
            "data/public and data/private bodies/manifests", "full-size sample teachers", "seed plans and freeze receipts",
            "analysis/calibration/validation/benchmark records and predictions", "other research chats or maintainer context"],
        "new_entry_sources": extra_sources,
        "functional_validation": None if report_metadata is None else {
            "status": report_metadata["status"], "scope": report_metadata.get("scope"),
            "docker_tested": report_metadata.get("docker_tested"),
            "report": "validation/delivery_guide_validation.json"},
        "release_status": "delivery documentation only; v3 remains draft; no Docker/resource/security acceptance asserted"}
    extras["MODEL_VISIBILITY.json"] = encoded(visibility)
    readme = """# IHVP v3 增量交付包（仅维护者）

请先阅读 [中文使用指南](USAGE_GUIDE.zh.md)。本总包包含新增的 15 条反馈序列和
15 条私有序列，以及参考答案、私有因子、种子和维护记录，不能交给被评模型。
仅提取 [model-visible.zip](model-visible.zip) 供模型使用；该小包恰好六个文件。
模型可见范围及禁止挂载内容见指南与维护端 MODEL_VISIBILITY.json。

本次仅修订交付说明和公开协议工具的宿主接线，不改变科学数据、冻结算法或预算，
不是新的算法／数据版本，也不激活 v3 正式评分。完整规模样例只能经宿主 HVP 服务访问；
smoke.json 的 4 维、3 步公开矩阵是唯一例外。目标 Linux/Docker 隔离仍需维护端验收。

原包全部条目内容逐字节保留：旧 README 和旧校验清单分别在
provenance/README.before-guide.md、provenance/FILE_SHA256.before-guide.json。
FILE_SHA256.json 是新包清单，并绑定原 ZIP 的 SHA-256；ZIP 旁的 .sha256 校验整个新包。
""".encode()
    extras["README.md"] = readme
    created, companion_created = False, False
    with zipfile.ZipFile(source_path) as source:
        old_entries = inventory(source)
        old_metadata, old_manifest_bytes, scientific = source_metadata(source, old_entries)
        preserved = {RENAMES.get(name, name): name for name in old_entries}
        require(len(preserved) == len(old_entries), "renamed original entries collide")
        records = {}
        for target, original in preserved.items():
            if original == "FILE_SHA256.json":
                records[target] = {"sha256": hashlib.sha256(old_manifest_bytes).hexdigest(), "bytes": len(old_manifest_bytes)}
            else:
                records[target] = dict(old_metadata["files"][original])
        for name, content in extras.items():
            safe_name(name)
            if name in preserved:
                original = preserved[name]
                require(source.read(original) == content, "new helper collides with different preserved source bytes")
                continue
            records[name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        new_metadata = {**old_metadata, "schema": "frontieram/ihvp-hard-v3-private-handoff-1",
            "delivery_revision": "usage-guide-1", "maintainer_only": True,
            "scientific_data_version": "v3", "scientific_data_changed": False, "frozen_algorithms_changed": False,
            "parent_archive": {"filename": source_path.name, "sha256": source_sha,
                               "original_file_manifest_sha256": hashlib.sha256(old_manifest_bytes).hexdigest()},
            "preserved_entry_count": len(old_entries), "preserved_entry_renames": RENAMES,
            "science_body_counts": {key: len(value) for key, value in scientific.items()},
            "guide_source": {"repo_path": "tasks/ihvp-sequence/versions/v3/USAGE_GUIDE.zh.md",
                             "sha256": records["USAGE_GUIDE.zh.md"]["sha256"]},
            "model_visible_zip_sha256": model_sha, "files": dict(sorted(records.items()))}
        manifest_bytes = encoded(new_metadata)
        out.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        created = True
        try:
            with os.fdopen(descriptor, "wb") as stream, zipfile.ZipFile(stream, "w", allowZip64=True) as child:
                for target, original in sorted(preserved.items()):
                    digest, size = hashlib.sha256(), 0
                    with source.open(original) as incoming, child.open(zip_info(target), "w", force_zip64=True) as outgoing:
                        for block in iter(lambda: incoming.read(CHUNK), b""):
                            digest.update(block); size += len(block); outgoing.write(block)
                    expected = records[target]
                    require(digest.hexdigest() == expected["sha256"] and size == expected["bytes"],
                            "original member differs from its source manifest")
                for name, content in sorted(extras.items()):
                    if name not in preserved:
                        child.writestr(zip_info(name), content)
                child.writestr(zip_info("FILE_SHA256.json"), manifest_bytes)
            verify_child(source, out, new_metadata, old_entries)
            current = source_path.stat()
            require((current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) ==
                    (source_stat.st_dev, source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns),
                    "source ZIP changed during delivery refresh")
            checksum = digest_file(out)
            companion_descriptor = os.open(companion, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            companion_created = True
            with os.fdopen(companion_descriptor, "w") as stream:
                stream.write(f"{checksum}  {out.name}\n")
            out.chmod(0o600); companion.chmod(0o600)
        except BaseException:
            if created:
                out.unlink(missing_ok=True)
            if companion_created:
                companion.unlink(missing_ok=True)
            raise
    return {"status": "PASS", "delivery_revision": "usage-guide-1", "data_changed": False,
            "algorithm_changed": False, "public_sequences": 15, "private_sequences": 15,
            "preserved_entries_byte_compared": len(old_entries), "model_visible_files": 6,
            "parent_zip_sha256": source_sha, "zip_sha256": checksum,
            "bytes": out.stat().st_size, "output": str(out)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--model-visible", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--validation-report", type=Path)
    args = parser.parse_args()
    try:
        result = refresh(args)
    except Exception as error:
        print(f"delivery refresh failed ({type(error).__name__}): {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
