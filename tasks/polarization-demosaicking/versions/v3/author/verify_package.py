#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def scene_ids(root: Path, numeric: bool = False) -> list[str]:
    values = [path.name for path in root.iterdir() if path.is_dir()]
    return sorted(values, key=lambda x: int(x) if numeric else x)


def check_arrays(root: Path, scenes: list[str]) -> None:
    for scene in scenes:
        folder = root / scene
        raw = np.load(folder / "raw.npy", allow_pickle=False)
        ref = np.load(folder / "reference.npy", allow_pickle=False)
        if raw.shape != (1024, 1024) or raw.dtype != np.uint8:
            raise SystemExit(f"invalid raw array: {folder}")
        if ref.shape != (4, 1024, 1024) or ref.dtype != np.uint8:
            raise SystemExit(f"invalid reference array: {folder}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise SystemExit(f"manifest mismatch: {name}")
    qiu = scene_ids(ROOT / "data/qiu32")
    wen5 = scene_ids(ROOT / "data/wen5", numeric=True)
    wen100 = scene_ids(ROOT / "data/wen100", numeric=True)
    if len(qiu) != 32 or len(wen5) != 5 or len(wen100) != 100:
        raise SystemExit(f"scene count mismatch: qiu={len(qiu)} wen5={len(wen5)} wen100={len(wen100)}")
    check_arrays(ROOT / "data/qiu32", qiu)
    check_arrays(ROOT / "data/wen5", wen5)
    check_arrays(ROOT / "data/wen100", wen100)
    runtime_checked = False
    if args.runtime:
        image = json.loads(subprocess.check_output(
            ["docker", "image", "inspect", "sha256:80b3867f1168dd170b4e7a8a06bb0fe98cb59b38116016641fee7a61fe823b42"]
        ))[0]
        if image["Architecture"] != "arm64" or image["Os"] != "linux":
            raise SystemExit("vNext worker image is not Linux ARM64")
        runtime_checked = True
    print(json.dumps({"status": "passed", "files_checked": len(manifest["files"]),
                      "qiu32": len(qiu), "wen5": len(wen5), "wen100": len(wen100),
                      "runtime_checked": runtime_checked}, indent=2))


if __name__ == "__main__":
    main()
