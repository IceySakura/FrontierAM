#!/usr/bin/env python3
"""Inventory an extracted Wen/CPDNet scene directory without decoding pixels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct


CHANNEL_DIRS = ("gt_0", "gt_45", "gt_90", "gt_135", "net_input")


def png_info(path: Path) -> dict:
    data = path.open("rb")
    with data:
        if data.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"not a PNG: {path}")
        length = struct.unpack(">I", data.read(4))[0]
        kind = data.read(4)
        if kind != b"IHDR" or length != 13:
            raise ValueError(f"missing IHDR: {path}")
        width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
            ">IIBBBBB", data.read(13)
        )
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
        "compression": compression,
        "filtering": filtering,
        "interlace": interlace,
    }


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(root: Path) -> dict:
    scenes = []
    errors = []
    for path in sorted(root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
        if not path.is_dir() or not path.name.isdigit():
            errors.append(f"unexpected top-level entry: {path.name}")
            continue
        scene = path.name
        record = {"scene": scene, "files": {}}
        for channel in CHANNEL_DIRS:
            channel_dir = path / channel
            expected = channel_dir / f"{scene}.png"
            if not expected.is_file():
                errors.append(f"missing {expected}")
                continue
            extras = sorted(p.name for p in channel_dir.iterdir() if p.name != expected.name)
            if extras:
                errors.append(f"unexpected files in {channel_dir}: {extras}")
            info = png_info(expected)
            record["files"][channel] = {
                "path": str(expected.relative_to(root)),
                "bytes": expected.stat().st_size,
                "sha256": digest(expected),
                **info,
            }
        scenes.append(record)
    return {
        "root_name": root.name,
        "scene_count": len(scenes),
        "scene_ids": [record["scene"] for record in scenes],
        "channel_dirs": list(CHANNEL_DIRS),
        "scenes": scenes,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = inventory(args.root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"scene_count": result["scene_count"], "errors": len(result["errors"])}, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
