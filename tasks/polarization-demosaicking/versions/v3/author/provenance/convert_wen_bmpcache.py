#!/usr/bin/env python3
"""Use a macOS sips-generated 24-bit BMP cache for fast, verified conversion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


ANGLES = (0, 45, 90, 135)


def read_bmp24(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    if blob[:2] != b"BM":
        raise ValueError(f"not a BMP: {path}")
    pixel_offset = struct.unpack_from("<I", blob, 10)[0]
    width, signed_height = struct.unpack_from("<ii", blob, 18)
    bit_count = struct.unpack_from("<H", blob, 28)[0]
    if bit_count != 24 or width <= 0 or signed_height == 0:
        raise ValueError(f"unsupported BMP: {path}")
    height = abs(signed_height)
    row_bytes = ((width * 3 + 3) // 4) * 4
    rows = np.frombuffer(blob, dtype=np.uint8, offset=pixel_offset, count=row_bytes * height)
    rows = rows.reshape(height, row_bytes)[:, : width * 3].reshape(height, width, 3)
    rows = rows[:, :, ::-1]  # BGR -> RGB
    if signed_height > 0:
        rows = rows[::-1]
    return rows.copy()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bmp_cache", type=Path)
    parser.add_argument("scene_file", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    crop = config["crop"]
    top, left = crop["top"], crop["left"]
    height, width = crop["height"], crop["width"]
    weights = np.asarray(config["grayscale"]["weights"], dtype=np.int16)
    denominator = int(config["grayscale"]["denominator"])
    scenes = [line.strip() for line in args.scene_file.read_text().splitlines() if line.strip()]
    records = []
    for scene in scenes:
        planes = []
        source_hashes = {}
        for angle in ANGLES:
            bmp = args.bmp_cache / f"{scene}_gt{angle}.bmp"
            rgb = read_bmp24(bmp)
            if rgb.shape != (1088, 1456, 3):
                raise ValueError(f"unexpected shape for {bmp}: {rgb.shape}")
            source_hashes[str(angle)] = digest(bmp)
            values = (rgb.astype(np.int32) * weights.reshape(1, 1, 3)).sum(axis=2)
            unrounded = values / denominator
            rounding = config["grayscale"]["rounding"]
            if rounding == "floor(x)":
                gray = np.floor(unrounded)
            elif rounding == "floor(x + 0.5)":
                gray = np.floor(unrounded + 0.5)
            elif rounding == "rint(x)":
                gray = np.rint(unrounded)
            else:
                raise ValueError(f"unsupported rounding rule: {rounding}")
            gray = gray.clip(0, 255).astype(np.uint8)
            planes.append(gray[top : top + height, left : left + width])
        reference = np.stack(planes, axis=0)
        raw = np.empty((height, width), dtype=np.uint8)
        raw[0::2, 0::2] = reference[2, 0::2, 0::2]
        raw[0::2, 1::2] = reference[1, 0::2, 1::2]
        raw[1::2, 0::2] = reference[3, 1::2, 0::2]
        raw[1::2, 1::2] = reference[0, 1::2, 1::2]
        scene_dir = args.output_root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        raw_path, reference_path = scene_dir / "raw.npy", scene_dir / "reference.npy"
        np.save(raw_path, raw, allow_pickle=False)
        np.save(reference_path, reference, allow_pickle=False)
        records.append({
            "scene": scene,
            "bmp_sha256": source_hashes,
            "raw_sha256": digest(raw_path),
            "reference_sha256": digest(reference_path),
            "raw_shape": list(raw.shape),
            "reference_shape": list(reference.shape),
            "dtype": "uint8"
        })
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps({
        "conversion_version": config["version"],
        "decoder": "macOS sips -> 24-bit BMP -> BGR parser",
        "config": config,
        "scenes": records,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"scenes": len(records), "output": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
