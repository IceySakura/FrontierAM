#!/usr/bin/env python3
"""Convert Wen RGB polarization references into the grayscale Qiu-style task."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np


ANGLES = (0, 45, 90, 135)


def read_png(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    if blob[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    pos = 8
    idat = []
    ihdr = None
    while pos < len(blob):
        length = struct.unpack(">I", blob[pos : pos + 4])[0]
        kind = blob[pos + 4 : pos + 8]
        payload = blob[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            idat.append(payload)
        elif kind == b"IEND":
            break
    if ihdr is None:
        raise ValueError(f"missing IHDR: {path}")
    width, height, bit_depth, color_type, compression, filtering, interlace = ihdr
    if (bit_depth, compression, filtering, interlace) != (8, 0, 0, 0):
        raise ValueError(f"unsupported PNG encoding in {path}: {ihdr}")
    channels = {0: 1, 2: 3}.get(color_type)
    if channels is None:
        raise ValueError(f"unsupported PNG color type {color_type}: {path}")
    raw = zlib.decompress(b"".join(idat))
    stride = width * channels
    output = np.empty((height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.uint8)
    offset = 0
    for row in range(height):
        filter_type = raw[offset]
        current = np.frombuffer(raw[offset + 1 : offset + stride + 1], dtype=np.uint8).copy()
        offset += stride + 1
        if filter_type not in (0, 1, 2, 3, 4):
            raise ValueError(f"unsupported PNG filter {filter_type}: {path}")
        # PNG filters are byte-recursive within a row. In particular, the
        # left predictor must use already reconstructed bytes, not filtered
        # bytes; doing this with a vectorized left array silently corrupts
        # almost every row for filters 1, 3 and 4.
        for index in range(stride):
            left = int(current[index - channels]) if index >= channels else 0
            upper = int(previous[index])
            upper_left = int(previous[index - channels]) if index >= channels else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = upper
            elif filter_type == 3:
                predictor = (left + upper) // 2
            else:
                estimate = left + upper - upper_left
                pa, pb, pc = abs(estimate - left), abs(estimate - upper), abs(estimate - upper_left)
                predictor = left if pa <= pb and pa <= pc else (upper if pb <= pc else upper_left)
            current[index] = (int(current[index]) + predictor) & 255
        output[row] = current
        previous = current
    return output.reshape(height, width, channels)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_scene(source_root: Path, output_root: Path, scene: str, config: dict) -> dict:
    crop = config["crop"]
    top, left = crop["top"], crop["left"]
    height, width = crop["height"], crop["width"]
    weights = np.asarray(config["grayscale"]["weights"], dtype=np.int16)
    denominator = int(config["grayscale"]["denominator"])
    planes = []
    sources = {}
    for angle in ANGLES:
        source = source_root / scene / f"gt_{angle}" / f"{scene}.png"
        rgb = read_png(source)
        if rgb.shape != (1088, 1456, 3):
            raise ValueError(f"unexpected shape for {source}: {rgb.shape}")
        sources[str(angle)] = sha256_file(source)
        values = (rgb.astype(np.int32) * weights.reshape(1, 1, 3)).sum(axis=2)
        gray = np.floor(values / denominator + 0.5).clip(0, 255).astype(np.uint8)
        planes.append(gray[top : top + height, left : left + width])
    reference = np.stack(planes, axis=0)
    raw = np.empty((height, width), dtype=np.uint8)
    raw[0::2, 0::2] = reference[2, 0::2, 0::2]
    raw[0::2, 1::2] = reference[1, 0::2, 1::2]
    raw[1::2, 0::2] = reference[3, 1::2, 0::2]
    raw[1::2, 1::2] = reference[0, 1::2, 1::2]
    scene_dir = output_root / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    raw_path = scene_dir / "raw.npy"
    reference_path = scene_dir / "reference.npy"
    np.save(raw_path, raw, allow_pickle=False)
    np.save(reference_path, reference, allow_pickle=False)
    return {
        "scene": scene,
        "source_sha256": sources,
        "raw_sha256": sha256_file(raw_path),
        "reference_sha256": sha256_file(reference_path),
        "raw_shape": list(raw.shape),
        "reference_shape": list(reference.shape),
        "dtype": "uint8"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("scene_file", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    scenes = [line.strip() for line in args.scene_file.read_text().splitlines() if line.strip()]
    records = [convert_scene(args.source_root, args.output_root, scene, config) for scene in scenes]
    manifest = {
        "conversion_version": config["version"],
        "config": config,
        "scenes": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"scenes": len(records), "output": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
