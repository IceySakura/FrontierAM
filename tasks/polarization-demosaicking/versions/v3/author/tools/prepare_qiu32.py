#!/usr/bin/env python3
"""Prepare Qiu32 NPY arrays from the frozen v1.2 BMP source directory."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


ANGLES = (0, 45, 90, 135)


def read_bmp(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    if blob[:2] != b"BM" or struct.unpack_from("<I", blob, 10)[0] != 1078:
        raise ValueError(f"unexpected BMP header: {path}")
    return np.frombuffer(blob, dtype=np.uint8, offset=1078).reshape(1024, 1024)[::-1].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="v1.2 table-iii-40/data directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("scene_file", type=Path)
    args = parser.parse_args()
    records = json.loads((args.source / "final_inputs.json").read_text())
    for scene in [line.strip() for line in args.scene_file.read_text().splitlines() if line.strip()]:
        planes = []
        for angle in ANGLES:
            record = next(row for row in records[scene] if row["angle"] == angle)
            planes.append(read_bmp(args.source / record["path"]))
        ref = np.stack(planes)
        raw = np.empty((1024, 1024), dtype=np.uint8)
        for row, col, channel in ((0, 0, 2), (0, 1, 1), (1, 0, 3), (1, 1, 0)):
            raw[row::2, col::2] = ref[channel, row::2, col::2]
        dest = args.output / scene
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / "raw.npy", raw, allow_pickle=False)
        np.save(dest / "reference.npy", ref, allow_pickle=False)
    print("prepared Qiu32 arrays")


if __name__ == "__main__":
    main()
