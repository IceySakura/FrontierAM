#!/usr/bin/env python3
"""Measure the internal bilinear/reflect references on converted vNext arrays."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("converted_root", type=Path)
    parser.add_argument("scene_file", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, "/workspace/results/polarization-demosaicking/full-frame-leic")
    sys.path.insert(0, "/workspace/results/polarization-demosaicking/probe")
    from full_frame import reconstruct
    from metrics import rmse_per_channel

    rows = []
    for scene in [line.strip() for line in args.scene_file.read_text().splitlines() if line.strip()]:
        scene_dir = args.converted_root / scene
        raw = np.load(scene_dir / "raw.npy", allow_pickle=False)
        ref = np.load(scene_dir / "reference.npy", allow_pickle=False)
        row = {"scene": scene}
        for method in ("bilinear", "reflect"):
            start = time.perf_counter()
            pred = reconstruct(raw, method)
            elapsed = time.perf_counter() - start
            channels = rmse_per_channel(pred, ref)
            for index, value in enumerate(channels):
                row[f"{method}_rmse_{index}"] = float(value)
            row[f"{method}_mean_rmse"] = float(np.mean(channels))
            row[f"{method}_seconds"] = elapsed
        rows.append(row)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"scenes": len(rows), "output": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
