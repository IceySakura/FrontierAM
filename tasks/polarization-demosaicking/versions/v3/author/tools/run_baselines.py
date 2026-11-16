#!/usr/bin/env python3
"""Measure the two maintenance references on fixed arrays."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reconstruct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reflect_root = ROOT / "references/reflect"
    sys.path.insert(0, str(reflect_root))
    bilinear = load(ROOT / "references/bilinear/solution.py", "vnext_bilinear")
    reflect = load(ROOT / "references/reflect/solution.py", "vnext_reflect")
    scenes = []
    for dataset, name in (("qiu", "qiu32"), ("wen", "wen100")):
        scene_file = ROOT / "tasks/polarization-demosaicking/vnext" / (
            "qiu_scenes.txt" if dataset == "qiu" else "wen_scenes_eval.txt")
        for scene in [line.strip() for line in scene_file.read_text().splitlines() if line.strip()]:
            scenes.append((dataset, scene, ROOT / "data" / name / scene))
    args.output.mkdir(parents=True, exist_ok=False)
    fields = ["dataset", "scene", "method", "rmse_0", "rmse_45", "rmse_90",
              "rmse_135", "mean_rmse", "elapsed_s"]
    with (args.output / "per_scene.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset, scene, path in scenes:
            raw = np.load(path / "raw.npy", allow_pickle=False)
            ref = np.load(path / "reference.npy", allow_pickle=False).astype(np.float64)
            for method, reconstruct in (("bilinear", bilinear), ("reflect", reflect)):
                start = time.perf_counter()
                pred = np.asarray(reconstruct(raw), dtype=np.float64)
                elapsed = time.perf_counter() - start
                channels = np.sqrt(np.mean((pred - ref) ** 2, axis=(1, 2)))
                writer.writerow({"dataset": dataset, "scene": scene, "method": method,
                                 "rmse_0": float(channels[0]), "rmse_45": float(channels[1]),
                                 "rmse_90": float(channels[2]), "rmse_135": float(channels[3]),
                                 "mean_rmse": float(np.mean(channels)), "elapsed_s": elapsed})
    print(f"wrote {len(scenes) * 2} baseline rows to {args.output / 'per_scene.csv'}")


if __name__ == "__main__":
    main()
