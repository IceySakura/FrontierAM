#!/usr/bin/env python3
"""Run a submission against the frozen vNext arrays."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scoring import CANONICAL_ADVANCED, CANONICAL_CLASSIC, FORMAL_ROSTER, aggregate
from runner import build_submission, run_scene


ROOT = Path(__file__).resolve().parents[2]


def run_one(build_dir: Path, data_dir: Path, result_dir: Path) -> dict:
    raw = data_dir / "raw.npy"
    reference = np.load(data_dir / "reference.npy", allow_pickle=False)
    execution = run_scene(build_dir, raw, result_dir)
    row = dict(execution)
    if execution["status"] == "ok":
        prediction = np.load(result_dir / "pred.npy", allow_pickle=False)
        delta = prediction.astype(np.float64) - reference.astype(np.float64)
        channels = np.sqrt(np.mean(delta * delta, axis=(1, 2)))
        row.update({
            "rmse_0": float(channels[0]),
            "rmse_45": float(channels[1]),
            "rmse_90": float(channels[2]),
            "rmse_135": float(channels[3]),
            "mean_rmse": float(np.mean(channels)),
        })
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--mode", choices=("dev", "formal"), default="formal")
    parser.add_argument("--classic", type=float, default=CANONICAL_CLASSIC)
    parser.add_argument("--advanced", type=float, default=CANONICAL_ADVANCED)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output exists: {args.output}")
    if args.mode == "formal" and (args.classic != CANONICAL_CLASSIC or
                                   args.advanced != CANONICAL_ADVANCED):
        raise SystemExit("formal anchors are frozen and cannot be overridden")

    args.output.mkdir(parents=True)
    build = build_submission(args.source, args.output / "build")
    records = []
    if build["status"] == "ok":
        if args.mode == "formal":
            qiu_scenes = list(FORMAL_ROSTER["qiu"])
            wen_scenes = list(FORMAL_ROSTER["wen"])
            groups = [("qiu", qiu_scenes, args.data_root / "qiu32"),
                      ("wen", wen_scenes, args.data_root / "wen100")]
        else:
            scenes_file = args.data_root / "wen5" / "scenes.txt"
            scenes = [line.strip() for line in scenes_file.read_text().splitlines() if line.strip()]
            groups = [("wen", scenes, args.data_root / "wen5")]
        for dataset, scenes, root in groups:
            for scene in scenes:
                case = args.output / f"{dataset}-{scene}"
                row = run_one(args.output / "build", root / scene, case)
                row.update(dataset=dataset, scene=scene)
                records.append(row)
    fields = ["dataset", "scene", "status", "reason", "wall_s", "peak_bytes",
              "rmse_0", "rmse_45", "rmse_90", "rmse_135", "mean_rmse"]
    with (args.output / "per_scene.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in records)
    if build["status"] != "ok":
        scored = {"status": "submission_failure", "reason": "build failure",
                  "E": None, "score": 0.0, "reward": 0.0}
    elif args.mode == "formal":
        scored = aggregate(records, args.classic, args.advanced)
    else:
        errors = [row["mean_rmse"] for row in records if row.get("status") == "ok"]
        scored = {"status": "ok" if len(errors) == len(records) else "submission_failure",
                  "scene_count": len(records),
                  "mean_rmse": float(np.mean(errors)) if errors else None}
    summary = {"mode": args.mode, "build": build, "scene_count": len(records),
               "score": scored,
               "max_wall_s": max((float(row["wall_s"]) for row in records
                                  if row.get("wall_s") is not None), default=None)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if scored["status"] == "ok" else 2)


if __name__ == "__main__":
    main()
