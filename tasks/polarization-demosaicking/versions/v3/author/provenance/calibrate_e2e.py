#!/usr/bin/env python3
"""Run a frozen reference source through the v1.2 one-scene worker contract.

This is staging-only.  It imports the released runner read-only and writes
receipts under a caller-provided output directory; it does not touch v1.2.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("converted_root", type=Path)
    ap.add_argument("scene_file", type=Path)
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--runner-root", type=Path,
                    default=Path("releases/polarization-demosaicking-maintenance-v1.2"))
    args = ap.parse_args()

    runner_dir = args.runner_root / "harbor/polarization-demosaicking"
    sys.path.insert(0, str(runner_dir))
    from runner import build_submission, run_scene, SCENE_LIMIT_S  # type: ignore

    args.output.mkdir(parents=True, exist_ok=False)
    build = build_submission(args.source, args.output / "build")
    rows: list[dict] = []
    if build["status"] != "ok":
        (args.output / "summary.json").write_text(json.dumps({
            "status": "build_failure", "build": build, "scene_limit_s": SCENE_LIMIT_S,
        }, indent=2) + "\n")
        raise SystemExit(2)

    for scene in [x.strip() for x in args.scene_file.read_text().splitlines() if x.strip()]:
        scene_out = args.output / f"scene-{scene}"
        execution = run_scene(args.output / "build", args.converted_root / scene / "raw.npy", scene_out)
        row: dict = {"scene": scene, **execution}
        if execution["status"] == "ok":
            pred = np.load(scene_out / "pred.npy", allow_pickle=False)
            ref = np.load(args.converted_root / scene / "reference.npy", allow_pickle=False)
            channels = np.sqrt(np.mean((pred.astype(np.float64) - ref.astype(np.float64)) ** 2,
                                       axis=(1, 2)))
            row["channel_rmse"] = [float(v) for v in channels]
            row["mean_rmse"] = float(np.mean(channels))
        rows.append(row)

    csv_path = args.output / "per_scene.csv"
    with csv_path.open("w", newline="") as stream:
        fields = ["scene", "status", "reason", "wall_s", "peak_bytes",
                  "rmse_0", "rmse_45", "rmse_90", "rmse_135", "mean_rmse"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            vals = dict(row)
            vals.update({
                "rmse_0": (row.get("channel_rmse") or [None] * 4)[0],
                "rmse_45": (row.get("channel_rmse") or [None] * 4)[1],
                "rmse_90": (row.get("channel_rmse") or [None] * 4)[2],
                "rmse_135": (row.get("channel_rmse") or [None] * 4)[3],
            })
            writer.writerow({key: vals.get(key) for key in fields})
    passed = sum(row["status"] == "ok" for row in rows)
    summary = {
        "status": "ok" if passed == len(rows) else "submission_failure",
        "scene_limit_s": SCENE_LIMIT_S,
        "expected_scenes": len(rows),
        "scenes_passed": passed,
        "max_wall_s": max((row.get("wall_s", 0.0) for row in rows), default=None),
        "mean_rmse": (sum(row["mean_rmse"] for row in rows if "mean_rmse" in row) / passed
                      if passed else None),
        "rows": rows,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("status", "expected_scenes", "scenes_passed", "max_wall_s", "mean_rmse")},
                     ensure_ascii=False))
    raise SystemExit(0 if summary["status"] == "ok" else 2)


if __name__ == "__main__":
    main()
