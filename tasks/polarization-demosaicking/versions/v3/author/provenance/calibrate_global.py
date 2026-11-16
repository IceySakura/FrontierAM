#!/usr/bin/env python3
"""Recompute provisional Qiu32+Wen100 anchors from calibration CSV receipts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scoring import aggregate, read_per_scene_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration_dir", type=Path)
    ap.add_argument("--classic-method", default="bilinear")
    ap.add_argument("--advanced-method", default="reflect")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.calibration_dir
    qiu_classic = read_per_scene_csv(root / "qiu32_reference_per_scene.csv", "qiu",
                                     args.classic_method)
    qiu_advanced = read_per_scene_csv(root / "qiu32_reference_per_scene.csv", "qiu",
                                      args.advanced_method)
    wen_classic = read_per_scene_csv(root / "wen100_e2e_bilinear.csv", "wen")
    wen_advanced = read_per_scene_csv(root / "wen100_e2e_reflect.csv", "wen")
    # The baseline values are the global means of their per-scene errors; use
    # the scorer itself for the arithmetic and endpoint checks.
    def mean(rows):
        return math.fsum(math.fsum(float(r[k]) for k in
                                   ("rmse_0", "rmse_45", "rmse_90", "rmse_135")) / 4
                         for r in rows) / len(rows)
    classic = mean(qiu_classic + wen_classic)
    advanced = mean(qiu_advanced + wen_advanced)
    classic_check = aggregate(qiu_classic + wen_classic, classic, advanced)
    advanced_check = aggregate(qiu_advanced + wen_advanced, classic, advanced)
    result = {
        "protocol": "wen-vnext-global-v1",
        "formal_scene_counts": {"qiu": len(qiu_classic), "wen": len(wen_classic),
                                 "total": len(qiu_classic) + len(wen_classic)},
        "per_scene_error": "arithmetic mean of four channel RMSEs",
        "global_error": "arithmetic mean of all 132 formal per-scene errors",
        "anchors": {"B_classic": classic, "H5_advanced": advanced, "perfect": 0.0},
        "endpoint_checks": {"classic": classic_check, "advanced": advanced_check},
    }
    if result["formal_scene_counts"] != {"qiu": 32, "wen": 100, "total": 132}:
        raise SystemExit("unexpected formal scene counts")
    if not classic > advanced > 0:
        raise SystemExit("anchors do not satisfy classic > advanced > 0")
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
