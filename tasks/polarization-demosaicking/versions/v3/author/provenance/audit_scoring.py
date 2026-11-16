#!/usr/bin/env python3
"""Independent audit of the vNext global score mapping."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scoring import aggregate, read_per_scene_csv, score_from_error


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration_dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.calibration_dir
    cal = json.loads((root / "wen_calibration.json").read_text())
    B = cal["anchors"]["B_classic"]
    H = cal["anchors"]["H5_advanced"]
    qiu = root / "qiu32_reference_per_scene.csv"
    checks = {}
    rows_b = read_per_scene_csv(qiu, "qiu", "bilinear") + read_per_scene_csv(
        root / "wen100_e2e_bilinear.csv", "wen")
    rows_h = read_per_scene_csv(qiu, "qiu", "reflect") + read_per_scene_csv(
        root / "wen100_e2e_reflect.csv", "wen")
    b = aggregate(rows_b, B, H)
    h = aggregate(rows_h, B, H)
    manual_e_b = math.fsum(math.fsum(float(row[name]) for name in
                                     ("rmse_0", "rmse_45", "rmse_90", "rmse_135")) / 4
                           for row in rows_b) / 132
    manual_e_h = math.fsum(math.fsum(float(row[name]) for name in
                                     ("rmse_0", "rmse_45", "rmse_90", "rmse_135")) / 4
                           for row in rows_h) / 132
    checks["independent_global_mean"] = abs(manual_e_b - B) <= 1e-12 and abs(manual_e_h - H) <= 1e-12
    checks["classic_endpoint_zero"] = b["status"] == "ok" and abs(b["E"] - B) <= 1e-12 and b["score"] <= 1e-12
    checks["advanced_endpoint_five"] = h["status"] == "ok" and abs(h["E"] - H) <= 1e-12 and abs(h["score"] - 5) <= 1e-12
    checks["perfect_endpoint_ten"] = abs(score_from_error(0, B, H) - 10) <= 1e-12
    failed = list(rows_b)
    failed[0] = dict(failed[0], status="submission_failure", reason="synthetic audit")
    f = aggregate(failed, B, H)
    checks["failure_is_zero"] = f["status"] == "submission_failure" and f["score"] == 0 and f["reward"] == 0
    report = {"passed": all(checks.values()), "checks": checks,
              "independent_errors": {"classic": manual_e_b, "advanced": manual_e_h},
              "classic": b, "advanced": h, "failed_fixture": f}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
