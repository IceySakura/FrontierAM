"""Private vNext scoring core for Qiu32 + Wen100.

The public v1.2 scorer is deliberately untouched.  This module is staging
code and makes the one-global-anchor rule explicit: four channel RMSEs are
averaged per scene, then all 132 scene errors are averaged once.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Mapping


QIU_FORMAL_ROSTER = (
    "ball", "blackstuff", "bottles", "cablelid", "caligraphset", "camera", "chairs",
    "containers", "dinosaur", "door", "drinkingfountain", "einstein", "fakefruit",
    "floor", "fruit2", "funnel", "glasscube", "kettle", "lock", "mirrorcard",
    "mirrorphone", "monitors", "paint", "plant", "plastik", "plastikcomp",
    "pomegranate", "printer", "ruler", "screen", "tools", "woodwall",
)
WEN_FORMAL_ROSTER = tuple(str(i) for i in range(1, 106) if i not in {43, 55, 72, 91, 99})
FORMAL_ROSTER = {"qiu": QIU_FORMAL_ROSTER, "wen": WEN_FORMAL_ROSTER}
FORMAL_COUNTS = {name: len(roster) for name, roster in FORMAL_ROSTER.items()}
FORMAL_TOTAL = sum(FORMAL_COUNTS.values())
# Canonical values are the math.fsum result from the frozen calibration CSVs.
CANONICAL_CLASSIC = 2.10798080792808
CANONICAL_ADVANCED = 1.5006853943125558
CHANNELS = ("rmse_0", "rmse_45", "rmse_90", "rmse_135")


def scene_error(channel_rmse: Iterable[float]) -> float:
    values = tuple(float(x) for x in channel_rmse)
    if len(values) != 4 or not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError("a scene must have four finite nonnegative RMSE values")
    return math.fsum(values) / 4.0


def score_from_error(error: float, classic: float, advanced: float) -> float:
    error, classic, advanced = map(float, (error, classic, advanced))
    if not all(math.isfinite(x) and x >= 0 for x in (error, classic, advanced)):
        raise ValueError("error and anchors must be finite and nonnegative")
    if not classic > advanced > 0:
        raise ValueError("expected classic > advanced > 0")
    # Endpoint values are often reconstructed from the same CSV with a
    # different summation order.  Treat a sub-ULP-scale drift as the intended
    # anchor so the trusted reference is exactly 0 or 5 rather than 3e-15.
    if error >= classic or math.isclose(error, classic, rel_tol=0.0, abs_tol=1e-12):
        return 0.0
    if math.isclose(error, advanced, rel_tol=0.0, abs_tol=1e-12):
        return 5.0
    if error >= advanced:
        return 5.0 * (classic - error) / (classic - advanced)
    return max(0.0, min(10.0, 10.0 - 5.0 * error / advanced))


def aggregate(rows: Iterable[Mapping[str, object]], classic: float, advanced: float) -> dict:
    """Aggregate candidate rows using the formal all-scenes rule.

    A non-``ok`` formal row is a submission failure and receives zero.  The
    caller must provide exactly one row for every `(dataset, scene)` key.
    """
    records = list(rows)
    keys = [(str(row.get("dataset")), str(row.get("scene"))) for row in records]
    expected_keys = {(dataset, scene) for dataset, roster in FORMAL_ROSTER.items()
                     for scene in roster}
    statuses = {str(row.get("status", "ok")) for row in records}
    if "infrastructure_failure" in statuses:
        return {"status": "infrastructure_failure", "reason": "formal worker/infrastructure failure",
                "E": None, "score": None, "reward": None}
    if not statuses <= {"ok", "submission_failure"}:
        return {"status": "infrastructure_failure", "reason": "invalid status in formal receipt",
                "E": None, "score": None, "reward": None}
    if len(records) != FORMAL_TOTAL or len(set(keys)) != FORMAL_TOTAL or set(keys) != expected_keys:
        return {"status": "infrastructure_failure", "reason": "formal roster does not match frozen manifest",
                "E": None, "score": None, "reward": None}
    failures = [row for row in records if str(row.get("status", "ok")) == "submission_failure"]
    if failures:
        return {
            "status": "submission_failure", "reason": "formal scene failure",
            "failed_scenes": [{"dataset": r.get("dataset"), "scene": r.get("scene"),
                                "reason": r.get("reason")} for r in failures],
            "E": None, "score": 0.0, "reward": 0.0,
        }
    errors = []
    try:
        for row in records:
            errors.append(scene_error(row[name] for name in CHANNELS))
    except (KeyError, TypeError, ValueError):
        return {"status": "submission_failure", "reason": "invalid formal scene metrics",
                "E": None, "score": 0.0, "reward": 0.0}
    error = math.fsum(errors) / FORMAL_TOTAL
    score = score_from_error(error, classic, advanced)
    return {
        "status": "ok", "E": error, "score": score, "reward": score / 10.0,
        "classic": float(classic), "advanced": float(advanced),
        "scene_count": FORMAL_TOTAL,
        "dataset_counts": dict(FORMAL_COUNTS),
    }


def read_per_scene_csv(path: Path, dataset: str, method: str | None = None) -> list[dict]:
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if method is not None and row.get("method") != method:
                continue
            values = {name: float(row[name]) for name in CHANNELS}
            rows.append({"dataset": dataset, "scene": str(row["scene"]),
                         "status": row.get("status", "ok"), **values,
                         "reason": row.get("reason", "")})
    return rows
