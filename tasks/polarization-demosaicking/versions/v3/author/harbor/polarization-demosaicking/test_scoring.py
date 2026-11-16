#!/usr/bin/env python3
from __future__ import annotations

from scoring import FORMAL_ROSTER, aggregate, score_from_error


B, H = 2.0, 1.0


def rows(value: float):
    roster = [("qiu", scene) for scene in FORMAL_ROSTER["qiu"]]
    roster += [("wen", scene) for scene in FORMAL_ROSTER["wen"]]
    return [{"dataset": dataset, "scene": scene,
             "status": "ok", "rmse_0": value, "rmse_45": value,
             "rmse_90": value, "rmse_135": value} for dataset, scene in roster]


def main():
    assert score_from_error(B, B, H) == 0.0
    assert score_from_error(H, B, H) == 5.0
    assert score_from_error(0, B, H) == 10.0
    out = aggregate(rows(1.0), B, H)
    assert out["E"] == 1.0 and out["score"] == 5.0 and out["reward"] == 0.5
    mixed = rows(0.0)
    mixed[-1]["status"] = "submission_failure"
    failed = aggregate(mixed, B, H)
    assert failed["score"] == 0.0 and failed["reward"] == 0.0
    infra = rows(0.0)
    infra[0]["status"] = "infrastructure_failure"
    infra_result = aggregate(infra, B, H)
    assert infra_result["status"] == "infrastructure_failure" and infra_result["score"] is None
    wrong_roster = rows(1.0)
    wrong_roster[0]["scene"] = "bogus"
    wrong_result = aggregate(wrong_roster, B, H)
    assert wrong_result["status"] == "infrastructure_failure"
    print("vnext scoring checks passed")


if __name__ == "__main__":
    main()
