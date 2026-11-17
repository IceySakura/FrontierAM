"""Pure, auditable aggregation and reward arithmetic for QP benchmark runs."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping


SCORING_VERSION = "frontieram-pdhcg-qp-v1"
OUTCOMES = {"accepted", "algorithm_failure", "evaluation_error"}


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _case(case: Any, limits: Mapping[str, float]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(case, dict):
        return {"case_id": None, "group": None, "valid": False, "success": None,
                "median_seconds": None, "penalized_seconds": None,
                "errors": ["case must be an object"]}
    case_id, group = case.get("case_id"), case.get("group")
    if not isinstance(case_id, str) or not case_id.strip():
        errors.append("case_id must be nonempty")
    if not isinstance(group, str) or not group.strip():
        errors.append("group must be nonempty")
    limit = limits.get(case_id) if isinstance(case_id, str) else None
    if limit is None:
        errors.append("missing time limit")
    repeats = case.get("repeats")
    if not isinstance(repeats, list) or len(repeats) != 3:
        errors.append("exactly three repeats are required")
        repeats = repeats if isinstance(repeats, list) else []
    numbers: list[int] = []
    accepted_times: list[float] = []
    statuses: list[str] = []
    accepted = 0
    for repeat in repeats:
        if not isinstance(repeat, dict):
            errors.append("repeat must be an object")
            continue
        number = repeat.get("repeat")
        if type(number) is not int:
            errors.append("repeat number must be an integer")
        else:
            numbers.append(number)
        outcome = repeat.get("outcome")
        if outcome not in OUTCOMES:
            errors.append("invalid outcome")
        if outcome == "evaluation_error":
            errors.append("evaluation_error: evaluator must repair before scoring")
            continue
        try:
            elapsed = _number(repeat.get("elapsed_seconds"), "elapsed_seconds")
            if elapsed <= 0:
                raise ValueError("elapsed_seconds must be positive")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if outcome == "accepted":
            accepted += 1
            accepted_times.append(elapsed)
            status = repeat.get("status")
            if status != "solved":
                errors.append("accepted status must be solved")
            else:
                statuses.append(status)
            if limit is not None and elapsed > limit + 1e-12:
                errors.append("accepted repeat exceeds time limit")
    if sorted(numbers) != [1, 2, 3]:
        errors.append("repeat numbers must be exactly 1,2,3")
    valid = not errors
    success = valid and accepted == 3 and len(statuses) == 3
    median = sorted(accepted_times)[1] if len(accepted_times) == 3 else None
    penalized = median if success else limit
    return {"case_id": case_id if isinstance(case_id, str) else None,
            "group": group if isinstance(group, str) else None, "valid": valid,
            "success": success if valid else None, "median_seconds": median,
            "penalized_seconds": penalized, "errors": errors}


def aggregate_cases(cases: Any, *, expected_case_ids: list[str],
                    time_limits: Mapping[str, float]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(cases, list):
        errors.append("cases must be a list")
        cases = []
    expected = list(expected_case_ids)
    if len(set(expected)) != len(expected) or not expected:
        errors.append("expected_case_ids must be unique and nonempty")
    rows = [_case(case, time_limits) for case in cases]
    seen = Counter(row["case_id"] for row in rows)
    for row in rows:
        if row["case_id"] not in set(expected):
            row["valid"] = False
            row["errors"].append("unexpected case_id")
        if seen[row["case_id"]] > 1:
            row["valid"] = False
            row["errors"].append("duplicate case_id")
    for case_id in expected:
        if seen[case_id] == 0:
            rows.append({"case_id": case_id, "group": None, "valid": False,
                         "success": None, "median_seconds": None,
                         "penalized_seconds": None, "errors": ["missing case"]})
    for row in rows:
        errors.extend(f"{row['case_id']}: {error}" for error in row["errors"])
    valid = not errors and len(rows) == len(expected)
    if not valid:
        return {"valid": False, "p": None, "g": None, "cases": rows,
                "groups": {}, "errors": errors, "scoring_version": SCORING_VERSION}
    successes = sum(row["success"] is True for row in rows)
    times = [row["penalized_seconds"] for row in rows]
    if any(not isinstance(t, (int, float)) or t <= 0 for t in times):
        return {"valid": False, "p": None, "g": None, "cases": rows,
                "groups": {}, "errors": ["invalid penalized time"],
                "scoring_version": SCORING_VERSION}
    g = math.exp(math.fsum(math.log(t) for t in times) / len(times))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    group_summary = {}
    for group, items in sorted(groups.items()):
        group_summary[group] = {"p": sum(i["success"] is True for i in items) / len(items),
                                "count": len(items),
                                "successful": sum(i["success"] is True for i in items)}
    return {"valid": True, "p": successes / len(expected), "g": g, "cases": rows,
            "groups": group_summary, "errors": [], "scoring_version": SCORING_VERSION}


def compute_reward(summary: Mapping[str, Any], calibration: Mapping[str, Any]) -> dict[str, float]:
    if not summary.get("valid"):
        raise ValueError("cannot score an invalid assessment")
    p = _number(summary.get("p"), "p")
    g = _number(summary.get("g"), "g")
    p_ref = _number(calibration.get("p_ref"), "p_ref")
    g_ref = _number(calibration.get("g_ref"), "g_ref")
    if not 0 <= p <= 1 or p_ref != 1 or not 0 < g_ref or not 0 < g:
        raise ValueError("invalid calibration or summary")
    # Aggregate all runs before mapping. Reference = 50/100; all solved with
    # vanishing runtime approaches 100. Failed cases retain their full deadline.
    ratio = g / g_ref
    reward = p / (1.0 + ratio)
    return {"reward": reward, "score": 100.0 * reward, "success_rate": p, "time_ratio": ratio,
            "p_ref": p_ref, "g_ref": g_ref}
