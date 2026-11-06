"""Success-only scoring with frozen, per-instance time budgets.

Only trusted runner records may enter this module. The runner must first mark
timeouts as algorithm failures; an ``accepted`` record beyond its frozen budget
is contradictory evidence and invalidates the assessment. Historical scoring-v1
record validation is reused unchanged, but its time metric is never returned.
"""

from collections import defaultdict
from collections.abc import Mapping

from .scoring import _number, _summary, aggregate_cases as _aggregate_v1

SCORING_VERSION = "success-v2"
MIN_TIME_LIMIT_SECONDS = 1.0
MAX_TIME_LIMIT_SECONDS = 300.0
TIME_ALLOWANCE = 1.2


def compute_score(p, p_B):
    """Return {S}; the caller supplies a verified success rate and frozen anchor."""
    p, p_B = _number(p, "p"), _number(p_B, "p_B")
    if not 0 <= p <= 1 or not 0 < p_B < 1:
        raise ValueError("Require 0 <= p <= 1 and 0 < p_B < 1")
    if p == p_B:
        score = 5.0
    elif p <= 0.9 * p_B:
        score = 0.0
    elif p < p_B:
        # Divide before subtracting to avoid underflow for very small anchors.
        score = 5 * ((p / p_B - 0.9) / 0.1)
    else:
        score = 5 + 5 * ((p - p_B) / (1 - p_B))
    return {"S": score}


def time_limit_for_reference(success, elapsed_seconds):
    """Derive a budget from three historical, independently checked repeats."""
    if type(success) is not bool:
        raise ValueError("success must be a boolean")
    if not isinstance(elapsed_seconds, (list, tuple)) or len(elapsed_seconds) != 3:
        raise ValueError("Exactly three reference elapsed times are required")
    times = [_number(value, "elapsed_seconds") for value in elapsed_seconds]
    if any(value <= 0 for value in times):
        raise ValueError("Reference elapsed times must be positive")
    if not success:
        return MAX_TIME_LIMIT_SECONDS
    return min(MAX_TIME_LIMIT_SECONDS, max(MIN_TIME_LIMIT_SECONDS,
                                          TIME_ALLOWANCE * max(times)))


def aggregate_cases(cases, *, expected_case_ids=None, time_limits):
    """Validate all three repeats and aggregate success, without a time score.

    ``time_limits`` maps case IDs to frozen budgets in [1, 300] seconds. Extra
    entries allow a development subset to use the complete budget mapping.
    Invalid records/configuration return valid_assessment=False and p=None.
    All elapsed-time fields are diagnostics; only p enters compute_score.
    """
    result = _aggregate_v1(cases, expected_case_ids=expected_case_ids)
    budget_errors, budgets = [], {}
    if not isinstance(time_limits, Mapping):
        budget_errors.append("time_limits must be a mapping of case IDs to budgets")
    else:
        for case_id, value in time_limits.items():
            try:
                if not isinstance(case_id, str) or not case_id.strip():
                    raise ValueError("time limit case ID must be a nonempty string")
                limit = _number(value, "time_limit_seconds")
                if not MIN_TIME_LIMIT_SECONDS <= limit <= MAX_TIME_LIMIT_SECONDS:
                    raise ValueError("time_limit_seconds must be between 1 and 300")
                budgets[case_id] = limit
            except ValueError as error:
                budget_errors.append(f"budget {case_id!r}: {error}")

    supplied = cases if isinstance(cases, list) else []
    for index, row in enumerate(result["cases"]):
        limit = budgets.get(row["case_id"])
        row["time_limit_seconds"] = limit
        issues = []
        if limit is None:
            issues.append(f"case {row['case_id']!r}: missing valid time budget")
        elif index < len(supplied) and isinstance(supplied[index], dict):
            records = supplied[index].get("repeats")
            for record in records if isinstance(records, list) else []:
                if not isinstance(record, dict) or record.get("outcome") != "accepted":
                    continue
                try:
                    elapsed = _number(record.get("elapsed_seconds"), "elapsed_seconds")
                except ValueError:
                    continue  # The unchanged record validator already reports this.
                if elapsed > limit:
                    issues.append(f"case {row['case_id']!r}, repeat {record.get('repeat')!r}: "
                                  f"accepted elapsed_seconds exceeds budget {limit}")
        if issues:
            row.update(valid_assessment=False, success=None, status=None,
                       penalized_seconds=None, reason="invalid_record")
            row["errors"].extend(issues)
            budget_errors.extend(issues)

    result["errors"].extend(budget_errors)
    total = result["counts"]["total_cases"]
    scope_counts = {key: result["counts"][key] for key in ("expected_cases", "received_cases")}
    result.update(_summary(result["cases"], errors=result["errors"], total_cases=total))
    result["counts"].update(scope_counts)
    groups = defaultdict(list)
    for row in result["cases"]:
        if row["group"]:
            groups[row["group"]].append(row)
    result["groups"] = {group: _summary(rows) for group, rows in sorted(groups.items())}
    result["scoring_version"] = SCORING_VERSION
    result.pop("G")
    for group in result["groups"].values():
        group.pop("G")
    for row in result["cases"]:
        row.pop("penalized_seconds")
    return result
