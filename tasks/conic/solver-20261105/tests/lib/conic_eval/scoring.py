"""Pure scoring-v1 arithmetic over independently validated runner records.

This module never reads candidate outputs or verifies mathematical certificates.
Only the trusted runner may assign ``accepted`` after independent verification,
resource checks, and result-file auditing. Candidate self-reported statuses must
not be passed through as accepted outcomes.

aggregate_cases schema:
  [{"case_id": nonempty str, "group": nonempty str, "repeats": [
      {"repeat": 1|2|3,
       "outcome": "accepted"|"algorithm_failure"|"evaluation_error",
       "status": "solved"|"primal_infeasible"|"dual_infeasible"|"unresolved"|None,
       "elapsed_seconds": positive finite number}, ...]}, ...]

Accepted records require a successful unified status and elapsed <= 1000.
Failure/error records may omit status. An evaluation_error may omit elapsed,
because no monitored process may have started. Algorithm failures require a
measured elapsed time; watchdog overshoot above 1000 is permitted for them.
Additional diagnostic fields are ignored. Inputs are never mutated.

Pass expected_case_ids from the fixed manifest to detect omitted whole cases;
without it the caller is responsible for supplying the entire planned set.
The function supports development subsets; the formal runner must enforce the
650-case held-out manifest. Invalid evaluations have p=G=None, never score zero.
Submission/build errors are the caller's separate whole-submission exception:
S=0 with p/G/C/V=None, not invented per-instance records.
"""

from collections import Counter, defaultdict
import math
import statistics

SCORING_VERSION = "scoring-v1"
TIME_LIMIT_SECONDS = 1000.0
REQUIRED_REPEATS = 3
ACCEPTED_STATUSES = frozenset({"solved", "primal_infeasible", "dual_infeasible"})
OUTCOMES = frozenset({"accepted", "algorithm_failure", "evaluation_error"})


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def compute_score(p, G, p_B, B):
    """Return unrounded {C,V,S}; invalid metrics/calibration raise ValueError.

    Calibration requires 0 < p_B < 1 and 0 < B <= 250. Metrics require
    0 <= p <= 1 and 0 <= G <= 1000. The ideal G=0 endpoint is allowed here,
    although actual accepted process measurements must be strictly positive.
    The caller must obtain p/G from a valid aggregate and p_B/B from its frozen
    calibration; this arithmetic function does not establish their provenance.
    """
    p, G, p_B, B = (_number(v, k) for k, v in (("p", p), ("G", G), ("p_B", p_B), ("B", B)))
    if not 0 <= p <= 1 or not 0 <= G <= TIME_LIMIT_SECONDS:
        raise ValueError("Metrics require 0 <= p <= 1 and 0 <= G <= 1000")
    if not 0 < p_B < 1 or not 0 < B <= 250:
        raise ValueError("Calibration requires 0 < p_B < 1 and 0 < B <= 250")
    C = 5 * (p / p_B) if p <= p_B else 5 + 5 * ((p - p_B) / (1 - p_B))
    if G <= B:
        V = 10 - 5 * (math.log1p(G) / math.log1p(B))
    elif G < 4 * B:
        # Difference of log1p avoids cancellation in log((1+4B)/(1+G)).
        high = math.log1p(4 * B)
        V = 5 * ((high - math.log1p(G)) / (high - math.log1p(B)))
    else:
        V = 0.0
    return {"C": C, "V": V, "S": 0.6 * C + 0.4 * V}


def _case_statistics(case, index):
    errors, elapsed, statuses, outcomes, numbers = [], [], [], Counter(), []
    if not isinstance(case, dict):
        case = {}
        errors.append("case must be an object")
    case_id, group = case.get("case_id"), case.get("group")
    for name, value in (("case_id", case_id), ("group", group)):
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{name} must be a nonempty string")
    repeats = case.get("repeats")
    if not isinstance(repeats, list):
        errors.append("repeats must be a list")
        repeats = []
    if len(repeats) != REQUIRED_REPEATS:
        errors.append("exactly three repeat records are required")
    for i, record in enumerate(repeats):
        prefix = f"repeat record {i}"
        if not isinstance(record, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        number, outcome, status = record.get("repeat"), record.get("outcome"), record.get("status")
        if type(number) is not int or number not in {1, 2, 3}:
            errors.append(f"{prefix}: repeat must be integer 1, 2 or 3")
        else:
            numbers.append(number)
        if not isinstance(outcome, str) or outcome not in OUTCOMES:
            errors.append(f"{prefix}: invalid outcome")
        else:
            outcomes[outcome] += 1
        if status is not None and (not isinstance(status, str) or status not in ACCEPTED_STATUSES | {"unresolved"}):
            errors.append(f"{prefix}: invalid unified status")
        if outcome == "accepted":
            if not isinstance(status, str) or status not in ACCEPTED_STATUSES:
                errors.append(f"{prefix}: accepted requires a successful unified status")
            else:
                statuses.append(status)
        if outcome == "evaluation_error":
            errors.append(f"{prefix}: evaluation_error invalidates assessment")
        value = record.get("elapsed_seconds")
        if value is None and outcome == "evaluation_error":
            continue
        try:
            seconds = _number(value, "elapsed_seconds")
            if seconds <= 0:
                raise ValueError("elapsed_seconds must be positive")
            if outcome == "accepted" and seconds > TIME_LIMIT_SECONDS:
                raise ValueError("accepted elapsed_seconds exceeds 1000")
            elapsed.append(seconds)
        except ValueError as error:
            errors.append(f"{prefix}: {error}")
    if len(numbers) != 3 or set(numbers) != {1, 2, 3}:
        errors.append("repeat numbers must be exactly {1,2,3}, with no duplicates")
    valid = not errors
    consistent = len(statuses) == 3 and len(set(statuses)) == 1
    success = valid and outcomes["accepted"] == 3 and consistent
    return {"case_id": case_id if isinstance(case_id, str) else None,
        "group": group if isinstance(group, str) else None, "valid_assessment": valid,
        "success": success if valid else None, "status": statuses[0] if success else None,
        "reason": "invalid_record" if errors else "accepted" if success else
                  "inconsistent_accepted_statuses" if outcomes["accepted"] == 3 else "algorithm_failure",
        "repeats_observed": len(repeats), "repeat_outcomes": dict(outcomes),
        "elapsed_seconds": elapsed,
        "median_elapsed_seconds": statistics.median(elapsed) if len(elapsed) == 3 else None,
        "penalized_seconds": statistics.median(elapsed) if success else TIME_LIMIT_SECONDS if valid else None,
        "errors": [f"case {case_id!r} (index {index}): {error}" for error in errors]}


def _summary(rows, *, errors=(), total_cases=None):
    total = len(rows) if total_cases is None else total_cases
    successes = sum(row["success"] is True for row in rows)
    failures = sum(row["valid_assessment"] and row["success"] is False for row in rows)
    valid = bool(rows) and not errors and all(row["valid_assessment"] for row in rows)
    G = None
    if valid:
        times = [row["penalized_seconds"] for row in rows]
        # Preserve exact equal-time endpoints, notably all failures -> G=1000.
        G = times[0] if all(t == times[0] for t in times) else math.expm1(
            math.fsum(math.log1p(t) for t in times) / total)
        G = min(TIME_LIMIT_SECONDS, max(0.0, G))  # Floating-point endpoint noise only.
    return {"valid_assessment": valid, "p": successes / total if valid else None, "G": G,
        "counts": {"total_cases": total, "successful_cases": successes,
            "algorithm_failed_cases": failures, "invalid_cases": total - successes - failures},
        "successful_status_counts": dict(Counter(row["status"] for row in rows if row["success"] is True))}


def aggregate_cases(cases, *, expected_case_ids=None):
    """Aggregate all planned cases, returning clear invalid/null metrics on errors.

    Result keys: valid_assessment, p, G, counts, successful_status_counts,
    cases, groups, errors, scoring_version. Individual case rows expose measured
    median and penalized_seconds separately. Group summaries are diagnostics,
    never group-weighted inputs to the overall score.
    """
    errors = []
    if not isinstance(cases, list):
        errors.append("cases must be a list")
        cases = []
    if not cases:
        errors.append("at least one planned case is required")
    expected = None
    if expected_case_ids is not None:
        try:
            if isinstance(expected_case_ids, (str, bytes)):
                raise ValueError("expected_case_ids must be a collection of IDs")
            expected_list = list(expected_case_ids)
            if not expected_list or any(not isinstance(x, str) or not x.strip() for x in expected_list):
                raise ValueError("expected_case_ids must contain nonempty strings")
            if len(set(expected_list)) != len(expected_list):
                raise ValueError("expected_case_ids contains duplicate IDs")
            expected = set(expected_list)
        except (TypeError, ValueError) as error:
            errors.append(str(error))
    rows = [_case_statistics(case, i) for i, case in enumerate(cases)]
    ids = Counter(row["case_id"] for row in rows if row["case_id"] is not None)
    for row in rows:
        issue = None
        if row["case_id"] is not None and ids[row["case_id"]] > 1:
            issue = f"duplicate case_id: {row['case_id']}"
        elif expected is not None and row["case_id"] not in expected:
            issue = f"unexpected case_id: {row['case_id']}"
        if issue:
            row.update(valid_assessment=False, success=None, status=None, penalized_seconds=None, reason="invalid_record")
            row["errors"].append(issue)
    if expected is not None:
        for case_id in sorted(expected - ids.keys()):
            row = _case_statistics({"case_id": case_id, "repeats": []}, len(rows))
            row["errors"] = [f"missing planned case: {case_id}"]
            rows.append(row)
    errors.extend(error for row in rows for error in row["errors"])
    total = len(expected) if expected is not None else len(rows)
    result = _summary(rows, errors=errors, total_cases=total)
    result["counts"].update(expected_cases=len(expected) if expected is not None else None, received_cases=len(cases))
    groups = defaultdict(list)
    for row in rows:
        if row["group"]:
            groups[row["group"]].append(row)
    result.update(scoring_version=SCORING_VERSION, cases=rows,
                  groups={group: _summary(items) for group, items in sorted(groups.items())}, errors=errors)
    return result
