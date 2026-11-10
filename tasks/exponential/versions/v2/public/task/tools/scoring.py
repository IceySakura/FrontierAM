"""V2 score for externally verified, complete submissions; not an isolation layer.

For each scored workload, r = candidate wall time / frozen reference wall time.
Average workloads within mother, mothers within family, families within scenario,
scenarios within precision, then precisions equally. Only after these five levels
is the complete-submission score mapped to 100 / (1 + R). There is no time floor,
memory utility, or build-cost amortization. Any mandatory case/control failure
makes the entire submission invalid with score zero.

The controller must run three fresh repetitions, independently verify every
output, and measure total execution-group resources. Successful aggregates must
contain valid=True, repeat_count=3, median time_seconds, maximum memory_bytes,
max_wall_seconds, and max_cpu_seconds. A one_shot repetition includes its complete
single request; a reuse repetition includes preparation once plus all 16 requests.
An aggregate is trusted evidence supplied by the evaluator, NEVER by the solver.
This module cannot establish accuracy, repetitions, provenance, or containment.

The canonical manifest defines the cells and denominators. It may contain a
subset of the possible scenario/precision grid; dataset completeness belongs to
manifest qualification, not this scorer. All manifest cases are mandatory.
Duplicate JSON keys must be rejected before constructing measurement dictionaries.
"""

import math
import numbers
from decimal import Decimal, localcontext


REPEATS = 3
MEMORY_CAP_BYTES = 4 * 1024**3
SCENARIO_LIMITS = {
    "one_shot": {"wall_seconds": 60.0, "cpu_seconds": 60.0, "requests": 1},
    "reuse": {"wall_seconds": 240.0, "cpu_seconds": 240.0, "requests": 16},
}
PRECISIONS = frozenset({"p4", "p7", "p10"})
GROUP_FIELDS = ("precision", "scenario", "family", "mother")


def _error(reason):
    raise ValueError(f"evaluator_error:{reason}")


def _number(value, *, positive=True):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    try:
        value = float(value)
    except (ValueError, OverflowError):
        return False
    return math.isfinite(value) and (value > 0 if positive else value >= 0)


def validate_measurement(record, *, scenario):
    """Validate a trusted aggregate; the external controller verifies its facts.

    max_wall_seconds and max_cpu_seconds enforce the cap on EACH repetition;
    a legal median cannot hide one over-budget repetition. memory_bytes is the
    maximum execution-group peak across the same three repetitions. Optional
    validation flags/resource fields are checked when supplied, never invented.
    """
    if not isinstance(scenario, str) or scenario not in SCENARIO_LIMITS:
        return _error("unknown_scenario")
    if not isinstance(record, dict):
        return {"valid": False, "reason": "missing_or_malformed_record"}
    if type(record.get("valid")) is not bool:
        return {"valid": False, "reason": "valid_flag_must_be_boolean"}
    if not record["valid"]:
        reason = record.get("reason", "reported_failure")
        return {"valid": False,
                "reason": reason if isinstance(reason, str) else "reported_failure"}
    if type(record.get("repeat_count")) is not int or record["repeat_count"] != REPEATS:
        return {"valid": False, "reason": "expected_three_repetitions"}
    if "build_cost_applied" in record:
        return {"valid": False, "reason": "legacy_build_accounting_not_supported"}
    for field in ("time_seconds", "memory_bytes", "max_wall_seconds"):
        if not _number(record.get(field)):
            return {"valid": False, "reason": f"invalid_{field}"}
    if not _number(record.get("max_cpu_seconds"), positive=False):
        return {"valid": False, "reason": "invalid_max_cpu_seconds"}
    limits = SCENARIO_LIMITS[scenario]
    if float(record["max_wall_seconds"]) < float(record["time_seconds"]):
        return {"valid": False, "reason": "median_exceeds_max_wall"}
    if float(record["max_wall_seconds"]) > limits["wall_seconds"]:
        return {"valid": False, "reason": "wall_time_limit"}
    if float(record["max_cpu_seconds"]) > limits["cpu_seconds"]:
        return {"valid": False, "reason": "cpu_time_limit"}
    if float(record["memory_bytes"]) > MEMORY_CAP_BYTES:
        return {"valid": False, "reason": "memory_limit"}
    for flag in ("all_repeats_valid", "accuracy_pass", "finite_output", "shape_valid"):
        if flag in record and record[flag] is not True:
            return {"valid": False, "reason": f"failed_{flag}"}
    if "status" in record and record["status"] != "ok":
        return {"valid": False, "reason": "non_ok_status"}
    if "request_count" in record:
        if (type(record["request_count"]) is not int
                or record["request_count"] != limits["requests"]):
            return {"valid": False, "reason": "wrong_request_count"}
    if "cpu_seconds" in record:
        if not _number(record["cpu_seconds"], positive=False):
            return {"valid": False, "reason": "invalid_cpu_seconds"}
        if float(record["cpu_seconds"]) > float(record["max_cpu_seconds"]):
            return {"valid": False, "reason": "cpu_exceeds_reported_max"}
    if "max_memory_bytes" in record:
        if (not _number(record["max_memory_bytes"])
                or float(record["max_memory_bytes"]) != float(record["memory_bytes"])):
            return {"valid": False, "reason": "inconsistent_peak_memory"}
    return {"valid": True, "reason": "passed"}


def _canonical_cases(cases):
    if not isinstance(cases, list) or not cases:
        return _error("canonical_cases_must_be_nonempty_list")
    canonical, mother_families = {}, {}
    for case in cases:
        if not isinstance(case, dict):
            return _error("malformed_canonical_case")
        for field in ("id", "family", "mother"):
            if not isinstance(case.get(field), str) or not case[field]:
                return _error(f"invalid_canonical_{field}")
        if case["id"] in canonical:
            return _error("duplicate_canonical_id")
        if case.get("role") not in ("scored", "control"):
            return _error("invalid_canonical_role")
        if (not isinstance(case.get("scenario"), str)
                or case["scenario"] not in SCENARIO_LIMITS):
            return _error("invalid_canonical_scenario")
        if (not isinstance(case.get("precision"), str)
                or case["precision"] not in PRECISIONS):
            return _error("invalid_canonical_precision")
        if "n" in case and (type(case["n"]) is not int or case["n"] < 1):
            return _error("invalid_canonical_n")
        if (case["mother"] in mother_families
                and mother_families[case["mother"]] != case["family"]):
            return _error("mother_assigned_to_multiple_families")
        mother_families[case["mother"]] = case["family"]
        canonical[case["id"]] = case
    if not any(c["role"] == "scored" for c in cases):
        return _error("at_least_one_scored_case_required")
    return canonical


def _finite_float(value):
    """JSON-safe representation; a mathematically finite ratio may exceed float."""
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _hierarchy(cases, ratios, valid, depth=0):
    """Return Decimal relative time, float coverage, and explanatory tree."""
    if depth == len(GROUP_FIELDS):
        complete = all(valid[c["id"]] for c in cases)
        coverage = math.fsum(float(valid[c["id"]]) for c in cases) / len(cases)
        ratio = (sum((ratios[c["id"]] for c in cases), Decimal(0)) / len(cases)
                 if complete else None)
        children = None
    else:
        field = GROUP_FIELDS[depth]
        groups = {}
        for case in cases:
            groups.setdefault(case[field], []).append(case)
        computed = {key: _hierarchy(group, ratios, valid, depth + 1)
                    for key, group in sorted(groups.items())}
        complete = all(item[2]["complete"] for item in computed.values())
        coverage = math.fsum(item[1] for item in computed.values()) / len(computed)
        ratio = (sum((item[0] for item in computed.values()), Decimal(0)) / len(computed)
                 if complete else None)
        children = {key: item[2] for key, item in computed.items()}
    shown = _finite_float(ratio)
    summary = {"relative_time": shown,
               "relative_time_overflow": ratio is not None and shown is None,
               "coverage": coverage, "complete": complete, "case_count": len(cases)}
    if children is not None:
        summary["grouped_by"] = GROUP_FIELDS[depth]
        summary["groups"] = children
    else:
        summary["case_ids"] = [c["id"] for c in cases]
    return ratio, coverage, summary


def score_submission(cases, candidate, reference):
    """Score fixed mandatory cases using externally authoritative aggregates.

    cases: list of {id, family, mother, scenario, precision, role}, optional n.
    candidate/reference: dictionaries keyed by exactly those canonical IDs.
    Missing/malformed/extra candidate records fail the submission, score zero.
    Missing/malformed/extra reference records and malformed canonical manifests
    raise ValueError('evaluator_error:...'), even if the candidate also fails.

    coverage uses only scored cases with the same five-level weights as R;
    controls_valid and per-case verdicts separately expose mandatory controls.
    An incomplete group has no relative-time estimate: successful-only averages
    must not substitute for the prescribed fixed denominator. Memory remains a
    separate per-case diagnostic/hard constraint, with no nonregression rule.
    """
    canonical = _canonical_cases(cases)
    if not isinstance(reference, dict):
        return _error("reference_must_be_case_dictionary")
    if set(reference) != set(canonical):
        return _error("reference_id_set_mismatch")
    for case_id, case in canonical.items():
        result = validate_measurement(reference[case_id], scenario=case["scenario"])
        if not result["valid"]:
            return _error(f"invalid_reference:{case_id}:{result['reason']}")

    malformed_candidate = not isinstance(candidate, dict)
    supplied = {} if malformed_candidate else candidate
    unexpected_ids = set(supplied) - set(canonical)
    verdicts = {case_id: validate_measurement(supplied.get(case_id), scenario=c["scenario"])
                for case_id, c in canonical.items()}
    valid = {case_id: result["valid"] for case_id, result in verdicts.items()}
    controls_valid = all(valid[c["id"]] for c in cases if c["role"] == "control")
    complete = (all(valid.values()) and not malformed_candidate and not unexpected_ids)

    # Decimal prevents finite positive clocks with extreme scale differences
    # from producing an Infinity/NaN JSON value or an overflow in aggregation.
    with localcontext() as context:
        context.prec = 50
        ratios, per_case = {}, {}
        for case_id, case in canonical.items():
            record, ref = supplied.get(case_id), reference[case_id]
            ratio = None
            if valid[case_id]:
                ratio = (Decimal.from_float(float(record["time_seconds"]))
                         / Decimal.from_float(float(ref["time_seconds"])))
                ratios[case_id] = ratio
            shown = _finite_float(ratio)
            per_case[case_id] = {
                **verdicts[case_id], "role": case["role"],
                "relative_time": shown,
                "relative_time_overflow": ratio is not None and shown is None,
                "time_seconds": float(record["time_seconds"]) if valid[case_id] else None,
                "memory_bytes": float(record["memory_bytes"]) if valid[case_id] else None,
                "reference_time_seconds": float(ref["time_seconds"]),
                "reference_memory_bytes": float(ref["memory_bytes"]),
            }
        scored = [case for case in cases if case["role"] == "scored"]
        ratio, coverage, tree = _hierarchy(scored, ratios, valid)
        score = float(Decimal(100) / (Decimal(1) + ratio)) if complete else 0.0
    if malformed_candidate:
        reason = "candidate_must_be_case_dictionary"
    elif unexpected_ids:
        reason = "unexpected_candidate_ids"
    elif not complete:
        reason = "mandatory_case_failure"
    else:
        reason = "complete_valid_submission"
    return {
        "status": "ok" if complete else "invalid_submission",
        "score": score, "complete": complete, "coverage": coverage,
        "relative_time": tree["relative_time"] if complete else None,
        "relative_time_overflow": tree["relative_time_overflow"] if complete else False,
        "controls_valid": controls_valid,
        "case_results": per_case,
        "failed_cases": {case_id: verdict["reason"] for case_id, verdict in verdicts.items()
                         if not verdict["valid"]},
        "unexpected_candidate_id_count": len(unexpected_ids),
        "hierarchy": tree, "reason": reason,
        "policy": {
            "formula": "100/(1+hierarchical_mean(T/Tref))",
            "aggregation_inner_to_outer": ["workload", "mother", "family", "scenario", "precision"],
            "weights": "equal at every level over the fixed canonical manifest",
            "memory_is_hard_constraint_only": True,
            "any_mandatory_failure_score": 0.0,
            "repeats": REPEATS,
            "research_superiority_inferred": False,
        },
    }
