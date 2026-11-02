"""RuleDetector base and the record -> TrajectoryView conversion.

``trajectory_view`` is copied line for line from
``bpo.core.sample.NormalizedSample.as_trajectory``; keep it that way so
verdicts match the upstream engine.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .detector import SEVERITY_ORDER, Finding
from .messages import Message, TrajectoryView, messages_from_record
from .tool_registry import build_tool_registry_timeline


def _count_tool_calls(messages: list[Message]) -> int:
    return sum(len(m.tool_calls) for m in messages)


def _explicit_nonnegative_int(value: Any) -> Optional[int]:
    """Normalize an explicitly exported integer without inventing a default."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 and value.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def trajectory_view(record: Dict[str, Any], sample_id: str = "") -> TrajectoryView:
    messages = messages_from_record(record)
    rec = record if isinstance(record, dict) else {}
    req = rec.get("request") if isinstance(rec.get("request"), dict) else {}
    end_status = (
        rec.get("end_status")
        or rec.get("finish_reason")
        or (req.get("end_status") if isinstance(req, dict) else None)
        or ""
    )
    finish_reason = rec.get("finish_reason") or ""
    extra = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
    sampling = (
        extra.get("sampling") if isinstance(extra.get("sampling"), dict) else {}
    )
    eval_detail = (
        extra.get("eval_detail")
        if isinstance(extra.get("eval_detail"), dict)
        else {}
    )
    request_tools = req.get("tools") if isinstance(req, dict) else None
    top_level_tools = rec.get("tools")
    tools_available = (
        list(request_tools)
        if isinstance(request_tools, list)
        else list(top_level_tools)
        if isinstance(top_level_tools, list)
        else []
    )
    generated_turn_index = _explicit_nonnegative_int(
        rec.get("generated_turn_index")
    )
    if generated_turn_index is not None and generated_turn_index >= len(messages):
        generated_turn_index = None
    return TrajectoryView(
        sample_id=sample_id,
        messages=messages,
        model=str(rec.get("model") or ""),
        tools_available=tools_available,
        tool_registry=build_tool_registry_timeline(rec, messages),
        finish_reason=str(finish_reason),
        end_status=str(end_status),
        n_tool_calls=_count_tool_calls(messages),
        infra_failure=bool(extra.get("infra_failure")),
        infra_failure_reason=str(extra.get("infra_failure_reason") or ""),
        infra_failure_detail=str(extra.get("infra_failure_detail") or ""),
        infra_failure_evidence=(
            dict(extra["infra_failure_evidence"])
            if isinstance(extra.get("infra_failure_evidence"), dict) else {}
        ),
        configured_max_iterations=_explicit_nonnegative_int(
            sampling.get("max_iterations")
        ),
        actual_iterations=_explicit_nonnegative_int(eval_detail.get("iterations")),
        evaluator_exit_status=(
            str(eval_detail.get("exit_status") or "").strip()
        ),
        generated_turn_index=generated_turn_index,
    )


class RuleDetector(ABC):
    detector_type = "rule"
    detector_id: str = ""
    thresholds: Dict[str, Any] = {}

    @abstractmethod
    def scan(self, view: TrajectoryView) -> List[Finding]:
        """Walk the view and return zero or more findings."""

    def detect_view(self, view: TrajectoryView) -> Dict[str, Any]:
        findings = self.scan(view)
        bad = [f for f in findings if f.is_issue]
        severity = (
            sorted({f.severity for f in bad}, key=lambda s: SEVERITY_ORDER.get(s, 9))[0]
            if bad else None
        )
        return {
            "is_issue": bool(bad),
            "severity": severity,
            "findings": [f.to_dict() for f in findings],
        }

    def detect(self, record: Dict[str, Any], sample_id: str = "") -> Dict[str, Any]:
        return self.detect_view(trajectory_view(record, sample_id))
