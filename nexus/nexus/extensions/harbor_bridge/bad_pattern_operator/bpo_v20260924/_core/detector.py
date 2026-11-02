"""``Detector`` Protocol + shared dataclasses.

Per §6 of ``docs/design.md`` (Wave 3a.5 split-ABC architecture):

- ``Detector`` is a *structural* :class:`typing.Protocol` — anything with
  ``detector_id``, ``detector_type``, ``manifest``, ``thresholds`` and a
  ``detect()`` method satisfies it. The runner uses
  ``isinstance(d, Detector)`` at registration time.
- ``RuleDetector`` (concrete ABC, single-stage) and ``LLMDetector``
  (concrete ABC, three-stage) are the two real superclasses. They both
  satisfy ``Detector`` but live in their own modules so each can carry
  workload-specific helpers.
- ``Finding`` is unchanged from the Wave 2 spike.
- ``Candidate`` is *new* in Wave 3a.5 — what an :class:`LLMDetector`
  ``prefilter()`` emits and what its ``judge_prompt()`` /
  ``finalize()`` consume.
- ``DetectorResult`` now carries:
    * ``detector_type`` — "rule" | "llm" — so the report can render
      stages differently per type.
    * ``stages`` — populated by RuleDetector with just ``{"final": ...}``
      and by LLMDetector with ``{"rule": ..., "llm": ..., "final": ...}``.

This module intentionally does NOT depend on ``rule_detector`` or
``llm_detector`` to avoid an import cycle. Subclasses live in those
modules and import these dataclasses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
)



Severity = Literal["high", "medium", "low", "candidate"]
SEVERITY_ORDER: Dict[str, int] = {
    "high": 0, "medium": 1, "low": 2, "candidate": 3,
}
VALID_SEVERITIES = frozenset(SEVERITY_ORDER.keys())

DetectorType = Literal["rule", "llm"]
VALID_DETECTOR_TYPES = frozenset({"rule", "llm"})


@dataclass(frozen=True)
class Finding:
    """A single quality issue found in a sample.

    The DQC successor to ``bpo.base_detector.Evidence``. The biggest
    cross-cutting rebrand-time change is ``is_bad_pattern`` →
    ``is_issue`` (option (b) in design_review §B2).
    """

    rule_id: str
    severity: Severity
    is_issue: bool
    description: str
    turn_index: Optional[int] = None
    confidence: float = 1.0
    quotes: List[Dict[str, Any]] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"Finding.severity must be one of {sorted(VALID_SEVERITIES)}, "
                f"got {self.severity!r} for rule_id={self.rule_id!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Candidate:
    """What an :class:`LLMDetector`'s ``prefilter()`` emits.

    The LLM judge consumes these one at a time;
    :meth:`LLMDetector.finalize` pairs each with a ``JudgeResult`` to
    produce a :class:`Finding` (or filter to ``None``).

    ``severity_hint`` is the rule prefilter's guess; the LLM judge may
    override. In ``--llm-mode skip``, ``finalize(c, verdict=None)`` uses
    ``severity_hint`` (falling back to ``"candidate"``) so that rule-only
    output still has reasonable severity stamps.
    """

    detector_id: str
    sample_id: str
    rule_id: str
    turn_index: Optional[int] = None
    quotes: List[Dict[str, Any]] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)
    severity_hint: Optional[Severity] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectorResult:
    """The output of one detector on one sample.

    For :class:`RuleDetector` results, ``stages`` carries only ``"final"``.
    For :class:`LLMDetector` results, ``stages`` carries ``"rule"``
    (candidates), ``"llm"`` (verdicts + cost), and ``"final"`` (findings).
    The two-phase runner emits intermediate :class:`DetectorResult`
    objects after phase 1 with ``stages.rule`` populated and
    ``stages.final`` empty, to be consumed by the LLM phase.
    """

    sample_id: str
    detector_id: str
    detector_type: DetectorType
    detector_version: str
    is_issue: bool
    severity: Optional[Severity]
    findings: List[Finding]
    n_messages: int
    thresholds: Dict[str, Any]
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error: Optional[str] = None
    # Populated by ``kind: indicator`` detectors instead of ``findings``:
    # a flat dict of per-trajectory measurements (counts / sizes / rates).
    # ``None`` for ordinary binary detectors. The run-level rollup
    # aggregates these across a run (see _overview.json ``metrics`` block).
    # See docs/design/detectors/indicators_alongside_detectors.md.
    metric: Optional[Dict[str, Any]] = None
    # Ephemeral, candidate-aligned LLM call records.  ``to_dict`` deliberately
    # omits them: the report writer moves full request/response bodies into the
    # run-local ``_llm_calls.jsonl`` sidecar and serializes byte ``call_refs``
    # into ``stages.llm`` instead.
    llm_calls: List[Any] = field(default_factory=list, repr=False, compare=False)
    # What part of the record this verdict covers, when the caller narrowed it
    # (see :mod:`bpo.core.scope`): ``{scope, generated_turn_index,
    # not_applicable_reason, n_filtered_findings}``. ``None`` for the default
    # whole-trajectory scope, so unscoped results serialize exactly as before.
    scope: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "canonical_name": self.detector_id,
            "detector_type": self.detector_type,
            "behavior_version": self.detector_version,
            "is_issue": self.is_issue,
            "severity": self.severity,
            "findings": [f.to_dict() for f in self.findings],
            "n_messages": self.n_messages,
            "thresholds": dict(self.thresholds),
            "stages": _normalize_stages(self.stages),
            "error": self.error,
            "metric": dict(self.metric) if self.metric is not None else None,
            **({"scope": dict(self.scope)} if self.scope is not None else {}),
        }

    @classmethod
    def error_for(
        cls,
        sample: "NormalizedSample",
        detector: "Detector",
        exc: BaseException,
    ) -> "DetectorResult":
        msg = f"{type(exc).__name__}: {exc}"
        return cls(
            sample_id=sample.sample_id,
            detector_id=getattr(detector, "detector_id", "<unknown>"),
            detector_type=getattr(detector, "detector_type", "rule"),
            detector_version=getattr(
                getattr(detector, "manifest", None), "version", "0.1.0"
            ),
            is_issue=False,
            severity=None,
            findings=[],
            n_messages=0,
            thresholds=dict(getattr(detector, "thresholds", {}) or {}),
            stages={},
            error=msg,
        )


def _normalize_stages(stages: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """JSON-friendly copy of stages, in case anything stuffed dataclasses in."""
    out: Dict[str, Dict[str, Any]] = {}
    for stage_name, stage_body in stages.items():
        if not isinstance(stage_body, dict):
            out[stage_name] = {"data": stage_body}
            continue
        out[stage_name] = {k: v for k, v in stage_body.items()}
    return out


@runtime_checkable
class Detector(Protocol):
    """Structural type that anything pluggable into the runner satisfies.

    Doesn't require explicit inheritance — :class:`RuleDetector` and
    :class:`LLMDetector` both satisfy it because they implement
    ``detect()`` and carry the expected attributes. The runner uses
    ``isinstance(d, Detector)`` at registration time.

    Implementations live in :mod:`bpo.core.rule_detector` and
    :mod:`bpo.core.llm_detector`.
    """

    detector_id: str
    detector_type: DetectorType
    manifest: "DetectorManifest"
    thresholds: Dict[str, Any]

    def detect(self, sample: "NormalizedSample") -> DetectorResult: ...


__all__ = [
    "Severity",
    "SEVERITY_ORDER",
    "VALID_SEVERITIES",
    "DetectorType",
    "VALID_DETECTOR_TYPES",
    "Finding",
    "Candidate",
    "DetectorResult",
    "Detector",
]
