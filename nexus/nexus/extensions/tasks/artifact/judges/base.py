"""Base classes and aggregation logic for artifact judges.

Two invariants every judge must uphold (both enforced at construction):
  * ``JudgeResult.score`` ∈ [0, 1] — ``aggregate_judge_results`` relies on it.
  * ``error`` ⇒ ``failure_kind`` — a fatal error MUST carry a structured kind so
    the task layer classifies the RL outcome without substring-matching free
    text. Non-fatal warnings belong in ``evidence`` / ``metadata``, not ``error``.
"""

from __future__ import annotations

import logging
import math

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


if TYPE_CHECKING:
    from nexus.core.openai_client import OpenAICompatibleClient
    from nexus.extensions.tasks.artifact.judges.capabilities.deploy import SandboxContext


logger = logging.getLogger(__name__)


# Structured failure categories; None denotes a valid score.
JudgeFailureKind = Literal[
    'no_client',  # required OpenAI client could not be resolved
    'no_artifact',  # empty archive_b64 reached the judge
    'no_sandbox',  # sandbox_ctx missing / deploy produced no URL or screenshots
    'vlm_failed',  # vision-LLM call errored (after client-layer retries)
    'parse_failed',  # LLM/agent output could not be parsed
    'agent_failed',  # claude-code run itself failed
    'config_error',  # wrong config type / missing required config (e.g. empty rubrics)
    'eval_timeout',  # judge execution exceeded its budget
    'internal_error',  # unexpected error in the judge implementation
]


class JudgeExecutionError(RuntimeError):
    """Carry a structured capability failure to the registry wrapper."""

    def __init__(self, kind: JudgeFailureKind, message: str):
        self.kind = kind
        super().__init__(message)


class RubricResultItem(BaseModel):
    """Per-rubric structured result for rubric-based judges."""

    index: int
    rubric: str
    passed: bool
    verdict: str = ''
    note: str = ''
    error: str | None = None
    trajectories: dict[str, Any] = Field(default_factory=dict)
    execution_time: float = 0.0


class JudgeResult(BaseModel):
    """One authoritative verdict, with optional diagnostic child results."""

    judge_name: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool | None = None
    evidence: str = ''
    raw_output: str = ''
    error: str | None = None
    failure_kind: JudgeFailureKind | None = None
    """Structured failure category (see :data:`JudgeFailureKind`). Set alongside
    ``error`` so the task layer classifies the RL outcome without parsing text."""
    metadata: dict[str, Any] = Field(default_factory=dict)
    rubric_results: list[RubricResultItem] = Field(default_factory=list)
    child_results: list[JudgeResult] = Field(default_factory=list)
    """Diagnostic child verdicts; consumers must not recompute the top-level score."""

    @field_validator('score', mode='before')
    @classmethod
    def _reject_non_numeric_score(cls, value: Any) -> Any:
        """Reject booleans and non-finite scores before numeric coercion."""
        if isinstance(value, bool):
            raise ValueError(
                f'score must be a real number, got bool {value!r} — '
                'a judge returning a flag as its score is a bug, not full marks'
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'score must be finite, got {value!r}')
        return value

    @model_validator(mode='after')
    def _failure_result_invariants(self) -> JudgeResult:
        """Require a failure kind for fatal errors and disallow passing failed results."""
        if self.error and not self.failure_kind:
            raise ValueError(f'JudgeResult({self.judge_name}) sets error but no failure_kind: {self.error!r}')
        if self.failure_kind and self.passed:
            raise ValueError(
                f'JudgeResult({self.judge_name}) has failure_kind={self.failure_kind!r} '
                'but claims passed=True — a failed evaluation cannot report a pass'
            )
        return self


class JudgeConfig(BaseModel):
    """Base config shared by all judge types."""

    model_config = {'arbitrary_types_allowed': True}

    name: str
    weight: float = Field(default=1.0, ge=0.0)
    required: bool = False


class BaseJudge(ABC):
    """Abstract base class for all judges."""

    @abstractmethod
    async def run(
        self,
        config: JudgeConfig,
        image: str,
        patch_text: str,
        working_dir: str,
        openai_client: OpenAICompatibleClient | None = None,
        **kwargs: Any,
    ) -> JudgeResult: ...


@dataclass(frozen=True)
class JudgeInput:
    """Frozen input for a registered judge. Use dataclasses.replace for derived inputs.

    Judge-specific fields live in instance; archive_b64 contains the deliverable tar.gz.
    """

    # Task identity and prompt.
    instance_id: str
    problem_statement: str

    # Clients resolved by the caller with per-task cache IDs.
    openai_clients: Mapping[str, OpenAICompatibleClient]

    # Base64-encoded tar.gz deliverable.
    archive_b64: str = ''

    # Resources supplied by the caller or capability preparation.
    image: str = ''
    runtime: Any = None
    sandbox_ctx: SandboxContext | None = None
    """Prepared by the wrapper when ``@register_judge`` declares
    ``capabilities=('deploy',)`` / ``('screenshot',)``."""

    # Strategy-specific data.
    instance: Mapping[str, Any] = field(default_factory=dict)
    """The full data row. Judges pick what they need (``rubrics``,
    ``anchor_screenshot``, ``aesthetic_mode`` — schema stays stable)."""
    save_dir: str | None = None
    model_configs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def flow(self) -> dict[str, Any]:
        """Convenience accessor for the ``web_judger`` flow contract.
        Non-web judges do not need to read this.
        """
        return dict(self.instance.get('flow') or {})

    @property
    def metadata(self) -> dict[str, Any]:
        """Convenience accessor for ``task.toml [metadata]``."""
        return dict(self.instance.get('_metadata') or {})


AggregationMode = Literal['mean', 'min', 'weighted']


def aggregate_judge_results(
    results: list[JudgeResult],
    configs: list[JudgeConfig],
    mode: AggregationMode = 'weighted',
) -> tuple[float, bool]:
    """Aggregate multiple JudgeResult into (final_score, resolved)."""
    if not results:
        return 0.0, False

    # Check required judges first — any required failure → overall 0
    for result, cfg in zip(results, configs, strict=False):
        if cfg.required and (result.error or result.score < 1.0):
            logger.warning(
                'Required judge %s failed (score=%.2f, error=%s)',
                result.judge_name,
                result.score,
                result.error,
            )
            return 0.0, False

    scores = [r.score for r in results]

    if mode == 'min':
        final = min(scores)
    elif mode == 'mean':
        final = sum(scores) / len(scores)
    else:  # weighted
        total_weight = sum(c.weight for c in configs)
        final = (
            0.0
            if total_weight == 0
            else sum(r.score * c.weight for r, c in zip(results, configs, strict=False)) / total_weight
        )

    resolved = final >= 1.0
    return round(final, 4), resolved


__all__ = [
    'RubricResultItem',
    'JudgeResult',
    'JudgeFailureKind',
    'JudgeExecutionError',
    'JudgeInput',
    'JudgeConfig',
    'BaseJudge',
    'AggregationMode',
    'aggregate_judge_results',
]
