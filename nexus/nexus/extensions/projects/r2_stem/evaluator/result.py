"""Structured results for R2 STEM evaluators."""

from dataclasses import asdict, dataclass
from typing import Any

from nexus.extensions.agents.base.evaluation import EvaluationDetails


@dataclass
class LLMJudgeResult(EvaluationDetails):
    """Structured details returned by the generic LLM judge."""

    method: str = ''
    judge_prompt: str = ''
    judge_response: str = ''
    parsed_result: dict[str, Any] | None = None
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ['LLMJudgeResult']
