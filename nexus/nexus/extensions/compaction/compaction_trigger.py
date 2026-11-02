"""Compaction condition abstractions for HyTuner context management.

Compaction decides *when* to truncate message history.
Compaction strategy decides *how* to compact.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class CompactionContext:
    """Runtime context used to evaluate whether compaction should happen."""

    current_prompt_tokens: int
    segment_count: int
    max_segments: int | None
    curr_segment_consumed_tokens: int | None = None


@dataclass
class CompactionDecision:
    """Result of compaction condition evaluation."""

    should_compact: bool
    reason: str = ''
    target_tokens: int | None = None


class CompactionCondition(ABC):
    """Base class for compaction trigger conditions."""

    @abstractmethod
    def should_compact(self, ctx: CompactionContext) -> CompactionDecision: ...


_compaction_condition_registry: dict[str, type[CompactionCondition]] = {}


def register_compaction_condition(name: str):
    """Decorator to register a compaction condition class."""

    def wrapper(cls: type[CompactionCondition]) -> type[CompactionCondition]:
        if name in _compaction_condition_registry:
            raise ValueError(f"Compaction condition '{name}' is already registered")
        _compaction_condition_registry[name] = cls
        return cls

    return wrapper


def get_compaction_condition(name: str, **kwargs) -> CompactionCondition:
    """Instantiate a registered compaction condition."""
    if name not in _compaction_condition_registry:
        available = ', '.join(sorted(_compaction_condition_registry.keys())) or '(none)'
        raise ValueError(f"Unknown compaction condition '{name}'. Available: {available}")
    return _compaction_condition_registry[name](**kwargs)


def _resolve_ratio_base(max_position_embeddings: int | None, max_model_len: int | None) -> tuple[str, int | None]:
    if max_position_embeddings is not None and max_model_len is not None:
        raise ValueError('ratio compaction takes either max_model_len or max_position_embeddings, not both')
    if max_model_len is not None:
        return 'max_model_len', max_model_len
    return 'max_position_embeddings', max_position_embeddings


@register_compaction_condition('remaining_tokens_exhausted')
class RemainingTokensExhaustedCondition(CompactionCondition):
    """Compact when current prompt tokens exceed a computed budget."""

    def __init__(
        self,
        ratio: float | None = None,
        max_position_embeddings: int | None = None,
        max_generation_length: int | None = None,
        max_model_len: int | None = None,
        **kwargs: Any,
    ):
        self.ratio = ratio
        self.ratio_base_name, self.ratio_base = _resolve_ratio_base(max_position_embeddings, max_model_len)
        self.max_generation_length = max_generation_length

    def should_compact(self, ctx: CompactionContext) -> CompactionDecision:
        # Priority 1: ratio * (max_model_len or max_position_embeddings)
        if self.ratio is not None and self.ratio_base is not None:
            budget = int(self.ratio * self.ratio_base)
            should = ctx.current_prompt_tokens >= budget
            return CompactionDecision(
                should_compact=should,
                reason=(
                    f'current_prompt_tokens({ctx.current_prompt_tokens}) '
                    f'>= ratio({self.ratio}) * {self.ratio_base_name}({self.ratio_base}) '
                    f'= {budget}'
                ),
            )

        # Priority 2: absolute max_generation_length threshold
        if self.max_generation_length is not None:
            should = ctx.current_prompt_tokens >= self.max_generation_length
            return CompactionDecision(
                should_compact=should,
                reason=(
                    f'current_prompt_tokens({ctx.current_prompt_tokens}) '
                    f'>= max_generation_length({self.max_generation_length})'
                ),
            )

        # Fallback: no budget available, never compact
        return CompactionDecision(should_compact=False, reason='no_budget_configured')


@register_compaction_condition('current_segment_consumed_tokens_exhausted')
class CurrentSegmentConsumedTokensExhaustedCondition(CompactionCondition):
    """Compact when current prompt tokens exceed a computed budget."""

    def __init__(
        self,
        ratio: float | None = None,
        max_position_embeddings: int | None = None,
        max_generation_length: int | None = None,
        max_model_len: int | None = None,
        **kwargs: Any,
    ):
        self.ratio = ratio
        self.ratio_base_name, self.ratio_base = _resolve_ratio_base(max_position_embeddings, max_model_len)
        self.max_generation_length = max_generation_length

    def should_compact(self, ctx: CompactionContext) -> CompactionDecision:
        assert ctx.curr_segment_consumed_tokens is not None

        # Priority 1: ratio * (max_model_len or max_position_embeddings)
        if self.ratio is not None and self.ratio_base is not None:
            budget = int(self.ratio * self.ratio_base)
            should = ctx.curr_segment_consumed_tokens >= budget
            return CompactionDecision(
                should_compact=should,
                reason=(
                    f'curr_segment_consumed_tokens({ctx.curr_segment_consumed_tokens}) '
                    f'>= ratio({self.ratio}) * {self.ratio_base_name}({self.ratio_base}) '
                    f'= {budget}'
                ),
            )

        # Priority 2: absolute max_generation_length threshold
        if self.max_generation_length is not None:
            should = ctx.curr_segment_consumed_tokens >= self.max_generation_length
            return CompactionDecision(
                should_compact=should,
                reason=(
                    f'curr_segment_consumed_tokens({ctx.curr_segment_consumed_tokens}) '
                    f'>= max_generation_length({self.max_generation_length})'
                ),
            )

        # Fallback: no budget available, never compact
        return CompactionDecision(should_compact=False, reason='no_budget_configured')


@register_compaction_condition('always')
class AlwaysCompactCondition(CompactionCondition):
    """Always compact when context management is enabled."""

    def __init__(self, **kwargs: Any) -> None:  # noqa: ARG002
        pass

    def should_compact(self, ctx: CompactionContext) -> CompactionDecision:
        return CompactionDecision(should_compact=True, reason='always')


@register_compaction_condition('always_not')
class AlwaysNotCompactCondition(CompactionCondition):
    """Never compact when context management is enabled."""

    def __init__(self, **kwargs: Any) -> None:  # noqa: ARG002
        pass

    def should_compact(self, ctx: CompactionContext) -> CompactionDecision:
        return CompactionDecision(should_compact=False, reason='always_not')
