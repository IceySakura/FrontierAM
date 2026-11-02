"""Independent CompactionManager for HyTuner context management.

This module provides a standalone CompactionManager class that encapsulates
all compaction logic (condition evaluation, strategy execution, segment
management) without depending on any client implementation.

External capabilities (token counting, LLM completion) are injected via
callbacks at construction time.
"""

from __future__ import annotations

import logging

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

from nexus.core import ContextLengthExceededError
from nexus.core.context import require_context
from nexus.extensions.compaction.compaction_strategy import get_compaction_strategy
from nexus.extensions.compaction.compaction_trigger import (
    CompactionContext,
    get_compaction_condition,
)
from nexus.extensions.compaction.runtime_state import (
    build_compaction_resume_message,
    build_trajectory_budget_state_message,
    is_compaction_runtime_state_message,
)
from nexus.utils.timeout import TimeBudget, await_completion_with_time_budget


logger = logging.getLogger(__name__)

_SELF_SUMMARIZE_STRATEGIES = frozenset({'self_summarize', 'self_summarize_v2', 'search/self_summarize'})


async def count_tokens_with_context_client(messages: list[dict], tools: Any) -> int:
    """Count prompt tokens with the current execution context's OpenAI client."""
    result = await require_context().openai_client.count_tokens(messages=messages, tools=tools)
    return result.input_tokens


def _read_trajectory_budget_state(client: Any) -> tuple[int | None, int] | None:
    """Read the public budget counters exposed by compatible training clients."""
    token_budget = getattr(client, 'token_budget', None)
    token_consumed = getattr(client, 'token_consumed', None)
    if (token_budget is not None and not isinstance(token_budget, int)) or not isinstance(token_consumed, int):
        return None
    return token_budget, token_consumed


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CompactionResult:
    """Return value of :meth:`CompactionManager.maybe_compact`."""

    messages: list[dict]
    compacted: bool
    prompt_token_count: int | None = None
    requested_strategy: str | None = None
    effective_strategy: str | None = None


# ---------------------------------------------------------------------------
# CompactionManager
# ---------------------------------------------------------------------------


class CompactionManager:
    """Stateful manager that decides *when* and *how* to compact messages.

    This class is fully independent of any client implementation.  All
    external capabilities are injected via constructor callbacks:

    * ``async token_count_fn(messages, tools) -> int`` – an async callable that
      counts tokens for a message list using the model's chat template.
    * ``completion_fn`` – an optional caller-owned summary callback. When it is
      omitted, self-summary strategies bind the current framework client and mark
      that internal request as ``summary_inference`` for Gateway policy.

    The manager maintains its own compaction state (segment count) and
    exposes helper methods consumed by the agent layer.
    """

    def __init__(
        self,
        *,
        strategy_name: str = 'keep_recent',
        condition_name: str = 'always_not',
        target_tokens: int = 4096,
        max_segments: int | None = None,
        token_count_fn: Callable[[list[dict], Any], Awaitable[int]],
        completion_fn: Callable | None = None,
        sampling_params: dict[str, Any] | None = None,
        condition_kwargs: dict[str, Any] | None = None,
        inject_trajectory_budget_state: bool = False,
        **strategy_kwargs: Any,
    ) -> None:
        self._strategy = get_compaction_strategy(strategy_name)
        self._condition = get_compaction_condition(condition_name, **(condition_kwargs or {}))
        self._strategy_name = strategy_name
        self._target_tokens = target_tokens
        self._max_segments = max_segments
        self._token_count_fn = token_count_fn
        self._completion_fn = completion_fn
        self._sampling_params = sampling_params or {}
        self._strategy_kwargs = strategy_kwargs
        self._inject_trajectory_budget_state = inject_trajectory_budget_state

        # Internal state
        self._segment_count: int = 0
        self._compaction_attempt_count: int = 0
        self._compaction_stats: list[dict[str, Any]] = []
        self._llm_summary_traces: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def segment_count(self) -> int:
        """Number of compaction segments produced so far."""
        return self._segment_count

    @property
    def compaction_stats(self) -> list[dict[str, Any]]:
        """Per-event compaction size stats collected during this trajectory."""
        return list(self._compaction_stats)

    @property
    def llm_summary_traces(self) -> list[dict[str, Any]]:
        """Opt-in summarizer inputs and outputs for trajectory dumps."""
        return list(self._llm_summary_traces)

    def inject_initial_runtime_context(self, messages: list[dict]) -> list[dict]:
        """Attach the initial trajectory budget snapshot before the first inference."""
        client = require_context().openai_client
        budget_state = _read_trajectory_budget_state(client)
        if not self._inject_trajectory_budget_state or budget_state is None:
            return messages
        token_budget, token_consumed = budget_state
        messages_without_runtime_state = [
            message for message in messages if not is_compaction_runtime_state_message(message)
        ]
        return messages_without_runtime_state + [
            build_trajectory_budget_state_message(
                token_budget=token_budget,
                token_consumed=token_consumed,
                completed_compactions=self._segment_count,
            ),
        ]

    def bind_completion_fn(self, fn: Callable) -> None:
        """Bind the completion callback after construction.

        This is used to inject the client's create_completion method
        once the client has been created.  Only needed for
        self-summary strategies.
        """
        self._completion_fn = fn

    # ------------------------------------------------------------------
    # Core compaction logic
    # ------------------------------------------------------------------

    async def maybe_compact(
        self,
        messages: list[dict],
        tools: Any,
        current_prompt_tokens: int | None = None,
        inference_budget: TimeBudget | None = None,
    ) -> CompactionResult:
        """Evaluate the compaction condition and, if triggered, execute the strategy.

        Args:
            messages: Current message history.
            tools: Tool schemas (passed to the strategy for token counting).
            current_prompt_tokens: Current prompt token count. When *None*
                the manager computes it via ``token_count_fn``.

        Returns:
            A :class:`CompactionResult` indicating whether compaction occurred
            and, if so, the compacted messages.
        """
        # 1. Internal token count calculation
        if current_prompt_tokens is None:
            current_prompt_tokens = await self._token_count_fn(messages, tools)

        client = require_context().openai_client
        curr_segment_consumed_tokens = None
        # Duck-typed dispatch (not isinstance): both HyTunerClient and GatewayOpenAIClient
        # expose this hook; an isinstance(HyTunerClient) guard would silently skip the
        # gateway client and kill segment-consumed accounting on the gateway lane.
        update_token_consumption = getattr(client, 'update_token_consumption', None)
        if callable(update_token_consumption):
            await update_token_consumption(messages, tools=tools)
            curr_segment_consumed_tokens = getattr(client, 'curr_segment_consumed_tokens', None)

        # 2. Evaluate condition
        ctx = CompactionContext(
            current_prompt_tokens=current_prompt_tokens,
            segment_count=self._segment_count,
            max_segments=self._max_segments,
            curr_segment_consumed_tokens=curr_segment_consumed_tokens,
        )
        decision = self._condition.should_compact(ctx)

        if not decision.should_compact:
            return CompactionResult(messages=messages, compacted=False)

        if getattr(client, 'strict_prefix_check', False):
            raise RuntimeError(
                'HyTunerClient strict_prefix_check is enabled, so compaction is not allowed because it rewrites '
                f'message history. Compaction condition matched: {decision.reason}. Disable strict_prefix_check '
                'or disable compaction for this run.'
            )

        # 3. Check segment limit
        if self._max_segments is not None and self._segment_count >= self._max_segments:
            raise ContextLengthExceededError(
                f'HyTuner context length exceeded and max_segments limit reached '
                f'({self._segment_count}/{self._max_segments}): '
                f'current_prompt_tokens={current_prompt_tokens}',
            )

        # 4. Check completion_fn requirement for self-summary strategies
        if self._strategy_name in _SELF_SUMMARIZE_STRATEGIES and self._completion_fn is None:
            summary_client = require_context().openai_client
            if hasattr(summary_client, 'create_completion'):
                # Only framework-owned callbacks receive Gateway metadata;
                # explicitly injected callbacks keep their public signature.
                self._completion_fn = partial(
                    summary_client.create_completion,
                    request_meta={'invocation_kind': 'summary_inference'},
                )
            else:
                raise RuntimeError(
                    f'{self._strategy_name} strategy requires a completion_fn but none '
                    'has been bound. Call bind_completion_fn() first.'
                )

        # 5. Execute strategy
        target = decision.target_tokens or self._target_tokens

        completed_compactions = self._segment_count + 1

        def _build_runtime_context(candidate_messages: list[dict]) -> list[dict[str, Any]]:
            resume_after_compaction = bool(
                self._strategy_name == 'self_summarize_v2'
                and candidate_messages
                and candidate_messages[-1].get('role') == 'assistant'
                and candidate_messages[-1].get('_is_compacted_summary')
            )
            runtime_context: list[dict[str, Any]] = []
            budget_state = _read_trajectory_budget_state(client)
            if self._inject_trajectory_budget_state and budget_state is not None:
                token_budget, token_consumed = budget_state
                runtime_context.append(
                    build_trajectory_budget_state_message(
                        token_budget=token_budget,
                        token_consumed=token_consumed,
                        completed_compactions=completed_compactions,
                    )
                )
            if resume_after_compaction:
                resume_message = build_compaction_resume_message()
                if runtime_context:
                    runtime_context[0]['content'] += f'\n{resume_message["content"]}'
                else:
                    runtime_context.append(resume_message)
            return runtime_context

        strategy_messages = [message for message in messages if not is_compaction_runtime_state_message(message)]

        async def _count_with_runtime_context(candidate_messages: list[dict], candidate_tools: Any) -> int:
            return await self._token_count_fn(
                candidate_messages + _build_runtime_context(candidate_messages),
                candidate_tools,
            )

        strategy_token_count_fn = _count_with_runtime_context

        # Build strategy kwargs – strategies receive all extra config;
        # Self-summary strategies additionally get completion_fn and sampling_params.
        kwargs: dict[str, Any] = dict(self._strategy_kwargs)
        if self._strategy_name == 'self_summarize_v2' and kwargs.get('self_summary_append_control', False):
            # strategy_messages excludes transient runtime-state messages for
            # handoff construction. Append mode needs the exact main-agent
            # history as its request prefix; runtime state is still removed
            # from the compacted result and regenerated below.
            kwargs['_summary_source_messages'] = messages
        if self._strategy_name in _SELF_SUMMARIZE_STRATEGIES and self._completion_fn is not None:
            completion_fn = self._completion_fn
            if inference_budget is not None:
                base_completion_fn = completion_fn

                async def _budgeted_completion(**completion_kwargs: Any) -> Any:
                    return await await_completion_with_time_budget(
                        inference_budget,
                        lambda: base_completion_fn(**completion_kwargs),
                    )

                completion_fn = _budgeted_completion
            kwargs['completion_fn'] = completion_fn
            effective_request_args = client.get_effective_request_args() if client is not None else {}
            effective_sampling_params = effective_request_args.get('sampling_params')
            kwargs['sampling_params'] = (
                dict(effective_sampling_params)
                if isinstance(effective_sampling_params, dict)
                else dict(self._sampling_params)
            )

        self._compaction_attempt_count += 1
        compaction_attempt_order = self._compaction_attempt_count
        truncated = await self._strategy.compact(
            strategy_messages,
            tools,
            target,
            strategy_token_count_fn,
            **kwargs,
        )
        strategy_info = dict(getattr(self._strategy, 'last_compact_info', None) or {})
        summary_trace = strategy_info.pop('summary_trace', None)
        if isinstance(summary_trace, dict):
            summary_trace = {**summary_trace, 'compaction_attempt_order': compaction_attempt_order}
        else:
            summary_trace = None

        if truncated == strategy_messages:
            logger.debug(
                f'Context management: compaction condition matched ({decision.reason}) but '
                f'strategy={self._strategy_name} returned unchanged history; skipping segment boundary.'
            )
            if strategy_info.get('fallback_reason'):
                self._compaction_stats.append({
                    'compacted': False,
                    'compaction_attempt_order': compaction_attempt_order,
                    'pre_prompt_tokens': int(current_prompt_tokens),
                    'requested_strategy': strategy_info.get('requested_strategy', self._strategy_name),
                    'effective_strategy': strategy_info.get('effective_strategy'),
                    **{key: value for key, value in strategy_info.items() if value is not None},
                })
                if summary_trace is not None:
                    self._llm_summary_traces.append(summary_trace)
            return CompactionResult(
                messages=messages,
                compacted=False,
                prompt_token_count=current_prompt_tokens,
                requested_strategy=self._strategy_name,
                effective_strategy=strategy_info.get('effective_strategy'),
            )

        logger.debug(
            f'Context management: compaction triggered ({decision.reason}). '
            f'Truncated {len(messages)} -> {len(truncated)} messages '
            f'(target_tokens={target}, strategy={self._strategy_name})'
        )

        # 6. Update internal state
        self._segment_count += 1

        # 7. Compute prompt_token_count for the compacted messages
        prompt_token_count = await strategy_token_count_fn(truncated, tools)
        dropped_messages = max(len(strategy_messages) - len(truncated), 0)
        truncated = truncated + _build_runtime_context(truncated)
        effective_strategy = strategy_info.get('effective_strategy', self._strategy_name)
        compaction_stat = {
            'compacted': True,
            'compaction_attempt_order': compaction_attempt_order,
            'pre_prompt_tokens': int(current_prompt_tokens),
            'post_prompt_tokens': int(prompt_token_count),
            'dropped_messages': dropped_messages,
            'requested_strategy': strategy_info.get('requested_strategy', self._strategy_name),
            'effective_strategy': effective_strategy,
        }
        compaction_stat.update({
            key: value for key, value in strategy_info.items() if key not in compaction_stat and value is not None
        })
        self._compaction_stats.append(compaction_stat)
        if summary_trace is not None:
            self._llm_summary_traces.append(summary_trace)

        if curr_segment_consumed_tokens is not None:
            cast(Any, client).curr_segment_consumed_tokens = 0

        return CompactionResult(
            messages=truncated,
            compacted=True,
            prompt_token_count=prompt_token_count,
            requested_strategy=self._strategy_name,
            effective_strategy=effective_strategy,
        )
