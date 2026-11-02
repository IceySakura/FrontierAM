"""Search-specific self-summary compaction strategy.

This strategy is specific to the search agent family (it reuses
``search_agent``'s structured ``COMPACTION_PROMPT`` / ``COMPACTION_CONTINUE_PROMPT``),
so it lives next to the agent that configures it rather than in the generic
``nexus.extensions.compaction`` package.

Importing this module registers the ``search/self_summarize`` strategy.  It is
imported eagerly by ``agent.py`` so registration always happens before
``_install_compaction_manager`` resolves the strategy by name.
"""

from __future__ import annotations

import logging

from collections.abc import Awaitable, Callable
from typing import Any

from nexus.extensions.compaction.compaction_strategy import (
    CompactionStrategy,
    register_compaction_strategy,
)
from nexus.utils.timeout import InferenceTimeoutError


logger = logging.getLogger(__name__)


@register_compaction_strategy('discard_all')
class DiscardAllCompactionStrategy(CompactionStrategy):
    """Hard reset: keep only the leading system message and the first user message.

    Equivalent to ``SearchAgent._reset_context()``: discards all assistant/tool
    output as well as any later user turns. ``target_tokens`` is ignored.
    """

    async def compact(
        self,
        messages: list[dict],
        tools: Any,
        target_tokens: int,
        token_count_fn: Callable[[list[dict], Any], Awaitable[int]],
        **kwargs,
    ) -> list[dict]:
        head: list[dict] = []
        if messages and messages[0].get('role') == 'system':
            head.append(messages[0])

        first_user_idx = next(
            (i for i, m in enumerate(messages) if m.get('role') == 'user'),
            None,
        )
        if first_user_idx is not None:
            head.append(messages[first_user_idx])

        if not head:
            return messages

        logger.debug(f'Context management: discard_all truncated {len(messages)} -> {len(head)} messages')
        return head


@register_compaction_strategy('search/self_summarize')
class SearchSelfSummarizeCompactionStrategy(CompactionStrategy):
    """Compact messages exactly like ``SearchAgent._compact_context``.

    This reproduces the search_agent compaction semantics one-to-one:

    1. The model is asked to summarize **the real conversation messages** (not a
       flattened text blob) by appending ``SearchAgent.COMPACTION_PROMPT`` (reused
       directly) as a final ``user`` message and running an internal completion
       via ``completion_fn``.
    2. The history is then **hard reset** to ``[system, first_user]`` (the
       original question), discarding all assistant/tool turns and any later
       user turns — identical to ``discard_all``.
    3. The summary is re-injected as a single ``user`` message wrapping the
       summary text together with a "continue working" instruction.  The summary
       is a ``user`` message (not ``assistant``) because it is context fed *to*
       the model, not something the model actually said.

    Unlike ``self_summarize`` (which keeps a recent tail and inserts the summary
    as an ``assistant`` message), this strategy keeps no tail by default and uses
    the structured four-section search prompt.

    The summarization completion runs via ``completion_fn``. ``CompactionManager``
    injects the current client's completion callback and applies the main agent's
    inference-time budget when available; direct strategy callers can still fall
    back to resolving the client from the current ``ExecutionContext``.

    Optional ``kwargs`` (named to match ``SelfSummarizeCompactionStrategy``):
        completion_fn:   Async ``(messages, tools, sampling_params) -> completion``.
            When absent, ``ctx.openai_client.create_completion`` is used.
        sampling_params:        Sampling parameters forwarded to ``completion_fn``.
        self_summary_prompt:    Override for the summarization request prompt. Used
            as a standalone final user message (no ``{text}`` placeholder needed).
        self_summary_keep_recent:  Recent messages to keep verbatim (default 0).
    """

    @staticmethod
    def _default_summary_prompt() -> str:
        """Reuse ``SearchAgent.COMPACTION_PROMPT`` directly (lazy import to avoid
        pulling the heavy search_agent module at import time)."""
        from nexus.extensions.agents.search_agent.agent import COMPACTION_PROMPT

        return COMPACTION_PROMPT

    @staticmethod
    def _default_continue_prompt() -> str:
        """Reuse ``SearchAgent.COMPACTION_CONTINUE_PROMPT`` (``{summary}`` placeholder)."""
        from nexus.extensions.agents.search_agent.agent import COMPACTION_CONTINUE_PROMPT

        return COMPACTION_CONTINUE_PROMPT

    @staticmethod
    def _resolve_completion_fn() -> Callable | None:
        """Resolve ``create_completion`` from the current ExecutionContext's client."""
        try:
            from nexus.core.context import require_context

            client = require_context().openai_client
        except Exception:
            return None
        return getattr(client, 'create_completion', None)

    @staticmethod
    def _reset_head(messages: list[dict]) -> list[dict]:
        """Mirror ``SearchAgent._reset_context``: keep up to and including the first user message."""
        head: list[dict] = []
        if messages and messages[0].get('role') == 'system':
            head.append(messages[0])
        first_user_idx = next((i for i, m in enumerate(messages) if m.get('role') == 'user'), None)
        if first_user_idx is not None:
            head.append(messages[first_user_idx])
        return head

    async def compact(
        self,
        messages: list[dict],
        tools: Any,
        target_tokens: int,
        token_count_fn: Callable[[list[dict], Any], Awaitable[int]],
        **kwargs,
    ) -> list[dict]:
        completion_fn = kwargs.get('completion_fn') or self._resolve_completion_fn()
        if completion_fn is None:
            logger.warning(
                'SearchSelfSummarizeCompactionStrategy: no completion_fn available; falling back to hard reset.'
            )
            head = self._reset_head(messages)
            return head or messages

        summary_prompt = str(kwargs.get('self_summary_prompt') or self._default_summary_prompt())
        keep_recent = int(kwargs.get('self_summary_keep_recent') or 0)
        sampling_params = kwargs.get('sampling_params', {})

        head = self._reset_head(messages)
        if not head:
            # No anchor to reset to; leave history unchanged.
            return messages

        # Tail kept verbatim (default: none, matching search_agent).  When
        # keep_recent > 0 the split point is nudged forward past any leading
        # ``tool`` messages so the tail never starts with an orphan tool
        # response whose ``assistant`` (with tool_calls) was summarized away —
        # such a sequence is rejected by the chat API.
        if keep_recent > 0:
            # Clamp to >= 0: keep_recent may exceed len(messages) (misconfig),
            # in which case a negative split would index out of range.
            split = max(len(messages) - keep_recent, 0)
            while split < len(messages) and messages[split].get('role') == 'tool':
                split += 1
            tail = messages[split:]
            to_summarize = messages[:split]
        else:
            tail = []
            to_summarize = messages

        # Build a one-shot summarization request from the REAL conversation
        # messages plus the structured prompt — identical to search_agent.
        summary_messages = list(to_summarize) + [{'role': 'user', 'content': summary_prompt}]

        try:
            completion = await completion_fn(
                messages=summary_messages,
                tools=None,
                sampling_params=sampling_params,
            )
            summary_text = (completion.choices[0].message.content or '').strip()
        except InferenceTimeoutError:
            raise
        except Exception:
            logger.exception(
                'SearchSelfSummarizeCompactionStrategy: internal completion failed; falling back to hard reset.'
            )
            # Hard reset without summary, mirroring search_agent fallback.
            return head + tail

        summary_msg: dict[str, Any] = {
            'role': 'user',
            # Use replace (not str.format) so a summary containing other ``{...}``
            # braces (e.g. JSON examples) cannot raise KeyError.
            'content': self._default_continue_prompt().replace('{summary}', summary_text),
            '_is_compacted_summary': True,
        }
        result = head + [summary_msg] + tail

        logger.debug(
            f'Context management(search/self_summarize): compressed from {len(messages)} to {len(result)} messages'
        )
        return result
