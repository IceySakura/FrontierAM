# /// script
# dependencies = []
#
# [project]
# name = "token_counter_perstep"
# authors = [
#     { name = "perryrupeng" },
# ]
#
# [tool.nexus]
# span_processors = { token_counter_perstep = "nexus.trace.processors.token_counter_perstep" }
# ///

from __future__ import annotations

from collections import defaultdict

from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.base import SpanProcessor, span_processor_registry
from nexus.trace.types import Span


@span_processor_registry.register('token_counter_perstep')
class TokenCounterPerStepProcessor(SpanProcessor):
    """Record token usage for each model call and aggregate it by client.

    Use openai_completion spans to record prompt, completion, reasoning, cached,
    and non-cached prompt tokens. Non-cached input measures cache misses, not
    unique trajectory content.

    Args:
        client_keys: Record only these client keys; None includes all clients.
    """

    def __init__(self, client_keys: list[str] | None = None) -> None:
        self._client_keys: set[str] | None = set(client_keys) if client_keys is not None else None
        self.stat_dict = defaultdict(
            lambda: {
                'completion_tokens': 0,
                'reasoning_tokens': 0,
                'prompt_tokens': 0,
                'cached_prompt_tokens': 0,
                'non_cached_prompt_tokens': 0,
                'llm_call_count': 0,
                'per_step': [],
            }
        )

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        if span.name != 'openai_completion':
            return

        cur_client_key = span.attributes.get('_client_key')
        if self._client_keys is not None and cur_client_key not in self._client_keys:
            return
        if cur_client_key is None:
            cur_client_key = '_unknown_client_'

        outputs = span.attributes.get('outputs')
        if not isinstance(outputs, dict):
            return

        usage = outputs.get('usage')
        if not isinstance(usage, dict):
            return

        completion_tokens = usage.get('completion_tokens') or 0
        reasoning_tokens = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0
        prompt_tokens = usage.get('prompt_tokens') or 0
        cached_prompt_tokens = (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
        non_cached_prompt_tokens = prompt_tokens - cached_prompt_tokens

        stats = self.stat_dict[cur_client_key]
        stats['completion_tokens'] += completion_tokens
        stats['reasoning_tokens'] += reasoning_tokens
        stats['prompt_tokens'] += prompt_tokens
        stats['cached_prompt_tokens'] += cached_prompt_tokens
        stats['non_cached_prompt_tokens'] += non_cached_prompt_tokens
        stats['llm_call_count'] += 1
        stats['per_step'].append({
            'call': stats['llm_call_count'],
            'prompt': prompt_tokens,
            'completion': completion_tokens,
            'reasoning': reasoning_tokens,
            'cached_prompt': cached_prompt_tokens,
            'non_cached_prompt': non_cached_prompt_tokens,
        })

    def get_result(self) -> dict:
        return dict(self.stat_dict)
