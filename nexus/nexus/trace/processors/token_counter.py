# /// script
# dependencies = []
#
# [project]
# name = "token_counter"
# authors = [
#     { name = "botongzhou" },
# ]
#
# [tool.nexus]
# span_processors = { token_counter = "nexus.trace.processors.token_counter" }
# ///

from __future__ import annotations

from collections import defaultdict

from nexus.core.types.usage import billed_prompt_usage
from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.base import SpanProcessor, span_processor_registry
from nexus.trace.types import Span


@span_processor_registry.register('token_counter')
class TokenCounterProcessor(SpanProcessor):
    """A processor that accumulates LLM token usage across all calls in a task.

    Listens for 'openai_completion' spans and sums up token counts from
    each call's usage field.

    Args:
        client_keys: If provided, only count tokens from LLM calls made by the
            specified client keys (e.g. ``{'main'}``). Pass ``None`` to count all
            clients.

    Example usage via RunInput config (count all clients):

        processors:
          token_counter:
            name: token_counter

    Example usage via RunInput config (count only the 'main' client):

        processors:
          token_counter:
            name: token_counter
            args:
              client_keys: [main]

    The result is a dictionary mapping client keys to their respective token usage stats:
    {
        "main": {
            "completion_tokens": 150,
            "reasoning_tokens": 50,
            "prompt_tokens": 350,
            "cached_prompt_tokens": 100,
            "non_cached_prompt_tokens": 250,
            "llm_call_count": 5
        },
        "other_client": {
            ...
        },
        "_unknown_client_": {
            ...
        }
    }
    if the client key is not attached to the span attributes, the usage will be aggregated under the "_unknown_client_" key.
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
        # A completion spanning several provider requests reports context-sized usage; count what was billed.
        billed = billed_prompt_usage(outputs)
        if billed is not None:
            prompt_tokens, cached_prompt_tokens = billed

        self.stat_dict[cur_client_key]['completion_tokens'] += completion_tokens
        self.stat_dict[cur_client_key]['reasoning_tokens'] += reasoning_tokens
        self.stat_dict[cur_client_key]['prompt_tokens'] += prompt_tokens
        self.stat_dict[cur_client_key]['cached_prompt_tokens'] += cached_prompt_tokens
        self.stat_dict[cur_client_key]['non_cached_prompt_tokens'] += prompt_tokens - cached_prompt_tokens
        self.stat_dict[cur_client_key]['llm_call_count'] += 1

    def get_result(self) -> dict:
        return dict(self.stat_dict)
