"""Span processor that collects token usage from openai_completion spans.

Monitors ``openai_completion`` spans emitted by ``OpenAICompatibleClient.create_completion()``
and accumulates token counts from ``ExtendedUsage`` stored in ``span.attributes['outputs']['usage']``.

Also tracks wall-clock time for model calls and total elapsed time.

Usage — instantiate directly and register with ``add_span_processor`` / ``remove_span_processor``::

    from nexus.extensions.tasks.claw_eval.token_processor import TokenUsageProcessor

    processor = TokenUsageProcessor()
    add_span_processor(processor)
    # ... run agent ...
    remove_span_processor(processor)
    stats = processor.get_stats()
"""

from __future__ import annotations

import time

from typing import Any

from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.base import SpanProcessor
from nexus.trace.types import Span


_NANOS_PER_SEC = 1_000_000_000.0


class TokenUsageProcessor(SpanProcessor):
    """Collects token usage and timing from openai_completion spans.

    This is a lightweight processor designed to be used inline within
    claw_eval task functions.  It does **not** need to be registered
    via ``span_processor_registry`` (though it could be).

    Attributes collected per ``openai_completion`` span:
    - prompt_tokens (model input tokens)
    - completion_tokens (model output tokens)
    - total_tokens

    Timing:
    - Per-call model time derived from span start/end timestamps.
    - Wall-clock elapsed time from processor creation to ``get_stats()`` call.
    """

    def __init__(self) -> None:
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._total_tokens: int = 0
        self._model_time_s: float = 0.0
        self._call_count: int = 0
        self._cost: float = 0.0
        self._start_time: float = time.monotonic()

    # -- SpanProcessor interface (only on_end needed) --

    def on_start(self, span: Span, event: Any) -> None:  # noqa: ARG002
        pass

    def on_update(self, span: Span, event: Any) -> None:  # noqa: ARG002
        pass

    def on_end(self, span: Span, event: SpanEndEvent) -> None:  # noqa: ARG002
        if span.name != 'openai_completion':
            return

        # Extract usage from serialised outputs
        outputs = span.attributes.get('outputs')
        if not outputs or not isinstance(outputs, dict):
            return

        usage = outputs.get('usage')
        if not usage or not isinstance(usage, dict):
            return

        self._prompt_tokens += usage.get('prompt_tokens', 0) or 0
        self._completion_tokens += usage.get('completion_tokens', 0) or 0
        self._total_tokens += usage.get('total_tokens', 0) or 0
        cost = usage.get('cost')
        if cost is not None:
            self._cost += float(cost)
        self._call_count += 1

        # Accumulate model call time from span timestamps
        if span.end_time_unix_nano is not None:
            duration = (span.end_time_unix_nano - span.start_time_unix_nano) / _NANOS_PER_SEC
            self._model_time_s += duration

    def finalize(self) -> None:
        pass

    # -- Public API --

    def get_stats(self) -> dict[str, Any]:
        """Return collected token and timing statistics.

        Returns a dict with keys matching the original claw-eval naming:
        - model_input_tokens
        - model_output_tokens
        - total_tokens
        - model_time_s
        - wall_time_s
        - model_call_count
        - cost (if available)
        """
        wall_time = time.monotonic() - self._start_time
        result: dict[str, Any] = {
            'model_input_tokens': self._prompt_tokens,
            'model_output_tokens': self._completion_tokens,
            'total_tokens': self._total_tokens,
            'model_time_s': round(self._model_time_s, 2),
            'wall_time_s': round(wall_time, 2),
            'model_call_count': self._call_count,
        }
        if self._cost > 0:
            result['cost'] = round(self._cost, 6)
        return result
