"""Span processor that collects main_agent metrics.

Derives per-step timing from ``agent.inference`` / ``agent.observation``
child spans and reads aggregate counters from the agent entity span
(matched by ``type == 'agent'`` attribute).

The processor also serves the role that hooks would traditionally play:
subclass ``MainAgentMetricsProcessor`` and override ``on_iteration_start`` /
``on_observation_end`` to inject custom logic at iteration boundaries — no
separate hook registry needed.

Usage — add to ``RunInput.processors``::

    processors:
      agent_metrics:
        name: main_agent_metrics
"""

import logging

from typing import Any

from nexus.trace.events import SpanEndEvent, SpanStartEvent
from nexus.trace.processors import SpanProcessor, span_processor_registry
from nexus.trace.types import Span


logger = logging.getLogger(__name__)

_NANOS_PER_SEC = 1_000_000_000.0


def _span_duration_sec(span: Span) -> float:
    """Return span duration in seconds, or 0 if timestamps are missing."""
    if span.end_time_unix_nano is None:
        return 0.0
    return (span.end_time_unix_nano - span.start_time_unix_nano) / _NANOS_PER_SEC


def _span_iteration(span: Span) -> int | None:
    """Best-effort parse of outer-loop iteration index from span attributes."""
    raw = span.attributes.get('iteration')
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return None
    return None


@span_processor_registry.register('main_agent_metrics')
class MainAgentMetricsProcessor(SpanProcessor):
    """Collects main_agent metrics from the span tree.

    Listens for span names emitted by the main_agent loop:

    * agent entity span (``type == 'agent'``) — carries final counters as attributes.
    * ``agent.iteration`` — one per outer-loop iteration.
    * ``agent.inference`` — one per LLM call attempt (retries included);
      each span carries ``iteration`` attribute (outer-loop index).
    * ``agent.observation`` — one per tool-execution phase; duration →
      observation time.

    **Hook replacement pattern** — subclass and override the ``on_iteration_start``
    / ``on_observation_end`` callbacks to run custom logic at iteration boundaries::

        class MyProcessor(MainAgentMetricsProcessor):
            def on_iteration_start(self, iteration: int) -> None:
                log.info('iteration %d started', iteration)

            def on_observation_end(self, iteration: int, duration: float) -> None:
                if duration > 30:
                    log.warning('slow observation at iteration %d', iteration)

    ``get_result()`` returns a flat dict suitable for logging / evaluation.
    """

    def __init__(self) -> None:
        self._inference_times: list[float] = []
        self._observation_times: list[float] = []
        self._agent_attrs: dict[str, Any] = {}
        self._current_iteration: int = 0

    # ------------------------------------------------------------------
    # Callbacks — override in subclasses for custom behaviour
    # ------------------------------------------------------------------

    def on_iteration_start(self, iteration: int) -> None:
        """Called when a new outer-loop iteration starts."""

    def on_observation_end(self, iteration: int, duration: float) -> None:
        """Called when an observation span ends (≈ all tools finished)."""

    # ------------------------------------------------------------------
    # SpanProcessor interface
    # ------------------------------------------------------------------

    def on_start(self, span: Span, event: SpanStartEvent) -> None:
        if span.name == 'agent.inference':
            iteration = _span_iteration(span)
            # Fallback for legacy spans without iteration attributes.
            if iteration is None:
                self._current_iteration += 1
                self.on_iteration_start(self._current_iteration)
                return

            # Retry attempts in the same outer loop share the same iteration id.
            if iteration != self._current_iteration:
                self._current_iteration = iteration
                self.on_iteration_start(self._current_iteration)

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        if span.name == 'agent.inference':
            self._inference_times.append(_span_duration_sec(span))

        elif span.name == 'agent.observation':
            duration = _span_duration_sec(span)
            self._observation_times.append(duration)
            self.on_observation_end(_span_iteration(span) or self._current_iteration, duration)

        elif span.attributes.get('type') == 'agent':
            attrs = {k: v for k, v in span.attributes.items() if k.startswith('agent.')}
            # Keep the last nonempty agent counters; empty wrappers must not erase them.
            if attrs:
                self._agent_attrs = attrs

    # ------------------------------------------------------------------

    def get_result(self) -> dict[str, Any]:
        return {
            **self._agent_attrs,
            'agent.total_inference_time': sum(self._inference_times),
            'agent.total_observation_time': sum(self._observation_times),
            'agent.inference_times': self._inference_times,
            'agent.observation_times': self._observation_times,
        }
