"""Tests for get_processor() function."""

from __future__ import annotations

from nexus.trace.context import context_processors, get_processor
from nexus.trace.processors.base import SpanProcessor
from nexus.trace.tracer import processor_context


class FakeProcessorA(SpanProcessor):
    pass


class FakeProcessorB(SpanProcessor):
    pass


class TestGetProcessor:
    def test_get_processor_returns_matching_type(self) -> None:
        """get_processor should return first isinstance match."""
        proc_a = FakeProcessorA()
        proc_b = FakeProcessorB()
        with processor_context([proc_a, proc_b], finalize_on_exit=False):
            assert get_processor(FakeProcessorA) is proc_a
            assert get_processor(FakeProcessorB) is proc_b

    def test_get_processor_returns_none_when_no_processors(self) -> None:
        """get_processor should return None when context_processors is empty."""
        with processor_context([], finalize_on_exit=False):
            assert get_processor(FakeProcessorA) is None

    def test_get_processor_not_found_returns_none(self) -> None:
        """get_processor should return None when no matching type."""
        proc_a = FakeProcessorA()
        with processor_context([proc_a], finalize_on_exit=False):
            assert get_processor(FakeProcessorB) is None

    def test_get_processor_matches_subclass_via_isinstance(self) -> None:
        """get_processor with base type should match subclass via isinstance."""
        proc = FakeProcessorA()
        with processor_context([proc], finalize_on_exit=False):
            assert get_processor(SpanProcessor) is proc

    def test_get_processor_outside_context_returns_none(self) -> None:
        """get_processor should return None when no context is active."""
        # Reset context_processors to None (no active processor_context)
        token = context_processors.set(None)
        try:
            assert get_processor(FakeProcessorA) is None
        finally:
            context_processors.reset(token)

    def test_get_processor_with_snapshot_result(self) -> None:
        """Full integration: get processor then call snapshot_result()."""
        proc = FakeProcessorA()
        with processor_context([proc], finalize_on_exit=False):
            found = get_processor(FakeProcessorA)
            assert found is proc
            assert found.snapshot_result() is None
