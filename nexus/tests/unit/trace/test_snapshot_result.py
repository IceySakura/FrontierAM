"""Tests for SpanProcessor.snapshot_result() across processor implementations."""

from __future__ import annotations

from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor
from nexus.trace.types import Span


def _completion_span(inputs: dict, outputs: dict, span_id: str) -> Span:
    return Span(
        trace_id='trace',
        span_id=span_id,
        parent_span_id='',
        name='openai_completion',
        start_time_unix_nano=0,
        attributes={'inputs': inputs, 'outputs': outputs},
    )


def _track_completion(
    tracker: ChatTrackerProcessor,
    inputs: dict,
    outputs: dict,
    span_id: str,
) -> None:
    tracker.on_end(_completion_span(inputs, outputs, span_id), SpanEndEvent())


class TestChatTrackerSnapshotResult:
    def test_snapshot_returns_trajectories_mid_execution(self) -> None:
        """snapshot_result should return trajectories without finalization."""
        tracker = ChatTrackerProcessor(tools_mode='branch_last')
        system = {'role': 'system', 'content': 'You are an agent.'}
        user = {'role': 'user', 'content': 'report env'}
        tools = [{'type': 'function', 'function': {'name': 'exec'}}]

        _track_completion(
            tracker,
            {'messages': [system, user], 'kwargs': {'tools': tools}},
            {'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]},
            'first',
        )

        snapshot = tracker.snapshot_result()
        assert 'trajectories' in snapshot
        assert len(snapshot['trajectories']) == 1
        assert 'calls' not in snapshot

    def test_snapshot_includes_tools_in_global_first_mode(self) -> None:
        """In global_first mode, snapshot should include tools."""
        tracker = ChatTrackerProcessor(tools_mode='global_first')
        tools = [{'type': 'function', 'function': {'name': 'exec'}}]
        system = {'role': 'system', 'content': 'You are an agent.'}
        user = {'role': 'user', 'content': 'hi'}

        _track_completion(
            tracker,
            {'messages': [system, user], 'kwargs': {'tools': tools}},
            {'choices': [{'message': {'role': 'assistant', 'content': 'hello'}}]},
            'first',
        )

        snapshot = tracker.snapshot_result()
        assert 'tools' in snapshot
        assert snapshot['tools'] == tools

    def test_snapshot_result_does_not_prevent_get_result(self) -> None:
        """Calling snapshot_result should not affect subsequent get_result."""
        tracker = ChatTrackerProcessor(tools_mode='global_first')
        system = {'role': 'system', 'content': 'You are an agent.'}
        user = {'role': 'user', 'content': 'hi'}

        _track_completion(
            tracker,
            {'messages': [system, user], 'kwargs': {}},
            {'choices': [{'message': {'role': 'assistant', 'content': 'hello'}}]},
            'first',
        )

        snapshot = tracker.snapshot_result()
        result = tracker.get_result()

        # Both should return trajectories
        assert len(snapshot['trajectories']) == 1
        assert len(result['trajectories']) == 1
