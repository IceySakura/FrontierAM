"""Tests for HytunerProcessor routed_experts propagation.

The client accumulates R3 routing deltas into a whole-segment numpy array. Each
completion span carries the latest array, and the processor keeps that value for
the final or snapshot result.
"""

from __future__ import annotations

import numpy as np

from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.hytuner import HytunerProcessor
from nexus.trace.types import Span


def _completion_span(training_data: dict, span_id: str) -> Span:
    return Span(
        trace_id='trace',
        span_id=span_id,
        parent_span_id='',
        name='openai_completion',
        start_time_unix_nano=0,
        attributes={'_hytuner_training_data': training_data},
    )


def _agent_inference_span(span_id: str) -> Span:
    return Span(
        trace_id='trace',
        span_id=span_id,
        parent_span_id='',
        name='agent_inference',
        start_time_unix_nano=0,
        attributes={'type': 'agent_inference'},
    )


def _turn(
    processor: HytunerProcessor,
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    routed_experts: np.ndarray | None,
    span_id: str,
) -> None:
    """Feed one completion turn: buffer the completion span, then flush via agent_inference."""
    training_data = {
        'request_id': 'req-0',
        'prompt_ids': prompt_ids,
        'raw_prompt_ids': prompt_ids,
        'response_ids': response_ids,
        'response_logprobs': [0.0] * len(response_ids),
        'routed_experts': routed_experts,
        'messages': [{'role': 'assistant', 'content': 'x'}],
        'extra_fields': None,
    }
    processor.on_end(_completion_span(training_data, f'{span_id}-c'), SpanEndEvent())
    processor.on_end(_agent_inference_span(f'{span_id}-a'), SpanEndEvent())


class TestHytunerRoutedExperts:
    def test_single_turn_surfaces_routed_experts(self) -> None:
        """get_result() exposes routed_experts from a single completion."""
        processor = HytunerProcessor(file_path=None, trajectory_id='0_test')
        routed_experts = np.arange(16, dtype=np.uint8).reshape(4, 2, 2)
        _turn(
            processor,
            prompt_ids=[1, 2, 3],
            response_ids=[4, 5],
            routed_experts=routed_experts,
            span_id='t1',
        )

        result = processor.get_result()
        np.testing.assert_array_equal(result['routed_experts'], routed_experts)

    def test_multi_turn_keeps_latest_whole_segment_array(self) -> None:
        """The latest completion carries the whole-segment routing array."""
        processor = HytunerProcessor(file_path=None, trajectory_id='0_test')
        first = np.full((4, 2, 2), 1, dtype=np.uint8)
        latest = np.concatenate((first, np.full((3, 2, 2), 2, dtype=np.uint8)))
        _turn(processor, prompt_ids=[1, 2, 3], response_ids=[4, 5], routed_experts=first, span_id='t1')
        # Turn 2: prompt grows by a tool token (6); new response is [7, 8].
        _turn(
            processor,
            prompt_ids=[1, 2, 3, 4, 5, 6],
            response_ids=[7, 8],
            routed_experts=latest,
            span_id='t2',
        )

        result = processor.get_result()
        np.testing.assert_array_equal(result['routed_experts'], latest)

    def test_missing_routed_experts_defaults_to_none(self) -> None:
        """Non-MoE completions leave routed_experts unset."""
        processor = HytunerProcessor(file_path=None, trajectory_id='0_test')
        _turn(processor, prompt_ids=[1, 2, 3], response_ids=[4, 5], routed_experts=None, span_id='t1')

        result = processor.get_result()
        assert result['routed_experts'] is None

    def test_snapshot_result_surfaces_routed_experts(self) -> None:
        """snapshot_result() also exposes the latest routing array."""
        processor = HytunerProcessor(file_path=None, trajectory_id='0_test')
        routed_experts = np.arange(16, dtype=np.uint8).reshape(4, 2, 2)
        # snapshot_result does not flush buffered completions, so flush via agent_inference first.
        _turn(
            processor,
            prompt_ids=[1, 2, 3],
            response_ids=[4, 5],
            routed_experts=routed_experts,
            span_id='t1',
        )

        snapshot = processor.snapshot_result()
        np.testing.assert_array_equal(snapshot['routed_experts'], routed_experts)
