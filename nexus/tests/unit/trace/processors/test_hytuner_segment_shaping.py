from __future__ import annotations

from nexus.runner.hytuner import HyTunerSegment
from nexus.trace.processors.hytuner import HytunerProcessor
from nexus.trace.processors.hytuner_segment_shaping import (
    SegmentShapingClassifier,
    normalize_segment_shaping_config,
)


def _shaping_processor() -> HytunerProcessor:
    return HytunerProcessor(
        file_path=None,
        trajectory_id='traj-1',
        segment_shaping_config={
            'mode': 'segment',
            'policy': 'tool_error_turn_penalty_only',
            'tool_error_turn_penalty': -0.02,
        },
    )


def test_processor_accepts_segment_shaping_config_dict() -> None:
    processor = HytunerProcessor(
        file_path=None,
        trajectory_id='traj-1',
        segment_shaping_config={
            'mode': 'segment',
            'policy': 'tool_error_turn_penalty_only',
            'tool_error_turn_penalty': -0.02,
        },
    )

    assert processor._shaping_enabled is True
    assert processor._segment_shaping.requires_tool_call_mapping is False


def test_finalize_current_segment_preserves_shaping_fields_without_marking_skipped() -> None:
    processor = _shaping_processor()
    processor._first_prompt_captured = True
    processor.prompt_ids = [1, 2]
    processor.raw_input_ids = [1, 2]
    processor.response_ids = [10, 11]
    processor.response_mask = [1, 1]
    processor.response_logprobs = [-0.1, -0.2]
    processor.num_turns = 1
    processor.messages = [
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call-1', 'function': {'name': 'search'}}]},
    ]
    processor.per_turn_shaping_rewards = [-0.02]
    processor.per_turn_response_end_indices = [0]
    processor._pending_boundary = 1

    processor._finalize_current_segment()

    assert len(processor.segments) == 1
    extra_fields = processor.segments[0]['extra_fields']
    assert extra_fields['per_turn_shaping_rewards'] == [-0.02, 0.0]
    assert extra_fields['per_turn_response_end_indices'] == [0, 1]
    assert extra_fields['shaping_trailing_tool_call_turns'] == 1
    assert extra_fields['shaping_multi_segment_skipped'] is False

    assert processor.per_turn_shaping_rewards == []
    assert processor.per_turn_response_end_indices == []
    assert processor._pending_boundary is None
    assert processor.shaping_trailing_tool_call_turns == 0
    assert processor.shaping_multi_segment_skipped is False


def test_standalone_summary_segment_gets_empty_shaping_fields() -> None:
    processor = _shaping_processor()

    processor._record_standalone_segment({
        'prompt_ids': [1],
        'raw_prompt_ids': [1],
        'response_ids': [2, 3],
        'response_logprobs': [-0.1, -0.2],
        'messages': [{'role': 'assistant', 'content': 'summary'}],
    })

    assert len(processor.segments) == 1
    extra_fields = processor.segments[0]['extra_fields']
    assert extra_fields == {
        'per_turn_shaping_rewards': [],
        'per_turn_response_end_indices': [],
        'shaping_trailing_tool_call_turns': 0,
        'shaping_multi_segment_skipped': False,
    }


def test_tool_error_turn_penalty_detects_multimodal_tool_error_prefix() -> None:
    classifier = SegmentShapingClassifier(
        normalize_segment_shaping_config({
            'mode': 'segment',
            'policy': 'tool_error_turn_penalty_only',
            'tool_error_turn_penalty': -0.02,
        })
    )

    reward = classifier.aggregate_turn(
        [
            {
                'role': 'tool',
                'content': [
                    {'type': 'text', 'text': 'Tool error: backend failed'},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc'}},
                ],
            }
        ],
        {},
    )

    assert reward == -0.02


def test_legacy_classifier_preserves_consecutive_escalation_state() -> None:
    classifier = SegmentShapingClassifier(
        normalize_segment_shaping_config({
            'mode': 'segment',
            'policy': 'legacy',
            'weights': {'invalid_params': -0.1},
            'error_taxonomy': 'controllable_v1',
            'escalation_factor': 1.0,
        })
    )
    tool_messages = [
        {
            'role': 'tool',
            'tool_call_id': 'call-1',
            'content': 'Tool error: empty query',
        }
    ]
    tool_call_mapping = {'call-1': 'search'}

    first_reward = classifier.aggregate_turn(tool_messages, tool_call_mapping)
    second_reward = classifier.aggregate_turn(tool_messages, tool_call_mapping)

    assert first_reward == -0.1
    assert second_reward == -0.2


def test_hytuner_segment_preserves_extra_fields_and_uses_independent_defaults() -> None:
    segment = HyTunerSegment(
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[1],
        extra_fields={'per_turn_shaping_rewards': [-0.02]},
    )
    assert segment.extra_fields == {'per_turn_shaping_rewards': [-0.02]}

    first_default = HyTunerSegment(prompt_ids=[1], response_ids=[2], response_mask=[1])
    second_default = HyTunerSegment(prompt_ids=[3], response_ids=[4], response_mask=[1])
    first_default.extra_fields['x'] = 1
    assert second_default.extra_fields == {}
