from __future__ import annotations

from nexus.extensions.harbor_bridge.agent_trajectory import chat_tracker_fallback_record
from nexus.extensions.harbor_bridge.bad_pattern_operator import detect_tool_call_bad_patterns, merge_tool_schemas
from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor
from nexus.trace.types import Span


def _call(name: str, arguments: object) -> dict:
    return {
        'id': 'call-1',
        'type': 'function',
        'function': {'name': name, 'arguments': arguments},
    }


def test_anthropic_tracker_schema_is_normalized_for_validation() -> None:
    tools = merge_tool_schemas([
        {
            'name': 'weather',
            'description': 'Get weather',
            'input_schema': {
                'type': 'object',
                'properties': {'city': {'type': 'string'}},
                'required': ['city'],
                'additionalProperties': False,
            },
        }
    ])
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': '', 'tool_calls': [_call('weather', {'city': 3, 'extra': 1})]}],
        tools,
        enabled_rules={'tool_call_args_invalid'},
    )

    # BPO v1.2.0 merges all violations from one tool call into one Finding.
    assert result['issue_counts'] == {'tool_call_args_invalid': 1}
    assert result['hits'][0]['detail']['n_violations'] == 2
    assert result['schema_registry_tool_count'] == 1
    assert result['schema_matched_call_count'] == 1
    assert result['schema_coverage'] == 1.0


def test_tracker_schema_overrides_incomplete_host_schema() -> None:
    host = [{'type': 'function', 'function': {'name': 'weather', 'parameters': {}}}]
    tracker = [{
        'name': 'weather',
        'input_schema': {
            'type': 'object',
            'properties': {'city': {'type': 'string'}},
            'required': ['city'],
        },
    }]
    tools = merge_tool_schemas(host, tracker)
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': '', 'tool_calls': [_call('weather', {})]}],
        tools,
        enabled_rules={'tool_call_args_invalid'},
    )

    assert result['issue_counts'] == {'tool_call_args_invalid': 1}
    assert tools[0]['function']['parameters']['required'] == ['city']


def test_offline_batch_replay_has_stable_schema_coverage_and_hit_rate() -> None:
    """Replay a batch-sized synthetic mix to lock down metric denominators."""
    tools = merge_tool_schemas([
        {
            'name': 'weather',
            'input_schema': {
                'type': 'object',
                'properties': {'city': {'type': 'string'}},
                'required': ['city'],
                'additionalProperties': False,
            },
        }
    ])
    detections = []
    for index in range(1000):
        arguments = {} if index < 25 else {'city': 'Shenzhen'}
        detections.append(
            detect_tool_call_bad_patterns(
                [
                    {'role': 'assistant', 'content': '', 'tool_calls': [_call('weather', arguments)]},
                    {'role': 'tool', 'tool_call_id': 'call-1', 'content': '{}'},
                ],
                tools,
                enabled_rules={'tool_call_args_invalid'},
            )
        )

    hit_rate = sum(bool(item['issue_counts'].get('tool_call_args_invalid')) for item in detections) / len(detections)
    mean_count = sum(item['issue_counts'].get('tool_call_args_invalid', 0) for item in detections) / len(detections)
    mean_schema_coverage = sum(item['schema_coverage'] for item in detections) / len(detections)

    assert hit_rate == 0.025
    assert mean_count == 0.025
    assert mean_schema_coverage == 1.0


def test_schema_coverage_is_zero_when_trajectory_has_no_tool_calls() -> None:
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': 'done'}],
        merge_tool_schemas([{'name': 'weather', 'input_schema': {'type': 'object'}}]),
        enabled_rules={'tool_call_args_invalid'},
    )

    assert result['named_tool_call_count'] == 0
    assert result['schema_matched_call_count'] == 0
    assert result['schema_coverage'] == 0.0


def test_schema_coverage_exposes_unmatched_tool_names() -> None:
    tools = merge_tool_schemas([
        {
            'name': 'weather',
            'input_schema': {'type': 'object', 'properties': {}},
        }
    ])
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                _call('weather', {}),
                {
                    'id': 'call-2',
                    'type': 'function',
                    'function': {'name': 'calendar', 'arguments': {}},
                },
            ],
        }],
        tools,
        enabled_rules={'tool_call_args_invalid', 'tool_call_hallucination'},
    )

    assert result['named_tool_call_count'] == 2
    assert result['schema_matched_call_count'] == 1
    assert result['schema_coverage'] == 0.5
    assert result['issue_counts'] == {'tool_call_hallucination': 1}


def test_chat_tracker_to_detector_offline_pipeline_uses_advertised_schema() -> None:
    """Exercise the same tracker -> selected branch -> detector path used by HyCode."""
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    tools = [{
        'name': 'calendar_create',
        'description': 'Create a calendar event',
        'input_schema': {
            'type': 'object',
            'properties': {
                'title': {'type': 'string'},
                'attendees': {'type': 'array', 'items': {'type': 'string'}},
            },
            'required': ['title'],
            'additionalProperties': False,
        },
    }]
    messages = [{'role': 'user', 'content': 'Create the event'}]
    output = {
        'choices': [{
            'message': {
                'role': 'assistant',
                'content': '',
                'tool_calls': [{
                    'id': 'call-1',
                    'type': 'function',
                    'function': {
                        'name': 'calendar_create',
                        'arguments': {'attendees': 'alice@example.com', 'unexpected': True},
                    },
                }],
            }
        }]
    }
    tracker.on_end(
        Span(
            trace_id='trace',
            span_id='span',
            parent_span_id='',
            name='openai_completion',
            start_time_unix_nano=0,
            attributes={'inputs': {'messages': messages, 'kwargs': {'tools': tools}}, 'outputs': output},
        ),
        SpanEndEvent(),
    )

    tracked_messages, tracked_tools, candidate_count = chat_tracker_fallback_record(tracker.snapshot_result())
    result = detect_tool_call_bad_patterns(
        tracked_messages,
        merge_tool_schemas(tracked_tools),
        enabled_rules={'tool_call_args_invalid'},
    )

    assert candidate_count == 1
    # Missing required + type mismatch + unexpected argument are one
    # call-level Finding with three structured violations.
    assert result['issue_counts'] == {'tool_call_args_invalid': 1}
    assert result['hits'][0]['detail']['n_violations'] == 3
    assert result['schema_registry_tool_count'] == 1
    assert result['schema_matched_call_count'] == 1
    assert result['schema_coverage'] == 1.0
