from __future__ import annotations

import copy

import pytest

from nexus.gateway.policies.hycode import COMPACT_SUMMARY_MARKER, COMPACT_TRIGGER_MARKER, HYCODE_POLICY
from nexus.gateway.scaffold_policy import RequestEvent, ResponseEvent, build_scaffold_policy
from nexus.gateway.trajectory_assembler import TrajectoryAssembler


@pytest.mark.parametrize('policy_name', ['generic', 'nexus_main_agent', 'hycode'])
@pytest.mark.parametrize(
    ('declared_kind', 'expected_kind', 'expected_source'),
    [
        (None, 'agent_inference', 'default'),
        ('summary_inference', 'summary_inference', 'request_meta'),
        ('compaction_resume', 'agent_inference', 'request_meta'),
        ('agent_inference', 'agent_inference', 'request_meta'),
        ('completion', 'agent_inference', 'request_meta'),
        ('worker', 'agent_inference', 'request_meta'),
    ],
)
def test_context_normalizes_explicit_invocation_without_mutating_wire_metadata(
    policy_name, declared_kind, expected_kind, expected_source
):
    metadata = {'request_id': 'request-1'}
    if declared_kind is not None:
        metadata['invocation_kind'] = declared_kind
    original = copy.deepcopy(metadata)
    assembler = TrajectoryAssembler(policy_name)
    context = assembler.extract_context([], request_meta=metadata)

    assert getattr(context, 'invocation_kind', None) == expected_kind
    assert context.invocation_source == expected_source
    assert context.request_meta == original
    assert metadata == original


@pytest.mark.parametrize('response_event', [False, True])
@pytest.mark.parametrize(
    ('messages', 'expected_kind', 'expected_source'),
    [
        ([{'role': 'user', 'content': COMPACT_TRIGGER_MARKER}], 'summary_inference', 'hycode_summary_inference'),
        (
            [{'role': 'user', 'content': [{'type': 'text', 'text': COMPACT_TRIGGER_MARKER}]}],
            'summary_inference',
            'hycode_summary_inference',
        ),
        ([{'role': 'user', 'content': COMPACT_SUMMARY_MARKER}], 'agent_inference', 'default'),
        (
            [
                {'role': 'user', 'content': COMPACT_TRIGGER_MARKER},
                {'role': 'assistant', 'content': 'old summary'},
                {'role': 'user', 'content': 'Continue the task'},
            ],
            'agent_inference',
            'default',
        ),
        (
            [
                {'role': 'user', 'content': COMPACT_TRIGGER_MARKER},
                {'role': 'assistant', 'content': 'old summary'},
            ],
            'agent_inference',
            'default',
        ),
    ],
)
def test_hycode_classifies_only_the_current_summary_request(messages, expected_kind, expected_source, response_event):
    request = RequestEvent(tuple(messages), None, {'agent_id': 'child-1'})
    event = ResponseEvent(request, {'role': 'assistant', 'content': 'answer'}) if response_event else request
    facts = HYCODE_POLICY.inspect_event(event)

    assert getattr(facts, 'invocation_kind', None) == expected_kind
    assert facts.invocation_source == expected_source
    assert facts.membership.chain_id == 'child-1'


@pytest.mark.parametrize('value', ['', 1, {}, []])
def test_malformed_explicit_invocation_is_rejected(value):
    assembler = TrajectoryAssembler('generic')
    with pytest.raises(ValueError, match='invocation_kind'):
        assembler.extract_context([], request_meta={'invocation_kind': value})


@pytest.mark.parametrize('declared_kind', ['agent_inference', 'compaction_resume', 'completion'])
def test_hycode_rejects_contradictory_current_summary_signals(declared_kind):
    request = RequestEvent(
        ({'role': 'user', 'content': COMPACT_TRIGGER_MARKER},),
        None,
        {'invocation_kind': declared_kind},
    )
    with pytest.raises(ValueError, match='invocation_kind'):
        HYCODE_POLICY.inspect_event(request)


def test_request_context_preserves_summary_classification_for_response_recording():
    assembler = TrajectoryAssembler('hycode')
    messages = [{'role': 'user', 'content': COMPACT_TRIGGER_MARKER}]
    context = assembler.extract_context(messages)
    request = RequestEvent(tuple(messages), None, None)
    response = ResponseEvent(request, {'role': 'assistant', 'content': 'summary'})

    # A response inspection without the request's detector result would classify
    # this untagged request as ordinary. The supplied snapshot must remain authoritative.
    assembler._policy = build_scaffold_policy('generic')
    facts = assembler._event_facts(response, context)

    assert getattr(context, 'invocation_kind', None) == 'summary_inference'
    assert facts.invocation_kind == context.invocation_kind
    assert facts.invocation_source == context.invocation_source
    assert context.request_meta is None


def test_nexus_compacted_input_is_an_ordinary_invocation():
    request = RequestEvent(({'role': 'user', 'content': 'history', '_is_compacted_summary': True},), None, None)
    facts = build_scaffold_policy('nexus_main_agent').inspect_event(request)

    assert facts.boundary_facts[0].segment_kind == 'compaction_resume'
    assert getattr(facts, 'invocation_kind', None) == 'agent_inference'


@pytest.mark.parametrize('has_resume_marker', [False, True])
def test_historical_summary_trigger_does_not_classify_the_current_output(has_resume_marker):
    assembler = TrajectoryAssembler('hycode')
    messages = [
        {'role': 'user', 'content': COMPACT_TRIGGER_MARKER},
        {'role': 'assistant', 'content': 'Historical summary'},
    ]
    if has_resume_marker:
        messages.insert(0, {'role': 'user', 'content': COMPACT_SUMMARY_MARKER})
    context = assembler.extract_context(messages)

    assembler.record_completion(messages, {'role': 'assistant', 'content': 'Continue'}, context=context)

    output = next(node for node in assembler.freeze().nodes.values() if node.generated_by == 'llm')
    assert context.invocation_kind == 'agent_inference'
    if has_resume_marker:
        assert output.segment_boundary.kind == 'compaction_resume'
    else:
        assert output.segment_boundary is None


@pytest.mark.parametrize('policy_name', ['generic', 'hycode', 'nexus_main_agent'])
@pytest.mark.parametrize('has_resume_marker', [False, True])
def test_explicit_summary_classification_also_drives_trajectory_segmentation(policy_name, has_resume_marker):
    assembler = TrajectoryAssembler(policy_name)
    content = COMPACT_SUMMARY_MARKER if has_resume_marker else 'A configured summary request'
    messages = [{'role': 'user', 'content': content}]
    metadata = {'invocation_kind': 'summary_inference'}
    context = assembler.extract_context(messages, request_meta=metadata)

    assembler.record_completion(messages, {'role': 'assistant', 'content': 'Summary'}, context=context)

    output = next(node for node in assembler.freeze().nodes.values() if node.generated_by == 'llm')
    assert context.invocation_kind == 'summary_inference'
    assert output.segment_boundary is not None
    assert output.segment_boundary.kind == 'summary_inference'


@pytest.mark.parametrize('policy_name', ['generic', 'hycode', 'nexus_main_agent'])
def test_repeated_summary_requests_do_not_leak_classification_into_ordinary_continuation(policy_name):
    assembler = TrajectoryAssembler(policy_name)
    messages = [{'role': 'user', 'content': 'A configured summary request'}]
    for index in range(2):
        assembler.record_completion(
            messages,
            {'role': 'assistant', 'content': f'Summary {index}'},
            request_meta={'invocation_kind': 'summary_inference'},
        )
    assembler.record_completion(
        [{'role': 'user', 'content': 'Continue'}],
        {'role': 'assistant', 'content': 'Ordinary output'},
    )

    generated = [node for node in assembler.freeze().nodes.values() if node.generated_by == 'llm']
    assert [node.segment_boundary.kind if node.segment_boundary else None for node in generated] == [
        'summary_inference',
        'summary_inference',
        None,
    ]
