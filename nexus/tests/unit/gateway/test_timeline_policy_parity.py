from __future__ import annotations

from collections import UserDict
from copy import deepcopy
from types import SimpleNamespace

import pytest

from _fakes import memory_store

from nexus.gateway.scaffold_policy import (
    GenericScaffoldPolicy,
    ObservationEvent,
    RequestEvent,
    ResponseEvent,
    build_scaffold_policy,
)
from nexus.gateway.session import GatewaySession
from nexus.gateway.trajectory_assembler import TrajectoryAssembler


class AttributeModel:
    def __init__(self, value):
        self.value = value

    def model_dump(self, *, exclude_none):
        return deepcopy(self.value)


class StableId:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


class BrokenId:
    def __str__(self):
        raise ValueError('broken timeline id')


class TimelineStubProcessor:
    is_token_level = False

    def __init__(self):
        self.calls = 0

    async def create_completion(self, **_kwargs):
        self.calls += 1
        return SimpleNamespace(
            message={'role': 'assistant', 'content': 'answer'},
            prompt_ids=None,
            raw_prompt_ids=None,
            token_ids=None,
            log_probs=None,
            routed_experts=None,
            stop_reason='stop',
            usage=None,
            id=None,
            created=None,
            model=None,
            object=None,
            system_fingerprint=None,
            service_tier=None,
            choice_logprobs=None,
            choice_provider_specific_fields=None,
            extra_fields={},
            completion_text=None,
            prompt_render=None,
        )


def _request(meta):
    return RequestEvent(
        messages=({'role': 'user', 'content': 'question'},),
        tools=None,
        request_meta=meta,
    )


@pytest.mark.parametrize('policy_name', ['generic', 'nexus_main_agent'])
async def test_gateway_session_full_dump_uses_policy_normalized_observability_scalars(monkeypatch, policy_name):
    clock = iter((10.0, 12.5))
    monkeypatch.setattr('nexus.gateway.session.time.time', lambda: next(clock))
    session = GatewaySession(
        'timeline-session', TimelineStubProcessor(), memory_store('timeline-session'), scaffold_policy=policy_name
    )
    user = {'role': 'user', 'content': 'question'}
    answer = {'role': 'assistant', 'content': 'answer'}

    await session.create_completion(
        [user],
        request_meta={
            'request_id': 101,
            'agent_id': StableId('agent-202'),
            'iteration': '7',
            'retry': 'invalid',
            'sequence': 0,
            'seq': '9',
        },
    )

    assert session.end().dump() == {
        'schema_version': 'gateway-trajectory-v1',
        'session_id': 'timeline-session',
        'state': 'completed',
        'roots': [0],
        'nodes': [
            {
                'node_id': 0,
                'message_id': 0,
                'parent': None,
                'children': [1],
                'generated_by': 'client',
                'original_message': user,
            },
            {
                'node_id': 1,
                'message_id': 1,
                'parent': 0,
                'children': [],
                'generated_by': 'llm',
                'original_message': answer,
                'invocation': {'iteration': 7},
                'provider_facts': {'finish_reason': 'stop', 'abort_wait_time': 0.0},
                'timeline': [
                    {
                        'event_type': 'llm_request',
                        'request_id': '101',
                        'agent_id': 'agent-202',
                        'is_sidechain': False,
                        'start_time': 10.0,
                        'end_time': 12.5,
                        'duration': 2.5,
                        'status': 'OK',
                        'iteration': 7,
                        'sequence': 9,
                        'attributes': {'type': 'agent_inference', 'abort_wait_time': 0.0},
                    },
                    {
                        'source': 'session',
                        'type': 'inference',
                        'name': 'create_completion',
                        'start': 10.0,
                        'end': 12.5,
                        'duration': 2.5,
                        'request_id': '101',
                        'agent_id': 'agent-202',
                        'is_sidechain': False,
                        'iteration': 7,
                        'sequence': 9,
                    },
                ],
            },
        ],
        'chains': [{'chain_id': 'main', 'kind': 'main', 'parent_chain_id': None, 'roots': [0]}],
        'timeline': [],
        'final_reward': None,
        'termination_reason': None,
        'metrics': {},
        'extra': {},
    }


@pytest.mark.parametrize('field_name', ['request_id', 'agent_id'])
async def test_gateway_session_rejects_invalid_observability_id_before_processor_or_state_mutation(field_name):
    processor = TimelineStubProcessor()
    session = GatewaySession(
        'invalid-timeline-id', processor, memory_store('invalid-timeline-id'), scaffold_policy='generic'
    )

    with pytest.raises(ValueError, match='broken timeline id'):
        await session.create_completion(
            [{'role': 'user', 'content': 'question'}],
            request_meta={field_name: BrokenId()},
        )

    assert processor.calls == 0
    assert session.is_ended is False
    assert session.segment_assembler.trajectory.nodes == {}
    assert session.get_trajectory().dump() == {
        'schema_version': 'gateway-trajectory-v1',
        'session_id': 'invalid-timeline-id',
        'state': 'active',
        'roots': [],
        'nodes': [],
        'chains': [{'chain_id': 'main', 'kind': 'main', 'parent_chain_id': None, 'roots': []}],
        'timeline': [],
        'final_reward': None,
        'termination_reason': None,
        'metrics': {},
        'extra': {},
    }


def test_generic_response_timeline_matches_legacy_scalar_and_attribute_normalization():
    policy = GenericScaffoldPolicy(policy_id='timeline-parity')
    meta = {
        'event_type': 909,
        'start_time': '1.25',
        'end_time': 'invalid',
        'duration': '2.5',
        'request_id': 101,
        'agent_id': 202,
        'parent_agent_id': 303,
        'status': 404,
        'span_id': 505,
        'parent_span_id': 606,
        'iteration': '7',
        'retry': 'invalid',
        'sequence': '8',
        'timeline_attributes': AttributeModel(
            {
                'model': {'keep': 1, 'drop': None},
                'items': [{'keep': 2, 'drop': None}, None],
                'drop': None,
            }
        ),
        'attributes': {'meta': {'keep': 3, 'drop': None}, 'drop': None},
    }
    response_meta = {'provider': {'keep': 4, 'drop': None}, 'drop': None}

    facts = policy.inspect_event(
        ResponseEvent(
            request=_request(meta),
            message={'role': 'assistant', 'content': 'answer'},
            start_time=10.0,
            end_time=11.0,
            status='fallback',
            response_meta=response_meta,
        )
    )

    assert dict(facts.timeline) == {
        'event_type': '909',
        'request_id': '101',
        'agent_id': '202',
        'parent_agent_id': '303',
        'is_sidechain': False,
        'start_time': 1.25,
        'duration': 2.5,
        'status': '404',
        'span_id': '505',
        'parent_span_id': '606',
        'iteration': 7,
        'sequence': 8,
        'attributes': {
            'model': {'keep': 1},
            'items': [{'keep': 2}, None],
            'meta': {'keep': 3},
            'provider': {'keep': 4},
        },
    }


@pytest.mark.parametrize('policy_name', ['generic', 'nexus_main_agent'])
def test_request_events_do_not_project_or_validate_timeline_metadata(policy_name):
    facts = build_scaffold_policy(policy_name).inspect_event(
        _request(
            {
                'start_time': object(),
                'timeline_attributes': object(),
                'attributes': 42,
            }
        )
    )

    assert facts.timeline == ()


def test_generic_timeline_ignores_non_mapping_attributes_and_invalid_timing():
    facts = build_scaffold_policy('generic').inspect_event(
        ResponseEvent(
            request=_request(
                {
                    'start_time': 'invalid',
                    'end_time': object(),
                    'duration': [],
                    'timeline_attributes': ['not', 'a', 'mapping'],
                    'attributes': 42,
                }
            ),
            message={'role': 'assistant', 'content': 'answer'},
        )
    )

    assert facts.timeline == ()


@pytest.mark.parametrize('policy_name', ['generic', 'nexus_main_agent'])
def test_frozen_dump_ignores_top_level_user_dict_timeline_attributes(policy_name):
    assembler = TrajectoryAssembler(scaffold_policy=policy_name)
    request_meta = {
        'request_id': 'request',
        'timeline_attributes': UserDict({'unexpected_timeline': True}),
        'attributes': UserDict({'unexpected_meta': True}),
    }
    if policy_name == 'nexus_main_agent':
        request_meta['declared_chain'] = {'chain_id': 'worker', 'kind': 'worker'}

    assembler.record_completion(
        messages=[{'role': 'user', 'content': 'question'}],
        output_message={'role': 'assistant', 'content': 'answer'},
        request_meta=request_meta,
    )

    llm_timeline = next(
        node['timeline'] for node in assembler.freeze().dump()['nodes'] if node['generated_by'] == 'llm'
    )
    assert llm_timeline == [
        {
            'event_type': 'llm_request',
            'request_id': 'request',
            'is_sidechain': policy_name == 'nexus_main_agent',
            'status': 'OK',
            'attributes': {'type': 'agent_inference'},
        }
    ]


@pytest.mark.parametrize('policy_name', ['generic', 'nexus_main_agent'])
def test_timeline_policy_deep_copies_nested_user_dict_without_normalizing_it(policy_name):
    nested_attributes = UserDict({'keep': {'value': 1}, 'none': None})
    request_meta = {
        'request_id': 'request',
        'attributes': {'nested': nested_attributes},
    }
    if policy_name == 'nexus_main_agent':
        request_meta['declared_chain'] = {'chain_id': 'worker', 'kind': 'worker'}

    facts = build_scaffold_policy(policy_name).inspect_event(
        ResponseEvent(
            request=_request(request_meta),
            message={'role': 'assistant', 'content': 'answer'},
        )
    )
    nested_attributes['keep']['value'] = 99

    copied_nested = dict(facts.timeline)['attributes']['nested']
    assert isinstance(copied_nested, UserDict)
    assert copied_nested == {'keep': {'value': 1}, 'none': None}


def test_generic_zero_and_numeric_context_values_follow_legacy_truthiness_and_string_conversion():
    facts = build_scaffold_policy('generic').inspect_event(
        ResponseEvent(
            request=_request(
                {
                    'request_id': 0,
                    'agent_id': 0,
                    'parent_agent_id': 0,
                    'parent_id': 7,
                    'status': 0,
                    'span_id': 0,
                    'parent_span_id': 0,
                    'sequence': 0,
                    'seq': 9,
                }
            ),
            message={'role': 'assistant', 'content': 'answer'},
            status=8,
        )
    )

    assert dict(facts.timeline) == {
        'event_type': 'llm_request',
        'request_id': '0',
        'agent_id': '0',
        'parent_agent_id': '7',
        'is_sidechain': False,
        'status': '8',
        'span_id': '0',
        'parent_span_id': '0',
        'sequence': 9,
    }


def test_status_and_span_values_alone_do_not_satisfy_legacy_has_facts():
    facts = build_scaffold_policy('generic').inspect_event(
        ResponseEvent(
            request=_request({'status': 200, 'span_id': 12, 'parent_span_id': 34}),
            message={'role': 'assistant', 'content': 'answer'},
        )
    )

    assert facts.timeline == ()


@pytest.mark.parametrize(
    ('policy_name', 'request_meta', 'is_sidechain'),
    [
        ('generic', {'agent_id': 12}, False),
        (
            'nexus_main_agent',
            {
                'agent_id': 12,
                'declared_chain': {'chain_id': 'worker', 'kind': 'worker', 'parent_chain_id': 'main'},
            },
            True,
        ),
    ],
)
def test_observation_timeline_normalization_is_inherited_by_generic_and_nexus(
    policy_name,
    request_meta,
    is_sidechain,
):
    request_meta.update(
        {
            'start_time': '3',
            'end_time': '5',
            'status': 200,
            'attributes': {'type': 'overridden', 'nested': {'drop': None, 'keep': True}},
        }
    )
    facts = build_scaffold_policy(policy_name).inspect_event(
        ObservationEvent(
            message={'role': 'tool', 'content': 'result'},
            parent_messages=({'role': 'user', 'content': 'question'},),
            tools=None,
            request_meta=request_meta,
        )
    )

    assert dict(facts.timeline) == {
        'event_type': 'observation',
        'agent_id': '12',
        'is_sidechain': is_sidechain,
        'start_time': 3.0,
        'end_time': 5.0,
        'duration': 2.0,
        'status': '200',
        'attributes': {'type': 'agent_observation', 'nested': {'keep': True}},
    }


@pytest.mark.parametrize('policy_name', ['generic', 'nexus_main_agent'])
def test_frozen_dump_preserves_legacy_timeline_normalization_and_deep_copy(policy_name):
    assembler = TrajectoryAssembler(scaffold_policy=policy_name, session_id=f'{policy_name}-timeline')
    attributes = {'nested': {'keep': 1, 'drop': None}, 'items': [None, {'drop': None, 'keep': 2}]}
    response_meta = {'provider': {'keep': 3, 'drop': None}}
    request_meta = {
        'request_id': 12,
        'agent_id': 34,
        'start_time': '1.5',
        'end_time': '4.0',
        'status': 200,
        'attributes': attributes,
    }
    if policy_name == 'nexus_main_agent':
        request_meta['declared_chain'] = {'chain_id': 'worker', 'kind': 'worker'}

    assembler.record_completion(
        messages=[{'role': 'user', 'content': 'question'}],
        output_message={'role': 'assistant', 'content': 'answer'},
        request_meta=request_meta,
        response_meta=response_meta,
    )
    attributes['nested']['keep'] = 99
    response_meta['provider']['keep'] = 99

    llm_timeline = next(
        node['timeline'] for node in assembler.freeze().dump()['nodes'] if node['generated_by'] == 'llm'
    )
    assert llm_timeline == [
        {
            'event_type': 'llm_request',
            'request_id': '12',
            'agent_id': '34',
            'is_sidechain': policy_name == 'nexus_main_agent',
            'start_time': 1.5,
            'end_time': 4.0,
            'duration': 2.5,
            'status': '200',
            'attributes': {
                'type': 'agent_inference',
                'nested': {'keep': 1},
                'items': [None, {'keep': 2}],
                'provider': {'keep': 3},
            },
        }
    ]


@pytest.mark.parametrize('policy_name', ['generic', 'nexus_main_agent'])
def test_invalid_timeline_values_do_not_raise_or_leak_into_frozen_dump(policy_name):
    assembler = TrajectoryAssembler(scaffold_policy=policy_name)
    assembler.record_completion(
        messages=[{'role': 'user', 'content': 'question'}],
        output_message={'role': 'assistant', 'content': 'answer'},
        request_meta={
            'start_time': 'invalid',
            'end_time': object(),
            'duration': {},
            'timeline_attributes': 42,
            'attributes': ['not', 'a', 'mapping'],
        },
    )

    llm_timeline = next(
        node['timeline'] for node in assembler.freeze().dump()['nodes'] if node['generated_by'] == 'llm'
    )
    assert llm_timeline == [
        {
            'event_type': 'llm_request',
            'is_sidechain': False,
            'status': 'OK',
            'attributes': {'type': 'agent_inference'},
        }
    ]
