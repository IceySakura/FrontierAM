from __future__ import annotations

import json

from types import SimpleNamespace

import pytest

from harbor.models.task.config import TaskOS

from nexus.extensions.harbor_bridge.agent_result import T3_AGENT_RESULT_ENV, verifier_env_with_t3_agent_result
from nexus.extensions.harbor_bridge.agent_trajectory import (
    HARBOR_TRAJECTORY_PROCESSOR_KEY,
    HARBOR_TRAJECTORY_PROCESSOR_NAME,
    T3_AGENT_TRAJECTORY_ENV,
    T3_AGENT_TRAJECTORY_SCHEMA_VERSION,
    build_t3_agent_trajectory,
    chat_tracker_fallback_record,
    chat_tracker_fallback_messages,
    strip_reasoning,
    trajectory_processor_config,
    upload_t3_agent_trajectory,
    verifier_env_with_t3_agent_trajectory,
    write_t3_agent_trajectory,
)
from nexus.trace.processors import span_processor_registry


class _FakeRuntime:
    def __init__(self, *, success: bool = True) -> None:
        self.success = success
        self.uploads = []

    async def upload_file(self, files, overwrite=False):
        self.uploads.append((files, overwrite))
        return SimpleNamespace(
            results=[
                SimpleNamespace(path=item.path, success=self.success, message='' if self.success else 'disk full')
                for item in files
            ]
        )


def test_dedicated_chat_tracker_config_is_stable() -> None:
    assert trajectory_processor_config() == {
        HARBOR_TRAJECTORY_PROCESSOR_KEY: {
            'name': HARBOR_TRAJECTORY_PROCESSOR_NAME,
            'args': {'tools_mode': 'branch_last'},
        }
    }
    processor = span_processor_registry.get(HARBOR_TRAJECTORY_PROCESSOR_NAME)(tools_mode='branch_last')
    assert processor.get_result() == {}


def test_chat_tracker_fallback_accepts_one_conversation_in_either_shape() -> None:
    messages = [{'role': 'assistant', 'content': 'tracked'}]

    assert chat_tracker_fallback_messages({'trajectories': [messages]}) == (messages, 1)
    assert chat_tracker_fallback_messages({'trajectories': [{'messages': messages, 'tools': []}]}) == (messages, 1)


def test_chat_tracker_fallback_record_adds_tools_without_changing_messages_api() -> None:
    messages = [{'role': 'assistant', 'content': 'tracked'}]
    tools = [{'name': 'calendar_create', 'input_schema': {'type': 'object'}}]
    snapshot = {'trajectories': [{'messages': messages, 'tools': tools}]}

    assert chat_tracker_fallback_record(snapshot) == (messages, tools, 1)
    assert chat_tracker_fallback_messages(snapshot) == (messages, 1)


def test_chat_tracker_fallback_record_accepts_snapshot_level_tools() -> None:
    messages = [{'role': 'assistant', 'content': 'tracked'}]
    tools = [{'name': 'calendar_create', 'input_schema': {'type': 'object'}}]
    snapshot = {'tools': tools, 'trajectories': [{'messages': messages}]}

    assert chat_tracker_fallback_record(snapshot) == (messages, tools, 1)
    assert chat_tracker_fallback_messages(snapshot) == (messages, 1)


def test_chat_tracker_fallback_refuses_to_guess_between_branches() -> None:
    first = [{'role': 'assistant', 'content': 'first'}]
    second = [{'role': 'assistant', 'content': 'second'}]

    assert chat_tracker_fallback_messages({'trajectories': [first, second]}) == ([], 2)
    assert chat_tracker_fallback_messages(None) == ([], 0)


def test_chat_tracker_fallback_selects_the_unique_main_branch() -> None:
    main = [{'role': 'assistant', 'content': 'main answer'}]
    child = [{'role': 'assistant', 'content': 'child answer'}]

    assert chat_tracker_fallback_messages({
        'trajectories': [
            {'role': 'main:child-1', 'messages': child},
            {'role': 'main', 'messages': main},
        ]
    }) == (main, 2)
    assert chat_tracker_fallback_messages({
        'trajectories': [
            {'role': 'main', 'messages': main},
            {'role': 'main', 'messages': child},
        ]
    }) == ([], 2)


def test_chat_tracker_fallback_matches_the_reported_final_answer_without_roles() -> None:
    draft = [{'role': 'assistant', 'content': 'draft answer'}]
    final = [{'role': 'assistant', 'content': 'final answer'}]

    assert chat_tracker_fallback_messages(
        {'trajectories': [{'messages': draft}, {'messages': final}]},
        final_answer='final answer',
    ) == (final, 2)


def test_messages_keep_tool_calls_and_drop_every_reasoning_shape() -> None:
    """Graders pair tool calls with results by id; reasoning must not reach a verifier."""
    messages = strip_reasoning([
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{'id': 'tooluse_abc', 'function': {'name': 'exec', 'arguments': '{}'}}],
            'reasoning_content': 'flat private',
            'reasoning_items': [
                {
                    'type': 'reasoning',
                    'summary': [{'type': 'summary_text', 'text': 'responses private'}],
                    'encrypted_content': 'opaque-private',
                }
            ],
            'thinking_blocks': [{'type': 'thinking', 'thinking': 'signed private'}],
            'reasoning': 'another private',
            'thinking': 'yet another private',
            'raw_response': '<think>raw private</think>',
        },
        {'role': 'tool', 'tool_call_id': 'tooluse_abc', 'content': 'ok'},
    ])

    assert messages == [
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{'id': 'tooluse_abc', 'function': {'name': 'exec', 'arguments': '{}'}}],
        },
        {'role': 'tool', 'tool_call_id': 'tooluse_abc', 'content': 'ok'},
    ]
    assert 'private' not in json.dumps(messages)


def test_responses_reasoning_is_removed_from_provider_specific_fields() -> None:
    messages = strip_reasoning([
        {
            'role': 'assistant',
            'content': 'visible answer',
            'provider_specific_fields': {
                'request_id': 'request-1',
                'reasoning_items': [{'type': 'reasoning', 'encrypted_content': 'nested-private'}],
                'raw_response': '<think>nested raw private</think>visible answer',
                'reasoningContentBlocks': [
                    {'reasoningText': {'text': 'bedrock private', 'signature': 'bedrock-secret-signature'}}
                ],
                'bedrock_metadata': {
                    'request_id': 'bedrock-request-1',
                    'reasoningContentBlocks': [
                        {'reasoningText': {'text': 'deep bedrock private', 'signature': 'deep-secret-signature'}}
                    ],
                },
                'anthropic_content': [
                    {'type': 'thinking', 'thinking': 'anthropic private', 'signature': 'secret-signature'},
                    {'type': 'text', 'text': 'visible answer'},
                    {'type': 'tool_use', 'id': 'toolu_1', 'name': 'Bash', 'input': {'command': 'ls'}},
                ],
                'responses_output_items': [
                    {
                        'type': 'reasoning',
                        'summary': [{'type': 'summary_text', 'text': 'duplicated private'}],
                        'encrypted_content': 'duplicated-opaque',
                    },
                    {'type': 'message', 'content': [{'type': 'output_text', 'text': 'visible answer'}]},
                ],
            },
        }
    ])

    assert messages == [
        {
            'role': 'assistant',
            'content': 'visible answer',
            'provider_specific_fields': {
                'request_id': 'request-1',
                'bedrock_metadata': {'request_id': 'bedrock-request-1'},
                'anthropic_content': [
                    {'type': 'text', 'text': 'visible answer'},
                    {'type': 'tool_use', 'id': 'toolu_1', 'name': 'Bash', 'input': {'command': 'ls'}},
                ],
                'responses_output_items': [
                    {'type': 'message', 'content': [{'type': 'output_text', 'text': 'visible answer'}]}
                ],
            },
        }
    ]
    serialized = json.dumps(messages)
    assert 'private' not in serialized
    assert 'opaque' not in serialized
    assert 'secret-signature' not in serialized
    assert 'reasoningContentBlocks' not in serialized


def test_reasoning_carried_as_a_content_block_is_stripped_with_its_message() -> None:
    """Anthropic-format Agents report thinking here, so field stripping alone would leak it."""
    messages = strip_reasoning([
        {'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 'private', 'signature': 'sig'}]},
        {
            'role': 'assistant',
            'content': [
                {'type': 'thinking', 'thinking': 'more private'},
                {'type': 'tool_use', 'id': 'toolu_1', 'name': 'Bash', 'input': {'command': 'ls'}},
            ],
        },
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_1', 'content': 'ok'}]},
    ])

    # The reasoning-only turn is gone; the one that also acted keeps its call and its pairing.
    assert messages == [
        {
            'role': 'assistant',
            'content': [{'type': 'tool_use', 'id': 'toolu_1', 'name': 'Bash', 'input': {'command': 'ls'}}],
        },
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_1', 'content': 'ok'}]},
    ]
    assert 'private' not in json.dumps(messages)


def test_a_turn_that_was_already_empty_is_kept() -> None:
    """Dropping applies to what stripping emptied, not to what the Agent recorded empty."""
    conversation = [{'role': 'assistant', 'content': []}]

    assert strip_reasoning(conversation) == conversation


def test_system_and_user_turns_are_handed_over_untouched() -> None:
    """Which turns a judge should see is the verifying task's call, not this module's."""
    conversation = [
        {'role': 'system', 'content': 'you are an agent'},
        {'role': 'user', 'content': 'the task prompt'},
        {'role': 'assistant', 'content': 'done'},
    ]

    assert strip_reasoning(conversation) == conversation


@pytest.mark.parametrize('trajectory', [None, [], 'not a list', [None, 'junk']])
def test_non_conversations_yield_no_messages(trajectory) -> None:
    assert strip_reasoning(trajectory) == []


def test_envelope_is_small_versioned_and_reasoning_free() -> None:
    trajectory = build_t3_agent_trajectory(
        mode='agent',
        status='success',
        agent_output=SimpleNamespace(
            main_trajectory=[{'role': 'assistant', 'content': 'answer', 'reasoning_content': 'private'}]
        ),
    )

    assert trajectory.model_dump() == {
        'schema_version': T3_AGENT_TRAJECTORY_SCHEMA_VERSION,
        'mode': 'agent',
        'status': 'success',
        'messages': [{'role': 'assistant', 'content': 'answer'}],
    }
    assert 'private' not in trajectory.model_dump_json()


def test_envelope_is_empty_when_the_agent_exposes_no_trajectory() -> None:
    trajectory = build_t3_agent_trajectory(mode='nop', status='skipped', agent_output=None)

    assert trajectory.messages == []


@pytest.mark.asyncio
async def test_trajectory_is_written_once_on_host_and_uploaded_to_standard_path(tmp_path) -> None:
    trajectory = build_t3_agent_trajectory(
        mode='agent',
        status='success',
        agent_output=SimpleNamespace(main_trajectory=[{'role': 'assistant', 'content': 'answer'}]),
    )
    host_path = write_t3_agent_trajectory(SimpleNamespace(agent_dir=tmp_path / 'agent'), trajectory)
    runtime = _FakeRuntime()

    container_path = await upload_t3_agent_trajectory(runtime, host_path, environment_os=TaskOS.LINUX)

    assert container_path == '/logs/agent/trajectory.json'
    assert json.loads(host_path.read_text())['messages'] == [{'role': 'assistant', 'content': 'answer'}]
    (files, overwrite) = runtime.uploads[0]
    assert overwrite is True
    assert files[0].path == '/logs/agent/trajectory.json'


@pytest.mark.asyncio
async def test_trajectory_upload_failure_is_not_ignored(tmp_path) -> None:
    trajectory = build_t3_agent_trajectory(mode='agent', status='success')
    host_path = write_t3_agent_trajectory(SimpleNamespace(agent_dir=tmp_path / 'agent'), trajectory)

    with pytest.raises(RuntimeError, match='Failed to upload T3 AgentTrajectory'):
        await upload_t3_agent_trajectory(_FakeRuntime(success=False), host_path, environment_os=TaskOS.LINUX)


def test_verifier_env_is_untouched_for_tasks_that_do_not_ask_for_a_trajectory() -> None:
    """A task family grading deliverables must see no new variable."""
    base = verifier_env_with_t3_agent_result({'JUDGE_MODEL': 'test'}, environment_os=TaskOS.LINUX)

    assert T3_AGENT_TRAJECTORY_ENV not in base
    assert base == {'JUDGE_MODEL': 'test', T3_AGENT_RESULT_ENV: '/logs/agent/result.json'}


def test_verifier_env_injects_reserved_trajectory_path_when_asked() -> None:
    env = verifier_env_with_t3_agent_trajectory({'JUDGE_MODEL': 'test'}, environment_os=TaskOS.LINUX)

    assert env == {'JUDGE_MODEL': 'test', T3_AGENT_TRAJECTORY_ENV: '/logs/agent/trajectory.json'}


def test_verifier_env_rejects_task_override_of_trajectory_path() -> None:
    with pytest.raises(ValueError, match='reserved by Harbor Bridge'):
        verifier_env_with_t3_agent_trajectory(
            {T3_AGENT_TRAJECTORY_ENV: '/app/fake-trajectory.json'},
            environment_os=TaskOS.LINUX,
        )
