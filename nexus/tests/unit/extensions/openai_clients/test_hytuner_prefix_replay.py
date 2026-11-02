from __future__ import annotations

import copy
import json

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.core import ContextLengthExceededError, OpenAIConfig
from nexus.extensions.openai_clients.hytuner import HyTunerClient
from nexus.runtime.server.services.llm_handler import LLMRequestHandler
from nexus.runtime.server.services.responses_proxy.converter import openai_to_responses_events, responses_to_openai
from nexus.runtime.server.services.rsio_llm_bridge import NEXUS_AGENT_COMPACTION_FIELD
from nexus.trace import processor_context
from nexus.trace.controller import SpanController
from nexus.trace.processors.hytuner import HytunerProcessor


class _Tokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):  # noqa: ANN001, ANN003, ANN202
        rendered = ''
        for message in messages:
            rendered += f'<{message["role"]}>'
            if message['role'] == 'assistant':
                rendered += f'<think>{message.get("reasoning_content") or ""}</think>'
            rendered += message.get('content') or ''
            for call in message.get('tool_calls') or []:
                rendered += json.dumps(call['function'], sort_keys=True)
        return rendered + ('<assistant>' if add_generation_prompt else '')

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        return 'thought</think>answer'


class _Server:
    def __init__(self) -> None:
        self.prompts: list[list[int]] = []

    async def generate(self, **kwargs: Any) -> SimpleNamespace:
        self.prompts.append(list(kwargs['prompt_ids']))
        # Sampled IDs deliberately differ from encoding the parsed output text.
        token_ids = [100001, *map(ord, '</think>'), 100002]
        return SimpleNamespace(
            token_ids=token_ids, log_probs=[-0.1] * len(token_ids), stop_reason='stop', extra_fields={}
        )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> HyTunerClient:
    monkeypatch.setattr(SpanController, '_instance', None)
    result = HyTunerClient(
        OpenAIConfig(
            client_args={'tokenizer': _Tokenizer(), 'server_manager': _Server(), 'max_model_len': 4096},
            request_args={},
        )
    )

    async def parse_tool_calls(token_ids: list[int], request: Any = None) -> tuple[str, str, list]:
        return '', 'thought</think>answer', []

    monkeypatch.setattr(result, '_parse_tool_calls', parse_tool_calls)
    return result


def _response_items(response: Any) -> list[dict]:
    return [
        event['item']
        for event in openai_to_responses_events(response.model_dump())
        if event['type'] == 'response.output_item.done'
    ]


def _transitions(caplog: pytest.LogCaptureFixture) -> list[dict]:
    return [
        json.loads(record.getMessage().split(': ', 1)[1])
        for record in caplog.records
        if record.name == 'nexus.extensions.openai_clients.hytuner'
        and record.getMessage().startswith('HyTunerClient prefix/budget transition: ')
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('replay_format', ['cached', 'responses'])
@pytest.mark.parametrize('append_message', [False, True], ids=['equal-length', 'appended'])
async def test_equivalent_replay_preserves_sampled_prefix_and_segment_budget(
    client: HyTunerClient,
    replay_format: str,
    append_message: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    processor = HytunerProcessor(file_path=None, trajectory_id='0_replay')
    initial = {'role': 'user', 'content': 'question'}
    with processor_context([processor]):
        response = await client.create_completion([initial])
        processor.get_result()
        baseline = client._initial_prompt_len
        limit = processor.effective_response_limit
        sampled_prefix = list(client.last_tokens)
        if replay_format == 'responses':
            replayed = responses_to_openai({'input': [{'type': 'message', **initial}, *_response_items(response)]})[
                'messages'
            ]
        else:
            replayed = copy.deepcopy(client.last_messages)
        if append_message:
            replayed.append({'role': 'user', 'content': 'continue'})
        await client.create_completion(replayed)
        result = processor.get_result()

    assert client.server_manager.prompts[1][: len(sampled_prefix)] == sampled_prefix
    assert client._initial_prompt_len == baseline
    assert processor.effective_response_limit == limit
    assert result['num_turns'] == 2
    assert result['unhandled_prefix_reset'] is False
    assert not result.get('segments')
    assert not _transitions(caplog)


@pytest.mark.asyncio
async def test_consecutive_responses_roundtrips_keep_reasoning_and_one_segment(
    client: HyTunerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(['first thought</think>first answer', 'second thought</think>second answer', 'third</think>done'])

    async def parse_tool_calls(token_ids: list[int], request: Any = None) -> tuple[str, str, list]:
        return '', next(outputs), []

    monkeypatch.setattr(client, '_parse_tool_calls', parse_tool_calls)
    processor = HytunerProcessor(file_path=None, trajectory_id='0_roundtrip')
    items = [{'type': 'message', 'role': 'user', 'content': 'question'}]
    with processor_context([processor]):
        for _ in range(3):
            response = await client.create_completion(responses_to_openai({'input': items})['messages'])
            items.extend(_response_items(response))
        result = processor.get_result()
    assert [(message['reasoning_content'], message['content']) for message in client.last_messages[1:]] == [
        ('first thought', 'first answer'),
        ('second thought', 'second answer'),
        ('third', 'done'),
    ]
    assert result['num_turns'] == 3
    assert not result.get('segments')


@pytest.mark.asyncio
async def test_direct_training_data_observer_records_without_span_processor_context(
    client: HyTunerClient,
) -> None:
    processor = HytunerProcessor(file_path=None, trajectory_id='0_direct')
    client.set_training_data_observer(processor)

    await client.create_completion([{'role': 'user', 'content': 'question'}])
    result = processor.get_result()

    assert result['prompt_ids']
    assert result['raw_input_ids']
    assert result['response_ids']
    assert result['response_mask'] == [1] * len(result['response_ids'])
    assert result['num_turns'] == 1


@pytest.mark.asyncio
async def test_direct_training_data_observer_deduplicates_normal_span_callback(
    client: HyTunerClient,
) -> None:
    processor = HytunerProcessor(file_path=None, trajectory_id='0_direct_dedup')
    client.set_training_data_observer(processor)

    with processor_context([processor]):
        await client.create_completion([{'role': 'user', 'content': 'question'}])
        result = processor.get_result()

    assert result['num_turns'] == 1
    assert result['response_mask'] == [1] * len(result['response_ids'])


def test_forked_hytuner_client_inherits_training_data_observer(client: HyTunerClient) -> None:
    processor = HytunerProcessor(file_path=None, trajectory_id='0_fork')
    client.set_training_data_observer(processor)

    child = client.fork_for_agent(agent_id='child', parent_agent_id=None)

    assert child._training_data_observer is processor


@pytest.mark.asyncio
@pytest.mark.parametrize('fixed_limit', [None, 100])
async def test_each_prefix_reset_logs_budget_and_content_free_evidence(
    client: HyTunerClient,
    fixed_limit: int | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    if fixed_limit is not None:
        client.sampling_params['max_tokens'] = fixed_limit
    await client.create_completion([{'role': 'user', 'content': 'private original question'}])
    first_baseline = client._initial_prompt_len
    for content in ['private rewritten question', 'private second rewrite']:
        await client.create_completion([{'role': 'user', 'content': content}])
    transitions = _transitions(caplog)
    assert len(transitions) == 2
    first = transitions[0]
    assert first['trigger'] == 'prefix_reset'
    assert first['request_id'] == client.request_id
    assert first['completion_span_id']
    assert first['old_initial_prompt_len'] == first_baseline
    assert first['new_initial_prompt_len'] > first_baseline
    assert first['message_diffs'] == [{'index': 0, 'fields': ['content']}]
    assert first['first_render_diff'] is not None
    if fixed_limit is not None:
        assert first['old_effective_response_limit'] == first['new_effective_response_limit'] == fixed_limit
    assert 'private original question' not in caplog.text
    assert 'private rewritten question' not in caplog.text


@pytest.mark.asyncio
async def test_limit_change_without_prefix_reset_is_distinguished(
    client: HyTunerClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    await client.create_completion([{'role': 'user', 'content': 'question'}])
    baseline = client._initial_prompt_len
    client.max_model_len = 4000
    await client.create_completion([*client.last_messages, {'role': 'user', 'content': 'continue'}])
    (transition,) = _transitions(caplog)
    assert transition['trigger'] == 'effective_limit_change'
    assert transition['old_initial_prompt_len'] == transition['new_initial_prompt_len'] == baseline
    assert transition['old_effective_response_limit'] - transition['new_effective_response_limit'] == 96
    assert transition['message_diffs'] == []
    assert transition['first_render_diff'] is None


@pytest.mark.asyncio
async def test_rejected_rewrite_still_logs_the_budget_transition(
    client: HyTunerClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    await client.create_completion([{'role': 'user', 'content': 'question'}])
    with pytest.raises(ContextLengthExceededError):
        await client.create_completion([{'role': 'user', 'content': 'x' * 5000}])
    (transition,) = _transitions(caplog)
    assert transition['trigger'] == 'prefix_reset'
    assert transition['phase'] == 'budget_check'
    assert transition['new_effective_response_limit'] < 0


@pytest.mark.asyncio
async def test_processor_mismatch_identifies_the_buffered_completion(
    client: HyTunerClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    processor = HytunerProcessor(file_path=None, trajectory_id='0_mismatch')
    with processor_context([processor]):
        await client.create_completion([{'role': 'user', 'content': 'question'}])
        processor.get_result()
        client.max_model_len = 4000
        await client.create_completion([*client.last_messages, {'role': 'user', 'content': 'continue'}])
        with pytest.raises(ValueError, match='effective_response_limit changed') as error:
            processor.get_result()
    (transition,) = _transitions(caplog)
    assert '0_mismatch' in str(error.value)
    assert client.request_id in str(error.value)
    assert transition['completion_span_id'] in str(error.value)


@pytest.mark.asyncio
async def test_plateau_prefix_rewrite_masks_trajectory(
    client: HyTunerClient,
) -> None:
    # Fixed max_tokens keeps effective_response_limit on its plateau, so the rewrite is invisible to
    # the effective_response_limit raise; prefix_shortened is the only signal that fires.
    client.sampling_params['max_tokens'] = 100
    processor = HytunerProcessor(file_path=None, trajectory_id='0_fork')
    with processor_context([processor]):
        await client.create_completion([{'role': 'user', 'content': 'original question'}])
        await client.create_completion([{'role': 'user', 'content': 'rewritten question'}])
        result = processor.get_result()
    assert processor._mask_unhandled_prefix_reset is True
    assert result['termination_reason'] == 'UNHANDLED_ERROR'
    assert result['unhandled_prefix_reset'] is True
    # Only the first (pre-rewrite) turn was stitched; the rewritten turn was dropped, not concatenated.
    assert result['num_turns'] == 1


@pytest.mark.asyncio
async def test_masked_trajectory_ignores_later_append_turns(
    client: HyTunerClient,
) -> None:
    # The rewrite (turn 2) latches the mask; a normal append after it (turn 3) carries
    # prefix_shortened=False, so only the _mask_unhandled_prefix_reset latch stops it from being
    # stitched onto the desynced stream. Guards against regressing the early-return in _append_turn.
    client.sampling_params['max_tokens'] = 100
    processor = HytunerProcessor(file_path=None, trajectory_id='0_fork_then_append')
    with processor_context([processor]):
        await client.create_completion([{'role': 'user', 'content': 'original question'}])
        await client.create_completion([{'role': 'user', 'content': 'rewritten question'}])
        await client.create_completion([*client.last_messages, {'role': 'user', 'content': 'continue'}])
        result = processor.get_result()
    assert processor._mask_unhandled_prefix_reset is True
    assert result['termination_reason'] == 'UNHANDLED_ERROR'
    # The post-rewrite append must NOT be concatenated: only the single pre-rewrite turn survives.
    assert result['num_turns'] == 1


@pytest.mark.asyncio
async def test_limit_moving_prefix_rewrite_masks_without_failing_later_requests(
    client: HyTunerClient,
) -> None:
    # Unfixed max_tokens: the rewrite also moves effective_response_limit. The direct observer runs on the
    # request path, so a limit raise here would fail every later completion instead of masking once.
    processor = HytunerProcessor(file_path=None, trajectory_id='0_moving_rewrite')
    client.set_training_data_observer(processor)
    await client.create_completion([{'role': 'user', 'content': 'original question'}])
    await client.create_completion([{'role': 'user', 'content': 'a longer rewritten question'}])
    await client.create_completion([*client.last_messages, {'role': 'user', 'content': 'continue'}])
    result = processor.get_result()
    assert processor._mask_unhandled_prefix_reset is True
    assert result['termination_reason'] == 'UNHANDLED_ERROR'
    assert result['num_turns'] == 1


@pytest.mark.asyncio
async def test_mask_from_prefix_rewrite_is_not_overwritten_by_span_reason(
    client: HyTunerClient,
) -> None:
    client.sampling_params['max_tokens'] = 100
    processor = HytunerProcessor(file_path=None, trajectory_id='0_fork_env_done')
    with processor_context([processor]):
        await client.create_completion([{'role': 'user', 'content': 'original question'}])
        await client.create_completion([{'role': 'user', 'content': 'rewritten question'}])
        processor.get_result()
    span = SimpleNamespace(
        name='task_phase',
        start_time_unix_nano=0,
        end_time_unix_nano=1,
        attributes={'type': 'task_phase', 'agent.termination_reason': 'ENV_DONE'},
    )
    processor.on_end(span, SimpleNamespace())
    assert processor.termination_reason == 'UNHANDLED_ERROR'


@pytest.mark.asyncio
async def test_explicit_compaction_preserves_summary_segment_and_logs_invocation(
    client: HyTunerClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    processor = HytunerProcessor(file_path=None, trajectory_id='0_compaction')
    handler = LLMRequestHandler(runtime=object(), openai_client=client)
    with processor_context([processor]):
        await handler._produce_response(
            'before', {'messages': [{'role': 'user', 'content': 'long question' * 10}]}, rsio_mode=False
        )
        await handler._produce_response(
            'summary',
            {
                'messages': [*client.last_messages, {'role': 'user', 'content': 'summarize'}],
                NEXUS_AGENT_COMPACTION_FIELD: True,
            },
            rsio_mode=False,
        )
        await handler._produce_response(
            'after', {'messages': [{'role': 'user', 'content': 'short summary'}]}, rsio_mode=False
        )
        result = processor.get_result()
    assert [segment['segment_type'] for segment in result['segments']] == ['conversation', 'summary_inference']
    assert any(transition['invocation_kind'] == 'summary_inference' for transition in _transitions(caplog))
