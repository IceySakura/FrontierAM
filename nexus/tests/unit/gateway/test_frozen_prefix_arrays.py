"""Keep frozen token spans as arrays until the backend request boundary."""

import copy
import pickle

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from _fakes import FakeTokenizer, memory_store

from nexus.gateway.mm_expansion import partition_expanded_ids
from nexus.gateway.processor import Processor
from nexus.gateway.session import GatewaySession
from nexus.gateway.token_budget import TokenBudgetState
from nexus.gateway.trajectory import TokenAccount, TrajectoryNode
from nexus.gateway.trajectory_assembler import TrajectoryAssembler
from nexus.gateway.types import BackendResult


class _RecordingBackend:
    def __init__(self) -> None:
        self.prompts: list[list[int]] = []

    async def create_completion(self, request_id: str, **kwargs: Any) -> BackendResult:
        del request_id
        prompt_ids = kwargs['prompt_ids']
        assert isinstance(prompt_ids, list), 'the backend retains its existing list interface'
        assert all(type(token) is int for token in prompt_ids), 'wire tokens must be native Python integers'
        self.prompts.append(prompt_ids)
        return BackendResult(token_ids=[ord('x')], log_probs=[-0.25])


@pytest.mark.parametrize('with_media', [False, True])
def test_frozen_prefix_borrows_sealed_arrays_without_scalar_conversion(with_media: bool) -> None:
    tokens = np.arange(8192, dtype=np.int32)
    raw_tokens = tokens[::2] if with_media else None
    node = TrajectoryNode(
        0,
        0,
        None,
        'client',
        children=[],
        templated_message='prompt',
        token_account=TokenAccount(token_ids=tokens, raw_token_ids=raw_tokens),
    )

    prefix = TrajectoryAssembler._frozen_prefix([node])

    assert prefix.message_token_ids[0] is tokens, 'frozen history must retain the sealed token array'
    assert prefix.rollout_message_token_ids[0] is (raw_tokens if with_media else tokens), 'retain the engine view'
    if not with_media:
        assert prefix.raw_message_token_ids == (), 'identical token views must share the frozen partition'


async def test_reused_prefix_stays_array_through_render_and_processor_output() -> None:
    tokenizer = FakeTokenizer()
    backend = _RecordingBackend()
    processor = Processor(backend, tokenizer=tokenizer)
    session = GatewaySession('frozen-arrays', processor, memory_store('frozen-arrays'), scaffold_policy='generic')
    messages = [{'role': 'user', 'content': 'question'}]
    first = await session.create_completion(messages)
    messages += [first.message, {'role': 'user', 'content': 'continue'}]
    prefix = session.segment_assembler.frozen_prefix(messages)
    assert prefix is not None, 'the second turn must reuse frozen history'
    raw_ids, training_ids, _, render = await processor.render_and_tokenize(messages, None, frozen_prefix=prefix)
    for stored, reused in zip(prefix.message_token_ids, render.message_token_ids, strict=False):
        assert reused is stored, 'rendering must not copy the frozen per-message token arrays'
    assert raw_ids is training_ids, 'text-only requests must share raw and training prompt storage'
    assert isinstance(render.generation_prompt_ids, np.ndarray), 'generation prompt IDs remain numeric arrays'
    seen_prompts = []

    def prepare(raw: np.ndarray, training: np.ndarray, request: dict, uses_default: bool) -> dict:
        del uses_default
        assert isinstance(raw, np.ndarray), 'budget preparation must receive arrays'
        assert isinstance(training, np.ndarray), 'budget preparation must receive arrays'
        seen_prompts.append(raw)
        return request

    result = await processor.create_completion('second', messages, frozen_prefix=prefix, prepare_generation=prepare)
    assert result.raw_prompt_ids is seen_prompts[0], 'processor output must retain its prepared array'
    assert result.prompt_ids is result.raw_prompt_ids, 'text-only processor output must share token views'
    np.testing.assert_array_equal(result.prompt_ids, backend.prompts[-1])
    np.testing.assert_array_equal(result.prompt_ids, training_ids)
    await session.finish_pending_publications()


async def test_prompt_list_conversion_precedes_inference_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []

    class TrackedTokenArray(np.ndarray):
        def tolist(self) -> list[int]:
            events.append('tolist')
            return super().tolist()

    clock = iter([10.0, 10.75])

    def perf_counter() -> float:
        events.append('clock')
        return next(clock)

    processor = Processor(_RecordingBackend(), tokenizer=FakeTokenizer())
    messages = [{'role': 'user', 'content': 'question'}]
    raw_ids, training_ids, request_mm, render = await processor.render_and_tokenize(messages, None)
    monkeypatch.setattr(
        processor,
        'render_and_tokenize',
        AsyncMock(return_value=(raw_ids.view(TrackedTokenArray), training_ids, request_mm, render)),
    )
    monkeypatch.setattr('nexus.gateway.processor.time', SimpleNamespace(perf_counter=perf_counter))

    result = await processor.create_completion('timing', messages)

    assert events == ['tolist', 'clock', 'clock'], 'prompt conversion must precede the inference timer'
    assert result.wall_inference_time == 0.75, 'the wall timer must still cover the backend call'


def test_media_partition_returns_views_without_expanded_token_lists() -> None:
    raw_parts = [np.array([1, 99, 2], dtype=np.int32), np.array([], dtype=np.int32), np.array([3], dtype=np.int32)]
    expanded = np.array([1, 99, 99, 99, 2, 3], dtype=np.int32)

    parts = partition_expanded_ids(raw_parts, expanded, 99)

    assert all(isinstance(part, np.ndarray) for part in parts), 'expanded message spans must remain arrays'
    assert parts[1].shape == (0,), 'empty spans remain one-dimensional'
    assert np.shares_memory(parts[0], expanded), 'expanded spans should slice the processor buffer'
    np.testing.assert_array_equal(np.concatenate(parts), expanded)


def test_token_budget_commits_array_snapshot_and_preserves_prefix_accounting() -> None:
    state = TokenBudgetState(token_budget=20, token_budget_include=['agent_output', 'tool_observation'])
    prompt = np.array([1, 2], dtype=np.int32)
    state.record_generation('main', prompt, prompt, [3, 4])
    committed = state._accounted_prompt_ids['main']
    assert isinstance(committed, np.ndarray), 'token budget must not reconstruct a Python token list'
    prompt[0] = 99
    np.testing.assert_array_equal(committed, [1, 2, 3, 4])
    next_prompt = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    state.account_tool_observation_suffix('main', next_prompt, [{'role': 'tool', 'content': 'observation'}], 2)
    request = state.prepare_generation('main', next_prompt, next_prompt, {'max_tokens': 100}, True)
    assert request['max_tokens'] == 16, 'two generated and two observation tokens consume the budget'


@pytest.mark.parametrize('offsets', [[0, 3, 3, 5], [0, 3, 1, 5]])
def test_token_attribution_preserves_monotone_ownership_with_overlapping_offsets(offsets: list[int]) -> None:
    class OffsetTokenizer:
        def encode(self, text: str, **kwargs: Any) -> list[int]:
            del kwargs
            return [10, 11, 12, 13] if text == 'abcdef' else [99]

        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            del text, kwargs
            return {'input_ids': [10, 11, 12, 13], 'offset_mapping': [(start, start + 1) for start in offsets]}

    processor = Processor(_RecordingBackend(), tokenizer=OffsetTokenizer())

    messages, generation = processor._canonical_span_ids(['ab', 'cd'], 'ef')

    assert [part.tolist() for part in messages] == [[10], [11, 12]], 'tokens must never move to an earlier span'
    assert generation.tolist() == [13], 'generation keeps the final span'


def test_frozen_prefix_pickle_and_copy_retain_numeric_array_values() -> None:
    tokens = np.arange(20, dtype=np.int32)[::2]
    node = TrajectoryNode(
        0,
        0,
        None,
        'client',
        children=[],
        templated_message='prompt',
        token_account=TokenAccount(token_ids=tokens, raw_token_ids=tokens[:3]),
    )
    prefix = TrajectoryAssembler._frozen_prefix([node])

    for restored in (copy.deepcopy(prefix), pickle.loads(pickle.dumps(prefix, protocol=5))):
        assert isinstance(restored.message_token_ids[0], np.ndarray), 'prefix round-trip must retain arrays'
        np.testing.assert_array_equal(restored.message_token_ids[0], tokens)
        np.testing.assert_array_equal(restored.rollout_message_token_ids[0], tokens[:3])


async def test_empty_prompt_render_keeps_int32_arrays() -> None:
    processor = Processor(_RecordingBackend(), tokenizer=FakeTokenizer())

    raw_ids, training_ids, _, render = await processor.render_and_tokenize([], None, add_generation_prompt=False)

    assert raw_ids is training_ids, 'empty text-only prompts must share their token view'
    assert raw_ids.shape == (0,), 'empty prompts must remain one-dimensional'
    assert raw_ids.dtype == np.int32, 'empty prompts must retain the token dtype'
    assert render.generation_prompt_ids.dtype == np.int32, 'empty generation prompts must retain the token dtype'


async def test_media_prefix_reuses_expanded_and_raw_arrays_while_only_expanding_new_images() -> None:
    class MediaPlugin:
        def __init__(self) -> None:
            self.images: list[list[str]] = []

        async def process_multi_modal_info(self, messages: list[dict]) -> dict[str, Any]:
            del messages
            return {'images': ['old-image', 'new-image']}

        def get_placeholder_token_ids(self) -> dict[str, int]:
            return {'images': 99}

        def expand_prompt(self, text: str, multi_modal_data: dict[str, Any]) -> Any:
            self.images.append(multi_modal_data['images'])
            tokens = [token for char in text for token in ([99, 99, 99] if char == 'c' else [ord(char)])]
            return SimpleNamespace(token_ids=tokens, multi_modal_items=None)

    plugin = MediaPlugin()
    tokens = np.array([97, 99, 99, 99, 98], dtype=np.int32)
    raw_tokens = np.array([97, 99, 98], dtype=np.int32)
    node = TrajectoryNode(
        0,
        0,
        None,
        'client',
        children=[],
        templated_message='acb',
        token_account=TokenAccount(token_ids=tokens, raw_token_ids=raw_tokens),
    )
    prefix = TrajectoryAssembler._frozen_prefix([node])
    processor = Processor(_RecordingBackend(), tokenizer=FakeTokenizer(), mm_plugin=plugin)

    render, _ = processor._build_prompt_render(
        [{'role': 'user', 'content': 'acb'}, {'role': 'user', 'content': 'dce'}],
        None,
        text='acbdce',
        add_generation_prompt=False,
        frozen=prefix,
        multi_modal_data={'images': ['old-image', 'new-image']},
    )

    assert render.message_token_ids[0] is tokens, 'expanded prefix must not be copied'
    assert render.raw_message_token_ids[0] is raw_tokens, 'raw prefix must not be copied'
    assert plugin.images == [['new-image']], 'previous images must not be reprocessed'
    np.testing.assert_array_equal(render.token_ids, [97, 99, 99, 99, 98, 100, 99, 99, 99, 101])
    np.testing.assert_array_equal(render.rollout_token_ids, [97, 99, 98, 100, 99, 101])
