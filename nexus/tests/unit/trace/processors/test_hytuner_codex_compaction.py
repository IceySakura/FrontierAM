from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.runtime.server.services.llm_handler import LLMRequestHandler
from nexus.runtime.server.services.rsio_llm_bridge import (
    NEXUS_AGENT_COMPACTION_FIELD,
    NEXUS_COMPACTION_DETECTOR_FIELD,
)
from nexus.trace import get_current_span_id, observe, processor_context, update_span
from nexus.trace.controller import SpanController
from nexus.trace.processors.hytuner import HytunerProcessor


def _training_data(prompt: list[int], response: list[int], label: str) -> dict[str, Any]:
    return {
        'prompt_ids': prompt,
        'raw_prompt_ids': prompt,
        'response_ids': response,
        'response_logprobs': [-0.1] * len(response),
        'messages': [
            {'role': 'user', 'content': label},
            {'role': 'assistant', 'content': f'{label} output'},
        ],
    }


class _TracedOpenAIClient:
    config = SimpleNamespace(request_args={}, client_args={})
    supports_request_meta = True

    def __init__(self, calls: dict[str, dict[str, Any]]) -> None:
        self.calls = calls
        self.request_kwargs: list[dict[str, Any]] = []

    @observe('openai_completion')
    async def create_completion(self, **kwargs):  # noqa: ANN003, ANN202
        self.request_kwargs.append(kwargs)
        label = kwargs['messages'][-1]['content']
        if label == 'failed-summary':
            raise RuntimeError('summary inference failed')
        span_id = get_current_span_id()
        assert span_id is not None
        update_span(span_id, attributes={'_hytuner_training_data': self.calls[label]})
        return {'choices': [{'message': {'role': 'assistant', 'content': f'{label} output'}}]}


@pytest.mark.asyncio
async def test_codex_compaction_produces_trainable_segments() -> None:
    SpanController._instance = None
    processor = HytunerProcessor(file_path=None, trajectory_id='0_codex')
    client = _TracedOpenAIClient(
        {
            'main-before': _training_data([1, 2], [10, 11], 'main-before'),
            'summary': _training_data([1, 2, 10, 11, 20], [21, 22], 'summary'),
            'main-after': _training_data([30, 31], [32, 33], 'main-after'),
        }
    )
    handler = LLMRequestHandler(runtime=object(), openai_client=client)
    compact_request = {
        'messages': [{'role': 'user', 'content': 'summary'}],
        NEXUS_AGENT_COMPACTION_FIELD: True,
        NEXUS_COMPACTION_DETECTOR_FIELD: 'turn_metadata',
        'request_meta': {
            'trajectory_id': '0_codex',
            'invocation_kind': 'stale_value',
        },
    }

    with processor_context([processor]):
        await handler._produce_response(
            'before',
            {'messages': [{'role': 'user', 'content': 'main-before'}]},
            rsio_mode=False,
        )
        await handler._produce_response('compact', compact_request, rsio_mode=False)
        await handler._produce_response(
            'after',
            {'messages': [{'role': 'user', 'content': 'main-after'}]},
            rsio_mode=False,
        )
        result = processor.get_result()

    assert [segment['segment_type'] for segment in result['segments']] == [
        'conversation',
        'summary_inference',
    ]
    assert result['segments'][0]['prompt_ids'] == [1, 2]
    assert result['segments'][0]['response_ids'] == [10, 11]
    assert result['segments'][1]['prompt_ids'] == [1, 2, 10, 11, 20]
    assert result['segments'][1]['response_ids'] == [21, 22]
    assert result['prompt_ids'] == [30, 31]
    assert result['response_ids'] == [32, 33]
    assert any(item['type'] == 'agent_compaction' for item in result['agent_timeline'])
    assert all(NEXUS_AGENT_COMPACTION_FIELD not in kwargs for kwargs in client.request_kwargs)
    assert all(NEXUS_COMPACTION_DETECTOR_FIELD not in kwargs for kwargs in client.request_kwargs)
    assert 'request_meta' not in client.request_kwargs[0]
    assert client.request_kwargs[1]['request_meta'] == {
        'invocation_kind': 'summary_inference',
        'trajectory_id': '0_codex',
    }
    assert 'request_meta' not in client.request_kwargs[2]


@pytest.mark.asyncio
async def test_failed_codex_compaction_does_not_cut_current_segment() -> None:
    SpanController._instance = None
    processor = HytunerProcessor(file_path=None, trajectory_id='0_codex')
    client = _TracedOpenAIClient({'main-before': _training_data([1, 2], [10, 11], 'main-before')})
    handler = LLMRequestHandler(runtime=object(), openai_client=client)

    with processor_context([processor]):
        await handler._produce_response(
            'before',
            {'messages': [{'role': 'user', 'content': 'main-before'}]},
            rsio_mode=False,
        )
        with pytest.raises(RuntimeError, match='summary inference failed'):
            await handler._produce_response(
                'compact',
                {
                    'messages': [{'role': 'user', 'content': 'failed-summary'}],
                    NEXUS_AGENT_COMPACTION_FIELD: True,
                    NEXUS_COMPACTION_DETECTOR_FIELD: 'turn_metadata',
                },
                rsio_mode=False,
            )
        result = processor.get_result()

    assert 'segments' not in result
    assert result['prompt_ids'] == [1, 2]
    assert result['response_ids'] == [10, 11]
    assert not any(item['type'] == 'agent_compaction' for item in result['agent_timeline'])
