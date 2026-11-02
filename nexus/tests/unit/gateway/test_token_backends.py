import asyncio

from collections.abc import Awaitable, Callable
from functools import partial
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.openai_clients.hytuner import _wait_for_abort_resume, capture_hytuner_execution_timeouts
from nexus.gateway.backend import HyTunerServerBackend, OpenAIAPIBackend, VllmCompletionsBackend
from nexus.gateway.types import BackendResult


def _output(token_id: int, stop_reason: str, *, log_probs=True, topk=True):
    extra = {}
    if topk:
        extra = {
            'resp_topk_logprob_indice': [[token_id + 100]],
            'resp_topk_logprob_value': [[-0.5]],
        }
    return SimpleNamespace(
        token_ids=[token_id],
        log_probs=[-0.1] if log_probs else None,
        routed_experts=None,
        num_preempted=0,
        stop_reason=stop_reason,
        extra_fields=extra,
    )


class _ServerManager:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.call_count = 0
        self.calls = []

    async def generate_tokens(self, **kwargs):
        self.call_count += 1
        self.calls.append(kwargs)
        return next(self.outputs)


class _LegacyServerManager:
    def __init__(self, output: SimpleNamespace | TypeError) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    async def generate_tokens(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Any = None,
        video_data: Any = None,
        audio_data: Any = None,
        priority: int = 0,
        request_meta: dict[str, Any] | None = None,
        wait_for_abort_resume: Callable[[asyncio.Event, str], Awaitable[float]] | None = None,
    ) -> SimpleNamespace:
        self.calls.append({
            'request_id': request_id,
            'prompt_ids': prompt_ids,
            'sampling_params': sampling_params,
            'image_data': image_data,
            'video_data': video_data,
            'audio_data': audio_data,
            'priority': priority,
            'request_meta': request_meta,
            'wait_for_abort_resume': wait_for_abort_resume,
        })
        if isinstance(self.output, TypeError):
            raise self.output
        return self.output


class _ModernServerManager(_LegacyServerManager):
    async def generate_tokens(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Any = None,
        video_data: Any = None,
        audio_data: Any = None,
        priority: int = 0,
        request_meta: dict[str, Any] | None = None,
        wait_for_abort_resume: Callable[[asyncio.Event, str], Awaitable[float]] | None = None,
        timeout_budget_provider: Callable[..., Any] | None = None,
    ) -> SimpleNamespace:
        self.timeout_budget_provider = timeout_budget_provider
        return await super().generate_tokens(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            priority=priority,
            request_meta=request_meta,
            wait_for_abort_resume=wait_for_abort_resume,
        )


class TestHyTunerServerBackend:
    @pytest.mark.parametrize('manager_type', [_LegacyServerManager, _ModernServerManager], ids=['legacy', 'modern'])
    async def test_strict_endpoint_signature_preserves_request_and_result(
        self, manager_type: type[_LegacyServerManager]
    ) -> None:
        manager = manager_type(_output(10, 'stop'))
        backend = HyTunerServerBackend(manager)
        request_meta = {'validate': True, 'trajectory_id': 'trajectory-7'}

        result = await backend.create_completion(
            'request',
            prompt_ids=[1, 2, 3],
            request_kwargs={'max_tokens': 1, 'temperature': 0.2, 'top_p': None, 'tools': []},
            image_data=['image'],
            video_data=['video'],
            audio_data=['audio'],
            priority=3,
            request_meta=request_meta,
        )

        assert manager.calls == [
            {
                'request_id': 'request',
                'prompt_ids': [1, 2, 3],
                'sampling_params': {'max_tokens': 1, 'temperature': 0.2},
                'image_data': ['image'],
                'video_data': ['video'],
                'audio_data': ['audio'],
                'priority': 3,
                'request_meta': {'validate': True, 'trajectory_id': 'trajectory-7'},
                'wait_for_abort_resume': _wait_for_abort_resume,
            }
        ], 'The endpoint should receive the original arguments in exactly one call'
        assert request_meta == {'validate': True, 'trajectory_id': 'trajectory-7'}, 'Caller metadata changed'
        assert result.token_ids == [10], 'The response tokens should pass through'
        assert result.log_probs == [-0.1], 'The response log probabilities should pass through'
        assert result.stop_reason == 'stop', 'The response stop reason should pass through'
        assert result.usage == {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4}, 'Usage changed'
        assert result.extra_fields == {
            'resp_topk_logprob_indice': [[110]],
            'resp_topk_logprob_value': [[-0.5]],
        }, 'Response extra fields should pass through'
        if isinstance(manager, _ModernServerManager):
            provider = manager.timeout_budget_provider
            assert isinstance(provider, partial), 'The modern endpoint should receive a bound timeout provider'
            assert provider.func is capture_hytuner_execution_timeouts, 'The provider should capture the execution timeouts'
            assert provider.keywords == {'track_shared_pause': True}, 'The provider should track shared pauses'

    @pytest.mark.parametrize('manager_type', [_LegacyServerManager, _ModernServerManager], ids=['legacy', 'modern'])
    async def test_endpoint_internal_type_error_propagates_without_retry(
        self, manager_type: type[_LegacyServerManager]
    ) -> None:
        error = TypeError('generation failed after dispatch')
        manager = manager_type(error)
        backend = HyTunerServerBackend(manager)

        with pytest.raises(TypeError) as raised:
            await backend.create_completion('request', prompt_ids=[1], request_kwargs={'max_tokens': 1})

        assert raised.value is error, 'The original endpoint error should propagate unchanged'
        assert len(manager.calls) == 1, 'An endpoint error must not trigger another generation'

    async def test_calls_complete_generation_endpoint_once(self):
        manager = _ServerManager([_output(10, 'stop')])
        backend = HyTunerServerBackend(manager)

        # request_meta is caller-owned and forwarded to the endpoint untouched
        # (validate rides it — the gateway never interprets it; see backend.py).
        result = await backend.create_completion(
            'request',
            prompt_ids=[1, 2, 3],
            request_kwargs={
                'max_tokens': 1,
                'temperature': 0.2,
                'top_p': None,
                'tools': [],
                'metadata': {'tenant': 'training'},
            },
            priority=3,
            request_meta={'validate': True},
        )

        assert manager.call_count == 1
        wait_for_abort_resume = manager.calls[0].pop('wait_for_abort_resume')
        assert wait_for_abort_resume is _wait_for_abort_resume
        timeout_budget_provider = manager.calls[0].pop('timeout_budget_provider')
        assert isinstance(timeout_budget_provider, partial)
        assert timeout_budget_provider.func is capture_hytuner_execution_timeouts
        assert timeout_budget_provider.keywords == {'track_shared_pause': True}
        assert manager.calls == [
            {
                'request_id': 'request',
                'prompt_ids': [1, 2, 3],
                'sampling_params': {'max_tokens': 1, 'temperature': 0.2},
                'image_data': None,
                'video_data': None,
                'audio_data': None,
                'priority': 3,
                'request_meta': {'validate': True},
            }
        ]
        assert result.token_ids == [10]
        assert result.log_probs == [-0.1]
        assert result.stop_reason == 'stop'
        assert result.usage == {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4}
        assert result.extra_fields == {
            'resp_topk_logprob_indice': [[110]],
            'resp_topk_logprob_value': [[-0.5]],
        }

    async def test_maps_reported_inference_time(self):
        output = _output(10, 'stop')
        output.extra_fields['_inference_time'] = 1.25
        backend = HyTunerServerBackend(_ServerManager([output]))

        result = await backend.create_completion(
            'request',
            prompt_ids=[1],
            request_kwargs={'max_tokens': 1},
        )

        assert result.inference_time == pytest.approx(1.25)

    async def test_forwards_empty_request_meta_when_caller_omits_it(self):
        # No request_meta from the caller -> endpoint still receives a dict (never None).
        manager = _ServerManager([_output(10, 'stop')])
        backend = HyTunerServerBackend(manager)

        await backend.create_completion('request', prompt_ids=[1], request_kwargs={'max_tokens': 1})

        assert manager.calls[0]['request_meta'] == {}

    async def test_agent_metadata_scopes_concurrent_backend_request_ids(self) -> None:
        manager = _ServerManager([_output(10, 'stop')])
        backend = HyTunerServerBackend(manager)
        request_meta = {
            'agent_id': 'agent-2',
            'parent_agent_id': 'main',
            'declared_chain': {
                'chain_id': 'agent-2',
                'parent_chain_id': 'main',
                'kind': 'subagent',
                'spawn_order': 2,
            },
            'iteration': 3,
            'retry': 1,
        }

        await backend.create_completion(
            'trajectory-7',
            prompt_ids=[1],
            request_kwargs={'max_tokens': 1},
            request_meta=request_meta,
        )

        assert manager.calls[0]['request_id'] == 'trajectory-7:agent-2:3:1:completion'
        assert manager.calls[0]['request_meta'] == request_meta

    async def test_agent_request_id_ignores_unrelated_topology_metadata(self) -> None:
        manager = _ServerManager([_output(10, 'stop')])
        backend = HyTunerServerBackend(manager)
        request_meta = {
            'agent_id': 'agent-2',
            'parent_agent_id': 'unexpected-parent',
            'declared_chain': {
                'chain_id': 'unexpected-chain',
                'parent_chain_id': 'different-parent',
                'kind': 'subagent',
                'spawn_order': -1,
            },
            'iteration': 3,
            'retry': 1,
        }

        await backend.create_completion('trajectory-7', prompt_ids=[1], request_meta=request_meta)

        assert manager.calls[0]['request_id'] == 'trajectory-7:agent-2:3:1:completion'
        assert manager.calls[0]['request_meta'] == request_meta

    async def test_generic_declared_chain_keeps_root_backend_request_id(self) -> None:
        manager = _ServerManager([_output(10, 'stop')])
        backend = HyTunerServerBackend(manager)
        request_meta = {
            'agent_id': 'worker-1',
            'parent_agent_id': 'main',
            'declared_chain': {'chain_id': 'worker-1', 'kind': 'worker'},
        }

        await backend.create_completion('trajectory-7', prompt_ids=[1], request_meta=request_meta)

        assert manager.calls[0]['request_id'] == 'trajectory-7'
        assert manager.calls[0]['request_meta'] == request_meta

    async def test_unhashable_generic_declared_chain_kind_keeps_root_backend_request_id(self) -> None:
        manager = _ServerManager([_output(10, 'stop')])
        backend = HyTunerServerBackend(manager)
        request_meta = {'declared_chain': {'chain_id': 'A', 'kind': []}}

        await backend.create_completion('trajectory-7', prompt_ids=[1], request_meta=request_meta)

        assert manager.calls[0]['request_id'] == 'trajectory-7'
        assert manager.calls[0]['request_meta'] == request_meta

    async def test_generic_declared_chain_mapping_keeps_root_backend_request_id(self) -> None:
        manager = _ServerManager([_output(10, 'stop')])
        backend = HyTunerServerBackend(manager)
        declared_chain = MappingProxyType({'chain_id': 'A', 'kind': 'worker'})
        request_meta = {'declared_chain': declared_chain}

        await backend.create_completion('trajectory-7', prompt_ids=[1], request_meta=request_meta)

        assert manager.calls[0]['request_id'] == 'trajectory-7'
        assert manager.calls[0]['request_meta'] == request_meta

    async def test_agent_metadata_accepts_declared_chain_mapping(self) -> None:
        manager = _ServerManager([_output(10, 'stop')])
        backend = HyTunerServerBackend(manager)
        declared_chain = MappingProxyType({
            'chain_id': 'agent-2',
            'parent_chain_id': 'main',
            'kind': 'subagent',
            'spawn_order': 2,
        })
        request_meta = {
            'agent_id': 'agent-2',
            'parent_agent_id': 'main',
            'declared_chain': declared_chain,
            'iteration': 3,
            'retry': 1,
        }

        await backend.create_completion('trajectory-7', prompt_ids=[1], request_meta=request_meta)

        assert manager.calls[0]['request_id'] == 'trajectory-7:agent-2:3:1:completion'
        assert manager.calls[0]['request_meta'] == request_meta

    async def test_generic_trace_scalars_without_agent_identity_keep_root_request_id(self) -> None:
        manager = _ServerManager([_output(10, 'stop')])
        backend = HyTunerServerBackend(manager)
        request_meta = {'iteration': 3, 'retry': 1, 'invocation_kind': 'compaction_resume'}

        await backend.create_completion('trajectory-7', prompt_ids=[1], request_meta=request_meta)

        assert manager.calls[0]['request_id'] == 'trajectory-7'
        assert manager.calls[0]['request_meta'] == request_meta

    async def test_distinct_concurrent_agents_receive_distinct_backend_request_ids(self) -> None:
        class ConcurrentServerManager:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []
                self.both_started = asyncio.Event()

            async def generate_tokens(self, **kwargs: Any) -> Any:
                self.calls.append(kwargs)
                if len(self.calls) == 2:
                    self.both_started.set()
                await self.both_started.wait()
                return _output(10, 'stop')

        manager = ConcurrentServerManager()
        backend = HyTunerServerBackend(manager)

        await asyncio.gather(
            backend.create_completion(
                'trajectory-7',
                prompt_ids=[1],
                request_meta={
                    'agent_id': 'agent-1',
                    'parent_agent_id': 'main',
                    'declared_chain': {
                        'chain_id': 'agent-1',
                        'parent_chain_id': 'main',
                        'kind': 'subagent',
                        'spawn_order': 1,
                    },
                    'iteration': 1,
                    'retry': 0,
                },
            ),
            backend.create_completion(
                'trajectory-7',
                prompt_ids=[1],
                request_meta={
                    'agent_id': 'agent-2',
                    'parent_agent_id': 'main',
                    'declared_chain': {
                        'chain_id': 'agent-2',
                        'parent_chain_id': 'main',
                        'kind': 'subagent',
                        'spawn_order': 2,
                    },
                    'iteration': 1,
                    'retry': 0,
                },
            ),
        )

        assert {call['request_id'] for call in manager.calls} == {
            'trajectory-7:agent-1:1:0:completion',
            'trajectory-7:agent-2:1:0:completion',
        }

    async def test_delimiter_fields_have_unambiguous_backend_request_ids(self) -> None:
        manager = _ServerManager([_output(10, 'stop'), _output(11, 'stop')])
        backend = HyTunerServerBackend(manager)

        await asyncio.gather(
            backend.create_completion(
                'trajectory:7',
                prompt_ids=[1],
                request_meta={
                    'agent_id': 'agent-2',
                    'parent_agent_id': 'main',
                    'declared_chain': {
                        'chain_id': 'agent-2',
                        'parent_chain_id': 'main',
                        'kind': 'subagent',
                        'spawn_order': 2,
                    },
                    'iteration': 3,
                    'retry': 1,
                    'invocation_kind': 'completion',
                },
            ),
            backend.create_completion(
                'trajectory',
                prompt_ids=[1],
                request_meta={
                    'agent_id': '7:agent-2',
                    'parent_agent_id': 'main',
                    'declared_chain': {
                        'chain_id': '7:agent-2',
                        'parent_chain_id': 'main',
                        'kind': 'subagent',
                        'spawn_order': 2,
                    },
                    'iteration': 3,
                    'retry': 1,
                    'invocation_kind': 'completion',
                },
            ),
        )

        assert [call['request_id'] for call in manager.calls] == [
            'trajectory%3A7:agent-2:3:1:completion',
            'trajectory:7%3Aagent-2:3:1:completion',
        ]
        assert len({call['request_id'] for call in manager.calls}) == 2

    @pytest.mark.parametrize(('field', 'value'), [
        ('agent_id', 1),
        ('agent_id', ''),
        ('iteration', True),
        ('retry', False),
        ('invocation_kind', 1),
        ('invocation_kind', ''),
    ])
    async def test_physical_request_id_rejects_invalid_fields_before_dispatch(
        self,
        field: str,
        value: Any,
    ) -> None:
        manager = _ServerManager([])
        backend = HyTunerServerBackend(manager)
        request_meta = {
            'agent_id': 'agent-1',
            'declared_chain': {'kind': 'subagent'},
            'iteration': 1,
            'retry': 0,
            field: value,
        }

        with pytest.raises(ValueError, match='declared agent request metadata'):
            await backend.create_completion('trajectory-7', prompt_ids=[1], request_meta=request_meta)

        assert manager.calls == []

    @pytest.mark.parametrize('missing_field', ['agent_id', 'iteration', 'retry'])
    async def test_physical_request_id_rejects_missing_fields_before_dispatch(self, missing_field: str) -> None:
        manager = _ServerManager([])
        backend = HyTunerServerBackend(manager)
        request_meta = {
            'agent_id': 'agent-1',
            'declared_chain': {'kind': 'subagent'},
            'iteration': 1,
            'retry': 0,
        }
        request_meta.pop(missing_field)

        with pytest.raises(ValueError, match='declared agent request metadata'):
            await backend.create_completion('trajectory-7', prompt_ids=[1], request_meta=request_meta)

        assert manager.calls == []

    async def test_passes_aborted_result_through_without_retry(self):
        manager = _ServerManager([_output(10, 'aborted'), _output(11, 'stop')])
        backend = HyTunerServerBackend(manager)

        result = await backend.create_completion('request', prompt_ids=[1], request_kwargs={'max_tokens': 2})

        assert manager.call_count == 1
        assert result.token_ids == [10]
        assert result.stop_reason == 'aborted'

    async def test_completed_response_at_token_budget_is_output_length(self):
        output = SimpleNamespace(
            token_ids=[10, 11, 12],
            log_probs=None,
            routed_experts=None,
            stop_reason='completed',
            extra_fields={},
        )
        backend = HyTunerServerBackend(_ServerManager([output]))

        result = await backend.create_completion('request', prompt_ids=[1], request_kwargs={'max_tokens': 3})

        assert result.stop_reason == 'length'

    async def test_requires_prompt_ids(self):
        backend = HyTunerServerBackend(_ServerManager([]))
        with pytest.raises(AssertionError, match='prompt_ids'):
            await backend.create_completion('request', messages=[{'role': 'user', 'content': 'hi'}])


def test_backend_result_rejects_misaligned_token_facts():
    with pytest.raises(ValueError, match='log_probs length'):
        BackendResult(token_ids=[1, 2], log_probs=[-0.1])
    with pytest.raises(ValueError, match='top-k rows'):
        BackendResult(
            token_ids=[1, 2],
            extra_fields={
                'resp_topk_logprob_indice': [[3]],
                'resp_topk_logprob_value': [[-0.1]],
            },
        )


async def test_message_backend_forwards_the_resolved_request_unchanged():
    calls = []

    class Client:
        async def create_completion(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content='done', reasoning_content=None, provider_specific_fields=None)
            choice = SimpleNamespace(message=message, finish_reason='stop', logprobs=None)
            return SimpleNamespace(choices=[choice], usage=None)

    await OpenAIAPIBackend(Client()).create_completion(
        'request',
        messages=[{'role': 'user', 'content': 'question'}],
        request_kwargs={'tools': [], 'temperature': None, 'stream': False},
    )

    assert calls == [
        {
            'messages': [{'role': 'user', 'content': 'question'}],
            'tools': [],
            'temperature': None,
            'stream': False,
        }
    ]


async def test_message_backend_serializes_structured_tool_arguments_for_openai_wire():
    calls = []

    class Client:
        async def create_completion(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content='done', reasoning_content=None, provider_specific_fields=None)
            choice = SimpleNamespace(message=message, finish_reason='stop', logprobs=None)
            return SimpleNamespace(choices=[choice], usage=None)

    messages = [
        {'role': 'user', 'content': 'Call the echo tool.'},
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {
                    'id': 'call-1',
                    'type': 'function',
                    'function': {'name': 'echo', 'arguments': {'text': 'nexus'}},
                },
                {
                    'id': 'call-2',
                    'type': 'function',
                    'function': {'name': 'echo', 'arguments': '{"text":"already-wire-ready"}'},
                },
            ],
        },
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'nexus'},
    ]

    await OpenAIAPIBackend(Client()).create_completion('request', messages=messages)

    wire_tool_calls = calls[0]['messages'][1]['tool_calls']
    assert wire_tool_calls[0]['function']['arguments'] == '{"text": "nexus"}'
    assert wire_tool_calls[1]['function']['arguments'] == '{"text":"already-wire-ready"}'
    assert messages[1]['tool_calls'][0]['function']['arguments'] == {'text': 'nexus'}


class TestVllmCompletionsBackend:
    async def test_preserves_provider_wire_text(self, monkeypatch: Any) -> None:
        backend = VllmCompletionsBackend('http://unused', 'test-model')
        request_body: dict[str, Any] = {}

        async def fake_post(body: dict[str, Any], request_id: str) -> dict[str, Any]:
            del request_id
            request_body.update(body)
            return {
                'id': 'cmpl-1',
                'model': 'test-model',
                'choices': [
                    {
                        'text': 'visible<special>',
                        'finish_reason': 'stop',
                        'stop_reason': 'STOP',
                        'logprobs': {
                            'tokens': ['token_id:10', 'token_id:11'],
                            'token_logprobs': [-0.1, -0.2],
                        },
                    }
                ],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 2},
            }

        monkeypatch.setattr(backend, '_post_with_retry', fake_post)
        try:
            result = await backend.create_completion('request-1', prompt_ids=[1])
        finally:
            await backend.aclose()

        assert request_body['skip_special_tokens'] is False
        assert result.token_ids == [10, 11]
        assert result.log_probs == [-0.1, -0.2]
        assert result.wire_text == 'visible<special>'

    async def test_nonempty_text_requires_generated_token_ids(self, monkeypatch: Any) -> None:
        backend = VllmCompletionsBackend('http://unused', 'test-model')

        async def fake_post(_body: dict[str, Any], _request_id: str) -> dict[str, Any]:
            return {
                'choices': [{'text': 'visible', 'finish_reason': 'stop', 'logprobs': {'tokens': []}}],
            }

        monkeypatch.setattr(backend, '_post_with_retry', fake_post)
        try:
            with pytest.raises(RuntimeError, match='missing generated token IDs'):
                await backend.create_completion('request-1', prompt_ids=[1])
        finally:
            await backend.aclose()
