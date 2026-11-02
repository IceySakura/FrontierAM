from types import SimpleNamespace
from typing import Any

import pytest

from _fakes import memory_store
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

from nexus.core import agent_registry
from nexus.core.context import ExecutionContext
from nexus.core.types.message import ExtendedMessage
from nexus.extensions.agents.main.agent import MainAgentInput
from nexus.extensions.agents.main.multi_agent import MultiAgentConfig, SubagentConfig
from nexus.gateway import client
from nexus.gateway.session_manager import session_manager
from nexus.gateway.types import GatewayCompletionResult
from nexus.utils.cache import cache_context
from nexus.utils.timeout import async_timeout, credit_runner_timeout_for_response


@pytest.fixture(autouse=True)
def _clean_sessions():
    session_manager.end_all()
    yield
    session_manager.end_all()


class _MultiAgentBackendClient:
    def __init__(self) -> None:
        self.root_calls = 0

    async def create_completion(self, *, messages: list[dict[str, Any]], **_kwargs: Any) -> SimpleNamespace:
        is_child = messages[0].get('content') == 'worker'
        if is_child:
            message = ExtendedMessage(content='child result')
        else:
            self.root_calls += 1
            if self.root_calls == 1:
                tool_call = ChatCompletionMessageToolCall(
                    id='spawn-1',
                    type='function',
                    function=Function(name='spawn_agent', arguments='{"prompt": "check the calculation"}'),
                )
                message = ExtendedMessage(content='', tool_calls=[tool_call])
            elif self.root_calls == 2:
                tool_call = ChatCompletionMessageToolCall(
                    id='wait-1',
                    type='function',
                    function=Function(name='wait_agent', arguments='{"agent_ids": ["agent-1"]}'),
                )
                message = ExtendedMessage(content='', tool_calls=[tool_call])
            else:
                message = ExtendedMessage(content='root result')
        choice = SimpleNamespace(message=message, finish_reason='tool_calls' if message.tool_calls else 'stop')
        return SimpleNamespace(choices=[choice], usage=None, model='model-a')


def _multi_agent_gateway(trajectory_id: str) -> client.GatewayOpenAIClient:
    return client.GatewayOpenAIClient(
        client.OpenAIConfig(
            client_args={
                'openai_client': _MultiAgentBackendClient(),
                'trajectory_id': trajectory_id,
                'record_store': memory_store(trajectory_id),
                'scaffold_policy': 'nexus_main_agent',
                'metadata': {'trajectory_id': trajectory_id, 'validate': False},
            },
            request_args={'model': 'model-a', 'temperature': 0.0},
        )
    )


class TestAgentForking:
    def test_fork_reuses_processor_config_and_active_session(self) -> None:
        parent = _multi_agent_gateway('trajectory-fork')
        active_session = parent._session()

        child = parent.fork_for_agent(agent_id='agent-1', parent_agent_id='main')

        assert child is not parent
        assert child._processor is parent._processor
        assert child.config is parent.config
        assert child._metadata is parent._metadata
        assert child._request_args is parent._request_args
        assert child._session() is active_session
        assert parent._client_key == 'gateway'
        assert child._client_key == 'gateway:agent-1'

    def test_fork_rejects_blank_agent_id(self) -> None:
        parent = _multi_agent_gateway('trajectory-blank-agent')

        with pytest.raises(ValueError, match='agent_id'):
            parent.fork_for_agent(agent_id=' ', parent_agent_id='main')

    async def test_registered_main_agent_forks_gateway_and_preserves_declared_chains(self) -> None:
        parent = _multi_agent_gateway('trajectory-registered-main')
        agent_input = MainAgentInput(
            user_prompt='solve collaboratively',
            tools=['multi_agent/spawn_agent', 'multi_agent/wait_agent'],
            multi_agent=MultiAgentConfig(
                max_agents=1,
                max_active_agents=1,
                subagent=SubagentConfig(system_prompt='worker', tools=['base/finish']),
            ),
        )

        async with ExecutionContext(openai_clients={'main': parent}):
            output = await agent_registry.get('main_agent')(agent_input)

        trajectory = session_manager.end_session('trajectory-registered-main')
        assert output.final_answer == 'root result'
        assert output.multi_agent_summary is not None
        assert output.multi_agent_summary['completed_agents'] == 1
        assert output.multi_agent_summary['agents'][0]['final_answer'] == 'child result'
        assert list(trajectory.chains) == ['main', 'agent-1']
        assert trajectory.chains['main'].kind == 'main'
        assert trajectory.chains['agent-1'].kind == 'subagent'
        assert trajectory.chains['agent-1'].parent_chain_id == 'main'
        assert trajectory.chains['agent-1'].spawn.tool_use_id == 'spawn-1'
        assert trajectory.chains['agent-1'].join.tool_use_id == 'wait-1'
        assert trajectory.chains['agent-1'].join.kind == 'poll'
        assert 'spawn_order' not in trajectory.chains['agent-1'].evidence['attributes']
        assert parent._client_key == 'gateway'


class TestRequestKwargs:
    @staticmethod
    def _gateway(captured, *, token_level=False):
        class RecordingSession:
            async def create_completion(self, **kwargs):
                captured.append(kwargs)
                return GatewayCompletionResult(message={'role': 'assistant', 'content': 'done'})

        gateway = object.__new__(client.GatewayOpenAIClient)
        gateway.config = SimpleNamespace(request_args={}, enforced_request_args={})
        gateway._trajectory_id = 'trajectory-1'
        gateway._metadata = None
        gateway._request_args = {}
        gateway._priority = None
        gateway._processor = SimpleNamespace(is_token_level=token_level)
        gateway._model_registry = None
        gateway._agent_request_meta = {}
        gateway._session = lambda: RecordingSession()
        return gateway

    async def test_resolves_one_request_mapping_for_the_session(self):
        captured = []
        gateway = self._gateway(captured)

        await gateway._create_completion(
            [{'role': 'user', 'content': 'question'}],
            model='model-a',
            tools=None,
            tool_choice=None,
            response_format=None,
            parallel_tool_calls=False,
            provider_extension=None,
            temperature=0.2,
        )

        assert 'sampling_params' not in captured[0]
        assert captured[0]['request_kwargs'] == {
            'model': 'model-a',
            'tools': None,
            'tool_choice': None,
            'response_format': None,
            'parallel_tool_calls': False,
            'provider_extension': None,
            'temperature': 0.2,
        }

    async def test_preserves_absent_null_and_empty_tools(self):
        captured = []
        gateway = self._gateway(captured)
        messages = [{'role': 'user', 'content': 'question'}]

        await gateway._create_completion(messages)
        await gateway._create_completion(messages, tools=None)
        await gateway._create_completion(messages, tools=[])

        assert captured[0]['request_kwargs'] is None
        assert captured[1]['request_kwargs'] == {'tools': None}
        assert captured[2]['request_kwargs'] == {'tools': []}

    async def test_legacy_tool_retry_control_is_not_forwarded(self):
        captured = []
        gateway = self._gateway(captured)

        await gateway._create_completion(
            [{'role': 'user', 'content': 'question'}],
            allow_retry_for_tool_call=True,
        )

        assert 'allow_retry_for_tool_call' not in captured[0]
        assert captured[0]['request_kwargs'] is None

    async def test_client_metadata_merges_under_per_call_request_meta(self):
        # The agent passes a fresh per-call request_meta (iteration/retry/...) EVERY
        # turn, so the client-level metadata (e.g. HyTuner's validate flag) must merge
        # UNDER it, not be dropped. A plain "fall back only when None" loses validate
        # and the endpoint raises 'requires request_meta[validate]'. Per-call keys win.
        captured = []
        gateway = self._gateway(captured)
        gateway._metadata = {'trajectory_id': 'trajectory-1', 'validate': False}

        await gateway._create_completion(
            [{'role': 'user', 'content': 'q'}],
            request_meta={'iteration': 2, 'retry': 0, 'trajectory_id': 'per-call-wins'},
        )

        merged = captured[0]['request_meta']
        assert merged['validate'] is False  # client-level context preserved
        assert merged['iteration'] == 2  # per-call added
        assert merged['retry'] == 0
        assert merged['trajectory_id'] == 'per-call-wins'  # per-call overrides on conflict

    @pytest.mark.parametrize(
        ('request_kwargs', 'token_level', 'message'),
        [
            ({'stream': True}, False, 'streaming'),
            ({'n': 2}, False, 'one completion choice'),
            ({'n': 2}, True, 'one completion choice'),
        ],
    )
    def test_unsupported_completion_modes_fail_loudly(self, request_kwargs, token_level, message):
        gateway = object.__new__(client.GatewayOpenAIClient)
        gateway.config = SimpleNamespace(request_args={}, enforced_request_args={})
        gateway._request_args = {}
        gateway._processor = SimpleNamespace(is_token_level=token_level)

        with pytest.raises(NotImplementedError, match=message):
            gateway._resolve_request_kwargs(None, request_kwargs)

    def test_configured_defaults_are_part_of_the_recorded_request(self):
        backend_client = SimpleNamespace(create_completion=lambda **_kwargs: None)
        gateway = client.GatewayOpenAIClient(
            client.OpenAIConfig(
                client_args={
                    'openai_client': backend_client,
                    'trajectory_id': 'configured-defaults',
                    'record_store': memory_store('configured-defaults'),
                    'scaffold_policy': 'generic',
                },
                request_args={'sampling_params': {'temperature': 0.4, 'max_tokens': 32}},
            )
        )

        request_kwargs, uses_default = gateway._resolve_request_kwargs(None, {})

        assert request_kwargs == {'temperature': 0.4, 'max_tokens': 32}
        assert uses_default

    def test_top_level_configured_request_defaults_are_preserved(self):
        backend_client = SimpleNamespace(create_completion=lambda **_kwargs: None)
        gateway = client.GatewayOpenAIClient(
            client.OpenAIConfig(
                client_args={
                    'openai_client': backend_client,
                    'trajectory_id': 'top-level-defaults',
                    'record_store': memory_store('top-level-defaults'),
                    'scaffold_policy': 'generic',
                },
                request_args={
                    'model': 'configured-model',
                    'metadata': {'tenant': 'training'},
                    'tools': None,
                    'sampling_params': {'temperature': 0.4},
                },
            )
        )

        request_kwargs, uses_default = gateway._resolve_request_kwargs(None, {})

        assert request_kwargs == {
            'model': 'configured-model',
            'metadata': {'tenant': 'training'},
            'tools': None,
            'temperature': 0.4,
        }
        assert uses_default

    def test_provider_specific_fields_stay_in_the_resolved_request(self):
        gateway = object.__new__(client.GatewayOpenAIClient)
        gateway.config = SimpleNamespace(request_args={}, enforced_request_args={})
        gateway._request_args = {
            'temperature': 0.4,
            'metadata': {'tenant': 'training'},
            'timeout': 10,
        }
        gateway._processor = SimpleNamespace(is_token_level=False)

        request_kwargs, _ = gateway._resolve_request_kwargs(None, {})

        assert request_kwargs == {
            'temperature': 0.4,
            'metadata': {'tenant': 'training'},
            'timeout': 10,
        }

    def test_explicit_null_removes_a_sampling_default(self):
        gateway = object.__new__(client.GatewayOpenAIClient)
        gateway.config = SimpleNamespace(request_args={}, enforced_request_args={})
        gateway._request_args = {'temperature': 0.7}
        gateway._processor = SimpleNamespace(is_token_level=False)

        request_kwargs, _ = gateway._resolve_request_kwargs(None, {'temperature': None})

        assert request_kwargs == {'temperature': None}

    def test_request_override_does_not_use_the_default_max_tokens(self):
        gateway = object.__new__(client.GatewayOpenAIClient)
        gateway.config = SimpleNamespace(request_args={}, enforced_request_args={})
        gateway._request_args = {}
        gateway._processor = SimpleNamespace(is_token_level=False)

        request_kwargs, uses_default = gateway._resolve_request_kwargs(
            None,
            {'_request_args_override': {'sampling_params': {'max_tokens': 17}}},
        )

        assert request_kwargs == {'max_tokens': 17}
        assert not uses_default

    def test_request_override_replaces_configured_request_defaults(self):
        gateway = object.__new__(client.GatewayOpenAIClient)
        gateway.config = SimpleNamespace(request_args={}, enforced_request_args={})
        gateway._request_args = {
            'model': 'thinking-model',
            'thinking': {'type': 'enabled'},
            'temperature': 1.0,
        }
        gateway._processor = SimpleNamespace(is_token_level=False)

        request_kwargs, _ = gateway._resolve_request_kwargs(
            None,
            {
                '_request_args_override': {
                    'model': 'plain-model',
                    'temperature': 0.0,
                }
            },
        )

        assert request_kwargs == {'model': 'plain-model', 'temperature': 0.0}

    def test_configured_model_matches_the_existing_openai_client_precedence(self):
        gateway = object.__new__(client.GatewayOpenAIClient)
        gateway.config = SimpleNamespace(request_args={}, enforced_request_args={})
        gateway._request_args = {'model': 'configured'}
        gateway._processor = SimpleNamespace(is_token_level=False)

        request_kwargs, _ = gateway._resolve_request_kwargs('per-call', {})

        assert request_kwargs == {'model': 'configured'}

    def test_enforced_sampling_replaces_request_sampling_and_owns_the_token_budget(self):
        backend_client = SimpleNamespace(create_completion=lambda **_kwargs: None)
        gateway = client.GatewayOpenAIClient(
            client.OpenAIConfig(
                client_args={
                    'openai_client': backend_client,
                    'trajectory_id': 'enforced-request',
                    'record_store': memory_store('enforced-request'),
                    'scaffold_policy': 'generic',
                },
                request_args={
                    'model': 'configured-model',
                    'metadata': {'tenant': 'training'},
                    'sampling_params': {'temperature': 1.0, 'top_p': 0.9},
                },
                enforced_request_args={
                    'model': 'enforced-model',
                    'max_tokens': 64,
                    'sampling_params': {'temperature': 0.0},
                },
            )
        )

        request_kwargs, uses_default = gateway._resolve_request_kwargs(
            'per-call-model',
            {'sampling_params': {'temperature': 0.8, 'top_k': 20}, 'timeout': 10},
        )

        assert request_kwargs == {
            'model': 'enforced-model',
            'metadata': {'tenant': 'training'},
            'timeout': 10,
            'temperature': 0.0,
            'max_tokens': 64,
        }
        assert not uses_default

    @pytest.mark.parametrize(
        ('call_kwargs', 'enforced_request_args', 'expected_sampling', 'uses_default'),
        [
            ({}, None, {'temperature': 1.0, 'max_tokens': 100}, True),
            ({'temperature': 0.2, 'max_tokens': 8}, None, {'temperature': 1.0, 'max_tokens': 100}, True),
            ({'sampling_params': {'max_tokens': 16}}, None, {'temperature': 1.0, 'max_tokens': 16}, False),
            (
                {'_request_args_override': {'model': 'aux-model', 'max_tokens': 17}},
                None,
                {'model': 'aux-model', 'max_tokens': 17},
                False,
            ),
            (
                {},
                {'sampling_params': {'temperature': 0.0, 'max_tokens': 32}},
                {'temperature': 0.0, 'max_tokens': 32},
                False,
            ),
            # Top-level enforced fields are the documented shape of the enforcement
            # contract; they must reach the sampler instead of being swept away with the
            # scaffold's wire fields.
            ({}, {'temperature': 0.0}, {'temperature': 0.0, 'max_tokens': 100}, True),
            ({}, {'max_tokens': 32}, {'temperature': 1.0, 'max_tokens': 32}, False),
            # A nested enforced policy replaces the configured one; top-level enforced
            # fields layer under it.
            (
                {},
                {'temperature': 0.0, 'sampling_params': {'top_p': 0.5}},
                {'temperature': 0.0, 'top_p': 0.5},
                False,
            ),
            # An explicit null enforces "no sampling policy" rather than falling back to
            # the configured one.
            ({}, {'sampling_params': None}, None, False),
        ],
        ids=[
            'configured',
            'wire-fields',
            'per-call-policy',
            'host-override',
            'enforced-policy',
            'enforced-top-level-temperature',
            'enforced-top-level-max-tokens',
            'enforced-top-level-under-nested',
            'enforced-null-policy',
        ],
    )
    def test_token_sampling_policy_precedence(
        self, call_kwargs, enforced_request_args, expected_sampling, uses_default
    ):
        configured_sampling = {'temperature': 1.0, 'max_tokens': 100}
        gateway = self._gateway([], token_level=True)
        gateway.config.request_args = {'sampling_params': configured_sampling}
        if enforced_request_args is not None:
            gateway.config.enforced_request_args = enforced_request_args
        gateway._request_args = dict(configured_sampling)

        request_kwargs, actual_uses_default = gateway._resolve_request_kwargs(None, dict(call_kwargs))

        assert request_kwargs == expected_sampling
        assert actual_uses_default is uses_default

    def test_message_backend_spec_preserves_enforced_request_args(self, monkeypatch):
        captured = {}

        class BackendClient:
            def __init__(self, config):
                captured['config'] = config

        monkeypatch.setattr(client.openai_client_registry, 'get', lambda _name: BackendClient)
        enforced_request_args = {'model': 'enforced-model', 'temperature': 0.0}

        client.GatewayOpenAIClient._build_openai_client(
            {
                'name': 'recording',
                'config': {
                    'request_args': {'temperature': 1.0},
                    'enforced_request_args': enforced_request_args,
                },
            }
        )

        assert captured['config'].enforced_request_args == enforced_request_args

    async def test_request_meta_is_forwarded_unchanged(self):
        captured = []
        gateway = self._gateway(captured)
        request_meta = {
            'request_id': 'request-1',
            'declared_chain': {'chain_id': 'worker-1', 'kind': 'worker'},
        }

        await gateway._create_completion(
            [{'role': 'user', 'content': 'question'}],
            request_meta=request_meta,
        )

        assert captured[0]['request_meta'] is request_meta

    def test_observation_request_meta_is_forwarded_unchanged(self):
        captured = {}
        session = SimpleNamespace(record_observation=lambda *args, **kwargs: captured.update(kwargs))
        gateway = object.__new__(client.GatewayOpenAIClient)
        gateway._trajectory_id = 'trajectory-1'
        gateway._metadata = None
        gateway._agent_request_meta = {}
        gateway._session = lambda: session
        request_meta = {
            'request_id': 'request-1',
            'declared_chain': {'chain_id': 'worker-1', 'kind': 'worker'},
        }

        gateway.record_observation(content='result', request_meta=request_meta)

        assert captured['request_meta'] is request_meta

    async def test_public_client_records_the_request_sent_to_the_message_backend(self):
        calls = []

        class BackendClient:
            async def create_completion(self, **kwargs):
                calls.append(kwargs)
                message = SimpleNamespace(content='done', reasoning_content=None, provider_specific_fields=None)
                choice = SimpleNamespace(message=message, finish_reason='stop', logprobs=None)
                return SimpleNamespace(choices=[choice], usage=None, model='model-a')

        gateway = client.GatewayOpenAIClient(
            client.OpenAIConfig(
                client_args={
                    'openai_client': BackendClient(),
                    'trajectory_id': 'trajectory-1',
                    'record_store': memory_store('trajectory-1'),
                    'scaffold_policy': 'generic',
                },
                request_args={'model': 'model-a', 'provider_extension': {'mode': 'strict'}},
            )
        )
        messages = [{'role': 'user', 'content': 'question'}]

        await gateway.create_completion(messages, tools=[], temperature=0.2)
        trajectory = session_manager.end_session('trajectory-1')

        expected = {
            'model': 'model-a',
            'provider_extension': {'mode': 'strict'},
            'temperature': 0.2,
            'tools': [],
        }
        assert calls == [{'messages': messages, **expected}]
        llm = next(node for node in trajectory.iter_nodes() if node.generated_by == 'llm')
        assert llm.request_kwargs == expected

    def test_client_requires_trajectory_id(self):
        backend_client = SimpleNamespace(create_completion=lambda **_kwargs: None)
        with pytest.raises(ValueError, match='trajectory_id'):
            client.GatewayOpenAIClient(
                client.OpenAIConfig(client_args={'openai_client': backend_client}, request_args={})
            )

    def test_token_backend_requires_tokenizer(self):
        with pytest.raises(ValueError, match='tokenizer'):
            client.GatewayOpenAIClient(
                client.OpenAIConfig(
                    client_args={
                        'server_manager': SimpleNamespace(generate_tokens=lambda **_kwargs: None),
                        'trajectory_id': 'missing-tokenizer',
                        'record_store': memory_store('missing-tokenizer'),
                    },
                    request_args={},
                )
            )

    def test_tokenizer_is_read_only_and_initialization_uses_debug_log(self, caplog, capsys):
        tokenizer = object()
        backend_client = SimpleNamespace(create_completion=lambda **_kwargs: None)

        with caplog.at_level('DEBUG', logger='nexus.gateway.client'):
            gateway = client.GatewayOpenAIClient(
                client.OpenAIConfig(
                    client_args={
                        'openai_client': backend_client,
                        'tokenizer': tokenizer,
                        'trajectory_id': 'logged-client',
                        'record_store': memory_store('logged-client'),
                    },
                    request_args={},
                )
            )

        assert gateway.tokenizer is tokenizer
        with pytest.raises(AttributeError):
            gateway.tokenizer = object()
        assert 'GatewayOpenAIClient initialized trajectory_id=logged-client' in caplog.text
        assert capsys.readouterr().out == ''


class TestFinishReason:
    async def test_output_length_takes_precedence_over_tool_calls(self):
        class BackendClient:
            async def create_completion(self, **_kwargs):
                tool_call = ChatCompletionMessageToolCall(
                    id='call-1',
                    type='function',
                    function=Function(name='calculate', arguments='{"a": 1}'),
                )
                message = ExtendedMessage(role='assistant', content=None, tool_calls=[tool_call])
                choice = SimpleNamespace(message=message, finish_reason='length', logprobs=None)
                return SimpleNamespace(choices=[choice], usage=None, model='model-a')

        gateway = client.GatewayOpenAIClient(
            client.OpenAIConfig(
                client_args={
                    'openai_client': BackendClient(),
                    'trajectory_id': 'trajectory-length',
                    'record_store': memory_store('trajectory-length'),
                    'scaffold_policy': 'generic',
                },
                request_args={'model': 'model-a'},
            )
        )

        response = await gateway.create_completion([{'role': 'user', 'content': 'question'}])
        trajectory = session_manager.end_session('trajectory-length')

        assert response.choices[0].finish_reason == 'length'
        llm = next(node for node in trajectory.iter_nodes() if node.generated_by == 'llm')
        assert llm.provider_facts.finish_reason == 'length'


class TestCompletionCache:
    async def test_gateway_completion_is_never_replayed_without_trajectory_side_effects(self):
        calls = []
        gateway = object.__new__(client.GatewayOpenAIClient)
        gateway._client_key = 'gateway'

        async def create(messages, model=None, **kwargs):
            calls.append((messages, model, kwargs))
            return SimpleNamespace(choices=[])

        gateway._create_completion = create
        cache = {}
        with cache_context(cache):
            await gateway.create_completion([{'role': 'user', 'content': 'question'}])

        with cache_context(cache):
            await gateway.create_completion([{'role': 'user', 'content': 'question'}])

        assert len(calls) == 2
        assert cache['entries'] == []


class TestToolCallArgumentNormalization:
    def test_string_arguments_become_dicts_on_a_copy(self):
        messages = [
            {'role': 'user', 'content': 'q'},
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {'id': 'c1', 'type': 'function', 'function': {'name': 'calc', 'arguments': '{"a": 3, "b": 9}'}}
                ],
            },
            {'role': 'tool', 'tool_call_id': 'c1', 'content': '27'},
        ]
        normalized = client.GatewayOpenAIClient._messages_with_normalized_tool_calls(messages)
        # The copy holds dict arguments (what HY templates iterate over)...
        assert normalized[1]['tool_calls'][0]['function']['arguments'] == {'a': 3, 'b': 9}
        # ...while the caller's own message keeps the JSON string (Function-model-safe).
        assert messages[1]['tool_calls'][0]['function']['arguments'] == '{"a": 3, "b": 9}'

    def test_dict_arguments_and_plain_messages_untouched(self):
        tool_call = {'id': 'c1', 'type': 'function', 'function': {'name': 'calc', 'arguments': {'a': 1}}}
        messages = [{'role': 'assistant', 'content': '', 'tool_calls': [tool_call]}, {'role': 'user', 'content': 'x'}]
        normalized = client.GatewayOpenAIClient._messages_with_normalized_tool_calls(messages)
        assert normalized[0]['tool_calls'][0]['function']['arguments'] == {'a': 1}

    async def test_create_completion_normalizes_at_the_interface_chokepoint(self):
        # _create_completion is the single point every generation request reaches — including
        # callers that bypass the public create_completion override (e.g. bfclv4/swe_setup call
        # it directly). Normalizing here (non-mutating copy) means the session/processor downstream
        # always see dict arguments and the render path needs no second conversion. The caller's
        # own messages must stay untouched (JSON string, Function-model-safe).
        captured = []

        class RecordingSession:
            async def create_completion(self, **kwargs):
                captured.append(kwargs['messages'])
                return GatewayCompletionResult(message={'role': 'assistant', 'content': 'done'})

        gateway = object.__new__(client.GatewayOpenAIClient)
        gateway.config = SimpleNamespace(request_args={}, enforced_request_args={})
        gateway._trajectory_id = 'trajectory-1'
        gateway._metadata = None
        gateway._request_args = {}
        gateway._priority = None
        gateway._processor = SimpleNamespace(is_token_level=True)
        gateway._model_registry = None
        gateway._agent_request_meta = {}
        gateway._session = lambda: RecordingSession()

        messages = [
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [{'function': {'name': 'calc', 'arguments': '{"a": 3}'}}],
            }
        ]
        await gateway._create_completion(messages)

        # downstream session/processor got dict arguments (renderable, no second conversion needed)
        assert captured[0][0]['tool_calls'][0]['function']['arguments'] == {'a': 3}
        # caller's original message untouched (still a JSON string for the Function model)
        assert messages[0]['tool_calls'][0]['function']['arguments'] == '{"a": 3}'


class TestProviderSpecificFields:
    """Regression: token-level training arrays (R3 routed_experts, token_ids, log_probs, per-token
    topk in extra_fields) must NOT land in the assistant message's provider_specific_fields. That
    dict is re-inspected for prefix matching / message identity, and a raw numpy array there raises
    'The truth value of an array with more than one element is ambiguous', which failed every R3
    nexus trajectory on the gateway path. The token channel (session.token_training) carries these
    to the trainer instead — matching the legacy client, which never attaches them to the message."""

    def test_numpy_token_fields_are_kept_out_of_the_message(self):
        np = pytest.importorskip('numpy')
        gateway = object.__new__(client.GatewayOpenAIClient)
        result = GatewayCompletionResult(
            message={'role': 'assistant', 'content': 'ok'},
            token_ids=[1, 2, 3],
            log_probs=[-0.1, -0.2, -0.3],
            routed_experts=np.arange(6).reshape(3, 2),
            extra_fields={
                'resp_topk_logprob_indice': np.zeros((3, 2)),  # per-token topk, also numpy
                'reasoning_effort': 'no_think',  # a genuine decoration field — must survive
            },
        )

        completion = gateway._to_chat_completion(result, model='m')

        message = completion.choices[0].message
        psf = message.provider_specific_fields or {}
        # no numpy arrays / token-level tensors leaked into the message
        assert 'routed_experts' not in psf
        assert 'token_ids' not in psf
        assert 'log_probs' not in psf
        assert 'resp_topk_logprob_indice' not in psf
        # genuine decoration fields still pass through
        assert psf.get('reasoning_effort') == 'no_think'
        # and the message content is intact
        assert message.content == 'ok'

    def test_inference_time_is_exposed_on_completion(self):
        gateway = object.__new__(client.GatewayOpenAIClient)
        result = GatewayCompletionResult(
            message={'role': 'assistant', 'content': 'ok'},
            inference_time=1.25,
        )

        completion = gateway._to_chat_completion(result, model='m')

        assert completion.provider_specific_fields == {'inference_time': 1.25}

    @pytest.mark.parametrize(
        ('extra_fields', 'expected_credit'),
        [
            ({'_abort_wait_time': 1.5}, 1.5),
            ({}, 3.0),
        ],
    )
    async def test_runner_timeout_credits_abort_wait_once_from_backend_accounting(self, extra_fields, expected_credit):
        gateway = object.__new__(client.GatewayOpenAIClient)
        result = GatewayCompletionResult(
            message={
                'role': 'assistant',
                'content': 'ok',
                'provider_specific_fields': {'_abort_wait_time': 50.0, '_runner_timeout_credited': 50.0},
            },
            inference_time=2.0,
            extra_fields=extra_fields,
        )

        async with ExecutionContext() as context:
            async with async_timeout(10.0) as timeout_handle:
                context.timeout_handle = timeout_handle
                original_deadline = timeout_handle.when()
                assert original_deadline is not None

                completion = gateway._to_chat_completion(result, model='m')
                used = credit_runner_timeout_for_response(completion, wall_time=5.0)

                # Of 5 s wall time, 2 s is inference and any backend abort wait
                # already extended this deadline. Message decoration is not accounting.
                assert used == pytest.approx(2.0)
                updated_deadline = timeout_handle.when()
                assert updated_deadline is not None
                assert updated_deadline - original_deadline == pytest.approx(expected_credit)
