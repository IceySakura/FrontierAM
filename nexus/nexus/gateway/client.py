# /// script
# dependencies = []
#
# [project]
# name = "gateway"
#
# [optional-dependencies]
# runtime = []
#
# [tool.nexus]
# openai_clients = { gateway = "nexus.gateway.client" }
# span_processors = { gateway_trajectory = "nexus.gateway.trajectory_processor" }
# ///

"""GatewayOpenAIClient: entry point for the Gateway module.

Registered as 'gateway' in the Nexus openai_client_registry.
Provides OpenAI-compatible completion interface with lazy session management
and trajectory collection.
"""

import copy
import json
import logging
import time
import uuid

from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam

from nexus.core.openai_client import OpenAICompatibleClient, OpenAIConfig, openai_client_registry
from nexus.core.types import ExtendedChatCompletion, ExtendedUsage, TokenCountResult
from nexus.core.types.completion import ExtendedChoice
from nexus.core.types.message import ExtendedMessage
from nexus.gateway.backend import (
    VLLM_SAMPLING_ARGS,
    HyTunerServerBackend,
    InferenceBackend,
    OpenAIAPIBackend,
)
from nexus.gateway.model_registry import ModelBinding, ModelRegistry
from nexus.gateway.processor import Processor
from nexus.gateway.scaffold_policy import InvocationKind
from nexus.gateway.session import GatewaySession
from nexus.gateway.session_manager import session_manager
from nexus.gateway.trajectory_store import TrajectoryRecordStore, build_trajectory_record_store
from nexus.gateway.types import GatewayCompletionResult
from nexus.trace import observe
from nexus.utils.timeout import credit_runner_timeout_for_response


logger = logging.getLogger(__name__)


_MISSING = object()


_ONE_CHOICE_ONLY = 'Gateway supports exactly one completion choice (n=1)'


def _is_array_like(value: Any) -> bool:
    """True for numpy arrays / torch tensors (multi-element buffers that must not enter a
    message's provider_specific_fields — their truthiness is ambiguous and they aren't
    JSON-safe). Detected structurally (by ``ndim``) to avoid importing numpy/torch here."""
    return hasattr(value, 'ndim') and not isinstance(value, (bool, int, float, str, bytes))


@openai_client_registry.register('gateway')
class GatewayOpenAIClient(OpenAICompatibleClient):
    """Gateway OpenAI-compatible client — one instance per trajectory.

    Symmetric with the other registered OpenAI clients (notably the legacy
    ``HyTunerClient``): runners instantiate one from a ``{'name': 'gateway',
    'config': {...}}`` spec per run, bound to a single trajectory via
    ``client_args['trajectory_id']``. Session data lives in the process-wide
    ``nexus.gateway.session_manager.session_manager`` singleton — the client
    only renders/tokenizes (its Processor) and routes completions to its
    trajectory's session. A ``GatewayTrajectoryProcessor`` owns session
    finalization and exposes the canonical trajectory through the runner's
    standard processor result channel.

    Usage:
        client = GatewayOpenAIClient(OpenAIConfig(client_args={
            'trajectory_id': 'traj-001', 'tokenizer': ..., 'server_manager': ...,
            'record_store': store, 'scaffold_policy': 'nexus_main_agent',
        }))
        response = await client.create_completion(messages, tools=tools)
        # Configure a gateway_trajectory processor with this trajectory_id to
        # finalize the session and return its canonical trajectory.
    """

    supports_request_meta = True

    _IGNORED_COMPLETION_KWARGS = frozenset({'allow_retry_for_tool_call'})

    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        client_args = config.client_args
        request_args = config.request_args
        trajectory_id = client_args.get('trajectory_id')
        if not trajectory_id:
            raise ValueError("GatewayOpenAIClient requires client_args['trajectory_id']")
        record_store: TrajectoryRecordStore | None = client_args.get('record_store')
        if record_store is None:
            # Self-contained runs configure a backend instead of threading a store in;
            # HyTuner still injects its own so workers share one store across runs.
            store_config = client_args.get('trajectory_store') or {'backend': 'memory'}
            record_store = build_trajectory_record_store({
                'namespace': f'gateway-{trajectory_id}',
                **store_config,
            })

        self._model_registry: ModelRegistry | None = None
        self._agent_request_meta: dict[str, Any] = {}
        if 'models' in client_args:
            mixed_fields = {
                'server_manager',
                'openai_client',
                'tokenizer',
                'tool_parser',
                'template_parser',
                'apply_chat_template_kwargs',
                'mm_plugin',
                'priority',
                'enable_rollout_routing_replay',
                'max_model_len',
            }.intersection(client_args)
            if mixed_fields:
                raise ValueError(f'Configure model fields {sorted(mixed_fields)} inside models[], not alongside models')
            models = client_args['models']
            if not isinstance(models, list) or not models:
                raise ValueError('models must be a non-empty list')
            bindings = {}
            for model_config in models:
                if not isinstance(model_config, dict):
                    raise ValueError('Each model configuration must be a mapping')
                model_id = model_config.get('id')
                if not isinstance(model_id, str) or not model_id.strip():
                    raise ValueError('Each model configuration requires a non-empty string id')
                if model_id in bindings:
                    raise ValueError(f'Duplicate model id {model_id!r}')
                bindings[model_id] = ModelBinding(
                    model_id=model_id,
                    processor=self._build_processor(model_config),
                    request_args=model_config.get('request_args', {}),
                    max_model_len=model_config.get('max_model_len'),
                )
            self._model_registry = ModelRegistry(bindings, default_model_id=client_args['default_model_id'])
            self._bound_model_id: str = self._model_registry.default_model_id
            compaction_model_id = client_args.get('compaction_model_id')
            subagent_model_id = client_args.get('subagent_model_id')
            self._compaction_model_id = self._model_registry.get(
                self._bound_model_id if compaction_model_id is None else compaction_model_id
            ).model_id
            self._subagent_model_id = self._model_registry.get(
                self._bound_model_id if subagent_model_id is None else subagent_model_id
            ).model_id
            self._processor = self._model_registry.get(self._bound_model_id).processor
        else:
            self._processor = self._build_processor(client_args)
        self._tokenizer = self._processor._tokenizer

        # Client key for observability
        self._client_key = 'gateway'
        self._trajectory_id = str(trajectory_id)
        self._record_store = record_store
        self._metadata: dict[str, Any] | None = client_args.get('metadata')
        self._scaffold_policy = client_args.get('scaffold_policy')
        self._request_args = self._flatten_request_args(request_args)
        self._priority: int | None = client_args.get('priority')
        self._max_model_len: int | None = client_args.get('max_model_len')
        self._token_budget: int | None = client_args.get('token_budget')
        self._token_budget_include: list[str] | None = client_args.get('token_budget_include')
        logger.debug(
            'GatewayOpenAIClient initialized trajectory_id=%s backend=%s token_level=%s '
            'has_tool_parser=%s has_template_parser=%s default_request_keys=%s',
            self._trajectory_id,
            type(self._processor._backend).__name__,
            self._processor.is_token_level,
            client_args.get('tool_parser') is not None,
            client_args.get('template_parser') is not None,
            sorted(self._request_args),
        )

    # TODO(jerryhrzhou): Rename client agent_id/parent_agent_id to chain_id/parent_chain_id
    # updating the fork API and client-owned metadata together to distinguish chain identity from scaffold agents.
    def fork_for_agent(self, *, agent_id: str, parent_agent_id: str | None) -> 'GatewayOpenAIClient':
        """Create an agent-specific wrapper over this trajectory's shared session runtime."""
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError('agent_id must be a non-empty string')
        child = copy.copy(self)
        child._client_key = f'{self._client_key}:{agent_id}'
        if self._model_registry is not None:
            child._bound_model_id = (
                self._subagent_model_id if parent_agent_id is not None else self._model_registry.default_model_id
            )
            child._next_agent_iteration = 0
            child._agent_request_meta = {
                'agent_id': agent_id,
                'parent_agent_id': parent_agent_id,
                'declared_chain': {
                    'chain_id': agent_id,
                    'parent_chain_id': parent_agent_id,
                    'kind': 'subagent' if parent_agent_id is not None else 'main',
                },
            }
        return child

    @property
    def tokenizer(self) -> Any:
        if self._model_registry is not None:
            return self._model_registry.get(self._bound_model_id).processor._tokenizer
        return self._tokenizer

    def _build_processor(self, client_args: dict[str, Any]) -> Processor:
        backend = self._build_backend(client_args)
        tokenizer = client_args.get('tokenizer')
        if isinstance(backend, HyTunerServerBackend) and tokenizer is None:
            raise ValueError("GatewayOpenAIClient requires client_args['tokenizer'] with server_manager")
        return Processor(
            backend=backend,
            tokenizer=tokenizer,
            tool_parser=client_args.get('tool_parser'),
            template_parser=client_args.get('template_parser'),
            apply_chat_template_kwargs=client_args.get('apply_chat_template_kwargs'),
            mm_plugin=client_args.get('mm_plugin'),
            priority=client_args.get('priority', 0),
            enable_rollout_routing_replay=client_args.get('enable_rollout_routing_replay', False),
        )

    def _merge_request_meta(self, request_meta: dict[str, Any] | None) -> dict[str, Any] | None:
        if self._metadata or self._agent_request_meta:
            return {**(self._metadata or {}), **self._agent_request_meta, **(request_meta or {})}
        return request_meta

    def _build_backend(self, client_args: dict[str, Any]) -> InferenceBackend:
        """Build the appropriate backend based on what's provided in client_args."""
        server_manager = client_args.get('server_manager')
        openai_client = client_args.get('openai_client')

        if client_args.get('llm_generator') is not None:
            raise ValueError(
                'GatewayOpenAIClient no longer accepts llm_generator: both RPC and HTTP transports are reached '
                'through the HyTuner-side generation endpoint passed as server_manager (partial rollout is owned '
                'by GeneratorWrapper, not the gateway backend)'
            )
        if server_manager is not None:
            return HyTunerServerBackend(server_manager)
        if openai_client is not None:
            return OpenAIAPIBackend(self._build_openai_client(openai_client))
        raise ValueError('GatewayOpenAIClient requires one of: server_manager or openai_client in config.client_args')

    @staticmethod
    def _build_openai_client(openai_client: Any) -> OpenAICompatibleClient:
        """Build a message-level backend client from a prebuilt object or JSON-serializable spec."""
        if isinstance(openai_client, OpenAICompatibleClient):
            return openai_client

        if not isinstance(openai_client, dict):
            if callable(getattr(openai_client, 'create_completion', None)):
                return openai_client
            raise TypeError(
                'Gateway openai_client must be an OpenAICompatibleClient instance or a client spec dict, '
                f'got {type(openai_client).__name__}'
            )

        prebuilt = openai_client.get('instance')
        if prebuilt is not None:
            if callable(getattr(prebuilt, 'create_completion', None)):
                return prebuilt
            raise TypeError(
                f'Gateway openai_client.instance must provide create_completion, got {type(prebuilt).__name__}'
            )

        name = openai_client.get('name', 'openai')
        config_dict = openai_client.get('config', {})
        client_cls = openai_client_registry.get(name)
        return client_cls(
            OpenAIConfig(
                client_args=config_dict.get('client_args', {}),
                request_args=config_dict.get('request_args', {}),
                enforced_request_args=config_dict.get('enforced_request_args', {}),
            )
        )

    # ---- OpenAI-compatible interface ----

    def _resolve_trajectory_id(self, trajectory_id: str | None = None) -> str:
        if trajectory_id is not None and trajectory_id != self._trajectory_id:
            raise ValueError(
                f'GatewayOpenAIClient is bound to trajectory_id={self._trajectory_id!r}, got {trajectory_id!r}'
            )
        return self._trajectory_id

    def _session(self) -> GatewaySession:
        """This trajectory's session in the process-wide manager (lazy creation)."""
        metadata = dict(self._metadata or {})
        metadata['token_budget_config'] = {
            'max_model_len': self._max_model_len,
            'token_budget': self._token_budget,
            'token_budget_include': self._token_budget_include,
        }
        return session_manager.get_or_create(
            self._trajectory_id,
            processor=self._processor,
            record_store=self._record_store,
            metadata=metadata,
            scaffold_policy=self._scaffold_policy,
        )

    @staticmethod
    def _flatten_request_args(request_args: dict[str, Any] | None) -> dict[str, Any]:
        if not request_args:
            return {}
        flattened = dict(request_args)
        nested_sampling_params = flattened.pop('sampling_params', None)
        if isinstance(nested_sampling_params, dict):
            flattened.update(nested_sampling_params)
        return flattened

    @staticmethod
    def _as_sampling_dict(sampling_params: Any) -> dict[str, Any]:
        """Copy one ``sampling_params`` field; null means "no policy". Raises on other types."""
        if sampling_params is None:
            return {}
        if not isinstance(sampling_params, dict):
            raise TypeError('sampling_params must be a dict or null')
        return dict(sampling_params)

    def _resolve_request_kwargs(
        self,
        model: str | Any,
        kwargs: dict[str, Any],
        tools: Any = _MISSING,
        *,
        binding: ModelBinding | None = None,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Resolve one request according to the backend contract.

        ``_request_args_override`` replaces the configured request. Token-level sampling is
        resolved from the configured ``sampling_params``, then per-call, then enforced;
        message-level backends additionally honor top-level OpenAI sampling kwargs.
        """
        token_level = (binding.processor if binding is not None else self._processor).is_token_level
        request_args_override = kwargs.pop('_request_args_override', None)
        per_call_sampling = kwargs.pop('sampling_params', None)
        enforced_request_args = self.config.enforced_request_args
        if request_args_override is None:
            source_request_args = self.config.request_args
            request_kwargs = dict(self._request_args)
            if binding is not None:
                source_request_args = {**source_request_args, **binding.request_args}
                request_kwargs.update(self._flatten_request_args(dict(binding.request_args)))
        else:
            # An override replaces the configured request, so it is the sampling source.
            source_request_args = request_args_override
            request_kwargs = self._flatten_request_args(request_args_override)

        # A host override declares a budget in either shape it accepts.
        override_sampling = (request_args_override or {}).get('sampling_params')
        override_declares_budget = request_args_override is not None and (
            'max_tokens' in request_args_override
            or (isinstance(override_sampling, dict) and 'max_tokens' in override_sampling)
        )
        # A configured ``max_tokens`` is a segment default the session accounts cumulatively; an
        # explicit source bounds only its own request, and an enforced policy replaces the
        # configured one outright. The session caps both by ``max_model_len`` and ``token_budget``.
        uses_default_max_tokens = not (
            (isinstance(per_call_sampling, dict) and 'max_tokens' in per_call_sampling)
            or override_declares_budget
            or 'sampling_params' in enforced_request_args
            or 'max_tokens' in enforced_request_args
            or (not token_level and 'max_tokens' in kwargs)
        )

        if token_level:
            # HyTuner's sampling policy owns token-level generation: top-level wire kwargs
            # belong to the scaffold's request and never reach the sampler.
            requested_sampling = self._as_sampling_dict(source_request_args.get('sampling_params'))
            if request_args_override is not None:
                # Host overrides use the flat shape legacy auxiliary requests accept.
                requested_sampling.update({
                    key: value
                    for key, value in request_args_override.items()
                    if key in VLLM_SAMPLING_ARGS and value is not None
                })
            per_call = self._as_sampling_dict(per_call_sampling)
            configured_max_tokens = requested_sampling.get('max_tokens')
            if configured_max_tokens is not None and per_call.get('max_tokens') is not None:
                # Per-call limits may tighten, never widen, the configured generation cap.
                per_call['max_tokens'] = min(int(per_call['max_tokens']), int(configured_max_tokens))
            requested_sampling.update(per_call)
            request_kwargs.update(requested_sampling)
        elif isinstance(per_call_sampling, dict):
            request_kwargs.update(per_call_sampling)

        for key, value in kwargs.items():
            if key not in self._IGNORED_COMPLETION_KWARGS:
                request_kwargs[key] = value

        if model is not None:
            request_kwargs.setdefault('model', model)
        if tools is not _MISSING:
            request_kwargs['tools'] = tools

        if token_level:
            request_kwargs = self._apply_enforced_request_args(request_kwargs)
            if request_kwargs.get('n') not in (None, 1):
                # The sweep below drops wire sampling fields; do not discard an n>1 silently.
                raise NotImplementedError(_ONE_CHOICE_ONLY)

            # Enforced sampling has the final say: a nested ``sampling_params`` replaces the
            # requested policy (null clears it), top-level enforced fields layer under it.
            enforced_replaces_policy = 'sampling_params' in request_kwargs
            enforced_nested_sampling = self._as_sampling_dict(request_kwargs.pop('sampling_params', None))
            effective_sampling = {} if enforced_replaces_policy else dict(requested_sampling)
            effective_sampling.update({
                key: value for key, value in enforced_request_args.items() if key in VLLM_SAMPLING_ARGS
            })
            effective_sampling.update(enforced_nested_sampling)

            # Drop the scaffold's wire sampling fields, then apply the policy through the
            # token backend's flat contract — enforced fields must survive this sweep.
            for key in VLLM_SAMPLING_ARGS:
                request_kwargs.pop(key, None)
            request_kwargs.update(effective_sampling)
        else:
            if 'sampling_params' in enforced_request_args:
                for key in VLLM_SAMPLING_ARGS:
                    request_kwargs.pop(key, None)
            request_kwargs = self._flatten_request_args(self._apply_enforced_request_args(request_kwargs))

        if request_kwargs.get('stream') not in (None, False):
            raise NotImplementedError('Gateway does not support streaming completions')
        if request_kwargs.get('n') not in (None, 1):
            raise NotImplementedError(_ONE_CHOICE_ONLY)

        return request_kwargs or None, uses_default_max_tokens

    @observe('openai_completion', self_attrs=['_client_key'])
    async def create_completion(
        self,
        messages: list[ChatCompletionMessageParam | dict],
        model: str | Any = None,
        request_meta: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        """Run live so every returned completion has a matching trajectory side effect.

        This override exists only to drop the base class's @cacheable: unlike a pure
        LLM client, every gateway completion mutates live trajectory state that is not
        persisted with the cache. A replay in a fresh process would return the cached
        result without recording the turn, corrupting training data.

        TODO(gateway-resume): the gateway does NOT support cache-based resume. This is a
        BIG open problem, not a nicety: a trajectory that crashes mid-rollout cannot be
        replayed from cache (the trajectory isn't restored, so we deliberately run live and
        re-generate everything). Real fixes require either (a) persisting/restoring the
        trajectory alongside the cache, or (b) moving the recording side effect outside the
        cacheable boundary so replay still rebuilds it. Until then, gateway rollouts
        are not crash-resumable. Must be solved before gateway can rely on resume.
        """
        # Tool-call argument normalization (str -> dict) is NOT done here: it happens once at
        # the sole interface chokepoint _create_completion (non-mutating copy), which every
        # generation request reaches — including callers that bypass this public override.
        if request_meta is not None:
            kwargs['request_meta'] = request_meta
        started = time.perf_counter()
        response = await self._create_completion(messages, model, **kwargs)
        credit_runner_timeout_for_response(response, time.perf_counter() - started, started_at=started)
        return response

    async def _create_completion(
        self,
        messages: list[ChatCompletionMessageParam | dict],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        """Implement OpenAICompatibleClient interface. Routes to session via trajectory_id."""
        self._resolve_trajectory_id(kwargs.pop('trajectory_id', None))
        per_call_meta = kwargs.pop('request_meta', None)
        trace_meta = kwargs.pop('trace_meta', None)
        per_call_meta = per_call_meta if per_call_meta is not None else trace_meta
        # Merge the client's construction-time metadata (integration-owned context set
        # once — e.g. HyTuner's validate flag) UNDER the per-call request_meta the agent
        # passes each turn (iteration/retry/invocation_kind). Per-call keys win, but the
        # client-level context is never dropped — the agent always supplies a
        # request_meta, so a plain "fall back only when None" would lose validate.
        # When the client has no metadata, forward the caller's request_meta unchanged
        # (identity preserved) so callers that own the full trace context are untouched.
        request_meta: dict[str, Any] | None = self._merge_request_meta(per_call_meta)
        if self._agent_request_meta:
            assert request_meta is not None, 'Fork identity must be present after merging request metadata'
            # NOTE(luxyhliu): A direct fork owns call ordinals when no harness declares them.
            if 'iteration' not in request_meta and 'retry' not in request_meta:
                request_meta = {**request_meta, 'iteration': self._next_agent_iteration, 'retry': 0}
            iteration = request_meta.get('iteration')
            if type(iteration) is int and iteration >= 0:
                self._next_agent_iteration = max(self._next_agent_iteration, iteration + 1)
        wire_tools = kwargs.pop('tools', _MISSING)
        priority = kwargs.pop('priority', self._priority)
        # Normalize tool-call arguments (str -> dict) HERE, at the single interface chokepoint
        # every generation request passes through — including callers that reach _create_completion
        # directly (e.g. bfclv4/swe_setup tasks) and bypass the public create_completion override.
        # Non-mutating and idempotent: valid JSON strings are parsed, while malformed strings
        # are preserved so an agent can send error feedback and retry without losing history.
        message_dicts = self._messages_with_normalized_tool_calls(cast(list[dict], list(messages)))
        session = self._session()
        context = None
        binding = None
        if self._model_registry is not None:
            context = session.segment_assembler.extract_context(
                message_dicts, tools=None if wire_tools is _MISSING else wire_tools, request_meta=request_meta
            )
            if context.invocation_kind == InvocationKind.SUMMARY_INFERENCE:
                model_id = self._compaction_model_id
            else:
                model_id = self._bound_model_id
            binding = self._model_registry.get(model_id)
        request_kwargs, uses_default_max_tokens = self._resolve_request_kwargs(
            model, kwargs, wire_tools, binding=binding
        )
        result = await session.create_completion(
            messages=message_dicts,
            request_kwargs=request_kwargs,
            priority=priority,
            uses_default_max_tokens=uses_default_max_tokens,
            request_meta=request_meta,
            binding=binding,
            context=context,
        )
        return self._to_chat_completion(result, model=model)

    async def _count_tokens(
        self,
        *,
        messages: list[ChatCompletionMessageParam | dict],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
    ) -> TokenCountResult:
        """Count the bound model's prompt: frozen sampled stream + rendered tail.

        A fresh canonical render is the wrong ruler for budget decisions: whenever a
        recorded emission violates ``render(parse(x)) == x`` (e.g. an invalid
        ``<tool_call>`` block the parser drops) the re-render undercounts the stream
        the next request will actually resend. Read-only mirror of the request path.
        """
        request_meta = self._merge_request_meta(None)
        binding = self._model_registry.get(self._bound_model_id) if self._model_registry is not None else None
        processor = binding.processor if binding is not None else self._processor
        if not processor.is_token_level:
            raise NotImplementedError('Token counting requires a token-level processor (tokenizer must be provided)')
        message_dicts = self._messages_with_normalized_tool_calls(cast(list[dict], list(messages)))
        tool_dicts = cast(list[dict] | None, tools)
        session = session_manager.get_session(self._trajectory_id)
        frozen_prefix = (
            session.segment_assembler.frozen_prefix(
                message_dicts,
                tools=tool_dicts,
                request_meta=request_meta,
                model_id=binding.model_id if binding is not None else None,
            )
            if session is not None
            else None
        )
        _, prompt_ids, _, _ = await processor.render_and_tokenize(
            message_dicts,
            tool_dicts,
            add_generation_prompt=add_generation_prompt,
            frozen_prefix=frozen_prefix,
        )
        return TokenCountResult(input_tokens=len(prompt_ids))

    @staticmethod
    def _messages_with_normalized_tool_calls(messages: list[dict]) -> list[dict]:
        # FOLLOWS HyTunerClient._messages_with_normalized_tool_calls (hytuner.py:152) —
        # the count path is read-only, so normalize on copies instead of in place.
        def _norm_tool_call(tool: dict) -> dict:
            args = tool['function']['arguments']
            if not isinstance(args, str):
                return tool
            try:
                normalized_args = json.loads(args)
            except json.JSONDecodeError:
                return tool
            return {**tool, 'function': {**tool['function'], 'arguments': normalized_args}}

        return [
            {**m, 'tool_calls': [_norm_tool_call(tc) for tc in m['tool_calls']]} if m.get('tool_calls') else m
            for m in messages
        ]

    # ---- Gateway-specific API (per-trajectory; lifecycle lives on the
    # session_manager singleton — end_session/has_session/leak accounting are
    # runner concerns, not client methods) ----

    async def update_token_consumption(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        trajectory_id: str | None = None,
    ) -> None:
        """Compaction accounting hook (legacy HyTunerClient parity, async per !920)."""
        self._resolve_trajectory_id(trajectory_id)
        if self.tokenizer is None:
            return
        # Non-mutating normalization, same as the count path (legacy parity).
        normalized = self._messages_with_normalized_tool_calls(cast(list[dict], list(messages)))
        await self._session().update_token_consumption(
            normalized,
            tools,
            request_meta=self._merge_request_meta(None),
            binding=self._model_registry.get(self._bound_model_id) if self._model_registry is not None else None,
        )

    @property
    def max_model_len(self) -> int | None:
        """Return the sequence limit of this client's bound model."""
        if self._model_registry is not None:
            return self._model_registry.get(self._bound_model_id).max_model_len
        return self._max_model_len

    @property
    def curr_segment_consumed_tokens(self) -> int | None:
        """Segment-local consumed-token counter (legacy HyTunerClient parity)."""
        if self.tokenizer is None:
            return None
        return self._session().segment_consumed_tokens(
            request_meta=self._merge_request_meta(None),
            binding=self._model_registry.get(self._bound_model_id) if self._model_registry is not None else None,
        )

    @curr_segment_consumed_tokens.setter
    def curr_segment_consumed_tokens(self, value: int) -> None:
        if self.tokenizer is None:
            return
        self._session().set_segment_consumed_tokens(
            value,
            request_meta=self._merge_request_meta(None),
            binding=self._model_registry.get(self._bound_model_id) if self._model_registry is not None else None,
        )

    def record_observation(
        self,
        trajectory_id: str | None = None,
        content: str | None = None,
        token_ids: list[int] | None = None,
        role: str = 'tool',
        parent_messages: list[dict] | None = None,
        request_meta: dict[str, Any] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        status: str | None = None,
        message: dict[str, Any] | None = None,
    ) -> None:
        """Notify gateway about an observation (tool/env output) that follows ``parent_messages``."""
        if content is None and message is None:
            content = trajectory_id
            trajectory_id = None
        if content is None and message is None:
            raise ValueError('record_observation requires content or message')
        self._resolve_trajectory_id(trajectory_id)
        session = self._session()
        session.record_observation(
            content,
            token_ids,
            role,
            message=message,
            parent_messages=parent_messages,
            request_meta=self._merge_request_meta(request_meta) if self._agent_request_meta else request_meta,
            start_time=start_time,
            end_time=end_time,
            status=status,
        )

    def finalize_segment(self, trajectory_id: str | None = None) -> None:
        """Mark a compaction boundary for this trajectory."""
        self._resolve_trajectory_id(trajectory_id)
        self._session().finalize_segment()

    # ---- Helpers ----

    def _to_chat_completion(self, result: GatewayCompletionResult, model: str | None = None) -> ExtendedChatCompletion:
        """Convert GatewayCompletionResult to ExtendedChatCompletion."""
        # Build tool_calls for ExtendedMessage if present
        tool_calls = None
        raw_tool_calls = result.message.get('tool_calls')
        if raw_tool_calls:
            from openai.types.chat.chat_completion_message_tool_call import (
                ChatCompletionMessageToolCall,
                Function,
            )

            tool_calls = []
            for tc in raw_tool_calls:
                if isinstance(tc, dict):
                    tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id=tc.get('id', str(uuid.uuid4())[:8]),
                            type='function',
                            function=Function(
                                name=tc.get('function', {}).get('name', ''),
                                arguments=tc.get('function', {}).get('arguments', ''),
                            ),
                        )
                    )
                else:
                    tool_calls.append(tc)

        # Token-level training data (token_ids / log_probs / routed_experts, plus per-token
        # topk arrays inside extra_fields) travels the token channel to the trainer
        # (session.token_training), NOT the assistant message. It must NOT ride in
        # provider_specific_fields: this dict becomes part of the message that the gateway
        # re-inspects for prefix matching / equality (MessageIdentity), and a raw numpy array
        # there raises "truth value of an array is ambiguous". The legacy HyTunerClient never
        # attaches these to the message (they live only on token_output). Keep only genuine,
        # message-relevant decoration fields; drop non-JSON-safe values (numpy arrays/tensors).
        provider_specific_fields = {**result.extra_fields}
        message_provider_fields = result.message.get('provider_specific_fields')
        if isinstance(message_provider_fields, dict):
            provider_specific_fields.update(message_provider_fields)
        provider_specific_fields = {
            key: value
            for key, value in provider_specific_fields.items()
            if value is not None and not _is_array_like(value)
        }

        raw_reasoning_content = result.message.get('reasoning_content')
        if raw_reasoning_content is None:
            raw_reasoning_content = provider_specific_fields.get('reasoning_content')
        reasoning_content = raw_reasoning_content if isinstance(raw_reasoning_content, str) else None

        message = ExtendedMessage(
            content=result.message.get('content'),
            role=result.message.get('role', 'assistant'),
            refusal=result.message.get('refusal'),
            reasoning_content=reasoning_content,
            tool_calls=tool_calls or None,
            raw_tool=result.message.get('raw_tool'),
            function_call=result.message.get('function_call'),
            annotations=result.message.get('annotations'),
            name=result.message.get('name'),
            thinking_blocks=result.message.get('thinking_blocks'),
            provider_specific_fields=provider_specific_fields or None,
            raw_response=result.message.get('raw_response'),
        )

        if result.stop_reason == 'length':
            finish_reason = 'length'
        elif tool_calls:
            finish_reason = 'tool_calls'
        else:
            finish_reason = result.stop_reason or 'stop'
        usage = self._normalize_usage(result.usage)

        completion_provider_fields = None
        if result.inference_time is not None:
            completion_provider_fields = {'inference_time': result.inference_time}
            if '_abort_wait_time' in result.extra_fields:
                # Abort waits already extended the runner deadline while generation was parked.
                completion_provider_fields['_runner_timeout_credited'] = result.extra_fields['_abort_wait_time']

        return ExtendedChatCompletion(
            id=result.id or f'gateway-{uuid.uuid4().hex[:12]}',
            choices=[
                ExtendedChoice(
                    finish_reason=finish_reason,
                    index=0,
                    message=message,
                    logprobs=result.choice_logprobs,
                    provider_specific_fields=result.choice_provider_specific_fields,
                )
            ],
            created=result.created or int(time.time()),
            model=result.model or model,
            object=result.object or 'chat.completion',
            system_fingerprint=result.system_fingerprint,
            usage=usage,
            service_tier=result.service_tier,
            provider_specific_fields=completion_provider_fields,
        )

    @staticmethod
    def _normalize_usage(usage: dict[str, Any] | ExtendedUsage | None) -> ExtendedUsage | None:
        """Normalize provider usage into ExtendedUsage without dropping common aliases."""
        if usage is None:
            return None
        if isinstance(usage, ExtendedUsage):
            return usage
        usage_dict = dict(usage)
        if 'prompt_tokens' not in usage_dict and 'input_tokens' in usage_dict:
            usage_dict['prompt_tokens'] = usage_dict['input_tokens']
        if 'completion_tokens' not in usage_dict and 'output_tokens' in usage_dict:
            usage_dict['completion_tokens'] = usage_dict['output_tokens']
        if 'total_tokens' not in usage_dict:
            usage_dict['total_tokens'] = (usage_dict.get('prompt_tokens') or 0) + (
                usage_dict.get('completion_tokens') or 0
            )
        return ExtendedUsage.model_validate(usage_dict)
