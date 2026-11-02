import logging
import time

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from openai.types.chat import ChatCompletionMessageParam

from nexus.core.registry import Registry
from nexus.core.types import ExtendedChatCompletion, TokenCountResult
from nexus.trace import get_current_span_id, observe, update_span
from nexus.utils.cache import cacheable
from nexus.utils.cache_task_id import get_cache_task_id, get_input_cache_task_id
from nexus.utils.provider_io import redact_provider_io
from nexus.utils.timeout import credit_runner_timeout_for_response


logger = logging.getLogger(__name__)

CONTEXT_LENGTH_EXCEEDED_CODE = 'context_length_exceeded'
"""Machine-readable code for a call refused on token budget.

An agent whose loop runs inside the sandbox cannot catch ContextLengthExceededError: the
error crosses the Socket.IO boundary and reaches it as an opaque HTTP 500. This code is
what travels that path instead, so the run can still be told apart from a real crash.
"""

UNPARSED_TOOL_CALL_FIELD = 'unparsed_tool_call'
"""Choice ``provider_specific_fields`` key set on a completion that began a tool call the parser
could not extract.

The completion still reaches the agent as plain text without ``tool_calls``, which a sandboxed
agent such as Codex takes as its final answer and exits cleanly on. The host reads this key to
tell that exit apart from a real answer.
"""


CONTENT_POLICY_VIOLATION_CODE = 'content_policy_violation'
"""Machine-readable code for a provider content-policy refusal."""


class ContentPolicyViolationError(Exception):
    """A provider refusal that must not be retried as a transport failure."""

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class ContextLengthExceededError(Exception):
    """Raised when the input prompt exceeds the maximum context length.

    This is a unified exception that wraps provider-specific context length errors
    from OpenAI, Azure OpenAI, LiteLLM, and other providers.
    """

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class TokenizerPanicError(RuntimeError):
    """A Rust-backed tokenizer panic converted into a normal Python exception."""


def as_tokenizer_panic_error(error: BaseException, *, operation: str) -> TokenizerPanicError | None:
    """Convert only PyO3 tokenizer panics, leaving cancellation and process signals untouched."""
    if type(error).__module__ != 'pyo3_runtime' or type(error).__name__ != 'PanicException':
        return None
    return TokenizerPanicError(f'Failed to tokenize {operation}: {error}')


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI-compatible clients."""

    client_args: dict[str, Any]  # args used to init client
    request_args: dict[str, Any]  # args used to call completion
    enforced_request_args: dict[str, Any] = field(default_factory=dict)


def apply_enforced_request_args(
    request_args: Mapping[str, Any],
    enforced_request_args: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply a client policy without mutating either input mapping.

    Typed enforced kwargs replace the same top-level kwargs and remove stale
    values with the same name from ``extra_body`` so the OpenAI SDK cannot
    override the policy during its final body merge. Provider-specific fields
    remain supported through ``enforced_request_args.extra_body``.
    """
    result = dict(request_args)
    if not enforced_request_args:
        return result

    enforced = dict(enforced_request_args)
    has_enforced_extra_body = 'extra_body' in enforced
    enforced_extra_body = enforced.pop('extra_body', None)
    has_existing_extra_body = 'extra_body' in result
    existing_extra_body = result.get('extra_body')

    if isinstance(existing_extra_body, Mapping):
        copied_extra_body = dict(existing_extra_body)
        for key in enforced:
            copied_extra_body.pop(key, None)
        result['extra_body'] = copied_extra_body

    result.update(enforced)

    if not has_enforced_extra_body:
        return result
    if enforced_extra_body is None:
        result['extra_body'] = None
        return result
    if not isinstance(enforced_extra_body, Mapping):
        raise TypeError('enforced_request_args.extra_body must be a mapping or null')
    if not has_existing_extra_body or existing_extra_body is None:
        merged_extra_body: dict[str, Any] = {}
    elif isinstance(existing_extra_body, Mapping):
        merged_extra_body = dict(result['extra_body'])
    else:
        raise TypeError('request_args.extra_body must be a mapping or null when an enforced extra_body is configured')
    merged_extra_body.update(enforced_extra_body)
    result['extra_body'] = merged_extra_body
    return result


def resolve_effective_request_args(
    request_args: Mapping[str, Any],
    enforced_request_args: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return values visible after the SDK's shallow ``extra_body`` merge."""
    effective = apply_enforced_request_args(request_args, enforced_request_args)
    extra_body = effective.pop('extra_body', None)
    if isinstance(extra_body, Mapping):
        effective.update(extra_body)
    return effective


def _convert_agent_message_item(item: Mapping[str, Any]) -> dict[str, Any] | None:
    """Convert the inter-agent envelope emitted by Codex 0.146.1.

    Codex splits a NEW_TASK into a plaintext ``input_text`` envelope and an
    ``encrypted_content`` payload. Custom providers receive the latter as the original task text,
    so preserve the ordered content parts and concatenate them without inserting separators.
    """
    content = item.get('content')
    if isinstance(content, str):
        return {'role': 'user', 'content': content}
    if not isinstance(content, list):
        return None

    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get('type')
        if part_type in {'input_text', 'output_text', 'text'}:
            text_parts.append(str(part.get('text', '')))
        elif part_type == 'encrypted_content':
            text_parts.append(str(part.get('encrypted_content', '')))
        else:
            logger.warning('dropping unsupported agent_message content type=%s', part_type)

    return {'role': 'user', 'content': ''.join(text_parts)}


def _messages_for_provider(
    messages: list[ChatCompletionMessageParam | dict], *, native_agent_message: bool
) -> list[ChatCompletionMessageParam | dict]:
    if native_agent_message:
        return messages
    return [
        (_convert_agent_message_item(message) or message)
        if isinstance(message, dict) and message.get('type') == 'agent_message'
        else message
        for message in messages
    ]


class OpenAICompatibleClient(ABC):
    """Base class for clients with OpenAI-compatible completion interface.

    Implementations must accept an OpenAIConfig parameter in __init__.
    Returns ExtendedChatCompletion which includes provider-specific fields
    for multi-provider compatibility.
    """

    # Whether _create_completion accepts the request_meta kwarg (client identity
    # metadata). Swallowed here otherwise so provider SDKs never see it.
    supports_request_meta: bool = False
    _client_key: str

    # Whether the client can forward the opaque state returned by the provider's
    # native ``/v1/responses/compact`` endpoint.  This is deliberately separate
    # from local summary compaction: training clients need token ids/logprobs for
    # their summary segment and therefore must keep this capability disabled.
    supports_native_responses_compaction: bool = False

    # Whether the provider reads Codex ``agent_message`` items natively.
    # Otherwise they are sent as user turns, while the traced input keeps them as recorded.
    supports_agent_message: bool = False

    def __init__(self, config: OpenAIConfig) -> None:
        self.config = config

    @property
    def input_cache_task_id(self) -> str | None:
        return get_input_cache_task_id(self)

    @property
    def cache_task_id(self) -> str | None:
        return get_cache_task_id(self)

    def _apply_enforced_request_args(self, request_args: Mapping[str, Any]) -> dict[str, Any]:
        return apply_enforced_request_args(request_args, self.config.enforced_request_args)

    def _resolve_effective_request_args(self, request_args: Mapping[str, Any]) -> dict[str, Any]:
        return resolve_effective_request_args(request_args, self.config.enforced_request_args)

    def get_effective_request_args(self) -> dict[str, Any]:
        """Return configured request args after the enforced policy."""
        return self._resolve_effective_request_args(self.config.request_args)

    def _record_provider_call(self, provider_call: dict[str, Any]) -> None:
        sanitized = redact_provider_io(provider_call, sensitive_source=self.config.client_args)
        if not isinstance(sanitized, dict):
            sanitized = {}
        provider_call.clear()
        provider_call.update(sanitized)
        span_id = get_current_span_id()
        if span_id is not None:
            update_span(span_id, attributes={'provider_call': sanitized})

    @observe('openai_completion', self_attrs=['_client_key', '_chat_phase', 'input_cache_task_id', 'cache_task_id'])
    @cacheable(tags=['llm', 'completion'], ignore_args=['self', 'request_meta'])
    async def create_completion(
        self,
        messages: list[ChatCompletionMessageParam | dict],
        model: str | Any = None,
        request_meta: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        if request_meta is not None and self.supports_request_meta:
            kwargs['request_meta'] = request_meta
        messages = _messages_for_provider(messages, native_agent_message=self.supports_agent_message)
        started = time.perf_counter()
        response = await self._create_completion(messages, model, **kwargs)
        credit_runner_timeout_for_response(response, time.perf_counter() - started, started_at=started)
        return response

    @abstractmethod
    async def _create_completion(
        self,
        messages: list[ChatCompletionMessageParam | dict],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion: ...

    @observe('openai_count_tokens', self_attrs=['_client_key'])
    async def count_tokens(
        self,
        *,
        messages: list[ChatCompletionMessageParam | dict],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
    ) -> TokenCountResult:
        messages = _messages_for_provider(messages, native_agent_message=self.supports_agent_message)
        return await self._count_tokens(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )

    async def _count_tokens(
        self,
        *,
        messages: list[ChatCompletionMessageParam | dict],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
    ) -> TokenCountResult:
        raise NotImplementedError(
            f'{type(self).__name__} does not support count_tokens; override _count_tokens() to enable it.'
        )


def validate_openai_compatible_client(obj: Any, _name: str, _meta: dict) -> None:
    if not callable(getattr(obj, '_create_completion', None)):
        raise TypeError(f'OpenAICompatibleClient must have a callable create_completion method, got {type(obj)}')


openai_client_registry = Registry[type[OpenAICompatibleClient]]('openai_client', validate_openai_compatible_client)
