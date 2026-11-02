# /// script
# dependencies = []
#
# [project]
# name = "litellm"
# authors = [
#     { name = "tarschen" },
#     { name = "edwardswu" },
# ]
#
# [optional-dependencies]
# host = ["boto3", "litellm>=1.83.14,<2"]
#
# [tool.nexus]
# openai_clients = { litellm = "nexus.extensions.openai_clients.litellm" }
# run_parts = { litellm = "nexus.extensions.openai_clients.litellm" }
# ///

from __future__ import annotations

import copy
import hashlib
import logging
import uuid

from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from typing import Any, cast
from urllib.parse import parse_qs, parse_qsl, quote, urlparse

import litellm

from litellm.exceptions import ContextWindowExceededError as LiteLLMContextWindowExceededError
from litellm.litellm_core_utils.prompt_templates.factory import THOUGHT_SIGNATURE_SEPARATOR

from nexus.core import ContextLengthExceededError, OpenAICompatibleClient, OpenAIConfig, openai_client_registry
from nexus.core.llm_stream_events import (
    STREAM_EVENT_CLIENT_ARG_KEYS,
    ChatCompletionStreamEventEmitter,
    LLMStreamEmitter,
)
from nexus.core.types import ExtendedChatCompletion
from nexus.extensions.openai_clients.cache_task_id import normalize_cache_task_id
from nexus.runner.compose import run_part
from nexus.trace import get_current_span_id, update_span
from nexus.utils.common import configurable_retry


litellm.suppress_debug_info = True  # ty: ignore[invalid-assignment]
logger = logging.getLogger(__name__)


@run_part('litellm')
def litellm_run_part(params: dict[str, Any]) -> dict[str, Any]:
    if params:
        raise ValueError(f'Run part litellm does not accept parameters: {sorted(params)}')
    return {
        'openai_clients': {
            'main': {
                'name': 'litellm',
                'config': {
                    'client_args': {
                        'api_key': '${oc.env:OPENAI_API_KEY,""}',
                        'base_url': '${oc.env:OPENAI_BASE_URL,""}',
                    },
                    'request_args': {'model': '${oc.env:OPENAI_MODEL,""}'},
                },
            },
        },
    }


def _patch_anthropic_response_id():
    """Patch litellm's Anthropic handler to preserve the original response id.

    The Anthropic transform_parsed_response method does not copy the 'id' field from the
    API response into the ModelResponse, causing it to always return a litellm-generated
    'chatcmpl-<uuid>' id instead of the original one from the provider.
    """
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    _original = AnthropicConfig.transform_parsed_response

    def _patched(self, completion_response, raw_response, model_response, **kwargs):
        result = _original(self, completion_response, raw_response, model_response, **kwargs)
        if 'id' in completion_response:
            result.id = completion_response['id']
        return result

    AnthropicConfig.transform_parsed_response = _patched  # ty: ignore[invalid-assignment]


_patch_anthropic_response_id()


def _patch_responses_extra_body():
    """Patch litellm's LiteLLMResponsesTransformationHandler to preserve unknown extra_body params.

    The _extract_extra_body_params method only promotes known Responses API params from extra_body
    to optional_params, but silently discards unknown params like 'advanced_options'.
    This patch wraps transform_request to capture the original extra_body before it is processed,
    then injects any unrecognized keys back into request_data['extra_body'] so that the openai SDK
    merges them into the JSON request body and forwards them to the downstream API.
    """
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    _original = LiteLLMResponsesTransformationHandler.transform_request

    def _patched(self, model, messages, optional_params, **kwargs):
        extra_body_before = dict(optional_params.get('extra_body', {}) or {})
        request_data = _original(self, model, messages, optional_params, **kwargs)
        remaining = {k: v for k, v in extra_body_before.items() if k not in request_data}
        if remaining and 'advanced_options' in remaining:
            request_data['extra_body'] = {**request_data.get('extra_body', {}), **remaining}
        return request_data

    LiteLLMResponsesTransformationHandler.transform_request = _patched  # ty: ignore[invalid-assignment]


_patch_responses_extra_body()


# ============== Gemini long decode (continuation loop) ==============
# Gemini 4 caps one generateContent hop (Argon: 65,536 output tokens). With max_tokens above that cap
# the hop stops with finishReason CONTINUATION and returns candidates[].continuationToken; nexus resumes
# by resending the unchanged request with the opaque token as a top-level body field, then folds the hops
# together. litellm 1.82 drops the token (and may map the stop to "stop"), so a hop that spent its output
# on thinking would otherwise look like an empty final answer; see
# https://openrouter.ai/docs/cookbook/evaluate-and-optimize/model-migrations/gemini-4#how-long-decode-works
# With max_tokens at or below the cap, the hop stops with MAX_TOKENS instead: nexus does not resume it but
# returns finish_reason "length" (native_finish_reason MAX_TOKENS) and leaves raising max_tokens to the caller.

GEMINI_CONTINUATION_TOKEN_FIELD = 'gemini_continuation_token'
GEMINI_CONTINUATION_INFO_FIELD = 'gemini_continuation'
_GEMINI_LENGTH_FINISH_REASONS = frozenset({'CONTINUATION', 'MAX_TOKENS'})
GEMINI_CONTINUATION_DEFAULT_MAX_HOPS = 8

# Streaming hops: litellm's Gemini stream parser drops finishReason/continuationToken, so the patched
# parser records them into the collector dict installed for the current hop.
_gemini_stream_collector: ContextVar[dict[str, Any] | None] = ContextVar('_gemini_stream_collector', default=None)


def _patch_gemini_continuation_token():
    """Expose Gemini's raw finishReason and continuationToken on each choice's provider_specific_fields."""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    original = VertexGeminiConfig._transform_google_generate_content_to_openai_model_response
    if getattr(original, '_nexus_continuation_patched', False):
        return

    def _patched(self, completion_response, model_response, *args, **kwargs):
        result = original(self, completion_response, model_response, *args, **kwargs)
        candidates = completion_response.get('candidates') if isinstance(completion_response, Mapping) else None
        choices = getattr(result, 'choices', None)
        if not isinstance(candidates, list) or not isinstance(choices, list):
            return result
        for idx, candidate in enumerate(candidates):
            if idx >= len(choices) or not isinstance(candidate, Mapping):
                continue
            native = candidate.get('finishReason')
            token = candidate.get('continuationToken')
            if not native and not token:
                continue
            fields = dict(getattr(choices[idx], 'provider_specific_fields', None) or {})
            if native:
                fields['native_finish_reason'] = native
            if token:
                fields[GEMINI_CONTINUATION_TOKEN_FIELD] = token
            choices[idx].provider_specific_fields = fields
        return result

    _patched._nexus_continuation_patched = True  # ty: ignore[unresolved-attribute]
    VertexGeminiConfig._transform_google_generate_content_to_openai_model_response = _patched  # ty: ignore[invalid-assignment]


_patch_gemini_continuation_token()


def _patch_gemini_stream_continuation_token():
    """Record each streamed chunk's raw finishReason and latest continuationToken for the current hop.

    A streaming hop can carry several tokens (one per checkpoint plus one at the stop); the last one wins.
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import ModelResponseIterator

    original = ModelResponseIterator.chunk_parser
    if getattr(original, '_nexus_continuation_patched', False):
        return

    def _patched(self, chunk, *args, **kwargs):
        collector = _gemini_stream_collector.get()
        if collector is not None and isinstance(chunk, Mapping):
            for candidate in chunk.get('candidates') or []:
                if not isinstance(candidate, Mapping) or int(candidate.get('index') or 0) != 0:
                    continue
                native = candidate.get('finishReason')
                token = candidate.get('continuationToken')
                if isinstance(native, str) and native:
                    collector['native_finish_reason'] = native
                if isinstance(token, str) and token:
                    collector[GEMINI_CONTINUATION_TOKEN_FIELD] = token
        return original(self, chunk, *args, **kwargs)

    _patched._nexus_continuation_patched = True  # ty: ignore[unresolved-attribute]
    ModelResponseIterator.chunk_parser = _patched  # ty: ignore[invalid-assignment]


_patch_gemini_stream_continuation_token()


def _first_choice(response_json: dict[str, Any]) -> dict[str, Any]:
    choices = response_json.get('choices') or []
    return choices[0] if choices and isinstance(choices[0], dict) else {}


def _gemini_pending_continuation(response_json: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (native_finish_reason, token); token is set only when the hop stopped at a length limit."""
    fields = _first_choice(response_json).get('provider_specific_fields') or {}
    native = fields.get('native_finish_reason')
    native = native if isinstance(native, str) else None
    token = fields.get(GEMINI_CONTINUATION_TOKEN_FIELD)
    if native in _GEMINI_LENGTH_FINISH_REASONS and isinstance(token, str) and token:
        return native, token
    return native, None


def _usage_int(usage: Any, *path: str) -> int:
    value: Any = usage
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return value if isinstance(value, int) else 0


def _merge_gemini_continuation_hops(hops: list[dict[str, Any]], *, unresolved: bool) -> dict[str, Any]:
    """Fold continuation hops into one completion.

    Text, reasoning, thinking blocks, tool calls and thought signatures concatenate in hop order.

    ``usage`` keeps OpenAI's per-request meaning, because downstream consumers (the Anthropic /
    Responses / Gemini proxies and the CLI agents behind them) read prompt tokens as the current context
    size: prompt and cached tokens come from the first hop (later hops resend that prompt plus the
    earlier hops' output), completion and reasoning tokens add up over all hops, so ``total_tokens`` is
    the context at the end of the turn. What was actually billed (every hop's prompt and cached tokens
    summed) goes to ``provider_specific_fields.gemini_continuation.billed_*`` together with per-hop usage;
    billing-oriented processors prefer those fields (see ``nexus.core.types.usage.billed_prompt_usage``).
    """
    merged = copy.deepcopy(hops[-1])
    choice = _first_choice(merged)
    messages = [_first_choice(h).get('message') or {} for h in hops]

    def _join(key: str) -> str | None:
        parts = [m[key] for m in messages if isinstance(m.get(key), str) and m[key]]
        return ''.join(parts) if parts else None

    def _concat(key: str) -> list[Any] | None:
        items = [item for m in messages for item in (m.get(key) or [])]
        return items or None

    message = choice.setdefault('message', {})
    message['content'] = _join('content')
    message['reasoning_content'] = _join('reasoning_content')
    message['thinking_blocks'] = _concat('thinking_blocks')
    message['tool_calls'] = _concat('tool_calls')
    for i, call in enumerate(message['tool_calls'] or []):
        if isinstance(call, dict) and 'index' in call:
            call['index'] = i
    signatures = [
        sig for m in messages for sig in ((m.get('provider_specific_fields') or {}).get('thought_signatures') or [])
    ]
    if signatures:
        message['provider_specific_fields'] = {
            **(message.get('provider_specific_fields') or {}),
            'thought_signatures': signatures,
        }

    hop_usage = []
    for h in hops:
        u = h.get('usage') or {}
        hop_usage.append({
            'prompt_tokens': _usage_int(u, 'prompt_tokens'),
            'completion_tokens': _usage_int(u, 'completion_tokens'),
            'reasoning_tokens': _usage_int(u, 'completion_tokens_details', 'reasoning_tokens'),
            'cached_tokens': _usage_int(u, 'prompt_tokens_details', 'cached_tokens'),
            'native_finish_reason': (_first_choice(h).get('provider_specific_fields') or {}).get(
                'native_finish_reason'
            ),
        })
    usage = copy.deepcopy(hops[0].get('usage') or {})
    if usage:
        completion = sum(x['completion_tokens'] for x in hop_usage)
        usage['completion_tokens'] = completion
        usage['total_tokens'] = hop_usage[0]['prompt_tokens'] + completion
        details = usage.get('completion_tokens_details')
        if isinstance(details, dict) and 'reasoning_tokens' in details:
            details['reasoning_tokens'] = sum(x['reasoning_tokens'] for x in hop_usage)
        merged['usage'] = usage

    fields = {
        k: v for k, v in (choice.get('provider_specific_fields') or {}).items() if k != GEMINI_CONTINUATION_TOKEN_FIELD
    }
    fields[GEMINI_CONTINUATION_INFO_FIELD] = {
        'hops': len(hops),
        'unresolved': unresolved,
        'context_prompt_tokens': hop_usage[0]['prompt_tokens'],
        'billed_prompt_tokens': sum(x['prompt_tokens'] for x in hop_usage),
        'billed_cached_tokens': sum(x['cached_tokens'] for x in hop_usage),
        'hop_usage': hop_usage,
    }
    choice['provider_specific_fields'] = fields
    if unresolved:
        choice['finish_reason'] = 'length'
    elif message['tool_calls']:
        choice['finish_reason'] = 'tool_calls'
    return merged


def _drop_gemini_continuation_token(response_json: dict[str, Any]) -> dict[str, Any]:
    for choice in response_json.get('choices') or []:
        fields = choice.get('provider_specific_fields') if isinstance(choice, dict) else None
        if isinstance(fields, dict):
            fields.pop(GEMINI_CONTINUATION_TOKEN_FIELD, None)
    return response_json


# ============== Gemini thought signature (synthetic tool call id) ==============
# Gemini's functionCall carries no OpenAI-style tool call id, so litellm mints one and appends the turn's thought signature to it. Every surface that echoes an id then carries the ciphertext: prompts, event logs, and the file names some harnesses derive from the id. nexus hands out an id of its own instead and replays the signature on the tool call itself, which is the other source litellm reads it from. The id litellm issued is not reused: it can repeat across turns, and a repeat would let a later turn's signature overwrite an earlier one keyed by it.

# Cap the per-client map so a long-lived runtime cannot grow it without bound.
GEMINI_THOUGHT_SIGNATURE_CACHE_SIZE = 8192


def _strip_gemini_thought_signatures(
    response_json: dict[str, Any], signatures: dict[str, str], renamed: dict[str, str] | None = None
) -> dict[str, Any]:
    """Move each tool call's thought signature off its id, onto the tool call and into ``signatures``.

    With ``renamed`` (a Gemini response), every tool call also gets a freshly minted id, recorded in ``renamed`` under the id litellm issued; pass the map the stream events already used so both surfaces carry the same ids.
    """
    for choice in response_json.get('choices') or []:
        message = choice.get('message') if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            continue
        used: set[str] = set()
        for tool_call in message.get('tool_calls') or []:
            raw_id = tool_call.get('id') if isinstance(tool_call, dict) else None
            if not isinstance(raw_id, str):
                continue
            clean_id, separator, signature = raw_id.partition(THOUGHT_SIGNATURE_SEPARATOR)
            if renamed is None and not separator:
                continue
            if renamed is not None:
                clean_id = renamed.setdefault(clean_id, f'call_{uuid.uuid4().hex[:28]}')
                # one response reusing an id for two calls still needs two
                if clean_id in used:
                    clean_id = f'call_{uuid.uuid4().hex[:28]}'
                used.add(clean_id)
            fields = tool_call.get('provider_specific_fields')
            signature = signature or (fields.get('thought_signature') if isinstance(fields, dict) else None)
            if signature:
                tool_call['provider_specific_fields'] = {
                    **(fields if isinstance(fields, dict) else {}),
                    'thought_signature': signature,
                }
                signatures[clean_id] = signature
            tool_call['id'] = clean_id
    while len(signatures) > GEMINI_THOUGHT_SIGNATURE_CACHE_SIZE:
        del signatures[next(iter(signatures))]
    return response_json


def _bare_gemini_stream_chunk(chunk: Any, renamed: dict[str, str] | None = None) -> Any:
    """Dump a streamed chunk for the stream event emitter, with the thought signature cut off every tool call id.

    With ``renamed``, each id is also swapped for the one :func:`_strip_gemini_thought_signatures` will hand out.
    """
    data = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
    if not isinstance(data, dict):
        return chunk

    def touched_tool_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            tool_call
            for choice in payload.get('choices') or []
            if isinstance(choice, dict) and isinstance(choice.get('delta'), dict)
            for tool_call in choice['delta'].get('tool_calls') or []
            if isinstance(tool_call, dict)
            and isinstance(tool_call.get('id'), str)
            and (renamed is not None or THOUGHT_SIGNATURE_SEPARATOR in tool_call['id'])
        ]

    if not touched_tool_calls(data):
        return data
    # Copy first: the same chunk still goes to stream_chunk_builder, whose ids the final strip reads.
    data = copy.deepcopy(data)
    for tool_call in touched_tool_calls(data):
        clean_id = tool_call['id'].partition(THOUGHT_SIGNATURE_SEPARATOR)[0]
        tool_call['id'] = clean_id if renamed is None else renamed.setdefault(clean_id, f'call_{uuid.uuid4().hex[:28]}')
    return data


def _restore_gemini_thought_signatures(
    messages: list[dict[str, Any]], signatures: dict[str, str]
) -> list[dict[str, Any]]:
    """Put each cached thought signature back on the replayed tool call a scaffold dropped it from."""
    if not signatures:
        return messages
    restored: list[dict[str, Any]] = []
    changed = False
    for message in messages:
        tool_calls = message.get('tool_calls') if isinstance(message, dict) else None
        if not tool_calls:
            restored.append(message)
            continue
        rebuilt: list[Any] = []
        for tool_call in tool_calls:
            fields = tool_call.get('provider_specific_fields') if isinstance(tool_call, dict) else None
            signature = signatures.get(tool_call.get('id')) if isinstance(tool_call, dict) else None
            if not signature or (isinstance(fields, dict) and fields.get('thought_signature')):
                rebuilt.append(tool_call)
                continue
            rebuilt.append({
                **tool_call,
                'provider_specific_fields': {
                    **(fields if isinstance(fields, dict) else {}),
                    'thought_signature': signature,
                },
            })
            changed = True
        restored.append({**message, 'tool_calls': rebuilt})
    return restored if changed else messages


# ============== Mid-Conversation System Message Support ==============

_preserve_mid_system: ContextVar[bool] = ContextVar('_preserve_mid_system', default=False)


def _has_mid_conversation_system(messages: list[dict[str, Any]]) -> bool:
    """Return True if messages contain a system message after the first non-system message."""
    seen_non_system = False
    for msg in messages:
        if msg.get('role') != 'system':
            seen_non_system = True
        elif seen_non_system:
            return True
    return False


def _model_supports_mid_conversation_system(model: str | None, custom_llm_provider: str | None) -> bool:
    if not model:
        return False
    try:
        from litellm.utils import _supports_factory

        return _supports_factory(
            model=model, custom_llm_provider=custom_llm_provider, key='supports_mid_conversation_system'
        )
    except Exception:
        return False


def _resolve_bedrock_invoke_route(model: str, custom_llm_provider: str | None) -> tuple[str, str | None]:
    """Resolve a Bedrock model to the Invoke route without losing its provider.

    Uses litellm's own provider resolution so that all model name formats
    (anthropic.claude-..., bedrock/anthropic.claude-..., etc.) are handled.
    """
    try:
        resolved_model, resolved_provider, _, _ = litellm.get_llm_provider(
            model=model, custom_llm_provider=custom_llm_provider
        )
    except Exception:
        resolved_model = model
        resolved_provider = custom_llm_provider

    if resolved_provider != 'bedrock' and custom_llm_provider != 'bedrock':
        return model, custom_llm_provider

    if resolved_model.startswith('invoke/'):
        invoke_model = resolved_model
    elif resolved_model.startswith('converse/'):
        invoke_model = f'invoke/{resolved_model.removeprefix("converse/")}'
    else:
        invoke_model = f'invoke/{resolved_model}'
    return invoke_model, 'bedrock'


def _format_mid_conversation_system_message(message: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI-style system message to an Anthropic message entry."""
    content = message.get('content', '')
    if isinstance(content, str):
        block: dict[str, Any] = {'type': 'text', 'text': content}
        if message.get('cache_control') is not None:
            block['cache_control'] = copy.deepcopy(message['cache_control'])
        formatted: list[Any] = [block]
    elif isinstance(content, list):
        formatted = copy.deepcopy(content)
    else:
        formatted = [{'type': 'text', 'text': str(content)}]
    return {'role': 'system', 'content': formatted}


def _demote_mid_conversation_system_for_gemini(messages: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    """Keep mid-conversation system messages in place for Gemini by sending them as user turns.

    litellm's Gemini transform hoists every system message into ``system_instruction``, so a per-turn
    reminder (e.g. Claude Code's ``<total_tokens>``) would change the prefix on every request and defeat
    prefix caching. Gemini has no in-conversation system role; a user turn keeps the prefix stable.
    """
    from nexus.extensions.openai_clients.vertex import _supports_mid_conversation_system

    if not isinstance(model, str) or 'gemini' not in model.lower() or not _supports_mid_conversation_system(model):
        return messages
    if not _has_mid_conversation_system(messages):
        return messages
    result: list[dict[str, Any]] = []
    started = False
    for message in messages:
        if message.get('role') in {'system', 'developer'} and started:
            content = message.get('content', '')
            if isinstance(content, list):
                content = [
                    {k: v for k, v in block.items() if k != 'cache_control'} if isinstance(block, dict) else block
                    for block in content
                ]
            result.append({'role': 'user', 'content': content})
            continue
        if message.get('role') not in {'system', 'developer'}:
            started = True
        result.append(message)
    return result


def _patch_anthropic_transform_for_mid_system():
    """Preserve mid-system ordering after LiteLLM's native request transform.

    LiteLLM 1.105.1 recognizes mid-conversation system messages, but aggregates
    consecutive messages and can move a boundary behind the following user turn.
    Run its complete transform first, then replace only the final ``messages``
    array with independently transformed non-system segments and in-place system
    entries. Request-wide validation, tools, thinking, and the leading top-level
    system field remain owned by LiteLLM.
    """
    from litellm.litellm_core_utils.prompt_templates import factory as prompt_factory
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    _original_transform_request = AnthropicConfig.transform_request
    _original_anthropic_messages_pt = prompt_factory.anthropic_messages_pt

    def _convert_segments(messages, model, llm_provider):
        converted_messages: list[dict[str, Any]] = []
        current_segment: list[dict[str, Any]] = []

        def _flush_segment() -> None:
            if not current_segment:
                return
            converted_messages.extend(
                cast(
                    list[dict[str, Any]],
                    _original_anthropic_messages_pt(
                        messages=cast(Any, list(current_segment)),
                        model=model,
                        llm_provider=llm_provider,
                    ),
                )
            )
            current_segment.clear()

        for message in messages:
            if message.get('role') == 'system':
                _flush_segment()
                converted_messages.append(_format_mid_conversation_system_message(message))
            else:
                current_segment.append(message)
        _flush_segment()

        return converted_messages

    def _patched_transform_request(self, model, messages, optional_params, litellm_params, headers):
        if not _preserve_mid_system.get():
            return _original_transform_request(
                self,
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )

        source_messages = copy.deepcopy(messages)
        payload = _original_transform_request(
            self,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        prefix_end = 0
        for message in source_messages:
            if message.get('role') != 'system':
                break
            prefix_end += 1
        payload['messages'] = _convert_segments(
            source_messages[prefix_end:],
            model=model,
            llm_provider=self._resolved_provider,
        )
        return payload

    AnthropicConfig.transform_request = _patched_transform_request  # ty: ignore[invalid-assignment]


_patch_anthropic_transform_for_mid_system()


_API_KEY_PROVIDER_TO_LITELLM_PROVIDER = {
    'aws_third': 'bedrock',
}


def _with_default_stream_options(request_kwargs: dict[str, Any]) -> dict[str, Any]:
    stream_options = request_kwargs.get('stream_options')
    if isinstance(stream_options, dict):
        stream_options = dict(stream_options)
        stream_options.setdefault('include_usage', True)
        return {**request_kwargs, 'stream_options': stream_options}
    if stream_options is None:
        return {**request_kwargs, 'stream_options': {'include_usage': True}}
    return request_kwargs


def _infer_custom_llm_provider(client_args: dict[str, Any], model: Any) -> str | None:
    if not isinstance(model, str) or not model or '/' in model:
        return None

    api_key = client_args.get('api_key')
    if not isinstance(api_key, str) or '?' not in api_key:
        return None

    parsed = urlparse(f'scheme://placeholder/{api_key.split(":", 1)[-1]}')
    provider = parse_qs(parsed.query).get('provider', [None])[0]
    if provider is None:
        return None

    return _API_KEY_PROVIDER_TO_LITELLM_PROVIDER.get(provider)


def _sanitize_request_kwargs(request_kwargs: dict[str, Any]) -> dict[str, Any]:
    custom_llm_provider = request_kwargs.get('custom_llm_provider')
    model = request_kwargs.get('model')
    if (
        custom_llm_provider == 'bedrock'
        and isinstance(model, str)
        and model.startswith('anthropic.')
        and request_kwargs.get('temperature') is not None
        and request_kwargs.get('top_p') is not None
    ):
        sanitized = dict(request_kwargs)
        sanitized.pop('top_p', None)
        return sanitized
    return request_kwargs


def _ensure_object_tool_parameters(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize function-tool JSON schemas so Gemini accepts them.

    Gemini's function-calling validator is stricter than OpenAI's: it rejects a
    ``parameters`` schema unless it is an object schema, and it also rejects an
    ``object`` schema that carries an *empty* ``properties: {}`` (raising
    ``functionDeclaration parameters schema should be of type OBJECT``). FastMCP
    emits ``{"type": "object", "properties": {}}`` for no-argument tools, which
    trips this check. Here we coerce ``parameters`` to ``{"type": "object"}`` and
    drop empty ``properties``. The transform is a no-op for well-formed schemas
    and stays valid JSON Schema, so it is safe for non-Gemini providers too.
    """
    if not isinstance(tools, list):
        return tools
    changed = False
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        fn = tool.get('function') if isinstance(tool, dict) else None
        params = fn.get('parameters') if isinstance(fn, dict) else None
        empty_props = (
            isinstance(params, dict) and isinstance(params.get('properties'), dict) and not params['properties']
        )
        if isinstance(fn, dict) and (not isinstance(params, dict) or params.get('type') != 'object' or empty_props):
            tool = copy.deepcopy(tool)
            new_params = tool['function'].get('parameters')
            if not isinstance(new_params, dict):
                new_params = {}
            new_params['type'] = 'object'
            props = new_params.get('properties')
            if isinstance(props, dict) and not props:
                new_params.pop('properties', None)
            tool['function']['parameters'] = new_params
            changed = True
        normalized.append(tool)
    return normalized if changed else tools


def _patch_stream_finish_reasons(response_json: dict[str, Any], chunks: list[Any]) -> dict[str, Any]:
    finish_reasons: dict[int, str] = {}
    for chunk in chunks:
        data = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
        if not isinstance(data, dict):
            continue
        for choice in data.get('choices') or []:
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get('finish_reason')
            if not isinstance(finish_reason, str) or not finish_reason:
                continue
            index = int(choice.get('index') or 0)
            finish_reasons[index] = finish_reason

    for choice in response_json.get('choices') or []:
        if not isinstance(choice, dict):
            continue
        index = int(choice.get('index') or 0)
        if index in finish_reasons:
            choice['finish_reason'] = finish_reasons[index]
    return response_json


def _copy_provider_reasoning_signature_to_thinking_blocks(response_json: dict[str, Any]) -> dict[str, Any]:
    for choice in response_json.get('choices') or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get('message')
        if not isinstance(message, dict):
            continue
        provider_specific_fields = message.get('provider_specific_fields')
        if not isinstance(provider_specific_fields, dict):
            continue
        reasoning_content = provider_specific_fields.get('reasoningContent')
        if not isinstance(reasoning_content, dict):
            continue
        signature = reasoning_content.get('signature')
        if not isinstance(signature, str) or not signature:
            continue

        thinking_blocks = message.get('thinking_blocks')
        if isinstance(thinking_blocks, list):
            for block in thinking_blocks:
                if isinstance(block, dict) and block.get('type', 'thinking') == 'thinking':
                    if not isinstance(block.get('thinking'), str):
                        block['thinking'] = ''
                    if not isinstance(block.get('signature'), str) or not block['signature']:
                        block['signature'] = signature
                    break
            else:
                thinking_blocks.append({'type': 'thinking', 'thinking': '', 'signature': signature})
        else:
            message['thinking_blocks'] = [{'type': 'thinking', 'thinking': '', 'signature': signature}]
    return response_json


def _append_api_key_query(api_key: str, **params: str) -> str:
    """Append or replace routing fields on an API-key query string."""
    base, separator, query = api_key.partition('?')
    query_items = [item for item in query.split('&') if item] if separator else []
    for key, value in params.items():
        if not value:
            continue
        query_items = [item for item in query_items if item.split('=', 1)[0] != key]
        query_items.append(f'{quote(key)}={quote(value)}')
    return f'{base}?{"&".join(query_items)}' if query_items else base


def _gemini_tool_call_thought_signature(tool_call: Any) -> str | None:
    """Read a Gemini tool signature from its restored field or legacy id suffix."""
    if not isinstance(tool_call, dict):
        return None
    fields = tool_call.get('provider_specific_fields')
    if isinstance(fields, dict):
        signature = fields.get('thought_signature')
        if isinstance(signature, str) and signature:
            return signature
    call_id = tool_call.get('id')
    if isinstance(call_id, str) and '__thought__' in call_id:
        signature = call_id.split('__thought__', 1)[1]
        if signature:
            return signature
    return None


def _deduplicate_gemini_reasoning_replay(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a replay-safe copy of Gemini assistant history without losing captured CoT.

    Nexus captures Gemini plaintext reasoning and signatures before this request
    transform. Codex history can then carry the same prior thought in three
    spellings: ``reasoning_content``, a synthetic plaintext ``reasoning_items``
    entry, and a signed thinking/tool-call part. Gemini requires signed parts to
    be replayed exactly where received; replaying the plaintext copy as well
    duplicates the prior reasoning and can trigger repetition loops.

    Signed thinking blocks take priority. A signature restored onto a tool
    call's ``provider_specific_fields.thought_signature`` (or still carried by
    the legacy id suffix) takes priority over a same-signature thinking block.
    Message-level ``provider_specific_fields.thought_signatures`` also proves a
    signed representation exists, but signatures already represented by a tool
    call or retained thinking block are removed from that list. Distinct
    message-level signatures and native encrypted reasoning items are
    preserved. The input list is never mutated.
    """
    replay = copy.deepcopy(messages)
    for message in replay:
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        tool_signatures = {
            signature
            for tool_call in message.get('tool_calls') or []
            if (signature := _gemini_tool_call_thought_signature(tool_call))
        }
        provider_fields = message.get('provider_specific_fields')
        message_signatures = {
            signature
            for signature in (
                provider_fields.get('thought_signatures')
                if isinstance(provider_fields, dict) and isinstance(provider_fields.get('thought_signatures'), list)
                else []
            )
            if isinstance(signature, str) and signature
        }
        blocks = message.get('thinking_blocks')
        signed_blocks = [
            block
            for block in blocks or []
            if isinstance(block, dict) and isinstance(block.get('signature'), str) and block['signature']
        ]
        if not tool_signatures and not signed_blocks and not message_signatures:
            continue

        message.pop('reasoning_content', None)

        reasoning_items = message.get('reasoning_items')
        if isinstance(reasoning_items, list):
            native_items = [
                item
                for item in reasoning_items
                if isinstance(item, dict)
                and (item.get('encrypted_content') or item.get('signature') or item.get('thoughtSignature'))
            ]
            if native_items:
                message['reasoning_items'] = native_items
            else:
                message.pop('reasoning_items', None)

        if tool_signatures and isinstance(blocks, list):
            distinct_blocks = [
                block
                for block in blocks
                if not (
                    isinstance(block, dict)
                    and isinstance(block.get('signature'), str)
                    and block['signature'] in tool_signatures
                )
            ]
            if distinct_blocks:
                message['thinking_blocks'] = distinct_blocks
            else:
                message.pop('thinking_blocks', None)

        represented_signatures = set(tool_signatures)
        represented_signatures.update(
            block['signature']
            for block in message.get('thinking_blocks') or []
            if isinstance(block, dict) and isinstance(block.get('signature'), str) and block['signature']
        )
        if isinstance(provider_fields, dict):
            raw_message_signatures = provider_fields.get('thought_signatures')
            if isinstance(raw_message_signatures, list):
                distinct_message_signatures = []
                seen_signatures = set(represented_signatures)
                for signature in raw_message_signatures:
                    if not isinstance(signature, str) or not signature or signature in seen_signatures:
                        continue
                    distinct_message_signatures.append(signature)
                    seen_signatures.add(signature)
                if distinct_message_signatures:
                    provider_fields['thought_signatures'] = distinct_message_signatures
                else:
                    provider_fields.pop('thought_signatures', None)
                if not provider_fields:
                    message.pop('provider_specific_fields', None)
    return replay


@openai_client_registry.register('litellm')
class LiteLLMClient(OpenAICompatibleClient):
    """LiteLLM client with optional Anthropic prompt caching support.

    Cache control can be enabled via client_args:
        - cache_control_enabled: bool = False
        - cache_control_last_n_messages: int = 2
        - cache_control_tagged_roles: list[str] = ["user", "tool"]

    strip_cache_control: bool = False drops every cache_control marker, including ones sent by
    agents such as Claude Code, so litellm's gemini/ provider does not hit the cachedContents API.

    Gateway routing via client_args:
        - provider: str, sent as ``provider=`` on the api-key query and replacing any
          provider the key already carries (e.g. pin an AIHub model to one upstream).
    """

    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        # Gemini thought signatures taken off the tool call ids handed out, keyed by the bare id.
        self._thought_signatures: dict[str, str] = {}
        internal = {
            'cache_task_id',
            'request_id_from_cache_task_id',
            'max_retries',
            'log_llm_usage',
            'strip_cache_control',
            'provider',
            'gemini_continuation',
            'gemini_continuation_max_hops',
            'gemini_deduplicate_reasoning_replay',
        } | STREAM_EVENT_CLIENT_ARG_KEYS
        self.client_args = {
            k: v for k, v in config.client_args.items() if not k.startswith('cache_control_') and k not in internal
        }
        cache_task_id = config.client_args.get('cache_task_id')
        api_key = self.client_args.get('api_key')
        provider = config.client_args.get('provider')
        if provider:
            if not isinstance(provider, str):
                raise TypeError('provider must be a string')
            if not isinstance(api_key, str) or not api_key:
                raise ValueError('provider requires a non-empty api_key')
            api_key = self.client_args['api_key'] = _append_api_key_query(api_key, provider=provider)
        if cache_task_id:
            if not isinstance(cache_task_id, str):
                raise TypeError('cache_task_id must be a string')
            # The routing hint travels in the api-key query string, so there must be a key to extend.
            if not isinstance(api_key, str) or not api_key:
                raise ValueError('cache_task_id requires a non-empty api_key')
            self.client_args['api_key'] = normalize_cache_task_id(api_key, cache_task_id)
            if config.client_args.get('request_id_from_cache_task_id'):
                request_id = hashlib.md5(
                    f'{cache_task_id}:routed-auth'.encode(),
                    usedforsecurity=False,
                ).hexdigest()
                self.client_args['api_key'] = _append_api_key_query(
                    self.client_args['api_key'],
                    request_id=request_id,
                )

    async def create_completion(self, messages, model=None, request_meta=None, **kwargs) -> ExtendedChatCompletion:
        response = await super().create_completion(messages, model=model, request_meta=request_meta, **kwargs)
        # A replayed completion never reaches _create_completion, so rebuild the map from the response itself.
        for choice in response.choices:
            for tool_call in choice.message.tool_calls or []:
                fields = getattr(tool_call, 'provider_specific_fields', None)
                if not isinstance(fields, dict):
                    continue
                signature = fields.get('thought_signature')
                if isinstance(signature, str) and signature:
                    self._thought_signatures[tool_call.id] = signature
        while len(self._thought_signatures) > GEMINI_THOUGHT_SIGNATURE_CACHE_SIZE:
            del self._thought_signatures[next(iter(self._thought_signatures))]
        return response

    async def _create_completion(self, messages, model=None, **kwargs):
        if self.config.client_args.get('cache_control_enabled'):
            strategy = self.config.client_args.get('cache_control_strategy', 'default')
            cache_control_kwargs = {
                k.removeprefix('cache_control_'): v
                for k, v in self.config.client_args.items()
                if k.startswith('cache_control_')
                and k not in {'cache_control_enabled', 'cache_control_strategy'}
                and v is not None
            }
            if strategy == 'claude_code':
                messages = apply_cache_control_claude_code(messages)
                if 'tools' in kwargs:
                    kwargs['tools'] = apply_tools_cache_control(kwargs['tools'])
            elif '::' in strategy:
                # Pluggable strategy: load a ``(messages, **kwargs) -> messages`` callable by
                # import path, using the same ``module::func`` convention as tool dispatch.
                # Lets a task ship its own cache strategy without hardcoding it here.
                from nexus.core.registry import import_from_path

                messages = import_from_path(strategy)(messages, **cache_control_kwargs)
            else:
                messages = apply_cache_control(messages, **cache_control_kwargs)
        if self.config.client_args.get('strip_cache_control'):
            messages = strip_cache_control(messages)
            if isinstance(kwargs.get('tools'), list):
                kwargs['tools'] = strip_cache_control(kwargs['tools'])

        request_args_override = kwargs.pop('_request_args_override', None)
        max_retries = self.config.client_args.get('max_retries', 15)
        request_kwargs = {
            'model': model,
            **self.client_args,
            **(request_args_override if request_args_override is not None else self.config.request_args),
            **kwargs,
            'messages': messages,
        }
        nexus_trace_id = request_kwargs.pop('nexus_trace_id', None)
        nexus_lb_strategy = request_kwargs.pop('nexus_llm_lb_strategy', None)
        if nexus_trace_id or nexus_lb_strategy:
            configured_headers = request_kwargs.get('extra_headers')
            extra_headers = dict(configured_headers) if isinstance(configured_headers, Mapping) else {}
            if nexus_trace_id:
                extra_headers.setdefault('SessionID', str(nexus_trace_id))
            if nexus_lb_strategy:
                extra_headers.setdefault('StrategyType', str(nexus_lb_strategy))
            request_kwargs['extra_headers'] = extra_headers
        request_kwargs.setdefault(
            'custom_llm_provider',
            _infer_custom_llm_provider(self.client_args, request_kwargs.get('model')),
        )
        request_kwargs = _sanitize_request_kwargs(request_kwargs)
        if self.config.client_args.get('stream'):
            request_kwargs['stream'] = True
        request_kwargs = self._apply_enforced_request_args(request_kwargs)
        if cache_task_id := self.config.client_args.get('cache_task_id'):
            request_kwargs['api_key'] = normalize_cache_task_id(request_kwargs.get('api_key'), cache_task_id)
        if isinstance(request_kwargs.get('messages'), list):
            request_kwargs['messages'] = _demote_mid_conversation_system_for_gemini(
                request_kwargs['messages'], request_kwargs.get('model')
            )
            request_kwargs['messages'] = _restore_gemini_thought_signatures(
                request_kwargs['messages'], self._thought_signatures
            )
        stream_enabled = bool(self.config.client_args.get('stream'))
        enforced_defaults = self._resolve_effective_request_args({})
        if 'stream' in enforced_defaults:
            stream_enabled = enforced_defaults['stream'] is True
        # Only Gemini needs these fixups; it reaches us via OpenAI-compatible
        # gateways, so the model name (e.g. api_google_gemini-...) is the only signal.
        model_name = request_kwargs.get('model')
        is_gemini = isinstance(model_name, str) and 'gemini' in model_name.lower()
        if is_gemini:
            if self.config.client_args.get('gemini_deduplicate_reasoning_replay', True) and isinstance(
                request_kwargs.get('messages'), list
            ):
                request_kwargs['messages'] = _deduplicate_gemini_reasoning_replay(request_kwargs['messages'])
            if 'tools' in request_kwargs:
                request_kwargs['tools'] = _ensure_object_tool_parameters(request_kwargs['tools'])
            # Gemini has no such control, yet litellm rejects False when several tools are given.
            if request_kwargs.get('parallel_tool_calls') is False:
                del request_kwargs['parallel_tool_calls']
            # Responses clients such as Codex send text.verbosity, but Gemini has no
            # equivalent and newer LiteLLM versions reject it before making the request.
            request_kwargs.pop('verbosity', None)

        # Mid-conversation system: force Invoke route + activate patch via ContextVar
        _mid_system_token = None
        _model_for_mid = request_kwargs.get('model')
        _provider_for_mid = request_kwargs.get('custom_llm_provider')
        _has_mid_sys = isinstance(request_kwargs.get('messages'), list) and _has_mid_conversation_system(
            request_kwargs['messages']
        )
        if _has_mid_sys and _model_supports_mid_conversation_system(_model_for_mid, _provider_for_mid):
            if isinstance(_model_for_mid, str):
                resolved_model, resolved_provider = _resolve_bedrock_invoke_route(_model_for_mid, _provider_for_mid)
                request_kwargs['model'] = resolved_model
                if resolved_provider is not None:
                    request_kwargs['custom_llm_provider'] = resolved_provider
            _mid_system_token = _preserve_mid_system.set(True)
        elif _has_mid_sys:
            logger.warning(
                'Mid-conversation system messages detected but model %s does not support '
                'supports_mid_conversation_system; system messages will be aggregated to the top (degraded).',
                _model_for_mid,
            )

        try:
            api_key = request_kwargs.get('api_key')
            # Keep the latest dispatched ID; each span records its own request's ID.
            self._request_cache_task_id = (
                dict(parse_qsl(api_key.partition('?')[2])).get('cache_task_id') if isinstance(api_key, str) else None
            )
            if (span_id := get_current_span_id()) is not None:
                update_span(span_id, attributes={'cache_task_id': self._request_cache_task_id})
            if stream_enabled:
                return await self._create_completion_with_stream_events(request_kwargs, max_retries=max_retries)
            if self._gemini_continuation_enabled(request_kwargs.get('model')):
                response_json = await self._acompletion_with_gemini_continuation(request_kwargs, max_retries)
            else:
                response = await configurable_retry(max_retries)(litellm.acompletion)(**request_kwargs)
                response_json = response.model_dump()
            response_json = _copy_provider_reasoning_signature_to_thinking_blocks(response_json)
            response_json = _strip_gemini_thought_signatures(
                response_json, self._thought_signatures, {} if is_gemini else None
            )
            if request_kwargs.get('model', '').startswith('openai/responses/'):
                for choice in response_json.get('choices') or []:
                    message = choice.get('message', {})
                    raw_annotations = message.get('annotations')
                    if raw_annotations:
                        message['annotations'] = [
                            {
                                'type': ann['type'],
                                'url_citation': {
                                    'url': ann.get('url', ''),
                                    'title': ann.get('title', ''),
                                    'start_index': ann.get('start_index', 0),
                                    'end_index': ann.get('end_index', 0),
                                },
                            }
                            for ann in raw_annotations
                            if ann.get('type') == 'url_citation'
                        ] or None
            return ExtendedChatCompletion.model_validate(response_json)
        except LiteLLMContextWindowExceededError as e:
            raise ContextLengthExceededError(f'LiteLLM context window exceeded: {str(e)}', original_error=e) from e
        finally:
            if _mid_system_token is not None:
                _preserve_mid_system.reset(_mid_system_token)

    def _gemini_continuation_enabled(self, model: Any) -> bool:
        return (
            isinstance(model, str)
            and 'gemini' in model.lower()
            and self.config.client_args.get('gemini_continuation', True) is not False
        )

    async def _acompletion_with_gemini_continuation(
        self,
        request_kwargs: dict[str, Any],
        max_retries: int,
        hop: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Run Gemini's long-decode continuation loop and return one merged response dict.

        Each hop resends the identical request plus the previous hop's continuationToken (top-level body
        field, merged in by litellm's extra_body handling). Only CONTINUATION hops are resumed: they mean
        max_tokens exceeds the per-request output cap and the hop hit that cap. A MAX_TOKENS hop means the
        caller's max_tokens itself is spent, so the loop ends and the turn reports finish_reason "length"
        (native_finish_reason MAX_TOKENS); the caller decides whether to raise max_tokens and continue.
        Hitting gemini_continuation_max_hops also ends with finish_reason "length".
        """
        max_hops = int(
            self.config.client_args.get('gemini_continuation_max_hops') or GEMINI_CONTINUATION_DEFAULT_MAX_HOPS
        )
        if hop is None:
            call = configurable_retry(max_retries)(litellm.acompletion)

            async def _plain_hop(kwargs: dict[str, Any]) -> dict[str, Any]:
                return (await call(**kwargs)).model_dump()

            hop = _plain_hop

        hops: list[dict[str, Any]] = []
        token: str | None = None
        native: str | None = None
        while True:
            hop_kwargs = request_kwargs
            if token is not None:
                extra_body = request_kwargs.get('extra_body')
                hop_kwargs = {
                    **request_kwargs,
                    'extra_body': {
                        **(extra_body if isinstance(extra_body, Mapping) else {}),
                        'continuationToken': token,
                    },
                }
            hops.append(await hop(hop_kwargs))
            native, token = _gemini_pending_continuation(hops[-1])
            if token is None or native != 'CONTINUATION' or len(hops) >= max_hops:
                break
            logger.info(
                'Gemini hop %d stopped with CONTINUATION (completion_tokens=%s); resuming',
                len(hops),
                _usage_int(hops[-1].get('usage'), 'completion_tokens'),
            )
        unresolved = native in _GEMINI_LENGTH_FINISH_REASONS
        if unresolved:
            logger.warning(
                'Gemini long decode stopped after %d hop(s) with native_finish_reason=%s; returning finish_reason=length',
                len(hops),
                native,
            )
        if len(hops) == 1 and not unresolved:
            return _drop_gemini_continuation_token(hops[0])
        return _merge_gemini_continuation_hops(hops, unresolved=unresolved)

    async def _create_completion_with_stream_events(
        self,
        request_kwargs: dict[str, Any],
        *,
        max_retries: int,
    ) -> ExtendedChatCompletion:
        streaming_kwargs = _with_default_stream_options({**request_kwargs, 'stream': True})
        model = streaming_kwargs.get('model')
        emitter = LLMStreamEmitter(
            provider='litellm',
            model=model if isinstance(model, str) else None,
            client_key=getattr(self, '_client_key', None),
            include_raw=bool(self.config.client_args.get('stream_events_include_raw', False)),
            max_delta_chars=int(self.config.client_args.get('stream_events_max_delta_chars', 4096) or 4096),
        )
        event_emitter = ChatCompletionStreamEventEmitter(emitter=emitter)
        call = configurable_retry(max_retries)(litellm.acompletion)
        renamed: dict[str, str] | None = {} if isinstance(model, str) and 'gemini' in model.lower() else None

        async def _stream_hop(kwargs: dict[str, Any]) -> dict[str, Any]:
            collector: dict[str, Any] = {}
            collector_token = _gemini_stream_collector.set(collector)
            try:
                stream = await call(**kwargs)
                chunks = []
                async for chunk in stream:
                    chunks.append(chunk)
                    event_emitter.consume_chunk(_bare_gemini_stream_chunk(chunk, renamed))
            finally:
                _gemini_stream_collector.reset(collector_token)
            response = litellm.stream_chunk_builder(chunks, messages=request_kwargs.get('messages'))
            if response is None:
                raise RuntimeError('LiteLLM stream_chunk_builder returned no response')
            response_json = _patch_stream_finish_reasons(response.model_dump(), chunks)
            if collector:
                choice = _first_choice(response_json)
                if choice:
                    choice['provider_specific_fields'] = {**(choice.get('provider_specific_fields') or {}), **collector}
            return response_json

        try:
            if self._gemini_continuation_enabled(model):
                response_json = await self._acompletion_with_gemini_continuation(
                    streaming_kwargs, max_retries, hop=_stream_hop
                )
            else:
                response_json = await _stream_hop(streaming_kwargs)
            response_json = _copy_provider_reasoning_signature_to_thinking_blocks(response_json)
            response_json = _strip_gemini_thought_signatures(response_json, self._thought_signatures, renamed)
            event_emitter.finish(response_json)
            return ExtendedChatCompletion.model_validate(response_json)
        except Exception as exc:
            event_emitter.fail(exc)
            raise


def apply_cache_control(
    messages: list[dict[str, Any]],
    last_n_messages: int = 2,
    tagged_roles: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Apply cache control marks to messages for Anthropic Claude prompt caching."""
    if tagged_roles is None:
        tagged_roles = ['user', 'tool']

    new_messages = copy.deepcopy(messages)
    n_tagged = 0

    for entry in reversed(new_messages):
        if n_tagged < last_n_messages and entry.get('role') in tagged_roles and _tag_message_cache_control(entry):
            n_tagged += 1

    return new_messages


def strip_cache_control(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of messages or tools without any message-, block- or tool-level cache_control."""
    stripped = []
    for item in items:
        if not isinstance(item, dict):
            stripped.append(item)
            continue
        item = {k: v for k, v in item.items() if k != 'cache_control'}
        content = item.get('content')
        if isinstance(content, list):
            item['content'] = [
                {k: v for k, v in block.items() if k != 'cache_control'} if isinstance(block, dict) else block
                for block in content
            ]
        stripped.append(item)
    return stripped


def _tag_message_cache_control(entry: dict[str, Any]) -> bool:
    """Tag a single message with cache_control. Returns True if successfully tagged."""
    # LiteLLM converts OpenAI tool messages into Anthropic tool_result blocks.
    # Keeping the marker at message level makes it land on tool_result itself
    # instead of the nested tool_result.content text block rejected by Bedrock.
    if entry.get('role') == 'tool':
        entry['cache_control'] = {'type': 'ephemeral'}
        return True

    content = entry.get('content')
    if content is None:
        return False

    if isinstance(content, str) and content.strip():
        entry['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        return True

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                text = block.get('text', '')
                if text.strip():
                    block['cache_control'] = {'type': 'ephemeral'}
                    return True

    return False


def apply_cache_control_claude_code(
    messages: list[dict[str, Any]],
    tagged_roles: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Apply cache control for Claude Code proxy requests.

    Tags only the last eligible user/tool message with content validation.
    """
    if tagged_roles is None:
        tagged_roles = ['user', 'tool']

    new_messages = copy.deepcopy(messages)

    for entry in reversed(new_messages):
        if entry.get('role') in tagged_roles and _tag_message_cache_control(entry):
            break

    return new_messages


def apply_tools_cache_control(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tag the last tool definition with cache_control.

    Cache prefix order is system → tools → messages. Tagging the last tool
    caches the entire system + tools prefix as one unit.
    """
    if not tools:
        return tools

    new_tools = copy.deepcopy(tools)
    new_tools[-1]['cache_control'] = {'type': 'ephemeral'}
    return new_tools
