"""Context Agent model clients with optional per-run prompt-cache isolation."""

from __future__ import annotations

import asyncio
import copy
import json
import uuid

from dataclasses import replace
from typing import Any

from nexus.core import OpenAIConfig, openai_client_registry
from nexus.extensions.openai_clients.openai_responses import OpenAIResponsesClient
from nexus.extensions.openai_clients.vertex import (
    VertexAnthropicAdapter,
    VertexAnthropicHTTPError,
    VertexClient,
    _is_anthropic_retryable,
    provider_attempt_error,
)


def _optional_cache_task_id(prefix: Any) -> str | None:
    normalized = str(prefix or '').strip()
    return f'{normalized}:{uuid.uuid4().hex}' if normalized else None


def _is_aihub_transient_auth_error(error: BaseException) -> bool:
    """Return whether AIHub reported its transient UserAuthError/code=2003 response."""

    if not isinstance(error, VertexAnthropicHTTPError) or error.status != 401:
        return False
    try:
        body = json.loads(error.body)
    except (TypeError, json.JSONDecodeError):
        return False
    details = body.get('error') if isinstance(body, dict) else None
    return isinstance(details, dict) and details.get('type') == 'UserAuthError' and str(details.get('code')) == '2003'


def _context_responses_replay_messages(messages: list[Any]) -> list[Any]:
    """Restore native Responses ordering for Context Agent reasoning turns.

    OpenCode represents one Responses turn as a chat-style assistant message
    containing ``reasoning_items`` and ``tool_calls``. Thinking-mode providers
    such as DSV4 require the original reasoning item, including
    ``content[].reasoning_text``, before the matching function calls on every
    subsequent request. Flatten only that Context Agent-specific shape here so
    the generic Responses client keeps its existing behavior.
    """
    replay: list[Any] = []
    for original in messages:
        if not isinstance(original, dict) or original.get('role') != 'assistant':
            replay.append(original)
            continue

        reasoning_items = [
            copy.deepcopy(item)
            for item in (original.get('reasoning_items') or [])
            if isinstance(item, dict) and item.get('type') == 'reasoning'
        ]
        if not reasoning_items:
            replay.append(original)
            continue

        replay.extend(reasoning_items)
        content = original.get('content')
        if content not in (None, '', []):
            replay.append({'role': 'assistant', 'content': copy.deepcopy(content)})

        for raw_tool_call in original.get('tool_calls') or []:
            if not isinstance(raw_tool_call, dict):
                continue
            function = raw_tool_call.get('function') or {}
            if not isinstance(function, dict):
                continue
            arguments = function.get('arguments', '')
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {}, ensure_ascii=False, separators=(',', ':'))
            replay.append({
                'type': 'function_call',
                'call_id': raw_tool_call.get('id') or '',
                'name': function.get('name') or '',
                'arguments': arguments,
            })
    return replay


class _ContextAgentVertexAnthropicAdapter(VertexAnthropicAdapter):
    """Anthropic adapter that retries AIHub's known transient 401 response."""

    async def _post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        provider_call: dict[str, Any],
    ) -> dict[str, Any]:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self._post_once(endpoint, payload)
            except asyncio.CancelledError as error:
                attempt_record = provider_attempt_error(attempt, error)
                captured_response = self._capture_error_response(provider_call, error)
                if captured_response is not None:
                    attempt_record['response'] = captured_response
                provider_call['attempts'].append(attempt_record)
                raise
            except Exception as error:
                attempt_record = provider_attempt_error(attempt, error)
                captured_response = self._capture_error_response(provider_call, error)
                if captured_response is not None:
                    attempt_record['response'] = captured_response
                provider_call['attempts'].append(attempt_record)
                if (
                    not _is_anthropic_retryable(error) and not _is_aihub_transient_auth_error(error)
                ) or attempt == self.max_retries:
                    raise
                await asyncio.sleep(min(2 ** (attempt - 1), 30))
                continue

            provider_call['attempts'].append({'attempt': attempt, 'status': 'success'})
            return response

        raise RuntimeError('Context Agent Vertex Anthropic retry loop exhausted without a response')


@openai_client_registry.register('context_agent_vertex')
class ContextAgentVertexClient(VertexClient):
    """Vertex client with optional per-run cache IDs and transient AIHub 401 retries."""

    def __init__(self, config: OpenAIConfig):
        client_args = dict(config.client_args)
        cache_task_id = _optional_cache_task_id(client_args.pop('cache_task_id_prefix', None))
        if cache_task_id is not None:
            gateway_routing = dict(client_args.get('gateway_routing') or {})
            gateway_routing['cache_task_id'] = cache_task_id
            client_args['gateway_routing'] = gateway_routing
        prepared_config = replace(config, client_args=client_args)
        super().__init__(prepared_config)
        if self.protocol == 'anthropic':
            self._adapter = _ContextAgentVertexAnthropicAdapter(prepared_config, self._record_provider_call)


@openai_client_registry.register('context_agent_openai_responses')
class ContextAgentOpenAIResponsesClient(OpenAIResponsesClient):
    """Responses client with an optional unique per-run gateway cache ID."""

    def __init__(self, config: OpenAIConfig):
        client_args = dict(config.client_args)
        cache_task_id = _optional_cache_task_id(client_args.pop('cache_task_id_prefix', None))
        if cache_task_id is not None:
            client_args['cache_task_id'] = cache_task_id
        super().__init__(replace(config, client_args=client_args))

    async def _create_completion(
        self,
        messages: list[Any],
        model: str | Any = None,
        **kwargs: Any,
    ) -> Any:
        return await super()._create_completion(
            _context_responses_replay_messages(messages),
            model=model,
            **kwargs,
        )


__all__ = [
    'ContextAgentOpenAIResponsesClient',
    'ContextAgentVertexClient',
    '_context_responses_replay_messages',
]
