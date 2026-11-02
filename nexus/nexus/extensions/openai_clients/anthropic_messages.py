# /// script
# dependencies = []
#
# [project]
# name = "anthropic_messages"
# authors = [
#     { name = "kechizhang" },
# ]
#
# [tool.nexus]
# openai_clients = { anthropic_messages = "nexus.extensions.openai_clients.anthropic_messages" }
# ///

"""Anthropic Messages client reachable from a platform EXTERNAL_URL model config.

WHY THIS EXISTS. ``vertex`` (and its deprecated ``vertex_anthropic`` alias) already
speak the Anthropic wire protocol, and all of the payload/response machinery below
is imported from it rather than reimplemented. What they cannot do is be *driven
from the platform*: hy_unify_eval builds a Nexus client from a flat
``parameter_configuration`` blob --

    {"nexus_main_client_name": "anthropic_messages",
     "main_client_args": {"ensure_v1_path": true, "max_retries": 5}}

plus ``external_url_custom_params.url`` / ``.model_name`` -- and merges
``main_client_args`` straight into ``client_args``. Four things blocked reusing
``vertex`` through that door:

1. ``extra_headers`` is silently dropped, so the betas that ``cua_anthropic_agent``
   ships per request (``anthropic-beta``) never reach the gateway.
2. Its ``anthropic-beta`` value is hardcoded to one computer-use revision, so a
   model on a different revision cannot be addressed at all.
3. No ``cache_task_id``, which OSWorld relies on for gateway cache affinity.
4. No base-URL normalization. The platform hands over
   ``http://.../v1``; naive appending yields ``/v1/v1/messages`` -> HTTP 404.

It also needs nested ``protocol`` + ``gateway_routing`` mappings, which a flat
``main_client_args`` cannot express. Everything here is therefore transport,
header and config plumbing; the protocol itself stays in ``vertex``.

Usage locally (secrets from env, never inline)::

    openai_clients:
      main:
        name: anthropic_messages
        config:
          client_args:
            api_key: '${oc.env:ANTHROPIC_API_KEY}'
            base_url: '${oc.env:ANTHROPIC_BASE_URL}'
          request_args:
            model: anthropic.claude-opus-4-8
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from nexus.core import (
    ContextLengthExceededError,
    OpenAICompatibleClient,
    OpenAIConfig,
    openai_client_registry,
    resolve_effective_request_args,
)
from nexus.core.types import ExtendedChatCompletion
from nexus.extensions.openai_clients.gateway_routing import merge_gateway_routing
from nexus.extensions.openai_clients.vertex import (
    VertexAnthropicHTTPError,
    _attach_provider_call,
    _attempt_error,
    _build_payload,
    _convert_response,
    _default_max_tokens_for_model,
    _error_status,
    _is_anthropic_retryable,
    _is_context_length_error,
    _normalize_output_config,
    _normalize_thinking,
    _provider_value,
    _read_stream_response,
    _set_final_error,
    apply_cache_control_claude_code,
    apply_tools_cache_control,
)
from nexus.utils.provider_io import redact_provider_io


logger = logging.getLogger(__name__)

_ANTHROPIC_VERSION = '2023-06-01'
_BEDROCK_ANTHROPIC_VERSION = 'bedrock-2023-05-31'
_DEFAULT_BASE_URL = 'https://api.anthropic.com'
_DEFAULT_TIMEOUT = 900.0
_DEFAULT_MAX_RETRIES = 3
_MESSAGES_STYLE = 'messages'
_BEDROCK_STYLE = 'bedrock_invoke'
_API_STYLES = (_MESSAGES_STYLE, _BEDROCK_STYLE)
_AUTH_STYLES = ('bearer', 'x-api-key')

# The gateway resolves the upstream account from the api_key's ?provider=, so the
# default has to follow the wire format we are about to speak.
_DEFAULT_PROVIDER_BY_STYLE = {_MESSAGES_STYLE: 'anthropic', _BEDROCK_STYLE: 'aws_third'}

_BETA_HEADER = 'anthropic-beta'
# Auth headers are owned by _build_headers and must never be overridden from
# extra_headers, which would strip the gateway routing off the credential.
_AUTH_HEADER_KEYS = frozenset({'authorization', 'x-api-key'})


class AnthropicMessagesRequestError(RuntimeError):
    """Sanitized failure surface: carries the status, never the upstream body."""

    def __init__(self, status: int | None) -> None:
        self.status = status
        suffix = f' (status {status})' if status is not None else ''
        super().__init__(f'Anthropic Messages request failed{suffix}')


def _normalize_base_url(base_url: Any) -> str:
    """Strip a trailing ``/v1`` so callers can pass either form.

    The platform sends ``http://host/v1`` while a local ``.env`` usually holds the
    bare origin. Both must end up addressing ``{origin}/v1/messages`` exactly once.
    """

    value = str(base_url or '').rstrip('/')
    if not value:
        return ''
    try:
        path = urlsplit(value).path
    except ValueError:
        path = value
    if path.rstrip('/').endswith('/v1'):
        return value[: -len('/v1')].rstrip('/')
    return value


def _normalize_betas(betas: Any) -> list[str]:
    if betas in (None, ''):
        return []
    if isinstance(betas, str):
        values = betas.split(',')
    elif isinstance(betas, Sequence):
        values = [str(item) for item in betas]
    else:
        raise TypeError('anthropic_messages client_args.betas must be a string or a sequence of strings')
    return [value.strip() for value in values if str(value).strip()]


def _merge_beta_values(*groups: Any) -> list[str]:
    """Union the beta flags, preserving first-seen order (duplicates rejected upstream)."""

    merged: list[str] = []
    for group in groups:
        for value in _normalize_betas(group):
            if value not in merged:
                merged.append(value)
    return merged


def _merge_native_request_args(payload: dict[str, Any], request_args: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay host-side request_args onto an agent-authored Anthropic body.

    The body wins for anything the agent expressed itself -- that is the whole
    point of taking it natively -- so configured values only fill gaps. The
    exceptions are the fields the host is authoritative for: ``model`` (chosen by
    host config, the agent only names what it wants) and the sampling/effort knobs
    an operator pins in ``request_args`` to make a run reproducible.
    """
    merged = dict(payload)
    for key in ('thinking', 'output_config', 'temperature', 'top_p', 'top_k', 'metadata'):
        if request_args.get(key) is not None:
            merged[key] = request_args[key]
    if merged.get('output_config') is None:
        effort = _normalize_output_config(dict(request_args))
        if effort is not None:
            merged['output_config'] = effort
        else:
            merged.pop('output_config', None)
    if merged.get('thinking') is not None:
        normalized_thinking = _normalize_thinking(merged['thinking'])
        if normalized_thinking is None:
            merged.pop('thinking', None)
        else:
            merged['thinking'] = normalized_thinking
    if merged.get('max_tokens') is None:
        merged['max_tokens'] = request_args.get('max_tokens')
    # `stream` is set by the transport, never inherited from the agent's body.
    merged.pop('stream', None)
    if request_args.get('stream'):
        merged['stream'] = True
    return merged


@openai_client_registry.register('anthropic_messages')
class AnthropicMessagesClient(OpenAICompatibleClient):
    """Claude over the Anthropic Messages API, configurable from flat client_args."""

    # This client speaks Anthropic Messages upstream, so an agent that also speaks it
    # (Claude Code via the bridge) can have its request body forwarded verbatim rather
    # than round-tripped through the OpenAI schema, which drops thinking config,
    # cache_control placement, structured system blocks and any newer field.
    native_request_protocol = 'anthropic'

    def __init__(self, config: OpenAIConfig) -> None:
        super().__init__(config)
        client_args = config.client_args

        self.api_style = str(client_args.get('api_style') or _MESSAGES_STYLE).lower()
        if self.api_style not in _API_STYLES:
            raise ValueError(f'anthropic_messages client_args.api_style must be one of {list(_API_STYLES)}')

        self.auth_style = str(client_args.get('auth_style') or 'bearer').lower()
        if self.auth_style not in _AUTH_STYLES:
            raise ValueError(f'anthropic_messages client_args.auth_style must be one of {list(_AUTH_STYLES)}')

        self._raw_api_key = str(client_args.get('api_key') or '')
        if not self._raw_api_key.strip():
            raise ValueError('anthropic_messages client_args.api_key is required')
        # The appid:appkey env chains in the example configs collapse to a bare ':'
        # when neither variable is set. That is non-empty, so it passes the check
        # above and the client starts happily, then every request 401s with nothing
        # pointing back at the unset env var. Reject it at construction instead.
        if not self._raw_api_key.strip(': '):
            raise ValueError(
                'anthropic_messages client_args.api_key resolved to an empty appid:appkey pair '
                '-- the credential env vars are unset'
            )

        # `ensure_v1_path` is accepted for config compatibility with the platform's
        # existing main_client_args blob, but normalization is unconditional here:
        # /v1/messages is the only Messages path, so there is no variant to opt into.
        self.base_url = _normalize_base_url(client_args.get('base_url') or _DEFAULT_BASE_URL)
        self.endpoint = client_args.get('endpoint')
        if not self.base_url and not self.endpoint:
            raise ValueError('anthropic_messages requires client_args.base_url or client_args.endpoint')

        self.timeout = float(client_args.get('timeout', _DEFAULT_TIMEOUT))
        self.max_retries = max(1, int(client_args.get('max_retries', _DEFAULT_MAX_RETRIES)))
        self.capture_raw_io = bool(client_args.get('capture_raw_io', False))
        self.anthropic_version = client_args.get('anthropic_version')
        self.betas = _normalize_betas(client_args.get('betas'))
        self.extra_headers = dict(client_args.get('extra_headers') or {})

        self._provider = client_args.get('provider')
        self._auto_provider = bool(client_args.get('auto_provider', True))
        self._gateway_model = client_args.get('gateway_model')
        self._gateway_timeout = client_args.get('gateway_timeout')
        self._cache_task_id = client_args.get('cache_task_id')

    # -- credential -------------------------------------------------------

    def _resolve_api_key(self, model: str) -> str:
        """Fold gateway routing into the api_key query string.

        Gateway credentials are ``credential?provider=...&model=...``, not URLs.
        An api_key that already carries a query string is treated as fully routed:
        only explicit client_args override it, so a hand-tuned key is never
        silently re-pointed at another account.
        """

        routing: dict[str, Any] = {}
        already_routed = '?' in self._raw_api_key

        provider = self._provider
        if provider in (None, '') and not already_routed and self._auto_provider:
            provider = _DEFAULT_PROVIDER_BY_STYLE[self.api_style]
        if provider not in (None, ''):
            routing['provider'] = str(provider)
            gateway_model = self._gateway_model if self._gateway_model not in (None, '') else model
            if gateway_model:
                routing['model'] = str(gateway_model)
            if self._gateway_timeout not in (None, ''):
                routing['timeout'] = self._gateway_timeout

        if self._cache_task_id not in (None, ''):
            routing['cache_task_id'] = hashlib.md5(str(self._cache_task_id).encode()).hexdigest()

        if not routing:
            return self._raw_api_key
        return merge_gateway_routing(self._raw_api_key, routing)

    # -- transport --------------------------------------------------------

    def _resolve_endpoint(self, model: str) -> str:
        if self.endpoint:
            return str(self.endpoint)
        if self.api_style == _BEDROCK_STYLE:
            return f'{self.base_url}/model/{model}/invoke'
        return f'{self.base_url}/v1/messages'

    def _build_headers(self, model: str, request_headers: Mapping[str, Any] | None) -> dict[str, str]:
        request_headers = dict(request_headers or {})
        # The agent ships its per-turn betas as extra_headers['anthropic-beta'];
        # union them with the configured ones instead of letting either side win.
        betas = _merge_beta_values(
            self.betas,
            self.extra_headers.get(_BETA_HEADER),
            request_headers.get(_BETA_HEADER),
        )

        headers: dict[str, str] = {'Content-Type': 'application/json'}
        api_key = self._resolve_api_key(model)
        if self.auth_style == 'x-api-key':
            headers['x-api-key'] = api_key
        else:
            headers['Authorization'] = f'Bearer {api_key}'

        if self.api_style == _MESSAGES_STYLE:
            headers['anthropic-version'] = str(self.anthropic_version or _ANTHROPIC_VERSION)

        for source in (self.extra_headers, request_headers):
            for key, value in source.items():
                if key == _BETA_HEADER or value is None:
                    continue
                # Never let a passed-through header clobber the auth header. The
                # resolved credential carries the gateway routing
                # (?provider=&model=&cache_task_id=); overwriting it silently drops
                # that routing and the gateway answers with an opaque
                # PlatformNoAvailableAccount 500 that looks nothing like a config bug.
                if str(key).lower() in _AUTH_HEADER_KEYS:
                    logger.warning(
                        'anthropic_messages ignoring %s from extra_headers: it would discard '
                        'the gateway-routed credential (set client_args.api_key instead)',
                        key,
                    )
                    continue
                headers[str(key)] = str(value)

        if betas:
            headers[_BETA_HEADER] = ','.join(betas)
        return headers

    async def _post_once(
        self,
        endpoint: str,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        model: str = '',
    ) -> dict[str, Any]:
        # `model` is passed separately rather than read back off the payload because on
        # the bedrock_invoke path the model lives in the URL and is popped from the
        # body -- reading payload['model'] there would label the stream with ''.
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, headers=dict(headers), json=payload) as response:
                if response.status < 400 and payload.get('stream'):
                    return await _read_stream_response(
                        response,
                        model or str(payload.get('model') or ''),
                        capture_raw_io=self.capture_raw_io,
                    )
                text = await response.text()
                if response.status >= 400:
                    raise VertexAnthropicHTTPError(response.status, text)
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise VertexAnthropicHTTPError(response.status, f'non-JSON response: {text[:500]}') from exc
                if not isinstance(data, dict):
                    raise VertexAnthropicHTTPError(
                        response.status,
                        f'unexpected response type {type(data).__name__}: {text[:500]}',
                    )
                return data

    async def _post_json(
        self,
        endpoint: str,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        provider_call: dict[str, Any],
        model: str = '',
    ) -> dict[str, Any]:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self._post_once(endpoint, headers, payload, model)
            except asyncio.CancelledError as error:
                provider_call['attempts'].append(self._attempt_record(provider_call, attempt, error))
                raise
            except Exception as error:
                provider_call['attempts'].append(self._attempt_record(provider_call, attempt, error))
                if not _is_anthropic_retryable(error) or attempt == self.max_retries:
                    raise
                await asyncio.sleep(min(2 ** (attempt - 1), 30))
                continue

            provider_call['attempts'].append({'attempt': attempt, 'status': 'success'})
            return response

        # Unreachable: max_retries is clamped to >=1, so the loop body always runs
        # and every path through it returns or raises. Kept as a guard so a future
        # change to the loop bounds can never make this silently return None.
        raise RuntimeError('anthropic_messages retry loop exhausted without a response')

    # -- tracing ----------------------------------------------------------

    def _new_provider_call(self, model: str = '', endpoint: str = '') -> dict[str, Any]:
        return {
            'provider': 'anthropic_messages',
            'protocol': 'anthropic',
            'api_style': self.api_style,
            'model': _provider_value(model),
            'endpoint': str(endpoint),
            'attempts': [],
        }

    def _attempt_record(
        self,
        provider_call: dict[str, Any],
        attempt: int,
        error: BaseException,
    ) -> dict[str, Any]:
        record = _attempt_error(attempt, error)
        captured = self._capture_error_response(provider_call, error)
        if captured is not None:
            record['response'] = captured
        return record

    def _capture_error_response(
        self,
        provider_call: dict[str, Any],
        error: BaseException,
    ) -> dict[str, Any] | None:
        if not self.capture_raw_io:
            return None
        body = getattr(error, 'body', None)
        if body is None and isinstance(error, VertexAnthropicHTTPError):
            body = error.message
        if body is None:
            return None
        response: dict[str, Any] = {'body': _provider_value(body)}
        status = _error_status(error)
        if status is not None:
            response['status_code'] = status
        provider_call['response'] = response
        return response

    def _record_failure(self, provider_call: dict[str, Any], error: BaseException) -> None:
        _set_final_error(provider_call, error)
        self._record_provider_call(provider_call)
        # AnthropicMessagesRequestError carries only the status, so the upstream body
        # -- which says WHY it was a 400 -- would otherwise never reach the run log
        # and a sandbox failure would be undiagnosable after the fact.
        # Set NEXUS_ANTHROPIC_LOG_ERROR_BODY=0 to suppress (mirrors vertex's gate).
        if os.environ.get('NEXUS_ANTHROPIC_LOG_ERROR_BODY', '1') == '0':
            return
        body = getattr(error, 'body', None) or getattr(error, 'message', None)
        if body is not None:
            text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, default=str)
            # Gateways echo the offending request -- Authorization header included --
            # back in 4xx bodies, and this log line is NOT covered by the
            # provider_call redaction pass. Scrub it against the credential before
            # writing, or the api_key lands in plaintext in every run log.
            safe = redact_provider_io(text, sensitive_source=self.config.client_args)
            logger.warning(
                'anthropic_messages upstream error detail (status=%s): %s',
                _error_status(error),
                str(safe)[:4000],
            )

    # -- request ----------------------------------------------------------

    def _prepare_native(
        self,
        native_request: Any,
        model: Any,
        request_args: dict[str, Any],
        request_headers: dict[str, Any],
    ) -> tuple[str, dict[str, Any], dict[str, str], str]:
        """Build the upstream request from a caller-supplied Anthropic body.

        The body already IS the protocol, so there is no conversion step: we only
        apply what the host is authoritative for (model, enforced/pinned args) and
        the transport details (api_style, endpoint, headers).
        """
        if not isinstance(native_request, Mapping):
            raise TypeError('anthropic_messages native_request must be a mapping')
        if native_request.get('protocol') != 'anthropic':
            raise ValueError("anthropic_messages native_request.protocol must be 'anthropic'")
        body = native_request.get('body')
        if not isinstance(body, Mapping):
            raise TypeError('anthropic_messages native_request.body must be a mapping')

        effective_args: dict[str, Any] = {**request_args, 'model': model or body.get('model')}
        if self.config.enforced_request_args:
            effective_args = resolve_effective_request_args(effective_args, self.config.enforced_request_args)
        final_model = str(effective_args.pop('model', '') or '')
        if not final_model:
            raise ValueError('anthropic_messages requires a model in request_args or create_completion(model=...)')

        payload = _merge_native_request_args(dict(body), effective_args)
        if payload.get('max_tokens') is None:
            payload['max_tokens'] = _default_max_tokens_for_model(final_model)

        # cache_control is NOT re-applied here. The agent placed its own breakpoints
        # in this body; a second pass would add another and Anthropic caps them at 4.
        if self.api_style == _MESSAGES_STYLE:
            payload['model'] = final_model
            payload.pop('anthropic_version', None)
        else:
            payload.pop('model', None)
            payload['anthropic_version'] = (
                effective_args.get('anthropic_version') or self.anthropic_version or _BEDROCK_ANTHROPIC_VERSION
            )

        headers = self._build_headers(final_model, request_headers)
        return final_model, payload, headers, self._resolve_endpoint(final_model)

    def _build_request_payload(self, *, messages, request_args):
        return _build_payload(messages=messages, request_args=request_args)

    def _prepare(
        self,
        messages: list[dict[str, Any]],
        model: str | None,
        kwargs: dict[str, Any],
    ) -> tuple[str, dict[str, Any], dict[str, str], str]:
        native_request = kwargs.pop('native_request', None)
        request_args_override = kwargs.pop('_request_args_override', None)
        configured_args = request_args_override if request_args_override is not None else self.config.request_args
        # extra_headers can arrive from EITHER side and both are load-bearing:
        # hy_unify_eval injects SessionID/StrategyType into request_args for
        # consistent-hash LB routing, while cua_anthropic_agent passes its
        # anthropic-beta flags per call. Take them from both before the dict merge
        # below collapses one onto the other, because _build_payload ignores
        # unknown keys -- dropped headers would raise no error at all.
        request_headers = {
            **(configured_args.get('extra_headers') or {}),
            **(kwargs.get('extra_headers') or {}),
        }
        request_args = {**configured_args, **kwargs}
        request_args.pop('extra_headers', None)
        configured_model = request_args.pop('model', None)

        if native_request is not None:
            return self._prepare_native(native_request, model or configured_model, request_args, request_headers)

        normalized_messages = [dict(message) for message in messages]
        if self.config.client_args.get('cache_control_enabled'):
            normalized_messages = apply_cache_control_claude_code(normalized_messages)
            if isinstance(request_args.get('tools'), list):
                request_args['tools'] = apply_tools_cache_control(request_args['tools'])

        effective_request: dict[str, Any] = {
            **request_args,
            'model': model or configured_model,
            'messages': normalized_messages,
        }
        if self.config.enforced_request_args:
            effective_request = resolve_effective_request_args(
                effective_request,
                self.config.enforced_request_args,
            )
        final_model = str(effective_request.pop('model', '') or '')
        normalized_messages = effective_request.pop('messages')
        if not final_model:
            raise ValueError('anthropic_messages requires a model in request_args or create_completion(model=...)')

        effective_args: dict[str, Any] = {**effective_request, 'model': final_model}
        if effective_args.get('max_tokens') is None:
            effective_args['max_tokens'] = _default_max_tokens_for_model(final_model)
        if self.api_style == _BEDROCK_STYLE and not effective_args.get('anthropic_version'):
            effective_args['anthropic_version'] = self.anthropic_version or _BEDROCK_ANTHROPIC_VERSION

        payload = self._build_request_payload(messages=normalized_messages, request_args=effective_args)
        if self.api_style == _MESSAGES_STYLE:
            # The Messages API takes the model in the body and rejects the
            # Vertex/Bedrock-only `anthropic_version` field that _build_payload emits.
            payload['model'] = final_model
            payload.pop('anthropic_version', None)
        else:
            # Bedrock InvokeModel names the model in the URL, not the body.
            payload.pop('model', None)

        headers = self._build_headers(final_model, request_headers)
        return final_model, payload, headers, self._resolve_endpoint(final_model)

    async def _create_completion(self, messages, model=None, **kwargs) -> ExtendedChatCompletion:
        provider_call = self._new_provider_call()
        try:
            final_model, payload, headers, endpoint = self._prepare(list(messages), model, kwargs)
            provider_call['model'] = final_model
            provider_call['endpoint'] = endpoint
            # The redaction pass scrubs any string it finds in the sensitive source
            # out of the whole trace. The payload alone is not enough: our credential
            # travels in the Authorization/x-api-key header, and gateways routinely
            # echo the offending request -- headers included -- into a 4xx body that
            # then lands in provider_call['attempts'] / ['response'] / ['error'].
            # Declaring the headers here is what keeps it out of the recorded trace.
            provider_call['_provider_sensitive_source'] = _provider_value({
                'payload': payload,
                'headers': dict(headers),
            })
            if self.capture_raw_io:
                provider_call['request'] = _provider_value(payload)
        except asyncio.CancelledError as error:
            self._record_failure(provider_call, error)
            raise
        except Exception as error:
            self._record_failure(provider_call, error)
            raise AnthropicMessagesRequestError(_error_status(error)) from error

        try:
            response = await self._post_json(endpoint, headers, payload, provider_call, final_model)
            if self.capture_raw_io:
                provider_call['response'] = _provider_value(response)
            completion = _convert_response(response, final_model)
        except asyncio.CancelledError as error:
            self._record_failure(provider_call, error)
            raise
        except Exception as error:
            self._record_failure(provider_call, error)
            if _is_context_length_error(error):
                raise ContextLengthExceededError('Anthropic Messages context length exceeded') from error
            raise AnthropicMessagesRequestError(_error_status(error)) from error
        return _attach_provider_call(completion, provider_call, self._record_provider_call)


__all__ = [
    'AnthropicMessagesClient',
    'AnthropicMessagesRequestError',
]
