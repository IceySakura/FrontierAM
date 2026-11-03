"""Authenticated loopback transport reusing Nexus's native Responses converter."""

from __future__ import annotations

import asyncio
import contextvars
import hmac
import json
import socket

from typing import Any

import uvicorn

from fastapi.responses import JSONResponse

from nexus.runtime.server.services.responses_proxy import ResponsesProxy
from nexus.runtime.server.services.responses_proxy.converter import (
    ResponsesProxyConversionError,
    openai_to_responses_events,
    responses_to_openai,
)

from .client import BoundaryError, BoundedOpenAIClient


MAX_ARGUMENT_BYTES = 900 * 1024
_ALLOWED_NAMES = {'mcp__workspace__exec'}


def _is_workspace_tool(name: str, identity: Any = None) -> bool:
    if identity is not None:
        return identity.tool_type == 'function' and (
            (identity.namespace == 'mcp__workspace' and identity.name == 'exec')
            or (identity.namespace is None and identity.name in _ALLOWED_NAMES)
        )
    return name in _ALLOWED_NAMES


def _reject_constant(_value: str) -> None:
    raise ValueError('Non-finite JSON is forbidden')


def validate_tool_response(response: dict[str, Any], allowed_names: set[str], record) -> None:
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        raise BoundaryError('choice_count')
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get('finish_reason') not in {'stop', 'tool_calls'}:
        raise BoundaryError('incomplete_response')
    message = choice.get('message')
    if not isinstance(message, dict):
        raise BoundaryError('message_structure')
    calls = message.get('tool_calls') or []
    if not isinstance(calls, list):
        raise BoundaryError('tool_structure')
    seen = set()
    for call in calls:
        if not isinstance(call, dict) or call.get('type') != 'function':
            raise BoundaryError('tool_identity')
        function = call.get('function')
        if not isinstance(function, dict) or function.get('name') not in allowed_names:
            raise BoundaryError('tool_identity')
        call_id = call.get('id')
        if call_id in (None, ''):
            record('native_normalization', category='missing_call_id')
        elif not isinstance(call_id, str) or call_id in seen:
            raise BoundaryError('call_id')
        else:
            seen.add(call_id)
        arguments = function.get('arguments')
        if isinstance(arguments, dict):
            record('native_normalization', category='object_arguments')
            parsed = arguments
        elif isinstance(arguments, str):
            if len(arguments.encode()) > MAX_ARGUMENT_BYTES:
                raise BoundaryError('argument_transport_limit')
            try:
                parsed = json.loads(arguments, parse_constant=_reject_constant)
            except (TypeError, ValueError, RecursionError):
                raise BoundaryError('argument_json') from None
        else:
            raise BoundaryError('argument_encoding')
        if not isinstance(parsed, dict):
            raise BoundaryError('argument_object')
        try:
            encoded = json.dumps(parsed, ensure_ascii=True, allow_nan=False).encode()
        except (TypeError, ValueError, RecursionError):
            raise BoundaryError('argument_json') from None
        if len(encoded) > MAX_ARGUMENT_BYTES:
            raise BoundaryError('argument_transport_limit')


class _HostBridge:
    def __init__(self, client: BoundedOpenAIClient):
        self.client = client
        # HTTP dispatch tasks do not reliably inherit the run that started
        # their listening socket. Match Nexus LLMRequestHandler's per-owner
        # context capture so native spans retain that run's processors/parent.
        self._request_context = contextvars.copy_context()

    async def request(self, params: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        del timeout  # The client's immutable task deadline is authoritative.
        task = asyncio.create_task(self._request(params), context=self._request_context.copy())
        return await task

    async def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        if params.get('_nexus_operation'):
            raise BoundaryError('compaction_disabled')
        params = dict(params)
        # Codex's local summary turn shares the run's deadline and HTTP budget.
        if params.pop('_nexus_agent_compaction', False) is True:
            self.client.record('compaction', detector=params.get('_nexus_compaction_detector'))
        params.pop('_nexus_compaction_detector', None)
        try:
            response = await self.client.create_completion(**params)
            result = response.model_dump(mode='json', warnings=False)
            allowed = {tool['function']['name'] for tool in params.get('tools', [])}
            validate_tool_response(result, allowed, self.client.record)
            self.client.check_deadline()
            return result
        except asyncio.CancelledError:
            raise
        except BoundaryError as error:
            self.client.record('error', category=error.error_code.removeprefix('frontieram_'))
            raise
        except Exception as error:
            self.client.record('error', category='proxy_protocol', exception_type=type(error).__name__)
            raise BoundaryError('proxy_protocol') from None


class OfflineResponsesProxy(ResponsesProxy):
    """Nexus converter and SSE routes with per-task local transport guards."""

    def __init__(
        self,
        client: BoundedOpenAIClient,
        *,
        token: str,
        port: int = 0,
        codex_version: str = '0.166.0-alpha.5.1',
    ):
        if not token or len(token) < 16:
            raise ValueError('A per-run loopback token of at least 16 characters is required')
        self.client = client
        self._token = token
        self._socket: socket.socket | None = None
        super().__init__(
            # Only the native routing/conversion is reused. The in-process
            # request transport below deliberately does not create a SIO server.
            sio=None,  # ty: ignore[invalid-argument-type]
            listen_host='127.0.0.1',
            listen_port=port,
            codex_version=codex_version,
            responses_to_openai_converter=self._request_converter,
            openai_to_responses_events_converter=self._response_converter,
        )
        self.bridge = _HostBridge(client)  # ty: ignore[invalid-assignment]

    @property
    def base_url(self) -> str:
        return f'http://127.0.0.1:{self.port}/v1'

    def _request_converter(self, body: dict[str, Any], **kwargs) -> dict[str, Any]:
        try:
            result = responses_to_openai(body, **kwargs)
            mappings = kwargs.get('collect_tool_name_mappings') or {}
            self.client.record(
                'advertised_tool_shapes',
                tools=[
                    {
                        'chat_name': tool.get('function', {}).get('name'),
                        'tool_type': getattr(mappings.get(tool.get('function', {}).get('name')), 'tool_type', None),
                        'namespace': getattr(mappings.get(tool.get('function', {}).get('name')), 'namespace', None),
                        'original_name': getattr(mappings.get(tool.get('function', {}).get('name')), 'name', None),
                    }
                    for tool in result.get('tools', [])
                ],
            )
            allowed_tools = []
            for tool in result.get('tools', []):
                name = tool.get('function', {}).get('name')
                if not _is_workspace_tool(name, mappings.get(name)):
                    identity = mappings.get(name)
                    self.client.record(
                        'tool_schema_filtered',
                        chat_name=name if isinstance(name, str) else None,
                        tool_type=getattr(identity, 'tool_type', None),
                        namespace=getattr(identity, 'namespace', None),
                        original_name=getattr(identity, 'name', None),
                    )
                    continue
                allowed_tools.append(tool)
            if len(allowed_tools) != 1:
                raise BoundaryError('workspace_tool_count')
            allowed_names = {tool['function']['name'] for tool in allowed_tools}
            result['tools'] = allowed_tools
            choice = result.get('tool_choice')
            if isinstance(choice, dict) and choice.get('function', {}).get('name') not in allowed_names:
                raise BoundaryError('selected_tool_identity')
            for message in result.get('messages', []):
                for call in message.get('tool_calls') or []:
                    if call.get('function', {}).get('name') not in allowed_names:
                        raise BoundaryError('history_tool_identity')
            return result
        except BoundaryError as error:
            self.client.record('error', category=error.error_code.removeprefix('frontieram_'))
            raise
        except Exception as error:
            self.client.record('error', category='request_conversion', exception_type=type(error).__name__)
            raise ResponsesProxyConversionError(
                'Invalid Responses structure', code='frontieram_request_conversion'
            ) from None

    def _response_converter(self, response: dict[str, Any], **kwargs) -> list[dict[str, Any]]:
        self.client.check_deadline()
        try:
            return openai_to_responses_events(response, **kwargs)
        except Exception as error:
            self.client.record('error', category='response_conversion', exception_type=type(error).__name__)
            raise BoundaryError('response_conversion') from None

    def _create_app(self):
        app = super()._create_app()

        @app.middleware('http')
        async def authenticate(request, call_next):
            expected = f'Bearer {self._token}'
            if not hmac.compare_digest(request.headers.get('authorization', ''), expected):
                return JSONResponse(status_code=401, content={'error': {'code': 'local_authentication_failed'}})
            if request.url.path == '/v1/responses/compact':
                return JSONResponse(status_code=400, content={'error': {'code': 'compaction_disabled'}})
            try:
                self.client.check_deadline()
            except BoundaryError:
                return JSONResponse(status_code=400, content={'error': {'code': 'development_stopped'}})
            return await call_next(request)

        return app

    async def start(self) -> None:
        self.client.check_deadline()
        self._socket = socket.socket()
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(('127.0.0.1', self.port))
        self._socket.listen(128)
        self._socket.setblocking(False)
        self.port = self._socket.getsockname()[1]
        config = uvicorn.Config(self.create_app(), log_level='error', access_log=False, timeout_graceful_shutdown=0)
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve(sockets=[self._socket]))
        for _ in range(100):
            if self._server.started:
                return
            if self._server_task.done():
                await self._server_task
                raise BoundaryError('proxy_startup')
            await asyncio.sleep(0.01)
        await self.stop()
        raise BoundaryError('proxy_startup')

    async def stop(self) -> None:
        await super().stop()
        if self._socket is not None:
            self._socket.close()
            self._socket = None
