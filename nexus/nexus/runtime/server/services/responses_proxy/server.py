"""Responses API proxy server for Codex."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nexus.runtime.server.services.anthropic_proxy.server import LLMBridge
from nexus.runtime.server.services.codex_compaction import CODEX_TURN_METADATA_HEADER, classify_codex_compaction
from nexus.runtime.server.services.rsio_llm_bridge import (
    NEXUS_AGENT_COMPACTION_FIELD,
    NEXUS_COMPACTION_DETECTOR_FIELD,
    bridge_error_payload,
)
from nexus.utils.event_loop import create_detached_task

from .codex import build_codex_request_meta
from .converter import (
    ResponsesProxyConversionError,
    ToolIdentity,
    normalize_exec_command_output_metadata,
    openai_to_responses_events,
    responses_to_openai,
)


if TYPE_CHECKING:
    from nexus.runtime.context import SIOBridge


logger = logging.getLogger(__name__)


ResponsesToOpenAIConverter = Callable[..., dict[str, Any]]
OpenAIToResponsesEventsConverter = Callable[..., list[dict[str, Any]]]


_COMPACT_REQUEST_FIELDS = frozenset({
    'input',
    'instructions',
    'model',
    'previous_response_id',
    'prompt_cache_key',
    'prompt_cache_options',
    'prompt_cache_retention',
    'service_tier',
})


def _extract_native_compaction_request_body(request_body: dict[str, Any]) -> dict[str, Any] | None:
    """Convert Codex's trailing compaction trigger into a native compact request."""
    items = request_body.get('input')
    if not isinstance(items, list) or not items:
        return None
    trigger_indexes = [
        index for index, item in enumerate(items) if isinstance(item, dict) and item.get('type') == 'compaction_trigger'
    ]
    if trigger_indexes != [len(items) - 1]:
        return None
    compact_body = {key: value for key, value in request_body.items() if key in _COMPACT_REQUEST_FIELDS}
    compact_body['input'] = items[:-1]
    return compact_body


def _native_compaction_events(compacted: dict[str, Any]) -> list[dict[str, Any]]:
    """Render a native compact response as the SSE sequence Codex expects."""
    response_id = compacted.get('id') or f'resp_{uuid.uuid4().hex}'
    events: list[dict[str, Any]] = [{'type': 'response.created', 'response': {'id': response_id}}]
    output = compacted.get('output')
    for item in output if isinstance(output, list) else []:
        if isinstance(item, dict):
            events.append({'type': 'response.output_item.done', 'item': item})
    events.append({
        'type': 'response.completed',
        'response': {
            'id': response_id,
            'usage': compacted.get('usage'),
        },
    })
    return events


def _uses_native_codex_catalog(version: str | None) -> bool:
    """Return whether this Codex version expects the native model-catalog shape."""
    if not version:
        return False

    numeric_version = version.split('-', maxsplit=1)[0]
    try:
        parts = tuple(int(part) for part in numeric_version.split('.'))
    except ValueError:
        return False
    if len(parts) < 2:
        return False

    return parts[:2] >= (0, 146)


def _format_sse_event(payload: dict[str, Any]) -> str:
    event_type = payload.get('type', 'message')
    body = json.dumps(payload, ensure_ascii=False)
    return f'event: {event_type}\ndata: {body}\n\n'


class ResponsesProxy:
    """Proxy Codex Responses API calls to the host-side OpenAI chat client."""

    def __init__(
        self,
        sio: SIOBridge,
        listen_port: int = 8083,
        listen_host: str = '127.0.0.1',
        request_timeout: float | None = 300,
        responses_to_openai_converter: ResponsesToOpenAIConverter = responses_to_openai,
        openai_to_responses_events_converter: OpenAIToResponsesEventsConverter = openai_to_responses_events,
        codex_version: str | None = None,
        compact_prompt: str | None = None,
        normalize_exec_command_metadata: bool = False,
    ):
        self.sio = sio
        self.port = listen_port
        self.host = listen_host
        self.request_timeout = request_timeout
        self.responses_to_openai_converter = responses_to_openai_converter
        self.openai_to_responses_events_converter = openai_to_responses_events_converter
        self.codex_version = codex_version
        self.compact_prompt = compact_prompt
        self.normalize_exec_command_metadata = normalize_exec_command_metadata
        self._pending_compaction_resumes: set[tuple[str | None, str | None]] = set()

        self.bridge: LLMBridge | None = None
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task | None = None

    def create_app(self) -> FastAPI:
        """Create the FastAPI application and initialize its LLM bridge."""
        if self.bridge is None:
            self.bridge = LLMBridge(self.sio)
        return self._create_app()

    def _create_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            logger.debug(f'Responses Proxy starting on {self.host}:{self.port}')
            yield
            logger.debug('Responses Proxy shutting down')

        app = FastAPI(
            title='Responses API Proxy',
            description='Proxy Codex Responses API calls to OpenAI chat completions',
            lifespan=lifespan,
        )

        @app.get('/')
        async def health():
            return {
                'status': 'ok',
                'bridge_ready': self.bridge is not None,
            }

        @app.get('/v1/models')
        async def models():
            if _uses_native_codex_catalog(self.codex_version):
                # Codex >=0.146 uses its native catalog contract. An empty remote catalog keeps
                # the bundled model definitions (including ``use_responses_lite``) authoritative.
                return {'models': []}
            return {'object': 'list', 'data': []}

        @app.post('/v1/responses')
        async def responses(request: Request):
            if self.bridge is None:
                raise HTTPException(status_code=503, detail='LLM bridge not initialized')

            try:
                request_body = await request.json()
                compact_body = _extract_native_compaction_request_body(request_body)
                if compact_body is not None:
                    compacted = await self.bridge.request(
                        {
                            '_nexus_operation': 'responses_compact',
                            'request_body': compact_body,
                        },
                        timeout=self.request_timeout,
                    )

                    async def compact_stream():
                        for event in _native_compaction_events(compacted):
                            yield _format_sse_event(event)
                            await asyncio.sleep(0)

                    return StreamingResponse(compact_stream(), media_type='text/event-stream')

                conversion_request_body = (
                    await asyncio.to_thread(normalize_exec_command_output_metadata, request_body)
                    if self.normalize_exec_command_metadata
                    else request_body
                )
                provider_request_id = str(uuid.uuid4())
                # Collect names of Responses `custom` tools (e.g. Codex freeform apply_patch) so
                # the matching function_call can be restored as a custom_tool_call on the return
                # trip. Local to this request -> per-request isolation (ResponsesProxy is shared).
                custom_tool_names: set[str] = set()
                tool_name_mappings: dict[str, ToolIdentity] = {}
                # Converters are synchronous and CPU-bound; large payloads can hold the proxy
                # event loop. Offload to a worker thread to keep the loop responsive.
                openai_params = await asyncio.to_thread(
                    self.responses_to_openai_converter,
                    conversion_request_body,
                    collect_custom_tools=custom_tool_names,
                    collect_tool_name_mappings=tool_name_mappings,
                )
                compaction_evidence = classify_codex_compaction(
                    request_body,
                    self.compact_prompt,
                    turn_metadata=request.headers.get(CODEX_TURN_METADATA_HEADER),
                )
                if compaction_evidence is not None:
                    openai_params[NEXUS_AGENT_COMPACTION_FIELD] = True
                    openai_params[NEXUS_COMPACTION_DETECTOR_FIELD] = compaction_evidence.detector
                compaction_thread = None
                is_compaction_resume = False
                if self.codex_version is not None:
                    openai_params['request_meta'] = build_codex_request_meta(
                        request_body,
                        request.headers,
                        configured_version=self.codex_version,
                        request_id=provider_request_id,
                        compaction_evidence=compaction_evidence,
                        tool_identities=tool_name_mappings,
                    )
                    request_meta = openai_params['request_meta']
                    codex_metadata = request_meta['codex']
                    session_id = codex_metadata.get('session_id')
                    thread_id = codex_metadata.get('thread_id')
                    compaction_thread = (
                        session_id if isinstance(session_id, str) else None,
                        thread_id if isinstance(thread_id, str) else None,
                    )
                    is_compaction_resume = (
                        compaction_evidence is None
                        and codex_metadata.get('request_kind') in (None, 'turn')
                        and compaction_thread in self._pending_compaction_resumes
                    )
                    if is_compaction_resume:
                        request_meta['invocation_kind'] = 'compaction_resume'
                logger.debug(
                    'Converted Responses request: model=%s messages=%s tools=%s tool_choice=%r '
                    'custom_tools=%s mapped_tools=%s',
                    openai_params.get('model'),
                    len(openai_params.get('messages') or []),
                    len(openai_params.get('tools') or []),
                    openai_params.get('tool_choice'),
                    sorted(custom_tool_names),
                    len(tool_name_mappings),
                )

                openai_response = await self.bridge.request(openai_params, timeout=self.request_timeout)
                events = await asyncio.to_thread(
                    self.openai_to_responses_events_converter,
                    openai_response,
                    custom_tool_names=custom_tool_names,
                    tool_name_mappings=tool_name_mappings,
                )
                # Only a successfully converted summary arms the next ordinary turn. Keep the
                # marker through failed resume requests, and keep separate Codex threads isolated.
                if compaction_thread is not None:
                    if compaction_evidence is not None:
                        self._pending_compaction_resumes.add(compaction_thread)
                    elif is_compaction_resume:
                        self._pending_compaction_resumes.discard(compaction_thread)

                async def stream():
                    for event in events:
                        yield _format_sse_event(event)
                        await asyncio.sleep(0)

                return StreamingResponse(stream(), media_type='text/event-stream')

            except ResponsesProxyConversionError as error:
                logger.warning('Rejected Responses request: %s', error)
                return JSONResponse(status_code=error.status_code, content=error.response_payload())
            except Exception as e:
                logger.error(f'Error processing Responses request: {e}')
                coded = bridge_error_payload(e)
                if coded is not None:
                    status_code, content = coded
                    return JSONResponse(status_code=status_code, content=content)
                raise HTTPException(status_code=500, detail=str(e)) from e

        @app.post('/v1/responses/compact')
        async def compact(request: Request):
            if self.bridge is None:
                raise HTTPException(status_code=503, detail='LLM bridge not initialized')

            try:
                request_body = await request.json()
                compacted = await self.bridge.request(
                    {
                        '_nexus_operation': 'responses_compact',
                        'request_body': request_body,
                    },
                    timeout=self.request_timeout,
                )
                return JSONResponse(content=compacted)
            except Exception as error:
                coded = bridge_error_payload(error)
                if coded is not None:
                    status_code, content = coded
                    return JSONResponse(status_code=status_code, content=content)
                raise HTTPException(status_code=500, detail=str(error)) from error

        @app.exception_handler(Exception)
        async def exception_handler(request: Request, exc: Exception):
            logger.error(f'Unhandled exception: {exc}')
            return JSONResponse(
                status_code=500,
                content={
                    'error': {
                        'type': 'internal_error',
                        'message': str(exc),
                    }
                },
            )

        return app

    async def start(self):
        self.bridge = LLMBridge(self.sio)

        app = self._create_app()
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level='warning',
            # This proxy stops with its command. Cancel outstanding ASGI/bridge
            # requests on shutdown; their active-time budget may be suspended.
            timeout_graceful_shutdown=0,
        )
        self._server = uvicorn.Server(config)
        self._server_task = create_detached_task(self._server.serve(), name='nexus-responses-proxy')

        await asyncio.sleep(0.5)
        logger.debug(f'Responses Proxy ready at http://{self.host}:{self.port}')

    async def stop(self):
        if self._server is not None:
            self._server.should_exit = True
            if self._server_task is not None:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5)
                except asyncio.TimeoutError:
                    self._server_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._server_task

        logger.debug('Responses Proxy stopped')

    @property
    def is_running(self) -> bool:
        return self._server is not None and self.bridge is not None


async def run_proxy(
    sio: SIOBridge,
    listen_port: int = 8083,
    listen_host: str = '127.0.0.1',
    responses_to_openai_converter: ResponsesToOpenAIConverter = responses_to_openai,
    openai_to_responses_events_converter: OpenAIToResponsesEventsConverter = openai_to_responses_events,
    codex_version: str | None = None,
    compact_prompt: str | None = None,
    normalize_exec_command_metadata: bool = False,
) -> ResponsesProxy:
    proxy = ResponsesProxy(
        sio=sio,
        listen_port=listen_port,
        listen_host=listen_host,
        responses_to_openai_converter=responses_to_openai_converter,
        openai_to_responses_events_converter=openai_to_responses_events_converter,
        codex_version=codex_version,
        compact_prompt=compact_prompt,
        normalize_exec_command_metadata=normalize_exec_command_metadata,
    )
    await proxy.start()
    return proxy


__all__ = [
    'OpenAIToResponsesEventsConverter',
    'ResponsesProxy',
    'ResponsesToOpenAIConverter',
    'run_proxy',
]
