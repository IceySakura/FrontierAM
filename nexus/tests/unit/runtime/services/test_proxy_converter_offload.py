"""Verify the LLM proxy servers offload their synchronous, CPU-bound converters
off the event loop thread.

The converters (`anthropic<->openai`, `openai<->responses`, `gemini<->openai`)
are pure-Python and CPU-bound; a large payload can hold the proxy event loop for
hundreds of milliseconds, which in a shared interactive Manager has previously
starved heartbeats/reporting hard enough to precede a coredump
(see docs/developing/case-webarena-glm-assignment-lost-coredump.md).

These tests pin the guarantee that each converter runs on a worker thread rather
than inline on the loop thread, by asserting the thread identity observed inside
the converter differs from the event loop's running thread.
"""

from __future__ import annotations

import asyncio
import threading

from typing import Any

import httpx
import pytest

from nexus.runtime.server.services.anthropic_proxy.server import AnthropicProxy
from nexus.runtime.server.services.gemini_proxy.server import GeminiProxy
from nexus.runtime.server.services.responses_proxy.server import ResponsesProxy


class _StaticBridge:
    """Minimal LLM bridge returning a canned OpenAI chat.completion response."""

    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    async def request(self, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        del params, timeout
        return self._response


def _openai_response(content: str = 'hello') -> dict[str, Any]:
    return {
        'id': 'chatcmpl-offload',
        'object': 'chat.completion',
        'created': 1,
        'model': 'openai-offload',
        'choices': [
            {
                'index': 0,
                'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': content},
            }
        ],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    }


async def _loop_thread_id() -> int:
    return threading.get_ident()


async def test_anthropic_proxy_runs_converters_off_loop_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    loop_thread_id = await _loop_thread_id()
    observed: dict[str, int] = {}

    import nexus.runtime.server.services.anthropic_proxy.server as server_mod

    real_to_openai = server_mod.anthropic_to_openai
    real_to_anthropic = server_mod.openai_to_anthropic

    def spy_to_openai(*args: Any, **kwargs: Any):
        observed['request'] = threading.get_ident()
        return real_to_openai(*args, **kwargs)

    def spy_to_anthropic(*args: Any, **kwargs: Any):
        observed['response'] = threading.get_ident()
        return real_to_anthropic(*args, **kwargs)

    monkeypatch.setattr(server_mod, 'anthropic_to_openai', spy_to_openai)
    monkeypatch.setattr(server_mod, 'openai_to_anthropic', spy_to_anthropic)

    proxy = AnthropicProxy(sio=object(), thinking_config='hunyuan')  # type: ignore[arg-type]
    proxy.bridge = _StaticBridge(_openai_response())  # type: ignore[assignment]
    transport = httpx.ASGITransport(app=proxy.create_app())

    async with httpx.AsyncClient(transport=transport, base_url='http://anthropic-proxy.test') as client:
        response = await client.post(
            '/v1/messages',
            json={
                'model': 'claude-offload',
                'max_tokens': 32,
                'messages': [{'role': 'user', 'content': 'hello'}],
            },
        )

    assert response.status_code == 200
    assert observed['request'] != loop_thread_id
    assert observed['response'] != loop_thread_id


async def test_responses_proxy_runs_converters_off_loop_thread() -> None:
    loop_thread_id = await _loop_thread_id()
    observed: dict[str, int] = {}

    def request_converter(
        request_body: dict[str, Any],
        *,
        collect_custom_tools: set[str],
        collect_tool_name_mappings: dict[str, Any],
    ) -> dict[str, Any]:
        del request_body, collect_custom_tools, collect_tool_name_mappings
        observed['request'] = threading.get_ident()
        return {'model': 'openai-offload', 'messages': [], 'tools': []}

    def events_converter(
        openai_response: dict[str, Any],
        *,
        custom_tool_names: set[str],
        tool_name_mappings: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del openai_response, custom_tool_names, tool_name_mappings
        observed['response'] = threading.get_ident()
        return [{'type': 'response.completed'}]

    proxy = ResponsesProxy(
        sio=object(),  # type: ignore[arg-type]
        responses_to_openai_converter=request_converter,
        openai_to_responses_events_converter=events_converter,
    )
    proxy.bridge = _StaticBridge(_openai_response())  # type: ignore[assignment]
    # ResponsesProxy 暴露的是公开的 create_app()（本分支重命名，见
    # responses_proxy/server.py:110）；GeminiProxy/AnthropicProxy 仍是 _create_app，
    # 所以本文件另两个用例保持原样。
    transport = httpx.ASGITransport(app=proxy.create_app())

    async with httpx.AsyncClient(transport=transport, base_url='http://responses-proxy.test') as client:
        response = await client.post('/v1/responses', json={'model': 'codex-offload', 'input': 'hello'})

    assert response.status_code == 200
    assert observed['request'] != loop_thread_id
    assert observed['response'] != loop_thread_id


async def test_gemini_proxy_runs_converters_off_loop_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    loop_thread_id = await _loop_thread_id()
    observed: dict[str, int] = {}

    import nexus.runtime.server.services.gemini_proxy.server as server_mod

    def spy_request(path: str, body: dict[str, Any]) -> dict[str, Any]:
        del path, body
        observed['request'] = threading.get_ident()
        return {'model': 'openai-offload', 'messages': []}

    def spy_response(openai_response: dict[str, Any], *, model: str | None = None) -> dict[str, Any]:
        del openai_response, model
        observed['response'] = threading.get_ident()
        return {'candidates': []}

    monkeypatch.setattr(server_mod, 'gemini_request_to_openai', spy_request)
    monkeypatch.setattr(server_mod, 'openai_response_to_gemini', spy_response)

    proxy = GeminiProxy(sio=object())  # type: ignore[arg-type]
    proxy.bridge = _StaticBridge(_openai_response())  # type: ignore[assignment]
    transport = httpx.ASGITransport(app=proxy._create_app())

    async with httpx.AsyncClient(transport=transport, base_url='http://gemini-proxy.test') as client:
        response = await client.post(
            '/v1/publishers/google/models/gemini-offload:generateContent',
            json={'contents': [{'role': 'user', 'parts': [{'text': 'hello'}]}]},
        )

    assert response.status_code == 200
    assert observed['request'] != loop_thread_id
    assert observed['response'] != loop_thread_id


def test_to_thread_actually_uses_a_worker_thread() -> None:
    """Sanity guard: asyncio.to_thread must run its callable off the loop thread.

    Protects the tests above from silently passing if the runtime ever changed
    to_thread semantics.
    """

    async def _run() -> tuple[int, int]:
        loop_tid = threading.get_ident()

        def _inner() -> int:
            return threading.get_ident()

        worker_tid = await asyncio.to_thread(_inner)
        return loop_tid, worker_tid

    loop_tid, worker_tid = asyncio.run(_run())
    assert loop_tid != worker_tid
