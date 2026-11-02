from __future__ import annotations

import pytest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from nexus.core.context import ExecutionContext
from nexus.core.tool import ToolContext, ToolExecutor, ToolMeta, ToolResponse, ToolType


class _Input(BaseModel):
    value: str


def _local_tool():
    async def run(params: _Input, _context: ToolContext) -> ToolResponse:
        return ToolResponse(output=[{'type': 'text', 'text': params.value}])

    run.__tool_input_model__ = _Input
    run.__tool_meta__ = ToolMeta(
        name='echo',
        description='Echo a value',
        tool_type=ToolType.LOCAL,
        input_model=_Input,
        func=run,
    )
    run.nexus_meta = {'type': 'tool', 'name': 'echo'}
    return run


@pytest.mark.asyncio
async def test_tool_executor_does_not_create_extra_otel_tool_call_span(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, 'get_tracer', provider.get_tracer)
    tracer = provider.get_tracer('tests.tool.executor')

    try:
        executor = ToolExecutor([_local_tool()])
        context = ExecutionContext(openai_clients={})
        with tracer.start_as_current_span('agent.observation'):
            async with context:
                response = await executor.call('echo', {'value': 'secret-value'})

        assert response.output[0]['text'] == 'secret-value'
        assert not any(span.name == 'tool.call' for span in exporter.get_finished_spans())
    finally:
        provider.shutdown()
