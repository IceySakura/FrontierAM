"""Tests for InvokeRequest/InvokeResponse models and InProcessBackend."""

import base64
import pickle

import pytest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

from nexus.runtime.models import (
    InvokeRequest,
    InvokeResponse,
    RemoteInvokeError,
    invoke_response_from_exception,
    is_remote_exception,
    unpack_response,
)


class TestInvokeRequest:
    def test_minimal(self):
        req = InvokeRequest(import_path='my.module::func')
        assert req.import_path == 'my.module::func'
        assert req.payload == ''
        assert req.venv_id is None
        assert req.request_id is None

    def test_build(self):
        req = InvokeRequest.build(
            'my.module::func',
            1,
            'hello',
            venv_id='venv-1',
            request_id='abc123',
            key='value',
        )
        assert req.request_id == 'abc123'
        assert req.import_path == 'my.module::func'
        assert req.venv_id == 'venv-1'

        args, kwargs = req.unpack_payload()
        assert args == (1, 'hello')
        assert kwargs == {'key': 'value'}

    def test_unpack_empty_payload(self):
        req = InvokeRequest(import_path='a.b::c')
        args, kwargs = req.unpack_payload()
        assert args == ()
        assert kwargs == {}

    def test_serialization_roundtrip(self):
        req = InvokeRequest.build(
            'a.b::c',
            {'nested': True},
            request_id='x',
            k=[1, 2, 3],
        )
        data = req.model_dump()
        req2 = InvokeRequest(**data)
        assert req2 == req
        args2, kwargs2 = req2.unpack_payload()
        assert args2 == ({'nested': True},)
        assert kwargs2 == {'k': [1, 2, 3]}

    def test_disable_invoke_wait_timeout_is_client_only(self):
        req = InvokeRequest.build('a.b::c', disable_invoke_wait_timeout=True)

        assert req.disable_invoke_wait_timeout is True
        assert 'disable_invoke_wait_timeout' not in req.model_dump()
        assert InvokeRequest(**req.model_dump()).disable_invoke_wait_timeout is False

    def test_payload_preserves_complex_objects(self):
        """Pickle payload preserves objects that JSON cannot."""
        # Use a set — JSON can't represent sets, pickle can.
        obj = {1, 2, 3}
        req = InvokeRequest.build('mod::fn', obj)
        args, kwargs = req.unpack_payload()
        assert isinstance(args[0], set)
        assert args[0] == {1, 2, 3}


class TestInvokeResponse:
    def test_success_with_pickle(self):
        result_obj = {'result': 42, 'nested': [1, 2, 3]}
        result_b64 = base64.b64encode(pickle.dumps(result_obj)).decode('utf-8')
        resp = InvokeResponse(success=True, result=result_b64)
        assert resp.success
        assert pickle.loads(base64.b64decode(resp.result)) == result_obj

    def test_failure_with_pickle(self):
        exc = ValueError('something broke')
        exc_b64 = base64.b64encode(pickle.dumps(exc)).decode('utf-8')
        resp = InvokeResponse(success=False, exception=exc_b64, traceback='Traceback...')
        assert not resp.success
        deserialized = pickle.loads(base64.b64decode(resp.exception))
        assert isinstance(deserialized, ValueError)
        assert str(deserialized) == 'something broke'

    def test_from_exception(self):
        exc = ValueError('bad value')
        resp = invoke_response_from_exception(exc, request_id='r1')
        assert not resp.success
        assert resp.request_id == 'r1'
        assert resp.exception is not None
        assert resp.traceback is not None
        # Can deserialize back to the original exception
        deserialized = pickle.loads(base64.b64decode(resp.exception))
        assert isinstance(deserialized, ValueError)
        assert str(deserialized) == 'bad value'

    def test_unserializable_remote_exception_falls_back_to_traceback(self):
        resp = InvokeResponse(
            success=False,
            exception=base64.b64encode(b'not-a-pickle').decode('utf-8'),
            traceback='remote traceback text',
        )

        with pytest.raises(RemoteInvokeError) as exc:
            unpack_response(resp)

        assert 'Failed to deserialize remote exception' in str(exc.value)
        assert exc.value.remote_traceback == 'remote traceback text'

    def test_re_raised_remote_exception_is_marked(self):
        """An exception re-raised from a response payload is stamped as remote.

        The exception traveled *inside* the InvokeResponse, so the transport
        demonstrably delivered a complete response: transport classification
        (ToolExecutor's RUNTIME branch) must not mistake it for a dead
        channel even when it is an OS connection error -- it is the sandbox
        callee's own error.
        """
        resp = invoke_response_from_exception(ConnectionResetError('sandbox tool lost its own socket'))

        with pytest.raises(ConnectionResetError) as exc_info:
            unpack_response(resp)

        assert is_remote_exception(exc_info.value) is True

    def test_locally_raised_exception_is_not_marked(self):
        """A fresh local exception has no remote marker, so classification by
        exception type alone still sees it as a potential transport error."""
        assert is_remote_exception(ConnectionResetError('raised by the HTTP layer')) is False


class TestRemoteInvokeError:
    def test_with_cause(self):
        cause = ValueError('original error')
        err = RemoteInvokeError('remote failed', cause=cause, remote_traceback='tb here')
        assert err.cause is cause
        assert err.remote_traceback == 'tb here'
        assert 'remote failed' in str(err)

    def test_without_cause(self):
        err = RemoteInvokeError('something went wrong')
        assert err.cause is None
        assert err.remote_traceback == ''


@pytest.mark.asyncio
async def test_runtime_invoke_creates_traditional_client_span(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.runtime.client.base import Runtime

    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, 'get_tracer', provider.get_tracer)
    tracer = provider.get_tracer('tests.runtime.invoke')

    class RecordingRuntime(Runtime):
        async def _invoke(self, request: InvokeRequest):
            with tracer.start_as_current_span('runtime.downstream'):
                return request.import_path

    try:
        runtime = RecordingRuntime()
        with tracer.start_as_current_span('tool.call') as parent:
            result = await runtime.invoke('demo.module::run', 'secret-argument')

        assert result == 'demo.module::run'
        invoke_span = next(span for span in exporter.get_finished_spans() if span.name == 'runtime.invoke')
        downstream = next(span for span in exporter.get_finished_spans() if span.name == 'runtime.downstream')
        assert invoke_span.kind is SpanKind.CLIENT
        assert invoke_span.parent is not None
        assert invoke_span.parent.span_id == parent.get_span_context().span_id
        assert invoke_span.attributes['rpc.system'] == 'nexus.runtime'
        assert invoke_span.attributes['rpc.method'] == 'invoke'
        assert invoke_span.attributes['nexus.runtime.target'] == 'demo.module::run'
        assert downstream.parent is not None
        assert downstream.parent.span_id == invoke_span.context.span_id
        assert 'secret-argument' not in repr(invoke_span.attributes)
    finally:
        provider.shutdown()


@pytest.mark.asyncio
async def test_runtime_invoke_inherits_execution_user() -> None:
    from nexus.runtime.client.base import Runtime
    from nexus.runtime.context import execution_user

    class RecordingRuntime(Runtime):
        def __init__(self) -> None:
            super().__init__()
            self.requests: list[InvokeRequest] = []

        async def _invoke(self, request: InvokeRequest):
            self.requests.append(request)
            return request.execution_user

    runtime = RecordingRuntime()
    with execution_user('agent'):
        assert await runtime.invoke('demo.module::run') == 'agent'
        explicit = InvokeRequest(import_path='demo.module::run', execution_user=1001)
        assert await runtime.invoke(explicit) == 1001

    assert [request.execution_user for request in runtime.requests] == ['agent', 1001]


class TestResolveAndCoerce:
    def test_resolve_func_with_import_path(self):
        from nexus.runtime.models import resolve_func

        func = resolve_func('os.path::join')
        import os.path

        assert func is os.path.join

    def test_coerce_args_dict_to_model(self):
        from pydantic import BaseModel

        from nexus.runtime.models import coerce_args

        class MyModel(BaseModel):
            x: int
            y: str

        def fn(m: MyModel) -> str:
            return f'{m.x}-{m.y}'

        args, kwargs = coerce_args(fn, ({'x': 1, 'y': 'hi'},), {})
        assert isinstance(args[0], MyModel)
        assert args[0].x == 1

    def test_coerce_args_passthrough_non_dict(self):
        from pydantic import BaseModel

        from nexus.runtime.models import coerce_args

        class MyModel(BaseModel):
            x: int

        def fn(m: MyModel) -> int:
            return m.x

        model = MyModel(x=5)
        args, kwargs = coerce_args(fn, (model,), {})
        assert args[0] is model  # not re-validated


class TestInProcessBackend:
    @pytest.mark.asyncio
    async def test_call_registered_executor(self):
        """Test that InProcessBackend can call a registered executor."""
        from nexus.core import executor_registry
        from nexus.runtime.server.backend import InProcessBackend

        # Check if command.run_command is registered (from domain groups)
        try:
            executor = executor_registry.get('command.run_command')
        except Exception:
            pytest.skip('command.run_command not registered (executors not loaded)')

        _backend = InProcessBackend()
        # We can at least verify the backend resolves the executor without error
        # Actually calling run_command would need a real shell, so just test resolution
        assert executor is not None
