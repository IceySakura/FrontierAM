from __future__ import annotations

import ast
import asyncio
import base64
import logging
import os
import pickle

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nexus.core.context import ExecutionContext
from nexus.runtime.client.models import VenvHandle
from nexus.runtime.providers.base import RuntimeProvider
from nexus.utils.otel.environment import is_managed_otel_env_key


class _Provider(RuntimeProvider):
    async def acquire(self, name: str, **params: Any) -> Any:
        raise NotImplementedError

    async def _release_impl(self, name: str) -> None:
        raise NotImplementedError


def _enable_otel(monkeypatch: pytest.MonkeyPatch, *, secret: str = 'otel-header-sentinel') -> None:
    for key in os.environ.copy():
        if is_managed_otel_env_key(key):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://collector.invalid:4318')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_HEADERS', f'authorization={secret}')
    monkeypatch.setenv('OTEL_PROPAGATORS', 'tracecontext')
    monkeypatch.setenv('OTEL_TRACES_SAMPLER', 'parentbased_always_on')
    monkeypatch.setenv('OTEL_RESOURCE_ATTRIBUTES', 'deployment.environment=test')


@pytest.mark.asyncio
async def test_worker_venv_adds_otel_extra_and_trusted_env_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable_otel(monkeypatch)
    wheel = tmp_path / 'nexus.whl'
    wheel.write_bytes(b'wheel')

    async def build_artifact(*_args: Any, **_kwargs: Any) -> Path:
        return wheel

    monkeypatch.setattr('nexus.runtime.client.packaging.build_artifact_cached', build_artifact)
    captured: dict[str, Any] = {}

    class Runtime:
        default_venv_id: str | None = None

        async def create_venv(self, spec: dict[str, Any]) -> VenvHandle:
            captured.update(spec)
            return VenvHandle(venv_id='venv-1', state='ready')

    runtime = Runtime()
    ctx = ExecutionContext()

    await ctx._setup_worker_venv(
        runtime,  # type: ignore[arg-type]
        {'source': str(tmp_path), 'extras': ['runtime', 'otel', 'runtime']},
    )

    assert captured['artifacts'][0]['extras'] == ['runtime', 'otel']
    assert captured['env_vars']['OTEL_SERVICE_NAME'] == 'nexus-runtime-worker'
    assert captured['env_vars']['NEXUS_OTEL_PROCESS_ROLE'] == 'runtime-worker'
    assert captured['env_vars']['OTEL_EXPORTER_OTLP_HEADERS'] == 'authorization=otel-header-sentinel'
    assert runtime.default_venv_id == 'venv-1'


@pytest.mark.asyncio
async def test_worker_venv_disabled_payload_shape_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv('NEXUS_OTEL_ENABLED', raising=False)
    wheel = tmp_path / 'nexus.whl'
    wheel.write_bytes(b'wheel')

    async def build_artifact(*_args: Any, **_kwargs: Any) -> Path:
        return wheel

    monkeypatch.setattr('nexus.runtime.client.packaging.build_artifact_cached', build_artifact)
    captured: dict[str, Any] = {}

    class Runtime:
        default_venv_id: str | None = None

        async def create_venv(self, spec: dict[str, Any]) -> VenvHandle:
            captured.update(spec)
            return VenvHandle(venv_id='venv-1', state='ready')

    await ExecutionContext()._setup_worker_venv(  # type: ignore[arg-type]
        Runtime(),
        {'source': str(tmp_path), 'extras': ['runtime', 'runtime']},
    )

    assert captured == {
        'python_version': captured['python_version'],
        'artifacts': [{'name': 'nexus', 'whl_data': 'd2hlZWw=', 'extras': ['runtime', 'runtime']}],
        'extra_requirements': [],
        'modules': None,
    }


@pytest.mark.asyncio
async def test_worker_launcher_merges_trusted_overlay_last(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable_otel(monkeypatch)
    from nexus.runtime.server.worker.launcher import WorkerLauncher

    python = tmp_path / 'bin' / 'python'
    python.parent.mkdir()
    python.touch()
    captured: dict[str, Any] = {}

    async def create_subprocess_exec(*args: str, **kwargs: Any) -> Any:
        captured.update(args=args, kwargs=kwargs)
        return SimpleNamespace(pid=1234, returncode=None, stdout=None, stderr=None)

    monkeypatch.setattr('asyncio.create_subprocess_exec', create_subprocess_exec)
    monkeypatch.setattr('nexus.utils.shell.get_clean_sandbox_env', lambda: {'BASE': 'yes'})

    await WorkerLauncher().start_worker(
        'venv-1',
        tmp_path,
        base_url='http://base.invalid',
        worker_id='worker-safe',
        env_vars={
            'USER_VALUE': 'kept',
            'OTEL_EXPORTER_OTLP_HEADERS': 'authorization=forged',
            'OTEL_SERVICE_NAME': 'forged',
            'OTEL_EXPORTER_OTLP_TRACES_HEADERS': 'authorization=absent-host-forgery',
            'OTEL_TRACES_SAMPLER_ARG': 'absent-host-forgery',
        },
    )

    env = captured['kwargs']['env']
    assert env['BASE'] == 'yes'
    assert env['USER_VALUE'] == 'kept'
    assert env['OTEL_EXPORTER_OTLP_HEADERS'] == 'authorization=otel-header-sentinel'
    assert env['OTEL_SERVICE_NAME'] == 'nexus-runtime-worker'
    assert env['NEXUS_OTEL_PROCESS_ROLE'] == 'runtime-worker'
    assert 'nexus.worker.id=worker-safe' in env['OTEL_RESOURCE_ATTRIBUTES']
    assert 'OTEL_EXPORTER_OTLP_TRACES_HEADERS' not in env
    assert 'OTEL_TRACES_SAMPLER_ARG' not in env
    assert 'close_fds' not in captured['kwargs']


@pytest.mark.asyncio
async def test_worker_launcher_disabled_preserves_exact_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv('NEXUS_OTEL_ENABLED', raising=False)
    # The launcher deliberately re-injects LD_LIBRARY_PATH (the worker is a nix binary and
    # needs nix runtime libraries). Drop it so the assertion does not depend on whether the
    # host running the tests happens to set it.
    monkeypatch.delenv('LD_LIBRARY_PATH', raising=False)
    from nexus.runtime.server.worker.launcher import WorkerLauncher

    python = tmp_path / 'bin' / 'python'
    python.parent.mkdir()
    python.touch()
    captured: dict[str, Any] = {}
    caller_env = {'USER_VALUE': 'kept', 'OTEL_SERVICE_NAME': 'user-owned'}

    async def create_subprocess_exec(*_args: str, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(pid=1234, returncode=None, stdout=None, stderr=None)

    monkeypatch.setattr('asyncio.create_subprocess_exec', create_subprocess_exec)
    monkeypatch.setattr('nexus.utils.shell.get_clean_sandbox_env', lambda: {'BASE': 'yes'})

    await WorkerLauncher().start_worker(
        'venv-1',
        tmp_path,
        base_url='http://base.invalid',
        env_vars=caller_env,
    )

    assert captured['env'] == {'BASE': 'yes', 'PYTHONSAFEPATH': '1', **caller_env}
    assert caller_env == {'USER_VALUE': 'kept', 'OTEL_SERVICE_NAME': 'user-owned'}


def test_runtime_image_installs_runtime_and_otel_extras() -> None:
    dockerfile = Path('deployment/docker/Dockerfile.runtime').read_text(encoding='utf-8')

    assert '"${wheel}[runtime,otel]"' in dockerfile
    assert '-e ".[runtime,otel]"' in dockerfile


@pytest.mark.asyncio
async def test_docker_disabled_readiness_preserves_legacy_container_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nexus.runtime.providers import docker

    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'false')
    provider = docker.DockerRuntimeProvider(image='runtime-image', docker_host='gateway.invalid')
    legacy_secret = 'legacy-readiness-sentinel'

    async def not_running(_container_name: str) -> bool:
        return False

    async def run_legacy(_cmd: str) -> str:
        return f'ordinary legacy container crash: {legacy_secret}'

    monkeypatch.setattr(docker, 'container_running', not_running)
    monkeypatch.setattr(docker, 'run_shell_cmd', run_legacy)

    with pytest.raises(RuntimeError) as exc_info:
        await provider._wait_for_container_ready('runtime-container', 'gateway.invalid')

    assert 'ordinary legacy container crash' in str(exc_info.value)
    assert legacy_secret in str(exc_info.value)


@pytest.mark.asyncio
async def test_replace_runtime_log_does_not_expose_environment_credentials(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _enable_otel(monkeypatch)
    secret = 'otel-header-sentinel'

    class Provider(_Provider):
        async def acquire(self, name: str, **params: Any) -> Any:
            return SimpleNamespace(base_url=f'http://{name}.invalid')

    caplog.set_level(logging.DEBUG, logger='nexus.core.context')
    ctx = ExecutionContext(runtime_provider=Provider())

    await ctx.replace_runtime('main', environment={'OTEL_EXPORTER_OTLP_HEADERS': secret})

    assert secret not in caplog.text


@pytest.mark.asyncio
async def test_replace_runtime_disabled_preserves_legacy_parameter_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'false')

    class Provider(_Provider):
        async def acquire(self, name: str, **params: Any) -> Any:
            return SimpleNamespace(base_url=f'http://{name}.invalid')

    caplog.set_level(logging.DEBUG, logger='nexus.core.context')

    await ExecutionContext(runtime_provider=Provider()).replace_runtime('main', user_option='legacy-value')

    assert "Acquiring new runtime 'main' with params: {'user_option': 'legacy-value'}" in caplog.text


@pytest.mark.asyncio
async def test_gongfeng_debug_logs_redact_only_exporter_headers(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from nexus.runtime.providers import gongfeng

    secret = 'otel-header-sentinel'
    _enable_otel(monkeypatch)
    captured: dict[str, Any] = {}

    class Response:
        status_code = 200
        text = '{"code": 0, "data": {}}'

        def json(self) -> dict[str, Any]:
            return {'code': 0, 'data': {}}

    async def post(_url: str, *, json: dict[str, Any], **_kwargs: Any) -> Response:
        captured['json'] = json
        return Response()

    monkeypatch.setattr(gongfeng.http_client, 'post', post)
    provider = gongfeng.GongfengRuntimeProvider(api_base_url='http://gongfeng.invalid', token='token')
    payload = {
        'image': 'image',
        'environment': {
            'OTEL_EXPORTER_OTLP_HEADERS': f'authorization={secret}',
            'NORMAL_VALUE': 'visible-value',
        },
    }
    caplog.set_level(logging.DEBUG, logger=gongfeng.__name__)

    await provider._call_api('/sandbox/v3/runtime/start', payload)

    assert captured['json'] is payload
    assert secret not in caplog.text
    assert 'visible-value' in caplog.text
    assert "'OTEL_EXPORTER_OTLP_HEADERS': '<redacted>'" in caplog.text
    assert payload['environment']['NORMAL_VALUE'] == 'visible-value'


@pytest.mark.asyncio
async def test_gongfeng_enabled_hides_backend_echo_from_logs_and_errors(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from nexus.runtime.providers import gongfeng

    _enable_otel(monkeypatch)
    secret = 'backend-echo-sentinel'

    class Response:
        status_code = 500
        text = f'ordinary diagnostic: backend echoed authorization={secret}'

    async def post(*_args: Any, **_kwargs: Any) -> Response:
        return Response()

    monkeypatch.setattr(gongfeng.http_client, 'post', post)
    provider = gongfeng.GongfengRuntimeProvider(api_base_url='http://gongfeng.invalid', token='token')
    caplog.set_level(logging.DEBUG, logger=gongfeng.__name__)

    with pytest.raises(RuntimeError) as exc_info:
        await provider._call_api(
            '/sandbox/v3/runtime/start',
            {'environment': {'OTEL_EXPORTER_OTLP_HEADERS': f'authorization={secret}'}},
        )

    assert secret not in caplog.text
    assert secret not in str(exc_info.value)
    assert '<redacted>' in caplog.text
    assert '<redacted>' in str(exc_info.value)
    assert 'ordinary diagnostic' in caplog.text
    assert 'ordinary diagnostic' in str(exc_info.value)


@pytest.mark.asyncio
async def test_gongfeng_enabled_redacts_host_config_echo_without_payload_environment(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from nexus.runtime.providers import gongfeng

    host_secret = 'host-config-sentinel'
    token = 'gongfeng-token-sentinel'
    _enable_otel(monkeypatch, secret=host_secret)

    class Response:
        status_code = 500
        text = f'ordinary control-plane diagnostic: authorization={host_secret}; token={token}'

    async def post(*_args: Any, **_kwargs: Any) -> Response:
        return Response()

    monkeypatch.setattr(gongfeng.http_client, 'post', post)
    provider = gongfeng.GongfengRuntimeProvider(api_base_url='http://gongfeng.invalid', token=token)
    caplog.set_level(logging.DEBUG, logger=gongfeng.__name__)

    with pytest.raises(RuntimeError) as exc_info:
        await provider._call_api('/sandbox/v3/runtime/get', {'runtime_id': 'runtime-1'})

    assert host_secret not in caplog.text
    assert host_secret not in str(exc_info.value)
    assert token not in caplog.text
    assert token not in str(exc_info.value)
    assert 'ordinary control-plane diagnostic' in caplog.text
    assert 'ordinary control-plane diagnostic' in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure_mode', ['invalid-json', 'api-error'])
async def test_gongfeng_enabled_hides_echo_from_json_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure_mode: str,
) -> None:
    from nexus.runtime.providers import gongfeng

    _enable_otel(monkeypatch)
    secret = f'{failure_mode}-echo-sentinel'

    class Response:
        status_code = 200
        text = f'ordinary response diagnostic: backend echoed authorization={secret}'

        def json(self) -> dict[str, Any]:
            if failure_mode == 'invalid-json':
                raise ValueError(f'ordinary parser diagnostic: echoed authorization={secret}')
            return {'code': 7, 'msg': f'ordinary API diagnostic: echoed authorization={secret}'}

    async def post(*_args: Any, **_kwargs: Any) -> Response:
        return Response()

    monkeypatch.setattr(gongfeng.http_client, 'post', post)
    provider = gongfeng.GongfengRuntimeProvider(api_base_url='http://gongfeng.invalid', token='token')
    caplog.set_level(logging.DEBUG, logger=gongfeng.__name__)

    with pytest.raises(RuntimeError) as exc_info:
        await provider._call_api(
            '/sandbox/v3/runtime/start',
            {'environment': {'OTEL_EXPORTER_OTLP_HEADERS': f'authorization={secret}'}},
        )

    assert secret not in caplog.text
    assert secret not in str(exc_info.value)
    assert 'ordinary response diagnostic' in caplog.text
    expected = 'ordinary parser diagnostic' if failure_mode == 'invalid-json' else 'ordinary API diagnostic'
    assert expected in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    if failure_mode == 'invalid-json':
        import traceback

        assert secret not in ''.join(traceback.format_exception(exc_info.value))


@pytest.mark.asyncio
async def test_gongfeng_disabled_preserves_legacy_response_body_and_request_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from nexus.runtime.providers import gongfeng

    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'false')
    secret = 'legacy-backend-echo-sentinel'
    token = 'gongfeng-token-sentinel'

    class Response:
        status_code = 500
        text = f'backend echoed {secret}; token={token}'

    async def post(*_args: Any, **_kwargs: Any) -> Response:
        return Response()

    monkeypatch.setattr(gongfeng.http_client, 'post', post)
    provider = gongfeng.GongfengRuntimeProvider(api_base_url='http://gongfeng.invalid', token=token)
    caplog.set_level(logging.DEBUG, logger=gongfeng.__name__)

    with pytest.raises(RuntimeError) as exc_info:
        await provider._call_api('/legacy', {'environment': {'USER_VALUE': secret}})

    assert secret in caplog.text
    assert secret in str(exc_info.value)
    assert token not in caplog.text
    assert token not in str(exc_info.value)


@pytest.mark.asyncio
async def test_remote_runtime_refreshes_existing_trace_context_carrier_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry.sdk.trace import TracerProvider

    from nexus.runtime.client import remote
    from nexus.runtime.models import InvokeRequest, pack_result
    from nexus.trace import RemoteTraceContext
    from nexus.utils.otel.propagation import inject_carrier

    _enable_otel(monkeypatch)
    monkeypatch.setattr(remote, 'inject_carrier', inject_carrier, raising=False)
    sent: dict[str, Any] = {}

    class Response:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return pack_result('ok', request_id='request-1').model_dump()

    async def post(_url: str, *, json: dict[str, Any], **_kwargs: Any) -> Response:
        sent.update(json)
        return Response()

    monkeypatch.setattr(remote.http_client, 'post', post)
    runtime = remote.RemoteRuntime(base_url='http://runtime.invalid', auto_connect_ws=False)
    original = RemoteTraceContext(
        trace_id='nexus-trace',
        parent_span_id='nexus-parent',
        run_id='run-1',
        run_span_id='run-span-1',
        run_kind='single',
        otel_carrier={'traceparent': '00-11111111111111111111111111111111-2222222222222222-01'},
    )
    request = InvokeRequest.build('unit::call', request_id='request-1', trace_context=original)
    tracer = TracerProvider().get_tracer('tests.runtime-host')

    with tracer.start_as_current_span('send-parent') as parent:
        assert await runtime._invoke(request) == 'ok'

    sent_context = RemoteTraceContext.model_validate(sent['trace_context'])
    assert sent_context.model_dump(exclude={'otel_carrier'}) == original.model_dump(exclude={'otel_carrier'})
    assert sent_context.otel_carrier is not None
    assert sent_context.otel_carrier['traceparent'].split('-')[2] == f'{parent.get_span_context().span_id:016x}'


@pytest.mark.asyncio
async def test_remote_runtime_keeps_existing_carrier_when_current_context_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nexus.runtime.client import remote
    from nexus.runtime.models import InvokeRequest, pack_result
    from nexus.trace import RemoteTraceContext

    _enable_otel(monkeypatch)
    monkeypatch.setattr(remote, 'inject_carrier', lambda: {}, raising=False)
    sent: dict[str, Any] = {}

    class Response:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return pack_result('ok', request_id='request-1').model_dump()

    async def post(_url: str, *, json: dict[str, Any], **_kwargs: Any) -> Response:
        sent.update(json)
        return Response()

    monkeypatch.setattr(remote.http_client, 'post', post)
    fallback = {'traceparent': '00-11111111111111111111111111111111-2222222222222222-01'}
    request = InvokeRequest.build(
        'unit::call',
        request_id='request-1',
        trace_context=RemoteTraceContext(
            trace_id='nexus-trace',
            parent_span_id='nexus-parent',
            otel_carrier=fallback,
        ),
    )

    await remote.RemoteRuntime(base_url='http://runtime.invalid', auto_connect_ws=False)._invoke(request)

    assert sent['trace_context']['otel_carrier'] == fallback


@pytest.mark.asyncio
async def test_remote_runtime_builds_payload_carrier_from_native_transport_span_without_nexus_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from nexus.runtime.client import remote
    from nexus.runtime.models import pack_result

    _enable_otel(monkeypatch)
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('tests.runtime-native-cleanup')
    monkeypatch.setattr(otel_trace, 'get_tracer', provider.get_tracer)
    sent: dict[str, Any] = {}

    class Response:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return pack_result('ok', request_id=sent['request_id']).model_dump()

    async def post(_url: str, *, json: dict[str, Any], **_kwargs: Any) -> Response:
        sent.update(json)
        return Response()

    monkeypatch.setattr(remote.http_client, 'post', post)
    runtime = remote.RemoteRuntime(base_url='http://runtime.invalid', auto_connect_ws=False)

    with tracer.start_as_current_span('cleanup-parent') as cleanup_parent:
        assert await runtime.invoke('unit::cleanup') == 'ok'

    [invoke_span] = [span for span in exporter.get_finished_spans() if span.name == 'runtime.invoke']
    sent_context = sent['trace_context']
    expected_trace_id = base64.urlsafe_b64encode(invoke_span.context.trace_id.to_bytes(16)).rstrip(b'=').decode()
    expected_span_id = base64.urlsafe_b64encode(invoke_span.context.span_id.to_bytes(8)).rstrip(b'=').decode()
    assert sent_context is not None
    assert sent_context['otel_carrier']['traceparent'].split('-')[2] == f'{invoke_span.context.span_id:016x}'
    assert sent_context['trace_id'] == expected_trace_id
    assert sent_context['parent_span_id'] == expected_span_id
    assert invoke_span.parent.span_id == cleanup_parent.get_span_context().span_id


@pytest.mark.asyncio
async def test_execute_invoked_func_passes_payload_carrier_as_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.runtime import models
    from nexus.trace import RemoteTraceContext
    from nexus.trace import tracer as nexus_tracer

    carrier = {'traceparent': '00-11111111111111111111111111111111-2222222222222222-01'}
    observed: dict[str, Any] = {}

    async def invoked() -> str:
        return 'ok'

    @contextmanager
    def capture_remote_context(*args: Any, **kwargs: Any):
        observed.update(args=args, kwargs=kwargs)
        yield

    monkeypatch.setattr(models, 'resolve_func', lambda _path: invoked)
    monkeypatch.setattr(nexus_tracer, 'with_remote_trace_context', capture_remote_context)
    payload = base64.b64encode(pickle.dumps(((), {}))).decode()

    result = await models.execute_invoked_func(
        'unit::invoked',
        payload,
        object(),
        trace_context=RemoteTraceContext(
            trace_id='nexus-trace',
            parent_span_id='nexus-parent',
            run_id='run-1',
            run_span_id='run-span-1',
            run_kind='single',
            otel_carrier=carrier,
        ),
    )

    assert result == 'ok'
    assert observed['kwargs']['otel_carrier'] == carrier


def test_runtime_base_forwards_payload_otel_carrier_to_all_fallbacks() -> None:
    source = Path('nexus/runtime/server/app.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'with_remote_trace_context'
    ]

    assert len(calls) == 4
    assert all(any(keyword.arg == 'otel_carrier' for keyword in call.keywords) for call in calls)


@pytest.mark.asyncio
async def test_runtime_invoke_payload_carrier_overrides_conflicting_http_context_for_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('NEXUS_OTEL_ENABLED', raising=False)
    from opentelemetry import context as otel_context
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from nexus.core import executor_registry
    from nexus.runtime import models as runtime_models
    from nexus.utils import log as nexus_log

    monkeypatch.setattr(nexus_log, 'setup_logging', lambda **_kwargs: None)
    monkeypatch.setattr(executor_registry, 'get_all', lambda: {})
    from nexus.runtime.server import app as runtime_app
    from nexus.runtime.server.worker import sdk as worker_sdk
    from nexus.trace import RemoteTraceContext, SpanController, span_context
    from nexus.utils.otel.nexus_bridge import NexusOTelBridge

    sdk = worker_sdk.WorkerSDK('http://base.invalid', 'venv-1')
    _enable_otel(monkeypatch)

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('tests.runtime-chain')
    monkeypatch.setattr(otel_trace, 'get_tracer', provider.get_tracer)
    controller = SpanController(otel_bridge=NexusOTelBridge(tracer_provider=provider))
    monkeypatch.setattr(SpanController, '_instance', controller)

    async def invoked() -> str:
        with span_context('worker.nexus', attributes={'type': 'tool'}):
            return 'ok'

    monkeypatch.setattr(runtime_models, 'resolve_func', lambda _path: invoked)

    async def flush() -> None:
        return None

    monkeypatch.setattr(sdk._span_processor, 'flush', flush)

    async def reject_hot_path_flush() -> None:
        pytest.fail('OTel flush ran on invoke hot path')

    monkeypatch.setattr(worker_sdk, 'async_force_flush_best_effort', reject_hot_path_flush, raising=False)
    worker_handler = sdk._sio.handlers['/worker']['invoke']
    sent_to_worker: dict[str, Any] = {}

    async def worker_emit(event: str, data: dict[str, Any], *, namespace: str) -> None:
        assert event == 'invoke:result'
        assert namespace == '/worker'
        runtime_app._worker_invoke_futures[data['request_id']].set_result(data)

    monkeypatch.setattr(sdk._sio, 'emit', worker_emit)

    async def base_emit(
        event: str,
        data: dict[str, Any],
        *,
        to: str,
        namespace: str,
    ) -> None:
        assert event == 'invoke'
        assert to == 'worker-sid'
        assert namespace == '/worker'
        sent_to_worker.update(data)
        token = otel_context.attach(otel_context.Context())
        try:
            await worker_handler(data)
        finally:
            otel_context.detach(token)

    monkeypatch.setattr(runtime_app.sio, 'emit', base_emit)
    monkeypatch.setitem(runtime_app._worker_sids, 'worker-venv-1', 'worker-sid')

    with span_context('host.nexus', attributes={'type': 'agent'}) as host_stub:
        remote_context = RemoteTraceContext.from_current_span()
        assert host_stub.span is not None
        host_nexus_span_id = host_stub.span.span_id

    assert remote_context is not None
    assert remote_context.otel_carrier is not None
    request = runtime_models.InvokeRequest.build(
        'unit::invoked',
        request_id='request-chain',
        venv_id='venv-1',
        trace_context=remote_context,
    )
    with tracer.start_as_current_span('runtime-base-asgi', kind=otel_trace.SpanKind.SERVER):
        response = await runtime_app._execute_invoke(request)

    assert response.success is True
    sent_context = RemoteTraceContext.model_validate(sent_to_worker['trace_context'])
    assert sent_context.model_dump(exclude={'otel_carrier'}) == remote_context.model_dump(exclude={'otel_carrier'})
    assert sent_context.otel_carrier != remote_context.otel_carrier

    spans = list(exporter.get_finished_spans())
    by_name = {span.name: span for span in spans if span.name != 'runtime.worker.invoke'}
    transports = [span for span in spans if span.name == 'runtime.worker.invoke']
    client = next(span for span in transports if span.kind is otel_trace.SpanKind.CLIENT)
    server = next(span for span in transports if span.kind is otel_trace.SpanKind.SERVER)
    host = by_name['host.nexus']
    base = by_name['runtime-base-asgi']
    runtime_server = by_name['runtime.invoke']
    worker_nexus = by_name['worker.nexus']

    assert host.attributes['nexus.span_id'] == host_nexus_span_id
    assert base.parent is None
    assert base.context.trace_id != host.context.trace_id
    assert runtime_server.kind is otel_trace.SpanKind.SERVER
    assert runtime_server.parent.span_id == host.context.span_id
    assert runtime_server.context.trace_id == host.context.trace_id
    assert client.parent.span_id == runtime_server.context.span_id
    assert client.context.trace_id == host.context.trace_id
    assert server.parent.span_id == client.context.span_id
    assert worker_nexus.parent.span_id == server.context.span_id
    assert client.attributes['nexus.worker.id'] == 'worker-venv-1'
    assert server.attributes['nexus.worker.id'] == 'worker-venv-1'


@pytest.mark.asyncio
async def test_runtime_invoke_payload_carrier_overrides_conflicting_http_context_for_direct_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('NEXUS_OTEL_ENABLED', raising=False)
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from nexus.runtime import models as runtime_models
    from nexus.runtime.server import app as runtime_app
    from nexus.trace import RemoteTraceContext

    _enable_otel(monkeypatch)
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('tests.runtime-direct-authority')
    monkeypatch.setattr(otel_trace, 'get_tracer', provider.get_tracer)

    async def invoked() -> str:
        with tracer.start_as_current_span('runtime.direct.work'):
            return 'ok'

    monkeypatch.setattr(runtime_models, 'resolve_func', lambda _path: invoked)
    payload_trace_id = '11111111111111111111111111111111'
    payload_parent_id = '2222222222222222'
    request = runtime_models.InvokeRequest.build(
        'unit::invoked',
        request_id='request-direct-authority',
        trace_context=RemoteTraceContext(
            trace_id='nexus-trace',
            parent_span_id='nexus-parent',
            otel_carrier={'traceparent': f'00-{payload_trace_id}-{payload_parent_id}-01'},
        ),
    )

    with tracer.start_as_current_span('runtime-base-asgi') as http_server:
        response = await asyncio.create_task(runtime_app._execute_invoke(request))

    assert response.success is True
    spans = exporter.get_finished_spans()
    [server] = [span for span in spans if span.name == 'runtime.invoke']
    [work] = [span for span in spans if span.name == 'runtime.direct.work']
    assert server.kind is otel_trace.SpanKind.SERVER
    assert server.context.trace_id == int(payload_trace_id, 16)
    assert server.parent is not None
    assert server.parent.span_id == int(payload_parent_id, 16)
    assert work.context.trace_id == int(payload_trace_id, 16)
    assert work.parent.span_id == server.context.span_id
    assert work.context.trace_id != http_server.get_span_context().trace_id


@pytest.mark.asyncio
async def test_runtime_invoke_keeps_matching_http_context_for_direct_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('NEXUS_OTEL_ENABLED', raising=False)
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from nexus.runtime import models as runtime_models
    from nexus.runtime.server import app as runtime_app
    from nexus.trace import RemoteTraceContext

    _enable_otel(monkeypatch)
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('tests.runtime-direct-matching-http')
    monkeypatch.setattr(otel_trace, 'get_tracer', provider.get_tracer)

    async def invoked() -> str:
        with tracer.start_as_current_span('runtime.direct.matching-http'):
            return 'ok'

    monkeypatch.setattr(runtime_models, 'resolve_func', lambda _path: invoked)

    with tracer.start_as_current_span('shared-trace'):
        with tracer.start_as_current_span('payload-parent') as payload_parent:
            payload_context = payload_parent.get_span_context()
            payload_carrier = {'traceparent': (f'00-{payload_context.trace_id:032x}-{payload_context.span_id:016x}-01')}

        request = runtime_models.InvokeRequest.build(
            'unit::invoked',
            request_id='request-direct-matching-http',
            trace_context=RemoteTraceContext(
                trace_id='nexus-trace',
                parent_span_id='nexus-parent',
                otel_carrier=payload_carrier,
            ),
        )
        with tracer.start_as_current_span('runtime-base-asgi') as http_server:
            response = await asyncio.create_task(runtime_app._execute_invoke(request))

    assert response.success is True
    spans = exporter.get_finished_spans()
    [server] = [span for span in spans if span.name == 'runtime.invoke']
    [work] = [span for span in spans if span.name == 'runtime.direct.matching-http']
    assert server.kind is otel_trace.SpanKind.SERVER
    assert server.context.trace_id == payload_context.trace_id
    assert server.parent is not None
    assert server.parent.span_id == http_server.get_span_context().span_id
    assert server.parent.span_id != payload_context.span_id
    assert work.parent.span_id == server.context.span_id


@pytest.mark.asyncio
async def test_runtime_invoke_invalid_payload_carrier_keeps_http_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('NEXUS_OTEL_ENABLED', raising=False)
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from nexus.runtime import models as runtime_models
    from nexus.runtime.server import app as runtime_app
    from nexus.trace import RemoteTraceContext

    _enable_otel(monkeypatch)
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('tests.runtime-direct-invalid-carrier')
    monkeypatch.setattr(otel_trace, 'get_tracer', provider.get_tracer)

    async def invoked() -> str:
        with tracer.start_as_current_span('runtime.direct.invalid-carrier'):
            return 'ok'

    monkeypatch.setattr(runtime_models, 'resolve_func', lambda _path: invoked)
    request = runtime_models.InvokeRequest.build(
        'unit::invoked',
        request_id='request-invalid-carrier',
        trace_context=RemoteTraceContext(
            trace_id='nexus-trace',
            parent_span_id='nexus-parent',
            otel_carrier={'traceparent': 'invalid'},
        ),
    )

    with tracer.start_as_current_span('runtime-base-asgi') as http_server:
        response = await asyncio.create_task(runtime_app._execute_invoke(request))

    assert response.success is True
    spans = exporter.get_finished_spans()
    [server] = [span for span in spans if span.name == 'runtime.invoke']
    [work] = [span for span in spans if span.name == 'runtime.direct.invalid-carrier']
    assert server.kind is otel_trace.SpanKind.SERVER
    assert server.parent is not None
    assert server.parent.span_id == http_server.get_span_context().span_id
    assert work.context.trace_id == http_server.get_span_context().trace_id
    assert work.parent.span_id == server.context.span_id


@pytest.mark.asyncio
async def test_worker_payload_carrier_replaces_stale_current_before_recording_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'false')
    import aiohttp

    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from nexus.runtime.server.worker import sdk as worker_sdk
    from nexus.trace import RemoteTraceContext

    sdk = worker_sdk.WorkerSDK('http://base.invalid', 'venv-1')
    _enable_otel(monkeypatch)

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel_trace, 'get_tracer', provider.get_tracer)

    class Session:
        async def __aenter__(self) -> Session:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def get(self, _url: str) -> Any:
            raise OSError('blob fetch failed')

    async def flush() -> None:
        return None

    emitted: dict[str, Any] = {}

    async def emit(event: str, data: dict[str, Any], *, namespace: str) -> None:
        emitted.update(event=event, data=data, namespace=namespace)

    monkeypatch.setattr(aiohttp, 'ClientSession', Session)
    monkeypatch.setattr(sdk._span_processor, 'flush', flush)

    async def reject_hot_path_flush() -> None:
        pytest.fail('OTel flush ran on invoke hot path')

    monkeypatch.setattr(worker_sdk, 'async_force_flush_best_effort', reject_hot_path_flush, raising=False)
    monkeypatch.setattr(sdk._sio, 'emit', emit)
    remote_parent_id = '2222222222222222'
    trace_context = RemoteTraceContext(
        trace_id='nexus-trace',
        parent_span_id='nexus-parent',
        otel_carrier={
            'traceparent': f'00-11111111111111111111111111111111-{remote_parent_id}-01',
        },
    )

    stale_tracer = provider.get_tracer('tests.worker-stale-current')
    with stale_tracer.start_as_current_span('stale-worker-current') as stale_current:
        await sdk._sio.handlers['/worker']['invoke']({
            'import_path': 'unit::never-reached',
            'request_id': 'request-payload-ref',
            'payload_ref': 'blob-secret-id',
            'trace_context': trace_context.model_dump(),
        })

    spans = exporter.get_finished_spans()
    [span] = [candidate for candidate in spans if candidate.name == 'runtime.worker.invoke']
    assert span.name == 'runtime.worker.invoke'
    assert span.kind is otel_trace.SpanKind.SERVER
    assert span.parent.span_id == int(remote_parent_id, 16)
    assert span.context.trace_id == int('11111111111111111111111111111111', 16)
    assert span.context.trace_id != stale_current.get_span_context().trace_id
    assert span.status.status_code is otel_trace.StatusCode.ERROR
    assert span.attributes['rpc.system'] == 'socket.io'
    assert span.attributes['rpc.method'] == 'invoke'
    assert span.attributes['nexus.worker.id'] == 'worker-venv-1'
    assert emitted['event'] == 'invoke:result'
    assert emitted['data']['success'] is False


@pytest.mark.asyncio
async def test_worker_and_base_otel_flush_runs_after_nexus_barrier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('NEXUS_OTEL_ENABLED', raising=False)
    from nexus.core import executor_registry
    from nexus.utils import log as nexus_log

    monkeypatch.setattr(nexus_log, 'setup_logging', lambda **_kwargs: None)
    monkeypatch.setattr(executor_registry, 'get_all', lambda: {})
    from nexus.runtime.server import app as runtime_app
    from nexus.runtime.server.worker import sdk as worker_sdk

    sdk = worker_sdk.WorkerSDK('http://base.invalid', 'venv-1')
    order: list[str] = []

    async def worker_flush() -> None:
        order.append('worker-nexus')

    async def worker_otel_flush() -> bool:
        order.append('worker-otel')
        return False

    monkeypatch.setattr(sdk._span_processor, 'flush', worker_flush)
    monkeypatch.setattr(worker_sdk, 'async_force_flush_best_effort', worker_otel_flush, raising=False)

    worker_response = await sdk._sio.handlers['/worker']['trace:flush']({})

    assert worker_response == {'ok': True}
    assert order == ['worker-nexus', 'worker-otel']

    async def base_barrier() -> int:
        order.append('base-nexus')
        return 7

    async def base_otel_flush() -> bool:
        order.append('base-otel')
        return False

    monkeypatch.setattr(runtime_app.span_broadcaster, 'barrier', base_barrier)
    monkeypatch.setattr(runtime_app, '_worker_sids', {})
    monkeypatch.setattr(runtime_app, 'async_force_flush_best_effort', base_otel_flush, raising=False)

    base_response = await runtime_app.trace_flush('sid')

    assert base_response == {'ok': True, 'span_batch_id': 7}
    assert order[-2:] == ['base-nexus', 'base-otel']


@pytest.mark.asyncio
async def test_worker_and_base_trace_barriers_use_shared_async_otel_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('NEXUS_OTEL_ENABLED', raising=False)
    from nexus.core import executor_registry
    from nexus.utils import log as nexus_log

    monkeypatch.setattr(nexus_log, 'setup_logging', lambda **_kwargs: None)
    monkeypatch.setattr(executor_registry, 'get_all', lambda: {})
    from nexus.runtime.server import app as runtime_app
    from nexus.runtime.server.worker import sdk as worker_sdk

    flushed: list[str] = []

    async def base_barrier() -> int:
        return 1

    async def worker_flush() -> None:
        return None

    async def base_otel_flush() -> bool:
        flushed.append('base')
        return True

    async def worker_otel_flush() -> bool:
        flushed.append('worker')
        return True

    monkeypatch.setattr(runtime_app, '_worker_sids', {})
    monkeypatch.setattr(runtime_app.span_broadcaster, 'barrier', base_barrier)
    monkeypatch.setattr(runtime_app, 'async_force_flush_best_effort', base_otel_flush, raising=False)

    await runtime_app.trace_flush('sid')

    sdk = worker_sdk.WorkerSDK('http://base.invalid', 'venv-1')
    monkeypatch.setattr(sdk._span_processor, 'flush', worker_flush)
    monkeypatch.setattr(worker_sdk, 'async_force_flush_best_effort', worker_otel_flush, raising=False)

    await sdk._sio.handlers['/worker']['trace:flush']({})

    assert flushed == ['base', 'worker']


@pytest.mark.asyncio
async def test_gongfeng_relies_on_http_instrumentation_and_preserves_legacy_disabled_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nexus.runtime.providers import gongfeng

    captured_headers: list[dict[str, str]] = []

    class Response:
        status_code = 200
        text = '{"code": 0, "data": {}}'

        def json(self) -> dict[str, Any]:
            return {'code': 0, 'data': {}}

    async def post(_url: str, *, headers: dict[str, str], **_kwargs: Any) -> Response:
        captured_headers.append(headers)
        return Response()

    monkeypatch.setattr(gongfeng.http_client, 'post', post)
    provider = gongfeng.GongfengRuntimeProvider(api_base_url='http://gongfeng.invalid', token='token')

    _enable_otel(monkeypatch)
    monkeypatch.setattr(
        gongfeng,
        'inject_carrier',
        lambda: (_ for _ in ()).throw(AssertionError('aiohttp instrumentation owns HTTP propagation')),
        raising=False,
    )
    monkeypatch.setattr(
        gongfeng,
        '_current_w3c_traceparent',
        lambda: (_ for _ in ()).throw(AssertionError('legacy conversion must not run')),
    )
    await provider._call_api('/enabled', {})

    assert captured_headers[-1] == {'token': 'token'}

    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'false')
    monkeypatch.setattr(gongfeng, '_current_w3c_traceparent', lambda: 'legacy-traceparent')
    await provider._call_api('/disabled', {})

    assert captured_headers[-1] == {'token': 'token', 'traceparent': 'legacy-traceparent'}
