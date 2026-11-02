from types import SimpleNamespace

import pytest

from aiohttp import web

from nexus.cli.commands import distributed
from nexus.cli.commands.distributed import (
    _instance_manager_endpoint,
    _instance_manager_metadata,
    _run_instance_manager,
    _start_instance_manager_http,
)
from nexus.distributed.config import DistributedConfig


def test_instance_manager_endpoint_and_metadata_use_pod_identity():
    environment = {
        'POD_IP': '21.162.151.155',
        'POD_NAME': 'nexus-instance-manager-0',
        'POD_NAMESPACE': 'nexus-system',
        'POD_UID': 'pod-uid-a',
        'NEXUS_DIST_PROVIDER_CONFIG_REF': 'tkex/shanghai',
        'NEXUS_DIST_CLUSTER_NAME': 'cls-g84u227v',
    }
    args = SimpleNamespace(endpoint_url=None, host='0.0.0.0', port=8091)

    endpoint = _instance_manager_endpoint(args, environ=environment)

    assert endpoint == 'http://21.162.151.155:8091'
    assert _instance_manager_metadata(endpoint, environ=environment) == {
        'endpoint_kind': 'pod_ip',
        'provider_config_ref': 'tkex/shanghai',
        'cluster_name': 'cls-g84u227v',
        'namespace': 'nexus-system',
        'pod_name': 'nexus-instance-manager-0',
        'pod_uid': 'pod-uid-a',
    }


def test_instance_manager_endpoint_formats_ipv6_pod_ip():
    args = SimpleNamespace(endpoint_url=None, host='0.0.0.0', port=8091)

    assert _instance_manager_endpoint(args, environ={'POD_IP': 'fd00::10'}) == 'http://[fd00::10]:8091'


def test_instance_manager_endpoint_requires_pod_ip_inside_pod():
    args = SimpleNamespace(endpoint_url=None, host='0.0.0.0', port=8091)

    with pytest.raises(ValueError, match='pod_ip_required'):
        _instance_manager_endpoint(args, environ={'POD_NAME': 'manager-0', 'POD_NAMESPACE': 'nexus-system'})


async def test_instance_manager_http_listens_before_registering(monkeypatch):
    events = []

    class FakeRunner:
        def __init__(self, app):
            self.app = app

        async def setup(self):
            events.append('runner.setup')

        async def cleanup(self):
            events.append('runner.cleanup')

    class FakeSite:
        def __init__(self, runner, host, port):
            assert runner.app == 'app'
            assert (host, port) == ('0.0.0.0', 8091)

        async def start(self):
            events.append('site.start')

    class FakeService:
        async def start(self):
            events.append('service.start')

        def start_background_tasks(self):
            events.append('service.background')

    monkeypatch.setattr(distributed.web, 'AppRunner', FakeRunner)
    monkeypatch.setattr(distributed.web, 'TCPSite', FakeSite)

    await _start_instance_manager_http(FakeService(), 'app', host='0.0.0.0', port=8091)

    assert events == ['runner.setup', 'site.start', 'service.start', 'service.background']


async def test_instance_manager_command_exits_after_service_self_fences(monkeypatch):
    from nexus.distributed.instance_manager import internal
    from nexus.distributed.instance_manager import service as service_module
    from nexus.distributed.persistence import db

    events = []

    class FakeEngine:
        async def dispose(self):
            events.append('engine.dispose')

    class FakeService:
        def __init__(self, **kwargs):
            del kwargs
            self.log_path = None
            self.allocation_id = None

        async def wait_stopped(self):
            events.append('service.wait_stopped')

        async def stop(self):
            events.append('service.stop')

    class FakeRunner:
        async def cleanup(self):
            events.append('runner.cleanup')

    async def start_http(service, app, *, host, port):
        del service, app, host, port
        events.append('http.started')
        return FakeRunner()

    monkeypatch.setattr(distributed, '_instance_manager_endpoint', lambda args: 'http://127.0.0.1:8091')
    monkeypatch.setattr(distributed, '_instance_manager_metadata', lambda endpoint: {'endpoint': endpoint})
    monkeypatch.setattr(distributed, '_create_executor', lambda args, config, capacity: object())
    monkeypatch.setattr(distributed, '_start_instance_manager_http', start_http)
    monkeypatch.setattr(db, 'create_engine', lambda config: FakeEngine())
    monkeypatch.setattr(db, 'create_session_factory', lambda engine: object())
    monkeypatch.setattr(service_module, 'InstanceManagerService', FakeService)
    monkeypatch.setattr(internal, 'create_instance_manager_internal_app', lambda **kwargs: object())

    args = SimpleNamespace(
        instance_manager_id='im-self-fenced',
        capacity=1,
        force_replace=False,
        environment_id=None,
        pool_id=None,
        workload='execution',
        label=[],
        bootstrap_version=None,
        artifact_ref_version=None,
        host='0.0.0.0',
        port=8091,
    )
    await _run_instance_manager(args, DistributedConfig(shard_count=8), log_path=None)

    assert events == [
        'http.started',
        'service.wait_stopped',
        'service.stop',
        'runner.cleanup',
        'engine.dispose',
    ]


async def test_web_app_loop_observability_starts_and_stops_both_monitors(monkeypatch):
    events: list[str] = []

    class Handle:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            events.append(f'{self.name}.close')

        def stop(self) -> None:
            events.append(f'{self.name}.stop')

    monkeypatch.setattr(
        distributed,
        '_start_main_loop_aiomonitor',
        lambda config, name: events.append(f'aiomonitor.start:{name}') or Handle('aiomonitor'),
    )
    monkeypatch.setattr(
        distributed,
        'start_event_loop_monitor',
        lambda name: events.append(f'lag.start:{name}') or Handle('lag'),
        raising=False,
    )
    app = web.Application()

    distributed._install_web_app_loop_observability(
        app,
        DistributedConfig(aiomonitor_enabled=False),
        'control-gateway',
    )
    context = app.cleanup_ctx[0](app)
    await anext(context)
    await context.aclose()

    assert events == [
        'aiomonitor.start:control-gateway',
        'lag.start:control-gateway',
        'lag.stop',
        'aiomonitor.close',
    ]


@pytest.mark.parametrize(
    'command',
    [
        'control-gateway',
        'application-gateway',
        'scheduler',
        'reconciler',
        'instance-manager',
        'dev',
    ],
)
def test_distributed_start_commands_do_not_install_uvloop(monkeypatch, command):
    config = SimpleNamespace(validate_for_service=lambda _command: None)
    args = SimpleNamespace(
        distributed_command=command,
        owner_id='owner-1',
        instance_manager_id='manager-1',
    )

    async def initialize_config(*_args, **_kwargs):
        return config

    async def noop(*_args, **_kwargs):
        return None

    from nexus.utils import event_loop as event_loop_module

    monkeypatch.setattr(distributed, '_config_scope_from_args', lambda _args: (command, None))
    monkeypatch.setattr(distributed, '_initialize_service_config', initialize_config)
    monkeypatch.setattr(distributed, '_config_from_args', lambda _args: config)
    monkeypatch.setattr(distributed, '_configure_distributed_logging', lambda *_args: None)
    monkeypatch.setattr(
        event_loop_module,
        'install_uvloop_required',
        lambda: pytest.fail(f'{command} must use the standard asyncio event loop'),
        raising=False,
    )
    monkeypatch.setattr(distributed, '_run_web_app', lambda *_args: 0)
    monkeypatch.setattr(distributed, '_run_scheduler', noop)
    monkeypatch.setattr(distributed, '_run_reconciler', noop)
    monkeypatch.setattr(distributed, '_run_monitored_instance_manager', noop)
    monkeypatch.setattr(distributed, '_run_dev', noop)

    assert distributed.distributed_command(args) == 0
