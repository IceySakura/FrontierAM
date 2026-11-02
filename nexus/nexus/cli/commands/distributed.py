from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import logging
import os
import shutil
import signal
import sys
import urllib.parse
import urllib.request

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web


if TYPE_CHECKING:
    from nexus.distributed.config import DistributedConfig


logger = logging.getLogger(__name__)


async def _run_until_terminated(operation):
    """Let service finalizers, including the last metrics export, finish on Pod termination."""
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    terminated = False
    installed = False
    previous = signal.getsignal(signal.SIGTERM)

    def terminate():
        nonlocal terminated
        if not terminated:
            terminated = True
            if task is not None:
                task.cancel()

    try:
        if previous == signal.SIG_DFL:
            try:
                loop.add_signal_handler(signal.SIGTERM, terminate)
                installed = True
            except (NotImplementedError, RuntimeError):
                pass
        try:
            return await operation
        except asyncio.CancelledError:
            if not terminated:
                raise
    finally:
        if installed:
            loop.remove_signal_handler(signal.SIGTERM)
            signal.signal(signal.SIGTERM, previous)


async def monitor_event_loop(name: str, operation, *, stall_timeout=None, on_stall=None):
    from nexus.distributed.observability.event_loop import monitor_event_loop as _monitor_event_loop

    return await _monitor_event_loop(
        name,
        operation,
        stall_timeout=stall_timeout,
        on_stall=on_stall,
    )


def _create_hard_exit_guard(*, timeout: float):
    from nexus.distributed.instance_manager.self_fence import HardExitGuard

    return HardExitGuard(timeout=timeout)


def start_event_loop_monitor(name: str):
    from nexus.distributed.observability.event_loop import start_event_loop_monitor as _start_event_loop_monitor

    return _start_event_loop_monitor(name)


async def _run_monitored_instance_manager(
    args: argparse.Namespace,
    config: DistributedConfig,
    *,
    log_path: Path | None,
) -> None:
    hard_exit_guard = _create_hard_exit_guard(timeout=config.manager_hard_exit_timeout.total_seconds())
    try:
        await monitor_event_loop(
            f'instance-manager:{args.instance_manager_id}',
            _run_instance_manager(args, config, log_path=log_path, hard_exit_guard=hard_exit_guard),
            stall_timeout=config.manager_self_fence_timeout.total_seconds(),
            on_stall=lambda _stalled_for: hard_exit_guard.arm('manager_loop_stall'),
        )
    finally:
        hard_exit_guard.disarm()


def _parse_labels(raw_labels: list[str] | None) -> dict[str, str]:
    labels: dict[str, str] = {}
    for raw_label in raw_labels or []:
        if '=' not in raw_label:
            raise ValueError(f'invalid label {raw_label!r}; expected key=value')
        key, value = raw_label.split('=', 1)
        if not key:
            raise ValueError(f'invalid label {raw_label!r}; label key cannot be empty')
        labels[key] = value
    return labels


def _allocation_liveness_from_args(args: argparse.Namespace):
    from nexus.distributed.environment.liveness import startup_liveness

    if getattr(args, 'distributed_command', None) != 'instance-manager':
        return None
    raw = os.environ.get('NEXUS_DIST_ALLOCATION_SPEC_JSON')
    spec = json.loads(raw) if raw else {}
    if not isinstance(spec, dict):
        raise ValueError('invalid_allocation_spec')
    return startup_liveness(
        spec,
        required=getattr(args, 'require_allocation_liveness_v1', False),
        allocation_id=os.environ.get('NEXUS_DIST_MANAGER_ALLOCATION_ID'),
        environment_id=getattr(args, 'environment_id', ''),
        pool_id=getattr(args, 'pool_id', ''),
        workload=getattr(args, 'workload', ''),
    )


def _request_policy_from_args(args: argparse.Namespace) -> dict[str, str] | None:
    if getattr(args, 'distributed_command', None) != 'instance-manager':
        return None
    cli_binding = getattr(args, 'request_policy_binding', None)
    cli_scope = getattr(args, 'request_policy_scope', None)
    if bool(cli_binding) != bool(cli_scope):
        raise ValueError('--request-policy-binding and --request-policy-scope must be provided together')
    cli_selection = {'binding': cli_binding, 'scope': cli_scope} if cli_binding and cli_scope else None

    raw = os.environ.get('NEXUS_DIST_ALLOCATION_SPEC_JSON')
    spec = json.loads(raw) if raw else {}
    if not isinstance(spec, dict):
        raise ValueError('invalid_allocation_spec')
    spec_selection = spec.get('request_policy')
    if spec_selection is not None and not isinstance(spec_selection, dict):
        raise ValueError('invalid_allocation_request_policy')
    if cli_selection is not None and spec_selection is not None and cli_selection != spec_selection:
        raise ValueError('conflicting_manager_request_policy')
    return cli_selection or spec_selection


def _config_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if policy := _allocation_liveness_from_args(args):
        updates.update(policy.config_overrides())
    if request_policy := _request_policy_from_args(args):
        updates['request_policy'] = request_policy
    if getattr(args, 'database_url', None):
        updates['database_url'] = args.database_url
    if getattr(args, 'shard_count', None):
        updates['shard_count'] = args.shard_count
    return updates


def _config_cli_secret_overrides(args: argparse.Namespace) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if getattr(args, 'internal_token', None):
        updates['internal_token'] = args.internal_token
    return updates


def _config_values_from_args(args: argparse.Namespace) -> dict[str, Any]:
    from nexus.distributed.config import DistributedConfig

    values = DistributedConfig.values_from_env()
    values.update(_config_cli_overrides(args))
    return values


def _config_from_args(args: argparse.Namespace) -> DistributedConfig:
    from nexus.distributed.config import DistributedConfig

    values = _config_values_from_args(args)
    secrets = _config_cli_secret_overrides(args)
    if secrets:
        values['secrets'] = secrets
    config = DistributedConfig.model_validate(values)
    policy = _allocation_liveness_from_args(args)
    if policy is not None:
        policy.validate_platform(config)
    return config


def _config_scope_from_args(args: argparse.Namespace) -> tuple[str, str | None]:
    role = args.distributed_command
    identity = None
    if role in {'scheduler', 'reconciler'}:
        identity = args.owner_id
    elif role == 'instance-manager':
        identity = args.instance_manager_id
    return role, identity


async def _initialize_service_config(
    args: argparse.Namespace,
    *,
    role: str,
    identity: str | None,
) -> DistributedConfig:
    from nexus.distributed.config import DistributedConfig
    from nexus.distributed.config.store import initialize_distributed_config
    from nexus.distributed.persistence.migrations import REQUEST_BUDGET_LEASES_REVISION

    config = await initialize_distributed_config(
        role=role,
        identity=identity,
        environment_values=DistributedConfig.values_from_env(),
        overrides=_config_cli_overrides(args),
        secret_overrides=_config_cli_secret_overrides(args),
        require_stored_config=_config_store_required_from_env(),
        minimum_schema_revision=REQUEST_BUDGET_LEASES_REVISION if role == 'instance-manager' else None,
    )
    if policy := _allocation_liveness_from_args(args):
        policy.validate_platform(config)
    return config


def _config_store_required_from_env() -> bool:
    raw = os.getenv('NEXUS_DIST_CONFIG_STORE_REQUIRED', '0').strip().lower()
    if raw in {'0', 'false', 'no', 'off'}:
        return False
    if raw in {'1', 'true', 'yes', 'on'}:
        return True
    raise ValueError('NEXUS_DIST_CONFIG_STORE_REQUIRED must be a boolean')


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--database-url', help='SQLAlchemy async database URL, defaults to NEXUS_DIST_DATABASE_URL')
    parser.add_argument('--internal-token', help='Internal bearer token for local development')
    parser.add_argument('--shard-count', type=int, help='Scheduler shard count')


def _add_web_arguments(parser: argparse.ArgumentParser, *, default_port: int) -> None:
    parser.add_argument('--host', default='127.0.0.1', help='Bind host')
    parser.add_argument('--port', type=int, default=default_port, help='Bind port')


def _local_connect_host(bind_host: str) -> str:
    return '127.0.0.1' if bind_host in {'0.0.0.0', '::'} else bind_host


def _url_host(host: str) -> str:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    return f'[{address.compressed}]' if address.version == 6 else address.compressed


def _instance_manager_endpoint(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    environment = os.environ if environ is None else environ
    if args.endpoint_url:
        return str(args.endpoint_url).rstrip('/')
    pod_ip = environment.get('POD_IP')
    if pod_ip:
        try:
            host = _url_host(str(ipaddress.ip_address(pod_ip)))
        except ValueError as exc:
            raise ValueError('invalid_pod_ip') from exc
        return f'http://{host}:{args.port}'
    if environment.get('POD_NAMESPACE'):
        raise ValueError('pod_ip_required')
    return f'http://{_url_host(_local_connect_host(args.host))}:{args.port}'


def _instance_manager_metadata(
    endpoint_url: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = os.environ if environ is None else environ
    pod_ip = environment.get('POD_IP')
    endpoint_host = urllib.parse.urlsplit(endpoint_url).hostname
    metadata = {
        'endpoint_kind': 'pod_ip' if pod_ip and endpoint_host == pod_ip else 'configured',
    }
    environment_fields = {
        'provider_config_ref': 'NEXUS_DIST_PROVIDER_CONFIG_REF',
        'cluster_name': 'NEXUS_DIST_CLUSTER_NAME',
        'namespace': 'POD_NAMESPACE',
        'pod_name': 'POD_NAME',
        'pod_uid': 'POD_UID',
    }
    for key, environment_name in environment_fields.items():
        if value := environment.get(environment_name):
            metadata[key] = value
    return metadata


def _add_executor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--executor', choices=['asyncio', 'thread'], default='asyncio')
    parser.add_argument('--thread-strategy', choices=['pooled', 'per_execution'], default='per_execution')
    parser.add_argument('--thread-max-tasks-per-thread', type=int, default=100)


def _add_environment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--environment-id', default='predefined-default')
    parser.add_argument('--pool-id', default='default')
    parser.add_argument('--workload', choices=['execution', 'orchestration'], default='execution')
    parser.add_argument('--label', action='append', default=[], help='InstanceManager label in key=value form')


def _add_placement_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--placement-environment', help='Default execution placement environment')
    parser.add_argument('--placement-pool', help='Default execution placement pool')
    parser.add_argument(
        '--placement-label', action='append', default=[], help='Default placement label in key=value form'
    )


def _placement_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    placement: dict[str, Any] = {}
    if getattr(args, 'placement_environment', None):
        placement['environment'] = args.placement_environment
    if getattr(args, 'placement_pool', None):
        placement['pool'] = args.placement_pool
    labels = _parse_labels(getattr(args, 'placement_label', None))
    if labels:
        placement['labels'] = labels
    return placement or None


def _placement_from_environment_args(args: argparse.Namespace) -> dict[str, Any]:
    placement: dict[str, Any] = {
        'environment': args.environment_id,
        'pool': args.pool_id,
    }
    labels = _parse_labels(args.label)
    if labels:
        placement['labels'] = labels
    return placement


def _create_executor(args: argparse.Namespace, config: DistributedConfig, *, capacity: int):
    from nexus.distributed.observability.event_loop import LoopAiomonitorConfig

    aiomonitor_config = LoopAiomonitorConfig.from_distributed_config(config)
    if args.executor == 'thread':
        from nexus.distributed.execution.executors.thread_executor import ThreadExecutor

        return ThreadExecutor(
            strategy=args.thread_strategy,
            max_threads=capacity,
            max_tasks_per_thread=args.thread_max_tasks_per_thread,
            aiomonitor_config=aiomonitor_config,
        )
    from nexus.distributed.execution.executors.asyncio_executor import AsyncioExecutor

    return AsyncioExecutor(capacity_total=capacity, aiomonitor_config=aiomonitor_config)


def _start_main_loop_aiomonitor(config: DistributedConfig, name: str, *, port_offset: int = 0):
    from nexus.distributed.observability.event_loop import LoopAiomonitorConfig, start_aiomonitor_for_loop

    return start_aiomonitor_for_loop(
        asyncio.get_running_loop(),
        config=LoopAiomonitorConfig.from_distributed_config(config),
        name=name,
        port_offset=port_offset,
    )


def _install_web_app_loop_observability(app: web.Application, config: DistributedConfig, name: str) -> None:
    async def event_loop_observability_context(_app: web.Application):
        aiomonitor_handle = _start_main_loop_aiomonitor(config, name)
        event_loop_monitor = start_event_loop_monitor(name)
        try:
            yield
        finally:
            if event_loop_monitor is not None:
                event_loop_monitor.stop()
            if aiomonitor_handle is not None:
                aiomonitor_handle.close()

    app.cleanup_ctx.append(event_loop_observability_context)


async def init_database(config: DistributedConfig) -> None:
    from nexus.distributed.persistence.db import init_database as _init_database

    await _init_database(config)


async def upgrade_database(config: DistributedConfig) -> None:
    from nexus.distributed.persistence.db import upgrade_database as _upgrade_database

    await _upgrade_database(config)


@dataclass(frozen=True)
class MycliInvocation:
    argv: list[str]
    env: dict[str, str]


def _normalize_mycli_args(extra_args: list[str] | None) -> list[str]:
    args = list(extra_args or [])
    if args[:1] == ['--']:
        return args[1:]
    return args


def _string_query_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        value = value[0] if value else None
    return str(value) if value is not None else None


def _mycli_invocation_from_sqlalchemy_url(
    database_url: str,
    *,
    prompt_password: bool,
    extra_args: list[str] | None,
) -> MycliInvocation:
    from sqlalchemy.engine import make_url

    url = make_url(database_url)
    if url.get_backend_name() != 'mysql':
        raise ValueError(f'distributed db shell only supports MySQL URLs, got {url.get_backend_name()!r}')

    argv = ['mycli']
    if url.host:
        argv.extend(['--host', url.host])
    if url.port is not None:
        argv.extend(['--port', str(url.port)])
    if url.username:
        argv.extend(['--user', url.username])
    if url.database:
        argv.extend(['--database', url.database])

    charset = _string_query_value(url.query.get('charset'))
    if charset:
        argv.extend(['--charset', charset])

    env = {}
    if url.password and not prompt_password:
        env['MYSQL_PWD'] = url.password

    argv.extend(_normalize_mycli_args(extra_args))
    return MycliInvocation(argv=argv, env=env)


def run_database_shell(
    config: DistributedConfig,
    *,
    prompt_password: bool,
    mycli_args: list[str] | None,
) -> int:
    if not config.database_url:
        raise ValueError('NEXUS_DIST_DATABASE_URL is required')
    if shutil.which('mycli') is None:
        print('mycli is not installed; install the distributed-db-shell extra first', file=sys.stderr)
        return 127

    invocation = _mycli_invocation_from_sqlalchemy_url(
        config.database_url,
        prompt_password=prompt_password,
        extra_args=mycli_args,
    )
    env = os.environ.copy()
    env.update(invocation.env)
    os.execvpe('mycli', invocation.argv, env)
    return 0


def _configure_distributed_logging(
    config: DistributedConfig,
    component: str,
    identity: str | None = None,
) -> Path | None:
    from nexus.distributed.observability.logging import setup_distributed_logging

    return setup_distributed_logging(
        log_dir=config.log_dir,
        log_level=config.log_level,
        component=component,
        identity=identity,
    )


def run_logs_tail(args: argparse.Namespace) -> int:
    url = f'{args.control_base_url.rstrip("/")}/v1/managers/{urllib.parse.quote(args.manager_id, safe="")}/logs'
    if args.follow:
        url = f'{url}/stream'
    params = {'lines': str(args.lines)}
    if args.start_time:
        params['start_time'] = args.start_time
    if args.end_time:
        params['end_time'] = args.end_time
    url = f'{url}?{urllib.parse.urlencode(params)}'
    request = urllib.request.Request(url, headers={'Accept': 'text/plain'}, method='GET')
    with urllib.request.urlopen(request, timeout=None if args.follow else 30) as response:
        if args.follow:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                sys.stdout.write(chunk.decode('utf-8', errors='replace'))
                sys.stdout.flush()
        else:
            sys.stdout.write(response.read().decode('utf-8', errors='replace'))
            sys.stdout.flush()
    return 0


def _read_json_object(path: str) -> dict[str, Any]:
    raw = sys.stdin.read() if path == '-' else Path(path).read_text(encoding='utf-8')
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f'config input is not valid JSON: {exc.msg}') from exc
    if not isinstance(value, dict):
        raise ValueError('config input must be a JSON object')
    return value


def _config_record_json(record, *, include_values: bool) -> dict[str, Any]:
    payload = {
        'role': record.role,
        'identity': record.identity,
        'version': record.version,
        'fields': list(record.fields),
    }
    if include_values:
        payload['values'] = record.values
    return payload


def _effective_config_json(config: DistributedConfig) -> dict[str, Any]:
    return config.model_dump(
        mode='json',
        exclude={'database_url', 'database_url_file', 'secrets'},
    )


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


async def _run_config_management(args: argparse.Namespace) -> None:
    from nexus.distributed.config.store import (
        delete_config_override,
        get_config_override,
        list_secret_override_summaries,
        load_distributed_config,
        set_config_override,
    )
    from nexus.distributed.persistence.db import create_engine, create_session_factory

    config = _config_from_args(args)
    config.validate_for_service('db-init')
    engine = create_engine(config)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            command = args.config_command
            if command in {'set', 'secret-set'}:
                secret = command == 'secret-set'
                record = await set_config_override(
                    session,
                    role=args.role,
                    identity=args.identity,
                    values=_read_json_object(args.file),
                    expected_version=args.expected_version,
                    actor=args.actor,
                    secret=secret,
                )
                await session.commit()
                _print_json(_config_record_json(record, include_values=not secret))
                return
            if command == 'show':
                record = await get_config_override(
                    session,
                    role=args.role,
                    identity=args.identity,
                )
                if record is None:
                    raise FileNotFoundError(f'distributed config scope not found: {args.role}/{args.identity}')
                effective = await load_distributed_config(
                    session_factory,
                    database_url=config.database_url,
                    role=args.role,
                    identity=None if args.identity == '*' else args.identity,
                )
                payload = _config_record_json(record, include_values=True)
                payload['effective_values'] = _effective_config_json(effective)
                _print_json(payload)
                return
            if command == 'secret-list':
                records = await list_secret_override_summaries(
                    session,
                    role=args.role,
                    identity=args.identity,
                )
                _print_json([_config_record_json(record, include_values=False) for record in records])
                return
            if command in {'delete', 'secret-delete'}:
                secret = command == 'secret-delete'
                deleted = await delete_config_override(
                    session,
                    role=args.role,
                    identity=args.identity,
                    expected_version=args.expected_version,
                    actor=args.actor,
                    secret=secret,
                )
                await session.commit()
                _print_json({'deleted': deleted, 'identity': args.identity, 'role': args.role})
                return
            raise ValueError(f'unsupported distributed config command: {command}')
    finally:
        await engine.dispose()


def distributed_command(args: argparse.Namespace) -> int:
    command = args.distributed_command
    if command == 'statistics':
        from nexus.distributed.analytics.backfill import run_backfill

        config = _config_from_args(args)
        config.validate_for_service('db-init')
        if not 1 <= args.max_pages <= 10000 or not 0 <= args.pause_seconds <= 60:
            raise ValueError('invalid statistics backfill limits')
        asyncio.run(monitor_event_loop('statistics-backfill', run_backfill(config, args)))
        return 0
    if command == 'db':
        config = _config_from_args(args)
        config.validate_for_service('db-init')
        if args.db_command == 'init':
            asyncio.run(monitor_event_loop('db-init', init_database(config)))
            return 0
        if args.db_command == 'upgrade':
            asyncio.run(monitor_event_loop('db-upgrade', upgrade_database(config)))
            return 0
        if args.db_command == 'shell':
            return run_database_shell(
                config,
                prompt_password=args.prompt_password,
                mycli_args=args.mycli_args,
            )
        return 0
    if command == 'logs':
        if args.logs_command == 'tail':
            return run_logs_tail(args)
        return 0
    if command == 'config':
        asyncio.run(monitor_event_loop('config-management', _run_config_management(args)))
        return 0
    service_commands = {'control-gateway', 'application-gateway', 'scheduler', 'reconciler', 'instance-manager'}
    if command in service_commands:
        role, identity = _config_scope_from_args(args)
        config = asyncio.run(_initialize_service_config(args, role=role, identity=identity))
        config.validate_for_service(command)
    else:
        config = _config_from_args(args)
    if command == 'control-gateway':
        _configure_distributed_logging(config, 'control-gateway')
        return _run_web_app(args, config, 'control')
    if command == 'application-gateway':
        _configure_distributed_logging(config, 'application-gateway')
        return _run_web_app(args, config, 'application')
    if command == 'scheduler':
        _configure_distributed_logging(config, 'scheduler', args.owner_id)
        asyncio.run(
            _run_until_terminated(monitor_event_loop(f'scheduler:{args.owner_id}', _run_scheduler(args, config)))
        )
        return 0
    if command == 'reconciler':
        _configure_distributed_logging(config, 'reconciler', args.owner_id)
        asyncio.run(
            _run_until_terminated(monitor_event_loop(f'reconciler:{args.owner_id}', _run_reconciler(args, config)))
        )
        return 0
    if command == 'instance-manager':
        log_path = _configure_distributed_logging(config, 'instance-manager', args.instance_manager_id)
        asyncio.run(_run_until_terminated(_run_monitored_instance_manager(args, config, log_path=log_path)))
        return 0
    if command == 'dev':
        log_path = _configure_distributed_logging(config, 'dev')
        asyncio.run(monitor_event_loop('dev', _run_dev(args, config, log_path=log_path)))
        return 0
    return 0


def _run_web_app(args: argparse.Namespace, config: DistributedConfig, kind: str) -> int:
    from aiohttp import web

    from nexus.distributed.gateways.application.app import create_application_app
    from nexus.distributed.gateways.common.platform_metrics import METRICS_KEY
    from nexus.distributed.gateways.control.app import create_control_app
    from nexus.distributed.persistence.db import create_engine, create_session_factory
    from nexus.utils.platform_metrics import process_recorder, set_process_recorder

    engine = create_engine(config)
    session_factory = create_session_factory(engine)
    if kind == 'control':
        app = create_control_app(session_factory=session_factory, config=config)
        _install_web_app_loop_observability(app, config, 'control-gateway')
    else:
        application_base_url = getattr(args, 'application_base_url', None)
        if not application_base_url:
            application_base_url = f'http://{_local_connect_host(args.host)}:{args.port}'
        app = create_application_app(
            session_factory=session_factory,
            config=config,
            live_compatibility_control_base_url=getattr(args, 'control_base_url', None)
            if getattr(args, 'enable_live_compat', False)
            else None,
            live_compatibility_application_base_url=application_base_url,
            live_compatibility_default_placement=_placement_from_args(args),
        )
        _install_web_app_loop_observability(app, config, 'application-gateway')

    async def engine_lifecycle(_app):
        yield
        await engine.dispose()

    app.cleanup_ctx.insert(0, engine_lifecycle)
    previous = process_recorder()
    set_process_recorder(app[METRICS_KEY])
    try:
        web.run_app(app, host=args.host, port=args.port)
    finally:
        set_process_recorder(previous)
    return 0


async def _run_scheduler(args: argparse.Namespace, config: DistributedConfig) -> None:
    from nexus.distributed.config.views import scheduler_config_view
    from nexus.distributed.control_plane.scheduler import SchedulerService
    from nexus.distributed.observability.service import PlatformControlPlaneMetrics, service_metrics
    from nexus.distributed.persistence.db import create_engine, create_session_factory

    aiomonitor_handle = _start_main_loop_aiomonitor(config, f'scheduler:{args.owner_id}')
    engine = create_engine(config)
    factory = create_session_factory(engine)
    try:
        async with service_metrics(config, 'scheduler', factory) as runtime:
            service = SchedulerService(
                session_factory=factory,
                config=scheduler_config_view(config),
                owner_id=args.owner_id,
                metrics=PlatformControlPlaneMetrics(runtime),
            )
            await service.run_forever()
    finally:
        if aiomonitor_handle is not None:
            aiomonitor_handle.close()
        await engine.dispose()


async def _run_reconciler(args: argparse.Namespace, config: DistributedConfig) -> None:
    from nexus.distributed.control_plane.reconciler import create_reconciler_service
    from nexus.distributed.observability.service import PlatformControlPlaneMetrics, service_metrics
    from nexus.distributed.persistence.db import create_engine, create_session_factory

    aiomonitor_handle = _start_main_loop_aiomonitor(config, 'reconciler')
    engine = create_engine(config)
    factory = create_session_factory(engine)
    try:
        async with service_metrics(config, 'reconciler', factory) as runtime:
            service = create_reconciler_service(
                session_factory=factory,
                config=config,
                owner_id=args.owner_id,
                metrics=PlatformControlPlaneMetrics(runtime),
            )
            await service.run_forever()
    finally:
        if aiomonitor_handle is not None:
            aiomonitor_handle.close()
        await engine.dispose()


async def _start_instance_manager_http(
    service: Any,
    app: web.Application,
    *,
    host: str,
    port: int,
) -> web.AppRunner:
    from aiohttp import web

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    service_started = False
    try:
        await site.start()
        await service.start()
        service_started = True
        service.start_background_tasks()
    except BaseException:
        if service_started:
            await service.stop()
        await runner.cleanup()
        raise
    return runner


async def _run_instance_manager(
    args: argparse.Namespace,
    config: DistributedConfig,
    *,
    log_path: Path | None,
    hard_exit_guard: Any | None = None,
) -> None:
    from nexus.distributed.config.views import instance_manager_config_view
    from nexus.distributed.instance_manager.internal import create_instance_manager_internal_app
    from nexus.distributed.instance_manager.service import InstanceManagerService
    from nexus.distributed.observability.service import service_metrics
    from nexus.distributed.persistence.db import create_engine, create_session_factory

    engine = create_engine(config)
    session_factory = create_session_factory(engine)
    async with service_metrics(config, 'instance-manager', session_factory):
        service = None
        runner = None
        try:
            endpoint_url = _instance_manager_endpoint(args)
            service = InstanceManagerService(
                session_factory=session_factory,
                config=instance_manager_config_view(config),
                manager_id=args.instance_manager_id,
                endpoint_url=endpoint_url,
                capacity_total=args.capacity,
                executor=_create_executor(args, config, capacity=args.capacity),
                force_replace=args.force_replace,
                environment_id=args.environment_id,
                pool_id=args.pool_id,
                workload=args.workload,
                labels=_parse_labels(args.label),
                metadata=_instance_manager_metadata(endpoint_url),
                bootstrap_version=getattr(args, 'bootstrap_version', None),
                artifact_ref_version=getattr(args, 'artifact_ref_version', None),
                hard_exit_guard=hard_exit_guard,
            )
            service.log_path = str(log_path) if log_path is not None else None
            app = create_instance_manager_internal_app(service=service, config=config)
            runner = await _start_instance_manager_http(service, app, host=args.host, port=args.port)
            logger.info(
                'Manager liveness: allocation=%s source=%s self_fence=%ss executor_hang=%ss dead=%ss '
                'heartbeat=%ss healthy=%ss hard_exit=%ss',
                service.allocation_id,
                'allocation' if getattr(args, 'require_allocation_liveness_v1', False) else 'platform',
                config.manager_self_fence_timeout.total_seconds(),
                config.manager_executor_hang_timeout.total_seconds(),
                config.manager_dead_ttl.total_seconds(),
                config.manager_heartbeat_interval,
                config.manager_healthy_ttl.total_seconds(),
                config.manager_hard_exit_timeout.total_seconds(),
            )
            logger.info('InstanceManager %s listening on %s', args.instance_manager_id, endpoint_url)
            await service.wait_stopped()
        finally:
            try:
                if runner is not None and service is not None:
                    await service.stop()
            finally:
                if runner is not None:
                    await runner.cleanup()
                await engine.dispose()


async def _run_dev(args: argparse.Namespace, config: DistributedConfig, *, log_path: Path | None) -> None:
    from aiohttp import web

    from nexus.distributed.config.views import instance_manager_config_view, scheduler_config_view
    from nexus.distributed.control_plane.reconciler import create_reconciler_service
    from nexus.distributed.control_plane.scheduler import SchedulerService
    from nexus.distributed.gateways.application.app import create_application_app
    from nexus.distributed.gateways.control.app import create_control_app
    from nexus.distributed.instance_manager.internal import create_instance_manager_internal_app
    from nexus.distributed.instance_manager.service import InstanceManagerService
    from nexus.distributed.persistence.db import create_engine, create_session_factory, init_database

    engine = create_engine(config)
    session_factory = create_session_factory(engine)
    await init_database(config)
    manager = InstanceManagerService(
        session_factory=session_factory,
        config=instance_manager_config_view(config),
        manager_id=args.instance_manager_id,
        endpoint_url=f'http://{args.host}:{args.instance_manager_port}',
        capacity_total=args.capacity,
        executor=_create_executor(args, config, capacity=args.capacity),
        force_replace=True,
        environment_id=args.environment_id,
        pool_id=args.pool_id,
        workload=args.workload,
        labels=_parse_labels(args.label),
    )
    manager.log_path = str(log_path) if log_path is not None else None
    await manager.start()
    manager.start_background_tasks()

    control_runner = web.AppRunner(create_control_app(session_factory=session_factory, config=config))
    connect_host = _local_connect_host(args.host)
    app_runner = web.AppRunner(
        create_application_app(
            session_factory=session_factory,
            config=config,
            live_compatibility_control_base_url=f'http://{connect_host}:{args.control_port}',
            live_compatibility_application_base_url=f'http://{connect_host}:{args.application_port}',
            live_compatibility_default_placement=_placement_from_environment_args(args),
        )
    )
    manager_runner = web.AppRunner(create_instance_manager_internal_app(service=manager, config=config))
    runners = [control_runner, app_runner, manager_runner]
    try:
        for runner in runners:
            await runner.setup()
        await web.TCPSite(control_runner, args.host, args.control_port).start()
        await web.TCPSite(app_runner, args.host, args.application_port).start()
        await web.TCPSite(manager_runner, args.host, args.instance_manager_port).start()
        scheduler = SchedulerService(
            session_factory=session_factory,
            config=scheduler_config_view(config),
            owner_id='dev-scheduler',
        )
        reconciler = create_reconciler_service(session_factory=session_factory, config=config)
        tasks = [
            asyncio.create_task(scheduler.run_forever()),
            asyncio.create_task(reconciler.run_forever()),
        ]
        logger.info(
            'Distributed dev stack: control=%s application=%s compatibility=%s manager=%s',
            args.control_port,
            args.application_port,
            f'{args.application_port}/compat/live',
            args.instance_manager_port,
        )
        await asyncio.Event().wait()
    finally:
        await manager.stop()
        for runner in runners:
            await runner.cleanup()
        for task in locals().get('tasks', []):
            task.cancel()
        await engine.dispose()


def register_distributed_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser('distributed', help='Run distributed Nexus services')
    distributed_subparsers = parser.add_subparsers(dest='distributed_command')

    statistics_parser = distributed_subparsers.add_parser('statistics', help='Backfill business statistics fields')
    _add_common_arguments(statistics_parser)
    statistics_commands = statistics_parser.add_subparsers(dest='statistics_command', required=True)
    backfill = statistics_commands.add_parser('backfill', help='Preview or apply a bounded, restartable backfill')
    backfill.add_argument('--apply', action='store_true', help='Write recovered fields; default is a read-only preview')
    backfill.add_argument('--batch-size', type=int, default=100)
    backfill.add_argument('--max-pages', type=int, default=1)
    backfill.add_argument('--pause-seconds', type=float, default=0.1)
    backfill.add_argument('--cursor', help='Resume cursor emitted after the last completed page')
    backfill.add_argument('--environment-id')
    backfill.add_argument('--parent-execution-id')
    backfill.add_argument(
        '--with-task-store', action='store_true', help='Read surviving Task Store records using stored configuration'
    )
    backfill.add_argument(
        '--task-list',
        action='append',
        default=[],
        metavar='PARENT_ID=LOCAL_PATH',
        help='Recover names from an existing local task list, verifying each item hash',
    )

    db_parser = distributed_subparsers.add_parser('db', help='Manage distributed database schema')
    _add_common_arguments(db_parser)
    db_subparsers = db_parser.add_subparsers(dest='db_command')
    db_subparsers.add_parser('init', help='Initialize database schema')
    db_subparsers.add_parser('upgrade', help='Upgrade database schema')
    db_shell = db_subparsers.add_parser('shell', help='Open mycli for the distributed database')
    db_shell.add_argument('--prompt-password', action='store_true', help='Let mycli prompt instead of using MYSQL_PWD')
    db_shell.add_argument('mycli_args', nargs=argparse.REMAINDER, help='Extra arguments passed to mycli after --')

    config_parser = distributed_subparsers.add_parser('config', help='Manage distributed startup configuration')
    config_subparsers = config_parser.add_subparsers(dest='config_command', required=True)

    def add_database_argument(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            '--database-url',
            help='SQLAlchemy async database URL, defaults to NEXUS_DIST_DATABASE_URL',
        )

    def add_scope_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument('--role', required=True)
        command_parser.add_argument('--identity', default='*')

    def add_mutation_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument('--expected-version', type=int, required=True)
        command_parser.add_argument('--actor', default='system:config-cli')

    for command_name, help_text in (
        ('set', 'Create or replace one ordinary config scope'),
        ('secret-set', 'Create or replace one secret config scope'),
    ):
        command_parser = config_subparsers.add_parser(command_name, help=help_text)
        add_database_argument(command_parser)
        add_scope_arguments(command_parser)
        add_mutation_arguments(command_parser)
        command_parser.add_argument('--file', required=True, help='JSON file path, or - for stdin')

    config_show = config_subparsers.add_parser('show', help='Show one ordinary config scope')
    add_database_argument(config_show)
    add_scope_arguments(config_show)

    secret_list = config_subparsers.add_parser('secret-list', help='List secret scope metadata without values')
    add_database_argument(secret_list)
    secret_list.add_argument('--role')
    secret_list.add_argument('--identity')

    for command_name, help_text in (
        ('delete', 'Delete one ordinary config scope'),
        ('secret-delete', 'Delete one secret config scope'),
    ):
        command_parser = config_subparsers.add_parser(command_name, help=help_text)
        add_database_argument(command_parser)
        add_scope_arguments(command_parser)
        add_mutation_arguments(command_parser)

    logs_parser = distributed_subparsers.add_parser('logs', help='Read distributed service logs')
    logs_subparsers = logs_parser.add_subparsers(dest='logs_command')
    logs_tail = logs_subparsers.add_parser('tail', help='Tail logs from a running InstanceManager')
    logs_tail.add_argument('--control-base-url', default='http://127.0.0.1:8081')
    logs_tail.add_argument('--component', choices=['instance-manager'], default='instance-manager')
    logs_tail.add_argument('--manager-id', required=True)
    logs_tail.add_argument('--lines', type=int, default=200)
    logs_tail.add_argument('--start-time', help='Only include log records at or after this local timestamp')
    logs_tail.add_argument('--end-time', help='Only include log records at or before this local timestamp')
    logs_tail.add_argument('-f', '--follow', action='store_true')

    control = distributed_subparsers.add_parser('control-gateway', help='Run Control Gateway')
    _add_common_arguments(control)
    _add_web_arguments(control, default_port=8081)

    application = distributed_subparsers.add_parser('application-gateway', help='Run Application Gateway')
    _add_common_arguments(application)
    _add_web_arguments(application, default_port=8082)
    application.add_argument('--enable-live-compat', action='store_true')
    application.add_argument('--control-base-url', default='http://127.0.0.1:8081')
    application.add_argument('--application-base-url')
    _add_placement_arguments(application)

    scheduler = distributed_subparsers.add_parser('scheduler', help='Run Scheduler')
    _add_common_arguments(scheduler)
    scheduler.add_argument('--owner-id', default='scheduler-1')

    reconciler = distributed_subparsers.add_parser('reconciler', help='Run Reconciler')
    _add_common_arguments(reconciler)
    reconciler.add_argument('--owner-id', default='reconciler-1')

    instance_manager = distributed_subparsers.add_parser(
        'instance-manager',
        help='Run predefined InstanceManager',
    )
    _add_common_arguments(instance_manager)
    _add_web_arguments(instance_manager, default_port=8091)
    _add_executor_arguments(instance_manager)
    _add_environment_arguments(instance_manager)
    instance_manager.add_argument('--instance-manager-id', default='instance-manager-1')
    instance_manager.add_argument('--endpoint-url', help='Public internal endpoint URL registered in DB')
    instance_manager.add_argument('--capacity', type=int, default=1)
    instance_manager.add_argument('--require-allocation-liveness-v1', action='store_true')
    instance_manager.add_argument('--force-replace', action='store_true')
    instance_manager.add_argument('--bootstrap-version', type=int)
    instance_manager.add_argument('--artifact-ref-version', type=int)
    instance_manager.add_argument('--request-policy-binding', help='Fixed request_rate policy binding for this Manager')
    instance_manager.add_argument('--request-policy-scope', help='Fixed request_rate policy scope for this Manager')

    dev = distributed_subparsers.add_parser('dev', help='Run a local all-in-one distributed stack')
    _add_common_arguments(dev)
    dev.add_argument('--host', default='127.0.0.1')
    dev.add_argument('--control-port', type=int, default=8081)
    dev.add_argument('--application-port', type=int, default=8082)
    dev.add_argument('--instance-manager-port', type=int, default=8091)
    dev.add_argument('--instance-manager-id', default='dev-instance-manager')
    dev.add_argument('--capacity', type=int, default=4)
    _add_executor_arguments(dev)
    _add_environment_arguments(dev)
