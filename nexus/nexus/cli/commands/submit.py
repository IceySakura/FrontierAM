from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import uuid

from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import aiohttp

from packaging.utils import canonicalize_name


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from nexus.cli.commands.run import add_compose_arguments
from nexus.dataset.auth import WedataKey
from nexus.distributed.batch.data_list import DEFAULT_MAX_DATA_FRAME_BYTES, DEFAULT_MAX_DATA_LIST_BYTES
from nexus.distributed.batch.materialize import (
    DEFAULT_MAX_TASK_RECORD_BYTES,
    MaterializedDataList,
    MaterializedTaskList,
    materialize_compose,
    materialize_generator,
)
from nexus.distributed.submission.specs import (
    DEFAULT_MANAGER_SLOTS,
    BatchManagerResources,
    BatchSubmission,
    LiveSubmission,
    RemoteComposeSpec,
    SingleManagerResources,
    SingleSubmission,
    normalize_host_env,
)
from nexus.distributed.submission.workload_image import validate_workload_image_version
from nexus.distributed.utils.identity import (
    validate_environment_id,
    validate_pool_id,
    validate_submission_idempotency_key,
)
from nexus.runner.compose import compile_single, parse_ordered_parts
from nexus.runner.compose.compose_hooks import parse_compose_hooks, submission_host_env
from nexus.runner.single import RunInput
from nexus.utils.otel.auto import is_requested as is_otel_requested


if TYPE_CHECKING:
    from nexus.distributed.submission.source_package import PackagedFile


_LOCAL_MAX_TASK_LIST_BYTES = 4 * 1024 * 1024 * 1024
_LOCAL_MAX_TASK_LIST_BYTES_ENV = 'NEXUS_SUBMIT_MAX_TASK_LIST_BYTES'
_LOCAL_MAX_DATA_FRAME_BYTES_ENV = 'NEXUS_SUBMIT_MAX_DATA_FRAME_BYTES'
_LOCAL_MAX_TASK_RECORD_BYTES_ENV = 'NEXUS_SUBMIT_MAX_TASK_RECORD_BYTES'
_LOCAL_MAX_DATA_LIST_BYTES_ENV = 'NEXUS_SUBMIT_MAX_DATA_LIST_BYTES'
_SUBMISSION_ATTEMPTS = 3
_WAIT_RETRY_ATTEMPTS = 5
_WAIT_MAX_BACKOFF_SECONDS = 8
_DEFAULT_CONTROL_URL = 'https://control-runner-nexus-agent.mcp.woa.com'
_DEFAULT_CONSOLE_URL = 'https://cloud.nexus-agent.woa.com'
_DYNAMIC_MANAGER_RESOURCE_DEFAULTS = {
    'manager_cpu_cores': 1.0,
    'manager_memory_gib': 2.0,
    'manager_slots': 20,
}
_DYNAMIC_BATCH_RESOURCE_DEFAULTS = {
    'manager_cpu_cores': 1.0,
    'manager_memory_gib': 2.0,
    'batch_manager_cpu_cores': 1.0,
    'batch_manager_memory_gib': 4.0,
}


def build_source_package(root: Path, output_path: Path, *, excluded_paths: list[Path]) -> PackagedFile:
    from nexus.distributed.submission.source_package import build_source_package as build

    return build(root, output_path, excluded_paths=excluded_paths)


def register_submit_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser('submit', help='Submit work to a distributed Nexus cluster')
    modes = parser.add_subparsers(dest='submit_command')

    single = modes.add_parser('single', help='Submit a single run')
    _add_single_or_live_arguments(single)
    single.add_argument('--wait', action='store_true')

    live = modes.add_parser('live', help='Submit a live run')
    _add_single_or_live_arguments(live)

    batch = modes.add_parser('batch', help='Submit a batch run')
    batch.add_argument('--generator', help='Generator Python path, relative to the project root')
    add_compose_arguments(batch, include_batch=True)
    _set_cloud_compose_help(batch)
    batch.add_argument('--generation', choices=['local', 'remote'], default='local')
    batch.add_argument('--wedata-key-file', type=Path, help='Private Wedata key JSON file for remote generation')
    batch.add_argument('--task-list-version', type=int, choices=[1, 2], default=2)
    batch.add_argument('--batch-manager-cpu-cores', type=_positive_float)
    batch.add_argument('--batch-manager-memory-gib', type=_positive_float)
    batch.add_argument('--manager-count', type=_positive_int)
    batch.add_argument('--manager-cpu-cores', type=_positive_float)
    batch.add_argument('--manager-memory-gib', type=_positive_float)
    batch.add_argument('--manager-slots', type=_positive_int)
    batch.add_argument('--max-inflight', type=_positive_int)
    batch.add_argument('--retry-failed', type=_non_negative_int, default=0)
    batch.add_argument('--fail-fast', action='store_true')
    batch.add_argument('--generation-timeout-seconds', type=_positive_int)
    batch.add_argument('--child-queue-ttl-seconds', type=_positive_int)
    batch.add_argument('--timeout-per-run', type=_positive_int)
    _add_common_submit_arguments(batch)
    batch.add_argument('--wait', action='store_true')


def _add_single_or_live_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('config', nargs='?', help='RunInput YAML configuration')
    add_compose_arguments(parser, include_batch=False)
    _set_cloud_compose_help(parser)
    parser.add_argument('--manager-cpu-cores', type=_positive_float)
    parser.add_argument('--manager-memory-gib', type=_positive_float)
    parser.add_argument('--manager-slots', type=_positive_int)
    _add_common_submit_arguments(parser)


def _add_common_submit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--auto-increase-cpu',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Auto-increase Manager CPU to ceil(memory GiB / 4); omitted uses the server default',
    )
    for field in ('self-fence', 'executor-hang', 'dead'):
        parser.add_argument(f'--manager-{field}-timeout-seconds', type=_positive_int)
    parser.add_argument('--environment', required=True, dest='environment_id')
    parser.add_argument('--pool')
    parser.add_argument('--scheduler-policy', help='Authorized Scheduler policy binding')
    parser.add_argument(
        '--policy-scope', action='append', default=[], help='Policy resource scope; repeat for multiple scopes'
    )
    parser.add_argument('--request-policy', help='Authorized request_rate policy binding')
    parser.add_argument('--request-scope', help='Request budget scope name')
    parser.add_argument(
        '--host-env',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='Environment variable for the remote task process; repeatable',
    )
    parser.add_argument('--config-only', action='store_true')
    parser.add_argument('--workload-image-version', help='Version tag in the platform workload image repository')
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--source', type=Path, help='Source package root; defaults to --project-root')
    parser.add_argument('--require-clean', action='store_true')
    parser.add_argument('--wheel-extra', action='append', default=[])
    parser.add_argument(
        '--run-trace-sink',
        action='append',
        default=[],
        help='Granted Run Trace Backend name; repeat to preserve sink order',
    )
    parser.add_argument('--control-url', default=os.getenv('NEXUS_CONTROL_URL', _DEFAULT_CONTROL_URL))
    parser.add_argument('--console-url', default=os.getenv('NEXUS_CONSOLE_URL', _DEFAULT_CONSOLE_URL))
    parser.add_argument('--taihu-token-file', type=Path, help='Private file containing a Taihu personal token')
    parser.add_argument('--taihu-token', help='Taihu personal token; prefer --taihu-token-file')
    parser.add_argument('--idempotency-key')
    parser.add_argument('--provisioning-timeout-seconds', type=_positive_int)
    parser.add_argument('--queue-ttl-seconds', type=_positive_int)
    parser.add_argument('--execution-ttl-seconds', type=_positive_int)
    parser.add_argument('--stop-timeout-seconds', type=_positive_int)
    parser.add_argument('--json', action='store_true')


def _set_cloud_compose_help(parser: argparse.ArgumentParser) -> None:
    for action in parser._actions:
        if '--data' in action.option_strings:
            action.help = 'Dataset Service URI; can be repeated and preserves command-line order.'
            return


def submit_command(args: argparse.Namespace) -> int:
    if not args.submit_command:
        return 0
    accepted: dict[str, Any] = {}

    def on_accepted(result: dict[str, Any]) -> None:
        accepted.update(result)
        if getattr(args, 'wait', False):
            _print_accepted_notice(result, json_output=args.json)

    try:
        result = asyncio.run(_submit(args, on_accepted=on_accepted))
    except KeyboardInterrupt:
        if accepted:
            result = {**accepted, 'wait_interrupted': True}
            _print_submit_result(result, json_output=args.json)
            print('wait interrupted; remote execution was not stopped', file=sys.stderr)
        else:
            print('submit interrupted before acceptance', file=sys.stderr)
        return 130
    except Exception as exc:
        if accepted:
            result = {
                **accepted,
                'wait_error': {
                    'type': type(exc).__name__,
                    'message': str(exc),
                },
            }
            _print_submit_result(result, json_output=args.json)
            print(f'submit accepted but wait failed: {exc}', file=sys.stderr)
            return 1
        print(f'submit failed: {exc}', file=sys.stderr)
        return 1
    _print_submit_result(result, json_output=args.json)
    return 0


def _print_accepted_notice(result: dict[str, Any], *, json_output: bool) -> None:
    notice = {key: result[key] for key in ('execution_id', 'batch_id', 'status') if key in result}
    if json_output:
        print(json.dumps({'event': 'accepted', **notice}, sort_keys=True), file=sys.stderr, flush=True)
        return
    identity = ' '.join(f'{key}={value}' for key, value in notice.items())
    print(f'accepted {identity}'.rstrip(), file=sys.stderr, flush=True)


def _print_submit_result(result: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


async def _submit(
    args: argparse.Namespace,
    *,
    on_accepted: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    idempotency_key = args.idempotency_key or f'req-{uuid.uuid4().hex}'
    validate_submission_idempotency_key(idempotency_key)
    validate_environment_id(args.environment_id)
    if args.pool is not None:
        validate_pool_id(args.pool)
    _validate_submission_args(args)
    project_root = args.project_root.resolve()
    source_root = (args.source or project_root).resolve()
    if args.require_clean:
        _require_clean_worktree(project_root)
    token = _read_taihu_token(args)
    from nexus.dataset.access import credentials_from_env

    data_credentials = credentials_from_env()
    wedata_credential = _read_wedata_credential(args)
    if wedata_credential is not None and 'wedata' in data_credentials:
        raise ValueError('WEDATA_KEY and --wedata-key-file are mutually exclusive')
    wedata_credential = wedata_credential or data_credentials.get('wedata')
    compose_parts = parse_ordered_parts(getattr(args, '_raw_argv', []))
    compose_hooks = parse_compose_hooks(getattr(args, 'hook_values', None))
    if compose_hooks and args.generation == 'remote':
        raise ValueError('--hook requires --generation local')
    compose_payload = (
        _validated_compose_payload(compose_parts, split=args.split, limit=getattr(args, 'limit', None))
        if args.submit_command != 'batch' or args.generation == 'remote'
        else None
    )
    local_batch = args.submit_command == 'batch' and args.generation == 'local'
    local_max_task_list_bytes = get_local_max_task_list_bytes() if local_batch else _LOCAL_MAX_TASK_LIST_BYTES
    local_max_data_frame_bytes = (
        _local_submit_limit(_LOCAL_MAX_DATA_FRAME_BYTES_ENV, DEFAULT_MAX_DATA_FRAME_BYTES)
        if local_batch
        else DEFAULT_MAX_DATA_FRAME_BYTES
    )
    local_max_task_record_bytes = (
        _local_submit_limit(_LOCAL_MAX_TASK_RECORD_BYTES_ENV, DEFAULT_MAX_TASK_RECORD_BYTES)
        if local_batch
        else DEFAULT_MAX_TASK_RECORD_BYTES
    )
    local_max_data_list_bytes = (
        _local_submit_limit(_LOCAL_MAX_DATA_LIST_BYTES_ENV, DEFAULT_MAX_DATA_LIST_BYTES)
        if local_batch
        else DEFAULT_MAX_DATA_LIST_BYTES
    )

    host_env = {item['name']: item['value'] for item in _parse_host_env(args.host_env)}
    with tempfile.TemporaryDirectory(prefix='nexus-submit-') as temporary_dir, submission_host_env(host_env):
        temporary = Path(temporary_dir)
        task_list: MaterializedTaskList | None = None
        data_list: MaterializedDataList | None = None
        run_input: dict[str, Any] | None = None
        generator_relative: str | None = None

        if args.submit_command in {'single', 'live'}:
            if args.config and compose_parts:
                raise ValueError('config cannot be combined with --data/--use/--set')
            if compose_parts:
                compiled = await asyncio.to_thread(
                    compile_single,
                    compose_parts,
                    item_id=args.id,
                    split=args.split,
                )
            elif args.config:
                compiled = await asyncio.to_thread(RunInput.from_file, args.config)
            else:
                raise ValueError('config or ordered compose arguments are required')
            run_input = compiled.model_dump(mode='json')
        else:
            if bool(args.generator) == bool(compose_parts):
                raise ValueError('exactly one of --generator or ordered compose arguments is required')
            if args.generator:
                if args.generation == 'remote' and args.workload_image_version is not None:
                    generator_relative = _resolve_image_generator(args.generator)
                    generator = None
                else:
                    generator, generator_relative = _resolve_generator(args.generator, source_root)
                if args.generation == 'local':
                    assert generator is not None
                    task_list = await asyncio.to_thread(
                        materialize_generator,
                        generator,
                        temporary / 'tasks.jsonl',
                        max_items=None,
                        max_bytes=local_max_task_list_bytes,
                        max_line_bytes=None,
                        task_list_version=args.task_list_version,
                        compose_hooks=compose_hooks,
                    )
            elif args.generation == 'local':
                materialized = await materialize_compose(
                    compose_parts,
                    temporary / 'tasks.jsonl',
                    split=args.split,
                    limit=args.limit,
                    max_items=None,
                    max_bytes=local_max_task_list_bytes,
                    max_line_bytes=None,
                    task_list_version=args.task_list_version,
                    auto_data_list=args.task_list_version == 2,
                    data_list_output_path=temporary / 'data_list.bin',
                    max_data_frame_bytes=local_max_data_frame_bytes,
                    max_task_record_bytes=local_max_task_record_bytes,
                    max_data_list_bytes=local_max_data_list_bytes,
                    reject_local_task_files=args.task_list_version == 1,
                    compose_hooks=compose_hooks,
                )
                task_list = materialized.task_list
                data_list = materialized.data_list

        auth_headers = {'Authorization': f'Bearer {token}'} if token else {}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
            source_ref: dict[str, Any] | None = None
            nexus_wheel_ref: dict[str, Any] | None = None
            uv_lock_ref: dict[str, Any] | None = None
            if not args.config_only and args.workload_image_version is None:
                excluded_paths = [
                    path for path in (args.taihu_token_file, getattr(args, 'wedata_key_file', None)) if path is not None
                ]
                source = build_source_package(source_root, temporary / 'source.tar.zst', excluded_paths=excluded_paths)
                wheel = _build_wheel(project_root, temporary / 'wheel')
                uv_lock = _prepare_uv_lock(project_root)
                source_ref = await _upload(
                    session,
                    args.control_url,
                    source.path,
                    kind='source_package',
                    sha256=source.sha256,
                    size_bytes=source.size_bytes,
                    idempotency_key=_artifact_upload_idempotency_key(
                        idempotency_key,
                        kind='source_package',
                        sha256=source.sha256,
                        size_bytes=source.size_bytes,
                    ),
                    environment_id=args.environment_id,
                    auth_headers=auth_headers,
                )
                nexus_wheel_ref = await _upload(
                    session,
                    args.control_url,
                    wheel.path,
                    kind='nexus_wheel',
                    sha256=wheel.sha256,
                    size_bytes=wheel.size_bytes,
                    idempotency_key=_artifact_upload_idempotency_key(
                        idempotency_key,
                        kind='nexus_wheel',
                        sha256=wheel.sha256,
                        size_bytes=wheel.size_bytes,
                    ),
                    environment_id=args.environment_id,
                    auth_headers=auth_headers,
                )
                uv_lock_ref = await _upload(
                    session,
                    args.control_url,
                    uv_lock.path,
                    kind='uv_lock',
                    sha256=uv_lock.sha256,
                    size_bytes=uv_lock.size_bytes,
                    idempotency_key=_artifact_upload_idempotency_key(
                        idempotency_key,
                        kind='uv_lock',
                        sha256=uv_lock.sha256,
                        size_bytes=uv_lock.size_bytes,
                    ),
                    environment_id=args.environment_id,
                    auth_headers=auth_headers,
                )

            data_list_ref = None
            data_list_manifest = None
            if data_list is not None:
                data_list_ref = await _upload(
                    session,
                    args.control_url,
                    data_list.path,
                    kind='data_list',
                    sha256=data_list.sha256,
                    size_bytes=data_list.size_bytes,
                    idempotency_key=_artifact_upload_idempotency_key(
                        idempotency_key,
                        kind='data_list',
                        sha256=data_list.sha256,
                        size_bytes=data_list.size_bytes,
                    ),
                    environment_id=args.environment_id,
                    auth_headers=auth_headers,
                )
                data_list_manifest = {
                    'version': 1,
                    'frame_count': data_list.frame_count,
                    'size_bytes': data_list.size_bytes,
                    'sha256': data_list.sha256,
                }

            task_list_ref = None
            task_list_manifest = None
            if task_list is not None:
                task_list_ref = await _upload(
                    session,
                    args.control_url,
                    task_list.path,
                    kind='task_list',
                    sha256=task_list.sha256,
                    size_bytes=task_list.size_bytes,
                    idempotency_key=_artifact_upload_idempotency_key(
                        idempotency_key,
                        kind='task_list',
                        sha256=task_list.sha256,
                        size_bytes=task_list.size_bytes,
                    ),
                    environment_id=args.environment_id,
                    auth_headers=auth_headers,
                )
                task_list_manifest = {
                    'version': 1,
                    'record_version': task_list.record_version,
                    'item_count': task_list.total_items,
                    'size_bytes': task_list.size_bytes,
                    'sha256': task_list.sha256,
                    'source': {
                        'mode': 'local_generator' if args.generator else 'local_compose',
                    },
                    'datasets': list(task_list.resolved_datasets),
                }
                if task_list.batch_metadata:
                    task_list_manifest['batch_metadata'] = task_list.batch_metadata

            payload = _submission_payload(
                args,
                idempotency_key=idempotency_key,
                source_ref=source_ref,
                nexus_wheel_ref=nexus_wheel_ref,
                uv_lock_ref=uv_lock_ref,
                run_input=run_input,
                task_list_ref=task_list_ref,
                task_list_manifest=task_list_manifest,
                data_list_ref=data_list_ref,
                data_list_manifest=data_list_manifest,
                generator_relative=generator_relative,
                compose_payload=compose_payload,
                wedata_credential=wedata_credential,
                cos_credential=data_credentials.get('cos'),
            )
            payload = _validate_and_normalize_payload(args.submit_command, payload)
            response = await _post_submission_with_retry(
                session,
                f'{args.control_url.rstrip("/")}/v1/submissions/{args.submit_command}',
                payload,
                auth_headers=auth_headers,
            )
            response = _with_web_ui_urls(
                response,
                command=args.submit_command,
                control_url=args.control_url,
                console_url=args.console_url,
            )
            if on_accepted is not None:
                on_accepted(dict(response))
            if getattr(args, 'wait', False):
                response['terminal'] = _terminal_execution_summary(
                    await _wait_for_execution(
                        session,
                        args.control_url,
                        response['execution_id'],
                        auth_headers=auth_headers,
                    )
                )
            return response


def _with_web_ui_urls(
    response: dict[str, Any],
    *,
    command: str,
    control_url: str,
    console_url: str,
) -> dict[str, Any]:
    result = dict(response)
    execution_id = result.get('execution_id')
    if not isinstance(execution_id, str) or not execution_id:
        return result
    control_base_url = control_url.rstrip('/')
    console_base_url = console_url.rstrip('/')
    encoded_execution_id = quote(execution_id, safe='')
    control_path = f'/v1/executions/{encoded_execution_id}'
    task_path = (
        f'/v1/console/batches/{encoded_execution_id}'
        if command == 'batch'
        else f'/v1/console/executions/{encoded_execution_id}'
    )
    environment_path = '/v1/console/my-environments'
    result['control_url'] = f'{control_base_url}{control_path}'
    if command == 'batch':
        result['batch_url'] = f'{control_base_url}{control_path}/batch'
    result.setdefault('task_web_ui', f'{console_base_url}{task_path}')
    result.setdefault('environment_web_ui', f'{console_base_url}{environment_path}')
    result.setdefault('tracedb_web_ui', f'{console_base_url}{environment_path}')
    return result


def _submission_payload(
    args: argparse.Namespace,
    *,
    idempotency_key: str,
    source_ref: dict[str, Any] | None,
    nexus_wheel_ref: dict[str, Any] | None,
    uv_lock_ref: dict[str, Any] | None,
    run_input: dict[str, Any] | None,
    task_list_ref: dict[str, Any] | None,
    task_list_manifest: dict[str, Any] | None,
    data_list_ref: dict[str, Any] | None,
    data_list_manifest: dict[str, Any] | None,
    generator_relative: str | None,
    compose_payload: dict[str, Any] | None,
    wedata_credential: dict[str, str] | None,
    cos_credential: dict[str, str] | None = None,
) -> dict[str, Any]:
    code: dict[str, Any]
    if args.config_only:
        code = {'mode': 'config_only'}
    elif args.workload_image_version is not None:
        code = {'mode': 'image', 'image_version': args.workload_image_version}
    else:
        if source_ref is None or nexus_wheel_ref is None or uv_lock_ref is None:
            raise ValueError('source_package code requires uploaded source, nexus wheel, and uv.lock artifacts')
        code = {
            'mode': 'source_package',
            'source_package': source_ref,
            'nexus_wheel': nexus_wheel_ref,
            'uv_lock': uv_lock_ref,
            'extras': _source_package_extras(args.wheel_extra),
        }

    payload: dict[str, Any] = {
        'submission_version': 1,
        'idempotency_key': idempotency_key,
        'environment_id': args.environment_id,
        'code': code,
    }
    if getattr(args, 'scheduler_policy', None):
        payload['scheduler_policy'] = {'binding': args.scheduler_policy, 'scopes': args.policy_scope}
    if getattr(args, 'request_policy', None):
        payload['request_policy'] = {'binding': args.request_policy, 'scope': args.request_scope}
    if host_env := _parse_host_env(args.host_env):
        payload['host_env'] = host_env
    if args.pool is not None:
        payload['pool'] = args.pool
    if wedata_credential is not None:
        payload['data_credentials'] = {'wedata': wedata_credential}
    if cos_credential is not None:
        payload.setdefault('data_credentials', {})['cos'] = cos_credential
    if args.run_trace_sink:
        payload['run_trace_backends'] = {'sinks': args.run_trace_sink}
    for field in (
        'provisioning_timeout_seconds',
        'queue_ttl_seconds',
        'execution_ttl_seconds',
        'stop_timeout_seconds',
    ):
        if (value := getattr(args, field)) is not None:
            payload[field] = value

    if args.pool is None:
        if args.submit_command in {'single', 'live'}:
            payload['resources'] = {
                'managers': {
                    'count': 1,
                    'cpu_cores': args.manager_cpu_cores,
                    'memory_gib': args.manager_memory_gib,
                    'slots': args.manager_slots,
                }
            }
        else:
            payload['resources'] = {
                'batch_manager': {
                    'cpu_cores': args.batch_manager_cpu_cores,
                    'memory_gib': args.batch_manager_memory_gib,
                },
                'managers': {
                    'count': args.manager_count,
                    'cpu_cores': args.manager_cpu_cores,
                    'memory_gib': args.manager_memory_gib,
                    'slots': args.manager_slots,
                },
            }

    if getattr(args, 'auto_increase_cpu', None) is not None and args.pool is None:
        for resource in payload['resources'].values():
            resource['auto_increase_cpu'] = args.auto_increase_cpu
    if liveness := _manager_liveness_args(args):
        payload['resources']['managers']['liveness'] = liveness

    if args.submit_command in {'single', 'live'}:
        payload['run_input'] = run_input
        if args.submit_command == 'live':
            payload['live'] = {}
        return payload

    generation: dict[str, Any]
    if task_list_ref is not None:
        if task_list_manifest is None:
            raise ValueError('local task list manifest is required')
        generation = {
            'mode': 'local_task_list',
            'task_list': {'ref': task_list_ref, 'format': 'jsonl', 'manifest': task_list_manifest},
        }
        if data_list_ref is not None:
            if data_list_manifest is None:
                raise ValueError('local data list manifest is required')
            generation['data_list'] = {
                'ref': data_list_ref,
                'format': 'nxdl',
                'manifest': data_list_manifest,
            }
        elif data_list_manifest is not None:
            raise ValueError('local data list ref is required')
    else:
        generation = {
            'mode': 'remote_materialize',
            'source': (
                {
                    'type': 'generator',
                    'path': generator_relative,
                    'task_list_version': args.task_list_version,
                }
                if generator_relative is not None
                else {
                    'type': 'compose',
                    'compose': compose_payload,
                    'task_list_version': args.task_list_version,
                }
            ),
        }
    if args.generation_timeout_seconds is not None:
        generation['timeout_seconds'] = args.generation_timeout_seconds
    payload['generation'] = generation

    batch: dict[str, Any] = {
        'retry_failed': args.retry_failed,
        'fail_fast': args.fail_fast,
    }
    if args.max_inflight is not None:
        batch['max_inflight'] = args.max_inflight
    if args.child_queue_ttl_seconds is not None:
        batch['child_queue_ttl_seconds'] = args.child_queue_ttl_seconds
    if args.timeout_per_run is not None:
        batch['timeout_per_run'] = args.timeout_per_run
    payload['batch'] = batch
    return payload


def _source_package_extras(extras: list[str]) -> list[str]:
    resolved = set(extras)
    if is_otel_requested():
        resolved.add('otel')
    return sorted(resolved)


def _manager_liveness_args(args: argparse.Namespace) -> dict[str, int] | None:
    from nexus.distributed.environment.liveness import ManagerLivenessPolicy

    values = {
        f'{name}_timeout_seconds': getattr(args, f'manager_{name}_timeout_seconds', None)
        for name in ('self_fence', 'executor_hang', 'dead')
    }
    if all(value is None for value in values.values()):
        return None
    if any(value is None for value in values.values()):
        raise ValueError('all three Manager liveness timeouts must be provided together')
    if args.pool is not None:
        raise ValueError('Manager liveness overrides require dynamic resources')
    return ManagerLivenessPolicy.model_validate(values).model_dump()


def _validate_submission_args(args: argparse.Namespace) -> None:
    if bool(getattr(args, 'scheduler_policy', None)) != bool(getattr(args, 'policy_scope', [])):
        raise ValueError('--scheduler-policy and --policy-scope must be provided together')
    if bool(getattr(args, 'request_policy', None)) != bool(getattr(args, 'request_scope', None)):
        raise ValueError('--request-policy and --request-scope must be provided together')
    _manager_liveness_args(args)
    host_env = _parse_host_env(args.host_env)
    if host_env and args.pool is not None:
        raise ValueError('--host-env cannot be combined with --pool')
    if args.workload_image_version is not None:
        validate_workload_image_version(args.workload_image_version)
        if args.config_only or args.source is not None or args.wheel_extra:
            raise ValueError(
                '--workload-image-version cannot be combined with --config-only, --source, or --wheel-extra'
            )
    if args.config_only and (args.source is not None or args.wheel_extra):
        raise ValueError('--config-only cannot be combined with --source or --wheel-extra')

    if args.pool is None:
        defaults = (
            _DYNAMIC_BATCH_RESOURCE_DEFAULTS if args.submit_command == 'batch' else _DYNAMIC_MANAGER_RESOURCE_DEFAULTS
        )
        for field, default in defaults.items():
            if getattr(args, field) is None:
                setattr(args, field, default)
        if args.submit_command == 'batch':
            _resolve_dynamic_batch_capacity(args)

    resource_fields = ['manager_cpu_cores', 'manager_memory_gib', 'manager_slots']
    if args.submit_command == 'batch':
        resource_fields.extend([
            'batch_manager_cpu_cores',
            'batch_manager_memory_gib',
            'manager_count',
        ])
    supplied_resources = [field for field in resource_fields if getattr(args, field) is not None]

    if args.pool is not None:
        if getattr(args, 'auto_increase_cpu', None) is not None:
            raise ValueError('--auto-increase-cpu requires dynamic resources')
        if not args.config_only:
            raise ValueError('predefined --pool submissions require --config-only')
        if supplied_resources:
            raise ValueError('--pool cannot be combined with dynamic resource flags')
    else:
        missing = [field for field in resource_fields if getattr(args, field) is None]
        if missing:
            options = ', '.join(f'--{field.replace("_", "-")}' for field in missing)
            raise ValueError(f'dynamic submission requires resource flags: {options}')
        auto_increase_cpu = getattr(args, 'auto_increase_cpu', None)
        managers = SingleManagerResources(
            cpu_cores=args.manager_cpu_cores,
            memory_gib=args.manager_memory_gib,
            slots=args.manager_slots,
            auto_increase_cpu=auto_increase_cpu,
        )
        args.manager_cpu_cores = managers.cpu_cores
        if args.submit_command == 'batch':
            batch_manager = BatchManagerResources(
                cpu_cores=args.batch_manager_cpu_cores,
                memory_gib=args.batch_manager_memory_gib,
                auto_increase_cpu=auto_increase_cpu,
            )
            args.batch_manager_cpu_cores = batch_manager.cpu_cores

    if args.submit_command == 'batch':
        if args.wedata_key_file is not None and args.generation != 'remote':
            raise ValueError('--wedata-key-file can only be used with --generation remote')
        if args.generation == 'remote' and args.generator and args.config_only:
            raise ValueError('remote generator requires source-package or workload-image code mode')
        if args.pool is None:
            capacity = args.manager_count * args.manager_slots
            if args.max_inflight is not None and args.max_inflight > capacity:
                raise ValueError('--max-inflight cannot exceed --manager-count * --manager-slots')


def _resolve_dynamic_batch_capacity(args: argparse.Namespace) -> None:
    if args.max_inflight is None:
        if args.manager_count is None:
            args.manager_count = 1
        if args.manager_slots is None:
            args.manager_slots = DEFAULT_MANAGER_SLOTS
        return
    if args.manager_count is None and args.manager_slots is None:
        args.manager_slots = DEFAULT_MANAGER_SLOTS
    if args.manager_count is None:
        args.manager_count = _ceil_div(args.max_inflight, args.manager_slots)
    elif args.manager_slots is None:
        args.manager_slots = _ceil_div(args.max_inflight, args.manager_count)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _parse_host_env(values: list[str]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for raw in values:
        name, separator, value = raw.partition('=')
        if not separator:
            raise ValueError('--host-env must use KEY=VALUE syntax')
        entries.append({'name': name, 'value': value})
    return normalize_host_env(entries)


def _validated_compose_payload(parts: list[Any], *, split: str | None, limit: int | None) -> dict[str, Any] | None:
    if not parts:
        return None
    from nexus.runner.compose.parser import parse_use_spec
    from nexus.runner.compose.resolver import resolve_non_data_part

    portable = []
    for part in parts:
        if part.kind == 'use' and Path(parse_use_spec(part.value).name).expanduser().is_file():
            resolved = resolve_non_data_part(part)
            portable.append({
                'kind': 'input',
                'value': resolved.body,
                'merge': resolved.merge,
                'source': '<inline profile>',
            })
        else:
            portable.append({'kind': part.kind, 'value': part.value, 'source': part.source, 'merge': part.merge})
    return RemoteComposeSpec.model_validate({
        'parts': portable,
        'split': split,
        'limit': limit,
    }).model_dump(mode='json', exclude_none=True)


def _validate_and_normalize_payload(command: str, payload: dict[str, Any]) -> dict[str, Any]:
    model = {
        'single': SingleSubmission,
        'live': LiveSubmission,
        'batch': BatchSubmission,
    }[command].model_validate(payload)
    normalized = model.model_dump(mode='json', exclude_none=True)
    if model.data_credentials is not None:
        normalized['data_credentials'] = model.data_credentials.storage_payloads()
    return normalized


def _resolve_generator(value: str, source_root: Path) -> tuple[Path, str]:
    generator = Path(value)
    if not generator.is_absolute():
        generator = source_root / generator
    generator = generator.resolve()
    try:
        relative = generator.relative_to(source_root).as_posix()
    except ValueError as exc:
        raise ValueError('generator must be inside the source root') from exc
    if not generator.is_file():
        raise ValueError(f'generator file does not exist: {value}')
    return generator, relative


def _resolve_image_generator(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '\\' in value or any(part in {'', '.', '..'} for part in path.parts):
        raise ValueError('generator must be a safe path relative to /workspace')
    return value


def _artifact_upload_idempotency_key(
    submission_key: str,
    *,
    kind: str,
    sha256: str,
    size_bytes: int,
) -> str:
    validate_submission_idempotency_key(submission_key)
    if not kind or not sha256 or isinstance(size_bytes, bool) or size_bytes < 0:
        raise ValueError('invalid artifact upload identity')
    digest = hashlib.sha256(
        f'nexus-artifact-upload:v1\0{submission_key}\0{kind}\0{sha256}\0{size_bytes}'.encode()
    ).hexdigest()
    return f'artifact:{digest}'


async def _post_submission_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    *,
    auth_headers: dict[str, str],
) -> dict[str, Any]:
    for attempt in range(_SUBMISSION_ATTEMPTS):
        try:
            return await _request_json(session, 'POST', url, json=payload, headers=auth_headers)
        except _HttpResponseError as exc:
            if exc.status < 500 or attempt + 1 == _SUBMISSION_ATTEMPTS:
                raise
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
            if attempt + 1 == _SUBMISSION_ATTEMPTS:
                raise
        await asyncio.sleep(2**attempt)
    raise AssertionError('submission retry loop exhausted')


async def _upload(
    session: aiohttp.ClientSession,
    control_url: str,
    path: Path,
    *,
    kind: str,
    sha256: str,
    size_bytes: int,
    idempotency_key: str,
    environment_id: str,
    auth_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    control_headers = auth_headers or {}
    upload = await _request_json(
        session,
        'POST',
        f'{control_url.rstrip("/")}/v1/artifact-uploads',
        json={
            'kind': kind,
            'sha256': sha256,
            'size_bytes': size_bytes,
            'idempotency_key': idempotency_key,
            'environment_id': environment_id,
        },
        headers=control_headers,
    )
    if ref := upload.get('ref'):
        return ref
    upload_id = upload['upload']['upload_id']
    ticket = upload['upload_ticket']
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with path.open('rb') as stream:
                return (
                    await _request_json(
                        session,
                        'PUT',
                        upload['upload_url'],
                        data=stream,
                        headers={
                            **control_headers,
                            'X-Nexus-Upload-Ticket': ticket,
                            'Content-Length': str(size_bytes),
                            'Content-Type': 'application/octet-stream',
                        },
                    )
                )['ref']
        except Exception as exc:
            last_error = exc
        await asyncio.sleep(2**attempt)
        status = await _request_json(
            session,
            'GET',
            f'{control_url.rstrip("/")}/v1/artifact-uploads/{upload_id}',
            headers=control_headers,
            params={'environment_id': environment_id},
        )
        if ref := status.get('ref'):
            return ref
        renewed = await _request_json(
            session,
            'POST',
            f'{control_url.rstrip("/")}/v1/artifact-uploads/{upload_id}/ticket',
            headers=control_headers,
            params={'environment_id': environment_id},
        )
        ticket = renewed['upload_ticket']
        upload['upload_url'] = renewed['upload_url']
    raise RuntimeError(f'artifact upload did not complete: {last_error}')


class _HttpResponseError(RuntimeError):
    def __init__(self, method: str, url: str, status: int, payload: object) -> None:
        super().__init__(f'{method} {url} returned {status}: {payload}')
        self.status = status


async def _request_json(session: aiohttp.ClientSession, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    async with session.request(method, url, **kwargs) as response:
        text = await response.text()
        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = {'body': text}
        if response.status >= 400:
            raise _HttpResponseError(method, url, response.status, payload)
        if not isinstance(payload, dict):
            raise RuntimeError(f'{method} {url} returned a non-object response')
        return payload


async def _wait_for_execution(
    session: aiohttp.ClientSession,
    control_url: str,
    execution_id: str,
    *,
    auth_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    terminal = {'succeeded', 'failed', 'killed', 'lost', 'expired'}
    consecutive_failures = 0
    while True:
        try:
            execution = await _request_json(
                session,
                'GET',
                f'{control_url.rstrip("/")}/v1/executions/{execution_id}',
                headers=auth_headers or {},
            )
        except _HttpResponseError as exc:
            if exc.status not in {408, 425, 429} and exc.status < 500:
                raise
            consecutive_failures += 1
            if consecutive_failures >= _WAIT_RETRY_ATTEMPTS:
                raise
            await asyncio.sleep(min(2 ** (consecutive_failures - 1), _WAIT_MAX_BACKOFF_SECONDS))
            continue
        except (aiohttp.ClientError, asyncio.TimeoutError):
            consecutive_failures += 1
            if consecutive_failures >= _WAIT_RETRY_ATTEMPTS:
                raise
            await asyncio.sleep(min(2 ** (consecutive_failures - 1), _WAIT_MAX_BACKOFF_SECONDS))
            continue
        consecutive_failures = 0
        if execution.get('status') in terminal:
            return execution
        await asyncio.sleep(1)


def _terminal_execution_summary(execution: dict[str, Any]) -> dict[str, Any]:
    fields = (
        'execution_id',
        'status',
        'phase',
        'allocation_id',
        'current_assignment_id',
        'current_attempt_id',
        'attempt_count',
        'result_artifact_id',
        'error_json',
        'stop_reason',
        'cleanup_done_at',
        'created_at',
        'updated_at',
    )
    return {field: execution[field] for field in fields if field in execution}


def _build_wheel(project_root: Path, output_dir: Path) -> PackagedFile:
    from nexus.distributed.submission.source_package import PackagedFile

    _validate_nexus_project(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(['uv', 'build', '--wheel', '--out-dir', str(output_dir)], cwd=project_root, check=True)
    wheels = list(output_dir.glob('*.whl'))
    if len(wheels) != 1:
        raise RuntimeError(f'expected one wheel, found {len(wheels)}')
    path = wheels[0]
    return PackagedFile(
        path=path,
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
        file_count=1,
        unpacked_size_bytes=path.stat().st_size,
    )


def _validate_nexus_project(project_root: Path) -> None:
    pyproject = project_root / 'pyproject.toml'
    if not pyproject.is_file():
        raise ValueError('source-package submit requires project-root/pyproject.toml for nexus-agent')
    with pyproject.open('rb') as stream:
        project = tomllib.load(stream).get('project') or {}
    if canonicalize_name(str(project.get('name') or '')) != 'nexus-agent':
        raise ValueError('source-package submit project must be nexus-agent')


def _prepare_uv_lock(project_root: Path) -> PackagedFile:
    from nexus.distributed.submission.source_package import PackagedFile

    path = project_root / 'uv.lock'
    if not path.is_file():
        raise ValueError('source-package submit requires project-root/uv.lock')
    subprocess.run(['uv', 'lock', '--check'], cwd=project_root, check=True)
    return PackagedFile(
        path=path,
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
        file_count=1,
        unpacked_size_bytes=path.stat().st_size,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_taihu_token(args: argparse.Namespace) -> str | None:
    if args.taihu_token and args.taihu_token_file:
        raise ValueError('--taihu-token and --taihu-token-file are mutually exclusive')
    if args.taihu_token:
        return args.taihu_token.strip()
    if args.taihu_token_file is None:
        return None
    path = args.taihu_token_file.expanduser().resolve()
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ValueError('--taihu-token-file must not be readable or writable by group/other')
    token = path.read_text(encoding='utf-8').strip()
    if not token:
        raise ValueError('--taihu-token-file is empty')
    return token


def _read_wedata_credential(args: argparse.Namespace) -> dict[str, str] | None:
    path_value = getattr(args, 'wedata_key_file', None)
    if path_value is None:
        return None
    path = path_value.expanduser().resolve()
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ValueError('--wedata-key-file must not be readable or writable by group/other')
    raw = path.read_text(encoding='utf-8').strip()
    if not raw:
        raise ValueError('--wedata-key-file is empty')
    return WedataKey.from_raw(raw, source=str(path)).to_auth_payload()


def _require_clean_worktree(project_root: Path) -> None:
    result = subprocess.run(
        ['git', 'status', '--porcelain'],
        cwd=project_root,
        check=True,
        text=True,
        capture_output=True,
    )
    if result.stdout.strip():
        raise ValueError('project worktree is not clean')


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return parsed


def get_local_max_task_list_bytes() -> int:
    return _local_submit_limit(_LOCAL_MAX_TASK_LIST_BYTES_ENV, _LOCAL_MAX_TASK_LIST_BYTES)


def _local_submit_limit(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f'{name} must be a positive integer') from None
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError('must be a non-negative integer')
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return parsed
