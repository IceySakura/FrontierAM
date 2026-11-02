from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import stat
import time
import uuid

from pathlib import Path
from typing import Any

from tests.e2e.distributed.run_tkex_cloud_batch import GongfengReleaseObserver
from tests.integration.distributed.cloud_batch_api import CloudBatchApi, CloudBatchApiError

from nexus.dataset.auth import clear_wedata_key_cache
from nexus.dataset.client import DatasetClient
from nexus.dataset.uri import parse_dataset_uri
from nexus.distributed.environment.kubernetes import WorkloadNotFoundError
from nexus.distributed.environment.tkex import TkexWorkloadClient


DEFAULT_DATASET_URI = 'nexus-test/terminal-bench-2@v4'
DEFAULT_GONGFENG_API_BASE_URL = 'http://hyrl-sandbox.prod.woa.com:8052'
TERMINAL_EXECUTION_STATUSES = {'succeeded', 'failed', 'killed', 'lost', 'expired'}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the real Taiji Dataset distributed batch E2E.')
    parser.add_argument('--control-url', required=True)
    parser.add_argument('--application-url', required=True)
    parser.add_argument('--environment-id', required=True)
    parser.add_argument('--taihu-token-file', type=Path, required=True)
    parser.add_argument('--wedata-key-file', type=Path, required=True)
    parser.add_argument('--wedata-secret-name', required=True)
    parser.add_argument('--wedata-secret-key', default='key')
    parser.add_argument('--dataset-uri', default=DEFAULT_DATASET_URI)
    parser.add_argument('--limit', type=_positive_int, default=2)
    parser.add_argument('--page-size', type=_positive_int, default=1)
    parser.add_argument('--runtime-api-base-url', default=DEFAULT_GONGFENG_API_BASE_URL)
    parser.add_argument('--gongfeng-token-file', type=Path, required=True)
    parser.add_argument('--tkex-base-url', required=True)
    parser.add_argument('--tkex-api-account', required=True)
    parser.add_argument('--tkex-api-token-file', type=Path, required=True)
    parser.add_argument('--tkex-cluster-name', required=True)
    parser.add_argument('--tkex-project-name', required=True)
    parser.add_argument('--tkex-namespace', required=True)
    parser.add_argument('--expected-workload-image')
    parser.add_argument('--poll-interval', type=_positive_float, default=2.0)
    parser.add_argument('--timeout-seconds', type=_positive_float, default=7200)
    parser.add_argument('--report-path', type=Path, required=True)
    return parser.parse_args(argv)


def dataset_batch_submission(
    *,
    environment_id: str,
    dataset_uri: str,
    limit: int,
    page_size: int,
    runtime_api_base_url: str,
    runtime_batch_id: str,
) -> dict[str, Any]:
    parsed = parse_dataset_uri(dataset_uri)
    if parsed.version in {'latest', 'dev'}:
        raise ValueError('real Dataset E2E requires a fixed release version')
    if page_size >= limit:
        raise ValueError('page_size must be smaller than limit so the E2E crosses a Dataset page boundary')
    runtime_config = {
        'api_base_url': runtime_api_base_url,
        'batch_id': runtime_batch_id,
        'purpose': 'test',
        'auto_connect_ws': False,
        'print_port_mappings': False,
        'queue_wait_max': 1800,
    }
    compose = {
        'parts': [
            {'kind': 'data', 'value': dataset_uri, 'source': f'--data {dataset_uri}'},
            {
                'kind': 'set_json',
                'value': f'imports={json.dumps(["nexus.extensions.tasks.terminal.task"])}',
                'source': 'taiji-dataset-e2e imports',
            },
            {
                'kind': 'set',
                'value': 'runtime_provider_name=gongfeng',
                'source': 'taiji-dataset-e2e runtime provider',
            },
            {
                'kind': 'set_json',
                'value': f'runtime_provider_config={json.dumps(runtime_config, sort_keys=True)}',
                'source': 'taiji-dataset-e2e runtime provider config',
            },
            {
                'kind': 'set_json',
                'value': 'task_config.run_oracle=true',
                'source': 'taiji-dataset-e2e oracle',
            },
            {
                'kind': 'set_json',
                'value': 'runner.timeout=5400',
                'source': 'taiji-dataset-e2e runner timeout',
            },
        ],
        'limit': limit,
        'page_size': page_size,
    }
    return {
        'submission_version': 1,
        'idempotency_key': f'taiji-dataset-e2e-{uuid.uuid4().hex}',
        'environment_id': environment_id,
        'code': {'mode': 'config_only'},
        'resources': {
            'batch_manager': {'cpu_cores': 1.0, 'memory_gib': 2.0},
            'managers': {
                'count': limit,
                'cpu_cores': 1.0,
                'memory_gib': 1.0,
                'slots': 1,
            },
        },
        'generation': {
            'mode': 'remote_materialize',
            'source': {'type': 'compose', 'compose': compose, 'task_list_version': 2},
            'timeout_seconds': 900,
        },
        'batch': {
            'max_inflight': limit,
            'retry_failed': 0,
            'fail_fast': False,
            'child_queue_ttl_seconds': 1800,
            'timeout_per_run': 5400,
        },
        'provisioning_timeout_seconds': 1800,
        'queue_ttl_seconds': 1800,
        'execution_ttl_seconds': 7200,
        'stop_timeout_seconds': 600,
    }


async def preflight_dataset(
    dataset_uri: str,
    *,
    wedata_key_file: Path,
    limit: int,
    page_size: int,
) -> dict[str, Any]:
    key_path = _readable_secret_file(wedata_key_file, label='Wedata key')
    previous_key_file = os.environ.get('WEDATA_KEY_FILE')
    previous_raw_key = os.environ.pop('WEDATA_KEY', None)
    os.environ['WEDATA_KEY_FILE'] = str(key_path)
    clear_wedata_key_cache()
    try:
        async with DatasetClient(timeout=120) as client:
            shown = await client.show(dataset_uri)
            versions = await client.versions(dataset_uri)
            ref = await client.resolve_ref(dataset_uri)
            items = [item async for item in client.iter_resolved_items(ref, limit=limit, page_size=page_size)]
    finally:
        if previous_key_file is None:
            os.environ.pop('WEDATA_KEY_FILE', None)
        else:
            os.environ['WEDATA_KEY_FILE'] = previous_key_file
        if previous_raw_key is not None:
            os.environ['WEDATA_KEY'] = previous_raw_key
        clear_wedata_key_cache()

    if len(items) != limit:
        raise AssertionError(f'Dataset preflight returned {len(items)} items, expected {limit}')
    item_summaries = []
    for item in items:
        task_files = item.resolved_task_files()
        if task_files is None or task_files.remote is None:
            raise AssertionError(f'Dataset item {item.id} has no remote task_files')
        if item.body.get('task_name') != 'terminal_task':
            raise AssertionError(f'Dataset item {item.id} is not terminal_task')
        materialized_metadata = dict(item.metadata)
        materialized_metadata.setdefault('dataset_item_id', item.id)
        item_summaries.append(
            {
                'item_id': item.id,
                'task_name': item.body['task_name'],
                'task_file': task_files.model_dump(exclude_none=True),
                'metadata_keys': sorted(materialized_metadata),
                'metadata_sha256': hashlib.sha256(
                    json.dumps(
                        materialized_metadata,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(',', ':'),
                    ).encode()
                ).hexdigest(),
            }
        )
    resolved_uri = f'{ref.namespace}/{ref.dataset}@{ref.version}'
    if ref.split:
        resolved_uri = f'{resolved_uri}?split={ref.split}'
    return {
        'dataset_uri': dataset_uri,
        'resolved_ref': ref.model_dump(exclude_none=True),
        'resolved_uri': resolved_uri,
        'uc_asset_keys': sorted((shown.get('asset') or {}).keys()),
        'io_overview_keys': sorted((shown.get('overview') or {}).keys()),
        'version_summary': {
            'latest_release': versions.get('latest_release'),
            'current_dev': versions.get('current_dev'),
            'release_count': len(versions.get('versions') or []),
        },
        'page_size': page_size,
        'minimum_pages': math.ceil(limit / page_size),
        'items': item_summaries,
    }


async def execute(args: argparse.Namespace) -> dict[str, Any]:
    report_path = args.report_path.resolve()
    report: dict[str, Any] = {
        'status': 'running',
        'target': {
            'control_url': args.control_url,
            'application_url': args.application_url,
            'environment_id': args.environment_id,
            'tkex_cluster_name': args.tkex_cluster_name,
            'tkex_namespace': args.tkex_namespace,
        },
        'preflight': None,
        'submission': None,
        'execution_id': None,
        'workloads': {},
        'validation': None,
        'gongfeng_cleanup': None,
        'cleanup': None,
    }
    _write_report(report_path, report)

    runtime_batch_id = f'nexus-taiji-dataset-e2e-{uuid.uuid4().hex}'
    payload = dataset_batch_submission(
        environment_id=args.environment_id,
        dataset_uri=args.dataset_uri,
        limit=args.limit,
        page_size=args.page_size,
        runtime_api_base_url=args.runtime_api_base_url,
        runtime_batch_id=runtime_batch_id,
    )
    report['submission'] = _sanitized_submission(payload)
    execution_id: str | None = None
    api: CloudBatchApi | None = None
    tkex = _create_tkex_workload_client(args)
    gongfeng = GongfengReleaseObserver(
        api_base_url=args.runtime_api_base_url,
        token_file=_readable_secret_file(args.gongfeng_token_file, label='Gongfeng token'),
    )
    try:
        report['preflight'] = await preflight_dataset(
            args.dataset_uri,
            wedata_key_file=args.wedata_key_file,
            limit=args.limit,
            page_size=args.page_size,
        )
        expected_items = {item['item_id']: item for item in report['preflight']['items']}
        _write_report(report_path, report)

        token = _readable_secret_file(args.taihu_token_file, label='Taihu token').read_text(encoding='utf-8').strip()
        api = CloudBatchApi(control_url=args.control_url, application_url=args.application_url, token=token)
        await api.__aenter__()
        await api.control_json('GET', '/v1/health')
        await api.application_json('GET', '/v1/health')
        submitted = await api.control_json('POST', '/v1/submissions/batch', body=payload)
        execution_id = str(submitted['execution_id'])
        report['execution_id'] = execution_id
        _write_report(report_path, report)

        execution, allocations = await _wait_for_terminal(
            api,
            tkex,
            execution_id=execution_id,
            report=report,
            report_path=report_path,
            expected_wedata_secret=args.wedata_secret_name,
            expected_wedata_secret_key=args.wedata_secret_key,
            expected_workload_image=args.expected_workload_image,
            timeout=args.timeout_seconds,
            poll_interval=args.poll_interval,
        )
        if execution['status'] != 'succeeded':
            raise AssertionError(f'Taiji Dataset batch ended as {execution["status"]}: {execution.get("error_json")}')
        report['validation'] = await _validate_completed_batch(
            api,
            execution_id=execution_id,
            expected_items=expected_items,
            expected_dataset_uri=args.dataset_uri,
            expected_resolved_uri=report['preflight']['resolved_uri'],
            expected_page_size=args.page_size,
        )
        report['gongfeng_cleanup'] = await gongfeng.wait_batch_released(
            runtime_batch_id,
            minimum_runtime_count=args.limit,
            timeout=1800,
            poll_interval=10,
        )
        report['cleanup'] = await _wait_for_cleanup(
            api,
            tkex,
            execution_id=execution_id,
            allocations=allocations,
            timeout=1800,
            poll_interval=args.poll_interval,
        )
        report['status'] = 'passed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        if api is not None and execution_id is not None:
            with contextlib.suppress(BaseException):
                execution = await api.control_json('GET', f'/v1/executions/{execution_id}')
                if execution.get('status') not in TERMINAL_EXECUTION_STATUSES:
                    await api.control_json('POST', f'/v1/executions/{execution_id}/stop')
            with contextlib.suppress(BaseException):
                allocations_response = await api.control_json('GET', f'/v1/executions/{execution_id}/allocations')
                report['cleanup'] = await _wait_for_cleanup(
                    api,
                    tkex,
                    execution_id=execution_id,
                    allocations=allocations_response.get('items') or [],
                    timeout=1800,
                    poll_interval=args.poll_interval,
                )
        with contextlib.suppress(BaseException):
            report['gongfeng_cleanup'] = await gongfeng.wait_batch_released(
                runtime_batch_id,
                minimum_runtime_count=0,
                timeout=900,
                poll_interval=10,
            )
        raise
    finally:
        report['finished_at'] = _utc_timestamp()
        _write_report(report_path, report)
        if api is not None:
            await api.close()
        await tkex.close()
    return report


async def _wait_for_terminal(
    api: CloudBatchApi,
    tkex: TkexWorkloadClient,
    *,
    execution_id: str,
    report: dict[str, Any],
    report_path: Path,
    expected_wedata_secret: str,
    expected_wedata_secret_key: str,
    expected_workload_image: str | None,
    timeout: float,
    poll_interval: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deadline = time.monotonic() + timeout
    latest_allocations: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        execution = await api.control_json('GET', f'/v1/executions/{execution_id}')
        allocation_response = await api.control_json('GET', f'/v1/executions/{execution_id}/allocations')
        latest_allocations = allocation_response.get('items') or []
        for allocation in latest_allocations:
            workload = str(allocation.get('workload') or '')
            if workload in report['workloads']:
                continue
            observed = await _observe_workload(
                tkex,
                allocation,
                expected_wedata_secret=expected_wedata_secret,
                expected_wedata_secret_key=expected_wedata_secret_key,
                expected_workload_image=expected_workload_image,
            )
            if observed is not None:
                report['workloads'][workload] = observed
                _write_report(report_path, report)
        if execution.get('status') in TERMINAL_EXECUTION_STATUSES:
            if set(report['workloads']) != {'execution', 'orchestration'}:
                raise AssertionError(f'did not observe both Manager workload roles: {sorted(report["workloads"])}')
            return execution, latest_allocations
        await asyncio.sleep(poll_interval)
    raise TimeoutError(f'Taiji Dataset batch did not finish within {timeout:g}s')


async def _observe_workload(
    tkex: TkexWorkloadClient,
    allocation: dict[str, Any],
    *,
    expected_wedata_secret: str,
    expected_wedata_secret_key: str,
    expected_workload_image: str | None,
) -> dict[str, Any] | None:
    reference = (allocation.get('provider_resource_ref') or {}).get('workload') or {}
    name = str(reference.get('name') or '')
    if not name:
        return None
    try:
        resource = await tkex.get_stateful_workload(name)
    except WorkloadNotFoundError:
        return None
    containers = (((resource.get('spec') or {}).get('template') or {}).get('spec') or {}).get('containers') or []
    if len(containers) != 1:
        raise AssertionError(f'workload {name} has {len(containers)} containers')
    container = containers[0]
    image = str(container.get('image') or '')
    if expected_workload_image and image != expected_workload_image:
        raise AssertionError(f'workload {name} used unexpected image {image}')
    env = {str(item.get('name')): item for item in container.get('env') or []}
    pod_spec = ((resource.get('spec') or {}).get('template') or {}).get('spec') or {}
    secret_sources = []
    for volume in pod_spec.get('volumes') or []:
        if volume.get('name') == 'nexus-secrets':
            secret_sources = (volume.get('projected') or {}).get('sources') or []
            break
    wedata_sources = [
        source for source in secret_sources if ((source.get('secret') or {}).get('name')) == expected_wedata_secret
    ]
    role = str(allocation.get('workload') or '')
    if role in {'orchestration', 'execution'}:
        if (env.get('WEDATA_KEY_FILE') or {}).get('value') != '/var/run/secrets/nexus/auth/wedata/key':
            raise AssertionError(f'{role} Manager has no canonical WEDATA_KEY_FILE')
        if len(wedata_sources) != 1:
            raise AssertionError(f'{role} Manager has no unique Wedata Secret projection')
        items = (wedata_sources[0].get('secret') or {}).get('items') or []
        if items != [{'key': expected_wedata_secret_key, 'path': 'auth/wedata/key'}]:
            raise AssertionError(f'{role} Manager has unexpected Wedata Secret items: {items}')
    else:
        raise AssertionError(f'unexpected allocation workload role {role!r}')
    return {
        'allocation_id': allocation['allocation_id'],
        'resource_name': name,
        'image': image,
        'desired_manager_count': allocation.get('desired_manager_count'),
        'wedata_secret_projected': bool(wedata_sources),
        'wedata_key_file': (env.get('WEDATA_KEY_FILE') or {}).get('value'),
    }


async def _validate_completed_batch(
    api: CloudBatchApi,
    *,
    execution_id: str,
    expected_items: dict[str, dict[str, Any]],
    expected_dataset_uri: str,
    expected_resolved_uri: str,
    expected_page_size: int,
) -> dict[str, Any]:
    summary = await api.control_json('GET', f'/v1/executions/{execution_id}/batch')
    items = await api.list_all_batch_items(execution_id)
    if summary.get('total_items') != len(expected_items) or summary.get('terminal_items') != len(expected_items):
        raise AssertionError(f'batch counts do not match Dataset items: {summary}')
    if {item['item_id'] for item in items} != set(expected_items):
        raise AssertionError('materialized batch item IDs do not match the fixed Dataset release')
    if any(item.get('status') != 'succeeded' for item in items):
        raise AssertionError('not every Dataset child succeeded')
    child_ids = {str(item['current_child_execution_id']) for item in items}
    if len(child_ids) != len(expected_items):
        raise AssertionError('Dataset items and child executions are not one-to-one')
    facts = await api.execution_facts(child_ids)
    result_artifact_ids: list[str] = []
    result_summaries: list[dict[str, Any]] = []
    for item in items:
        item_id = str(item['item_id'])
        metadata = item.get('metadata')
        if not isinstance(metadata, dict):
            raise AssertionError(f'batch item {item_id} has invalid metadata')
        metadata_sha256 = hashlib.sha256(
            json.dumps(
                metadata,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(',', ':'),
            ).encode()
        ).hexdigest()
        if metadata_sha256 != expected_items[item_id]['metadata_sha256']:
            raise AssertionError(f'batch item {item_id} metadata does not match the Dataset item')
        child = facts[str(item['current_child_execution_id'])]
        run_input = (child.get('spec') or {}).get('run_input') or {}
        task_files = run_input.get('task_files') or {}
        if task_files != expected_items[item_id]['task_file']:
            raise AssertionError(f'child {child["execution_id"]} task_files do not match Dataset item {item_id}')
        if child.get('status') != 'succeeded' or not child.get('result_artifact_id'):
            raise AssertionError(f'child {child["execution_id"]} has no durable successful result')
        result = await api.resolved_result(child)
        if result.get('success') is not True or not isinstance(result.get('task_output'), dict):
            raise AssertionError(f'child {child["execution_id"]} returned an unsuccessful result')
        result_artifact_ids.append(str(child['result_artifact_id']))
        task_output = result['task_output']
        result_summaries.append(
            {
                'item_id': item_id,
                'child_execution_id': child['execution_id'],
                'result_artifact_id': child['result_artifact_id'],
                'metadata_keys': sorted(metadata),
                'task_output_keys': sorted(task_output),
                'score': task_output.get('score'),
            }
        )
    if len(set(result_artifact_ids)) != len(expected_items):
        raise AssertionError('Dataset child results are not one-to-one')
    manifest = summary.get('task_list_manifest') or {}
    if manifest.get('record_version') != 2:
        raise AssertionError(f'batch manifest did not retain task-list v2: {manifest.get("record_version")}')
    datasets = manifest.get('datasets') or []
    if datasets != [{'input_uri': expected_dataset_uri, 'resolved_uri': expected_resolved_uri}]:
        raise AssertionError(f'batch manifest did not pin the expected Dataset release: {datasets}')
    compose = ((manifest.get('source') or {}).get('definition') or {}).get('compose') or {}
    if compose.get('page_size') != expected_page_size:
        raise AssertionError('batch manifest did not retain the requested Dataset page_size')
    return {
        'total_items': len(items),
        'child_execution_count': len(child_ids),
        'result_artifact_count': len(result_artifact_ids),
        'dataset_manifest': datasets,
        'task_list_version': manifest['record_version'],
        'page_size': expected_page_size,
        'results': result_summaries,
    }


async def _wait_for_cleanup(
    api: CloudBatchApi,
    tkex: TkexWorkloadClient,
    *,
    execution_id: str,
    allocations: list[dict[str, Any]],
    timeout: float,
    poll_interval: float,
) -> dict[str, Any]:
    resource_names = {
        str(((allocation.get('provider_resource_ref') or {}).get('workload') or {}).get('name') or '')
        for allocation in allocations
    } - {''}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        execution = await api.control_json('GET', f'/v1/executions/{execution_id}')
        response = await api.control_json('GET', f'/v1/executions/{execution_id}/allocations')
        current = response.get('items') or []
        resources_gone = True
        for name in resource_names:
            try:
                await tkex.get_stateful_workload(name)
            except WorkloadNotFoundError:
                continue
            resources_gone = False
        if (
            execution.get('status') in TERMINAL_EXECUTION_STATUSES
            and all(item.get('status') == 'released' for item in current)
            and resources_gone
        ):
            return {
                'execution_status': execution['status'],
                'execution_phase': execution.get('phase'),
                'allocation_statuses': [item['status'] for item in current],
                'workload_resources_gone': True,
            }
        await asyncio.sleep(poll_interval)
    raise TimeoutError(f'Taiji Dataset batch resources were not released within {timeout:g}s')


def _sanitized_submission(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = json.loads(json.dumps(payload))
    sanitized.pop('idempotency_key', None)
    return sanitized


def _readable_secret_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f'{label} file does not exist: {resolved}')
    mode = stat.S_IMODE(resolved.stat().st_mode)
    if mode & 0o077:
        raise ValueError(f'{label} file must not be group/world accessible: {resolved}')
    return resolved


def _read_secret_file(path: Path, *, label: str) -> str:
    resolved = _readable_secret_file(path, label=label)
    value = resolved.read_text(encoding='utf-8').strip()
    if not value:
        raise ValueError(f'{label} file is empty')
    return value


def _create_tkex_workload_client(args: argparse.Namespace) -> TkexWorkloadClient:
    return TkexWorkloadClient(
        base_url=args.tkex_base_url,
        namespace=args.tkex_namespace,
        api_account=args.tkex_api_account,
        api_token=_read_secret_file(args.tkex_api_token_file, label='TKEx API token'),
        cluster_name=args.tkex_cluster_name,
        project_name=args.tkex_project_name,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def _utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'{path.suffix}.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        asyncio.run(execute(args))
    except (AssertionError, CloudBatchApiError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f'{type(exc).__name__}: {exc}')
        return 1
    print(args.report_path.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
