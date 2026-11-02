from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import re
import stat
import tempfile
import time
import uuid

from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from tests.integration.distributed.cloud_batch_api import CloudBatchApi, CloudBatchApiError
from tests.integration.distributed.cloud_batch_scenarios import (
    DEFAULT_WHEELHOUSE_EXTRAS,
    GIB,
    SOURCE_FIXTURE_MODULE,
    STRESS_STOP_POLICY,
    TKEX_LOAD_PROFILES,
    BatchLoadProfile,
    StressObservation,
    StressStopPolicy,
    batch_submission,
    environment_demand,
    fit_profile_to_container_budget,
    prepare_source_project,
    stress_stop_reasons,
    validate_profile_for_cluster,
    write_config_task_list,
)

from nexus.cli.commands.submit import _build_wheel, _prepare_uv_lock
from nexus.distributed.environment.kubernetes import WorkloadNotFoundError
from nexus.distributed.environment.naming import allocation_resource_name
from nexus.distributed.environment.tkex import TkexWorkloadClient
from nexus.distributed.submission.source_package import build_source_package
from nexus.runtime.providers.gongfeng import (
    RUNTIME_STATUS_NOT_FOUND,
    RUNTIME_STATUS_STOPPED,
    GongfengRuntimeProvider,
)


ROOT = Path(__file__).parents[3]
TERMINAL_EXECUTION_STATUSES = {'succeeded', 'failed', 'killed', 'lost', 'expired'}
TKEX_OBSERVER_READ_ATTEMPTS = 5
TKEX_OBSERVER_RETRY_BASE_SECONDS = 0.5
TKEX_TRANSIENT_API_ERROR = re.compile(r'^tkex_api_(?:429|5\d\d):')
GONGFENG_API_BASE_URL = 'http://hyrl-sandbox.prod.woa.com:8052'
STAGE_SUBMIT_RETRY_TIMEOUT_SECONDS = 120.0
STAGE_SUBMIT_RETRY_MAX_DELAY_SECONDS = 5.0


def _monotonic() -> float:
    return time.monotonic()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run guarded TKEx dynamic cloud-batch E2E and load scenarios.')
    parser.add_argument('--control-url', required=True)
    parser.add_argument('--application-url', required=True)
    parser.add_argument('--environment-id', required=True)
    parser.add_argument('--taihu-token-file', type=Path, required=True)
    parser.add_argument('--profile', action='append', choices=tuple(TKEX_LOAD_PROFILES), required=True)
    parser.add_argument('--typed-smoke', action='store_true')
    parser.add_argument('--containers-per-manager-pod', type=_positive_int, default=1)
    parser.add_argument('--available-cpu-cores', type=_positive_float)
    parser.add_argument('--available-memory-gib', type=_positive_float)
    parser.add_argument('--allow-extreme', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--database-url-file', type=Path)
    parser.add_argument(
        '--control-database-metrics',
        action='store_true',
        help='Read database guardrail counters from the admin-only control system status endpoint.',
    )
    parser.add_argument('--tkex-base-url')
    parser.add_argument('--tkex-api-account')
    parser.add_argument('--tkex-api-token-file', type=Path)
    parser.add_argument('--tkex-cluster-name')
    parser.add_argument('--tkex-project-name')
    parser.add_argument('--tkex-namespace')
    parser.add_argument('--universal-image')
    parser.add_argument('--workload-image')
    parser.add_argument('--builder-image')
    parser.add_argument('--gongfeng-api-base-url', default=GONGFENG_API_BASE_URL)
    parser.add_argument('--gongfeng-token-file', type=Path)
    parser.add_argument('--manager-crash', choices=('none', 'batch', 'execution'), default='none')
    parser.add_argument('--execution-manager-crash-count', type=_positive_int, default=20)
    parser.add_argument('--confirm-manager-pod-delete', action='store_true')
    parser.add_argument('--manager-crash-approval-file', type=Path)
    parser.add_argument('--poll-interval', type=_positive_float, default=1.0)
    parser.add_argument('--observation-interval', type=_positive_float, default=10.0)
    parser.add_argument('--report-path', type=Path)
    return parser.parse_args(argv)


def selected_profiles(names: list[str], *, allow_extreme: bool) -> list[BatchLoadProfile]:
    requested = set(names)
    profiles = [profile for name, profile in TKEX_LOAD_PROFILES.items() if name in requested]
    if any(profile.opt_in for profile in profiles) and not allow_extreme:
        raise ValueError('opt-in stress profiles require --allow-extreme')
    return profiles


def fit_profiles_to_container_budget(
    profiles: list[BatchLoadProfile],
    *,
    containers_per_manager_pod: int,
) -> list[BatchLoadProfile]:
    return [
        fit_profile_to_container_budget(
            profile,
            containers_per_manager_pod=containers_per_manager_pod,
        )
        for profile in profiles
    ]


def build_execution_plan(
    profiles: list[BatchLoadProfile],
    *,
    containers_per_manager_pod: int,
    available_cpu_cores: float,
    available_memory_gib: float,
) -> dict[str, Any]:
    if not profiles:
        raise ValueError('at least one profile is required')
    stages: list[dict[str, Any]] = []
    for profile in profiles:
        validate_profile_for_cluster(profile, containers_per_manager_pod=containers_per_manager_pod)
        demand = environment_demand(profile)
        stages.append(
            {
                'name': profile.name,
                'code_mode': profile.code_mode,
                'item_count': profile.item_count,
                'max_inflight': profile.max_inflight,
                'minimum_peak': profile.minimum_peak,
                'opt_in': profile.opt_in,
                'resources': {
                    'manager_pods': demand['manager_count'],
                    'running_manager_containers': profile.manager_container_count(
                        containers_per_pod=containers_per_manager_pod
                    ),
                    'cpu_cores': demand['cpu_millicores'] / 1000,
                    'memory_gib': demand['memory_bytes'] / GIB,
                    'execution_slots': profile.execution_slot_count,
                    'environment_slots': demand['slots'],
                },
            }
        )
    maximum_cpu = max(stage['resources']['cpu_cores'] for stage in stages)
    maximum_memory = max(stage['resources']['memory_gib'] for stage in stages)
    minimum_cpu = maximum_cpu * (1 + STRESS_STOP_POLICY.minimum_cluster_headroom_ratio)
    minimum_memory = maximum_memory * (1 + STRESS_STOP_POLICY.minimum_cluster_headroom_ratio)
    if available_cpu_cores < minimum_cpu:
        raise ValueError(f'CPU headroom requires at least {minimum_cpu:g} available cores')
    if available_memory_gib < minimum_memory:
        raise ValueError(f'memory headroom requires at least {minimum_memory:g} available GiB')
    return {
        'execution_mode': 'sequential',
        'maximum_running_manager_containers': max(stage['resources']['running_manager_containers'] for stage in stages),
        'maximum_requested_cpu_cores': maximum_cpu,
        'maximum_requested_memory_gib': maximum_memory,
        'minimum_available_cpu_cores': minimum_cpu,
        'minimum_available_memory_gib': minimum_memory,
        'stages': stages,
    }


def typed_source_submission(
    *,
    environment_id: str,
    source_refs: dict[str, dict[str, Any]],
    run_input: dict[str, Any],
    live: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'submission_version': 1,
        'idempotency_key': f'tkex-typed-{"live" if live else "single"}-{uuid.uuid4().hex}',
        'environment_id': environment_id,
        'code': {
            'mode': 'source_package',
            'source_package': source_refs['source_package'],
            'nexus_wheel': source_refs['nexus_wheel'],
            'uv_lock': source_refs['uv_lock'],
            'extras': list(DEFAULT_WHEELHOUSE_EXTRAS),
        },
        'resources': {
            'managers': {
                'count': 1,
                'cpu_cores': 1.0,
                'memory_gib': 1.0,
                'slots': 1,
            }
        },
        'run_input': run_input,
        'provisioning_timeout_seconds': 1800,
        'queue_ttl_seconds': 1800,
        'execution_ttl_seconds': 1800,
        'stop_timeout_seconds': 300,
    }
    if live:
        payload['live'] = {'working_dir': '/workspace'}
    return payload


def typed_single_run_input() -> dict[str, Any]:
    return {
        'imports': [SOURCE_FIXTURE_MODULE.removesuffix('.py')],
        'task_name': 'cloud_source_package_echo',
        'task_config': {'index': 900001, 'sleep_seconds': 0},
    }


def typed_live_run_input() -> dict[str, Any]:
    return {
        'imports': ['nexus.extensions.tasks.identity', SOURCE_FIXTURE_MODULE.removesuffix('.py')],
        'task_name': 'identity',
        'agent_name': 'cloud_interactive_agent',
        'agent_config': {'prompt': 'tkex-typed-live'},
    }


@dataclass(frozen=True, slots=True)
class TkexObserverConfig:
    base_url: str
    namespace: str
    api_account: str
    api_token_file: Path
    cluster_name: str
    project_name: str
    expected_workload_image: str | None = None


@dataclass(frozen=True, slots=True)
class ManagerCrashTarget:
    manager_id: str
    pod_name: str
    previous_pod_uid: str
    previous_session_id: str
    reserved_slots: int

    def as_report(self) -> dict[str, Any]:
        return {
            'manager_id': self.manager_id,
            'pod_name': self.pod_name,
            'previous_pod_uid': self.previous_pod_uid,
            'previous_session_id': self.previous_session_id,
            'reserved_slots': self.reserved_slots,
        }


def _manager_id_for_pod(allocation_id: str, pod_name: str) -> str:
    ordinal = pod_name.rsplit('-', 1)[-1]
    if not ordinal.isdigit():
        raise ValueError(f'pod name does not end with an ordinal: {pod_name}')
    return f'{allocation_id}-{int(ordinal)}'


def select_manager_crash_targets(
    pods_payload: dict[str, Any],
    managers_payload: dict[str, Any],
    *,
    allocation_id: str,
    workload: str,
    count: int,
    required_reserved_slots: int | None,
) -> list[ManagerCrashTarget]:
    managers = {str(item['manager_id']): item for item in managers_payload.get('items') or []}
    targets: list[ManagerCrashTarget] = []
    for pod in sorted(pods_payload.get('items') or [], key=lambda item: (item.get('metadata') or {}).get('name', '')):
        metadata = pod.get('metadata') or {}
        annotations = metadata.get('annotations') or {}
        labels = metadata.get('labels') or {}
        if annotations.get('nexus.io/allocation-id') != allocation_id:
            continue
        if labels.get('nexus.io/workload') != workload:
            continue
        if metadata.get('deletionTimestamp') or not _pod_is_ready(pod):
            continue
        pod_name = str(metadata.get('name') or '')
        pod_uid = str(metadata.get('uid') or '')
        if not pod_name or not pod_uid:
            continue
        manager_id = _manager_id_for_pod(allocation_id, pod_name)
        manager = managers.get(manager_id)
        if not manager or not manager.get('healthy') or not manager.get('session_id'):
            continue
        reserved_slots = int(manager.get('reserved_slots') or 0)
        if required_reserved_slots is not None and reserved_slots != required_reserved_slots:
            continue
        targets.append(
            ManagerCrashTarget(
                manager_id=manager_id,
                pod_name=pod_name,
                previous_pod_uid=pod_uid,
                previous_session_id=str(manager['session_id']),
                reserved_slots=reserved_slots,
            )
        )
        if len(targets) == count:
            break
    if len(targets) != count:
        raise ValueError(
            f'requested {count} {workload} manager crash targets for {allocation_id}, found {len(targets)}'
        )
    return targets


def manager_crash_recovery(
    targets: list[ManagerCrashTarget],
    pods_payload: dict[str, Any],
    managers_payload: dict[str, Any],
) -> tuple[list[dict[str, str]], list[str]]:
    pods = {
        str((item.get('metadata') or {}).get('name')): item
        for item in pods_payload.get('items') or []
        if (item.get('metadata') or {}).get('name')
    }
    managers = {str(item['manager_id']): item for item in managers_payload.get('items') or []}
    recovered: list[dict[str, str]] = []
    pending: list[str] = []
    for target in targets:
        pod = pods.get(target.pod_name)
        manager = managers.get(target.manager_id)
        pod_uid = str(((pod or {}).get('metadata') or {}).get('uid') or '')
        session_id = str((manager or {}).get('session_id') or '')
        if (
            pod is None
            or pod_uid == target.previous_pod_uid
            or not _pod_is_ready(pod)
            or manager is None
            or not manager.get('healthy')
            or session_id == target.previous_session_id
        ):
            pending.append(target.manager_id)
            continue
        recovered.append(
            {
                'manager_id': target.manager_id,
                'pod_name': target.pod_name,
                'replacement_pod_uid': pod_uid,
                'replacement_session_id': session_id,
            }
        )
    return recovered, pending


def manager_crash_trigger_ready(
    mode: str,
    *,
    profile: BatchLoadProfile,
    summary: dict[str, Any],
    peak_running_items: int,
    live_ready_sandboxes: int | None,
) -> bool:
    generation_complete = summary.get('generation_status') == 'complete'
    submitted = int(summary.get('submitted_items') or 0)
    total = int(summary.get('total_items') or profile.item_count)
    if mode == 'batch':
        return generation_complete and 0 < submitted < total and peak_running_items >= profile.minimum_peak
    if mode == 'execution':
        return (
            generation_complete
            and live_ready_sandboxes is not None
            and profile.minimum_sandbox_peak is not None
            and live_ready_sandboxes >= profile.minimum_sandbox_peak
        )
    return False


def unplanned_unreachable_managers(unreachable_managers: int, *, planned_pending: int) -> int:
    return max(0, unreachable_managers - planned_pending)


class TkexManagerCrashController:
    def __init__(
        self,
        client: TkexWorkloadClient,
        *,
        mode: str,
        count: int,
        approval_file: Path | None = None,
    ) -> None:
        if mode not in {'batch', 'execution'}:
            raise ValueError(f'unsupported manager crash mode: {mode}')
        if count <= 0 or (mode == 'batch' and count != 1):
            raise ValueError(f'invalid {mode} manager crash count: {count}')
        self.client = client
        self.mode = mode
        self.count = count
        self.approval_file = approval_file
        self.approval_request_file = Path(f'{approval_file}.request.json') if approval_file is not None else None
        self.targets: list[ManagerCrashTarget] = []
        self._injected_at_monotonic: float | None = None
        self.report: dict[str, Any] = {
            'mode': mode,
            'target_count': count,
            'status': 'armed',
            'targets': [],
            'replacements': [],
        }

    @property
    def injected(self) -> bool:
        return self._injected_at_monotonic is not None

    @property
    def awaiting_confirmation(self) -> bool:
        return not self.injected and self.report['status'] == 'awaiting_confirmation'

    @property
    def recovered(self) -> bool:
        return self.report['status'] == 'recovered'

    @property
    def planned_pending(self) -> int:
        return len(self.targets) if self.report['status'] == 'recovering' else 0

    @property
    def recovery_elapsed_seconds(self) -> float:
        if self._injected_at_monotonic is None or self.recovered:
            return 0
        return _monotonic() - self._injected_at_monotonic

    async def inject(
        self,
        api: CloudBatchApi,
        *,
        allocations: list[dict[str, Any]],
        profile: BatchLoadProfile,
        task_list_ref: dict[str, Any] | None,
    ) -> None:
        if self.injected:
            raise RuntimeError('manager crash was already injected')
        if not self.targets:
            workload = 'orchestration' if self.mode == 'batch' else 'execution'
            matching = [item for item in allocations if item.get('workload') == workload]
            if len(matching) != 1:
                raise ValueError(f'expected one {workload} allocation, found {len(matching)}')
            allocation_id = str(matching[0]['allocation_id'])
            pods_payload, managers_payload = await asyncio.gather(
                self.client.list_pods(),
                api.control_json('GET', '/v1/managers'),
            )
            self.targets = select_manager_crash_targets(
                pods_payload,
                managers_payload,
                allocation_id=allocation_id,
                workload=workload,
                count=self.count,
                required_reserved_slots=profile.slots_per_manager if self.mode == 'execution' else None,
            )
            self.report.update(
                {
                    'allocation_id': allocation_id,
                    'targets': [target.as_report() for target in self.targets],
                    'task_list_ref_before_crash': task_list_ref,
                }
            )
        stale_reason = self._allocation_stale_reason(allocations)
        if stale_reason is not None:
            self._mark_approval_stale(stale_reason)
            return
        if self.approval_file is not None and not self._exact_delete_is_approved():
            self.report['status'] = 'awaiting_confirmation'
            return
        stale_reason = await self._target_stale_reason(api)
        if stale_reason is not None:
            self._mark_approval_stale(stale_reason)
            return
        await asyncio.gather(
            *(self.client.delete_pod(target.pod_name, uid=target.previous_pod_uid) for target in self.targets)
        )
        self._injected_at_monotonic = _monotonic()
        self.report.update(
            {
                'status': 'recovering',
                'injected_at': datetime.now(timezone.utc).isoformat(),
            }
        )

    def _allocation_stale_reason(self, allocations: list[dict[str, Any]]) -> str | None:
        allocation_id = str(self.report['allocation_id'])
        allocation = next(
            (item for item in allocations if str(item.get('allocation_id') or '') == allocation_id),
            None,
        )
        if allocation is None:
            return f'allocation {allocation_id} is no longer present'
        status = str(allocation.get('status') or 'unknown')
        if status != 'ready':
            return f'allocation {allocation_id} entered {status}'
        return None

    async def _target_stale_reason(self, api: CloudBatchApi) -> str | None:
        managers_payload = await api.control_json('GET', '/v1/managers')
        managers = {str(item['manager_id']): item for item in managers_payload.get('items') or []}
        for target in self.targets:
            try:
                pod = await self.client.get_pod(target.pod_name)
            except WorkloadNotFoundError:
                return f'pod {target.pod_name} no longer exists'
            metadata = pod.get('metadata') or {}
            if str(metadata.get('uid') or '') != target.previous_pod_uid:
                return f'pod {target.pod_name} UID changed'
            if metadata.get('deletionTimestamp') or not _pod_is_ready(pod):
                return f'pod {target.pod_name} is no longer ready'
            manager = managers.get(target.manager_id)
            if (
                manager is None
                or not manager.get('healthy')
                or manager.get('draining')
                or str(manager.get('session_id') or '') != target.previous_session_id
            ):
                return f'manager {target.manager_id} session changed or became unhealthy'
        return None

    def _mark_approval_stale(self, reason: str) -> None:
        self.report.update(
            {
                'status': 'approval_stale',
                'stale_reason': reason,
                'stale_at': datetime.now(timezone.utc).isoformat(),
            }
        )

    def _exact_delete_is_approved(self) -> bool:
        assert self.approval_file is not None
        assert self.approval_request_file is not None
        operations = [self._delete_operation(target) for target in self.targets]
        approval_id = hashlib.sha256(json.dumps(operations, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        request = {
            'approval_id': approval_id,
            'mode': self.mode,
            'operations': operations,
        }
        self.report.update(
            {
                'approval_id': approval_id,
                'approval_file': str(self.approval_file),
                'approval_request_file': str(self.approval_request_file),
                'delete_operations': operations,
            }
        )
        self.approval_request_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.approval_request_file.exists():
            self.approval_request_file.write_text(json.dumps(request, indent=2, sort_keys=True) + '\n')
        try:
            approval = json.loads(self.approval_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False
        if approval.get('approval_id') != approval_id:
            self.report['approval_error'] = 'approval_id_mismatch'
            return False
        self.report.pop('approval_error', None)
        self.report['approved_at'] = datetime.now(timezone.utc).isoformat()
        return True

    def _delete_operation(self, target: ManagerCrashTarget) -> dict[str, Any]:
        path, body = self.client.pod_delete_request(target.pod_name, uid=target.previous_pod_uid)
        return {
            'method': 'DELETE',
            'url': f'{self.client.base_url}{path}',
            'path': path,
            'cluster': self.client.cluster_name,
            'project': self.client.project_name,
            'namespace': self.client.namespace,
            'resource': {
                'kind': 'Pod',
                'name': target.pod_name,
                'uid': target.previous_pod_uid,
            },
            'body': body,
            'expected_effect': f'Delete Pod {target.pod_name} and let its StatefulSetPlus recreate it.',
            'verification': (f'Require a new Pod UID and a new healthy manager session for {target.manager_id}.'),
        }

    async def observe_recovery(self, api: CloudBatchApi) -> bool:
        injected_at = self._injected_at_monotonic
        if injected_at is None:
            return False
        if self.recovered:
            return True
        pods_payload, managers_payload = await asyncio.gather(
            self.client.list_pods(),
            api.control_json('GET', '/v1/managers'),
        )
        recovered, pending = manager_crash_recovery(self.targets, pods_payload, managers_payload)
        self.report['pending_manager_ids'] = pending
        if pending:
            return False
        self.report.update(
            {
                'status': 'recovered',
                'recovered_at': datetime.now(timezone.utc).isoformat(),
                'recovery_seconds': _monotonic() - injected_at,
                'replacements': recovered,
                'pending_manager_ids': [],
            }
        )
        return True

    def validate_completed(self, *, parent: dict[str, Any], summary: dict[str, Any]) -> None:
        if not self.injected or not self.recovered:
            raise AssertionError(f'{self.mode} manager crash did not recover: {self.report}')
        if self.mode != 'batch':
            return
        if int(parent.get('attempt_count') or 0) < 2:
            raise AssertionError('batch manager crash did not create a new parent attempt')
        if summary.get('task_list_ref') != self.report.get('task_list_ref_before_crash'):
            raise AssertionError('batch manager crash changed task_list_ref')


async def _advance_manager_crash(
    controller: TkexManagerCrashController,
    api: CloudBatchApi,
    *,
    allocations: list[dict[str, Any]],
    profile: BatchLoadProfile,
    summary: dict[str, Any],
    peak_running_items: int,
    live_ready_sandboxes: int | None,
    stage: dict[str, Any],
) -> None:
    if not controller.injected:
        if controller.report.get('status') == 'approval_stale':
            stage['manager_crash'] = dict(controller.report)
            return
        if not getattr(controller, 'awaiting_confirmation', False) and not manager_crash_trigger_ready(
            controller.mode,
            profile=profile,
            summary=summary,
            peak_running_items=peak_running_items,
            live_ready_sandboxes=live_ready_sandboxes,
        ):
            stage['manager_crash'] = dict(controller.report)
            return
        await controller.inject(
            api,
            allocations=allocations,
            profile=profile,
            task_list_ref=summary.get('task_list_ref'),
        )
        stage['manager_crash'] = dict(controller.report)
        return
    if not controller.recovered:
        await controller.observe_recovery(api)
    stage['manager_crash'] = dict(controller.report)


async def _poll_manager_crash(
    controller: TkexManagerCrashController,
    api: CloudBatchApi,
    *,
    observed: bool,
    allocations: list[dict[str, Any]],
    profile: BatchLoadProfile,
    summary: dict[str, Any],
    peak_running_items: int,
    live_ready_sandboxes: int | None,
    stage: dict[str, Any],
) -> None:
    if not observed and not getattr(controller, 'awaiting_confirmation', False):
        return
    await _advance_manager_crash(
        controller,
        api,
        allocations=allocations,
        profile=profile,
        summary=summary,
        peak_running_items=peak_running_items,
        live_ready_sandboxes=live_ready_sandboxes,
        stage=stage,
    )


def _tkex_observation_due(
    *,
    allocations: list[dict[str, Any]],
    now: float,
    next_observation_at: float,
    manager_crash_controller: TkexManagerCrashController | None,
) -> bool:
    return (
        bool(allocations)
        and now >= next_observation_at
        and not (manager_crash_controller is not None and manager_crash_controller.awaiting_confirmation)
    )


def _create_tkex_workload_client(config: TkexObserverConfig) -> TkexWorkloadClient:
    return TkexWorkloadClient(
        base_url=config.base_url,
        namespace=config.namespace,
        api_account=config.api_account,
        api_token=_read_secret_file(config.api_token_file, label='TKEx API token'),
        cluster_name=config.cluster_name,
        project_name=config.project_name,
    )


class TkexObserver:
    def __init__(self, config: TkexObserverConfig) -> None:
        self.client = _create_tkex_workload_client(config)
        self.expected_workload_image = config.expected_workload_image
        self.read_concurrency = 50

    async def close(self) -> None:
        await self.client.close()

    async def _read(self, operation: Callable[[], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
        for attempt in range(TKEX_OBSERVER_READ_ATTEMPTS):
            try:
                return await operation()
            except (aiohttp.ClientError, TimeoutError) as exc:
                error = exc
            except RuntimeError as exc:
                if TKEX_TRANSIENT_API_ERROR.match(str(exc)) is None:
                    raise
                error = exc
            if attempt + 1 == TKEX_OBSERVER_READ_ATTEMPTS:
                raise error
            await asyncio.sleep(TKEX_OBSERVER_RETRY_BASE_SECONDS * (2**attempt))
        raise AssertionError('unreachable')

    async def observe(self, allocations: list[dict[str, Any]]) -> dict[str, Any]:
        allocation_ids = {str(item['allocation_id']) for item in allocations}
        semaphore = asyncio.Semaphore(max(1, int(getattr(self, 'read_concurrency', 50))))

        async def read_allocation(allocation: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
            name = allocation_resource_name(str(allocation['allocation_id']))
            async with semaphore:
                try:
                    workload = await self._read(lambda name=name: self.client.get_stateful_workload(name))
                except WorkloadNotFoundError:
                    return None, None
                try:
                    await self._read(lambda name=name: self.client.get_service(name))
                except WorkloadNotFoundError:
                    service_name = None
                else:
                    service_name = name
                return workload, service_name

        allocation_observations = await asyncio.gather(*(read_allocation(item) for item in allocations))
        workloads = [workload for workload, _service in allocation_observations if workload is not None]
        services_present = [service for _workload, service in allocation_observations if service is not None]
        pod_summary = summarize_tkex_pods(
            await self._read(self.client.list_pods),
            allocation_ids=allocation_ids,
        )
        template_container_counts: list[int] = []
        workload_images: set[str] = set()
        db_pool_sizes: set[str] = set()
        db_pool_settings: dict[str, dict[str, str]] = {}
        cluster_names: set[str] = set()
        pod_ip_downward_api = True
        desired_replicas = 0
        ready_replicas = 0
        for workload in workloads:
            spec = workload.get('spec') or {}
            status = workload.get('status') or {}
            desired_replicas += int(spec.get('replicas') or 0)
            ready_replicas += int(status.get('readyReplicas') or 0)
            containers = ((spec.get('template') or {}).get('spec') or {}).get('containers') or []
            template_container_counts.append(len(containers))
            image = containers[0].get('image') if containers else None
            workload_images.add(str(image) if image else '<missing>')
            env = {item.get('name'): item for item in ((containers[0].get('env') or []) if containers else [])}
            workload_role = (env.get('NEXUS_DIST_MANAGER_WORKLOAD') or {}).get('value')
            db_pool_size = (env.get('NEXUS_DIST_DB_POOL_SIZE') or {}).get('value')
            db_max_overflow = (env.get('NEXUS_DIST_DB_MAX_OVERFLOW') or {}).get('value')
            if db_pool_size is not None:
                db_pool_sizes.add(str(db_pool_size))
            if workload_role and db_pool_size is not None and db_max_overflow is not None:
                setting = {'pool_size': str(db_pool_size), 'max_overflow': str(db_max_overflow)}
                previous = db_pool_settings.setdefault(str(workload_role), setting)
                if previous != setting:
                    raise RuntimeError(f'inconsistent DB pool settings for workload role {workload_role}')
            if value := (env.get('NEXUS_DIST_CLUSTER_NAME') or {}).get('value'):
                cluster_names.add(str(value))
            pod_ip = env.get('POD_IP') or {}
            pod_ip_downward_api = pod_ip_downward_api and (
                (((pod_ip.get('valueFrom') or {}).get('fieldRef') or {}).get('fieldPath')) == 'status.podIP'
            )
        return {
            **pod_summary,
            'workload_count': len(workloads),
            'desired_replicas': desired_replicas,
            'ready_replicas': ready_replicas,
            'unreachable_managers': max(0, desired_replicas - ready_replicas),
            'template_container_counts': sorted(set(template_container_counts)),
            'workload_images': sorted(workload_images),
            'unexpected_workload_images': sorted(
                workload_images - {self.expected_workload_image}
                if getattr(self, 'expected_workload_image', None)
                else set()
            ),
            'db_pool_sizes': sorted(db_pool_sizes),
            'db_pool_settings': db_pool_settings,
            'cluster_names': sorted(cluster_names),
            'pod_ip_downward_api': pod_ip_downward_api,
            'services_present': sorted(services_present),
        }


class GongfengReleaseObserver:
    def __init__(
        self,
        *,
        api_base_url: str,
        token_file: Path,
        list_chunk_size: int = 100,
        list_concurrency: int = 10,
        batch_list_page_size: int = 5000,
    ) -> None:
        self.provider = GongfengRuntimeProvider(
            api_base_url=api_base_url,
            token=_read_secret_file(token_file, label='Gongfeng token'),
            auto_connect_ws=False,
            print_port_mappings=False,
            request_timeout=120,
        )
        self.list_chunk_size = list_chunk_size
        self.list_concurrency = list_concurrency
        self.batch_list_page_size = batch_list_page_size
        self.batch_snapshot_attempts = 10
        self.batch_snapshot_retry_seconds = 1.0

    async def batch_snapshot(self, batch_id: str) -> dict[str, Any]:
        attempts = max(1, int(getattr(self, 'batch_snapshot_attempts', 10)))
        for attempt in range(attempts):
            runtimes, totals = await self._batch_snapshot_pages(batch_id)
            runtime_ids = [str(item.get('runtime_id') or '') for item in runtimes]
            if any(not runtime_id for runtime_id in runtime_ids):
                raise AssertionError(f'Gongfeng batch {batch_id} returned a runtime without an ID')
            duplicate_ids = sorted(runtime_id for runtime_id, count in Counter(runtime_ids).items() if count > 1)
            known_totals = {total for total in totals if total > 0}
            inconsistent_reason = None
            if duplicate_ids:
                inconsistent_reason = f'duplicate runtime IDs: {duplicate_ids[:10]}'
            elif len(known_totals) > 1:
                inconsistent_reason = f'page totals changed: {sorted(known_totals)}'
            elif known_totals and len(runtime_ids) != next(iter(known_totals)):
                inconsistent_reason = (
                    f'unique runtime count {len(runtime_ids)} did not match total {next(iter(known_totals))}'
                )
            if inconsistent_reason is None:
                return self._batch_snapshot_result(batch_id, runtimes=runtimes, runtime_ids=runtime_ids)
            if attempt + 1 == attempts:
                return self._conservative_batch_snapshot_result(
                    batch_id,
                    runtimes=runtimes,
                    runtime_ids=runtime_ids,
                )
            await asyncio.sleep(float(getattr(self, 'batch_snapshot_retry_seconds', 1.0)))
        raise AssertionError('unreachable')

    async def _batch_snapshot_pages(self, batch_id: str) -> tuple[list[dict[str, Any]], list[int]]:
        runtimes: list[dict[str, Any]] = []
        totals: list[int] = []
        page = 1
        while True:
            chunk, total = await self.provider.list_runtimes(
                batch_id=batch_id,
                page=page,
                per_page=self.batch_list_page_size,
            )
            if any(str(item.get('batch_id') or '') != batch_id for item in chunk):
                raise AssertionError(f'Gongfeng batch query returned a runtime outside {batch_id}')
            runtimes.extend(chunk)
            totals.append(int(total or 0))
            if len(chunk) < self.batch_list_page_size:
                break
            page += 1
        return runtimes, totals

    def _batch_snapshot_result(
        self,
        batch_id: str,
        *,
        runtimes: list[dict[str, Any]],
        runtime_ids: list[str],
    ) -> dict[str, Any]:
        status_counts = Counter(str(item.get('status') or 'unknown') for item in runtimes)
        active_runtime_ids = [
            str(item['runtime_id'])
            for item in runtimes
            if str(item.get('status') or '') not in {RUNTIME_STATUS_STOPPED, RUNTIME_STATUS_NOT_FOUND}
        ]
        return {
            'batch_id': batch_id,
            'runtime_count': len(runtimes),
            'ready_runtime_count': status_counts.get('scheduled', 0),
            'active_runtime_ids': active_runtime_ids,
            'status_counts': dict(sorted(status_counts.items())),
            'runtime_ids': runtime_ids,
        }

    def _conservative_batch_snapshot_result(
        self,
        batch_id: str,
        *,
        runtimes: list[dict[str, Any]],
        runtime_ids: list[str],
    ) -> dict[str, Any]:
        grouped_statuses: dict[str, set[str]] = {}
        for runtime, runtime_id in zip(runtimes, runtime_ids, strict=True):
            grouped_statuses.setdefault(runtime_id, set()).add(str(runtime.get('status') or 'unknown'))
        statuses = {
            runtime_id: next(iter(runtime_statuses)) if len(runtime_statuses) == 1 else 'inconsistent'
            for runtime_id, runtime_statuses in grouped_statuses.items()
        }
        terminal_statuses = {RUNTIME_STATUS_STOPPED, RUNTIME_STATUS_NOT_FOUND}
        active_runtime_ids = [
            runtime_id
            for runtime_id, runtime_statuses in grouped_statuses.items()
            if any(status not in terminal_statuses for status in runtime_statuses)
        ]
        status_counts = Counter(statuses.values())
        return {
            'batch_id': batch_id,
            'runtime_count': len(grouped_statuses),
            'ready_runtime_count': status_counts.get('scheduled', 0),
            'active_runtime_ids': active_runtime_ids,
            'status_counts': dict(sorted(status_counts.items())),
            'runtime_ids': list(grouped_statuses),
        }

    async def wait_batch_released(
        self,
        batch_id: str,
        *,
        minimum_runtime_count: int,
        timeout: float,
        poll_interval: float,
    ) -> dict[str, Any]:
        deadline = _monotonic() + timeout
        snapshot: dict[str, Any] = {}
        while True:
            snapshot = await self.batch_snapshot(batch_id)
            active = snapshot['active_runtime_ids']
            if snapshot['runtime_count'] >= minimum_runtime_count and not active:
                return {
                    'batch_id': batch_id,
                    'runtime_count': snapshot['runtime_count'],
                    'active_runtime_count': 0,
                    'status_counts': snapshot['status_counts'],
                }
            if _monotonic() >= deadline:
                if active:
                    await self._stop_runtimes(active)
                raise TimeoutError(
                    f'Gongfeng batch {batch_id} did not release after {timeout:g}s: '
                    f'runtime_count={snapshot.get("runtime_count")}, active={len(active)}'
                )
            await asyncio.sleep(poll_interval)

    async def wait_released(
        self,
        runtime_ids: list[str],
        *,
        timeout: float,
        poll_interval: float,
    ) -> dict[str, Any]:
        unique_runtime_ids = list(dict.fromkeys(runtime_ids))
        if len(unique_runtime_ids) != len(runtime_ids):
            raise AssertionError('Gongfeng release validation requires unique runtime IDs')
        deadline = _monotonic() + timeout
        latest_statuses: dict[str, str] = {}
        while True:
            latest_statuses = await self._read_statuses(unique_runtime_ids)
            active = [
                runtime_id
                for runtime_id, status in latest_statuses.items()
                if status not in {RUNTIME_STATUS_STOPPED, RUNTIME_STATUS_NOT_FOUND}
            ]
            if not active:
                return {
                    'checked_runtime_count': len(unique_runtime_ids),
                    'active_runtime_count': 0,
                    'release_status_counts': dict(sorted(Counter(latest_statuses.values()).items())),
                }
            if _monotonic() >= deadline:
                await self._stop_runtimes(active)
                raise TimeoutError(
                    f'{len(active)} Gongfeng runtimes remain active after {timeout:g}s; sample={active[:10]}'
                )
            await asyncio.sleep(poll_interval)

    async def _read_statuses(self, runtime_ids: list[str]) -> dict[str, str]:
        semaphore = asyncio.Semaphore(self.list_concurrency)

        async def read_chunk(chunk: list[str]) -> dict[str, str]:
            async with semaphore:
                runtimes, _total = await self.provider.list_runtimes(
                    runtime_ids=chunk,
                    page=1,
                    per_page=len(chunk),
                )
            statuses = {
                str(runtime['runtime_id']): str(runtime.get('status') or '')
                for runtime in runtimes
                if runtime.get('runtime_id')
            }
            return {runtime_id: statuses.get(runtime_id, RUNTIME_STATUS_NOT_FOUND) for runtime_id in chunk}

        chunks = [
            runtime_ids[offset : offset + self.list_chunk_size]
            for offset in range(0, len(runtime_ids), self.list_chunk_size)
        ]
        results = await asyncio.gather(*(read_chunk(chunk) for chunk in chunks))
        return {runtime_id: status for result in results for runtime_id, status in result.items()}

    async def _stop_runtimes(self, runtime_ids: list[str]) -> None:
        chunks = [
            runtime_ids[offset : offset + self.list_chunk_size]
            for offset in range(0, len(runtime_ids), self.list_chunk_size)
        ]
        for chunk in chunks:
            await self.provider._stop_runtime(chunk)


class StageRunError(RuntimeError):
    def __init__(self, stage: dict[str, Any], cause: BaseException) -> None:
        super().__init__(f'{stage["name"]} failed: {cause}')
        self.stage = stage
        self.cause = cause


async def read_database_metrics(database_url_file: Path) -> dict[str, Any]:
    database_url = _read_secret_file(database_url_file, label='database URL')
    engine = create_async_engine(database_url, poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            max_connections = int((await connection.execute(text("SHOW VARIABLES LIKE 'max_connections'"))).one()[1])
            metrics: dict[str, Any] = {
                'available': True,
                'dialect': 'mysql',
                'max_connections': max_connections,
            }
            names = {
                'Threads_connected': 'threads_connected',
                'Threads_running': 'threads_running',
                'Max_used_connections': 'max_used_connections',
                'Slow_queries': 'slow_queries',
                'Aborted_connects': 'aborted_connects',
            }
            for name, normalized_name in names.items():
                row = (await connection.execute(text(f"SHOW GLOBAL STATUS LIKE '{name}'"))).one()
                metrics[normalized_name] = int(row[1])
            return metrics
    finally:
        await engine.dispose()


async def read_control_database_metrics(api: CloudBatchApi) -> dict[str, Any]:
    status = await api.control_json('GET', '/v1/system/status')
    metrics = status.get('database')
    if not isinstance(metrics, dict) or metrics.get('available') is not True:
        raise RuntimeError('control database metrics are unavailable')
    required_integer_fields = (
        'max_connections',
        'threads_connected',
        'threads_running',
        'max_used_connections',
        'slow_queries',
        'aborted_connects',
    )
    try:
        normalized: dict[str, Any] = {
            'available': True,
            'dialect': str(metrics['dialect']),
        }
        normalized.update({name: int(metrics[name]) for name in required_integer_fields})
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError('control database metrics are incomplete') from exc
    return normalized


async def read_control_artifact_metrics(api: CloudBatchApi) -> dict[str, int]:
    status = await api.control_json('GET', '/v1/system/status')
    metrics = status.get('artifacts')
    uploads_by_status = metrics.get('uploads_by_status') if isinstance(metrics, dict) else None
    if not isinstance(metrics, dict) or not isinstance(uploads_by_status, dict):
        raise RuntimeError('control artifact metrics are incomplete')
    try:
        return {
            'completed_uploads': int(uploads_by_status.get('completed', 0)),
            'failed_uploads': int(metrics['failed_uploads']),
            'checksum_failures': int(metrics['checksum_failures']),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError('control artifact metrics are incomplete') from exc


DatabaseMetricsReader = Callable[[], Awaitable[dict[str, Any]]]


class _BatchConcurrencySampler:
    def __init__(
        self,
        api: CloudBatchApi,
        *,
        execution_id: str,
        stage: dict[str, Any],
        minimum_peak: int,
        interval: float,
    ) -> None:
        self.api = api
        self.execution_id = execution_id
        self.stage = stage
        self.minimum_peak = minimum_peak
        self.interval = max(interval, 0.001)
        self.latest_summary: dict[str, Any] = {}
        self.minimum_peak_reached_at: float | None = None
        self._first_sample = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(
            self._run(),
            name=f'tkex-batch-concurrency:{self.execution_id}',
        )

    async def wait_for_first_sample(self) -> None:
        await self._first_sample.wait()
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self.raise_if_failed()

    async def _run(self) -> None:
        next_sample_at = time.monotonic()
        try:
            while not self._stop.is_set():
                summary = await _read_batch_summary(self.api, self.execution_id)
                sampled_at = time.monotonic()
                self.latest_summary = summary
                active_items = int(summary.get('active_items') or 0)
                self.stage['peak_active_items'] = max(self.stage['peak_active_items'], active_items)
                if summary.get('generation_status') == 'complete':
                    status_counts = summary.get('child_execution_status_counts') or {}
                    running_items = int(status_counts.get('running') or 0)
                    self.stage['peak_running_items'] = max(self.stage['peak_running_items'], running_items)
                    if running_items >= self.minimum_peak and self.minimum_peak_reached_at is None:
                        self.minimum_peak_reached_at = sampled_at
                    self.stage['concurrency_observations'].append(
                        {
                            'observed_at': datetime.now(timezone.utc).isoformat(),
                            'active_items': active_items,
                            'running_items': running_items,
                        }
                    )
                self._first_sample.set()
                next_sample_at += self.interval
                timeout = max(0, next_sample_at - time.monotonic())
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._error = exc
            self._first_sample.set()


async def _read_batch_summary(api: CloudBatchApi, execution_id: str) -> dict[str, Any]:
    try:
        return await api.control_json(
            'GET',
            f'/v1/executions/{execution_id}/batch?include_status_counts=true',
        )
    except CloudBatchApiError as exc:
        if exc.status != 404:
            raise
        return {}


async def _submit_batch_with_retry(
    api: CloudBatchApi,
    payload: dict[str, Any],
    *,
    timeout_seconds: float = STAGE_SUBMIT_RETRY_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    deadline = _monotonic() + timeout_seconds
    delay = 1.0
    while True:
        try:
            return await api.control_json('POST', '/v1/submissions/batch', body=payload)
        except CloudBatchApiError as exc:
            response = exc.payload
            retryable = exc.status == 429 and isinstance(response, dict) and response.get('retryable') is True
            remaining = deadline - _monotonic()
            if not retryable or remaining <= 0:
                raise
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2, STAGE_SUBMIT_RETRY_MAX_DELAY_SECONDS)


def result_upload_failure_ratio(
    baseline: dict[str, int],
    current: dict[str, int],
) -> float:
    completed = max(0, current['completed_uploads'] - baseline['completed_uploads'])
    failed = max(0, current['failed_uploads'] - baseline['failed_uploads'])
    if failed == 0:
        return 0
    return failed / max(1, completed + failed)


def _stress_policy_for_profile(profile: BatchLoadProfile) -> StressStopPolicy:
    if profile.task_kind != 'sandbox_agent_echo':
        return STRESS_STOP_POLICY
    return replace(
        STRESS_STOP_POLICY,
        no_progress_seconds=max(STRESS_STOP_POLICY.no_progress_seconds, profile.task_sleep_seconds + 300),
    )


def _stage_progress_fingerprint(
    execution: dict[str, Any],
    summary: dict[str, Any],
    *,
    completed_uploads: int,
) -> tuple[Any, ...]:
    return (
        execution.get('status'),
        summary.get('generation_status'),
        summary.get('materialized_count'),
        summary.get('submitted_items'),
        summary.get('terminal_items'),
        completed_uploads,
    )


def _monitor_no_progress_seconds(
    *,
    monitor_active: bool,
    monitor_was_active: bool,
    last_progress_at: float,
    now: float,
) -> tuple[float, float]:
    if not monitor_active:
        return 0, last_progress_at
    if not monitor_was_active:
        return 0, now
    return now - last_progress_at, last_progress_at


def _tkex_observation_stop_reasons(observation: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if observation.get('services_present'):
        reasons.append(f'unexpected allocation Services: {observation["services_present"]}')
    if observation.get('workload_count') and not observation.get('pod_ip_downward_api'):
        reasons.append('allocation workload is missing the POD_IP Downward API')
    unexpected_images = observation.get('unexpected_workload_images') or []
    if unexpected_images:
        reasons.append(f'unexpected workload images: {", ".join(unexpected_images)}')
    return reasons


def database_metrics_reader(args: argparse.Namespace, api: CloudBatchApi) -> DatabaseMetricsReader | None:
    if args.database_url_file is not None:
        return lambda: read_database_metrics(args.database_url_file)
    if args.control_database_metrics:
        return lambda: read_control_database_metrics(api)
    return None


def _validate_large_profile_db_pool(
    profile: BatchLoadProfile,
    observed_pool_settings: dict[str, dict[str, str]],
) -> None:
    expected = {
        'execution': {'pool_size': '1', 'max_overflow': '0'},
        'orchestration': {'pool_size': '4', 'max_overflow': '2'},
    }
    if profile.manager_pod_count > 100 and observed_pool_settings != expected:
        raise RuntimeError(
            f'{profile.name} requires prior small-stage role-specific DB pools {expected}; '
            f'observed={observed_pool_settings}'
        )


def _merge_observed_db_pool_settings(
    current: dict[str, dict[str, str]],
    observations: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    merged = {role: dict(setting) for role, setting in current.items()}
    for observation in observations:
        for role, setting in (observation.get('db_pool_settings') or {}).items():
            normalized = {str(key): str(value) for key, value in setting.items()}
            previous = merged.setdefault(str(role), normalized)
            if previous != normalized:
                raise RuntimeError(f'inconsistent DB pool settings for workload role {role}')
    return merged


def _read_secret_file(path: Path, *, label: str) -> str:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ValueError(f'{label} file must not be readable or writable by group/other')
    value = path.read_text(encoding='utf-8').strip()
    if not value:
        raise ValueError(f'{label} file is empty')
    return value


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'{path.suffix}.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    temporary.replace(path)


async def _poll_typed_execution(
    api: CloudBatchApi,
    execution_id: str,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = await api.control_json('GET', f'/v1/executions/{execution_id}')
        if predicate(latest):
            return latest
        if latest.get('status') in TERMINAL_EXECUTION_STATUSES:
            raise AssertionError(f'{execution_id} reached unexpected terminal state: {latest}')
        await asyncio.sleep(1)
    raise TimeoutError(f'{execution_id} did not reach the expected state: {latest}')


async def _run_typed_smoke(
    api: CloudBatchApi,
    observer: TkexObserver,
    *,
    environment_id: str,
    source_refs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        'status': 'running',
        'started_at': datetime.now(timezone.utc).isoformat(),
        'single': None,
        'live': None,
    }
    submitted_ids: list[str] = []
    released_ids: set[str] = set()
    try:
        single_submission = await api.control_json(
            'POST',
            '/v1/submissions/single',
            body=typed_source_submission(
                environment_id=environment_id,
                source_refs=source_refs,
                run_input=typed_single_run_input(),
                live=False,
            ),
        )
        single_id = str(single_submission['execution_id'])
        submitted_ids.append(single_id)
        single = await _poll_typed_execution(
            api,
            single_id,
            lambda item: item.get('status') == 'succeeded',
            timeout=3600,
        )
        single_result = await api.resolved_result(single)
        single_output = single_result.get('task_output') or {}
        if single_output.get('marker') != 'source-package-ok:900001':
            raise AssertionError(f'typed single returned an unexpected result: {single_result}')
        if not single_output.get('dependency_version') or '/manager-venvs/' not in str(single_output.get('sys_prefix')):
            raise AssertionError(f'typed single did not use the built wheelhouse Manager venv: {single_result}')
        await _wait_stage_released(api, observer, execution_id=single_id, timeout=900)
        released_ids.add(single_id)
        report['single'] = {
            'execution_id': single_id,
            'status': single['status'],
            'result_artifact_id': single.get('result_artifact_id'),
            'dependency_build_id': single.get('dependency_build_id'),
            'marker': single_output['marker'],
        }

        live_submission = await api.control_json(
            'POST',
            '/v1/submissions/live',
            body=typed_source_submission(
                environment_id=environment_id,
                source_refs=source_refs,
                run_input=typed_live_run_input(),
                live=True,
            ),
        )
        live_id = str(live_submission['execution_id'])
        submitted_ids.append(live_id)
        await _poll_typed_execution(
            api,
            live_id,
            lambda item: item.get('status') == 'waiting_input',
            timeout=3600,
        )
        required = await api.read_application_sse_until(
            f'/v1/executions/{live_id}/live/events?cursor=0',
            'input.required',
            timeout=300,
        )
        if required.get('type') != 'input.required':
            raise AssertionError(f'typed live returned an unexpected SSE event: {required}')
        await api.application_json(
            'POST',
            f'/v1/executions/{live_id}/commands',
            body={
                'type': 'input.submit',
                'payload': {'text': 'tkex-typed-live-ok'},
                'idempotency_key': f'{live_id}-input',
            },
        )
        live = await _poll_typed_execution(
            api,
            live_id,
            lambda item: item.get('status') == 'succeeded',
            timeout=1800,
        )
        live_result = await api.resolved_result(live)
        if (live_result.get('agent_output') or {}).get('choice') != 'tkex-typed-live-ok':
            raise AssertionError(f'typed live returned an unexpected result: {live_result}')
        await _wait_stage_released(api, observer, execution_id=live_id, timeout=900)
        released_ids.add(live_id)
        report['live'] = {
            'execution_id': live_id,
            'status': live['status'],
            'result_artifact_id': live.get('result_artifact_id'),
            'dependency_build_id': live.get('dependency_build_id'),
            'sse_event': required['type'],
            'choice': live_result['agent_output']['choice'],
        }
        report['status'] = 'passed'
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        return report
    except Exception as exc:
        report['status'] = 'failed'
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        if report['status'] != 'passed':
            for execution_id in reversed(submitted_ids):
                if execution_id not in released_ids:
                    await _request_stop(api, execution_id)
                    with contextlib.suppress(Exception):
                        await _wait_stage_released(api, observer, execution_id=execution_id, timeout=900)


async def execute_profiles(
    args: argparse.Namespace,
    *,
    profiles: list[BatchLoadProfile],
    plan: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    token = _read_secret_file(args.taihu_token_file, label='Taihu token')
    observer = TkexObserver(_tkex_observer_config(args))
    tkex_config = _tkex_observer_config(args)
    sandbox_observer = None
    if any(profile.task_kind == 'sandbox_agent_echo' for profile in profiles):
        sandbox_observer = GongfengReleaseObserver(
            api_base_url=args.gongfeng_api_base_url,
            token_file=args.gongfeng_token_file,
        )
    report: dict[str, Any] = {
        'started_at': datetime.now(timezone.utc).isoformat(),
        'status': 'running',
        'target': {
            'environment_id': args.environment_id,
            'control_url': args.control_url,
            'application_url': args.application_url,
            'tkex_cluster_name': args.tkex_cluster_name,
            'tkex_namespace': args.tkex_namespace,
            'images': {
                'universal': args.universal_image,
                'workload': args.workload_image,
                'builder': args.builder_image,
            },
        },
        'plan': plan,
        'manager_crash': {
            'mode': getattr(args, 'manager_crash', 'none'),
            'execution_manager_count': getattr(args, 'execution_manager_crash_count', 20),
        },
        'source_inputs': None,
        'typed_smoke': None,
        'stages': [],
    }
    _write_report(report_path, report)
    source_refs: dict[str, dict[str, Any]] | None = None
    source_dependency_builds_by_workload: dict[str, list[str]] | None = None
    runtime_db_pool_settings: dict[str, dict[str, str]] = {}
    try:
        async with CloudBatchApi(
            control_url=args.control_url,
            application_url=args.application_url,
            token=token,
        ) as api:
            metrics_reader = database_metrics_reader(args, api)
            await api.control_json('GET', '/v1/health')
            await api.application_json('GET', '/v1/health')
            if metrics_reader is not None:
                report['database_before'] = await metrics_reader()
            with tempfile.TemporaryDirectory(prefix='nexus-tkex-cloud-batch-') as temporary_dir:
                temporary = Path(temporary_dir)
                typed_smoke = bool(getattr(args, 'typed_smoke', False))
                if typed_smoke or any(profile.code_mode == 'source_package' for profile in profiles):
                    source_refs, source_report = await _prepare_source_inputs(
                        api,
                        project_root=args.project_root.resolve(),
                        temporary=temporary,
                        environment_id=args.environment_id,
                    )
                    report['source_inputs'] = source_report
                    _write_report(report_path, report)
                if typed_smoke:
                    assert source_refs is not None
                    report['typed_smoke'] = await _run_typed_smoke(
                        api,
                        observer,
                        environment_id=args.environment_id,
                        source_refs=source_refs,
                    )
                    _write_report(report_path, report)
                for profile in profiles:
                    _validate_large_profile_db_pool(profile, runtime_db_pool_settings)
                    if profile.code_mode == 'config_only':
                        task_list_path = temporary / f'{profile.name}.jsonl'
                        run_id_prefix = f'tkex-{profile.name}-{uuid.uuid4().hex[:8]}'
                        manifest = write_config_task_list(
                            task_list_path,
                            profile=profile,
                            run_id_prefix=run_id_prefix,
                        )
                        task_list_ref = await api.upload_artifact(
                            task_list_path,
                            kind='task_list',
                            sha256=manifest['sha256'],
                            size_bytes=manifest['size_bytes'],
                            idempotency_key=f'tkex-e2e-task-list-{manifest["sha256"]}',
                            environment_id=args.environment_id,
                        )
                        payload = batch_submission(
                            profile=profile,
                            environment_id=args.environment_id,
                            idempotency_key=f'tkex-e2e-{profile.name}-{uuid.uuid4().hex}',
                            task_list_ref=task_list_ref,
                            task_list_manifest=manifest,
                        )
                    else:
                        run_id_prefix = None
                        assert source_refs is not None
                        payload = batch_submission(
                            profile=profile,
                            environment_id=args.environment_id,
                            idempotency_key=f'tkex-e2e-{profile.name}-{uuid.uuid4().hex}',
                            source_package_ref=source_refs['source_package'],
                            nexus_wheel_ref=source_refs['nexus_wheel'],
                            uv_lock_ref=source_refs['uv_lock'],
                        )
                    crash_controller = None
                    crash_client = None
                    crash_mode = getattr(args, 'manager_crash', 'none')
                    if profile.task_kind == 'sandbox_agent_echo' and crash_mode != 'none':
                        crash_client = _create_tkex_workload_client(tkex_config)
                        crash_controller = TkexManagerCrashController(
                            crash_client,
                            mode=crash_mode,
                            count=(1 if crash_mode == 'batch' else args.execution_manager_crash_count),
                            approval_file=getattr(args, 'manager_crash_approval_file', None),
                        )
                    try:
                        try:
                            stage = await _run_stage(
                                api,
                                observer,
                                profile=profile,
                                payload=payload,
                                available_cpu_cores=args.available_cpu_cores,
                                available_memory_gib=args.available_memory_gib,
                                database_metrics_reader=metrics_reader,
                                poll_interval=args.poll_interval,
                                observation_interval=args.observation_interval,
                                sandbox_release_observer=sandbox_observer,
                                gongfeng_batch_id=run_id_prefix if profile.task_kind == 'sandbox_agent_echo' else None,
                                manager_crash_controller=crash_controller,
                            )
                        except StageRunError as exc:
                            report['stages'].append(exc.stage)
                            _write_report(report_path, report)
                            raise
                    finally:
                        if crash_client is not None:
                            await crash_client.close()
                    report['stages'].append(stage)
                    runtime_db_pool_settings = _merge_observed_db_pool_settings(
                        runtime_db_pool_settings,
                        stage['tkex_observations'],
                    )
                    if profile.code_mode == 'source_package':
                        stage_builds = stage['dependency_build_ids_by_workload']
                        if source_dependency_builds_by_workload is None:
                            source_dependency_builds_by_workload = stage_builds
                        elif source_dependency_builds_by_workload != stage_builds:
                            raise AssertionError(
                                'source profiles did not reuse DependencyBuilds by workload: '
                                f'expected={source_dependency_builds_by_workload}, observed={stage_builds}'
                            )
                    _write_report(report_path, report)
            if metrics_reader is not None:
                report['database_after'] = await metrics_reader()
            report['maximum_consecutive_gateway_5xx'] = api.maximum_consecutive_gateway_5xx
        report['status'] = 'passed'
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        _write_report(report_path, report)
        return report
    except Exception as exc:
        report['status'] = 'failed'
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        _write_report(report_path, report)
        raise
    finally:
        await observer.close()


async def _prepare_source_inputs(
    api: CloudBatchApi,
    *,
    project_root: Path,
    temporary: Path,
    environment_id: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    started = time.monotonic()
    source_root = temporary / 'source'
    prepare_source_project(source_root, project_root=project_root)
    source = await asyncio.to_thread(build_source_package, source_root, temporary / 'source.tar.zst')
    wheel = await asyncio.to_thread(_build_wheel, project_root, temporary / 'wheel')
    uv_lock = await asyncio.to_thread(_prepare_uv_lock, project_root)
    packaged = {
        'source_package': source,
        'nexus_wheel': wheel,
        'uv_lock': uv_lock,
    }
    upload_scope = uuid.uuid4().hex
    refs: dict[str, dict[str, Any]] = {}
    for kind, item in packaged.items():
        refs[kind] = await api.upload_artifact(
            item.path,
            kind=kind,
            sha256=item.sha256,
            size_bytes=item.size_bytes,
            idempotency_key=f'tkex-e2e-{upload_scope}-{kind}-{item.sha256}',
            environment_id=environment_id,
        )
    return refs, {
        'build_and_upload_seconds': time.monotonic() - started,
        'artifacts': {
            kind: {
                'artifact_id': ref['artifact_id'],
                'kind': ref['kind'],
                'sha256': ref['sha256'],
                'size_bytes': ref['size_bytes'],
            }
            for kind, ref in refs.items()
        },
    }


async def _run_stage(
    api: CloudBatchApi,
    observer: TkexObserver,
    *,
    profile: BatchLoadProfile,
    payload: dict[str, Any],
    available_cpu_cores: float,
    available_memory_gib: float,
    database_metrics_reader: DatabaseMetricsReader | None,
    poll_interval: float,
    observation_interval: float,
    sandbox_release_observer: GongfengReleaseObserver | None = None,
    gongfeng_batch_id: str | None = None,
    manager_crash_controller: TkexManagerCrashController | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    artifact_baseline = await read_control_artifact_metrics(api)
    submitted = await _submit_batch_with_retry(api, payload)
    execution_id = str(submitted['execution_id'])
    stage: dict[str, Any] = {
        'name': profile.name,
        'execution_id': execution_id,
        'status': 'running',
        'submitted_at': datetime.now(timezone.utc).isoformat(),
        'peak_active_items': 0,
        'peak_running_items': 0,
        'scheduler_fill_seconds': None,
        'dependency_build_ids': [],
        'dependency_build_ids_by_workload': {},
        'concurrency_observations': [],
        'tkex_observations': [],
        'database_observations': [],
        'artifact_observations': [],
        'gongfeng_observations': [],
    }
    if manager_crash_controller is not None:
        stage['manager_crash'] = dict(manager_crash_controller.report)
    concurrency_sampler = _BatchConcurrencySampler(
        api,
        execution_id=execution_id,
        stage=stage,
        minimum_peak=profile.minimum_peak,
        interval=poll_interval,
    )
    concurrency_sampler.start()
    allocations: list[dict[str, Any]] = []
    allocations_ever_ready = False
    all_ready_at: float | None = None
    latest_tkex: dict[str, Any] = {
        'running_manager_containers': 0,
        'desired_replicas': 0,
        'ready_replicas': 0,
        'unreachable_managers': 0,
        'restart_count': 0,
        'oom_killed_container_count': 0,
        'services_present': [],
        'pod_ip_downward_api': True,
    }
    latest_db: dict[str, Any] | None = None
    latest_result_upload_failure_ratio = 0.0
    latest_completed_uploads = artifact_baseline['completed_uploads']
    next_observation_at = 0.0
    latest_live_ready_sandboxes: int | None = None
    progress_fingerprint: tuple[Any, ...] | None = None
    last_progress_at = time.monotonic()
    monitor_was_active = False
    demand = environment_demand(profile)
    stop_policy = _stress_policy_for_profile(profile)
    cluster_headroom_ratio = (
        min(
            available_cpu_cores / (demand['cpu_millicores'] / 1000),
            available_memory_gib / (demand['memory_bytes'] / GIB),
        )
        - 1
    )
    try:
        await concurrency_sampler.wait_for_first_sample()
        while True:
            concurrency_sampler.raise_if_failed()
            execution = await api.control_json('GET', f'/v1/executions/{execution_id}')
            allocation_payload = await api.control_json('GET', f'/v1/executions/{execution_id}/allocations')
            allocations = allocation_payload.get('items') or []
            now = time.monotonic()
            observed = False
            if _tkex_observation_due(
                allocations=allocations,
                now=now,
                next_observation_at=next_observation_at,
                manager_crash_controller=manager_crash_controller,
            ):
                observed = True
                latest_tkex = await observer.observe(allocations)
                latest_tkex = {'observed_at': datetime.now(timezone.utc).isoformat(), **latest_tkex}
                stage['tkex_observations'].append(latest_tkex)
                artifact_metrics = await read_control_artifact_metrics(api)
                latest_completed_uploads = artifact_metrics['completed_uploads']
                latest_result_upload_failure_ratio = result_upload_failure_ratio(
                    artifact_baseline,
                    artifact_metrics,
                )
                stage['artifact_observations'].append(
                    {
                        'observed_at': datetime.now(timezone.utc).isoformat(),
                        **artifact_metrics,
                        'result_upload_failure_ratio': latest_result_upload_failure_ratio,
                    }
                )
                if database_metrics_reader is not None:
                    latest_db = await database_metrics_reader()
                    stage['database_observations'].append(
                        {
                            'observed_at': datetime.now(timezone.utc).isoformat(),
                            **latest_db,
                        }
                    )
                next_observation_at = now + observation_interval
            concurrency_sampler.raise_if_failed()
            summary = concurrency_sampler.latest_summary
            generation_complete = summary.get('generation_status') == 'complete'
            now = time.monotonic()
            fingerprint = _stage_progress_fingerprint(
                execution,
                summary,
                completed_uploads=latest_completed_uploads,
            )
            if fingerprint != progress_fingerprint:
                progress_fingerprint = fingerprint
                last_progress_at = now
            if not allocations_ever_ready and _allocations_are_observed_ready(allocations, latest_tkex):
                allocations_ever_ready = True
                all_ready_at = now
            if (
                stage['scheduler_fill_seconds'] is None
                and all_ready_at is not None
                and concurrency_sampler.minimum_peak_reached_at is not None
            ):
                stage['scheduler_fill_seconds'] = max(0, concurrency_sampler.minimum_peak_reached_at - all_ready_at)
            if manager_crash_controller is not None:
                if (
                    observed
                    and manager_crash_controller.mode == 'execution'
                    and not manager_crash_controller.injected
                    and not manager_crash_controller.awaiting_confirmation
                    and stage['peak_running_items'] >= profile.minimum_peak
                ):
                    if sandbox_release_observer is None or gongfeng_batch_id is None:
                        raise AssertionError('execution manager crash requires Gongfeng batch observation')
                    gongfeng_snapshot = await sandbox_release_observer.batch_snapshot(gongfeng_batch_id)
                    latest_live_ready_sandboxes = int(gongfeng_snapshot['ready_runtime_count'])
                    stage['gongfeng_observations'].append(
                        {
                            'observed_at': datetime.now(timezone.utc).isoformat(),
                            'batch_id': gongfeng_batch_id,
                            'runtime_count': gongfeng_snapshot['runtime_count'],
                            'ready_runtime_count': latest_live_ready_sandboxes,
                            'active_runtime_count': len(gongfeng_snapshot['active_runtime_ids']),
                            'status_counts': gongfeng_snapshot['status_counts'],
                        }
                    )
                await _poll_manager_crash(
                    manager_crash_controller,
                    api,
                    observed=observed,
                    allocations=allocations,
                    profile=profile,
                    summary=summary,
                    peak_running_items=stage['peak_running_items'],
                    live_ready_sandboxes=latest_live_ready_sandboxes,
                    stage=stage,
                )
                if manager_crash_controller.recovery_elapsed_seconds > 600:
                    raise TimeoutError(
                        f'{manager_crash_controller.mode} manager crash did not recover within 600 seconds'
                    )
            monitor_active = allocations_ever_ready and generation_complete
            no_progress_seconds, last_progress_at = _monitor_no_progress_seconds(
                monitor_active=monitor_active,
                monitor_was_active=monitor_was_active,
                last_progress_at=last_progress_at,
                now=now,
            )
            monitor_was_active = monitor_active
            restart_or_oom = max(
                int(latest_tkex.get('restart_count') or 0),
                int(latest_tkex.get('oom_killed_container_count') or 0),
            )
            database_ratio = None
            if latest_db is not None and latest_db['max_connections'] > 0:
                database_ratio = latest_db['threads_connected'] / latest_db['max_connections']
            observation = StressObservation(
                running_manager_containers=int(latest_tkex.get('running_manager_containers') or 0),
                cluster_headroom_ratio=cluster_headroom_ratio,
                database_connection_ratio=database_ratio,
                unreachable_managers=(
                    unplanned_unreachable_managers(
                        int(latest_tkex.get('unreachable_managers') or 0),
                        planned_pending=(
                            manager_crash_controller.planned_pending if manager_crash_controller is not None else 0
                        ),
                    )
                    if allocations_ever_ready
                    else 0
                ),
                no_progress_seconds=no_progress_seconds,
                oom_or_restarts=restart_or_oom,
                result_upload_failure_ratio=latest_result_upload_failure_ratio,
                sustained_gateway_5xx=api.consecutive_gateway_5xx >= 3,
            )
            reasons = stress_stop_reasons(observation, policy=stop_policy)
            reasons.extend(_tkex_observation_stop_reasons(latest_tkex))
            if reasons:
                stage['guardrail_stop_reasons'] = reasons
                raise RuntimeError(f'{profile.name} stress guardrail triggered: {reasons}')
            if execution.get('status') in TERMINAL_EXECUTION_STATUSES:
                if execution['status'] != 'succeeded':
                    raise AssertionError(f'{profile.name} execution ended as {execution["status"]}: {execution}')
                break
            await asyncio.sleep(poll_interval)
    except BaseException as exc:
        with contextlib.suppress(BaseException):
            await concurrency_sampler.stop()
        await _raise_failed_stage(api, observer, stage=stage, execution_id=execution_id, cause=exc)

    try:
        await concurrency_sampler.stop()
    except BaseException as exc:
        await _raise_failed_stage(api, observer, stage=stage, execution_id=execution_id, cause=exc)

    try:
        final_summary = await api.control_json('GET', f'/v1/executions/{execution_id}/batch')
        items = await api.list_all_batch_items(execution_id)
        child_ids = {
            str(item['current_child_execution_id']) for item in items if item.get('current_child_execution_id')
        }
        facts = await api.execution_facts(child_ids)
        stage['validation'] = validate_completed_batch(profile, summary=final_summary, items=items, facts=facts)
        if manager_crash_controller is not None:
            manager_crash_controller.validate_completed(parent=execution, summary=final_summary)
            stage['manager_crash'] = dict(manager_crash_controller.report)
        if stage['peak_running_items'] < profile.minimum_peak:
            raise AssertionError(
                f'{profile.name} peak running items {stage["peak_running_items"]} did not reach {profile.minimum_peak}'
            )
        if profile.code_mode == 'source_package':
            outputs = await _resolve_outputs(api, items=items, facts=facts)
            stage['source_validation'] = validate_source_outputs(profile, outputs)
        elif profile.task_kind == 'sandbox_agent_echo':
            if sandbox_release_observer is None:
                raise AssertionError(f'{profile.name} requires a Gongfeng release observer')
            outputs = await _resolve_outputs(api, items=items, facts=facts)
            stage['sandbox_validation'] = validate_sandbox_outputs(profile, outputs)
            runtime_ids = [str((output.get('task_output') or {}).get('runtime_id') or '') for output in outputs]
            stage['sandbox_validation'].update(
                await sandbox_release_observer.wait_released(
                    runtime_ids,
                    timeout=900,
                    poll_interval=10,
                )
            )
            if gongfeng_batch_id is None:
                raise AssertionError(f'{profile.name} requires a Gongfeng batch ID')
            stage['sandbox_batch_release'] = await sandbox_release_observer.wait_batch_released(
                gongfeng_batch_id,
                minimum_runtime_count=profile.item_count,
                timeout=900,
                poll_interval=10,
            )
        dependency_builds = dependency_build_ids_by_workload(allocations)
        if profile.code_mode == 'source_package':
            dependency_builds = validate_source_dependency_builds(profile.name, allocations)
        stage['dependency_build_ids_by_workload'] = dependency_builds
        stage['dependency_build_ids'] = sorted(
            {dependency_build_id for build_ids in dependency_builds.values() for dependency_build_id in build_ids}
        )
        await _wait_stage_released(api, observer, execution_id=execution_id, timeout=1800)
    except BaseException as exc:
        await _raise_failed_stage(api, observer, stage=stage, execution_id=execution_id, cause=exc)
    stage['status'] = 'passed'
    stage['cleanup'] = {'status': 'released'}
    stage['total_seconds'] = time.monotonic() - started
    stage['finished_at'] = datetime.now(timezone.utc).isoformat()
    return stage


async def _raise_failed_stage(
    api: CloudBatchApi,
    observer: TkexObserver,
    *,
    stage: dict[str, Any],
    execution_id: str,
    cause: BaseException,
) -> None:
    stage['status'] = 'failed'
    stage['error'] = {'type': type(cause).__name__, 'message': str(cause)}
    await _request_stop(api, execution_id)
    try:
        await _wait_stage_released(api, observer, execution_id=execution_id, timeout=1800)
    except BaseException as cleanup_error:
        stage['cleanup'] = {
            'status': 'failed',
            'error': {'type': type(cleanup_error).__name__, 'message': str(cleanup_error)},
        }
    else:
        stage['cleanup'] = {'status': 'released'}
    stage['finished_at'] = datetime.now(timezone.utc).isoformat()
    raise StageRunError(stage, cause) from cause


async def _resolve_outputs(
    api: CloudBatchApi,
    *,
    items: list[dict[str, Any]],
    facts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(50)

    async def resolve(item: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await api.resolved_result(facts[str(item['current_child_execution_id'])])

    return list(await asyncio.gather(*(resolve(item) for item in items)))


_resolve_source_outputs = _resolve_outputs


async def _request_stop(api: CloudBatchApi, execution_id: str) -> None:
    with contextlib.suppress(Exception):
        await api.control_json('POST', f'/v1/executions/{execution_id}/stop', body={})


async def _wait_stage_released(
    api: CloudBatchApi,
    observer: TkexObserver,
    *,
    execution_id: str,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    latest_allocations: list[dict[str, Any]] = []
    latest_tkex: dict[str, Any] = {}
    while time.monotonic() < deadline:
        payload = await api.control_json('GET', f'/v1/executions/{execution_id}/allocations')
        latest_allocations = payload.get('items') or []
        if latest_allocations:
            latest_tkex = await observer.observe(latest_allocations)
        if (
            latest_allocations
            and all(item.get('status') == 'released' for item in latest_allocations)
            and int(latest_tkex.get('active_pod_count') or 0) == 0
            and int(latest_tkex.get('workload_count') or 0) == 0
        ):
            return
        await asyncio.sleep(2)
    raise TimeoutError(
        f'allocation resources were not fully released for {execution_id}: '
        f'allocations={latest_allocations}, tkex={latest_tkex}'
    )


def _allocations_are_observed_ready(
    allocations: list[dict[str, Any]],
    tkex_observation: dict[str, Any],
) -> bool:
    if not allocations or not all(
        item.get('status') == 'ready'
        and int(item.get('ready_manager_count') or 0) == int(item.get('desired_manager_count') or 0)
        for item in allocations
    ):
        return False
    desired_managers = sum(int(item.get('desired_manager_count') or 0) for item in allocations)
    return (
        desired_managers > 0
        and int(tkex_observation.get('desired_replicas') or 0) == desired_managers
        and int(tkex_observation.get('ready_replicas') or 0) == desired_managers
        and int(tkex_observation.get('unreachable_managers') or 0) == 0
    )


def summarize_tkex_pods(payload: dict[str, Any], *, allocation_ids: set[str]) -> dict[str, Any]:
    pods = [
        pod
        for pod in payload.get('items') or []
        if ((pod.get('metadata') or {}).get('annotations') or {}).get('nexus.io/allocation-id') in allocation_ids
    ]
    active = [pod for pod in pods if (pod.get('status') or {}).get('phase') not in {'Succeeded', 'Failed'}]
    ready = [pod for pod in active if _pod_is_ready(pod)]
    pod_ips = sorted(
        {str((pod.get('status') or {}).get('podIP')) for pod in ready if (pod.get('status') or {}).get('podIP')}
    )
    statuses = [status for pod in active for status in ((pod.get('status') or {}).get('containerStatuses') or [])]
    container_counts = [len((pod.get('spec') or {}).get('containers') or []) for pod in active]
    return {
        'pod_count': len(pods),
        'active_pod_count': len(active),
        'running_manager_containers': sum(container_counts),
        'maximum_containers_per_pod': max(container_counts, default=0),
        'ready_pod_count': len(ready),
        'ready_pods_missing_ip': sum(not (pod.get('status') or {}).get('podIP') for pod in ready),
        'pod_ip_count': len(pod_ips),
        'pod_ip_sample': pod_ips[:5],
        'restart_count': sum(int(status.get('restartCount') or 0) for status in statuses),
        'oom_killed_container_count': sum(_container_was_oom_killed(status) for status in statuses),
    }


def dependency_build_ids_by_workload(allocations: list[dict[str, Any]]) -> dict[str, list[str]]:
    build_ids: dict[str, set[str]] = {}
    for allocation in allocations:
        workload = allocation.get('workload')
        if not isinstance(workload, str) or not workload:
            continue
        workload_build_ids = build_ids.setdefault(workload, set())
        direct_id = allocation.get('dependency_build_id')
        if isinstance(direct_id, str) and direct_id:
            workload_build_ids.add(direct_id)
        nested_id = (allocation.get('dependency_build') or {}).get('dependency_build_id')
        if isinstance(nested_id, str) and nested_id:
            workload_build_ids.add(nested_id)
    return {workload: sorted(ids) for workload, ids in sorted(build_ids.items())}


def validate_source_dependency_builds(
    profile_name: str,
    allocations: list[dict[str, Any]],
) -> dict[str, list[str]]:
    build_ids = dependency_build_ids_by_workload(allocations)
    expected_workloads = {'execution', 'orchestration'}
    if set(build_ids) != expected_workloads or any(len(ids) != 1 for ids in build_ids.values()):
        raise AssertionError(f'{profile_name} did not use one DependencyBuild per workload: {build_ids}')
    return build_ids


def validate_completed_batch(
    profile: BatchLoadProfile,
    *,
    summary: dict[str, Any],
    items: list[dict[str, Any]],
    facts: dict[str, dict[str, Any]],
) -> dict[str, int]:
    final = summary.get('summary') or {}
    if final.get('total') != profile.item_count or final.get('succeeded') != profile.item_count:
        raise AssertionError(f'{profile.name} final summary is incomplete: {final}')
    failed = sum(int(final.get(key) or 0) for key in ('failed', 'killed', 'lost', 'expired', 'cancelled'))
    if failed:
        raise AssertionError(f'{profile.name} has {failed} non-success terminal items')
    if len(items) != profile.item_count or len({item['item_id'] for item in items}) != profile.item_count:
        raise AssertionError(f'{profile.name} item ledger is incomplete or contains duplicates')
    if any(item.get('status') != 'succeeded' for item in items):
        raise AssertionError(f'{profile.name} item ledger contains non-success rows')
    child_ids = [item.get('current_child_execution_id') for item in items]
    if any(not child_id for child_id in child_ids) or len(set(child_ids)) != profile.item_count:
        raise AssertionError(f'{profile.name} child executions are missing or not unique')
    if set(child_ids) != set(facts):
        raise AssertionError(f'{profile.name} child execution facts do not match the item ledger')
    if any(fact.get('status') != 'succeeded' for fact in facts.values()):
        raise AssertionError(f'{profile.name} child execution facts contain non-success rows')
    result_ids = [fact.get('result_artifact_id') for fact in facts.values()]
    if any(not result_id for result_id in result_ids):
        raise AssertionError(f'{profile.name} child execution is missing a result artifact')
    if len(set(result_ids)) != profile.item_count:
        raise AssertionError(f'{profile.name} result artifacts are not unique')
    return {
        'item_count': len(items),
        'child_execution_count': len(facts),
        'unique_result_artifact_count': len(set(result_ids)),
    }


def validate_source_outputs(profile: BatchLoadProfile, outputs: list[dict[str, Any]]) -> dict[str, list[str]]:
    if len(outputs) != profile.item_count:
        raise AssertionError(f'{profile.name} source output count does not match the profile')
    task_outputs = [output.get('task_output') or {} for output in outputs]
    markers = {task_output.get('marker') for task_output in task_outputs}
    expected_markers = {f'source-package-ok:{index}' for index in range(profile.item_count)}
    if markers != expected_markers:
        raise AssertionError(f'{profile.name} source markers do not match generated items')
    dependency_versions = sorted({str(output.get('dependency_version') or '') for output in task_outputs})
    sys_prefixes = sorted({str(output.get('sys_prefix') or '') for output in task_outputs})
    nexus_files = sorted({str(output.get('nexus_file') or '') for output in task_outputs})
    if not dependency_versions or '' in dependency_versions:
        raise AssertionError(f'{profile.name} source output is missing the wheelhouse dependency version')
    if not sys_prefixes or any('/manager-venvs/' not in value for value in sys_prefixes):
        raise AssertionError(f'{profile.name} source output did not run from a Manager venv')
    if not nexus_files or any(
        not any(nexus_file.startswith(f'{prefix.rstrip("/")}/') for prefix in sys_prefixes)
        for nexus_file in nexus_files
    ):
        raise AssertionError(f'{profile.name} Nexus wheel was not imported from the Manager venv')
    return {
        'dependency_versions': dependency_versions,
        'sys_prefixes': sys_prefixes,
        'nexus_files': nexus_files,
    }


def validate_sandbox_outputs(profile: BatchLoadProfile, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(outputs) != profile.item_count:
        raise AssertionError(f'{profile.name} sandbox output count does not match the profile')
    if profile.minimum_sandbox_peak is None:
        raise AssertionError(f'{profile.name} does not define a minimum sandbox peak')

    task_outputs = [output.get('task_output') or {} for output in outputs]
    expected_markers = {f'{profile.name}-{index:012d}' for index in range(profile.item_count)}
    markers = {str(output.get('marker') or '') for output in task_outputs}
    if markers != expected_markers:
        raise AssertionError(f'{profile.name} sandbox markers do not match generated items')
    if any(
        int(output.get('command_return_code', -1)) != 0
        or str(output.get('command_stdout') or '') != str(output.get('marker') or '')
        for output in task_outputs
    ):
        raise AssertionError(f'{profile.name} contains an unverified agent shell command')

    runtime_ids = [str(output.get('runtime_id') or '') for output in task_outputs]
    if any(not runtime_id for runtime_id in runtime_ids) or len(set(runtime_ids)) != profile.item_count:
        raise AssertionError(f'{profile.name} runtime IDs are missing or not unique')

    events: list[tuple[float, int]] = []
    for output in task_outputs:
        try:
            ready_at = float(output['sandbox_ready_at'])
            finished_at = float(output['sandbox_hold_finished_at'])
        except (KeyError, TypeError, ValueError) as exc:
            raise AssertionError(f'{profile.name} contains an invalid sandbox ready interval') from exc
        if finished_at <= ready_at:
            raise AssertionError(f'{profile.name} contains a non-positive sandbox ready interval')
        events.extend(((ready_at, 1), (finished_at, -1)))

    ready = 0
    peak_ready = 0
    for _timestamp, delta in sorted(events, key=lambda event: (event[0], event[1])):
        ready += delta
        if ready < 0:
            raise AssertionError(f'{profile.name} contains inconsistent sandbox ready intervals')
        peak_ready = max(peak_ready, ready)
    if peak_ready < profile.minimum_sandbox_peak:
        raise AssertionError(
            f'{profile.name} peak ready sandboxes {peak_ready} did not reach {profile.minimum_sandbox_peak}'
        )

    exit_statuses = Counter(str(output.get('agent_exit_status') or 'unknown') for output in task_outputs)
    return {
        'agent_exit_statuses': dict(sorted(exit_statuses.items())),
        'peak_ready_sandboxes': peak_ready,
        'unique_marker_count': len(markers),
        'unique_runtime_id_count': len(set(runtime_ids)),
    }


def _pod_is_ready(pod: dict[str, Any]) -> bool:
    return any(
        condition.get('type') == 'Ready' and condition.get('status') == 'True'
        for condition in ((pod.get('status') or {}).get('conditions') or [])
    )


def _container_was_oom_killed(status: dict[str, Any]) -> bool:
    for state_name in ('state', 'lastState'):
        terminated = (status.get(state_name) or {}).get('terminated') or {}
        if terminated.get('reason') == 'OOMKilled':
            return True
    return False


def _tkex_observer_config(args: argparse.Namespace) -> TkexObserverConfig:
    values = {
        'base_url': args.tkex_base_url,
        'api_account': args.tkex_api_account,
        'api_token_file': args.tkex_api_token_file,
        'cluster_name': args.tkex_cluster_name,
        'project_name': args.tkex_project_name,
        'namespace': args.tkex_namespace,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(f'TKEx observation arguments are required for execute: {", ".join(missing)}')
    return TkexObserverConfig(**values, expected_workload_image=args.workload_image)


def _validate_execute_prerequisites(args: argparse.Namespace, profiles: list[BatchLoadProfile]) -> None:
    _tkex_observer_config(args)
    _read_secret_file(args.taihu_token_file, label='Taihu token')
    images = {
        'universal': args.universal_image,
        'workload': args.workload_image,
        'builder': args.builder_image,
    }
    missing_images = [name for name, value in images.items() if not value]
    if missing_images:
        raise ValueError(f'image references are required for execute: {", ".join(missing_images)}')
    invalid_images = [name for name, value in images.items() if not re.fullmatch(r'.+@sha256:[0-9a-f]{64}', str(value))]
    if invalid_images:
        raise ValueError(f'image references must be digest pinned: {", ".join(invalid_images)}')
    if args.database_url_file is not None and args.control_database_metrics:
        raise ValueError('--database-url-file and --control-database-metrics are mutually exclusive')
    if any(profile.task_kind == 'sandbox_agent_echo' for profile in profiles):
        if args.gongfeng_token_file is None:
            raise ValueError('sandbox_agent_echo profiles require --gongfeng-token-file')
        _read_secret_file(args.gongfeng_token_file, label='Gongfeng token')
    crash_mode = getattr(args, 'manager_crash', 'none')
    if crash_mode != 'none':
        if (
            not getattr(args, 'confirm_manager_pod_delete', False)
            and getattr(args, 'manager_crash_approval_file', None) is None
        ):
            raise ValueError(
                'manager crash execution requires --confirm-manager-pod-delete or --manager-crash-approval-file'
            )
        sandbox_profiles = [profile for profile in profiles if profile.task_kind == 'sandbox_agent_echo']
        if len(sandbox_profiles) != 1:
            raise ValueError('manager crash execution requires exactly one sandbox_agent_echo profile')
        crash_count = int(getattr(args, 'execution_manager_crash_count', 20))
        if crash_mode == 'execution' and (
            crash_count > 100 or crash_count > sandbox_profiles[0].execution_manager_count
        ):
            raise ValueError('execution manager crash count must not exceed 100 or the execution manager count')
    large_profiles = [profile for profile in profiles if profile.manager_pod_count > 100]
    if large_profiles:
        if args.database_url_file is None and not args.control_database_metrics:
            raise ValueError(
                'profiles above 100 Manager pods require database metrics via '
                '--control-database-metrics or --database-url-file'
            )
        if args.database_url_file is not None:
            _read_secret_file(args.database_url_file, label='database URL')
        large_source_profiles = [profile for profile in large_profiles if profile.code_mode == 'source_package']
        if large_source_profiles and (not profiles or profiles[0].name != 'wheelhouse-smoke'):
            raise ValueError(
                'source-package profiles above 100 Manager pods require wheelhouse-smoke as the first stage'
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profiles = fit_profiles_to_container_budget(
        selected_profiles(args.profile, allow_extreme=args.allow_extreme),
        containers_per_manager_pod=args.containers_per_manager_pod,
    )
    if args.available_cpu_cores is None or args.available_memory_gib is None:
        raise SystemExit('--available-cpu-cores and --available-memory-gib are required for a guarded plan')
    plan = build_execution_plan(
        profiles,
        containers_per_manager_pod=args.containers_per_manager_pod,
        available_cpu_cores=args.available_cpu_cores,
        available_memory_gib=args.available_memory_gib,
    )
    plan['typed_smoke'] = args.typed_smoke
    report_path = args.report_path or Path(tempfile.gettempdir()) / (
        f'nexus-tkex-{args.tkex_cluster_name or "unselected"}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json'
    )
    print(
        json.dumps({'execute': args.execute, 'plan': plan, 'report_path': str(report_path)}, indent=2, sort_keys=True)
    )
    if args.execute:
        _validate_execute_prerequisites(args, profiles)
        asyncio.run(execute_profiles(args, profiles=profiles, plan=plan, report_path=report_path))
        print(f'TKEx cloud-batch report: {report_path}')
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return parsed


if __name__ == '__main__':
    raise SystemExit(main())
