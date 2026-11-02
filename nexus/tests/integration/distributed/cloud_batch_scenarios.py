from __future__ import annotations

import hashlib
import json
import math
import shutil

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal


GIB = 1024**3
MAX_STRESS_CONTAINERS = 3001
CodeMode = Literal['config_only', 'source_package']
TaskKind = Literal['sleep_echo', 'sandbox_agent_echo']
DEFAULT_WHEELHOUSE_EXTRAS = ('distributed-db-shell',)
SOURCE_FIXTURE_MODULE = 'cloud_source_fixture.py'

STRESS_ENVIRONMENT_LIMITS = {
    'manager_count': 3001,
    'cpu_millicores': 3_010_000,
    'memory_bytes': 3010 * GIB,
    'slots': 60_001,
    'active_batches': 1,
    'max_items': 100_000,
}


@dataclass(frozen=True, slots=True)
class StressStopPolicy:
    max_running_containers: int = MAX_STRESS_CONTAINERS
    minimum_cluster_headroom_ratio: float = 0.10
    maximum_database_connection_ratio: float = 0.70
    maximum_unreachable_managers: int = 2
    no_progress_seconds: int = 120
    maximum_oom_or_restarts: int = 5
    maximum_result_upload_failure_ratio: float = 0.01


STRESS_STOP_POLICY = StressStopPolicy()


@dataclass(frozen=True, slots=True)
class StressObservation:
    running_manager_containers: int
    cluster_headroom_ratio: float | None = None
    database_connection_ratio: float | None = None
    unreachable_managers: int = 0
    no_progress_seconds: float = 0
    oom_or_restarts: int = 0
    result_upload_failure_ratio: float = 0
    sustained_gateway_5xx: bool = False


def stress_stop_reasons(
    observation: StressObservation,
    *,
    policy: StressStopPolicy = STRESS_STOP_POLICY,
) -> list[str]:
    reasons: list[str] = []
    if observation.running_manager_containers > policy.max_running_containers:
        reasons.append(
            f'running_manager_containers={observation.running_manager_containers} exceeds '
            f'{policy.max_running_containers}'
        )
    if (
        observation.cluster_headroom_ratio is not None
        and observation.cluster_headroom_ratio < policy.minimum_cluster_headroom_ratio
    ):
        reasons.append(
            f'cluster_headroom_ratio={observation.cluster_headroom_ratio:.3f} below '
            f'{policy.minimum_cluster_headroom_ratio:.3f}'
        )
    if (
        observation.database_connection_ratio is not None
        and observation.database_connection_ratio > policy.maximum_database_connection_ratio
    ):
        reasons.append(
            f'database_connection_ratio={observation.database_connection_ratio:.3f} exceeds '
            f'{policy.maximum_database_connection_ratio:.3f}'
        )
    if observation.unreachable_managers > policy.maximum_unreachable_managers:
        reasons.append(
            f'unreachable_managers={observation.unreachable_managers} exceeds {policy.maximum_unreachable_managers}'
        )
    if observation.no_progress_seconds > policy.no_progress_seconds:
        reasons.append(f'no_progress_seconds={observation.no_progress_seconds:g} exceeds {policy.no_progress_seconds}')
    if observation.oom_or_restarts >= policy.maximum_oom_or_restarts:
        reasons.append(f'oom_or_restarts={observation.oom_or_restarts} reached {policy.maximum_oom_or_restarts}')
    if observation.result_upload_failure_ratio > policy.maximum_result_upload_failure_ratio:
        reasons.append(
            f'result_upload_failure_ratio={observation.result_upload_failure_ratio:.3f} exceeds '
            f'{policy.maximum_result_upload_failure_ratio:.3f}'
        )
    if observation.sustained_gateway_5xx:
        reasons.append('sustained_gateway_5xx=true')
    return reasons


@dataclass(frozen=True, slots=True)
class BatchLoadProfile:
    name: str
    code_mode: CodeMode
    item_count: int
    execution_manager_count: int
    slots_per_manager: int
    max_inflight: int
    minimum_peak: int
    task_kind: TaskKind = 'sleep_echo'
    minimum_sandbox_peak: int | None = None
    task_sleep_seconds: int = 20
    timeout_per_run_seconds: int = 120
    batch_manager_cpu_cores: float = 1.0
    batch_manager_memory_gib: float = 1.0
    execution_manager_cpu_cores: float = 1.0
    execution_manager_memory_gib: float = 1.0
    retry_failed: int = 0
    stop_timeout_seconds: int | None = None
    opt_in: bool = False

    def __post_init__(self) -> None:
        if self.code_mode not in {'config_only', 'source_package'}:
            raise ValueError(f'unsupported code_mode: {self.code_mode}')
        if self.task_kind not in {'sleep_echo', 'sandbox_agent_echo'}:
            raise ValueError(f'unsupported task_kind: {self.task_kind}')
        if self.code_mode == 'source_package' and self.task_kind != 'sleep_echo':
            raise ValueError('source_package profiles only support sleep_echo tasks')
        if self.item_count <= 0:
            raise ValueError('item_count must be positive')
        if self.execution_manager_count <= 0 or self.manager_pod_count > MAX_STRESS_CONTAINERS:
            raise ValueError(f'manager pod count must be between 2 and {MAX_STRESS_CONTAINERS}')
        if self.slots_per_manager <= 0:
            raise ValueError('slots_per_manager must be positive')
        if self.max_inflight <= 0 or self.max_inflight > self.execution_slot_count:
            raise ValueError('max_inflight must be positive and no greater than execution slots')
        if self.minimum_peak <= 0 or self.minimum_peak > self.max_inflight:
            raise ValueError('minimum_peak must be positive and no greater than max_inflight')
        if self.task_kind == 'sandbox_agent_echo':
            if self.minimum_sandbox_peak is None:
                raise ValueError('sandbox_agent_echo profiles require minimum_sandbox_peak')
            if self.minimum_sandbox_peak <= 0 or self.minimum_sandbox_peak > self.max_inflight:
                raise ValueError('minimum_sandbox_peak must be positive and no greater than max_inflight')
        elif self.minimum_sandbox_peak is not None:
            raise ValueError('minimum_sandbox_peak is only valid for sandbox_agent_echo profiles')
        if self.item_count < self.max_inflight:
            raise ValueError('item_count must be no less than max_inflight')
        if self.task_sleep_seconds < 0:
            raise ValueError('task_sleep_seconds must be non-negative')
        if self.timeout_per_run_seconds < self.task_sleep_seconds + 60:
            raise ValueError('timeout_per_run_seconds must include at least 60 seconds beyond task_sleep_seconds')
        if self.retry_failed < 0:
            raise ValueError('retry_failed must be non-negative')
        if self.stop_timeout_seconds is not None and self.stop_timeout_seconds <= 0:
            raise ValueError('stop_timeout_seconds must be positive')
        if self.batch_manager_cpu_cores <= 0 or self.batch_manager_memory_gib <= 0:
            raise ValueError('batch manager resources must be positive')
        if self.execution_manager_cpu_cores <= 0 or self.execution_manager_memory_gib <= 0:
            raise ValueError('execution manager resources must be positive')

    @property
    def manager_pod_count(self) -> int:
        return 1 + self.execution_manager_count

    @property
    def execution_slot_count(self) -> int:
        return self.execution_manager_count * self.slots_per_manager

    def manager_container_count(self, *, containers_per_pod: int) -> int:
        if containers_per_pod <= 0:
            raise ValueError('containers_per_pod must be positive')
        return self.manager_pod_count * containers_per_pod


def validate_profile_for_cluster(profile: BatchLoadProfile, *, containers_per_manager_pod: int) -> None:
    actual = profile.manager_container_count(containers_per_pod=containers_per_manager_pod)
    if actual > MAX_STRESS_CONTAINERS:
        raise ValueError(
            f'{profile.name} would create {actual} running manager containers; limit is {MAX_STRESS_CONTAINERS}'
        )
    demand = environment_demand(profile)
    exceeded = {
        key: value
        for key, value in demand.items()
        if key in STRESS_ENVIRONMENT_LIMITS and value > STRESS_ENVIRONMENT_LIMITS[key]
    }
    if profile.item_count > STRESS_ENVIRONMENT_LIMITS['max_items']:
        exceeded['max_items'] = profile.item_count
    if exceeded:
        raise ValueError(f'{profile.name} exceeds stress environment limits: {exceeded}')


def fit_profile_to_container_budget(
    profile: BatchLoadProfile,
    *,
    containers_per_manager_pod: int,
) -> BatchLoadProfile:
    if containers_per_manager_pod <= 0:
        raise ValueError('containers_per_manager_pod must be positive')
    maximum_manager_pods = MAX_STRESS_CONTAINERS // containers_per_manager_pod
    if maximum_manager_pods < 2:
        raise ValueError('container budget cannot fit a batch manager and an execution manager')
    if profile.manager_pod_count <= maximum_manager_pods:
        return profile

    execution_manager_count = maximum_manager_pods - 1
    execution_slots = execution_manager_count * profile.slots_per_manager
    max_inflight = min(profile.max_inflight, execution_slots)
    minimum_peak = max(1, int(profile.minimum_peak * max_inflight / profile.max_inflight))
    minimum_sandbox_peak = (
        max(1, int(profile.minimum_sandbox_peak * max_inflight / profile.max_inflight))
        if profile.minimum_sandbox_peak is not None
        else None
    )
    fitted = replace(
        profile,
        execution_manager_count=execution_manager_count,
        max_inflight=max_inflight,
        minimum_peak=minimum_peak,
        minimum_sandbox_peak=minimum_sandbox_peak,
    )
    validate_profile_for_cluster(fitted, containers_per_manager_pod=containers_per_manager_pod)
    return fitted


def environment_demand(profile: BatchLoadProfile) -> dict[str, int]:
    cpu_cores = profile.batch_manager_cpu_cores + profile.execution_manager_count * profile.execution_manager_cpu_cores
    memory_gib = (
        profile.batch_manager_memory_gib + profile.execution_manager_count * profile.execution_manager_memory_gib
    )
    return {
        'manager_count': profile.manager_pod_count,
        'cpu_millicores': round(cpu_cores * 1000),
        'memory_bytes': round(memory_gib * GIB),
        'slots': profile.execution_slot_count + 1,
        'active_batches': 1,
    }


def source_generator_path(profile: BatchLoadProfile) -> str:
    return f'cloud_generators/{profile.name.replace("-", "_")}.py'


def write_config_task_list(path: Path, *, profile: BatchLoadProfile, run_id_prefix: str) -> dict[str, Any]:
    if profile.code_mode != 'config_only':
        raise ValueError('config task list requires a config_only profile')
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open('wb') as stream:
        for index in range(profile.item_count):
            stable_id = f'{index:012d}'
            if profile.task_kind == 'sandbox_agent_echo':
                marker = f'{profile.name}-{stable_id}'
                run_input = _sandbox_agent_run_input(
                    profile=profile,
                    run_id=f'{run_id_prefix}-{stable_id}',
                    batch_id=run_id_prefix,
                    index=index,
                    marker=marker,
                )
            else:
                run_input = {
                    'run_id': f'{run_id_prefix}-{stable_id}',
                    'imports': ['tests.integration.distributed.cloud_tasks'],
                    'task_name': 'sleep_echo',
                    'task_config': {
                        'index': index,
                        'message': f'{profile.name}-{stable_id}',
                        'sleep_seconds': profile.task_sleep_seconds,
                    },
                }
            record = {
                'version': 1,
                'item_id': f'item-{stable_id}',
                'run_input': run_input,
            }
            line = json.dumps(record, sort_keys=True, separators=(',', ':')).encode() + b'\n'
            stream.write(line)
            digest.update(line)
            size_bytes += len(line)
    return {
        'version': 1,
        'item_count': profile.item_count,
        'size_bytes': size_bytes,
        'sha256': digest.hexdigest(),
        'source': {'mode': 'cloud_batch_e2e', 'profile': profile.name},
        'datasets': [],
    }


def _sandbox_agent_run_input(
    *,
    profile: BatchLoadProfile,
    run_id: str,
    batch_id: str,
    index: int,
    marker: str,
) -> dict[str, Any]:
    return {
        'run_id': run_id,
        'imports': [
            'tests.integration.distributed.cloud_tasks',
            'nexus.extensions.agents.main',
            'nexus.extensions.agents.swe_agent.tools',
        ],
        'task_name': 'sandbox_agent_echo',
        'task_config': {
            'index': index,
            'marker': marker,
            'hold_seconds': profile.task_sleep_seconds,
        },
        'runtime_provider_name': 'gongfeng',
        'runtime_provider_config': {
            'api_base_url': 'http://hyrl-sandbox.prod.woa.com:8052',
            'token__env': 'GONGFENG_TOKEN',
            'runtime_image': 'current',
            'runtime_provider': 'gongfeng',
            'resource_spec': {'cpu': 1, 'memory': 1},
            'request_timeout': 1800,
            'queue_wait_max': 1800,
            'startup_jitter_max': 30,
            'auto_connect_ws': False,
            'rpc_protocol': 'http',
            'batch_id': batch_id,
            'purpose': 'test',
            'print_port_mappings': False,
        },
        'runtime_configs': {
            'main': {'image': 'mirrors.tencent.com/tlinux/tlinux4.0-minimal'},
        },
        'agent_name': 'main_agent',
        'agent_config': {
            'tools': ['swe_agent/bash'],
            'max_iterations': 3,
            'parallel_tool_calls': False,
        },
        'openai_clients': {
            'main': {
                'name': 'openai',
                'config': {
                    'client_args': {
                        'api_key': (
                            '${oc.env:MODEL_SERVICE_API_APP_ID}:${oc.env:MODEL_SERVICE_API_APP_KEY}'
                            '?provider=azure_openai&model=${oc.env:NEXUS_MODEL,"gpt-5.5"}'
                            '&timeout=${oc.env:MODEL_SERVICE_API_TIMEOUT,"60"}'
                        ),
                        'base_url': '${oc.env:MODEL_SERVICE_API_BASE_URL}',
                    },
                    'request_args': {'model': '${oc.env:NEXUS_MODEL,"gpt-5.5"}'},
                },
            }
        },
    }


def batch_submission(
    *,
    profile: BatchLoadProfile,
    environment_id: str,
    idempotency_key: str,
    task_list_ref: dict[str, Any] | None = None,
    task_list_manifest: dict[str, Any] | None = None,
    source_package_ref: dict[str, Any] | None = None,
    nexus_wheel_ref: dict[str, Any] | None = None,
    uv_lock_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if profile.code_mode == 'config_only':
        if task_list_ref is None or task_list_manifest is None:
            raise ValueError('config_only batch requires task list inputs')
        code = {'mode': 'config_only'}
        generation = {
            'mode': 'local_task_list',
            'task_list': {'ref': task_list_ref, 'format': 'jsonl', 'manifest': task_list_manifest},
            'timeout_seconds': 900,
        }
    else:
        if source_package_ref is None or nexus_wheel_ref is None or uv_lock_ref is None:
            raise ValueError('source_package batch requires all source package inputs')
        code = {
            'mode': 'source_package',
            'source_package': source_package_ref,
            'nexus_wheel': nexus_wheel_ref,
            'uv_lock': uv_lock_ref,
            'extras': list(DEFAULT_WHEELHOUSE_EXTRAS),
        }
        generation = {
            'mode': 'remote_materialize',
            'source': {'type': 'generator', 'path': source_generator_path(profile)},
            'timeout_seconds': 900,
        }
    execution_waves = math.ceil(profile.item_count / profile.max_inflight)
    sandbox_profile = profile.task_kind == 'sandbox_agent_echo'
    return {
        'submission_version': 1,
        'idempotency_key': idempotency_key,
        'environment_id': environment_id,
        'code': code,
        'resources': {
            'batch_manager': {
                'cpu_cores': profile.batch_manager_cpu_cores,
                'memory_gib': profile.batch_manager_memory_gib,
            },
            'managers': {
                'count': profile.execution_manager_count,
                'cpu_cores': profile.execution_manager_cpu_cores,
                'memory_gib': profile.execution_manager_memory_gib,
                'slots': profile.slots_per_manager,
            },
        },
        'generation': generation,
        'batch': {
            'max_inflight': profile.max_inflight,
            'retry_failed': profile.retry_failed,
            'fail_fast': False,
            'child_queue_ttl_seconds': 3600 if sandbox_profile else 1800,
            'timeout_per_run': profile.timeout_per_run_seconds,
        },
        'provisioning_timeout_seconds': 3600 if sandbox_profile else 1800,
        'queue_ttl_seconds': 3600 if sandbox_profile else 1800,
        'execution_ttl_seconds': max(7200, profile.timeout_per_run_seconds * execution_waves + 1800),
        'stop_timeout_seconds': (
            profile.stop_timeout_seconds
            if profile.stop_timeout_seconds is not None
            else (900 if sandbox_profile else 300)
        ),
    }


def prepare_source_project(source_root: Path, *, project_root: Path) -> None:
    source_root.mkdir(parents=True, exist_ok=False)
    for name in ('pyproject.toml', 'uv.lock', 'README.md'):
        shutil.copyfile(project_root / name, source_root / name)
    (source_root / SOURCE_FIXTURE_MODULE).write_text(_source_fixture_text(), encoding='utf-8')
    for profile in TKEX_LOAD_PROFILES.values():
        if profile.code_mode != 'source_package':
            continue
        generator = source_root / source_generator_path(profile)
        generator.parent.mkdir(parents=True, exist_ok=True)
        generator.write_text(_source_generator_text(profile), encoding='utf-8')


def _source_fixture_text() -> str:
    return """from __future__ import annotations

import asyncio
import importlib.metadata
import sys

import nexus

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    TaskRunInput,
    TaskRunOutput,
    UserInputRequest,
    WithInputProvider,
    agent_registry,
    task_registry,
)
from nexus.runtime.client.base import Runtime
from nexus.runtime.providers.base import RuntimeProvider
from nexus.runtime.providers.registry import runtime_provider_registry


class CloudSourceEchoInput(TaskRunInput):
    index: int
    sleep_seconds: float = 0


class CloudSourceEchoOutput(TaskRunOutput):
    index: int
    marker: str
    dependency_version: str
    sys_prefix: str
    nexus_file: str


@task_registry.register("cloud_source_package_echo")
async def cloud_source_package_echo(task_input: CloudSourceEchoInput) -> CloudSourceEchoOutput:
    await asyncio.sleep(task_input.sleep_seconds)
    return CloudSourceEchoOutput(
        index=task_input.index,
        marker=f"source-package-ok:{task_input.index}",
        dependency_version=importlib.metadata.version("mycli"),
        sys_prefix=sys.prefix,
        nexus_file=str(nexus.__file__),
    )


class CloudInteractiveAgentInput(AgentRunInput, WithInputProvider):
    prompt: str = "choose"


class CloudInteractiveAgentOutput(AgentRunOutput):
    choice: str | None


@agent_registry.register("cloud_interactive_agent")
async def cloud_interactive_agent(agent_input: CloudInteractiveAgentInput) -> CloudInteractiveAgentOutput:
    choice = await agent_input.get_user_input(UserInputRequest(hint=agent_input.prompt))
    return CloudInteractiveAgentOutput(choice=choice)


@runtime_provider_registry.register("cloud_fake_runtime")
class CloudFakeRuntimeProvider(RuntimeProvider):
    async def acquire(self, name: str, **params):
        import os

        base_url = params.get("base_url") or os.environ["NEXUS_DIST_FAKE_RUNTIME_BASE_URL"]
        runtime = CloudFakeRuntime(base_url=base_url)
        self._active_runtimes[name] = runtime
        return runtime

    async def _release_impl(self, name: str) -> None:
        self._active_runtimes.pop(name, None)


class CloudFakeRuntime(Runtime):
    def __init__(self, *, base_url: str) -> None:
        super().__init__()
        self._base_url = base_url

    @property
    def base_url(self) -> str | None:
        return self._base_url

    async def _invoke(self, request):
        raise NotImplementedError("cloud fake runtime only supports proxy tests")

    async def ws_connect(self) -> None:
        return None

    def on(self, event, listener):
        return lambda: None
"""


def _source_generator_text(profile: BatchLoadProfile) -> str:
    return f'''from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput


def get_total_count():
    return {profile.item_count}


def generate_configs():
    for index in range({profile.item_count}):
        stable_id = f"{{index:012d}}"
        yield BatchItem(
            config=RunInput(
                imports=["{SOURCE_FIXTURE_MODULE.removesuffix('.py')}"],
                task_name="cloud_source_package_echo",
                task_config={{"index": index, "sleep_seconds": {profile.task_sleep_seconds}}},
            ),
            run_id=f"{profile.name}-{{stable_id}}",
        )
'''


TKEX_LOAD_PROFILES = {
    profile.name: profile
    for profile in (
        BatchLoadProfile(
            name='wheelhouse-smoke',
            code_mode='source_package',
            item_count=20,
            execution_manager_count=2,
            slots_per_manager=2,
            max_inflight=4,
            minimum_peak=4,
        ),
        BatchLoadProfile(
            name='sandbox-agent-smoke',
            code_mode='config_only',
            item_count=20,
            execution_manager_count=5,
            slots_per_manager=4,
            max_inflight=20,
            minimum_peak=18,
            task_kind='sandbox_agent_echo',
            minimum_sandbox_peak=20,
            task_sleep_seconds=360,
            timeout_per_run_seconds=1200,
        ),
        BatchLoadProfile(
            name='wheelhouse-fanout',
            code_mode='source_package',
            item_count=499,
            execution_manager_count=499,
            slots_per_manager=1,
            max_inflight=499,
            minimum_peak=450,
        ),
        BatchLoadProfile(
            name='config-baseline',
            code_mode='config_only',
            item_count=2000,
            execution_manager_count=25,
            slots_per_manager=20,
            max_inflight=500,
            minimum_peak=450,
        ),
        BatchLoadProfile(
            name='source-heavy',
            code_mode='source_package',
            item_count=2000,
            execution_manager_count=25,
            slots_per_manager=20,
            max_inflight=500,
            minimum_peak=450,
        ),
        BatchLoadProfile(
            name='container-scale',
            code_mode='config_only',
            item_count=4000,
            execution_manager_count=199,
            slots_per_manager=10,
            max_inflight=1990,
            minimum_peak=1800,
            task_sleep_seconds=22,
        ),
        BatchLoadProfile(
            name='container-ceiling',
            code_mode='config_only',
            item_count=4000,
            execution_manager_count=499,
            slots_per_manager=4,
            max_inflight=1996,
            minimum_peak=1800,
            task_sleep_seconds=22,
        ),
        BatchLoadProfile(
            name='manager-2000',
            code_mode='config_only',
            item_count=16_000,
            execution_manager_count=1999,
            slots_per_manager=4,
            max_inflight=7996,
            minimum_peak=7200,
            task_sleep_seconds=22,
            timeout_per_run_seconds=2400,
            batch_manager_cpu_cores=4,
            batch_manager_memory_gib=4,
            opt_in=True,
        ),
        BatchLoadProfile(
            name='sandbox-agent-10000',
            code_mode='config_only',
            item_count=16_000,
            execution_manager_count=1999,
            slots_per_manager=6,
            max_inflight=10_100,
            minimum_peak=10_000,
            task_kind='sandbox_agent_echo',
            minimum_sandbox_peak=10_000,
            task_sleep_seconds=1200,
            timeout_per_run_seconds=3600,
            batch_manager_cpu_cores=4,
            batch_manager_memory_gib=4,
            retry_failed=2,
            opt_in=True,
        ),
        BatchLoadProfile(
            name='sandbox-agent-60000',
            code_mode='config_only',
            item_count=100_000,
            execution_manager_count=3000,
            slots_per_manager=20,
            max_inflight=60_000,
            minimum_peak=60_000,
            task_kind='sandbox_agent_echo',
            minimum_sandbox_peak=60_000,
            task_sleep_seconds=3600,
            timeout_per_run_seconds=7200,
            batch_manager_cpu_cores=4,
            batch_manager_memory_gib=4,
            retry_failed=2,
            stop_timeout_seconds=3600,
            opt_in=True,
        ),
        BatchLoadProfile(
            name='extreme',
            code_mode='config_only',
            item_count=20_000,
            execution_manager_count=499,
            slots_per_manager=20,
            max_inflight=9000,
            minimum_peak=8500,
            opt_in=True,
        ),
    )
}


__all__ = [
    'DEFAULT_WHEELHOUSE_EXTRAS',
    'GIB',
    'MAX_STRESS_CONTAINERS',
    'STRESS_STOP_POLICY',
    'STRESS_ENVIRONMENT_LIMITS',
    'SOURCE_FIXTURE_MODULE',
    'TKEX_LOAD_PROFILES',
    'BatchLoadProfile',
    'StressObservation',
    'StressStopPolicy',
    'batch_submission',
    'environment_demand',
    'fit_profile_to_container_budget',
    'prepare_source_project',
    'source_generator_path',
    'stress_stop_reasons',
    'validate_profile_for_cluster',
    'write_config_task_list',
]
