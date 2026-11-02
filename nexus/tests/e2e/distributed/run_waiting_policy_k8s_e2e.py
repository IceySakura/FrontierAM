"""Exercise waiting_policy phase projection and Batch/Single policy observability on K8s (kind) via Helm chart.

Deploys the nexus-distributed Helm chart (deployment/distributed/helm/nexus-distributed) with the
Kubernetes dynamic environment provider enabled, and verifies end-to-end:
1. Single execution phase lifecycle under scheduler policy gate:
   - waiting_resources -> queued -> waiting_policy (under quota_exhausted, hard_brake, signal_incomplete) -> running -> succeeded;
   - GET /v1/executions/{id} and GET /v1/executions both project phase="waiting_policy";
   - GET /v1/executions/{id}/scheduler-policy and GET /v1/executions/{id}/request-policy;
   - GET /v1/executions/{id}/control-events records scheduler.policy_deferred and scheduler.policy_applied;
   - Queue TTL expiration while waiting on policy: waiting_policy -> expired (status="expired", phase="expired",
     attempt_count=0, with scheduler.policy_deferred + execution.status.changed(to="expired") and no policy_applied);
   - Unbound Single bypasses waiting_policy and succeeds while gated Single remains in waiting_policy.
2. Batch execution policy observability, tail release under slot contention, and runtime below_minimum pause:
   - Batch parent stays in phase="running_children" while children are in phase="waiting_policy";
   - GET /v1/executions/{parent_id}/batch?include_status_counts=true reports child_waiting_policy_count=2;
   - GET /v1/executions/{parent_id}/scheduler-policy resolves child template policy with next_schedule_at=None;
   - GET /v1/executions/{parent_id}/request-policy resolves child template request policy;
   - When scheduler policy allows under 1-slot contention while request_rate is paused at below_minimum (factor=0.001),
     the 1st child enters phase="running" (paused on request budget with reason_code="below_minimum" without changing phase)
     and the unassigned 2nd child immediately clears policy_deferred and transitions from phase="waiting_policy"
     to phase="queued", with child_waiting_policy_count=0;
   - When factor=2.5 resumes request budget, /request-policy returns reason_code="resumed", effective_rate_per_minute=240.0,
     and both children complete with succeeded.
3. Complete dynamic K8s Allocation & StatefulSet cleanup verification.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import uuid

from contextlib import ExitStack
from pathlib import Path
from typing import Any

import aiohttp

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from tests.integration.distributed.cloud_batch_api import CloudBatchApi
from tests.integration.distributed.cluster_config import (
    create_database_bootstrap_secrets,
    local_database_url,
    platform_config_values,
    platform_secret_values,
    seed_platform_config,
)

from nexus.distributed.environment.naming import allocation_resource_names
from nexus.distributed.persistence.orm import (
    DistributedConfigOverrideRow,
    ExecutionRow,
    PolicyScopeRow,
)
from nexus.distributed.policies.collections.infer_proxy_request_rate import (
    REQUEST_CAPACITY_SOURCE,
    REQUEST_RATE_POLICY,
)
from nexus.distributed.repositories.policy import policy_scope_key


PROJECT_ROOT = Path(__file__).parents[3]
CHART_DIR = PROJECT_ROOT / 'deployment/distributed/helm/nexus-distributed'
DYNAMIC_VALUES = CHART_DIR / 'examples/values-local-dynamic.yaml'
SIGNAL_SERVER_SCRIPT = Path(__file__).with_name('request_budget_signal_server.py')

MODEL = 'hy3-offline'
QUOTA_QPM = 240.0
ADMIN_SIGNAL_KEY = 'budget-e2e-admin-key'
PLATFORM_API_KEY = 'budget-e2e-quota-group-key'
CLIENT_API_KEY = 'budget-e2e-client-key'

PROBE_CODE = """import asyncio
from nexus.core import TaskRunInput, TaskRunOutput, require_context, task_registry


class WaitingPolicyProbeInput(TaskRunInput):
    index: int = 0
    requests: int = 0
    sleep_seconds: float = 0.5


class WaitingPolicyProbeOutput(TaskRunOutput):
    marker: str
    requests: int


@task_registry.register('waiting_policy_probe')
async def waiting_policy_probe(value: WaitingPolicyProbeInput) -> WaitingPolicyProbeOutput:
    if value.sleep_seconds > 0:
        await asyncio.sleep(value.sleep_seconds)
    if value.requests > 0:
        client = require_context().get_openai_client('main')
        for step in range(value.requests):
            await client.create_completion([{'role': 'user', 'content': f'{value.index}:{step}'}])
    return WaitingPolicyProbeOutput(marker=f'waiting-policy-e2e:{value.index}', requests=value.requests)
"""


def run_cmd(
    command: list[str],
    *,
    input_text: str | None = None,
    capture: bool = True,
    check: bool = True,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        input=input_text,
        capture_output=capture,
        text=True,
        check=False,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or '')[-2000:]
        raise RuntimeError(f'command failed ({completed.returncode}): {" ".join(command)}\n{detail}')
    return completed


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def wait_tcp(host: str, port: int, *, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f'TCP {host}:{port} not ready')


async def wait_http(url: str, *, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            with contextlib.suppress(Exception):
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        return
            await asyncio.sleep(0.5)
    raise TimeoutError(f'HTTP {url} not ready')


class PortForward:
    def __init__(
        self,
        *,
        kubectl: str,
        namespace: str,
        resource: str,
        remote_port: int,
        local_port: int | None = None,
    ) -> None:
        self.kubectl = kubectl
        self.namespace = namespace
        self.resource = resource
        self.remote_port = remote_port
        self.local_port = local_port or free_port()
        self._stop = False
        self._process: subprocess.Popen[str] | None = None
        self._thread: Any = None

    def _loop(self) -> None:
        while not self._stop:
            self._process = subprocess.Popen(
                [
                    self.kubectl,
                    '-n',
                    self.namespace,
                    'port-forward',
                    self.resource,
                    f'{self.local_port}:{self.remote_port}',
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self._process.wait()
            if not self._stop:
                time.sleep(0.3)

    def __enter__(self) -> PortForward:
        import threading

        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        wait_tcp('127.0.0.1', self.local_port, timeout=30.0)
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop = True
        if self._process is not None:
            self._process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._process.wait(timeout=5)
            if self._process.poll() is None:
                self._process.kill()
        if self._thread is not None:
            self._thread.join(timeout=2)


def deploy_signal_server(kubectl: str, namespace: str, *, image_tag: str) -> str:
    manifest = {
        'apiVersion': 'v1',
        'kind': 'List',
        'items': [
            {
                'apiVersion': 'v1',
                'kind': 'ConfigMap',
                'metadata': {'name': 'budget-signal-server', 'namespace': namespace},
                'data': {'server.py': SIGNAL_SERVER_SCRIPT.read_text(encoding='utf-8')},
            },
            {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': {'name': 'budget-signal-server', 'namespace': namespace},
                'spec': {
                    'replicas': 1,
                    'selector': {'matchLabels': {'app': 'budget-signal-server'}},
                    'template': {
                        'metadata': {'labels': {'app': 'budget-signal-server'}},
                        'spec': {
                            'containers': [
                                {
                                    'name': 'server',
                                    'image': f'nexus-distributed-universal:{image_tag}',
                                    'imagePullPolicy': 'IfNotPresent',
                                    'command': ['/opt/nexus-venv/bin/python', '/app-fixture/server.py'],
                                    'env': [
                                        {'name': 'POLICY_TEST_KEY', 'value': ADMIN_SIGNAL_KEY},
                                        {'name': 'TIDE_PLATFORM_API_KEY', 'value': PLATFORM_API_KEY},
                                        {'name': 'TIDE_CLIENT_API_KEY', 'value': CLIENT_API_KEY},
                                    ],
                                    'ports': [{'containerPort': 8080, 'name': 'http'}],
                                    'readinessProbe': {
                                        'httpGet': {'path': '/health', 'port': 8080},
                                        'initialDelaySeconds': 1,
                                        'periodSeconds': 2,
                                    },
                                    'volumeMounts': [{'name': 'code', 'mountPath': '/app-fixture', 'readOnly': True}],
                                }
                            ],
                            'volumes': [{'name': 'code', 'configMap': {'name': 'budget-signal-server'}}],
                        },
                    },
                },
            },
            {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {'name': 'budget-signal-server', 'namespace': namespace},
                'spec': {
                    'selector': {'app': 'budget-signal-server'},
                    'ports': [{'name': 'http', 'port': 8080, 'targetPort': 8080}],
                },
            },
        ],
    }
    run_cmd([kubectl, 'apply', '-f', '-'], input_text=json.dumps(manifest))
    run_cmd([kubectl, '-n', namespace, 'rollout', 'status', 'deployment/budget-signal-server', '--timeout=120s'])
    return f'http://budget-signal-server.{namespace}.svc.cluster.local:8080'


def policy_executor_config(
    *,
    environment_id: str,
    scheduler_binding: str,
    scheduler_scope: str,
    request_binding: str,
    request_scope: str,
    capacity_url: str,
    revision: int = 1,
) -> dict[str, Any]:
    return {
        'imports': [
            'nexus.distributed.policies.collections.infer_proxy',
            'nexus.distributed.policies.collections.infer_proxy_request_rate',
        ],
        'workers': 2,
        'poll_seconds': 0.5,
        'idle_seconds': 3600,
        'worker_startup_seconds': 30,
        'bindings': {
            scheduler_binding: {
                'kind': 'scheduler',
                'policy': 'infer_proxy_tide',
                'source': 'infer_proxy_capacity',
                'revision': revision,
                'mode': 'enforce',
                'on_unavailable': 'defer',
                'environments': [environment_id],
                'principals': [],
                'refresh_seconds': 2,
                'timeout_seconds': 4,
                'max_age_seconds': 30,
                'scopes': {
                    scheduler_scope: {
                        'source_config': {
                            'model': MODEL,
                            'url': capacity_url,
                            'api_key': PLATFORM_API_KEY,
                        }
                    }
                },
                'params': {},
            },
            request_binding: {
                'kind': 'request_rate',
                'policy': REQUEST_RATE_POLICY,
                'source': REQUEST_CAPACITY_SOURCE,
                'revision': revision,
                'mode': 'enforce',
                'on_unavailable': 'defer',
                'environments': [environment_id],
                'principals': [],
                'refresh_seconds': 2,
                'timeout_seconds': 4,
                'max_age_seconds': 30,
                'scopes': {
                    request_scope: {
                        'source_config': {
                            'model': MODEL,
                            'url': capacity_url,
                            'timeout_seconds': 3,
                            'api_key': PLATFORM_API_KEY,
                        }
                    }
                },
                'params': {
                    'quota_qpm': QUOTA_QPM,
                    'min_quota_ratio': 0.01,
                },
            },
        },
        'request_defaults': {},
    }


def bootstrap_cluster(args: argparse.Namespace, *, app_local_port: int) -> str:
    kubectl = args.kubectl
    helm = args.helm
    namespace = args.namespace
    release = args.release_name
    db_secret = f'{release}-database'
    workload_db_secret = f'{release}-workload-database'
    mysql_secret = f'{release}-mysql-auth'
    mysql_password = 'nexus_dist_local'

    run_cmd([helm, 'uninstall', release, '--namespace', namespace, '--wait'], check=False)
    run_cmd([kubectl, '-n', namespace, 'delete', 'statefulset,job', '--all', '--ignore-not-found=true'], check=False)
    run_cmd([kubectl, 'create', 'namespace', namespace], check=False)
    create_database_bootstrap_secrets(
        run_cmd,
        kubectl=kubectl,
        namespace=namespace,
        release_name=release,
        database_secret_name=db_secret,
        mysql_secret_name=mysql_secret,
        password=mysql_password,
    )
    signal_cluster_url = deploy_signal_server(kubectl, namespace, image_tag=args.image_tag)

    helm_base = [
        helm,
        'upgrade',
        '--install',
        release,
        str(CHART_DIR),
        '--namespace',
        namespace,
        '-f',
        str(DYNAMIC_VALUES),
        '--set-string',
        f'fullnameOverride={release}',
        '--set-string',
        f'controlImage.tag={args.image_tag}',
        '--set-string',
        f'workloadImage.tag={args.image_tag}',
        '--set-string',
        f'builderImage.tag={args.image_tag}',
        '--set-string',
        f'secrets.existingSecret={db_secret}',
        '--set-string',
        f'mysql.auth.existingSecret={mysql_secret}',
        '--set',
        'secrets.create=false',
        '--set-string',
        f'dynamicEnvironment.namespace={namespace}',
        '--set-string',
        'dynamicEnvironment.enabledProviders[0]=kubernetes',
        '--timeout',
        '240s',
    ]
    run_cmd(
        [
            *helm_base,
            '--set',
            'controlGateway.replicas=0',
            '--set',
            'applicationGateway.replicas=0',
            '--set',
            'scheduler.replicas=0',
            '--set',
            'reconciler.replicas=0',
            '--set',
            'fakeRuntime.replicas=0',
            '--wait',
        ]
    )
    run_cmd([kubectl, '-n', namespace, 'rollout', 'status', f'deployment/{release}-mysql', '--timeout=180s'])
    run_cmd([kubectl, '-n', namespace, 'wait', '--for=condition=complete', f'job/{release}-db-init', '--timeout=180s'])

    with PortForward(kubectl=kubectl, namespace=namespace, resource=f'service/{release}-mysql', remote_port=3306) as pf:
        db_url = local_database_url(port=pf.local_port, password=mysql_password)
        config_values = platform_config_values(
            application_public_base_url=f'http://127.0.0.1:{app_local_port}',
            application_internal_base_url=f'http://{release}-application-gateway.{namespace}.svc.cluster.local:8082',
            workload_image=f'nexus-distributed-workload:{args.image_tag}',
            builder_image=f'nexus-distributed-builder:{args.image_tag}',
            namespace=namespace,
            release_name=release,
            database_secret_name=workload_db_secret,
            enabled_providers=['kubernetes'],
            fake_runtime_base_url=f'http://{release}-fake-runtime:8080',
        )
        config_values['reconciler_poll_interval'] = 0.5
        config_values['scheduler_poll_interval'] = 0.5
        config_values['reconciler_lease_ttl'] = 6.0
        config_values['reconciler_lease_renew_interval'] = 2.0
        config_values['reconciler_shard_discover_interval'] = 2.0
        config_values['scheduler_lease_ttl'] = 6.0
        config_values['scheduler_lease_renew_interval'] = 2.0
        config_values['scheduler_shard_discover_interval'] = 2.0
        config_values['request_budget'] = {
            'lease_seconds': 4.0,
            'renew_before_seconds': 1.5,
            'safety_margin_seconds': 0.1,
        }
        config_values['policy_executor'] = policy_executor_config(
            environment_id=args.environment,
            scheduler_binding=args.scheduler_binding,
            scheduler_scope=args.scheduler_scope,
            request_binding=args.request_binding,
            request_scope=args.request_scope,
            capacity_url=f'{signal_cluster_url}/capacity',
            revision=1,
        )
        seed_platform_config(
            run_cmd,
            python=sys.executable,
            database_url=db_url,
            config_values=config_values,
            secret_values=platform_secret_values(os.environ),
        )

    run_cmd([*helm_base, '--set', 'dbInit.enabled=false', '--wait'])
    run_cmd([kubectl, '-n', namespace, 'rollout', 'status', f'deployment/{release}-scheduler', '--timeout=120s'])
    return signal_cluster_url


async def ensure_dynamic_environment(control_url: str, *, namespace: str, environment_id: str) -> None:
    provider_ref = 'kubernetes/local-kind'
    provider_body = {
        'config': {
            'provider_type': 'kubernetes',
            'namespace': namespace,
            'auth_type': 'in_cluster',
            'tls_strict': False,
            'manager_resource_request_ratio': 0.1,
        },
        'status': 'enabled',
    }
    encoded_ref = urllib.parse.quote(provider_ref, safe='')
    async with aiohttp.ClientSession() as session:
        async with session.get(f'{control_url}/v1/provider-configs') as resp:
            resp.raise_for_status()
            configs = await resp.json()
        if any(item['provider_config_ref'] == provider_ref for item in configs.get('items', [])):
            async with session.put(f'{control_url}/v1/provider-configs/{encoded_ref}', json=provider_body) as resp:
                resp.raise_for_status()
        else:
            async with session.post(
                f'{control_url}/v1/provider-configs',
                json={'provider_config_ref': provider_ref, **provider_body},
            ) as resp:
                resp.raise_for_status()

        limits = {
            'manager_count': 20,
            'cpu_millicores': 20_000,
            'memory_bytes': 32 * 1024**3,
            'slots': 100,
            'active_batches': 6,
            'batch_max_inflight': 16,
            'max_provisioning_timeout_seconds': 900,
            'max_queue_ttl_seconds': 1800,
            'max_execution_ttl_seconds': 3600,
            'max_stop_timeout_seconds': 120,
            'max_generation_timeout_seconds': 600,
            'max_child_queue_ttl_seconds': 1800,
            'max_timeout_per_run': 1800,
        }
        env_body = {
            'name': 'Kind Dynamic Environment',
            'allocation_type': 'dynamic',
            'provider_type': 'kubernetes',
            'provider_config_ref': provider_ref,
            'resource_limits': limits,
            'default_timeouts': {
                'provisioning_timeout_seconds': 600,
                'queue_ttl_seconds': 1800,
                'execution_ttl_seconds': 3600,
                'stop_timeout_seconds': 60,
                'generation_timeout_seconds': 600,
                'child_queue_ttl_seconds': 1800,
                'timeout_per_run': 600,
            },
            'default_quota': {
                'manager_count': 20,
                'cpu_millicores': 20_000,
                'memory_bytes': 32 * 1024**3,
                'slots': 100,
                'active_batches': 6,
            },
            'status': 'enabled',
        }
        async with session.get(f'{control_url}/v1/environments') as resp:
            resp.raise_for_status()
            envs = await resp.json()
        if any(item['environment_id'] == environment_id for item in envs.get('items', [])):
            async with session.put(f'{control_url}/v1/environments/{environment_id}', json=env_body) as resp:
                resp.raise_for_status()
        else:
            async with session.post(
                f'{control_url}/v1/environments',
                json={'environment_id': environment_id, **env_body},
            ) as resp:
                resp.raise_for_status()
        async with session.put(f'{control_url}/v1/my-environments/{environment_id}/quota') as resp:
            resp.raise_for_status()


def prepare_source(
    output: Path,
    *,
    project_root: Path,
    llm_base_url: str,
) -> Path:
    source = output / 'source'
    if source.exists():
        shutil.rmtree(source)
    source.mkdir(parents=True, exist_ok=True)
    for name in ('pyproject.toml', 'uv.lock', 'README.md'):
        shutil.copyfile(project_root / name, source / name)
    (source / 'waiting_policy_probe.py').write_text(PROBE_CODE, encoding='utf-8')
    client_block = {
        'main': {
            'name': 'infer_proxy',
            'config': {
                'client_args': {
                    'api_key': CLIENT_API_KEY,
                    'base_url': llm_base_url,
                },
                'request_args': {
                    'model': MODEL,
                },
            },
        }
    }
    (source / 'single_probe.json').write_text(
        json.dumps(
            {
                'imports': ['waiting_policy_probe'],
                'task_name': 'waiting_policy_probe',
                'task_config': {'index': 100, 'requests': 2, 'sleep_seconds': 0.5},
                'openai_clients': client_block,
            }
        ),
        encoding='utf-8',
    )
    generator_code = f"""from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput

def get_total_count():
    return 2

def generate_configs():
    for i in range(2):
        yield BatchItem(
            run_id=f'item-{{i + 1:012d}}',
            config=RunInput.from_dict({{
                'imports': ['waiting_policy_probe'],
                'task_name': 'waiting_policy_probe',
                'task_config': {{'index': i, 'requests': 2, 'sleep_seconds': 0.5}},
                'openai_clients': {{
                    'main': {{
                        'name': 'infer_proxy',
                        'config': {{
                            'client_args': {{
                                'base_url': {llm_base_url!r},
                                'api_key': {CLIENT_API_KEY!r},
                            }},
                            'request_args': {{'model': {MODEL!r}}},
                        }},
                    }},
                }},
            }}),
        )
"""
    (source / 'generator.py').write_text(generator_code, encoding='utf-8')
    return source


def submit_single(
    *,
    source: Path,
    project_root: Path,
    environment_id: str,
    control_url: str,
    scheduler_policy: str | None = None,
    policy_scope: str | None = None,
    request_policy: str | None = None,
    request_scope: str | None = None,
    queue_ttl_seconds: int = 1800,
) -> str:
    command = [
        sys.executable,
        '-m',
        'nexus.cli.main',
        'submit',
        'single',
        str(source / 'single_probe.json'),
        '--environment',
        environment_id,
        '--control-url',
        control_url,
        '--source',
        str(source),
        '--project-root',
        str(project_root),
        '--manager-cpu-cores',
        '1',
        '--manager-memory-gib',
        '2',
        '--manager-slots',
        '1',
        '--queue-ttl-seconds',
        str(queue_ttl_seconds),
        '--execution-ttl-seconds',
        '3600',
        '--idempotency-key',
        f'waiting-policy-single-{uuid.uuid4().hex}',
        '--json',
    ]
    if scheduler_policy and policy_scope:
        command += ['--scheduler-policy', scheduler_policy, '--policy-scope', policy_scope]
    if request_policy and request_scope:
        command += ['--request-policy', request_policy, '--request-scope', request_scope]
    completed = run_cmd(command)
    payload = json.loads(completed.stdout)
    return str(payload.get('execution_id') or payload['execution']['execution_id'])


def submit_batch(
    *,
    source: Path,
    project_root: Path,
    environment_id: str,
    control_url: str,
    scheduler_policy: str,
    policy_scope: str,
    request_policy: str,
    request_scope: str,
) -> str:
    command = [
        sys.executable,
        '-m',
        'nexus.cli.main',
        'submit',
        'batch',
        '--generator',
        'generator.py',
        '--generation',
        'remote',
        '--environment',
        environment_id,
        '--control-url',
        control_url,
        '--source',
        str(source),
        '--project-root',
        str(project_root),
        '--manager-count',
        '1',
        '--manager-cpu-cores',
        '1',
        '--manager-memory-gib',
        '2',
        '--manager-slots',
        '2',
        '--batch-manager-cpu-cores',
        '1',
        '--batch-manager-memory-gib',
        '2',
        '--max-inflight',
        '2',
        '--timeout-per-run',
        '600',
        '--queue-ttl-seconds',
        '1800',
        '--execution-ttl-seconds',
        '3600',
        '--child-queue-ttl-seconds',
        '1800',
        '--scheduler-policy',
        scheduler_policy,
        '--policy-scope',
        policy_scope,
        '--request-policy',
        request_policy,
        '--request-scope',
        request_scope,
        '--idempotency-key',
        f'waiting-policy-batch-{uuid.uuid4().hex}',
        '--json',
    ]
    completed = run_cmd(command)
    payload = json.loads(completed.stdout)
    return str(payload.get('execution_id') or payload['execution']['execution_id'])


async def set_signal_state(signal_url: str, **updates: Any) -> dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f'{signal_url}/state',
            headers={'X-Policy-Test-Key': ADMIN_SIGNAL_KEY},
            json=updates,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()


async def wait_for_scope_snapshot(
    db_url: str,
    *,
    scope_key: str,
    predicate: Any,
    timeout: float = 45.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    engine = create_async_engine(db_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        while time.monotonic() < deadline:
            async with sessions() as session:
                from nexus.distributed.utils.clock import utcnow

                row = await session.get(PolicyScopeRow, scope_key)
                if row is not None:
                    row.last_demand_at = utcnow()
                    await session.commit()
                    if isinstance(row.snapshot_json, dict) and predicate(row.snapshot_json, row):
                        return row.snapshot_json
            await asyncio.sleep(0.4)
    finally:
        await engine.dispose()
    raise TimeoutError(f'Timed out waiting for scope snapshot {scope_key}')


async def run_e2e(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    app_local_port = free_port()
    kubectl = args.kubectl
    namespace = args.namespace
    release = args.release_name
    if args.skip_bootstrap:
        signal_cluster_url = f'http://budget-signal-server.{namespace}.svc.cluster.local:8080'
    else:
        signal_cluster_url = bootstrap_cluster(args, app_local_port=app_local_port)
    scheduler_scope_key = policy_scope_key('scheduler', args.scheduler_binding, args.scheduler_scope)
    request_scope_key = policy_scope_key('request_rate', args.request_binding, args.request_scope)

    with ExitStack() as stack:
        pf_mysql = stack.enter_context(
            PortForward(kubectl=kubectl, namespace=namespace, resource=f'service/{release}-mysql', remote_port=3306)
        )
        db_url = local_database_url(port=pf_mysql.local_port, password='nexus_dist_local')
        if args.skip_bootstrap:
            engine = create_async_engine(db_url)
            sessions = async_sessionmaker(engine, expire_on_commit=False)
            try:
                async with sessions() as session:
                    cfg_row = await session.scalar(
                        select(DistributedConfigOverrideRow).where(
                            DistributedConfigOverrideRow.role == '*',
                            DistributedConfigOverrideRow.identity == '*',
                        )
                    )
                    if cfg_row and isinstance(cfg_row.config_json, dict):
                        pub_url = cfg_row.config_json.get('application_public_base_url', '')
                        parsed_port = urllib.parse.urlsplit(str(pub_url)).port
                        if parsed_port:
                            app_local_port = int(parsed_port)
            finally:
                await engine.dispose()

        pf_control = stack.enter_context(
            PortForward(
                kubectl=kubectl, namespace=namespace, resource=f'service/{release}-control-gateway', remote_port=8081
            )
        )
        stack.enter_context(
            PortForward(
                kubectl=kubectl,
                namespace=namespace,
                resource=f'service/{release}-application-gateway',
                remote_port=8082,
                local_port=app_local_port,
            )
        )
        pf_signal = stack.enter_context(
            PortForward(kubectl=kubectl, namespace=namespace, resource='service/budget-signal-server', remote_port=8080)
        )

        control_url = f'http://127.0.0.1:{pf_control.local_port}'
        signal_url = f'http://127.0.0.1:{pf_signal.local_port}'

        await wait_http(f'{control_url}/v1/health')
        await wait_http(f'{signal_url}/health')
        await ensure_dynamic_environment(control_url, namespace=namespace, environment_id=args.environment)

        source = prepare_source(
            args.output,
            project_root=PROJECT_ROOT,
            llm_base_url=f'{signal_cluster_url}/v1',
        )

        async with CloudBatchApi(
            control_url=control_url,
            application_url=f'http://127.0.0.1:{app_local_port}',
            token='local-e2e',
        ) as api:
            # Clean up any leftover active executions from prior interrupted runs
            existing_list = await api.control_json('GET', '/v1/executions')
            for item in existing_list.get('items', []):
                if item.get('status') in {'queued', 'running'}:
                    with contextlib.suppress(Exception):
                        await api.control_json('POST', f'/v1/executions/{item["execution_id"]}/stop')

            # ==================================================================
            # Scenario 1: Single Execution Phase Lifecycle, Policy Reason
            #             Transitions, Queue TTL Expiration, and Unbound Bypass
            # ==================================================================
            await set_signal_state(
                signal_url,
                remaining=0,
                hard_brake=False,
                factor=1.0,
                mode='ok',
                delay=0.0,
                reset_counts=True,
            )

            gated_single_id = submit_single(
                source=source,
                project_root=PROJECT_ROOT,
                environment_id=args.environment,
                control_url=control_url,
                scheduler_policy=args.scheduler_binding,
                policy_scope=args.scheduler_scope,
                request_policy=args.request_binding,
                request_scope=args.request_scope,
                queue_ttl_seconds=1800,
            )
            expired_single_id = submit_single(
                source=source,
                project_root=PROJECT_ROOT,
                environment_id=args.environment,
                control_url=control_url,
                scheduler_policy=args.scheduler_binding,
                policy_scope=args.scheduler_scope,
                request_policy=args.request_binding,
                request_scope=args.request_scope,
                queue_ttl_seconds=15,
            )

            single_observed_phases: list[str] = []
            expired_observed_phases: list[str] = []
            deadline = time.monotonic() + 300.0
            single_waiting_policy_view: dict[str, Any] = {}
            expired_waiting_policy_view: dict[str, Any] = {}
            while time.monotonic() < deadline:
                exe = await api.control_json('GET', f'/v1/executions/{gated_single_id}')
                exp_exe = await api.control_json('GET', f'/v1/executions/{expired_single_id}')
                phase = str(exe.get('phase'))
                exp_phase = str(exp_exe.get('phase'))
                if not single_observed_phases or single_observed_phases[-1] != phase:
                    single_observed_phases.append(phase)
                    print(f'[Single {gated_single_id}] status={exe["status"]} phase={phase}', flush=True)
                if not expired_observed_phases or expired_observed_phases[-1] != exp_phase:
                    expired_observed_phases.append(exp_phase)
                    print(
                        f'[ExpiredSingle {expired_single_id}] status={exp_exe["status"]} phase={exp_phase}', flush=True
                    )
                if phase == 'waiting_policy' and not single_waiting_policy_view:
                    single_waiting_policy_view = exe
                if exp_phase == 'waiting_policy' and not expired_waiting_policy_view:
                    expired_waiting_policy_view = exp_exe
                if single_waiting_policy_view and expired_waiting_policy_view:
                    break
                if exe['status'] in {'failed', 'cancelled', 'expired', 'lost'}:
                    raise RuntimeError(f'Single {gated_single_id} entered terminal status early: {exe}')
                await asyncio.sleep(0.4)
            else:
                raise TimeoutError(
                    f'Singles did not reach waiting_policy; gated={single_observed_phases}, expired={expired_observed_phases}'
                )

            assert single_waiting_policy_view['status'] == 'queued'
            assert single_waiting_policy_view['phase'] == 'waiting_policy'
            assert single_waiting_policy_view['attempt_count'] == 0
            assert single_waiting_policy_view['current_assignment_id'] is None

            # Verify list endpoint also projects waiting_policy
            list_resp = await api.control_json('GET', '/v1/executions')
            listed_single = next(
                (item for item in list_resp.get('items', []) if item['execution_id'] == gated_single_id),
                None,
            )
            assert listed_single is not None
            assert listed_single['phase'] == 'waiting_policy'

            # Verify /scheduler-policy and /request-policy on gated Single while quota_exhausted
            await wait_for_scope_snapshot(
                db_url,
                scope_key=scheduler_scope_key,
                predicate=lambda s, _: s.get('reason_code') == 'quota_exhausted',
            )
            await wait_for_scope_snapshot(
                db_url,
                scope_key=request_scope_key,
                predicate=lambda s, _: s.get('reason_code') == 'quota_exhausted',
            )
            single_sched_policy_quota = await api.control_json(
                'GET', f'/v1/executions/{gated_single_id}/scheduler-policy'
            )
            assert single_sched_policy_quota['enabled'] is True
            assert single_sched_policy_quota['binding'] == args.scheduler_binding
            assert [s['scope_id'] for s in single_sched_policy_quota['scopes']] == [args.scheduler_scope]
            assert single_sched_policy_quota['mode'] == 'enforce'
            assert single_sched_policy_quota['allow'] is False
            assert single_sched_policy_quota['reason_code'] == 'quota_exhausted'
            assert single_sched_policy_quota['next_schedule_at'] is not None

            single_req_policy_quota = await api.control_json('GET', f'/v1/executions/{gated_single_id}/request-policy')
            assert single_req_policy_quota['enabled'] is True
            assert single_req_policy_quota['binding'] == args.request_binding
            assert single_req_policy_quota['scope'] == args.request_scope
            assert single_req_policy_quota['allow'] is False
            assert single_req_policy_quota['effective_rate_per_minute'] == 0.0
            assert single_req_policy_quota['reason_code'] == 'quota_exhausted'

            # Switch signal to hard_brake=True, remaining=600 -> reason_code='hard_brake', phase stays 'waiting_policy'
            await set_signal_state(signal_url, remaining=600, hard_brake=True, factor=1.0)
            await wait_for_scope_snapshot(
                db_url,
                scope_key=scheduler_scope_key,
                predicate=lambda s, _: s.get('reason_code') == 'hard_brake',
            )
            single_sched_policy_brake = await api.control_json(
                'GET', f'/v1/executions/{gated_single_id}/scheduler-policy'
            )
            exe_during_brake = await api.control_json('GET', f'/v1/executions/{gated_single_id}')
            assert single_sched_policy_brake['allow'] is False
            assert single_sched_policy_brake['reason_code'] == 'hard_brake'
            assert exe_during_brake['status'] == 'queued'
            assert exe_during_brake['phase'] == 'waiting_policy'

            # Switch signal to remaining=-1 -> reason_code='signal_incomplete' on both /scheduler-policy & /request-policy
            await set_signal_state(signal_url, remaining=-1, hard_brake=False, factor=1.0)
            await wait_for_scope_snapshot(
                db_url,
                scope_key=scheduler_scope_key,
                predicate=lambda s, _: s.get('reason_code') == 'signal_incomplete',
            )
            await wait_for_scope_snapshot(
                db_url,
                scope_key=request_scope_key,
                predicate=lambda s, _: s.get('reason_code') == 'signal_incomplete',
            )
            single_sched_policy_incomplete = await api.control_json(
                'GET', f'/v1/executions/{gated_single_id}/scheduler-policy'
            )
            single_req_policy_incomplete = await api.control_json(
                'GET', f'/v1/executions/{gated_single_id}/request-policy'
            )
            assert single_sched_policy_incomplete['allow'] is False
            assert single_sched_policy_incomplete['reason_code'] == 'signal_incomplete'
            assert single_req_policy_incomplete['allow'] is False
            assert single_req_policy_incomplete['effective_rate_per_minute'] == 0.0
            assert single_req_policy_incomplete['reason_code'] == 'signal_incomplete'

            # Submit unbound Single while tide-gate is blocked -> must bypass waiting_policy and succeed!
            unbound_single_id = submit_single(
                source=source,
                project_root=PROJECT_ROOT,
                environment_id=args.environment,
                control_url=control_url,
            )
            unbound_sched_policy = await api.control_json('GET', f'/v1/executions/{unbound_single_id}/scheduler-policy')
            assert unbound_sched_policy == {'enabled': False}

            unbound_observed_phases: list[str] = []
            deadline = time.monotonic() + 240.0
            while time.monotonic() < deadline:
                u_exe = await api.control_json('GET', f'/v1/executions/{unbound_single_id}')
                u_phase = str(u_exe.get('phase'))
                if not unbound_observed_phases or unbound_observed_phases[-1] != u_phase:
                    unbound_observed_phases.append(u_phase)
                if u_phase in {'succeeded', 'failed', 'cancelled', 'expired'}:
                    assert u_exe['status'] == 'succeeded', u_exe
                    assert u_exe['phase'] == 'succeeded', u_exe
                    break
                await asyncio.sleep(0.5)
            else:
                raise TimeoutError('Unbound single timed out')
            assert 'waiting_policy' not in unbound_observed_phases

            # Verify expired_single_id reached status='expired', phase='expired' while waiting on policy
            deadline = time.monotonic() + 60.0
            expired_final_view: dict[str, Any] = {}
            while time.monotonic() < deadline:
                exp_exe = await api.control_json('GET', f'/v1/executions/{expired_single_id}')
                exp_phase = str(exp_exe.get('phase'))
                if not expired_observed_phases or expired_observed_phases[-1] != exp_phase:
                    expired_observed_phases.append(exp_phase)
                    print(
                        f'[ExpiredSingle {expired_single_id}] status={exp_exe["status"]} phase={exp_phase}', flush=True
                    )
                if exp_phase == 'expired':
                    expired_final_view = exp_exe
                    break
                await asyncio.sleep(0.4)
            else:
                raise TimeoutError(
                    f'Expired single {expired_single_id} did not expire; observed={expired_observed_phases}'
                )

            assert expired_final_view['status'] == 'expired'
            assert expired_final_view['phase'] == 'expired'
            assert expired_final_view['attempt_count'] == 0
            assert expired_final_view['current_assignment_id'] is None
            expired_events_resp = await api.control_json('GET', f'/v1/executions/{expired_single_id}/control-events')
            expired_event_types = [ev['type'] for ev in expired_events_resp.get('items', [])]
            assert 'scheduler.policy_deferred' in expired_event_types
            assert 'execution.status.changed' in expired_event_types
            assert 'scheduler.policy_applied' not in expired_event_types

            # Verify gated Single is STILL in waiting_policy while unbound Single finished and short-TTL Single expired
            exe_still_waiting = await api.control_json('GET', f'/v1/executions/{gated_single_id}')
            assert exe_still_waiting['status'] == 'queued'
            assert exe_still_waiting['phase'] == 'waiting_policy'

            # Restore signal: remaining=600, hard_brake=False -> allow=True, reason_code='capacity_available'
            await set_signal_state(signal_url, remaining=600, hard_brake=False, factor=1.0)
            deadline = time.monotonic() + 240.0
            while time.monotonic() < deadline:
                exe = await api.control_json('GET', f'/v1/executions/{gated_single_id}')
                phase = str(exe.get('phase'))
                if not single_observed_phases or single_observed_phases[-1] != phase:
                    single_observed_phases.append(phase)
                    print(f'[Single {gated_single_id}] status={exe["status"]} phase={phase}', flush=True)
                if phase in {'succeeded', 'failed', 'cancelled', 'expired'}:
                    assert exe['status'] == 'succeeded', exe
                    assert exe['phase'] == 'succeeded', exe
                    assert exe['attempt_count'] == 1, exe
                    break
                await asyncio.sleep(0.4)
            else:
                raise TimeoutError('Gated single did not complete after policy recovery')

            single_events_resp = await api.control_json('GET', f'/v1/executions/{gated_single_id}/control-events')
            single_event_types = [ev['type'] for ev in single_events_resp.get('items', [])]
            assert 'scheduler.policy_deferred' in single_event_types
            assert 'scheduler.policy_applied' in single_event_types

            # ==================================================================
            # Scenario 2 & 3: Batch Children waiting_policy, Batch Summary
            #                 child_waiting_policy_count, Parent /scheduler-policy
            #                 & /request-policy, 1-slot contention transition
            #                 from waiting_policy -> queued, and runtime
            #                 below_minimum pause & resume
            # ==================================================================
            await set_signal_state(signal_url, remaining=0, hard_brake=False, factor=1.0)
            await wait_for_scope_snapshot(
                db_url,
                scope_key=scheduler_scope_key,
                predicate=lambda s, _: s.get('reason_code') == 'quota_exhausted',
            )
            await wait_for_scope_snapshot(
                db_url,
                scope_key=request_scope_key,
                predicate=lambda s, _: s.get('reason_code') == 'quota_exhausted',
            )

            batch_id = submit_batch(
                source=source,
                project_root=PROJECT_ROOT,
                environment_id=args.environment,
                control_url=control_url,
                scheduler_policy=args.scheduler_binding,
                policy_scope=args.scheduler_scope,
                request_policy=args.request_binding,
                request_scope=args.request_scope,
            )

            batch_observed_phases: list[str] = []
            children_ids: list[str] = []
            children_waiting_views: list[dict[str, Any]] = []
            batch_summary_waiting: dict[str, Any] = {}
            deadline = time.monotonic() + 240.0
            while time.monotonic() < deadline:
                b_exe = await api.control_json('GET', f'/v1/executions/{batch_id}')
                b_phase = str(b_exe.get('phase'))
                if not batch_observed_phases or batch_observed_phases[-1] != b_phase:
                    batch_observed_phases.append(b_phase)
                    print(f'[Batch {batch_id}] status={b_exe["status"]} phase={b_phase}', flush=True)
                if b_exe['status'] in {'failed', 'cancelled', 'expired', 'lost'}:
                    raise RuntimeError(f'Batch {batch_id} entered terminal status early: {b_exe}')
                items_resp = await api.control_json('GET', f'/v1/executions/{batch_id}/batch/items')
                c_ids = [
                    i['current_child_execution_id']
                    for i in items_resp.get('items', [])
                    if i.get('current_child_execution_id')
                ]
                if len(c_ids) == 2:
                    c_views = [await api.control_json('GET', f'/v1/executions/{cid}') for cid in c_ids]
                    if all(c['status'] == 'queued' and c['phase'] == 'waiting_policy' for c in c_views):
                        b_summary = await api.control_json(
                            'GET', f'/v1/executions/{batch_id}/batch?include_status_counts=true'
                        )
                        assert b_exe['status'] == 'running'
                        assert b_exe['phase'] == 'running_children'
                        assert b_summary['child_waiting_policy_count'] == 2
                        assert b_summary['child_execution_status_counts'] == {'queued': 2}
                        children_ids = c_ids
                        children_waiting_views = c_views
                        batch_summary_waiting = b_summary
                        break
                await asyncio.sleep(0.4)
            else:
                raise TimeoutError('Batch children did not both reach waiting_policy')

            # Verify list endpoint also projects waiting_policy for children and running_children for parent
            list_resp_batch = await api.control_json('GET', '/v1/executions')
            by_id = {item['execution_id']: item for item in list_resp_batch.get('items', [])}
            assert by_id[batch_id]['phase'] == 'running_children'
            for cid in children_ids:
                assert by_id[cid]['phase'] == 'waiting_policy'

            # Inspect /scheduler-policy and /request-policy directly on the Batch parent!
            batch_sched_policy_quota = await api.control_json('GET', f'/v1/executions/{batch_id}/scheduler-policy')
            assert batch_sched_policy_quota['enabled'] is True
            assert batch_sched_policy_quota['binding'] == args.scheduler_binding
            assert [s['scope_id'] for s in batch_sched_policy_quota['scopes']] == [args.scheduler_scope]
            assert batch_sched_policy_quota['mode'] == 'enforce'
            assert batch_sched_policy_quota['allow'] is False
            assert batch_sched_policy_quota['reason_code'] == 'quota_exhausted'
            assert batch_sched_policy_quota['next_schedule_at'] is None

            child0_sched_policy_quota = await api.control_json(
                'GET', f'/v1/executions/{children_ids[0]}/scheduler-policy'
            )
            assert child0_sched_policy_quota['allow'] is False
            assert child0_sched_policy_quota['reason_code'] == 'quota_exhausted'
            assert child0_sched_policy_quota['next_schedule_at'] is not None

            batch_req_policy_quota = await api.control_json('GET', f'/v1/executions/{batch_id}/request-policy')
            assert batch_req_policy_quota['enabled'] is True
            assert batch_req_policy_quota['binding'] == args.request_binding
            assert batch_req_policy_quota['scope'] == args.request_scope
            assert batch_req_policy_quota['configured_quota_qpm'] == QUOTA_QPM
            assert batch_req_policy_quota['allow'] is False
            assert batch_req_policy_quota['effective_rate_per_minute'] == 0.0
            assert batch_req_policy_quota['reason_code'] == 'quota_exhausted'

            # Switch signal to hard_brake=True, remaining=600 -> Batch parent /scheduler-policy updates to hard_brake
            await set_signal_state(signal_url, remaining=600, hard_brake=True, factor=1.0)
            await wait_for_scope_snapshot(
                db_url,
                scope_key=scheduler_scope_key,
                predicate=lambda s, _: s.get('reason_code') == 'hard_brake',
            )
            batch_sched_policy_brake = await api.control_json('GET', f'/v1/executions/{batch_id}/scheduler-policy')
            batch_summary_brake = await api.control_json(
                'GET', f'/v1/executions/{batch_id}/batch?include_status_counts=true'
            )
            assert batch_sched_policy_brake['allow'] is False
            assert batch_sched_policy_brake['reason_code'] == 'hard_brake'
            assert batch_sched_policy_brake['next_schedule_at'] is None
            assert batch_summary_brake['child_waiting_policy_count'] == 2

            # Set slot_cost=2 on both children so the 2-slot Manager can only run 1 child at a time
            # when scheduler policy unblocks, forcing the 2nd child from waiting_policy -> queued (no_capacity)
            # and verifying child_waiting_policy_count immediately drops to 0.
            # Simultaneously set factor=0.001 (< min_quota_ratio=0.01) so request_rate enters below_minimum
            # while the 1st child is in phase="running" (and parent is in phase="running_children").
            engine = create_async_engine(db_url)
            sessions = async_sessionmaker(engine, expire_on_commit=False)
            try:
                async with sessions() as session:
                    for cid in children_ids:
                        c_row = await session.get(ExecutionRow, cid)
                        assert c_row is not None
                        c_row.slot_cost = 2
                    await session.commit()
            finally:
                await engine.dispose()

            await set_signal_state(signal_url, remaining=600, hard_brake=False, factor=0.001)
            await wait_for_scope_snapshot(
                db_url,
                scope_key=request_scope_key,
                predicate=lambda s, _: s.get('reason_code') == 'below_minimum',
            )
            slot_contention_snapshot: dict[str, Any] = {}
            batch_req_policy_below_min: dict[str, Any] = {}
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline:
                b_exe = await api.control_json('GET', f'/v1/executions/{batch_id}')
                b_phase = str(b_exe.get('phase'))
                if not batch_observed_phases or batch_observed_phases[-1] != b_phase:
                    batch_observed_phases.append(b_phase)
                    print(f'[Batch {batch_id}] status={b_exe["status"]} phase={b_phase}', flush=True)
                c_views = [await api.control_json('GET', f'/v1/executions/{cid}') for cid in children_ids]
                phases_set = {c['phase'] for c in c_views}
                if b_phase == 'running_children' and phases_set == {'running', 'queued'}:
                    b_summary_contention = await api.control_json(
                        'GET', f'/v1/executions/{batch_id}/batch?include_status_counts=true'
                    )
                    assert b_summary_contention['child_waiting_policy_count'] == 0
                    assert b_summary_contention['child_execution_status_counts'] == {'running': 1, 'queued': 1}
                    running_child = next(c for c in c_views if c['phase'] == 'running')
                    b_req_below = await api.control_json('GET', f'/v1/executions/{batch_id}/request-policy')
                    c_req_below = await api.control_json(
                        'GET', f'/v1/executions/{running_child["execution_id"]}/request-policy'
                    )
                    assert b_req_below['allow'] is False
                    assert b_req_below['effective_rate_per_minute'] == 0.0
                    assert b_req_below['reason_code'] == 'below_minimum'
                    assert c_req_below['allow'] is False
                    assert c_req_below['effective_rate_per_minute'] == 0.0
                    assert c_req_below['reason_code'] == 'below_minimum'
                    batch_req_policy_below_min = b_req_below
                    slot_contention_snapshot = {
                        'batch_phase': b_phase,
                        'child_waiting_policy_count': b_summary_contention['child_waiting_policy_count'],
                        'child_execution_status_counts': b_summary_contention['child_execution_status_counts'],
                        'children': [
                            {
                                'execution_id': c['execution_id'],
                                'status': c['status'],
                                'phase': c['phase'],
                                'current_assignment_id': c['current_assignment_id'],
                            }
                            for c in c_views
                        ],
                    }
                    break
                await asyncio.sleep(0.3)
            else:
                raise TimeoutError('Did not observe 1 running + 1 queued child with child_waiting_policy_count=0')

            # Resume request budget with factor=2.5 (> 1.0 triggers reason_code='resumed' to full 240.0 QPM)
            await set_signal_state(signal_url, remaining=600, hard_brake=False, factor=2.5)
            await wait_for_scope_snapshot(
                db_url,
                scope_key=request_scope_key,
                predicate=lambda s, _: s.get('reason_code') == 'resumed',
            )
            batch_req_policy_resumed = await api.control_json('GET', f'/v1/executions/{batch_id}/request-policy')
            assert batch_req_policy_resumed['allow'] is True
            assert batch_req_policy_resumed['effective_rate_per_minute'] == QUOTA_QPM
            assert batch_req_policy_resumed['reason_code'] == 'resumed'

            # Wait for Batch and all children to complete with succeeded
            deadline = time.monotonic() + 240.0
            while time.monotonic() < deadline:
                b_exe = await api.control_json('GET', f'/v1/executions/{batch_id}')
                b_phase = str(b_exe.get('phase'))
                if not batch_observed_phases or batch_observed_phases[-1] != b_phase:
                    batch_observed_phases.append(b_phase)
                    print(f'[Batch {batch_id}] status={b_exe["status"]} phase={b_phase}', flush=True)
                b_summary = await api.control_json('GET', f'/v1/executions/{batch_id}/batch?include_status_counts=true')
                if b_summary['control_state'] == 'finalized' and b_phase in {
                    'succeeded',
                    'failed',
                    'cancelled',
                    'expired',
                }:
                    assert b_summary['outcome'] == 'succeeded', b_summary
                    assert b_summary['summary']['succeeded'] == 2, b_summary
                    assert b_summary['child_waiting_policy_count'] == 0, b_summary
                    assert b_exe['status'] == 'succeeded', b_exe
                    assert b_exe['phase'] == 'succeeded', b_exe
                    break
                await asyncio.sleep(0.5)
            else:
                raise TimeoutError('Batch did not complete after policy recovery')

            for cid in children_ids:
                c_events_resp = await api.control_json('GET', f'/v1/executions/{cid}/control-events')
                c_event_types = [ev['type'] for ev in c_events_resp.get('items', [])]
                assert 'scheduler.policy_deferred' in c_event_types
                assert 'scheduler.policy_applied' in c_event_types

            # ==================================================================
            # Scenario 4: Dynamic K8s Allocations & StatefulSets Release
            # ==================================================================
            all_allocations: list[dict[str, Any]] = []
            for top_id in (gated_single_id, expired_single_id, unbound_single_id, batch_id):
                deadline = time.monotonic() + 120.0
                while time.monotonic() < deadline:
                    alloc_resp = await api.control_json('GET', f'/v1/executions/{top_id}/allocations')
                    allocs = alloc_resp.get('items', [])
                    if allocs and all(a['status'] == 'released' for a in allocs):
                        all_allocations.extend(allocs)
                        break
                    await asyncio.sleep(1.0)
                else:
                    raise TimeoutError(f'Allocations for {top_id} were not released')

            for alloc in all_allocations:
                sts_name, _ = allocation_resource_names(alloc['allocation_id'])
                check = run_cmd([kubectl, '-n', namespace, 'get', 'statefulset', sts_name], check=False)
                assert check.returncode != 0, f'StatefulSet {sts_name} still exists'

        report = {
            'single': {
                'gated_single_id': gated_single_id,
                'observed_phases': single_observed_phases,
                'waiting_policy_view': {
                    k: single_waiting_policy_view.get(k)
                    for k in ('execution_id', 'status', 'phase', 'dependency_status', 'attempt_count')
                },
                'scheduler_policy_quota_exhausted': single_sched_policy_quota,
                'scheduler_policy_hard_brake': single_sched_policy_brake,
                'scheduler_policy_signal_incomplete': single_sched_policy_incomplete,
                'request_policy_quota_exhausted': single_req_policy_quota,
                'request_policy_signal_incomplete': single_req_policy_incomplete,
                'control_event_types': single_event_types,
            },
            'expired_single': {
                'expired_single_id': expired_single_id,
                'observed_phases': expired_observed_phases,
                'final_view': {
                    k: expired_final_view.get(k)
                    for k in ('execution_id', 'status', 'phase', 'attempt_count', 'current_assignment_id')
                },
                'control_event_types': expired_event_types,
            },
            'unbound_single': {
                'unbound_single_id': unbound_single_id,
                'observed_phases': unbound_observed_phases,
                'scheduler_policy': unbound_sched_policy,
            },
            'batch': {
                'batch_id': batch_id,
                'children_ids': children_ids,
                'observed_phases': batch_observed_phases,
                'batch_summary_waiting': {
                    k: batch_summary_waiting.get(k)
                    for k in (
                        'parent_execution_id',
                        'control_state',
                        'submitted_items',
                        'active_items',
                        'child_waiting_policy_count',
                        'child_execution_status_counts',
                    )
                },
                'children_waiting_views': [
                    {k: c.get(k) for k in ('execution_id', 'status', 'phase', 'attempt_count')}
                    for c in children_waiting_views
                ],
                'parent_scheduler_policy_quota_exhausted': batch_sched_policy_quota,
                'child0_scheduler_policy_quota_exhausted': child0_sched_policy_quota,
                'parent_scheduler_policy_hard_brake': batch_sched_policy_brake,
                'parent_request_policy_quota_exhausted': batch_req_policy_quota,
                'parent_request_policy_below_minimum': batch_req_policy_below_min,
                'parent_request_policy_resumed': batch_req_policy_resumed,
                'slot_contention_snapshot': slot_contention_snapshot,
            },
            'released_allocations': [a['allocation_id'] for a in all_allocations],
        }
        (args.output / 'waiting-policy-k8s-e2e-report.json').write_text(
            json.dumps(report, indent=2),
            encoding='utf-8',
        )
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run K8s E2E for waiting_policy phase and Batch policy observability.')
    parser.add_argument('--kubectl', default='kubectl')
    parser.add_argument('--helm', default='helm')
    parser.add_argument('--namespace', default='nexus-dist')
    parser.add_argument('--release-name', default='nexus-dist')
    parser.add_argument('--environment', default='kind-dynamic')
    parser.add_argument('--scheduler-binding', default='tide-gate')
    parser.add_argument('--scheduler-scope', default='hy3')
    parser.add_argument('--request-binding', default='team-a-hy3-rate')
    parser.add_argument('--request-scope', default='capacity')
    parser.add_argument('--image-tag', default='waiting-policy-e2e')
    parser.add_argument('--skip-bootstrap', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('/tmp/nexus-waiting-policy-k8s-e2e'))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kubeconfig = args.output / 'kubeconfig'
    if kubeconfig.exists() and not os.environ.get('KUBECONFIG'):
        os.environ['KUBECONFIG'] = str(kubeconfig)
    report = asyncio.run(run_e2e(args))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
