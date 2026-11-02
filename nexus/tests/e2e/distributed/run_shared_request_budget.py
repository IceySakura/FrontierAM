"""Exercise K8s dynamic allocation and shared request budget (kind=request_rate) end-to-end via Helm chart.

Deploys the nexus-distributed Helm chart (deployment/distributed/helm/nexus-distributed) with the
Kubernetes dynamic environment provider enabled, and drives real source-package Batches and Singles through:
1. Shadow mode observation (requests pass without lease gating; /request-policy reports observed rate);
2. Enforce mode across multiple dynamic InstanceManager StatefulSets (Batch with 2 Manager pods + concurrent
   Single with 1 Manager pod) sharing one 240 QPM budget in MySQL without duplicating total quota;
3. Multiplicative factor adjustments & floor pause:
   - Ordinary decrease uses factor directly (factor=0.5 -> 120 QPM);
   - Another decrease applies immediately (factor=0.8 -> 96 QPM);
   - hard_brake=true is ignored; falling below the floor pauses requests (0 QPM);
4. Quota exhaustion (remaining=0 -> allow=False, rate=0.0, reason=quota_exhausted, 0 completions issued),
   Scheduler Pod restart survival while blocked, and recovery on positive balance;
5. Online mode switch to disabled and complete K8s StatefulSet / Allocation cleanup verification.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
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
    PolicyScopeRow,
    RequestBudgetLeaseRow,
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

PROBE_CODE = """from nexus.core import TaskRunInput, TaskRunOutput, require_context, task_registry


class BudgetK8sProbeInput(TaskRunInput):
    index: int = 0
    requests: int = 0


class BudgetK8sProbeOutput(TaskRunOutput):
    marker: str
    requests: int


@task_registry.register('budget_k8s_probe')
async def budget_k8s_probe(value: BudgetK8sProbeInput) -> BudgetK8sProbeOutput:
    if value.requests > 0:
        client = require_context().get_openai_client('main')
        for step in range(value.requests):
            await client.create_completion([{'role': 'user', 'content': f'{value.index}:{step}'}])
    return BudgetK8sProbeOutput(marker=f'budget-k8s-e2e:{value.index}', requests=value.requests)
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

    def restart(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._process.wait(timeout=3)
            if self._process.poll() is None:
                self._process.kill()
        time.sleep(0.5)
        wait_tcp('127.0.0.1', self.local_port, timeout=30.0)

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
    binding_name: str,
    scope_name: str,
    capacity_url: str,
    mode: str,
    revision: int,
    quota_qpm: float = QUOTA_QPM,
    include_request_defaults: bool = True,
) -> dict[str, Any]:
    return {
        'imports': ['nexus.distributed.policies.collections.infer_proxy_request_rate'],
        'workers': 2,
        'poll_seconds': 0.5,
        'idle_seconds': 3600,
        'worker_startup_seconds': 30,
        'bindings': {
            binding_name: {
                'kind': 'request_rate',
                'policy': REQUEST_RATE_POLICY,
                'source': REQUEST_CAPACITY_SOURCE,
                'revision': revision,
                'mode': mode,
                'on_unavailable': 'defer',
                'environments': [environment_id],
                'principals': [],
                'refresh_seconds': 2,
                'timeout_seconds': 4,
                'max_age_seconds': 30,
                'scopes': {
                    scope_name: {
                        'source_config': {
                            'model': MODEL,
                            'url': capacity_url,
                            'timeout_seconds': 3,
                            'api_key': PLATFORM_API_KEY,
                        }
                    }
                },
                'params': {
                    'quota_qpm': quota_qpm,
                    'min_quota_ratio': 0.01,
                },
            },
            'team-b-fixed-rate': {
                'kind': 'request_rate',
                'policy': 'fixed_request_rate',
                'source': 'static_request_capacity',
                'revision': 1,
                'mode': 'enforce',
                'on_unavailable': 'defer',
                'environments': [environment_id],
                'principals': [],
                'refresh_seconds': 2,
                'timeout_seconds': 4,
                'max_age_seconds': 30,
                'scopes': {
                    'fixed': {
                        'source_config': {
                            'rate_per_minute': 360.0,
                            'allow': True,
                        }
                    }
                },
                'params': {},
            },
        },
        'request_defaults': (
            {
                environment_id: {
                    'binding': binding_name,
                    'scope': scope_name,
                }
            }
            if include_request_defaults
            else {}
        ),
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
    run_cmd([
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
    ])
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
            binding_name=args.binding,
            scope_name=args.scope,
            capacity_url=f'{signal_cluster_url}/capacity',
            mode='shadow',
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
    (source / 'budget_k8s_probe.py').write_text(PROBE_CODE, encoding='utf-8')
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
        json.dumps({
            'imports': ['budget_k8s_probe'],
            'task_name': 'budget_k8s_probe',
            'task_config': {'index': 100, 'requests': 6},
            'openai_clients': client_block,
        }),
        encoding='utf-8',
    )
    for gen_name, item_reqs in (
        ('generator.py', [10, 10, 8, 8]),
        ('generator_batch2.py', [6, 6]),
    ):
        generator_code = f"""from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput

ITEM_REQUESTS = {item_reqs!r}

def get_total_count():
    return len(ITEM_REQUESTS)

def generate_configs():
    for i, reqs in enumerate(ITEM_REQUESTS):
        yield BatchItem(
            run_id=f'item-{{i + 1:012d}}',
            config=RunInput.from_dict({{
                'imports': ['budget_k8s_probe'],
                'task_name': 'budget_k8s_probe',
                'task_config': {{'index': i, 'requests': reqs}},
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
        (source / gen_name).write_text(generator_code, encoding='utf-8')
    return source


def submit_batch(
    *,
    source: Path,
    project_root: Path,
    environment_id: str,
    control_url: str,
    generator_name: str = 'generator.py',
    manager_count: int = 2,
    max_inflight: int = 4,
) -> str:
    command = [
        sys.executable,
        '-m',
        'nexus.cli.main',
        'submit',
        'batch',
        '--generator',
        generator_name,
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
        str(manager_count),
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
        str(max_inflight),
        '--timeout-per-run',
        '600',
        '--queue-ttl-seconds',
        '1800',
        '--execution-ttl-seconds',
        '3600',
        '--child-queue-ttl-seconds',
        '1800',
        '--idempotency-key',
        f'budget-k8s-batch-{uuid.uuid4().hex}',
        '--json',
    ]
    completed = run_cmd(command)
    payload = json.loads(completed.stdout)
    return str(payload.get('execution_id') or payload['execution']['execution_id'])


def submit_single(
    *,
    source: Path,
    project_root: Path,
    environment_id: str,
    control_url: str,
    request_policy: str | None = None,
    request_scope: str | None = None,
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
        '2',
        '--queue-ttl-seconds',
        '1800',
        '--execution-ttl-seconds',
        '3600',
        '--idempotency-key',
        f'budget-k8s-single-{uuid.uuid4().hex}',
        '--json',
    ]
    if request_policy and request_scope:
        command += ['--request-policy', request_policy, '--request-scope', request_scope]
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


async def get_signal_state(signal_url: str) -> dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f'{signal_url}/state',
            headers={'X-Policy-Test-Key': ADMIN_SIGNAL_KEY},
        ) as resp:
            resp.raise_for_status()
            return await resp.json()


async def update_policy_config(
    *,
    kubectl: str,
    namespace: str,
    release: str,
    db_url: str,
    policy_executor: dict[str, Any],
    pf_control: PortForward | None = None,
    control_url: str | None = None,
) -> int:
    from nexus.distributed.config.store import set_config_override

    engine = create_async_engine(db_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as session:
            row = await session.scalar(
                select(DistributedConfigOverrideRow).where(
                    DistributedConfigOverrideRow.role == '*',
                    DistributedConfigOverrideRow.identity == '*',
                )
            )
            assert row is not None
            config_json = dict(row.config_json)
            expected_version = int(row.version)
            config_json['policy_executor'] = policy_executor
            record = await set_config_override(
                session,
                role='*',
                identity='*',
                values=config_json,
                expected_version=expected_version,
                actor='e2e:shared-request-budget',
            )
            await session.commit()
            new_version = record.version
    finally:
        await engine.dispose()

    run_cmd([
        kubectl,
        '-n',
        namespace,
        'rollout',
        'restart',
        f'deployment/{release}-scheduler',
        f'deployment/{release}-control-gateway',
    ])
    run_cmd([kubectl, '-n', namespace, 'rollout', 'status', f'deployment/{release}-scheduler', '--timeout=120s'])
    run_cmd([kubectl, '-n', namespace, 'rollout', 'status', f'deployment/{release}-control-gateway', '--timeout=120s'])
    if pf_control is not None:
        pf_control.restart()
    if control_url is not None:
        await wait_http(f'{control_url}/v1/health')
    return int(new_version)


async def get_request_policy_status(control_url: str, execution_id: str) -> dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.get(f'{control_url}/v1/executions/{execution_id}/request-policy') as resp:
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
            await asyncio.sleep(0.5)
    finally:
        await engine.dispose()
    raise TimeoutError(f'Timed out waiting for scope snapshot {scope_key}')


async def run_e2e(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    app_local_port = free_port()
    signal_cluster_url = bootstrap_cluster(args, app_local_port=app_local_port)

    kubectl = args.kubectl
    namespace = args.namespace
    release = args.release_name
    budget_key = policy_scope_key('request_rate', args.binding, args.scope)

    with ExitStack() as stack:
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
        pf_mysql = stack.enter_context(
            PortForward(kubectl=kubectl, namespace=namespace, resource=f'service/{release}-mysql', remote_port=3306)
        )

        control_url = f'http://127.0.0.1:{pf_control.local_port}'
        signal_url = f'http://127.0.0.1:{pf_signal.local_port}'
        db_url = local_database_url(port=pf_mysql.local_port, password='nexus_dist_local')

        await wait_http(f'{control_url}/v1/health')
        await wait_http(f'{signal_url}/health')
        await ensure_dynamic_environment(control_url, namespace=namespace, environment_id=args.environment)

        source = prepare_source(
            args.output,
            project_root=PROJECT_ROOT,
            llm_base_url=f'{signal_cluster_url}/v1',
        )

        # ------------------------------------------------------------------
        # Scenario 1: Shadow mode observation with dynamic K8s Single
        # ------------------------------------------------------------------
        await set_signal_state(
            signal_url, factor=0.5, hard_brake=False, remaining=600, mode='ok', delay=0.0, reset_counts=True
        )
        snap_shadow = await wait_for_scope_snapshot(
            db_url,
            scope_key=budget_key,
            predicate=lambda s, _: s.get('payload', {}).get('allow') is True,
        )
        shadow_single_id = submit_single(
            source=source,
            project_root=PROJECT_ROOT,
            environment_id=args.environment,
            control_url=control_url,
        )
        async with CloudBatchApi(
            control_url=control_url,
            application_url=f'http://127.0.0.1:{app_local_port}',
            token='local-e2e',
        ) as api:
            deadline = time.monotonic() + 240.0
            while time.monotonic() < deadline:
                exe = await api.control_json('GET', f'/v1/executions/{shadow_single_id}')
                if exe['status'] in {'succeeded', 'failed', 'cancelled', 'expired'}:
                    assert exe['status'] == 'succeeded', exe
                    break
                await asyncio.sleep(1.0)
            else:
                raise TimeoutError('shadow single timed out')

        shadow_policy_status = await get_request_policy_status(control_url, shadow_single_id)
        assert shadow_policy_status['enabled'] is True
        assert shadow_policy_status['mode'] == 'shadow'
        assert shadow_policy_status['binding'] == args.binding
        assert shadow_policy_status['scope'] == args.scope

        # ------------------------------------------------------------------
        # Scenario 2: Switch to enforce mode (revision=2) and verify factor
        #             combinations (direct decrease, ignored hard_brake, floor pause)
        # ------------------------------------------------------------------
        await set_signal_state(
            signal_url, factor=1.0, hard_brake=False, remaining=600, mode='ok', delay=0.0, reset_counts=True
        )
        await update_policy_config(
            kubectl=kubectl,
            namespace=namespace,
            release=release,
            db_url=db_url,
            pf_control=pf_control,
            control_url=control_url,
            policy_executor=policy_executor_config(
                environment_id=args.environment,
                binding_name=args.binding,
                scope_name=args.scope,
                capacity_url=f'{signal_cluster_url}/capacity',
                mode='enforce',
                revision=2,
            ),
        )
        snap_full = await wait_for_scope_snapshot(
            db_url,
            scope_key=budget_key,
            predicate=lambda s, r: (
                r.config_revision == 2 and math.isclose(s.get('payload', {}).get('rate_per_minute', 0.0), 240.0)
            ),
        )

        # 2a. Ordinary decrease factor=0.5 -> 120 QPM
        await set_signal_state(signal_url, factor=0.5, hard_brake=False, remaining=600)
        snap_decreased = await wait_for_scope_snapshot(
            db_url,
            scope_key=budget_key,
            predicate=lambda s, _: (
                s.get('reason_code') == 'adopted'
                and math.isclose(s.get('payload', {}).get('rate_per_minute', 0.0), 120.0)
            ),
        )

        # 2b. Another ordinary decrease applies immediately -> 96 QPM
        await set_signal_state(signal_url, factor=0.8, hard_brake=False, remaining=600)
        snap_decreased_again = await wait_for_scope_snapshot(
            db_url,
            scope_key=budget_key,
            predicate=lambda s, _: (
                s.get('reason_code') == 'adopted'
                and math.isclose(s.get('payload', {}).get('rate_per_minute', 0.0), 96.0)
            ),
        )

        # 2c. Hard brake is ignored: the factor still decreases the rate to 48 QPM.
        await set_signal_state(signal_url, factor=0.5, hard_brake=True, remaining=600)
        snap_hard = await wait_for_scope_snapshot(
            db_url,
            scope_key=budget_key,
            predicate=lambda s, _: (
                s.get('payload', {}).get('allow') is True
                and math.isclose(s.get('payload', {}).get('rate_per_minute', 0.0), 48.0)
            ),
        )
        await set_signal_state(signal_url, factor=0.001, hard_brake=True, remaining=600)
        snap_brake = await wait_for_scope_snapshot(
            db_url,
            scope_key=budget_key,
            predicate=lambda s, _: s.get('reason_code') == 'below_minimum',
        )
        await set_signal_state(signal_url, factor=1.0, hard_brake=True, remaining=600)
        await wait_for_scope_snapshot(
            db_url,
            scope_key=budget_key,
            predicate=lambda s, _: (
                s.get('snapshot_version') != snap_brake['snapshot_version']
                and s.get('reason_code') == 'below_minimum'
                and s.get('payload', {}).get('rate_per_minute') == 0.0
            ),
        )

        # 2d. Slow signal timeout (delay=6.0s > timeout_seconds=4.0s): records failure without extending valid_until
        await set_signal_state(signal_url, factor=1.0, hard_brake=False, remaining=600, delay=6.0)
        timeout_failure_observed: dict[str, Any] = {}
        engine = create_async_engine(db_url)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        try:
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                async with sessions() as session:
                    from nexus.distributed.utils.clock import utcnow

                    row = await session.get(PolicyScopeRow, budget_key)
                    if row is not None:
                        row.last_demand_at = utcnow()
                        await session.commit()
                        if row.failures > 0 and row.last_error is not None:
                            timeout_failure_observed = {
                                'failures': row.failures,
                                'last_error': row.last_error,
                                'valid_until': row.valid_until.isoformat() if row.valid_until else None,
                            }
                            break
                await asyncio.sleep(0.5)
            else:
                raise TimeoutError('Timed out waiting for slow signal policy_timeout failure')
        finally:
            await engine.dispose()

        # Restore healthy signal (factor=2.5, delay=0.0) so rate scales back up to 240.0 QPM
        await set_signal_state(
            signal_url,
            factor=2.5,
            hard_brake=False,
            remaining=600,
            delay=0.0,
            mode='ok',
            reset_counts=True,
        )
        await wait_for_scope_snapshot(
            db_url,
            scope_key=budget_key,
            predicate=lambda s, r: (
                r.failures == 0 and math.isclose(s.get('payload', {}).get('rate_per_minute', 0.0), 240.0)
            ),
        )

        # ------------------------------------------------------------------
        # Scenario 3: Multi-Batch (Batch 1 with 2 Managers + Batch 2 with 1 Manager)
        #             + concurrent Single sharing the same budget_key simultaneously
        # ------------------------------------------------------------------
        batch_id = submit_batch(
            source=source,
            project_root=PROJECT_ROOT,
            environment_id=args.environment,
            control_url=control_url,
            generator_name='generator.py',
            manager_count=2,
            max_inflight=4,
        )
        batch2_id = submit_batch(
            source=source,
            project_root=PROJECT_ROOT,
            environment_id=args.environment,
            control_url=control_url,
            generator_name='generator_batch2.py',
            manager_count=1,
            max_inflight=2,
        )
        single_id = submit_single(
            source=source,
            project_root=PROJECT_ROOT,
            environment_id=args.environment,
            control_url=control_url,
        )

        # Wait until at least 2 dynamic Managers simultaneously hold UNEXPIRED leases (r.valid_until > now_db)
        # splitting the 240 QPM budget, and verify sum(unexpired granted_qpm) never exceeds 240 QPM!
        observed_multi_manager_leases: list[dict[str, Any]] = []
        engine = create_async_engine(db_url)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        try:
            deadline = time.monotonic() + 180.0
            while time.monotonic() < deadline:
                async with sessions() as session:
                    from nexus.distributed.utils.clock import utcnow

                    now_db = utcnow()
                    rows = (
                        await session.scalars(
                            select(RequestBudgetLeaseRow).where(RequestBudgetLeaseRow.budget_key == budget_key)
                        )
                    ).all()
                    unexpired = [
                        r for r in rows if r.valid_until is not None and r.valid_until > now_db and r.granted_qpm > 0
                    ]
                    unexpired_sum = sum(r.granted_qpm for r in unexpired)
                    assert unexpired_sum <= QUOTA_QPM + 1e-6, f'Over-allocated: {unexpired_sum} > {QUOTA_QPM}'
                    if len(unexpired) >= 2:
                        observed_multi_manager_leases = [
                            {
                                'manager_id': r.manager_id,
                                'session_id': r.manager_session_id,
                                'granted_qpm': r.granted_qpm,
                                'lease_version': r.lease_version,
                                'request_seq': r.request_seq,
                                'valid_until': r.valid_until.isoformat() if r.valid_until else None,
                            }
                            for r in unexpired
                        ]
                        break
                await asyncio.sleep(0.4)
            else:
                raise TimeoutError('Did not observe >= 2 simultaneous unexpired Manager leases splitting budget')
        finally:
            await engine.dispose()

        # ------------------------------------------------------------------
        # Scenario 4: Quota exhaustion (remaining=0) pause, zero-request window,
        #             Scheduler Pod restart survival while paused, and recovery
        # ------------------------------------------------------------------
        await set_signal_state(signal_url, factor=1.0, hard_brake=False, remaining=0)
        snap_zero = await wait_for_scope_snapshot(
            db_url,
            scope_key=budget_key,
            predicate=lambda s, _: (
                s.get('payload', {}).get('allow') is False and s.get('reason_code') == 'quota_exhausted'
            ),
        )
        # Wait for the 4s short leases to expire, then verify 0 completions occur over a 2.5s window.
        await asyncio.sleep(4.2)
        paused_before = (await get_signal_state(signal_url))['completions']
        await asyncio.sleep(2.5)
        paused_after = (await get_signal_state(signal_url))['completions']
        assert paused_after == paused_before, f'Requests leaked while paused: {paused_before} -> {paused_after}'

        # Restart Scheduler Pod while quota is exhausted: state must survive in MySQL!
        run_cmd([kubectl, '-n', namespace, 'rollout', 'restart', f'deployment/{release}-scheduler'])
        run_cmd([kubectl, '-n', namespace, 'rollout', 'status', f'deployment/{release}-scheduler', '--timeout=120s'])
        policy_while_blocked = await get_request_policy_status(control_url, batch_id)
        assert policy_while_blocked['allow'] is False
        assert policy_while_blocked['effective_rate_per_minute'] == 0.0
        assert policy_while_blocked['reason_code'] == 'quota_exhausted'

        # Restore positive quota (remaining=600, factor=2.5) and wait for Batch 1, Batch 2, and Single to succeed!
        await set_signal_state(signal_url, factor=2.5, hard_brake=False, remaining=600)
        async with CloudBatchApi(
            control_url=control_url,
            application_url=f'http://127.0.0.1:{app_local_port}',
            token='local-e2e',
        ) as api:
            deadline = time.monotonic() + 300.0
            while time.monotonic() < deadline:
                b1_summary = await api.control_json('GET', f'/v1/executions/{batch_id}/batch')
                b2_summary = await api.control_json('GET', f'/v1/executions/{batch2_id}/batch')
                s_exe = await api.control_json('GET', f'/v1/executions/{single_id}')
                if (
                    b1_summary['control_state'] == 'finalized'
                    and b2_summary['control_state'] == 'finalized'
                    and s_exe['status'] in {'succeeded', 'failed', 'cancelled', 'expired'}
                ):
                    assert b1_summary['outcome'] == 'succeeded', b1_summary
                    assert b1_summary['summary']['succeeded'] == 4, b1_summary
                    assert b2_summary['outcome'] == 'succeeded', b2_summary
                    assert b2_summary['summary']['succeeded'] == 2, b2_summary
                    assert s_exe['status'] == 'succeeded', s_exe
                    break
                await asyncio.sleep(1.0)
            else:
                raise TimeoutError('Batch 1, Batch 2, or Single did not complete after quota recovery')

            # Verify client never polled capacity directly, total completions match (36 + 12 + 6 = 54),
            # and verify §7.4 sliding-window rate & burst bound across all recorded timestamps!
            sig_final = await get_signal_state(signal_url)
            assert sig_final['client_calls'] == 0
            assert sig_final['platform_keyed_calls'] > 0
            assert sig_final['completions'] == (36 + 12 + 6)
            timestamps = sorted(float(t) for t in sig_final.get('recent_completions', []))
            assert len(timestamps) >= 20
            # Under 240 QPM (4 req/s) across at most 4 active RequestBudgetControls, any window [t_i, t_j]
            # can contain at most 240 * (t_j - t_i) / 60 + 4 requests.
            max_burst_excess = 0.0
            for idx_i, t_i in enumerate(timestamps):
                for idx_j in range(idx_i, len(timestamps)):
                    t_j = timestamps[idx_j]
                    count = idx_j - idx_i + 1
                    rate_allowance = QUOTA_QPM * max(0.0, t_j - t_i) / 60.0
                    burst_excess = count - rate_allowance
                    max_burst_excess = max(max_burst_excess, burst_excess)
                    assert count <= rate_allowance + 4.5, (idx_i, idx_j, count, rate_allowance)

            # ------------------------------------------------------------------
            # Scenario 5: Switch team-a to mode=disabled (revision=3), exercise
            #             second built-in FixedRequestRatePolicy (team-b-fixed-rate),
            #             and verify all dynamic Allocations & StatefulSets are released
            # ------------------------------------------------------------------
            await update_policy_config(
                kubectl=kubectl,
                namespace=namespace,
                release=release,
                db_url=db_url,
                pf_control=pf_control,
                control_url=control_url,
                policy_executor=policy_executor_config(
                    environment_id=args.environment,
                    binding_name=args.binding,
                    scope_name=args.scope,
                    capacity_url=f'{signal_cluster_url}/capacity',
                    mode='disabled',
                    revision=3,
                    include_request_defaults=False,
                ),
            )
            disabled_policy_status = await get_request_policy_status(control_url, batch_id)
            assert disabled_policy_status['enabled'] is True
            assert disabled_policy_status['mode'] == 'disabled'
            assert disabled_policy_status['config_revision'] == 3

            fixed_single_id = submit_single(
                source=source,
                project_root=PROJECT_ROOT,
                environment_id=args.environment,
                control_url=control_url,
                request_policy='team-b-fixed-rate',
                request_scope='fixed',
            )
            deadline = time.monotonic() + 240.0
            while time.monotonic() < deadline:
                f_exe = await api.control_json('GET', f'/v1/executions/{fixed_single_id}')
                if f_exe['status'] in {'succeeded', 'failed', 'cancelled', 'expired'}:
                    assert f_exe['status'] == 'succeeded', f_exe
                    break
                await asyncio.sleep(1.0)
            else:
                raise TimeoutError('Fixed-rate policy single timed out')

            fixed_policy_status = await get_request_policy_status(control_url, fixed_single_id)
            assert fixed_policy_status['enabled'] is True
            assert fixed_policy_status['binding'] == 'team-b-fixed-rate'
            assert fixed_policy_status['scope'] == 'fixed'
            assert fixed_policy_status['effective_rate_per_minute'] == 360.0
            assert fixed_policy_status['budget_key'] != budget_key
            assert len(fixed_policy_status['managers']) == 1

            all_allocations: list[dict[str, Any]] = []
            for top_id in (shadow_single_id, batch_id, batch2_id, single_id, fixed_single_id):
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
            'shadow_single_id': shadow_single_id,
            'batch_id': batch_id,
            'batch2_id': batch2_id,
            'single_id': single_id,
            'fixed_single_id': fixed_single_id,
            'snapshots': {
                'shadow': snap_shadow,
                'full_240': snap_full,
                'decreased_120': snap_decreased,
                'decreased_96': snap_decreased_again,
                'hard_ignored_48': snap_hard,
                'below_minimum_0': snap_brake,
                'timeout_failure': timeout_failure_observed,
                'quota_exhausted': snap_zero,
            },
            'observed_multi_manager_leases': observed_multi_manager_leases,
            'paused_window_completions': {'before': paused_before, 'after_2_5s': paused_after},
            'max_burst_excess_requests': round(max_burst_excess, 4),
            'policy_while_blocked': policy_while_blocked,
            'disabled_policy_status': disabled_policy_status,
            'fixed_policy_status': fixed_policy_status,
            'signal_stats': sig_final,
            'released_allocations': [a['allocation_id'] for a in all_allocations],
        }
        (args.output / 'shared-request-budget-k8s-e2e-report.json').write_text(
            json.dumps(report, indent=2),
            encoding='utf-8',
        )
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run K8s Helm dynamic shared request budget E2E.')
    parser.add_argument('--kubectl', default='kubectl')
    parser.add_argument('--helm', default='helm')
    parser.add_argument('--namespace', default='nexus-dist')
    parser.add_argument('--release-name', default='nexus-dist')
    parser.add_argument('--environment', default='kind-dynamic')
    parser.add_argument('--binding', default='team-a-hy3-rate')
    parser.add_argument('--scope', default='capacity')
    parser.add_argument('--image-tag', default='cursor-local')
    parser.add_argument('--output', type=Path, default=Path('/tmp/nexus-cursor-budget-k8s-e2e'))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = asyncio.run(run_e2e(args))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
