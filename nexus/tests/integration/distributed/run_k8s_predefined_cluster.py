from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid

from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import yaml

from tencentcloud.common import credential as tencent_cloud_credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.dcdb.v20180411 import dcdb_client
from tencentcloud.dcdb.v20180411 import models as dcdb_models
from tests.integration.distributed.cluster_config import (
    TDSQLTraceConfig,
    create_database_bootstrap_secrets,
    load_tdsql_trace_config,
    local_database_url,
    platform_config_values,
    platform_control_gateway_secret_values,
    platform_secret_values,
    seed_platform_config,
)

from nexus.cli.commands.submit import _upload
from nexus.distributed.access.trace_database_providers.tencent_cloud_dcdb import (
    TencentCloudDCDBTraceDatabaseAccountProvider,
)
from nexus.distributed.config import DistributedConfig
from nexus.runner.single import RunInput
from nexus.runner.trace import principal_trace_scope_id
from nexus.runner.trace.tdsql._driver import connect as mysql_connect
from nexus.runner.trace.tdsql.config import build_db_config
from nexus.runner.trace.tdsql.schema import SHARED_TRACE_TABLE, principal_batch_trace_table_name


ROOT = Path(__file__).parents[3]
CHART_DIR = ROOT / 'deployment/distributed/helm/nexus-distributed'
LOCAL_VALUES = CHART_DIR / 'examples/values-local-predefined.yaml'
OPUS48_PAYLOAD = CHART_DIR / 'examples/opus48-live-execution.yaml'
TEST_WORKLOAD_DOCKERFILE = ROOT / 'tests/integration/distributed/Dockerfile.test-workload'
DEFAULT_DEVCLOUD_KUBECONFIG = Path('/root/.kube/devcloud-nexus-online-service.conf')
DEFAULT_DEVCLOUD_NAMESPACE = 'nexus-online-service'
DEFAULT_CONTROL_IMAGE = 'mirrors.tencent.com/nexus_utopia/nexus-distributed-universal:phase2-devcloud'
DEFAULT_WORKLOAD_IMAGE = 'mirrors.tencent.com/nexus_utopia/nexus-distributed-workload:phase2-devcloud'
DEFAULT_BUILDER_IMAGE = 'mirrors.tencent.com/nexus_utopia/nexus-distributed-builder:phase2-devcloud'
DEFAULT_MYSQL_IMAGE = 'mirrors.tencent.com/nexus_utopia/nexus-mysql:devcloud-8.0.45-20260723190651'
EXPECTED_CLUSTER_SLOT_COUNT = 2
EXPECTED_SCHEDULER_SHARD_COUNT = DistributedConfig().shard_count
PREDEFINED_BATCH_ITEM_COUNT = 4
TERMINAL_STATUSES = {'succeeded', 'failed', 'killed', 'lost', 'expired'}
AIOMONITOR_STATIC_PORT_CHECKS = (
    ('control-gateway', 50100),
    ('application-gateway', 50100),
    ('scheduler', 50100),
    ('reconciler', 50100),
    ('instance-manager', 50100),
    ('instance-manager', 50101),
    ('batch-manager', 50100),
)
REQUIRED_SECRET_ENV = (
    'GONGFENG_TOKEN',
    'MODEL_SERVICE_API_APP_ID',
    'MODEL_SERVICE_API_APP_KEY',
    'MODEL_SERVICE_API_BASE_URL',
)
DEFAULT_OPUS48_GONGFENG_API_BASE_URL = 'http://hyrl-sandbox.online.prod.woa.com:8052'


def parse_duration_seconds(value: str) -> float:
    raw = value.strip().lower()
    if raw.endswith('ms'):
        return float(raw[:-2]) / 1000
    if raw.endswith('s'):
        return float(raw[:-1])
    if raw.endswith('m'):
        return float(raw[:-1]) * 60
    if raw.endswith('h'):
        return float(raw[:-1]) * 3600
    return float(raw)


@dataclass(frozen=True)
class RepeatedClusterCase:
    task_key: str
    iteration: int
    execution_id: str
    mode: str
    kill: bool = False


def build_repeated_cluster_cases(run_tag: str) -> list[RepeatedClusterCase]:
    cases: list[RepeatedClusterCase] = []
    for iteration in range(1, 4):
        cases.append(
            RepeatedClusterCase(
                task_key='live_alpha',
                iteration=iteration,
                execution_id=f'{run_tag}-live-alpha-{iteration}',
                mode='live',
            )
        )
        cases.append(
            RepeatedClusterCase(
                task_key='live_beta',
                iteration=iteration,
                execution_id=f'{run_tag}-live-beta-{iteration}',
                mode='live',
                kill=iteration == 2,
            )
        )
        cases.append(
            RepeatedClusterCase(
                task_key='desktop_single',
                iteration=iteration,
                execution_id=f'{run_tag}-desktop-single-{iteration}',
                mode='job',
            )
        )
    return cases


def run(
    args: list[str],
    *,
    input_text: str | None = None,
    timeout: float | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    display = ' '.join(args)
    print(f'+ {display}', flush=True)
    return subprocess.run(
        args,
        cwd=ROOT,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        timeout=timeout,
        check=check,
    )


def cleanup_release_resources(
    *,
    helm: str,
    kubectl: str,
    namespace: str,
    release_name: str,
    database_secret_name: str,
    mysql_secret_name: str,
) -> None:
    pods = run(
        [
            kubectl,
            '-n',
            namespace,
            'get',
            'pods',
            '-l',
            f'app.kubernetes.io/instance={release_name}',
            '-o',
            'json',
        ],
        capture=True,
        check=False,
    )
    if pods.returncode == 0:
        try:
            items = json.loads(pods.stdout or '').get('items', [])
        except (AttributeError, json.JSONDecodeError):
            items = []
        for item in items:
            metadata = item.get('metadata') or {}
            finalizers = metadata.get('finalizers') or []
            if (
                item.get('status', {}).get('phase') == 'Succeeded'
                and metadata.get('deletionTimestamp')
                and 'batch.kubernetes.io/job-tracking' in finalizers
            ):
                run(
                    [
                        kubectl,
                        '-n',
                        namespace,
                        'patch',
                        'pod',
                        metadata['name'],
                        '--subresource=status',
                        '--type=json',
                        '-p',
                        '[{"op":"remove","path":"/metadata/finalizers"}]',
                    ],
                    check=False,
                )
    run([helm, 'uninstall', release_name, '--namespace', namespace, '--wait'], check=False)
    run(
        [
            kubectl,
            '-n',
            namespace,
            'delete',
            'secret',
            database_secret_name,
            mysql_secret_name,
            '--ignore-not-found=true',
        ],
        check=False,
    )
    for resource in ('pvc', 'pods'):
        run(
            [
                kubectl,
                '-n',
                namespace,
                'delete',
                resource,
                '-l',
                f'app.kubernetes.io/instance={release_name}',
                '--ignore-not-found=true',
                '--wait=true',
            ],
            check=False,
        )


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def merged_env(env_file: Path) -> dict[str, str]:
    values = load_env_file(env_file)
    for key, value in os.environ.items():
        values[key] = value
    if values.get('MODEL_SERVICE_API_BASE_URL'):
        values['MODEL_SERVICE_API_BASE_URL'] = normalize_model_service_base_url(values['MODEL_SERVICE_API_BASE_URL'])
    return values


def normalize_model_service_base_url(value: str) -> str:
    value = value.rstrip('/')
    suffix = '/openai/v1'
    if value.endswith(suffix):
        value = value[: -len(suffix)]
    return value


def require_binary(name: str, explicit: str | None = None) -> str:
    candidate = explicit or shutil.which(name)
    if not candidate:
        raise SystemExit(f'{name} not found; install it or pass --{name}-bin')
    return candidate


def split_image(image: str) -> tuple[str, str]:
    repository, separator, tag = image.rpartition(':')
    if separator and '/' not in tag:
        return repository, tag
    return image, ''


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def wait_tcp_ok(host: str, port: int, *, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    raise TimeoutError(f'{host}:{port} did not accept TCP connections: {last_error}')


def create_bootstrap_secrets(
    kubectl: str,
    namespace: str,
    env_values: dict[str, str],
    *,
    database_secret_name: str,
    mysql_secret_name: str,
    release_name: str,
) -> str:
    mysql_root_password = env_values.get('NEXUS_DIST_MYSQL_ROOT_PASSWORD', 'nexus_dist_local')
    create_database_bootstrap_secrets(
        run,
        kubectl=kubectl,
        namespace=namespace,
        release_name=release_name,
        database_secret_name=database_secret_name,
        mysql_secret_name=mysql_secret_name,
        password=mysql_root_password,
    )
    return mysql_root_password


def configure_devcloud(kubectl: str, *, kubeconfig: Path, namespace: str) -> None:
    if not kubeconfig.is_file():
        raise SystemExit(f'DevCloud kubeconfig not found: {kubeconfig}')
    os.environ['KUBECONFIG'] = str(kubeconfig)
    run([kubectl, 'get', 'namespace', namespace, '-o', 'name'], timeout=30)


def push_images(images: tuple[str, ...]) -> None:
    for image in images:
        registry = image.split('/', 1)[0]
        if '/' not in image or '.' not in registry:
            raise SystemExit(f'DevCloud image must use an explicit registry: {image}')
        run(['docker', 'push', image], timeout=1800)


class PortForward(AbstractContextManager['PortForward']):
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
        self.process: subprocess.Popen[str] | None = None

    @property
    def url(self) -> str:
        return f'http://127.0.0.1:{self.local_port}'

    def __enter__(self) -> PortForward:
        self.process = subprocess.Popen(
            [
                self.kubectl,
                '-n',
                self.namespace,
                'port-forward',
                self.resource,
                f'{self.local_port}:{self.remote_port}',
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


def seed_cluster_platform_config(
    kubectl: str,
    *,
    namespace: str,
    release_name: str,
    mysql_password: str,
    env_values: dict[str, str],
    application_public_base_url: str,
    workload_image: str,
    builder_image: str,
    database_secret_name: str,
    artifact_routing_enabled: bool = False,
    artifact_routing_seed: str = 'artifact-store-v1',
    artifact_routing_version: int = 1,
    artifact_routes: list[dict[str, Any]] | None = None,
    artifact_service_base_url: str = '',
    artifact_service_system_prefix: str = '/system',
    artifact_service_namespace_prefix: str = '/_nexus_dist',
    tdsql_trace: TDSQLTraceConfig | None = None,
) -> None:
    control_gateway_secrets = platform_control_gateway_secret_values(tdsql_trace=tdsql_trace)
    with PortForward(
        kubectl=kubectl,
        namespace=namespace,
        resource=f'service/{release_name}-mysql',
        remote_port=3306,
    ) as mysql:
        wait_tcp_ok('127.0.0.1', mysql.local_port, timeout=60)
        seed_platform_config(
            run,
            python=sys.executable,
            database_url=local_database_url(port=mysql.local_port, password=mysql_password),
            config_values=platform_config_values(
                application_public_base_url=application_public_base_url,
                application_internal_base_url=(
                    f'http://{release_name}-application-gateway.{namespace}.svc.cluster.local:8082'
                ),
                workload_image=workload_image,
                builder_image=builder_image,
                namespace=namespace,
                release_name=release_name,
                database_secret_name=database_secret_name,
                artifact_routing_enabled=artifact_routing_enabled,
                artifact_routing_seed=artifact_routing_seed,
                artifact_routing_version=artifact_routing_version,
                artifact_routes=artifact_routes or [],
                artifact_service_base_url=artifact_service_base_url,
                artifact_service_system_prefix=artifact_service_system_prefix,
                artifact_service_namespace_prefix=artifact_service_namespace_prefix,
                fake_runtime_base_url=f'http://{release_name}-fake-runtime:8080',
                tdsql_trace=tdsql_trace,
            ),
            secret_values=platform_secret_values(env_values, tdsql_trace=tdsql_trace),
            role_secret_values=(
                {('control-gateway', '*'): control_gateway_secrets} if control_gateway_secrets else None
            ),
        )


async def wait_http_ok(url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return
                    last_error = RuntimeError(f'{url} returned {response.status}')
            except Exception as exc:
                last_error = exc
            await asyncio.sleep(0.2)
    raise TimeoutError(f'{url} did not become ready: {last_error}')


async def assert_runtime_aiomonitor_webui(app_url: str, execution_id: str, *, timeout: float = 60.0) -> None:
    url = f'{app_url}/v1/executions/{execution_id}/runtimes/main/proxy/debug/aiomonitor/main/'
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(url) as response:
                    text = await response.text()
                    if response.status == 200 and ('aiomonitor' in text.lower() or '<html' in text.lower()):
                        print(f'runtime aiomonitor webui ready: {url}', flush=True)
                        return
                    last_error = RuntimeError(f'{url} returned {response.status}: {text[:200]}')
            except Exception as exc:
                last_error = exc
            await asyncio.sleep(1)
    raise TimeoutError(f'runtime aiomonitor webui did not become ready: {last_error}')


async def request_json(method: str, url: str, *, body: dict[str, Any] | None = None, timeout: float = 120.0):
    attempts = 3 if method.upper() == 'GET' else 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.request(method, url, json=body) as response:
                    text = await response.text()
                    payload = json.loads(text) if text else {}
                    if response.status >= 400:
                        raise RuntimeError(f'{method} {url} -> {response.status}: {payload}')
                    return payload
        except (TimeoutError, aiohttp.ClientError) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                raise
            await asyncio.sleep(1)
    raise RuntimeError(f'{method} {url} failed: {last_error}')


async def request_json_error(
    method: str,
    url: str,
    *,
    body: dict[str, Any],
    expected_statuses: set[int],
    timeout: float = 120.0,
) -> dict[str, Any]:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.request(method, url, json=body) as response:
            text = await response.text()
            payload = json.loads(text) if text else {}
            if response.status not in expected_statuses:
                raise RuntimeError(f'{method} {url} -> {response.status}, expected {expected_statuses}: {payload}')
            if not isinstance(payload, dict):
                raise RuntimeError(f'{method} {url} returned a non-object error')
            return payload


def predefined_environment_definition() -> dict[str, Any]:
    return {
        'environment_id': 'predefined-local',
        'name': 'Predefined Local',
        'allocation_type': 'predefined',
        'provider_type': 'predefined',
        'resource_limits': {'batch_max_inflight': EXPECTED_CLUSTER_SLOT_COUNT},
        'default_timeouts': {},
        'status': 'enabled',
    }


async def ensure_predefined_environment(control_url: str) -> None:
    environment_url = f'{control_url}/v1/environments/predefined-local'
    payload = predefined_environment_definition()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with session.get(environment_url) as response:
            if response.status == 200:
                current = await response.json()
                if current.get('allocation_type') != 'predefined' or current.get('provider_type') != 'predefined':
                    raise RuntimeError(f'predefined-local has incompatible configuration: {current}')
                update = {
                    field: payload[field]
                    for field in ('resource_limits', 'default_timeouts', 'status')
                    if current.get(field) != payload[field]
                }
                if update:
                    await request_json('PUT', environment_url, body=update)
            elif response.status != 404:
                raise RuntimeError(f'GET {environment_url} -> {response.status}: {await response.text()}')
            else:
                await request_json('POST', f'{control_url}/v1/environments', body=payload)
    await request_json(
        'PUT',
        f'{environment_url}/grants/local-dev',
        body={'limits': {}, 'enabled': True},
    )


async def resolved_execution_result(app_url: str, item: dict[str, Any]) -> dict[str, Any]:
    result = item.get('result_json') or {}
    if isinstance(result, dict) and result.get('_external') is True:
        return await request_json('GET', f'{app_url}/v1/executions/{item["execution_id"]}/result')
    return result


async def poll_execution(
    control_url: str,
    execution_id: str,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout: float = 120.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = await request_json('GET', f'{control_url}/v1/executions/{execution_id}')
        if predicate(last):
            return last
        if last.get('status') in {'failed', 'lost', 'expired'}:
            raise AssertionError(f'{execution_id} reached unexpected terminal state: {last}')
        await asyncio.sleep(0.5)
    raise TimeoutError(f'{execution_id} did not reach expected state; last={last}')


async def read_sse_until(url: str, event_type: str, *, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout + 5)) as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise RuntimeError(f'SSE {url} returned {response.status}: {await response.text()}')
            current_event = None
            current_data = None
            while time.monotonic() < deadline:
                line = await response.content.readline()
                if not line:
                    await asyncio.sleep(0.05)
                    continue
                text_line = line.decode().strip()
                if text_line.startswith('event:'):
                    current_event = text_line.split(':', 1)[1].strip()
                elif text_line.startswith('data:'):
                    current_data = json.loads(text_line.split(':', 1)[1].strip())
                elif text_line == '' and current_event == event_type and current_data is not None:
                    return current_data
    raise TimeoutError(f'SSE event {event_type} not received from {url}')


def fake_live_payload(
    execution_id: str,
    *,
    fake_runtime_base_url: str,
    route_status: str = 'cached',
) -> dict[str, Any]:
    return {
        'execution_id': execution_id,
        'mode': 'live',
        'spec_type': 'nexus_live',
        'instance_mode': 'asyncio',
        'route_status': route_status,
        'placement': {'environment': 'predefined-local', 'pool': 'interactive', 'labels': {'region': 'local'}},
        'spec': {
            'imports': ['nexus.extensions.tasks.identity', 'tests.integration.distributed.fixtures'],
            'runtime_provider_name': 'distributed_fake_runtime',
            'runtime_provider_config': {'base_url': fake_runtime_base_url},
            'runtime_configs': {'main': {}},
            'task_name': 'identity',
            'agent_name': 'distributed_interactive_agent',
            'agent_config': {'prompt': 'choose'},
        },
    }


def fake_live_run_request(run_id: str, *, fake_runtime_base_url: str) -> dict[str, Any]:
    return {
        'run_id': run_id,
        'imports': ['nexus.extensions.tasks.identity', 'tests.integration.distributed.fixtures'],
        'runtime_provider_name': 'distributed_fake_runtime',
        'runtime_provider_config': {'base_url': fake_runtime_base_url},
        'runtime_configs': {'main': {}},
        'task_name': 'identity',
        'agent_name': 'distributed_interactive_agent',
        'agent_config': {'prompt': 'choose'},
        'live': {'working_dir': '/workspace'},
    }


def fake_job_payload(execution_id: str, message: str = 'ok') -> dict[str, Any]:
    return {
        'execution_id': execution_id,
        'mode': 'job',
        'spec_type': 'nexus_single',
        'instance_mode': 'asyncio',
        'placement': {'environment': 'predefined-local', 'pool': 'interactive', 'labels': {'region': 'local'}},
        'spec': {
            'imports': ['tests.integration.distributed.fixtures'],
            'task_name': 'distributed_success',
            'task_config': {'message': message},
        },
    }


def typed_predefined_single_submission(run_tag: str) -> dict[str, Any]:
    return {
        'submission_version': 1,
        'idempotency_key': f'{run_tag}-typed-single',
        'environment_id': 'predefined-local',
        'pool': 'interactive',
        'code': {'mode': 'config_only'},
        'run_input': fake_job_payload(f'{run_tag}-typed-single')['spec'],
        'queue_ttl_seconds': 120,
        'execution_ttl_seconds': 300,
        'stop_timeout_seconds': 30,
    }


def typed_predefined_live_submission(run_tag: str, *, fake_runtime_base_url: str) -> dict[str, Any]:
    return {
        'submission_version': 1,
        'idempotency_key': f'{run_tag}-typed-live',
        'environment_id': 'predefined-local',
        'pool': 'interactive',
        'code': {'mode': 'config_only'},
        'run_input': fake_live_payload(
            f'{run_tag}-typed-live',
            fake_runtime_base_url=fake_runtime_base_url,
        )['spec'],
        'live': {'working_dir': '/workspace'},
        'queue_ttl_seconds': 120,
        'execution_ttl_seconds': 300,
        'stop_timeout_seconds': 30,
    }


def write_predefined_task_list(path: Path, run_tag: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open('wb') as stream:
        for index in range(PREDEFINED_BATCH_ITEM_COUNT):
            run_input = fake_job_payload(
                f'{run_tag}-typed-batch-child-{index}',
                message=f'predefined-batch-ok:{index}',
            )['spec']
            run_input['run_id'] = f'{run_tag}-typed-batch-child-{index}'
            record = {
                'version': 1,
                'item_id': f'item-{index:04d}',
                'run_input': run_input,
            }
            line = json.dumps(record, sort_keys=True, separators=(',', ':')).encode() + b'\n'
            stream.write(line)
            digest.update(line)
            size_bytes += len(line)
    return {
        'version': 1,
        'item_count': PREDEFINED_BATCH_ITEM_COUNT,
        'size_bytes': size_bytes,
        'sha256': digest.hexdigest(),
        'source': {'mode': 'predefined_config_only_e2e'},
        'datasets': [],
    }


def typed_predefined_batch_submission(
    *,
    task_list_ref: dict[str, Any],
    manifest: dict[str, Any],
    run_tag: str,
) -> dict[str, Any]:
    return {
        'submission_version': 1,
        'idempotency_key': f'{run_tag}-typed-batch',
        'environment_id': 'predefined-local',
        'pool': 'interactive',
        'code': {'mode': 'config_only'},
        'generation': {
            'mode': 'local_task_list',
            'task_list': {
                'ref': task_list_ref,
                'format': 'jsonl',
                'manifest': manifest,
            },
            'timeout_seconds': 120,
        },
        'batch': {
            'max_inflight': EXPECTED_CLUSTER_SLOT_COUNT,
            'retry_failed': 0,
            'fail_fast': False,
            'child_queue_ttl_seconds': 120,
            'timeout_per_run': 120,
        },
        'queue_ttl_seconds': 120,
        'execution_ttl_seconds': 600,
        'stop_timeout_seconds': 30,
    }


def predefined_cli_batch_args(generator: Path, *, control_url: str, run_tag: str) -> list[str]:
    return [
        'batch',
        '--generator',
        str(generator),
        '--generation',
        'local',
        '--environment',
        'predefined-local',
        '--pool',
        'interactive',
        '--config-only',
        '--control-url',
        control_url,
        '--max-inflight',
        str(EXPECTED_CLUSTER_SLOT_COUNT),
        '--generation-timeout-seconds',
        '120',
        '--child-queue-ttl-seconds',
        '120',
        '--timeout-per-run',
        '120',
        '--queue-ttl-seconds',
        '120',
        '--execution-ttl-seconds',
        '600',
        '--stop-timeout-seconds',
        '30',
        '--idempotency-key',
        f'{run_tag}-cli-batch',
        '--wait',
    ]


def write_predefined_cli_generator(path: Path, *, run_tag: str) -> None:
    path.write_text(
        '\n'.join([
            'from nexus.runner.batch import BatchItem',
            'from nexus.runner.single import RunInput',
            '',
            'def generate_configs():',
            f'    for index in range({PREDEFINED_BATCH_ITEM_COUNT}):',
            '        yield BatchItem(',
            '            config=RunInput(',
            "                imports=['tests.integration.distributed.fixtures'],",
            "                task_name='distributed_success',",
            "                task_config={'message': f'predefined-cli-ok:{index}'},",
            '            ),',
            f"            run_id=f'{run_tag}-cli-item-{{index:04d}}',",
            '        )',
            '',
        ]),
        encoding='utf-8',
    )


def _sanitize_cli_args(args: list[str]) -> list[str]:
    sanitized: list[str] = []
    replace_next = False
    for value in args:
        if replace_next:
            sanitized.append(f'<{sanitized[-1].removeprefix("--")}>')
            replace_next = False
            continue
        sanitized.append(value)
        replace_next = value in {'--control-url', '--generator'}
    return sanitized


def _run_submit_cli(args: list[str], *, timeout: float) -> dict[str, Any]:
    executable = ROOT / '.venv/bin/nexus'
    if not executable.is_file():
        raise FileNotFoundError(f'Nexus CLI is unavailable: {executable}')
    command = [str(executable), 'submit', *args, '--json']
    print(f'+ {" ".join(command)}', flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr[-2000:] or completed.stdout[-2000:]
        raise RuntimeError(f'Nexus submit CLI failed ({completed.returncode}): {detail}')
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        raise RuntimeError('Nexus submit CLI returned no JSON result')
    result = json.loads(stdout_lines[-1])
    accepted: list[dict[str, Any]] = []
    for line in completed.stderr.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get('event') == 'accepted':
            accepted.append(payload)
    return {
        'args': _sanitize_cli_args(args),
        'accepted': accepted,
        'result': result,
    }


async def assert_cli_predefined_batch(
    control_url: str,
    *,
    generator: Path,
    run_tag: str,
) -> dict[str, Any]:
    cli = await asyncio.to_thread(
        _run_submit_cli,
        predefined_cli_batch_args(generator, control_url=control_url, run_tag=run_tag),
        timeout=900,
    )
    result = cli['result']
    execution_id = result['execution_id']
    assert len(cli['accepted']) == 1, cli
    assert cli['accepted'][0]['execution_id'] == execution_id, cli
    assert result['terminal']['status'] == 'succeeded', result

    execution = await request_json('GET', f'{control_url}/v1/executions/{execution_id}')
    assert execution['status'] == 'succeeded', execution
    assert execution['workload'] == 'orchestration', execution
    summary = await request_json('GET', result['batch_url'])
    items = await list_batch_items(control_url, execution_id)
    assert summary['summary']['total'] == len(items), summary
    assert summary['summary']['succeeded'] == len(items), summary
    assert all(item['status'] == 'succeeded' for item in items), items
    batch_result = await request_json('GET', result['result_url'])
    assert batch_result['batch']['succeeded'] == len(items), batch_result
    child_execution_ids = [item['current_child_execution_id'] for item in items]
    return {
        'submission_path': 'nexus submit batch',
        'args': cli['args'],
        'execution_id': execution_id,
        'owner_principal_id': execution['owner_principal_id'],
        'status': execution['status'],
        'summary': summary['summary'],
        'child_execution_ids': child_execution_ids,
    }


def invalid_predefined_source_submission(run_tag: str) -> dict[str, Any]:
    def artifact(kind: str, suffix: str) -> dict[str, Any]:
        return {
            'artifact_id': f'art-{run_tag}-{suffix}',
            'kind': kind,
            'sha256': '0' * 64,
            'size_bytes': 1,
        }

    payload = typed_predefined_single_submission(run_tag)
    payload['idempotency_key'] = f'{run_tag}-typed-source-rejected'
    payload['code'] = {
        'mode': 'source_package',
        'source_package': artifact('source_package', 'source'),
        'nexus_wheel': artifact('nexus_wheel', 'wheel'),
        'uv_lock': artifact('uv_lock', 'lock'),
        'extras': [],
    }
    return payload


def _set_gongfeng_api_base_url(spec: dict[str, Any], api_base_url: str) -> None:
    runtime_provider_config = spec.get('runtime_provider_config')
    if isinstance(runtime_provider_config, dict):
        runtime_provider_config['api_base_url'] = api_base_url


def _set_gongfeng_runtime_image(spec: dict[str, Any], runtime_image: str) -> None:
    runtime_provider_config = spec.get('runtime_provider_config')
    if isinstance(runtime_provider_config, dict):
        runtime_provider_config['runtime_image'] = runtime_image


def resolve_run_input_environment(spec: dict[str, Any], environment_values: dict[str, str]) -> dict[str, Any]:
    previous = {key: os.environ.get(key) for key in environment_values}
    os.environ.update(environment_values)
    try:
        return RunInput.from_dict(spec).model_dump(mode='json', exclude={'compaction_manager'})
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def opus48_live_payload(
    execution_id: str,
    *,
    task_key: str,
    iteration: int,
    gongfeng_api_base_url: str = DEFAULT_OPUS48_GONGFENG_API_BASE_URL,
    runtime_image: str = 'current',
    environment_values: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = yaml.safe_load(OPUS48_PAYLOAD.read_text(encoding='utf-8'))
    payload['execution_id'] = execution_id
    payload['slot_cost'] = 1
    spec = payload['spec']
    _set_gongfeng_api_base_url(spec, gongfeng_api_base_url)
    _set_gongfeng_runtime_image(spec, runtime_image)
    spec['run_id'] = execution_id
    spec['task_config']['instance_id'] = execution_id
    spec['task_config']['prompt'] = f'写一个极简静态页面，标题为 {task_key}-{iteration}，只包含一行正文和一点 CSS。'
    spec['agent_config']['max_turns'] = 8
    spec['agent_config']['timeout'] = 600
    spec['runner']['timeout'] = 900
    if environment_values is not None:
        payload['spec'] = resolve_run_input_environment(spec, environment_values)
    return payload


def desktop_single_payload(
    execution_id: str,
    *,
    iteration: int,
    gongfeng_api_base_url: str = DEFAULT_OPUS48_GONGFENG_API_BASE_URL,
    runtime_image: str = 'current',
    environment_values: dict[str, str] | None = None,
) -> dict[str, Any]:
    marker = f'desktop-single-{iteration}'
    payload: dict[str, Any] = {
        'execution_id': execution_id,
        'mode': 'job',
        'spec_type': 'nexus_single',
        'instance_mode': 'asyncio',
        'slot_cost': 1,
        'max_attempts': 1,
        'placement': {'environment': 'predefined-local', 'pool': 'interactive', 'labels': {'region': 'local'}},
        'spec': {
            'run_id': execution_id,
            'imports': [
                'nexus.extensions',
                'nexus.extensions.openai_clients.litellm',
                'nexus.extensions.tasks.identity',
                'nexus.extensions.tools.desktop',
                'nexus.extensions.agents.main',
            ],
            'task_name': 'identity',
            'task_config': {},
            'runtime_provider_name': 'gongfeng',
            'runtime_provider_config': {
                'api_base_url': gongfeng_api_base_url,
                'image': 'mirrors.tencent.com/hunyuan_yanguan/nexus-ubuntu-session:1.4.1',
                'request_timeout': 2400,
                'resource_spec': {'cpu': 4, 'memory': 16},
                'runtime_image': runtime_image,
                'runtime_provider': 'gongfeng',
                'token': '${oc.env:GONGFENG_TOKEN}',
            },
            'runtime_configs': {
                'main': {
                    'exposed_ports': [5900, 6080],
                    'environment': {
                        'GTK_A11Y': 'none',
                        'NEXUS_GUI_REQUIRE_SYSTEM_DBUS': '1',
                        'UV_CACHE_DIR': '/tmp/nexus-uv-cache',
                        'XDG_CACHE_HOME': '/tmp/nexus-cache',
                    },
                    'use_new_cache_proxy': True,
                    'worker_venv': {'modules': []},
                }
            },
            'agent_name': 'main_agent',
            'agent_config': {
                'system_prompt': ('You are a precise desktop test agent. Use the requested tools exactly, then stop.'),
                'user_prompt': (
                    'Use tools in order: '
                    '1. call desktop/start with {"desktop_session":"gnome-session",'
                    '"session_options":{"gnome_session_name":"ubuntu"}}; '
                    '2. call desktop/screen_size; '
                    f'3. call desktop/bash with command: printf %s {marker!r} > '
                    f'/tmp/nexus_dist_desktop_job.txt && cat /tmp/nexus_dist_desktop_job.txt; '
                    f'4. reply exactly "{marker} done" and stop.'
                ),
                'max_iterations': 8,
                'tools': ['desktop/start', 'desktop/screen_size', 'desktop/bash'],
                'parallel_tool_calls': False,
            },
            'openai_clients': {
                'main': {
                    'name': 'litellm',
                    'config': {
                        'client_args': {
                            'api_key': (
                                '${oc.env:MODEL_SERVICE_API_APP_ID}:${oc.env:MODEL_SERVICE_API_APP_KEY}'
                                '?provider=aws_third&model=${oc.env:NEXUS_MODEL,"anthropic.claude-opus-4-8"}'
                                '&timeout=${oc.env:MODEL_SERVICE_API_TIMEOUT,"60"}'
                            ),
                            'base_url': '${oc.env:MODEL_SERVICE_API_BASE_URL}',
                        },
                        'request_args': {'model': '${oc.env:NEXUS_MODEL,"anthropic.claude-opus-4-8"}'},
                    },
                }
            },
            'processors': {
                'chat': {'name': 'chat_tracker', 'args': {'tools_mode': 'branch_last'}},
                'agent_metrics': {'name': 'main_agent_metrics', 'args': {}},
            },
            'runner': {'timeout': 1200, 'hold_after_task': None, 'early_output_path': None, 'cache': None},
        },
    }
    if environment_values is not None:
        payload['spec'] = resolve_run_input_environment(payload['spec'], environment_values)
    return payload


async def assert_inventory(control_url: str, release_name: str) -> None:
    print('checking predefined environment inventory', flush=True)
    environments = await request_json('GET', f'{control_url}/v1/environments')
    names = {item['environment_id'] for item in environments['items']}
    assert 'predefined-local' in names, environments

    stats = await request_json('GET', f'{control_url}/v1/environments/predefined-local/stats?pool_id=interactive')
    assert stats['manager_count'] == 2, stats
    assert stats['healthy_manager_count'] == 2, stats
    assert stats['capacity_total'] == EXPECTED_CLUSTER_SLOT_COUNT, stats

    orchestration_stats = await request_json(
        'GET',
        f'{control_url}/v1/environments/predefined-local/stats?pool_id=orchestration',
    )
    assert orchestration_stats['manager_count'] == 1, orchestration_stats
    assert orchestration_stats['healthy_manager_count'] == 1, orchestration_stats
    assert orchestration_stats['capacity_total'] == 1, orchestration_stats

    managers = await request_json('GET', f'{control_url}/v1/managers?environment=predefined-local')
    manager_ids = {item['manager_id'] for item in managers['items']}
    assert manager_ids == {
        f'{release_name}-instance-manager-0',
        f'{release_name}-instance-manager-1',
        f'{release_name}-batch-manager-0',
    }, managers
    batch_manager = next(item for item in managers['items'] if item['manager_id'].endswith('-batch-manager-0'))
    assert batch_manager['pool_id'] == 'orchestration', batch_manager
    assert batch_manager['workload'] == 'orchestration', batch_manager


async def wait_allocations_released(control_url: str, execution_id: str, *, timeout: float = 180.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    latest: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        latest = await request_json('GET', f'{control_url}/v1/executions/{execution_id}/allocations')
        if latest['items'] and all(item['status'] == 'released' for item in latest['items']):
            return
        await asyncio.sleep(0.5)
    raise TimeoutError(f'allocations were not released: {execution_id}: {latest}')


async def list_batch_items(control_url: str, execution_id: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        suffix = f'&cursor={urllib.parse.quote(cursor)}' if cursor else ''
        page = await request_json(
            'GET',
            f'{control_url}/v1/executions/{execution_id}/batch/items?limit=100{suffix}',
        )
        items.extend(page['items'])
        cursor = page.get('next_cursor')
        if not cursor:
            return items


def assert_execution_manager(item: dict[str, Any], *, release_name: str) -> None:
    assignment = item.get('current_assignment') or {}
    manager_id = assignment.get('manager_id')
    assert item['workload'] == 'execution', item
    assert item['pool_id'] == 'interactive', item
    assert isinstance(manager_id, str), item
    assert manager_id.startswith(f'{release_name}-instance-manager-'), item


async def assert_typed_predefined_submissions(
    control_url: str,
    app_url: str,
    *,
    release_name: str,
    run_tag: str,
) -> dict[str, Any]:
    print('checking typed config-only predefined single, live, and batch', flush=True)
    rejected = await request_json_error(
        'POST',
        f'{control_url}/v1/submissions/single',
        body=invalid_predefined_source_submission(run_tag),
        expected_statuses={400, 422},
    )
    assert 'config_only' in json.dumps(rejected), rejected

    single_payload = typed_predefined_single_submission(run_tag)
    single_submission = await request_json(
        'POST',
        f'{control_url}/v1/submissions/single',
        body=single_payload,
    )
    single = await poll_execution(
        control_url,
        single_submission['execution_id'],
        lambda item: item['status'] == 'succeeded',
    )
    assert_execution_manager(single, release_name=release_name)
    single_result = await resolved_execution_result(app_url, single)
    assert single_result['task_output']['message'] == 'ok', single_result
    await wait_allocations_released(control_url, single['execution_id'])

    fake_runtime_base_url = f'http://{release_name}-fake-runtime:8080'
    live_payload = typed_predefined_live_submission(
        run_tag,
        fake_runtime_base_url=fake_runtime_base_url,
    )
    live_submission = await request_json(
        'POST',
        f'{control_url}/v1/submissions/live',
        body=live_payload,
    )
    live = await poll_execution(
        control_url,
        live_submission['execution_id'],
        lambda item: item['status'] == 'waiting_input',
    )
    assert_execution_manager(live, release_name=release_name)
    event = await read_sse_until(
        f'{app_url}/v1/executions/{live["execution_id"]}/live/events?cursor=0',
        'input.required',
    )
    assert event['type'] == 'input.required', event
    await request_json(
        'POST',
        f'{app_url}/v1/executions/{live["execution_id"]}/commands',
        body={
            'type': 'input.submit',
            'payload': {'text': 'predefined-live-ok'},
            'idempotency_key': f'{live["execution_id"]}-input',
        },
    )
    live = await poll_execution(
        control_url,
        live['execution_id'],
        lambda item: item['status'] == 'succeeded',
    )
    live_result = await resolved_execution_result(app_url, live)
    assert live_result['agent_output']['choice'] == 'predefined-live-ok', live_result
    await wait_allocations_released(control_url, live['execution_id'])

    with tempfile.TemporaryDirectory(prefix='nexus-predefined-task-list-') as directory:
        task_list_path = Path(directory) / 'tasks.jsonl'
        manifest = write_predefined_task_list(task_list_path, run_tag)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
            task_list_ref = await _upload(
                session,
                control_url,
                task_list_path,
                kind='task_list',
                sha256=manifest['sha256'],
                size_bytes=manifest['size_bytes'],
                idempotency_key=f'{run_tag}-typed-task-list-{uuid.uuid4().hex}',
                environment_id='predefined-local',
            )
    batch_payload = typed_predefined_batch_submission(
        task_list_ref=task_list_ref,
        manifest=manifest,
        run_tag=run_tag,
    )
    batch_submission = await request_json(
        'POST',
        f'{control_url}/v1/submissions/batch',
        body=batch_payload,
    )
    batch = await poll_execution(
        control_url,
        batch_submission['execution_id'],
        lambda item: item.get('current_assignment') is not None,
    )
    batch_manager_id = batch['current_assignment']['manager_id']
    assert batch['workload'] == 'orchestration', batch
    assert batch['pool_id'] == 'orchestration', batch
    assert batch_manager_id == f'{release_name}-batch-manager-0', batch
    batch = await poll_execution(
        control_url,
        batch['execution_id'],
        lambda item: item['status'] == 'succeeded',
        timeout=300,
    )
    summary = await request_json('GET', f'{control_url}/v1/executions/{batch["execution_id"]}/batch')
    assert summary['summary']['total'] == PREDEFINED_BATCH_ITEM_COUNT, summary
    assert summary['summary']['succeeded'] == PREDEFINED_BATCH_ITEM_COUNT, summary
    items = await list_batch_items(control_url, batch['execution_id'])
    assert len(items) == PREDEFINED_BATCH_ITEM_COUNT, items
    assert all(item['status'] == 'succeeded' for item in items), items
    child_results: list[str] = []
    child_execution_ids: list[str] = []
    child_manager_ids: set[str] = set()
    for index, item in enumerate(items):
        child_execution_id = item['current_child_execution_id']
        child_execution_ids.append(child_execution_id)
        child = await request_json(
            'GET',
            f'{control_url}/v1/executions/{child_execution_id}',
        )
        assert_execution_manager(child, release_name=release_name)
        child_manager_ids.add(child['current_assignment']['manager_id'])
        child_result = await resolved_execution_result(app_url, child)
        child_results.append(child_result['task_output']['message'])
        assert child_result['task_output']['message'] == f'predefined-batch-ok:{index}', child_result
    batch_result = await resolved_execution_result(app_url, batch)
    assert batch_result['batch']['total'] == PREDEFINED_BATCH_ITEM_COUNT, batch_result
    assert batch_result['batch']['succeeded'] == PREDEFINED_BATCH_ITEM_COUNT, batch_result
    await wait_allocations_released(control_url, batch['execution_id'])

    interactive_stats = await request_json(
        'GET',
        f'{control_url}/v1/environments/predefined-local/stats?pool_id=interactive',
    )
    orchestration_stats = await request_json(
        'GET',
        f'{control_url}/v1/environments/predefined-local/stats?pool_id=orchestration',
    )
    assert interactive_stats['reserved_slots'] == 0, interactive_stats
    assert orchestration_stats['reserved_slots'] == 0, orchestration_stats
    return {
        'source_package_rejected': True,
        'single': {
            'submission': single_payload,
            'execution_id': single['execution_id'],
            'status': single['status'],
            'manager_id': single['current_assignment']['manager_id'],
            'result': single_result,
        },
        'live': {
            'submission': live_payload,
            'execution_id': live['execution_id'],
            'status': live['status'],
            'manager_id': live['current_assignment']['manager_id'],
            'sse_event': event['type'],
            'result': live_result,
        },
        'batch': {
            'submission': batch_payload,
            'execution_id': batch['execution_id'],
            'owner_principal_id': batch['owner_principal_id'],
            'status': batch['status'],
            'manager_id': batch_manager_id,
            'child_execution_ids': child_execution_ids,
            'child_manager_ids': sorted(child_manager_ids),
            'task_list_ref': task_list_ref,
            'summary': summary['summary'],
            'child_results': child_results,
        },
    }


async def assert_fake_live_interaction(
    control_url: str,
    app_url: str,
    run_tag: str,
    *,
    fake_runtime_base_url: str,
) -> None:
    print('checking fake live SSE, input command, runtime proxy, and completion', flush=True)
    execution_id = f'{run_tag}-fake-live'
    await request_json(
        'POST',
        f'{control_url}/v1/executions',
        body=fake_live_payload(execution_id, fake_runtime_base_url=fake_runtime_base_url),
    )
    await poll_execution(control_url, execution_id, lambda item: item['status'] == 'waiting_input')
    await request_json('POST', f'{control_url}/v1/executions/{execution_id}/activate')

    event = await read_sse_until(f'{app_url}/v1/executions/{execution_id}/live/events?cursor=0', 'input.required')
    assert event['type'] == 'input.required', event

    proxied = await request_json('GET', f'{app_url}/v1/executions/{execution_id}/runtimes/main/proxy/ping')
    assert proxied == {'pong': True, 'source': 'k8s-fake-runtime'}, proxied

    command = await request_json(
        'POST',
        f'{app_url}/v1/executions/{execution_id}/commands',
        body={'type': 'input.submit', 'payload': {'text': 'Alpha'}, 'idempotency_key': f'{execution_id}-input'},
    )
    assert command['assignment_id'], command
    completed = await poll_execution(control_url, execution_id, lambda item: item['status'] == 'succeeded')
    result = await resolved_execution_result(app_url, completed)
    assert completed['result_json'] == result
    assert result['success'] is True, completed

    # Once the assignment is closed this endpoint must replay the live SSE projection from the raw trace source.
    replayed = await read_sse_until(
        f'{app_url}/v1/executions/{execution_id}/live/events?cursor=0',
        'run.output',
    )
    assert replayed['type'] == 'run.output', replayed
    assert replayed['data']['output']['success'] is True, replayed


async def assert_legacy_live_compatibility(
    control_url: str,
    app_url: str,
    compatibility_url: str,
    run_tag: str,
    *,
    fake_runtime_base_url: str,
) -> None:
    print('checking application /compat/live compatibility API', flush=True)
    requested_run_id = f'{run_tag}-compat-live'
    created = await request_json(
        'POST',
        f'{compatibility_url}/runs',
        body=fake_live_run_request(requested_run_id, fake_runtime_base_url=fake_runtime_base_url),
    )
    run_id = created['run_id']
    assert isinstance(run_id, str), created
    assert run_id, created
    assert created['status'] in {'starting', 'running'}, created

    await poll_execution(control_url, run_id, lambda item: item['status'] == 'waiting_input')
    run_info = await request_json('GET', f'{compatibility_url}/runs/{run_id}')
    assert run_info['status'] == 'idle', run_info
    assert run_info['runtimes']['main']['proxy_url'].startswith(f'/compat/live/v1/runs/{run_id}/runtimes/main/proxy'), (
        run_info
    )

    config = await request_json('GET', f'{compatibility_url}/runs/{run_id}/config')
    assert config['request']['task_name'] == 'identity', config
    assert config['request']['run_id'] == requested_run_id, config

    event = await read_sse_until(f'{compatibility_url}/runs/{run_id}/events?cursor=0', 'input.required')
    assert event['type'] == 'input.required', event

    proxied = await request_json('GET', f'{compatibility_url}/runs/{run_id}/runtimes/main/proxy/ping')
    assert proxied == {'pong': True, 'source': 'k8s-fake-runtime'}, proxied

    turn = await request_json('POST', f'{compatibility_url}/runs/{run_id}/turns', body={'text': 'Beta'})
    assert turn['run_id'] == run_id, turn
    assert turn['turn_id'], turn
    completed = await poll_execution(control_url, run_id, lambda item: item['status'] == 'succeeded')
    result = await resolved_execution_result(app_url, completed)
    assert result['success'] is True, completed

    summary = await request_json('GET', f'{compatibility_url}/runs?summary=1')
    assert any(item['run_id'] == run_id for item in summary['runs']), summary


async def assert_capacity_and_reschedule(
    control_url: str,
    app_url: str,
    run_tag: str,
    *,
    fake_runtime_base_url: str,
) -> None:
    print('checking capacity reservation and reschedule after slot release', flush=True)
    live_ids = [f'{run_tag}-capacity-live-{index}' for index in range(EXPECTED_CLUSTER_SLOT_COUNT)]
    for execution_id in live_ids:
        await request_json(
            'POST',
            f'{control_url}/v1/executions',
            body=fake_live_payload(execution_id, fake_runtime_base_url=fake_runtime_base_url),
        )
    for execution_id in live_ids:
        await poll_execution(control_url, execution_id, lambda item: item['status'] == 'waiting_input')

    stats = await request_json('GET', f'{control_url}/v1/environments/predefined-local/stats?pool_id=interactive')
    assert stats['reserved_slots'] == EXPECTED_CLUSTER_SLOT_COUNT, stats
    assert stats['available_slots'] == 0, stats

    overflow_id = f'{run_tag}-overflow-job'
    await request_json('POST', f'{control_url}/v1/executions', body=fake_job_payload(overflow_id, message='after-free'))
    await asyncio.sleep(5)
    overflow = await request_json('GET', f'{control_url}/v1/executions/{overflow_id}')
    assert overflow['status'] == 'queued', overflow
    assert overflow['current_assignment_id'] is None, overflow

    await request_json('POST', f'{control_url}/v1/executions/{live_ids[0]}/stop')
    await poll_execution(control_url, live_ids[0], lambda item: item['status'] == 'killed')
    finished = await poll_execution(control_url, overflow_id, lambda item: item['status'] == 'succeeded')
    result = await resolved_execution_result(app_url, finished)
    assert result['task_output']['message'] == 'after-free', finished

    for execution_id in live_ids[1:]:
        await request_json('POST', f'{control_url}/v1/executions/{execution_id}/stop')
    for execution_id in live_ids[1:]:
        await poll_execution(control_url, execution_id, lambda item: item['status'] == 'killed')


def _live_request(item: dict[str, Any]) -> dict[str, Any]:
    assignment = item.get('current_assignment') or {}
    runtime_info = assignment.get('runtime_info') or {}
    input_info = runtime_info.get('input') or {}
    request = input_info.get('request') or {}
    return request if isinstance(request, dict) else {}


def _live_turn_id(item: dict[str, Any]) -> str | None:
    turn_id = _live_request(item).get('turn_id')
    return turn_id if isinstance(turn_id, str) else None


def _live_response_text(item: dict[str, Any]) -> str:
    request = _live_request(item)
    message = request.get('message')
    if isinstance(message, dict) and isinstance(message.get('result'), str):
        return message['result']
    hint = request.get('hint')
    return hint if isinstance(hint, str) else ''


def ready_live_cases(
    cases: list[RepeatedClusterCase],
    snapshots: dict[str, dict[str, Any]],
    *,
    completed_execution_ids: set[str],
) -> list[RepeatedClusterCase]:
    return [
        case
        for case in cases
        if case.mode == 'live'
        and case.execution_id not in completed_execution_ids
        and snapshots.get(case.execution_id, {}).get('status') == 'waiting_input'
    ]


async def _wait_for_queue_pressure(control_url: str, cases: list[RepeatedClusterCase], timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    last_snapshot: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        last_snapshot = [
            await request_json('GET', f'{control_url}/v1/executions/{case.execution_id}') for case in cases
        ]
        queued = [
            item for item in last_snapshot if item['status'] == 'queued' and item['current_assignment_id'] is None
        ]
        reserved = [
            item
            for item in last_snapshot
            if item['status'] in {'assigned', 'starting', 'running', 'waiting_input'}
            and item['current_assignment_id'] is not None
        ]
        stats = await request_json(
            'GET',
            f'{control_url}/v1/environments/predefined-local/stats?pool_id=interactive',
        )
        if queued and len(reserved) >= EXPECTED_CLUSTER_SLOT_COUNT and stats['available_slots'] == 0:
            return
        await asyncio.sleep(1)
    raise TimeoutError(f'queue pressure was not observed; last={last_snapshot}')


async def _wait_for_live_response(
    control_url: str,
    app_url: str,
    case: RepeatedClusterCase,
    *,
    timeout: float,
    check_runtime_aiomonitor: bool,
) -> None:
    first_waiting = await poll_execution(
        control_url,
        case.execution_id,
        lambda item: item['status'] == 'waiting_input',
        timeout=timeout,
    )
    if check_runtime_aiomonitor:
        await assert_runtime_aiomonitor_webui(app_url, case.execution_id)
    if case.kill:
        await request_json('POST', f'{control_url}/v1/executions/{case.execution_id}/stop')
        stopped = await poll_execution(
            control_url,
            case.execution_id,
            lambda item: item['status'] == 'killed' and item['route_status'] == 'closed',
            timeout=120,
        )
        assert stopped['route_status'] == 'closed', stopped
        return

    first_turn_id = _live_turn_id(first_waiting)
    first_response_text = _live_response_text(first_waiting)
    await request_json(
        'POST',
        f'{app_url}/v1/executions/{case.execution_id}/commands',
        body={
            'type': 'input.submit',
            'payload': {'text': '已经验证，可以结束。'},
            'idempotency_key': f'{case.execution_id}-finish',
        },
    )

    deadline = time.monotonic() + timeout
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = await request_json('GET', f'{control_url}/v1/executions/{case.execution_id}')
        if last['status'] in TERMINAL_STATUSES:
            if last['status'] != 'succeeded':
                raise RuntimeError(f'{case.execution_id} ended with {last["status"]}: {last.get("error_json")}')
            return
        if last['status'] == 'waiting_input':
            response_text = _live_response_text(last)
            if response_text and (
                _live_turn_id(last) != first_turn_id or response_text.strip() != first_response_text.strip()
            ):
                await request_json('POST', f'{control_url}/v1/executions/{case.execution_id}/stop')
                stopped = await poll_execution(
                    control_url,
                    case.execution_id,
                    lambda item: item['status'] in {'killed', 'succeeded'} and item['route_status'] == 'closed',
                    timeout=120,
                )
                assert stopped['route_status'] == 'closed', stopped
                return
        await asyncio.sleep(5)
    raise TimeoutError(f'{case.execution_id} did not produce a live response within {timeout}s; last={last}')


async def _wait_for_repeated_cluster_done(
    control_url: str,
    app_url: str,
    cases: list[RepeatedClusterCase],
    *,
    timeout: float,
    check_runtime_aiomonitor: bool,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    completed_execution_ids: set[str] = set()
    snapshots: dict[str, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        snapshots = {
            case.execution_id: await request_json('GET', f'{control_url}/v1/executions/{case.execution_id}')
            for case in cases
        }

        for case in cases:
            if case.execution_id in completed_execution_ids or case.mode != 'job':
                continue
            item = snapshots[case.execution_id]
            if item['status'] in TERMINAL_STATUSES:
                assert item['status'] == 'succeeded', item
                result = await resolved_execution_result(app_url, item)
                assert item['result_json'] == result
                assert result['success'] is True, item
                completed_execution_ids.add(case.execution_id)

        for case in cases:
            if case.execution_id in completed_execution_ids or case.mode != 'live':
                continue
            item = snapshots[case.execution_id]
            if item['status'] in TERMINAL_STATUSES:
                if item['status'] != 'succeeded':
                    raise RuntimeError(f'{case.execution_id} ended before handling: {item}')
                completed_execution_ids.add(case.execution_id)

        ready = ready_live_cases(cases, snapshots, completed_execution_ids=completed_execution_ids)
        if ready:
            remaining = max(1.0, deadline - time.monotonic())
            case = ready[0]
            await _wait_for_live_response(
                control_url,
                app_url,
                case,
                timeout=remaining,
                check_runtime_aiomonitor=check_runtime_aiomonitor,
            )
            completed_execution_ids.add(case.execution_id)
            continue

        if len(completed_execution_ids) == len(cases):
            return [snapshots[case.execution_id] for case in cases]
        await asyncio.sleep(2)

    raise TimeoutError(f'repeated Opus 4.8 cluster cases did not finish within {timeout}s; last={snapshots}')


async def assert_repeated_opus48_cluster(
    control_url: str,
    app_url: str,
    run_tag: str,
    timeout: float,
    gongfeng_api_base_url: str,
    runtime_image: str,
    check_runtime_aiomonitor: bool,
    environment_values: dict[str, str],
) -> None:
    print('checking repeated Opus 4.8 live/live/desktop executions with slot=2 and one mid-run kill', flush=True)
    cases = build_repeated_cluster_cases(run_tag)
    for case in cases:
        payload = (
            desktop_single_payload(
                case.execution_id,
                iteration=case.iteration,
                gongfeng_api_base_url=gongfeng_api_base_url,
                runtime_image=runtime_image,
                environment_values=environment_values,
            )
            if case.mode == 'job'
            else opus48_live_payload(
                case.execution_id,
                task_key=case.task_key,
                iteration=case.iteration,
                gongfeng_api_base_url=gongfeng_api_base_url,
                runtime_image=runtime_image,
                environment_values=environment_values,
            )
        )
        await request_json('POST', f'{control_url}/v1/executions', body=payload)
        if case.mode == 'live':
            await request_json('POST', f'{control_url}/v1/executions/{case.execution_id}/activate')

    await _wait_for_queue_pressure(control_url, cases)
    final = await _wait_for_repeated_cluster_done(
        control_url,
        app_url,
        cases,
        timeout=timeout,
        check_runtime_aiomonitor=check_runtime_aiomonitor,
    )
    assert sum(1 for case in cases if case.kill) == 1, cases
    assert all(item['status'] in TERMINAL_STATUSES for item in final), final
    assert all(item['status'] == 'succeeded' for case, item in zip(cases, final, strict=True) if case.mode == 'job'), (
        final
    )


async def assert_opus48_live(
    control_url: str,
    app_url: str,
    run_tag: str,
    timeout: float,
    gongfeng_api_base_url: str,
    runtime_image: str,
    check_runtime_aiomonitor: bool,
) -> None:
    print('checking Opus 4.8 Gongfeng live execution', flush=True)
    payload = yaml.safe_load(OPUS48_PAYLOAD.read_text(encoding='utf-8'))
    execution_id = f'{run_tag}-opus48-live'
    payload['execution_id'] = execution_id
    _set_gongfeng_api_base_url(payload['spec'], gongfeng_api_base_url)
    _set_gongfeng_runtime_image(payload['spec'], runtime_image)
    payload['spec']['task_config']['instance_id'] = execution_id

    await request_json('POST', f'{control_url}/v1/executions', body=payload)
    await request_json('POST', f'{control_url}/v1/executions/{execution_id}/activate')

    deadline = time.monotonic() + timeout
    terminal = {'succeeded', 'failed', 'killed', 'lost', 'expired'}
    last: dict[str, Any] | None = None

    def live_request(item: dict[str, Any]) -> dict[str, Any]:
        assignment = item.get('current_assignment') or {}
        runtime_info = assignment.get('runtime_info') or {}
        input_info = runtime_info.get('input') or {}
        return input_info.get('request') or {}

    def live_turn_id(item: dict[str, Any]) -> str | None:
        request = live_request(item)
        turn_id = request.get('turn_id')
        return turn_id if isinstance(turn_id, str) else None

    def live_response_text(item: dict[str, Any]) -> str:
        request = live_request(item)
        message = request.get('message')
        if isinstance(message, dict) and isinstance(message.get('result'), str):
            return message['result']
        hint = request.get('hint')
        return hint if isinstance(hint, str) else ''

    first_waiting = await poll_execution(
        control_url,
        execution_id,
        lambda item: item['status'] == 'waiting_input',
        timeout=timeout,
    )
    if check_runtime_aiomonitor:
        await assert_runtime_aiomonitor_webui(app_url, execution_id)
    first_turn_id = live_turn_id(first_waiting)
    first_response_text = live_response_text(first_waiting)
    await request_json(
        'POST',
        f'{app_url}/v1/executions/{execution_id}/commands',
        body={
            'type': 'input.submit',
            'payload': {'text': '不需要继续扩展，请直接结束任务。'},
            'idempotency_key': f'{execution_id}-finish',
        },
    )

    while time.monotonic() < deadline:
        last = await request_json('GET', f'{control_url}/v1/executions/{execution_id}')
        if last['status'] in terminal:
            if last['status'] != 'succeeded':
                raise RuntimeError(f'Opus 4.8 live ended with {last["status"]}: {last.get("error_json")}')
            return
        if last['status'] == 'waiting_input':
            response_text = live_response_text(last)
            if response_text and (
                live_turn_id(last) != first_turn_id or response_text.strip() != first_response_text.strip()
            ):
                assert response_text.strip(), response_text
                await request_json('POST', f'{control_url}/v1/executions/{execution_id}/stop')
                stopped = await poll_execution(
                    control_url,
                    execution_id,
                    lambda item: item['status'] in {'killed', 'succeeded'} and item['route_status'] == 'closed',
                )
                assert stopped['route_status'] == 'closed', stopped
                return
        await asyncio.sleep(5)
    raise TimeoutError(f'Opus 4.8 live did not finish within {timeout}s; last={last}')


def compatibility_health_url(compatibility_url: str) -> str:
    compatibility_url = compatibility_url.rstrip('/')
    if compatibility_url.endswith('/v1'):
        return f'{compatibility_url[:-3]}/health'
    return f'{compatibility_url}/health'


async def assert_tdsql_batch_trace(
    config: TDSQLTraceConfig,
    *,
    owner_principal_id: str,
    parent_execution_id: str,
    child_execution_ids: list[str],
) -> dict[str, Any]:
    principal_scope_id = principal_trace_scope_id(owner_principal_id)
    table_name = principal_batch_trace_table_name(principal_scope_id)
    expected_run_ids = set(child_execution_ids)
    if not expected_run_ids:
        raise AssertionError('TDSQL batch trace assertion requires child executions')

    connection = await mysql_connect(
        **build_db_config(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            db=config.db,
        )
    )
    parent_rows_removed = False
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                'SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s',
                (config.db, table_name),
            )
            table_count = int((await cursor.fetchone())[0])
            assert table_count == 1, f'TDSQL batch trace table is missing: {table_name}'

            await cursor.execute(
                'SELECT COLUMN_NAME FROM information_schema.COLUMNS '
                'WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s ORDER BY ORDINAL_POSITION',
                (config.db, table_name),
            )
            columns = [str(row[0]) for row in await cursor.fetchall()]
            assert 'trajectory_group_id' not in columns, columns
            assert 'parent_execution_id' in columns, columns

            await cursor.execute(f'SHOW CREATE TABLE {table_name}')
            create_statement = str((await cursor.fetchone())[1])
            normalized_create = re.sub(r'[\s`]+', '', create_statement.lower())
            assert 'shardkey=run_id' in normalized_create, create_statement

            await cursor.execute(
                f'SELECT COUNT(*), COUNT(DISTINCT run_id) FROM {table_name} WHERE parent_execution_id = %s',
                (parent_execution_id,),
            )
            row_count, distinct_run_count = (int(value) for value in await cursor.fetchone())
            assert row_count > 0, table_name
            assert distinct_run_count == len(expected_run_ids), (distinct_run_count, expected_run_ids)

            await cursor.execute(
                f'SELECT DISTINCT run_id FROM {table_name} WHERE parent_execution_id = %s',
                (parent_execution_id,),
            )
            actual_run_ids = {str(row[0]) for row in await cursor.fetchall()}
            assert actual_run_ids == expected_run_ids, (actual_run_ids, expected_run_ids)

            placeholders = ', '.join(['%s'] * len(child_execution_ids))
            await cursor.execute(
                f'SELECT COUNT(*) FROM {SHARED_TRACE_TABLE} WHERE run_id IN ({placeholders})',
                tuple(child_execution_ids),
            )
            shared_table_rows = int((await cursor.fetchone())[0])
            assert shared_table_rows == 0, 'TDSQL batch child traces leaked into the shared table'

    finally:
        try:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    f'DELETE FROM {table_name} WHERE parent_execution_id = %s',
                    (parent_execution_id,),
                )
            await connection.commit()
            parent_rows_removed = True
        finally:
            connection.close()

    return {
        'table_name': table_name,
        'row_count': row_count,
        'distinct_child_runs': distinct_run_count,
        'expected_child_runs': len(expected_run_ids),
        'has_trajectory_group_id': False,
        'has_parent_execution_id': True,
        'shard_key': 'run_id',
        'shared_table_rows': shared_table_rows,
        'parent_rows_removed': parent_rows_removed,
        'table_preserved': True,
        'principal_scope_id': principal_scope_id,
        'target_fingerprint': config.target_fingerprint(),
    }


def _tdsql_batch_assertion_script(
    config: TDSQLTraceConfig,
    *,
    owner_principal_id: str,
    parent_execution_id: str,
    child_execution_ids: list[str],
) -> str:
    payload = json.dumps(
        {
            'config': {
                'host': config.host,
                'port': config.port,
                'user': config.user,
                'password': config.password,
                'db': config.db,
            },
            'owner_principal_id': owner_principal_id,
            'parent_execution_id': parent_execution_id,
            'child_execution_ids': child_execution_ids,
        },
        separators=(',', ':'),
    )
    return f"""import asyncio
import json
import re

from nexus.runner.trace.tdsql._driver import connect
from nexus.runner.trace.tdsql.config import build_db_config
from nexus.runner.trace import principal_trace_scope_id
from nexus.runner.trace.tdsql.schema import SHARED_TRACE_TABLE, principal_batch_trace_table_name


payload = json.loads({json.dumps(payload)})


async def verify():
    config = payload['config']
    owner_principal_id = payload['owner_principal_id']
    parent_execution_id = payload['parent_execution_id']
    child_execution_ids = payload['child_execution_ids']
    expected_run_ids = set(child_execution_ids)
    principal_scope_id = principal_trace_scope_id(owner_principal_id)
    table_name = principal_batch_trace_table_name(principal_scope_id)
    connection = await connect(**build_db_config(**config))
    parent_rows_removed = False
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                'SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s',
                (config['db'], table_name),
            )
            table_count = int((await cursor.fetchone())[0])
            assert table_count == 1, f'TDSQL batch trace table is missing: {{table_name}}'

            await cursor.execute(
                'SELECT COLUMN_NAME FROM information_schema.COLUMNS '
                'WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s ORDER BY ORDINAL_POSITION',
                (config['db'], table_name),
            )
            columns = [str(row[0]) for row in await cursor.fetchall()]
            assert 'trajectory_group_id' not in columns, columns
            assert 'parent_execution_id' in columns, columns

            await cursor.execute(f'SHOW CREATE TABLE `{{table_name}}`')
            create_statement = str((await cursor.fetchone())[1])
            normalized_create = re.sub(r'[\\s`]+', '', create_statement.lower())
            assert 'shardkey=run_id' in normalized_create, create_statement

            await cursor.execute(
                f'SELECT COUNT(*), COUNT(DISTINCT run_id) FROM `{{table_name}}` '
                'WHERE parent_execution_id = %s',
                (parent_execution_id,),
            )
            row_count, distinct_run_count = (int(value) for value in await cursor.fetchone())
            assert row_count > 0, table_name
            assert distinct_run_count == len(expected_run_ids), (distinct_run_count, expected_run_ids)

            await cursor.execute(
                f'SELECT DISTINCT run_id FROM `{{table_name}}` WHERE parent_execution_id = %s',
                (parent_execution_id,),
            )
            actual_run_ids = {{str(row[0]) for row in await cursor.fetchall()}}
            assert actual_run_ids == expected_run_ids, (actual_run_ids, expected_run_ids)

            placeholders = ', '.join(['%s'] * len(child_execution_ids))
            await cursor.execute(
                f'SELECT COUNT(*) FROM {{SHARED_TRACE_TABLE}} WHERE run_id IN ({{placeholders}})',
                tuple(child_execution_ids),
            )
            shared_table_rows = int((await cursor.fetchone())[0])
            assert shared_table_rows == 0, 'TDSQL batch child traces leaked into the shared table'

    finally:
        try:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    f'DELETE FROM `{{table_name}}` WHERE parent_execution_id = %s',
                    (parent_execution_id,),
                )
            await connection.commit()
            parent_rows_removed = True
        finally:
            connection.close()

    report = {{
        'table_name': table_name,
        'row_count': row_count,
        'distinct_child_runs': distinct_run_count,
        'expected_child_runs': len(expected_run_ids),
        'has_trajectory_group_id': False,
        'has_parent_execution_id': True,
        'shard_key': 'run_id',
        'shared_table_rows': shared_table_rows,
        'parent_rows_removed': parent_rows_removed,
        'table_preserved': True,
        'principal_scope_id': principal_scope_id,
    }}
    print('NEXUS_TDSQL_ASSERTION=' + json.dumps(report, sort_keys=True))


asyncio.run(verify())
"""


async def assert_tdsql_batch_trace_in_cluster(
    config: TDSQLTraceConfig,
    *,
    kubectl: str,
    namespace: str,
    release_name: str,
    owner_principal_id: str,
    parent_execution_id: str,
    child_execution_ids: list[str],
) -> dict[str, Any]:
    resource = ready_pod_resource(kubectl, namespace, 'batch-manager', release_name=release_name)
    command = [
        kubectl,
        '-n',
        namespace,
        'exec',
        '-i',
        resource,
        '--',
        '/opt/nexus-venv/bin/python',
        '-c',
        "exec(__import__('sys').stdin.read())",
    ]
    completed = await asyncio.to_thread(
        run,
        command,
        input_text=_tdsql_batch_assertion_script(
            config,
            owner_principal_id=owner_principal_id,
            parent_execution_id=parent_execution_id,
            child_execution_ids=child_execution_ids,
        ),
        capture=True,
        timeout=90,
    )
    marker = 'NEXUS_TDSQL_ASSERTION='
    for line in reversed((completed.stdout or '').splitlines()):
        if line.startswith(marker):
            report = json.loads(line.removeprefix(marker))
            if isinstance(report, dict):
                report['target_fingerprint'] = config.target_fingerprint()
                return report
            break
    raise AssertionError(f'TDSQL batch assertion returned no report: {(completed.stdout or "")[-1000:]}')


def _tdsql_readonly_credential_script(
    credential: dict[str, Any],
    *,
    parent_execution_id: str,
    rejected_passwords: list[str],
) -> str:
    payload = json.dumps(
        {
            'host': credential['read_host'],
            'port': credential['read_port'],
            'db': credential['database'],
            'user': credential['username'],
            'password': credential['password'],
            'table_name': credential['batch_table_name'],
            'parent_execution_id': parent_execution_id,
            'rejected_passwords': rejected_passwords,
        },
        separators=(',', ':'),
    )
    return f"""import asyncio
import json

from nexus.runner.trace.tdsql._driver import connect
from nexus.runner.trace.tdsql.config import build_db_config


payload = json.loads({json.dumps(payload)})


async def open_connection(password):
    return await connect(**build_db_config(
        host=payload['host'],
        port=payload['port'],
        user=payload['user'],
        password=password,
        db=payload['db'],
    ))


async def verify():
    rejected = 0
    for old_password in payload['rejected_passwords']:
        try:
            stale = await open_connection(old_password)
        except Exception:
            rejected += 1
        else:
            stale.close()
            raise AssertionError('rotated trace database password still works')

    connection = await open_connection(payload['password'])
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                f"SELECT COUNT(*) FROM `{{payload['table_name']}}` WHERE parent_execution_id = %s",
                (payload['parent_execution_id'],),
            )
            selected_rows = int((await cursor.fetchone())[0])
            assert selected_rows > 0, payload['parent_execution_id']
            try:
                await cursor.execute(
                    f"DELETE FROM `{{payload['table_name']}}` WHERE parent_execution_id = %s AND 1 = 0",
                    (payload['parent_execution_id'],),
                )
            except Exception:
                write_denied = True
            else:
                raise AssertionError('trace database read-only account accepted DELETE')
    finally:
        connection.close()

    print('NEXUS_TDSQL_ACCOUNT_ASSERTION=' + json.dumps({{
        'selected_rows': selected_rows,
        'write_denied': write_denied,
        'rejected_password_count': rejected,
    }}, sort_keys=True))


asyncio.run(verify())
"""


async def assert_tdsql_readonly_credential_in_cluster(
    config: TDSQLTraceConfig,
    *,
    kubectl: str,
    namespace: str,
    release_name: str,
    credential: dict[str, Any],
    parent_execution_id: str,
    rejected_passwords: list[str],
) -> dict[str, Any]:
    if config.read_host is None or config.read_port is None:
        raise ValueError('TDSQL read endpoint is required for account E2E')
    if credential['read_host'] != config.read_host or credential['read_port'] != config.read_port:
        raise AssertionError('issued trace database endpoint does not match configured read endpoint')
    if credential['database'] != config.db:
        raise AssertionError('issued trace database does not match configured backend')
    resource = ready_pod_resource(kubectl, namespace, 'batch-manager', release_name=release_name)
    completed = await asyncio.to_thread(
        run,
        [
            kubectl,
            '-n',
            namespace,
            'exec',
            '-i',
            resource,
            '--',
            '/opt/nexus-venv/bin/python',
            '-c',
            "exec(__import__('sys').stdin.read())",
        ],
        input_text=_tdsql_readonly_credential_script(
            credential,
            parent_execution_id=parent_execution_id,
            rejected_passwords=rejected_passwords,
        ),
        capture=True,
        timeout=90,
    )
    marker = 'NEXUS_TDSQL_ACCOUNT_ASSERTION='
    for line in reversed((completed.stdout or '').splitlines()):
        if line.startswith(marker):
            report = json.loads(line.removeprefix(marker))
            if isinstance(report, dict):
                return report
            break
    raise AssertionError(f'TDSQL account assertion returned no report: {(completed.stdout or "")[-1000:]}')


def _tdsql_drop_test_account_script(config: TDSQLTraceConfig, *, username: str) -> str:
    if config.maintenance_user is None or config.maintenance_password is None:
        raise ValueError('TDSQL maintenance credential is required for account cleanup')
    payload = json.dumps(
        {
            'host': config.host,
            'port': config.port,
            'db': config.db,
            'user': config.maintenance_user,
            'password': config.maintenance_password,
            'username': username,
        },
        separators=(',', ':'),
    )
    return f"""import asyncio
import json

from nexus.runner.trace.tdsql._driver import connect
from nexus.runner.trace.tdsql.config import build_db_config


payload = json.loads({json.dumps(payload)})


async def cleanup():
    connection = await connect(**build_db_config(
        host=payload['host'],
        port=payload['port'],
        user=payload['user'],
        password=payload['password'],
        db=payload['db'],
    ))
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(f"DROP USER IF EXISTS '{{payload['username']}}'@'%'")
        await connection.commit()
    finally:
        connection.close()
    print('NEXUS_TDSQL_ACCOUNT_CLEANUP=ok')


asyncio.run(cleanup())
"""


async def drop_tdsql_test_account_in_cluster(
    config: TDSQLTraceConfig,
    *,
    kubectl: str,
    namespace: str,
    release_name: str,
    username: str,
) -> str:
    if re.fullmatch(r'nxs_ro_[0-9a-f]{24}', username) is None:
        raise ValueError('invalid TDSQL test account username')
    if config.account_provider_type == 'tencent_cloud_dcdb':
        client = _create_dcdb_e2e_client(config)
        request = dcdb_models.DeleteAccountRequest()
        request.InstanceId = config.dcdb_instance_id
        request.UserName = username
        request.Host = config.dcdb_account_host
        try:
            await asyncio.wait_for(
                asyncio.to_thread(client.DeleteAccount, request),
                timeout=30,
            )
        except TencentCloudSDKException as exc:
            if exc.code != 'UnauthorizedOperation.PermissionDenied':
                raise
            assert config.dcdb_region is not None
            assert config.dcdb_instance_id is not None
            assert config.dcdb_secret_id is not None
            assert config.dcdb_secret_key is not None
            provider = TencentCloudDCDBTraceDatabaseAccountProvider(
                region=config.dcdb_region,
                instance_id=config.dcdb_instance_id,
                api_endpoint=config.dcdb_api_endpoint or 'dcdb.tencentcloudapi.com',
                database=config.db,
                account_host=config.dcdb_account_host,
                secret_id=config.dcdb_secret_id,
                secret_key=config.dcdb_secret_key,
                client=client,
            )
            await provider.disable_account(username=username)
            return 'deactivated'
        return 'removed'
    resource = ready_pod_resource(kubectl, namespace, 'batch-manager', release_name=release_name)
    completed = await asyncio.to_thread(
        run,
        [
            kubectl,
            '-n',
            namespace,
            'exec',
            '-i',
            resource,
            '--',
            '/opt/nexus-venv/bin/python',
            '-c',
            "exec(__import__('sys').stdin.read())",
        ],
        input_text=_tdsql_drop_test_account_script(config, username=username),
        capture=True,
        timeout=90,
    )
    if 'NEXUS_TDSQL_ACCOUNT_CLEANUP=ok' not in (completed.stdout or '').splitlines():
        raise AssertionError('TDSQL test account cleanup returned no success marker')
    return 'removed'


def _create_dcdb_e2e_client(config: TDSQLTraceConfig):
    if not all((config.dcdb_region, config.dcdb_secret_id, config.dcdb_secret_key)):
        raise ValueError('TDSQL DCDB E2E credentials are unavailable')
    http_profile = HttpProfile()
    http_profile.endpoint = config.dcdb_api_endpoint or 'dcdb.tencentcloudapi.com'
    http_profile.reqTimeout = 30
    return dcdb_client.DcdbClient(
        tencent_cloud_credential.Credential(config.dcdb_secret_id, config.dcdb_secret_key),
        config.dcdb_region,
        ClientProfile(httpProfile=http_profile),
    )


async def assert_trace_database_account_e2e(
    control_url: str,
    config: TDSQLTraceConfig,
    *,
    kubectl: str,
    namespace: str,
    release_name: str,
    parent_execution_id: str,
    owner_principal_id: str,
) -> dict[str, Any]:
    if config.access_config() is None:
        raise ValueError('TDSQL account E2E requires read and maintenance configuration')
    expected_table = principal_batch_trace_table_name(principal_trace_scope_id(owner_principal_id))
    catalog = await request_json('GET', f'{control_url}/v1/my-environments')
    account = next(
        (item for item in catalog['trace_database_accounts'] if item['backend_name'] == 'tdsql'),
        None,
    )
    if account is None:
        raise AssertionError('TDSQL account is absent from My Environments')
    assert account['batch_table_name'] == expected_table, account

    issued_username: str | None = None
    cleanup_status: str | None = None
    try:
        created = await request_json(
            'POST',
            f'{control_url}/v1/my-environments/predefined-local/trace-database-credentials',
            body={'backend_name': 'tdsql', 'operation': 'create'},
        )
        assert created['status'] == 'ready', created
        assert created['batch_table_name'] == expected_table, created
        issued_username = created['username']
        create_report = await assert_tdsql_readonly_credential_in_cluster(
            config,
            kubectl=kubectl,
            namespace=namespace,
            release_name=release_name,
            credential=created,
            parent_execution_id=parent_execution_id,
            rejected_passwords=[],
        )

        rotated = await request_json(
            'POST',
            f'{control_url}/v1/my-environments/predefined-local/trace-database-credentials',
            body={'backend_name': 'tdsql', 'operation': 'reset'},
        )
        assert rotated['status'] == 'ready', rotated
        assert rotated['username'] == created['username'], rotated
        assert rotated['credential_version'] > created['credential_version'], rotated
        reset_report = await assert_tdsql_readonly_credential_in_cluster(
            config,
            kubectl=kubectl,
            namespace=namespace,
            release_name=release_name,
            credential=rotated,
            parent_execution_id=parent_execution_id,
            rejected_passwords=[created['password']],
        )
        report = {
            'backend_name': 'tdsql',
            'username': rotated['username'],
            'batch_table_name': expected_table,
            'credential_versions': [created['credential_version'], rotated['credential_version']],
            'create': create_report,
            'reset': reset_report,
        }
    finally:
        if issued_username is not None:
            cleanup_status = await drop_tdsql_test_account_in_cluster(
                config,
                kubectl=kubectl,
                namespace=namespace,
                release_name=release_name,
                username=issued_username,
            )
    assert cleanup_status is not None
    report['account_cleanup'] = cleanup_status
    report['account_removed'] = cleanup_status == 'removed'
    return report


async def run_api_tests(
    control_url: str,
    app_url: str,
    compatibility_url: str,
    *,
    release_name: str,
    kubectl: str,
    namespace: str,
    run_opus48: bool,
    opus48_timeout: float,
    opus48_gongfeng_api_base_url: str,
    opus48_runtime_image: str,
    check_runtime_aiomonitor: bool,
    environment_values: dict[str, str],
    tdsql_trace: TDSQLTraceConfig | None = None,
    tdsql_batch_e2e: bool = False,
) -> dict[str, Any]:
    run_tag = f'k8s-{int(time.time())}'
    fake_runtime_base_url = f'http://{release_name}-fake-runtime:8080'
    print(f'api test run tag: {run_tag}', flush=True)
    await wait_http_ok(f'{control_url}/v1/health')
    await wait_http_ok(f'{app_url}/v1/health')
    await wait_http_ok(compatibility_health_url(compatibility_url))
    await assert_inventory(control_url, release_name)
    typed = await assert_typed_predefined_submissions(
        control_url,
        app_url,
        release_name=release_name,
        run_tag=run_tag,
    )
    cli_batch: dict[str, Any] | None = None
    if tdsql_batch_e2e:
        output_root = ROOT / 'output/distributed-submit-e2e'
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='predefined-cli-', dir=output_root) as directory:
            generator = Path(directory) / 'generator.py'
            write_predefined_cli_generator(generator, run_tag=run_tag)
            cli_batch = await assert_cli_predefined_batch(
                control_url,
                generator=generator,
                run_tag=run_tag,
            )
    trace_database_account_report: dict[str, Any] | None = None
    if tdsql_trace is not None and cli_batch is not None and tdsql_trace.access_config() is not None:
        trace_database_account_report = await assert_trace_database_account_e2e(
            control_url,
            tdsql_trace,
            kubectl=kubectl,
            namespace=namespace,
            release_name=release_name,
            parent_execution_id=cli_batch['execution_id'],
            owner_principal_id=cli_batch['owner_principal_id'],
        )
    tdsql_report: dict[str, Any] | None = None
    if tdsql_trace is not None:
        batch = cli_batch or typed['batch']
        tdsql_report = await assert_tdsql_batch_trace_in_cluster(
            tdsql_trace,
            kubectl=kubectl,
            namespace=namespace,
            release_name=release_name,
            owner_principal_id=batch['owner_principal_id'],
            parent_execution_id=batch['execution_id'],
            child_execution_ids=batch['child_execution_ids'],
        )
    if tdsql_batch_e2e:
        print('TDSQL batch E2E completed', flush=True)
        return {
            'run_tag': run_tag,
            'typed_predefined': typed,
            'cli_predefined_batch': cli_batch,
            'tdsql_batch_trace': tdsql_report,
            'trace_database_account': trace_database_account_report,
            'legacy_live_compatibility': False,
            'capacity_and_reschedule': False,
            'opus48': False,
        }
    await assert_fake_live_interaction(
        control_url,
        app_url,
        run_tag,
        fake_runtime_base_url=fake_runtime_base_url,
    )
    await assert_legacy_live_compatibility(
        control_url,
        app_url,
        compatibility_url,
        run_tag,
        fake_runtime_base_url=fake_runtime_base_url,
    )
    await assert_capacity_and_reschedule(
        control_url,
        app_url,
        run_tag,
        fake_runtime_base_url=fake_runtime_base_url,
    )
    if run_opus48:
        await assert_repeated_opus48_cluster(
            control_url,
            app_url,
            run_tag,
            timeout=opus48_timeout,
            gongfeng_api_base_url=opus48_gongfeng_api_base_url,
            runtime_image=opus48_runtime_image,
            check_runtime_aiomonitor=check_runtime_aiomonitor,
            environment_values=environment_values,
        )
    print('api tests completed', flush=True)
    return {
        'run_tag': run_tag,
        'typed_predefined': typed,
        'cli_predefined_batch': cli_batch,
        'tdsql_batch_trace': tdsql_report,
        'trace_database_account': trace_database_account_report,
        'legacy_live_compatibility': True,
        'capacity_and_reschedule': True,
        'opus48': run_opus48,
    }


def wait_for_job_pod_success(
    kubectl: str,
    namespace: str,
    *,
    job_name: str,
    timeout: str,
) -> None:
    deadline = time.monotonic() + parse_duration_seconds(timeout)
    job = json.loads(
        run(
            [kubectl, '-n', namespace, 'get', 'job', job_name, '-o', 'json'],
            capture=True,
        ).stdout
    )
    job_uid = str(job.get('metadata', {}).get('uid') or '')
    if not job_uid:
        raise RuntimeError(f'{job_name} has no UID')
    phases: dict[str, str] = {}
    while time.monotonic() < deadline:
        pods = json.loads(
            run(
                [
                    kubectl,
                    '-n',
                    namespace,
                    'get',
                    'pods',
                    '-l',
                    f'job-name={job_name}',
                    '-o',
                    'json',
                ],
                capture=True,
            ).stdout
        )
        phases = {}
        for item in pods.get('items', []):
            metadata = item.get('metadata', {})
            owners = metadata.get('ownerReferences') or []
            if not any(owner.get('kind') == 'Job' and owner.get('uid') == job_uid for owner in owners):
                continue
            phases[str(metadata['name'])] = item.get('status', {}).get('phase', 'Unknown')
        if any(phase == 'Succeeded' for phase in phases.values()):
            return
        failed = [name for name, phase in phases.items() if phase == 'Failed']
        if failed:
            raise RuntimeError(f'{job_name} failed: {failed}')
        time.sleep(1)
    raise TimeoutError(f'{job_name} did not succeed within {timeout}: {phases}')


def assert_database_ready(kubectl: str, namespace: str, release_name: str, timeout: str) -> None:
    run([kubectl, '-n', namespace, 'rollout', 'status', f'deployment/{release_name}-mysql', f'--timeout={timeout}'])
    wait_for_job_pod_success(
        kubectl,
        namespace,
        job_name=f'{release_name}-db-init',
        timeout=timeout,
    )


def assert_control_ready(kubectl: str, namespace: str, release_name: str, timeout: str) -> None:
    run([
        kubectl,
        '-n',
        namespace,
        'rollout',
        'status',
        f'deployment/{release_name}-control-gateway',
        f'--timeout={timeout}',
    ])


def assert_workload_ready(kubectl: str, namespace: str, release_name: str, timeout: str) -> None:
    for deployment in (
        f'{release_name}-fake-runtime',
        f'{release_name}-application-gateway',
        f'{release_name}-scheduler',
        f'{release_name}-reconciler',
    ):
        run([kubectl, '-n', namespace, 'rollout', 'status', f'deployment/{deployment}', f'--timeout={timeout}'])
    run([
        kubectl,
        '-n',
        namespace,
        'rollout',
        'status',
        f'statefulset/{release_name}-instance-manager',
        f'--timeout={timeout}',
    ])
    run([
        kubectl,
        '-n',
        namespace,
        'rollout',
        'status',
        f'statefulset/{release_name}-batch-manager',
        f'--timeout={timeout}',
    ])

    schedulers = json.loads(
        run(
            [
                kubectl,
                '-n',
                namespace,
                'get',
                'pods',
                '-l',
                f'app.kubernetes.io/component=scheduler,app.kubernetes.io/instance={release_name}',
                '-o',
                'json',
            ],
            capture=True,
        ).stdout
    )
    ready_schedulers = [
        item
        for item in schedulers['items']
        if all(
            condition['type'] != 'Ready' or condition['status'] == 'True' for condition in item['status']['conditions']
        )
    ]
    assert len(ready_schedulers) == 3, [item['metadata']['name'] for item in schedulers['items']]

    managers = json.loads(
        run(
            [
                kubectl,
                '-n',
                namespace,
                'get',
                'pods',
                '-l',
                f'app.kubernetes.io/component=instance-manager,app.kubernetes.io/instance={release_name}',
                '-o',
                'json',
            ],
            capture=True,
        ).stdout
    )
    assert len(managers['items']) == 2, [item['metadata']['name'] for item in managers['items']]
    batch_managers = json.loads(
        run(
            [
                kubectl,
                '-n',
                namespace,
                'get',
                'pods',
                '-l',
                f'app.kubernetes.io/component=batch-manager,app.kubernetes.io/instance={release_name}',
                '-o',
                'json',
            ],
            capture=True,
        ).stdout
    )
    assert len(batch_managers['items']) == 1, [item['metadata']['name'] for item in batch_managers['items']]

    deadline = time.monotonic() + parse_duration_seconds(timeout)
    leases = ''
    while time.monotonic() < deadline:
        leases = run(
            [
                kubectl,
                '-n',
                namespace,
                'exec',
                f'deploy/{release_name}-mysql',
                '--',
                'sh',
                '-c',
                (
                    'MYSQL_PWD="$MYSQL_ROOT_PASSWORD" mysql -h127.0.0.1 -uroot -N -B nexus_dist '
                    '-e "select count(*), count(distinct owner_id) from shard_leases '
                    "where shard_group = 'execution-scheduler'\""
                ),
            ],
            capture=True,
        ).stdout.strip()
        lease_count, owner_count = [int(part) for part in leases.split('\t')]
        if lease_count == EXPECTED_SCHEDULER_SHARD_COUNT and owner_count >= 1:
            return
        time.sleep(1)
    raise AssertionError(f'scheduler leases not ready: {leases}')


def ready_pod_resource(
    kubectl: str,
    namespace: str,
    component: str,
    *,
    release_name: str | None = None,
) -> str:
    selector = f'app.kubernetes.io/component={component}'
    if release_name:
        selector += f',app.kubernetes.io/instance={release_name}'
    pods = json.loads(
        run(
            [kubectl, '-n', namespace, 'get', 'pods', '-l', selector, '-o', 'json'],
            capture=True,
        ).stdout
    )
    for item in sorted(pods['items'], key=lambda pod: pod['metadata']['name']):
        conditions = item['status'].get('conditions') or []
        ready = any(condition['type'] == 'Ready' and condition['status'] == 'True' for condition in conditions)
        if ready:
            return f'pod/{item["metadata"]["name"]}'
    raise AssertionError(f'no ready {component} pod: {[item["metadata"]["name"] for item in pods["items"]]}')


def assert_aiomonitor_static_ports(kubectl: str, namespace: str, *, release_name: str) -> None:
    print('checking static aiomonitor ports', flush=True)
    for component, remote_port in AIOMONITOR_STATIC_PORT_CHECKS:
        resource = ready_pod_resource(kubectl, namespace, component, release_name=release_name)
        with PortForward(kubectl=kubectl, namespace=namespace, resource=resource, remote_port=remote_port) as forward:
            wait_tcp_ok('127.0.0.1', forward.local_port, timeout=30)
        print(f'aiomonitor ready: {component}:{remote_port}', flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run predefined distributed tests on the DevCloud Kubernetes cluster.')
    parser.add_argument('--kubeconfig', type=Path, default=DEFAULT_DEVCLOUD_KUBECONFIG)
    parser.add_argument('--namespace', default=DEFAULT_DEVCLOUD_NAMESPACE)
    parser.add_argument('--release-name', default='nexus-phase2-devcloud-predefined')
    parser.add_argument('--control-image', default=DEFAULT_CONTROL_IMAGE)
    parser.add_argument('--workload-image', default=DEFAULT_WORKLOAD_IMAGE)
    parser.add_argument('--builder-image', default=DEFAULT_BUILDER_IMAGE)
    parser.add_argument('--workload-base-image', default='nexus-distributed-workload-base:devcloud-local')
    parser.add_argument('--workload-dockerfile', type=Path, default=TEST_WORKLOAD_DOCKERFILE)
    parser.add_argument('--mysql-image', default=DEFAULT_MYSQL_IMAGE)
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    parser.add_argument('--kubectl-bin')
    parser.add_argument('--helm-bin')
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--skip-push', action='store_true')
    parser.add_argument('--skip-opus48', '--skip-gpt55', action='store_true')
    parser.add_argument('--tdsql-config-file', type=Path)
    parser.add_argument(
        '--tdsql-batch-e2e',
        action='store_true',
        help='Run a focused CLI batch flow and verify its principal-scoped TDSQL trace table.',
    )
    parser.add_argument('--keep-release', action='store_true')
    parser.add_argument('--report-path', type=Path)
    parser.add_argument('--wait-timeout', default='600s')
    parser.add_argument('--opus48-timeout', '--opus-timeout', type=float, default=1800)
    parser.add_argument(
        '--opus48-gongfeng-api-base-url',
        default=os.environ.get('NEXUS_E2E_OPUS48_GONGFENG_API_BASE_URL', DEFAULT_OPUS48_GONGFENG_API_BASE_URL),
    )
    parser.add_argument(
        '--opus48-runtime-image',
        default=os.environ.get('NEXUS_E2E_OPUS48_RUNTIME_IMAGE', 'current'),
    )
    parser.add_argument('--check-runtime-aiomonitor', action='store_true')
    parser.add_argument('--artifact-routing-enabled', action='store_true')
    parser.add_argument('--artifact-routing-seed', default='artifact-store-v1')
    parser.add_argument('--artifact-routing-version', type=int, default=1)
    parser.add_argument('--artifact-routing-routes')
    parser.add_argument('--artifact-service-base-url', default='')
    parser.add_argument('--artifact-service-system-prefix', default='/system')
    parser.add_argument('--artifact-service-namespace-prefix', default='/_nexus_dist')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_routes: list[dict[str, Any]] = []
    if args.artifact_routing_enabled:
        if not args.artifact_routing_routes:
            raise SystemExit('--artifact-routing-enabled requires --artifact-routing-routes')
        try:
            decoded_routes = json.loads(args.artifact_routing_routes)
        except json.JSONDecodeError as exc:
            raise SystemExit('--artifact-routing-routes must be valid JSON') from exc
        if not isinstance(decoded_routes, list) or not all(isinstance(route, dict) for route in decoded_routes):
            raise SystemExit('--artifact-routing-routes must be a JSON array of objects')
        artifact_routes = decoded_routes
    kubectl = require_binary('kubectl', args.kubectl_bin)
    helm = require_binary('helm', args.helm_bin)
    configure_devcloud(kubectl, kubeconfig=args.kubeconfig, namespace=args.namespace)
    env_values = merged_env(args.env_file)
    if args.tdsql_batch_e2e and args.tdsql_config_file is None:
        raise SystemExit('--tdsql-batch-e2e requires --tdsql-config-file')
    tdsql_trace = load_tdsql_trace_config(args.tdsql_config_file) if args.tdsql_config_file is not None else None
    run_opus48 = not args.skip_opus48 and not args.tdsql_batch_e2e
    missing = [name for name in REQUIRED_SECRET_ENV if not env_values.get(name)]
    if run_opus48 and missing:
        raise SystemExit(f'missing required Opus 4.8 secrets in environment or {args.env_file}: {", ".join(missing)}')
    print(
        'secret inputs: '
        + ', '.join(f'{name}={"set" if env_values.get(name) else "missing"}' for name in REQUIRED_SECRET_ENV),
        flush=True,
    )

    if not args.skip_build:
        run(
            ['docker', 'build', '-f', 'deployment/distributed/Dockerfile.universal', '-t', args.control_image, '.'],
            timeout=900,
        )
        run(
            [
                'docker',
                'build',
                '-f',
                'deployment/distributed/Dockerfile.workload',
                '--build-arg',
                f'NEXUS_UNIVERSAL_BASE_IMAGE={args.control_image}',
                '-t',
                args.workload_base_image,
                '.',
            ],
            timeout=900,
        )
        run(
            [
                'docker',
                'build',
                '-f',
                str(args.workload_dockerfile),
                '--build-arg',
                f'NEXUS_WORKLOAD_BASE_IMAGE={args.workload_base_image}',
                '-t',
                args.workload_image,
                '.',
            ],
            timeout=900,
        )
        run(
            [
                'docker',
                'build',
                '-f',
                'deployment/distributed/Dockerfile.builder',
                '--build-arg',
                f'NEXUS_WORKLOAD_BASE_IMAGE={args.workload_image}',
                '-t',
                args.builder_image,
                '.',
            ],
            timeout=1200,
        )

    if not args.skip_push:
        push_images((args.control_image, args.workload_image, args.builder_image))

    app_local_port = free_port()
    database_secret_name = f'{args.release_name}-secrets'
    mysql_secret_name = f'{args.release_name}-mysql-auth'
    report: dict[str, Any] | None = None
    try:
        run([helm, 'uninstall', args.release_name, '--namespace', args.namespace, '--wait'], check=False)
        run(
            [
                kubectl,
                '-n',
                args.namespace,
                'delete',
                'secret',
                database_secret_name,
                mysql_secret_name,
                '--ignore-not-found=true',
            ],
            check=False,
        )
        mysql_password = create_bootstrap_secrets(
            kubectl,
            args.namespace,
            env_values,
            database_secret_name=database_secret_name,
            mysql_secret_name=mysql_secret_name,
            release_name=args.release_name,
        )
        control_image_repository, control_image_tag = split_image(args.control_image)
        workload_image_repository, workload_image_tag = split_image(args.workload_image)
        builder_image_repository, builder_image_tag = split_image(args.builder_image)
        helm_args = [
            helm,
            'upgrade',
            '--install',
            args.release_name,
            str(CHART_DIR),
            '--namespace',
            args.namespace,
            '-f',
            str(LOCAL_VALUES),
            '--set-string',
            f'fullnameOverride={args.release_name}',
            '--set-string',
            f'controlImage.repository={control_image_repository}',
            '--set-string',
            f'workloadImage.repository={workload_image_repository}',
            '--set-string',
            f'builderImage.repository={builder_image_repository}',
            '--set-string',
            f'secrets.existingSecret={database_secret_name}',
            '--set-string',
            f'mysql.auth.existingSecret={mysql_secret_name}',
            '--set',
            'secrets.create=false',
            '--set-string',
            'controlImage.pullPolicy=Always',
            '--set-string',
            'workloadImage.pullPolicy=Always',
            '--set-string',
            'builderImage.pullPolicy=Always',
            '--set',
            'liveCompatibility.enabled=true',
            '--timeout',
            args.wait_timeout,
        ]
        if control_image_tag:
            helm_args.extend(['--set-string', f'controlImage.tag={control_image_tag}'])
        if workload_image_tag:
            helm_args.extend(['--set-string', f'workloadImage.tag={workload_image_tag}'])
        if builder_image_tag:
            helm_args.extend(['--set-string', f'builderImage.tag={builder_image_tag}'])
        if args.mysql_image:
            mysql_image_repository, mysql_image_tag = split_image(args.mysql_image)
            helm_args.extend(['--set-string', f'mysql.image.repository={mysql_image_repository}'])
            if mysql_image_tag:
                helm_args.extend(['--set-string', f'mysql.image.tag={mysql_image_tag}'])
        bootstrap_helm_args = [
            *helm_args,
            '--set',
            'controlGateway.replicas=0',
            '--set',
            'applicationGateway.replicas=0',
            '--set',
            'scheduler.replicas=0',
            '--set',
            'reconciler.replicas=0',
            '--set',
            'instanceManager.replicas=0',
            '--set',
            'batchManager.replicas=0',
            '--set',
            'fakeRuntime.replicas=0',
            '--wait',
        ]
        run(bootstrap_helm_args, timeout=900)
        assert_database_ready(kubectl, args.namespace, args.release_name, args.wait_timeout)
        seed_cluster_platform_config(
            kubectl,
            namespace=args.namespace,
            release_name=args.release_name,
            mysql_password=mysql_password,
            env_values=env_values,
            application_public_base_url=f'http://127.0.0.1:{app_local_port}',
            workload_image=args.workload_image,
            builder_image=args.builder_image,
            database_secret_name=database_secret_name,
            artifact_routing_enabled=args.artifact_routing_enabled,
            artifact_routing_seed=args.artifact_routing_seed,
            artifact_routing_version=args.artifact_routing_version,
            artifact_routes=artifact_routes,
            artifact_service_base_url=args.artifact_service_base_url,
            artifact_service_system_prefix=args.artifact_service_system_prefix,
            artifact_service_namespace_prefix=args.artifact_service_namespace_prefix,
            tdsql_trace=tdsql_trace,
        )
        run([*helm_args, '--set', 'dbInit.enabled=false', '--wait'], timeout=900)
        assert_control_ready(kubectl, args.namespace, args.release_name, args.wait_timeout)
        bootstrap_control_resource = ready_pod_resource(
            kubectl,
            args.namespace,
            'control-gateway',
            release_name=args.release_name,
        )
        with PortForward(
            kubectl=kubectl,
            namespace=args.namespace,
            resource=bootstrap_control_resource,
            remote_port=8081,
        ) as bootstrap_control:
            asyncio.run(wait_http_ok(f'{bootstrap_control.url}/v1/health'))
            asyncio.run(ensure_predefined_environment(bootstrap_control.url))
        assert_workload_ready(kubectl, args.namespace, args.release_name, args.wait_timeout)
        assert_aiomonitor_static_ports(kubectl, args.namespace, release_name=args.release_name)
        control_resource = ready_pod_resource(
            kubectl,
            args.namespace,
            'control-gateway',
            release_name=args.release_name,
        )
        app_resource = ready_pod_resource(
            kubectl,
            args.namespace,
            'application-gateway',
            release_name=args.release_name,
        )

        with ExitStack() as stack:
            control = stack.enter_context(
                PortForward(
                    kubectl=kubectl,
                    namespace=args.namespace,
                    resource=control_resource,
                    remote_port=8081,
                )
            )
            app = stack.enter_context(
                PortForward(
                    kubectl=kubectl,
                    namespace=args.namespace,
                    resource=app_resource,
                    remote_port=8082,
                    local_port=app_local_port,
                )
            )
            report = asyncio.run(
                run_api_tests(
                    control.url,
                    app.url,
                    f'{app.url}/compat/live/v1',
                    release_name=args.release_name,
                    kubectl=kubectl,
                    namespace=args.namespace,
                    run_opus48=run_opus48,
                    opus48_timeout=args.opus48_timeout,
                    opus48_gongfeng_api_base_url=args.opus48_gongfeng_api_base_url,
                    opus48_runtime_image=args.opus48_runtime_image,
                    check_runtime_aiomonitor=args.check_runtime_aiomonitor,
                    environment_values=env_values,
                    tdsql_trace=tdsql_trace,
                    tdsql_batch_e2e=args.tdsql_batch_e2e,
                )
            )
    finally:
        if not args.keep_release:
            cleanup_release_resources(
                helm=helm,
                kubectl=kubectl,
                namespace=args.namespace,
                release_name=args.release_name,
                database_secret_name=database_secret_name,
                mysql_secret_name=mysql_secret_name,
            )
    assert report is not None
    report['cluster'] = {
        'kubeconfig': str(args.kubeconfig),
        'namespace': args.namespace,
        'release_name': args.release_name,
    }
    report['images'] = {
        'control': args.control_image,
        'workload': args.workload_image,
        'builder': args.builder_image,
        'mysql': args.mysql_image,
    }
    report['cleanup'] = {'release_removed': not args.keep_release}
    report_path = args.report_path or Path(tempfile.gettempdir()) / f'nexus-{args.release_name}-report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(f'{report_path.suffix}.tmp')
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    temporary_report.replace(report_path)
    print(f'predefined Kubernetes report: {report_path}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
