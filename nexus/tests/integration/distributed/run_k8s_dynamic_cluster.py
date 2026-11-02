from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid

from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import aiohttp
import yaml

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
from tests.integration.distributed.run_generic_repository_contract import (
    StaticCredentials,
    artifact_credentials,
    run_contract,
)
from tests.integration.distributed.run_k8s_predefined_cluster import (
    CHART_DIR,
    DEFAULT_BUILDER_IMAGE,
    DEFAULT_CONTROL_IMAGE,
    DEFAULT_DEVCLOUD_KUBECONFIG,
    DEFAULT_DEVCLOUD_NAMESPACE,
    DEFAULT_MYSQL_IMAGE,
    DEFAULT_WORKLOAD_IMAGE,
    PortForward,
    assert_database_ready,
    configure_devcloud,
    free_port,
    merged_env,
    poll_execution,
    push_images,
    read_sse_until,
    ready_pod_resource,
    request_json,
    require_binary,
    resolved_execution_result,
    run,
    split_image,
    wait_http_ok,
    wait_tcp_ok,
)

from nexus.cli.commands.submit import _build_wheel, _prepare_uv_lock, _upload
from nexus.dataset.zip_index import write_nexus_zip_directory_to_file
from nexus.distributed.artifacts.store import GenericRepositoryArtifactStore, StoredObject
from nexus.distributed.environment.naming import allocation_resource_names
from nexus.distributed.repositories.artifact import artifact_object_key
from nexus.distributed.submission.source_package import build_source_package
from nexus.distributed.submission.workload_image import resolve_workload_image, workload_image_repository


ROOT = Path(__file__).parents[3]
DYNAMIC_VALUES = CHART_DIR / 'examples/values-local-dynamic.yaml'
TEST_WORKLOAD_DOCKERFILE = ROOT / 'tests/integration/distributed/Dockerfile.test-workload'
TEST_USER_WORKLOAD_DOCKERFILE = ROOT / 'tests/integration/distributed/Dockerfile.test-user-workload'
DYNAMIC_ENVIRONMENT_ID = 'devcloud-dynamic'
DYNAMIC_PROVIDER_CONFIG_REF = 'kubernetes/devcloud'
BATCH_MANAGER_WORKLOAD = 'orchestration'
SOURCE_FIXTURE_MODULE = 'devcloud_source_package_fixture.py'
SOURCE_SMOKE_ITEM_COUNT = 20
SOURCE_SMOKE_TASK_SLEEP_SECONDS = 20
IMAGE_FIXTURE_PATH = 'tests/integration/distributed/image_mode_fixture.py'
IMAGE_FIXTURE_MODULE = 'tests.integration.distributed.image_mode_fixture'
IMAGE_SMOKE_ITEM_COUNT = 20
DATASET_ITEM_COUNT = 23
DATASET_PAGE_SIZE = 7
DATASET_SERVICE_PORT = 8084
DATASET_AUTH_TOKEN = 'devcloud-dataset-contract-token'
DATASET_URI = 'devcloud/echo@latest?split=test'
LOCAL_DATASET_ITEM_IDS = ('local-directory', 'local-zip')
LOCAL_DATASET_MARKERS = {
    'local-directory': 'local-directory-ok',
    'local-zip': 'local-zip-ok',
}
LOCAL_DATASET_BARRIER_EXPECTED_ARRIVALS = len(LOCAL_DATASET_ITEM_IDS)
HARBOR_DATASET_ITEM_IDS = ('harbor-directory', 'harbor-zip')
HARBOR_DATASET_MARKERS = {
    'harbor-directory': 'harbor-directory-ok',
    'harbor-zip': 'harbor-zip-ok',
}
HARBOR_DATASET_BARRIER_EXPECTED_ARRIVALS = len(HARBOR_DATASET_ITEM_IDS)
HEAVY_ITEM_COUNT = 2000
HEAVY_MANAGER_COUNT = 25
HEAVY_SLOTS_PER_MANAGER = 20
HEAVY_MAX_INFLIGHT = HEAVY_MANAGER_COUNT * HEAVY_SLOTS_PER_MANAGER
HEAVY_TASK_SLEEP_SECONDS = 20
HEAVY_TIMEOUT_PER_RUN_SECONDS = 300
HEAVY_MIN_PEAK_RESERVED = 450
HEAVY_RECOVERY_INJECTION_RESERVED = 400


class _AllocationWorkloadReleasedError(RuntimeError):
    """The allocation StatefulSet was removed as part of normal release."""


class _AllocationWorkloadTransitionError(RuntimeError):
    """The allocation workload is in a normal deletion transition."""


def _is_kubernetes_resource_not_found(
    completed: subprocess.CompletedProcess[str],
    *,
    resource_kind: str,
    resource_name: str,
    resource_group: str | None = None,
) -> bool:
    qualified_kind = f'{resource_kind}.{resource_group}' if resource_group else resource_kind
    expected_message = f'Error from server (NotFound): {qualified_kind} "{resource_name}" not found'
    outputs = [value for value in (completed.stdout, completed.stderr) if isinstance(value, str)]
    if any(line.strip() == expected_message for output in outputs for line in output.splitlines()):
        return True
    for output in outputs:
        try:
            status = json.loads(output)
        except json.JSONDecodeError:
            continue
        details = status.get('details') if isinstance(status, dict) else None
        if not isinstance(details, dict):
            continue
        if (
            status.get('kind') == 'Status'
            and status.get('reason') == 'NotFound'
            and details.get('kind') == resource_kind
            and details.get('name') == resource_name
            and (details.get('group') or None) == resource_group
        ):
            return True
    return False


def _remaining_timeout(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def heavy_peak_target(*, recovery_mode: str | None) -> int:
    if recovery_mode not in {None, 'execution', 'batch'}:
        raise ValueError(f'unsupported recovery mode: {recovery_mode}')
    return HEAVY_MIN_PEAK_RESERVED if recovery_mode is None else HEAVY_RECOVERY_INJECTION_RESERVED


def source_smoke_submission(
    *,
    source_ref: dict[str, Any],
    nexus_wheel_ref: dict[str, Any],
    uv_lock_ref: dict[str, Any],
    idempotency_key: str,
) -> dict[str, Any]:
    return {
        'submission_version': 1,
        'idempotency_key': idempotency_key,
        'environment_id': DYNAMIC_ENVIRONMENT_ID,
        'code': {
            'mode': 'source_package',
            'source_package': source_ref,
            'nexus_wheel': nexus_wheel_ref,
            'uv_lock': uv_lock_ref,
            'extras': [],
        },
        'resources': {
            'batch_manager': {'cpu_cores': 0.25, 'memory_gib': 1.0},
            'managers': {'count': 2, 'cpu_cores': 0.25, 'memory_gib': 1.0, 'slots': 2},
        },
        'generation': {
            'mode': 'remote_materialize',
            'source': {'type': 'generator', 'path': SOURCE_FIXTURE_MODULE},
            'timeout_seconds': 300,
        },
        'batch': {
            'max_inflight': 4,
            'retry_failed': 1,
            'fail_fast': False,
            'timeout_per_run': 120,
        },
        'provisioning_timeout_seconds': 600,
        'queue_ttl_seconds': 600,
        'execution_ttl_seconds': 1800,
        'stop_timeout_seconds': 60,
    }


def image_single_submission(*, image_version: str, idempotency_key: str) -> dict[str, Any]:
    return {
        'submission_version': 1,
        'idempotency_key': idempotency_key,
        'environment_id': DYNAMIC_ENVIRONMENT_ID,
        'code': {'mode': 'image', 'image_version': image_version},
        'resources': {
            'managers': {'count': 1, 'cpu_cores': 0.25, 'memory_gib': 1.0, 'slots': 1},
        },
        'run_input': {
            'imports': [IMAGE_FIXTURE_MODULE],
            'task_name': 'distributed_image_echo',
            'task_config': {'index': 999, 'sleep_seconds': 10},
        },
        'provisioning_timeout_seconds': 600,
        'queue_ttl_seconds': 600,
        'execution_ttl_seconds': 900,
        'stop_timeout_seconds': 60,
    }


def image_batch_submission(*, image_version: str, idempotency_key: str) -> dict[str, Any]:
    return {
        'submission_version': 1,
        'idempotency_key': idempotency_key,
        'environment_id': DYNAMIC_ENVIRONMENT_ID,
        'code': {'mode': 'image', 'image_version': image_version},
        'resources': {
            'batch_manager': {'cpu_cores': 0.25, 'memory_gib': 1.0},
            'managers': {'count': 2, 'cpu_cores': 0.25, 'memory_gib': 1.0, 'slots': 2},
        },
        'generation': {
            'mode': 'remote_materialize',
            'source': {'type': 'generator', 'path': IMAGE_FIXTURE_PATH},
            'timeout_seconds': 300,
        },
        'batch': {
            'max_inflight': 4,
            'retry_failed': 1,
            'fail_fast': False,
            'timeout_per_run': 120,
        },
        'provisioning_timeout_seconds': 600,
        'queue_ttl_seconds': 600,
        'execution_ttl_seconds': 1800,
        'stop_timeout_seconds': 60,
    }


def write_heavy_task_list(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open('wb') as stream:
        for index in range(HEAVY_ITEM_COUNT):
            record = {
                'version': 1,
                'item_id': f'item-{index:012d}',
                'run_input': {
                    'run_id': f'devcloud-heavy-{index:04d}',
                    'imports': ['tests.integration.distributed.cloud_tasks'],
                    'task_name': 'sleep_echo',
                    'task_config': {
                        'index': index,
                        'message': f'heavy-{index:04d}',
                        'sleep_seconds': HEAVY_TASK_SLEEP_SECONDS,
                    },
                },
            }
            line = json.dumps(record, sort_keys=True, separators=(',', ':')).encode() + b'\n'
            stream.write(line)
            digest.update(line)
            size_bytes += len(line)
    return {
        'version': 1,
        'item_count': HEAVY_ITEM_COUNT,
        'size_bytes': size_bytes,
        'sha256': digest.hexdigest(),
        'source': {'mode': 'devcloud_heavy'},
        'datasets': [],
    }


def heavy_batch_submission(
    *,
    task_list_ref: dict[str, Any],
    manifest: dict[str, Any],
    idempotency_key: str,
    retry_failed: int = 0,
) -> dict[str, Any]:
    return {
        'submission_version': 1,
        'idempotency_key': idempotency_key,
        'environment_id': DYNAMIC_ENVIRONMENT_ID,
        'code': {'mode': 'config_only'},
        'resources': {
            'batch_manager': {'cpu_cores': 1.0, 'memory_gib': 1.0},
            'managers': {
                'count': HEAVY_MANAGER_COUNT,
                'cpu_cores': 1.0,
                'memory_gib': 1.0,
                'slots': HEAVY_SLOTS_PER_MANAGER,
            },
        },
        'generation': {
            'mode': 'local_task_list',
            'task_list': {'ref': task_list_ref, 'format': 'jsonl', 'manifest': manifest},
            'timeout_seconds': 600,
        },
        'batch': {
            'max_inflight': HEAVY_MAX_INFLIGHT,
            'retry_failed': retry_failed,
            'fail_fast': False,
            'child_queue_ttl_seconds': 900,
            'timeout_per_run': HEAVY_TIMEOUT_PER_RUN_SECONDS,
        },
        'provisioning_timeout_seconds': 900,
        'queue_ttl_seconds': 900,
        'execution_ttl_seconds': 3600,
        'stop_timeout_seconds': 120,
    }


def dataset_batch_submission(*, idempotency_key: str) -> dict[str, Any]:
    return {
        'submission_version': 1,
        'idempotency_key': idempotency_key,
        'environment_id': DYNAMIC_ENVIRONMENT_ID,
        'code': {'mode': 'config_only'},
        'data_credentials': {
            'wedata': {
                'subject': 'devcloud-dataset-contract',
                'key': 'devcloud-dataset-contract-key',
                'id': 'devcloud-dataset-contract-key-v1',
            }
        },
        'resources': {
            'batch_manager': {'cpu_cores': 0.25, 'memory_gib': 1.0},
            'managers': {'count': 2, 'cpu_cores': 0.25, 'memory_gib': 1.0, 'slots': 2},
        },
        'generation': {
            'mode': 'remote_materialize',
            'source': {
                'type': 'compose',
                'compose': {
                    'parts': [
                        {
                            'kind': 'data',
                            'value': DATASET_URI,
                            'source': f'--data {DATASET_URI}',
                        }
                    ],
                },
                'task_list_version': 2,
            },
            'timeout_seconds': 300,
        },
        'batch': {
            'max_inflight': 4,
            'retry_failed': 0,
            'fail_fast': False,
            'timeout_per_run': 120,
        },
        'provisioning_timeout_seconds': 600,
        'queue_ttl_seconds': 600,
        'execution_ttl_seconds': 1800,
        'stop_timeout_seconds': 60,
    }


def heavy_child_diagnostics(facts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(fact.get('status')) for fact in facts.values())
    artifact_counts = Counter(str(fact.get('status')) for fact in facts.values() if fact.get('result_artifact_id'))
    margins: dict[str, list[float]] = {}
    expired_by_manager: Counter[str] = Counter()
    stop_reason_counts: Counter[str] = Counter()
    expired_samples: list[dict[str, Any]] = []

    for fact in facts.values():
        status = str(fact.get('status'))
        updated_at = fact.get('updated_at')
        execution_expires_at = fact.get('execution_expires_at')
        if isinstance(updated_at, str) and isinstance(execution_expires_at, str):
            updated = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            expires = datetime.fromisoformat(execution_expires_at.replace('Z', '+00:00'))
            margins.setdefault(status, []).append((expires - updated).total_seconds())
        stop_reason = fact.get('stop_reason')
        if isinstance(stop_reason, str) and stop_reason:
            stop_reason_counts[stop_reason] += 1
        if status != 'expired':
            continue
        assignment = fact.get('current_assignment')
        manager_id = assignment.get('manager_id') if isinstance(assignment, dict) else None
        if isinstance(manager_id, str) and manager_id:
            expired_by_manager[manager_id] += 1
        if len(expired_samples) < 10:
            expired_samples.append(
                {
                    'execution_id': fact.get('execution_id'),
                    'updated_at': updated_at,
                    'execution_expires_at': execution_expires_at,
                    'result_artifact_id': fact.get('result_artifact_id'),
                    'stop_reason': stop_reason,
                    'manager_id': manager_id,
                    'error_json': fact.get('error_json'),
                }
            )

    margin_stats = {
        status: {
            'min': round(min(values), 3),
            'median': round(median(values), 3),
            'max': round(max(values), 3),
        }
        for status, values in sorted(margins.items())
        if values
    }
    return {
        'status_counts': dict(sorted(status_counts.items())),
        'result_artifact_counts': {status: artifact_counts.get(status, 0) for status in sorted(status_counts)},
        'deadline_margin_seconds': margin_stats,
        'expired_by_manager': dict(sorted(expired_by_manager.items())),
        'stop_reason_counts': dict(sorted(stop_reason_counts.items())),
        'expired_samples': sorted(expired_samples, key=lambda item: str(item['execution_id'])),
    }


def fake_dataset_server_code() -> str:
    return r"""
import hashlib
import os
import re
import tempfile
from pathlib import Path

from nexus.dataset.zip_index import write_nexus_zip_from_directory

from aiohttp import web


TOKEN = os.environ["DATASET_AUTH_TOKEN"]
ITEM_COUNT = int(os.environ["DATASET_ITEM_COUNT"])
PAGE_SIZE = int(os.environ["DATASET_PAGE_SIZE"])
BASE_URL = os.environ["DATASET_SERVICE_BASE_URL"].rstrip("/")
PORT = int(os.environ.get("DATASET_SERVICE_PORT", "8084"))
RANGE_PATTERN = re.compile(r"bytes=(\d+)-(\d+)")
STATS = {
    "pagination_requests": 0,
    "sign_requests": 0,
    "range_requests": 0,
    "full_blob_requests": 0,
}


def build_zip(marker):
    with tempfile.TemporaryDirectory(prefix="dataset-zip-") as directory:
        root = Path(directory)
        (root / "marker.txt").write_text(marker + "\n", encoding="utf-8")
        return write_nexus_zip_from_directory(root)


BLOBS = {}
for index in range(ITEM_COUNT):
    path = f"blobs/dataset-{index:04d}.zip"
    content = build_zip(f"remote-dataset-ok:{index}")
    BLOBS[path] = {
        "content": content,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }


def authorize(request):
    if request.headers.get("secure-authentication") != TOKEN:
        raise web.HTTPUnauthorized(
            text='{"code":401,"message":"invalid dataset test token"}',
            content_type="application/json",
        )


def validate_dataset(request):
    if request.match_info["namespace"] != "devcloud" or request.match_info["dataset"] != "echo":
        raise web.HTTPNotFound()


async def health(request):
    return web.json_response({"ok": True})


def barrier_payload(state):
    return {
        "arrived": len(state["members"]),
        "expected": int(os.environ.get("DATASET_BARRIER_EXPECTED_ARRIVALS", "2")),
        "released": state["released"],
    }


def barrier_state(name):
    return BARRIERS.setdefault(name, {"members": set(), "released": False})


async def barrier_arrive(request):
    authorize(request)
    member = request.query.get("member")
    if not member:
        raise web.HTTPBadRequest(text="missing barrier member")
    state = barrier_state(request.match_info["name"])
    state["members"].add(member)
    return web.json_response(barrier_payload(state))


async def barrier_status(request):
    authorize(request)
    return web.json_response(barrier_payload(barrier_state(request.match_info["name"])))


async def barrier_release(request):
    authorize(request)
    state = barrier_state(request.match_info["name"])
    state["released"] = True
    return web.json_response(barrier_payload(state))


async def dataset(request):
    authorize(request)
    validate_dataset(request)
    return web.json_response({"code": 0, "data": {"latest_release": "v1", "releases": ["v1"]}})


async def version(request):
    authorize(request)
    validate_dataset(request)
    if request.match_info["version"] != "v1":
        raise web.HTTPNotFound()
    return web.json_response({"code": 0, "data": {"default_split": "test", "splits": ["test"]}})


async def items(request):
    authorize(request)
    validate_dataset(request)
    if request.match_info["version"] != "v1" or request.query.get("split") != "test":
        raise web.HTTPNotFound()
    try:
        start = int(request.query.get("cursor", "0"))
        requested = int(request.query.get("limit", str(PAGE_SIZE)))
    except ValueError as exc:
        raise web.HTTPBadRequest(text="invalid pagination") from exc
    if start < 0 or requested < 1:
        raise web.HTTPBadRequest(text="invalid pagination")
    STATS["pagination_requests"] += 1
    end = min(ITEM_COUNT, start + min(requested, PAGE_SIZE))
    records = [
        {
            "id": f"dataset-{index:04d}",
            "body": {
                "imports": ["tests.integration.distributed.cloud_tasks"],
                "task_name": "dataset_task_files_echo",
                "task_config": {
                    "index": index,
                    "expected_marker": f"remote-dataset-ok:{index}",
                },
            },
            "files": {
                "type": "zip",
                "path": f"blobs/dataset-{index:04d}.zip",
                "size": BLOBS[f"blobs/dataset-{index:04d}.zip"]["size"],
                "sha256": BLOBS[f"blobs/dataset-{index:04d}.zip"]["sha256"],
            },
        }
        for index in range(start, end)
    ]
    has_more = end < ITEM_COUNT
    pagination = {"has_more": has_more}
    if has_more:
        pagination["next_cursor"] = str(end)
    return web.json_response({"code": 0, "data": {"items": records, "pagination": pagination}})


async def sign_files(request):
    authorize(request)
    validate_dataset(request)
    if request.match_info["version"] != "v1" or request.query.get("split") != "test":
        raise web.HTTPNotFound()
    payload = await request.json()
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list) or not files or not all(isinstance(path, str) for path in files):
        raise web.HTTPBadRequest(text="invalid files")
    missing = [path for path in files if path not in BLOBS]
    if missing:
        raise web.HTTPNotFound(text="unknown blob")
    STATS["sign_requests"] += 1
    return web.json_response({
        "code": 0,
        "data": {"urls": {path: f"{BASE_URL}/{path}" for path in files}},
    })


async def blob(request):
    path = f"blobs/{request.match_info['name']}"
    record = BLOBS.get(path)
    if record is None:
        raise web.HTTPNotFound()
    content = record["content"]
    requested_range = request.headers.get("Range")
    common_headers = {"Accept-Ranges": "bytes"}
    if requested_range is None:
        STATS["full_blob_requests"] += 1
        return web.Response(body=content, headers=common_headers, content_type="application/zip")
    match = RANGE_PATTERN.fullmatch(requested_range)
    if match is None:
        raise web.HTTPRequestRangeNotSatisfiable(headers={"Content-Range": f"bytes */{len(content)}"})
    start, end = (int(value) for value in match.groups())
    if start > end or start >= len(content):
        raise web.HTTPRequestRangeNotSatisfiable(headers={"Content-Range": f"bytes */{len(content)}"})
    end = min(end, len(content) - 1)
    body = content[start : end + 1]
    STATS["range_requests"] += 1
    return web.Response(
        status=206,
        body=body,
        headers={
            **common_headers,
            "Content-Range": f"bytes {start}-{end}/{len(content)}",
            "Content-Length": str(len(body)),
        },
        content_type="application/zip",
    )


async def stats(request):
    return web.json_response(STATS)


app = web.Application()
BARRIERS = {}
app.router.add_get("/health", health)
app.router.add_post("/barriers/{name}/arrive", barrier_arrive)
app.router.add_get("/barriers/{name}", barrier_status)
app.router.add_post("/barriers/{name}/release", barrier_release)
app.router.add_get("/api/datasets/{namespace}/{dataset}", dataset)
app.router.add_get("/api/datasets/{namespace}/{dataset}/versions/{version}", version)
app.router.add_get("/api/datasets/{namespace}/{dataset}/versions/{version}/items", items)
app.router.add_post("/api/datasets/{namespace}/{dataset}/versions/{version}/files:sign", sign_files)
app.router.add_get("/blobs/{name}", blob)
app.router.add_get("/stats", stats)
web.run_app(app, host="0.0.0.0", port=PORT, access_log=None)
""".strip()


def fake_dataset_manifest(*, release_name: str, image: str) -> str:
    name = f'{release_name}-fake-dataset'
    labels = {
        'app.kubernetes.io/name': 'nexus-distributed',
        'app.kubernetes.io/instance': release_name,
        'app.kubernetes.io/component': 'fake-dataset',
        'app.kubernetes.io/managed-by': 'nexus-test',
    }
    server_code = fake_dataset_server_code()
    return json.dumps(
        {
            'apiVersion': 'v1',
            'kind': 'List',
            'items': [
                {
                    'apiVersion': 'v1',
                    'kind': 'Service',
                    'metadata': {'name': name, 'labels': labels},
                    'spec': {
                        'selector': labels,
                        'ports': [{'name': 'http', 'port': DATASET_SERVICE_PORT, 'targetPort': 'http'}],
                    },
                },
                {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'metadata': {'name': name, 'labels': labels},
                    'spec': {
                        'replicas': 1,
                        'selector': {'matchLabels': labels},
                        'template': {
                            'metadata': {'labels': labels},
                            'spec': {
                                'automountServiceAccountToken': False,
                                'enableServiceLinks': False,
                                'containers': [
                                    {
                                        'name': 'fake-dataset',
                                        'image': image,
                                        'imagePullPolicy': 'Always',
                                        'command': ['/opt/nexus-venv/bin/python', '-u', '-c'],
                                        'args': [server_code],
                                        'env': [
                                            {'name': 'DATASET_AUTH_TOKEN', 'value': DATASET_AUTH_TOKEN},
                                            {'name': 'DATASET_ITEM_COUNT', 'value': str(DATASET_ITEM_COUNT)},
                                            {'name': 'DATASET_PAGE_SIZE', 'value': str(DATASET_PAGE_SIZE)},
                                            {
                                                'name': 'DATASET_BARRIER_EXPECTED_ARRIVALS',
                                                'value': str(LOCAL_DATASET_BARRIER_EXPECTED_ARRIVALS),
                                            },
                                            {
                                                'name': 'DATASET_SERVICE_BASE_URL',
                                                'value': f'http://{name}:{DATASET_SERVICE_PORT}',
                                            },
                                            {'name': 'DATASET_SERVICE_PORT', 'value': str(DATASET_SERVICE_PORT)},
                                        ],
                                        'ports': [{'name': 'http', 'containerPort': DATASET_SERVICE_PORT}],
                                        'readinessProbe': {
                                            'httpGet': {'path': '/health', 'port': 'http'},
                                            'periodSeconds': 2,
                                        },
                                        'resources': {
                                            'requests': {'cpu': '25m', 'memory': '64Mi'},
                                            'limits': {'cpu': '250m', 'memory': '256Mi'},
                                        },
                                        'securityContext': {
                                            'allowPrivilegeEscalation': False,
                                            'capabilities': {'drop': ['ALL']},
                                        },
                                    }
                                ],
                            },
                        },
                    },
                },
            ],
        },
        sort_keys=True,
    )


def create_fake_dataset_service(kubectl: str, namespace: str, *, release_name: str, image: str) -> None:
    name = f'{release_name}-fake-dataset'
    run(
        [kubectl, '-n', namespace, 'apply', '-f', '-'],
        input_text=fake_dataset_manifest(release_name=release_name, image=image),
    )
    run(
        [kubectl, '-n', namespace, 'rollout', 'status', f'deployment/{name}', '--timeout=180s'],
        timeout=210,
    )


def delete_fake_dataset_service(kubectl: str, namespace: str, *, release_name: str) -> None:
    name = f'{release_name}-fake-dataset'
    run(
        [
            kubectl,
            '-n',
            namespace,
            'delete',
            f'deployment/{name}',
            f'service/{name}',
            '--ignore-not-found=true',
            '--wait=false',
        ],
        check=False,
    )


def dataset_environment_config(*, release_name: str) -> list[dict[str, str]]:
    return [
        {
            'name': 'TAIJI_DATASET_SERVER_URL',
            'value': f'http://{release_name}-fake-dataset:{DATASET_SERVICE_PORT}',
        },
        {'name': 'TAIJI_DATASET_AUTH_TOKEN', 'value': DATASET_AUTH_TOKEN},
    ]


def dataset_service_enabled(scenarios: set[str]) -> bool:
    return bool(scenarios & {'dataset', 'local-dataset', 'harbor'})


def seed_dynamic_platform_config(
    kubectl: str,
    *,
    namespace: str,
    release_name: str,
    mysql_password: str,
    environment: dict[str, str],
    application_public_base_url: str,
    workload_image: str,
    builder_image: str,
    database_secret_name: str,
    artifact_provider: str,
    artifact_base_url: str,
    artifact_repository: str,
    artifact_credentials: StaticCredentials | None,
    include_dataset: bool,
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
        config_values = platform_config_values(
            application_public_base_url=application_public_base_url,
            application_internal_base_url=(
                f'http://{release_name}-application-gateway.{namespace}.svc.cluster.local:8082'
            ),
            workload_image=workload_image,
            builder_image=builder_image,
            namespace=namespace,
            release_name=release_name,
            database_secret_name=database_secret_name,
            enabled_providers=['kubernetes'],
            fake_runtime_base_url=f'http://{release_name}-fake-runtime:8080',
            environment_extra_env=(dataset_environment_config(release_name=release_name) if include_dataset else []),
            artifact_provider=artifact_provider,
            artifact_base_url=artifact_base_url,
            artifact_repository=artifact_repository,
            tdsql_trace=tdsql_trace,
        )
        if tdsql_trace is not None:
            config_values['default_run_trace_sinks'] = []
        seed_platform_config(
            run,
            python=sys.executable,
            database_url=local_database_url(port=mysql.local_port, password=mysql_password),
            config_values=config_values,
            secret_values=platform_secret_values(
                environment,
                artifact_username=artifact_credentials.username if artifact_credentials is not None else None,
                artifact_token=artifact_credentials.token if artifact_credentials is not None else None,
                tdsql_trace=tdsql_trace,
            ),
            role_secret_values=(
                {('control-gateway', '*'): control_gateway_secrets} if control_gateway_secrets else None
            ),
        )


MYSQL_STATUS_KEYS = frozenset({'Connections', 'Max_used_connections', 'Slow_queries', 'Threads_connected'})


def _mysql_query(kubectl: str, namespace: str, release_name: str, sql: str) -> str:
    resource = ready_pod_resource(kubectl, namespace, 'mysql', release_name=release_name)
    command = [
        kubectl,
        '-n',
        namespace,
        'exec',
        resource,
        '--',
        'sh',
        '-c',
        'exec mysql --protocol=tcp -h127.0.0.1 -uroot -p"$MYSQL_ROOT_PASSWORD" --batch --skip-column-names -e "$1"',
        'mysql-status',
        sql,
    ]
    attempts = 5
    last_output = ''
    for attempt in range(attempts):
        result = run(command, capture=True, check=False)
        if result.returncode == 0:
            return result.stdout
        last_output = result.stdout
        if attempt + 1 < attempts:
            time.sleep(1)
    raise RuntimeError(f'MySQL query failed after {attempts} attempts: {last_output[-2000:]}')


def _parse_mysql_status(output: str) -> dict[str, int]:
    status = {}
    for line in output.splitlines():
        key, separator, value = line.partition('\t')
        if separator and key in MYSQL_STATUS_KEYS:
            status[key] = int(value)
    assert set(status) == MYSQL_STATUS_KEYS, {'mysql_status_output': output}
    return status


def mysql_status(kubectl: str, namespace: str, release_name: str) -> dict[str, int]:
    sql = (
        'SHOW GLOBAL STATUS WHERE Variable_name IN '
        "('Connections','Max_used_connections','Slow_queries','Threads_connected');"
    )
    return _parse_mysql_status(_mysql_query(kubectl, namespace, release_name, sql))


def reset_mysql_status(kubectl: str, namespace: str, release_name: str) -> None:
    _mysql_query(kubectl, namespace, release_name, 'FLUSH STATUS;')


def _source_fixture_text() -> str:
    return f"""from __future__ import annotations

import asyncio

from nexus.core import TaskRunInput, TaskRunOutput, task_registry
from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    UserInputRequest,
    WithInputProvider,
    agent_registry,
    require_context,
)
from nexus.runtime.client.base import Runtime
from nexus.runtime.providers.base import RuntimeProvider
from nexus.runtime.providers.registry import runtime_provider_registry
from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput


class SourceEchoInput(TaskRunInput):
    index: int
    sleep_seconds: float = 0


class SourceEchoOutput(TaskRunOutput):
    index: int
    marker: str


@task_registry.register("distributed_source_package_echo")
async def distributed_source_package_echo(task_input: SourceEchoInput) -> SourceEchoOutput:
    await asyncio.sleep(task_input.sleep_seconds)
    return SourceEchoOutput(index=task_input.index, marker=f"source-package-ok:{{task_input.index}}")


class DatasetFilesInput(TaskRunInput):
    expected_marker: str
    barrier_name: str | None = None
    barrier_member: str | None = None


class DatasetFilesOutput(TaskRunOutput):
    marker: str
    paths: list[str]


@task_registry.register("distributed_dataset_files_echo")
async def distributed_dataset_files_echo(task_input: DatasetFilesInput) -> DatasetFilesOutput:
    if task_input.barrier_name and task_input.barrier_member:
        import json
        import os
        import urllib.error
        import urllib.parse
        import urllib.request

        base_url = os.environ["TAIJI_DATASET_SERVER_URL"].rstrip("/")
        token = os.environ["TAIJI_DATASET_AUTH_TOKEN"]
        barrier_path = urllib.parse.quote(task_input.barrier_name, safe="")
        member = urllib.parse.quote(task_input.barrier_member, safe="")

        def barrier_request(path, method="GET"):
            request = urllib.request.Request(
                f"{{base_url}}/barriers/{{barrier_path}}{{path}}",
                method=method,
                headers={{'secure-authentication': token}},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read())

        await asyncio.to_thread(barrier_request, f"/arrive?member={{member}}", "POST")
        while True:
            state = await asyncio.to_thread(barrier_request, "")
            if state["released"]:
                break
            await asyncio.sleep(0.5)
    ctx = require_context()
    if ctx.task_files is None:
        raise RuntimeError("dataset task files are missing")
    paths = sorted(await ctx.task_files.list())
    marker = (await ctx.task_files.read_text("marker.txt")).strip()
    if marker != task_input.expected_marker:
        raise RuntimeError("dataset marker mismatch: %r != %r" % (marker, task_input.expected_marker))
    return DatasetFilesOutput(marker=marker, paths=paths)


class HostEnvProbeOutput(TaskRunOutput):
    value: str


@task_registry.register("distributed_host_env_probe")
async def distributed_host_env_probe(_task_input: TaskRunInput) -> HostEnvProbeOutput:
    import os

    return HostEnvProbeOutput(value=os.environ.get("HOST_ENV_MVP_PROBE", ""))


class SourceInteractiveAgentInput(AgentRunInput, WithInputProvider):
    prompt: str = "choose"


class SourceInteractiveAgentOutput(AgentRunOutput):
    choice: str | None


@agent_registry.register("distributed_interactive_agent")
async def distributed_interactive_agent(agent_input: SourceInteractiveAgentInput) -> SourceInteractiveAgentOutput:
    choice = await agent_input.get_user_input(UserInputRequest(hint=agent_input.prompt))
    return SourceInteractiveAgentOutput(choice=choice)


@runtime_provider_registry.register("distributed_fake_runtime")
class DistributedFakeRuntimeProvider(RuntimeProvider):
    async def acquire(self, name: str, **params):
        import os

        base_url = params.get("base_url") or os.environ["NEXUS_DIST_FAKE_RUNTIME_BASE_URL"]
        runtime = DistributedFakeRuntime(base_url=base_url)
        self._active_runtimes[name] = runtime
        return runtime

    async def _release_impl(self, name: str) -> None:
        self._active_runtimes.pop(name, None)


class DistributedFakeRuntime(Runtime):
    def __init__(self, *, base_url: str) -> None:
        super().__init__()
        self._base_url = base_url

    @property
    def base_url(self) -> str | None:
        return self._base_url

    async def _invoke(self, request):
        raise NotImplementedError("distributed fake runtime only supports proxy tests")

    async def ws_connect(self) -> None:
        return None

    def on(self, event, listener):
        return lambda: None


def generate_configs():
    for index in range({SOURCE_SMOKE_ITEM_COUNT}):
        yield BatchItem(
            config=RunInput(
                imports=["devcloud_source_package_fixture"],
                task_name="distributed_source_package_echo",
                task_config={{"index": index, "sleep_seconds": {SOURCE_SMOKE_TASK_SLEEP_SECONDS}}},
            ),
            run_id=f"source-smoke-{{index:03d}}",
        )
"""


def write_local_dataset_fixture(root: Path, *, barrier_name: str | None = None) -> Path:
    """Create the minimal local Dataset used by the Cloud data-list E2E."""

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_module = SOURCE_FIXTURE_MODULE.removesuffix('.py')
    item_paths: list[str] = []

    directory_id = LOCAL_DATASET_ITEM_IDS[0]
    directory_item_dir = root / 'items' / 'test' / directory_id
    directory_files = directory_item_dir / 'files'
    directory_files.mkdir(parents=True, exist_ok=True)
    (directory_files / 'marker.txt').write_text(f'{LOCAL_DATASET_MARKERS[directory_id]}\n', encoding='utf-8')
    directory_item = {
        'id': directory_id,
        'metadata': {'fixture': 'local-task-files'},
        'body': {
            'imports': [source_module],
            'task_name': 'distributed_dataset_files_echo',
            'task_config': {
                'expected_marker': LOCAL_DATASET_MARKERS[directory_id],
                **(
                    {
                        'barrier_name': barrier_name,
                        'barrier_member': directory_id,
                    }
                    if barrier_name is not None
                    else {}
                ),
            },
        },
        'files': {'type': 'directory', 'path': 'files'},
    }
    (directory_item_dir / 'item.yaml').write_text(
        json.dumps(directory_item, indent=2) + '\n',
        encoding='utf-8',
    )
    item_paths.append(f'items/test/{directory_id}/item.yaml')

    zip_id = LOCAL_DATASET_ITEM_IDS[1]
    zip_item_dir = root / 'items' / 'test' / zip_id
    zip_item_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='nexus-local-dataset-zip-') as temporary_dir:
        zip_source = Path(temporary_dir) / 'files'
        zip_source.mkdir()
        (zip_source / 'marker.txt').write_text(f'{LOCAL_DATASET_MARKERS[zip_id]}\n', encoding='utf-8')
        zip_path = zip_item_dir / 'files.zip'
        write_nexus_zip_directory_to_file(zip_source, zip_path)
    zip_bytes = zip_path.read_bytes()
    zip_item = {
        'id': zip_id,
        'metadata': {'fixture': 'local-task-files'},
        'body': {
            'imports': [source_module],
            'task_name': 'distributed_dataset_files_echo',
            'task_config': {
                'expected_marker': LOCAL_DATASET_MARKERS[zip_id],
                **(
                    {
                        'barrier_name': barrier_name,
                        'barrier_member': zip_id,
                    }
                    if barrier_name is not None
                    else {}
                ),
            },
        },
        'files': {
            'type': 'zip',
            'path': 'files.zip',
            'sha256': hashlib.sha256(zip_bytes).hexdigest(),
            'size': len(zip_bytes),
        },
    }
    (zip_item_dir / 'item.yaml').write_text(
        json.dumps(zip_item, indent=2) + '\n',
        encoding='utf-8',
    )
    item_paths.append(f'items/test/{zip_id}/item.yaml')

    dataset = {
        'schema_version': 1,
        'name': 'devcloud-local-task-files',
        'default_split': 'test',
        'splits': {'test': {'items': item_paths}},
    }
    (root / 'dataset.yaml').write_text(json.dumps(dataset, indent=2) + '\n', encoding='utf-8')
    return root


def write_harbor_dataset_fixture(
    root: Path,
    *,
    docker_image: str,
    release_name: str,
    barrier_name: str,
) -> Path:
    """Create two tiny Harbor tasks for the distributed Reader/data-list E2E."""

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    service_url = f'http://{release_name}-fake-dataset:{DATASET_SERVICE_PORT}'

    def toml_string(value: str) -> str:
        return json.dumps(str(value))

    def verifier_script(item_id: str, marker: str) -> str:
        return f"""#!/bin/sh
set -eu
test "$(cat /app/fixture/marker.txt)" = {toml_string(marker)}
python3 - <<'PY'
import json
import os
import time
import urllib.parse
import urllib.request

base = os.environ["DATASET_BARRIER_URL"].rstrip("/")
name = urllib.parse.quote(os.environ["DATASET_BARRIER_NAME"], safe="")
member = urllib.parse.quote(os.environ["DATASET_BARRIER_MEMBER"], safe="")
headers = {{"secure-authentication": os.environ["DATASET_AUTH_TOKEN"]}}

def request(path, method="GET"):
    request = urllib.request.Request(base + path, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())

request(f"/barriers/{{name}}/arrive?member={{member}}", "POST")
deadline = time.monotonic() + 300
while True:
    state = request(f"/barriers/{{name}}")
    if state["released"]:
        break
    if time.monotonic() >= deadline:
        raise TimeoutError("Harbor E2E barrier was not released")
    time.sleep(0.25)
PY
cat > /logs/verifier/reward.json <<'JSON'
{{"reward": 1.0, "item_id": {toml_string(item_id)}, "marker": {toml_string(marker)}}}
JSON
"""

    for item_id in HARBOR_DATASET_ITEM_IDS:
        marker = HARBOR_DATASET_MARKERS[item_id]
        task_dir = root / item_id
        (task_dir / 'fixture').mkdir(parents=True, exist_ok=True)
        (task_dir / 'tests').mkdir()
        (task_dir / 'task.toml').write_text(
            f"""schema_version = "1.2"

[environment]
docker_image = {toml_string(docker_image)}
workdir = "/app"
cpus = 1
memory_mb = 1024
network_mode = "public"

[environment.env]
DATASET_BARRIER_URL = {toml_string(service_url)}
DATASET_BARRIER_NAME = {toml_string(barrier_name)}
DATASET_BARRIER_MEMBER = {toml_string(item_id)}
DATASET_AUTH_TOKEN = {toml_string(DATASET_AUTH_TOKEN)}

[environment.upload]
source = "fixture/marker.txt"
target = "/app/fixture/marker.txt"

[agent]
timeout_sec = 60

[verifier]
environment_mode = "shared"
timeout_sec = 120
""",
            encoding='utf-8',
        )
        (task_dir / 'instruction.md').write_text(f'Verify {item_id}.\n', encoding='utf-8')
        (task_dir / 'fixture' / 'marker.txt').write_text(f'{marker}\n', encoding='utf-8')
        test_path = task_dir / 'tests' / 'test.sh'
        test_path.write_text(verifier_script(item_id, marker), encoding='utf-8')
        test_path.chmod(0o755)
    return root


def _typed_single_live_submission(
    *,
    source_ref: dict[str, Any],
    nexus_wheel_ref: dict[str, Any],
    uv_lock_ref: dict[str, Any],
    run_input: dict[str, Any],
    live: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'submission_version': 1,
        'idempotency_key': f'devcloud-{uuid.uuid4().hex}',
        'environment_id': DYNAMIC_ENVIRONMENT_ID,
        'code': {
            'mode': 'source_package',
            'source_package': source_ref,
            'nexus_wheel': nexus_wheel_ref,
            'uv_lock': uv_lock_ref,
            'extras': [],
        },
        'resources': {
            'managers': {'count': 1, 'cpu_cores': 0.25, 'memory_gib': 1.0, 'slots': 1},
        },
        'run_input': run_input,
        'provisioning_timeout_seconds': 600,
        'queue_ttl_seconds': 600,
        'execution_ttl_seconds': 900,
        'stop_timeout_seconds': 60,
    }
    if live:
        payload['live'] = {'working_dir': '/workspace'}
    return payload


def _live_run_input() -> dict[str, Any]:
    return {
        'imports': ['nexus.extensions.tasks.identity', SOURCE_FIXTURE_MODULE.removesuffix('.py')],
        'runtime_provider_name': 'distributed_fake_runtime',
        'runtime_configs': {'main': {}},
        'task_name': 'identity',
        'agent_name': 'distributed_interactive_agent',
        'agent_config': {'prompt': 'choose'},
    }


def _write_cli_smoke_inputs(source_root: Path) -> tuple[Path, Path]:
    single_config = source_root / 'devcloud-cli-single.yaml'
    single_config.write_text(
        '\n'.join(
            [
                'imports:',
                '  - devcloud_source_package_fixture',
                'task_name: distributed_host_env_probe',
                'task_config:',
                '  index: 1001',
                '',
            ]
        ),
        encoding='utf-8',
    )
    live_config = source_root / 'devcloud-cli-live.yaml'
    live_config.write_text(
        '\n'.join(
            [
                'imports:',
                '  - devcloud_source_package_fixture',
                'runtime_configs:',
                '  main: {}',
                'task_name: identity',
                'agent_name: distributed_interactive_agent',
                'agent_config:',
                '  prompt: choose',
                '',
            ]
        ),
        encoding='utf-8',
    )
    return single_config, live_config


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
        raise RuntimeError(
            f'Nexus submit CLI failed ({completed.returncode}): {completed.stderr[-2000:] or completed.stdout[-2000:]}'
        )
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


def _run_dataset_fetch_cli(dataset: str, output: Path) -> dict[str, Any]:
    """Materialize one local Dataset fixture through the public dataset CLI."""

    executable = ROOT / '.venv/bin/nexus'
    if not executable.is_file():
        raise FileNotFoundError(f'Nexus CLI is unavailable: {executable}')
    completed = subprocess.run(
        [str(executable), 'dataset', 'fetch', dataset, '--output', str(output), '--force', '--json'],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f'Nexus dataset fetch CLI failed ({completed.returncode}): '
            f'{completed.stderr[-2000:] or completed.stdout[-2000:]}'
        )
    output = completed.stdout.strip()
    decoder = json.JSONDecoder()
    for offset in (index for index in range(len(output) - 1, -1, -1) if output[index] == '{'):
        try:
            payload, end = decoder.raw_decode(output[offset:])
        except json.JSONDecodeError:
            continue
        if output[offset + end :].strip():
            continue
        if isinstance(payload, dict):
            return payload
        raise RuntimeError(f'Nexus dataset fetch CLI returned an invalid payload: {payload!r}')
    raise RuntimeError('Nexus dataset fetch CLI returned no JSON result')


def _sanitize_cli_args(args: list[str]) -> list[str]:
    sanitized: list[str] = []
    replace_next = False
    for value in args:
        if replace_next:
            sanitized.append(f'<{sanitized[-1].removeprefix("--")}>')
            replace_next = False
            continue
        sanitized.append(value)
        replace_next = value in {'--control-url', '--data', '--project-root', '--source'}
    return sanitized


async def ensure_dynamic_environment(control_url: str, *, namespace: str) -> None:
    provider_configs = await request_json('GET', f'{control_url}/v1/provider-configs')
    provider_payload = {
        'config': {
            'provider_type': 'kubernetes',
            'namespace': namespace,
            'auth_type': 'in_cluster',
            # DevCloud's serving certificate needs relaxed Python strict-X509 flags.
            'tls_strict': False,
            'manager_resource_request_ratio': 0.1,
        },
        'status': 'enabled',
    }
    provider_path = urllib.parse.quote(DYNAMIC_PROVIDER_CONFIG_REF, safe='')
    if any(item['provider_config_ref'] == DYNAMIC_PROVIDER_CONFIG_REF for item in provider_configs['items']):
        await request_json('PUT', f'{control_url}/v1/provider-configs/{provider_path}', body=provider_payload)
    else:
        await request_json(
            'POST',
            f'{control_url}/v1/provider-configs',
            body={'provider_config_ref': DYNAMIC_PROVIDER_CONFIG_REF, **provider_payload},
        )
    environments = await request_json('GET', f'{control_url}/v1/environments')
    payload = {
        'name': 'DevCloud Dynamic',
        'allocation_type': 'dynamic',
        'provider_type': 'kubernetes',
        'provider_config_ref': DYNAMIC_PROVIDER_CONFIG_REF,
        'resource_limits': {
            'manager_count': 60,
            'cpu_millicores': 60_000,
            'memory_bytes': 64 * 1024**3,
            'slots': 1200,
            'active_batches': 4,
            'max_provisioning_timeout_seconds': 900,
            'max_queue_ttl_seconds': 900,
            'max_execution_ttl_seconds': 3600,
            'max_stop_timeout_seconds': 120,
            'max_generation_timeout_seconds': 600,
            'max_child_queue_ttl_seconds': 900,
            'max_timeout_per_run': HEAVY_TIMEOUT_PER_RUN_SECONDS,
        },
        'default_timeouts': {
            'provisioning_timeout_seconds': 900,
            'queue_ttl_seconds': 900,
            'execution_ttl_seconds': 3600,
            'stop_timeout_seconds': 120,
            'generation_timeout_seconds': 600,
            'child_queue_ttl_seconds': 900,
            'timeout_per_run': HEAVY_TIMEOUT_PER_RUN_SECONDS,
        },
        'default_quota': {
            'manager_count': 60,
            'cpu_millicores': 60_000,
            'memory_bytes': 64 * 1024**3,
            'slots': 1200,
            'active_batches': 4,
        },
        'status': 'enabled',
    }
    if any(item['environment_id'] == DYNAMIC_ENVIRONMENT_ID for item in environments['items']):
        await request_json('PUT', f'{control_url}/v1/environments/{DYNAMIC_ENVIRONMENT_ID}', body=payload)
    else:
        await request_json(
            'POST',
            f'{control_url}/v1/environments',
            body={'environment_id': DYNAMIC_ENVIRONMENT_ID, **payload},
        )
    catalog = await request_json('GET', f'{control_url}/v1/environment-catalog')
    entry = next(
        (item for item in catalog['items'] if item['environment_id'] == DYNAMIC_ENVIRONMENT_ID),
        None,
    )
    if entry is None:
        raise AssertionError('dynamic Environment is missing from the user catalog')
    grant = entry.get('grant')
    if grant is None:
        if not entry.get('claimable'):
            raise AssertionError('dynamic Environment default quota is not claimable')
        claimed = await request_json(
            'PUT',
            f'{control_url}/v1/my-environments/{DYNAMIC_ENVIRONMENT_ID}/quota',
        )
        grant = claimed.get('grant')
    if not isinstance(grant, dict) or not grant.get('enabled') or grant.get('limits') != payload['default_quota']:
        raise AssertionError(f'dynamic Environment default quota claim returned an invalid Grant: {grant}')


async def _upload_artifact(
    session: aiohttp.ClientSession,
    control_url: str,
    path: Path,
    *,
    kind: str,
    sha256: str,
    size_bytes: int,
) -> dict[str, Any]:
    return await _upload(
        session,
        control_url,
        path,
        kind=kind,
        sha256=sha256,
        size_bytes=size_bytes,
        idempotency_key=f'devcloud-{kind}-{uuid.uuid4().hex}',
        environment_id=DYNAMIC_ENVIRONMENT_ID,
    )


async def _wait_batch_summary(control_url: str, execution_id: str, predicate, *, timeout: float) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout
    latest: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        latest = await request_json('GET', f'{control_url}/v1/executions/{execution_id}/batch')
        if predicate(latest):
            return latest
        if latest.get('control_state') == 'finalized':
            raise AssertionError(f'batch entered unexpected terminal state: {execution_id}: {latest}')
        await asyncio.sleep(1)
    raise TimeoutError(f'batch summary did not reach expected state: {execution_id}: {latest}')


async def _wait_allocations_ready(control_url: str, execution_id: str, *, timeout: float = 900) -> dict[str, dict]:
    deadline = asyncio.get_running_loop().time() + timeout
    latest: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        latest = await request_json('GET', f'{control_url}/v1/executions/{execution_id}/allocations')
        by_workload = {item['workload']: item for item in latest['items']}
        if by_workload and all(
            item['status'] == 'ready' and item['ready_manager_count'] == item['desired_manager_count']
            for item in by_workload.values()
        ):
            return by_workload
        await asyncio.sleep(1)
    raise TimeoutError(f'allocations did not become ready: {execution_id}: {latest}')


async def _wait_allocations_released(control_url: str, execution_id: str, *, timeout: float = 300) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    latest: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        latest = await request_json('GET', f'{control_url}/v1/executions/{execution_id}/allocations')
        if latest['items'] and all(item['status'] == 'released' for item in latest['items']):
            return
        await asyncio.sleep(1)
    raise TimeoutError(f'allocations were not released: {execution_id}: {latest}')


async def _wait_no_active_manager_sessions(
    control_url: str,
    *,
    allocation_ids: set[str],
    timeout: float = 90,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    latest: list[dict[str, Any]] = []
    while asyncio.get_running_loop().time() < deadline:
        managers = await request_json('GET', f'{control_url}/v1/managers')
        latest = [item for item in managers['items'] if item['allocation_id'] in allocation_ids and item['healthy']]
        if not latest:
            return
        await asyncio.sleep(1)
    raise TimeoutError(f'dynamic manager sessions remained healthy after release: {latest[:10]}')


def _tdsql_v3_task_record_assertion_script(
    config: TDSQLTraceConfig,
    *,
    parent_execution_id: str,
    forbidden_path: Path,
    expect_present: bool,
    expected_item_ids: tuple[str, ...] = LOCAL_DATASET_ITEM_IDS,
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
            'parent_execution_id': parent_execution_id,
            'forbidden_path': str(forbidden_path),
            'expected_item_ids': list(expected_item_ids),
            'expect_present': expect_present,
        },
        separators=(',', ':'),
    )
    return f"""import asyncio
import inspect
import json
import re

from nexus.distributed.batch.task_store import BatchTaskRecord, decode_task_record
from nexus.runner.trace.tdsql._driver import connect
from nexus.runner.trace.tdsql.config import build_db_config


payload = json.loads({json.dumps(payload)})


async def maybe_await(value):
    return await value if inspect.isawaitable(value) else value


async def verify():
    config = payload['config']
    parent_execution_id = payload['parent_execution_id']
    forbidden_path = payload['forbidden_path']
    expected_item_ids = set(payload['expected_item_ids'])
    connection = await connect(**build_db_config(**config))
    try:
        async with connection.cursor() as cursor:
            await cursor.execute('SHOW CREATE TABLE batch_task_records')
            create_row = await maybe_await(cursor.fetchone())
            create_statement = str(create_row[1])
            normalized_create = re.sub(r'[\\s`]+', '', create_statement.lower())
            assert 'shardkey=parent_execution_id' in normalized_create, create_statement
            await cursor.execute(
                'SELECT item_id, record_version, record_sha256, record_data '
                'FROM batch_task_records WHERE parent_execution_id = %s ORDER BY item_id',
                (parent_execution_id,),
            )
            rows = await maybe_await(cursor.fetchall())

        records = {{}}
        for item_id, version, record_sha256, record_data in rows:
            item_id = str(item_id)
            record_sha256 = str(record_sha256)
            if isinstance(record_data, memoryview):
                record_data = record_data.tobytes()
            elif not isinstance(record_data, bytes):
                record_data = bytes(record_data)
            record = BatchTaskRecord(
                parent_execution_id=parent_execution_id,
                item_id=item_id,
                record_version=int(version),
                record_sha256=record_sha256,
                record_data=record_data,
            )
            decoded = decode_task_record(record, expected_sha256=record_sha256)
            assert record.record_version == 3 and decoded.frame is not None, (item_id, record.record_version)
            assert decoded.run_input.get('task_files') is None, item_id
            assert forbidden_path not in json.dumps(decoded.identity, sort_keys=True), item_id
            records[item_id] = {{
                'record_version': record.record_version,
                'record_size_bytes': len(record_data),
                'data_ref': decoded.identity['data_ref'],
                'frame_zip_size': decoded.frame.zip_size,
                'frame_zip_sha256': decoded.frame.zip_sha256,
            }}
        if payload['expect_present']:
            assert set(records) == expected_item_ids, (sorted(records), sorted(expected_item_ids))
        else:
            assert not records, sorted(records)
        await connection.rollback()
    finally:
        connection.close()

    report = {{
        'table_name': 'batch_task_records',
        'shard_key': 'parent_execution_id',
        'records': records,
        'remaining_parent_rows': len(records),
    }}
    print('NEXUS_TDSQL_TASK_RECORDS=' + json.dumps(report, sort_keys=True))


asyncio.run(verify())
"""


def _read_v3_task_records(
    kubectl: str,
    namespace: str,
    release_name: str,
    *,
    config: TDSQLTraceConfig,
    parent_execution_id: str,
    forbidden_path: Path,
    expect_present: bool,
    expected_item_ids: tuple[str, ...] = LOCAL_DATASET_ITEM_IDS,
) -> dict[str, Any]:
    """Inspect real TDSQL NXBR records in-cluster without exposing credentials or record bytes."""

    resource = ready_pod_resource(kubectl, namespace, 'control-gateway', release_name=release_name)
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
    completed = run(
        command,
        input_text=_tdsql_v3_task_record_assertion_script(
            config,
            parent_execution_id=parent_execution_id,
            forbidden_path=forbidden_path,
            expect_present=expect_present,
            expected_item_ids=expected_item_ids,
        ),
        capture=True,
        timeout=90,
    )
    marker = 'NEXUS_TDSQL_TASK_RECORDS='
    for line in reversed((completed.stdout or '').splitlines()):
        if line.startswith(marker):
            report = json.loads(line.removeprefix(marker))
            if isinstance(report, dict):
                report['target_fingerprint'] = config.target_fingerprint()
                return report
            break
    raise AssertionError('TDSQL task record assertion returned no sanitized report')


async def _wait_v3_task_records_removed(
    kubectl: str,
    namespace: str,
    release_name: str,
    *,
    config: TDSQLTraceConfig,
    parent_execution_id: str,
    forbidden_path: Path,
    expected_item_ids: tuple[str, ...] = LOCAL_DATASET_ITEM_IDS,
    timeout: float = 120,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout
    latest_error: BaseException | None = None
    while asyncio.get_running_loop().time() < deadline:
        try:
            return await asyncio.to_thread(
                _read_v3_task_records,
                kubectl,
                namespace,
                release_name,
                config=config,
                parent_execution_id=parent_execution_id,
                forbidden_path=forbidden_path,
                expect_present=False,
                expected_item_ids=expected_item_ids,
            )
        except (AssertionError, subprocess.CalledProcessError) as exc:
            latest_error = exc
        await asyncio.sleep(1)
    raise TimeoutError(f'Batch Task Store records were not removed automatically: {latest_error}')


def _ready_allocation_manager_pod(
    kubectl: str,
    namespace: str,
    *,
    allocation_id: str,
    deadline: float | None = None,
) -> tuple[str, str]:
    workload_name, _ = allocation_resource_names(allocation_id)
    workload_result = run(
        [kubectl, '-n', namespace, 'get', 'statefulset', workload_name, '-o', 'json'],
        capture=True,
        check=False,
        timeout=_remaining_timeout(deadline),
    )
    if workload_result.returncode != 0:
        if _is_kubernetes_resource_not_found(
            workload_result,
            resource_kind='statefulsets',
            resource_name=workload_name,
            resource_group='apps',
        ):
            raise _AllocationWorkloadReleasedError(workload_name)
        raise subprocess.CalledProcessError(
            workload_result.returncode,
            workload_result.args,
            output=workload_result.stdout,
            stderr=workload_result.stderr,
        )
    workload = json.loads(workload_result.stdout)
    selector = workload['spec']['selector']['matchLabels']
    selector_value = ','.join(f'{key}={value}' for key, value in sorted(selector.items()))
    pods = json.loads(
        run(
            [kubectl, '-n', namespace, 'get', 'pods', '-l', selector_value, '-o', 'json'],
            capture=True,
            timeout=_remaining_timeout(deadline),
        ).stdout
    )['items']
    ready = [
        pod
        for pod in pods
        if any(
            condition.get('type') == 'Ready' and condition.get('status') == 'True'
            for condition in pod.get('status', {}).get('conditions', [])
        )
    ]
    if not ready:
        raise _AllocationWorkloadTransitionError(
            {'workload': workload_name, 'pods': [pod['metadata']['name'] for pod in pods]}
        )
    pod_name = sorted(pod['metadata']['name'] for pod in ready)[0]
    return workload_name, pod_name


def _assert_batch_workspace_sanitized(
    kubectl: str,
    namespace: str,
    *,
    allocation_id: str,
    forbidden_path: Path,
) -> dict[str, Any]:
    """Inspect the downloaded task-list and reject submit-host path leakage."""

    _, pod_name = _ready_allocation_manager_pod(
        kubectl,
        namespace,
        allocation_id=allocation_id,
    )
    script = (
        'import json,os,sys; from pathlib import Path; '
        'root=Path(os.environ.get("NEXUS_DIST_BATCH_CONTROL_SPOOL_DIR","/tmp/nexus-batch-control")); '
        'files=sorted(root.rglob("tasks.jsonl")); '
        'needle=sys.argv[1].encode(); '
        'bad=[str(path) for path in files if needle in path.read_bytes()]; '
        'print(json.dumps({"files":[str(path) for path in files],"bad":bad})); '
        'raise SystemExit(0 if files and not bad else 1)'
    )
    inspected = run(
        [
            kubectl,
            '-n',
            namespace,
            'exec',
            f'pod/{pod_name}',
            '-c',
            'instance-manager',
            '--',
            '/opt/nexus-venv/bin/python',
            '-c',
            script,
            str(forbidden_path),
        ],
        capture=True,
    )
    payload = json.loads(inspected.stdout)
    return {'pod': pod_name, 'task_list_files': payload['files'], 'forbidden_path_matches': payload['bad']}


def _assert_assignment_task_files_clean(
    kubectl: str,
    namespace: str,
    *,
    allocation_id: str,
    timeout: float = 120,
    poll_interval: float = 1,
    require_files: bool = False,
) -> dict[str, Any]:
    """Wait until a live manager observes assignment ZIPs in the requested state."""

    expected_state = 'roots and files' if require_files else 'roots and not files'
    script = (
        'import json,os; from pathlib import Path; '
        'root=Path(os.environ.get("NEXUS_DIST_BATCH_CONTROL_SPOOL_DIR","/tmp/nexus-batch-control")); '
        'roots=sorted(path for path in root.glob("assignment-files-*") if path.is_dir()); '
        'files=sorted(str(path) for root in roots for path in root.rglob("*") '
        'if path.is_file() and path.name in {"files.zip",".files.zip.tmp"}); '
        'print(json.dumps({"roots":[str(path) for path in roots],"files":files})); '
        f'raise SystemExit(0 if {expected_state} else 1)'
    )
    deadline = time.monotonic() + timeout
    latest: object | None = None

    def timeout_error() -> TimeoutError:
        expected = 'present' if require_files else 'cleaned'
        return TimeoutError(
            f'assignment task files were not {expected} before allocation release: {allocation_id}: {latest}'
        )

    while True:
        try:
            _, pod_name = _ready_allocation_manager_pod(
                kubectl,
                namespace,
                allocation_id=allocation_id,
                deadline=deadline,
            )
        except subprocess.TimeoutExpired as exc:
            raise timeout_error() from exc
        except _AllocationWorkloadReleasedError as exc:
            raise AssertionError(
                f'assignment cleanup was not observed before allocation release: {allocation_id}'
            ) from exc
        except _AllocationWorkloadTransitionError as exc:
            latest = exc
        else:
            try:
                inspected = run(
                    [
                        kubectl,
                        '-n',
                        namespace,
                        'exec',
                        f'pod/{pod_name}',
                        '-c',
                        'instance-manager',
                        '--',
                        '/opt/nexus-venv/bin/python',
                        '-c',
                        script,
                    ],
                    capture=True,
                    check=False,
                    timeout=_remaining_timeout(deadline),
                )
            except subprocess.TimeoutExpired as exc:
                raise timeout_error() from exc
            if inspected.returncode == 0:
                payload = json.loads(inspected.stdout)
                return {
                    'pod': pod_name,
                    'assignment_temp_roots': payload['roots'],
                    'stale_assignment_task_files': payload['files'],
                }
            if _is_kubernetes_resource_not_found(
                inspected,
                resource_kind='pods',
                resource_name=pod_name,
            ):
                latest = _AllocationWorkloadTransitionError(f'pod disappeared before cleanup inspection: {pod_name}')
            else:
                try:
                    payload = json.loads(inspected.stdout)
                except (TypeError, json.JSONDecodeError):
                    raise subprocess.CalledProcessError(
                        inspected.returncode,
                        inspected.args,
                        output=inspected.stdout,
                        stderr=inspected.stderr,
                    ) from None
                latest = payload
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise timeout_error()
        time.sleep(min(poll_interval, remaining))


async def _wait_local_dataset_terminal_cleanup(
    kubectl: str,
    namespace: str,
    control_url: str,
    execution_id: str,
    *,
    release_name: str,
    tdsql_trace: TDSQLTraceConfig,
    forbidden_path: Path,
    allocation_id: str,
    expected_item_ids: tuple[str, ...] = LOCAL_DATASET_ITEM_IDS,
) -> dict[str, Any]:
    terminal = await poll_execution(
        control_url,
        execution_id,
        lambda item: item['status'] == 'succeeded',
        timeout=1200,
    )
    assignment_cleanup = await asyncio.to_thread(
        _assert_assignment_task_files_clean,
        kubectl,
        namespace,
        allocation_id=allocation_id,
    )
    terminal_task_store = await _wait_v3_task_records_removed(
        kubectl,
        namespace,
        release_name,
        config=tdsql_trace,
        parent_execution_id=execution_id,
        forbidden_path=forbidden_path,
        expected_item_ids=expected_item_ids,
    )
    return {
        'terminal': terminal,
        'task_store': terminal_task_store,
        'assignment': assignment_cleanup,
    }


async def _wait_for_local_dataset_barrier_and_release(
    kubectl: str,
    namespace: str,
    *,
    release_name: str,
    barrier_name: str,
    expected_arrivals: int,
    timeout: float = 120,
) -> dict[str, Any]:
    """Hold dataset tasks until live rows and assignment files have been inspected."""

    with PortForward(
        kubectl=kubectl,
        namespace=namespace,
        resource=f'service/{release_name}-fake-dataset',
        remote_port=DATASET_SERVICE_PORT,
    ) as dataset_service:
        await wait_http_ok(f'{dataset_service.url}/health', timeout=30)
        deadline = time.monotonic() + timeout
        headers = {'secure-authentication': DATASET_AUTH_TOKEN}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            latest: dict[str, Any] | None = None
            while time.monotonic() < deadline:
                async with session.get(
                    f'{dataset_service.url}/barriers/{urllib.parse.quote(barrier_name)}', headers=headers
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(f'barrier status returned {response.status}: {await response.text()}')
                    latest = await response.json()
                if latest['arrived'] >= expected_arrivals:
                    async with session.post(
                        f'{dataset_service.url}/barriers/{urllib.parse.quote(barrier_name)}/release',
                        headers=headers,
                    ) as response:
                        if response.status != 200:
                            raise RuntimeError(f'barrier release returned {response.status}: {await response.text()}')
                        return await response.json()
                await asyncio.sleep(0.5)
        raise TimeoutError(f'local dataset barrier did not receive all arrivals: {latest}')


async def _observe_local_dataset_active_state(
    kubectl: str,
    namespace: str,
    control_url: str,
    execution_id: str,
    *,
    orchestration_allocation_id: str,
    execution_allocation_id: str,
    release_name: str,
    barrier_name: str,
    dataset_root: Path,
    tdsql_trace: TDSQLTraceConfig,
    expected_item_ids: tuple[str, ...] = LOCAL_DATASET_ITEM_IDS,
    expected_barrier_arrivals: int = LOCAL_DATASET_BARRIER_EXPECTED_ARRIVALS,
) -> dict[str, Any]:
    """Capture all live workload evidence before releasing the deterministic task barrier."""

    workspace = await asyncio.to_thread(
        _assert_batch_workspace_sanitized,
        kubectl,
        namespace,
        allocation_id=orchestration_allocation_id,
        forbidden_path=dataset_root,
    )
    active_parent = await request_json('GET', f'{control_url}/v1/executions/{execution_id}')
    if active_parent['status'] in {'succeeded', 'failed', 'killed', 'lost', 'expired'}:
        raise AssertionError(active_parent)
    active_task_store = await asyncio.to_thread(
        _read_v3_task_records,
        kubectl,
        namespace,
        release_name,
        config=tdsql_trace,
        parent_execution_id=execution_id,
        forbidden_path=dataset_root,
        expect_present=True,
        expected_item_ids=expected_item_ids,
    )
    barrier = await _wait_for_local_dataset_barrier_and_release(
        kubectl,
        namespace,
        release_name=release_name,
        barrier_name=barrier_name,
        expected_arrivals=expected_barrier_arrivals,
    )
    return {
        'workspace': workspace,
        'active_task_store': active_task_store,
        'execution_allocation_id': execution_allocation_id,
        'barrier': barrier,
    }


async def _list_all_batch_items(control_url: str, execution_id: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        suffix = f'&cursor={urllib.parse.quote(cursor)}' if cursor else ''
        page = await request_json(
            'GET',
            f'{control_url}/v1/executions/{execution_id}/batch/items?limit=500{suffix}',
        )
        items.extend(page['items'])
        cursor = page.get('next_cursor')
        if not cursor:
            return items


async def _execution_facts(
    control_url: str,
    execution_ids: set[str],
) -> dict[str, dict[str, Any]]:
    facts: dict[str, dict[str, Any]] = {}
    offset = 0
    while len(facts) < len(execution_ids):
        page = await request_json(
            'GET',
            f'{control_url}/v1/executions?limit=500&offset={offset}',
        )
        for item in page['items']:
            if item['execution_id'] in execution_ids:
                facts[item['execution_id']] = item
        offset += len(page['items'])
        if not page['items'] or offset >= page['total']:
            break
    assert set(facts) == execution_ids, {'missing_execution_ids': sorted(execution_ids - set(facts))[:20]}
    return facts


async def _manager_session(control_url: str, manager_id: str) -> str:
    managers = await request_json('GET', f'{control_url}/v1/managers')
    manager = next((item for item in managers['items'] if item['manager_id'] == manager_id), None)
    assert manager is not None, manager_id
    return manager['session_id']


async def _wait_manager_session_replaced(
    control_url: str,
    *,
    manager_id: str,
    previous_session_id: str,
    timeout: float = 300,
) -> str:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        managers = await request_json('GET', f'{control_url}/v1/managers')
        manager = next((item for item in managers['items'] if item['manager_id'] == manager_id), None)
        if manager is not None and manager['healthy'] and manager['session_id'] != previous_session_id:
            return manager['session_id']
        await asyncio.sleep(1)
    raise TimeoutError(f'manager session was not replaced: {manager_id}')


def _manager_id_for_pod(allocation_id: str, pod_name: str) -> str:
    ordinal = pod_name.rsplit('-', 1)[-1]
    assert ordinal.isdigit(), pod_name
    return f'{allocation_id}-{int(ordinal)}'


def _delete_dynamic_manager_pod(
    kubectl: str,
    namespace: str,
    *,
    release_name: str,
    workload: str,
) -> tuple[str, str]:
    payload = json.loads(
        run(
            [
                kubectl,
                '-n',
                namespace,
                'get',
                'pods',
                '-l',
                (
                    f'app.kubernetes.io/managed-by=nexus,app.kubernetes.io/instance={release_name},'
                    f'nexus.io/workload={workload}'
                ),
                '-o',
                'json',
            ],
            capture=True,
        ).stdout
    )
    ready = [
        item
        for item in payload['items']
        if any(
            condition.get('type') == 'Ready' and condition.get('status') == 'True'
            for condition in item['status'].get('conditions', [])
        )
    ]
    assert ready, payload
    pod = sorted(ready, key=lambda item: item['metadata']['name'])[0]
    name = pod['metadata']['name']
    uid = pod['metadata']['uid']
    run([kubectl, '-n', namespace, 'delete', 'pod', name, '--wait=false'])
    return name, uid


async def _wait_pod_replaced(
    kubectl: str,
    namespace: str,
    *,
    name: str,
    previous_uid: str,
    timeout: float = 300,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        result = run(
            [kubectl, '-n', namespace, 'get', 'pod', name, '-o', 'json'],
            capture=True,
            check=False,
        )
        if result.returncode == 0:
            pod = json.loads(result.stdout)
            ready = any(
                condition.get('type') == 'Ready' and condition.get('status') == 'True'
                for condition in pod.get('status', {}).get('conditions', [])
            )
            if pod['metadata']['uid'] != previous_uid and ready:
                return
        await asyncio.sleep(1)
    raise TimeoutError(f'pod was not replaced: {name}')


def _allocation_workload_image(
    kubectl: str,
    namespace: str,
    *,
    allocation_id: str,
) -> str:
    workload_name, _ = allocation_resource_names(allocation_id)
    payload = json.loads(
        run(
            [kubectl, '-n', namespace, 'get', 'statefulset', workload_name, '-o', 'json'],
            capture=True,
        ).stdout
    )
    containers = payload['spec']['template']['spec']['containers']
    manager = next(container for container in containers if container['name'] == 'instance-manager')
    return str(manager['image'])


async def _run_single_live_smoke(
    control_url: str,
    app_url: str,
    *,
    source_ref: dict[str, Any],
    nexus_wheel_ref: dict[str, Any],
    uv_lock_ref: dict[str, Any],
) -> None:
    single = await request_json(
        'POST',
        f'{control_url}/v1/submissions/single',
        body=_typed_single_live_submission(
            source_ref=source_ref,
            nexus_wheel_ref=nexus_wheel_ref,
            uv_lock_ref=uv_lock_ref,
            run_input={
                'imports': ['devcloud_source_package_fixture'],
                'task_name': 'distributed_source_package_echo',
                'task_config': {'index': 999},
            },
            live=False,
        ),
    )
    finished = await poll_execution(
        control_url,
        single['execution_id'],
        lambda item: item['status'] == 'succeeded',
        timeout=900,
    )
    result = await resolved_execution_result(app_url, finished)
    assert result['task_output']['marker'] == 'source-package-ok:999', result
    await _wait_allocations_released(control_url, single['execution_id'])

    live = await request_json(
        'POST',
        f'{control_url}/v1/submissions/live',
        body=_typed_single_live_submission(
            source_ref=source_ref,
            nexus_wheel_ref=nexus_wheel_ref,
            uv_lock_ref=uv_lock_ref,
            run_input=_live_run_input(),
            live=True,
        ),
    )
    await poll_execution(
        control_url,
        live['execution_id'],
        lambda item: item['status'] == 'waiting_input',
        timeout=900,
    )
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
            'payload': {'text': 'dynamic-live-ok'},
            'idempotency_key': f'{live["execution_id"]}-input',
        },
    )
    await poll_execution(control_url, live['execution_id'], lambda item: item['status'] == 'succeeded', timeout=900)
    await _wait_allocations_released(control_url, live['execution_id'])


async def _run_image_smoke(
    control_url: str,
    app_url: str,
    kubectl: str,
    namespace: str,
    *,
    image_version: str,
    expected_image: str,
) -> dict[str, Any]:
    capabilities = await request_json('GET', f'{control_url}/v1/capabilities')
    assert capabilities['workload_image'] == {
        'enabled': True,
        'repository': workload_image_repository(expected_image),
        'base_image': capabilities['workload_image']['base_image'],
    }, capabilities
    assert resolve_workload_image(capabilities['workload_image']['base_image'], image_version) == expected_image
    system_before = await request_json('GET', f'{control_url}/v1/system/status')

    single = await request_json(
        'POST',
        f'{control_url}/v1/submissions/single',
        body=image_single_submission(
            image_version=image_version,
            idempotency_key=f'devcloud-image-single-{uuid.uuid4().hex}',
        ),
    )
    single_allocations = await _wait_allocations_ready(control_url, single['execution_id'])
    assert set(single_allocations) == {'execution'}, single_allocations
    assert 'dependency_build' not in single_allocations['execution'], single_allocations
    single_image = _allocation_workload_image(
        kubectl,
        namespace,
        allocation_id=single_allocations['execution']['allocation_id'],
    )
    assert single_image == expected_image
    single_terminal = await poll_execution(
        control_url,
        single['execution_id'],
        lambda item: item['status'] == 'succeeded',
        timeout=900,
    )
    single_result = await resolved_execution_result(app_url, single_terminal)
    assert single_result['task_output'] == {'index': 999, 'marker': 'image-mode-ok:999'}, single_result
    await _wait_allocations_released(control_url, single['execution_id'])

    batch = await request_json(
        'POST',
        f'{control_url}/v1/submissions/batch',
        body=image_batch_submission(
            image_version=image_version,
            idempotency_key=f'devcloud-image-batch-{uuid.uuid4().hex}',
        ),
    )
    batch_allocations = await _wait_allocations_ready(control_url, batch['execution_id'])
    assert set(batch_allocations) == {'orchestration', 'execution'}, batch_allocations
    allocation_images = {
        workload: _allocation_workload_image(
            kubectl,
            namespace,
            allocation_id=allocation['allocation_id'],
        )
        for workload, allocation in batch_allocations.items()
    }
    assert allocation_images == {'orchestration': expected_image, 'execution': expected_image}
    assert all('dependency_build' not in allocation for allocation in batch_allocations.values())
    await poll_execution(
        control_url,
        batch['execution_id'],
        lambda item: item['status'] == 'succeeded',
        timeout=1200,
    )
    summary = await request_json('GET', f'{control_url}/v1/executions/{batch["execution_id"]}/batch')
    assert summary['summary'] == {
        'total': IMAGE_SMOKE_ITEM_COUNT,
        'succeeded': IMAGE_SMOKE_ITEM_COUNT,
        'failed': 0,
        'killed': 0,
        'lost': 0,
        'expired': 0,
        'cancelled': 0,
    }, summary
    assert summary['task_list_ref']['kind'] == 'task_list', summary
    items = await _list_all_batch_items(control_url, batch['execution_id'])
    assert len(items) == IMAGE_SMOKE_ITEM_COUNT
    assert len({item['item_id'] for item in items}) == IMAGE_SMOKE_ITEM_COUNT
    assert all(item['status'] == 'succeeded' for item in items)

    async def resolve_output(item: dict[str, Any]) -> tuple[int, str]:
        child = await request_json('GET', f'{control_url}/v1/executions/{item["current_child_execution_id"]}')
        result = await resolved_execution_result(app_url, child)
        output = result['task_output']
        return int(output['index']), str(output['marker'])

    outputs = await asyncio.gather(*(resolve_output(item) for item in items))
    assert sorted(outputs) == [(index, f'image-mode-ok:{index}') for index in range(IMAGE_SMOKE_ITEM_COUNT)]
    await _wait_allocations_released(control_url, batch['execution_id'])

    missing_version = f'missing-{uuid.uuid4().hex[:16]}'
    missing_payload = image_single_submission(
        image_version=missing_version,
        idempotency_key=f'devcloud-image-missing-{uuid.uuid4().hex}',
    )
    missing_payload['provisioning_timeout_seconds'] = 30
    missing_payload['queue_ttl_seconds'] = 180
    missing_payload['execution_ttl_seconds'] = 180
    missing = await request_json(
        'POST',
        f'{control_url}/v1/submissions/single',
        body=missing_payload,
    )
    missing_terminal = await poll_execution(
        control_url,
        missing['execution_id'],
        lambda item: item['status'] == 'failed',
        timeout=180,
    )
    missing_events = await request_json(
        'GET',
        f'{control_url}/v1/executions/{missing["execution_id"]}/control-events',
    )
    failure_events = [
        event
        for event in missing_events['items']
        if event['type'] == 'execution.status.changed'
        and event.get('payload', {}).get('reason') == 'EnvironmentProvisioningFailed'
    ]
    assert len(failure_events) == 1, missing_events
    await _wait_allocations_released(control_url, missing['execution_id'])
    system_after = await request_json('GET', f'{control_url}/v1/system/status')
    assert system_after['dependency_builds']['total'] == system_before['dependency_builds']['total']

    return {
        'image_version': image_version,
        'resolved_image': expected_image,
        'capabilities': capabilities['workload_image'],
        'dependency_builds_before': system_before['dependency_builds'],
        'dependency_builds_after': system_after['dependency_builds'],
        'single': {
            'execution_id': single['execution_id'],
            'image': single_image,
            'marker': single_result['task_output']['marker'],
        },
        'batch': {
            'execution_id': batch['execution_id'],
            'images': allocation_images,
            'summary': summary['summary'],
            'unique_item_count': len({item['item_id'] for item in items}),
            'unique_output_count': len(set(outputs)),
        },
        'missing_image': {
            'execution_id': missing['execution_id'],
            'image_version': missing_version,
            'status': missing_terminal['status'],
            'failure_reason': failure_events[0]['payload']['reason'],
        },
    }


async def _run_cli_smoke(
    control_url: str,
    app_url: str,
    *,
    source_root: Path,
) -> dict[str, Any]:
    single_config, live_config = _write_cli_smoke_inputs(source_root)
    common = [
        '--environment',
        DYNAMIC_ENVIRONMENT_ID,
        '--manager-cpu-cores',
        '0.25',
        '--manager-memory-gib',
        '1',
        '--project-root',
        str(ROOT),
        '--source',
        str(source_root),
        '--control-url',
        control_url,
        '--provisioning-timeout-seconds',
        '600',
        '--queue-ttl-seconds',
        '600',
        '--execution-ttl-seconds',
        '1800',
        '--stop-timeout-seconds',
        '120',
        '--host-env',
        'HOST_ENV_MVP_PROBE=ok',
    ]

    single = await asyncio.to_thread(
        _run_submit_cli,
        [
            'single',
            str(single_config),
            *common,
            '--idempotency-key',
            f'devcloud-cli-single-{uuid.uuid4().hex}',
            '--wait',
        ],
        timeout=1800,
    )
    single_result = single['result']
    assert len(single['accepted']) == 1, single
    assert single['accepted'][0]['execution_id'] == single_result['execution_id'], single
    assert single_result['terminal']['status'] == 'succeeded', single_result
    single_output = await request_json('GET', single_result['result_url'])
    assert single_output['task_output']['value'] == 'ok', single_output
    await _wait_allocations_released(control_url, single_result['execution_id'])

    live = await asyncio.to_thread(
        _run_submit_cli,
        ['live', str(live_config), *common, '--idempotency-key', f'devcloud-cli-live-{uuid.uuid4().hex}'],
        timeout=1800,
    )
    live_result = live['result']
    assert not live['accepted'], live
    await poll_execution(
        control_url,
        live_result['execution_id'],
        lambda item: item['status'] == 'waiting_input',
        timeout=900,
    )
    required = await read_sse_until(live_result['events_url'], 'input.required')
    assert required['type'] == 'input.required', required
    await request_json(
        'POST',
        live_result['command_url'],
        body={
            'type': 'input.submit',
            'payload': {'text': 'devcloud-cli-live-ok'},
            'idempotency_key': f'{live_result["execution_id"]}-input',
        },
    )
    live_terminal = await poll_execution(
        control_url,
        live_result['execution_id'],
        lambda item: item['status'] == 'succeeded',
        timeout=900,
    )
    live_output = await request_json('GET', live_result['result_url'])
    assert live_output['agent_output']['choice'] == 'devcloud-cli-live-ok', live_output
    await _wait_allocations_released(control_url, live_result['execution_id'])

    batch = await asyncio.to_thread(
        _run_submit_cli,
        [
            'batch',
            '--generator',
            SOURCE_FIXTURE_MODULE,
            '--generation',
            'local',
            '--batch-manager-cpu-cores',
            '0.25',
            '--batch-manager-memory-gib',
            '1',
            '--manager-count',
            '2',
            '--manager-cpu-cores',
            '0.25',
            '--manager-memory-gib',
            '1',
            '--manager-slots',
            '2',
            '--max-inflight',
            '4',
            '--timeout-per-run',
            '120',
            *common,
            '--idempotency-key',
            f'devcloud-cli-batch-{uuid.uuid4().hex}',
            '--wait',
        ],
        timeout=2400,
    )
    batch_result = batch['result']
    assert len(batch['accepted']) == 1, batch
    assert batch['accepted'][0]['execution_id'] == batch_result['execution_id'], batch
    assert batch_result['terminal']['status'] == 'succeeded', batch_result
    summary = await request_json('GET', batch_result['batch_url'])
    assert summary['summary']['total'] == SOURCE_SMOKE_ITEM_COUNT, summary
    assert summary['summary']['succeeded'] == SOURCE_SMOKE_ITEM_COUNT, summary
    batch_output = await request_json('GET', batch_result['result_url'])
    assert batch_output['batch']['succeeded'] == SOURCE_SMOKE_ITEM_COUNT, batch_output
    await _wait_allocations_released(control_url, batch_result['execution_id'])

    return {
        'single': {
            'args': single['args'],
            'accepted': single['accepted'][0],
            'execution_id': single_result['execution_id'],
            'terminal_status': single_result['terminal']['status'],
            'host_env_probe': single_output['task_output']['value'],
        },
        'live': {
            'args': live['args'],
            'execution_id': live_result['execution_id'],
            'terminal_status': live_terminal['status'],
            'choice': live_output['agent_output']['choice'],
        },
        'batch': {
            'args': batch['args'],
            'accepted': batch['accepted'][0],
            'execution_id': batch_result['execution_id'],
            'terminal_status': batch_result['terminal']['status'],
            'summary': summary['summary'],
        },
    }


async def _run_dataset_batch_case(
    control_url: str,
    app_url: str,
    kubectl: str,
    namespace: str,
    *,
    release_name: str,
) -> dict[str, Any]:
    submission = await request_json(
        'POST',
        f'{control_url}/v1/submissions/batch',
        body=dataset_batch_submission(idempotency_key=f'devcloud-dataset-{uuid.uuid4().hex}'),
    )
    execution_id = submission['execution_id']
    assert submission['credential_kinds'] == ['wedata'], submission
    parent = await request_json('GET', f'{control_url}/v1/executions/{execution_id}')
    generation = parent['spec']['generation']
    assert generation['mode'] == 'remote_materialize', generation
    assert generation['source']['task_list_version'] == 2, generation
    assert 'data_list' not in generation, generation
    await poll_execution(
        control_url,
        execution_id,
        lambda item: item['status'] == 'succeeded',
        timeout=1200,
    )
    summary = await request_json('GET', f'{control_url}/v1/executions/{execution_id}/batch')
    assert summary['summary']['total'] == DATASET_ITEM_COUNT, summary
    assert summary['summary']['succeeded'] == DATASET_ITEM_COUNT, summary
    assert sum(value for key, value in summary['summary'].items() if key not in {'total', 'succeeded'}) == 0, summary
    manifest = summary['task_list_manifest']
    assert manifest['record_version'] == 2, manifest
    assert manifest['item_count'] == DATASET_ITEM_COUNT, manifest
    assert manifest['datasets'] == [
        {
            'input_uri': DATASET_URI,
            'resolved_uri': 'devcloud/echo@v1?split=test',
        }
    ], manifest

    items = await _list_all_batch_items(control_url, execution_id)
    assert len(items) == DATASET_ITEM_COUNT
    assert [item['item_id'] for item in items] == [f'dataset-{index:04d}' for index in range(DATASET_ITEM_COUNT)]
    assert all(item['status'] == 'succeeded' for item in items)

    async def read_result(item: dict[str, Any]) -> tuple[int, str, list[str]]:
        child = await request_json(
            'GET',
            f'{control_url}/v1/executions/{item["current_child_execution_id"]}',
        )
        result = await resolved_execution_result(app_url, child)
        output = result['task_output']
        return int(output['index']), str(output['marker']), list(output['paths'])

    outputs = await asyncio.gather(*(read_result(item) for item in items))
    assert outputs == [(index, f'remote-dataset-ok:{index}', ['marker.txt']) for index in range(DATASET_ITEM_COUNT)]
    with PortForward(
        kubectl=kubectl,
        namespace=namespace,
        resource=f'service/{release_name}-fake-dataset',
        remote_port=DATASET_SERVICE_PORT,
    ) as dataset_service:
        await wait_http_ok(f'{dataset_service.url}/health', timeout=30)
        stats = await request_json('GET', f'{dataset_service.url}/stats')
    minimum_page_count = (DATASET_ITEM_COUNT + DATASET_PAGE_SIZE - 1) // DATASET_PAGE_SIZE
    assert stats['pagination_requests'] >= minimum_page_count, stats
    assert stats['sign_requests'] >= DATASET_ITEM_COUNT, stats
    assert stats['range_requests'] >= DATASET_ITEM_COUNT, stats
    assert stats['full_blob_requests'] == 0, stats
    await _wait_allocations_released(control_url, execution_id)
    return {
        'execution_id': execution_id,
        'credential_kinds': submission['credential_kinds'],
        'input_uri': DATASET_URI,
        'resolved_uri': manifest['datasets'][0]['resolved_uri'],
        'item_count': DATASET_ITEM_COUNT,
        'task_list_record_version': manifest['record_version'],
        'data_list_present': False,
        'minimum_page_count': minimum_page_count,
        'service_stats': stats,
        'summary': summary['summary'],
    }


async def _run_local_dataset_batch_case(
    control_url: str,
    app_url: str,
    kubectl: str,
    namespace: str,
    *,
    release_name: str,
    source_root: Path,
    dataset_root: Path,
    tdsql_trace: TDSQLTraceConfig,
    barrier_name: str,
) -> dict[str, Any]:
    cli = await asyncio.to_thread(
        _run_submit_cli,
        [
            'batch',
            '--generation',
            'local',
            '--data',
            str(dataset_root),
            '--batch-manager-cpu-cores',
            '0.25',
            '--batch-manager-memory-gib',
            '1',
            '--manager-count',
            '1',
            '--manager-cpu-cores',
            '0.25',
            '--manager-memory-gib',
            '1',
            '--manager-slots',
            '2',
            '--max-inflight',
            '2',
            '--timeout-per-run',
            '120',
            '--environment',
            DYNAMIC_ENVIRONMENT_ID,
            '--project-root',
            str(ROOT),
            '--source',
            str(source_root),
            '--control-url',
            control_url,
            '--provisioning-timeout-seconds',
            '600',
            '--queue-ttl-seconds',
            '600',
            '--execution-ttl-seconds',
            '1800',
            '--stop-timeout-seconds',
            '120',
            '--idempotency-key',
            f'devcloud-local-dataset-{uuid.uuid4().hex}',
        ],
        timeout=2400,
    )
    assert not cli['accepted'], cli
    execution_id = cli['result']['execution_id']
    parent = await request_json('GET', f'{control_url}/v1/executions/{execution_id}')
    generation = parent['spec']['generation']
    assert generation['mode'] == 'local_task_list', generation
    task_list = generation['task_list']
    data_list = generation['data_list']
    assert task_list['manifest']['record_version'] == 3, task_list
    assert task_list['manifest']['item_count'] == len(LOCAL_DATASET_ITEM_IDS), task_list
    assert task_list['manifest']['datasets'] == [], task_list
    assert data_list['manifest']['frame_count'] == len(LOCAL_DATASET_ITEM_IDS), data_list
    assert data_list['ref']['kind'] == 'data_list', data_list
    if str(dataset_root) in json.dumps(parent['spec'], sort_keys=True):
        raise AssertionError('local Dataset path leaked into parent execution spec')

    artifacts: dict[str, dict[str, Any]] = {}
    for kind, descriptor in (('task_list', task_list), ('data_list', data_list)):
        artifact = await request_json(
            'GET',
            f'{control_url}/v1/artifacts/{descriptor["ref"]["artifact_id"]}'
            f'?environment_id={urllib.parse.quote(DYNAMIC_ENVIRONMENT_ID)}',
        )
        assert artifact['status'] == 'ready', artifact
        assert artifact['ref'] == descriptor['ref'], artifact
        artifacts[kind] = artifact['ref']

    allocations = await _wait_allocations_ready(control_url, execution_id, timeout=1200)
    assert set(allocations) == {'orchestration', 'execution'}, allocations
    await _wait_batch_summary(
        control_url,
        execution_id,
        lambda value: (
            value['generation_status'] == 'complete' and value['materialized_count'] == len(LOCAL_DATASET_ITEM_IDS)
        ),
        timeout=1200,
    )
    observed = await _observe_local_dataset_active_state(
        kubectl,
        namespace,
        control_url,
        execution_id,
        orchestration_allocation_id=allocations['orchestration']['allocation_id'],
        execution_allocation_id=allocations['execution']['allocation_id'],
        release_name=release_name,
        barrier_name=barrier_name,
        dataset_root=dataset_root,
        tdsql_trace=tdsql_trace,
    )
    terminal_cleanup = await _wait_local_dataset_terminal_cleanup(
        kubectl,
        namespace,
        control_url,
        execution_id,
        release_name=release_name,
        tdsql_trace=tdsql_trace,
        forbidden_path=dataset_root,
        allocation_id=observed['execution_allocation_id'],
    )
    terminal = terminal_cleanup['terminal']
    terminal_task_store = terminal_cleanup['task_store']
    assignment_cleanup = terminal_cleanup['assignment']
    summary = await _wait_batch_summary(
        control_url,
        execution_id,
        lambda value: value['summary']['succeeded'] == len(LOCAL_DATASET_ITEM_IDS),
        timeout=1200,
    )
    assert summary['task_list_manifest']['record_version'] == 3, summary

    items = await _list_all_batch_items(control_url, execution_id)
    assert [item['item_id'] for item in items] == list(LOCAL_DATASET_ITEM_IDS), items
    assert all(item['status'] == 'succeeded' for item in items), items

    async def read_result(item: dict[str, Any]) -> dict[str, Any]:
        child = await request_json(
            'GET',
            f'{control_url}/v1/executions/{item["current_child_execution_id"]}',
        )
        child_spec = child['spec']
        if 'run_input_ref' not in child_spec or 'run_input' in child_spec:
            raise AssertionError({'item_id': item['item_id'], 'spec': child_spec})
        if str(dataset_root) in json.dumps(child_spec, sort_keys=True):
            raise AssertionError(f'local Dataset path leaked into child spec: {item["item_id"]}')
        output = (await resolved_execution_result(app_url, child))['task_output']
        assert output['marker'] == LOCAL_DATASET_MARKERS[item['item_id']], output
        assert output['paths'] == ['marker.txt'], output
        return {
            'item_id': item['item_id'],
            'child_execution_id': child['execution_id'],
            'run_input_ref': child_spec['run_input_ref'],
            'marker': output['marker'],
            'paths': output['paths'],
        }

    outputs = await asyncio.gather(*(read_result(item) for item in items))
    await _wait_allocations_released(control_url, execution_id)
    return {
        'args': cli['args'],
        'execution_id': execution_id,
        'terminal_status': terminal['status'],
        'summary': summary['summary'],
        'task_list_manifest': summary['task_list_manifest'],
        'data_list_manifest': data_list['manifest'],
        'artifacts': artifacts,
        'task_store': {
            'active': observed['active_task_store'],
            'after_terminal': terminal_task_store,
        },
        'workspace': observed['workspace'],
        'assignment_cleanup': assignment_cleanup,
        'barrier': observed['barrier'],
        'outputs': outputs,
    }


async def _run_harbor_dataset_batch_case(
    control_url: str,
    app_url: str,
    kubectl: str,
    namespace: str,
    *,
    release_name: str,
    source_root: Path,
    direct_root: Path,
    fetched_root: Path,
    tdsql_trace: TDSQLTraceConfig,
    direct_barrier_name: str,
    fetched_barrier_name: str,
) -> dict[str, Any]:
    """Run Harbor Reader and fetched-local-Dataset batches through distributed data-list."""

    async def run_one(
        *,
        label: str,
        data_value: str,
        forbidden_path: Path,
        barrier_name: str,
    ) -> dict[str, Any]:
        cli = await asyncio.to_thread(
            _run_submit_cli,
            [
                'batch',
                '--generation',
                'local',
                '--task-list-version',
                '2',
                '--data',
                data_value,
                '--use',
                'harbor_bridge:mode=nop',
                '--batch-manager-cpu-cores',
                '0.25',
                '--batch-manager-memory-gib',
                '1',
                '--manager-count',
                '1',
                '--manager-cpu-cores',
                '0.25',
                '--manager-slots',
                '2',
                '--max-inflight',
                '2',
                '--timeout-per-run',
                '120',
                '--environment',
                DYNAMIC_ENVIRONMENT_ID,
                '--project-root',
                str(ROOT),
                '--source',
                str(source_root),
                '--control-url',
                control_url,
                '--provisioning-timeout-seconds',
                '600',
                '--queue-ttl-seconds',
                '600',
                '--execution-ttl-seconds',
                '1800',
                '--stop-timeout-seconds',
                '120',
                '--idempotency-key',
                f'devcloud-harbor-{label}-{uuid.uuid4().hex}',
            ],
            timeout=2400,
        )
        assert not cli['accepted'], cli
        execution_id = cli['result']['execution_id']
        parent = await request_json('GET', f'{control_url}/v1/executions/{execution_id}')
        generation = parent['spec']['generation']
        assert generation['mode'] == 'local_task_list', generation
        task_list = generation['task_list']
        data_list = generation['data_list']
        assert task_list['manifest']['record_version'] == 3, task_list
        assert task_list['manifest']['item_count'] == len(HARBOR_DATASET_ITEM_IDS), task_list
        assert task_list['manifest']['datasets'] == [], task_list
        assert data_list['manifest']['frame_count'] == len(HARBOR_DATASET_ITEM_IDS), data_list
        assert data_list['ref']['kind'] == 'data_list', data_list
        serialized_spec = json.dumps(parent['spec'], sort_keys=True)
        if str(forbidden_path) in serialized_spec:
            raise AssertionError(f'Harbor Dataset path leaked into parent execution spec: {forbidden_path}')

        artifacts: dict[str, dict[str, Any]] = {}
        for kind, descriptor in (('task_list', task_list), ('data_list', data_list)):
            artifact = await request_json(
                'GET',
                f'{control_url}/v1/artifacts/{descriptor["ref"]["artifact_id"]}'
                f'?environment_id={urllib.parse.quote(DYNAMIC_ENVIRONMENT_ID)}',
            )
            assert artifact['status'] == 'ready', artifact
            assert artifact['ref'] == descriptor['ref'], artifact
            artifacts[kind] = artifact['ref']

        allocations = await _wait_allocations_ready(control_url, execution_id, timeout=1200)
        assert set(allocations) == {'orchestration', 'execution'}, allocations
        await _wait_batch_summary(
            control_url,
            execution_id,
            lambda value: (
                value['generation_status'] == 'complete' and value['materialized_count'] == len(HARBOR_DATASET_ITEM_IDS)
            ),
            timeout=1200,
        )
        assignment_before_release = await asyncio.to_thread(
            _assert_assignment_task_files_clean,
            kubectl,
            namespace,
            allocation_id=allocations['execution']['allocation_id'],
            require_files=True,
        )
        observed = await _observe_local_dataset_active_state(
            kubectl,
            namespace,
            control_url,
            execution_id,
            orchestration_allocation_id=allocations['orchestration']['allocation_id'],
            execution_allocation_id=allocations['execution']['allocation_id'],
            release_name=release_name,
            barrier_name=barrier_name,
            dataset_root=forbidden_path,
            tdsql_trace=tdsql_trace,
            expected_item_ids=HARBOR_DATASET_ITEM_IDS,
            expected_barrier_arrivals=HARBOR_DATASET_BARRIER_EXPECTED_ARRIVALS,
        )
        terminal_cleanup = await _wait_local_dataset_terminal_cleanup(
            kubectl,
            namespace,
            control_url,
            execution_id,
            release_name=release_name,
            tdsql_trace=tdsql_trace,
            forbidden_path=forbidden_path,
            allocation_id=observed['execution_allocation_id'],
            expected_item_ids=HARBOR_DATASET_ITEM_IDS,
        )
        summary = await _wait_batch_summary(
            control_url,
            execution_id,
            lambda value: value['summary']['succeeded'] == len(HARBOR_DATASET_ITEM_IDS),
            timeout=1200,
        )
        assert summary['task_list_manifest']['record_version'] == 3, summary

        items = await _list_all_batch_items(control_url, execution_id)
        assert [item['item_id'] for item in items] == list(HARBOR_DATASET_ITEM_IDS), items
        assert all(item['status'] == 'succeeded' for item in items), items

        async def read_result(item: dict[str, Any]) -> dict[str, Any]:
            child = await request_json(
                'GET',
                f'{control_url}/v1/executions/{item["current_child_execution_id"]}',
            )
            child_spec = child['spec']
            if 'run_input_ref' not in child_spec or 'run_input' in child_spec:
                raise AssertionError({'item_id': item['item_id'], 'spec': child_spec})
            if str(forbidden_path) in json.dumps(child_spec, sort_keys=True):
                raise AssertionError(f'Harbor Dataset path leaked into child spec: {item["item_id"]}')
            output = (await resolved_execution_result(app_url, child))['task_output']
            expected_marker = HARBOR_DATASET_MARKERS[item['item_id']]
            assert float(output['reward']) == 1.0, output
            assert float(output['rewards']['reward']) == 1.0, output
            assert output['reward_metadata']['item_id'] == item['item_id'], output
            assert output['reward_metadata']['marker'] == expected_marker, output
            return {
                'item_id': item['item_id'],
                'child_execution_id': child['execution_id'],
                'run_input_ref': child_spec['run_input_ref'],
                'reward': output['reward'],
                'marker': output['reward_metadata']['marker'],
            }

        outputs = await asyncio.gather(*(read_result(item) for item in items))
        await _wait_allocations_released(control_url, execution_id)
        return {
            'args': cli['args'],
            'execution_id': execution_id,
            'terminal_status': terminal_cleanup['terminal']['status'],
            'summary': summary['summary'],
            'task_list_manifest': summary['task_list_manifest'],
            'data_list_manifest': data_list['manifest'],
            'artifacts': artifacts,
            'task_store': {
                'active': observed['active_task_store'],
                'after_terminal': terminal_cleanup['task_store'],
            },
            'workspace': observed['workspace'],
            'assignment_files_before_release': assignment_before_release,
            'assignment_cleanup_after_terminal': terminal_cleanup['assignment'],
            'barrier': observed['barrier'],
            'outputs': outputs,
        }

    assert direct_root.is_dir()
    assert fetched_root.is_dir()
    return {
        'direct_reader': await run_one(
            label='direct',
            data_value=f't3-harbor://{direct_root}?flat=true',
            forbidden_path=direct_root,
            barrier_name=direct_barrier_name,
        ),
        'fetched_local_zip': await run_one(
            label='fetched',
            data_value=str(fetched_root),
            forbidden_path=fetched_root,
            barrier_name=fetched_barrier_name,
        ),
    }


async def _run_source_batch_case(
    control_url: str,
    app_url: str,
    kubectl: str,
    namespace: str,
    *,
    release_name: str,
    source_ref: dict[str, Any],
    nexus_wheel_ref: dict[str, Any],
    uv_lock_ref: dict[str, Any],
    replace_workload: str,
) -> dict[str, Any]:
    submission = await request_json(
        'POST',
        f'{control_url}/v1/submissions/batch',
        body=source_smoke_submission(
            source_ref=source_ref,
            nexus_wheel_ref=nexus_wheel_ref,
            uv_lock_ref=uv_lock_ref,
            idempotency_key=f'devcloud-source-smoke-{replace_workload}-{uuid.uuid4().hex}',
        ),
    )
    execution_id = submission['execution_id']
    allocations = await _wait_allocations_ready(control_url, execution_id)
    assert allocations['orchestration']['desired_manager_count'] == 1, allocations
    assert allocations['execution']['desired_manager_count'] == 2, allocations
    before = await _wait_batch_summary(
        control_url,
        execution_id,
        lambda value: value['generation_status'] == 'complete' and value['materialized_count'] > 0,
        timeout=900,
    )
    task_list_ref = before['task_list_ref']
    name, uid = _delete_dynamic_manager_pod(
        kubectl,
        namespace,
        release_name=release_name,
        workload=replace_workload,
    )
    await _wait_pod_replaced(kubectl, namespace, name=name, previous_uid=uid)
    await poll_execution(control_url, execution_id, lambda item: item['status'] == 'succeeded', timeout=1800)
    summary = await request_json('GET', f'{control_url}/v1/executions/{execution_id}/batch')
    assert summary['summary'] == {
        'total': SOURCE_SMOKE_ITEM_COUNT,
        'succeeded': SOURCE_SMOKE_ITEM_COUNT,
        'failed': 0,
        'killed': 0,
        'lost': 0,
        'expired': 0,
        'cancelled': 0,
    }, summary
    assert summary['task_list_ref'] == task_list_ref, summary
    items = await _list_all_batch_items(control_url, execution_id)
    assert len(items) == SOURCE_SMOKE_ITEM_COUNT
    assert len({item['item_id'] for item in items}) == SOURCE_SMOKE_ITEM_COUNT
    assert all(item['status'] == 'succeeded' for item in items)

    async def resolve_marker(item: dict[str, Any]) -> str:
        child = await poll_execution(
            control_url,
            item['current_child_execution_id'],
            lambda value: value['status'] == 'succeeded',
        )
        result = await resolved_execution_result(app_url, child)
        return result['task_output']['marker']

    markers = await asyncio.gather(*(resolve_marker(item) for item in items))
    assert set(markers) == {f'source-package-ok:{index}' for index in range(SOURCE_SMOKE_ITEM_COUNT)}
    await _wait_allocations_released(control_url, execution_id)
    return summary


async def _run_heavy_batch_case(
    control_url: str,
    kubectl: str,
    namespace: str,
    *,
    release_name: str,
    task_list_ref: dict[str, Any],
    manifest: dict[str, Any],
    recovery_mode: str | None,
) -> dict[str, Any]:
    assert recovery_mode in {None, 'execution', 'batch'}
    reset_mysql_status(kubectl, namespace, release_name)
    db_before = mysql_status(kubectl, namespace, release_name)
    started = asyncio.get_running_loop().time()
    submission = await request_json(
        'POST',
        f'{control_url}/v1/submissions/batch',
        body=heavy_batch_submission(
            task_list_ref=task_list_ref,
            manifest=manifest,
            idempotency_key=f'devcloud-heavy-{recovery_mode or "normal"}-{uuid.uuid4().hex}',
            retry_failed=1 if recovery_mode == 'execution' else 0,
        ),
    )
    execution_id = submission['execution_id']
    allocations = await _wait_allocations_ready(control_url, execution_id, timeout=1200)
    child_allocation = allocations['execution']
    assert child_allocation['desired_manager_count'] == HEAVY_MANAGER_COUNT, child_allocation
    pool_id = child_allocation['pool_id']
    all_ready_at = asyncio.get_running_loop().time()
    peak_reserved = 0
    peak_manager_reserved = 0
    replaced = False
    replaced_manager_id: str | None = None
    previous_manager_session_id: str | None = None
    replacement_manager_session_id: str | None = None
    task_list_ref_before_recovery: dict[str, Any] | None = None
    fill_seconds: float | None = None
    peak_target = heavy_peak_target(recovery_mode=recovery_mode)

    if recovery_mode == 'batch':
        before_recovery = await _wait_batch_summary(
            control_url,
            execution_id,
            lambda value: value['generation_status'] == 'complete' and 0 < value['submitted_items'] < HEAVY_ITEM_COUNT,
            timeout=900,
        )
        task_list_ref_before_recovery = before_recovery['task_list_ref']
        current_managers = await request_json('GET', f'{control_url}/v1/managers')
        name, uid = _delete_dynamic_manager_pod(
            kubectl,
            namespace,
            release_name=release_name,
            workload=BATCH_MANAGER_WORKLOAD,
        )
        replaced_manager_id = _manager_id_for_pod(allocations['orchestration']['allocation_id'], name)
        manager_before = next(item for item in current_managers['items'] if item['manager_id'] == replaced_manager_id)
        previous_manager_session_id = manager_before['session_id']
        await _wait_pod_replaced(kubectl, namespace, name=name, previous_uid=uid)
        replacement_manager_session_id = await _wait_manager_session_replaced(
            control_url,
            manager_id=replaced_manager_id,
            previous_session_id=previous_manager_session_id,
        )
        replaced = True

    while True:
        execution = await request_json('GET', f'{control_url}/v1/executions/{execution_id}')
        managers = await request_json('GET', f'{control_url}/v1/managers')
        child_managers = [item for item in managers['items'] if item['pool_id'] == pool_id and item['healthy']]
        assert len(child_managers) <= HEAVY_MANAGER_COUNT, child_managers
        peak_manager_reserved = max(
            peak_manager_reserved, max((item['reserved_slots'] for item in child_managers), default=0)
        )
        reserved = sum(item['reserved_slots'] for item in child_managers)
        peak_reserved = max(peak_reserved, reserved)
        if fill_seconds is None and peak_reserved >= peak_target:
            fill_seconds = asyncio.get_running_loop().time() - all_ready_at
        if recovery_mode == 'execution' and not replaced and peak_reserved >= peak_target:
            name, uid = _delete_dynamic_manager_pod(
                kubectl,
                namespace,
                release_name=release_name,
                workload='execution',
            )
            replaced_manager_id = _manager_id_for_pod(child_allocation['allocation_id'], name)
            manager_before = next(item for item in managers['items'] if item['manager_id'] == replaced_manager_id)
            previous_manager_session_id = manager_before['session_id']
            await _wait_pod_replaced(kubectl, namespace, name=name, previous_uid=uid)
            replacement_manager_session_id = await _wait_manager_session_replaced(
                control_url,
                manager_id=replaced_manager_id,
                previous_session_id=previous_manager_session_id,
            )
            replaced = True
        if execution['status'] in {'succeeded', 'failed', 'killed', 'lost', 'expired'}:
            break
        await asyncio.sleep(1)
    summary = await request_json('GET', f'{control_url}/v1/executions/{execution_id}/batch')
    items = await _list_all_batch_items(control_url, execution_id)
    child_execution_id_values = [item['current_child_execution_id'] for item in items]
    missing_child_items = [item['item_id'] for item in items if not item['current_child_execution_id']]
    child_execution_ids = {str(value) for value in child_execution_id_values if value}
    facts = await _execution_facts(control_url, child_execution_ids)
    if execution['status'] != 'succeeded':
        diagnostics = heavy_child_diagnostics(facts)
        event_samples: dict[str, Any] = {}
        for sample in diagnostics['expired_samples'][:3]:
            child_execution_id = str(sample['execution_id'])
            event_samples[child_execution_id] = await request_json(
                'GET',
                f'{control_url}/v1/executions/{child_execution_id}/control-events?cursor=0&limit=100',
            )
        failure = {
            'parent_execution': {
                'execution_id': execution_id,
                'status': execution['status'],
                'phase': execution.get('phase'),
                'result_json': execution.get('result_json'),
                'error_json': execution.get('error_json'),
            },
            'batch': summary,
            'missing_child_item_ids': missing_child_items[:20],
            'children': diagnostics,
            'event_samples': event_samples,
        }
        failure_path = Path('/tmp/nexus-phase2-devcloud-heavy-failure.json')
        failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding='utf-8')
        print(json.dumps({'devcloud_heavy_failure': failure}, sort_keys=True), flush=True)
        raise AssertionError(f'heavy batch failed; diagnostics: {failure_path}')

    assert summary['summary']['total'] == HEAVY_ITEM_COUNT, summary
    assert summary['summary']['succeeded'] == HEAVY_ITEM_COUNT, summary
    assert sum(value for key, value in summary['summary'].items() if key not in {'total', 'succeeded'}) == 0, summary
    assert len(items) == HEAVY_ITEM_COUNT, len(items)
    assert len({item['item_id'] for item in items}) == HEAVY_ITEM_COUNT
    assert all(item['status'] == 'succeeded' for item in items)
    assert not missing_child_items, missing_child_items[:20]
    assert len(child_execution_ids) == HEAVY_ITEM_COUNT
    assert all(item['status'] == 'succeeded' for item in facts.values())
    result_artifact_ids = [item['result_artifact_id'] for item in facts.values()]
    assert all(result_artifact_ids)
    assert len(set(result_artifact_ids)) == HEAVY_ITEM_COUNT
    extra_assignment_attempts = sum(max(0, int(item['attempt_count']) - 1) for item in facts.values())
    if recovery_mode == 'execution':
        assert extra_assignment_attempts <= HEAVY_SLOTS_PER_MANAGER, extra_assignment_attempts
    else:
        assert extra_assignment_attempts == 0, extra_assignment_attempts
    assert peak_manager_reserved <= HEAVY_SLOTS_PER_MANAGER, peak_manager_reserved
    assert peak_reserved <= HEAVY_MAX_INFLIGHT, peak_reserved
    assert peak_reserved >= peak_target, peak_reserved
    if recovery_mode is not None:
        assert replaced
        assert previous_manager_session_id
        assert replacement_manager_session_id
        assert previous_manager_session_id != replacement_manager_session_id
    parent = await request_json('GET', f'{control_url}/v1/executions/{execution_id}')
    if recovery_mode == 'batch':
        assert parent['attempt_count'] >= 2, parent
        assert summary['task_list_ref'] == task_list_ref_before_recovery, summary
    await _wait_allocations_released(control_url, execution_id, timeout=600)
    db_after = mysql_status(kubectl, namespace, release_name)
    metrics = {
        'execution_id': execution_id,
        'allocation_ids': sorted(item['allocation_id'] for item in allocations.values()),
        'peak_reserved_slots': peak_reserved,
        'peak_reserved_per_manager': peak_manager_reserved,
        'peak_target_slots': peak_target,
        'scheduler_fill_seconds': fill_seconds,
        'total_seconds': asyncio.get_running_loop().time() - started,
        'recovery_mode': recovery_mode,
        'replaced_manager_id': replaced_manager_id,
        'previous_manager_session_id': previous_manager_session_id,
        'replacement_manager_session_id': replacement_manager_session_id,
        'child_execution_count': len(facts),
        'unique_result_artifact_count': len(set(result_artifact_ids)),
        'extra_assignment_attempts': extra_assignment_attempts,
        'mysql': {
            'before': db_before,
            'after': db_after,
            'max_used_connections': db_after['Max_used_connections'],
            'slow_queries_delta': max(0, db_after['Slow_queries'] - db_before['Slow_queries']),
        },
    }
    print(json.dumps({'devcloud_heavy_metrics': metrics}, sort_keys=True), flush=True)
    return metrics


async def run_dynamic_api_tests(
    control_url: str,
    app_url: str,
    kubectl: str,
    namespace: str,
    *,
    release_name: str,
    scenarios: set[str],
    uploaded_refs: list[dict[str, Any]],
    workload_image: str,
    image_mode_version: str | None = None,
    image_mode_image: str | None = None,
    tdsql_trace: TDSQLTraceConfig | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'scenarios': sorted(scenarios),
        'cli_smoke': None,
        'dataset': None,
        'local_dataset': None,
        'harbor': None,
        'source_smoke': None,
        'dependency_build': None,
        'image_smoke': None,
        'heavy_runs': [],
    }
    await ensure_dynamic_environment(control_url, namespace=namespace)
    with tempfile.TemporaryDirectory(prefix='nexus-devcloud-dynamic-submit-') as temporary_dir:
        temporary = Path(temporary_dir)
        source_root: Path | None = None
        source_ref: dict[str, Any] | None = None
        nexus_wheel_ref: dict[str, Any] | None = None
        uv_lock_ref: dict[str, Any] | None = None
        heavy_ref: dict[str, Any] | None = None
        heavy_manifest: dict[str, Any] | None = None
        dataset_root: Path | None = None
        local_dataset_barrier_name: str | None = None
        harbor_root: Path | None = None
        harbor_fetch_source_root: Path | None = None
        harbor_local_root: Path | None = None
        harbor_direct_barrier_name: str | None = None
        harbor_fetched_barrier_name: str | None = None
        if scenarios & {'cli-smoke', 'source-smoke', 'local-dataset', 'harbor'}:
            source_root = temporary / 'source'
            source_root.mkdir()
            for name in ('pyproject.toml', 'uv.lock', 'README.md'):
                shutil.copyfile(ROOT / name, source_root / name)
            (source_root / SOURCE_FIXTURE_MODULE).write_text(_source_fixture_text(), encoding='utf-8')
        if 'local-dataset' in scenarios:
            assert source_root is not None
            local_dataset_barrier_name = f'local-dataset-{uuid.uuid4().hex}'
            dataset_root = write_local_dataset_fixture(
                temporary / 'local-dataset',
                barrier_name=local_dataset_barrier_name,
            )
            assert dataset_root.parent == source_root.parent
            assert not dataset_root.is_relative_to(source_root)
        if 'harbor' in scenarios:
            assert source_root is not None
            harbor_direct_barrier_name = f'harbor-direct-{uuid.uuid4().hex}'
            harbor_fetched_barrier_name = f'harbor-fetched-{uuid.uuid4().hex}'
            harbor_root = write_harbor_dataset_fixture(
                temporary / 'harbor-direct',
                docker_image=workload_image,
                release_name=release_name,
                barrier_name=harbor_direct_barrier_name,
            )
            harbor_fetch_source_root = write_harbor_dataset_fixture(
                temporary / 'harbor-fetch-source',
                docker_image=workload_image,
                release_name=release_name,
                barrier_name=harbor_fetched_barrier_name,
            )
            harbor_local_root = temporary / 'harbor-local'
            fetch_payload = await asyncio.to_thread(
                _run_dataset_fetch_cli,
                f't3-harbor://{harbor_fetch_source_root}?flat=true',
                harbor_local_root,
            )
            fetched_items = sorted(harbor_local_root.glob('items/test/*/item.yaml'))
            assert len(fetched_items) == len(HARBOR_DATASET_ITEM_IDS), fetch_payload
            assert all(
                yaml.safe_load(item.read_text(encoding='utf-8'))['files']['type'] == 'zip' for item in fetched_items
            ), fetch_payload
        if 'source-smoke' in scenarios:
            assert source_root is not None
            source = await asyncio.to_thread(build_source_package, source_root, temporary / 'source.tar.zst')
            wheel = await asyncio.to_thread(_build_wheel, ROOT, temporary / 'wheel')
            uv_lock = await asyncio.to_thread(_prepare_uv_lock, ROOT)
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
                source_ref = await _upload_artifact(
                    session,
                    control_url,
                    source.path,
                    kind='source_package',
                    sha256=source.sha256,
                    size_bytes=source.size_bytes,
                )
                nexus_wheel_ref = await _upload_artifact(
                    session,
                    control_url,
                    wheel.path,
                    kind='nexus_wheel',
                    sha256=wheel.sha256,
                    size_bytes=wheel.size_bytes,
                )
                uv_lock_ref = await _upload_artifact(
                    session,
                    control_url,
                    uv_lock.path,
                    kind='uv_lock',
                    sha256=uv_lock.sha256,
                    size_bytes=uv_lock.size_bytes,
                )
            uploaded_refs.extend((source_ref, nexus_wheel_ref, uv_lock_ref))
        if scenarios & {'heavy', 'execution-recovery', 'batch-recovery'}:
            heavy_path = temporary / 'heavy-tasks.jsonl'
            heavy_manifest = write_heavy_task_list(heavy_path)
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
                heavy_ref = await _upload_artifact(
                    session,
                    control_url,
                    heavy_path,
                    kind='task_list',
                    sha256=heavy_manifest['sha256'],
                    size_bytes=heavy_manifest['size_bytes'],
                )
            uploaded_refs.append(heavy_ref)
        report['direct_upload_kinds'] = [ref['kind'] for ref in uploaded_refs]

        if 'dependency-build' in scenarios:
            from tests.integration.distributed.dependency_build_contract import run_dependency_build_contract

            report['dependency_build'] = await run_dependency_build_contract(
                control_url,
                app_url,
                root=ROOT,
                workspace=temporary / 'dependency-build',
                uploaded_refs=uploaded_refs,
                kubectl=kubectl,
                namespace=namespace,
                release_name=release_name,
            )
        if 'cli-smoke' in scenarios:
            assert source_root is not None
            report['cli_smoke'] = await _run_cli_smoke(
                control_url,
                app_url,
                source_root=source_root,
            )
        if 'dataset' in scenarios:
            report['dataset'] = await _run_dataset_batch_case(
                control_url,
                app_url,
                kubectl,
                namespace,
                release_name=release_name,
            )
        if 'local-dataset' in scenarios:
            assert source_root is not None
            assert dataset_root is not None
            assert tdsql_trace is not None
            assert local_dataset_barrier_name is not None
            report['local_dataset'] = await _run_local_dataset_batch_case(
                control_url,
                app_url,
                kubectl,
                namespace,
                release_name=release_name,
                source_root=source_root,
                dataset_root=dataset_root,
                tdsql_trace=tdsql_trace,
                barrier_name=local_dataset_barrier_name,
            )
        if 'harbor' in scenarios:
            assert source_root is not None
            assert harbor_root is not None
            assert harbor_local_root is not None
            assert harbor_direct_barrier_name is not None
            assert harbor_fetched_barrier_name is not None
            assert tdsql_trace is not None
            report['harbor'] = await _run_harbor_dataset_batch_case(
                control_url,
                app_url,
                kubectl,
                namespace,
                release_name=release_name,
                source_root=source_root,
                direct_root=harbor_root,
                fetched_root=harbor_local_root,
                tdsql_trace=tdsql_trace,
                direct_barrier_name=harbor_direct_barrier_name,
                fetched_barrier_name=harbor_fetched_barrier_name,
            )
        if 'source-smoke' in scenarios:
            assert source_ref is not None
            assert nexus_wheel_ref is not None
            assert uv_lock_ref is not None
            await _run_single_live_smoke(
                control_url,
                app_url,
                source_ref=source_ref,
                nexus_wheel_ref=nexus_wheel_ref,
                uv_lock_ref=uv_lock_ref,
            )
            report['source_smoke'] = await _run_source_batch_case(
                control_url,
                app_url,
                kubectl,
                namespace,
                release_name=release_name,
                source_ref=source_ref,
                nexus_wheel_ref=nexus_wheel_ref,
                uv_lock_ref=uv_lock_ref,
                replace_workload='execution',
            )
        if 'image-smoke' in scenarios:
            if image_mode_version is None or image_mode_image is None:
                raise ValueError('image-smoke requires image mode version and image')
            report['image_smoke'] = await _run_image_smoke(
                control_url,
                app_url,
                kubectl,
                namespace,
                image_version=image_mode_version,
                expected_image=image_mode_image,
            )
        if 'heavy' in scenarios:
            assert heavy_ref is not None
            assert heavy_manifest is not None
            report['heavy_runs'].append(
                await _run_heavy_batch_case(
                    control_url,
                    kubectl,
                    namespace,
                    release_name=release_name,
                    task_list_ref=heavy_ref,
                    manifest=heavy_manifest,
                    recovery_mode=None,
                )
            )
        if 'execution-recovery' in scenarios:
            assert heavy_ref is not None
            assert heavy_manifest is not None
            report['heavy_runs'].append(
                await _run_heavy_batch_case(
                    control_url,
                    kubectl,
                    namespace,
                    release_name=release_name,
                    task_list_ref=heavy_ref,
                    manifest=heavy_manifest,
                    recovery_mode='execution',
                )
            )
        if 'batch-recovery' in scenarios:
            assert heavy_ref is not None
            assert heavy_manifest is not None
            report['heavy_runs'].append(
                await _run_heavy_batch_case(
                    control_url,
                    kubectl,
                    namespace,
                    release_name=release_name,
                    task_list_ref=heavy_ref,
                    manifest=heavy_manifest,
                    recovery_mode='batch',
                )
            )

        allocation_payload = await request_json('GET', f'{control_url}/v1/allocations?limit=500')
        dynamic_allocations = [
            item for item in allocation_payload['items'] if item['environment_id'] == DYNAMIC_ENVIRONMENT_ID
        ]
        assert dynamic_allocations
        assert all(item['status'] == 'released' for item in dynamic_allocations), dynamic_allocations
        allocation_ids = {item['allocation_id'] for item in dynamic_allocations}
        await _wait_no_active_manager_sessions(control_url, allocation_ids=allocation_ids)
        report['released_allocation_ids'] = sorted(allocation_ids)
        report['mysql_final'] = mysql_status(kubectl, namespace, release_name)
        return report


async def cleanup_submission_artifacts(
    *,
    refs: list[dict[str, Any]],
    base_url: str,
    repository: str,
    credentials: StaticCredentials,
) -> None:
    store = GenericRepositoryArtifactStore(
        base_url=base_url,
        repository=repository,
        credentials=credentials,
    )
    try:
        for ref in refs:
            key = artifact_object_key(
                owner_principal_id='local-dev',
                artifact_id=ref['artifact_id'],
                sha256=ref['sha256'],
                kind=ref['kind'],
            )
            await store.delete(StoredObject(uri=store.object_uri(key)))
    finally:
        await store.close()


def assert_no_dynamic_provider_resources(
    kubectl: str,
    namespace: str,
    *,
    release_name: str,
    timeout: float = 180.0,
) -> list[str]:
    resource_types = ('statefulsets', 'jobs', 'services', 'secrets', 'pods')
    deadline = time.monotonic() + timeout
    remaining: dict[str, list[dict[str, Any]]] = {}
    tolerated_tombstones: set[str] = set()
    while time.monotonic() < deadline:
        remaining = {}
        for resource_type in resource_types:
            payload = json.loads(
                run(
                    [
                        kubectl,
                        '-n',
                        namespace,
                        'get',
                        resource_type,
                        '-l',
                        f'app.kubernetes.io/managed-by=nexus,app.kubernetes.io/instance={release_name}',
                        '-o',
                        'json',
                    ],
                    capture=True,
                ).stdout
            )
            items = payload['items']
            if resource_type == 'pods':
                tolerated_tombstones.update(
                    item['metadata']['name'] for item in items if is_tolerated_job_tracking_tombstone(item)
                )
                items = [item for item in items if not is_tolerated_job_tracking_tombstone(item)]
            if items:
                remaining[resource_type] = items
        if not remaining:
            tombstones = sorted(tolerated_tombstones)
            if tombstones:
                print(
                    'DevCloud retained terminating dependency-builder Pod tombstones: '
                    f'{tombstones}; batch.kubernetes.io/job-tracking is owned by the cluster Job controller',
                    flush=True,
                )
            return tombstones
        time.sleep(1)
    raise AssertionError(remaining)


def assert_database_secret_synced(
    kubectl: str,
    namespace: str,
    *,
    source_secret_name: str,
    workload_secret_name: str,
    release_name: str,
) -> dict[str, str]:
    source = json.loads(
        run(
            [kubectl, '-n', namespace, 'get', 'secret', source_secret_name, '-o', 'json'],
            capture=True,
        ).stdout
    )
    workload = json.loads(
        run(
            [kubectl, '-n', namespace, 'get', 'secret', workload_secret_name, '-o', 'json'],
            capture=True,
        ).stdout
    )
    if (source.get('data') or {}).get('database-url') != (workload.get('data') or {}).get('database-url'):
        raise AssertionError('dynamic workload database Secret content mismatch')
    metadata = workload.get('metadata') or {}
    labels = metadata.get('labels') or {}
    annotations = metadata.get('annotations') or {}
    if labels.get('app.kubernetes.io/instance') != release_name:
        raise AssertionError('dynamic workload database Secret release label mismatch')
    if annotations.get('nexus.io/shared-secret') != 'database-url':
        raise AssertionError('dynamic workload database Secret ownership marker missing')
    return {
        'secret_name': workload_secret_name,
        'managed_by': str(annotations.get('nexus.io/managed-by') or ''),
    }


def is_tolerated_job_tracking_tombstone(resource: dict[str, Any]) -> bool:
    metadata = resource.get('metadata') or {}
    labels = metadata.get('labels') or {}
    return (
        bool(metadata.get('deletionTimestamp'))
        and metadata.get('finalizers') == ['batch.kubernetes.io/job-tracking']
        and bool(labels.get('batch.kubernetes.io/job-name'))
    )


def delete_dynamic_provider_resources(kubectl: str, namespace: str, *, release_name: str) -> None:
    run(
        [
            kubectl,
            '-n',
            namespace,
            'delete',
            'statefulsets,jobs,services,secrets',
            '-l',
            f'app.kubernetes.io/managed-by=nexus,app.kubernetes.io/instance={release_name}',
            '--ignore-not-found=true',
            '--wait=true',
        ],
        check=False,
    )
    run(
        [
            kubectl,
            '-n',
            namespace,
            'delete',
            'pods',
            '-l',
            f'app.kubernetes.io/managed-by=nexus,app.kubernetes.io/instance={release_name}',
            '--ignore-not-found=true',
            '--wait=false',
        ],
        check=False,
    )


def test_workload_build_command(
    *,
    workload_image: str,
    dockerfile: Path,
    workload_base_image: str,
) -> list[str]:
    return [
        'docker',
        'build',
        '-f',
        str(dockerfile),
        '--build-arg',
        f'NEXUS_WORKLOAD_BASE_IMAGE={workload_base_image}',
        '-t',
        workload_image,
        '.',
    ]


def builder_build_command(*, workload_image: str, builder_image: str) -> list[str]:
    return [
        'docker',
        'build',
        '-f',
        'deployment/distributed/Dockerfile.builder',
        '--build-arg',
        f'NEXUS_WORKLOAD_BASE_IMAGE={workload_image}',
        '-t',
        builder_image,
        '.',
    ]


def user_workload_build_command(*, workload_image: str, user_image: str, dockerfile: Path) -> list[str]:
    return [
        'docker',
        'build',
        '-f',
        str(dockerfile),
        '--build-arg',
        f'NEXUS_WORKLOAD_BASE_IMAGE={workload_image}',
        '-t',
        user_image,
        '.',
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run dynamic distributed single/live/batch tests on the DevCloud Kubernetes cluster.'
    )
    parser.add_argument('--kubeconfig', type=Path, default=DEFAULT_DEVCLOUD_KUBECONFIG)
    parser.add_argument('--namespace', default=DEFAULT_DEVCLOUD_NAMESPACE)
    parser.add_argument('--release-name', default='nexus-phase2-devcloud-dynamic')
    parser.add_argument('--control-image', default=DEFAULT_CONTROL_IMAGE)
    parser.add_argument('--workload-image', default=DEFAULT_WORKLOAD_IMAGE)
    parser.add_argument('--builder-image', default=DEFAULT_BUILDER_IMAGE)
    parser.add_argument('--workload-dockerfile', type=Path, default=TEST_WORKLOAD_DOCKERFILE)
    parser.add_argument('--image-mode-dockerfile', type=Path, default=TEST_USER_WORKLOAD_DOCKERFILE)
    parser.add_argument('--image-mode-version')
    parser.add_argument('--workload-base-image', default='nexus-distributed-workload-base:devcloud-local')
    parser.add_argument('--mysql-image', default=DEFAULT_MYSQL_IMAGE)
    parser.add_argument('--artifact-base-url', default='https://mirrors.tencent.com/repository/generic')
    parser.add_argument('--artifact-repository', default='nexus-agent')
    parser.add_argument('--artifact-provider', choices=('local', 'generic_repository'), default='local')
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    parser.add_argument('--tdsql-config-file', type=Path)
    parser.add_argument('--kubectl-bin')
    parser.add_argument('--helm-bin')
    parser.add_argument(
        '--scenario',
        action='append',
        choices=(
            'cli-smoke',
            'dataset',
            'local-dataset',
            'harbor',
            'dependency-build',
            'source-smoke',
            'image-smoke',
            'heavy',
            'execution-recovery',
            'batch-recovery',
        ),
        help='Scenario to run; repeat the flag. The default runs all scenarios.',
    )
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--skip-push', action='store_true')
    parser.add_argument('--keep-release', action='store_true')
    parser.add_argument('--wait-timeout', default='600s')
    parser.add_argument('--report-path', type=Path)
    args = parser.parse_args(argv)
    image_smoke_enabled = args.scenario is None or 'image-smoke' in args.scenario
    if image_smoke_enabled and (args.skip_build or args.skip_push) and not args.image_mode_version:
        parser.error('--image-mode-version is required when image-smoke skips build or push')
    data_list_enabled = args.scenario is None or bool(set(args.scenario or ()) & {'local-dataset', 'harbor'})
    if data_list_enabled and args.tdsql_config_file is None:
        parser.error('--local-dataset/--harbor requires --tdsql-config-file')
    return args


def main() -> int:
    args = parse_args()
    scenarios = set(
        args.scenario
        or (
            'cli-smoke',
            'dataset',
            'local-dataset',
            'harbor',
            'dependency-build',
            'source-smoke',
            'image-smoke',
            'heavy',
            'execution-recovery',
            'batch-recovery',
        )
    )
    image_mode_version: str | None = None
    image_mode_image: str | None = None
    if 'image-smoke' in scenarios:
        image_mode_version = args.image_mode_version or (
            f'devcloud-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{uuid.uuid4().hex[:8]}'
        )
        image_mode_image = resolve_workload_image(args.workload_image, image_mode_version)
    kubectl = require_binary('kubectl', args.kubectl_bin)
    helm = require_binary('helm', args.helm_bin)
    configure_devcloud(kubectl, kubeconfig=args.kubeconfig, namespace=args.namespace)
    values = merged_env(args.env_file)
    credentials: StaticCredentials | None = None
    if args.artifact_provider == 'generic_repository':
        credentials = artifact_credentials(values)
        asyncio.run(
            run_contract(
                base_url=args.artifact_base_url,
                repository=args.artifact_repository,
                credentials=credentials,
            )
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
            test_workload_build_command(
                workload_image=args.workload_image,
                dockerfile=args.workload_dockerfile,
                workload_base_image=args.workload_base_image,
            ),
            timeout=900,
        )
        if scenarios & {'cli-smoke', 'source-smoke', 'local-dataset', 'harbor', 'dependency-build'}:
            run(
                builder_build_command(workload_image=args.workload_image, builder_image=args.builder_image),
                timeout=1200,
            )
        if image_mode_image is not None:
            run(
                user_workload_build_command(
                    workload_image=args.workload_image,
                    user_image=image_mode_image,
                    dockerfile=args.image_mode_dockerfile,
                ),
                timeout=900,
            )

    refs: list[dict[str, Any]] = []
    if not args.skip_push:
        images = [args.control_image, args.workload_image]
        if scenarios & {'cli-smoke', 'source-smoke', 'local-dataset', 'harbor', 'dependency-build'}:
            images.append(args.builder_image)
        if image_mode_image is not None:
            images.append(image_mode_image)
        push_images(images)

    app_local_port = free_port()
    database_secret_name = f'{args.release_name}-database'
    workload_database_secret_name = f'{args.release_name}-workload-database'
    mysql_secret_name = f'{args.release_name}-mysql-auth'
    mysql_password = values.get('NEXUS_DIST_MYSQL_ROOT_PASSWORD', 'nexus_dist_local')
    tdsql_trace = load_tdsql_trace_config(args.tdsql_config_file) if scenarios & {'local-dataset', 'harbor'} else None
    try:
        run([helm, 'uninstall', args.release_name, '--namespace', args.namespace, '--wait'], check=False)
        delete_dynamic_provider_resources(kubectl, args.namespace, release_name=args.release_name)
        delete_fake_dataset_service(kubectl, args.namespace, release_name=args.release_name)
        run(
            [
                kubectl,
                '-n',
                args.namespace,
                'delete',
                'secret',
                database_secret_name,
                workload_database_secret_name,
                mysql_secret_name,
                '--ignore-not-found=true',
            ],
            check=False,
        )
        create_database_bootstrap_secrets(
            run,
            kubectl=kubectl,
            namespace=args.namespace,
            release_name=args.release_name,
            database_secret_name=database_secret_name,
            mysql_secret_name=mysql_secret_name,
            password=mysql_password,
        )
        control_repository, control_tag = split_image(args.control_image)
        workload_repository, workload_tag = split_image(args.workload_image)
        builder_repository, builder_tag = split_image(args.builder_image)
        mysql_repository, mysql_tag = split_image(args.mysql_image)
        helm_args = [
            helm,
            'upgrade',
            '--install',
            args.release_name,
            str(CHART_DIR),
            '--namespace',
            args.namespace,
            '-f',
            str(DYNAMIC_VALUES),
            '--set-string',
            f'fullnameOverride={args.release_name}',
            '--set-string',
            f'controlImage.repository={control_repository}',
            '--set-string',
            f'workloadImage.repository={workload_repository}',
            '--set-string',
            f'builderImage.repository={builder_repository}',
            '--set-string',
            f'mysql.image.repository={mysql_repository}',
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
            '--set-string',
            f'dynamicEnvironment.namespace={args.namespace}',
            '--set-string',
            'dynamicEnvironment.enabledProviders[0]=kubernetes',
            '--timeout',
            args.wait_timeout,
        ]
        if control_tag:
            helm_args.extend(['--set-string', f'controlImage.tag={control_tag}'])
        if workload_tag:
            helm_args.extend(['--set-string', f'workloadImage.tag={workload_tag}'])
        if builder_tag:
            helm_args.extend(['--set-string', f'builderImage.tag={builder_tag}'])
        if mysql_tag:
            helm_args.extend(['--set-string', f'mysql.image.tag={mysql_tag}'])
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
            'fakeRuntime.replicas=0',
            '--wait',
        ]
        run(bootstrap_helm_args, timeout=900)
        assert_database_ready(kubectl, args.namespace, args.release_name, args.wait_timeout)
        seed_dynamic_platform_config(
            kubectl,
            namespace=args.namespace,
            release_name=args.release_name,
            mysql_password=mysql_password,
            environment=values,
            application_public_base_url=f'http://127.0.0.1:{app_local_port}',
            workload_image=args.workload_image,
            builder_image=args.builder_image,
            database_secret_name=workload_database_secret_name,
            artifact_provider=args.artifact_provider,
            artifact_base_url=args.artifact_base_url,
            artifact_repository=args.artifact_repository,
            artifact_credentials=credentials,
            include_dataset=dataset_service_enabled(scenarios),
            tdsql_trace=tdsql_trace,
        )
        run([*helm_args, '--set', 'dbInit.enabled=false', '--wait'], timeout=900)
        if dataset_service_enabled(scenarios):
            create_fake_dataset_service(
                kubectl,
                args.namespace,
                release_name=args.release_name,
                image=args.workload_image,
            )

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
            asyncio.run(wait_http_ok(f'{control.url}/v1/health', timeout=60))
            asyncio.run(wait_http_ok(f'{app.url}/v1/health', timeout=60))
            report = asyncio.run(
                run_dynamic_api_tests(
                    control.url,
                    app.url,
                    kubectl,
                    args.namespace,
                    release_name=args.release_name,
                    scenarios=scenarios,
                    uploaded_refs=refs,
                    workload_image=args.workload_image,
                    image_mode_version=image_mode_version,
                    image_mode_image=image_mode_image,
                    tdsql_trace=tdsql_trace,
                )
            )
        report['database_secret_sync'] = assert_database_secret_synced(
            kubectl,
            args.namespace,
            source_secret_name=database_secret_name,
            workload_secret_name=workload_database_secret_name,
            release_name=args.release_name,
        )
        run(
            [
                kubectl,
                '-n',
                args.namespace,
                'delete',
                'secret',
                workload_database_secret_name,
                '--ignore-not-found=true',
            ]
        )
        if dataset_service_enabled(scenarios):
            delete_fake_dataset_service(kubectl, args.namespace, release_name=args.release_name)
        tolerated_tombstones = assert_no_dynamic_provider_resources(
            kubectl,
            args.namespace,
            release_name=args.release_name,
        )
        report['cleanup'] = {
            'tolerated_job_tracking_tombstones': tolerated_tombstones,
        }
        report_path = args.report_path or Path(tempfile.gettempdir()) / f'nexus-{args.release_name}-report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_report = report_path.with_suffix(f'{report_path.suffix}.tmp')
        temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
        temporary_report.replace(report_path)
        print(f'dynamic Kubernetes report: {report_path}', flush=True)
        print(f'dynamic Kubernetes scenarios passed: {sorted(scenarios)}', flush=True)
        return 0
    finally:
        if not args.keep_release:
            run([helm, 'uninstall', args.release_name, '--namespace', args.namespace, '--wait'], check=False)
            delete_dynamic_provider_resources(kubectl, args.namespace, release_name=args.release_name)
            delete_fake_dataset_service(kubectl, args.namespace, release_name=args.release_name)
            run(
                [
                    kubectl,
                    '-n',
                    args.namespace,
                    'delete',
                    'secret',
                    database_secret_name,
                    workload_database_secret_name,
                    mysql_secret_name,
                    '--ignore-not-found=true',
                ],
                check=False,
            )
            if refs and credentials is not None:
                asyncio.run(
                    cleanup_submission_artifacts(
                        refs=refs,
                        base_url=args.artifact_base_url,
                        repository=args.artifact_repository,
                        credentials=credentials,
                    )
                )


if __name__ == '__main__':
    sys.exit(main())
