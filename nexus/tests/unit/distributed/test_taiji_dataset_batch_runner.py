from __future__ import annotations

import json

from types import SimpleNamespace

import pytest

from tests.e2e.distributed import run_taiji_dataset_batch as taiji_runner
from tests.e2e.distributed.run_taiji_dataset_batch import (
    _observe_workload,
    dataset_batch_submission,
)

from nexus.distributed.submission.specs import BatchSubmission


def test_taiji_runner_reads_tkex_api_token_file_before_creating_client(monkeypatch, tmp_path) -> None:
    token_file = tmp_path / 'tkex-token'
    token_file.write_text('tkex-secret', encoding='utf-8')
    token_file.chmod(0o600)
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(taiji_runner, 'TkexWorkloadClient', FakeClient)

    taiji_runner._create_tkex_workload_client(
        SimpleNamespace(
            tkex_base_url='https://tkex.example',
            tkex_namespace='nexus-test',
            tkex_api_account='account',
            tkex_api_token_file=token_file,
            tkex_cluster_name='cluster',
            tkex_project_name='project',
        )
    )

    assert captured['api_token'] == 'tkex-secret'
    assert 'api_token_file' not in captured


def test_dataset_batch_submission_is_fixed_paginated_and_secret_free() -> None:
    payload = dataset_batch_submission(
        environment_id='runner-dev',
        dataset_uri='nexus-test/terminal-bench-2@v4',
        limit=2,
        page_size=1,
        runtime_api_base_url='https://runtime.example',
        runtime_batch_id='dataset-e2e-batch',
    )

    submission = BatchSubmission.model_validate(payload)
    compose = submission.generation.source.compose  # type: ignore[union-attr]
    assert compose is not None
    assert submission.generation.source.task_list_version == 2  # type: ignore[union-attr]
    assert compose.limit == 2
    assert compose.page_size == 1
    assert compose.parts[0].value == 'nexus-test/terminal-bench-2@v4'
    assert submission.code.mode == 'config_only'
    assert submission.resources is not None
    assert submission.resources.batch_manager is not None
    assert submission.resources.managers.count == 2
    assert submission.batch.max_inflight == 2
    serialized = json.dumps(payload)
    assert 'WEDATA_KEY' not in serialized
    assert 'GONGFENG_TOKEN' not in serialized


@pytest.mark.parametrize(
    ('dataset_uri', 'limit', 'page_size', 'message'),
    [
        ('nexus-test/terminal-bench-2@latest', 2, 1, 'fixed release'),
        ('nexus-test/terminal-bench-2@v4', 2, 2, 'page_size'),
    ],
)
def test_dataset_batch_submission_rejects_unpinned_or_single_page_inputs(
    dataset_uri: str,
    limit: int,
    page_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        dataset_batch_submission(
            environment_id='runner-dev',
            dataset_uri=dataset_uri,
            limit=limit,
            page_size=page_size,
            runtime_api_base_url='https://runtime.example',
            runtime_batch_id='dataset-e2e-batch',
        )


class _WorkloadClient:
    def __init__(self, workload: dict) -> None:
        self.workload = workload

    async def get_stateful_workload(self, name: str) -> dict:
        assert name == 'nexus-allocation'
        return self.workload


def _workload(*, image: str, include_wedata: bool) -> dict:
    env = []
    sources = [{'secret': {'name': 'database', 'items': [{'key': 'url', 'path': 'database/url'}]}}]
    if include_wedata:
        env.append({'name': 'WEDATA_KEY_FILE', 'value': '/var/run/secrets/nexus/auth/wedata/key'})
        sources.append(
            {
                'secret': {
                    'name': 'runner-dev-wedata',
                    'items': [{'key': 'key', 'path': 'auth/wedata/key'}],
                }
            }
        )
    return {
        'spec': {
            'template': {
                'spec': {
                    'containers': [{'name': 'instance-manager', 'image': image, 'env': env}],
                    'volumes': [{'name': 'nexus-secrets', 'projected': {'sources': sources}}],
                }
            }
        }
    }


def _allocation(workload: str) -> dict:
    return {
        'allocation_id': f'allocation-{workload}',
        'workload': workload,
        'desired_manager_count': 1,
        'provider_resource_ref': {'workload': {'name': 'nexus-allocation'}},
    }


async def test_observe_workload_projects_wedata_key_to_both_manager_roles() -> None:
    image = 'registry.example/nexus-workload@sha256:1234'
    orchestration = await _observe_workload(
        _WorkloadClient(_workload(image=image, include_wedata=True)),  # type: ignore[arg-type]
        _allocation('orchestration'),
        expected_wedata_secret='runner-dev-wedata',
        expected_wedata_secret_key='key',
        expected_workload_image=image,
    )
    execution = await _observe_workload(
        _WorkloadClient(_workload(image=image, include_wedata=True)),  # type: ignore[arg-type]
        _allocation('execution'),
        expected_wedata_secret='runner-dev-wedata',
        expected_wedata_secret_key='key',
        expected_workload_image=image,
    )

    assert orchestration is not None
    assert orchestration['wedata_secret_projected'] is True
    assert execution is not None
    assert execution['wedata_secret_projected'] is True


async def test_observe_workload_rejects_missing_wedata_key_on_execution_manager() -> None:
    image = 'registry.example/nexus-workload@sha256:1234'
    with pytest.raises(AssertionError, match='execution Manager'):
        await _observe_workload(
            _WorkloadClient(_workload(image=image, include_wedata=False)),  # type: ignore[arg-type]
            _allocation('execution'),
            expected_wedata_secret='runner-dev-wedata',
            expected_wedata_secret_key='key',
            expected_workload_image=image,
        )
