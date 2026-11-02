from __future__ import annotations

import pytest

from pydantic import ValidationError

from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.submission.specs import (
    BatchSubmission,
    ConfigOnlyCode,
    ImageCode,
    SingleSubmission,
    SourcePackageCode,
    artifact_refs,
)


def _artifact(kind: str, suffix: str) -> dict[str, object]:
    return {
        'artifact_id': f'artifact-{suffix}',
        'kind': kind,
        'sha256': suffix * 64,
        'size_bytes': 17,
    }


def _task_list_spec(suffix: str = 'c', *, item_count: int = 1) -> dict[str, object]:
    ref = _artifact('task_list', suffix)
    return {
        'ref': ref,
        'format': 'jsonl',
        'manifest': {
            'version': 1,
            'item_count': item_count,
            'size_bytes': ref['size_bytes'],
            'sha256': ref['sha256'],
            'source': {'mode': 'local_task_list'},
            'datasets': [],
        },
    }


def _data_list_spec(suffix: str = 'e', *, frame_count: int = 1) -> dict[str, object]:
    ref = _artifact('data_list', suffix)
    return {
        'ref': ref,
        'format': 'nxdl',
        'manifest': {
            'version': 1,
            'frame_count': frame_count,
            'size_bytes': ref['size_bytes'],
            'sha256': ref['sha256'],
        },
    }


def _source_code() -> dict[str, object]:
    return {
        'mode': 'source_package',
        'source_package': _artifact('source_package', 'a'),
        'nexus_wheel': _artifact('nexus_wheel', 'b'),
        'uv_lock': _artifact('uv_lock', 'd'),
        'extras': ['extensions'],
    }


def _image_code() -> dict[str, object]:
    return {'mode': 'image', 'image_version': 'user-build-20260802-001'}


def _dynamic_single(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        'submission_version': 1,
        'idempotency_key': 'request-1',
        'environment_id': 'environment-dynamic',
        'code': _source_code(),
        'resources': {
            'managers': {
                'count': 1,
                'cpu_cores': 8,
                'memory_gib': 16,
                'slots': 1,
            }
        },
        'run_input': {'task_name': 'echo', 'task_config': {}},
    }
    payload.update(updates)
    return payload


def _dynamic_batch(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        'submission_version': 1,
        'idempotency_key': 'batch-1',
        'environment_id': 'environment-dynamic',
        'code': _source_code(),
        'resources': {
            'batch_manager': {'cpu_cores': 4, 'memory_gib': 8},
            'managers': {
                'count': 3,
                'cpu_cores': 2,
                'memory_gib': 4,
                'slots': 5,
            },
        },
        'generation': {
            'mode': 'remote_materialize',
            'timeout_seconds': 120,
            'source': {'type': 'generator', 'path': 'tasks.py'},
        },
        'batch': {
            'max_inflight': 15,
            'retry_failed': 1,
            'fail_fast': False,
            'child_queue_ttl_seconds': 300,
            'timeout_per_run': 20,
        },
    }
    payload.update(updates)
    return payload


def test_dynamic_single_uses_typed_code_resources_and_timeouts():
    submission = SingleSubmission.model_validate(
        _dynamic_single(
            provisioning_timeout_seconds=60,
            queue_ttl_seconds=120,
            execution_ttl_seconds=600,
            stop_timeout_seconds=30,
        )
    )

    assert isinstance(submission.code, SourcePackageCode)
    assert submission.resources.managers.count == 1
    assert submission.resources.managers.slots == 1
    assert submission.provisioning_timeout_seconds == 60
    dumped = submission.model_dump(mode='json')
    assert dumped['environment_id'] == 'environment-dynamic'
    assert not ({'placement', 'slot_cost', 'max_attempts', 'execution_environment'} & dumped.keys())


def test_dynamic_single_accepts_submit_host_environment():
    submission = SingleSubmission.model_validate(
        _dynamic_single(
            host_env=[
                {'name': 'HYTUNER', 'value': '1'},
                {'name': 'HOST_ENV_MVP_PROBE', 'value': 'ok'},
            ]
        )
    )

    assert submission.model_dump(mode='json')['host_env'] == [
        {'name': 'HYTUNER', 'value': '1'},
        {'name': 'HOST_ENV_MVP_PROBE', 'value': 'ok'},
    ]


@pytest.mark.parametrize(
    'host_env',
    [
        [{'name': 'bad-name', 'value': 'x'}],
        [{'name': 'POD_NAME', 'value': 'x'}],
        [{'name': 'FOO', 'value': 'x'}, {'name': 'FOO', 'value': 'y'}],
        [{'name': 'FOO', 'value': '\x00'}],
    ],
)
def test_submission_rejects_invalid_submit_host_environment(host_env):
    with pytest.raises(ValidationError):
        SingleSubmission.model_validate(_dynamic_single(host_env=host_env))


def test_predefined_submission_rejects_submit_host_environment():
    with pytest.raises(ValidationError, match='host_env_requires_dynamic_allocation'):
        SingleSubmission.model_validate(
            {
                'idempotency_key': 'request-predefined-host-env',
                'environment_id': 'environment-predefined',
                'pool': 'execution-pool',
                'code': {'mode': 'config_only'},
                'host_env': [{'name': 'FOO', 'value': 'bar'}],
                'run_input': {'task_name': 'echo', 'task_config': {}},
            }
        )


def test_submission_accepts_an_ordered_run_trace_backend_selection() -> None:
    submission = SingleSubmission.model_validate(
        _dynamic_single(
            run_trace_backends={
                'sinks': ['tdsql-team-a', 'cos-archive'],
            }
        )
    )

    assert submission.run_trace_backends is not None
    assert submission.run_trace_backends.sinks == ['tdsql-team-a', 'cos-archive']
    assert submission.model_dump(mode='json')['run_trace_backends'] == {'sinks': ['tdsql-team-a', 'cos-archive']}


def test_submission_accepts_unicode_run_trace_backend_names() -> None:
    submission = SingleSubmission.model_validate(_dynamic_single(run_trace_backends={'sinks': ['团队归档']}))

    assert submission.run_trace_backends is not None
    assert submission.run_trace_backends.sinks == ['团队归档']


@pytest.mark.parametrize(
    'sinks',
    [
        [],
        ['primary', 'primary'],
        ['invalid backend'],
        [f'backend-{index}' for index in range(9)],
    ],
)
def test_submission_rejects_invalid_run_trace_backend_selections(sinks: list[str]) -> None:
    with pytest.raises(ValidationError):
        SingleSubmission.model_validate(_dynamic_single(run_trace_backends={'sinks': sinks}))


def test_submission_identity_fields_accept_database_and_api_maximum_lengths():
    submission = SingleSubmission.model_validate({
        'idempotency_key': 'i' * 128,
        'environment_id': 'e' * 64,
        'pool': f'p{"x" * 62}z',
        'code': {'mode': 'config_only'},
        'run_input': {'task_name': 'echo', 'task_config': {}},
    })

    assert len(submission.idempotency_key) == 128
    assert len(submission.environment_id) == 64
    assert submission.pool is not None
    assert len(submission.pool) == 64


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('idempotency_key', 'i' * 129),
        ('environment_id', 'e' * 65),
        ('pool', f'p{"x" * 63}z'),
    ],
)
def test_submission_identity_fields_reject_values_longer_than_persisted_contract(field: str, value: str):
    payload = {
        'idempotency_key': 'request-predefined',
        'environment_id': 'environment-predefined',
        'pool': 'execution-pool',
        'code': {'mode': 'config_only'},
        'run_input': {'task_name': 'echo', 'task_config': {}},
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        SingleSubmission.model_validate(payload)


def test_dynamic_single_manager_count_is_fixed():
    payload = _dynamic_single()
    payload['resources']['managers']['count'] = 2  # type: ignore[index]

    with pytest.raises(ValidationError):
        SingleSubmission.model_validate(payload)


def test_dynamic_single_manager_slots_are_configurable():
    payload = _dynamic_single()
    payload['resources']['managers']['slots'] = 20  # type: ignore[index]

    submission = SingleSubmission.model_validate(payload)

    assert submission.resources.managers.slots == 20
    assert submission.resources.managers.normalized().slots == 20


def test_dynamic_single_manager_slots_default_to_20():
    payload = _dynamic_single()
    del payload['resources']['managers']['slots']  # type: ignore[index]

    submission = SingleSubmission.model_validate(payload)

    assert submission.resources.managers.slots == 20


def test_predefined_accepts_only_config_only_pool_shape():
    submission = SingleSubmission.model_validate({
        'idempotency_key': 'request-predefined',
        'environment_id': 'environment-predefined',
        'pool': 'execution-pool',
        'code': {'mode': 'config_only'},
        'run_input': {'task_name': 'echo', 'task_config': {}},
    })

    assert isinstance(submission.code, ConfigOnlyCode)
    assert submission.pool == 'execution-pool'
    assert submission.resources is None


@pytest.mark.parametrize(
    'updates',
    [
        {'pool': 'predefined-pool'},
        {'resources': None},
    ],
)
def test_dynamic_and_predefined_shapes_cannot_be_mixed(updates: dict[str, object]):
    with pytest.raises(ValidationError):
        SingleSubmission.model_validate(_dynamic_single(**updates))


def test_predefined_rejects_source_package_code():
    with pytest.raises(ValidationError, match='config_only'):
        SingleSubmission.model_validate({
            'idempotency_key': 'request-predefined',
            'environment_id': 'environment-predefined',
            'pool': 'execution-pool',
            'code': _source_code(),
            'run_input': {'task_name': 'identity'},
        })


def test_dynamic_single_accepts_image_code_without_artifact_refs():
    submission = SingleSubmission.model_validate(_dynamic_single(code=_image_code()))

    assert isinstance(submission.code, ImageCode)
    assert submission.code.image_version == 'user-build-20260802-001'
    assert artifact_refs(submission) == []


@pytest.mark.parametrize('image_version', ['latest', 'registry.example/workload:v1', '../v1'])
def test_image_code_rejects_floating_or_full_image_references(image_version: str):
    with pytest.raises(ValidationError, match='image_version'):
        SingleSubmission.model_validate(_dynamic_single(code={'mode': 'image', 'image_version': image_version}))


def test_predefined_rejects_image_code():
    with pytest.raises(ValidationError, match='config_only'):
        SingleSubmission.model_validate({
            'idempotency_key': 'request-predefined-image',
            'environment_id': 'environment-predefined',
            'pool': 'execution-pool',
            'code': _image_code(),
            'run_input': {'task_name': 'identity'},
        })


@pytest.mark.parametrize('field', ['placement', 'slot_cost', 'max_attempts', 'owner_principal_id'])
def test_forbidden_public_scheduling_fields_are_rejected(field: str):
    with pytest.raises(ValidationError):
        SingleSubmission.model_validate(_dynamic_single(**{field: 1}))


def test_source_package_code_does_not_accept_requirements_or_python_controls():
    code = _source_code()
    code['requirements'] = ['arbitrary-package']

    with pytest.raises(ValidationError):
        SourcePackageCode.model_validate(code)


def test_source_package_code_rejects_legacy_or_arbitrary_wheel_fields():
    code = _source_code()
    code['wheel'] = code.pop('nexus_wheel')

    with pytest.raises(ValidationError):
        SourcePackageCode.model_validate(code)


def test_v1_dynamic_batch_defaults_missing_policy_to_execution_capacity():
    payload = _dynamic_batch()
    payload.pop('batch')

    submission = BatchSubmission.model_validate(payload)

    assert submission.batch.max_inflight == 15
    assert submission.batch.retry_failed == 0
    assert submission.batch.fail_fast is False


@pytest.mark.parametrize(
    ('manager_updates', 'expected_count', 'expected_slots'),
    [
        ({'count': None, 'slots': None}, 1, 20),
        ({'count': None}, 3, 5),
        ({'slots': None}, 3, 5),
    ],
)
def test_dynamic_batch_infers_missing_manager_capacity(
    manager_updates: dict[str, int | None],
    expected_count: int,
    expected_slots: int,
):
    payload = _dynamic_batch()
    managers = payload['resources']['managers']  # type: ignore[index]
    for field, value in manager_updates.items():
        if value is None:
            managers.pop(field)
        else:
            managers[field] = value

    submission = BatchSubmission.model_validate(payload)

    assert submission.resources is not None
    assert submission.resources.managers.count == expected_count
    assert submission.resources.managers.slots == expected_slots


def test_batch_rejects_max_inflight_above_execution_capacity():
    payload = _dynamic_batch()
    payload['batch']['max_inflight'] = 16  # type: ignore[index]

    with pytest.raises(ValidationError, match='max_inflight'):
        BatchSubmission.model_validate(payload)


def test_batch_manager_count_and_slots_are_not_public_fields():
    payload = _dynamic_batch()
    payload['resources']['batch_manager']['count'] = 1  # type: ignore[index]

    with pytest.raises(ValidationError):
        BatchSubmission.model_validate(payload)


def test_v1_predefined_batch_leaves_missing_policy_for_environment_default():
    submission = BatchSubmission.model_validate({
        'idempotency_key': 'batch-predefined',
        'environment_id': 'environment-predefined',
        'pool': 'execution-pool',
        'code': {'mode': 'config_only'},
        'generation': {
            'mode': 'local_task_list',
            'task_list': _task_list_spec(),
        },
    })

    assert submission.batch.max_inflight is None


def test_local_task_list_requires_manifest_matching_artifact_identity():
    payload = _dynamic_batch(
        generation={
            'mode': 'local_task_list',
            'task_list': {'ref': _artifact('task_list', 'c'), 'format': 'jsonl'},
        }
    )

    with pytest.raises(ValidationError, match='manifest'):
        BatchSubmission.model_validate(payload)


def test_remote_generator_requires_source_package_code():
    payload = _dynamic_batch(code={'mode': 'config_only'})

    with pytest.raises(ValidationError, match='remote generator'):
        BatchSubmission.model_validate(payload)


def test_remote_generator_accepts_image_code():
    submission = BatchSubmission.model_validate(_dynamic_batch(code=_image_code()))

    assert isinstance(submission.code, ImageCode)
    assert submission.generation.source.path == 'tasks.py'  # type: ignore[union-attr]


@pytest.mark.parametrize(
    'uri',
    [
        'research/tasks@v1.2.3',
        'taiji.nexus.research/tasks@latest?split=train',
    ],
)
def test_cloud_compose_accepts_dataset_service_uris(uri: str):
    payload = _dynamic_batch(
        code={'mode': 'config_only'},
        generation={
            'mode': 'remote_materialize',
            'source': {
                'type': 'compose',
                'compose': {
                    'parts': [{'kind': 'data', 'value': uri, 'source': f'--data {uri}'}],
                    'page_size': 17,
                },
            },
        },
    )

    submission = BatchSubmission.model_validate(payload)

    assert submission.generation.source.compose.parts[0].value == uri  # type: ignore[union-attr]
    assert submission.generation.source.compose.page_size == 17  # type: ignore[union-attr]
    assert submission.generation.source.task_list_version == 2  # type: ignore[union-attr]


def test_remote_materialize_accepts_explicit_task_list_v2():
    payload = _dynamic_batch()
    payload['generation']['source']['task_list_version'] = 2  # type: ignore[index]

    submission = BatchSubmission.model_validate(payload)

    assert submission.generation.source.task_list_version == 2  # type: ignore[union-attr]


def test_remote_materialize_rejects_unknown_task_list_version():
    payload = _dynamic_batch()
    payload['generation']['source']['task_list_version'] = 3  # type: ignore[index]

    with pytest.raises(ValidationError, match='task_list_version'):
        BatchSubmission.model_validate(payload)


@pytest.mark.parametrize('page_size', [0, 10_001])
def test_cloud_compose_rejects_out_of_range_page_size(page_size: int):
    uri = 'research/tasks@v1'
    payload = _dynamic_batch(
        code={'mode': 'config_only'},
        generation={
            'mode': 'remote_materialize',
            'source': {
                'type': 'compose',
                'compose': {
                    'parts': [{'kind': 'data', 'value': uri, 'source': f'--data {uri}'}],
                    'page_size': page_size,
                },
            },
        },
    )

    with pytest.raises(ValidationError, match='page_size'):
        BatchSubmission.model_validate(payload)


@pytest.mark.parametrize('path', ['./tasks.jsonl', '/tmp/tasks.jsonl', '../tasks.jsonl', 'file:///tmp/tasks.jsonl'])
def test_cloud_compose_rejects_local_dataset_paths(path: str):
    payload = _dynamic_batch(
        code={'mode': 'config_only'},
        generation={
            'mode': 'remote_materialize',
            'source': {
                'type': 'compose',
                'compose': {'parts': [{'kind': 'data', 'value': path, 'source': f'--data {path}'}]},
            },
        },
    )

    with pytest.raises(ValidationError, match='Dataset Service URI'):
        BatchSubmission.model_validate(payload)


def test_artifact_refs_include_only_code_and_task_list_artifacts():
    submission = BatchSubmission.model_validate({
        **_dynamic_batch(),
        'generation': {
            'mode': 'local_task_list',
            'task_list': _task_list_spec(),
        },
    })

    assert [(ref.kind, ref.artifact_id) for ref in artifact_refs(submission)] == [
        ('source_package', 'artifact-a'),
        ('nexus_wheel', 'artifact-b'),
        ('uv_lock', 'artifact-d'),
        ('task_list', 'artifact-c'),
    ]


def test_v3_local_task_list_requires_data_list_and_returns_both_artifact_refs():
    task_list = _task_list_spec()
    task_list['manifest']['record_version'] = 3  # type: ignore[index]
    submission = BatchSubmission.model_validate({
        **_dynamic_batch(),
        'generation': {
            'mode': 'local_task_list',
            'task_list': task_list,
            'data_list': _data_list_spec(),
        },
    })

    assert submission.generation.data_list is not None  # type: ignore[union-attr]
    assert [(ref.kind, ref.artifact_id) for ref in artifact_refs(submission)][-2:] == [
        ('task_list', 'artifact-c'),
        ('data_list', 'artifact-e'),
    ]


def test_v3_local_task_list_rejects_missing_data_list():
    task_list = _task_list_spec()
    task_list['manifest']['record_version'] = 3  # type: ignore[index]

    with pytest.raises(ValidationError, match='record_version=3'):
        BatchSubmission.model_validate({
            **_dynamic_batch(),
            'generation': {'mode': 'local_task_list', 'task_list': task_list},
        })


def test_v1_or_v2_local_task_list_rejects_data_list():
    with pytest.raises(ValidationError, match='record_version=3'):
        BatchSubmission.model_validate({
            **_dynamic_batch(),
            'generation': {
                'mode': 'local_task_list',
                'task_list': _task_list_spec(),
                'data_list': _data_list_spec(),
            },
        })


@pytest.mark.parametrize(
    'data_list',
    [
        {
            **_data_list_spec(),
            'ref': _artifact('task_list', 'e'),
        },
        {
            **_data_list_spec(),
            'format': 'jsonl',
        },
        {
            **_data_list_spec(),
            'manifest': {**_data_list_spec()['manifest'], 'sha256': 'f' * 64},  # type: ignore[dict-item]
        },
        {
            **_data_list_spec(),
            'manifest': {**_data_list_spec()['manifest'], 'size_bytes': 18},  # type: ignore[dict-item]
        },
        _data_list_spec(frame_count=0),
    ],
)
def test_data_list_requires_kind_format_identity_and_nonempty_frames(data_list):
    task_list = _task_list_spec()
    task_list['manifest']['record_version'] = 3  # type: ignore[index]

    with pytest.raises(ValidationError):
        BatchSubmission.model_validate({
            **_dynamic_batch(),
            'generation': {
                'mode': 'local_task_list',
                'task_list': task_list,
                'data_list': data_list,
            },
        })


def test_dataset_is_not_an_artifact_kind_but_run_result_is():
    with pytest.raises(ValidationError):
        ArtifactRef.model_validate(_artifact('dataset', 'd'))

    assert ArtifactRef.model_validate(_artifact('run_result', 'd')).kind == 'run_result'
