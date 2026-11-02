from __future__ import annotations

import pytest

from nexus.distributed.submission.workload_image import (
    resolve_workload_image,
    validate_workload_image_version,
    workload_image_repository,
)


@pytest.mark.parametrize('value', ['user-build-20260802-001', 'Release_1.2.3', 'a' * 128])
def test_workload_image_version_accepts_docker_tags(value: str):
    assert validate_workload_image_version(value) == value


@pytest.mark.parametrize(
    'value',
    [
        '',
        'latest',
        'LATEST',
        '.release',
        '-release',
        'release candidate',
        'registry.example/team/workload',
        'workload:release',
        'workload@sha256:' + 'a' * 64,
        'a' * 129,
    ],
)
def test_workload_image_version_rejects_floating_or_full_image_references(value: str):
    with pytest.raises(ValueError, match='image_version'):
        validate_workload_image_version(value)


@pytest.mark.parametrize(
    ('base_image', 'repository'),
    [
        ('registry.example/team/workload@sha256:' + 'a' * 64, 'registry.example/team/workload'),
        ('registry.example/team/workload:platform-v1@sha256:' + 'a' * 64, 'registry.example/team/workload'),
        ('registry.example:5000/team/workload:platform-v1', 'registry.example:5000/team/workload'),
        ('nexus-distributed-workload:platform-v1', 'nexus-distributed-workload'),
        ('nexus-distributed-workload', 'nexus-distributed-workload'),
    ],
)
def test_workload_image_repository_strips_only_digest_or_final_tag(base_image: str, repository: str):
    assert workload_image_repository(base_image) == repository
    assert resolve_workload_image(base_image, 'user-build-1') == f'{repository}:user-build-1'


def test_production_workload_image_repository_requires_digest_pinned_base():
    with pytest.raises(ValueError, match='digest-pinned'):
        workload_image_repository('registry.example/team/workload:platform-v1', require_digest=True)

    digest_base = 'registry.example/team/workload@sha256:' + 'a' * 64
    assert workload_image_repository(digest_base, require_digest=True) == 'registry.example/team/workload'


@pytest.mark.parametrize('base_image', ['', 'https://registry.example/workload:tag', 'registry.example/team/'])
def test_workload_image_repository_rejects_invalid_deployment_images(base_image: str):
    with pytest.raises(ValueError, match='base image'):
        workload_image_repository(base_image)
