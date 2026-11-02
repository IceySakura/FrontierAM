from pathlib import Path

import pytest

from nexus.extensions.harbor_bridge.identity import build_task_identity, task_content_sha256, task_run_id


def _write_identity_task(root: Path) -> Path:
    task_dir = root / 'example'
    (task_dir / 'tests' / 'nested').mkdir(parents=True)
    (task_dir / 'environment').mkdir()
    (task_dir / 'solution').mkdir()
    (task_dir / 'task.toml').write_text('[environment]\ndocker_image = "agent:v1"\n')
    (task_dir / 'instruction.md').write_text('Complete the task.\n')
    (task_dir / 'tests' / 'test.sh').write_text('#!/bin/sh\n')
    (task_dir / 'tests' / 'nested' / 'assert.py').write_text('assert True\n')
    (task_dir / 'environment' / 'Dockerfile').write_text('FROM python:3.13\n')
    (task_dir / 'solution' / 'solve.sh').write_text('#!/bin/sh\n')
    return task_dir


@pytest.mark.parametrize(
    ('relative_path', 'replacement'),
    [
        ('task.toml', '[environment]\ndocker_image = "agent:v2"\n'),
        ('instruction.md', 'Complete a different task.\n'),
        ('tests/test.sh', '#!/bin/sh\nexit 1\n'),
        ('tests/nested/assert.py', 'assert False\n'),
    ],
)
def test_task_identity_changes_for_canonical_inputs(
    tmp_path: Path,
    relative_path: str,
    replacement: str,
) -> None:
    task_dir = _write_identity_task(tmp_path)
    before = task_content_sha256(task_dir)

    (task_dir / relative_path).write_text(replacement)

    assert task_content_sha256(task_dir) != before


def test_task_identity_ignores_noncanonical_task_files(tmp_path: Path) -> None:
    task_dir = _write_identity_task(tmp_path)
    before = task_content_sha256(task_dir)

    (task_dir / 'environment' / 'Dockerfile').write_text('FROM ubuntu:latest\n')
    (task_dir / 'solution' / 'solve.sh').write_text('exit 1\n')
    (task_dir / 'notes.md').write_text('Local notes.\n')

    assert task_content_sha256(task_dir) == before


def test_task_identity_is_location_independent(tmp_path: Path) -> None:
    first = _write_identity_task(tmp_path / 'first')
    second = _write_identity_task(tmp_path / 'second')

    assert task_content_sha256(first) == task_content_sha256(second)
    assert task_run_id(first.name, task_content_sha256(first)) == task_run_id(
        second.name,
        task_content_sha256(second),
    )


def test_task_identity_builds_run_id_and_shared_metadata(tmp_path: Path) -> None:
    task_dir = _write_identity_task(tmp_path)

    identity = build_task_identity(task_dir, 'example')

    assert identity.run_id == task_run_id('example', identity.content_sha256)
    assert identity.metadata(
        task_dir=task_dir,
        config={
            'environment': {'image_digest': 'sha256:image'},
            'judge': {'reward_kind': 'continuous'},
        },
    ) == {
        'format': 't3-harbor',
        'source_path': str(task_dir),
        'task_identity_version': 't3h1',
        'task_content_sha256': identity.content_sha256,
        'task_toml_sha256': identity.task_toml_sha256,
        'image_digest': 'sha256:image',
        'reward_kind': 'continuous',
    }
