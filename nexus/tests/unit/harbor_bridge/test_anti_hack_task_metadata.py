"""``_anti_hack_task_metadata`` must hand the strategy what task.toml actually says.

The two tables it publishes -- ``harbor`` from ``[metadata]`` and ``other`` from
``[other]`` -- are the only anchors the detector has. Harbor's own loader drops
``[other]`` from the parsed config object, so task.py re-reads the file for it; if that
fallback ever regresses, every SWE task silently loses its base_commit and the whole
answer-lookup tier gates itself off while still reporting ``completed``.
"""

from __future__ import annotations

from nexus.extensions.harbor_bridge.task import HarborBridgeTaskInput, _anti_hack_task_metadata


def test_task_reads_harbor_metadata_and_other() -> None:
    harbor_task = type(
        'HarborTask',
        (),
        {
            'name': 'owner__project-123',
            'config': {
                'metadata': {'source_repo': 'owner/project'},
                'other': {
                    'repo': 'owner/project',
                    'base_commit': 'abc1234def5678',
                    'history_truncated': False,
                },
            },
        },
    )()
    task_input = HarborBridgeTaskInput(
        workdir='/testbed',
        instance_id='owner__project-123',
        run_agent=lambda **_: None,
    )

    assert _anti_hack_task_metadata(harbor_task, task_input) == {
        'instance_id': 'owner__project-123',
        'task_name': 'owner__project-123',
        'workdir': '/testbed',
        'harbor': {
            'source_repo': 'owner/project',
        },
        'other': {
            'repo': 'owner/project',
            'base_commit': 'abc1234def5678',
            'history_truncated': False,
        },
    }


def test_task_reads_other_discarded_by_real_harbor_loader(tmp_path) -> None:
    from harbor.models.task.task import Task

    task_dir = tmp_path / 'owner__project-123'
    task_dir.mkdir()
    (task_dir / 'instruction.md').write_text('Complete the task.\n')
    (task_dir / 'task.toml').write_text(
        """
[metadata]
source_repo = "owner/project"

[other]
repo = "owner/project"
base_commit = "abc1234def5678"
history_truncated = false
""".lstrip()
    )
    harbor_task = Task(task_dir=task_dir, disable_verification=True)
    assert not hasattr(harbor_task.config, 'other')

    task_input = HarborBridgeTaskInput(
        workdir='/testbed',
        instance_id='owner__project-123',
        run_agent=lambda **_: None,
    )

    assert _anti_hack_task_metadata(harbor_task, task_input) == {
        'instance_id': 'owner__project-123',
        'task_name': 'owner__project-123',
        'workdir': '/testbed',
        'harbor': {
            'source_repo': 'owner/project',
        },
        'other': {
            'repo': 'owner/project',
            'base_commit': 'abc1234def5678',
            'history_truncated': False,
        },
    }
