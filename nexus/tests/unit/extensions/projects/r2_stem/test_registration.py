from pathlib import Path

from nexus.core import agent_registry, task_registry, tool_registry
from nexus.core.module_meta import parse_directory_module
from nexus.extensions.projects.r2_stem import (
    run_mini_swe_agent_stem,
    run_science_agent_task,
)
from nexus.extensions.projects.r2_stem.generator import get_batch_metadata
from nexus.extensions.projects.r2_stem.tools import bash_tool


def test_r2_stem_entries_use_stem_suffix() -> None:
    assert agent_registry.get('mini_swe_agent_stem') is run_mini_swe_agent_stem
    assert task_registry.get('science_agent_task') is run_science_agent_task
    assert tool_registry.get('mini_swe_agent_stem/bash') is bash_tool
    assert get_batch_metadata()['task_name'] == 'science_agent_task'


def test_task_name_avoids_swe_substring() -> None:
    # terminus2 treats any task whose name contains 'swe' as a SWE repository task and
    # replaces the user prompt with its PR-fixing template, which STEM samples must not get.
    assert 'swe' not in run_science_agent_task.nexus_meta['name']
    assert 'swe' not in get_batch_metadata()['task_name']


def test_r2_stem_manifest_exposes_renamed_entries() -> None:
    project_root = Path(__file__).resolve().parents[5]
    metadata = parse_directory_module(project_root / 'nexus/extensions/projects/r2_stem', project_root)

    assert metadata is not None
    assert metadata.module_path == 'nexus.extensions.projects.r2_stem'
    assert metadata.objects == {
        'agents': {'mini_swe_agent_stem': 'nexus.extensions.projects.r2_stem.agent'},
        'tasks': {'science_agent_task': 'nexus.extensions.projects.r2_stem.task'},
        'tools': {'mini_swe_agent_stem/bash': 'nexus.extensions.projects.r2_stem.tools'},
    }
