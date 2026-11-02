from pathlib import Path

from nexus.core import agent_registry, task_registry
from nexus.core.module_meta import parse_directory_module
from nexus.extensions.projects.hy_live.agent import live_agent
from nexus.extensions.projects.hy_live.task import live_agent_task


def test_live_agent_registry_entries() -> None:
    assert agent_registry.get('hy-live') is live_agent
    assert task_registry.get('hy-live') is live_agent_task


def test_live_agent_manifest_splits_agent_and_task() -> None:
    project_root = Path(__file__).resolve().parents[5]
    metadata = parse_directory_module(project_root / 'nexus/extensions/projects/hy_live', project_root)

    assert metadata is not None
    assert metadata.name == 'hy-live'
    assert metadata.module_path == 'nexus.extensions.projects.hy_live'
    assert metadata.objects == {
        'agents': {'hy-live': 'nexus.extensions.projects.hy_live.agent'},
        'tasks': {'hy-live': 'nexus.extensions.projects.hy_live.task'},
        'span_processors': {'hy_live_langfuse_tracker': 'nexus.extensions.projects.hy_live.langfuse_trace'},
    }
