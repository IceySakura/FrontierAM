from nexus.utils.tool_analyzer import _DEFAULT_TOOL_STATS_TASKS


def test_harbor_tasks_collect_tool_statistics_by_default() -> None:
    assert 'harbor_bridge_task' in _DEFAULT_TOOL_STATS_TASKS
    assert 'harbor_trial_task' in _DEFAULT_TOOL_STATS_TASKS
