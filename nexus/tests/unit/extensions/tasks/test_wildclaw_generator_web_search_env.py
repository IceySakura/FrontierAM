from pathlib import Path

from nexus.extensions.tasks.wildclaw_bench.generator import _build_env_vars, _build_run_input


def test_build_env_vars_passes_serper_bridge_configuration(monkeypatch, tmp_path: Path) -> None:
    task_file = tmp_path / 'task.md'
    task_file.write_text(
        """---
id: task
---
## Env
```
CUSTOM_ENV
```
""",
        encoding='utf-8',
    )
    for key, value in {
        'SERPER_API_KEY': 'serper-key',
        'SERPER_API_TYPE': 'distillation',
        'SEARCH_SERPER_URL': 'http://search-gateway.local/search',
        'SEARCH_SERPER_APP_ID': 'app-id',
        'SEARCH_SERPER_APP_KEY': 'app-key',
        'SERPER_KEY_ID': 'key-id',
        'CUSTOM_ENV': 'custom-value',
    }.items():
        monkeypatch.setenv(key, value)

    env_vars = _build_env_vars(task_file)

    assert env_vars['SERPER_API_KEY'] == 'serper-key'
    assert env_vars['SERPER_API_TYPE'] == 'distillation'
    assert env_vars['SEARCH_SERPER_URL'] == 'http://search-gateway.local/search'
    assert env_vars['SEARCH_SERPER_APP_ID'] == 'app-id'
    assert env_vars['SEARCH_SERPER_APP_KEY'] == 'app-key'
    assert env_vars['SERPER_KEY_ID'] == 'key-id'
    assert env_vars['CUSTOM_ENV'] == 'custom-value'


def test_build_run_input_injects_serper_bridge_configuration_into_docker_args(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / 'wildclaw'
    task_dir = root_dir / 'tasks' / '04_Search_Retrieval'
    workspace_dir = root_dir / 'workspaces' / 'task'
    (workspace_dir / 'exec').mkdir(parents=True)
    task_dir.mkdir(parents=True)
    task_file = task_dir / 'task.md'
    task_file.write_text(
        """---
id: search_task
name: Search Task
category: 04_Search_Retrieval
timeout_seconds: 60
---
## Workspace Path
```
workspaces/task
```
## Env
```
CUSTOM_ENV
```
""",
        encoding='utf-8',
    )
    for key, value in {
        'OPENCLAW_LITELLM_API_KEY': 'llm-key',
        'OPENCLAW_LITELLM_BASE_URL': 'http://llm.local/v1',
        'OPENCLAW_MODEL': 'gpt-test',
        'WILDCLAW_IMAGE': 'wildclawbench-ubuntu:test',
        'SERPER_API_KEY': 'serper-key',
        'SERPER_API_TYPE': 'distillation',
        'SEARCH_SERPER_URL': 'http://search-gateway.local/search',
        'SEARCH_SERPER_APP_ID': 'app-id',
        'SEARCH_SERPER_APP_KEY': 'app-key',
        'SERPER_KEY_ID': 'key-id',
        'CUSTOM_ENV': 'custom-value',
    }.items():
        monkeypatch.setenv(key, value)

    run_input, _metadata = _build_run_input(task_file, root_dir)

    extra_args = run_input.runtime_configs['main']['extra_args']
    env_args = {extra_args[index + 1] for index, item in enumerate(extra_args) if item == '-e'}
    assert 'SERPER_API_KEY=serper-key' in env_args
    assert 'SERPER_API_TYPE=distillation' in env_args
    assert 'SEARCH_SERPER_URL=http://search-gateway.local/search' in env_args
    assert 'SEARCH_SERPER_APP_ID=app-id' in env_args
    assert 'SEARCH_SERPER_APP_KEY=app-key' in env_args
    assert 'SERPER_KEY_ID=key-id' in env_args
    assert 'CUSTOM_ENV=custom-value' in env_args
