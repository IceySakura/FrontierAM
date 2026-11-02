import logging

from pathlib import Path

import pytest

from nexus.core.module_dependencies import (
    ModuleDependencyError,
    ModuleSelector,
    resolve_host_module_dependencies,
    selectors_from_run_input,
)
from nexus.core.module_meta import ModuleMeta, discover_all_modules
from nexus.runner.single import RunInput


ROOT = Path(__file__).resolve().parents[2]


def test_runner_hooks_resolve_from_config_and_discovered_metadata():
    selectors = selectors_from_run_input(
        {'runner': {'hooks': {'video': {'name': 'gui_recording', 'args': {'fps': 12}}}}}
    )
    assert selectors == (ModuleSelector(kind='object', object_type='run_hooks', value='gui_recording'),)
    resolved = resolve_host_module_dependencies(selectors, discover_all_modules(ROOT))
    assert [module.module_path for module in resolved.modules] == ['nexus.runner.hooks.gui_recording']


def _frontend_run_input() -> dict:
    return RunInput(
        imports=[
            'nexus.extensions.tasks.frontend',
            'nexus.extensions.agents.claude_code',
        ],
        task_name='frontend_task',
        agent_name='claude_code',
        runtime_provider_name='gongfeng',
        openai_clients={'main': {'name': 'litellm', 'config': {}}},
        processors={'chat': {'name': 'chat_tracker', 'args': {}}},
    ).model_dump(mode='json', exclude={'compaction_manager'})


def test_frontend_run_input_resolves_litellm_host_requirements() -> None:
    selectors = selectors_from_run_input(_frontend_run_input())
    resolved = resolve_host_module_dependencies(
        selectors,
        discover_all_modules(project_root=ROOT, include_compiled_manifest=False),
    )

    assert selectors == (
        ModuleSelector(kind='import', value='nexus.extensions.agents.claude_code'),
        ModuleSelector(kind='import', value='nexus.extensions.tasks.frontend'),
        ModuleSelector(kind='object', object_type='agents', value='claude_code'),
        ModuleSelector(kind='object', object_type='openai_clients', value='litellm'),
        ModuleSelector(kind='object', object_type='runtime_providers', value='gongfeng'),
        ModuleSelector(kind='object', object_type='span_processors', value='chat_tracker'),
        ModuleSelector(kind='object', object_type='tasks', value='frontend_task'),
    )
    assert tuple(module.name for module in resolved.modules) == (
        'claude_code',
        'litellm',
        'frontend',
        'gongfeng',
        'chat_tracker',
    )
    assert resolved.requirements == ('boto3', 'litellm>=1.83.14,<2')


def test_trajectory_observer_resolves_its_optional_host_dependency() -> None:
    selectors = selectors_from_run_input({
        'gateway_trajectory_observers': {'langfuse': {'name': 'langfuse', 'args': {}}},
    })
    resolved = resolve_host_module_dependencies(
        selectors,
        discover_all_modules(project_root=ROOT, include_compiled_manifest=False),
    )

    assert selectors == (ModuleSelector(kind='object', object_type='trajectory_observers', value='langfuse'),)
    assert tuple(module.name for module in resolved.modules) == ('langfuse',)
    assert resolved.requirements == ('langfuse>=4.0.0',)


def test_unknown_registry_object_is_warned_and_skipped(caplog) -> None:
    selectors = selectors_from_run_input({'task_name': 'not_declared'})

    with caplog.at_level(logging.WARNING, logger='nexus.core.module_dependencies'):
        resolved = resolve_host_module_dependencies(selectors, [])

    assert resolved.modules == ()
    assert resolved.requirements == ()
    assert 'module_dependency_selector_unresolved: tasks:not_declared' in caplog.messages


def test_invalid_import_is_rejected_before_module_resolution() -> None:
    with pytest.raises(ModuleDependencyError) as exc_info:
        selectors_from_run_input({'task_name': 'identity', 'imports': ['nexus.extensions..broken']})

    assert exc_info.value.code == 'module_import_invalid'
    assert exc_info.value.detail == 'nexus.extensions..broken'


def test_unmatched_imports_are_warned_and_known_objects_still_resolve(caplog) -> None:
    module = ModuleMeta(
        name='identity',
        module_path='nexus.extensions.tasks.identity',
        fs_path=Path('/tmp/identity'),
        objects={'tasks': {'identity': 'nexus.extensions.tasks.identity'}},
    )
    external = selectors_from_run_input({'task_name': 'identity', 'imports': ['customer.tasks']})
    with caplog.at_level(logging.WARNING, logger='nexus.core.module_dependencies'):
        assert resolve_host_module_dependencies(external, [module]).modules == (module,)
    assert 'module_dependency_selector_unresolved: imports:customer.tasks' in caplog.messages

    caplog.clear()
    internal = selectors_from_run_input({'task_name': 'identity', 'imports': ['nexus.extensions.tasks.missing']})
    with caplog.at_level(logging.WARNING, logger='nexus.core.module_dependencies'):
        assert resolve_host_module_dependencies(internal, [module]).modules == (module,)
    assert 'module_dependency_selector_unresolved: imports:nexus.extensions.tasks.missing' in caplog.messages


def test_broad_import_and_undeclared_object_do_not_hide_known_module_dependencies(caplog) -> None:
    module = ModuleMeta(
        name='one_agent_infer',
        module_path='nexus.extensions.tasks.one_agent_infer',
        fs_path=Path('/tmp/one_agent_infer'),
        host_extras=('httpx>=0.27',),
        objects={'tasks': {'one_agent_infer': 'nexus.extensions.tasks.one_agent_infer.task'}},
    )
    selectors = selectors_from_run_input({
        'task_name': 'one_agent_infer',
        'imports': ['nexus.extensions'],
        'openai_clients': {'main': {'name': 'one_agent_direct'}},
    })

    with caplog.at_level(logging.WARNING, logger='nexus.core.module_dependencies'):
        resolved = resolve_host_module_dependencies(selectors, [module])

    assert resolved.modules == (module,)
    assert resolved.requirements == ('httpx>=0.27',)
    assert set(caplog.messages) == {
        'module_dependency_selector_unresolved: imports:nexus.extensions',
        'module_dependency_selector_unresolved: openai_clients:one_agent_direct',
    }


def test_direct_url_host_requirement_is_rejected() -> None:
    module = ModuleMeta(
        name='unsafe',
        module_path='nexus.extensions.tasks.unsafe',
        fs_path=Path('/tmp/unsafe'),
        host_extras=('demo @ https://example.test/demo.whl',),
        objects={'tasks': {'unsafe': 'nexus.extensions.tasks.unsafe'}},
    )

    with pytest.raises(ModuleDependencyError) as exc_info:
        resolve_host_module_dependencies(
            (ModuleSelector(kind='object', object_type='tasks', value='unsafe'),),
            [module],
        )

    assert exc_info.value.code == 'module_requirement_unsafe'
    assert exc_info.value.detail == 'demo @ https://example.test/demo.whl'
