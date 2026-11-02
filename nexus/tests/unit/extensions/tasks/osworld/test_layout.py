"""OSWorld layout regressions."""

from __future__ import annotations

import ast
import importlib
import json

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import tomllib

import nexus
import nexus.extensions.tasks.os_world as os_world

from nexus.core.module_meta import parse_directory_module
from nexus.extensions.tasks.os_world.data import paths, task_source
from nexus.extensions.tasks.os_world.execution import large_sandbox, run_dir_hook
from nexus.extensions.tasks.os_world.generators import config, generator_native
from nexus.extensions.tasks.os_world.scripts import build_osworld_dataset


ROOT = Path(__file__).resolve().parents[5]


PACKAGE = ROOT / 'nexus/extensions/tasks/os_world'


def test_root_contains_task_registration_and_output_contract():
    assert {p.name for p in PACKAGE.glob('*.py')} == {'task.py', '__init__.py', 'outputs.py'}
    # Tests belong to the repository, not the installed task package.
    assert not list((PACKAGE / 'unit_tests').glob('*.py'))
    assert (PACKAGE / 'prompts/claude_code').is_dir()
    assert (PACKAGE / 'assets/codex').is_dir()


def test_six_fixed_entry_files_and_no_other_generator():
    expected = {f'generator_native_{family}_{harness}.py' for family, harness in generator_native.SUPPORTED}
    assert {p.name for p in (PACKAGE / 'generators').glob('generator*.py')} == expected | {'generator_native.py'}
    assert len(expected) == 6


def test_registry_only_advertises_native_entry_and_live_hook():
    metadata = tomllib.loads((PACKAGE / 'pyproject.toml').read_text())['tool']['nexus']
    assert metadata['run_parts'] == {'osworld_native': 'nexus.extensions.tasks.os_world.generators.run_parts'}
    assert metadata['run_hooks'] == {
        'osworld_run_dir': 'nexus.extensions.tasks.os_world.execution.run_dir_hook',
        'osworld_output_contract': 'nexus.extensions.tasks.os_world.outputs',
    }
    assert 'commands' not in metadata
    for category in ('run_parts', 'run_hooks', 'tools', 'tasks'):
        for module in metadata[category].values():
            importlib.import_module(module)


def test_moved_config_checks_the_original_module_dependencies(monkeypatch):
    seen = []
    original = parse_directory_module

    def parse(directory, project):
        seen.append((directory, project))
        return original(directory, project)

    monkeypatch.setattr(config, 'parse_directory_module', parse)
    monkeypatch.setattr(config, 'check_module', lambda meta: type('Report', (), {'missing_host': []})())
    assert config._missing_evaluator_host_deps() == []
    assert seen == [(PACKAGE, ROOT)]
    assert paths.DEFAULT_CACHE_ROOT == PACKAGE / 'cache'


def test_selection_requires_dataset_even_with_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv('OSWORLD_DATASET_URI', raising=False)
    monkeypatch.setenv('OSWORLD_EXAMPLES_ROOT', str(tmp_path))
    monkeypatch.delenv('OSWORLD_LIMIT', raising=False)
    with pytest.raises(ValueError, match='OSWORLD_DATASET_URI'):
        list(task_source.iter_selected_tasks())


def test_only_packaging_enumerates_checkout_with_explicit_filters(monkeypatch, tmp_path):
    root = tmp_path / 'examples'
    (root / 'chrome').mkdir(parents=True)
    for task_id in ('a', 'b', 'c'):
        (root / 'chrome' / f'{task_id}.json').write_text('{}')
    (tmp_path / 'test_nogdrive.json').write_text(json.dumps({'chrome': ['a', 'b', 'c']}))
    for name in ('OSWORLD_TEST_META_PATH', 'OSWORLD_TASK_IDS', 'OSWORLD_DOMAIN'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('OSWORLD_EXAMPLES_ROOT', str(root))
    monkeypatch.setenv('OSWORLD_SKIP', '1')
    monkeypatch.setenv('OSWORLD_LIMIT', '1')
    assert build_osworld_dataset._selected_tasks() == [('chrome', 'b', root / 'chrome/b.json')]


@pytest.mark.asyncio
async def test_moved_run_hook_stamps_the_actual_checkout(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(run_dir_hook, 'source_commit', lambda root: seen.append(root) or 'fixture-commit')
    hook = run_dir_hook.OSWorldRunDirHook(run_dir=str(tmp_path))
    await hook.on_context_ready(None)
    assert seen == [ROOT]
    assert hook.get_result()['run_dir'] == str(tmp_path)
    assert (tmp_path / 'source_commit.txt').read_text() == 'fixture-commit\n'


@pytest.mark.asyncio
async def test_bare_large_sandbox_does_not_install_a_cli(monkeypatch):
    execute = AsyncMock(return_value=(0, 'GUEST_FREE_BYTES=999999999999', ''))
    monkeypatch.setattr(large_sandbox, '_run_detached', execute)
    provider = large_sandbox.OSWorldV1LargeRuntimeProvider(cli_harness='bare', secret_id='test', secret_key='test')
    await provider._bootstrap_with_retries(object())
    script = execute.call_args.args[1]
    assert 'GUEST_FREE_BYTES' in script
    for forbidden in ('claude', 'codex', 'npm install', 'pip install'):
        assert forbidden not in script


NEXUS_ROOT = Path(nexus.__file__).parent


V1_ROOT = Path(os_world.__file__).parent


GENERATOR_STEMS = ('generator',)


V1_OWNED_KNOBS = frozenset({
    'OSWORLD_AGS_PROVISION_MODE',
    'OSWORLD_ARTIFACT_DIR',
    'OSWORLD_EXAMPLES_ROOT',
    'OSWORLD_MAX_ITERATIONS',
    'OSWORLD_RECORD_SCREENSHOTS',
    'OSWORLD_SESSION_NAMESPACE',
    'OSWORLD_TUNNEL_PORT',
})


OUTSIDE_SOURCE_FILES = sorted(
    path
    for path in NEXUS_ROOT.rglob('*.py')
    if V1_ROOT not in path.parents and not path.name.startswith(GENERATOR_STEMS)
)


def _uppercase_call_names(source: str) -> set[str]:
    """Uppercase string names this module passes to any call, plus ``os.environ[...]``.

    Deliberately blunt: an env read wrapped in a module-local helper looks like
    ``_env_first('OSWORLD_CLAUDE_MODEL')``, which no amount of call-shape matching catches —
    that wrapper is precisely how the agent stayed invisible to the 阶段 2 gate. Over-matching
    an unrelated uppercase argument is harmless here, because the filter below only keeps
    names the V1 task owns.
    """

    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.isupper():
                names.add(first.value)
        elif isinstance(node, ast.Subscript) and getattr(node.value, 'attr', None) == 'environ':
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                names.add(key.value)
    return names


def test_no_module_outside_os_world_reads_a_v1_knob() -> None:
    # V2 is an independent task with an explicit generator entry boundary. It
    # retains these public CLI knob names, not a runtime/global env fallback.
    v2_entry = NEXUS_ROOT / 'extensions/tasks/os_world_v2/generators/policy.py'
    violations = {
        str(path.relative_to(NEXUS_ROOT)): sorted(leaked)
        for path in OUTSIDE_SOURCE_FILES
        if path != v2_entry
        if (leaked := V1_OWNED_KNOBS & _uppercase_call_names(path.read_text()))
    }
    assert not violations, f'Pass task-owned options explicitly: {violations}'


def test_every_owned_knob_is_still_a_v1_knob() -> None:
    """A name V1 stopped reading is no longer V1's to own.

    Without this the list drifts into a set of names that guard nothing, and the first real
    breach lands on a name someone quietly dropped from it.
    """
    read_by_v1 = set()
    for path in V1_ROOT.rglob('*.py'):
        read_by_v1 |= _uppercase_call_names(path.read_text())

    assert sorted(V1_OWNED_KNOBS - read_by_v1) == []


TASK_DOMAIN_PREFIXES = (
    'OSWORLD_',
    'OPUS_CC_',
    'CC_',
    'CODEX_',
    'HUNYUAN_',
    'AGS_',
    'AIHUB_',
)


ENTRY_FILES = frozenset({'task.py', 'task_source.py'})


TASK_SOURCE_FILES = sorted(
    path
    for path in V1_ROOT.rglob('*.py')
    if 'generators' not in path.relative_to(V1_ROOT).parts
    and 'scripts' not in path.relative_to(V1_ROOT).parts
    and path.name not in ENTRY_FILES
)


def test_only_the_entries_read_a_task_domain_variable() -> None:
    violations = {}
    for path in TASK_SOURCE_FILES:
        leaked = sorted(
            name for name in _uppercase_call_names(path.read_text()) if name.startswith(TASK_DOMAIN_PREFIXES)
        )
        if leaked:
            violations[str(path.relative_to(V1_ROOT))] = leaked
    assert not violations, f'Resolve task settings at entrypoints: {violations}'
