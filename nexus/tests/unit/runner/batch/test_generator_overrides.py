from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys

from pathlib import Path

import pytest

from nexus.runner.batch import BatchItem
from nexus.runner.batch.runner import _iter_generator_items
from nexus.runner.compose.compose_hooks import postprocess_generated_items
from nexus.runner.compose.parser import parse_ordered_parts
from nexus.runner.compose.resolver import resolve_non_data_part
from nexus.runner.single import RunInput


def test_overrides_merge_in_order_without_mutating_generator_items():
    original = BatchItem(
        run_id='case',
        metadata={'label': 'kept'},
        config=RunInput(task_name='noop', task_config={'nested': {'keep': 1, 'drop': 2}, 'items': ['base']}),
    )
    parts = parse_ordered_parts([
        '--set',
        'task_config.items+=first',
        '--set-json',
        'task_config={"items":["replace"],"nested":{"drop":null}}',
        '--set=task_config.items+=last',
        '--set',
        'task_config.enabled=false',
    ])
    first, second = postprocess_generated_items([original, original], overrides=[resolve_non_data_part(p) for p in parts])
    assert first.config.task_config == {'nested': {'keep': 1}, 'items': ['replace', 'last'], 'enabled': False}
    assert first.run_id == 'case'
    assert first.metadata == original.metadata
    assert second.config == first.config
    assert original.config.task_config['items'] == ['base']
    assert original.config.task_config['nested']['drop'] == 2


@pytest.mark.parametrize(('command', 'workers'), [(['run', 'batch'], 1), (['run-batch'], 2)])
def test_cli_generator_overrides_persist_execute_export_and_resume(tmp_path, command, workers):
    generator = tmp_path / 'generator.py'
    generator.write_text("""
from nexus.core import TaskRunInput, TaskRunOutput, task_registry
from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput

class Input(TaskRunInput):
    marker: str
    items: list[str]
    enabled: bool

class Output(TaskRunOutput):
    seen: dict

@task_registry.register('generator_override_probe')
async def task(task_input: Input) -> Output:
    return Output(seen=task_input.model_dump())

def get_batch_metadata():
    return {'source': 'original-generator'}

def generate_configs():
    for i in range(4):
        yield BatchItem(run_id=f'case-{i}', metadata={'ordinal': i},
            config=RunInput(task_name='generator_override_probe', imports=['probe_task'],
                task_config={'marker': 'base', 'items': ['base'], 'enabled': True}))
""")
    task_source, generator_source = generator.read_text().split('def get_batch_metadata():', 1)
    (tmp_path / 'probe_task.py').write_text(task_source)
    generator.write_text(
        'import probe_task\nfrom nexus.runner.batch import BatchItem\nfrom nexus.runner.single import RunInput\n'
        + 'def get_batch_metadata():'
        + generator_source
    )
    env = dict(os.environ, PYTHONPATH=str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    part = tmp_path / ('part.yaml' if workers == 1 else 'part.json')
    part.write_text(
        'body:\n  agent_config:\n    working_dir: /from-part\n  task_config:\n    marker: from-part\n    items: [part]\n'
    )
    if workers == 2:
        part.write_text(
            json.dumps({
                'body': {
                    'agent_config': {'working_dir': '/from-part'},
                    'task_config': {'marker': 'from-part', 'items': ['part']},
                }
            })
        )
    db = tmp_path / 'runs.db'
    output = tmp_path / 'results.jsonl'
    argv = [
        sys.executable,
        '-m',
        'nexus.cli.main',
        *command,
        '--use',
        'worker_venv',
        '--limit',
        '2',
        '--generator',
        str(generator),
        '--database',
        str(db),
        '--max-concurrent',
        '2',
        '--worker-processes',
        str(workers),
        '--no-persist',
        'trace',
        '--no-persist',
        'log',
        '--export-format',
        'jsonl',
        '--export-path',
        str(output),
    ]
    overrides = [
        '--set',
        'task_config.marker=first',
        '--use',
        str(part),
        '--set-json',
        'task_config={"marker":"json","items":["reset"]}',
        '--set',
        'task_config.items+=tail',
        '--set=task_config.enabled=false',
        '--set-json',
        'extra={"note":"from-cli"}',
    ]
    result = subprocess.run([*argv, *overrides], capture_output=True, text=True, timeout=60, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in output.read_text().splitlines()]
    expected = {'marker': 'json', 'items': ['reset', 'tail'], 'enabled': False}
    assert {r['run_id'] for r in records} == {'case-0', 'case-1'}
    assert all(r['status'] == 'success' for r in records)
    assert all(r['task_output']['seen'] == expected for r in records)
    assert all(r['extra'] == {'note': 'from-cli'} for r in records)
    with sqlite3.connect(db) as conn:
        assert json.loads(conn.execute("SELECT value FROM batch_info WHERE key = 'metadata'").fetchone()[0]) == {
            'source': 'original-generator'
        }
    paths = [db] if workers == 1 else sorted(Path(str(db) + '.shards').glob('*.db'))
    assert len(paths) == workers
    for path in paths:
        with sqlite3.connect(path) as conn:
            for run_id, config, metadata in conn.execute('SELECT run_id, config, metadata FROM runs'):
                assert json.loads(config)['task_config'] == expected
                assert json.loads(config)['extra'] == {'note': 'from-cli'}
                assert json.loads(config)['agent_config']['working_dir'] == '/from-part'
                assert json.loads(config)['runtime_configs']['main']['worker_venv'] is True
                assert json.loads(metadata) == {'ordinal': int(run_id[-1])}
    # Rerunning uses locked stored configs; it must not append again or overwrite them.
    resumed = subprocess.run(
        [*argv, *overrides, '--set', 'task_config.marker=changed', '--limit', '0', '--no-resume'],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert 'ignoring generator options' in resumed.stdout + resumed.stderr
    assert all(json.loads(line)['task_output']['seen'] == expected for line in output.read_text().splitlines())


def test_overrides_do_not_resolve_unrelated_environment_references(monkeypatch):
    monkeypatch.setenv('GENERATOR_OVERRIDE_ENV', 'resolved-value')
    item = BatchItem(config=RunInput(task_name='noop', agent_config={'token': '${oc.env:GENERATOR_OVERRIDE_ENV}'}))
    parts = parse_ordered_parts(['--set', 'task_config.enabled=true'])
    updated = next(postprocess_generated_items([item], overrides=[resolve_non_data_part(part) for part in parts]))
    assert updated.config.agent_config == {'token': '${oc.env:GENERATOR_OVERRIDE_ENV}'}
    assert updated.config.task_config == {'enabled': True}


@pytest.mark.parametrize(
    ('field', 'value', 'expected'),
    [
        ('timeout', '600', 600.0),
        ('hold_signal_handlers', 'false', False),
        ('trace_drain_error_policy', 'raise', 'raise'),
    ],
)
def test_typed_environment_overrides_match_compose_without_resolving_untouched_sections(
    monkeypatch, field, value, expected
):
    from nexus.runner.compose import compile_single

    monkeypatch.setenv('GENERATOR_TYPED_OVERRIDE', value)
    monkeypatch.delenv('GENERATOR_UNTOUCHED_SECRET', raising=False)
    item = BatchItem(config=RunInput(task_name='noop', agent_config={'token': '${oc.env:GENERATOR_UNTOUCHED_SECRET}'}))
    parts = parse_ordered_parts(['--set', f'runner.{field}=${{oc.env:GENERATOR_TYPED_OVERRIDE}}'])
    updated = next(postprocess_generated_items([item], overrides=[resolve_non_data_part(part) for part in parts]))
    composed = compile_single([*parse_ordered_parts(['--set', 'task_name=noop']), *parts])
    assert getattr(updated.config.runner, field) == getattr(composed.runner, field) == expected
    assert updated.config.agent_config == item.config.agent_config


def test_changed_sections_preserve_opaque_extra_and_prompt(monkeypatch):
    monkeypatch.delenv('GENERATOR_OPAQUE_VALUE', raising=False)
    literal = '${oc.env:GENERATOR_OPAQUE_VALUE}'
    item = BatchItem(config=RunInput(task_name='noop'))
    parts = parse_ordered_parts([
        '--set-json',
        'extra=' + json.dumps({'literal': literal}),
        '--set-json',
        'task_config=' + json.dumps({'user_prompt': literal}),
    ])
    updated = next(postprocess_generated_items([item], overrides=[resolve_non_data_part(part) for part in parts]))
    assert updated.config.extra == {'literal': literal}
    assert updated.config.task_config == {'user_prompt': literal}


@pytest.mark.asyncio
@pytest.mark.parametrize('workers', [1, 2])
async def test_overlay_resolver_runs_once_for_generation_and_never_for_locked_resume(tmp_path, workers):
    from nexus.runner.batch import BatchConfig, run_batch

    generator = tmp_path / 'empty.py'
    generator.write_text('def generate_configs():\n    return []\n')
    calls = []

    def resolve():
        calls.append(True)
        return []

    config = BatchConfig(worker_processes=workers, enable_trace_db=False, enable_log_capture=False)
    database = tmp_path / 'runs.db'
    first = await run_batch(generator, database, config, generator_overrides=resolve)
    assert first.total == 0
    assert calls == [True]

    def unavailable():
        raise AssertionError('locked resume must not resolve overlays')

    resumed = await run_batch(generator, database, config, generator_overrides=unavailable)
    assert resumed.total == 0


@pytest.mark.parametrize('limit', [0, 1, 3, 8, None])
def test_generator_limit_does_not_overconsume_and_closes_iterator(limit):
    seen = []
    closed = []
    calls = []

    class Loader:
        def generate(self):
            calls.append(True)
            return self.items()

        def items(self):
            try:
                for index in range(3):
                    if limit is not None and len(seen) >= limit:
                        raise AssertionError('consumed beyond limit')
                    seen.append(index)
                    yield BatchItem(config=RunInput(task_name='noop'), run_id=str(index))
            finally:
                closed.append(True)

    result = list(_iter_generator_items(Loader(), None, limit))
    expected = 3 if limit is None else min(limit, 3)
    assert len(result) == expected
    assert seen == list(range(expected))
    assert bool(calls) is (limit != 0)
    assert bool(closed) is (limit != 0)


@pytest.mark.parametrize('workers', [1, 2])
def test_cli_zero_limit_never_calls_generate_configs(tmp_path, workers):
    generator = tmp_path / 'zero.py'
    generator.write_text("def generate_configs():\n    raise AssertionError('must not generate')\n")
    output = tmp_path / 'zero.jsonl'
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'nexus.cli.main',
            'run',
            'batch',
            '--generator',
            str(generator),
            '--limit',
            '0',
            '--worker-processes',
            str(workers),
            '--database',
            str(tmp_path / 'zero.db'),
            '--export-format',
            'jsonl',
            '--export-path',
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with sqlite3.connect(tmp_path / 'zero.db') as conn:
        assert conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 0
    if output.exists():
        assert output.read_text() == ''
