"""Unit coverage for run CLI commands."""

from __future__ import annotations

import argparse

import pytest

from nexus.cli.commands.run import register_run_batch_parser, register_run_parser, run_batch
from nexus.runner.batch import BatchSummary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    register_run_batch_parser(subparsers)
    register_run_parser(subparsers)
    return parser


@pytest.mark.parametrize(
    ('command', 'enabled'),
    [
        (['run-batch', '--database', 'results.db'], False),
        (['run-batch', '--database', 'results.db', '--enable-conflict-groups'], True),
        (['run', 'batch', '--database', 'results.db'], False),
        (['run', 'batch', '--database', 'results.db', '--enable-conflict-groups'], True),
    ],
)
def test_conflict_groups_argument(command, enabled):
    args = _parser().parse_args(command)

    assert args.enable_conflict_groups is enabled


@pytest.mark.parametrize('command', ['run-batch', 'run'])
def test_batch_accepts_compose_config_path(command):
    argv = [command, 'batch', 'compose.yaml'] if command == 'run' else [command, 'compose.yaml']

    args = _parser().parse_args(argv)

    assert args.config == 'compose.yaml'


def test_run_batch_passes_conflict_groups_to_batch_config(tmp_path, monkeypatch):
    captured = {}

    async def fake_run_batch(**kwargs):
        captured.update(kwargs)
        return BatchSummary(total=0)

    monkeypatch.setattr('nexus.runner.batch.run_batch', fake_run_batch)
    monkeypatch.setattr('nexus.cli.commands.run.setup_logging', lambda **_kwargs: None)
    args = _parser().parse_args(['run-batch', '--database', str(tmp_path / 'results.db'), '--enable-conflict-groups'])

    assert run_batch(args) == 0
    assert captured['batch_config'].enable_conflict_groups is True


@pytest.mark.parametrize(
    'command',
    [
        ['run-batch'],
        ['run', 'batch'],
    ],
)
@pytest.mark.parametrize(
    ('omitted', 'trace_enabled', 'log_enabled'),
    [
        ([], True, True),
        (['trace'], False, True),
        (['log'], True, False),
        (['trace', 'log'], False, False),
    ],
)
def test_run_batch_passes_no_persist_to_batch_config(
    command, omitted, trace_enabled, log_enabled, tmp_path, monkeypatch
):
    captured = {}

    async def fake_run_batch(**kwargs):
        captured.update(kwargs)
        return BatchSummary(total=0)

    monkeypatch.setattr('nexus.runner.batch.run_batch', fake_run_batch)
    monkeypatch.setattr('nexus.cli.commands.run.setup_logging', lambda **_kwargs: None)
    argv = [*command, '--database', str(tmp_path / 'results.db')]
    for store in omitted:
        argv.extend(['--no-persist', store])
    args = _parser().parse_args(argv)

    assert run_batch(args) == 0
    assert captured['batch_config'].enable_trace_db is trace_enabled
    assert captured['batch_config'].enable_log_capture is log_enabled


def test_run_batch_compiles_compose_config_file(tmp_path, monkeypatch):
    captured = {}
    compose_path = tmp_path / 'batch.yaml'
    compose_path.write_text(
        """
compose:
  - input:
      task_name: identity
      task_config: {}
""".lstrip(),
        encoding='utf-8',
    )

    async def fake_run_batch(**kwargs):
        captured.update(kwargs)
        return BatchSummary(total=1, success=1)

    monkeypatch.setattr('nexus.runner.batch.run_batch', fake_run_batch)
    monkeypatch.setattr('nexus.cli.commands.run.setup_logging', lambda **_kwargs: None)
    args = _parser().parse_args(['run', 'batch', str(compose_path)])

    assert run_batch(args) == 0
    assert len(captured['batch_items']) == 1
    assert captured['batch_items'][0].config.task_name == 'identity'


def test_conflict_groups_rejects_multiple_worker_processes(tmp_path, monkeypatch, caplog):
    async def unexpected_run_batch(**_kwargs):
        raise AssertionError('batch runner should not be called')

    monkeypatch.setattr('nexus.runner.batch.run_batch', unexpected_run_batch)
    monkeypatch.setattr('nexus.cli.commands.run.setup_logging', lambda **_kwargs: None)
    args = _parser().parse_args([
        'run-batch',
        '--database',
        str(tmp_path / 'results.db'),
        '--enable-conflict-groups',
        '--worker-processes',
        '2',
    ])

    assert run_batch(args) == 1
    assert '--enable-conflict-groups currently requires --worker-processes 1' in caplog.text


@pytest.mark.parametrize(('option', 'value'), [('--data', 'other'), ('--use', 'data:other')])
def test_generator_rejects_other_compose_sources_before_loading(tmp_path, option, value, caplog, monkeypatch):
    monkeypatch.setattr('nexus.cli.commands.run.setup_logging', lambda **_kwargs: None)
    generator = tmp_path / 'generator.py'
    generator.write_text("raise AssertionError('must not import')\n")
    argv = ['run', 'batch', '--generator', str(generator), option, value]
    args = _parser().parse_args(argv)
    args._raw_argv = argv
    assert run_batch(args) == 1
    assert '--generator cannot be combined with' in caplog.text


@pytest.mark.parametrize('override', [['--set', 'missing-equals'], ['--set-json', 'task_config={bad}']])
def test_generator_rejects_malformed_overrides_before_loading(tmp_path, override, caplog, monkeypatch):
    monkeypatch.setattr('nexus.cli.commands.run.setup_logging', lambda **_kwargs: None)
    generator = tmp_path / 'generator.py'
    generator.write_text("raise AssertionError('must not import')\n")
    argv = ['run', 'batch', '--generator', str(generator), *override]
    args = _parser().parse_args(argv)
    args._raw_argv = argv
    assert run_batch(args) == 1
    assert 'Invalid generator override' in caplog.text


def test_generator_rejects_negative_limit_before_loading(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr('nexus.cli.commands.run.setup_logging', lambda **_kwargs: None)
    generator = tmp_path / 'generator.py'
    generator.write_text("raise AssertionError('must not import')\n")
    argv = ['run', 'batch', '--generator', str(generator), '--limit', '-1']
    args = _parser().parse_args(argv)
    args._raw_argv = argv
    assert run_batch(args) == 1
    assert '--limit must be non-negative' in caplog.text


@pytest.mark.parametrize('part', ['missing_run_part_for_test', 'run_timeout:seconds="unterminated'])
def test_generator_reports_invalid_use_before_loading(tmp_path, monkeypatch, caplog, part):
    monkeypatch.setattr('nexus.cli.commands.run.setup_logging', lambda **_kwargs: None)
    generator = tmp_path / 'generator.py'
    generator.write_text("raise AssertionError('must not import')\n")
    argv = ['run', 'batch', '--generator', str(generator), '--use', part]
    args = _parser().parse_args(argv)
    args._raw_argv = argv
    assert run_batch(args) == 1
    assert 'Invalid generator override' in caplog.text
