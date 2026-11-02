"""Local E2E: CLI -> generation/selection -> workers -> real task -> DB -> CLI export."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys

from pathlib import Path

import pytest


pytestmark = pytest.mark.e2e
_REPO = Path(__file__).resolve().parents[2]
_TASK_MODULE = 'tests.e2e._generator_run_parts_fixture'


class BatchCLI:
    def __init__(self, root: Path):
        self.root = root
        self.sequence = 0
        self.generator = root / 'generator.py'
        self.generator.write_text(f"""
import json
from pathlib import Path
from nexus import BatchItem, RunInput

ROOT = Path(__file__).parent

def get_batch_metadata():
    return {{'suite': 'generator-parts-e2e'}}

def generate_configs():
    try:
        for index in range(5):
            case = f'case-{{index}}'
            with (ROOT / 'generated.jsonl').open('a') as log:
                log.write(json.dumps(case) + '\\n')
            yield BatchItem(run_id=case,
                config=RunInput(task_name='generator_parts_e2e', imports=[{_TASK_MODULE!r}],
                    task_config={{'case': case, 'state_dir': str(ROOT / 'state'), 'values': [1, 2], 'factor': 2}}),
                metadata={{'suite': 'e2e', 'dataset_item_id': case}})
    finally:
        (ROOT / 'generator.closed').write_text('closed')
""")
        (root / 'common.yaml').write_text('body:\n  task_config:\n    values: [3]\n    factor: 5\n')
        (root / 'complete.json').write_text(
            json.dumps({
                'task_name': 'generator_parts_e2e',
                'task_config': {'factor': 7},
            })
        )
        self.overlays = [
            '--use',
            str(root / 'common.yaml'),
            '--set',
            'task_config.factor=6',
            '--use',
            str(root / 'complete.json'),
            '--set',
            'task_config.values+=4',
            '--set-json',
            'extra={"source":"generator-e2e"}',
        ]

    def call(self, *args: str) -> str:
        self.sequence += 1
        log = self.root / f'command-{self.sequence}.log'
        env = dict(os.environ, PYTHONPATH=str(_REPO), NEXUS_OTEL_ENABLED='false')
        with log.open('w') as stream:
            try:
                completed = subprocess.run(
                    [sys.executable, '-m', 'nexus.cli.main', *args],
                    cwd=self.root,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=60,
                )
            except subprocess.TimeoutExpired:
                pytest.fail(f'CLI timed out after 60s; log: {log}\n{log.read_text()[-5000:]}', pytrace=False)
        assert completed.returncode == 0, f'CLI failed; log: {log}\n{log.read_text()[-5000:]}'
        return log.read_text()

    def run(self, database: str, *, workers: int = 1, limit: int = 3, extra: tuple[str, ...] = ()) -> str:
        return self.call(
            'run',
            'batch',
            '--generator',
            str(self.generator),
            '--database',
            str(self.root / database),
            '--limit',
            str(limit),
            '--worker-processes',
            str(workers),
            '--max-concurrent',
            '2',
            *self.overlays,
            *extra,
        )

    def export(self, database: str) -> list[dict]:
        target = self.root / f'{database}.jsonl'
        self.call('export', str(self.root / database), '--format', 'jsonl', '--output', str(target))
        return [json.loads(line) for line in target.read_text().splitlines()]

    def rows(self, database: str) -> list[dict]:
        path = self.root / database
        with sqlite3.connect(path) as conn:
            info = {key: json.loads(value) for key, value in conn.execute('SELECT key, value FROM batch_info')}
        paths = [path]
        if info.get('sharded_batch'):
            paths = [Path(str(path) + '.shards') / shard['database'] for shard in info['shards']]
        rows = []
        for path in paths:
            with sqlite3.connect(path) as conn:
                conn.row_factory = sqlite3.Row
                for row in conn.execute('SELECT * FROM runs ORDER BY run_id'):
                    item = dict(row)
                    for field in ['config', 'metadata', 'task_output']:
                        item[field] = json.loads(item[field]) if item[field] else None
                    rows.append(item)
        return sorted(rows, key=lambda row: row['run_id'])

    def assert_results(self, database: str, *, count: int, attempt: int = 1) -> list[dict]:
        rows = self.rows(database)
        exported = sorted(self.export(database), key=lambda row: row['run_id'])
        assert len(rows) == len(exported) == count
        for index, (row, record) in enumerate(zip(rows, exported, strict=True)):
            assert row['run_id'] == f'case-{index}'
            assert row['status'] == record['status'] == 'success'
            assert row['metadata'] == record['metadata'] == {'suite': 'e2e', 'dataset_item_id': f'case-{index}'}
            assert row['config']['task_config']['factor'] == 7
            assert row['config']['task_config']['values'] == [3, 4]
            assert record['extra'] == {'source': 'generator-e2e'}
            output = record['task_output']
            assert output['answer'] == 49
            assert output['score'] == 1.0
            assert output['attempt'] == attempt
            assert output['child_pid'] != output['worker_pid']
            state = Path(row['config']['task_config']['state_dir'])
            assert json.loads((state / f'case-{index}.json').read_text()) == output
            assert int((state / f'case-{index}.attempts').read_text()) == attempt
        return rows


@pytest.fixture
def cli(tmp_path):
    return BatchCLI(tmp_path)


@pytest.mark.parametrize('environment_timeout', [False, True])
def test_generator_and_data_produce_equivalent_effective_tasks(cli, monkeypatch, environment_timeout):
    if environment_timeout:
        monkeypatch.setenv('GENERATOR_E2E_TIMEOUT', '30')
        cli.overlays.extend(['--set', 'runner.timeout=${oc.env:GENERATOR_E2E_TIMEOUT}'])
    cli.run('generator.db', limit=2)
    generated = cli.assert_results('generator.db', count=2)
    dataset = cli.root / 'dataset'
    dataset.mkdir()
    files = []
    for index in range(5):
        case = f'case-{index}'
        filename = f'{case}.json'
        files.append(filename)
        (dataset / filename).write_text(
            json.dumps({
                'id': case,
                'body': {
                    'task_name': 'generator_parts_e2e',
                    'imports': [_TASK_MODULE],
                    'task_config': {
                        'case': case,
                        'state_dir': str(cli.root / 'data-state'),
                        'values': [1, 2],
                        'factor': 2,
                    },
                },
                'metadata': {'suite': 'e2e', 'dataset_item_id': case},
            })
        )
    (dataset / 'dataset.yaml').write_text(json.dumps({'default_split': 'train', 'splits': {'train': {'items': files}}}))
    cli.call(
        'run',
        'batch',
        '--data',
        str(dataset),
        '--split',
        'train',
        '--limit',
        '2',
        '--database',
        str(cli.root / 'data.db'),
        *cli.overlays,
    )
    data = cli.assert_results('data.db', count=2)
    for left, right in zip(generated, data, strict=True):
        left['config']['task_config'].pop('state_dir')
        right['config']['task_config'].pop('state_dir')
        assert left['config'] == right['config']
        assert left['config']['runner']['timeout'] == (30.0 if environment_timeout else None)


def test_two_workers_limit_generation_and_export_full_artifacts(cli):
    cli.run('parallel.db', workers=2, limit=3)
    rows = cli.assert_results('parallel.db', count=3)
    assert len({row['task_output']['worker_pid'] for row in rows}) == 2
    assert (cli.root / 'generated.jsonl').read_text().splitlines() == ['"case-0"', '"case-1"', '"case-2"']
    assert (cli.root / 'generator.closed').read_text() == 'closed'
    full = cli.root / 'full'
    cli.call('export', str(cli.root / 'parallel.db'), '--format', 'full', '--output', str(full))
    for row in rows:
        assert json.loads((full / 'runs' / row['run_id'] / 'task_output.json').read_text()) == row['task_output']
        assert (full / 'configs' / f'{row["run_id"]}.yaml').is_file()


def test_real_failure_retries_keep_effective_configuration(cli):
    cli.run('retry.db', limit=2, extra=('--retry-failed', '1', '--set', 'task_config.fail_first=true'))
    rows = cli.assert_results('retry.db', count=2, attempt=2)
    assert all(row['retry_count'] == 1 for row in rows)
    assert all(row['error'] is None for row in rows)
    assert len((cli.root / 'generated.jsonl').read_text().splitlines()) == 2


def test_resume_keeps_saved_tasks_despite_new_limit_and_parts(cli):
    cli.run('resume.db', limit=2)
    before = cli.assert_results('resume.db', count=2)
    (cli.root / 'complete.json').write_text(
        json.dumps({'task_name': 'generator_parts_e2e', 'task_config': {'factor': 999}})
    )
    log = cli.run('resume.db', limit=0, extra=('--no-resume',))
    assert 'ignoring generator options' in log
    after = cli.assert_results('resume.db', count=2, attempt=2)
    assert [r['config'] for r in before] == [r['config'] for r in after]
    assert len((cli.root / 'generated.jsonl').read_text().splitlines()) == 2


@pytest.mark.parametrize('workers', [1, 2])
def test_locked_resume_does_not_read_missing_or_invalid_overlays(cli, workers):
    cli.run('resume.db', workers=workers, limit=2)
    before = cli.assert_results('resume.db', count=2)
    (cli.root / 'common.yaml').unlink()
    (cli.root / 'complete.json').write_text('{invalid json')
    log = cli.run(
        'resume.db',
        workers=workers,
        limit=0,
        extra=(
            '--no-resume',
            '--use',
            'unavailable_review_part',
            '--use',
            'run_timeout:seconds="unterminated',
            '--set',
            'missing-equals',
        ),
    )
    assert 'ignoring generator options' in log
    after = cli.assert_results('resume.db', count=2, attempt=2)
    assert [row['config'] for row in before] == [row['config'] for row in after]
    assert len((cli.root / 'generated.jsonl').read_text().splitlines()) == 2


@pytest.mark.parametrize('workers', [1, 2])
def test_zero_limit_has_no_generation_or_task_side_effects(cli, workers):
    cli.run('zero.db', workers=workers, limit=0)
    assert cli.rows('zero.db') == []
    assert cli.export('zero.db') == []
    assert not (cli.root / 'generated.jsonl').exists()
    assert not (cli.root / 'state').exists()
