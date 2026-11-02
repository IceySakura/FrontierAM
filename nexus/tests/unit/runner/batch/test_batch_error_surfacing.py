from __future__ import annotations

import io
import json

from pathlib import Path

import pytest

from pydantic import BaseModel

from nexus.cli.commands.export import export_run_to_full, write_run_to_jsonl
from nexus.runner.batch import runner as batch_runner
from nexus.runner.batch.batch_config import BatchConfig
from nexus.runner.batch.database import BatchDatabase
from nexus.runner.single import ErrorInfo, RunInput, RunOutput


def _write_generator(path: Path) -> None:
    path.write_text(
        """
from nexus.runner.batch.batch_config import BatchItem
from nexus.runner.single import RunInput


def generate_configs():
    yield BatchItem(
        config=RunInput(task_name='noop', task_config={}),
        run_id='run-0',
        metadata={},
    )
"""
    )


@pytest.mark.asyncio
async def test_exception_escaping_run_single_is_recorded_as_a_failed_run(tmp_path, monkeypatch):
    """Execution gathers with ``return_exceptions``, so an escape must be caught per run.

    Otherwise the run stays at the 'running' status written before execution, carries no
    error, and is counted as neither success nor failure -- the batch just reports fewer
    results than it had tasks.
    """
    generator_path = tmp_path / 'generator.py'
    db_path = tmp_path / 'batch.db'
    _write_generator(generator_path)

    async def exploding_run_single(config):
        raise ImportError("Module 'nexus.extensions.missing' not found")

    monkeypatch.setattr(batch_runner, 'run_single', exploding_run_single)

    summary = await batch_runner.run_batch(
        generator_path=generator_path,
        db_path=db_path,
        batch_config=BatchConfig(
            max_concurrent=1,
            worker_processes=1,
            retry_failed=0,
            enable_trace_db=False,
            enable_log_capture=False,
        ),
    )

    assert summary.total == 1
    assert summary.success == 0
    assert summary.failed == 1

    async with BatchDatabase(db_path) as db:
        run_record = await db.get_run('run-0')

    assert run_record['status'] == 'failed'
    assert run_record['end_time']
    assert "Module 'nexus.extensions.missing' not found" in run_record['error']
    assert 'Traceback' in run_record['error']


@pytest.mark.asyncio
async def test_escaping_exception_still_propagates_under_fail_fast(tmp_path, monkeypatch):
    generator_path = tmp_path / 'generator.py'
    db_path = tmp_path / 'batch.db'
    _write_generator(generator_path)

    async def exploding_run_single(config):
        raise ImportError('boom')

    monkeypatch.setattr(batch_runner, 'run_single', exploding_run_single)

    with pytest.raises(ImportError, match='boom'):
        await batch_runner.run_batch(
            generator_path=generator_path,
            db_path=db_path,
            batch_config=BatchConfig(
                max_concurrent=1,
                worker_processes=1,
                retry_failed=0,
                fail_fast=True,
                enable_trace_db=False,
                enable_log_capture=False,
            ),
        )

    async with BatchDatabase(db_path) as db:
        run_record = await db.get_run('run-0')

    assert run_record['status'] == 'failed'
    assert 'boom' in run_record['error']


class TaskResult(BaseModel):
    score: float = 0.0
    status: str = 'success'


class AgentResult(BaseModel):
    exit_status: str = 'max_iterations'
    iterations: int = 1


class JudgeResult(BaseModel):
    exit_status: str = 'finished'
    exit_code: int = 0
    error: str | None = None


@pytest.mark.asyncio
@pytest.mark.parametrize('dictionary_outputs', [False, True])
async def test_batch_persists_model_and_dictionary_outputs(tmp_path, monkeypatch, dictionary_outputs):
    generator_path = tmp_path / 'generator.py'
    db_path = tmp_path / 'batch.db'
    _write_generator(generator_path)
    task = TaskResult()
    agent = AgentResult()
    judge = JudgeResult()

    async def completed_run_single(config):
        return RunOutput(
            task_output=task.model_dump() if dictionary_outputs else task,
            agent_output=agent.model_dump() if dictionary_outputs else agent,
            judge_output=judge.model_dump() if dictionary_outputs else judge,
        )

    monkeypatch.setattr(batch_runner, 'run_single', completed_run_single)
    summary = await batch_runner.run_batch(
        generator_path=generator_path,
        db_path=db_path,
        batch_config=BatchConfig(enable_trace_db=False, enable_log_capture=False),
    )

    assert summary.success == 1
    async with BatchDatabase(db_path) as db:
        record = await db.get_run('run-0')
    assert record['status'] == 'success'
    assert record['task_output'] == task.model_dump()
    assert record['agent_output'] == agent.model_dump()
    assert record['judge_output'] == judge.model_dump()
    stream = io.StringIO()
    write_run_to_jsonl(record, stream)
    assert json.loads(stream.getvalue())['judge_output'] == judge.model_dump()
    export_run_to_full(record, tmp_path / 'export')
    assert json.loads((tmp_path / 'export' / 'runs' / 'run-0' / 'judge_output.json').read_text()) == judge.model_dump()


@pytest.mark.asyncio
async def test_failed_batch_run_keeps_partial_task_and_agent_outputs(tmp_path, monkeypatch):
    generator_path = tmp_path / 'generator.py'
    db_path = tmp_path / 'batch.db'
    _write_generator(generator_path)
    task = {'status': 'judge_error', 'completed': False, 'patch_text': 'partial patch'}
    agent = AgentResult()
    judge = JudgeResult(exit_status='judge_error', exit_code=-1, error='judge interrupted')

    async def failed_run_single(config):
        return RunOutput(
            success=False,
            error=ErrorInfo(type='RuntimeError', message='judge interrupted'),
            task_output=task,
            agent_output=agent,
            judge_output=judge,
        )

    monkeypatch.setattr(batch_runner, 'run_single', failed_run_single)
    summary = await batch_runner.run_batch(
        generator_path=generator_path,
        db_path=db_path,
        batch_config=BatchConfig(enable_trace_db=False, enable_log_capture=False),
    )

    assert summary.failed == 1
    async with BatchDatabase(db_path) as db:
        record = await db.get_run('run-0')
    assert record['status'] == 'failed'
    assert 'judge interrupted' in record['error']
    assert record['task_output'] == task
    assert record['agent_output'] == agent.model_dump()
    assert record['judge_output'] == judge.model_dump()


@pytest.mark.asyncio
async def test_real_batch_retries_and_persists_standard_initialization_failure(tmp_path):
    config = RunInput(
        imports=['missing_platform_initialization_fixture_module'],
        task_name='superswe_task',
        agent_name='swe_agent',
        agent_config={'platform_exit_status': True},
        runner={'hooks': {'superswe_platform': {'name': 'superswe_platform'}}},
    )
    generator_path = tmp_path / 'generator.py'
    generator_path.write_text(
        'from nexus.runner.single import RunInput\n'
        'from nexus.runner.batch.batch_config import BatchItem\n'
        'def generate_configs():\n'
        f'    yield BatchItem(config=RunInput.model_validate({config.model_dump(mode="json")!r}), run_id="setup")\n'
    )
    db_path = tmp_path / 'batch.db'
    summary = await batch_runner.run_batch(
        generator_path=generator_path,
        db_path=db_path,
        batch_config=BatchConfig(retry_failed=1, enable_trace_db=False, enable_log_capture=False),
    )
    assert summary.success == 0
    assert summary.failed == 1
    async with BatchDatabase(db_path) as db:
        record = await db.get_run('setup')
    assert record['status'] == 'failed'
    assert record['retry_count'] == 1
    assert record['agent_output']['exit_status'] == 'environment_error'
    assert record['agent_output']['exit_code'] == -1
    assert record['task_output']['status'] == 'agent_error'
    assert record['task_output']['error'] == record['agent_output']['error']
    assert record['judge_output'] is None
