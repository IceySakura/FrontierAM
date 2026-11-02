from __future__ import annotations

import pytest

from nexus.core import WorkflowState
from nexus.runner.batch import runner as batch_runner
from nexus.runner.batch.batch_config import BatchConfig, BatchItem
from nexus.runner.single import ErrorInfo, RunInput, RunOutput


@pytest.mark.asyncio
async def test_batch_retry_restarts_from_original_input_by_default(monkeypatch) -> None:
    inputs: list[RunInput] = []
    state = WorkflowState(workflow='main', version='1', cursor='work')

    async def fake_run_single(run_input: RunInput) -> RunOutput:
        inputs.append(run_input)
        if len(inputs) == 1:
            return RunOutput(
                success=False,
                error=ErrorInfo(type='RuntimeError', message='retry me'),
                workflow_states={'task/main': state},
            )
        return RunOutput(success=True, workflow_states=run_input.workflow_states)

    monkeypatch.setattr(batch_runner, 'run_single', fake_run_single)

    summary = await batch_runner.run_batch(
        generator_path=None,
        db_path=':memory:',
        batch_config=BatchConfig(
            retry_failed=1,
            enable_trace_db=False,
            enable_log_capture=False,
        ),
        batch_items=[BatchItem(config=RunInput(task_name='unused'))],
    )

    assert summary.success == 1
    assert [run_input.workflow_states for run_input in inputs] == [{}, {}]
    assert inputs[0] is not inputs[1]
    assert inputs[0].workflow_states is not inputs[1].workflow_states


@pytest.mark.asyncio
async def test_batch_retry_can_resume_workflow_from_previous_attempt(monkeypatch) -> None:
    calls: list[dict[str, WorkflowState]] = []
    state = WorkflowState(workflow='main', version='1', cursor='work')

    async def fake_run_single(run_input: RunInput) -> RunOutput:
        calls.append(dict(run_input.workflow_states))
        if len(calls) == 1:
            return RunOutput(
                success=False,
                error=ErrorInfo(type='RuntimeError', message='retry me'),
                workflow_states={'task/main': state},
            )
        return RunOutput(success=True, workflow_states=run_input.workflow_states)

    monkeypatch.setattr(batch_runner, 'run_single', fake_run_single)

    summary = await batch_runner.run_batch(
        generator_path=None,
        db_path=':memory:',
        batch_config=BatchConfig(
            retry_failed=1,
            resume_workflow_on_retry=True,
            enable_trace_db=False,
            enable_log_capture=False,
        ),
        batch_items=[BatchItem(config=RunInput(task_name='unused'))],
    )

    assert summary.success == 1
    assert calls == [{}, {'task/main': state}]
