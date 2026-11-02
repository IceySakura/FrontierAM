"""Task-owned inputs, workspace setup, and actual runner/visual-agent composition."""

from __future__ import annotations

import copy

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pydantic import ValidationError

from nexus.core import (
    ExecutionContext,
)
from nexus.extensions.tasks.visual.task import VisualTaskInput, visual_task
from nexus.runner.single import RunInput, run


async def test_visual_task_passes_text_to_codex_input(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.extensions.agents.codex import agent as codex_module

    captured = []

    async def runtime_stub(target, /, **kwargs):
        captured.append(kwargs)
        return {
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 0.1,
            'timed_out': False,
        }

    monkeypatch.setattr(codex_module, 'call_runtime_agent', runtime_stub)
    result = await run(
        RunInput(
            task_name='visual',
            task_config={'user_prompt': 'question from visual task'},
            agent_name='codex',
            agent_config={'working_dir': '/workspace'},
        )
    )
    assert result.success, result.error
    assert result.agent_output.exit_status == 'completed'
    assert captured[0]['task'] == 'question from visual task'
    assert captured[0]['working_dir'] == '/workspace'


@pytest.mark.parametrize('structured', [False, True])
async def test_prompt_and_history_pass_through_without_runtime(
    structured: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = (
        [
            {'type': 'text', 'text': 'Analyze'},
            {'type': 'file', 'file': {'url': '/host/input.pdf', 'filename': 'report'}},
            {'type': 'video_url', 'video_url': {'url': '/host/input.mp4'}},
        ]
        if structured
        else 'Analyze'
    )
    history = [{'role': 'user', 'content': 'earlier'}, {'role': 'assistant', 'content': 'ready'}]
    captured = []

    async def run_agent(**kwargs):
        assert set(kwargs) == {'user_prompt', 'history', 'upload_workspace'}
        data = kwargs
        captured.append(copy.deepcopy(data))
        if isinstance(data['user_prompt'], list):
            data['user_prompt'][0]['text'] = 'changed by agent'
        data['history'][0]['content'] = 'changed by agent'

    task_input = VisualTaskInput(user_prompt=prompt, history=history, run_agent=run_agent)
    async with ExecutionContext():
        output = await visual_task(task_input)

    assert output.model_dump() == {'score': None, 'evaluation_details': None}
    assert captured == [{'user_prompt': prompt, 'history': history, 'upload_workspace': None}]
    assert task_input.user_prompt == prompt
    assert task_input.history == history


@pytest.mark.parametrize('structured', [False, True])
async def test_task_passes_source_without_reading_paths(tmp_path: Path, structured: bool) -> None:
    source = str(tmp_path / 'not-on-task-host.zip')
    prompt = [{'type': 'text', 'text': 'Analyze'}] if structured else 'Analyze'
    run_agent = AsyncMock()
    task_input = VisualTaskInput(user_prompt=prompt, upload_workspace=source, run_agent=run_agent)
    output = await visual_task(task_input)
    run_agent.assert_awaited_once_with(user_prompt=prompt, history=None, upload_workspace=source)
    assert not Path(source).exists()
    assert output.model_dump() == {'score': None, 'evaluation_details': None}


def test_task_input_does_not_accept_agent_workspace() -> None:
    with pytest.raises(ValidationError, match='Extra inputs are not permitted'):
        VisualTaskInput.model_validate({'user_prompt': 'hi', 'workspace': '/task/workspace', 'run_agent': AsyncMock()})


def test_task_rejects_agent_policy_fields_and_invalid_history() -> None:
    with pytest.raises(ValidationError, match='Extra inputs are not permitted'):
        VisualTaskInput.model_validate({'user_prompt': 'hi', 'tools': [], 'run_agent': AsyncMock()})
    with pytest.raises(ValidationError, match='history must end'):
        VisualTaskInput(user_prompt='hi', history=[{'role': 'user', 'content': 'hi'}], run_agent=AsyncMock())


def test_task_rejects_old_mapping_list() -> None:
    with pytest.raises(ValidationError, match='valid string'):
        VisualTaskInput.model_validate({
            'user_prompt': 'hi',
            'upload_workspace': [{'source': '/src', 'target': '/dst'}],
            'run_agent': AsyncMock(),
        })


def test_task_text_is_literal_but_workspace_paths_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_TASK_INPUT_PATH', '/host/input')
    text = 'literal ${oc.env:NEXUS_MISSING_HISTORY_VAR} and ${template}'
    config = {
        'task_name': 'visual',
        'task_config': {
            'user_prompt': [{'type': 'text', 'text': text}],
            'history': [{'role': 'assistant', 'content': 'Previous answer'}],
            'upload_workspace': '${oc.env:NEXUS_TASK_INPUT_PATH}',
        },
    }
    result = RunInput.from_dict(config)
    assert result.task_config['user_prompt'] == config['task_config']['user_prompt']
    assert result.task_config['history'] == config['task_config']['history']
    assert result.task_config['upload_workspace'] == '/host/input'
