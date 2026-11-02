from __future__ import annotations

import sys

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import pytest  # noqa: E402

from nexus.extensions.tasks.mm_agent import utils as mm_agent_utils  # noqa: E402
from nexus.extensions.tasks.mm_agent.utils import TaskArtifact  # noqa: E402
from nexus.extensions.tasks.ppt_rubric import task as ppt_task  # noqa: E402
from nexus.extensions.tasks.ppt_rubric.task import PPTTaskInput  # noqa: E402
from nexus.runtime.executors.models.file import FileOperationResult, UploadFileResponse  # noqa: E402


class _ScoringNeverRuns(RuntimeError):
    pass


@pytest.fixture
def runtime_context(monkeypatch):
    async def upload(files, overwrite):
        return UploadFileResponse(results=[FileOperationResult(path=item.path, success=True) for item in files])

    runtime = SimpleNamespace(
        upload_file=AsyncMock(side_effect=upload),
        run_command=AsyncMock(return_value=SimpleNamespace(stdout='', exit_code=1)),
        invoke=AsyncMock(side_effect=_ScoringNeverRuns('scoring is out of scope for these tests')),
        default_venv_id='venv0ab12',
    )
    context = SimpleNamespace(get_runtime=AsyncMock(return_value=runtime), agent_name='codex', tool_state={})
    monkeypatch.setattr(ppt_task, 'require_context', lambda: context)
    monkeypatch.setattr(mm_agent_utils, 'require_context', lambda: context)
    return context, runtime


def _row(tmp_path, *, artifacts=None, agent_tools=None, **kwargs):
    from PIL import Image

    rows: list[TaskArtifact] = []
    for artifact in artifacts or []:
        source = tmp_path / Path(artifact['target']).name
        if source.suffix == '.png' and not source.exists():
            Image.new('RGB', (32, 18), '#4488cc').save(source)
        elif not source.exists():
            source.write_bytes(b'file bytes')
        rows.append(TaskArtifact(source=str(source), target=artifact['target']))
    fields: dict = {
        'question': 'Reproduce the slide.',
        'agent_tools': agent_tools or [],
        'run_agent': AsyncMock(),
    }
    fields.update(kwargs)
    return PPTTaskInput(artifacts=rows, **fields)


def _run_agent_kwargs(task) -> dict:
    return task.run_agent.await_args.kwargs


@pytest.mark.asyncio
async def test_codex_row_with_ppt12_mounts_the_mcp_server(tmp_path, runtime_context):
    task = _row(
        tmp_path,
        artifacts=[{'target': '/workspace/inputs/slide.png'}],
        agent_tools=['ppt12'],
    )

    await ppt_task.run_ppt_topic(task)

    servers = _run_agent_kwargs(task)['mcp_servers']
    assert len(servers) == 1
    assert servers[0]['name'] == 'ppt12'
    assert servers[0]['command'] == '${NEXUS_WORKER_PYTHON}'
    assert servers[0]['args'] == ['-m', 'nexus.extensions.tasks.ppt_rubric.ppt_mcp_server']
    assert servers[0]['env']['PPT12_REFERENCE_PNG'] == '/workspace/inputs/slide.png'
    assert servers[0]['env']['PPT12_WORKSPACE'] == '/workspace'
    assert servers[0]['required'] is True
    # The image artifact still travels as a native codex attachment too.
    assert _run_agent_kwargs(task)['images'] == ['/workspace/inputs/slide.png']


@pytest.mark.asyncio
async def test_codex_row_without_agent_tools_stays_bare(tmp_path, runtime_context):
    task = _row(tmp_path)

    await ppt_task.run_ppt_topic(task)

    assert 'mcp_servers' not in _run_agent_kwargs(task)
    assert 'tools' not in _run_agent_kwargs(task)


@pytest.mark.asyncio
async def test_codex_row_without_a_worker_venv_fails_before_the_harness_runs(tmp_path, runtime_context, monkeypatch):
    _, runtime = runtime_context
    monkeypatch.setattr(runtime, 'default_venv_id', None)
    task = _row(tmp_path, agent_tools=['ppt12'])

    with pytest.raises(ValueError, match='worker_venv'):
        await ppt_task.run_ppt_topic(task)
    task.run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_codex_installed_harness_with_ppt12_raises_before_the_harness_runs(tmp_path, runtime_context):
    """Review decision: the ppt12 switch is codex-only. A non-codex installed
    harness carrying it used to warn and run on with no tools, quietly putting
    a tool-less arm into the scores; it now fails like every other unsupported
    combination in this task (installed harness with images, unknown agent)."""
    context, _ = runtime_context
    context.agent_name = 'claude_code'
    task = _row(tmp_path, agent_tools=['ppt12'])

    with pytest.raises(ValueError, match='ppt12'):
        await ppt_task.run_ppt_topic(task)
    task.run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_builtin_loop_expansion_is_unchanged_by_the_switch(tmp_path, runtime_context):
    context, _ = runtime_context
    context.agent_name = 'main_agent'
    # The loop's real image channel is artifacts; a ppt12 row without an image
    # artifact now fails loud (review decision) instead of seeding nothing.
    task = _row(
        tmp_path,
        artifacts=[{'target': '/workspace/inputs/slide.png'}],
        agent_tools=['ppt12'],
    )

    await ppt_task.run_ppt_topic(task)

    kwargs = _run_agent_kwargs(task)
    from nexus.extensions.tasks.ppt_rubric.ppt_tools import PPT12_TOOLS

    assert kwargs['tools'] == list(PPT12_TOOLS)
    assert 'mcp_servers' not in kwargs


@pytest.mark.asyncio
async def test_builtin_loop_ppt12_without_an_image_artifact_raises(tmp_path, runtime_context):
    """Review decision: seeding nothing meant all 12 tools would answer
    "no reference image" while the rollout scored normally -- fail instead."""
    context, _ = runtime_context
    context.agent_name = 'main_agent'
    task = _row(tmp_path, agent_tools=['ppt12'])

    with pytest.raises(ValueError, match='no image artifact'):
        await ppt_task.run_ppt_topic(task)
    task.run_agent.assert_not_awaited()
