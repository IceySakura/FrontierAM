import base64

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.agents.installed_common import InstalledAgentOutput
from nexus.extensions.tasks.mm_agent import task as task_module
from nexus.extensions.tasks.mm_agent import utils as utils_module
from nexus.runtime.executors.models.file import FileOperationResult, UploadFileResponse


IMAGE_TARGET = '/workspace/inputs/chart.png'
PDF_TARGET = '/workspace/inputs/report.pdf'


class CodexLikeOutput(InstalledAgentOutput):
    """Codex reports termination_reason on top of the shared installed-agent fields."""

    termination_reason: str | None = None


@pytest.fixture
def runtime_context(monkeypatch):
    async def upload(files, overwrite):
        return UploadFileResponse(results=[FileOperationResult(path=item.path, success=True) for item in files])

    runtime = SimpleNamespace(upload_file=AsyncMock(side_effect=upload))
    context = SimpleNamespace(get_runtime=AsyncMock(return_value=runtime), agent_name='codex')
    monkeypatch.setattr(task_module, 'require_context', lambda: context)
    monkeypatch.setattr(utils_module, 'require_context', lambda: context)
    monkeypatch.setattr(task_module, '_emit_task_span_attrs', lambda **kwargs: None)
    monkeypatch.setattr(task_module, 'vl_reward_system', AsyncMock(return_value=(0.75, {})))
    return context, runtime


def task_input(tmp_path, target, question, **kwargs):
    source = tmp_path / target.rsplit('/', 1)[-1]
    source.write_bytes(b'file bytes')
    return task_module.SearchTaskInput(
        question=question,
        answer='42',
        artifacts=[{'source': str(source), 'target': target}],
        run_agent=AsyncMock(
            return_value=InstalledAgentOutput(success=True, output='42', stderr='', exit_code=0, execution_time=1.0)
        ),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_artifact_is_staged_at_its_declared_target(tmp_path, runtime_context):
    _, runtime = runtime_context

    result = await task_module.mm_agent_task(task_input(tmp_path, PDF_TARGET, f'Summarize {PDF_TARGET}'))

    item = runtime.upload_file.call_args.args[0][0]
    assert item.path == PDF_TARGET
    assert base64.b64decode(item.content) == b'file bytes'
    assert result.score == 0.75


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('agent_name', 'target', 'question', 'image_mode', 'expected_kwargs'),
    [
        # Only Codex images become native attachments; everything else is reached by path.
        ('codex', IMAGE_TARGET, 'Which page?', 'attach', {'user_prompt': 'Which page?', 'images': [IMAGE_TARGET]}),
        ('codex', PDF_TARGET, f'Read {PDF_TARGET}', 'attach', {'user_prompt': f'Read {PDF_TARGET}'}),
        ('claude_code', IMAGE_TARGET, f'Read {IMAGE_TARGET}', 'attach', {'user_prompt': f'Read {IMAGE_TARGET}'}),
        ('hycode', IMAGE_TARGET, f'Read {IMAGE_TARGET}', 'attach', {'user_prompt': f'Read {IMAGE_TARGET}'}),
        ('codex', IMAGE_TARGET, f'Read {IMAGE_TARGET}', 'path', {'user_prompt': f'Read {IMAGE_TARGET}'}),
    ],
)
async def test_harness_receives_the_bare_question_and_only_codex_image_attachments(
    tmp_path, runtime_context, agent_name, target, question, image_mode, expected_kwargs
):
    context, _ = runtime_context
    context.agent_name = agent_name
    task = task_input(tmp_path, target, question, codex_image_inject_mode=image_mode)

    await task_module.mm_agent_task(task)

    assert task.run_agent.await_args.kwargs == expected_kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize('agent_name', ['gemini_cli'])
async def test_harness_without_a_gateway_policy_is_rejected(tmp_path, runtime_context, agent_name):
    context, _ = runtime_context
    context.agent_name = agent_name
    task = task_input(tmp_path, PDF_TARGET, f'Read {PDF_TARGET}')

    with pytest.raises(ValueError, match='unsupported nexus.agent_name'):
        await task_module.mm_agent_task(task)
    task.run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_image_artifacts_reach_the_reward_platform(tmp_path, runtime_context):
    task = task_input(tmp_path, IMAGE_TARGET, 'Which page?', reward_system_extra={'scoring_mode': 'auto'})

    await task_module.mm_agent_task(task)

    kwargs = task_module.vl_reward_system.await_args.kwargs
    assert kwargs['images'][0].startswith('data:image/png;base64,')
    assert kwargs['reward_extra'] == {'scoring_mode': 'auto'}


@pytest.mark.asyncio
async def test_judge_output_is_exported_as_reward_extra_info(tmp_path, runtime_context):
    task_module.vl_reward_system.return_value = (
        1.0,
        {
            'raw_response': '{"是否一致": 1}',
            'analysis': 'matches',
            'platform_request_id': 'req-1',
            'question': 'Which page?',
            'model_response': '42',
        },
    )

    result = await task_module.mm_agent_task(task_input(tmp_path, PDF_TARGET, f'Read {PDF_TARGET}'))

    assert result.extra_fields['reward_extra_info'] == {'rm_response': '{"是否一致": 1}', 'rm_request_id': 'req-1'}


@pytest.mark.asyncio
async def test_failed_judge_call_keeps_error_text_out_of_rm_response(tmp_path, runtime_context):
    task_module.vl_reward_system.return_value = (
        -100.0,
        {'error': 'reward platform request failed: boom', 'platform_request_id': 'req-2'},
    )

    result = await task_module.mm_agent_task(task_input(tmp_path, PDF_TARGET, f'Read {PDF_TARGET}'))

    assert result.extra_fields['reward_extra_info'] == {'rm_response': '', 'rm_request_id': 'req-2'}


@pytest.mark.asyncio
async def test_unjudged_run_exports_empty_reward_extra_info(tmp_path, runtime_context):
    task = task_input(tmp_path, PDF_TARGET, f'Read {PDF_TARGET}')
    task.run_agent = AsyncMock(
        return_value=CodexLikeOutput(success=False, output='', stderr='', exit_code=1, execution_time=1.0)
    )

    result = await task_module.mm_agent_task(task)

    assert result.extra_fields['reward_extra_info'] == {'rm_response': '', 'rm_request_id': ''}


@pytest.mark.asyncio
async def test_strip_multimodal_content_keeps_the_judge_text_only(tmp_path, runtime_context):
    task = task_input(tmp_path, IMAGE_TARGET, 'Which page?', reward_system_strip_multimodal_content=True)

    await task_module.mm_agent_task(task)

    assert task_module.vl_reward_system.await_args.kwargs['images'] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('failed_fields', 'expected_score', 'expected_exit_status'),
    [
        ({'termination_reason': 'TRUNCATION'}, 0.0, 'TRUNCATION'),
        (
            {'termination_reason': 'TASK_TIMEOUT', 'failure_kind': 'agent'},
            task_module._TASK_FAILED_SCORE,
            'UNHANDLED_ERROR',
        ),
        (
            {'termination_reason': 'CODEX_TURN_FAILED', 'failure_kind': 'agent'},
            task_module._TASK_FAILED_SCORE,
            'UNHANDLED_ERROR',
        ),
        ({'failure_kind': 'agent'}, task_module._TASK_FAILED_SCORE, 'UNHANDLED_ERROR'),
        ({'failure_kind': 'infrastructure', 'failure_stage': 'install'}, task_module._TASK_FAILED_SCORE, 'ENV_ERROR'),
    ],
)
async def test_unfinished_runs_skip_the_judge(
    tmp_path, runtime_context, failed_fields, expected_score, expected_exit_status
):
    task = task_input(tmp_path, PDF_TARGET, f'Read {PDF_TARGET}')
    task.run_agent = AsyncMock(
        return_value=CodexLikeOutput(
            success=False, output='half an ans', stderr='', exit_code=1, execution_time=1.0, **failed_fields
        )
    )

    result = await task_module.mm_agent_task(task)

    assert (result.score, result.exit_status) == (expected_score, expected_exit_status)
    task_module.vl_reward_system.assert_not_awaited()


@pytest.mark.asyncio
async def test_harness_reported_scoreable_exit_is_judged(tmp_path, runtime_context):
    context, _ = runtime_context
    context.agent_name = 'hycode'
    task = task_input(tmp_path, PDF_TARGET, f'Read {PDF_TARGET}')
    task.run_agent = AsyncMock(
        return_value=CodexLikeOutput(
            success=True, output='42', stderr='', exit_code=-5, execution_time=1.0, termination_reason='ENV_DONE'
        )
    )

    result = await task_module.mm_agent_task(task)

    assert (result.score, result.exit_status) == (0.75, 'ENV_DONE')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('failure', 'match'),
    [
        ('missing_source', 'No such file'),
        ('duplicate_target', 'duplicate targets'),
        ('failed_upload', 'Failed to upload'),
        ('unreferenced', 'not mentioned in the question'),
        ('inline_images', 'take task_config.artifacts'),
    ],
)
async def test_bad_input_never_runs_the_harness(tmp_path, runtime_context, failure, match):
    context, runtime = runtime_context
    task = task_input(tmp_path, PDF_TARGET, f'Read {PDF_TARGET}')
    if failure == 'missing_source':
        (tmp_path / 'report.pdf').unlink()
    elif failure == 'duplicate_target':
        task.artifacts = [*task.artifacts, task.artifacts[0]]
    elif failure == 'failed_upload':
        runtime.upload_file.side_effect = lambda files, overwrite: UploadFileResponse(
            results=[FileOperationResult(path=files[0].path, success=False)]
        )
    elif failure == 'unreferenced':
        task.question = 'Read the document'
    else:
        task.artifacts, task.images = None, ['/host/a.png']

    with pytest.raises((RuntimeError, ValueError, OSError), match=match):
        await task_module.mm_agent_task(task)
    task.run_agent.assert_not_awaited()
