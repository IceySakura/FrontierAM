"""Independent V2 task schema, registration and artifact lifecycle."""

import asyncio
import logging

from typing import Any, Literal

from pydantic import Field, SecretStr

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore, require_context, task_registry
from nexus.extensions.tasks.os_world_v2.adapter import claude_multiturn as _claude_multiturn  # noqa: F401
from nexus.extensions.tasks.os_world_v2.adapter import codex as _codex  # noqa: F401
from nexus.extensions.tasks.os_world_v2.adapter.adapter import Options, run
from nexus.extensions.tasks.os_world_v2.adapter.outputs import save_agent_evidence, summarize
from nexus.extensions.tasks.os_world_v2.outputs import (
    CONTRACT_ACTIVE_STATE,
    OSWorldJudgeOutput,
    classify_exception,
    project_task_output,
    wrap_inner_timeout,
)


class OSWorldV2TaskInput(TaskRunInput, WithAgentRunner):
    task_id: str
    benchmark_release: Literal['osworld-v2.1'] = 'osworld-v2.1'
    v21: Options
    runtime_name: str = 'main'
    domain: str = 'tasks'
    run_setup: bool = True
    run_evaluator: bool = True
    skip_gui_start: bool = True
    command_as_user: bool = False
    client_password: str = 'osworld-public-evaluation'
    action_space: Literal['pyautogui', 'computer_13'] = 'pyautogui'
    include_initial_screenshot: bool = True
    setup_wait_time: float = Field(default=5, ge=0)
    step_pause: float = Field(default=1, ge=0)
    artifact_dataset_uri: str = ''
    artifact_taihu_pat: SecretStr | None = Field(default=None, exclude=True, repr=False)
    artifact_service_url: str | None = None
    bundle_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    wedata_key: str | dict[str, Any] | None = Field(default=None, repr=False)


class OSWorldV2TaskOutput(TaskRunOutput, WithScore):
    status: Literal['success', 'agent_error', 'judge_error'] = 'success'
    completed: bool
    task_id: str
    domain: str = 'tasks'
    instruction: str
    evaluator_result: dict[str, Any] | None = None
    agent_output: Any = None
    artifacts: dict[str, Any] | None = None
    judge_output: OSWorldJudgeOutput | None = None
    error: str | None = None
    # A failed agent/judge stage has no actual score (standalone failure output).
    score: float | None = None  # type: ignore[assignment]


@task_registry.register('os_world_v2')
async def run_os_world_v2(task_input: OSWorldV2TaskInput) -> OSWorldV2TaskOutput:
    from nexus.extensions.tasks.os_world_v2.adapter.artifacts import RunArtifacts

    ctx = require_context()
    for key in ('osworld_partial_task_output', 'osworld_agent_output', 'osworld_judge_output'):
        ctx.tool_state.pop(key, None)
    ctx.tool_state['osworld_output_stage'] = 'prepare'
    artifacts = RunArtifacts(
        taihu_pat=task_input.artifact_taihu_pat.get_secret_value() if task_input.artifact_taihu_pat else None,
        service_url=task_input.artifact_service_url,
    )
    from nexus.extensions.tasks.os_world_v2.execution.release import install_release

    release_records = []
    ctx.tool_state['osworld_release_records'] = release_records
    artifacts.reference['release'] = release_records
    install_release(ctx.runtime_provider, task_input.runtime_name, release_records, artifacts.root)
    if task_input.artifact_dataset_uri:
        logging.getLogger(__name__).warning(
            'artifact_dataset_uri is deprecated; V2 now publishes through the shared OSWorld artifact service'
        )
    previous = ctx.tool_state.get('osworld_artifact_directory')
    previous_publisher = ctx.tool_state.get('osworld_artifact_publisher')
    ctx.tool_state['osworld_artifact_directory'] = str(artifacts.files)
    ctx.tool_state['osworld_artifact_publisher'] = artifacts
    standalone = not ctx.tool_state.get(CONTRACT_ACTIVE_STATE)
    result, primary = None, None
    try:
        result = await run(task_input)
    except BaseException as exc:
        primary = wrap_inner_timeout(
            exc,
            stage=ctx.tool_state.get('osworld_output_stage', 'prepare'),
            timeout_handle=getattr(ctx, 'timeout_handle', None),
        )
        if not standalone or not isinstance(primary, Exception):
            if primary is exc:
                raise
            raise primary from exc
        logging.getLogger(__name__).warning(
            'V2 task failed (%s: %s); reporting the classified failure as task output',
            type(primary).__name__,
            primary,
        )
    finally:
        try:
            if result is None:
                save_agent_evidence(ctx, artifacts.files)
            artifacts.finish(task_id=task_input.task_id, run_id=ctx.run_id, result=result, error=primary)
            publication = asyncio.create_task(artifacts.publish())
            try:
                reference = await asyncio.shield(publication)
            except asyncio.CancelledError:
                await publication
                raise
            if result is not None:
                result.artifacts = reference
                result.artifacts['release'] = release_records
        except asyncio.CancelledError:
            if primary is None:
                raise
        except Exception as exc:
            artifacts.reference.update(status='upload_failed', upload_failed=True)
            if result is not None:
                result.artifacts = artifacts.reference
            logging.getLogger(__name__).warning(
                'V2 artifact publication failed (%s); preserving task outcome', type(exc).__name__
            )
        finally:
            ctx.tool_state['osworld_output_publisher'] = artifacts
            if result is None:
                stage = ctx.tool_state.get('osworld_output_stage', 'prepare')
                agent = ctx.tool_state.get('osworld_agent_output')
                judge = ctx.tool_state.get('osworld_judge_output')
                detail = f'{type(primary).__name__}: {primary}' if primary else 'Task did not return a result'
                if isinstance(primary, Exception) and (agent is None or (stage != 'judge' and agent.exit_code == 0)):
                    agent = classify_exception(primary, stage=stage)
                if stage == 'judge' and judge is None:
                    judge = OSWorldJudgeOutput(exit_status='judge_error', exit_code=-1, error=detail)
                ctx.tool_state['osworld_partial_task_output'] = summarize({
                    'status': 'judge_error' if stage == 'judge' else 'agent_error',
                    'completed': False,
                    'task_id': task_input.task_id,
                    'domain': task_input.domain,
                    'instruction': '',
                    'score': None,
                    'agent_output': agent.model_dump(mode='json') if agent is not None else None,
                    'judge_output': judge.model_dump(mode='json') if judge is not None else None,
                    'error': detail,
                    'artifacts': artifacts.reference,
                })
            for key in ('osworld_output_stage', 'osworld_agent_output', 'osworld_judge_output'):
                ctx.tool_state.pop(key, None)
            if previous is None:
                ctx.tool_state.pop('osworld_artifact_directory', None)
            else:
                ctx.tool_state['osworld_artifact_directory'] = previous
            if previous_publisher is None:
                ctx.tool_state.pop('osworld_artifact_publisher', None)
            else:
                ctx.tool_state['osworld_artifact_publisher'] = previous_publisher
    # Without the output-contract hook nothing projects RunOutput afterwards, so the
    # task itself returns the bounded, classified result (also on failure).
    if result is not None and (not standalone or not isinstance(result, OSWorldV2TaskOutput)):
        return result
    value = result if result is not None else ctx.tool_state.pop('osworld_partial_task_output')
    projected = project_task_output(value, summarize=summarize)
    projected.setdefault('instruction', '')
    return OSWorldV2TaskOutput.model_validate(projected)
