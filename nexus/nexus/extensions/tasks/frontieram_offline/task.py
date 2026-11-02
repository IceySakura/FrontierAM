"""Nexus Task lifecycle for the frozen exponential-action pilot."""

from __future__ import annotations

import asyncio
import json

from pathlib import Path
from typing import Any, cast

from pydantic import ConfigDict, Field

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, require_context, task_registry

from .agent import OfflineCodexOutput
from .grading import grade
from .runtime import OfflineDockerRuntime


class OfflineTaskInput(TaskRunInput, WithAgentRunner):
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)
    output_dir: str
    user_prompt: str = (
        'Read /workspace/task/PILOT.md, INSTRUCTION.md and API.md. Implement the numerical task using only '
        'the offline workspace tool. Put complete source, run.sh and README in /workspace/submission. '
        'A final turn freezes this one submission. Hidden evaluation is unavailable during development.'
    )
    evaluate: bool = True
    verification_only: bool = True
    grading_timeout: float = Field(default=900, gt=0, le=28800)


class OfflineTaskOutput(TaskRunOutput):
    status: str
    exit_status: str
    score: float | None = None
    score_status: str | None = None
    verification_only: bool
    result_dir: str
    agent_status: str | None = None
    reason: str | None = None
    valid_runs: int | None = None
    total_runs: int | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


@task_registry.register('frontieram_exponential_offline')
async def exponential_offline_task(value: OfflineTaskInput) -> OfflineTaskOutput:
    output_dir = Path(value.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    ctx = require_context()
    provider = ctx.runtime_provider
    if provider is None or type(provider).__name__ != 'FrontierAMOfflineProvider':
        raise ValueError('This task requires the FrontierAM networkless Docker RuntimeProvider')
    provider.configure('main', output_dir=str(output_dir / 'runtime'), stage_public=True)
    runtime = None
    provenance = {}
    cleanup_error = None
    try:
        runtime = cast(OfflineDockerRuntime, await ctx.get_runtime())
        provenance = runtime.assets.provenance()
        provenance['nexus_components'] = [
            'runner.single',
            'Task/Agent registries',
            'RuntimeProvider',
            'OpenAIClient',
            'ResponsesProxy',
            'token_counter',
        ]
        inspection = await runtime.inspect()
        provenance['container_id'] = inspection['Id']
        provenance['isolation'] = runtime.configuration_checks
        runtime.assets.verify_private_data()
        agent_output = cast(
            OfflineCodexOutput,
            await value.run_agent(user_prompt=value.user_prompt, run_dir=str(output_dir / 'agent')),
        )
        freeze_error = None
        invalid_submission = False
        try:
            await runtime.freeze(output_dir, label='source')
        except Exception as error:
            freeze_error = type(error).__name__
            invalid_submission = isinstance(error, runtime.assets.controller.SubmissionFailure)
        if not agent_output.evaluation_eligible:
            result = OfflineTaskOutput(
                status='unmeasured',
                exit_status=agent_output.exit_status,
                verification_only=value.verification_only,
                result_dir=str(output_dir),
                agent_status=agent_output.stop_reason,
                reason='agent_or_inference_interrupted',
                provenance=provenance,
            )
        elif invalid_submission:
            result = OfflineTaskOutput(
                status='invalid_submission',
                exit_status='finished',
                score=0.0,
                verification_only=value.verification_only,
                result_dir=str(output_dir),
                agent_status=agent_output.stop_reason,
                reason=freeze_error,
                provenance=provenance,
            )
        elif freeze_error:
            result = OfflineTaskOutput(
                status='unmeasured',
                exit_status='runtime_error_by_env',
                verification_only=value.verification_only,
                result_dir=str(output_dir),
                agent_status=agent_output.stop_reason,
                reason=freeze_error,
                provenance=provenance,
            )
        elif not value.evaluate:
            result = OfflineTaskOutput(
                status='verification_completed',
                exit_status='finished',
                verification_only=value.verification_only,
                result_dir=str(output_dir),
                agent_status=agent_output.stop_reason,
                provenance=provenance,
            )
        else:
            metrics = await grade(runtime, output_dir, timeout=value.grading_timeout)
            result = OfflineTaskOutput(
                status='success' if metrics['exit_status'] == 'finished' else 'unmeasured',
                exit_status=metrics['exit_status'],
                score=metrics['score'],
                score_status=metrics.get('score_status'),
                verification_only=value.verification_only,
                result_dir=str(output_dir),
                agent_status=agent_output.stop_reason,
                reason=metrics.get('reason'),
                valid_runs=metrics.get('valid_runs'),
                total_runs=metrics.get('total_runs'),
                provenance=provenance,
            )
    except asyncio.CancelledError:
        (output_dir / 'task_interrupted.json').write_text(json.dumps({'reason': 'runner_cancelled'}) + '\n')
        raise
    except Exception as error:
        result = OfflineTaskOutput(
            status='unmeasured',
            exit_status='runtime_error_by_env',
            verification_only=value.verification_only,
            result_dir=str(output_dir),
            reason=type(error).__name__,
            provenance=provenance,
        )
    finally:
        try:
            await provider.release('main', force=True)
        except Exception as error:
            cleanup_error = type(error).__name__
            (output_dir / 'cleanup_error.json').write_text(json.dumps({'error_category': cleanup_error}) + '\n')
    if cleanup_error:
        result.provenance['result_before_cleanup_failure'] = {'exit_status': result.exit_status, 'score': result.score}
        result.status, result.exit_status, result.score = 'unmeasured', 'runtime_error_by_env', None
        result.reason = 'cleanup_failure:' + cleanup_error
    (output_dir / 'task_result.json').write_text(result.model_dump_json(indent=2) + '\n')
    return result
