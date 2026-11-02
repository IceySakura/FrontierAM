"""Thin Nexus task entrypoint for Harbor-native Trial execution."""

from __future__ import annotations

import re
import tempfile

from pathlib import Path
from typing import Any

from pydantic import Field

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore, require_context, task_registry
from nexus.dataset.task_files import TaskFiles
from nexus.extensions.harbor_trial.context import HarborTrialBinding, harbor_trial_binding


class HarborTrialTaskInput(TaskRunInput, WithAgentRunner):
    """Portable input for Harbor Trial execution on Nexus Runtimes."""

    task_dir: str | None = None
    instance_id: str = ''
    agent_timeout: float | None = Field(default=None, gt=0)
    verifier_timeout: float | None = Field(default=None, gt=0)
    timeout_multiplier: float = Field(default=1.0, gt=0)
    force_build: bool = False


class HarborTrialTaskOutput(TaskRunOutput, WithScore):
    """Harbor-native result plus a Nexus-compatible scalar score."""

    reward: float = 0.0
    rewards: dict[str, float] = Field(default_factory=dict)
    trial_result: dict[str, Any]
    step_agent_outputs: list[dict[str, Any]] = Field(default_factory=list)


def _safe_task_scope_name(instance_id: str | None) -> str:
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(instance_id or '').strip()).strip('._')
    return value or 'task'


async def materialize_task_directory(
    task_files: TaskFiles | None,
    *,
    task_dir: str | None,
    destination: Path,
) -> Path:
    """Materialize a portable task bundle or validate a local development path."""
    if task_files is not None:
        destination.mkdir(parents=True, exist_ok=False)
        await task_files.download('**/*', destination)
        return destination

    if not task_dir:
        raise ValueError('harbor_trial_task requires task_files or task_dir')
    resolved = Path(task_dir).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f'Harbor task directory not found: {resolved}')
    return resolved


def _reward_values(result: Any) -> dict[str, float]:
    verifier_result = result.verifier_result
    if verifier_result is None:
        return {}
    return {key: float(value) for key, value in (verifier_result.rewards or {}).items()}


def _step_agent_outputs(result: Any) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for step in result.step_results or []:
        metadata = step.agent_result.metadata if step.agent_result is not None else None
        output = metadata.get('nexus_agent_output') if metadata else None
        if isinstance(output, dict):
            outputs.append({'step_name': step.step_name, 'output': output})
    if result.agent_result is not None and result.agent_result.metadata:
        output = result.agent_result.metadata.get('nexus_agent_output')
        if isinstance(output, dict):
            outputs.append({'step_name': None, 'output': output})
    return outputs


def _raise_for_trial_failure(result: Any) -> None:
    if result.exception_info is None:
        return
    error = result.exception_info
    raise RuntimeError(f'Harbor Trial failed with {error.exception_type}: {error.exception_message}')


@task_registry.register('harbor_trial_task')
async def harbor_trial_task(task_input: HarborTrialTaskInput) -> HarborTrialTaskOutput:
    """Run a Harbor Trial while Nexus supplies the Agent and single Runtime."""
    from harbor.models.task.config import TaskConfig as HarborTaskConfig
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig, VerifierConfig
    from harbor.trial.trial import Trial

    from nexus.extensions.harbor_trial.network import validate_separate_verifier_networks

    context = require_context()
    if context.runtime_provider is None:
        raise RuntimeError('harbor_trial_task requires a Nexus runtime provider')

    with tempfile.TemporaryDirectory(prefix='nexus_harbor_task_') as task_scope:
        task_path = await materialize_task_directory(
            context.task_files,
            task_dir=task_input.task_dir,
            destination=Path(task_scope) / _safe_task_scope_name(task_input.instance_id),
        )

        with tempfile.TemporaryDirectory(prefix='nexus_harbor_trial_') as trials_dir:
            agent_info = getattr(task_input.run_agent, 'nexus_meta', {})
            agent_class = 'NexusCodexAgent' if agent_info.get('name') == 'codex' else 'NexusAgent'
            trial_config = TrialConfig(
                task=TaskConfig(path=task_path),
                trial_name=f'{_safe_task_scope_name(task_input.instance_id or task_path.name)}__nexus',
                trials_dir=Path(trials_dir),
                timeout_multiplier=task_input.timeout_multiplier,
                agent=AgentConfig(
                    import_path=f'nexus.extensions.harbor_trial.agent:{agent_class}',
                    override_timeout_sec=task_input.agent_timeout,
                    resume_trajectory=False,
                ),
                environment=EnvironmentConfig(
                    import_path='nexus.extensions.harbor_trial.environment:NexusEnvironment',
                    force_build=task_input.force_build,
                ),
                verifier=VerifierConfig(override_timeout_sec=task_input.verifier_timeout),
            )
            validate_separate_verifier_networks(
                HarborTaskConfig.model_validate_toml((task_path / 'task.toml').read_text()),
                trial_config,
                context.runtime_provider,
            )
            binding = HarborTrialBinding(
                execution_context=context,
                run_agent=task_input.run_agent,
            )
            async with harbor_trial_binding(binding):
                trial = await Trial.create(trial_config)
                result = await trial.run()

            _raise_for_trial_failure(result)
            rewards = _reward_values(result)
            reward = rewards.get('reward', 0.0)
            result_payload = result.model_dump(mode='json')
            result_payload.pop('trial_uri', None)
            return HarborTrialTaskOutput(
                score=reward,
                reward=reward,
                rewards=rewards,
                trial_result=result_payload,
                step_agent_outputs=_step_agent_outputs(result),
            )


__all__ = [
    'HarborTrialTaskInput',
    'HarborTrialTaskOutput',
    'harbor_trial_task',
    'materialize_task_directory',
]
