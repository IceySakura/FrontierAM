"""VitaBench task for nexus framework.

Runs a VitaBench simulation (agent + user simulator + environment) and evaluates
the trajectory using sliding-window rubric-based LLM evaluation.
"""

import json
import logging

from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os

from pydantic import Field

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.tasks.vita.data_model.tasks import Task
from nexus.extensions.tasks.vita.environment.environment import Environment, get_cross_environment
from nexus.extensions.tasks.vita.vita_utils.utils import get_weekday


logger = logging.getLogger(__name__)


def _build_environment(domain: str, env_data: dict, language: str) -> Environment:
    """Build a VitaBench Environment from domain name and task environment data."""
    if ',' in domain:
        return get_cross_environment(domain, env_data, language)

    from nexus.extensions.tasks.vita.domains import get_env_constructor

    constructor = get_env_constructor(domain)
    return constructor(env_data, language)


async def load_instance_from_dataset(dataset_path: str, instance_id: str) -> dict[str, Any]:
    """Load a single VitaBench task instance from a local JSON/JSONL dataset file.

    For JSONL files, scans line-by-line for a matching ``instance_id``.
    For JSON files, handles both arrays and single-object formats.
    """
    path = Path(dataset_path)
    if not await aiofiles.os.path.exists(path):
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')

    if path.suffix == '.jsonl':
        async with aiofiles.open(path, encoding='utf-8', errors='replace') as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f'Invalid JSONL line in dataset {dataset_path}: {e}') from e
                if data.get('instance_id') == instance_id:
                    return data
    else:
        async with aiofiles.open(path, encoding='utf-8', errors='replace') as f:
            text = await f.read()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid JSON in dataset {dataset_path}: {e}') from e
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get('instance_id') == instance_id:
                    return item
        elif isinstance(data, dict) and data.get('instance_id') == instance_id:
            return data

    raise ValueError(f'Instance {instance_id!r} not found in dataset {dataset_path}')


def _infer_language_from_instance_id(instance_id: str) -> str:
    """Infer language from instance_id format: ``{domain}__{lang}-{task_id}``.

    Returns ``'english'`` for ``en``, ``'chinese'`` for ``zh`` or unknown codes.
    """
    # instance_id pattern: "delivery__zh-10711001" or "ota__en-A0812005"
    if '__' in instance_id:
        after = instance_id.split('__', 1)[1]  # "zh-10711001"
        lang_code = after.split('-', 1)[0]  # "zh"
        if lang_code == 'en':
            return 'english'
    return 'chinese'


class VitaTaskInput(TaskRunInput, WithAgentRunner):
    """Input for VitaBench task evaluation.

    Typical usage: provide ``dataset_path`` and ``instance_id`` to load a task
    from a JSONL/JSON file.  The ``domain``, ``task_id``, and ``language`` are
    automatically inferred from the loaded task data and ``instance_id``.

    These fields can still be supplied explicitly to override the inferred values.
    """

    dataset_path: str | None = Field(
        default=None, description='Path to a JSON or JSONL file containing VitaBench task data'
    )
    instance_id: str | None = Field(
        default=None, description='Task instance ID within the dataset file (matches instance_id field in JSON)'
    )
    task_data: dict | None = Field(default=None, description='Full task JSON, used directly if provided')
    language: str | None = Field(
        default=None, description='Override language (chinese or english). Auto-inferred if omitted'
    )
    evaluation_type: str = Field(default='trajectory', description='Evaluation type: trajectory')


class VitaTaskOutput(TaskRunOutput, WithScore):
    """Output from VitaBench task evaluation."""

    termination_reason: str = ''
    rubric_score: float = 0.0
    reward_details: dict[str, Any] | None = None
    simulation_summary: dict[str, Any] | None = None


@task_registry.register('vita_task')
async def run_vita_task(task_input: VitaTaskInput) -> VitaTaskOutput:
    """Run a VitaBench simulation and evaluate the trajectory."""
    ctx = require_context()

    # --- Load task data ---
    task_data = task_input.task_data
    if not task_data:
        if task_input.dataset_path and task_input.instance_id:
            task_data = await load_instance_from_dataset(task_input.dataset_path, task_input.instance_id)
            logger.debug(
                'Loaded task from dataset: %s (instance_id=%s)', task_input.dataset_path, task_input.instance_id
            )
        else:
            raise ValueError('Either task_data or both dataset_path and instance_id must be provided')

    # --- Derive domain, task_id, language from task_data / instance_id ---
    domain: str = task_data.get('domain', '')
    task_id: str = task_data.get('id', '') or task_data.get('instance_id', '')

    if task_input.language:
        language = task_input.language
    elif task_input.instance_id:
        language = _infer_language_from_instance_id(task_input.instance_id)
    else:
        language = 'chinese'

    # --- Populate context ---
    ctx.task_config['domain'] = domain
    ctx.task_config['task'] = task_data
    ctx.task_config['language'] = language

    task = Task.model_validate(task_data)
    environment = _build_environment(domain, task.environment, language)

    env_time = environment.tools.db.time if environment.tools and environment.tools.db else None
    weekday = get_weekday(env_time, language) if env_time else ''
    time_str = f'{env_time} {weekday}'.strip() if env_time else ''

    agent_policy = environment.get_policy()
    system_prompt = agent_policy.format(time=time_str) if '{time}' in agent_policy else agent_policy

    ctx.tool_state['vita_env'] = environment
    ctx.task_config['system_prompt'] = system_prompt
    ctx.task_config['env_time'] = env_time

    logger.debug('VitaBench task started: domain=%s, task_id=%s', domain, task_id)

    agent_output = await task_input.run_agent(
        user_prompt=task_data.get('instructions', ''),
    )

    simulation = getattr(agent_output, 'simulation', None)
    termination_reason = getattr(agent_output, 'termination_reason', '')

    from nexus.extensions.tasks.vita.data_model.simulation import SimulationRun
    from nexus.extensions.tasks.vita.evaluator import evaluate_trajectory

    eval_client_name = 'evaluator' if 'evaluator' in (ctx.openai_clients or {}) else 'main'
    eval_client = ctx.get_openai_client(eval_client_name)

    sim_messages = []
    if simulation and 'messages' in simulation:
        sim_run = SimulationRun.model_validate(simulation)
        sim_messages = sim_run.messages or []

    reward_info = await evaluate_trajectory(
        openai_client=eval_client,
        simulation_messages=sim_messages,
        task=task,
        termination_reason=termination_reason,
        evaluation_type=task_input.evaluation_type,
        language=language,
    )

    score = reward_info.reward
    rubric_score = 0.0
    if reward_info.reward_breakdown:
        rubric_score = next(iter(reward_info.reward_breakdown.values()), 0.0)

    reward_details = reward_info.info or {}
    if reward_info.nl_rubrics:
        reward_details['rubrics'] = [
            {'rubric': r.nl_rubric, 'met': r.met, 'justification': r.justification} for r in reward_info.nl_rubrics
        ]

    simulation_summary = None
    if simulation:
        simulation_summary = {
            'task_id': simulation.get('task_id', task_id),
            'termination_reason': termination_reason,
            'duration': simulation.get('duration'),
            'agent_cost': simulation.get('agent_cost'),
            'user_cost': simulation.get('user_cost'),
            'num_messages': len(simulation.get('messages', [])),
        }

    logger.info(
        'VitaBench task completed: task_id=%s, score=%.2f, rubric_score=%.2f, termination=%s',
        task_id,
        score,
        rubric_score,
        termination_reason,
    )

    return VitaTaskOutput(
        score=score,
        termination_reason=termination_reason,
        rubric_score=rubric_score,
        reward_details=reward_details,
        simulation_summary=simulation_summary,
    )
