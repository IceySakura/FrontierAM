"""Rubric generation task: run an agent to produce rubrics.yaml for a SWE-bench instance.

Requires a runtime (Docker sandbox) for the agent to explore the codebase.
Uses WithAgentRunner for standard nexus agent injection.
"""

import logging

from typing import Any, cast

import yaml

from pydantic import Field, ValidationError

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    require_context,
    task_registry,
)
from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.dataset_handler import get_handler
from nexus.extensions.tasks.swe.infer import run_pre_agent_script
from nexus.extensions.tasks.swe.rubrics.shared.schema import RubricSchema
from nexus.extensions.tasks.swe.task import load_instance_from_dataset, load_instance_from_server


logger = logging.getLogger(__name__)


def _get_gold_patch(instance: SWEbenchInstance) -> str:
    """Extract gold patch from instance using common field names."""
    return instance.get('patch') or instance.get('gold_patch') or instance.get('fix_patch') or ''


class SweRubricGenInput(TaskRunInput, WithAgentRunner):
    """Input for rubric generation task."""

    dataset_path: str = ''
    remote_dataset: str = ''
    split: str = ''
    instance_id: str
    instance: dict | None = None
    problem_statement: str | None = None
    image_prefix: str = 'mirrors.tencent.com/hunyuan_yanguan'
    image_override: str | None = None
    pre_agent_script: list[str] | None = None
    rubrics_filename: str = Field(default='rubrics.yaml', description='Name of the rubrics file to generate')
    gen_client_name: str = Field(default='main', description='Which openai_client to use for generation')
    infer_runtime_name: str = 'main'


class SweRubricGenOutput(TaskRunOutput):
    """Output from rubric generation task."""

    completed: bool = False
    rubrics_yaml: str = ''
    rubrics: dict | None = None  # Parsed and validated rubric schema as dict
    error: str | None = None


@task_registry.register('swe_rubric_gen_task')
async def run_swe_rubric_gen_task(task_input: SweRubricGenInput) -> SweRubricGenOutput:
    """Generate evaluation rubrics for a SWE-bench instance.

    Flow: load instance -> configure runtime -> run agent -> extract rubrics.yaml -> validate
    """
    ctx = require_context()

    # --- Input validation ---
    if not task_input.instance_id:
        return SweRubricGenOutput(error='missing instance_id')

    if not ctx.runtime_provider:
        return SweRubricGenOutput(error='Rubric generation requires a runtime provider')

    # --- Load instance ---
    instance = task_input.instance
    if instance is None:
        if task_input.dataset_path:
            instance = await load_instance_from_dataset(task_input.dataset_path, task_input.instance_id)
        elif task_input.remote_dataset:
            instance = await load_instance_from_server(
                task_input.remote_dataset, task_input.instance_id, task_input.split or None
            )
        else:
            return SweRubricGenOutput(error='Instance is required: provide instance, dataset_path, or remote_dataset')

    instance = cast(SWEbenchInstance, instance)
    dataset_handler = get_handler(instance.get('dataset_type'))
    instance_workdir = instance.get('workdir', '/testbed')
    instance_image = instance.get('image') or instance.get('image_name', '')

    logger.debug(f'[rubric_gen] instance_workdir: {instance_workdir}')

    # --- Configure runtime ---
    infer_runtime_name = task_input.infer_runtime_name
    configure_params: dict[str, Any] = {'image': instance_image}
    if task_input.image_override:
        configure_params['image'] = task_input.image_override

    ctx.runtime_provider.configure(infer_runtime_name, **configure_params)
    infer_runtime = await ctx.get_runtime(infer_runtime_name)

    # Dataset-specific initialization
    try:
        await dataset_handler.infer_runtime_init(instance, infer_runtime)
    except Exception as e:
        return SweRubricGenOutput(error=f'infer_runtime_init failed: {e}')

    # Optional pre-agent script
    if task_input.pre_agent_script:
        formatted_commands = [cmd.format(**instance) for cmd in task_input.pre_agent_script]
        try:
            await run_pre_agent_script(infer_runtime, formatted_commands, working_dir=instance_workdir)
        except Exception as e:
            return SweRubricGenOutput(error=f'Pre-agent script failed: {e}')

    # --- Prepare task_config for agent ---
    problem_statement = task_input.problem_statement or dataset_handler.get_problem_statement(instance)
    if not problem_statement:
        return SweRubricGenOutput(error=f'No problem statement for instance {task_input.instance_id}')

    gold_patch = _get_gold_patch(instance)
    if not gold_patch:
        logger.warning(f'[rubric_gen] No gold patch found for {task_input.instance_id}')

    ctx.task_config['problem_statement'] = problem_statement
    ctx.task_config['working_dir'] = instance_workdir
    ctx.task_config['gold_patch'] = gold_patch
    ctx.task_config['gen_client_name'] = task_input.gen_client_name
    ctx.tool_state.setdefault('working_dir', instance_workdir)

    # --- Run agent ---
    agent_output = await task_input.run_agent(user_prompt=problem_statement)
    logger.info(f'[rubric_gen] Agent finished: {getattr(agent_output, "exit_status", "unknown")}')

    # --- Extract rubrics.yaml ---
    rubrics_path = f'{instance_workdir}/{task_input.rubrics_filename}'
    rubrics_yaml = ''
    error = None

    try:
        result = await infer_runtime.call_executor(
            'run_command',
            {'command': f'cat {rubrics_path}', 'cwd': instance_workdir},
        )
        rubrics_yaml = result.get('stdout', '') or ''
        if not rubrics_yaml:
            error = f'rubrics.yaml empty or not found at {rubrics_path}'
    except Exception as e:
        error = f'Failed to read rubrics.yaml: {e}'

    # --- Validate schema ---
    rubrics_dict = None
    if rubrics_yaml and not error:
        try:
            schema = RubricSchema.from_yaml(rubrics_yaml)
            rubrics_dict = schema.model_dump()
            logger.debug(f'[rubric_gen] Rubrics validated: {len(schema.get_all_items())} items')
        except (yaml.YAMLError, ValidationError, ValueError) as e:
            logger.warning(f'[rubric_gen] Rubrics YAML validation failed: {e}')
            # Still return the raw YAML, just note the validation error
            error = f'Rubrics generated but validation failed: {e}'

    # --- Cleanup ---
    await ctx.runtime_provider.release(name=infer_runtime_name)

    return SweRubricGenOutput(
        completed=bool(rubrics_yaml),
        rubrics_yaml=rubrics_yaml,
        rubrics=rubrics_dict,
        error=error,
    )
