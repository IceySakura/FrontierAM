"""Integrated rubric task: generate rubrics + evaluate patch in one task.

Supports flexible inputs:
- patch_text directly, or extracted from swe_output dict
- rubrics_yaml directly (skip generation), or generate via agent
- gen/eval use separate openai_clients
"""

import logging

from typing import Any, cast

import yaml

from pydantic import Field, ValidationError

from nexus.core import (
    Registry,
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    agent_registry,
    require_context,
    task_registry,
)
from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.dataset_handler import get_handler
from nexus.extensions.tasks.swe.infer import run_pre_agent_script
from nexus.extensions.tasks.swe.rubrics.agentic_patch.eval_task import score_patch_against_rubrics
from nexus.extensions.tasks.swe.rubrics.shared.schema import RubricSchema
from nexus.extensions.tasks.swe.task import load_instance_from_dataset, load_instance_from_server


logger = logging.getLogger(__name__)


def _get_gold_patch(instance: SWEbenchInstance) -> str:
    return instance.get('patch') or instance.get('gold_patch') or instance.get('fix_patch') or ''


class SweRubricInput(TaskRunInput, WithAgentRunner):
    """Input for the integrated rubric task (gen + eval).

    Uses WithAgentRunner so the runner injects openai_clients into ExecutionContext.
    The run_agent callable is not used directly — agent is called explicitly via registry.
    """

    # Instance
    dataset_path: str = ''
    remote_dataset: str = ''
    split: str = ''
    instance_id: str
    instance: dict | None = None
    problem_statement: str | None = None

    # Patch source
    patch_text: str = Field(default='', description='Patch to evaluate (direct)')
    swe_output: dict | None = Field(default=None, description='SwebenchTaskOutput dict to extract patch_text from')

    # Runtime config (for generation phase)
    image_prefix: str = 'mirrors.tencent.com/hunyuan_yanguan'
    image_override: str | None = None
    pre_agent_script: list[str] | None = None
    rubrics_filename: str = 'rubrics.yaml'
    infer_runtime_name: str = 'main'

    # Skip generation if rubrics already available
    rubrics_yaml: str = Field(default='', description='Pre-existing rubrics YAML (skip generation)')

    # Client config
    gen_client_name: str = Field(default='main', description='openai_client for rubric generation')
    eval_client_name: str = Field(default='rubric_eval', description='openai_client for rubric evaluation')

    # Agent config (explicit, no WithAgentRunner)
    rubric_agent_name: str = Field(default='swe_agent_rubrics', description='Agent registry name for generation')
    rubric_agent_config: dict = Field(default_factory=dict, description='Extra agent config')


class SweRubricOutput(TaskRunOutput, WithScore):
    """Output from integrated rubric task."""

    rubrics_yaml: str = ''
    rubrics: dict | None = None  # Parsed rubric schema as dict (metadata + axes)
    gen_completed: bool = False
    item_scores: dict[str, float] = Field(default_factory=dict)
    weighted_sum: float = 0.0
    max_possible_score: float = 0.0
    # score inherited from WithScore = weighted_sum / max_possible_score
    patch_text: str = ''
    raw_response: str = ''  # Raw LLM eval response for debugging
    error: str | None = None


@task_registry.register('swe_rubric_task')
async def run_swe_rubric_task(task_input: SweRubricInput) -> SweRubricOutput:
    """Integrated rubric task: generate rubrics (if needed) + evaluate patch.

    Flow:
    1. Resolve patch_text from input or swe_output
    2. Generate rubrics (if rubrics_yaml not provided) via agent in runtime
    3. Evaluate patch against rubrics via LLM judge
    4. Return scores
    """
    ctx = require_context()

    # --- Resolve patch ---
    patch_text = task_input.patch_text
    if not patch_text and task_input.swe_output:
        patch_text = task_input.swe_output.get('patch_text', '')

    if not patch_text:
        return SweRubricOutput(score=0.0, error='missing patch_text: provide patch_text or swe_output with patch_text')

    if not task_input.instance_id:
        return SweRubricOutput(score=0.0, patch_text=patch_text, error='missing instance_id')

    # --- Load instance ---
    instance = task_input.instance
    if instance is None:
        try:
            if task_input.dataset_path:
                instance = await load_instance_from_dataset(task_input.dataset_path, task_input.instance_id)
            elif task_input.remote_dataset:
                instance = await load_instance_from_server(
                    task_input.remote_dataset, task_input.instance_id, task_input.split or None
                )
            else:
                return SweRubricOutput(
                    score=0.0,
                    patch_text=patch_text,
                    error='Instance required: provide instance, dataset_path, or remote_dataset',
                )
        except Exception as e:
            return SweRubricOutput(score=0.0, patch_text=patch_text, error=f'Failed to load instance: {e}')

    instance = cast(SWEbenchInstance, instance)
    dataset_handler = get_handler(instance.get('dataset_type'))
    problem_statement = task_input.problem_statement or dataset_handler.get_problem_statement(instance)

    # --- Phase 1: Generate rubrics (if needed) ---
    rubrics_yaml = task_input.rubrics_yaml
    gen_completed = bool(rubrics_yaml)

    if not rubrics_yaml:
        gen_result = await _generate_rubrics(ctx, task_input, instance, dataset_handler, problem_statement)
        if gen_result.get('error') and not gen_result.get('rubrics_yaml'):
            return SweRubricOutput(
                score=0.0,
                patch_text=patch_text,
                error=f'Rubric generation failed: {gen_result["error"]}',
            )
        rubrics_yaml = gen_result.get('rubrics_yaml', '')
        gen_completed = bool(rubrics_yaml)

    if not rubrics_yaml:
        return SweRubricOutput(score=0.0, patch_text=patch_text, error='No rubrics generated or provided')

    # Validate rubrics and parse to dict
    rubrics_dict = None
    try:
        rubric_schema = RubricSchema.from_yaml(rubrics_yaml)
        rubrics_dict = rubric_schema.model_dump()
    except (yaml.YAMLError, ValidationError, ValueError) as e:
        return SweRubricOutput(
            score=0.0,
            rubrics_yaml=rubrics_yaml,
            gen_completed=gen_completed,
            patch_text=patch_text,
            error=f'invalid rubrics_yaml: {e}',
        )

    # --- Phase 2: Evaluate ---
    try:
        scores = await score_patch_against_rubrics(
            rubrics_yaml=rubrics_yaml,
            patch_text=patch_text,
            problem_statement=problem_statement or '',
            eval_client_name=task_input.eval_client_name,
        )
    except Exception as e:
        logger.exception(f'[rubric_task] Evaluation failed: {e}')
        return SweRubricOutput(
            score=0.0,
            rubrics_yaml=rubrics_yaml,
            gen_completed=gen_completed,
            patch_text=patch_text,
            error=f'evaluation failed: {e}',
        )

    return SweRubricOutput(
        score=scores.score,
        rubrics_yaml=rubrics_yaml,
        rubrics=rubrics_dict,
        gen_completed=gen_completed,
        item_scores=scores.item_scores,
        weighted_sum=scores.weighted_sum,
        max_possible_score=scores.max_possible_score,
        patch_text=patch_text,
        raw_response=scores.raw_response,
    )


async def _generate_rubrics(
    ctx: Any,
    task_input: SweRubricInput,
    instance: SWEbenchInstance,
    dataset_handler: Any,
    problem_statement: str,
) -> dict[str, str | None]:
    """Run rubric generation agent in runtime. Returns {'rubrics_yaml': str, 'error': str|None}."""

    if not ctx.runtime_provider:
        return {'rubrics_yaml': '', 'error': 'Rubric generation requires a runtime provider'}

    instance_workdir = instance.get('workdir', '/testbed')
    instance_image = instance.get('image') or instance.get('image_name', '')

    # Configure runtime
    infer_runtime_name = task_input.infer_runtime_name
    configure_params: dict[str, Any] = {'image': instance_image}
    if task_input.image_override:
        configure_params['image'] = task_input.image_override

    ctx.runtime_provider.configure(infer_runtime_name, **configure_params)
    infer_runtime = await ctx.get_runtime(infer_runtime_name)

    try:
        await dataset_handler.infer_runtime_init(instance, infer_runtime)
    except Exception as e:
        return {'rubrics_yaml': '', 'error': f'infer_runtime_init failed: {e}'}

    # Pre-agent script
    if task_input.pre_agent_script:
        formatted_commands = [cmd.format(**instance) for cmd in task_input.pre_agent_script]
        try:
            await run_pre_agent_script(infer_runtime, formatted_commands, working_dir=instance_workdir)
        except Exception as e:
            return {'rubrics_yaml': '', 'error': f'Pre-agent script failed: {e}'}

    # Set task_config for agent
    gold_patch = _get_gold_patch(instance)
    ctx.task_config['problem_statement'] = problem_statement
    ctx.task_config['working_dir'] = instance_workdir
    ctx.task_config['gold_patch'] = gold_patch
    ctx.task_config['gen_client_name'] = task_input.gen_client_name
    ctx.tool_state.setdefault('working_dir', instance_workdir)

    # Get and call agent directly from registry
    Registry.import_module('nexus.extensions.tasks.swe.swe_agent_rubrics')
    agent_fn = agent_registry.get(task_input.rubric_agent_name)

    # Build agent input
    from nexus.extensions.tasks.swe.swe_agent_rubrics.agent import SWEAgentRubricsInput

    agent_config: dict[str, Any] = {'user_prompt': problem_statement, **task_input.rubric_agent_config}
    agent_input = SWEAgentRubricsInput(**agent_config)

    try:
        agent_output = await agent_fn(agent_input)
        logger.info(f'[rubric_task] Agent finished: {getattr(agent_output, "exit_status", "unknown")}')
    except Exception as e:
        await ctx.runtime_provider.release(name=infer_runtime_name)
        return {'rubrics_yaml': '', 'error': f'Agent failed: {e}'}

    # Extract rubrics.yaml
    rubrics_path = f'{instance_workdir}/{task_input.rubrics_filename}'
    try:
        result = await infer_runtime.call_executor(
            'run_command',
            {'command': f'cat {rubrics_path}', 'cwd': instance_workdir},
        )
        rubrics_yaml = result.get('stdout', '') or ''
    except Exception as e:
        rubrics_yaml = ''
        logger.warning(f'[rubric_task] Failed to read rubrics.yaml: {e}')

    await ctx.runtime_provider.release(name=infer_runtime_name)

    error = None if rubrics_yaml else f'rubrics.yaml empty or not found at {rubrics_path}'
    return {'rubrics_yaml': rubrics_yaml, 'error': error}
