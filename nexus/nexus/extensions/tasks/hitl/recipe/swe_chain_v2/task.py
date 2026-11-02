"""The chained SWE task: one instance, its requirements handed over one at a time.

Around the agent it does what any SWE HITL task does -- put the repo at the base
commit, extract the patch, evaluate it. What is specific here is the chain: an
instance carries several problem statements and several patches, so the statements
are preprocessed and handed to the agent as a list rather than combined into one
prompt, and the reference patches are concatenated for evaluation.

Reads no files. The dataset record and the rubrics both arrive on the task input,
from a generator that has already read the dataset once for the whole batch.
"""

import asyncio
import contextlib
import logging
import os

from typing import Any, Literal

from pydantic import Field

from nexus.core import (
    ExecutionContext,
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.tasks.hitl.agents.agent_with_user import AgentWithUserOutput
from nexus.extensions.tasks.hitl.eval import (
    EvaluationResult,
    SWEScorePolicy,
    SWEUnitScorePolicy,
    aggregate_swe_evaluations,
    detect_worker_reward_hacking,
    eval_script_of,
    run_swe_patch_rubrics_eval,
    run_unit_eval_in_sandbox,
)
from nexus.extensions.tasks.hitl.recipe.swe_chain_v2.dataset_preprocess import (
    preprocess_eval_script,
    preprocess_problem_statement,
)
from nexus.extensions.tasks.hitl.recipe.swe_chain_v2.hook import HOOK_NAME
from nexus.extensions.tasks.hitl.recipe.swe_chain_v2.requirement_groups import (
    DEFAULT_GROUP_MAX_CHARS,
    DEFAULT_GROUP_SIZE,
    group_requirements,
)
from nexus.extensions.tasks.hitl.utils.runtime_git import cleanup_test_conflicts
from nexus.extensions.tasks.swe.dataset_handler import get_handler
from nexus.extensions.tasks.swe.infer import (
    get_patch_from_runtime,
    run_pre_agent_script,
    verify_git_truncation,
)
from nexus.trace.phases import phase
from nexus.trace.tracer import get_current_span_id, update_span
from nexus.utils.timeout import async_timeout


logger = logging.getLogger(__name__)

# Which evaluators a run may ask for. Adding one means adding it here and preparing
# its inputs below; see ``hitl/eval/evaluators``.
EvalType = Literal['unit_test', 'patch_rubrics']


class SWEChainV2TaskInput(TaskRunInput, WithAgentRunner):
    """Input for long-horizon chained SWE HITL runs.

    What the conversation runs under -- the persona, the worker backend and its
    config, the client names, the turn cap -- is ``agent_config``. This carries the
    task: the dataset record, the requirement chain, and how the result is scored.
    """

    instance_id: str
    instance: dict[str, Any] = Field(default_factory=dict)
    """The whole dataset record, which the generator reads and passes through.

    Defaulted rather than required, so constructing this input to reach one of the
    helpers does not mean supplying a record they never look at. What a run needs of
    it is checked in :func:`_chain_instance`, which can name the missing field.

    A plain dict rather than ``swe.task.SWEbenchInstanceDict``: a chain record carries
    ``patch`` and ``problem_statements`` as lists, and that type declares both as
    ``str``, so validating a real record against it fails outright.

    What this task itself reads::

        instance_id                         which record this is
        problem_statements  list[str]       one per thing the user asks for
        patch               list[str]       the reference patch per statement
        source_dataset_type, dataset_type   selects the dataset handler
        image | image_name                  the sandbox image
        base_commit, runtime_base_commit    what the worker's diff is taken against
        workdir             default /testbed
        language            default 'py', and only when the handler declines
        repo, test_patch, spec_dict         evaluation and reward-hacking checks
        rubrics_yaml                        read by the generator, not here

    Not an exhaustive schema: the record also goes to the dataset handler
    ``source_dataset_type`` selects, which reads fields of its own -- the
    ``multi_swe_bench`` one takes ``scripts``, ``f2p_tests`` and ``base.sha``. So the
    record is passed whole; picking fields out of it here would break a handler
    silently, since a missing field reads as a default rather than an error.
    """
    problem_statement: str | None = None

    infer_runtime_name: str = 'main'
    eval_runtime_name: str = 'eval'
    image_override: str | None = None
    pre_agent_script: list[str] | None = None
    skip_git_truncate: bool = False
    remove_git_dir: bool = False

    agent_timeout_seconds: int = 3600
    dataset_preset: str = ''

    eval_types: list[EvalType] = Field(default_factory=lambda: ['unit_test'])
    eval_timeout_seconds: int = 3600
    skip_eval: bool = False
    # Reward-hacking detection over the worker trajectory. Flag-only today (writes
    # eval_details['reward_hack']; never changes the score). Kept as a separate input
    # so an RL run can later branch on it to mask/penalize from the same flags.
    enable_reward_hacking_detection: bool = True
    skip_empty_patch_eval: bool = False
    score_policy: SWEScorePolicy = 'mean'
    unit_score_policy: SWEUnitScorePolicy = 'official'

    rubrics_yaml: str = ''
    rubrics_eval_client_name: str = 'rubric_eval'

    chain_problem_statements: list[str] | None = None
    chain_patches: list[str] | None = None
    include_chain_step_prefixes: bool = True

    # How many dataset entries one released requirement carries. 1 releases the chain
    # as the dataset holds it.
    requirement_group_size: int = DEFAULT_GROUP_SIZE
    requirement_group_max_chars: int = DEFAULT_GROUP_MAX_CHARS


class SWEChainV2TaskOutput(TaskRunOutput, WithScore):
    """Output of a chained SWE HITL run."""

    completed: bool
    resolved: bool
    patch_text: str = ''
    worker_rounds: int = 0
    usage_summary: dict[str, Any] = Field(default_factory=dict)
    run_meta: dict[str, Any] = Field(default_factory=dict)
    """How the agent ran and what it did, passed through from the agent output."""
    eval_details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    termination_reason: str | None = None
    completed_steps: int = 0
    total_steps: int = 0


@task_registry.register('hitl/swe_chain_v2_task')
async def run_swe_chain_v2_task(
    task_input: SWEChainV2TaskInput,
) -> SWEChainV2TaskOutput:
    """Run a chained SWE instance and evaluate the final cumulative patch.

    The agent is handed every requirement rather than one combined prompt, which is
    what this recipe varies; the trajectories are dropped from the output because the
    run-level chat tracker exports the same content under ``processor_results.chat``.
    """

    ctx = require_context()
    if not ctx.runtime_provider:
        raise RuntimeError('HITL SWE chain recipe requires a runtime provider.')

    instance = _chain_instance(task_input)
    source_dataset_type = _source_dataset_type(instance)
    chain_dataset_type = str(instance.get('dataset_type') or '')
    instance['chain_dataset_type'] = chain_dataset_type
    instance['dataset_type'] = source_dataset_type

    dataset_preset = task_input.dataset_preset or os.environ.get('HITL_DATASET_PRESET', 'default')
    raw_problem_statements = task_input.chain_problem_statements or _string_list(instance.get('problem_statements'))
    problem_statements = [
        preprocess_problem_statement(statement, dataset_preset) for statement in raw_problem_statements
    ]
    if not problem_statements:
        raise ValueError(f'No problem_statements found for chain instance {task_input.instance_id}.')
    released_requirements = group_requirements(
        problem_statements,
        size=task_input.requirement_group_size,
        max_chars=task_input.requirement_group_max_chars,
    )

    chain_patches = task_input.chain_patches or _string_list(instance.get('patch'))
    instance['chain_patches'] = chain_patches
    instance['patch'] = _combine_patches(chain_patches) if chain_patches else ''

    dataset_handler = get_handler(source_dataset_type)
    instance_language = dataset_handler.get_repo_language(instance) or str(  # ty: ignore[invalid-argument-type]
        instance.get('language') or 'py'
    )
    instance_image = instance.get('image') or instance.get('image_name', '')
    base_commit = _base_commit(instance)
    instance_workdir = instance.get('workdir', '/testbed')
    instance['workdir'] = instance_workdir

    image = task_input.image_override or instance_image
    configure_params: dict[str, Any] = {'image': image}
    request_timeout = int(getattr(ctx.runtime_provider, 'request_timeout', 1800))
    infer_runtime_name = task_input.infer_runtime_name
    eval_runtime_name = task_input.eval_runtime_name

    patch_text = ''
    agent_output: AgentWithUserOutput | None = None
    eval_results: dict[str, EvaluationResult] = {}

    try:
        phase('task.init')
        logger.info('[swe_chain_v2] [%s] Acquiring infer runtime (image=%s)', task_input.instance_id, image)
        ctx.runtime_provider.configure(infer_runtime_name, **configure_params)
        with phase('task.init.get_runtime'):
            infer_runtime = await ctx.get_runtime(infer_runtime_name)
        _record_runtime_id(ctx, infer_runtime_name)
        logger.info('[swe_chain_v2] [%s] Infer runtime ready, initializing repo', task_input.instance_id)

        try:
            async with async_timeout(request_timeout):
                with phase('task.init.runtime_init'):
                    await dataset_handler.infer_runtime_init(
                        instance,  # ty: ignore[invalid-argument-type]
                        infer_runtime,
                        skip_git_truncate=task_input.skip_git_truncate,
                        remove_git_dir=task_input.remove_git_dir,
                    )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise TimeoutError(f'infer_runtime_init timed out: {exc}') from exc
        logger.info('[swe_chain_v2] [%s] Repo initialized, starting agent', task_input.instance_id)

        if task_input.pre_agent_script:
            formatted_commands = [command.format(**instance) for command in task_input.pre_agent_script]
            with phase('task.init.pre_agent_script'):
                await run_pre_agent_script(infer_runtime, formatted_commands, working_dir=instance_workdir)

        # Read after everything that can rewrite history -- the handler, and a
        # pre-agent script that may truncate on its own -- so it describes the tree the
        # worker starts from. Only when the detection that consumes it is on. Left unset
        # on failure, which is the same unverified state as not measuring at all.
        if task_input.enable_reward_hacking_detection:
            try:
                async with async_timeout(request_timeout):
                    with phase('task.init.verify_truncation'):
                        instance['history_truncated'] = await verify_git_truncation(
                            infer_runtime,
                            instance.get('runtime_base_commit') or base_commit,
                            working_dir=instance_workdir,
                        )
            except Exception as exc:  # noqa: BLE001 - unverified is a state the detector handles
                logger.warning('[swe_chain_v2] could not verify git truncation: %s', exc)

        problem_statement = _combined_problem_statement(
            problem_statements,
            include_prefixes=task_input.include_chain_step_prefixes,
        )
        eval_script = preprocess_eval_script(eval_script_of(dataset_handler, instance), dataset_preset)

        ctx.task_config['problem_statement'] = problem_statement
        ctx.task_config['working_dir'] = instance_workdir
        ctx.task_config['instance'] = instance
        ctx.task_config['eval_script'] = eval_script
        ctx.task_config['chain_problem_statements'] = problem_statements
        ctx.task_config['chain_patches'] = chain_patches
        ctx.task_config['source_dataset_type'] = source_dataset_type
        ctx.task_config['dataset_preset'] = dataset_preset
        ctx.tool_state.setdefault('working_dir', instance_workdir)

        logger.info(
            '[swe_chain_v2] [%s] handing over %d requirements in %d releases',
            task_input.instance_id,
            len(problem_statements),
            len(released_requirements),
        )
        phase('task.agent')
        produced = await asyncio.wait_for(
            task_input.run_agent(
                user_prompt=released_requirements[0],
                # The whole chain, handed to the user one at a time; the queue releases
                # requirement N+1 once N is accepted. This recipe's hook rides along
                # here so the one call site carries everything the run computes.
                requirements=released_requirements,
                hooks=[{'name': HOOK_NAME, 'args': {'instance_id': task_input.instance_id}}],
                working_dir=instance_workdir,
            ),
            timeout=task_input.agent_timeout_seconds,
        )
        # run_agent is typed to the base AgentRunOutput; this recipe reads the
        # AgentWithUserOutput subclass fields, so the run must actually be that agent.
        if not isinstance(produced, AgentWithUserOutput):
            raise TypeError(
                f'the chain task reads {AgentWithUserOutput.__name__} fields, and this run produced '
                f'{type(produced).__name__}. Set agent_name to one that returns it.'
            )
        agent_output = produced

        logger.info('[swe_chain_v2] [%s] Agent finished, extracting patch', task_input.instance_id)
        with phase('task.agent.cleanup_test_conflicts'):
            await cleanup_test_conflicts(
                infer_runtime,
                working_dir=instance_workdir,
                test_patch=str(instance.get('test_patch') or ''),
                base_commit=base_commit,
            )

        if task_input.remove_git_dir:
            await ctx.runtime_provider.release(name=infer_runtime_name)
            return _output(
                agent_output,
                total_steps=len(released_requirements),
                eval_details={'skipped': {'reason': 'remove_git_dir is true; patch extraction skipped'}},
                error='Skipping patch extraction because remove_git_dir is true.',
            )

        patch_base = instance.get('runtime_base_commit') or base_commit
        with phase('task.agent.get_patch'):
            patch_text = await get_patch_from_runtime(
                infer_runtime,
                working_dir=instance_workdir,
                language=instance_language,
                base_commit=patch_base,
            )
        await ctx.runtime_provider.release(name=infer_runtime_name)
        logger.info(
            '[swe_chain_v2] [%s] Patch extracted (%d chars), starting eval',
            task_input.instance_id,
            len(patch_text or ''),
        )

        if task_input.skip_eval or not task_input.eval_types:
            return _output(
                agent_output,
                total_steps=len(released_requirements),
                patch_text=patch_text or '',
                eval_details={'skipped': {'reason': 'skip_eval or empty eval_types'}},
            )

        if not patch_text and task_input.skip_empty_patch_eval:
            return _output(
                agent_output,
                total_steps=len(released_requirements),
                eval_details={'skipped': {'reason': 'empty patch'}},
                error='Agent did not produce any changes. Skipping evaluation.',
            )

        phase('task.scoring')
        if 'unit_test' in task_input.eval_types:
            logger.info('[swe_chain_v2] [%s] Running unit_test eval', task_input.instance_id)
            with phase('task.scoring.unit_test'):
                eval_results['unit_test'] = await run_unit_eval_in_sandbox(
                    instance=instance,
                    instance_id=task_input.instance_id,
                    patch_text=patch_text or '',
                    dataset_handler=dataset_handler,
                    configure_params=configure_params,
                    runtime_name=eval_runtime_name,
                    working_dir=instance_workdir,
                    base_commit=base_commit,
                    init_timeout_seconds=request_timeout,
                    eval_timeout_seconds=task_input.eval_timeout_seconds,
                )
            logger.info('[swe_chain_v2] [%s] unit_test eval done', task_input.instance_id)

        if 'patch_rubrics' in task_input.eval_types:
            rubrics_yaml = task_input.rubrics_yaml
            logger.info(
                '[swe_chain_v2] [%s] Running patch_rubrics eval (rubrics=%dchars)',
                task_input.instance_id,
                len(rubrics_yaml or ''),
            )
            with phase('task.scoring.patch_rubrics'):
                eval_results['patch_rubrics'] = await run_swe_patch_rubrics_eval(
                    patch_text=patch_text or '',
                    problem_statement=problem_statement,
                    rubrics_yaml=rubrics_yaml,
                    eval_client_name=task_input.rubrics_eval_client_name,
                )
            logger.info('[swe_chain_v2] [%s] patch_rubrics eval done', task_input.instance_id)

        score, resolved, eval_details = aggregate_swe_evaluations(
            eval_results,
            score_policy=task_input.score_policy,
            unit_score_policy=task_input.unit_score_policy,
        )

        if task_input.enable_reward_hacking_detection:
            # Flag-only: record whether the worker fetched the answer, from the network or
            # from the runtime itself. An RL run would branch here to mask or penalize.
            with phase('task.scoring.reward_hacking'):
                reward_hack = detect_worker_reward_hacking(
                    agent_output.trajectories,
                    task_input.instance_id,
                    instance.get('repo'),
                    patch_text or '',
                    score,
                    base_commit=patch_base,
                    history_truncated=instance.get('history_truncated'),
                )
            if reward_hack:
                eval_details['reward_hack'] = reward_hack

        return _output(
            agent_output,
            total_steps=len(released_requirements),
            patch_text=patch_text or '',
            score=score,
            resolved=resolved,
            eval_details=eval_details,
        )
    except Exception as exc:
        logger.error('[swe_chain_v2] Task failed: %s: %s', type(exc).__name__, exc, exc_info=True)
        # On the span as well as in the output: the run record is read by whoever
        # scores a batch, and the span by whoever decides to retry one.
        _record_termination(type(exc).__name__)
        with contextlib.suppress(Exception):
            await ctx.runtime_provider.release(name=infer_runtime_name)
        with contextlib.suppress(Exception):
            await ctx.runtime_provider.release(name=eval_runtime_name)
        return _output(
            agent_output,
            total_steps=len(released_requirements),
            patch_text=patch_text or '',
            eval_details={name: result.model_dump() for name, result in eval_results.items()},
            error=f'{type(exc).__name__}: {exc}',
            termination_reason=type(exc).__name__,
        )
    finally:
        # Closes whichever phase was open, including on the error path. Without it the
        # last phase span never ends and its duration reads as the whole run.
        phase()


def _record_runtime_id(ctx: ExecutionContext, runtime_name: str) -> None:
    """Note which sandbox served this run on the current span.

    Links a sandbox-side failure to the run that hit it; without it the only join is
    by timestamp across two log streams.
    """
    runtime_id = ctx.get_runtime_id(runtime_name)
    span_id = get_current_span_id()
    if runtime_id and span_id:
        update_span(span_id, attributes={'hytuner.runtime_id': runtime_id})


def _record_termination(reason: str) -> None:
    """Put the termination reason where the trace consumers look for it."""
    span_id = get_current_span_id()
    if span_id:
        update_span(span_id, attributes={'hytuner.termination_reason': reason})


def _chain_instance(task_input: 'SWEChainV2TaskInput') -> dict[str, Any]:
    """A working copy of the record, refused if a field the task needs is missing.

    Copied because the task writes derived values back onto it (``chain_patches``,
    the combined ``patch``, ``workdir``) and the input is shared with whatever else
    reads the run config.

    The three fields checked are the ones with no usable default: without an image
    there is no sandbox, without a base commit the worker's diff is taken against
    nothing, and without statements there is nothing for the user to ask for. The
    rest have defaults in the code that reads them, so a check here would only
    duplicate them.
    """
    record = dict(task_input.instance)
    if not record:
        raise ValueError(
            f'chain instance {task_input.instance_id} arrived with no dataset record; '
            'the generator passes it as task_config["instance"]'
        )
    if not (record.get('image') or record.get('image_name')):
        raise ValueError(f'chain instance {task_input.instance_id} carries no image')
    if not _base_commit(record):
        raise ValueError(f'chain instance {task_input.instance_id} carries no base_commit')
    if not (task_input.chain_problem_statements or _string_list(record.get('problem_statements'))):
        raise ValueError(f'chain instance {task_input.instance_id} carries no problem_statements')
    return record


def _combined_problem_statement(problem_statements: list[str], *, include_prefixes: bool) -> str:
    if not include_prefixes:
        return '\n\n'.join(problem_statements)
    return '\n\n'.join(f'Chain request {index + 1}:\n{statement}' for index, statement in enumerate(problem_statements))


def _combine_patches(patches: list[str]) -> str:
    return '\n'.join(patch.strip() for patch in patches if patch and patch.strip())


def _source_dataset_type(instance: dict[str, Any]) -> str:
    return str(instance.get('source_dataset_type') or instance.get('dataset_type') or '')


def _base_commit(instance: dict[str, Any]) -> str:
    """The commit the repo is put at before the worker starts."""
    return str(instance.get('base_commit') or '')


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [item for item in (str(item) for item in value) if item]
    if isinstance(value, str) and value:
        return [value]
    return []


def _output(
    agent_output: AgentWithUserOutput | None,
    *,
    total_steps: int,
    patch_text: str = '',
    score: float = 0.0,
    resolved: bool = False,
    eval_details: dict[str, Any] | None = None,
    error: str | None = None,
    termination_reason: str | None = None,
) -> SWEChainV2TaskOutput:
    """Shape one return path's outcome as this recipe's output.

    ``worker_rounds`` and ``completed_steps`` come off the agent output's own fields
    rather than out of ``run_meta``, which carries a copy of both: an agent that raised
    before filling that bundle still reports its counters, and there is one place to
    change when a counter changes.

    ``trajectories`` is not among the fields: it is excluded from the agent output's dump
    for the same reason it is not copied here -- the run-level chat tracker already
    exports those conversations under ``processor_results.chat``.
    """
    return SWEChainV2TaskOutput(
        score=score,
        completed=agent_output is not None and error is None,
        resolved=resolved,
        patch_text=patch_text,
        worker_rounds=agent_output.worker_rounds if agent_output else 0,
        usage_summary=dict(agent_output.usage_summary) if agent_output else {},
        run_meta=dict(agent_output.run_meta) if agent_output else {},
        eval_details=eval_details or {},
        error=error or (agent_output.stderr or None if agent_output else None),
        termination_reason=termination_reason,
        completed_steps=len(agent_output.accepted) if agent_output else 0,
        total_steps=total_steps,
    )


__all__ = ['SWEChainV2TaskInput', 'SWEChainV2TaskOutput', 'run_swe_chain_v2_task']
