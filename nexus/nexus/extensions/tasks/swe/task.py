import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shlex
import time

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, cast

import aiofiles
import aiofiles.os

from pydantic import Field, TypeAdapter, WrapValidator

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.agents.swe_agent.platform_status import (
    PlatformExitStatus,
    exception_details,
    sandbox_failure_status,
)
from nexus.extensions.agents.swe_agent.tools import resolve_bash_backend
from nexus.extensions.harbor_bridge.agent_result import T3AgentResult, agent_termination_reason
from nexus.extensions.tasks.swe.constants import (
    GOLD_FIELDS,
    KEY_INSTANCE_ID,
    SWEbenchInstance,
    anti_hack_prompt_for_backend,
)
from nexus.extensions.tasks.swe.dataset_handler import get_handler
from nexus.extensions.tasks.swe.eval import _compute_partial_score, evaluate_and_grade_in_runtime
from nexus.extensions.tasks.swe.infer import (
    get_patch_from_runtime,
    run_pre_agent_script,
    verify_git_truncation,
)
from nexus.extensions.tasks.swe.reward_hacking import detect_reward_hacking
from nexus.extensions.tasks.swe.reward_hacking.monitors import run_swe_monitors
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse
from nexus.trace import get_current_span_id, get_processor, update_span
from nexus.trace.phases import phase
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor
from nexus.utils.timeout import async_timeout


def _keep_extra_fields(v: Any, handler: Any) -> dict:
    """Validate known fields via TypedDict, preserve extra fields.

    Pydantic strips unknown fields when validating TypedDict. This validator
    ensures that extra fields from upstream (e.g., dataset-specific fields)
    are preserved after validation.
    """
    validated = handler(v)
    if isinstance(v, dict) and isinstance(validated, dict):
        for key, value in v.items():
            if key not in validated:
                validated[key] = value
    return validated


# Type alias that preserves extra fields during Pydantic validation
SWEbenchInstanceDict = Annotated[SWEbenchInstance, WrapValidator(_keep_extra_fields)]

# TypeAdapter for validating instances loaded from files/server
_instance_adapter: TypeAdapter[SWEbenchInstanceDict] = TypeAdapter(SWEbenchInstanceDict)


logger = logging.getLogger(__name__)


class _AgentInfrastructureError(RuntimeError):
    """Structured agent failure using the shared training classification."""

    def __init__(self, message: str, *, failure_stage: str | None) -> None:
        super().__init__(message)
        self.termination_reason = agent_termination_reason(
            T3AgentResult(
                mode='agent',
                status='error',
                agent_exit_status='env_error',
                failure_kind='infrastructure',
                failure_stage=failure_stage,
            )
        )


async def load_instance_from_dataset(dataset_path: str, instance_id: str) -> SWEbenchInstanceDict:
    """Load a single instance from a local JSON/JSONL dataset."""
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
                    return _instance_adapter.validate_python(data)
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
                    return _instance_adapter.validate_python(item)
        elif isinstance(data, dict) and data.get('instance_id') == instance_id:
            return _instance_adapter.validate_python(data)

    raise ValueError(f'Instance {instance_id} not found in dataset {dataset_path}')


async def load_instance_from_server(dataset: str, instance_id: str, split: str | None = None) -> SWEbenchInstanceDict:
    """Load a single instance from the dataset server."""
    from nexus.utils.legacy_dataset_client import DatasetClient, parse_dataset_path

    namespace, dataset_name = parse_dataset_path(dataset)
    async with DatasetClient() as client:
        start_time = time.perf_counter()
        instance = await client.get_instance(namespace, dataset_name, instance_id, split)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug('get_instance(%s/%s, %s) took %.2f ms', namespace, dataset_name, instance_id, elapsed_ms)
        return _instance_adapter.validate_python(instance)


class SwebenchTaskInput(TaskRunInput, WithAgentRunner):
    """Inputs for single-run SWE-bench evaluation."""

    dataset_path: str = ''
    remote_dataset: str = ''  # Format: "namespace/dataset_name", e.g., "swe-public/SWE-bench_Verified"
    split: str = ''  # Optional split name for remote dataset
    instance_id: str
    instance: SWEbenchInstanceDict | None = None
    problem_statement: str | None = None
    infer_runtime_name: str = 'main'
    eval_runtime_name: str = 'eval'
    image_prefix: str = 'mirrors.tencent.com/hunyuan_yanguan'
    eval_timeout_seconds: int = 3600
    run_oracle: bool = False
    patch_text: str | None = None
    skip_empty_patch_eval: bool = False
    skip_infer_patch_eval: bool = False
    skip_git_truncate: bool = False  # Skip git history truncation in infer_runtime_init
    remove_git_dir: bool = False  # Remove git directory in infer_runtime_init
    raise_on_runtime_error: bool = False  # Re-raise RuntimeError for retry instead of swallowing
    image_override: str | None = None
    pre_agent_script: list[str] | None = None
    use_partial_score: bool = False
    # Record violation evidence only when detection is enabled.
    enable_reward_hacking_detection: bool = False
    # Enable the output checks used by quality penalties.
    enable_output_monitoring: bool = False
    # Quality penalty coefficients; zero leaves the score unchanged.
    codeswitch_penalty: float = 0.0
    codeswitch_penalty_reward_threshold: float = 1.0
    str_replace_old_eq_new_penalty: float = 0.0
    str_replace_old_eq_new_reward_threshold: float = 1.0
    output_format_bad_pattern_penalty: float = 0.0
    output_format_bad_pattern_reward_threshold: float = 1.0
    let_me_loop_penalty: float = 0.0
    let_me_loop_reward_threshold: float = 1.0
    enable_anti_hack: bool = False
    # Replace the score of a detected violation when enable_anti_hack is set.
    hack_reward_override: float | None = None
    # Skip reward overrides during validation.
    is_validation: bool = False
    # Control notice injection independently of violation detection and reward adjustment.
    enable_anti_hack_prompt: bool = False
    agent_instance_exclude_fields: list[str] = Field(
        default_factory=list, description='Additional instance fields withheld from the agent; answers remain hidden'
    )
    artifact_cleanup_globs: list[str] = Field(
        default_factory=list, description='Explicit artifact basename patterns to remove before inference'
    )
    patch_exclude_patterns: list[str] = Field(
        default_factory=list, description='Additional gitignore patterns applied when extracting the submitted patch'
    )
    eval_max_attempts: int = 1


class SwebenchTaskOutput(TaskRunOutput, WithScore):
    """Result of SWE-bench evaluation."""

    completed: bool
    resolved: bool
    # In-memory artifacts (no workspace reads/writes).
    patch_text: str = ''
    report_json: str = ''  # JSON string of evaluation report (when available)
    test_output: str = ''  # raw test output from evaluation
    apply_log: str = ''  # raw log of applying patch
    error: str | None = None
    error_code: str | None = None  # Error code from evaluation (e.g., PATCH_APPLY_FAILED)
    termination_reason: str | None = None  # Termination reason from runtime
    # Extra fields passed through to HyTuner (e.g. reward-hacking flags). Runner merges
    # these into the HyTuner output, so adding a key here flows it to training unchanged.
    extra_fields: dict[str, Any] = Field(default_factory=dict)


def _chat_messages(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a ChatTrackerProcessor snapshot into one message list for detection.

    snapshot_result returns {'trajectories': [...]}; each trajectory is either a list of
    messages (global_first mode) or a {'messages': [...]} dict (branch_last mode). Every
    trajectory's messages are concatenated, so detection sees the full chat history,
    including turns moved out of the live buffer by context-management compaction.
    """
    messages: list[dict[str, Any]] = []
    for trajectory in snapshot.get('trajectories') or []:
        if isinstance(trajectory, dict):
            messages.extend(trajectory.get('messages') or [])
        elif isinstance(trajectory, list):
            messages.extend(trajectory)
    return messages


_MIRRORED_HACK_FLAGS = ('reward_hack_mask', 'reward_hack_attempted', 'reward_hack_effective')


def _mirror_hack_flags(extra_fields: dict[str, Any]) -> None:
    """Copy the detector's flags into ``reward_extra_info`` as well.

    The flags already reach the trainer's batch through its sweep over every extra field, which
    is what the step-level hack metrics read. Two other consumers read the ``reward_extra_info``
    channel and nothing else: the jsonl sample dumps, and the per-data-source aggregation over
    the whole validation set rather than one batch of it. Mirroring puts a flagged run in front
    of both. Scalars only: ``reward_hack_rules`` is a list and has no column to go to.
    """
    reward_extra_info = extra_fields.get('reward_extra_info')
    if not isinstance(reward_extra_info, dict):
        reward_extra_info = {}
        extra_fields['reward_extra_info'] = reward_extra_info
    for flag in _MIRRORED_HACK_FLAGS:
        reward_extra_info[flag] = 1.0 if extra_fields.get(flag) else 0.0


def _build_leak_needles(instance: Mapping[str, Any], *, cap_kept: int = 60, cap_added: int = 60) -> list[str]:
    """Build bounded fingerprints used to find possible reference artifacts.

    Keep signed diff lines and stripped additions, excluding context and headers.
    Neither form proves that a file is an answer copy: ordinary dependencies can
    contain identical code. The cleanup must also confirm a standalone unified
    diff before deleting a content match. Known evaluation artifacts are handled
    separately by path, with runtime and build-cache exclusions.
    """
    _skip_head = ('//', '#', '*', '/*', '+++', '---', '@@', 'diff ', 'index ')
    sources = [str(instance.get(k) or '') for k in ('patch', 'test_patch')]

    kept: list[str] = []
    added: list[str] = []
    seen_kept: set[str] = set()
    seen_added: set[str] = set()

    for blob in sources:
        for raw in blob.splitlines():
            line = raw.strip()
            if len(line) < 24 or line.startswith(_skip_head):
                continue
            # Do not use unchanged context lines as fingerprint candidates.
            if raw.startswith(('+', '-')) and line not in seen_kept:
                seen_kept.add(line)
                kept.append(line)
            # Stripped additions are candidates, not proof of unique content.
            if raw.startswith('+') and not raw.startswith('+++'):
                body = raw[1:].strip()
                if len(body) >= 24 and not body.startswith(_skip_head) and body not in seen_added:
                    seen_added.add(body)
                    added.append(body)

    added.sort(key=len, reverse=True)
    return kept[:cap_kept] + added[:cap_added]


def _checked_cleanup_response(result: Any, operation: str) -> RunCommandResponse:
    response = RunCommandResponse.model_validate(result)
    details = (
        f'SWE reference cleanup {operation}: status={response.status.value}, '
        f'return_code={response.return_code}; {(response.stderr or "")[:2000]}'
    )
    if response.status == CommandRunStatus.TimeLimitExceeded:
        raise TimeoutError(details)
    if response.status != CommandRunStatus.Finished or response.return_code != 0:
        raise RuntimeError(details)
    return response


@task_registry.register('swe_task')
async def run_swe_task(task_input: SwebenchTaskInput) -> SwebenchTaskOutput:
    """Run a single SWE-bench inference with agent and evaluation using the runtime provider."""
    ctx = require_context()
    platform_state = getattr(ctx, 'tool_state', {}).get('_superswe_platform')

    def set_platform_stage(stage: str) -> None:
        if isinstance(platform_state, dict):
            platform_state['stage'] = stage
            if stage == 'agent':
                platform_state['agent_started'] = True
            if stage.startswith('judge.'):
                platform_state['judge_started'] = True
                platform_state['patch_text'] = patch_text or ''

    if not ctx.runtime_provider:
        raise RuntimeError('SWE-bench task requires a runtime provider.')

    # Reward-hacking detection is off by default. When enabled, the flags default to
    # all-clear here and are populated on the success path; defining them up front means
    # every return path (early skips, success, outer error) carries a consistent
    # detection contract. When disabled, detection adds no extra fields.
    reward_hack_extra: dict[str, Any] = (
        {
            'reward_hack_mask': False,
            'reward_hack_rules': [],
            'reward_hack_attempted': False,
            'reward_hack_effective': False,
        }
        if task_input.enable_reward_hacking_detection
        else {}
    )

    instance = task_input.instance
    if instance is None:
        if task_input.dataset_path:
            instance = await load_instance_from_dataset(task_input.dataset_path, task_input.instance_id)
        elif task_input.remote_dataset:
            instance = await load_instance_from_server(
                task_input.remote_dataset, task_input.instance_id, task_input.split or None
            )
        else:
            raise ValueError('Instance is required either in task input, dataset path, or remote_dataset.')

    dataset_handler = get_handler(instance.get('dataset_type'))
    instance_language = dataset_handler.get_repo_language(instance) or 'py'
    instance_image = instance.get('image') or instance.get('image_name', '')
    # Fall back to the dataset base revision when base_commit is absent.
    base_commit = instance.get('base_commit', '')
    if not base_commit:
        base_commit = instance.get('base', {}).get('sha', '')

    infer_runtime_name = task_input.infer_runtime_name
    eval_runtime_name = task_input.eval_runtime_name

    image = instance_image
    logger.debug(f'Using image {image} for instance {task_input.instance_id}.')
    configure_params: dict[str, Any] = {'image': image}
    if task_input.image_override:
        configure_params['image'] = task_input.image_override
        image = task_input.image_override

    instance_workdir = instance.get('workdir', '/testbed')
    logger.debug(f'instance_workdir, {instance_workdir}')
    # Preserve the work directory for grading initialization.
    instance['workdir'] = instance_workdir

    # Get patch text and reverse flag for evaluation
    patch_text: str | None = None
    reverse_patch: bool = False
    is_patch_from_base_commit: bool = False

    term_reason = 'ENV_DONE'
    execution_stage = 'task.init'
    runtime_id = 'unknown'
    eval_runtime_id = 'unknown'
    request_timeout = 1800

    reward_time = -1.0

    async def finish_agent_runtime_failure(sandbox: dict, error: str) -> SwebenchTaskOutput:
        # This is a terminal zero-score adjudication, not a successful test run.
        # Keep completed=False and the skipped components visible to audit tools.
        assert isinstance(platform_state, dict)
        assert ctx.runtime_provider is not None
        platform_state['terminal_agent_failure'] = {'sandbox': dict(sandbox), 'error': error}
        platform_state['patch_text'] = patch_text or ''
        for name in (infer_runtime_name, eval_runtime_name):
            with contextlib.suppress(Exception):
                await ctx.runtime_provider.release(name=name)
            ctx._runtimes.pop(name, None)
        return SwebenchTaskOutput(
            score=0.0,
            completed=False,
            resolved=False,
            patch_text=patch_text or '',
            error=error,
            termination_reason='ENV_DONE',
            extra_fields={**reward_hack_extra, 'grading_skipped': 'agent_runtime_failure'},
        )

    if hasattr(ctx.runtime_provider, 'request_timeout'):
        request_timeout = cast(float | None, ctx.runtime_provider.request_timeout)

    try:
        if task_input.run_oracle:
            logger.debug(f'Running in oracle mode for instance {task_input.instance_id}')

            # Check dataset type
            dataset_type = instance.get('dataset_type', '')

            if dataset_type == 'multi-swe-bench':
                # For multi-swe-bench, use fix_patch
                fix_patch = instance.get('fix_patch', '')
                if fix_patch:
                    patch_text = fix_patch
                    logger.debug(f'Using fix_patch as oracle for multi-swe-bench (length={len(patch_text)}).')
                else:
                    raise ValueError(f'No fix_patch available for multi-swe-bench instance {task_input.instance_id}.')
            else:
                # Try patch first (most datasets), then bug_patch (SWE-Smith/SWE-Cosmos)
                patch_text = instance.get('patch', '')
                reverse_patch = instance.get('dataset', '').lower().startswith('swe-cosmos/')
                if patch_text:
                    logger.debug(f'Using oracle patch from instance["patch"] (length={len(patch_text)}).')
                else:
                    bug_patch = instance.get('bug_patch', '')
                    if bug_patch:
                        # SWE-Smith: use bug_patch with reverse apply
                        patch_text = bug_patch
                        reverse_patch = True
                        logger.debug(f'Using reverse bug_patch as oracle (length={len(patch_text)}).')
                    else:
                        raise ValueError(f'No oracle patch available for instance {task_input.instance_id}.')
        elif task_input.patch_text is not None:
            logger.debug(f'Using provided patch text (length={len(task_input.patch_text)}).')
            patch_text = task_input.patch_text
        else:
            logger.debug(f'Running inference for instance {task_input.instance_id}.')
            set_platform_stage('runtime.create')
            ctx.runtime_provider.configure(infer_runtime_name, **configure_params)
            phase('task.init')
            with phase('task.init.get_runtime'):
                infer_runtime = await ctx.get_runtime(infer_runtime_name)

            # Remove known evaluation artifacts before the agent starts. Content
            # matches alone are insufficient: require standalone patch structure
            # so imports, decorators and embedded examples in dependencies survive.
            # Preserve runtime/worker environments and precompiled project caches.
            cleanup_cmd = r"""
set -euo pipefail
# grep returns 1 for an empty result, which is not a cleanup failure.
grep_or_empty() { grep "$@" || [ "$?" -eq 1 ]; }
# Anchor exclusions at directory boundaries to preserve only the intended paths.
PRUNE='^(/tmp/nexus|/tmp/project_build|/tmp/node-compile-cache|/tmp/\.X11-unix)(/|$)|^/tmp/tmux-'
NEEDLES=/tmp/.b4_needles

# Known evaluation/build artifacts (only direct children of these roots).
for ROOT in /tmp /root /home; do
  [ -d "$ROOT" ] || continue
  find "$ROOT" -mindepth 1 -maxdepth 1 \
    \( __NEXUS_ARTIFACT_NAME_FILTERS__ \) \
  | grep_or_empty -Ev "$PRUNE" \
  | while IFS= read -r p; do rm -rf -- "$p"; echo "B4_RM_GLOB:$p"; done
done

# Fingerprints select candidates; only a standalone unified diff confirms a copy.
# In particular, a Python module quoting a diff in a docstring is not a patch.
if [ -s "$NEEDLES" ]; then
  grep_or_empty -rlFf "$NEEDLES" /tmp /root \
    | grep_or_empty -Ev "$PRUNE" \
    | grep_or_empty -Ev '\.(o|a|so|lo|la|obj|pyc|class|jar|rlib|rmeta)$' \
    | while IFS= read -r f; do
        [ -f "$f" ] || continue
        if awk '
          /^[[:space:]]*$/ { next }
          !started {
            started = 1
            if ($0 !~ /^(diff --git |--- )/) { invalid = 1; exit }
          }
          /^--- / { old_path = 1 }
          /^\+\+\+ / { if (old_path) new_path = 1 }
          /^@@ -[0-9]+(,[0-9]+)? \+[0-9]+(,[0-9]+)? @@/ { if (new_path) hunk = 1 }
          END { exit (invalid || !hunk) }
        ' "$f"; then
          rm -f -- "$f"
          echo "B4_RM_FP:$f"
        else
          status=$?
          [ "$status" -eq 1 ] || exit "$status"
        fi
      done
fi
rm -f "$NEEDLES"

rm -rf /tmp/nexus/tools/editor-history
"""
            # Pattern cleanup is an explicit dataset choice, not an implicit filename policy.
            name_filters = '-false' + ''.join(
                f' -o -name {shlex.quote(pattern)}' for pattern in task_input.artifact_cleanup_globs
            )
            cleanup_cmd = cleanup_cmd.replace('__NEXUS_ARTIFACT_NAME_FILTERS__', name_filters)
            set_platform_stage('environment.cleanup')
            try:
                needles = _build_leak_needles(instance)
                material = '\n'.join(needles)
                # Keep full fingerprints out of shell arguments. Writing even an empty
                # payload also prevents stale fingerprints from a previous attempt.
                transfer = _checked_cleanup_response(
                    await infer_runtime.call_executor(
                        'run_command',
                        {
                            'command': 'umask 077; cat > /tmp/.b4_needles && sha256sum /tmp/.b4_needles',
                            'stdin': material,
                            'cwd': '/',
                        },
                    ),
                    'fingerprint transfer',
                )
                expected_digest = hashlib.sha256(material.encode()).hexdigest()
                if (transfer.stdout or '').split(maxsplit=1)[:1] != [expected_digest]:
                    raise RuntimeError('SWE reference cleanup fingerprint checksum mismatch')
                res = _checked_cleanup_response(
                    await infer_runtime.call_executor('run_command', {'command': cleanup_cmd, 'cwd': '/'}),
                    'artifact removal',
                )
                removed = [ln for ln in (res.stdout or '').splitlines() if ln.startswith(('B4_RM_GLOB:', 'B4_RM_FP:'))]
                if removed:
                    logger.info(f'[cleanup] B-4 removed {len(removed)} leak artifact(s): {removed[:20]}')
                logger.info(f'[cleanup] B-4 needles={len(needles)}')
            except (asyncio.TimeoutError, TimeoutError):
                raise
            except Exception as e:
                raise RuntimeError(f'Failed to prepare SWE reference cleanup: {e}') from e

            set_platform_stage('environment.initialize')
            # Run dataset-specific initialization (e.g., SWE-Smith applies bug patch and sanitizes git history)
            # Note: Handler may modify instance['base_commit'] if git history is changed
            try:
                async with async_timeout(request_timeout):
                    await dataset_handler.infer_runtime_init(
                        instance,
                        infer_runtime,
                        skip_git_truncate=task_input.skip_git_truncate,
                        remove_git_dir=task_input.remove_git_dir,
                    )
                if hasattr(infer_runtime, 'config') and hasattr(infer_runtime.config, 'base_url'):
                    runtime_id = ctx.get_runtime_id(infer_runtime_name) or 'unknown'
                    span_id = get_current_span_id()
                    if span_id:
                        update_span(span_id, attributes={'hytuner.runtime_id': runtime_id})
            except (asyncio.TimeoutError, TimeoutError) as e:
                raise TimeoutError(f'Failed to run infer_runtime_init timeout: {e}') from e
            except Exception as e:
                raise RuntimeError(f'Failed to run infer_runtime_init: {e}') from e

            # Optional user-provided pre-agent script (e.g., truncate history to base commit)
            if task_input.pre_agent_script:
                formatted_commands = [cmd.format(**instance) for cmd in task_input.pre_agent_script]
                try:
                    with phase('task.init.runtime_setup'):
                        log_output = await run_pre_agent_script(
                            infer_runtime,
                            formatted_commands,
                            working_dir=instance_workdir,
                        )
                    logger.debug('Pre-agent script output:\n%s', log_output)
                except Exception as e:
                    raise RuntimeError(f'Pre-agent script failed: {e}') from e

            # Verify the repository state after dataset initialization and setup commands.
            try:
                async with async_timeout(request_timeout):
                    with phase('task.init.verify_truncation'):
                        instance['history_truncated'] = await verify_git_truncation(
                            infer_runtime,
                            instance.get('runtime_base_commit') or base_commit,
                            working_dir=instance_workdir,
                        )
                if instance.get('history_truncated') is False:
                    logger.warning(
                        'git truncation NOT effective for %s: refs can still reach commits outside '
                        'base_commit ancestry (golden patch may be readable via git history)',
                        task_input.instance_id,
                    )
                else:
                    logger.info(
                        'history_truncated=%s for %s', instance.get('history_truncated'), task_input.instance_id
                    )
            except Exception as e:
                # Leave truncation unverified when the diagnostic fails.
                logger.warning('Could not verify git truncation: %s', e)

            # Build the task text shared by the agent entry points.
            problem_statement = dataset_handler.get_problem_statement(instance)
            if not problem_statement:
                instance_id = instance.get(KEY_INSTANCE_ID, task_input.instance_id)
                raise ValueError(f'No problem statement found for instance {instance_id}.')
            # Use validated configuration, and match the backend the runner will pass to the agent.
            if task_input.enable_anti_hack_prompt:
                backend = resolve_bash_backend(ctx.agent_config.get('bash_backend'))
                notice = anti_hack_prompt_for_backend(backend)
                problem_statement = f'{problem_statement}\n\n{notice}'
            ctx.task_config['problem_statement'] = problem_statement
            ctx.task_config['working_dir'] = instance_workdir
            ctx.task_config['language'] = instance_language
            # Same precedence patch extraction uses below. It can only be resolved here, after
            # infer_runtime_init: handlers that rebuild history (SWE-Cosmos / SWE-Smith) mint the
            # commit the agent actually starts from, and those datasets ship no base_commit at all.
            ctx.task_config['base_commit'] = instance.get('runtime_base_commit') or base_commit
            # Only the agent-facing copy is filtered; oracle and grading retain the original instance.
            keep_gold = os.environ.get('SWE_TASK_CONFIG_KEEP_GOLD', '').strip().lower() in ('1', 'true', 'yes')
            excluded_fields = set(task_input.agent_instance_exclude_fields)
            if not keep_gold:
                excluded_fields.update(GOLD_FIELDS)
            ctx.task_config['instance'] = cast(
                SWEbenchInstance, {key: value for key, value in instance.items() if key not in excluded_fields}
            )
            ctx.tool_state.setdefault('working_dir', instance_workdir)

            execution_stage = 'task.agent'
            set_platform_stage('agent')
            phase('task.agent')
            agent_output = await task_input.run_agent(
                user_prompt=problem_statement,
                working_dir=instance_workdir,
            )
            logger.info('Agent finished with exit status: %s', getattr(agent_output, 'exit_status', 'unknown'))

            sandbox = getattr(agent_output, 'sandbox', None)
            if (
                isinstance(platform_state, dict)
                and getattr(agent_output, 'exit_status', None) == PlatformExitStatus.RUNTIME_ERROR_BY_AGENT
                and getattr(agent_output, 'exit_code', None) == 0
                and isinstance(sandbox, dict)
                and sandbox_failure_status(sandbox, during_tool=True) == PlatformExitStatus.RUNTIME_ERROR_BY_AGENT
            ):
                return await finish_agent_runtime_failure(
                    sandbox, str(getattr(agent_output, 'error', None) or 'Sandbox terminated during agent execution')
                )

            if getattr(agent_output, 'failure_kind', None) == 'infrastructure':
                failure_stage = getattr(agent_output, 'failure_stage', None)
                execution_stage = f'task.agent.{failure_stage or "infrastructure"}'
                detail = (
                    getattr(agent_output, 'error', None)
                    or getattr(agent_output, 'stderr', '')
                    or 'No diagnostic was returned.'
                )
                raise _AgentInfrastructureError(
                    f'Agent infrastructure failure during {failure_stage or "execution"}: {detail}',
                    failure_stage=failure_stage,
                )

            if task_input.remove_git_dir:
                # skip git patch extraction if remove_git_dir is set to true
                execution_stage = 'task.agent.release_runtime'
                await ctx.runtime_provider.release(name=infer_runtime_name)
                ctx._runtimes.pop(infer_runtime_name, None)
                return SwebenchTaskOutput(
                    score=0.0,
                    completed=True,
                    resolved=False,
                    # Preserve the extracted patch when returning before grading.
                    patch_text=patch_text or '',
                    error='Skipping inference patch extraction since remove_git_dir is set to true.',
                    extra_fields=reward_hack_extra,
                )
            # Extract git patch from the runtime.
            # Diff against runtime_base_commit if the handler reinitialized history (e.g. SWE-Cosmos / SWE-Smith collapse history to a single commit in infer_runtime_init). Diffing against that fixed commit instead of HEAD keeps the patch correct even when the agent makes its own commits during rollout.
            patch_base = instance.get('runtime_base_commit') or base_commit
            execution_stage = 'task.agent.get_patch'
            set_platform_stage('patch.extract')
            with phase('task.agent.get_patch'):
                patch_text = await get_patch_from_runtime(
                    infer_runtime,
                    working_dir=instance_workdir,
                    language=instance_language,
                    base_commit=patch_base,
                    extra_ignore_patterns=task_input.patch_exclude_patterns,
                )
            is_patch_from_base_commit = True
            execution_stage = 'task.agent.release_runtime'
            await ctx.runtime_provider.release(name=infer_runtime_name)
            # Remove the context reference after releasing the runtime.
            ctx._runtimes.pop(infer_runtime_name, None)
            if task_input.skip_infer_patch_eval:
                return SwebenchTaskOutput(
                    score=0.0,
                    completed=True,
                    resolved=False,
                    patch_text=patch_text or '',
                    error='Skipping inference patch evaluation since skip_infer_patch_eval is set to true.',
                    extra_fields=reward_hack_extra,
                )
            if not patch_text and task_input.skip_empty_patch_eval:
                error_msg = 'Agent did not produce any changes in the working directory. Skipping evaluation.'
                logger.info(error_msg)
                return SwebenchTaskOutput(
                    score=0.0,
                    completed=True,
                    resolved=False,
                    patch_text=patch_text or '',
                    error=error_msg,
                    extra_fields=reward_hack_extra,
                )

        logger.debug(f'Got patch length: {len(patch_text)}')

        # Retry raised runtime errors; retain every returned grading verdict.
        eval_max_attempts = max(1, task_input.eval_max_attempts)
        result: dict[str, Any] = {}
        for eval_attempt in range(1, eval_max_attempts + 1):
            if eval_attempt > 1:
                logger.debug('Eval attempt %d/%d for %s', eval_attempt, eval_max_attempts, task_input.instance_id)

            execution_stage = 'task.scoring'
            set_platform_stage('judge.create_runtime')
            phase('task.scoring')
            ctx.runtime_provider.configure(eval_runtime_name, **configure_params)
            try:
                try:
                    async with async_timeout(request_timeout):
                        execution_stage = 'task.scoring.get_runtime'
                        with phase('task.scoring.get_runtime'):
                            rt = await ctx.get_runtime(eval_runtime_name)
                        if hasattr(rt, 'config') and hasattr(rt.config, 'base_url'):
                            eval_runtime_id = ctx.get_runtime_id(eval_runtime_name) or 'unknown'
                            span_id = get_current_span_id()
                            if span_id:
                                update_span(span_id, attributes={'hytuner.eval_runtime_id': eval_runtime_id})
                except (asyncio.TimeoutError, TimeoutError) as e:
                    raise TimeoutError(f'Failed to get eval_runtime timeout: {e}') from e
                except Exception as e:
                    raise RuntimeError(f'Failed to get eval_runtime: {e}') from e

                span_id = get_current_span_id()

                if is_patch_from_base_commit and base_commit:
                    # Restore the base revision and normalize checkout errors for grading retries.
                    try:
                        execution_stage = 'task.scoring.checkout'
                        checkout_result = RunCommandResponse.model_validate(
                            await rt.call_executor(
                                'run_command', {'command': 'git checkout ' + base_commit, 'cwd': instance_workdir}
                            )
                        )
                        checkout_details = (
                            f'Checkout command status={checkout_result.status.value}, '
                            f'return_code={checkout_result.return_code}\n'
                            f'stdout:\n{checkout_result.stdout or ""}\nstderr:\n{checkout_result.stderr or ""}'
                        )
                        if checkout_result.status == CommandRunStatus.TimeLimitExceeded:
                            raise TimeoutError(checkout_details)
                        if checkout_result.status != CommandRunStatus.Finished or checkout_result.return_code != 0:
                            raise RuntimeError(checkout_details)
                    except (asyncio.TimeoutError, TimeoutError) as e:
                        raise TimeoutError(f'Failed to checkout base_commit timeout: {e}') from e
                    except Exception as e:
                        raise RuntimeError(f'Failed to checkout base_commit: {e}') from e
                try:
                    execution_stage = 'task.scoring.runtime_init'
                    set_platform_stage('judge.initialize')
                    with phase('task.scoring.runtime_init'):
                        await dataset_handler.eval_runtime_init(instance, rt)
                    logger.debug('eval_runtime_init')
                except Exception as e:
                    raise RuntimeError(f'Failed to eval_runtime_init: {e}') from e

                start_time = time.time()
                try:
                    execution_stage = 'task.scoring.grade'
                    set_platform_stage('judge.evaluate')
                    with phase('task.scoring.grade'):
                        async with async_timeout(task_input.eval_timeout_seconds):
                            result = await evaluate_and_grade_in_runtime(
                                rt,
                                patch_text or '',
                                instance,
                                task_input.instance_id,
                                task_input.eval_timeout_seconds,
                                include_tests_status=True,
                                reverse_patch=reverse_patch,
                            )
                except (asyncio.TimeoutError, TimeoutError) as e:
                    reward_time = time.time() - start_time
                    raise TimeoutError(
                        f'Failed to evaluate_and_grade_in_runtime timeout: {e}, infer_runtime_id:{runtime_id}'
                    ) from e
                except Exception as e:
                    reward_time = time.time() - start_time
                    raise RuntimeError(
                        f'Failed to evaluate_and_grade_in_runtime: {e}, infer_runtime_id:{runtime_id}'
                    ) from e
                reward_time = time.time() - start_time

                if span_id and reward_time > 0:
                    update_span(span_id, attributes={'hytuner.reward_time': reward_time})

                execution_stage = 'task.scoring.release_runtime'
                await ctx.runtime_provider.release(name=eval_runtime_name)
                ctx._runtimes.pop(eval_runtime_name, None)

                # Returned verdicts and structured errors stop here; only raised runtime errors are retried.
                if result.get('error'):
                    logger.warning(
                        'Eval returned error on attempt %d/%d for %s (error_code=%s): %s',
                        eval_attempt,
                        eval_max_attempts,
                        task_input.instance_id,
                        result.get('error_code'),
                        result['error'],
                    )
                elif result.get('resolved', False):
                    logger.info(
                        'Eval resolved on attempt %d/%d for %s', eval_attempt, eval_max_attempts, task_input.instance_id
                    )
                else:
                    logger.info(
                        'Eval completed on attempt %d/%d, not resolved (score=%.2f, error_code=%s) for %s '
                        '— verdict, not a failure; no retry',
                        eval_attempt,
                        eval_max_attempts,
                        float(result.get('score', 0.0)),
                        result.get('error_code'),
                        task_input.instance_id,
                    )
                break

            except (TimeoutError, RuntimeError) as e:
                with contextlib.suppress(Exception):
                    await ctx.runtime_provider.release(name=eval_runtime_name)
                ctx._runtimes.pop(eval_runtime_name, None)
                if eval_attempt < eval_max_attempts:
                    logger.warning(
                        'Eval attempt %d/%d failed with %s for %s, retrying: %s',
                        eval_attempt,
                        eval_max_attempts,
                        type(e).__name__,
                        task_input.instance_id,
                        e,
                    )
                    continue
                raise

        execution_stage = 'task.result'
        logger.debug(f'Evaluation result: {result}')

        error = result.get('error')
        resolved = bool(result.get('resolved', False))
        if error:
            logger.warning('SWE-bench evaluation failed: %s', error)
        else:
            logger.info('SWE-bench evaluation completed. Resolved=%s', resolved)

        score = float(result.get('score', 0.0))
        if task_input.use_partial_score and not resolved and not error:
            report = json.loads(result.get('report_json', '{}'))
            partial = _compute_partial_score(report.get(task_input.instance_id, {}))
            if partial > 0:
                score = partial
                logger.info('Using partial score %.4f for %s', score, task_input.instance_id)

        # Record violation evidence from the complete chat trajectory.
        if task_input.enable_reward_hacking_detection:
            try:
                processor = get_processor(ChatTrackerProcessor)
                if processor is None:
                    # Detection was explicitly enabled but the chat tracker it reads from is
                    # not configured — surface the misconfiguration rather than silently skip.
                    logger.warning(
                        'reward-hacking detection enabled but no ChatTrackerProcessor is registered; '
                        'add the chat_tracker processor to the run config to enable detection'
                    )
                else:
                    messages = _chat_messages(processor.snapshot_result() or {})
                    if messages:
                        reward_hack_extra = (
                            detect_reward_hacking(
                                messages,
                                task_input.instance_id,
                                instance.get('repo'),
                                patch_text,
                                score,
                                # Handlers that rebuild history (SWE-Smith / SWE-Cosmos) leave the
                                # dataset's base_commit unreachable in the inference runtime, so the
                                # anchor has to be the commit the agent actually started from —
                                # the same base patch extraction diffs against.
                                base_commit=instance.get('runtime_base_commit') or base_commit,
                                history_truncated=instance.get('history_truncated'),
                            )
                            or reward_hack_extra
                        )
            except Exception as exc:
                logger.warning('reward-hacking detection skipped (fail-open): %s', exc)

            # detect_reward_hacking hands back a fresh dict, so the mirror has to run after it,
            # and outside the try so a fail-open skip still reports its all-clear default.
            _mirror_hack_flags(reward_hack_extra)

        # Output quality monitoring: writes the codeswitch / bad_pattern / str_replace keys the
        # quality penalties below consume. Off by default, which leaves those keys absent.
        if task_input.enable_output_monitoring:
            try:
                _processor = get_processor(ChatTrackerProcessor)
                if _processor is not None:
                    _msgs = _chat_messages(_processor.snapshot_result() or {})
                    if _msgs:
                        reward_hack_extra.update(run_swe_monitors(_msgs))
            except Exception as _exc:
                logger.warning('output monitoring skipped (fail-open): %s', _exc)

        # Apply violation and quality penalties independently of the monitoring switch.
        try:
            from nexus.extensions.tasks.swe.reward_hacking.penalties import apply_swe_penalties

            penalized_score, penalty_term = apply_swe_penalties(
                score,
                reward_hack_extra,
                term_reason,
                codeswitch_penalty=task_input.codeswitch_penalty,
                codeswitch_penalty_reward_threshold=task_input.codeswitch_penalty_reward_threshold,
                output_format_bad_pattern_penalty=task_input.output_format_bad_pattern_penalty,
                output_format_bad_pattern_reward_threshold=task_input.output_format_bad_pattern_reward_threshold,
                str_replace_old_eq_new_penalty=task_input.str_replace_old_eq_new_penalty,
                str_replace_old_eq_new_reward_threshold=task_input.str_replace_old_eq_new_reward_threshold,
                let_me_loop_penalty=task_input.let_me_loop_penalty,
                let_me_loop_reward_threshold=task_input.let_me_loop_reward_threshold,
                enable_anti_hack=task_input.enable_anti_hack,
                hack_reward_override=(None if task_input.is_validation else task_input.hack_reward_override),
            )
            if penalized_score is not None:
                score = penalized_score
            if penalty_term:
                if penalty_term != term_reason and span_id:
                    update_span(span_id, attributes={'hytuner.termination_reason': penalty_term})
                term_reason = penalty_term
        except Exception as _exc:
            logger.warning('penalty application skipped (fail-open): %s', _exc)

        phase()  # close remaining phase spans

        return SwebenchTaskOutput(
            score=score,
            completed=bool(result.get('completed', False)),
            resolved=resolved,
            patch_text=patch_text or '',
            report_json=result.get('report_json', ''),
            test_output=result.get('test_output', ''),
            apply_log=result.get('apply_log', ''),
            error=error,
            error_code=result.get('error_code'),
            termination_reason=term_reason,
            extra_fields=reward_hack_extra,
        )
    except Exception as e:
        if isinstance(platform_state, dict):
            platform_state['failure'] = exception_details(e)
            platform_state['patch_text'] = patch_text or ''
            sandbox = platform_state['failure'].get('sandbox')
            if (
                platform_state.get('stage') == 'agent'
                and isinstance(sandbox, dict)
                and sandbox_failure_status(sandbox, during_tool=True) == PlatformExitStatus.RUNTIME_ERROR_BY_AGENT
            ):
                # Also cover a runtime exception escaping run_agent without an
                # agent output. Do not let raise_on_runtime_error retry this trial.
                return await finish_agent_runtime_failure(sandbox, f'{type(e).__name__}: {e}')
        err_str = str(e)
        # Structured failure evidence takes precedence over exception names in stderr.
        is_agent_infra_error = isinstance(e, _AgentInfrastructureError)
        if not is_agent_infra_error and (isinstance(e, UnicodeDecodeError) or 'UnicodeDecodeError' in err_str):
            error_type = 'UnicodeDecodeError'
            term_reason = 'PATCH_ERROR'
        elif not is_agent_infra_error and (isinstance(e, TimeoutError) or 'TimeoutError' in err_str):
            error_type = 'TimeoutError'
            term_reason = 'TASK_TIMEOUT'
        elif isinstance(e, RuntimeError) or 'RuntimeError' in err_str:
            if task_input.raise_on_runtime_error:
                # Infra error (sandbox crash, proxy failure) — re-raise for retry
                logger.error(f'RuntimeError during task execution: {e}', exc_info=True)
                with contextlib.suppress(Exception):
                    await ctx.runtime_provider.release(name=infer_runtime_name)
                # Drop the runtime reference even when release fails.
                ctx._runtimes.pop(infer_runtime_name, None)
                with contextlib.suppress(Exception):
                    await ctx.runtime_provider.release(name=eval_runtime_name)
                ctx._runtimes.pop(eval_runtime_name, None)
                raise
            error_type = 'RuntimeError'
            # A runtime can die after setup; the exception class alone does not identify initialization.
            if isinstance(e, _AgentInfrastructureError):
                term_reason = e.termination_reason
            else:
                term_reason = 'INITIALIZATION_ERROR' if execution_stage == 'task.init' else 'ENV_ERROR'
        else:
            # Re-raise non-skippable errors if any (though here we mostly want to skip to keep training alive)
            logger.error(f'Unexpected error encountered during task execution: {e}', exc_info=True)
            term_reason = 'UNHANDLED_ERROR'
            error_type = type(e).__name__

        error_msg = f'{error_type} encountered during task execution: {e}. Skipping this sample.'
        logger.error('%s Stage: %s.', error_msg, execution_stage)

        # Update span with the specific termination reason for HyTuner to skip training
        span_id = get_current_span_id()
        if span_id:
            update_span(
                span_id,
                attributes={'hytuner.termination_reason': term_reason, 'swe.error_stage': execution_stage},
            )
            if reward_time > 0:
                update_span(span_id, attributes={'hytuner.reward_time': reward_time})
        # Release runtimes if they were acquired
        with contextlib.suppress(Exception):
            await ctx.runtime_provider.release(name=infer_runtime_name)
        ctx._runtimes.pop(infer_runtime_name, None)
        with contextlib.suppress(Exception):
            await ctx.runtime_provider.release(name=eval_runtime_name)
        ctx._runtimes.pop(eval_runtime_name, None)
        return SwebenchTaskOutput(
            score=0.0,
            completed=False,
            resolved=False,
            patch_text=patch_text or '',
            error=error_msg,
            termination_reason=term_reason,
            extra_fields={**reward_hack_extra, 'error_stage': execution_stage},
        )


__all__ = [
    'SwebenchTaskInput',
    'SwebenchTaskOutput',
    'run_swe_task',
]
