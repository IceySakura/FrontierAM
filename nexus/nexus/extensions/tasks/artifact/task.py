"""Deprecated host-side artifact task.

.. deprecated::
    Use harbor_entry through harbor_bridge_task. This entrypoint is retained
    for compatibility and is no longer validated with judge changes.
"""

import contextlib
import json
import logging
import os

from typing import Any

from pydantic import Field, field_validator

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.tasks.artifact import config
from nexus.extensions.tasks.artifact.agent_setup import cos_io
from nexus.extensions.tasks.artifact.judges import (
    JudgeResult,
)
from nexus.extensions.tasks.artifact.judges.capabilities.agent import (
    _provision_runtime,
    release_runtime,
)
from nexus.extensions.tasks.artifact.judges.capabilities.deploy import (
    archive_to_code_content_dict,
    collect_project_archive,
    resolve_screenshot_save_dir,
)
from nexus.extensions.tasks.artifact.reward_protocol import (
    PENALTY_MISSING,
    PENALTY_UNDEPLOYABLE,
    Judge,
    archive_size_bytes,
    build_reward_extra_info,
    status_value,
    term_reason_for,
)
from nexus.extensions.tasks.swe.infer import run_pre_agent_script
from nexus.trace import get_current_span_id, processor_context, update_span
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor
from nexus.utils.reward_score import ReasoningReward


logger = logging.getLogger(__name__)

# Optional result persistence alongside judge evidence.
JUDGE_RESULTS_FILENAME = 'judge_results.json'

# Registered strategies declare deployment and screenshot requirements.


def _persist_judge_results(
    judge_results: list['JudgeResult'],
    instance_id: str,
    final_score: float,
    save_dir: str | None,
    trajectory_id: str | None = None,
) -> None:
    """Dump full judge results to disk alongside screenshots.

    Falls back to $ARTIFACT_SAVE_SCREENSHOTS_DIR/{trajectory_id or instance_id}
    when save_dir is None. Silently skips if no output directory is configured.
    """
    if not save_dir:
        root = config.get().obs.save_screenshots_dir
        if not root:
            return
        save_dir = os.path.join(root, trajectory_id or instance_id or 'unknown')

    try:
        os.makedirs(save_dir, exist_ok=True)
        payload = {
            'instance_id': instance_id,
            'final_score': final_score,
            'judges': [r.model_dump() for r in judge_results],
        }
        filepath = os.path.join(save_dir, JUDGE_RESULTS_FILENAME)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info('[%s] Saved judge results to %s', instance_id, filepath)
    except Exception as e:
        logger.warning('[%s] Failed to persist judge results: %s', instance_id, e)


ArtifactInstance = dict[str, Any]


class ArtifactTaskInput(TaskRunInput, WithAgentRunner):
    """Input for the artifact task."""

    # ── Instance identification ──────────────────────────────────────────────
    instance_id: str
    """Unique identifier for this artifact instance."""

    dataset_path: str = ''
    """Path to a local JSONL/JSON dataset file."""

    instance: ArtifactInstance | None = None
    """Pre-loaded instance dict (takes priority over dataset_path lookup)."""

    # ── Runtime configuration ────────────────────────────────────────────────
    infer_runtime_name: str = 'main'

    image: str = ''
    """Docker image for both infer and eval runtimes."""

    image_override: str | None = None
    """Override image (takes priority over `image`)."""

    working_dir: str = '/workspace'
    """Working directory inside the runtime."""

    # ── Infer phase ──────────────────────────────────────────────────────────
    problem_statement: str = ''
    """The task prompt given to the agent during inference."""

    pre_agent_script: list[str] | None = None
    """Shell commands to run inside the infer runtime before the agent starts."""

    skip_infer: bool = False
    """Skip the infer phase entirely (use a pre-existing archive via ``archive_b64``)."""

    archive_b64: str = ''
    """External base64 tar.gz to evaluate when ``skip_infer=True`` (e.g. from a previous infer DB)."""

    skip_eval: bool = False
    """Skip the eval phase entirely (only run infer, store archive for later eval)."""

    # ── Eval phase ───────────────────────────────────────────────────────────
    setup_script: str = ''
    """Shell command passed through to judges (e.g. start web server before evaluation)."""

    judge_name: str = ''
    """Which registered judge scores this row — **the only judge entry**.

    Every row (dataset + RL) must set this. The named judge (under
    ``judges/strategies/``) decides which sub-judges run, their order /
    concurrency and how the scores aggregate, and returns a single
    :class:`JudgeResult` — e.g. ``aesthetic_general`` gates on the aesthetic
    score before spending an agent run on ``general_agent``. A missing /
    unknown name is an eval-config fault → maskout.
    """

    @field_validator('judge_name', mode='before')
    @classmethod
    def _accept_null_judge_name(cls, value: Any) -> Any:
        """``None`` 当「没写」，与键缺失行为一致。

        数据行里显式写 ``null`` 时 ``.get(k, '')`` 返回的是 ``None`` 而不是默认值。
        字段声明成 ``str`` 会在这里就 ValidationError，整行到不了
        ``resolve_judge_name`` —— 而那个函数会给出一条说得清的错误。
        """
        return '' if value is None else value


class ArtifactTaskOutput(TaskRunOutput, WithScore):
    """Output from the artifact task."""

    completed: bool
    resolved: bool

    archive_b64: str = ''
    """Base64-encoded tar.gz of the working directory (the infer deliverable)."""

    file_list: list[str] = Field(default_factory=list)
    """Relative file paths contained in the archive."""

    archive_cos_url: str = ''
    """``cos://`` URL of the uploaded archive, or '' when upload is off or failed.

    Backup only: eval reads ``archive_b64`` from memory in the same call, so nothing in
    the RL path depends on this. It exists so a rollout's deliverable can be fetched back
    later (re-judging, debugging a score) without re-running infer.
    """

    judge_results: list[dict[str, Any]] = Field(default_factory=list)
    """Per-judge JudgeResult dicts for downstream inspection."""

    infer_trajectories: dict[str, Any] = Field(default_factory=dict)
    """LLM call trajectories from the infer phase only."""

    extra_fields: dict[str, Any] = Field(default_factory=dict)
    """Fields propagated to HyTuner reward_extra_info (appears in traj jsonl).

    Carries ``reward_extra_info`` (training_action / judge_code / error_code /
    extra_reward_metrics from ``build_judge_metrics`` / artifact stats). HyTuner's
    ``runner/hytuner.py`` merges this into the trajectory's extra_fields.
    """

    infer_metadata: dict[str, Any] = Field(default_factory=dict)
    """Diagnostic metadata from the infer phase (agent status, artifact stats):
    agent_exit_status / agent_termination_reason / agent_execution_status /
    artifact_collect_status / artifact_file_count / artifact_archive_bytes. Feeds
    the empty/incomplete-artifact short-circuit and the RL reward_extra_info.
    """

    error: str | None = None
    termination_reason: str | None = None

    zero_reason: str | None = None
    """Internal outcome diagnosis, independent of the HyTuner maskout signal.

    Main families: ``ARTIFACT_*`` model faults, ``EMPTY:<cause>`` /
    ``INCOMPLETE:<cause>`` unfinished rollouts, ``EVAL_*`` evaluator faults,
    ``EVAL_TASK_ERROR:<reason>`` task-level failures, and ``ALL_ZERO`` for a
    legitimate all-zero judge result. Usually ``None`` when score > 0.
    """


# ── Helpers ────────────────────────────────────────────────────────────────────


async def _maybe_upload_archive_to_cos(archive_b64: str, *, instance_id: str) -> str:
    """Back up the deliverable to COS without affecting evaluation.

    FRONTEND_UPLOAD_TO_COS=0 disables the upload.
    """
    if not archive_b64 or not cos_io.is_upload_enabled():
        return ''
    try:
        key = cos_io.build_archive_key(
            instance_id=instance_id,
            archive_name='workspace.tar.gz',
            span_id=get_current_span_id() or '',
        )
        url = await cos_io.upload_archive_b64_async(key, archive_b64)
        logger.info('[%s] Archive backed up to %s (%.1f KB)', instance_id, url, len(archive_b64) * 3 / 4 / 1024)
        return url
    except Exception as exc:
        logger.warning('[%s] COS backup failed (%s); archive still passed in memory', instance_id, exc)
        return ''


def _infer_termination_for_installed_agent(agent_output: Any) -> str:
    """Infer termination from explicit timeout or max-turn signals, not exit_code alone."""
    stderr = str(getattr(agent_output, 'stderr', '') or '')
    if '[timeout]' in stderr:
        return 'TIMEOUT'
    if getattr(agent_output, 'reached_max_turns', False):
        return 'MAX_STEPS'
    return ''


def _agent_status(agent_output: Any) -> dict[str, str]:
    """Normalize agent-specific status fields into shared outcome metadata."""
    exit_status = status_value(getattr(agent_output, 'exit_status', None))
    execution_status = status_value(getattr(agent_output, 'execution_status', None))
    if not exit_status:
        success = getattr(agent_output, 'success', None)
        code = getattr(agent_output, 'exit_code', None)
        if success is not None:
            exit_status = 'success' if success else 'failed'
        if not execution_status and code is not None:
            execution_status = f'exit_code={code}'
    termination_reason = status_value(getattr(agent_output, 'termination_reason', None))
    if not termination_reason:
        termination_reason = _infer_termination_for_installed_agent(agent_output)
    return {
        'agent_exit_status': exit_status,
        'agent_termination_reason': termination_reason,
        'agent_execution_status': execution_status,
    }


async def _load_instance(task_input: ArtifactTaskInput) -> ArtifactInstance:
    """Load instance dict from dataset_path or return the pre-loaded one."""
    if task_input.instance is not None:
        return task_input.instance

    if not task_input.dataset_path:
        # No dataset — build a minimal instance from task input fields
        return {
            'instance_id': task_input.instance_id,
            'problem_statement': task_input.problem_statement,
            'working_dir': task_input.working_dir,
            'image': task_input.image,
        }

    from pathlib import Path

    import aiofiles
    import aiofiles.os

    path = Path(task_input.dataset_path)
    if not await aiofiles.os.path.exists(path):
        raise FileNotFoundError(f'Dataset file not found: {task_input.dataset_path}')

    if path.suffix == '.jsonl':
        async with aiofiles.open(path, encoding='utf-8', errors='replace') as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get('instance_id') == task_input.instance_id:
                    return data
    else:
        async with aiofiles.open(path, encoding='utf-8', errors='replace') as f:
            text = await f.read()
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get('instance_id') == task_input.instance_id:
                    return item
        elif isinstance(data, dict) and data.get('instance_id') == task_input.instance_id:
            return data

    raise ValueError(f'Instance {task_input.instance_id!r} not found in {task_input.dataset_path}')


def _rl_extra_fields(
    judge_code: int,
    *,
    term_reason: str = '',
    infer_metadata: dict[str, Any] | None = None,
    weighted_scores: dict[str, float] | None = None,
    empty_cause: str = '',
    incomplete_cause: str = '',
) -> dict[str, Any]:
    """Build ``{'reward_extra_info': ...}`` for the RL training contract.

    ``build_reward_extra_info`` derives ``training_action`` (use_score / penalty /
    maskout) from ``judge_code`` and attaches ``extra_reward_metrics`` (the 0/1
    ``build_judge_metrics`` indicators). HyTuner reads this off ``extra_fields``.
    """
    judge_response: dict[str, Any] = {}
    if weighted_scores:
        judge_response['weighted_scores'] = weighted_scores
    return {
        'reward_extra_info': build_reward_extra_info(
            request_id='',
            judge_code=judge_code,
            judge_response=judge_response,
            term_reason=term_reason,
            artifact_empty_cause=empty_cause,
            artifact_incomplete_cause=incomplete_cause,
            infer_metadata=infer_metadata or {},
        )
    }


def _classify_zero_reason(zero_reason: str | None) -> tuple[int, str]:
    """Map diagnostic reasons to reward codes and termination reasons.

    Clean zero scores remain trainable; artifact faults are penalties and evaluator
    faults are masked out.
    """
    if not zero_reason or zero_reason == 'ALL_ZERO':
        return Judge.SUCCESS, term_reason_for(Judge.SUCCESS)
    # Artifact penalties remain trainable with no termination reason.
    if zero_reason == 'ARTIFACT_UNDEPLOYABLE':
        return Judge.ARTIFACT_UNDEPLOYABLE, term_reason_for(Judge.ARTIFACT_UNDEPLOYABLE)
    if zero_reason == 'ARTIFACT_MISSING':
        return Judge.ARTIFACT_MISSING, term_reason_for(Judge.ARTIFACT_MISSING)
    # Unfinished rollouts are masked out.
    if zero_reason.startswith(('EMPTY:', 'INCOMPLETE:')):
        return Judge.ROLLOUT_UNFINISHED, term_reason_for(Judge.ROLLOUT_UNFINISHED)
    # Evaluator and infrastructure failures.
    if zero_reason == 'EVAL_DEPLOY_SANDBOX_UNAVAILABLE':
        code = Judge.EVAL_DEPLOY_SANDBOX_UNAVAILABLE
    elif zero_reason in ('EVAL_VLM_CALL_FAILED', 'EVAL_PARSE_FAILED'):
        code = Judge.EVAL_VLM_CALL_FAILED
    elif zero_reason.startswith(('EVAL_ERROR:', 'EVAL_NO_SANDBOX')):
        code = Judge.EVAL_AGENT_CALL_FAILED
    elif zero_reason in ('EVAL_NO_ARTIFACT', 'EVAL_NO_PIPELINE') or zero_reason.startswith('EVAL_UNKNOWN_PIPELINE:'):
        code = Judge.EVAL_CONFIG_FAILED
    elif zero_reason == 'EVAL_TASK_ERROR:TASK_TIMEOUT':
        code = Judge.NEXUS_TASK_TIMEOUT
    elif zero_reason.startswith('EVAL_TASK_ERROR:'):
        code = Judge.NEXUS_TASK_FAILED
    else:
        code = Judge.EVAL_UNKNOWN_FAILED
    return code, term_reason_for(code)


def _maskout_output(
    task_input: ArtifactTaskInput,
    archive_b64: str,
    file_list: list[str],
    infer_metadata: dict[str, Any],
    *,
    archive_cos_url: str = '',
    judge_code: int,
    term_reason: str,
    message: str,
    zero_reason: str,
    empty_cause: str = '',
    incomplete_cause: str = '',
) -> 'ArtifactTaskOutput':
    """Skip judges and emit a ``JUDGE_*`` maskout (score below THRESHOLD: no gradient,
    and excluded from the GRPO group baseline).
    """
    logger.info('[%s] Skipping judges (%s): %s', task_input.instance_id, term_reason, message)
    span_id = get_current_span_id()
    if span_id:
        update_span(span_id, attributes={'hytuner.termination_reason': term_reason})
    return ArtifactTaskOutput(
        # Use ReasoningReward.THRESHOLD to exclude maskout samples from the GRPO baseline.
        score=float(ReasoningReward.THRESHOLD.value),
        completed=False,
        resolved=False,
        archive_b64=archive_b64,
        archive_cos_url=archive_cos_url,
        file_list=file_list,
        infer_metadata=infer_metadata,
        extra_fields=_rl_extra_fields(
            judge_code,
            term_reason=term_reason,
            infer_metadata=infer_metadata,
            empty_cause=empty_cause,
            incomplete_cause=incomplete_cause,
        ),
        error=message,
        termination_reason=term_reason,
        zero_reason=zero_reason,
    )


def _model_penalty_output(
    task_input: ArtifactTaskInput,
    archive_b64: str,
    file_list: list[str],
    infer_metadata: dict[str, Any],
    *,
    judge_code: int,
    score: float,
    message: str,
    zero_reason: str,
    empty_cause: str = '',
    archive_cos_url: str = '',
) -> 'ArtifactTaskOutput':
    """Emit a trainable penalty with output and span termination reasons unset."""
    assert Judge(judge_code).action == 'penalty', f'{Judge(judge_code).name} is not a penalty code'
    logger.info('[%s] %s (penalty=%.2f): %s', task_input.instance_id, Judge(judge_code).name, score, message)
    return ArtifactTaskOutput(
        score=float(score),
        completed=True,
        resolved=False,
        archive_b64=archive_b64,
        archive_cos_url=archive_cos_url,
        file_list=file_list,
        infer_metadata=infer_metadata,
        extra_fields=_rl_extra_fields(judge_code, infer_metadata=infer_metadata, empty_cause=empty_cause),
        error=message,
        zero_reason=zero_reason,
    )


def _zero_reason_from_results(judge_results: list['JudgeResult']) -> str | None:
    """Map the first structured failure to a diagnostic reason; clean zeros yield ALL_ZERO."""
    if not judge_results:
        return 'ALL_ZERO'
    kinds = [r.failure_kind for r in judge_results if r.failure_kind]
    errors = [r.error for r in judge_results if r.error]
    if kinds:
        kind = kinds[0]
        if kind in ('vlm_failed', 'parse_failed'):
            return 'EVAL_VLM_CALL_FAILED' if kind == 'vlm_failed' else 'EVAL_PARSE_FAILED'
        # Keep sandbox infrastructure failures separately observable.
        if kind == 'no_sandbox':
            return 'EVAL_DEPLOY_SANDBOX_UNAVAILABLE'
        # Preserve TIMEOUT in the reason for judge/eval_timeout metrics.
        if kind == 'eval_timeout':
            return 'EVAL_TIMEOUT'
        return f'EVAL_ERROR:{kind}'
    if errors:
        # Rarely reached: JudgeResult's validator enforces error ⇒ failure_kind, so
        # ``kinds`` is usually non-empty above. Belt-and-suspenders.
        return 'EVAL_ERROR:' + errors[0][:80]
    return 'ALL_ZERO'


def resolve_eval_outcome(judge_result: 'JudgeResult') -> tuple[float, bool, str | None]:
    """Translate the top-level verdict into score, resolved status, and diagnostic reason.

    Failures precede artifact penalties and scores. Child results are diagnostic only;
    a clean zero remains trainable.
    """
    final_score = judge_result.score

    if judge_result.failure_kind:
        # Failed evaluations cannot report a resolved task.
        return final_score, False, _zero_reason_from_results([judge_result])

    resolved = judge_result.passed if judge_result.passed is not None else final_score >= 1.0

    # Only the top-level verdict may assign a deployment penalty.
    if judge_result.metadata.get('deploy_failed'):
        return PENALTY_UNDEPLOYABLE, resolved, 'ARTIFACT_UNDEPLOYABLE'

    if final_score == 0.0:
        # A clean top-level zero remains trainable.
        return final_score, resolved, _zero_reason_from_results([judge_result])

    return final_score, resolved, None


def _finalize_eval(
    task_input: ArtifactTaskInput,
    *,
    judge_results: list['JudgeResult'],
    final_score: float,
    resolved: bool,
    zero_reason: str | None,
    archive_b64: str,
    archive_cos_url: str,
    file_list: list[str],
    infer_trajectories: dict[str, Any],
    infer_metadata: dict[str, Any],
    save_dir: str | None,
    trajectory_id: str | None,
) -> 'ArtifactTaskOutput':
    """Persist judge results and convert the verdict to the task reward contract."""
    _persist_judge_results(judge_results, task_input.instance_id, final_score, save_dir, trajectory_id=trajectory_id)

    judge_code, eval_term_reason = _classify_zero_reason(zero_reason)

    # Maskout uses ReasoningReward.THRESHOLD; preserve all other scores.
    try:
        _is_maskout_verdict = Judge(judge_code).action == 'maskout'
    except ValueError:
        logger.warning('[%s] unknown judge_code=%r, keeping score as-is', task_input.instance_id, judge_code)
        _is_maskout_verdict = False
    if _is_maskout_verdict:
        final_score = float(ReasoningReward.THRESHOLD.value)

    weighted_scores = {r.judge_name: r.score for r in judge_results}
    extra_fields = _rl_extra_fields(
        judge_code,
        term_reason=eval_term_reason,
        infer_metadata=infer_metadata,
        weighted_scores=weighted_scores,
    )
    if eval_term_reason:
        span_id = get_current_span_id()
        if span_id:
            update_span(span_id, attributes={'hytuner.termination_reason': eval_term_reason})

    return ArtifactTaskOutput(
        score=final_score,
        completed=True,
        resolved=resolved,
        archive_b64=archive_b64,
        archive_cos_url=archive_cos_url,
        file_list=file_list,
        judge_results=[r.model_dump() for r in judge_results],
        infer_trajectories=infer_trajectories,
        infer_metadata=infer_metadata,
        extra_fields=extra_fields,
        termination_reason=eval_term_reason or None,
        zero_reason=zero_reason,
    )


def _archive_short_circuit(
    task_input: ArtifactTaskInput,
    instance: ArtifactInstance,
    archive_b64: str,
    file_list: list[str],
    infer_metadata: dict[str, Any],
    archive_cos_url: str = '',
) -> 'ArtifactTaskOutput | None':
    """Classify empty or incomplete artifacts before provisioning evaluation resources."""
    from nexus.extensions.tasks.artifact.judges.base import JudgeInput
    from nexus.extensions.tasks.artifact.judges.registry import _artifact_precheck

    verdict = _artifact_precheck(
        # The name only labels the verdict; this runs before dispatch and must
        # not depend on a judge being resolved yet.
        str(getattr(task_input, 'judge_name', '') or 'artifact'),
        JudgeInput(
            instance_id=task_input.instance_id,
            problem_statement='',
            openai_clients={},
            archive_b64=archive_b64,
            instance={**instance, 'file_list': file_list, 'infer_metadata': infer_metadata},
        ),
    )
    if verdict is None:
        return None

    meta = dict(verdict.metadata or {})
    cause = meta.get('empty_cause') or 'unknown'

    # ``agent_failed`` here always means "the rollout did not finish" — the
    # precheck only uses that kind for timeout / max-steps / truncation /
    # collect-failure, never for a judging fault.
    if verdict.failure_kind == 'agent_failed':
        return _maskout_output(
            task_input,
            archive_b64,
            file_list,
            infer_metadata,
            archive_cos_url=archive_cos_url,
            judge_code=Judge.ROLLOUT_UNFINISHED,
            term_reason=term_reason_for(Judge.ROLLOUT_UNFINISHED),
            message=verdict.error or 'rollout did not finish',
            zero_reason=f'EMPTY:{cause}',
            empty_cause=cause,
        )

    return _model_penalty_output(
        task_input,
        archive_b64,
        file_list,
        infer_metadata,
        judge_code=Judge.ARTIFACT_MISSING,
        score=PENALTY_MISSING,
        message=verdict.error or 'agent exited cleanly but wrote no files',
        zero_reason='ARTIFACT_MISSING',
        empty_cause=cause if cause != 'unknown' else 'no_files_written',
        archive_cos_url=archive_cos_url,
    )


@task_registry.register('artifact_task')
async def run_artifact_task(task_input: ArtifactTaskInput) -> ArtifactTaskOutput:
    """Run optional inference, dispatch one registered judge, and finalize the reward."""
    ctx = require_context()
    if not ctx.runtime_provider:
        raise RuntimeError('artifact_task requires a runtime_provider.')

    term_reason = 'ENV_DONE'
    infer_runtime_name = task_input.infer_runtime_name
    # Initialized here so the except block can reference it even if the failure
    # happens before the infer phase populates it.
    infer_metadata: dict[str, Any] = {}

    try:
        # ── Load instance ────────────────────────────────────────────────────
        instance = await _load_instance(task_input)

        image = task_input.image_override or instance.get('image') or task_input.image
        working_dir = instance.get('working_dir') or task_input.working_dir
        problem_statement = task_input.problem_statement or instance.get('problem_statement', '')

        configure_params: dict[str, Any] = {}
        if image:
            configure_params['image'] = image

        # ── Phase 1: Infer ───────────────────────────────────────────────────
        archive_b64 = ''
        archive_cos_url = ''
        file_list: list[str] = []
        infer_trajectories: dict[str, Any] = {}
        infer_metadata = {
            'working_dir': working_dir,
            'image': image,
            'artifact_file_count': 0,
            'artifact_archive_bytes': 0,
        }

        if not task_input.skip_infer:
            logger.debug('[%s] Starting infer phase', task_input.instance_id)
            infer_chat_tracker = ChatTrackerProcessor()
            with processor_context([infer_chat_tracker], finalize_on_exit=True):
                ctx.runtime_provider.configure(infer_runtime_name, **configure_params)
                infer_runtime = await ctx.get_runtime(infer_runtime_name)

                if task_input.pre_agent_script:
                    formatted = [cmd.format(**instance) for cmd in task_input.pre_agent_script]
                    await run_pre_agent_script(infer_runtime, formatted, working_dir=working_dir)

                if not problem_statement:
                    raise ValueError(f'No problem_statement for instance {task_input.instance_id!r}')

                ctx.task_config['problem_statement'] = problem_statement
                ctx.task_config['working_dir'] = working_dir
                ctx.task_config['instance'] = instance
                ctx.tool_state.setdefault('working_dir', working_dir)

                agent_output = await task_input.run_agent(user_prompt=problem_statement)
                infer_metadata.update(_agent_status(agent_output))
                logger.info(
                    '[%s] Agent finished: %s',
                    task_input.instance_id,
                    infer_metadata.get('agent_exit_status') or 'unknown',
                )

                # Collect a tar.gz of the whole working_dir (excludes node_modules/.git/dist).
                try:
                    try:
                        archive_b64, file_list = await collect_project_archive(infer_runtime, working_dir)
                        infer_metadata['artifact_collect_status'] = 'success'
                        infer_metadata['artifact_file_count'] = len(file_list)
                        infer_metadata['artifact_archive_bytes'] = archive_size_bytes(archive_b64)
                    except Exception as collect_err:
                        infer_metadata['artifact_collect_status'] = 'failed'
                        infer_metadata['artifact_collect_error'] = f'{type(collect_err).__name__}: {collect_err}'
                        raise
                finally:
                    await ctx.runtime_provider.release(name=infer_runtime_name)
                logger.info(
                    '[%s] Infer phase done, files=%d, archive_b64_len=%d',
                    task_input.instance_id,
                    len(file_list),
                    len(archive_b64),
                )
                archive_cos_url = await _maybe_upload_archive_to_cos(archive_b64, instance_id=task_input.instance_id)
            infer_trajectories = infer_chat_tracker.get_result()
        else:
            logger.debug('[%s] Skipping infer phase (skip_infer=True)', task_input.instance_id)
            infer_metadata['infer_skipped'] = True
            archive_b64 = task_input.archive_b64
            if archive_b64:
                # External archive supplied via skip_infer: recover file_list so the
                # empty/incomplete short-circuit and judges see the real file set.
                try:
                    code_dict, _ = archive_to_code_content_dict(archive_b64)
                    file_list = sorted(code_dict.keys())
                    infer_metadata['artifact_file_count'] = len(file_list)
                    infer_metadata['artifact_archive_bytes'] = archive_size_bytes(archive_b64)
                except Exception as decode_err:
                    logger.warning('[%s] Failed to decode external archive_b64: %s', task_input.instance_id, decode_err)

        # ── Dispatch key: validate BEFORE the artifact short-circuit ─────────
        # 判分配置错误与产物好坏无关，所以要在「产物为空 → 提前返回惩罚」之前查。
        # 放在后面的话，一个空产物会先返回产物惩罚，配置错误永远查不到 —— 于是
        # 「忘了写 judge_name」表现成一个看起来正常的产物低分，只在产物非空时才
        # 偶尔暴露。
        #
        # ``skip_eval`` 时不强制判分配置：那条路径只跑 infer 存档，判分名字可以
        # 留空，之后再配。
        if not task_input.skip_eval:
            from nexus.extensions.tasks.artifact.judges.registry import resolve_judge_name as _resolve_name

            try:
                _resolve_name({'judge_name': task_input.judge_name})
            except ValueError as name_err:
                logger.error('[%s] bad judge dispatch config: %s', task_input.instance_id, name_err)
                return _finalize_eval(
                    task_input,
                    judge_results=[
                        JudgeResult(
                            judge_name='',
                            score=0.0,
                            error=str(name_err),
                            failure_kind='config_error',
                        )
                    ],
                    final_score=0.0,
                    resolved=False,
                    zero_reason='EVAL_ERROR:config_error',
                    archive_b64=archive_b64,
                    archive_cos_url=archive_cos_url,
                    file_list=file_list,
                    infer_metadata=infer_metadata,
                    infer_trajectories=infer_trajectories,
                    save_dir=None,
                    trajectory_id=ctx.task_config.get('trajectory_id'),
                )

        # ── Short-circuit: skip judges on empty/incomplete artifact ──────────
        # Saves compute and emits a JUDGE_* maskout signal instead of deploying
        # an empty project.
        if not task_input.skip_eval:
            short_circuit = _archive_short_circuit(
                task_input, instance, archive_b64, file_list, infer_metadata, archive_cos_url=archive_cos_url
            )
            if short_circuit is not None:
                short_circuit.infer_trajectories = infer_trajectories
                return short_circuit

        if task_input.skip_eval:
            logger.debug('[%s] Skipping eval phase (skip_eval=True)', task_input.instance_id)
            return ArtifactTaskOutput(
                score=0.0,
                completed=True,
                resolved=False,
                archive_b64=archive_b64,
                archive_cos_url=archive_cos_url,
                file_list=file_list,
                infer_trajectories=infer_trajectories,
                infer_metadata=infer_metadata,
            )

        # Shared save_dir for screenshots + judge evidence. When global_step is set
        # (RL training), group per-step + per-rollout so n>1 rollouts don't overwrite:
        # $ARTIFACT_SAVE_SCREENSHOTS_DIR/{step}/{trajectory_id}/
        _trajectory_id = ctx.task_config.get('trajectory_id') or task_input.instance_id
        _global_step = ctx.task_config.get('global_step')
        _effective_iid = os.path.join(str(_global_step), _trajectory_id) if _global_step else _trajectory_id
        save_dir = resolve_screenshot_save_dir(None, _effective_iid)

        # ── Eval: the registered judge is the ONLY judge-orchestration entry ──
        # 每行必须给 judge_name（RL 和 dataset 都是）。
        # judge 内部决定跑哪些子 judge、先后/并发、如何聚合，返回单个 JudgeResult。
        # task.py 只负责起容器 + 分派 + 收尾。
        if not archive_b64:
            out = _maskout_output(
                task_input,
                archive_b64,
                file_list,
                infer_metadata,
                archive_cos_url=archive_cos_url,
                judge_code=Judge.EVAL_CONFIG_FAILED,
                term_reason=term_reason_for(Judge.EVAL_CONFIG_FAILED),
                message='no archive reached the eval phase',
                zero_reason='EVAL_NO_ARTIFACT',
            )
            out.infer_trajectories = infer_trajectories
            return out
        from nexus.extensions.tasks.artifact.judges.base import JudgeInput
        from nexus.extensions.tasks.artifact.judges.registry import resolve_judge_name, run_judge

        # 分派键归一化：与其他数据行入口（generator / parquet builder）共用
        # ``resolve_judge_name``，规则只定义一次。这里本该已经被上面的前置校验
        # 拦下，保留是因为 ``skip_eval`` 之外的路径也可能走到。
        try:
            judge_name = resolve_judge_name({'judge_name': task_input.judge_name})
        except ValueError as exc:
            out = _maskout_output(
                task_input,
                archive_b64,
                file_list,
                infer_metadata,
                archive_cos_url=archive_cos_url,
                judge_code=Judge.EVAL_CONFIG_FAILED,
                term_reason=term_reason_for(Judge.EVAL_CONFIG_FAILED),
                message=str(exc),
                zero_reason='EVAL_NO_PIPELINE',
            )
            out.infer_trajectories = infer_trajectories
            return out

        eval_runtime_name = f'eval_{task_input.instance_id}'
        logger.debug('[%s] Starting judge "%s"', task_input.instance_id, judge_name)
        try:
            runtime = await _provision_runtime(eval_runtime_name, image)
            judge_result = await run_judge(
                judge_name,
                JudgeInput(
                    instance_id=task_input.instance_id,
                    problem_statement=problem_statement,
                    openai_clients=ctx.openai_clients,
                    archive_b64=archive_b64,
                    image=image,
                    runtime=runtime,
                    # ``file_list`` + ``infer_metadata`` feed the wrapper's
                    # artifact pre-check (empty deliverable). Both paths get it
                    # from ``run_judge``, so both must supply the same facts.
                    instance={**instance, 'file_list': file_list, 'infer_metadata': infer_metadata},
                    save_dir=save_dir,
                ),
            )
        finally:
            await release_runtime(eval_runtime_name)

        # One judge, one result. Composite judges aggregate internally and
        # expose their legs via ``child_results`` — task.py does not re-derive
        # the score from them (see JudgeResult.child_results).
        judge_results = [judge_result, *judge_result.child_results]
        final_score, resolved, zero_reason = resolve_eval_outcome(judge_result)

        logger.info('[%s] Eval done: score=%.4f resolved=%s', task_input.instance_id, final_score, resolved)

        return _finalize_eval(
            task_input,
            judge_results=judge_results,
            final_score=final_score,
            resolved=resolved,
            zero_reason=zero_reason,
            archive_b64=archive_b64,
            archive_cos_url=archive_cos_url,
            file_list=file_list,
            infer_trajectories=infer_trajectories,
            infer_metadata=infer_metadata,
            save_dir=save_dir,
            trajectory_id=ctx.task_config.get('trajectory_id'),
        )

    except Exception as e:
        err_str = str(e)
        if isinstance(e, TimeoutError) or 'TimeoutError' in err_str:
            term_reason = 'TASK_TIMEOUT'
        elif isinstance(e, RuntimeError):
            term_reason = 'INITIALIZATION_ERROR'
        else:
            term_reason = 'UNHANDLED_ERROR'

        error_msg = f'{type(e).__name__}: {e}'
        logger.exception('[%s] Task failed: %s', task_input.instance_id, error_msg)

        span_id = get_current_span_id()
        if span_id:
            update_span(span_id, attributes={'hytuner.termination_reason': term_reason})

        with contextlib.suppress(Exception):
            await ctx.runtime_provider.release(name=infer_runtime_name)

        # For infrastructure-level failures (e.g. Gongfeng quota exhausted, sandbox
        # creation failed) we re-raise so that single.run reports success=False and
        # the training side can route the trajectory through the None-reward path
        # (handle_nexus_none_reward_score) instead of treating it as a real 0-score
        # rollout. TimeoutError and other task-level errors keep the original
        # behaviour of returning a score=0 ArtifactTaskOutput so existing
        # downstream metrics (TASK_TIMEOUT termination_reason, etc.) still flow.
        if term_reason == 'INITIALIZATION_ERROR':
            raise

        # Task timeout and unhandled task errors are separate maskout verdicts for
        # analysis. Their generic termination reasons already sit in HyTuner's
        # MASKOUT_TERMINATION_REASONS, so they need no JUDGE_ prefix.
        err_judge_code = Judge.NEXUS_TASK_TIMEOUT if term_reason == 'TASK_TIMEOUT' else Judge.NEXUS_TASK_FAILED
        term_reason = term_reason_for(err_judge_code)
        return ArtifactTaskOutput(
            score=float(ReasoningReward.THRESHOLD.value),
            completed=False,
            resolved=False,
            archive_b64=archive_b64,
            archive_cos_url=archive_cos_url,
            file_list=file_list,
            infer_trajectories=infer_trajectories,
            error=error_msg,
            infer_metadata=infer_metadata,
            extra_fields=_rl_extra_fields(
                err_judge_code,
                term_reason=term_reason,
                infer_metadata=infer_metadata,
            ),
            termination_reason=term_reason,
            zero_reason=f'EVAL_TASK_ERROR:{term_reason}',
        )


__all__ = ['ArtifactTaskInput', 'ArtifactTaskOutput', 'run_artifact_task']
