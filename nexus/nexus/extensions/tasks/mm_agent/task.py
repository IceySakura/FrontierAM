import base64
import logging
import mimetypes
import os
import time

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.agents.base.schema import TerminationReason
from nexus.extensions.agents.base.utils import extract_text_from_content
from nexus.extensions.agents.installed_common import InstalledAgentOutput
from nexus.extensions.agents.main.agent import MainAgentOutput, MainExitStatus
from nexus.extensions.tasks.mm_agent.prompts import get_default_system_prompt
from nexus.extensions.tasks.mm_agent.reward_system import vl_reward_system
from nexus.extensions.tasks.mm_agent.utils import TaskArtifact, upload_artifacts
from nexus.extensions.tasks.mm_agent.utils.video_utils import prepare_video_source, remove_staged_video
from nexus.runtime.executors import FileItem
from nexus.trace import get_current_span_id, update_span
from nexus.utils.reward_system import resolve_strip_multimodal_config


logger = logging.getLogger(__name__)

_RUNTIME_IMAGE_DIR = '/tmp'
_FORCE_FINISH_HOOK_PATH = 'nexus.extensions.tasks.mm_agent.hooks::force_finish_hook'

# Invoked by import path rather than imported: the tool module reaches jupyter_client through
# ``runtime.executors.jupyter``, which only the sandbox needs to have installed.
_CLOSE_VIDEOCUT_SESSION = 'nexus.extensions.tools.code_interpreter.jupyter_with_videocut::close_videocut_session'
# Task-local; the tool keys its kernels off ``run_id`` and never reads this.
_VIDEO_STATE_KEY = 'mm_agent.video_source'

# for MD5-suffixed structural tokens like <think:abc123>
_THINK_BEGIN_TOKEN = os.environ.get('THINK_BEGIN_TOKEN', '<think>')
_THINK_END_TOKEN = os.environ.get('THINK_END_TOKEN', '</think>')

# ---------------------------------------------------------------------------
# Reward score constants for abnormal exits
# ---------------------------------------------------------------------------
# ``-100`` matches ``ReasoningReward.UNJUDGED`` in hytuner; samples with a
# score at or below ``ReasoningReward.THRESHOLD = -99`` are masked out by
# ``mask_invalid_samples`` BEFORE advantage computation, so they carry no
# gradient. We use this for environment-side failures the model should not
# be punished for.
_UNJUDGED_SCORE = -100.0

# ``ReasoningReward.THRESHOLD``: at or below this a judge score is unusable, not a training signal.
_UNJUDGED_SCORE_THRESHOLD = -99.0

# ``-300`` matches ``ReasoningReward.TASK_FAILED`` in hytuner: the task execution failed.
_TASK_FAILED_SCORE = -300.0

# ``exit_status`` values that always warrant masking (env-side failures).
_ALWAYS_MASK_STATUSES: set[str] = {
    MainExitStatus.TOOL_EXECUTION_ERROR.value,  # tool exec / network / OOM / output parsing — env-side
    MainExitStatus.INFERENCE_ERROR.value,  # LLM backend errored out
    MainExitStatus.INFERENCE_TIMEOUT.value,  # LLM backend timeout
    MainExitStatus.OBSERVATION_TIMEOUT.value,  # tool execution timeout
    MainExitStatus.RUNTIME_ERROR.value,  # runtime unhealthy / environment failure
}

# ``exit_status`` values that always short-circuit to ``score=0.0`` without
# invoking the reward system. Two flavors live here:
#
# 1. *Model-side protocol violations* — trivially wrong, would only waste
#    judge/LM calls to evaluate:
#      - TOOL_ERROR : malformed tool call / bad args / unknown tool / schema
#                     or model-format violations
#
# 2. *Trajectory cutoffs* — the rollout was interrupted before the
#    model could produce a final answer. We do **not** want to invoke the
#    reward system on a half-finished trajectory (the score would be
#    arbitrary), so we short-circuit to 0.0. HyTuner separately decides whether
#    to train or mask it based on the public termination reason.
#      - MAX_ITERATIONS          : agent ran out of iteration budget
#      - CONTEXT_LENGTH_EXCEEDED : prompt blew past max_model_len mid-run
#      - OUTPUT_LENGTH_EXCEEDED  : model output hit max_tokens mid-response
_ALWAYS_ZERO_STATUSES: set[str] = {
    MainExitStatus.TOOL_ERROR.value,
    MainExitStatus.MAX_ITERATIONS.value,
    MainExitStatus.CONTEXT_LENGTH_EXCEEDED.value,
    MainExitStatus.OUTPUT_LENGTH_EXCEEDED.value,
}

# nexus.agent_name selects the harness: claude_code / codex / hycode (installed CLI) or main_agent (built-in).
# Anything else is rejected in mm_agent_task, never silently graded as a harness. A harness joins this set
# only once it declares meta.gateway.scaffold_policy and its output fits _run_installed_harness's classification.
_BUILTIN_AGENT_NAMES = frozenset({'main_agent', 'main'})
_INSTALLED_HARNESS_NAMES = frozenset({'claude_code', 'codex', 'hycode'})


def _selected_agent_name() -> str:
    """Return the harness chosen via ``nexus.agent_name`` (empty string if unset)."""
    return (require_context().agent_name or '').strip()


def _is_installed_harness(agent_name: str) -> bool:
    """True only for a harness mm_agent supports (claude_code / codex / hycode).

    Allowlist, not ``not in _BUILTIN_AGENT_NAMES``: fail-open would grade other agents' output as an empty prediction.
    """
    return agent_name.strip().lower() in _INSTALLED_HARNESS_NAMES


def extract_final_answer(text: str) -> str:
    """Extract the final answer from model prediction.

    If the think-end token (``</think>`` by default, override via the
    ``THINK_END_TOKEN`` env var) is present, return everything after its
    *last* occurrence. Otherwise return the original text stripped.
    """
    if not text:
        return text

    idx = text.rfind(_THINK_END_TOKEN)
    if idx != -1:
        return text[idx + len(_THINK_END_TOKEN) :].strip()

    return text.strip()


def encode_image(image_source: str) -> str:
    """Encode an image to base64 string.

    Accepts:
    - A file path (reads and base64-encodes the file)
    - A data URI (data:image/...;base64,...) — extracts the base64 portion directly
    - A raw base64 string — returns as-is
    """
    if image_source.startswith('data:image'):
        # Already a data URI, extract the base64 part
        _, base64_data = image_source.split(',', 1)
        return base64_data
    looks_like_path = image_source.startswith(('/', './', '../'))
    if looks_like_path:
        if not os.path.exists(image_source):
            raise FileNotFoundError(
                f'encode_image: local image path does not exist: {image_source!r}. '
                'Refusing to treat a filesystem path as raw base64.'
            )
        with open(image_source, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    if os.path.exists(image_source):
        with open(image_source, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    # Last resort: preserve legacy callers that pass raw base64 strings.
    return image_source


def _is_local_file(image_source: str) -> bool:
    return not image_source.startswith('data:image') and Path(image_source).is_file()


def _guess_image_mime(image_source: str) -> str:
    """Best-effort MIME type for a local image path.

    The policy side can hardcode ``image/jpeg`` because vLLM sniffs the
    decoded bytes, but the reward platform is a separate consumer that may
    well trust the declared type — labelling a PNG as JPEG there risks a
    silently broken image (and therefore a bogus score).
    """
    mime, _ = mimetypes.guess_type(image_source)
    if mime is not None and mime.startswith('image/'):
        return mime
    # Unknown suffix: keep the historical default rather than inventing a
    # type the platform may reject outright.
    return 'image/jpeg'


def _to_reward_image_url(image_source: str) -> str:
    """Normalise one image into something the reward platform can fetch.

    The reward service runs elsewhere, so a local path would silently score
    as a broken image. Data URIs and HTTP(S) URLs pass through; local files
    become data URIs.
    """
    if image_source.startswith(('data:image', 'http://', 'https://')):
        return image_source
    return f'data:{_guess_image_mime(image_source)};base64,{encode_image(image_source)}'


def _build_reward_images(images: list[str] | None) -> list[str]:
    """Normalise the rollout images for scoring, skipping the unusable ones.

    A single unreadable image must not sentinel the whole sample to -100, so
    failures are logged and dropped rather than raised.
    """
    reward_images: list[str] = []
    for image in images or []:
        try:
            reward_images.append(_to_reward_image_url(image))
        except (OSError, ValueError) as exc:
            logger.warning(f'Skipping image for reward scoring, cannot encode {image!r}: {exc}')
    return reward_images


def _build_runtime_image_locations(images: list[str]) -> dict[str, str]:
    runtime_image_locations: dict[str, str] = {}

    for image in images:
        if not _is_local_file(image):
            continue
        runtime_image_locations[image] = f'{_RUNTIME_IMAGE_DIR}/{Path(image).name}'

    return runtime_image_locations


async def _upload_images_to_runtime(images: list[str]) -> dict[str, str]:
    runtime_image_locations = _build_runtime_image_locations(images)
    if not runtime_image_locations:
        return {}

    ctx = require_context()
    runtime = await ctx.get_runtime('main')
    file_items = [
        FileItem(
            path=runtime_image_path,
            content=base64.b64encode(Path(image).read_bytes()).decode('utf-8'),
            encoding='base64',
        )
        for image, runtime_image_path in runtime_image_locations.items()
    ]
    await runtime.upload_file(file_items, overwrite=True)
    logger.debug(f'Uploaded {len(file_items)} image(s) to runtime work dir for multimodal task')
    return runtime_image_locations


# ---------------------------------------------------------------------------
# Video source transfer (for tools that cut clips out of the source video)
# ---------------------------------------------------------------------------


async def _close_video_session(ctx: Any) -> None:
    """Release the clip tool's kernel, its clip directory, and the staged source video.

    Never raises: a sample that answered correctly must not fail over a leftover temp file.
    """
    try:
        runtime = await ctx.get_runtime('main')
    except Exception:
        logger.warning('Failed to reach the runtime for video session cleanup', exc_info=True)
        return

    # Two independent cleanups: a kernel that refuses to close must not strand the staged source.
    try:
        await runtime.invoke(_CLOSE_VIDEOCUT_SESSION, ctx.run_id)
    except Exception:
        logger.warning('Failed to close the video tool session', exc_info=True)

    staged_path = (ctx.tool_state.get(_VIDEO_STATE_KEY) or {}).get('source_path')
    if staged_path:
        try:
            await runtime.invoke(remove_staged_video, staged_path)
        except Exception:
            logger.warning('Failed to remove the staged source video', exc_info=True)


class SearchTaskInput(TaskRunInput, WithAgentRunner):
    """Search task input.

    Scoring is performed exclusively by ``vl_reward_system``; the
    ``reward_func`` field selects the specific reward routine
    (e.g. ``exact_match``, ``consist_judge``).
    """

    question: str
    # Installed-harness input files ({source, target}), staged in the sandbox before the harness
    # starts. Replaces ``images`` on that path; the built-in main_agent still uses ``images``.
    artifacts: list[TaskArtifact] | None = None
    # How image artifacts get into the Codex conversation: 'attach' via `codex exec -i`, or 'path'
    # as a sandbox path in the question so the model opens it itself. Other harnesses only have paths.
    codex_image_inject_mode: Literal['attach', 'path'] = 'attach'
    images: list[str] | None = None  # Optional images associated with the question
    input_image_paths: bool = True  # whether to use image paths as input
    # Videos in the dataset's own form, e.g.
    # [{"video": "/path/a.mp4", "fps": 2, "max_frames": 384, "min_frames": 2}].
    # Passed to the model verbatim so the rollout decoder and the HF processor own the
    # frame sampling and the resize.
    videos: list[dict[str, Any]] | None = None
    # Copy the first video into the runtime so a clip tool can cut from it, and name the
    # staged path in the first-turn prompt. Off by default: the transfer plus installing
    # ffmpeg in the sandbox is only worth it when such a tool is actually mounted.
    enable_video_staging: bool = False
    # Presigned URLs for ``videos``, positionally aligned. When present the sandbox
    # downloads the source itself instead of the Host uploading it.
    video_cos_urls: list[str] | None = None
    video_timeout: float = 1800.0  # source transfer + probe budget, in seconds
    answer: str
    instance_id: str | None = None
    history: list[dict[str, Any]] | None = None
    new_sizes: list[list[int] | tuple[int, int] | None] | None = None
    # vl_reward_system routing
    reward_func: str | None = None
    # Optional reward-platform router id, e.g. "MultimodalReward-ToolUseModelScoring-0".
    reward_system_router_id: str | None = None
    # Free-form sub-service config forwarded as the request's ``extra`` field
    # (e.g. {"scoring_mode": "auto", "checklist_mode": "soft"}).
    reward_system_extra: dict[str, Any] | None = None
    # ``images`` reach the reward platform by default so a multimodal router
    # judges against the same picture the policy saw. Set True to drop them for
    # every router, or list the text-only router_ids that must stay text-only.
    reward_system_strip_multimodal_content: bool | list[str] | None = None
    instruction_template: str | None = None
    # System prompt configuration. If ``system_prompt`` is set, it takes
    # precedence; otherwise the task picks a pre-canned template by name
    # via ``use_default_system_prompt_template`` (e.g. 'search', 'code',
    # 'generic_training', 'generic_inference'). ``None`` → 'code'.
    system_prompt: str | None = None
    use_default_system_prompt_template: str | None = None
    # Force-finish: inject FORCE_FINISH_PROMPT just before the final
    # iteration so the model converges to a final answer instead of
    # hitting MAX_ITERATIONS. Opt-in; typical usage is inference only.
    enable_force_finish: bool = False
    max_iterations: int = 10


class SearchTaskOutput(TaskRunOutput, WithScore):
    """Output containing both result and ground truth for immediate debuggability."""

    prediction: str
    answer: str
    trajectory: list[dict[str, Any]] | None = None
    evaluation_details: dict[str, Any] | None = None
    # Exposed on the task output so consumers (viz / downstream training)
    # don't have to dig into ``agent_output`` again.
    exit_status: str | None = None
    instance_id: str | None = None
    reward_func: str | None = None
    question: str | None = None
    system_prompt_template: str | None = None
    # ``runner/hytuner.py`` merges this into the trajectory's extra_fields; its
    # ``reward_extra_info`` becomes per-sample columns of the rollout_data_dir jsonl.
    extra_fields: dict[str, Any] = Field(default_factory=dict)


def _build_reward_extra_info(evaluation_details: dict[str, Any] | None) -> dict[str, str]:
    """Expose the judge's raw output under the reward-system dump columns.

    Every exit path emits the same keys (empty when the judge was never called)
    so the rollout batch columns stay uniform across samples.
    """
    details = evaluation_details or {}
    return {
        'rm_response': str(details.get('raw_response') or ''),
        'rm_request_id': str(details.get('platform_request_id') or ''),
    }


def _emit_task_span_attrs(
    *,
    exit_status: str | None = None,
    termination_reason: str | None = None,
    reward_time: float | None = None,
) -> None:
    """Publish termination_reason / reward_time onto the current span for
    ``HytunerProcessor`` to consume.

    ``exit_status`` is translated via ``MainExitStatus.termination_reason_for``
    and written as ``hytuner.termination_reason`` so hytuner's
    ``maskout_samples_by_termination_reason`` matches on the uppercase
    ``TerminationReason`` strings rather than nexus's lowercase
    ``MainExitStatus`` values. Unknown values (e.g. our minted
    ``REWARD_ERROR``) pass through unchanged.
    """
    span_id = get_current_span_id()
    if not span_id:
        return
    attrs: dict[str, Any] = {}
    if termination_reason is not None:
        attrs['hytuner.termination_reason'] = termination_reason
    elif exit_status is not None:
        attrs['hytuner.termination_reason'] = (
            MainExitStatus.termination_reason_for(exit_status) if exit_status != 'REWARD_ERROR' else exit_status
        )
    if reward_time is not None and reward_time > 0:
        attrs['hytuner.reward_time'] = float(reward_time)
    if attrs:
        update_span(span_id, attributes=attrs)


def _build_task_output(
    task_input: SearchTaskInput,
    *,
    score: float,
    prediction: str,
    exit_status: str,
    trajectory: list[dict[str, Any]] | None = None,
    evaluation_details: dict[str, Any] | None = None,
) -> SearchTaskOutput:
    """Build the task output. Every exit path goes through here so they all carry the same fields."""
    return SearchTaskOutput(
        score=score,
        prediction=prediction,
        answer=task_input.answer.strip(),
        trajectory=trajectory,
        evaluation_details=evaluation_details,
        exit_status=exit_status,
        instance_id=task_input.instance_id,
        reward_func=task_input.reward_func,
        question=task_input.question,
        system_prompt_template=task_input.use_default_system_prompt_template,
        extra_fields={'reward_extra_info': _build_reward_extra_info(evaluation_details)},
    )


async def _score_with_reward_system(
    task_input: SearchTaskInput,
    *,
    prediction: str,
    trajectory: list[dict[str, Any]] | None,
    image_sources: list[str] | None,
) -> tuple[float, dict[str, Any], float]:
    """Judge one rollout, returning (score, evaluation_details, elapsed_seconds).

    ``image_sources`` are host paths on the rollout worker. An unusable judge score is clamped to
    ``_UNJUDGED_SCORE``, leaving the caller to relabel its own exit status.
    """
    strip_all_routers, strip_router_ids = resolve_strip_multimodal_config(
        task_input.reward_system_strip_multimodal_content
    )
    strip_multimodal = strip_all_routers or task_input.reward_system_router_id in strip_router_ids
    reward_images = [] if strip_multimodal else _build_reward_images(image_sources)

    reward_start = time.monotonic()
    score, evaluation_details = await vl_reward_system(
        task_input.reward_func,
        question=task_input.question,
        response=prediction,
        target_answer=task_input.answer.strip(),
        trajectory=trajectory,
        instance_id=task_input.instance_id,
        router_id=task_input.reward_system_router_id,
        images=reward_images,
        reward_extra=task_input.reward_system_extra,
    )
    reward_time = time.monotonic() - reward_start

    if score is None or float(score) <= _UNJUDGED_SCORE_THRESHOLD:
        logger.warning(
            f'[RewardError] vl_reward_system returned invalid score={score}; clamping to {_UNJUDGED_SCORE}. '
            f'instance_id={task_input.instance_id}, reward_func={task_input.reward_func!r}'
        )
        score = _UNJUDGED_SCORE
    return score, evaluation_details, reward_time


async def _run_installed_harness(task_input: SearchTaskInput) -> SearchTaskOutput:
    """Agent-agnostic path for installed CLI harnesses (claude_code / codex / hycode): stage the sample's
    artifacts in the sandbox, run the harness on the bare question, then score via vl_reward_system
    like the main_agent path. Exit comes from success/exit_code plus the harness's own failure_kind
    and termination_reason (no MainExitStatus here).
    """
    question = task_input.question
    if task_input.images:
        raise ValueError(
            'mm_agent installed harnesses take task_config.artifacts ({source, target}), not images; '
            'inline image content cannot reach a CLI harness.'
        )
    if task_input.videos or task_input.enable_video_staging:
        # Dispatch here happens before the staging checks, so without this the sample answers
        # a video question having never seen the video, and still gets scored and trained on.
        raise ValueError(
            'mm_agent installed harnesses cannot take videos or enable_video_staging; the clip '
            'tool and the staged source are both main_agent-only.'
        )
    artifacts = task_input.artifacts or []
    if artifacts:
        await upload_artifacts(artifacts)

    # Honor the dataset instruction_template like the main_agent path.
    user_prompt = (
        task_input.instruction_template.replace('{question}', question) if task_input.instruction_template else question
    )

    # Installed agents take a plain-text user_prompt; working_dir/timeout/version come from
    # nexus.agent_config. main-only kwargs are omitted (run_agent filters them anyway).
    harness_kwargs: dict[str, Any] = {'user_prompt': user_prompt}
    # Only Codex can attach images the way its own users do (`codex exec -i`), and only when the
    # dataset asks for it. Everything else is reached by path, so the dataset must name it in the
    # question or the harness cannot find the file.
    can_attach_images = task_input.codex_image_inject_mode == 'attach' and _selected_agent_name().lower() == 'codex'
    attached = [artifact.target for artifact in artifacts if artifact.is_image] if can_attach_images else []
    unreferenced = [
        artifact.target
        for artifact in artifacts
        if artifact.target not in attached and artifact.target not in user_prompt
    ]
    if unreferenced:
        raise ValueError(f'mm_agent artifacts are not mentioned in the question and cannot be found: {unreferenced}')
    if attached:
        harness_kwargs['images'] = attached
    agent_output = await task_input.run_agent(**harness_kwargs)

    if not isinstance(agent_output, InstalledAgentOutput):
        raise TypeError(
            f'installed harness path expected InstalledAgentOutput, got {type(agent_output).__name__} '
            f'(agent_name={_selected_agent_name()!r}); check nexus.agent_name routing.'
        )

    # output is the harness's raw stdout, so hycode must run in text mode, not --output-format stream-json.
    prediction_val = extract_final_answer(agent_output.output or '')
    success = agent_output.success
    exit_code = int(agent_output.exit_code or 0)
    # vl_reward_system is an ORM (no PRM yet) and ignores ``trajectory``; installed harnesses expose
    # none anyway, so pass None explicitly rather than a fake empty list.
    trajectory = None

    # Classify before scoring: only a run that finished is worth judging. failure_kind tells a harness
    # that never ran from a model that ran badly; codex marks context-window refusals TRUNCATION, and any
    # other harness-private reason (e.g. CODEX_TURN_FAILED) means nothing to HyTuner. hycode also reports
    # success on a non-zero exit it deems scoreable (missing exit code, SIGKILL after a full trajectory).
    harness_reason = getattr(agent_output, 'termination_reason', None)
    if success and (exit_code == 0 or harness_reason == TerminationReason.ENV_DONE.value):
        termination_reason = TerminationReason.ENV_DONE.value
    elif agent_output.failure_kind == 'infrastructure':
        termination_reason = TerminationReason.ENV_ERROR.value
    elif harness_reason == TerminationReason.TRUNCATION.value:
        termination_reason = TerminationReason.TRUNCATION.value
    else:
        termination_reason = TerminationReason.UNHANDLED_ERROR.value

    evaluation_details: dict[str, Any] | None = None
    reward_time = 0.0
    if termination_reason == TerminationReason.ENV_DONE.value:
        # The reward platform reads from the rollout worker, so ``source``, not the sandbox ``target``.
        score, evaluation_details, reward_time = await _score_with_reward_system(
            task_input,
            prediction=prediction_val,
            trajectory=trajectory,
            image_sources=[artifact.source for artifact in artifacts if artifact.is_image],
        )
        if score <= _UNJUDGED_SCORE_THRESHOLD:
            termination_reason = 'REWARD_ERROR'
    elif termination_reason == TerminationReason.TRUNCATION.value:
        # Scored like main_agent's CONTEXT_LENGTH_EXCEEDED, which also maps to TRUNCATION.
        score = 0.0
    else:
        # An infra failure, timeout or crash leaves no answer to judge; scoring partial output would be arbitrary.
        score = _TASK_FAILED_SCORE
        logger.warning(
            f'[HarnessFailure] termination_reason={termination_reason} stage={agent_output.failure_stage!r}; '
            f'masking (score={score}). instance_id={task_input.instance_id}'
        )

    logger.info(
        f'[mm_agent/installed harness={_selected_agent_name() or "unknown"}] '
        f'InstanceID: {task_input.instance_id or "N/A"} | '
        f'success={success} exit_code={exit_code} | Score: {score} | Question: {question} | '
        f'Prediction[:300]: {prediction_val[:300]}'
    )
    _emit_task_span_attrs(termination_reason=termination_reason, reward_time=reward_time)

    return _build_task_output(
        task_input,
        score=score,
        prediction=prediction_val,
        exit_status=termination_reason,
        trajectory=trajectory,
        evaluation_details=evaluation_details,
    )


@task_registry.register('mm_agent')
async def mm_agent_task(task_input: SearchTaskInput) -> SearchTaskOutput:
    """
    Executes a search task and evaluates the prediction via vl_reward_system.

    Agent-agnostic: nexus.agent_name selects the harness — main_agent runs the native multimodal
    tool loop; any other value is an installed CLI harness via _run_installed_harness.
    """
    question = task_input.question
    instance_id = task_input.instance_id
    agent_name = _selected_agent_name()
    logger.debug(
        f'Search task started: reward_func={task_input.reward_func!r}, instance_id={instance_id}, '
        f'agent_name={agent_name or "unknown"}, question={question}'
    )
    # Standardize ground truth once
    target_answer = task_input.answer.strip()

    if _is_installed_harness(agent_name):
        return await _run_installed_harness(task_input)

    # Empty / main_agent -> built-in path below; any other name is unsupported and must fail loud.
    normalized_agent = agent_name.strip().lower()
    if normalized_agent and normalized_agent not in _BUILTIN_AGENT_NAMES:
        raise ValueError(
            f'mm_agent: unsupported nexus.agent_name={agent_name!r}. Expected empty / '
            f'{sorted(_BUILTIN_AGENT_NAMES)} (built-in main_agent) or {sorted(_INSTALLED_HARNESS_NAMES)}.'
        )
    if task_input.artifacts:
        raise ValueError('mm_agent artifacts are installed-harness only; the built-in main_agent takes images.')

    # 1. Execution
    template = task_input.instruction_template
    if template is None:
        logger.warning('instruction_template is None, falling back to question as prompt')
    prompt = template.replace('{question}', question) if template else question

    runtime_image_locations: dict[str, str] = {}
    ctx = require_context()
    if task_input.images and ctx.runtime_provider:
        runtime_image_locations = await _upload_images_to_runtime(task_input.images)
        for image, runtime_image_path in runtime_image_locations.items():
            prompt = prompt.replace(image, runtime_image_path)

    # Staged before the prompt is built: clip tools locate the source through the path this
    # puts in the first-turn message.
    video_state: dict[str, Any] | None = None
    if task_input.enable_video_staging:
        if not task_input.videos:
            raise ValueError('enable_video_staging is set but the sample carries no videos')
        if not ctx.runtime_provider:
            raise ValueError('enable_video_staging needs a runtime provider to stage the source into')
        video_state = await prepare_video_source(task_input.videos, task_input.video_cos_urls, task_input.video_timeout)
        # Set here, with the ``try`` on the next line, so the staged copy is owned by that
        # cleanup from the moment it exists — everything after this point can raise.
        ctx.tool_state[_VIDEO_STATE_KEY] = video_state

    try:
        requests = []
        if task_input.images is not None:
            for image in task_input.images:
                if task_input.input_image_paths:
                    image_name = Path(image).stem
                    image_location = runtime_image_locations.get(image, image)
                    requests.append({'type': 'text', 'text': f'\n![{image_name}]({image_location})'})
                base64_data = encode_image(image)
                requests.append({
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{base64_data}'},
                })
        # Two paths for one video, because two processes resolve them: the decoder runs on the
        # Host and reads the original, while the model's code runs in the runtime and needs the
        # staged copy, which goes in as a separate text part.
        for index, video in enumerate(task_input.videos or []):
            if video_state is not None and index == 0:
                staged = video_state['source_path']
                requests.append({'type': 'text', 'text': f'\n![{Path(staged).stem}]({staged})'})
            requests.append({'type': 'video', **video})
        requests.append({'type': 'text', 'text': prompt})

        # Pick default system_prompt from the named template registry in
        # prompts.py. Explicit ``system_prompt`` wins if provided.
        system_prompt = task_input.system_prompt or get_default_system_prompt(
            task_input.use_default_system_prompt_template
        )

        # Expose the current-turn user question and the user-supplied image list
        # to tools via tool_state. Consumed by ``yb_web_search`` (forwards the
        # question as the high-level ``prompt`` to the oneagent prompt_search
        # backend) and ``yb_vision_search`` (resolves ``user_image K`` ids to
        # local paths). Use a ``yb_search.*`` namespace to avoid colliding with
        # other consumers of ``tool_state``.
        ctx.tool_state['yb_search.user_question'] = question
        ctx.tool_state['yb_search.images'] = list(task_input.images or [])

        agent_kwargs: dict[str, Any] = {
            'user_prompt': requests,
            'system_prompt': system_prompt,
            'history': task_input.history,
            'max_iterations': task_input.max_iterations,
        }
        if task_input.enable_force_finish:
            agent_kwargs['post_iteration_hook'] = _FORCE_FINISH_HOOK_PATH

        agent_output = await task_input.run_agent(**agent_kwargs)
    finally:
        if _VIDEO_STATE_KEY in ctx.tool_state:
            await _close_video_session(ctx)

    if not isinstance(agent_output, MainAgentOutput):
        raise RuntimeError(f'mm_agent task only supports main_agent output, got {type(agent_output)}')

    trajectory = agent_output.main_trajectory

    # -------------------------------------------------------------------
    # Handle abnormal exit statuses.
    #
    # main_agent's ``MainExitStatus`` (str-subclass Enum) now distinguishes
    # *model-side* vs *environment-side* failures. For the former we want a
    # trainable 0 signal; for the latter we mask the sample out via
    # ``_UNJUDGED_SCORE`` (which is below hytuner's mask_invalid_samples
    # threshold of -99).
    # -------------------------------------------------------------------
    exit_status = getattr(agent_output, 'exit_status', None)
    exit_status_value = exit_status.value if hasattr(exit_status, 'value') else (exit_status or '')

    if exit_status_value in _ALWAYS_MASK_STATUSES:
        assistant_turns = len([x for x in trajectory if x['role'] == 'assistant'])
        logger.info(
            f'[ExitStatus: {exit_status_value}] Masked (score={_UNJUDGED_SCORE}), '
            f'turn={assistant_turns}, question={question}'
        )
        final_answer = getattr(agent_output, 'final_answer', None) or ''
        _emit_task_span_attrs(exit_status=exit_status_value)
        return _build_task_output(
            task_input,
            score=_UNJUDGED_SCORE,
            prediction=final_answer,
            exit_status=exit_status_value,
            trajectory=trajectory,
        )

    # Model-side protocol violations (e.g. malformed tool call, leftover image
    # special tokens in a text-only rollout): skip the reward system entirely
    # and return score=0.0. These are unambiguously wrong responses — invoking
    # a judge LM or semantic matcher would only waste compute.
    if exit_status_value in _ALWAYS_ZERO_STATUSES:
        assistant_turns = len([x for x in trajectory if x['role'] == 'assistant'])
        logger.info(
            f'[ExitStatus: {exit_status_value}] Short-circuited (score=0.0), '
            f'turn={assistant_turns}, question={question}'
        )
        final_answer = getattr(agent_output, 'final_answer', None) or ''
        _emit_task_span_attrs(exit_status=exit_status_value)
        return _build_task_output(
            task_input,
            score=0.0,
            prediction=final_answer,
            exit_status=exit_status_value,
            trajectory=trajectory,
        )

    last_msg = agent_output.main_trajectory[-1]
    raw_content = last_msg.get('content') or last_msg.get('reasoning_content')
    prediction_val = extract_text_from_content(raw_content) if raw_content is not None else ''

    # Normalise raw prediction: strip <think> wrappers so the
    # reward system always sees the final answer in its bare form.
    prediction_val = extract_final_answer(prediction_val)
    logger.debug(f'Extracted prediction: {prediction_val}')

    # 2. Evaluation — always via vl_reward_system.
    score, evaluation_details, reward_time = await _score_with_reward_system(
        task_input,
        prediction=prediction_val,
        trajectory=agent_output.main_trajectory,
        image_sources=task_input.images,
    )

    logger.info(
        f'**vl_reward_system** InstanceID: {instance_id or "N/A"} | Score: {score} | '
        f'Question: {question} | Prediction: {prediction_val} | Reference: {target_answer} | '
        f'Evaluation Details: {evaluation_details}'
    )

    # An unusable judge score was clamped to _UNJUDGED_SCORE (masked by value); label it a scoring failure.
    if score <= _UNJUDGED_SCORE_THRESHOLD:
        exit_status_value = 'REWARD_ERROR'
    else:
        exit_status_value = exit_status_value or MainExitStatus.FINISHED.value

    _emit_task_span_attrs(exit_status=exit_status_value, reward_time=reward_time)

    return _build_task_output(
        task_input,
        score=score,
        prediction=prediction_val,
        exit_status=exit_status_value,
        trajectory=trajectory,
        evaluation_details=evaluation_details,
    )
