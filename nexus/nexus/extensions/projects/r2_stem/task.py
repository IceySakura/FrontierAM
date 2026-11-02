"""Run one R2 STEM sample with an agent and evaluate it after execution."""

import asyncio
import contextlib
import json
import logging
import math
import time

from pathlib import PurePosixPath
from typing import Any

from pydantic import Field

from nexus.core import (
    ContextLengthExceededError,
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.agents.base.schema import TerminationReason
from nexus.extensions.projects.r2_stem.adapter import capture_answer_file, hide_verifier_from_agent
from nexus.extensions.projects.r2_stem.evaluator import (
    VerifierArtifact,
    build_evaluator,
    unit_cases_evaluate,
    verifier_image_evaluate,
)
from nexus.extensions.projects.r2_stem.evaluator.common import load_ref_answer
from nexus.extensions.projects.r2_stem.evaluator.reward_system import rmsys_evaluate
from nexus.trace import get_current_span_id, update_span


logger = logging.getLogger(__name__)


SUPPORTED_EVALUATION_METHODS = {'llm_judge', 'none', 'rmsys', 'unit_cases', 'verifier_image'}
TEXT_EVALUATION_METHODS = {'llm_judge', 'rmsys'}
SANDBOX_EVALUATION_METHODS = {'unit_cases', 'verifier_image'}
SUCCESS_EXIT_STATUSES = {
    'mini_swe_agent_stem': {'submitted'},
    'terminus2': {'completed'},
    # HyCode is a CLI agent and reports ``termination_reason`` instead of ``exit_status``.
    'hycode': {'ENV_DONE'},
}
MAX_ERROR_TRACEBACK_CHARS = 20_000
EVALUATION_FAILURE_SCORE = -200.0
AGENT_FAILURE_SCORE = -300.0

# A rollout that ran out of room, or out of time, never reached the answer it was working
# towards, so anything the evaluator can see is whichever fragment the model had emitted
# when the limit hit. Scoring that fragment spends a judge call on a number that describes
# nothing about the policy, and for the length cutoffs it still lands in the GRPO group
# statistics its siblings are centred on. Short-circuit to 0.0 instead; whether the partial
# sequence trains or is masked stays HyTuner's decision, taken from the termination reason.
#
# A mapping rather than a set of cutoff statuses, because the three must stay distinct for
# the training side: 'output_length_exceeded' says max_tokens is too small for one turn,
# 'truncation' says the context window or trajectory budget is, and 'timeout' says the
# wall-clock budget is. Every agent in SUCCESS_EXIT_STATUSES spells these the same way.
#
# Running out of turns is deliberately absent. HyCode's MAX_STEPS belongs here only if the
# alternative were AGENT_FAILURE_SCORE, but a non-success status merely records an error and
# still gets judged, which is what Mini-SWE's max_iterations already does. Judging both means
# a run that reached its answer and then burned its remaining turns keeps the score it earned.
_CUTOFF_EXIT_STATUSES = {
    'truncation': TerminationReason.TRUNCATION.value,
    'output_length_exceeded': TerminationReason.OUTPUT_LENGTH_EXCEEDED.value,
    'timeout': TerminationReason.TASK_TIMEOUT.value,
}
_CUTOFF_TERMINATION_REASONS = frozenset(_CUTOFF_EXIT_STATUSES.values())

# The exception to the short-circuit above, for rows that set ``verifier_score_truncated``: a
# verifier grades files on disk, so whatever the agent had written by the time it stopped is
# still there to grade, and scoring 0 unseen throws that work away.
#
# A timeout is here on the same reasoning but is not the same risk. Truncation stops the agent
# between turns, while a timeout cancels it mid-step, so the sandbox may still be changing while
# the verifier reads it. The grade is then whatever the run happened to have reached, which is
# the trade the flag buys.
_ARTIFACT_GRADABLE_CUTOFF_REASONS = frozenset({
    TerminationReason.TRUNCATION.value,
    TerminationReason.OUTPUT_LENGTH_EXCEEDED.value,
    TerminationReason.TASK_TIMEOUT.value,
})

# Score a verifier gives when every test passed.
FULL_VERIFIER_SCORE = 1.0

# Reasons a full verifier pass replaces with ENV_DONE. Every member is in HyTuner's
# MASKOUT_TERMINATION_REASONS (hytuner/utils/termination_reason.py), so keeping it would drop the
# row from the gradient and throw the verifier run away -- while the grade itself says the work
# was finished, whatever else went wrong.
#
# What is deliberately absent matters as much. TRUNCATION, OUTPUT_LENGTH_EXCEEDED and MAX_STEPS
# are not masked, so they already train on their real score and relabelling would only cost the
# diagnosis. UNHANDLED_ERROR is masked but stays out: it names a failure nobody has explained
# yet, and a passing grade is not evidence that the failure was harmless.
_FULL_PASS_CLEARS_REASON = frozenset({
    TerminationReason.TASK_TIMEOUT.value,
    TerminationReason.ENV_ERROR.value,
})

# Bounds the error text copied into the one-line verifier summary.
MAX_SUMMARY_ERROR_CHARS = 500


class ScienceAgentTaskInput(TaskRunInput, WithAgentRunner):
    """One R2 STEM sample and its post-agent evaluation configuration."""

    question: str = Field(description='Question prompt to solve.')
    ref_answer: Any = Field(default=None, description='Reference answer copied through for downstream judge.')
    image: str = Field(default='', description='Runtime image for this sample.')
    sample_id: str = Field(default='', description='Stable sample id/run id.')
    agent_name: str = Field(
        default='',
        description='Agent driving this sample; must match the runner-level agent_name when both are set.',
    )
    runtime_name: str = Field(default='main', description='Runtime name to acquire.')
    agent_timeout: int = Field(default=7200, description='Maximum seconds for agent execution.')
    agent_wall_time_reserve_seconds: float = Field(
        default=600.0,
        description='Seconds withheld from agent_timeout so the agent can stop itself before being cancelled.',
    )
    stop_at_trajectory_budget: bool = Field(
        default=False,
        description=(
            'With agent_name=hycode, end the rollout at the model call that fills the trajectory budget, so it '
            'trains as TRUNCATION. The budget prefers sampling_params.max_tokens when set, else max_model_len. '
            'Without this flag HyCode retries on a rewritten history, which opens a new token segment and gets '
            'the whole rollout masked as UNHANDLED_ERROR. Other agents ignore this flag.'
        ),
    )
    working_dir: str = Field(default='/workspace', description='Workspace directory for the agent.')
    evaluation_method: str = Field(
        default='none',
        description='Evaluation method: llm_judge, unit_cases, verifier_image, rmsys, or none.',
    )
    task_dir: str = Field(
        default='',
        description='Local task directory containing host-side evaluator resources.',
    )
    test_timeout: int = Field(default=3600, description='Maximum seconds for post-agent sandbox tests.')
    docker_timeout: int = Field(default=3600, description='Maximum seconds for the inner verifier image pull.')
    verifier_runtime_name: str = Field(default='verifier', description='Trusted verifier runtime name.')
    verifier_image: str = Field(
        default='',
        description='Digest-pinned image that both backs the verifier runtime and owns /tests.',
    )
    verifier_dind: bool = Field(
        default=False,
        description=(
            'Run the image-owned tests in a nested container on the verifier runtime instead of its rootfs, '
            'with the inner docker daemon exposed to them.'
        ),
    )
    verifier_network_access: bool = Field(
        default=False,
        description='Allow audited egress rules in the otherwise deny-by-default verifier runtime.',
    )
    verifier_artifacts_read_only: bool = Field(
        default=True,
        description='Expose transferred agent artifacts read-only during verification.',
    )
    verifier_score_truncated: bool = Field(
        default=False,
        description=(
            'With evaluation_method=verifier_image, grade the artifacts of a rollout that was cut '
            'short (TRUNCATION, OUTPUT_LENGTH_EXCEEDED or TASK_TIMEOUT, including a timeout that '
            'cancelled the agent) instead of scoring it 0 unseen. The reason is kept, and a reason '
            'HyTuner masks still drops the sample, with one exception: a reason in '
            '_FULL_PASS_CLEARS_REASON that scored full marks is reported as ENV_DONE, since masking '
            'it would throw away a grade that says the work was finished. Grading a cancelled agent '
            'races it -- run_agent never returned, so the sandbox may still be changing while the '
            'verifier reads it. Other evaluation methods ignore this flag.'
        ),
    )
    verifier_mounts: list[str | VerifierArtifact] = Field(
        default_factory=list,
        description='Explicit agent artifact paths transferred to the trusted verifier.',
    )
    verifier_allowed_protected_mounts: list[str] = Field(
        default_factory=list,
        description='Exact audited protected-path exceptions for verifier artifacts.',
    )
    verifier_env: dict[str, str] = Field(
        default_factory=dict,
        description='Verifier-only environment templates resolved immediately before scoring.',
    )
    verifier_env_refs: dict[str, str] = Field(
        default_factory=dict,
        description='Verifier env name to host env name, resolved immediately before scoring.',
    )
    reward_system_user_token: str | None = Field(
        default=None,
        description='Reward system user token; falls back to NEXUS_REWARD_SYSTEM_TOKEN.',
    )
    reward_system_exp_id: str = Field(
        default='',
        description='Reward system experiment id; falls back to NEXUS_REWARD_SYSTEM_EXP_ID.',
    )
    reward_system_router_id: str = Field(
        default='',
        description='Reward system router id, e.g. "class1-class2-version".',
    )
    reward_system_extra: dict[str, Any] | None = Field(
        default=None,
        description='Optional reward system extra parameters.',
    )


class ScienceAgentTaskOutput(TaskRunOutput, WithScore):
    """Nexus result payload for downstream reporting."""

    answer: str | None = None
    ref_answer: Any = None
    image: str = ''
    question: str = ''
    sample_id: str = ''
    raw_terminal_output: str | None = None
    exit_status: str = 'unknown'
    iterations: int = 0
    test_results: dict[str, Any] | None = None
    judge_error: str | None = None
    error: str | None = None
    agent_error: str | None = None
    agent_error_type: str | None = None
    agent_error_traceback: str | None = None
    extra_fields: dict[str, Any] = Field(default_factory=dict)


def _normalize_evaluation_method(evaluation_method: str) -> str:
    method = evaluation_method.strip().lower()
    if method not in SUPPORTED_EVALUATION_METHODS:
        raise ValueError(f'Unsupported evaluation_method: {evaluation_method}')
    return method


def _answer_path(working_dir: str) -> str:
    directory = PurePosixPath(working_dir)
    if not directory.is_absolute():
        raise ValueError(f'working_dir must be an absolute runtime path: {working_dir}')
    return str(directory / 'answer.txt')


def _build_agent_prompt(
    *,
    question: str,
    evaluation_method: str,
    agent_name: str,
    working_dir: str,
) -> str:
    """Attach the submission contract that text judging needs from the selected agent."""
    if evaluation_method not in TEXT_EVALUATION_METHODS:
        return question

    if agent_name == 'terminus2':
        answer_path = _answer_path(working_dir)
        return (
            f'{question}\n\n'
            f'When you finish, write exactly one output file: {answer_path}\n'
            'The file must contain only the final answer, without analysis, logs, or Markdown fences.'
        )

    if agent_name == 'hycode':
        # The CLI ends the session on the first turn that carries no tool call, and text
        # judging then reads that turn. Mini-SWE runs on the same rule and makes it work by
        # stating it as an invariant (see agent.py:_build_user_prompt); HyCode's own system
        # prompt never does, so a turn that merely pauses to think ends the run and leaves
        # the judge whatever the model last happened to say. Spell the rule out here.
        return (
            f'{question}\n\n'
            'While working, every response must include at least one tool call. '
            'A response without a tool call ends the session immediately, so never stop at a '
            'thought, a plan, or a progress note.\n'
            'When you have the final answer, respond with no tool call, and make the final answer '
            'the entire content of that message, without analysis, logs, or Markdown fences. '
            'That message is the only thing that gets read, so state the answer there even if you '
            'have also written it to a file.'
        )

    return question


def _extract_assistant_answer(agent_output: Any) -> str:
    submission = getattr(agent_output, 'submission', None)
    if isinstance(submission, str) and submission.strip():
        return submission.strip()

    trajectory = getattr(agent_output, 'main_trajectory', None)
    if not trajectory:
        return ''
    last_message = trajectory[-1]
    if not isinstance(last_message, dict) or last_message.get('role') != 'assistant':
        return ''
    content = last_message.get('content')
    return content.strip() if isinstance(content, str) else ''


def _hycode_blocks_text(content: list[Any], block_type: str, field: str) -> str:
    chunks = [
        item[field]
        for item in content
        if isinstance(item, dict) and item.get('type') == block_type and isinstance(item.get(field), str)
    ]
    return ''.join(chunks)


def _hycode_message_text(event: dict[str, Any]) -> str:
    """Read the usable text of one HyCode ``assistant`` event.

    HyCode nests the payload under ``message`` like the Anthropic wire format, but some
    builds put ``content`` directly on the event, so both shapes are accepted.

    A block list with no text falls back to its thinking. When the model omits its think-end
    token the backend reports the whole turn as reasoning, so the answer or tool call the
    model actually produced arrives as a thinking block and the turn has no text at all.
    Treating such a turn as empty makes the caller reach further back and judge an unrelated
    line from the middle of the run, which is worse than judging the reasoning itself: the
    reward system scores reasoning-wrapped answers exactly like bare ones.
    """
    for container in (event.get('message'), event):
        if not isinstance(container, dict):
            continue
        content = container.get('content')
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            text = _hycode_blocks_text(content, 'text', 'text')
            if text:
                return text
            thinking = _hycode_blocks_text(content, 'thinking', 'thinking')
            if thinking:
                return thinking
    return ''


def _extract_hycode_answer(output: str) -> str:
    """Read the final answer out of HyCode's stream-json stdout.

    HyCode print mode emits newline-delimited events and closes with a ``result`` event
    whose ``result`` field carries the answer the CLI settled on, so the last such event
    wins. Runs that stop without a usable ``result`` payload still leave the answer in the
    newest ``assistant`` message, which is then the only remaining source.

    ``_hycode_message_text`` counts a thinking-only turn as usable, which keeps that search
    from walking past the turn the session actually ended on.
    """
    result_text = ''
    assistant_text = ''
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get('type')
        if event_type == 'result':
            result_text = str(event.get('result') or '')
        elif event_type == 'assistant':
            text = _hycode_message_text(event)
            if text.strip():
                assistant_text = text
    return (result_text or assistant_text).strip()


def _rollout_tool_calls_end_token(ctx: Any) -> str | None:
    """Return the token that closes a tool-call block for the rollout's tool parser, if it names one.

    Both rollout clients (legacy HyTuner and gateway) are built with ``client_args['tool_parser']``.
    HyTuner's HY4 wrapper names only the opening token, so the closing one is spelled from it.
    """
    client = (getattr(ctx, 'openai_clients', None) or {}).get('main')
    client_args = getattr(getattr(client, 'config', None), 'client_args', None)
    parser = client_args.get('tool_parser') if isinstance(client_args, dict) else None
    end_token = getattr(parser, 'tool_calls_end_token', None)
    if isinstance(end_token, str) and end_token:
        return end_token
    start_token = getattr(parser, 'tool_calls_start_token', None)
    if isinstance(start_token, str) and start_token.startswith('<') and not start_token.startswith('</'):
        return '</' + start_token[1:]
    return None


def _hycode_ended_on_unparsed_tool_call(ctx: Any, agent_output: Any) -> bool:
    """Whether HyCode's final reply still holds a tool call the rollout parser fell back on.

    The parser hands a malformed tool call (a dropped <arg_value> tag, say) back as plain text, so
    HyCode gets a reply with no tool call, takes it as its answer and exits success. A parsed block
    leaves no markup behind, so a closing token in that reply means the block was written out and
    then not parsed. A block cut short by max_tokens never got its closing token, and is not
    counted. Only a run that closed with a result event ended on a reply at all.
    """
    end_token = _rollout_tool_calls_end_token(ctx)
    if end_token is None or getattr(agent_output, 'result_subtype', None) is None:
        return False
    return end_token in _extract_hycode_answer(getattr(agent_output, 'output', '') or '')


async def _extract_answer(
    *,
    agent_name: str,
    agent_output: Any,
    runtime: Any,
    evaluation_method: str,
    working_dir: str,
    test_timeout: int,
) -> str:
    """Resolve a text answer using the adapter bound to the selected agent."""
    if agent_name == 'mini_swe_agent_stem':
        return _extract_assistant_answer(agent_output)

    if agent_name == 'hycode':
        return _extract_hycode_answer(getattr(agent_output, 'output', '') or '')

    if agent_name == 'terminus2':
        if evaluation_method not in TEXT_EVALUATION_METHODS:
            return ''
        return await capture_answer_file(
            runtime=runtime,
            answer_path=_answer_path(working_dir),
            test_timeout=test_timeout,
        )

    raise ValueError(f'Unsupported agent: {agent_name}')


def _agent_succeeded(agent_name: str, exit_status: str) -> bool:
    statuses = SUCCESS_EXIT_STATUSES.get(agent_name)
    if statuses is None:
        raise ValueError(f'Unsupported agent: {agent_name}')
    return exit_status in statuses


def _resolve_agent_name(task_agent_name: str, context_agent_name: str | None) -> str:
    """Bind the sample to an agent, rejecting a task_config that contradicts the runner."""
    configured = (task_agent_name or '').strip()
    from_runner = (context_agent_name or '').strip()
    if configured and from_runner and configured != from_runner:
        raise ValueError(
            f'task_config.agent_name ({configured}) does not match the configured agent_name ({from_runner}).'
        )
    agent_name = configured or from_runner
    if not agent_name:
        raise ValueError('science_agent_task requires task_config.agent_name.')
    if agent_name not in SUCCESS_EXIT_STATUSES:
        raise ValueError(f'Unsupported agent: {agent_name}')
    return agent_name


def _status_to_text(status: Any) -> str:
    if status is None:
        return ''
    value = getattr(status, 'value', status)
    return str(value).strip()


def _canonical_agent_reason(agent_name: str, agent_output: Any) -> str | None:
    """Return the reason HyTuner trains on when the agent publishes one, else None.

    HyCode carries both: ``termination_reason`` is the canonical value it puts on its span, and
    ``exit_status`` is a lowercased display name (``env_done``, or ``sigkill`` for a scoreable
    process exit) that never matches the canonical sets below.
    """
    if agent_name != 'hycode':
        return None
    return _status_to_text(getattr(agent_output, 'termination_reason', None)) or None


def _cutoff_termination_reason(agent_output: Any) -> str | None:
    """Return the canonical termination reason when the rollout was cut short, else None.

    Prefer the agent's own ``termination_reason`` when it already speaks the canonical
    uppercase vocabulary, since that is what reaches HyTuner; otherwise translate the
    lowercase exit status.
    """
    reason = _status_to_text(getattr(agent_output, 'termination_reason', None)).upper()
    if reason in _CUTOFF_TERMINATION_REASONS:
        return reason

    exit_status = _status_to_text(getattr(agent_output, 'exit_status', None)).lower()
    return _CUTOFF_EXIT_STATUSES.get(exit_status)


def _grades_cut_off_artifacts(task_input: ScienceAgentTaskInput, evaluation_method: str, cutoff_reason: str) -> bool:
    """Whether a rollout cut short by ``cutoff_reason`` still has its artifacts verified."""
    return (
        task_input.verifier_score_truncated
        and evaluation_method == 'verifier_image'
        and cutoff_reason in _ARTIFACT_GRADABLE_CUTOFF_REASONS
    )


def _agent_wall_time_limit(task_input: ScienceAgentTaskInput) -> float:
    """Return the wall-clock budget handed to the agent, shorter than the outer timeout.

    The outer timeout below cancels ``run_agent``, which takes the trajectory, the
    submission and the termination reason down with it and leaves the rollout unexplained.
    An agent that honours this budget ends its own loop first and returns all three; the
    outer timeout stays as the backstop for one that ignores it or wedges in its epilogue.

    The reserve is capped at a tenth of the timeout so a short ``agent_timeout`` — a test,
    or a smoke run — cannot fall to a non-positive budget, which agents read as 'no limit'.
    """
    reserve = min(task_input.agent_wall_time_reserve_seconds, task_input.agent_timeout / 10)
    return max(task_input.agent_timeout - reserve, 0.0)


# How often a deadline that follows a paused runner deadline checks whether it has resumed.
_PAUSED_DEADLINE_POLL_SECONDS = 5.0


def _runner_deadline(runner_timeout: Any) -> float | None:
    """Return the runner's absolute deadline, or None while it is paused."""
    when = getattr(runner_timeout, 'when', None)
    if callable(when):
        return when()
    # COMPAT: async-timeout on Python 3.10 keeps the deadline as an attribute instead of when().
    return runner_timeout.deadline


async def _await_agent_in_active_time(agent_run: Any, timeout: float, runner_timeout: Any) -> Any:
    """Await ``agent_run`` for at most ``timeout`` seconds of active time, like ``asyncio.wait_for``.

    Partial rollout parks model calls at training boundaries, and the HyTuner client suspends the
    runner deadline (``ExecutionContext.timeout_handle``) for exactly as long as each call stays
    parked. This deadline keeps the distance to the runner deadline it starts with, so every pause
    pushes it out by the same amount. Without a runner deadline there is nothing to follow and it
    is a plain wall-clock timeout.
    """
    runner_deadline = _runner_deadline(runner_timeout) if runner_timeout is not None else None
    if runner_deadline is None:
        return await asyncio.wait_for(agent_run, timeout=timeout)

    loop = asyncio.get_running_loop()
    lead = runner_deadline - (loop.time() + timeout)
    agent_task = asyncio.ensure_future(agent_run)
    try:
        while True:
            runner_deadline = _runner_deadline(runner_timeout)
            if runner_deadline is None:
                wait = _PAUSED_DEADLINE_POLL_SECONDS
            else:
                wait = runner_deadline - lead - loop.time()
                if wait <= 0:
                    raise asyncio.TimeoutError(f'agent used its {timeout:g}s of active time')
            done, _ = await asyncio.wait({agent_task}, timeout=wait)
            if done:
                return agent_task.result()
    finally:
        if not agent_task.done():
            agent_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await agent_task


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _configured_max_tokens(client: Any) -> int | None:
    """Return the request-level ``max_tokens`` cap, not a per-call remaining budget."""
    sampling = getattr(client, 'sampling_params', None)
    if not isinstance(sampling, dict):
        request_args = getattr(getattr(client, 'config', None), 'request_args', None) or {}
        sampling = request_args.get('sampling_params') if isinstance(request_args, dict) else None
    if not isinstance(sampling, dict):
        return None
    return _positive_int(sampling.get('max_tokens'))


def _trajectory_token_limit(client: Any) -> int | None:
    """Return the ``usage.total_tokens`` at which the trajectory budget is used up.

    Measured the way HyTunerClient spends it: ``max_tokens`` covers everything after the initial
    prompt (responses and tool observations), and the context never passes ``max_model_len``.
    ``usage.total_tokens`` includes that initial prompt, so it is added back before comparing.
    A client that does not report its initial prompt length is measured from zero.

    The length is HyTunerClient's private ``_initial_prompt_len``, the value its own budget is
    measured from; the client is shared and exposes no accessor, so a test pins the name instead.
    """
    max_tokens = _configured_max_tokens(client)
    max_model_len = _positive_int(getattr(client, 'max_model_len', None))
    if max_tokens is None:
        return max_model_len
    limit = max_tokens + (_positive_int(getattr(client, '_initial_prompt_len', None)) or 0)
    return min(limit, max_model_len) if max_model_len is not None else limit


class _TrajectoryBudgetStop:
    """Stands in for the rollout client while HyCode runs; see ``stop_at_trajectory_budget``.

    HyCode reads ``finish_reason=length`` as a per-call output limit and keeps going on a rewritten
    history (max-output recovery, queued task-notifications). The training client answers a rewritten
    history with a fresh budget, and the trajectory processor then masks the whole rollout. So once a
    call fills the trajectory budget -- its tokens are already recorded by then -- HyCode is told
    ``context_length_exceeded``, and every later call is refused before it reaches the client.

    The budget prefers request-level ``sampling_params.max_tokens`` (the per-sample response cap from
    extra_info or the trainer) when it is set, then ``max_model_len``; see ``_trajectory_token_limit``
    for how it is compared with ``usage.total_tokens``. A per-call ``max_tokens`` that is tighter than
    that budget still passes through, so HyCode's max-output recovery is unchanged.
    """

    def __init__(self, client: Any, *, sample_id: str, trajectory_id: str | None) -> None:
        self.client = client
        self.sample_id = sample_id
        self.trajectory_id = trajectory_id
        self.exhausted = False
        self.refused_calls = 0

    @classmethod
    def install(cls, ctx: Any, task_input: ScienceAgentTaskInput, agent_name: str) -> '_TrajectoryBudgetStop | None':
        """Put the stop in front of the rollout client for one agent run, when the row asks for it."""
        if not task_input.stop_at_trajectory_budget or agent_name != 'hycode':
            return None
        client = ctx.openai_clients.get('main')
        if getattr(client, 'max_model_len', None) is None:
            logger.info(
                '[trajectory-budget-stop] not installed: main client %s has no max_model_len', type(client).__name__
            )
            return None
        stop = cls(client, sample_id=task_input.sample_id, trajectory_id=ctx.task_config.get('traj_id'))
        ctx.openai_clients['main'] = stop
        return stop

    def restore(self, ctx: Any, agent_exit_status: str) -> None:
        ctx.openai_clients['main'] = self.client
        if self.exhausted:
            logger.warning(
                '[trajectory-budget-stop] rollout ended at the trajectory budget: sample_id=%s traj_id=%s '
                'refused_calls=%d agent_exit_status=%s',
                self.sample_id,
                self.trajectory_id,
                self.refused_calls,
                agent_exit_status,
            )

    def _is_auxiliary(self, model: Any) -> bool:
        # HyTunerClient's private routing check, pinned by a test: the shared client has no public one.
        is_auxiliary_model = getattr(self.client, '_is_auxiliary_model', None)
        return is_auxiliary_model is not None and bool(is_auxiliary_model(model))

    def __getattr__(self, name: str) -> Any:
        # Only reached for names not defined here: everything the LLM bridge uses besides create_completion.
        client = self.__dict__.get('client')
        if client is None:
            raise AttributeError(name)
        return getattr(client, name)

    async def create_completion(self, *args: Any, **kwargs: Any) -> Any:
        if self.exhausted:
            self.refused_calls += 1
            logger.warning(
                '[trajectory-budget-stop] refused model call #%d after the trajectory budget was used up: '
                'sample_id=%s traj_id=%s model=%s',
                self.refused_calls,
                self.sample_id,
                self.trajectory_id,
                kwargs.get('model'),
            )
            raise ContextLengthExceededError(
                f'trajectory budget already used up; refused model call #{self.refused_calls}'
            )

        completion = await self.client.create_completion(*args, **kwargs)
        if self._is_auxiliary(kwargs.get('model', args[1] if len(args) > 1 else None)):
            # Auxiliary requests never extend the main trajectory, so their own length limit is
            # returned to the caller instead of ending the rollout.
            return completion
        budget = _trajectory_token_limit(self.client)
        max_tokens = _configured_max_tokens(self.client)
        max_model_len = getattr(self.client, 'max_model_len', None)
        usage = getattr(completion, 'usage', None)
        total_tokens = getattr(usage, 'total_tokens', None)
        if (
            completion.choices[0].finish_reason != 'length'
            or total_tokens is None
            or budget is None
            or total_tokens < budget
        ):
            return completion

        self.exhausted = True
        logger.warning(
            '[trajectory-budget-stop] model call filled the trajectory budget; ending the rollout: '
            'sample_id=%s traj_id=%s total_tokens=%s budget=%s max_tokens=%s max_model_len=%s '
            'prompt_tokens=%s completion_tokens=%s initial_prompt_len=%s',
            self.sample_id,
            self.trajectory_id,
            total_tokens,
            budget,
            max_tokens,
            max_model_len,
            getattr(usage, 'prompt_tokens', None),
            getattr(usage, 'completion_tokens', None),
            getattr(self.client, '_initial_prompt_len', None),
        )
        raise ContextLengthExceededError(
            f'trajectory budget used up: total_tokens={total_tokens} reached budget={budget}'
        )


def _append_error(current: str | None, new_error: str) -> str:
    return f'{current}; {new_error}' if current else new_error


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _build_verifier_observability(
    test_results: Any,
    *,
    attempted: bool,
    scoring_failed: bool,
    total_duration_seconds: float | None,
    error: str | None = None,
    graded_cutoff_reason: str | None = None,
) -> dict[str, Any]:
    """Build bounded, numeric-only training metrics from verifier output.

    This is deliberately tolerant: verifier observability must never turn a valid
    score into a task failure. Full CTRF results and the bounded log tail remain in
    ``test_results``; only a compact summary crosses into rollout ``extra_fields``.
    """
    metrics: dict[str, float] = {
        'verifier_attempted': 1.0 if attempted else 0.0,
        'verifier_completed': 0.0,
        'verifier_scoring_failed': 1.0 if scoring_failed else 0.0,
        'verifier_reward_present': 0.0,
        'verifier_graded_cutoff': 1.0 if graded_cutoff_reason else 0.0,
    }
    compact: dict[str, Any] = {
        'attempted': attempted,
        'completed': False,
        'scoring_failed': scoring_failed,
    }
    if graded_cutoff_reason:
        compact['graded_cutoff_reason'] = graded_cutoff_reason
    if error:
        compact['error'] = error[:MAX_SUMMARY_ERROR_CHARS]
    if isinstance(test_results, dict) and test_results.get('skipped_evaluator_reason'):
        compact['skipped_reason'] = str(test_results['skipped_evaluator_reason'])

    total_duration = _finite_float(total_duration_seconds)
    if total_duration is not None:
        total_duration = max(total_duration, 0.0)
        metrics['verifier_total_duration_s'] = total_duration
        compact['total_duration_seconds'] = round(total_duration, 3)

    if not isinstance(test_results, dict):
        return {
            'reward_extra_info': {'extra_reward_metrics': metrics},
            'verifier_summary': compact,
        }

    verifier = test_results.get('nexus_verifier')
    if isinstance(verifier, dict):
        metrics['verifier_completed'] = 1.0
        compact['completed'] = True

        reward = _finite_float(verifier.get('reward'))
        if reward is not None:
            metrics['verifier_reward_present'] = 1.0
            compact['reward'] = reward

        test_duration = _finite_float(verifier.get('duration_seconds'))
        if test_duration is not None:
            test_duration = max(test_duration, 0.0)
            metrics['verifier_test_duration_s'] = test_duration
            compact['test_duration_seconds'] = test_duration
            if total_duration is not None:
                setup_duration = max(total_duration - test_duration, 0.0)
                metrics['verifier_setup_artifact_duration_s'] = setup_duration
                compact['setup_artifact_duration_seconds'] = round(setup_duration, 3)

        return_code = verifier.get('return_code')
        numeric_return_code = _finite_float(return_code)
        if numeric_return_code is not None:
            metrics['verifier_nonzero_return_code'] = 1.0 if numeric_return_code != 0 else 0.0
            compact['return_code'] = int(numeric_return_code)
        elif return_code is not None:
            compact['return_code'] = str(return_code)

        status = verifier.get('status')
        if status is not None:
            compact['status'] = str(status)
        log_truncated = bool(verifier.get('log_truncated', False))
        metrics['verifier_log_truncated'] = 1.0 if log_truncated else 0.0
        compact['log_truncated'] = log_truncated

    # A CTRF report keeps its counts under ``results.summary``; a bare ``summary`` is still read
    # for verifiers that write one.
    ctrf_results = test_results.get('results')
    ctrf_summary = ctrf_results.get('summary') if isinstance(ctrf_results, dict) else None
    if not isinstance(ctrf_summary, dict):
        ctrf_summary = test_results.get('summary')
    if isinstance(ctrf_summary, dict):
        compact_tests: dict[str, int] = {}
        for source_key, metric_key in (
            ('tests', 'verifier_tests_total'),
            ('passed', 'verifier_tests_passed'),
            ('failed', 'verifier_tests_failed'),
            ('pending', 'verifier_tests_pending'),
            ('skipped', 'verifier_tests_skipped'),
            ('other', 'verifier_tests_other'),
        ):
            value = _finite_float(ctrf_summary.get(source_key))
            if value is None:
                continue
            value = max(value, 0.0)
            metrics[metric_key] = value
            compact_tests[source_key] = int(value)
        total = metrics.get('verifier_tests_total')
        passed = metrics.get('verifier_tests_passed')
        if total is not None and passed is not None and total > 0:
            metrics['verifier_tests_pass_ratio'] = passed / total
        if compact_tests:
            compact['tests'] = compact_tests

    return {
        'reward_extra_info': {'extra_reward_metrics': metrics},
        'verifier_summary': compact,
    }


def _record_reward_time(start_time: float) -> None:
    span_id = get_current_span_id()
    if span_id:
        update_span(span_id, attributes={'hytuner.reward_time': time.time() - start_time})


@task_registry.register('science_agent_task')
async def run_science_agent_task(task_input: ScienceAgentTaskInput) -> ScienceAgentTaskOutput:
    ctx = require_context()
    if not ctx.runtime_provider:
        raise RuntimeError('science_agent_task requires a runtime provider.')
    agent_name = _resolve_agent_name(task_input.agent_name, ctx.agent_name)

    evaluation_method = _normalize_evaluation_method(task_input.evaluation_method)
    prompt = _build_agent_prompt(
        question=task_input.question,
        evaluation_method=evaluation_method,
        agent_name=agent_name,
        working_dir=task_input.working_dir,
    )

    runtime_name = task_input.runtime_name
    if task_input.image:
        ctx.runtime_provider.configure(runtime_name, image=task_input.image)

    runtime = await ctx.get_runtime(runtime_name)
    logger.debug('Runtime initialized: %s', runtime)
    runtime_id = ctx.get_runtime_id(runtime_name)
    span_id = get_current_span_id()
    if span_id and runtime_id:
        update_span(span_id, attributes={'hytuner.runtime_id': runtime_id})

    agent_output: Any = None
    answer: str | None = None if evaluation_method in SANDBOX_EVALUATION_METHODS else ''
    score = EVALUATION_FAILURE_SCORE
    test_results: dict[str, Any] | None = None
    judge_error: str | None = None
    error: str | None = None
    agent_error: str | None = None
    agent_error_type: str | None = None
    agent_error_traceback: str | None = None
    exit_status_str = 'unknown'
    canonical_reason: str | None = None
    iterations = 0
    raw_terminal_output: str | None = None
    skip_evaluation_reason: str | None = None
    verifier_attempted = False
    verifier_scoring_failed = False
    verifier_total_duration_seconds: float | None = None
    graded_cutoff_reason: str | None = None

    try:
        # This is the hard isolation boundary: no verifier script is present
        # while either Terminus2 or mini-swe-agent is running.
        await hide_verifier_from_agent(runtime)

        budget_stop = _TrajectoryBudgetStop.install(ctx, task_input, agent_name)
        try:
            agent_run = task_input.run_agent(
                user_prompt=prompt,
                working_dir=task_input.working_dir,
                wall_time_limit_seconds=_agent_wall_time_limit(task_input),
            )
            if agent_name == 'hycode':
                # HyCode's own timeout already excludes partial-rollout pauses; this backstop has to
                # as well, or it would fire first with active time left.
                agent_output = await _await_agent_in_active_time(
                    agent_run, task_input.agent_timeout, getattr(ctx, 'timeout_handle', None)
                )
            else:
                agent_output = await asyncio.wait_for(agent_run, timeout=task_input.agent_timeout)
        except asyncio.TimeoutError as exc:
            exit_status_str = 'timeout'
            agent_error = f'Agent timed out after {task_input.agent_timeout}s: {exc}'
            agent_error_type = type(exc).__name__
            agent_error_traceback = str(exc)
            error = agent_error
            if _grades_cut_off_artifacts(task_input, evaluation_method, TerminationReason.TASK_TIMEOUT.value):
                # The row asked for the artifacts of a cut-off run to be graded, and this is the
                # harshest cut of all: run_agent never returned, so the cancelled agent may still
                # be writing while the verifier reads. What gets graded is whatever it had reached.
                graded_cutoff_reason = TerminationReason.TASK_TIMEOUT.value
            else:
                score = AGENT_FAILURE_SCORE
                skip_evaluation_reason = 'AGENT_TIMEOUT'
            # The cancelled agent never published a reason, so without this the rollout
            # reaches HyTuner as ENV_DONE. Reporting it stays unconditional: grading the
            # artifacts says what the run produced, not that it finished.
            span_id = get_current_span_id()
            if span_id:
                update_span(span_id, attributes={'hytuner.termination_reason': TerminationReason.TASK_TIMEOUT.value})
        except Exception as exc:
            logger.exception('Agent execution failed: %s', exc)
            exit_status_str = 'agent_error'
            agent_error = f'Agent execution failed: {exc}'
            agent_error_type = type(exc).__name__
            agent_error_traceback = str(exc)[:MAX_ERROR_TRACEBACK_CHARS]
            error = agent_error
            score = AGENT_FAILURE_SCORE
            skip_evaluation_reason = 'AGENT_ERROR'
            # Same hole the timeout branch above closes: an agent that raised before publishing a
            # reason leaves the span saying nothing, and HyTuner reads that as ENV_DONE -- which it
            # does not mask, so AGENT_FAILURE_SCORE would train as a real -300 on a rollout that
            # never ran. UNHANDLED_ERROR is what a task-side failure is called elsewhere.
            span_id = get_current_span_id()
            if span_id:
                update_span(span_id, attributes={'hytuner.termination_reason': TerminationReason.UNHANDLED_ERROR.value})
        else:
            exit_status = getattr(agent_output, 'exit_status', None)
            if exit_status is None:
                # CLI agents such as HyCode carry no exit_status and report termination_reason.
                exit_status = getattr(agent_output, 'termination_reason', None)
            exit_status_str = getattr(exit_status, 'value', str(exit_status or 'unknown'))
            canonical_reason = _canonical_agent_reason(agent_name, agent_output)
            if canonical_reason == TerminationReason.ENV_DONE.value and _hycode_ended_on_unparsed_tool_call(
                ctx, agent_output
            ):
                logger.warning(
                    'HyCode ended on a reply whose tool call the rollout parser could not read; reporting %s: '
                    'sample_id=%s',
                    TerminationReason.FORMAT_ERROR.value,
                    task_input.sample_id,
                )
                canonical_reason = TerminationReason.FORMAT_ERROR.value
                exit_status_str = canonical_reason.lower()
                span_id = get_current_span_id()
                if span_id:
                    update_span(
                        span_id,
                        attributes={
                            'hytuner.termination_reason': canonical_reason,
                            'hytuner.cleared_termination_reason': TerminationReason.ENV_DONE.value,
                        },
                    )
            # HyCode counts turns instead of iterations; without it every HyCode row reports 0.
            iterations = getattr(
                agent_output,
                'iterations',
                getattr(agent_output, 'n_iterations', getattr(agent_output, 'num_turns', 0)) or 0,
            )
            raw_terminal_output = getattr(agent_output, 'raw_terminal_output', None)
            agent_error = getattr(agent_output, 'error', None)
            agent_error_type = getattr(agent_output, 'error_type', None)
            agent_error_traceback = getattr(agent_output, 'error_traceback', None)
            cutoff_reason = _cutoff_termination_reason(agent_output)
            if cutoff_reason is not None:
                if _grades_cut_off_artifacts(task_input, evaluation_method, cutoff_reason):
                    logger.info(
                        'Agent output is %s; verifier_score_truncated is set, grading its artifacts', cutoff_reason
                    )
                    graded_cutoff_reason = cutoff_reason
                else:
                    logger.info('Agent output is %s; skip evaluator and assign score=0', cutoff_reason)
                    score = 0.0
                    answer = None if evaluation_method in SANDBOX_EVALUATION_METHODS else ''
                    skip_evaluation_reason = cutoff_reason
                error = _append_error(error, f'agent_exit_status:{exit_status_str}')
                # Every supported agent already publishes this on its own span, but an
                # agent that only returns ``exit_status`` would otherwise reach HyTuner
                # as ENV_DONE.
                span_id = get_current_span_id()
                if span_id:
                    update_span(span_id, attributes={'hytuner.termination_reason': cutoff_reason})
            elif not _agent_succeeded(agent_name, canonical_reason or exit_status_str):
                error = f'agent_exit_status:{exit_status_str}'
                if agent_error:
                    error = f'{error}; {agent_error}'
        finally:
            if budget_stop is not None:
                budget_stop.restore(ctx, exit_status_str)

        if skip_evaluation_reason:
            logger.info(
                'Skipping %s evaluation because agent stopped with %s', evaluation_method, skip_evaluation_reason
            )
            test_results = {
                'skipped_evaluator_reason': skip_evaluation_reason,
                'exit_status': exit_status_str,
            }
        elif evaluation_method in SANDBOX_EVALUATION_METHODS:
            start_time = time.time()
            verifier_attempted = evaluation_method == 'verifier_image'
            try:
                if evaluation_method == 'unit_cases':
                    # This is the only point at which task-owned tests enter the
                    # agent sandbox, and it always occurs after run_agent has returned.
                    score, test_results = await unit_cases_evaluate(
                        runtime_name=runtime_name,
                        task_dir=task_input.task_dir,
                        test_timeout=task_input.test_timeout,
                        verifier_env=task_input.verifier_env,
                        verifier_env_refs=task_input.verifier_env_refs,
                    )
                else:
                    # The declared artifacts enter only the separate verifier runtime after
                    # run_agent has returned; the tests come from the verifier image itself.
                    score, test_results = await verifier_image_evaluate(
                        agent_runtime_name=runtime_name,
                        verifier_runtime_name=task_input.verifier_runtime_name,
                        sample_id=task_input.sample_id,
                        verifier_image=task_input.verifier_image,
                        verifier_dind=task_input.verifier_dind,
                        verifier_network_access=task_input.verifier_network_access,
                        verifier_artifacts_read_only=task_input.verifier_artifacts_read_only,
                        verifier_mounts=task_input.verifier_mounts,
                        verifier_allowed_protected_mounts=task_input.verifier_allowed_protected_mounts,
                        verifier_env=task_input.verifier_env,
                        verifier_env_refs=task_input.verifier_env_refs,
                        test_timeout=task_input.test_timeout,
                        docker_timeout=task_input.docker_timeout,
                    )
            except Exception as exc:
                logger.exception('%s evaluation failed: %s', evaluation_method, exc)
                judge_error = f'{evaluation_method}_error:{exc}'
                score = EVALUATION_FAILURE_SCORE
                error = _append_error(error, judge_error)
                verifier_scoring_failed = evaluation_method == 'verifier_image'
                span_id = get_current_span_id()
                if span_id:
                    update_span(span_id, attributes={'hytuner.termination_reason': 'REWARD_ERROR'})
            finally:
                if evaluation_method == 'verifier_image':
                    verifier_total_duration_seconds = max(time.time() - start_time, 0.0)
                _record_reward_time(start_time)
        else:
            try:
                answer = await _extract_answer(
                    agent_name=agent_name,
                    agent_output=agent_output,
                    runtime=runtime,
                    evaluation_method=evaluation_method,
                    working_dir=task_input.working_dir,
                    test_timeout=task_input.test_timeout,
                )
                if answer:
                    score = 0.0
            except Exception as exc:
                logger.exception('Answer extraction failed for agent %s: %s', agent_name, exc)
                error = _append_error(error, f'answer_extraction_error:{exc}')
                score = EVALUATION_FAILURE_SCORE
    finally:
        # Host-side LLM/RM evaluation does not need the sandbox. Release it as
        # soon as the post-agent answer/test adapter has finished.
        try:
            await ctx.runtime_provider.release(name=runtime_name)
        except Exception as exc:
            logger.debug('Failed to release runtime %s: %s', runtime_name, exc)

    if skip_evaluation_reason:
        logger.info(
            'Skipping host-side %s evaluation because agent stopped with %s', evaluation_method, skip_evaluation_reason
        )
    elif evaluation_method == 'llm_judge':
        start_time = time.time()
        try:
            evaluator = build_evaluator(evaluation_method)
            reference = load_ref_answer(reference=task_input.ref_answer, task_dir=task_input.task_dir)
            score, details = await evaluator.evaluate_with_details(
                prediction=answer or '',
                reference=reference,
                question=task_input.question,
            )
            score = max(0.0, min(1.0, float(score)))
            test_results = details.to_dict() if details else {}
        except Exception as exc:
            logger.exception('llm_judge evaluation failed: %s', exc)
            judge_error = f'llm_judge_error:{exc}'
            score = EVALUATION_FAILURE_SCORE
            error = _append_error(error, judge_error)
        finally:
            _record_reward_time(start_time)
    elif evaluation_method == 'rmsys' and agent_name == 'hycode' and not answer:
        # An agent that produced nothing to judge failed at the task, not at evaluation.
        # EVALUATION_FAILURE_SCORE would drop it below the trainer's filter threshold and
        # delete the sample, so the policy would never learn that ending empty is bad; 0
        # keeps it in the batch as a usable negative.
        #
        # Scoped to hycode. The other agents reach an empty answer through their own exit
        # statuses and keep the EVALUATION_FAILURE_SCORE they were tuned against.
        logger.info('No answer to judge; scoring 0 instead of calling the reward system')
        score = 0.0
        test_results = {'skipped_evaluator_reason': 'EMPTY_ANSWER', 'exit_status': exit_status_str}
    elif evaluation_method == 'rmsys':
        start_time = time.time()
        try:
            if not answer:
                raise ValueError('rmsys evaluation requires a non-empty answer')
            reference = load_ref_answer(reference=task_input.ref_answer, task_dir=task_input.task_dir)
            score = await rmsys_evaluate(
                model_answer=answer,
                reference=reference,
                question=task_input.question,
                router_id=task_input.reward_system_router_id,
                user_token=task_input.reward_system_user_token,
                exp_id=task_input.reward_system_exp_id,
                extra_info=task_input.reward_system_extra,
            )
        except Exception as exc:
            logger.exception('rmsys evaluation failed: %s', exc)
            judge_error = f'rmsys_error:{exc}'
            score = EVALUATION_FAILURE_SCORE
            error = _append_error(error, judge_error)
        finally:
            _record_reward_time(start_time)

    # What HyTuner will see: a cut-off row carries the reason task.py published for it, and every
    # other row carries whatever the agent put on its own span -- HyCode's canonical
    # termination_reason, or for other agents what exit_status_str holds.
    #
    # Compared without case folding, on purpose. Only an agent that already speaks the canonical
    # vocabulary can be rewritten this way -- HyCode does, while Mini-SWE reports short lowercase
    # statuses whose upper-cased spelling is a different reason than the one it published
    # ('timeout' is TASK_TIMEOUT, 'error' is UNHANDLED_ERROR). Matching those would relabel a
    # rollout on a name collision. Failing closed costs nothing: the only rewritable reason
    # Mini-SWE can reach is TASK_TIMEOUT, and it arrives through the cut-off branch already mapped.
    reported_reason = graded_cutoff_reason or canonical_reason or exit_status_str
    if (
        evaluation_method == 'verifier_image'
        and reported_reason in _FULL_PASS_CLEARS_REASON
        and _finite_float(score) == FULL_VERIFIER_SCORE
    ):
        # Every test passed, so the artifacts say the work was finished. Left alone the reason
        # would have HyTuner mask the row and throw this grade away.
        logger.info(
            '[verifier-summary] sample_id=%s graded %s at full marks; reporting %s',
            task_input.sample_id,
            reported_reason,
            TerminationReason.ENV_DONE.value,
        )
        span_id = get_current_span_id()
        if span_id:
            update_span(
                span_id,
                attributes={
                    'hytuner.termination_reason': TerminationReason.ENV_DONE.value,
                    # The reason this replaced. Without it the dumped trajectory says ENV_DONE and
                    # nothing else, so a run cannot tell an ordinary success from one this rule
                    # rescued, and cannot be audited for how often it fires.
                    'hytuner.cleared_termination_reason': reported_reason,
                },
            )

    extra_fields: dict[str, Any] = {}
    if evaluation_method == 'verifier_image':
        try:
            extra_fields = _build_verifier_observability(
                test_results,
                attempted=verifier_attempted,
                scoring_failed=verifier_scoring_failed,
                total_duration_seconds=verifier_total_duration_seconds,
                error=judge_error,
                graded_cutoff_reason=graded_cutoff_reason,
            )
            logger.info(
                '[verifier-summary] sample_id=%s %s',
                task_input.sample_id,
                json.dumps(extra_fields.get('verifier_summary', {}), sort_keys=True, default=str),
            )
        except Exception:
            # Metrics and logging are strictly best effort and must never affect reward.
            logger.warning(
                'Could not build verifier observability for sample_id=%s', task_input.sample_id, exc_info=True
            )
            extra_fields = {}

    return ScienceAgentTaskOutput(
        score=score,
        answer=answer,
        ref_answer=task_input.ref_answer,
        image=task_input.image,
        question=task_input.question,
        sample_id=task_input.sample_id,
        raw_terminal_output=raw_terminal_output,
        exit_status=exit_status_str,
        iterations=iterations,
        test_results=test_results,
        judge_error=judge_error,
        error=error,
        agent_error=agent_error,
        agent_error_type=agent_error_type,
        agent_error_traceback=agent_error_traceback,
        extra_fields=extra_fields,
    )


__all__ = ['ScienceAgentTaskInput', 'ScienceAgentTaskOutput', 'run_science_agent_task']
