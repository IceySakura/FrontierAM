"""Context Agent General Task — orchestrates the path-native rollout for Nexus.

This task:
1. Receives ``context_files`` (URIs only — no content) and ``context_question``
2. Runs the ``context_agent_general`` agent, which fetches / converts / reads
   whatever it decides it needs using the seven path-native tools
3. Scores the answer with the harness's own evaluator suite (:mod:`.eval`)

Scoring is **routed by ``dataset_name``**, not by which reference field the row
happens to carry. That is the whole point of the ported suite: three grading
styles are in play and mixing them is misleading.

* *rule-based* (no judge model, deterministic, free) — ``GraphWalks`` computes
  set F1 over the nodes in a strict ``Final Answer: [...]`` line,
  ``LongBenchV2`` matches the official answer-letter regex, ``MRCR_*`` is a
  prefix gate then a difflib ratio, and ``oolong`` gives partial credit for
  numeric distance.
* *reference-graded* — ``AALCR`` is correct against **any** of N official
  answers, ``LongBenchPro`` demands **all** components, ``BrowseCompLongContext``
  uses the official grader prompt, ``TableMixV2`` / ``DocLonger`` use the generic
  answer judge.
* *rubric-graded* — ``CLB`` / ``CLBLife`` / ``Pluto20`` / ``SurgeOODV21`` /
  ``surge_docs`` / ``GDPpdf``, and ``TableMixLonger`` with its own checklist
  template.
* *deliverable-graded* — ``GDPval`` receives partial credit equal to the
  fraction of rubric criteria passed, and a Judge agent opens the actual files
  in the Runtime ``workspace/`` instead of trusting final-message claims.

An unregistered ``dataset_name`` falls back to inferring from the row: rubrics
win over an answer when both are present. ``judge_backend`` forces one judge
regardless, which is how you grade a corpus the suite has no entry for.

``AALCR`` and ``LongBenchPro`` retain their benchmark-specific prompts and
answer-list semantics while using the routed Judge client configured for the
run.

The score is returned in ``WithScore.score``. Under HyTuner RL training,
``nexus.runner.hytuner`` reads ``task_output.score`` as ``reward_score``; when
it is not None, HyTuner's own reward_loop is skipped and this score flows
straight into the RL advantage computation.

AgentIF rows additionally carry ``constraint_rubrics``. Their existing routed
evaluator remains the answer axis. Only a fully correct answer invokes the
fixed trajectory-constraint judge, producing rewards 1.0 (both axes pass),
``constraint_fail_reward`` (answer passes but any constraint fails; 0.5 by
default, 0.0 for a binary reward), or 0.0 (answer fails).
"""

import logging
import uuid

from typing import Any

from pydantic import Field

from nexus.core import (
    Registry,
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.agents.context_agent.general_lang import resolve_lang
from nexus.extensions.tasks.context_agent.termination import report_task_termination

from .bad_patterns import ToolCallBadPatternPolicy, apply_bad_pattern_score
from .constraint_scorer import score_constraints
from .eval import STYLES, evaluate
from .eval._judge_llm import resolve_judge_llm
from .output import agent_termination_reason as _agent_termination_reason
from .output import agent_tool_error_extra_fields, apply_tool_name_resolution_score, judge_skip_reason
from .output import extract_final_answer as _extract_final_answer
from .task_files_auth import bind_task_wedata_key
from .workspace_artifacts import DEFAULT_MAX_BYTES, DEFAULT_MAX_FILES, persist_runtime_workspace


logger = logging.getLogger(__name__)

# Ensure the agent is registered.
Registry.import_module('nexus.extensions.agents.context_agent.general_agent')

# ``eval/judge.py`` and ``eval/answer_judge.py`` return this rationale prefix
# when the judge could not be reached or parsed after all retries. They pair it
# with ``score: 0``, which for RL is indistinguishable from a genuine zero — so
# it is detected here and surfaced instead of being averaged in silently.
_JUDGE_FAILED_PREFIX = 'Judge failed'


class ContextAgentGeneralTaskInput(TaskRunInput, WithAgentRunner):
    """Input for the general context agent task.

    Attributes:
        context_question: The question to answer.
        context_files: The task's file list — ``{"name", "uri"}`` items where
            ``uri`` is an http(s) URL, local path, or ``task-files://`` path.
            Contents are NOT part of the row; the agent fetches and reads what
            it wants.
        context_dialogue: Optional multi-turn history. When set, the final user
            turn is the question sent to the policy; ``context_question`` stays
            as evaluator metadata/fallback and is not appended again.
        dataset_name: **The scoring routing key.** One of the names in
            ``eval.STYLES`` (e.g. ``LongBenchV2``, ``AALCR``, ``CLB``,
            ``surge_docs``). Note the benchmark names and the data filenames
            disagree on purpose: ``CLB_200`` carries ``CLB``, ``doc_longer_406``
            carries ``DocLonger``, ``table_longer_66`` carries
            ``TableMixLonger``. Read from ``metadata.dataset_name`` when the row
            nests it there.
        metadata: Passed through to the evaluators verbatim. Carries the
            per-benchmark keys they need — ``random_string_to_prepend`` (MRCR),
            ``oolong_answer_type`` / ``oolong_task*`` (oolong), ``checklist`` /
            ``judge_template`` / ``final_question`` (TableMixLonger),
            ``sub_category``.
        answer: Reference answer. A ``list[str]`` for AALCR (any match counts)
            and LongBenchPro (all components required), a ``str`` elsewhere.
        rubrics: Criteria list for the rubric-graded benchmarks.
        checklist / judge_template: TableMixLonger's own template inputs; also
            accepted nested under ``metadata``.
        evaluate: Whether to run the routed evaluator after the Agent returns.
            Set to ``False`` for rollout-only jobs that need the answer and
            trajectory without constructing or calling a Judge client.
        judge_backend: ``auto`` (route on ``dataset_name``), ``rubrics``,
            ``rubric_accuracy``, ``answer``, ``deliverable`` or ``rm_system``
            to force one judge.
        reward_system_router_id / reward_prompt / ref_answer / rm_extra:
            RewardService inputs used only by the RL ``rm_system`` backend.
        constraint_rubrics: AgentIF SP constraints judged from the full public
            trajectory after (and only after) the answer axis passes.
        constraint_fail_reward: Reward when the answer passes but a constraint
            fails. ``0.0`` makes the AgentIF reward binary.
        WEDATA_KEY: Optional task-scoped Wedata credential for remote
            TaskFiles. It takes precedence over the process environment and
            default key file, and is excluded from the serialized Task input.
        artifact_root: Optional Host directory under which the Runtime
            ``workspace/`` is persisted as ``<artifact_root>/<run_id>/workspace``.
        system_messages: Optional task-side persona, appended to the agent's
            own system prompt.
        language: Optional pre-tagged language ("zh" / "en"; "cn" accepted).
        files_base: Base directory for resolving relative local input paths.
        working_dir: Runtime workspace passed explicitly to the Agent. Defaults
            to ``/tmp/ctxagent_general/<run_id>``.
        max_iterations / max_tool_rounds / max_parallel_calls /
        file_search_top_k / file_search_chunk_lines /
        python_timeout / curl_timeout / curl_max_bytes / system_prompt_lang /
        system_prompt_path / enabled_tools:
            Forwarded to the agent. Most are also overridable via
            ``CONTEXTAGENT_GENERAL_*`` env vars set by the launch script.
    """

    context_question: str
    context_files: list = Field(default_factory=list)
    context_dialogue: list[dict[str, Any]] | None = None
    dataset_name: str | None = None
    metadata: dict = Field(default_factory=dict)
    answer: str | list[str] | None = None
    rubrics: list[str] | None = None
    checklist: list[str] | None = None
    judge_template: str | None = None
    evaluate: bool = True
    tool_call_bad_pattern_policy: ToolCallBadPatternPolicy | None = None
    zero_score_on_tool_name_resolution_error: bool = False
    judge_backend: str = 'auto'
    reward_system_router_id: str | None = None
    reward_prompt: list[dict[str, Any]] | None = None
    ref_answer: str | None = None
    rm_extra: dict[str, Any] | None = None
    constraint_rubrics: list[dict[str, Any]] = Field(default_factory=list)
    constraint_fail_reward: float = Field(default=0.5, ge=0.0, le=1.0)
    WEDATA_KEY: str | dict[str, Any] | None = Field(default=None, exclude=True, repr=False)
    artifact_root: str | None = None
    artifact_max_files: int = DEFAULT_MAX_FILES
    artifact_max_bytes: int = DEFAULT_MAX_BYTES
    system_messages: list[dict] | str | None = None
    language: str | None = None
    files_base: str | None = None
    working_dir: str | None = None
    max_iterations: int = 20
    max_tool_rounds: int | None = None
    max_parallel_calls: int = 0
    file_search_top_k: int = 5
    file_search_chunk_lines: int = 50
    python_timeout: int = 120
    curl_timeout: int = 120
    curl_max_bytes: int = 200 * 1024 * 1024
    system_prompt_lang: str | None = None
    system_prompt_path: str | None = None
    enabled_tools: list[str] | None = None


class ContextAgentGeneralTaskOutput(TaskRunOutput, WithScore):
    """Output from the general context agent task.

    Attributes:
        final_answer: The agent's extracted final answer.
        score: Always a ``float`` in [0, 1] for this pipeline. 0.0 covers an
            empty answer, a row with nothing to grade against, AND a judge
            outage. We deliberately do NOT emit ``None``: under an importlib
            reward_loop a ``None`` reward on the SUCCESS path reaches
            ``overlong_mask_weighted``, which cannot handle the nexus
            placeholder row (``reward_model`` is a JSON string,
            ``reward_style="nexus"`` is unsupported) and would raise, killing
            the whole training run.
        grading_style: ``rule`` / ``reference`` / ``rubric`` — which style
            produced the score. Averaging across styles is misleading, so this
            is reported rather than left to be inferred from the dataset name.
        judge_unavailable: True when the score is a degraded 0.0 rather than a
            graded one. Filter on this before reading a mean.
        rationale: The evaluator's complete explanation.
        requirement_status: Per-rubric yes/no list, when the evaluator produced
            one.
        graded_on: ``deliverables`` for the GDPval agentic Judge, or
            ``final_message`` when a deliverable Judge had to fall back.
        criteria_passed / criteria_total: Per-rubric counts returned by rubric
            evaluators. GDPval's Agentic Judge and ordinary text rubric judges
            both provide them.
        judge_rounds: Agentic Judge model rounds.
        criteria: Per-criterion agentic Judge verdicts and concrete evidence.
        evaluation_skipped: True when evaluation is disabled or skipped because
            the Agent hit an input/output token limit, or exhausted its tool
            rounds without producing a final answer. In that mode ``score`` is
            the required ``0.0`` placeholder and is not a Judge result.
        workspace_artifact: Persistent Host-side mapping for the downloaded
            Runtime ``workspace/``, when ``artifact_root`` is configured.
        termination_reason: ``REWARD_ERROR`` when grading infrastructure fails,
            allowing HyTuner to mask the rollout instead of learning from a
            fabricated zero.
        answer_score / constraint_score: Independent audit axes. The final
            AgentIF reward is 1.0 when both pass, ``constraint_fail_reward``
            when only the answer passes, and 0.0 when the answer fails. ``constraint_score`` is
            ``None`` when no constraint grading was needed.
    """

    extra_fields: dict[str, Any] = Field(default_factory=dict)
    final_answer: str | None = None
    score: float = 0.0
    answer_score: float = 0.0
    constraint_score: float | None = None
    answer_judge: dict[str, Any] = Field(default_factory=dict)
    constraint_judge: dict[str, Any] = Field(default_factory=dict)
    grading_style: str | None = None
    judge_unavailable: bool = False
    evaluation_skipped: bool = False
    rationale: str | None = None
    requirement_status: list = Field(default_factory=list)
    graded_on: str | None = None
    criteria_passed: int | None = None
    criteria_total: int | None = None
    judge_rounds: int | None = None
    criteria: list[dict[str, Any]] = Field(default_factory=list)
    workspace_artifact: dict[str, Any] | None = None
    termination_reason: str = 'ENV_DONE'


def _has_items(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return len(value) > 0
    except TypeError:
        return False


def _build_eval_task(task_input: ContextAgentGeneralTaskInput) -> tuple[dict[str, Any], str]:
    """Assemble the dict the evaluators read, and resolve the routing key.

    ``dataset_name`` may arrive top-level or nested under ``metadata``; the
    evaluators only look at the nested one, so it is normalised into there.
    """
    metadata = dict(task_input.metadata or {})
    dataset_name = str(metadata.get('dataset_name') or task_input.dataset_name or '')
    if dataset_name:
        metadata['dataset_name'] = dataset_name

    eval_task: dict[str, Any] = {
        'metadata': metadata,
        'context_question': task_input.context_question,
        'context_files': list(task_input.context_files or []),
        'answer': task_input.answer,
        'rubrics': list(task_input.rubrics or []),
        'reward_system_router_id': task_input.reward_system_router_id,
        'reward_prompt': list(task_input.reward_prompt or []),
        'ref_answer': task_input.ref_answer,
        'rm_extra': dict(task_input.rm_extra or {}),
    }
    # TableMixLonger reads these from either level; pass whichever the row gave.
    if task_input.checklist is not None:
        eval_task['checklist'] = task_input.checklist
    if task_input.judge_template is not None:
        eval_task['judge_template'] = task_input.judge_template
    return eval_task, dataset_name


def _normalize_score(raw: Any) -> float | None:
    """Coerce an evaluator score to a float in [0, 1], or None if unusable."""
    if raw is None or isinstance(raw, bool):
        return float(raw) if isinstance(raw, bool) else None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return min(1.0, max(0.0, value))


@task_registry.register('context_agent_general_task')
@report_task_termination
async def run_context_agent_general_task(
    task_input: ContextAgentGeneralTaskInput,
) -> ContextAgentGeneralTaskOutput:
    """Run the general context agent task and score it with the eval suite."""
    ctx = require_context()
    if ctx.runtime_provider is None:
        raise RuntimeError('context_agent_general_task requires a runtime provider')
    ctx.task_config.pop('WEDATA_KEY', None)
    ctx.task_files = bind_task_wedata_key(ctx.task_files, task_input.WEDATA_KEY)

    lang = resolve_lang(task_input.language, task_input.context_question)
    eval_task, dataset_name = _build_eval_task(task_input)
    working_dir = task_input.working_dir or f'/tmp/ctxagent_general/{ctx.run_id or uuid.uuid4().hex}'
    ctx.task_config['working_dir'] = working_dir

    logger.debug(
        'Running context_agent_general_task: lang=%s dataset=%s files=%d working_dir=%s question="%s..."',
        lang,
        dataset_name or '(unrouted)',
        len(task_input.context_files or []),
        working_dir,
        (task_input.context_question or '')[:80],
    )

    agent_output = await task_input.run_agent(
        user_prompt=task_input.context_question,
        context_files=task_input.context_files,
        context_dialogue=task_input.context_dialogue,
        system_messages=task_input.system_messages,
        language=lang,
        files_base=task_input.files_base,
        working_dir=working_dir,
        max_iterations=task_input.max_iterations,
        max_tool_rounds=task_input.max_tool_rounds,
        max_parallel_calls=task_input.max_parallel_calls,
        file_search_top_k=task_input.file_search_top_k,
        file_search_chunk_lines=task_input.file_search_chunk_lines,
        python_timeout=task_input.python_timeout,
        curl_timeout=task_input.curl_timeout,
        curl_max_bytes=task_input.curl_max_bytes,
        system_prompt_lang=task_input.system_prompt_lang,
        system_prompt_path=task_input.system_prompt_path,
        enabled_tools=task_input.enabled_tools,
    )

    def build_output(**kwargs: Any) -> ContextAgentGeneralTaskOutput:
        output = ContextAgentGeneralTaskOutput(**kwargs)
        output.score, output.extra_fields = apply_bad_pattern_score(
            output.score,
            agent_output,
            task_input.tool_call_bad_pattern_policy,
            extra_fields=output.extra_fields,
            termination_reason=output.termination_reason,
        )
        return output

    final_answer = _extract_final_answer(agent_output)
    agent_termination_reason = _agent_termination_reason(agent_output)
    style = STYLES.get(dataset_name)
    staged_files = list(getattr(agent_output, 'staged_files', None) or [])
    workspace_artifact: dict[str, Any] | None = None
    if task_input.artifact_root:
        workspace_artifact = await persist_runtime_workspace(
            ctx,
            working_dir=working_dir,
            artifact_root=task_input.artifact_root,
            context_files=list(task_input.context_files or []),
            staged_files=staged_files,
            task_id=str((task_input.metadata or {}).get('task_id') or '') or None,
            max_files=task_input.artifact_max_files,
            max_bytes=task_input.artifact_max_bytes,
        )
    if not task_input.evaluate:
        logger.debug('context_agent_general_task: evaluation disabled; returning rollout output only')
        return build_output(
            extra_fields=agent_tool_error_extra_fields(agent_output),
            final_answer=final_answer,
            score=0.0,
            evaluation_skipped=True,
            rationale='Evaluation disabled',
            workspace_artifact=workspace_artifact,
            termination_reason=agent_termination_reason,
        )

    skip_reason = judge_skip_reason(agent_output, final_answer)
    if skip_reason is not None:
        logger.info('context_agent_general_task: %s score=0.0', skip_reason)
        return build_output(
            extra_fields=agent_tool_error_extra_fields(agent_output),
            final_answer=final_answer,
            score=0.0,
            grading_style=style,
            judge_unavailable=False,
            evaluation_skipped=True,
            rationale=skip_reason,
            workspace_artifact=workspace_artifact,
            termination_reason=agent_termination_reason,
        )

    deliverable_evaluation = task_input.judge_backend == 'deliverable' or (
        task_input.judge_backend == 'auto' and dataset_name == 'GDPval'
    )
    if (final_answer is None or not str(final_answer).strip()) and not deliverable_evaluation:
        # Agent produced no answer — genuine 0.0 (a valid training signal), and
        # no judge call to pay for.
        logger.info('context_agent_general_task: no answer produced; score=0.0')
        return build_output(
            extra_fields=agent_tool_error_extra_fields(agent_output),
            final_answer=final_answer,
            score=0.0,
            grading_style=style,
            rationale='No model output',
            workspace_artifact=workspace_artifact,
            termination_reason=agent_termination_reason,
        )

    if not _has_items(task_input.rubrics) and not _has_items(task_input.answer) and dataset_name not in STYLES:
        logger.warning(
            'context_agent_general_task: nothing to grade against (dataset=%r has no evaluator '
            'and the row carries neither rubrics nor an answer); score=0.0',
            dataset_name,
        )
        return build_output(
            extra_fields=agent_tool_error_extra_fields(agent_output),
            final_answer=final_answer,
            score=0.0,
            judge_unavailable=True,
            rationale='nothing to grade against',
            workspace_artifact=workspace_artifact,
            termination_reason=agent_termination_reason,
        )

    # The routed judge is never the policy: see eval/_judge_llm.py. It is built
    # lazily, so the five rule-based benchmarks never open a client — and a
    # missing judge credential cannot fail a benchmark that needs no judge.
    judge_llm = resolve_judge_llm(ctx)

    try:
        result = await evaluate(
            str(final_answer or ''),
            eval_task,
            judge_llm,
            backend=task_input.judge_backend,
            runtime_context=ctx,
            working_dir=working_dir,
            staged_files=staged_files,
            artifact_max_files=task_input.artifact_max_files,
            artifact_max_bytes=task_input.artifact_max_bytes,
        )
    except Exception:
        # An evaluator raising is an infrastructure failure, not a zero answer.
        logger.exception(
            'context_agent_general_task: evaluator raised for dataset=%r; degrading to 0.0',
            dataset_name,
        )
        return build_output(
            extra_fields=agent_tool_error_extra_fields(agent_output),
            final_answer=final_answer,
            score=0.0,
            grading_style=style,
            judge_unavailable=True,
            rationale='evaluator raised',
            workspace_artifact=workspace_artifact,
            termination_reason='REWARD_ERROR',
        )

    rationale = str(result.get('rationale') or '')
    answer_score = _normalize_score(result.get('score'))
    reward_error = bool(result.get('reward_error'))
    judge_unavailable = answer_score is None or rationale.startswith(_JUDGE_FAILED_PREFIX)
    answer_judge_valid = not (reward_error or judge_unavailable)

    if answer_score is None or judge_unavailable:
        # Must NOT stay None on the success path — see the output docstring.
        logger.warning(
            'context_agent_general_task: judge UNSCORED (dataset=%r, rationale=%r) — '
            'degrading to 0.0 to avoid crashing the reward_loop',
            dataset_name,
            rationale[:200],
        )
        answer_score = 0.0

    score = float(answer_score or 0.0)
    constraint_score: float | None = None
    constraint_result: dict[str, Any] = {}
    if task_input.constraint_rubrics and not (reward_error or judge_unavailable):
        if answer_score == 1.0:
            try:
                constraint_result = await score_constraints(
                    rubrics=task_input.constraint_rubrics,
                    trajectory=list(getattr(agent_output, 'main_trajectory', None) or []),
                    final_answer=str(final_answer or ''),
                    context_files=list(task_input.context_files or []),
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception('context_agent_general_task: constraint judge raised')
                constraint_result = {
                    'binary': 0,
                    'ok': False,
                    'error_msg': f'{type(exc).__name__}: {exc}',
                    'verdicts': [],
                }

            normalized_constraint = _normalize_score(constraint_result.get('binary'))
            if not constraint_result.get('ok') or normalized_constraint not in {0.0, 1.0}:
                judge_unavailable = True
                reward_error = True
                score = 0.0
                constraint_result.setdefault(
                    'error_msg',
                    'constraint judge returned an invalid result',
                )
                rationale = (
                    f'{rationale}; constraint judge failed: {constraint_result.get("error_msg") or "unknown error"}'
                ).strip('; ')
            else:
                constraint_score = normalized_constraint
                score = 1.0 if constraint_score == 1.0 else task_input.constraint_fail_reward
        else:
            # Constraint correctness cannot change a wrong answer's zero reward.
            # Skipping this call avoids spending judge capacity on an axis whose
            # result cannot affect the training signal.
            score = 0.0
            constraint_result = {
                'ok': True,
                'skipped': True,
                'reason': 'answer axis did not pass',
            }

    score, extra_fields = apply_tool_name_resolution_score(
        score,
        agent_output,
        enabled=task_input.zero_score_on_tool_name_resolution_error,
        eligible=not (reward_error or judge_unavailable),
    )
    if task_input.constraint_rubrics:
        # The HyTuner runner forwards extra_fields, not the Task's top-level audit axes.
        # Missing or failed constraint grades must not contribute fabricated zero metrics.
        axis_metrics: dict[str, float] = {}
        if answer_judge_valid and answer_score is not None:
            axis_metrics['answer_score'] = answer_score
        if constraint_score is not None:
            axis_metrics['constraint_score'] = constraint_score
        extra_fields.setdefault('reward_extra_info', {})['extra_reward_metrics'] = axis_metrics
        extra_fields['agentif_judge'] = {'answer': dict(result), 'constraint': constraint_result}

    logger.info(
        'context_agent_general_task: score=%.3f answer=%.3f constraint=%s style=%s dataset=%s',
        score,
        answer_score,
        'not-scored' if constraint_score is None else f'{constraint_score:.0f}',
        style or '(unrouted)',
        dataset_name or '(unrouted)',
    )
    return build_output(
        extra_fields=extra_fields,
        final_answer=final_answer,
        score=score,
        answer_score=answer_score,
        constraint_score=constraint_score,
        answer_judge=dict(result),
        constraint_judge=constraint_result,
        grading_style=style,
        judge_unavailable=judge_unavailable,
        rationale=rationale or None,
        requirement_status=list(result.get('requirement_status') or []),
        graded_on=result.get('graded_on'),
        criteria_passed=result.get('criteria_passed'),
        criteria_total=result.get('criteria_total'),
        judge_rounds=result.get('judge_rounds'),
        criteria=list(result.get('criteria') or []),
        workspace_artifact=workspace_artifact,
        termination_reason='REWARD_ERROR' if reward_error or judge_unavailable else agent_termination_reason,
    )
