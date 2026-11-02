"""Multi-dimension scoring for code_mt_task.

Contract:
- `score_dimensions = []` → binary/partial coercion of raw verifier score.
- `score_dimensions = [...]` → each dimension produces a float; dimensions
  merge via `score_merge`. LLM-judge dimensions write `<name>_score` and
  `<name>_reason` back into `eval_result` so the next MT turn's feedback
  template can reference them as `{<name>_reason}`.
- LLM judges are fully field-agnostic: the template (inline or file) is
  rendered with `_build_format_vars` (reward.json scalars + task_input.extra
  scalars + a few code_mt standards). No code_mt code reads dataset-specific
  field names.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re

from pathlib import Path
from typing import Any

from nexus.extensions.projects.code_mt.task import loglib
from nexus.extensions.projects.code_mt.task.models import CodeMTTaskInput, LLMJudgeConfig, ScoreDimension


logger = logging.getLogger(__name__)


# ============================================================================
# Format vars (shared by prompt_template / mt_feedback_template / llm_judge)
# ============================================================================


def build_format_vars(
    task_input: CodeMTTaskInput,
    eval_result: dict[str, Any] | None = None,
    *,
    attempt: int = 1,
    original_message: str = '',
    extra_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the format_vars dict used wherever code_mt renders a template.

    Order of precedence (later overrides earlier):
      1. code_mt standards (instance_id, task_type, attempt, max_turns, ...)
      2. task_input.extra scalars (dataset fields written into task.toml)
      3. reward.json scalars (eval_result; written by tests/judge.py)
      4. extra_overrides (call-site additions)
    """
    eval_result = eval_result or {}
    out: dict[str, Any] = {
        'instance_id': task_input.instance_id,
        'task_type': task_input.task_type,
        'attempt': attempt,
        'max_turns': task_input.mt_max_turns,
        'original_message': original_message or task_input.message,
    }
    for k, v in task_input.extra.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = '' if v is None else v
    for k, v in eval_result.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = '' if v is None else v
    if extra_overrides:
        out.update(extra_overrides)
    return out


def render_template(template: str, format_vars: dict[str, Any]) -> str:
    """Render a `{key}`-style template, tolerating missing keys.

    Missing keys are replaced with an empty string (NOT preserved as literal
    `{key}`) so that retry/judge prompts never leak unfilled placeholders to
    the agent or LLM judge. A WARNING is logged for each missing key.
    """
    try:
        return template.format(**format_vars)
    except (KeyError, IndexError, ValueError):
        missing: set[str] = set()

        def _replacer(m: re.Match) -> str:
            key = m.group(1)
            if key in format_vars:
                return str(format_vars[key])
            missing.add(key)
            return ''

        rendered = re.sub(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}', _replacer, template)
        if missing:
            logger.warning('render_template: missing keys replaced with empty: %s', sorted(missing))
        return rendered


# ============================================================================
# Main entry point
# ============================================================================


async def compute_score(
    raw_score: float,
    eval_result: dict[str, Any],
    task_input: CodeMTTaskInput,
    ctx: Any,
) -> tuple[float, dict[str, float]]:
    """Compute final score with multi-dimension support.

    Mutates `eval_result` to add `<name>_score`/`<name>_reason` entries for
    LLM-judge dimensions so downstream MT feedback templates can reference them.
    """
    if not task_input.score_dimensions:
        coerced = _coerce_score(raw_score, task_input.score_kind)
        return coerced, {'verifier': coerced}

    final, scores = await _compute_multi_dimension(raw_score, eval_result, task_input, ctx)
    # BINARY VERDICT (pre-penalty): score_kind='binary' snaps the MERGED rule+llm
    # base to 0/1 here — "做对了才是1，不到1就是0" (取int). Penalty then runs on the
    # 0/1 base downstream and turns it back into a float. This is why the snap MUST
    # be here (pre-penalty) and NOT at task.py's post-penalty step (which would
    # erase the penalty). 'partial' keeps the merged float untouched.
    if task_input.score_kind == 'binary':
        final = 1.0 if final >= 1.0 else 0.0
    return final, scores


def _coerce_score(raw_score: float, kind: str) -> float:
    if kind == 'binary':
        return 1.0 if raw_score >= 1.0 else 0.0
    return max(0.0, min(1.0, raw_score))


async def _compute_multi_dimension(
    raw_score: float,
    eval_result: dict[str, Any],
    task_input: CodeMTTaskInput,
    ctx: Any,
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}

    if task_input.score_merge == 'sequential':
        for dim in task_input.score_dimensions:
            dim_score = await _eval_dimension(dim, raw_score, eval_result, task_input, ctx)
            scores[dim.name] = dim_score
            if dim_score == 0.0:
                loglib.phase_fail(
                    logger,
                    'SCORE',
                    f'sequential short-circuit: {dim.name!r} failed',
                )
                return 0.0, scores
        final = min(scores.values()) if scores else 0.0
        return final, scores

    for dim in task_input.score_dimensions:
        scores[dim.name] = await _eval_dimension(dim, raw_score, eval_result, task_input, ctx)

    if not scores:
        return 0.0, {}

    if task_input.score_merge == 'multiply':
        final = math.prod(scores.values())
    elif task_input.score_merge == 'min':
        final = min(scores.values())
    else:  # weighted_sum
        total_weight = sum(d.weight for d in task_input.score_dimensions)
        final = (
            sum(scores.get(d.name, 0.0) * d.weight for d in task_input.score_dimensions) / total_weight
            if total_weight
            else 0.0
        )
    return final, scores


async def _eval_dimension(
    dim: ScoreDimension,
    raw_score: float,
    eval_result: dict[str, Any],
    task_input: CodeMTTaskInput,
    ctx: Any,
) -> float:
    if dim.type == 'rule':
        s = _coerce_score(raw_score, task_input.score_kind)
        loglib.score_line(logger, dim.name, 'rule', s, detail=f'raw={raw_score:.2f}')
        return s

    if dim.type == 'llm_judge':
        judge_config = LLMJudgeConfig(**dim.config) if dim.config else task_input.llm_judge
        # Display the RESOLVED backend (not the raw use_self, which defaults True and
        # made a correctly host/sandbox-routed judge log a misleading "model=self").
        _be = (judge_config.judge_backend or '').lower()
        if _be not in ('self', 'host', 'sandbox'):
            _be = 'self' if judge_config.use_self else 'host'
        _model_disp = 'self(policy)' if _be == 'self' else ('sandbox' if _be == 'sandbox' else judge_config.model)
        loglib.phase_start(
            logger,
            'JUDGE',
            f'{dim.name} (llm_judge) — backend={_be} model={_model_disp}',
        )
        score, reason, prompt, raw_response = await _run_llm_judge(
            eval_result,
            judge_config,
            task_input,
            ctx,
        )
        # Write back so next MT turn's feedback template can use them.
        eval_result[f'{dim.name}_score'] = score
        eval_result[f'{dim.name}_reason'] = reason
        # Persist the full judge conversation so it can be inspected from the
        # db (task_output.eval_result) without needing live trace spans —
        # _create_completion intentionally bypasses observation to keep
        # judge calls out of agent training data.
        eval_result[f'{dim.name}_prompt'] = prompt
        eval_result[f'{dim.name}_response'] = raw_response
        loglib.score_line(logger, dim.name, 'llm_judge', score, detail=reason[:200])
        return score

    logger.warning('Unknown score dimension type: %s', dim.type)
    return 0.0


# ============================================================================
# Agent trajectory extraction (for judges whose input is the model's output,
# not an on-disk / reward.json field — e.g. RepoQA/rubric judges).
# ============================================================================


def _msg_text(content: Any) -> str:
    """Flatten an OpenAI-style message `content` (str | list of parts) to text."""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(str(p.get('text', '')))
            else:
                parts.append(str(p))
        return '\n'.join(x for x in parts if x)
    return str(content)


def _chat_tracker_messages() -> list[dict[str, Any]]:
    """Flatten the live ChatTrackerProcessor snapshot into one OpenAI-chat message list.

    The processor listens on openai_completion spans (every scaffold's LLM calls
    route through the host openai_client) and rebuilds the chat tree, so its
    snapshot carries the assistant tool_calls + role='tool' observations for
    codex / hycode / mini_swe / swe_agent alike. branch_last mode wraps each
    trajectory as {'messages': [...]}; global_first yields a bare list. Concatenate
    every trajectory so detection sees the full history (incl. compacted turns).
    Returns [] when the processor is absent or on any error (fail-open).
    """
    try:
        from nexus.trace import get_processor
        from nexus.trace.processors.chat_tracker import ChatTrackerProcessor

        processor = get_processor(ChatTrackerProcessor)
        if processor is None:
            return []
        snapshot = processor.snapshot_result() or {}
    except Exception:
        logger.debug('chat_tracker snapshot unavailable; no fallback trajectory', exc_info=True)
        return []

    messages: list[dict[str, Any]] = []
    for trajectory in snapshot.get('trajectories') or []:
        if isinstance(trajectory, dict):
            messages.extend(trajectory.get('messages') or [])
        elif isinstance(trajectory, list):
            messages.extend(trajectory)
    return messages


def extract_agent_trajectory(ctx: Any) -> tuple[str, str, list[dict[str, Any]]]:
    """Pull the agent's runtime output off the shared ExecutionContext.

    The tool_call / text_parse loops stash the running conversation under
    ctx.tool_state['_code_mt_messages'] (MESSAGES_STATE_KEY) so it survives
    across MT turns. This is the ONLY place the agent's natural-language answer
    exists — it never touches the container filesystem, so no grader/tests can
    surface it. Returns:
      - answer:     the last assistant message's text (the "final answer")
      - transcript: the whole conversation rendered as text (for rubrics that
                    judge the entire trajectory, not just the final answer)
      - messages:   the raw message list (for fully-custom prompt builders)
    Everything degrades to '' / [] when the trajectory is unavailable.
    """
    tool_state = getattr(ctx, 'tool_state', None) or {}
    messages = tool_state.get('_code_mt_messages') or []

    # Agent-agnostic fallback: scaffolds OTHER than code_mt (codex / hycode /
    # mini_swe / swe_agent) never populate _code_mt_messages — they run their own
    # loop and their LLM calls flow through the host openai_client, which the
    # ChatTrackerProcessor records as openai_completion spans. Reconstruct the full
    # OpenAI-chat conversation (assistant tool_calls + role='tool' observations)
    # from that snapshot so the reward-hacking / anti-hack monitors + host-judge
    # answer seeding see a real trajectory for those agents too. Purely additive:
    # code_mt short-circuits (its _code_mt_messages is already set), and this is a
    # no-op when chat_tracker is not registered (returns []). Never raises.
    if not messages:
        messages = _chat_tracker_messages() or []

    # Prefer an explicitly submitted answer. The submit_answer tool
    # (code_mt/submit_answer) stashes its `answer` field on tool_state via
    # context_updates; that is the authoritative final answer for QA-style tasks
    # (RepoQA), whose answer lives in the tool_call arguments and NOT in any
    # assistant `content` text. Fall back to the last non-empty assistant message
    # when no answer was submitted (backward compatible: tasks that finish by
    # writing the answer as plain assistant text still grade as before).
    answer = ''
    submitted_answer = tool_state.get('answer')
    if isinstance(submitted_answer, str) and submitted_answer.strip():
        answer = submitted_answer
    else:
        for m in reversed(messages):
            if m.get('role') == 'assistant':
                text = _msg_text(m.get('content'))
                if text.strip():
                    answer = text
                    break

    lines: list[str] = []
    for m in messages:
        role = m.get('role', '?')
        text = _msg_text(m.get('content'))
        if text.strip():
            lines.append(f'[{role}]\n{text}')
        for tc in m.get('tool_calls') or []:
            fn = tc.get('function', {}) if isinstance(tc, dict) else {}
            name = fn.get('name', '') if isinstance(fn, dict) else ''
            args = fn.get('arguments', '') if isinstance(fn, dict) else ''
            lines.append(f'[{role}:tool_call {name}]\n{args}')
    transcript = '\n\n'.join(lines)

    return answer, transcript, messages


# ============================================================================
# LLM judge
# ============================================================================


def _resolve_judge_params(judge_config: LLMJudgeConfig, task_input: CodeMTTaskInput) -> dict[str, Any]:
    """Resolve the dedicated LLM judge's model/request knobs.

    Precedence (first non-empty wins): process env (JUDGE_*) > judge_config
    (dim.config / task_input.llm_judge, i.e. the value the hytuner index / offline
    generator / task.toml baked into the task) > task.toml [verifier.env] > field
    defaults.

    Why env > baked config: the launch environment (and the hytuner index, which
    overrides task.toml one layer up in loader.apply_task_spec) must be able to
    OVERRIDE a task.toml-baked judge field at run time — e.g. inject/rotate the
    gateway secret via JUDGE_OPENAI_API_KEY, or repoint the model/base for a whole
    run — without editing every task dir. This makes string knobs consistent with
    the numeric _num knobs below, which already prefer env. The old "baked config
    wins over env" ordering was there to stop a stray JUDGE_OPENAI_MODEL from
    hijacking a self-judge onto the external gateway, but that backend hijack is now
    prevented independently by the explicit `judge_backend` field (a self judge
    never reads JUDGE_* at all — it uses ctx.openai_client), so the string-first
    ordering only added an inconsistency. Trade-off: a WRONG JUDGE_OPENAI_MODEL/BASE
    exported in the shell will now override a correct task.toml value, so keep the
    launch env clean (the run scripts export only JUDGE_OPENAI_API_KEY). expandvars
    still resolves `$APP_ID:$APP_KEY` at every layer, so the real secret is supplied
    at run time and never baked into the task dir/index.
    """
    env = task_input.verifier_env or {}

    def _num(key: str, fallback: float, cast: Any) -> Any:
        # env (JUDGE_*) then task.toml [verifier.env] override the baked config
        # `fallback`; LLMJudgeConfig always carries a non-None numeric default, so
        # `fallback` is the resolved judge_config field and applies only when
        # neither env nor verifier.env pinned the value.
        for candidate in (os.environ.get(key), env.get(key)):
            if candidate not in (None, ''):
                try:
                    return cast(candidate)
                except (TypeError, ValueError):
                    return fallback
        return fallback

    def _resolve(key: str, cfg_val: str) -> str:
        # env (JUDGE_*) > baked judge_config (cfg_val) > task.toml [verifier.env].
        for candidate in (os.environ.get(key), cfg_val, env.get(key)):
            if candidate not in (None, ''):
                return os.path.expandvars(candidate)
        return ''

    api_base = _resolve('JUDGE_OPENAI_BASE_URL', judge_config.api_base)
    model = _resolve('JUDGE_OPENAI_MODEL', judge_config.model)
    api_key = _resolve('JUDGE_OPENAI_API_KEY', judge_config.api_key) or 'EMPTY'

    extra_body: dict[str, Any] = dict(judge_config.extra_body or {})
    raw_extra = env.get('JUDGE_EXTRA_BODY_JSON')
    if raw_extra:
        try:
            parsed = json.loads(raw_extra) if isinstance(raw_extra, str) else raw_extra
            if isinstance(parsed, dict):
                extra_body = parsed
        except (TypeError, ValueError) as e:
            logger.warning('JUDGE_EXTRA_BODY_JSON not valid JSON, ignoring: %s', e)

    # Resolve WHERE the judge runs. An explicit judge_backend wins and is
    # authoritative; '' falls back to the legacy use_self inference — but
    # use_self is now authoritative there too. The old
    # `use_self = use_self and not has_env_judge` coupling is DELETED on purpose:
    # a stray JUDGE_* in the launch env used to silently flip a self-judge onto
    # the external gateway (builder anti_hack hijacked itself onto repoqa's
    # deepseek endpoint whenever both ran in the same job). Now 'self' means self.
    backend = (judge_config.judge_backend or '').lower()
    if backend not in ('self', 'host', 'sandbox'):
        backend = 'self' if judge_config.use_self else 'host'

    return {
        'backend': backend,
        'use_self': backend == 'self',
        'api_base': api_base,
        'model': model,
        'api_key': api_key,
        'max_tokens': int(_num('JUDGE_MAX_TOKENS', judge_config.max_tokens, int)),
        'timeout': float(_num('JUDGE_TIMEOUT', judge_config.timeout, float)),
        'max_retries': int(_num('JUDGE_MAX_RETRIES', judge_config.max_retries, int)),
        'extra_body': extra_body,
    }


async def _run_llm_judge(
    eval_result: dict[str, Any],
    judge_config: LLMJudgeConfig,
    task_input: CodeMTTaskInput,
    ctx: Any,
) -> tuple[float, str, str, str]:
    """Returns (score, reason, rendered_prompt, raw_response_content).

    The latter two are returned (not just logged) so the caller can persist them
    into eval_result for offline inspection — _create_completion bypasses span
    observation on purpose, so live traces don't carry the judge transcript.
    """
    if not judge_config.enabled:
        return 1.0, '', '', ''

    params = _resolve_judge_params(judge_config, task_input)
    if params['backend'] == 'sandbox':
        # In-sandbox judging (harbor Verifier test.sh → judge.py → reward.json)
        # already produced the score; the host side must not re-invoke an LLM.
        # A rule dimension surfaces the reward.json field. Neutral no-op here.
        logger.debug('llm_judge backend=sandbox → host judge no-op (score from reward.json)')
        return 1.0, '', '', ''

    template = _load_template(judge_config, task_input)
    if not template:
        # Enabled judge with no resolvable template is a MISCONFIGURATION, not a
        # pass. Fail-closed (0.0) to match the empty-choices path below (:513);
        # the old fail-open 1.0 silently granted full marks to every sample of a
        # broken judge dimension.
        loglib.phase_fail(logger, 'JUDGE', 'LLM judge enabled but no template/template_path resolved')
        return 0.0, 'LLM judge enabled but no template/template_path resolved', '', ''

    if judge_config.prompt_builder_import:
        # Judge input is the agent's runtime output (not an on-disk field). A
        # dataset-supplied builder pulls it off ctx and renders the prompt —
        # e.g. RepoQA rubric templates that keep verbatim [QUESTION]/[REFERENCE]/
        # [ANSWER] placeholders, or future rubrics fed the whole trajectory.
        from nexus.core.registry import import_from_path
        from nexus.extensions.projects.code_mt.task._module_aliases import canonicalize_import_path

        resolved = canonicalize_import_path(judge_config.prompt_builder_import)
        assert resolved is not None  # guarded by outer if
        builder = import_from_path(resolved)
        prompt = builder(template, ctx, task_input, eval_result, judge_config)
    else:
        # Default path (builder anti_hack, etc.): render {key} against
        # reward.json + task.extra scalars. Agent answer/transcript are also
        # exposed so a plain {agent_answer}/{agent_trajectory} template works
        # without a custom builder.
        answer, transcript, _ = extract_agent_trajectory(ctx)
        format_vars = build_format_vars(
            task_input,
            eval_result,
            extra_overrides={'agent_answer': answer, 'agent_trajectory': transcript},
        )
        prompt = render_template(template, format_vars)

    if params['backend'] == 'self':
        client = ctx.openai_client
    else:
        from nexus.core.openai_client import OpenAIConfig
        from nexus.extensions.openai_clients.litellm import LiteLLMClient

        request_args: dict[str, Any] = {
            'model': params['model'],
            'max_tokens': params['max_tokens'],
            'temperature': judge_config.temperature,
            'timeout': params['timeout'],
            'num_retries': params['max_retries'],
        }
        if params['extra_body']:
            request_args['extra_body'] = params['extra_body']
        client = LiteLLMClient(
            config=OpenAIConfig(
                client_args={
                    'api_key': params['api_key'],
                    'base_url': params['api_base'],
                    # Cap the OUTER tenacity retry wrapper (_create_completion reads
                    # client_args['max_retries'], default 15). Without this a judge
                    # that keeps failing (e.g. gateway 401) retries 15× with
                    # exponential backoff up to 10 min each → the run row stays
                    # `running` for hours and blocks a concurrency slot. Default 3,
                    # controllable via the llm_judge.max_retries field or
                    # JUDGE_MAX_RETRIES.
                    'max_retries': params['max_retries'],
                },
                request_args=request_args,
            )
        )

    loglib.phase_info(logger, 'JUDGE', f'prompt ({len(prompt):,} chars), template_path={judge_config.template_path}')

    # self-mode judging shares the SAME HyTunerClient instance that produced the
    # trajectory (client is ctx.openai_client). _create_completion mutates that
    # client's per-trajectory state — prefix cache (last_messages/last_tokens/
    # last_expanded_tokens/_last_rendered_text/_last_num_*), token counters, and
    # the R3 cross-turn routing accumulator (self._routed_experts). Running the
    # judge prompt through it would corrupt the trajectory's recorded routing and
    # prefix cache. Snapshot the mutated fields before the call and restore them
    # after, so the judge leaves ZERO trace on the shared client. The litellm
    # backend uses a throwaway client, so this only guards the 'self' backend.
    _judge_shares_training_client = params['backend'] == 'self'
    _client_state_snapshot: dict[str, Any] | None = None
    if _judge_shares_training_client:
        _snapshot_fields = (
            'last_messages',
            'last_tokens',
            'last_expanded_tokens',
            '_last_rendered_text',
            '_last_num_images',
            '_last_num_videos',
            '_last_num_audios',
            '_initial_prompt_len',
            'token_consumed',
            'curr_segment_consumed_tokens',
            '_routed_experts',
        )
        _client_state_snapshot = {f: getattr(client, f) for f in _snapshot_fields if hasattr(client, f)}

    try:
        # Use the underscore-prefixed `_create_completion` to bypass the
        # `@observe('openai_completion')` decorator on the public method.
        # Why: HyTunerClient (training path) writes token-level training_data
        # into the current `openai_completion` span on every call to the
        # public `create_completion`. HytunerProcessor then buffers that span
        # and flushes it as a conversation turn — which would pollute the
        # agent's trajectory with the judge's user prompt + response. By
        # calling `_create_completion` we skip span creation entirely; the
        # judge call leaves zero trace in the training data.
        response = await client._create_completion(messages=[{'role': 'user', 'content': prompt}])
        if not getattr(response, 'choices', None):
            loglib.phase_fail(logger, 'JUDGE', 'LLM judge returned empty choices')
            return 0.0, 'LLM judge returned empty choices', prompt, ''
        content = response.choices[0].message.content or ''
        loglib.phase_info(logger, 'JUDGE', f'response: {content[:300]}')
        score, reason = _parse_judge_response(content, judge_config)
        return score, reason, prompt, content
    except Exception as e:
        loglib.phase_fail(logger, 'JUDGE', f'LLM judge call failed: {e}')
        return 0.0, f'LLM judge call failed: {e}', prompt, ''
    finally:
        # Restore the shared training client's per-trajectory state regardless of
        # judge success/failure, so scoring never perturbs the recorded trajectory.
        if _client_state_snapshot is not None:
            for f, v in _client_state_snapshot.items():
                setattr(client, f, v)


def _load_template(judge_config: LLMJudgeConfig, task_input: CodeMTTaskInput) -> str:
    """Resolve judge template: file path takes precedence over inline string.

    template_path is resolved relative to task_dir if not absolute.
    """
    if judge_config.template_path:
        p = Path(judge_config.template_path)
        if not p.is_absolute() and task_input.task_dir:
            p = Path(task_input.task_dir) / judge_config.template_path
        try:
            return p.read_text(encoding='utf-8')
        except OSError as e:
            logger.warning('LLM judge template_path %s unreadable: %s', p, e)
    return judge_config.template


def _parse_judge_response(content: str, judge_config: LLMJudgeConfig) -> tuple[float, str]:
    """Parse the judge response. Dataset-defined overrides take precedence.

    Precedence: parser_import > value_map > score_scale > built-in coercion.
    """
    if judge_config.parser_import:
        from nexus.core.registry import import_from_path
        from nexus.extensions.projects.code_mt.task._module_aliases import canonicalize_import_path

        try:
            resolved = canonicalize_import_path(judge_config.parser_import)
            assert resolved is not None  # guarded by outer if
            parser = import_from_path(resolved)
            score, reason = parser(content, judge_config)
            return float(score), str(reason or '')
        except Exception as e:
            logger.warning('Custom judge parser %s failed: %s', judge_config.parser_import, e)

    fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(content.strip())

    for raw in candidates:
        try:
            data = json.loads(_strip_trailing_commas(raw))
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        score = _coerce_judge_value(data.get(judge_config.score_field), judge_config)
        reason = str(data.get(judge_config.reason_field, '') or '')
        if score is not None:
            return score, reason

    # Loose fallback: "Score: X" pattern
    m = re.search(r'[Ss]core\s*[:=]\s*([0-9]*\.?[0-9]+)', content)
    if m:
        try:
            v = float(m.group(1))
            return _apply_score_scale(v, judge_config), content[:300]
        except ValueError:
            pass

    logger.warning('Could not parse judge response: %s', content[:200])
    return 0.0, content[:300]


def _strip_trailing_commas(s: str) -> str:
    return re.sub(r',\s*([}\]])', r'\1', s)


def _apply_score_scale(v: float, judge_config: LLMJudgeConfig) -> float:
    if judge_config.score_scale:
        lo, hi = judge_config.score_scale
        if hi > lo:
            v = (v - lo) / (hi - lo)
    return min(max(v, 0.0), 1.0)


def _coerce_judge_value(v: Any, judge_config: LLMJudgeConfig) -> float | None:
    # 1. Dataset-defined enum mapping wins outright — case-insensitive on the key.
    if judge_config.value_map:
        key = str(v).strip() if not isinstance(v, bool) else str(v).lower()
        if key in judge_config.value_map:
            return float(judge_config.value_map[key])
        lower_map = {k.lower(): val for k, val in judge_config.value_map.items()}
        if key.lower() in lower_map:
            return float(lower_map[key.lower()])

    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return _apply_score_scale(float(v), judge_config)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ('pass', 'true', 'yes', 'approved'):
            return 1.0
        if s in ('fail', 'false', 'no', 'rejected'):
            return 0.0
        try:
            return _apply_score_scale(float(s), judge_config)
        except ValueError:
            return None
    return None
