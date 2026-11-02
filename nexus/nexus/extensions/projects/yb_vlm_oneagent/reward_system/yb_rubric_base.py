"""Rubric scoring base module.

Contains shared infrastructure used by all rubric scoring versions:

* ``parse_json_block``       — robust JSON extraction from judge output
* ``render_history``         — flatten conversation history into prompt text
* ``call_judge_llm``         — asynchronously call the judge LLM and parse its JSON
* ``yb_rubric_score_impl``   — main framework: parse rubrics → run
  ``N_ROUNDS`` concurrent judge calls → take mean of valid rounds

Design notes:

1. The judge transport is :func:`nexus.extensions.tasks.mm_agent.\
reward_system.taiji_api.get_model_response` — the same Polaris-backed
   path ``consist_judge`` uses. No IP-list / OpenAI client fallback.

2. Env knobs use the ``YB_RUBRIC_*`` namespace:

   * ``YB_RUBRIC_JUDGE_MODEL`` — judge model name(s) understood by
     ``taiji_api`` (string or comma-separated). Defaults to
     ``Qwen35-397B-A17B-Rubric-Judge-RL``. The Polaris service name is
     resolved by ``taiji_api`` itself via the standard
     ``{model_name}_AIDE`` convention.
   * ``YB_RUBRIC_JUDGE_MAX_RETRIES`` — retries on request failure,
     empty / unparseable judge output, or parsed output with an invalid
     ``scored_checklist``.
   * ``YB_RUBRIC_JUDGE_MAX_TOKENS`` — max output tokens for each judge
     response. Defaults to 8192; rubric JSON can be long when the
     checklist has many items, but should not need the old 20000-token
     generation budget in normal cases.
   * ``YB_RUBRIC_JUDGE_REQUEST_TIMEOUT`` — per HTTP request timeout in
     seconds. Defaults to 120.
   * ``YB_RUBRIC_JUDGE_TEMPERATURE`` — judge sampling temperature.
     Defaults to 0.7, matching the previous OpenAI/vLLM RM caller.
   * ``YB_RUBRIC_JUDGE_TOP_P`` — judge nucleus sampling parameter.
     Defaults to 0.8, matching the previous OpenAI/vLLM RM caller.
   * ``YB_RUBRIC_JUDGE_TOP_K`` — optional judge top-k sampling parameter.
     Defaults to 20, matching the previous OpenAI/vLLM RM caller. Set
     to 0 or a negative value to omit ``top_k`` from the request body.
   * ``YB_RUBRIC_N_ROUNDS`` — number of scoring rounds; final reward is
     the mean of all valid rounds.
   * ``YB_RUBRIC_N_ROUNDS_VAL`` — validation-only override for the round
     count. Used only when the caller sets ``obj['validate'] = True``
     (yb_vlm_oneagent forwards this from ``task_input.validate``).
     Falls back to ``YB_RUBRIC_N_ROUNDS`` when unset. Training rollouts
     always use ``YB_RUBRIC_N_ROUNDS``.
   * ``YB_RUBRIC_MAX_CONCURRENT`` — maximum number of in-flight judge
     transport requests across samples, rounds, and retries on one event
     loop. Defaults to 1.

3. Each rubric topic module (e.g. ``shiwu_topic`` / ``medical_topic``)
   supplies its own ``SCORE_SYSTEM_PROMPT`` / ``build_scoring_prompt`` /
   ``get_overall_score`` and calls into ``yb_rubric_score_impl`` here.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback

from collections.abc import Callable
from typing import Any

from nexus.extensions.tasks.mm_agent.reward_system.taiji_api import get_model_response


# ---------------------------------------------------------------------------
# Environment knobs
# ---------------------------------------------------------------------------

# Default judge model — vLLM deployment with rubric-specific RL post-training.
# Polaris service name is resolved by ``taiji_api`` via the standard
# ``{model_name}_AIDE`` convention.
_DEFAULT_JUDGE_MODEL = 'Qwen35-397B-A17B-Rubric-Judge-RL'


_JUDGE_REQUEST_SEMAPHORES: dict[int, tuple[int, asyncio.Semaphore]] = {}


def _resolve_judge_model() -> Any:
    """Resolve ``YB_RUBRIC_JUDGE_MODEL``.

    Accepts either a single name or a comma-separated list. Returns a
    string for the single-name case (most common for vLLM deployments)
    and a ``list[str]`` when multiple names are configured (matches
    ``taiji_api.get_model_response`` semantics, which round-robins).
    """
    raw = os.environ.get('YB_RUBRIC_JUDGE_MODEL') or _DEFAULT_JUDGE_MODEL
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    if len(parts) == 1:
        return parts[0]
    return parts


def _resolve_int(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name)
    if raw is None or raw == '':
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _resolve_float(env_name: str, default: float) -> float:
    raw = os.environ.get(env_name)
    if raw is None or raw == '':
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def get_judge_request_semaphore() -> asyncio.Semaphore:
    """Return the event-loop-local semaphore for actual judge requests."""
    limit = _resolve_int('YB_RUBRIC_MAX_CONCURRENT', 1)
    if limit < 1:
        limit = 1

    loop_key = id(asyncio.get_running_loop())
    entry = _JUDGE_REQUEST_SEMAPHORES.get(loop_key)
    if entry is None or entry[0] != limit:
        semaphore = asyncio.Semaphore(limit)
        _JUDGE_REQUEST_SEMAPHORES[loop_key] = (limit, semaphore)
        return semaphore
    return entry[1]


def resolve_n_rounds(is_validate: bool) -> int:
    """Resolve the number of judge scoring rounds.

    During validation the task layer sets ``obj['validate'] = True``. In
    that case prefer ``YB_RUBRIC_N_ROUNDS_VAL`` (higher round count for
    lower-variance eval scores) and fall back to ``YB_RUBRIC_N_ROUNDS``
    when the val-specific knob is unset. During training we always use
    ``YB_RUBRIC_N_ROUNDS`` — the validation override never leaks into
    training rollouts.

    Shared by ``yb_rubric_score_impl`` and every topic scorer, including
    the writing scorer's own multi-round path.
    """
    default = _resolve_int('YB_RUBRIC_N_ROUNDS', 1)
    raw = os.environ.get('YB_RUBRIC_N_ROUNDS_VAL') if is_validate else None
    if raw is None or raw == '':
        n_rounds = default
    else:
        try:
            n_rounds = int(raw)
        except (TypeError, ValueError):
            n_rounds = default
    if n_rounds < 1:
        n_rounds = 1
    return n_rounds


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def parse_json_block(text: str) -> dict[str, Any]:
    """Robustly extract a top-level JSON object from a judge response.

    Tries, in order:
      1. Direct ``json_repair.loads`` on cleaned text.
      2. Bracket-match all top-level ``{...}`` candidates and parse from
         the last one (judge often emits reasoning JSON before the verdict).
      3. Outermost ``{``…``}`` substring fallback.
      4. ``ast.literal_eval`` on candidates as a last resort.

    If the parse yields a list, the last dict containing
    ``scored_checklist`` is returned (else the last dict).
    """
    import ast  # local — keeps top-level imports lean

    import json_repair

    if not text:
        raise ValueError('LLM response is empty')
    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = cleaned[3:].lstrip()
        if cleaned.lower().startswith('json'):
            cleaned = cleaned[4:].lstrip()
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    def _extract_brace(s: str) -> str:
        start = s.find('{')
        end = s.rfind('}')
        if start != -1 and end != -1 and end > start:
            return s[start : end + 1]
        return s

    def _extract_all_json_objects(s: str) -> list:
        objects = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                depth = 0
                in_str = False
                escape = False
                start = i
                for j in range(i, len(s)):
                    ch = s[j]
                    if escape:
                        escape = False
                        continue
                    if ch == '\\' and in_str:
                        escape = True
                        continue
                    if ch == '"' and not escape:
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            objects.append(s[start : j + 1])
                            i = j + 1
                            break
                else:
                    break  # unclosed, bail
            else:
                i += 1
        return objects

    def _ensure_dict(result):
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            for item in reversed(result):
                if isinstance(item, dict) and 'scored_checklist' in item:
                    return item
            for item in reversed(result):
                if isinstance(item, dict):
                    return item
        raise ValueError(f'解析结果非 dict: {type(result)}')

    # Step 1: direct json_repair.loads
    try:
        return _ensure_dict(json_repair.loads(cleaned))
    except Exception:
        pass

    # Step 2: extract all JSON objects, try from last
    json_objects = _extract_all_json_objects(cleaned)
    for obj_str in reversed(json_objects):
        try:
            return _ensure_dict(json_repair.loads(obj_str))
        except Exception:
            continue

    # Step 3: outermost-brace fallback
    snippet = _extract_brace(cleaned)
    try:
        return _ensure_dict(json_repair.loads(snippet))
    except Exception:
        pass

    # Step 4: ast.literal_eval on remaining candidates
    for candidate in reversed(json_objects) if json_objects else [snippet]:
        try:
            result = ast.literal_eval(candidate)
            if isinstance(result, dict):
                return _ensure_dict(json_repair.loads(json.dumps(result, ensure_ascii=False)))
        except Exception:
            pass

    raise ValueError(f'无法解析 JSON:\n{cleaned[:500]}...')


# ---------------------------------------------------------------------------
# History rendering
# ---------------------------------------------------------------------------


def render_history(messages: list) -> str:
    """Flatten a multi-turn conversation into a plain-text history block."""
    if not messages:
        return '（无历史对话）'
    lines = []
    for msg in messages:
        role = msg.get('role', 'user')
        if role == 'system':
            continue
        content = msg.get('content', '')
        if not content:
            continue
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'text':
                    text_parts.append(part.get('text', ''))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = '\n'.join(text_parts)
        lines.append(f'{role}: {content}')
    return '\n'.join(lines) if lines else '（无历史对话）'


# ---------------------------------------------------------------------------
# LLM caller — backed by taiji_api (Polaris) instead of an IP list.
# ---------------------------------------------------------------------------


async def call_judge_llm(
    prompt: str,
    system_prompt: str,
    max_retries: int | None = None,
) -> dict[str, Any]:
    """Call the judge LLM and parse the JSON response.

    Uses :func:`taiji_api.get_model_response` (Polaris-backed, same
    transport as ``consist_judge``). The system role is propagated
    via ``system_text`` so the rubric grader instructions land in
    their intended slot rather than being crammed into the user turn.

    Retries up to ``max_retries`` times on empty or unparseable
    responses. Each underlying transport call is allowed a single retry
    (``max_retry=1``) so the outer loop owns the retry budget.
    """
    if max_retries is None:
        max_retries = _resolve_int('YB_RUBRIC_JUDGE_MAX_RETRIES', 2)
    if max_retries < 1:
        max_retries = 1

    judge_model = _resolve_judge_model()
    judge_max_tokens = _resolve_int('YB_RUBRIC_JUDGE_MAX_TOKENS', 8192)
    if judge_max_tokens < 1:
        judge_max_tokens = 8192
    judge_request_timeout = _resolve_int('YB_RUBRIC_JUDGE_REQUEST_TIMEOUT', 120)
    if judge_request_timeout < 1:
        judge_request_timeout = 120
    judge_temperature = _resolve_float('YB_RUBRIC_JUDGE_TEMPERATURE', 0.7)
    if judge_temperature < 0:
        judge_temperature = 0.7
    judge_top_p = _resolve_float('YB_RUBRIC_JUDGE_TOP_P', 0.8)
    if judge_top_p <= 0:
        judge_top_p = 0.8
    judge_top_k = _resolve_int('YB_RUBRIC_JUDGE_TOP_K', 20)
    if judge_top_k < 1:
        judge_top_k = None

    last_err: Exception | None = None
    for attempt in range(max_retries):
        attempt_label = f'{attempt + 1}/{max_retries}'
        action = 'retrying...' if attempt + 1 < max_retries else 'no retries left'
        try:
            async with get_judge_request_semaphore():
                raw_response = await get_model_response(
                    img_path=None,
                    input_text=prompt,
                    max_token=judge_max_tokens,
                    max_retry=1,
                    MODEL=judge_model,
                    system_text=system_prompt,
                    request_timeout=judge_request_timeout,
                    temperature=judge_temperature,
                    top_p=judge_top_p,
                    top_k=judge_top_k,
                    enable_thinking=False,
                )

            if not raw_response or not str(raw_response).strip():
                print(
                    f'[YbRubric] Attempt {attempt_label}: empty LLM response, {action}',
                    file=sys.stderr,
                    flush=True,
                )
                last_err = ValueError('empty LLM response')
                continue

            parsed = parse_json_block(raw_response)
            scored_checklist = parsed.get('scored_checklist')
            if not isinstance(scored_checklist, list) or not scored_checklist:
                last_err = ValueError('parsed judge response missing non-empty scored_checklist')
                print(
                    f'[YbRubric] Attempt {attempt_label}: judge response parsed but invalid ({last_err}), {action}',
                    file=sys.stderr,
                    flush=True,
                )
                continue

            parsed['_raw_content'] = raw_response
            return parsed

        except json.JSONDecodeError as e:
            print(
                f'[YbRubric] Attempt {attempt_label}: JSON parse failed ({e}), {action}',
                file=sys.stderr,
                flush=True,
            )
            last_err = e
            continue
        except Exception as e:
            print(
                f'[YbRubric] Attempt {attempt_label}: error ({e}), {action}',
                file=sys.stderr,
                flush=True,
            )
            last_err = e
            continue

    raise ValueError(
        f'[YbRubric] All {max_retries} retries exhausted, LLM returned no valid response (last_err={last_err})'
    )


# ---------------------------------------------------------------------------
# Main scoring function framework
# ---------------------------------------------------------------------------


async def yb_rubric_score_impl(
    obj: dict,
    build_scoring_prompt_fn: Callable,
    get_overall_score_fn: Callable,
    system_prompt: str,
    version_tag: str = '',
) -> dict:
    """Rubric-based scoring framework (shared across rubric versions).

    Each version passes in its own:
      * ``build_scoring_prompt_fn``: ``(question, messages, checklist, answer) -> str``
      * ``get_overall_score_fn``: ``(scored_checklist) -> float``
      * ``system_prompt``: the system prompt string for the judge
      * ``version_tag``: short version label for logging (e.g. ``"v4"``)

    Expects ``obj`` to contain:
      * ``response``    — model's answer text
      * ``ref_answer``  — rubrics JSON string (from the dataset's
        ``answer`` / ground_truth field)
      * ``prompt`` / ``question`` — the user question text
      * ``history``     — conversation history (optional)

    Sets on ``obj``:
      * ``reward``              — float in [0, 1], or -1.0 on failure
      * ``critic_resp``         — combined raw judge response(s)
      * ``model_output``        — alias for ``critic_resp``
      * ``scoring_mode``        — ``"yb_rubric"`` on success, else
        ``"yb_rubric_failed"`` / ``"yb_rubric_empty_response"``
      * ``overall_score``       — same as ``reward`` on success
      * ``round_scores``        — per-round scores
      * ``scored_checklist``    — last valid scored checklist
      * ``original_request``    — back-reference to ``obj`` itself
    """
    obj.update({'original_request': obj})
    obj['reward'] = -1.0

    # During validation the task layer sets ``obj['validate'] = True``.
    # ``resolve_n_rounds`` prefers ``YB_RUBRIC_N_ROUNDS_VAL`` in that case
    # and falls back to ``YB_RUBRIC_N_ROUNDS`` otherwise; training rollouts
    # always use ``YB_RUBRIC_N_ROUNDS``.
    is_validate = bool(obj.get('validate', False))
    n_rounds = resolve_n_rounds(is_validate)

    try:
        answer = obj.get('response', '')
        if not answer or not str(answer).strip():
            obj['reward'] = 0.0
            obj['critic_resp'] = 'Model response is empty'
            obj['model_output'] = obj['critic_resp']
            obj['scoring_mode'] = 'yb_rubric_empty_response'
            return obj

        # Strip tool-call/response envelope; the oneagent protocol no
        # longer wraps the final answer in <answer></answer> tags so
        # we only need to drop everything up to the last
        # ``</tool_responses>`` block (if any).
        tool_token = '</tool_responses>'
        if tool_token in answer:
            answer = answer.split(tool_token)[-1].strip()

        # Parse rubrics: prefer ``rubrics`` (already a list), fall back
        # to ``ref_answer`` (JSON string).
        checklist = obj.get('rubrics', [])
        if not checklist:
            ref_answer_raw = obj.get('ref_answer', '')
            if not ref_answer_raw:
                print(
                    '[YbRubric] WARNING: no rubrics found (neither rubrics nor ref_answer); reward=-1',
                    file=sys.stderr,
                    flush=True,
                )
                obj['critic_resp'] = 'No rubrics provided'
                obj['model_output'] = obj['critic_resp']
                obj['scoring_mode'] = 'yb_rubric_failed'
                return obj
            try:
                if isinstance(ref_answer_raw, str):
                    checklist = json.loads(ref_answer_raw)
                elif isinstance(ref_answer_raw, list):
                    checklist = ref_answer_raw
                else:
                    checklist = ref_answer_raw
            except (json.JSONDecodeError, TypeError) as e:
                print(
                    f'[YbRubric] Failed to parse rubrics from ref_answer: {e}',
                    file=sys.stderr,
                    flush=True,
                )
                obj['critic_resp'] = f'Failed to parse rubrics: {e}'
                obj['model_output'] = obj['critic_resp']
                obj['scoring_mode'] = 'yb_rubric_failed'
                return obj

        if not checklist or not isinstance(checklist, list):
            print(
                '[YbRubric] WARNING: checklist is empty or not a list',
                file=sys.stderr,
                flush=True,
            )
            obj['critic_resp'] = 'Checklist is empty or invalid'
            obj['model_output'] = obj['critic_resp']
            obj['scoring_mode'] = 'yb_rubric_failed'
            return obj

        question = obj.get('question', '') or obj.get('prompt', '')
        history = obj.get('history', [])

        scoring_prompt = build_scoring_prompt_fn(
            question=question,
            messages=history,
            checklist=checklist,
            answer=answer,
        )

        # n rounds judge calls, run concurrently on the current event loop.
        round_scores: list[float] = []
        round_raw_contents: list[str] = [''] * n_rounds
        round_scored_checklists: list[list] = [[]] * n_rounds
        last_valid_scored_checklist: list = []

        async def _score_one_round(round_idx: int):
            parsed = await call_judge_llm(scoring_prompt, system_prompt=system_prompt)
            raw_content = parsed.get('_raw_content', '')
            scored_checklist = parsed.get('scored_checklist', [])
            score = None
            if scored_checklist:
                score = get_overall_score_fn(scored_checklist)
            return round_idx, score, raw_content, scored_checklist

        round_results = await asyncio.gather(
            *(_score_one_round(idx) for idx in range(n_rounds)),
            return_exceptions=True,
        )
        for round_idx, round_result in enumerate(round_results):
            try:
                if isinstance(round_result, BaseException):
                    raise round_result

                _, score, raw_content, scored_checklist = round_result
                round_raw_contents[round_idx] = raw_content
                round_scored_checklists[round_idx] = scored_checklist

                if not scored_checklist:
                    print(
                        f'[YbRubric] WARNING: round {round_idx + 1}/{n_rounds} '
                        f'scored_checklist is empty, skipping this round',
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

                round_scores.append(score)
                last_valid_scored_checklist = scored_checklist

            except Exception as round_err:
                print(
                    f'[YbRubric] WARNING: round {round_idx + 1}/{n_rounds} failed ({round_err}), skipping this round',
                    file=sys.stderr,
                    flush=True,
                )
                round_raw_contents[round_idx] = f'Error: {round_err}'
                continue

        if not round_scores:
            print(
                '[YbRubric] WARNING: all scoring rounds failed',
                file=sys.stderr,
                flush=True,
            )
            obj['reward'] = -1.0
            combined_raw = '\n\n'.join(f'=== Round {i + 1} ===\n{c}' for i, c in enumerate(round_raw_contents))
            obj['critic_resp'] = f'All {n_rounds} scoring rounds failed. Raw:\n{combined_raw[:2000]}'
            obj['model_output'] = obj['critic_resp']
            obj['scoring_mode'] = 'yb_rubric_failed'
            return obj

        overall_score = sum(round_scores) / len(round_scores)

        if n_rounds == 1:
            combined_critic_resp = round_raw_contents[0] if round_raw_contents else ''
        else:
            combined_critic_resp = json.dumps(
                {
                    'n_rounds': n_rounds,
                    'n_valid_rounds': len(round_scores),
                    'round_scores': round_scores,
                    'mean_score': overall_score,
                    'round_responses': [
                        {'round': i + 1, 'raw_content': c, 'scored_checklist': sc}
                        for i, (c, sc) in enumerate(zip(round_raw_contents, round_scored_checklists, strict=False))
                    ],
                },
                ensure_ascii=False,
            )

        obj['reward'] = overall_score
        obj['critic_resp'] = combined_critic_resp
        obj['model_output'] = combined_critic_resp
        obj['scoring_mode'] = 'yb_rubric'
        obj['overall_score'] = overall_score
        obj['round_scores'] = round_scores
        obj['scored_checklist'] = last_valid_scored_checklist

        log_prefix = f'version={version_tag}, ' if version_tag else ''
        scores_str = ', '.join(f'{s:.4f}' for s in round_scores)
        print(
            f'[YbRubric] {log_prefix}n_rounds={n_rounds}, '
            f'valid_rounds={len(round_scores)}, '
            f'round_scores=[{scores_str}], mean_score={overall_score:.4f}, '
            f'checklist_items={len(last_valid_scored_checklist)}',
            flush=True,
        )

        return obj

    except Exception as e:
        print(f'[YbRubric] Error: {e}', file=sys.stderr, flush=True)
        traceback.print_exc()

        obj['critic_resp'] = f'Error in yb_rubric_score: {e}'
        obj['model_output'] = obj['critic_resp']
        obj['reward'] = -1.0
        obj['scoring_mode'] = 'yb_rubric_failed'
        return obj
