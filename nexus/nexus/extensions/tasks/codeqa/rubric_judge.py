"""Pure rubric-judge logic: prompt assembly + strict rubric-score parsing.

Ported from hytuner.utils.reward_score.aihub (the message-assembly and parse
helpers only). The HTTP layer of aihub is intentionally dropped -- the LLM call
is issued host-side through a Nexus OpenAICompatibleClient by task.py, not a raw
aiohttp session inside the sandbox.
"""

import json
import re


# Fixed judge system prompt, BYTE-IDENTICAL to the reference hytuner aihub.py
# (_JUDGE_SYSTEM_PROMPT). Used verbatim to keep exact scoring parity with aihub.
_JUDGE_SYSTEM_PROMPT = (
    'You are a strict grader for code-related question answering. '
    'Follow the user-provided rubric exactly and output ONLY the JSON object '
    'specified by the rubric, with no markdown fences and no extra text.'
)


def build_rubric_messages(question, reference, answer, critic_rule):
    """rubric-mode: critic_rule contains [QUESTION]/[REFERENCE]/[ANSWER] placeholders; fill them
    and use as the user message. The system prompt is the fixed _JUDGE_SYSTEM_PROMPT, matching
    the reference aihub.py verbatim. answer is substituted LAST so it cannot smuggle further
    placeholders."""
    user_msg = (
        critic_rule
        .replace('[QUESTION]', question or '')
        .replace('[REFERENCE]', reference or '')
        .replace('[ANSWER]', answer or '')
    )
    return [{'role': 'system', 'content': _JUDGE_SYSTEM_PROMPT}, {'role': 'user', 'content': user_msg}]


def _strip_json_fence(content: str) -> str:
    s = (content or '').strip()
    if s.startswith('```'):
        s = re.sub(r'^```[a-zA-Z0-9_]*\s*', '', s)
        s = re.sub(r'\s*```$', '', s).strip()
    return s


def parse_rubric_score_strict(content: str, score_key: str) -> float | None:
    """Strict parse: the whole response must be a JSON dict; take ONLY the exact score_key
    field, range-check to [0,100], divide by 100 -> [0,1]. Any of unparseable / not-a-dict /
    missing score_key / non-numeric -> None (caller treats as unparseable = real 0.0).
    NO %/N-100/near-score fallback: it would grab a number injected in the answer or a rubric's
    own literals (0.4 / 1 / 0-100)."""
    if not content:
        return None
    try:
        obj = json.loads(_strip_json_fence(content))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    raw = obj.get(score_key)
    if raw is None:
        return None
    try:
        s = float(raw)
    except (TypeError, ValueError):
        return None
    s = max(0.0, min(100.0, s))
    return s / 100.0
