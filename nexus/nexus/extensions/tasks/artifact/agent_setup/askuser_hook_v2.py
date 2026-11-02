"""AskUserQuestion hook v2 — uses an LLM via the in-sandbox Anthropic proxy
to choose context-aware answers, instead of always picking the first option.

Falls back to the v1 'first option' behavior on any failure, so the batch run
keeps progressing even if the proxy is down or the LLM output is malformed.

Env knobs:
- NEXUS_ASKUSER_HOOK_DISABLE_LLM=1 → force v1 first-option behavior
- NEXUS_ASKUSER_HOOK_MODEL          → override the simulated-user model
- NEXUS_ASKUSER_HOOK_TIMEOUT        → urllib timeout in seconds (default 30)
"""

import json
import os
import re
import sys
import urllib.request


HOOK_TIMEOUT = float(os.environ.get('NEXUS_ASKUSER_HOOK_TIMEOUT', '30'))
HOOK_MODEL = os.environ.get('NEXUS_ASKUSER_HOOK_MODEL', 'claude-opus-4-7')
# Max tokens for the simulated-user answer. Was hardcoded 500, which truncated verbose
# models (e.g. opus-4-8) — the strict-JSON reply then fails to parse and falls back to the
# first option. Kept generous + env-configurable, mirroring stop_hook's HOOK_MAX_TOKENS.
HOOK_MAX_TOKENS = int(os.environ.get('NEXUS_ASKUSER_HOOK_MAX_TOKENS', '64000'))
MAX_CONTEXT_CHARS = int(os.environ.get('NEXUS_ASKUSER_HOOK_CONTEXT_CHARS', '3000'))
RECENT_ASSISTANT_TURNS = int(os.environ.get('NEXUS_ASKUSER_HOOK_RECENT_TURNS', '2'))


def _detect_prompt_language(text: str) -> str:
    """Return 'zh' if the prompt is mostly Chinese, else 'en'. Mirrors stop_hook.py."""
    if not text:
        return 'en'
    cjk_count = sum(1 for c in text if '一' <= c <= '鿿')
    return 'zh' if cjk_count > 20 else 'en'


SYSTEM_PROMPT = (
    'You are simulating a user collaborating with an AI coding assistant during a '
    'NON-INTERACTIVE batch run. The assistant just invoked AskUserQuestion. Your '
    "job is to choose answer(s) for each question on the user's behalf so the "
    'assistant can keep working.\n\n'
    '## Rules\n'
    '- Use the recent assistant context (in <context> tags) to ground your choice '
    'in the actual task at hand.\n'
    '- Pick option label(s) that best match the spirit of the original task and '
    'keep the work moving toward IMPLEMENTATION.\n'
    '- Single-select: choose exactly 1 option. Multi-select: choose 1+ options as '
    'appropriate.\n'
    '- Prefer options labeled "(Recommended)" or that look like sensible defaults '
    'unless the context strongly suggests another choice.\n'
    '- Never invent labels or modify the text. Use the EXACT label string as it '
    'appears in the options list — do NOT translate or rewrite labels even when '
    'the original task is in another language (e.g. Chinese). Labels stay verbatim.\n'
    '- Reply with STRICT JSON only — no preamble, no markdown, no explanation.\n\n'
    '## Output schema\n'
    '{"answers": [{"choices": ["<exact label 1>", ...]}, ...]}\n\n'
    'There is one entry in "answers" per question, in the same order as the '
    'questions appear. Each entry\'s "choices" is a list of label strings (length '
    '1 for single-select, 1+ for multi-select).\n\n'
    'Examples:\n'
    '  Single-select x2: '
    '{"answers": [{"choices": ["Right sticky panel (Recommended)"]}, '
    '{"choices": ["Color block placeholder (Recommended)"]}]}\n'
    '  Multi-select:     '
    '{"answers": [{"choices": ["Email", "SMS"]}]}'
)


def _emit(reason: str) -> None:
    print(
        json.dumps({
            'hookSpecificOutput': {
                'hookEventName': 'PreToolUse',
                'permissionDecision': 'deny',
                'permissionDecisionReason': reason,
            }
        })
    )
    sys.exit(0)


def _read_original_user_prompt(transcript_path: str) -> str:
    """Return the first 'real' user prompt — the actual task brief.

    The simulated user must anchor its choices to the original task. Skip
    system-reminder blocks and hook-injection echoes.
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return ''
    try:
        with open(transcript_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                msg = evt.get('message') or evt
                if msg.get('role') != 'user':
                    continue
                content = msg.get('content', '')
                texts: list[str] = []
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get('type') == 'text':
                            t = b.get('text', '')
                            if t:
                                texts.append(t)
                elif isinstance(content, str):
                    texts.append(content)
                for t in texts:
                    s = t.strip()
                    if not s:
                        continue
                    if s.startswith('<system-reminder>') or 'PreToolUse:' in s[:200]:
                        continue
                    if s.startswith('User has answered'):
                        continue
                    raw = s
                    for marker in ('<command-args>', 'ARGUMENTS:'):
                        idx = raw.find(marker)
                        if idx >= 0:
                            raw = raw[idx + len(marker) :]
                    raw = raw.strip()
                    if len(raw) > 1500:
                        raw = raw[:1500] + '...[truncated]'
                    return raw
    except Exception:
        return ''
    return ''


def _read_recent_assistant_context(transcript_path: str, max_turns: int) -> str:
    """Return the last N assistant text turns concatenated (most recent last)."""
    if not transcript_path or not os.path.exists(transcript_path):
        return ''
    texts: list[str] = []
    try:
        with open(transcript_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                msg = evt.get('message') or evt
                if msg.get('role') != 'assistant':
                    continue
                content = msg.get('content', '')
                text = ''
                if isinstance(content, list):
                    parts = [b.get('text', '') for b in content if isinstance(b, dict) and b.get('type') == 'text']
                    text = '\n'.join(p for p in parts if p)
                elif isinstance(content, str):
                    text = content
                if text.strip():
                    texts.append(text)
    except Exception:
        return ''
    recent = texts[-max_turns:] if max_turns > 0 else texts
    joined = '\n\n---\n\n'.join(recent)
    if len(joined) > MAX_CONTEXT_CHARS:
        joined = joined[-MAX_CONTEXT_CHARS:]
    return joined


def _format_questions_for_prompt(questions: list[dict]) -> str:
    blocks = []
    for i, q in enumerate(questions, 1):
        opts = q.get('options', []) or []
        opts_text = (
            '\n'.join(f'    - {opt.get("label", "")}: {opt.get("description", "")}' for opt in opts)
            or '    (no options provided)'
        )
        kind = 'multi-select' if q.get('multiSelect') else 'single-select'
        blocks.append(
            f'Q{i} ({kind}): {q.get("question", "")}\n  Header: {q.get("header", "")}\n  Options:\n{opts_text}'
        )
    return '\n\n'.join(blocks)


def _ask_proxy(questions: list[dict], context: str, original_task: str = '') -> list[list[str]]:
    """Call the in-sandbox Anthropic proxy. Return list-of-choices-per-question."""
    base_url = os.environ.get('ANTHROPIC_BASE_URL', '').rstrip('/')
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not base_url:
        raise RuntimeError('ANTHROPIC_BASE_URL not set')

    task_block = f'<original_task>\n{original_task}\n</original_task>\n\n' if original_task else ''
    user_msg = (
        f'{task_block}'
        f'<recent_assistant_context>\n{context or "(no recent assistant context)"}\n</recent_assistant_context>\n\n'
        f'The assistant is asking:\n\n{_format_questions_for_prompt(questions)}\n\n'
        'Anchor your choices to the original task brief above. Reply with strict JSON.'
    )

    payload = {
        'model': HOOK_MODEL,
        'max_tokens': HOOK_MAX_TOKENS,
        'system': [
            {
                'type': 'text',
                'text': SYSTEM_PROMPT,
                'cache_control': {'type': 'ephemeral'},
            }
        ],
        'messages': [{'role': 'user', 'content': user_msg}],
    }
    req = urllib.request.Request(
        f'{base_url}/v1/messages',
        method='POST',
        headers={
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        },
        data=json.dumps(payload).encode('utf-8'),
    )
    with urllib.request.urlopen(req, timeout=HOOK_TIMEOUT) as resp:
        body = json.loads(resp.read())
    blocks = body.get('content', [])
    raw = '\n'.join(b.get('text', '') for b in blocks if isinstance(b, dict) and b.get('type') == 'text').strip()

    # LLM occasionally wraps in ```json ... ``` despite instructions; strip it.
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    m = re.search(r'\{[\s\S]*\}', raw)
    if not m:
        raise ValueError(f'no JSON object in LLM output: {raw[:200]}')
    parsed = json.loads(m.group())
    answers = parsed.get('answers', [])
    out = []
    for ans in answers:
        if isinstance(ans, dict):
            out.append([str(c) for c in ans.get('choices', [])])
        else:
            out.append([])
    return out


def _validate_choices(questions: list[dict], chosen: list[list[str]]) -> list[list[str]]:
    """Map LLM-returned choices to actual option labels; fall back to first option."""
    fixed: list[list[str]] = []
    for i, q in enumerate(questions):
        opts = q.get('options', []) or []
        labels = [str(opt.get('label', '')) for opt in opts]
        first_label = labels[0] if labels else 'first option'

        got = chosen[i] if i < len(chosen) else []
        valid = []
        for c in got:
            if c in labels:
                valid.append(c)
            else:
                # Tolerate small drift: substring match in either direction.
                hit = next(
                    (lbl for lbl in labels if lbl and (lbl in c or c in lbl)),
                    None,
                )
                if hit and hit not in valid:
                    valid.append(hit)
        if not valid:
            valid = [first_label]
        fixed.append(valid)
    return fixed


def _v1_first_option(questions: list[dict]) -> list[list[str]]:
    out = []
    for q in questions:
        opts = q.get('options', []) or []
        out.append([opts[0].get('label', 'first option') if opts else 'first option'])
    return out


def _format_reason(questions: list[dict], chosen: list[list[str]], original_task: str = '') -> str:
    """Format the permissionDecisionReason carrying the chosen answers.

    Wrapper text is rendered in the original task's natural language so the
    agent's downstream system-reminder mirrors the user's language. The
    question text and chosen labels stay verbatim — translating them would
    change semantics (labels in particular must round-trip exactly).
    """
    parts = []
    for q, choices in zip(questions, chosen, strict=False):
        ans_text = ', '.join(choices) if choices else 'first option'
        parts.append(f'"{q.get("question", "?")}"="{ans_text}"')
    if _detect_prompt_language(original_task) == 'zh':
        return '用户已回答你的问题：' + '，'.join(parts) + '。请基于这些答案继续。'
    return (
        'User has answered your questions: '
        + ', '.join(parts)
        + ". You can now continue with the user's answers in mind."
    )


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        # If we can't parse hook input, deny silently with no answer (Claude will see
        # the deny + no permissionDecisionReason and stop trying).
        _emit(
            'User has answered your questions: (unparseable hook input). '
            "You can now continue with the user's answers in mind."
        )

    tool_input = data.get('tool_input') or data.get('toolInput') or {}
    questions = tool_input.get('questions', []) or []
    transcript_path = data.get('transcript_path') or data.get('transcriptPath') or ''

    # Read once and reuse for both the LLM call and the reason wrapper. Cheap
    # (one-time scan of the transcript) and we want the wrapper to know the
    # task language even on the v1 fallback path.
    original_task = _read_original_user_prompt(transcript_path)

    if not questions:
        _emit(_format_reason([], [], original_task=original_task))

    use_llm = os.environ.get('NEXUS_ASKUSER_HOOK_DISABLE_LLM') not in ('1', 'true', 'True')

    chosen: list[list[str]]
    if use_llm:
        try:
            ctx = _read_recent_assistant_context(transcript_path, RECENT_ASSISTANT_TURNS)
            raw = _ask_proxy(questions, ctx, original_task=original_task)
            chosen = _validate_choices(questions, raw)
        except Exception as e:
            sys.stderr.write(f'askuser_hook_v2: LLM call failed, falling back to v1: {e}\n')
            chosen = _v1_first_option(questions)
    else:
        chosen = _v1_first_option(questions)

    _emit(_format_reason(questions, chosen, original_task=original_task))


if __name__ == '__main__':
    main()
