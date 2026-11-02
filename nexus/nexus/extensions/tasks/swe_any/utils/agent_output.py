"""Read back the output an agentic pass wrote (Parse seed policy, Select agentic output).

Both passes end the same way — the agent writes one JSON object to a pinned sandbox
path — so both need the same tolerance for a file that carries prose or fences around
the object. Kept out of the two ``prompts`` modules because it belongs to reading the
answer, not to asking the question.
"""

from __future__ import annotations

import json

from typing import Any


def parse_agent_json(text: str) -> dict[str, Any]:
    """Parse an agent-written JSON object from file contents (or noisy text).

    Accepts a raw JSON object or the first embedded ``{...}`` block.
    Raises ``ValueError`` when no object can be parsed.
    """
    raw = (text or '').strip()
    if not raw:
        raise ValueError('empty agent output')

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback: scan for the first balanced {...} span, tracking string/escape
    # state so braces inside JSON string literals are not miscounted.
    span = _first_json_object_span(raw)
    if span is None:
        raise ValueError('no JSON object in agent output')
    data = json.loads(span)
    if not isinstance(data, dict):
        raise ValueError('parsed JSON is not an object')
    return data


def _first_json_object_span(text: str) -> str | None:
    """Return the first top-level ``{...}`` substring, ignoring braces in strings."""
    depth = 0
    start = None
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None
