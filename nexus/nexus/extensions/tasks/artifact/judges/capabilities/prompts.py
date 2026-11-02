"""Prompt capability — load a judge's default prompt template.

Every strategy ships a default prompt under ``judges/prompts/judge_*.txt`` that
its ``config.prompt`` override falls back to. This module is the single place
that knows where those files live, replacing the per-strategy ``_PROMPTS_DIR``
+ ``_load_default_prompt()`` boilerplate.
"""

from __future__ import annotations

import re

from pathlib import Path


# ``capabilities/`` sits one level below ``judges/``; prompts live in ``judges/prompts/``.
PROMPTS_DIR = Path(__file__).resolve().parents[1] / 'prompts'

# A placeholder is ``{word}`` — letters/digits/underscore only. JSON/CSS/JS braces
# like ``{"winner": "A"}`` or ``{ margin: 0 }`` don't match, so they pass through
# untouched.
_PLACEHOLDER_RE = re.compile(r'\{(\w+)\}')


def load_prompt(name: str) -> str:
    """Read ``prompts/<name>`` (``name`` includes the ``.txt`` suffix), stripped."""
    return (PROMPTS_DIR / name).read_text(encoding='utf-8').strip()


def render_prompt(template: str, /, **values: object) -> str:
    """Substitute ``{key}`` placeholders in a single pass, without ``str.format``'s fragility.

    Done as one regex scan over the ORIGINAL template so a substituted value that
    itself contains ``{another_key}`` is NOT re-substituted (the naive
    ``for k: result = result.replace(...)`` loop had that injection bug — e.g. a
    rubric text containing the literal ``{query}`` got clobbered by the later
    ``query=`` pass). Only ``{word}`` matches, so JSON/CSS/JS braces
    (``{"winner": "A"}``) pass through untouched, and an unknown/unfilled
    placeholder stays visible as literal ``{key}`` rather than crashing or
    emptying the prompt.
    """

    def _sub(m: re.Match[str]) -> str:
        key = m.group(1)
        return str(values[key]) if key in values else m.group(0)

    return _PLACEHOLDER_RE.sub(_sub, template)


__all__ = ['PROMPTS_DIR', 'load_prompt', 'render_prompt']
