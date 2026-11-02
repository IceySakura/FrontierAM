"""Rule: empty_response."""

from .base import Rule
from ..models import Finding


def scan(ctx):
    out = []
    for i, m in ctx.assistant_messages():
        calls = m.get('tool_calls') if isinstance(m.get('tool_calls'), list) else []
        if not ctx.content(i).strip() and not ctx.reasoning(i).strip() and not calls:
            out.append(Finding(i, 'assistant 的 content、reasoning_content、tool_calls 全为空', ['content']))
    return out


RULE = Rule('empty_response', '空回包', scan)
