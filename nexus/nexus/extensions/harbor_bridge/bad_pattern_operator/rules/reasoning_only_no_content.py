"""Rule: reasoning_only_no_content."""

from .base import Rule
from ..models import Finding


def scan(ctx):
    out = []
    for i, m in ctx.assistant_messages():
        calls = m.get('tool_calls') if isinstance(m.get('tool_calls'), list) else []
        reasoning = ctx.reasoning(i)
        if reasoning.strip() and not ctx.content(i).strip() and not calls:
            out.append(Finding(i, '存在 reasoning，但没有正文或结构化工具调用', ['reasoning_content'], reasoning))
    return out


RULE = Rule('reasoning_only_no_content', '仅 reasoning、无正文或结构化行动', scan)
