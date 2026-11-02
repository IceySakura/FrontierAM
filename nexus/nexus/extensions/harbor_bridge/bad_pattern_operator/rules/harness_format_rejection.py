"""Rule: harness_format_rejection."""

from .base import Rule
from ..models import Finding
from ..patterns import HARD_REJECTION


def scan(ctx):
    out = []
    for i, _ in ctx.assistant_messages():
        for j in range(i + 1, min(i + 3, len(ctx.messages))):
            nxt = ctx.messages[j]
            if not isinstance(nxt, dict):
                continue
            if nxt.get('role') == 'tool':
                continue
            if nxt.get('role') in {'user', 'system'} and HARD_REJECTION.search(ctx.content(j)):
                out.append(Finding(i, '上一条 assistant 输出被脚手架判为格式或解析错误', ['content'], ctx.content(i)))
            break
    return out


RULE = Rule('harness_format_rejection', '脚手架报格式或解析错误', scan)
