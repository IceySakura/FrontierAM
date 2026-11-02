"""Rule: terminus_tool_parse_fail."""

import json
import re

from .base import Rule
from ..models import Finding
from ..patterns import HARD_REJECTION


def scan(ctx):
    out = []
    for i, _ in ctx.assistant_messages():
        if not re.search(r'terminus|terminal[_-]?bench', json.dumps(ctx.messages[: i + 1], ensure_ascii=False), re.I):
            continue
        for j in range(i + 1, min(i + 3, len(ctx.messages))):
            nxt = ctx.messages[j]
            if not isinstance(nxt, dict):
                continue
            if nxt.get('role') == 'tool':
                continue
            if nxt.get('role') in {'user', 'system'} and HARD_REJECTION.search(ctx.content(j)):
                out.append(Finding(i, 'Terminus 命令块被脚手架判为无法解析', ['content'], ctx.content(i)))
            break
    return out


RULE = Rule('terminus_tool_parse_fail', 'Terminus 工具调用解析失败', scan)
