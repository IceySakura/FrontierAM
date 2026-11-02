"""Rule: reasoning_filler_loop."""

import re

from .base import Rule
from ..models import Finding


FILLER_SPLIT = re.compile(r'[\n。.!?！？]+')
FILLER_HEAD = re.compile(
    r"^(?:let me|let['’]s|i['’]ll|i will|now|ok|okay|here|here goes|emit|emitting|go|going|write|writing|produce|producing|output|final|finally|done|yes|right|good|alright|so)\b",
    re.I,
)


def scan(ctx):
    out = []
    for i, _ in ctx.assistant_messages():
        reasoning = ctx.reasoning(i)
        current: list[str] = []
        best: list[str] = []
        for segment in FILLER_SPLIT.split(reasoning):
            line = re.sub(r'\s+', ' ', re.sub(r'^[\s>*\-#`]+', '', segment)).strip()
            tokens = len(re.findall(r'[A-Za-z0-9_]+', line)) + len(re.findall(r'[\u4e00-\u9fff]', line))
            if line and tokens <= 6 and FILLER_HEAD.match(line) and not re.search(r'[`/]', line):
                current.append(line)
                if len(current) > len(best):
                    best = list(current)
            else:
                current = []
        if len(best) >= 8 and len({x.casefold() for x in best}) >= 3:
            out.append(Finding(i, 'reasoning 连续出现至少 8 句极短填充句', ['reasoning_content'], reasoning))
    return out


RULE = Rule('reasoning_filler_loop', '思维链填充词空转', scan)
