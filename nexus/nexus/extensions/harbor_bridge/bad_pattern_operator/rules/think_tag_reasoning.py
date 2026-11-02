"""Rule: think_tag_reasoning."""

from .base import Rule
from ..models import Finding
from ..patterns import HASHED_THINK_TAG, THINK_TAG


def scan(ctx):
    out = []
    for i, _ in ctx.assistant_messages():
        reasoning = ctx.reasoning(i)
        for match in THINK_TAG.finditer(reasoning):
            if (
                HASHED_THINK_TAG.fullmatch(match.group(0))
                or not reasoning[: match.start()].strip()
                or not reasoning[match.end() :].strip()
            ):
                out.append(Finding(i, 'think 标签泄漏到 reasoning', ['reasoning_content'], match.group(0)))
                break
    return out


RULE = Rule('think_tag_reasoning', 'think 标签泄漏到 reasoning', scan)
