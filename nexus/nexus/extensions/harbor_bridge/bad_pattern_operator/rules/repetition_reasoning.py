"""Rule: repetition_reasoning."""

import difflib
import re

from .base import Rule
from ..models import Finding
from ..utils import call_signature, text


def scan(ctx):
    out = []
    by_turn = {}
    for e in ctx.calls:
        by_turn.setdefault(e.message_index, []).append(e)
    previous = None
    for i, _ in ctx.assistant_messages():
        reasoning = re.sub(r'\s+', ' ', ctx.reasoning(i)).strip()
        events = by_turn.get(i, [])
        signatures = []
        results = []
        valid = bool(events)
        for e in events:
            sig = call_signature(e.call)
            if sig is None or e.result is None:
                valid = False
                break
            signatures.append(sig)
            results.append(text(e.result))
        state = (signatures, results, re.sub(r'\s+', ' ', ctx.content(i)).strip()) if valid else None
        if (
            len(reasoning) >= 60
            and state is not None
            and previous
            and previous[1] == state
            and difflib.SequenceMatcher(None, previous[0], reasoning).ratio() >= 0.96
        ):
            out.append(
                Finding(
                    i, '跨 assistant 轮 reasoning 近似重复，且工具动作和结果无进展', ['reasoning_content'], reasoning
                )
            )
        previous = (reasoning, state) if len(reasoning) >= 60 and state is not None else None
    return out


RULE = Rule('repetition_reasoning', '思维链复读且状态无进展', scan)
