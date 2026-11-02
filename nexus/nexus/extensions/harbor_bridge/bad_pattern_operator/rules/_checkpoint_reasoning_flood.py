"""Thinking-only findings aligned with HyX checkpoint_reasoning_flood@1.

Only assistant thinking is inspected, independently per message. Existing
self-talk and literal-repetition rules retain their broader channel coverage.
"""

from ._reasoning_flood import detect_thinking
from ..context import DetectionContext
from ..models import Finding


def scan(ctx: DetectionContext) -> list[Finding]:
    findings = []
    for index, message in ctx.assistant_messages():
        # Match the source rule's field precedence, including its reasoning alias.
        for field in ('thinking', 'reasoning_content', 'reasoning'):
            thinking = message.get(field)
            if isinstance(thinking, str) and thinking.strip():
                break
        else:
            continue
        result = detect_thinking(thinking)
        if result is None:
            continue
        findings.append(
            Finding(
                message_index=index,
                message=result['detail'],
                path=[field],
                excerpt=result['excerpt'],
                severity='medium',
                detail={
                    'source_rule_id': 'checkpoint_reasoning_flood',
                    'source_rule_version': 1,
                    'channel': 'reasoning',
                    'kind': result['kind'],
                    'thinking_chars': result['thinking_chars'],
                    'filler_window': result['filler_window'],
                    'template': result['template'],
                },
                quotes=[
                    {
                        'kind': result['kind'],
                        'turn_index': index,
                        'role': 'assistant',
                        'channel': 'reasoning',
                        'snippet': result['excerpt'],
                    }
                ],
            )
        )
    return findings
