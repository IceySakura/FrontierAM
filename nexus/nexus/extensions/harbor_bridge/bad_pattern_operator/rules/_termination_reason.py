"""Helpers for trajectory-level rules driven by normalized termination reasons."""

from __future__ import annotations

from collections.abc import Iterable

from .base import Rule
from ..models import Finding


def _evidence_message_index(ctx) -> int:
    """Anchor trajectory-level evidence to the final assistant turn when possible."""
    return next(
        (
            index
            for index in range(len(ctx.messages) - 1, -1, -1)
            if isinstance(ctx.messages[index], dict) and ctx.messages[index].get('role') == 'assistant'
        ),
        max(len(ctx.messages) - 1, 0),
    )


def termination_reason_rule(
    rule_id: str,
    title: str,
    reasons: Iterable[str],
) -> Rule:
    """Build a configurable bad-pattern rule from canonical termination reasons."""
    normalized_reasons = frozenset(str(reason).strip().upper() for reason in reasons if str(reason).strip())
    if not normalized_reasons:
        raise ValueError('termination reason rule requires at least one reason')

    def scan(ctx) -> list[Finding]:
        if ctx.termination_reason not in normalized_reasons:
            return []
        return [
            Finding(
                message_index=_evidence_message_index(ctx),
                message=f'轨迹因 {ctx.termination_reason} 提前结束',
                detail={
                    'trajectory_level': True,
                    'termination_reason': ctx.termination_reason,
                },
            )
        ]

    return Rule(rule_id, title, scan)


__all__ = ['termination_reason_rule']
