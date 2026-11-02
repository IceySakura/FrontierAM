"""Opt-in bad-pattern settlement shared by Context Agent task wrappers."""

from __future__ import annotations

import logging

from typing import Any

from nexus.extensions.harbor_bridge.anti_hack.strategies.anthropic_blocks import normalize_anthropic_block_messages
from nexus.extensions.harbor_bridge.bad_pattern_operator import (
    ToolCallBadPatternPolicy,
    apply_tool_call_bad_pattern_penalty,
    detect_tool_call_bad_patterns,
    merge_tool_schemas,
)

from .output import agent_termination_reason


logger = logging.getLogger(__name__)


def apply_bad_pattern_score(
    score: float,
    agent_output: Any,
    policy: ToolCallBadPatternPolicy | None,
    *,
    extra_fields: dict[str, Any] | None = None,
    termination_reason: str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Settle configured rules once, preserving Judge evidence and existing metrics."""
    extra = dict(extra_fields or {})
    if policy is None:
        return score, extra
    try:
        messages = (
            agent_output.get('main_trajectory')
            if isinstance(agent_output, dict)
            else getattr(agent_output, 'main_trajectory', None)
        )
        if not isinstance(messages, list) or not messages:
            raise ValueError('Bad-pattern detection requires a non-empty agent main_trajectory')
        tools = (
            agent_output.get('tool_schemas')
            if isinstance(agent_output, dict)
            else getattr(agent_output, 'tool_schemas', None)
        )
        messages = normalize_anthropic_block_messages(messages) or messages
        detection = detect_tool_call_bad_patterns(
            messages,
            merge_tool_schemas(tools),
            termination_reason=termination_reason or agent_termination_reason(agent_output),
            **policy.detection_kwargs(),
        )
        penalized_score, metrics = apply_tool_call_bad_pattern_penalty(score, detection, policy)
    except Exception as exc:
        # Match Harbor's fail-open policy: detector failures must not change task reward.
        logger.exception('Context Agent bad-pattern detection failed: policy=%s', policy.id)
        extra['tool_call_bad_pattern'] = {'policy_id': policy.id, 'error': str(exc), 'error_type': type(exc).__name__}
        return score, extra

    logger.info(
        'Context Agent bad-pattern policy=%s issues=%s original_score=%s score=%s',
        policy.id,
        detection['issue_count'],
        score,
        penalized_score,
    )
    detection['policy_id'] = policy.id
    extra['tool_call_bad_pattern'] = detection
    reward_info = dict(extra.get('reward_extra_info') or {})
    reward_info.update(metrics)
    reward_info['extra_reward_metrics'] = {**(reward_info.get('extra_reward_metrics') or {}), **metrics}
    extra['reward_extra_info'] = reward_info
    return penalized_score, extra
