"""Shared public output fields for Context Agent serving profiles."""

from __future__ import annotations

from nexus.core import AgentRunOutput, WithFinalAnswer, WithMainTrajectory
from nexus.extensions.agents.base.schema import TerminationReason


class ContextAgentRunOutput(AgentRunOutput, WithMainTrajectory, WithFinalAnswer):
    """Common rollout result and termination metadata."""

    exit_status: TerminationReason = TerminationReason.SUCCESS
    tool_rounds_used: int = 0
    tool_calls_requested: int = 0
    tool_calls_executed: int = 0
    tool_calls_skipped: int = 0
    hit_max_turn_limit: bool = False
    forced_finalization: bool = False
    finalization_failed: bool = False
    response_budget_exhausted: bool = False


__all__ = ['ContextAgentRunOutput']
