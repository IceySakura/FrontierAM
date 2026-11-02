"""Typed contract shared by Harbor anti-hack strategies and consumers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from nexus.extensions.harbor_bridge.agent_trajectory import T3AgentTrajectory


class AntiHackContext(BaseModel):
    """Host-side facts available to a trajectory strategy."""

    model_config = ConfigDict(extra='forbid')

    trajectory: T3AgentTrajectory
    task_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class AntiHackResult(BaseModel):
    """Strategy-independent result returned through TaskOutput.extra_fields."""

    model_config = ConfigDict(extra='forbid')

    status: Literal['completed', 'unavailable', 'error']
    detected: bool = False
    violations: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


__all__ = ['AntiHackContext', 'AntiHackResult']
