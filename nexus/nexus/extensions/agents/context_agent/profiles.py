"""Stable serving profiles layered over the unified Context Agent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ContextAgentProfile:
    key: str
    agent_name: str
    task_name: str
    document_mode: Literal['preloaded', 'path-native']
    execution_mode: Literal['host', 'runtime']
    tool_names: tuple[str, ...]
    runtime_namespace: str | None = None
    sandbox_root: str | None = None
    frozen_sft_contract: bool = False
    max_parallel_calls: int = 0


GENERAL_PROFILE = ContextAgentProfile(
    key='general',
    agent_name='context_agent_general',
    task_name='context_agent_general_task',
    document_mode='path-native',
    execution_mode='runtime',
    tool_names=(
        'curl',
        'bash',
        'python',
        'file_search_msearch',
        'file_search_mclick',
        'grep',
        'read_lines',
    ),
    runtime_namespace='context_agent_general',
)

PROFILES = {GENERAL_PROFILE.key: GENERAL_PROFILE}


def get_profile(key: str) -> ContextAgentProfile:
    """Return a profile by its internal key or public agent name."""
    if key in PROFILES:
        return PROFILES[key]
    for profile in PROFILES.values():
        if key == profile.agent_name:
            return profile
    raise KeyError(f'unknown Context Agent profile: {key!r}')
