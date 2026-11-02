"""Shared clients, scaffolding, judge launch factories, and infer harness builders."""

from __future__ import annotations

from nexus.extensions.tasks.artifact.agent_setup import claude_config, claude_hooks, clients
from nexus.extensions.tasks.artifact.agent_setup.claude_config import (
    default_code_review_claude_input,
    default_judge_claude_input,
)


__all__ = [
    'clients',
    'claude_config',
    'claude_hooks',
    'default_judge_claude_input',
    'default_code_review_claude_input',
]
