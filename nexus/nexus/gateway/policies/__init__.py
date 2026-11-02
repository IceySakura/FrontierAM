"""Built-in direct scaffold policies."""

from __future__ import annotations

from nexus.gateway.policies.claude_code import CLAUDE_CODE_POLICY
from nexus.gateway.policies.codex import CODEX_POLICY
from nexus.gateway.policies.generic import GENERIC_POLICY
from nexus.gateway.policies.hycode import HYCODE_POLICY
from nexus.gateway.policies.nexus_main_agent import NEXUS_MAIN_AGENT_POLICY
from nexus.gateway.policies.workbuddy import WORKBUDDY_POLICY
from nexus.gateway.scaffold_policy import register_scaffold_policy


register_scaffold_policy(CLAUDE_CODE_POLICY)
register_scaffold_policy(CODEX_POLICY)
register_scaffold_policy(NEXUS_MAIN_AGENT_POLICY)
register_scaffold_policy(GENERIC_POLICY)
register_scaffold_policy(HYCODE_POLICY)
register_scaffold_policy(WORKBUDDY_POLICY)

__all__ = [
    'CLAUDE_CODE_POLICY',
    'CODEX_POLICY',
    'GENERIC_POLICY',
    'HYCODE_POLICY',
    'NEXUS_MAIN_AGENT_POLICY',
    'WORKBUDDY_POLICY',
]
