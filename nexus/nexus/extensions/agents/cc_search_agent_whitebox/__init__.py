"""White-box Claude Code style search agent.

Reproduces the behavior of the packaged Claude Code ``search`` subagent
(prompt + ``Bash`` / ``WebSearch`` / ``WebFetch`` tool surface) but
drives the LLM and tool-call loop entirely inside Nexus on top of
``main_agent``. This gives full visibility into every inference, tool
input, tool output, and termination signal — unlike the black-box CC
search agent which delegates to the opaque ``claude`` CLI.
"""

from nexus.extensions.agents.cc_search_agent_whitebox.agent import (
    DEFAULT_REQUIRED_TOOLS,
    CCSearchAgentWhiteboxInput,
    CCSearchAgentWhiteboxOutput,
    run_cc_search_agent_whitebox,
)


__all__ = [
    'CCSearchAgentWhiteboxInput',
    'CCSearchAgentWhiteboxOutput',
    'DEFAULT_REQUIRED_TOOLS',
    'run_cc_search_agent_whitebox',
]
