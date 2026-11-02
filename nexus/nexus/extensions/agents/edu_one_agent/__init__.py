"""edu_one_agent implementation for nexus framework."""

from nexus.extensions.agents.edu_one_agent.agent import (
    EduOneAgent,
    run_edu_one_agent,
)
from nexus.extensions.agents.edu_one_agent.tools import (
    edu_search_tool,
    web_search_tool,
)


__all__ = [
    # Agent
    'EduOneAgent',
    'run_edu_one_agent',
    # Tools
    'edu_search_tool',
    'web_search_tool',
]
