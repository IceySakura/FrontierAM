"""R2 STEM mini-SWE-agent project extension."""

from nexus.extensions.projects.r2_stem.agent import (
    MiniSWEAgentStemExitStatus,
    MiniSWEAgentStemInput,
    MiniSWEAgentStemOutput,
    run_mini_swe_agent_stem,
)
from nexus.extensions.projects.r2_stem.task import (
    ScienceAgentTaskInput,
    ScienceAgentTaskOutput,
    run_science_agent_task,
)


__all__ = [
    'MiniSWEAgentStemExitStatus',
    'MiniSWEAgentStemInput',
    'MiniSWEAgentStemOutput',
    'ScienceAgentTaskInput',
    'ScienceAgentTaskOutput',
    'run_mini_swe_agent_stem',
    'run_science_agent_task',
]
