"""Yb VLM oneagent task module.

Importing this package registers the task-local reward functions into the
shared :data:`nexus.extensions.tasks.mm_agent.reward_system.reward.REWARD_FUNC_REGISTRY`,
so that ``vl_reward_system`` can route to them whenever ``task_input.reward_func``
matches — both for ``yb_vlm_oneagent`` and for ``mm_agent_task``.

``yb_rubric`` is registered in its original in-process form. Deferred scoring
is now controlled explicitly by ``task_input.reward_mode`` inside
``task.py`` instead of swapping the registry implementation at import time.
This keeps the default behavior inert when the package is imported from main.
"""

from nexus.extensions.projects.yb_vlm_oneagent.reward_system.tool_use_label import (
    tool_use_label_reward,
)
from nexus.extensions.projects.yb_vlm_oneagent.reward_system.yb_rubric import (
    yb_rubric_reward,
)
from nexus.extensions.tasks.mm_agent.reward_system.reward import (
    REWARD_FUNC_REGISTRY,
)


# Idempotent: if this package is imported twice, the second registration is
# a no-op overwrite of the same function reference.
REWARD_FUNC_REGISTRY['yb_rubric'] = yb_rubric_reward
REWARD_FUNC_REGISTRY['tool_use_label'] = tool_use_label_reward
