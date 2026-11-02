"""Windows Agent Arena (WAA) evaluation task for Nexus.

This module implements the WAA benchmark evaluation pipeline:
- Task setup (environment initialization)
- Agent execution (via navi_agent)
- Task evaluation (getters + metrics)
"""

from nexus.extensions.tasks.waa.task import waa_task  # noqa: F401
