"""BFCL v4 (Berkeley Function Calling Leaderboard) integration for Nexus.

Provides a 'bfcl' task that evaluates LLM function calling capabilities
across 30+ test categories using BFCL's inference and evaluation pipeline.
"""

from nexus.extensions.tasks.bfclv4.task import bfcl_task as bfcl_task
