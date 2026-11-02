"""WebArena-Verified benchmark task implementation.

WebArena-Verified is a verified version of the WebArena benchmark with:
- Deterministic evaluation (no LLM-as-judge)
- Type-aware value normalization
- Structured comparison for nested objects/arrays
- 812 verified tasks with human-audited reference answers
"""

from .task import (
    WebArenaVerifiedTaskInput,
    WebArenaVerifiedTaskOutput,
    run_webarena_verified_task,
)


__all__ = [
    'WebArenaVerifiedTaskInput',
    'WebArenaVerifiedTaskOutput',
    'run_webarena_verified_task',
]
