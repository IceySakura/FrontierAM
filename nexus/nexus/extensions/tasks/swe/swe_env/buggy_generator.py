#!/usr/bin/env python3
"""SWE-Env *buggy* (no-op patch) batch generator for `nexus run-batch`.

Sanity check that "doing nothing" does NOT pass: every instance is run in oracle mode, but its oracle patch is replaced with a no-op patch, so the agent's effective change is nothing. The task still runs the real evaluation; a well-formed instance should end up ``unresolved`` (status ``unresolved`` / score 0). Any instance that comes back ``resolved`` here is suspect — its tests pass without any fix.

Like the oracle generator, every loaded instance has its ``dataset_type`` forced to ``"swe-env"``.

Why a no-op patch instead of an empty string:
  In oracle mode the task reads ``instance['patch']``; an empty patch raises "No oracle patch available" and the evaluation never runs. To force a real eval run we instead apply a patch that changes nothing meaningful: it creates a single throwaway file ``.nexus_empty_patch`` at the repo root. This applies cleanly under both ``git apply`` and ``patch -p1`` (verified), so the eval always executes; the created file does not affect any test outcome.

Usage:
  uv run nexus run-batch -g nexus/extensions/tasks/swe/swe_env/buggy_generator.py \
      -d ./outputs/swe_env_buggy_batch.db --max-concurrent 20

Environment variables: identical to oracle_generator.py (see its module docstring).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.swe.swe_env.oracle_generator import (
    batch_metadata,
    build_configs,
)
from nexus.extensions.tasks.swe.swe_env.oracle_generator import (
    get_total_count as _get_total_count,
)


# A model patch that changes nothing relevant: it only creates a throwaway file at the repo root. Kept non-empty so oracle mode does not bail out before evaluation, yet semantically a no-op.
_NOOP_PATCH = (
    'diff --git a/.nexus_empty_patch b/.nexus_empty_patch\n'
    'new file mode 100644\n'
    '--- /dev/null\n'
    '+++ b/.nexus_empty_patch\n'
    '@@ -0,0 +1 @@\n'
    '+empty\n'
)


def _replace_with_noop_patch(instance: dict[str, Any]) -> None:
    """Swap the oracle patch for the no-op patch so applying it changes nothing meaningful."""
    instance['patch'] = _NOOP_PATCH
    # Drop alternative oracle sources so the task can't fall back to a real fix.
    instance.pop('bug_patch', None)
    instance.pop('fix_patch', None)


def generate_configs() -> Iterator[BatchItem]:
    return build_configs(_replace_with_noop_patch)


def get_total_count() -> int:
    return _get_total_count()


def get_batch_metadata() -> dict[str, Any]:
    return batch_metadata('buggy')
