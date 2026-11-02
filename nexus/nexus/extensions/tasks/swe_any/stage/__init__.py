"""SWE-Any pipeline stages: ``parse`` → ``select`` → ``mask`` → ``eval``.

Each stage is its own subpackage with the same ``heuristics`` (deterministic) +
``prompts`` (agentic) split. Stages depend only on the shared libraries in
``..utils`` (graph model, SCIP toolchain, source editing, test classification),
never on each other.
"""

from __future__ import annotations
