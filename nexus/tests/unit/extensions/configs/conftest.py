"""Missing optional runtime dependencies skip a case here instead of failing it.

``load_config`` resolves a runtime yaml's ``imports:``, and ``runtime/swe_runtime.yaml``
imports ``nexus.trace.processors.langfuse_tracker``. Neither ``langfuse`` nor ``litellm``
is declared in the root ``pyproject.toml`` -- they live in the per-agent sub-packages --
so the documented local flow (``uv sync --extra dev`` then ``uv run pytest tests/unit``)
left 115 of 265 cases red while CI, whose image carries both, stayed green.

A module-level ``pytest.importorskip`` would instead turn the whole file green on such an
interpreter, and this file holds the guard that catches a shared agent profile drifting
from production -- silently skipping all of it is worse than failing. Converting only the
import error, per case, keeps the other 150 running.
"""

from __future__ import annotations

import pytest


# Declared by the agent sub-packages, not by the root project.
OPTIONAL_RUNTIME_MODULES = frozenset({'langfuse', 'litellm'})


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item):
    try:
        return (yield)
    except ModuleNotFoundError as exc:
        if (exc.name or '').split('.')[0] in OPTIONAL_RUNTIME_MODULES:
            pytest.skip(f'runtime import needs {exc.name}')
        raise
