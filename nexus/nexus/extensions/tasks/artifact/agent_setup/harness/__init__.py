"""Harness registry: how artifact builds the infer agent's launch config.

One place decides what a harness needs to start, so the offline generator and the RL parquet
builder cannot drift apart. Before this existed the hycode flags lived only in
``generator/generator.py``, while RL read a hand-written copy from a notebook -- and a copy
that omits ``--disallowedTools EnterPlanMode`` produces 1 file instead of 32 without any error.

Usage::

    from nexus.extensions.tasks.artifact.agent_setup import harness

    spec = harness.HarnessSpec(working_dir='/workspace', timeout=1800.0)
    agent_config = harness.build_infer_agent_config('hycode', spec)

See ``docs/developing/plan-artifact-harness-refactor.md`` for the design and the staged
migration this module is part of.
"""

from __future__ import annotations

from . import claude_code, hycode, hycode_dynamicagent
from .base import HARNESS_BUILDERS, RUNNER_OWNED_KEYS, HarnessSpec, build_infer_agent_config


# Registered here rather than by each module importing base and mutating it: one visible list
# beats scattered side effects when you need to know what is supported.
HARNESS_BUILDERS['claude_code'] = claude_code.build_agent_config
HARNESS_BUILDERS['hycode'] = hycode.build_agent_config
HARNESS_BUILDERS['hycode_dynamicagent'] = hycode_dynamicagent.build_agent_config


__all__ = [
    'HARNESS_BUILDERS',
    'RUNNER_OWNED_KEYS',
    'HarnessSpec',
    'build_infer_agent_config',
]
