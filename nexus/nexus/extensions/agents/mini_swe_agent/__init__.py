"""White-box mini-swe-agent for coding tasks (SWE-bench) on the nexus framework.

A transparent, Python-level port of upstream mini-swe-agent
(https://github.com/SWE-agent/mini-swe-agent) that runs the whole agent loop on
nexus interfaces. The loop runs host-side and drives the LLM through
``ctx.openai_client``; bash commands run in the sandbox via the RUNTIME
``mini_swe_agent/bash`` tool, each in a fresh ``bash -c`` subshell (mini's
non-persistent environment). It plugs into the standard ``swe_task``, which
re-derives the patch via ``git diff``.

Import discipline (why this file has no imports):
    The RUNTIME ``bash`` tool is resolved *inside the sandbox* via
    ``importlib.import_module('nexus.extensions.agents.mini_swe_agent.tools')``,
    which first executes this package ``__init__``. To keep the sandbox import
    chain free of host-only dependencies (notably ``jinja2``, pulled in by
    ``agent`` -> ``mini_core``), this module must NOT import ``agent``. The agent
    and tool are registered lazily from ``pyproject.toml`` metadata
    (``module_registry.discover_and_register()``), so no eager import is needed:

        from nexus.core import agent_registry
        agent_fn = agent_registry.get('mini_swe_agent')  # imports .agent on demand

    Import the concrete types straight from the submodule when needed:

        from nexus.extensions.agents.mini_swe_agent.agent import MiniSweAgentInput
"""
