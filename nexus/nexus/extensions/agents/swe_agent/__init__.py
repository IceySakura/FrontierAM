"""SWE-agent implementation for nexus framework.

This module provides a port of the SWE-agent (Software Engineering Agent) to the
nexus framework. It includes:

- swe_agent: Main agent that runs the SWE-agent loop
- ch_swe_agent: The same loop and tools addressed to the model in Chinese
- Tools: bash, str_replace_editor, review_submit, each registered under both
  the `swe_agent/` and `ch_swe_agent/` namespaces

Usage:
    from nexus.core import Registry
    Registry.import_module('nexus.extensions.agents.swe_agent')

    agent_fn = agent_registry.get('swe_agent')

Example:
    from nexus.extensions.agents.swe_agent import SWEAgentInput, SWEAgentOutput

    input = SWEAgentInput(
        user_prompt='Fix the bug in function foo...',
        working_dir='/testbed',
        max_iterations=30,
    )

    # Run with ExecutionContext
    output = await run_swe_agent(input)
    print(output.submission)
"""
