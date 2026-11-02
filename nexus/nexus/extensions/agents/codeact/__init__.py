"""CodeAct agent implementation for nexus framework.

This module provides the CodeAct agent which uses executable code actions to
interact with the environment. It includes:

- codeact: Main agent that runs the CodeAct loop
- Tools: terminal, file_editor, task_tracker, think, finish

Usage:
    from nexus.core import Registry
    Registry.import_module('nexus.extensions.agents.codeact')

    agent_fn = agent_registry.get('codeact')

Example:
    from nexus.extensions.agents.codeact import CodeActAgentInput, CodeActAgentOutput

    input = CodeActAgentInput(
        user_prompt='Fix the bug in function foo...',
        max_iterations=100,
    )

    # Run with ExecutionContext
    output = await run_codeact_agent(input)
    print(output.execution_status)
"""
