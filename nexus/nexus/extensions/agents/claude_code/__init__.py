"""Claude Code Agent.

This module provides an agent that runs Claude Code in a runtime sandbox,
using the host-side OpenAI client for LLM calls via an Anthropic API proxy.

Example:
    from nexus.extensions.agents.claude_code import ClaudeCodeInput

    agent = agent_registry.get('claude_code')
    result = await agent(ClaudeCodeInput(
        user_prompt='Create a hello world Python script',
        working_dir='/workspace',
    ))

    print(f'Success: {result.success}')
    print(f'Output: {result.output}')
"""
