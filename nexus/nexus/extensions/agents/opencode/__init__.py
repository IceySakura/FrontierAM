"""OpenCode Agent.

Runs the OpenCode CLI (opencode) in a runtime sandbox using headless
mode (``opencode run``).  LLM calls are routed through a local
OpenAI-compatible proxy back to the host-side Nexus OpenAI client via
Socket.IO.

Example:
    from nexus.extensions.agents.opencode import OpenCodeInput

    agent = agent_registry.get('opencode')
    result = await agent(OpenCodeInput(
        user_prompt='Create a hello world Python script',
        working_dir='/workspace',
    ))

    print(f'Success: {result.success}')
    print(f'Output: {result.output}')
"""
