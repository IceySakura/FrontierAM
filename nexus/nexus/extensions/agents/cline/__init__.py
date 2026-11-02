"""Cline Agent.

Runs the Cline CLI (cline) in a runtime sandbox using headless mode.
LLM calls are routed through a local OpenAI-compatible proxy back to
the host-side Nexus OpenAI client via Socket.IO.

Example:
    from nexus.extensions.agents.cline import ClineInput

    agent = agent_registry.get('cline')
    result = await agent(ClineInput(
        user_prompt='Create a hello world Python script',
        working_dir='/workspace',
    ))

    print(f'Success: {result.success}')
    print(f'Output: {result.output}')
"""
