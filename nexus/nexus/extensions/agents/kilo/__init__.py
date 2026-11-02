"""Kilo Code Agent.

Runs the Kilo Code CLI (@kilocode/cli) in a runtime sandbox using headless
mode (``kilo run``).  LLM calls are routed through a local OpenAI-compatible
proxy back to the host-side Nexus OpenAI client via Socket.IO.

Example:
    from nexus.extensions.agents.kilo import KiloInput

    agent = agent_registry.get('kilo')
    result = await agent(KiloInput(
        user_prompt='Create a hello world Python script',
        working_dir='/workspace',
    ))

    print(f'Success: {result.success}')
    print(f'Output: {result.output}')
"""
