"""CodeBuddy CLI Agent.

Runs the Tencent CodeBuddy CLI (@tencent-ai/codebuddy-code) in a runtime
sandbox.  LLM calls are routed through a local OpenAI-compatible proxy back
to the host-side Nexus OpenAI client via Socket.IO.

Example:
    from nexus.extensions.agents.codebuddy import CodeBuddyInput

    agent = agent_registry.get('codebuddy')
    result = await agent(CodeBuddyInput(
        user_prompt='Create a hello world Python script',
        working_dir='/workspace',
    ))

    print(f'Success: {result.success}')
    print(f'Output: {result.output}')
"""
