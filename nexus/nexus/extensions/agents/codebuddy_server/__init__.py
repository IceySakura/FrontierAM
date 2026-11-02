"""CodeBuddy Agent Server integration.

Runs the ``@tencent-ai/agent-server`` HTTP server inside a runtime sandbox.
The agent-server exposes REST endpoints for code generation and SWE-Bench
style tasks.  LLM calls are routed through an OpenAI-compatible proxy back
to the host-side Nexus OpenAI client via Socket.IO.

Two modes are available through the ``/api/v1/swe/completions`` (SWE-Bench)
and ``/api/v1/craft/completions`` (general code generation) endpoints.

Example:
    from nexus.extensions.agents.codebuddy_server import CodeBuddyServerInput

    agent = agent_registry.get('codebuddy_server')
    result = await agent(CodeBuddyServerInput(
        user_prompt='Fix the bug in utils/math.py where square(n) returns n*2',
        working_dir='/workspace',
        mode='swe',
    ))

    print(f'Success: {result.success}')
    print(f'Output: {result.output}')
"""
