"""Main Agent — 通用的 tool-calling agent 循环。

通过 ``tools`` 参数配置任意工具组合，支持顺序/并行工具执行、
多种退出条件。

Usage:
    from nexus.core import Registry
    Registry.import_module('nexus.extensions.agents.main')

    agent_fn = agent_registry.get('main_agent')

Example:
    from nexus.extensions.agents.main import MainAgentInput, MainAgentOutput

    input = MainAgentInput(
        user_prompt='Fix the bug in function foo...',
        tools=['swe_agent/bash', 'swe_agent/str_replace_editor'],
        max_iterations=30,
    )

    # Run with ExecutionContext
    output = await run_main_agent(input)
    print(output.final_answer)
"""
