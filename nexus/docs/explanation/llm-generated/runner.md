# Nexus 执行流程教程

本教程将详细解释 Nexus 框架的执行流程，帮助你理解从配置文件到任务执行的完整链路。

## 目录

1. [概述](#概述)
2. [执行流程图](#执行流程图)
3. [详细步骤解析](#详细步骤解析)
4. [代码示例](#代码示例)
5. [数据流向](#数据流向)

## 概述

Nexus 的执行流程遵循以下核心链路：

```
配置文件 (YAML) → Runner → ExecutionContext → Task → Agent → Tools
```

整个流程由 **单任务运行器 (Single Runner)** 协调，通过 **ExecutionContext** 提供依赖注入，支持 **本地 (LOCAL)** 和 **远程隔离 (RUNTIME)** 两种工具执行模式。

## 执行流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              配置文件 (YAML)                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   imports   │  │  task_name  │  │ agent_name  │  │ runtime_provider    │ │
│  │   模块导入   │  │  task_config│  │ agent_config│  │ openai_client_*     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Runner (nexus/runner/single.py)                   │
│                                                                             │
│  1. 导入模块 (Registry.import_module)                                        │
│  2. 创建 RuntimeProvider                                                    │
│  3. 创建 OpenAI Client                                                      │
│  4. 构建 run_agent 工厂函数                                                  │
│  5. 构建 TaskInput                                                          │
│  6. 创建 SpanProcessors                                                     │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ExecutionContext (async context manager)                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  • runtime_provider: RuntimeProvider | None                          │   │
│  │  • openai_clients: dict[str, OpenAICompatibleClient]                │   │
│  │  • task_name, task_config: 任务元数据                                 │   │
│  │  • tool_state: dict[str, Any]  # 工具间共享状态                       │   │
│  │  • _runtimes: dict[str, Runtime]  # 懒加载的运行时实例                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  async with ctx:                                                            │
│      with processor_context([...]):                                         │
│          task_output = await task_fn(task_input)                           │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Task 执行                                       │
│                                                                             │
│  @task_registry.register('echo')                                            │
│  async def echo_task(task_input: EchoTaskInput) -> EchoTaskOutput:         │
│      ctx = require_context()                                                │
│      runtime = await ctx.get_runtime()  # 懒加载 Runtime                    │
│      await task_input.run_agent(user_prompt='...')  # 调用 Agent            │
│      result = await runtime.call_function(...)  # 远程执行函数              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Agent 执行循环                                   │
│                                                                             │
│  @agent_registry.register('demo_agent')                                     │
│  async def run_demo_agent(agent_input: DemoAgentInput) -> DemoAgentOutput:  │
│      ctx = require_context()                                                │
│      client = ctx.openai_client                                             │
│      executor = ToolExecutor([...])                                         │
│                                                                             │
│      for iteration in range(max_iterations):                                │
│          response = await client.create_completion(messages, tools)         │
│          if tool_calls:                                                     │
│              for tool_call in tool_calls:                                   │
│                  result = await executor.call(tool_name, params)            │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
┌───────────────────────────────┐   ┌───────────────────────────────────────┐
│      LOCAL Tool 执行           │   │          RUNTIME Tool 执行             │
│                               │   │                                       │
│  直接在 Agent 进程内执行        │   │  1. 构建 ToolCallRequest               │
│                               │   │  2. 获取 Runtime (HTTP Client)         │
│  await tool_func(params, ctx) │   │  3. POST /api/v1/call_tool             │
│                               │   │  4. Runtime Server 执行工具            │
│  示例: think, 内存计算等        │   │  5. 返回 ToolResponse                  │
│                               │   │                                       │
│                               │   │  示例: bash, 文件操作等                 │
└───────────────────────────────┘   └───────────────────────────────────────┘
```

## 详细步骤解析

### 步骤 1: 配置加载与模块导入

Runner 首先加载 YAML 配置文件，然后导入指定的模块：

```python
# nexus/runner/single.py
for module_path in run_input.imports:
    Registry.import_module(module_path)  # 例如: 'nexus.extensions'
```

模块导入会触发各种 `@xxx_registry.register()` 装饰器，将 Task、Agent、Tool 注册到全局注册表中。

### 步骤 2: RuntimeProvider 初始化

如果配置了 runtime_provider，Runner 会创建对应的 Provider 实例：

```python
if run_input.runtime_provider_name:
    provider_cls = runtime_provider_registry.get(run_input.runtime_provider_name)
    runtime_provider = provider_cls(**run_input.runtime_provider_config)

    # 预配置各个 Runtime 的参数
    for runtime_name, runtime_config in run_input.runtime_configs.items():
        runtime_provider.configure(runtime_name, **runtime_config)
```

### 步骤 3: OpenAI Client 创建

为 Agent 创建 LLM 客户端：

```python
client_cls = openai_client_registry.get(run_input.openai_client_name)
openai_config = OpenAIConfig(
    client_args=run_input.openai_client_config.get('client_args', {}),
    request_args=run_input.openai_client_config.get('request_args', {}),
)
openai_clients['main'] = client_cls(openai_config)
```

### 步骤 4: run_agent 工厂函数

Runner 创建一个闭包函数 `run_agent`，供 Task 调用 Agent：

```python
async def run_agent(**task_kwargs: Any) -> AgentRunOutput:
    merged_kwargs = {**run_input.agent_config, **task_kwargs}
    agent_input = _build_input(agent_input_type, merged_kwargs)
    return await agent_fn(agent_input)
```

### 步骤 5: ExecutionContext 激活

通过 `async with` 激活 ExecutionContext：

```python
ctx = ExecutionContext(
    runtime_provider=runtime_provider,
    openai_clients=openai_clients,
    task_name=run_input.task_name,
    task_config=run_input.task_config,
)

async with ctx:  # 设置为当前上下文，通过 contextvars 实现协程安全
    with processor_context([...]):  # 注册 Span 处理器
        task_output = await task_fn(task_input)
# 退出时自动释放所有 Runtime
```

### 步骤 6: Task 执行

Task 通过 `require_context()` 获取当前上下文：

```python
@task_registry.register('echo')
async def echo_task(task_input: EchoTaskInput) -> EchoTaskOutput:
    ctx = require_context()  # 获取当前 ExecutionContext

    # 配置并获取 Runtime（懒加载）
    ctx.runtime_provider.configure('main', image='node')
    runtime = await ctx.get_runtime()

    # 调用 Agent
    await task_input.run_agent(user_prompt='...')

    # 在 Runtime 中执行函数
    result = await runtime.call_function(lambda: open('/tmp/echo.txt').read())
```

### 步骤 7: Agent 执行循环

Agent 通过 ToolExecutor 统一调用工具：

```python
@agent_registry.register('demo_agent')
async def run_demo_agent(agent_input: DemoAgentInput) -> DemoAgentOutput:
    ctx = require_context()
    client = ctx.openai_client

    # 创建工具执行器
    executor = ToolExecutor([
        tool_registry.get('demo/think'),  # LOCAL 工具
        tool_registry.get('demo/bash'),   # RUNTIME 工具
    ])

    # 获取工具 Schema 供 LLM 使用
    tools = executor.get_tool_schemas()

    for _ in range(max_iterations):
        response = await client.create_completion(messages=history, tools=tools)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                # ToolExecutor 自动路由到 LOCAL 或 RUNTIME
                result = await executor.call(tool_name, tool_args)
```

### 步骤 8: Tool 执行

ToolExecutor 根据工具类型选择执行路径：

```python
# LOCAL Tool: 直接调用
if meta.tool_type == ToolType.LOCAL:
    response = await tool_func(params, context)

# RUNTIME Tool: 通过 HTTP 远程执行
else:
    runtime = await ctx.get_runtime()
    request = ToolCallRequest(
        import_path=entry.import_path,  # 工具的导入路径
        params=params.model_dump(),
        context=tool_context,
    )
    response = await runtime.call_tool(request)
```

**重要**：
- Tool 函数禁止调用 `require_context()`，必须使用参数传入的 `ToolContext`
- `ToolContext` 是不可变的，不能直接修改 `context.tool_state`
- 状态更新必须通过 `ToolResponse.context_updates` 返回

## 代码示例

### 完整配置文件示例

```yaml
# workspace/example.yaml
imports:
  - nexus.extensions

task_name: "echo"
task_config:
  message: "Hello, Nexus!"
  run_oracle: false

runtime_provider_name: "docker"
runtime_provider_config: {}
runtime_configs:
  main:
    image: "node"

agent_name: "demo_agent"
agent_config:
  max_iterations: 10

openai_clients:
  main:
    name: "openai"
    config:
      client_args:
        api_key: "<your-api-key>"
        base_url: "https://api.openai.com/v1"
      request_args:
        model: "gpt-4o"

processors:
  trace:
    name: "json_line"
    args:
      file_path: "workspace/trace.jsonl"
```

### 自定义 Task 示例

```python
from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    require_context,
    task_registry,
)

class MyTaskInput(TaskRunInput, WithAgentRunner):
    problem_statement: str

class MyTaskOutput(TaskRunOutput):
    solution: str
    success: bool

@task_registry.register('my_task')
async def my_task(task_input: MyTaskInput) -> MyTaskOutput:
    ctx = require_context()

    # 获取并配置 Runtime
    ctx.runtime_provider.configure('main', image='python:3.11')
    runtime = await ctx.get_runtime()

    # 调用 Agent 解决问题
    agent_output = await task_input.run_agent(
        user_prompt=task_input.problem_statement
    )

    # 验证结果
    result = await runtime.call_function(verify_solution)

    return MyTaskOutput(solution=result, success=True)
```

## 数据流向

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            数据流向概览                                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  RunInput (配置)                                                          │
│       │                                                                  │
│       ├──► imports ──► Registry.import_module() ──► 注册 Task/Agent/Tool │
│       │                                                                  │
│       ├──► task_config ──► TaskInput ──► task_fn()                       │
│       │                                                                  │
│       ├──► agent_config ──► run_agent() ──► AgentInput ──► agent_fn()    │
│       │                                                                  │
│       └──► openai_client_config ──► OpenAI Client ──► LLM API            │
│                                                                          │
│  ExecutionContext (运行时依赖)                                             │
│       │                                                                  │
│       ├──► runtime_provider ──► get_runtime() ──► Runtime (HTTP/Docker)  │
│       │                                                                  │
│       ├──► openai_clients ──► openai_client ──► LLM 调用                  │
│       │                                                                  │
│       └──► tool_state ──► 工具间共享状态                                   │
│                                                                          │
│  ToolExecutor (工具执行)                                                   │
│       │                                                                  │
│       ├──► LOCAL ──► 直接调用 tool_func()                                 │
│       │                                                                  │
│       └──► RUNTIME ──► HTTP POST ──► Runtime Server ──► tool_func()      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## 关键概念总结

| 概念 | 说明 | 生命周期 |
|------|------|----------|
| **ExecutionContext** | 统一的执行上下文，提供依赖注入 | 整个 Task 执行期间 |
| **RuntimeProvider** | 管理 Runtime 的生命周期 | Task 级别 |
| **Runtime** | 隔离的执行环境（如 Docker 容器） | 懒加载，Task 结束时释放 |
| **ToolExecutor** | 统一的工具调用接口 | Agent 执行期间 |
| **Registry** | 全局注册表，存储 Task/Agent/Tool | 进程级别 |

## 下一步

- [Registry 系统详解](../concepts/01-registry.md)
- [Tool 系统详解](../concepts/05-tool.md)
- [Runtime 系统详解](../concepts/06-runtime.md)
