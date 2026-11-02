# Agent 系统

Agent 是 Nexus 框架中的智能执行单元，负责与 LLM 交互并通过工具调用完成任务。

## 目录

1. [概述](#概述)
2. [Agent 结构](#agent-结构)
3. [注册与使用](#注册与使用)
4. [输入输出 Mixin](#输入输出-mixin)
5. [实现模式](#实现模式)
6. [与 ToolExecutor 集成](#与-toolexecutor-集成)

## 概述

Agent 在 Nexus 架构中的位置：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Agent 在架构中的位置                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Task                                                                   │
│    │                                                                    │
│    │ run_agent()                                                        │
│    ▼                                                                    │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                           Agent                                    │ │
│  │                                                                    │ │
│  │   ┌─────────────────────────────────────────────────────────────┐ │ │
│  │   │                    执行循环                                   │ │ │
│  │   │                                                             │ │ │
│  │   │   for iteration in range(max_iterations):                   │ │ │
│  │   │       │                                                     │ │ │
│  │   │       │  1. 调用 LLM                                         │ │ │
│  │   │       ▼                                                     │ │ │
│  │   │   ┌──────────────────┐                                      │ │ │
│  │   │   │  OpenAI Client   │◄──── ctx.openai_client               │ │ │
│  │   │   │  create_completion│                                      │ │ │
│  │   │   └────────┬─────────┘                                      │ │ │
│  │   │            │                                                 │ │ │
│  │   │            ▼                                                 │ │ │
│  │   │       2. 处理响应                                            │ │ │
│  │   │            │                                                 │ │ │
│  │   │            ├──► 有 tool_calls?                               │ │ │
│  │   │            │         │                                       │ │ │
│  │   │            │         ▼                                       │ │ │
│  │   │            │   ┌───────────────┐                            │ │ │
│  │   │            │   │ ToolExecutor  │                            │ │ │
│  │   │            │   │   .call()     │                            │ │ │
│  │   │            │   └───────┬───────┘                            │ │ │
│  │   │            │           │                                     │ │ │
│  │   │            │           ├──► LOCAL Tool                       │ │ │
│  │   │            │           │    (进程内执行)                      │ │ │
│  │   │            │           │                                     │ │ │
│  │   │            │           └──► RUNTIME Tool                     │ │ │
│  │   │            │                (远程沙箱执行)                    │ │ │
│  │   │            │                                                 │ │ │
│  │   │            └──► 无 tool_calls → 退出循环                      │ │ │
│  │   │                                                             │ │ │
│  │   └─────────────────────────────────────────────────────────────┘ │ │
│  │                                                                    │ │
│  │   返回 AgentRunOutput (包含 trajectory)                           │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Agent 的职责**：
- 管理与 LLM 的对话
- 解析和执行工具调用
- 维护对话历史（trajectory）
- 控制执行循环（迭代次数限制）

## Agent 结构

### 基础类型

```python
# 输入基类
class AgentRunInput(BaseModel):
    """Agent 输入基类

    只包含业务参数，运行时依赖通过 ExecutionContext 获取
    """
    model_config = {'arbitrary_types_allowed': True}

# 输出基类
class AgentRunOutput(BaseModel):
    """Agent 输出基类

    包含执行结果
    """
    model_config = {'arbitrary_types_allowed': True}

# Agent 函数协议
class AgentFunc(Protocol):
    async def __call__(self, agent_input: AgentRunInput) -> AgentRunOutput: ...
```

### 典型 Agent 结构

```python
# 1. 定义输入
class MyAgentInput(AgentRunInput, WithUserPrompt):
    max_iterations: int = 10
    temperature: float = 0.7

# 2. 定义输出
class MyAgentOutput(AgentRunOutput, WithMainTrajectory):
    final_answer: str

# 3. 实现 Agent 函数
@agent_registry.register('my_agent')
async def my_agent(agent_input: MyAgentInput) -> MyAgentOutput:
    ctx = require_context()
    client = ctx.openai_client

    executor = ToolExecutor([...])
    history = [{'role': 'user', 'content': agent_input.user_prompt}]

    for _ in range(agent_input.max_iterations):
        response = await client.create_completion(messages=history, tools=executor.get_tool_schemas())
        # ... 处理响应和工具调用

    return MyAgentOutput(
        main_trajectory=history,
        final_answer=history[-1].get('content', '')
    )
```

## 注册与使用

### 注册 Agent

使用 `@agent_registry.register()` 装饰器：

```python
from nexus.core import agent_registry, AgentRunInput, AgentRunOutput

@agent_registry.register('demo_agent')
async def demo_agent(agent_input: DemoAgentInput) -> DemoAgentOutput:
    """Demo Agent 实现"""
    ...
```

**注册时会自动验证**：
- 必须是 async 函数
- 必须恰好有一个参数
- 参数类型必须是 AgentRunInput 子类
- 返回类型必须是 AgentRunOutput 子类

### 配置运行

在 YAML 配置文件中引用：

```yaml
agent_name: "demo_agent"
agent_config:
  max_iterations: 10
  temperature: 0.7
```

### 获取 Agent

```python
from nexus.core import agent_registry

# 获取 Agent 函数
agent_fn = agent_registry.get('demo_agent')

# 查看 Agent 元数据
print(agent_fn.nexus_meta)
# {'name': 'demo_agent', 'type': 'agent', 'meta': {}}
```

## 输入输出 Mixin

### 输入 Mixin

#### WithUserPrompt

提供用户提示字段：

```python
class WithUserPrompt(BaseModel):
    """接收用户提示的 Agent"""
    user_prompt: str | Iterable[ChatCompletionContentPartParam]
```

**使用示例**：

```python
class MyAgentInput(AgentRunInput, WithUserPrompt):
    max_iterations: int = 10

# user_prompt 可以是字符串
input = MyAgentInput(user_prompt='Hello')

# 也可以是多模态内容
input = MyAgentInput(user_prompt=[
    {'type': 'text', 'text': 'What is in this image?'},
    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,...'}}
])
```

#### WithInputProvider

提供交互式输入接口：

```python
class WithInputProvider(BaseModel):
    """支持交互式输入的 Agent"""
    get_user_input: Callable[[str | None], Awaitable[str]]
```

**使用示例**：

```python
class InteractiveAgentInput(AgentRunInput, WithInputProvider):
    pass

@agent_registry.register('interactive_agent')
async def interactive_agent(agent_input: InteractiveAgentInput) -> InteractiveAgentOutput:
    # 在执行过程中请求用户输入
    user_response = await agent_input.get_user_input('Please confirm (yes/no):')
```

### 输出 Mixin

#### WithMainTrajectory

提供对话轨迹：

```python
class WithMainTrajectory(BaseModel):
    """带对话轨迹的 Agent 输出"""
    main_trajectory: list[ChatCompletionMessageParam | dict]
```

**使用示例**：

```python
class MyAgentOutput(AgentRunOutput, WithMainTrajectory):
    pass

# 返回完整对话历史
return MyAgentOutput(main_trajectory=[
    {'role': 'system', 'content': '...'},
    {'role': 'user', 'content': '...'},
    {'role': 'assistant', 'content': '...'},
    {'role': 'tool', 'tool_call_id': '...', 'content': '...'},
    {'role': 'assistant', 'content': 'Final answer'},
])
```

#### WithMainTmuxSessionId

提供 tmux 会话 ID：

```python
class WithMainTmuxSessionId(BaseModel):
    """在 tmux 会话中执行操作的 Agent"""
    main_tmux_session_id: str
```

## 实现模式

### 模式 1：基础对话 Agent

```python
@agent_registry.register('basic_agent')
async def basic_agent(agent_input: BasicAgentInput) -> BasicAgentOutput:
    """基础对话 Agent，无工具调用"""
    ctx = require_context()
    client = ctx.openai_client

    history = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': agent_input.user_prompt},
    ]

    response = await client.create_completion(messages=history)
    message = response.choices[0].message

    history.append(message.model_dump(exclude_none=True))

    return BasicAgentOutput(
        main_trajectory=history,
        response=message.content or ''
    )
```

### 模式 2：工具调用 Agent

```python
@agent_registry.register('tool_agent')
async def tool_agent(agent_input: ToolAgentInput) -> ToolAgentOutput:
    """带工具调用的 Agent"""
    ctx = require_context()
    client = ctx.openai_client

    # 创建工具执行器
    executor = ToolExecutor([
        tool_registry.get('think'),
        tool_registry.get('bash'),
        tool_registry.get('file_edit'),
    ])

    history = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': agent_input.user_prompt},
    ]

    tools = executor.get_tool_schemas()

    for _ in range(agent_input.max_iterations):
        # 调用 LLM
        response = await client.create_completion(
            messages=history,
            tools=tools or None
        )
        message = response.choices[0].message
        history.append(message.model_dump(exclude_none=True))

        # 无工具调用则结束
        if not message.tool_calls:
            break

        # 执行工具调用
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # ToolExecutor 自动路由到 LOCAL 或 RUNTIME
            tool_response = await executor.call(tool_name, tool_args)

            history.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'content': tool_response.output,
            })

    return ToolAgentOutput(main_trajectory=history)
```

### 模式 3：ReAct Agent

```python
@agent_registry.register('react_agent')
async def react_agent(agent_input: ReActAgentInput) -> ReActAgentOutput:
    """ReAct (Reasoning + Acting) Agent"""
    ctx = require_context()
    client = ctx.openai_client

    executor = ToolExecutor([
        tool_registry.get('think'),  # 用于推理
        tool_registry.get('search'),
        tool_registry.get('calculate'),
    ])

    system_prompt = """You are a ReAct agent. For each step:
1. Use the Think tool to reason about the problem
2. Use appropriate tools to take action
3. Observe the results
4. Repeat until you have the answer

Always think before acting."""

    history = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': agent_input.user_prompt},
    ]

    thoughts = []  # 收集推理过程

    for _ in range(agent_input.max_iterations):
        response = await client.create_completion(
            messages=history,
            tools=executor.get_tool_schemas()
        )
        message = response.choices[0].message
        history.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            break

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            tool_response = await executor.call(tool_name, tool_args)

            # 记录思考过程
            if tool_name == 'think':
                thoughts.append(tool_args.get('thought', ''))

            history.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'content': tool_response.output,
            })

    return ReActAgentOutput(
        main_trajectory=history,
        thoughts=thoughts,
    )
```

### 模式 4：多轮交互 Agent

```python
@agent_registry.register('interactive_agent')
async def interactive_agent(agent_input: InteractiveAgentInput) -> InteractiveAgentOutput:
    """支持与用户多轮交互的 Agent"""
    ctx = require_context()
    client = ctx.openai_client

    history = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': agent_input.user_prompt},
    ]

    for _ in range(agent_input.max_turns):
        response = await client.create_completion(messages=history)
        message = response.choices[0].message
        history.append(message.model_dump(exclude_none=True))

        # 检查是否需要用户输入
        if '[NEED_INPUT]' in (message.content or ''):
            user_input = await agent_input.get_user_input(
                'Agent needs your input:'
            )
            history.append({'role': 'user', 'content': user_input})
        elif '[DONE]' in (message.content or ''):
            break

    return InteractiveAgentOutput(main_trajectory=history)
```

## 与 ToolExecutor 集成

ToolExecutor 是 Agent 调用工具的统一接口：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ToolExecutor 工作流程                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Agent                                                                  │
│    │                                                                    │
│    │ executor = ToolExecutor([tool1, tool2, tool3])                     │
│    │                                                                    │
│    ▼                                                                    │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      ToolExecutor                                  │ │
│  │                                                                    │ │
│  │   get_tool_schemas()                                               │ │
│  │   ┌─────────────────────────────────────────────────────────────┐ │ │
│  │   │ 返回 OpenAI function calling 格式的工具定义                   │ │ │
│  │   │ [                                                           │ │ │
│  │   │   {'type': 'function', 'function': {'name': 'think', ...}}, │ │ │
│  │   │   {'type': 'function', 'function': {'name': 'bash', ...}},  │ │ │
│  │   │ ]                                                           │ │ │
│  │   └─────────────────────────────────────────────────────────────┘ │ │
│  │                                                                    │ │
│  │   call(tool_name, params)                                          │ │
│  │   ┌─────────────────────────────────────────────────────────────┐ │ │
│  │   │                                                             │ │ │
│  │   │   1. 查找工具                                                │ │ │
│  │   │   2. 验证参数 (Pydantic)                                     │ │ │
│  │   │   3. 构建 ToolContext                                        │ │ │
│  │   │   4. 根据 ToolType 路由:                                     │ │ │
│  │   │                                                             │ │ │
│  │   │      LOCAL ──► 直接调用 tool_func(params, context)           │ │ │
│  │   │                                                             │ │ │
│  │   │      RUNTIME ──► runtime.call_tool(ToolCallRequest)          │ │ │
│  │   │                  ──► HTTP POST /api/v1/call_tool             │ │ │
│  │   │                                                             │ │ │
│  │   │   5. 返回 ToolResponse                                       │ │ │
│  │   │   6. 应用 context_updates 到 ExecutionContext                │ │ │
│  │   │                                                             │ │ │
│  │   └─────────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 基本用法

```python
from nexus.core import ToolExecutor, tool_registry

# 创建执行器
executor = ToolExecutor([
    tool_registry.get('think'),
    tool_registry.get('bash'),
])

# 获取工具 Schema（传给 LLM）
tools = executor.get_tool_schemas()

# 调用工具
response = await executor.call('bash', {'command': 'ls -la'})
print(response.output)  # [{'type': 'text', 'text': '...'}]
print(response.meta)    # {'exit_code': 0, 'stdout': '...'}
```

### 上下文传递

ToolExecutor 自动从 ExecutionContext 构建 ToolContext：

```python
def _build_tool_context(self) -> ToolContext:
    ctx = require_context()
    return ToolContext(
        task_config=ctx.task_config.copy(),  # 任务参数（只读）
        tool_state=ctx.tool_state.copy(),    # 工具共享状态
    )
```

工具返回的 `context_updates` 会自动合并回 ExecutionContext：

```python
def _apply_context_updates(self, updates: dict[str, Any]) -> None:
    if updates:
        ctx = require_context()
        ctx.tool_state.update(updates)
```

## 完整示例

```python
"""完整的 Agent 实现示例"""

import json
import logging

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    Registry,
    ToolExecutor,
    WithMainTrajectory,
    WithUserPrompt,
    agent_registry,
    require_context,
    tool_registry,
)


logger = logging.getLogger(__name__)

# 确保工具已注册
Registry.import_module('nexus.extensions.agents.demo.tools')


class CodeAssistantInput(AgentRunInput, WithUserPrompt):
    """代码助手输入"""
    max_iterations: int = 15
    working_directory: str = '/workspace'


class CodeAssistantOutput(AgentRunOutput, WithMainTrajectory):
    """代码助手输出"""
    files_modified: list[str] = []


@agent_registry.register('code_assistant')
async def code_assistant(agent_input: CodeAssistantInput) -> CodeAssistantOutput:
    """代码助手 Agent

    能够：
    - 思考问题（think 工具）
    - 执行命令（bash 工具）
    - 读写文件（通过 bash）
    """
    ctx = require_context()
    client = ctx.openai_client

    # 创建工具执行器
    executor = ToolExecutor([
        tool_registry.get('demo/think'),
        tool_registry.get('demo/bash'),
    ])

    system_prompt = f"""You are a code assistant working in {agent_input.working_directory}.

Available tools:
- Think: Use this to reason through problems step by step
- Bash: Execute shell commands to read/write files, run tests, etc.

Guidelines:
1. Always think through the problem before taking action
2. Use bash to explore the codebase first
3. Make small, incremental changes
4. Test your changes when possible"""

    history = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': agent_input.user_prompt},
    ]

    tools = executor.get_tool_schemas()
    files_modified = []

    for iteration in range(agent_input.max_iterations):
        logger.info(f'Iteration {iteration + 1}/{agent_input.max_iterations}')

        # 调用 LLM
        response = await client.create_completion(
            messages=history,
            tools=tools or None
        )
        message = response.choices[0].message
        history.append(message.model_dump(exclude_none=True))

        # 记录输出
        if message.content:
            logger.info(f'Assistant: {message.content[:200]}...')

        # 无工具调用则结束
        if not message.tool_calls:
            logger.info('No tool calls, finishing')
            break

        # 执行工具调用
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            logger.info(f'Tool: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:100]}...)')

            tool_response = await executor.call(tool_name, tool_args)

            # 追踪文件修改
            if tool_name == 'Bash':
                cmd = tool_args.get('command', '')
                if any(op in cmd for op in ['>', '>>', 'tee', 'sed -i', 'echo']):
                    # 简单的文件修改检测
                    files_modified.append(cmd)

            history.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'content': tool_response.output,
            })

    return CodeAssistantOutput(
        main_trajectory=history,
        files_modified=files_modified,
    )
```

## 相关文档

- [Tool 系统详解](./05-tool.md)
- [Task 系统详解](./03-task.md)
- [ExecutionContext 详解](./02-execution-context.md)
