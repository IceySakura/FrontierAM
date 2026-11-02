# ExecutionContext 执行上下文

ExecutionContext 是 Nexus 框架的依赖注入核心，为 Task、Agent、Tool 提供统一的运行时依赖访问接口。

## 目录

1. [概述](#概述)
2. [核心功能](#核心功能)
3. [生命周期](#生命周期)
4. [API 详解](#api-详解)
5. [使用模式](#使用模式)
6. [协程安全](#协程安全)

## 概述

ExecutionContext 解决了 AI Agent 系统中的依赖传递问题：

- **问题**：Task → Agent → Tool 调用链中需要传递 Runtime、LLM Client 等依赖
- **方案**：通过 Context Variable 实现隐式传递，任何层级都可以通过 `require_context()` 获取

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ExecutionContext 作用域                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  async with ExecutionContext(...) as ctx:                               │
│      │                                                                  │
│      │  ┌────────────────────────────────────────────────────────────┐ │
│      │  │ Task                                                       │ │
│      │  │   ctx = require_context()  ✓ 可访问                        │ │
│      │  │   │                                                        │ │
│      │  │   │  ┌─────────────────────────────────────────────────┐  │ │
│      │  │   │  │ Agent                                           │  │ │
│      │  │   │  │   ctx = require_context()  ✓ 可访问              │  │ │
│      │  │   │  │   │                                             │  │ │
│      │  │   │  │   │  ┌──────────────────────────────────────┐  │  │ │
│      │  │   │  │   │  │ Tool                                 │  │  │ │
│      │  │   │  │   │  │   使用参数传入的 ToolContext          │  │  │ │
│      │  │   │  │   │  │   ✗ 禁止调用 require_context()       │  │  │ │
│      │  │   │  │   │  └──────────────────────────────────────┘  │  │ │
│      │  │   │  └─────────────────────────────────────────────────┘  │ │
│      │  └────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  # 退出 async with 后                                                   │
│  require_context()  ✗ RuntimeError: No active ExecutionContext         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 核心功能

### 1. Runtime 管理

懒加载方式获取隔离的执行环境：

```python
ctx = require_context()

# 获取默认 'main' Runtime
runtime = await ctx.get_runtime()

# 获取指定名称的 Runtime
eval_runtime = await ctx.get_runtime('eval', memory='4G')

# Runtime 只在首次调用时创建，后续返回缓存实例
```

### 2. LLM Client 访问

获取预配置的 OpenAI 兼容客户端：

```python
ctx = require_context()

# 获取默认 'main' 客户端
client = ctx.openai_client  # 便捷属性

# 或通过名称获取
client = ctx.get_openai_client('main')
grm_client = ctx.get_openai_client('grm')
```

### 3. 共享状态

工具间可以共享状态。**注意**：Tool 不能调用 `require_context()`，需要通过 `ToolResponse.context_updates` 返回状态更新：

```python
# 在 Tool A 中 - 通过 context_updates 返回更新
async def tool_a(params: ToolAInput, context: ToolContext) -> ToolResponse:
    return ToolResponse(
        output=[{'type': 'text', 'text': 'done'}],
        context_updates={'file_list': ['a.py', 'b.py']},  # 返回状态更新
    )

# 在 Tool B 中 - 从 ToolContext 参数读取状态
async def tool_b(params: ToolBInput, context: ToolContext) -> ToolResponse:
    files = context.tool_state.get('file_list', [])  # 从参数读取
    # ...
```

### 4. Task 元数据

访问当前任务的信息：

```python
ctx = require_context()
print(ctx.task_name)    # 'echo'
print(ctx.task_config)  # {'message': 'Hello'}
```

## 生命周期

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ExecutionContext 生命周期                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. 创建阶段                                                             │
│     ┌────────────────────────────────────────────────────────────┐     │
│     │ ctx = ExecutionContext(                                    │     │
│     │     runtime_provider=provider,                             │     │
│     │     openai_clients={'main': client},                       │     │
│     │     task_name='echo',                                      │     │
│     │     task_config={'message': 'Hello'},                      │     │
│     │ )                                                          │     │
│     └────────────────────────────────────────────────────────────┘     │
│                           │                                             │
│                           ▼                                             │
│  2. 激活阶段 (async with __aenter__)                                     │
│     ┌────────────────────────────────────────────────────────────┐     │
│     │ • 设置 _current_context ContextVar                          │     │
│     │ • 保存 token 用于恢复                                        │     │
│     │ • 此时 require_context() 可用                                │     │
│     └────────────────────────────────────────────────────────────┘     │
│                           │                                             │
│                           ▼                                             │
│  3. 执行阶段                                                             │
│     ┌────────────────────────────────────────────────────────────┐     │
│     │ • Task/Agent 通过 require_context() 获取                     │     │
│     │ • Tool 使用参数传入的 ToolContext（禁止 require_context）    │     │
│     │ • get_runtime() 懒加载 Runtime 实例                          │     │
│     │ • _runtimes 缓存已创建的 Runtime                             │     │
│     │ • tool_state 在工具间共享                                    │     │
│     └────────────────────────────────────────────────────────────┘     │
│                           │                                             │
│                           ▼                                             │
│  4. 清理阶段 (async with __aexit__)                                      │
│     ┌────────────────────────────────────────────────────────────┐     │
│     │ • runtime_provider.release_all() 释放所有 Runtime            │     │
│     │ • _runtimes.clear() 清空缓存                                 │     │
│     │ • _current_context.reset(token) 恢复之前的上下文              │     │
│     │ • require_context() 不再可用（除非有外层上下文）               │     │
│     └────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## API 详解

### ExecutionContext 属性

```python
class ExecutionContext(BaseModel):
    # Runtime 管理
    runtime_provider: RuntimeProvider | None = None

    # OpenAI 客户端
    openai_clients: dict[str, OpenAICompatibleClient] = {}

    # Task 元数据
    task_name: str | None = None
    task_config: dict[str, Any] = {}

    # Agent 元数据
    agent_name: str | None = None

    # 工具共享状态
    tool_state: dict[str, Any] = {}

    # 追踪 ID
    trace_id: str | None = None

    # 私有：Runtime 缓存
    _runtimes: dict[str, Runtime]  # PrivateAttr
```

### get_runtime()

```python
async def get_runtime(self, name: str = 'main', **params: Any) -> Runtime:
    """获取 Runtime 实例，懒加载方式

    Args:
        name: Runtime 名称，默认 'main'
        **params: 首次创建时的额外参数（后续调用忽略）

    Returns:
        Runtime 实例

    Raises:
        RuntimeError: 未配置 runtime_provider
    """
```

**特性**：
- 首次调用时通过 `runtime_provider.acquire()` 创建
- 后续调用返回缓存实例，忽略新参数
- 支持预配置 + 运行时参数合并

```python
# 预配置
ctx.runtime_provider.configure('main', image='python:3.11', cpu=2)

# 运行时获取，可传入额外参数
runtime = await ctx.get_runtime('main', memory='4G')
# 最终参数：{image: 'python:3.11', cpu: 2, memory: '4G'}
```

### get_openai_client()

```python
def get_openai_client(self, name: str = 'main') -> OpenAICompatibleClient:
    """获取 OpenAI 兼容客户端

    Args:
        name: 客户端名称，默认 'main'

    Returns:
        OpenAICompatibleClient 实例

    Raises:
        KeyError: 客户端名称不存在
    """
```

### require_context()

```python
def require_context() -> ExecutionContext:
    """获取当前活动的 ExecutionContext

    Returns:
        当前 ExecutionContext

    Raises:
        RuntimeError: 没有活动的上下文
    """
```

## 使用模式

### 基础用法

```python
from nexus.core import ExecutionContext, require_context

# Runner 层：创建并激活上下文
async def run_task():
    ctx = ExecutionContext(
        runtime_provider=docker_provider,
        openai_clients={'main': openai_client},
        task_name='my_task',
        task_config={'key': 'value'},
    )

    async with ctx:
        await task_fn(task_input)
    # 退出时自动清理 Runtime
```

```python
# Task/Agent/Tool 层：使用上下文
@task_registry.register('my_task')
async def my_task(input: MyTaskInput) -> MyTaskOutput:
    ctx = require_context()

    # 访问 Task 信息
    print(f'Running task: {ctx.task_name}')

    # 获取 Runtime
    runtime = await ctx.get_runtime()

    # 获取 LLM Client
    client = ctx.openai_client

    # 使用共享状态
    ctx.tool_state['initialized'] = True
```

### 多 Runtime 场景

```python
@task_registry.register('eval_task')
async def eval_task(input: EvalTaskInput) -> EvalTaskOutput:
    ctx = require_context()

    # Agent 使用 'main' Runtime
    ctx.runtime_provider.configure('main', image='python:3.11')
    main_runtime = await ctx.get_runtime('main')

    # 评估使用独立的 'eval' Runtime
    ctx.runtime_provider.configure('eval', image='eval-sandbox')
    eval_runtime = await ctx.get_runtime('eval')

    # 两个 Runtime 完全隔离
    await main_runtime.call_function(agent_work)
    score = await eval_runtime.call_function(evaluate)
```

### 工具状态共享

**重要**：Tool 函数禁止调用 `require_context()`，必须使用参数传入的 `ToolContext`。状态更新通过 `ToolResponse.context_updates` 返回：

```python
# Tool: 读取文件列表
@tool_registry.register('list_files')
async def list_files(params: ListFilesInput, context: ToolContext) -> ToolResponse:
    # 注意：Tool 内部不能调用 require_context()
    # 如需 Runtime，应通过 ToolExecutor 在调用前获取并传入
    files = os.listdir(params.path)

    return ToolResponse(
        output=[{'type': 'text', 'text': '\n'.join(files)}],
        context_updates={'current_files': files},  # 通过返回值更新状态
    )

# Tool: 使用文件列表
@tool_registry.register('process_files')
async def process_files(params: ProcessInput, context: ToolContext) -> ToolResponse:
    files = context.tool_state.get('current_files', [])  # 从参数读取共享状态

    # 处理文件...
```

## 协程安全

ExecutionContext 使用 Python 的 `contextvars` 实现协程安全的上下文传递：

```python
from contextvars import ContextVar

# 全局 ContextVar
_current_context: ContextVar[ExecutionContext | None] = ContextVar('execution_context', default=None)
```

### 并发安全性

```python
async def concurrent_example():
    ctx1 = ExecutionContext(task_name='task1')
    ctx2 = ExecutionContext(task_name='task2')

    async def work_in_ctx1():
        async with ctx1:
            ctx = require_context()
            assert ctx.task_name == 'task1'  # 正确
            await asyncio.sleep(1)
            ctx = require_context()
            assert ctx.task_name == 'task1'  # 仍然正确

    async def work_in_ctx2():
        async with ctx2:
            ctx = require_context()
            assert ctx.task_name == 'task2'  # 正确，不受 ctx1 影响

    # 并发执行，互不干扰
    await asyncio.gather(work_in_ctx1(), work_in_ctx2())
```

### 上下文嵌套

```python
async def nested_example():
    outer_ctx = ExecutionContext(task_name='outer')
    inner_ctx = ExecutionContext(task_name='inner')

    async with outer_ctx:
        ctx = require_context()
        assert ctx.task_name == 'outer'

        async with inner_ctx:
            ctx = require_context()
            assert ctx.task_name == 'inner'  # 内层覆盖外层

        ctx = require_context()
        assert ctx.task_name == 'outer'  # 恢复外层
```

## 最佳实践

1. **始终使用 async with**：确保 Runtime 正确清理
2. **懒加载 Runtime**：不要在 Task 开始时就获取所有 Runtime
3. **预配置参数**：使用 `configure()` 设置默认参数，运行时只传必要的覆盖参数
4. **避免直接传递 ctx**：使用 `require_context()` 获取，保持代码解耦
5. **合理使用 tool_state**：只共享必要的状态，避免过度耦合

## 相关文档

- [Runtime 系统详解](./06-runtime.md)
- [Tool 系统详解](./05-tool.md)
- [执行流程教程](../getting-started/02-execution-flow.md)
