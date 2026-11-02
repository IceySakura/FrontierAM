# Interact with Runtime

本节介绍如何与 Runtime 容器进行交互，执行各种操作如命令执行、文件管理、Jupyter 会话、终端会话和浏览器控制等。

## Runtime 基类

Runtime 是 Agent/Task 与执行环境（沙箱、容器等）交互的统一接口。 所有 Runtime 实现都继承自 `Runtime` 基类（`nexus/runtime/client/base.py`）。

### 核心方法

`Runtime` 基类定义了以下核心方法：

```python
class Runtime(ExecutorMixin, ABC):
    """Runtime 提供统一的接口与执行环境交互。

    Attributes:
        port_mappings: 内部端口到外部 URL 的映射
                      例如：{8023: 'http://localhost:32768', 8080: 'http://localhost:32769'}
        default_venv_id: 默认 Worker Venv ID（设置后所有 invoke 调用路由到该 Worker）
    """

    async def invoke(
        self,
        target: str | Callable,
        *args: Any,
        venv_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """在 Runtime 环境中调用指定函数。

        Args:
            target: import path 字符串（如 'my_module::my_func'）或函数对象
            venv_id: 指定在哪个 Worker Venv 中执行，None 表示在 Base 进程执行

        Returns:
            函数返回值
        """
        ...

    async def call_tool(
        self,
        import_path: str,
        params: dict,
        context: ToolContext,
    ) -> ToolResponse:
        """在 Runtime 中调用已注册的工具。"""
        ...

    async def call_executor(
        self,
        executor_name: str,
        arg: Any,
    ) -> Any:
        """在 Runtime 环境中调用注册的 Executor。"""
        ...

    async def create_venv(
        self,
        spec: dict | None = None,
        **kwargs,
    ) -> VenvHandle:
        """创建 Worker Venv。"""
        ...
```

### 四种交互方式的区别

`invoke`、`call_tool`、`call_executor` 和 ExecutorMixin 便捷方法提供了不同层次的抽象。

#### 1. `invoke`：统一远程调用原语（推荐）

**用途**：在 Runtime 环境中调用指定的 Python 函数，支持路由到 Worker Venv。

**特点**：

- 通过 import path 定位函数，参数用 pickle 序列化——不依赖 dill
- 支持 `venv_id` 参数路由到特定 Worker 进程
- 自动将 dict 参数强转为对应的 Pydantic BaseModel
- HTTP long-poll 减少延迟，大 payload 走 blob sideband

**示例**：

```python
# 通过 import path 调用
result = await runtime.invoke('my_module::my_func', arg1, arg2)

# 传入函数对象（自动提取 import path）
from my_module import my_func
result = await runtime.invoke(my_func, arg1, arg2)

# 在特定 Worker Venv 中执行
result = await runtime.invoke(my_func, arg1, venv_id='worker-1')

# 执行 Python 代码字符串
result = await runtime.exec('import os; os.listdir("/tmp")')
```

**实现位置**：`nexus/runtime/client/base.py`

#### 2. `call_tool`：面向 LLM 的工具调用

**用途**：调用注册在 `tool_registry` 中的工具，这些工具专为 LLM Function Calling 设计。

**特点**：

- 输入符合 OpenAI Function Call Schema（由 Pydantic 模型生成）
- 输出是字符串或 `ChatCompletionContentPartParam` 列表（适合直接返回给 LLM）
- 支持 `ToolContext` 进行状态管理（如工作目录、会话 ID 等）
- 工具可以标记为 `LOCAL` 或 `RUNTIME` 类型

**示例**：

```python
# 定义工具
@tool_registry.register(
    'swe_agent/bash',
    meta={
        'name': 'bash',
        'description': 'Execute a bash command',
        'tool_type': ToolType.RUNTIME,
    },
)
async def bash_tool(params: BashInput, context: ToolContext) -> ToolResponse:
    # 工具内部通常会调用 executor 来完成实际工作
    result = await run_command_in_session(
        RunCommandInSessionRequest(
            session_id=context.tool_state['terminal_session_id'],
            command=params.command,
            timeout=300,
        )
    )
    return ToolResponse(
        output=[{'type': 'text', 'text': result.output}],
        meta={'exit_code': result.exit_code},
    )

# 调用工具
response = await runtime.call_tool(
    ToolCallRequest(
        import_path='swe_agent/bash',
        params={'command': 'ls -la'},
        context=tool_context,
    )
)
```

**实现位置**：`nexus/extensions/agents/swe_agent/tools.py:76-163`

#### 3. `call_executor`：底层结构化接口

**用途**：调用注册在 `executor_registry` 中的执行器，提供结构化的输入输出接口。

**特点**：

- 输入和输出都是结构化的 Pydantic 模型（而非字符串）
- 更底层，是构建 Tool 的基础组件
- Tool 通常直接调用 Executor 完成实际工作

**调用关系**：`Tool` → `Executor`（Tool 是面向 LLM 的封装，Executor 是实际执行逻辑）

**示例**：

```python
# Executor 定义在 nexus/extensions/executors/terminal.py
async def run_command_in_session(request: RunCommandInSessionRequest) -> CommandInfo:
    """在终端会话中执行命令并等待完成。"""
    # ... 实现逻辑
    return CommandInfo(
        command_id=cmd_id,
        output=output,
        exit_code=exit_code,
        status='completed'
    )

# Tool 调用 Executor
@tool_registry.register('swe_agent/bash', ...)
async def bash_tool(params: BashInput, context: ToolContext) -> ToolResponse:
    # Tool 内部调用 Executor
    result = await run_command_in_session(
        RunCommandInSessionRequest(
            session_id=session_id,
            command=params.command,
            timeout=300,
        )
    )
    # 将结构化结果转换为面向 LLM 的字符串输出
    return ToolResponse(output=[{'type': 'text', 'text': result.output}])

# 也可以通过 runtime 直接调用 Executor
result = await runtime.call_executor(
    'run_command_in_terminal_session',
    {
        'session_id': 'my-session',
        'command': 'ls -la',
        'timeout': 300,
    }
)
# result 是 CommandInfo 对象（结构化数据）
```

**实现位置**：

- Executor 定义：`nexus/extensions/executors/terminal.py`
- Tool 调用示例：`nexus/extensions/agents/swe_agent/tools.py:126-132`

#### 总结对比

| 维度 | `invoke` | `call_tool` | `call_executor` |
|------|----------|-------------|-----------------|
| **抽象层次** | 统一原语 | 面向 LLM | 底层基础设施 |
| **输入** | import path + 任意参数 | Function Call Schema | Pydantic 模型 |
| **输出** | 任意 Python 对象 | 字符串（LLM 可读） | Pydantic 模型 |
| **Worker Venv** | 支持 `venv_id` 路由 | 支持 | 支持 |
| **典型用途** | 远程执行任意函数 | LLM Agent 工具 | 构建 Tool 的基础 |
| **限制** | 需要目标函数可 import | 需要注册到 registry | 需要注册到 registry |

### Port Mappings

`port_mappings` 属性用于暴露容器内部服务的外部访问地址。 例如在 WebArena 场景中，浏览器需要访问容器内的 Web 服务，可以通过此属性获取外部 URL：

```python
# 假设配置中指定了 exposed_ports: [8023, 8080]
runtime = await provider.acquire('main', image='...')
print(runtime.port_mappings)
# 输出：{8023: 'http://localhost:32768', 8080: 'http://localhost:32769'}
```

## ExecutorMixin：便捷的交互方法

`Runtime` 继承了 `ExecutorMixin`（`nexus/runtime/executor_mixin.py`），提供了一系列类型化的便捷方法。 这些方法内部调用 `call_executor()`，但提供了更友好的参数展开形式。

### 为什么使用 ExecutorMixin？

不使用 Mixin 时，你需要手动构造 Pydantic 请求模型：

```python
from nexus.extensions.executors.command import RunCommandRequest

# 繁琐的方式
request = RunCommandRequest(command='ls', timeout=10, cwd='/tmp')
response = await runtime.call_executor('run_command', request.model_dump())
```

使用 Mixin 后，可以直接使用展开的参数：

```python
# 简洁的方式
response = await runtime.run_command(command='ls', timeout=10, cwd='/tmp')
```

### Executor 分类

`Executor` 方法按功能分为以下几类：

1. **命令执行**（Shell Executors，[使用示例](shell-executor.ipynb)）：在容器中执行 Shell 命令
2. **文件操作**（File Executors，[使用示例](file-executor.ipynb)）：上传和下载文件
3. **Jupyter**（Jupyter Executors，[使用示例](jupyter-executor.ipynb)）：管理和执行 Jupyter 会话
4. **浏览器控制**（Browser Executors，[使用示例](browser-executor.ipynb)）：控制 Playwright 浏览器
