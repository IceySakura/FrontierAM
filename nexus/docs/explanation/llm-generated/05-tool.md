# Tool 工具系统

Tool 是 Nexus 框架中的原子操作单元，通过 Pydantic 模型定义输入，支持本地执行和远程沙箱执行两种模式。 111

## 目录

1. [概述](#概述)
2. [Tool 结构](#tool-结构)
3. [执行模式](#执行模式)
4. [注册与使用](#注册与使用)
5. [ToolExecutor](#toolexecutor)
6. [Schema 自动生成](#schema-自动生成)
7. [实现示例](#实现示例)

## 概述

Tool 在 Nexus 架构中的位置：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Tool 在架构中的位置                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Agent                                                                  │
│    │                                                                    │
│    │ ToolExecutor.call(tool_name, params)                               │
│    ▼                                                                    │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                         ToolExecutor                               │ │
│  │                                                                    │ │
│  │   根据 ToolType 路由:                                               │ │
│  │                                                                    │ │
│  │   ┌─────────────────────┐     ┌─────────────────────────────────┐ │ │
│  │   │    LOCAL Tool       │     │        RUNTIME Tool             │ │ │
│  │   │                     │     │                                 │ │ │
│  │   │  在 Agent 进程内执行  │     │  在远程沙箱中执行                │ │ │
│  │   │                     │     │                                 │ │ │
│  │   │  适用于:             │     │  适用于:                        │ │ │
│  │   │  • 思考/推理         │     │  • Shell 命令                   │ │ │
│  │   │  • 内存计算          │     │  • 文件操作                     │ │ │
│  │   │  • 快速操作          │     │  • 不信任代码执行                │ │ │
│  │   │                     │     │                                 │ │ │
│  │   │  直接调用:           │     │  HTTP 调用:                     │ │ │
│  │   │  tool_func(params,  │     │  POST /api/v1/call_tool         │ │ │
│  │   │            context) │     │  {import_path, params, context} │ │ │
│  │   └─────────────────────┘     └─────────────────────────────────┘ │ │
│  │                                                                    │ │
│  │   返回统一的 ToolResponse                                           │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Tool 的特点**：
- **类型安全**：使用 Pydantic 模型定义输入
- **Schema 自动生成**：自动生成 OpenAI function calling 格式
- **双模式执行**：LOCAL（进程内）或 RUNTIME（远程沙箱）
- **上下文传递**：通过 ToolContext 参数传递状态（只读）
- **不可变上下文**：ToolContext 是不可变的，状态更新通过 `ToolResponse.context_updates` 返回
- **隔离性**：Tool 内部禁止调用 `require_context()`，必须使用参数传入的 ToolContext

## Tool 结构

### 核心类型

```python
# 工具类型枚举
class ToolType(str, Enum):
    LOCAL = 'local'      # 在 Agent 进程内执行
    RUNTIME = 'runtime'  # 在远程沙箱中执行

# 工具上下文（不可变）
class ToolContext(BaseModel):
    task_config: ImmutableDict = {}   # Task 参数（只读）
    tool_state: ImmutableDict = {}    # 工具共享状态（只读，通过 context_updates 更新）

# 工具响应
class ToolResponse(BaseModel):
    output: list[ChatCompletionContentPartParam | dict]  # LLM 可见的输出
    meta: dict[str, Any] = {}                            # 内部元数据（不给 LLM）
    context_updates: dict[str, Any] = {}                 # 状态更新

# 工具元数据
class ToolMeta(BaseModel):
    name: str
    description: str
    tool_type: ToolType
    input_model: type[BaseModel]
    func: Callable[..., Awaitable[ToolResponse]]
```

### 工具函数签名

```python
async def tool_func(params: InputModel, context: ToolContext) -> ToolResponse:
    """工具函数标准签名

    Args:
        params: Pydantic 模型实例，包含工具输入参数
        context: 工具上下文，包含任务参数和共享状态（只读）

    Returns:
        ToolResponse 包含输出、元数据和状态更新

    注意:
        - Tool 内部禁止调用 require_context()，会抛出 RuntimeError
        - ToolContext 是不可变的，直接修改会抛出 TypeError
        - 状态更新必须通过 ToolResponse.context_updates 返回
    """
    ...
```

## 执行模式

### LOCAL 模式

直接在 Agent 进程内执行，适用于：
- 纯计算任务
- 推理/思考工具
- 不需要隔离的快速操作

```
Agent Process
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   Agent Loop                                                            │
│      │                                                                  │
│      │ await executor.call('think', {'thought': '...'})                 │
│      ▼                                                                  │
│   ToolExecutor._call_local()                                            │
│      │                                                                  │
│      │ await tool_func(params, context)                                 │
│      ▼                                                                  │
│   think_tool() ──► ToolResponse                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### RUNTIME 模式

通过 HTTP 在远程沙箱中执行，适用于：
- Shell 命令执行
- 文件系统操作
- 不信任代码执行
- 需要隔离的操作

```
Agent Process                              Runtime Container
┌───────────────────────────┐              ┌────────────────────────────┐
│                           │              │                            │
│ Agent Loop                │              │  FastAPI Server            │
│    │                      │              │                            │
│    │ executor.call(       │  HTTP POST   │  POST /api/v1/call_tool    │
│    │   'bash',            │ ──────────►  │    │                       │
│    │   {'command': 'ls'}) │              │    ▼                       │
│    ▼                      │              │  import_from_path()        │
│ ToolExecutor              │              │    │                       │
│    │                      │              │    ▼                       │
│    │ _call_runtime()      │              │  bash_tool(params, ctx)    │
│    │  │                   │              │    │                       │
│    │  │ runtime.call_tool │              │    ▼                       │
│    │  │ (request)         │  ◄────────── │  ToolResponse              │
│    │  │                   │  HTTP 200    │                            │
│    ▼  ▼                   │              │                            │
│ ToolResponse              │              │                            │
│                           │              │                            │
└───────────────────────────┘              └────────────────────────────┘
```

## 注册与使用

### 注册工具

使用 `@tool_registry.register()` 装饰器：

```python
from nexus.core import tool_registry, ToolContext, ToolResponse, ToolType
from pydantic import BaseModel, Field


class MyToolInput(BaseModel):
    """工具输入模型 - 自动生成 OpenAI Schema"""
    param1: str = Field(description='第一个参数')
    param2: int = Field(default=10, description='第二个参数')


@tool_registry.register(
    'my_tool',
    meta={
        'name': 'MyTool',           # OpenAI 显示名称
        'description': '工具描述',    # OpenAI 工具描述
        'tool_type': ToolType.LOCAL,  # 执行模式
    }
)
async def my_tool(params: MyToolInput, context: ToolContext) -> ToolResponse:
    """工具实现"""
    result = f'param1={params.param1}, param2={params.param2}'

    return ToolResponse(
        output=[{'type': 'text', 'text': result}],
        meta={'internal_data': 'xxx'},
        context_updates={'last_result': result},
    )
```

**注册时会自动验证**：
- 必须是 async 函数
- 必须至少有 2 个参数
- 第一个参数必须是 Pydantic BaseModel 子类
- 第二个参数必须是 ToolContext
- 返回类型必须是 ToolResponse

### 获取工具

```python
from nexus.core import tool_registry

# 获取工具函数
tool = tool_registry.get('my_tool')

# 获取工具元数据
meta = tool.__tool_meta__
print(meta.name)         # 'MyTool'
print(meta.description)  # '工具描述'
print(meta.tool_type)    # ToolType.LOCAL
print(meta.schema)       # OpenAI function schema
```

## ToolExecutor

ToolExecutor 是 Agent 调用工具的统一接口：

### 创建执行器

```python
from nexus.core import ToolExecutor, tool_registry

executor = ToolExecutor([
    tool_registry.get('think'),
    tool_registry.get('bash'),
    tool_registry.get('file_edit'),
])
```

### 获取 Schema

```python
# 获取所有工具的 OpenAI function calling schema
schemas = executor.get_tool_schemas()
# [
#   {
#     'type': 'function',
#     'function': {
#       'name': 'Think',
#       'description': 'Use this tool to think...',
#       'parameters': {
#         'type': 'object',
#         'properties': {
#           'thought': {'type': 'string', 'description': '...'}
#         },
#         'required': ['thought']
#       }
#     }
#   },
#   ...
# ]
```

### 调用工具

```python
# 调用工具（自动路由到 LOCAL 或 RUNTIME）
response = await executor.call('bash', {'command': 'ls -la'})

# 响应结构
print(response.output)           # LLM 可见的输出
print(response.meta)             # 内部元数据
print(response.context_updates)  # 状态更新（已自动应用到 ExecutionContext）
```

### 内部工作流程

```python
async def call(self, tool_name: str, params: dict[str, Any]) -> ToolResponse:
    # 1. 查找工具
    tool_func = self._tools[tool_name]
    meta = get_tool_meta(tool_func)

    # 2. 验证参数
    parsed_params = meta.input_model.model_validate(params)

    # 3. 构建上下文
    tool_context = self._build_tool_context()

    # 4. 根据类型路由
    if meta.tool_type == ToolType.LOCAL:
        response = await self._call_local(tool_func, parsed_params, tool_context)
    else:
        response = await self._call_runtime(tool_func, parsed_params, tool_context)

    # 5. 应用状态更新
    self._apply_context_updates(response.context_updates)

    return response
```

## Schema 自动生成

Nexus 自动从 Pydantic 模型生成 OpenAI function calling schema：

### 输入模型定义

```python
from pydantic import BaseModel, Field


class BashInput(BaseModel):
    """Bash 工具输入"""
    command: str = Field(description='The bash command to execute')
    timeout: int = Field(default=60, description='Timeout in seconds')
    working_dir: str | None = Field(default=None, description='Working directory')
```

### 自动生成的 Schema

```python
# tool.__tool_meta__.schema 返回:
{
    'type': 'function',
    'function': {
        'name': 'Bash',
        'description': 'Execute a bash command in the sandbox environment.',
        'parameters': {
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to execute'
                },
                'timeout': {
                    'type': 'integer',
                    'default': 60,
                    'description': 'Timeout in seconds'
                },
                'working_dir': {
                    'type': 'string',
                    'description': 'Working directory'
                }
            },
            'required': ['command']  # 无默认值的字段为必填
        }
    }
}
```

### 支持的类型映射

| Python 类型 | JSON Schema 类型 |
|------------|------------------|
| `str` | `string` |
| `int` | `integer` |
| `float` | `number` |
| `bool` | `boolean` |
| `list[T]` | `array` |
| `dict[str, T]` | `object` |
| `T \| None` | 可选字段 |
| `Literal['a', 'b']` | `enum` |

## 实现示例

### LOCAL 工具：Think

```python
from pydantic import BaseModel, Field
from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


class ThinkInput(BaseModel):
    """思考工具输入"""
    thought: str = Field(description='Your thought or reasoning process')


@tool_registry.register(
    'think',
    meta={
        'name': 'Think',
        'description': 'Use this tool to think through a problem step by step.',
        'tool_type': ToolType.LOCAL,
    }
)
async def think_tool(params: ThinkInput, context: ToolContext) -> ToolResponse:
    """思考工具 - 不执行任何操作，只记录思考过程"""
    return ToolResponse(
        output=[{'type': 'text', 'text': 'Thought recorded.'}],
        meta={'thought': params.thought},  # 保存在 meta 中供后续分析
    )
```

### RUNTIME 工具：Bash

```python
import subprocess
from pydantic import BaseModel, Field
from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


class BashInput(BaseModel):
    """Bash 工具输入"""
    command: str = Field(description='The bash command to execute')
    timeout: int = Field(default=60, description='Timeout in seconds')


@tool_registry.register(
    'bash',
    meta={
        'name': 'Bash',
        'description': 'Execute a bash command in the sandbox environment.',
        'tool_type': ToolType.RUNTIME,  # 在远程沙箱执行
    }
)
async def bash_tool(params: BashInput, context: ToolContext) -> ToolResponse:
    """Bash 工具 - 在沙箱中执行命令"""
    try:
        result = subprocess.run(
            params.command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=params.timeout,
        )

        output = result.stdout
        if result.stderr:
            output += f'\n[stderr]\n{result.stderr}'

        return ToolResponse(
            output=[{'type': 'text', 'text': output or '(no output)'}],
            meta={
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
            },
        )

    except subprocess.TimeoutExpired:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Command timed out after {params.timeout}s'}],
            meta={'exit_code': -1, 'timeout': True},
        )
```

### 带状态更新的工具

**重要**：ToolContext 是不可变的，不能直接修改 `context.tool_state`。状态更新必须通过 `ToolResponse.context_updates` 返回：

```python
class FileReadInput(BaseModel):
    """文件读取输入"""
    path: str = Field(description='File path to read')


@tool_registry.register(
    'file_read',
    meta={
        'name': 'ReadFile',
        'description': 'Read contents of a file',
        'tool_type': ToolType.RUNTIME,
    }
)
async def file_read_tool(params: FileReadInput, context: ToolContext) -> ToolResponse:
    """文件读取工具 - 更新共享状态"""
    try:
        with open(params.path) as f:
            content = f.read()

        # 从 context 读取现有状态（只读）
        read_files = list(context.tool_state.get('read_files', []))
        read_files.append(params.path)

        # 通过 context_updates 返回状态更新（不能直接修改 context.tool_state）
        return ToolResponse(
            output=[{'type': 'text', 'text': content}],
            meta={'file_size': len(content)},
            context_updates={'read_files': read_files},  # 状态更新通过返回值
        )

    except FileNotFoundError:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'File not found: {params.path}'}],
            meta={'error': 'file_not_found'},
        )
```

### 多模态输出工具

```python
import base64
from pydantic import BaseModel, Field


class ScreenshotInput(BaseModel):
    """截图工具输入"""
    selector: str | None = Field(default=None, description='CSS selector')


@tool_registry.register(
    'screenshot',
    meta={
        'name': 'Screenshot',
        'description': 'Take a screenshot of the current page',
        'tool_type': ToolType.RUNTIME,
    }
)
async def screenshot_tool(params: ScreenshotInput, context: ToolContext) -> ToolResponse:
    """截图工具 - 返回图像"""
    # 假设有截图逻辑
    image_bytes = take_screenshot(params.selector)
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    return ToolResponse(
        output=[
            {'type': 'text', 'text': 'Screenshot taken'},
            {
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/png;base64,{base64_image}'
                }
            }
        ],
        meta={'width': 1920, 'height': 1080},
    )
```

## ToolResponse 详解

### output 字段

LLM 可见的输出，支持多模态：

```python
# 文本输出
output=[{'type': 'text', 'text': 'Hello'}]

# 图像输出
output=[{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,...'}}]

# 混合输出
output=[
    {'type': 'text', 'text': 'Here is the screenshot:'},
    {'type': 'image_url', 'image_url': {'url': '...'}}
]
```

### meta 字段

内部元数据，不传给 LLM：

```python
meta={
    'exit_code': 0,      # 命令退出码
    'execution_time': 1.5,  # 执行时间
    'raw_output': '...',    # 原始输出
}
```

### context_updates 字段

状态更新，会合并到 `ExecutionContext.tool_state`：

```python
context_updates={
    'files_modified': ['a.py', 'b.py'],
    'last_command': 'git commit',
}
```

## 相关文档

- [Agent 系统详解](./04-agent.md)
- [Runtime 系统详解](./06-runtime.md)
- [ExecutionContext 详解](./02-execution-context.md)
