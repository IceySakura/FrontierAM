# Runtime 运行时系统

Runtime 是 Nexus 框架的沙箱执行环境，提供代码隔离执行能力，支持工具调用、函数序列化执行等功能。

## 目录

1. [概述](#概述)
2. [架构设计](#架构设计)
3. [Runtime 接口](#runtime-接口)
4. [RuntimeProvider](#runtimeprovider)
5. [Docker 实现](#docker-实现)
6. [Runtime Server](#runtime-server)
7. [使用模式](#使用模式)

## 概述

Runtime 系统解决 AI Agent 的安全执行问题：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Runtime 系统概述                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   问题：AI Agent 需要执行不信任的代码（Shell 命令、脚本等）               │
│                                                                         │
│   解决方案：                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                     Agent Process (Host)                        │   │
│   │                                                                 │   │
│   │   Agent Loop                                                    │   │
│   │      │                                                          │   │
│   │      │ Tool: bash 'rm -rf /'                                    │   │
│   │      ▼                                                          │   │
│   │   ToolExecutor                                                  │   │
│   │      │                                                          │   │
│   │      │ RUNTIME Tool                                             │   │
│   │      ▼                                                          │   │
│   │   HttpRuntime.call_tool()  ──────────────────────────────────►  │   │
│   │                                                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│                                    │ HTTP                               │
│                                    ▼                                    │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │              Runtime Container (Isolated Sandbox)               │   │
│   │                                                                 │   │
│   │   FastAPI Server                                                │   │
│   │      │                                                          │   │
│   │      │ POST /api/v1/call_tool                                   │   │
│   │      ▼                                                          │   │
│   │   bash_tool()                                                   │   │
│   │      │                                                          │   │
│   │      │ subprocess.run('rm -rf /')                               │   │
│   │      ▼                                                          │   │
│   │   只影响容器内部，宿主机安全！                                     │   │
│   │                                                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Runtime 系统的组件**：

| 组件 | 职责 |
|------|------|
| `Runtime` | 抽象接口，定义 call_function/call_tool/call_executor |
| `RuntimeProvider` | 管理 Runtime 生命周期（创建、释放） |
| `HttpRuntime` | HTTP 客户端实现，与 Runtime Server 通信 |
| `Runtime Server` | FastAPI 服务，在容器内执行代码 |
| `DockerRuntimeProvider` | Docker 容器管理，创建隔离环境 |

## 架构设计

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Runtime 架构                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ExecutionContext                                                      │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                 │   │
│   │   runtime_provider: RuntimeProvider                             │   │
│   │        │                                                        │   │
│   │        │ configure('main', image='python:3.11')                 │   │
│   │        │ acquire('main') ──────────────────────────────┐        │   │
│   │        │                                               │        │   │
│   │        ▼                                               ▼        │   │
│   │   _runtime_params                              _active_runtimes │   │
│   │   {'main': {image: 'python:3.11'}}             {'main': Runtime}│   │
│   │                                                                 │   │
│   │   get_runtime('main') ──► 返回缓存的 Runtime 或调用 acquire     │   │
│   │                                                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   RuntimeProvider (Abstract)                                            │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                 │   │
│   │   configure(name, **params)   # 预配置参数                       │   │
│   │   acquire(name, **params)     # 创建 Runtime                    │   │
│   │   release(name)               # 释放 Runtime                    │   │
│   │   release_all()               # 释放所有                        │   │
│   │                                                                 │   │
│   │   实现：                                                        │   │
│   │   • DockerRuntimeProvider  ──► Docker 容器                      │   │
│   │   • KubernetesRuntimeProvider (future)                          │   │
│   │   • RemoteRuntimeProvider (future)                              │   │
│   │                                                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   Runtime (Abstract)                                                    │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                 │   │
│   │   call_function(func, *args, **kwargs)  # 序列化执行函数         │   │
│   │   call_tool(ToolCallRequest)            # 执行工具              │   │
│   │   call_executor(name, arg)              # 调用执行器            │   │
│   │                                                                 │   │
│   │   实现：                                                        │   │
│   │   • HttpRuntime  ──► HTTP 通信，dill 序列化                     │   │
│   │                                                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Runtime 接口

### 基类定义

```python
class Runtime(ABC):
    """Runtime 抽象基类"""

    @abstractmethod
    async def call_function(
        self,
        func: Callable[..., T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """序列化并在远程执行函数"""
        ...

    @abstractmethod
    async def call_tool(
        self,
        request: ToolCallRequest,
    ) -> ToolResponse:
        """执行工具调用"""
        ...

    @abstractmethod
    async def call_executor(
        self,
        executor_name: str,
        arg: Any,
    ) -> Any:
        """调用注册的执行器"""
        ...
```

### call_function

使用 `dill` 序列化任意 Python 函数并远程执行：

```python
runtime = await ctx.get_runtime()

# 执行简单函数
result = await runtime.call_function(lambda x: x + 1, 41)
# result = 42

# 执行文件操作
content = await runtime.call_function(
    lambda path: open(path).read(),
    '/workspace/file.txt'
)

# 执行复杂逻辑
def process_data(data):
    import pandas as pd
    df = pd.DataFrame(data)
    return df.describe().to_dict()

stats = await runtime.call_function(process_data, my_data)
```

**注意**：函数必须是可序列化的（使用 dill）

### call_tool

执行注册的工具：

```python
from nexus.core.tool_executor import ToolCallRequest
from nexus.core.tool import ToolContext

request = ToolCallRequest(
    import_path='nexus.extensions.tools::bash_tool',
    params={'command': 'ls -la'},
    context=ToolContext(task_config={}, tool_state={}),
)

response = await runtime.call_tool(request)
print(response.output)  # 命令输出
```

### call_executor

调用注册的执行器：

```python
# 执行文件操作
result = await runtime.call_executor('file', {
    'operation': 'read',
    'path': '/workspace/data.json',
})

# 执行 Jupyter 代码
result = await runtime.call_executor('jupyter', {
    'code': 'import pandas as pd; pd.DataFrame({"a": [1,2,3]})',
})
```

## RuntimeProvider

### 基类定义

```python
class RuntimeProvider(ABC):
    """RuntimeProvider 抽象基类"""

    def __init__(self, **config: Any) -> None:
        self.config = config
        self._active_runtimes: dict[str, Runtime] = {}
        self._runtime_params: dict[str, dict[str, Any]] = {}

    def configure(self, name: str, **params: Any) -> None:
        """预配置 Runtime 参数"""
        if name not in self._runtime_params:
            self._runtime_params[name] = {}
        self._runtime_params[name].update(params)

    def get_params(self, name: str, **override: Any) -> dict[str, Any]:
        """获取合并后的参数"""
        base = self._runtime_params.get(name, {}).copy()
        base.update(override)
        return base

    @abstractmethod
    async def acquire(self, name: str, **params: Any) -> Runtime:
        """创建 Runtime 实例"""
        ...

    @abstractmethod
    async def release(self, name: str) -> None:
        """释放 Runtime 实例"""
        ...

    async def release_all(self) -> None:
        """释放所有 Runtime"""
        for name in list(self._active_runtimes.keys()):
            await self.release(name)
```

### 两阶段参数配置

```python
# 阶段 1：预配置默认参数
provider.configure('main', image='python:3.11', cpu=2)
provider.configure('eval', image='eval-sandbox:latest')

# 阶段 2：获取时可覆盖/扩展参数
runtime = await provider.acquire('main', memory='4G')
# 最终参数: {image: 'python:3.11', cpu: 2, memory: '4G'}
```

## Docker 实现

### DockerRuntimeProvider

```python
@runtime_provider_registry.register('docker')
class DockerRuntimeProvider(RuntimeProvider):
    """Docker 容器管理

    配置选项：
        runtime_image: Runtime 镜像名
        force_rebuild: 强制重建镜像
        python_version: Python 版本

    acquire 参数：
        image: Docker 镜像（必填）
        port: 主机端口（可选，自动分配）
        extra_args: 额外 docker run 参数
    """
```

### 容器架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Docker 容器架构                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   nexus-runtime:{version} (预构建镜像)                                   │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                 │   │
│   │   /nix/runtime/                                                 │   │
│   │   ├── nix-env/         # Nix 环境                               │   │
│   │   ├── nexus/           # Nexus 包                               │   │
│   │   └── python-{ver}/    # Python 环境                            │   │
│   │                                                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                       │                                 │
│                                       │ docker run --volumes-from       │
│                                       ▼                                 │
│   nexus-user-{name} (用户容器)                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                 │   │
│   │   Base Image: {user-specified, e.g., 'python:3.11'}             │   │
│   │                                                                 │   │
│   │   挂载:                                                          │   │
│   │   /nix/runtime/  ◄── 来自 nexus-runtime 容器                     │   │
│   │                                                                 │   │
│   │   运行:                                                          │   │
│   │   /nix/runtime/nexus/deployment/runtime/start.sh               │   │
│   │       ├── 启动 FastAPI Server                                   │   │
│   │       └── 监听端口 8000                                         │   │
│   │                                                                 │   │
│   │   端口映射: localhost:{auto} ──► container:8000                  │   │
│   │                                                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 使用示例

```python
# 配置
runtime_provider_name: "docker"
runtime_provider_config:
  force_rebuild: false
runtime_configs:
  main:
    image: "python:3.11"
  eval:
    image: "eval-sandbox:latest"
```

```python
# 代码中使用
ctx = require_context()

# 配置并获取 Runtime
ctx.runtime_provider.configure('main', extra_args=['--network=none'])
runtime = await ctx.get_runtime('main')

# 执行代码
result = await runtime.call_function(lambda: 'Hello from container!')
```

## Runtime Server

Runtime Server 是运行在容器内的 FastAPI 服务：

### 端点定义

```python
# /health - 健康检查
@app.get('/health')
def health():
    return {'status': 'ok'}

# /api/v1/call_tool - 工具执行
@api_v1_router.post('/call_tool')
async def call_tool(request: CallToolRequest):
    tool_func = import_from_path(request.import_path)
    meta = get_tool_meta(tool_func)
    parsed_params = meta.input_model.model_validate(request.params)
    return await tool_func(parsed_params, request.context)

# /api/v1/call_function - 函数执行
@api_v1_router.post('/call_function')
async def call_function(request: CallFunctionRequest):
    func = dill.loads(base64.b64decode(request.func))
    args = dill.loads(base64.b64decode(request.args))
    kwargs = dill.loads(base64.b64decode(request.kwargs))
    result = func(*args, **kwargs)
    return CallFunctionResponse(
        success=True,
        result=base64.b64encode(dill.dumps(result)).decode('utf-8'),
    )

# /api/v1/call_executor - 执行器调用
@api_v1_router.post('/call_executor')
async def call_executor(request: CallExecutorRequest):
    executor = executor_registry.get(request.executor_name)
    return await executor(request.arg)
```

### 请求/响应模型

```python
class CallToolRequest(BaseModel):
    import_path: str  # 'nexus.extensions.tools::bash_tool'
    params: dict[str, Any]
    context: ToolContext

class CallFunctionRequest(BaseModel):
    func: str   # base64(dill.dumps(function))
    args: str   # base64(dill.dumps(args))
    kwargs: str # base64(dill.dumps(kwargs))

class CallFunctionResponse(BaseModel):
    success: bool
    result: str | None = None    # base64(dill.dumps(result))
    exception: str | None = None # base64(dill.dumps(exception))
    traceback: str | None = None
```

## 使用模式

### 模式 1：基础使用

```python
@task_registry.register('basic_task')
async def basic_task(task_input: BasicTaskInput) -> BasicTaskOutput:
    ctx = require_context()

    # 配置并获取 Runtime
    runtime = await ctx.get_runtime()

    # 执行函数
    result = await runtime.call_function(lambda: 'Hello!')

    return BasicTaskOutput(result=result)
```

### 模式 2：多 Runtime

```python
@task_registry.register('multi_runtime_task')
async def multi_runtime_task(task_input: MultiRuntimeInput) -> MultiRuntimeOutput:
    ctx = require_context()

    # Agent 使用 Python 环境
    ctx.runtime_provider.configure('main', image='python:3.11')
    main_runtime = await ctx.get_runtime('main')

    # 评估使用专用沙箱
    ctx.runtime_provider.configure('eval', image='eval-sandbox:latest')
    eval_runtime = await ctx.get_runtime('eval')

    # 两个 Runtime 完全隔离
    await main_runtime.call_function(agent_work)
    score = await eval_runtime.call_function(evaluate)

    return MultiRuntimeOutput(score=score)
```

### 模式 3：Oracle 模式

```python
@task_registry.register('oracle_task')
async def oracle_task(task_input: OracleTaskInput) -> OracleTaskOutput:
    ctx = require_context()
    runtime = await ctx.get_runtime()

    if task_input.run_oracle:
        # 直接在 Runtime 中写入正确答案
        await runtime.call_function(
            lambda answer: open('/tmp/answer.txt', 'w').write(answer),
            task_input.expected_answer
        )
    else:
        # 正常调用 Agent
        await task_input.run_agent(user_prompt='...')

    # 统一评估
    result = await runtime.call_function(check_answer)
    return OracleTaskOutput(score=result)
```

### 模式 4：复杂文件操作

```python
async def complex_file_operations():
    ctx = require_context()
    runtime = await ctx.get_runtime()

    # 创建工作目录
    await runtime.call_function(
        lambda: os.makedirs('/workspace/project', exist_ok=True)
    )

    # 写入多个文件
    files = {'main.py': 'print("Hello")', 'test.py': 'assert True'}
    for name, content in files.items():
        await runtime.call_function(
            lambda n, c: open(f'/workspace/project/{n}', 'w').write(c),
            name, content
        )

    # 执行测试
    result = await runtime.call_function(
        lambda: subprocess.run(['python', '-m', 'pytest'], capture_output=True)
    )
```

## 安全考虑

1. **网络隔离**：可通过 `extra_args=['--network=none']` 禁用网络
2. **资源限制**：可设置 CPU、内存限制
3. **文件系统隔离**：容器内的文件操作不影响宿主机
4. **进程隔离**：容器内的进程与宿主机隔离

```python
# 安全配置示例
ctx.runtime_provider.configure('secure',
    image='python:3.11',
    extra_args=[
        '--network=none',           # 禁用网络
        '--memory=2g',              # 内存限制
        '--cpus=1',                 # CPU 限制
        '--read-only',              # 只读文件系统
        '--tmpfs=/tmp:size=100m',   # 临时文件系统
    ]
)
```

## 相关文档

- [Tool 系统详解](./05-tool.md)
- [ExecutionContext 详解](./02-execution-context.md)
- [执行流程教程](../getting-started/02-execution-flow.md)
