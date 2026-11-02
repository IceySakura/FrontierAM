# Runtime Client/Server 分离

本文介绍 Nexus 1.0 中将 Runtime 通信层从单体 `protocol/` 拆分为 `client/` 和 `server/` 两个独立包的设计决策。

## 背景：旧 `protocol/` 的问题

在 1.0 之前，`nexus/runtime/protocol/` 是一个包含约 4500 行代码的单体模块：

```
nexus/runtime/protocol/
├── client.py    # ~3800 行：HttpRuntime, SocketIoRuntime, 所有 AGS Runtime 变体
├── server.py    # ~700 行：FastAPI server + Socket.IO handlers
└── ags.py       # AGS 协议适配
```

这带来了几个问题：

1. **Host/Runtime 代码耦合**：`client.py` 运行在 Host，`server.py` 运行在沙盒，两者在同一个 package 下导致 import 污染——Base 镜像不得不安装 Host 侧依赖
2. **单文件过大**：`client.py` 接近 4000 行，包含了所有 Runtime 实现（HTTP、Socket.IO、AGS 各种变体），难以维护
3. **模型定义分散**：请求/响应模型散落在 client 和 server 中，缺乏统一的协议层
4. **Provider-specific 代码位置错误**：AGS 的 Browser/CodeInterpreter/OSWorld Runtime 实现本应属于 AGS provider，却放在了通用 protocol 层

## 新架构

### 目录结构

```
nexus/runtime/
├── client/                     # Host 侧（orchestrator 上运行）
│   ├── base.py                 # Runtime ABC + ExecutorMixin
│   ├── remote.py               # RemoteRuntime（HTTP + Socket.IO）
│   ├── local.py                # LocalRuntime（进程内执行）
│   ├── models.py               # Host 侧模型（VenvHandle, WorkerVenvConfig 等）
│   └── packaging.py            # Artifact 打包逻辑
├── server/                     # 沙盒侧（容器内运行）
│   ├── app.py                  # FastAPI + Socket.IO server
│   ├── backend.py              # ExecutorBackend
│   ├── venv/                   # Worker venv 管理
│   │   ├── manager.py          # VenvManager
│   │   └── models.py           # VenvSpec, VenvStatus
│   ├── worker/                 # Worker 子进程
│   │   ├── launcher.py         # WorkerLauncher
│   │   └── sdk.py              # WorkerSDK（Worker 侧 SIO client）
│   └── services/               # LLM API 代理
├── models.py                   # 共享协议模型（InvokeRequest/Response）
├── context.py                  # RuntimeControl + SIOBridge
├── providers/                  # Provider 工厂层（各自独立 package）
│   ├── docker.py
│   ├── gongfeng.py
│   ├── ags/                    # AGS provider + 所有 AGS Runtime 实现
│   ├── agr_mobile/             # AGR Mobile provider + client
│   ├── android_cloudtest/      # CloudTest provider + client
│   └── android_websocket/      # WebSocket Android provider + client
└── executors/                  # Executor 实现
    ├── models/                 # Executor Pydantic 模型（Host/Runtime 共享）
    └── ...                     # 具体 executor 实现
```

### 核心原则

**按部署边界切分**：`client/` 的代码只在 Host 上运行，`server/` 的代码只在沙盒内运行。 两者通过 `models.py` 中定义的协议模型通信。

**Provider 内聚**：AGS、Android 等专用 Runtime 实现移入对应的 provider 包。 通用 protocol 层只保留 `RemoteRuntime` 和 `LocalRuntime`。

**Executor Models 独立**：请求/响应的 Pydantic 模型从 executor 实现中提取到 `executors/models/`，作为 Host 和 Runtime 共享的协议契约。

## 关键变更

### Runtime 统一

旧架构中有两个独立的 Runtime 类：

| 旧 | 新 | 说明 |
|---|---|---|
| `HttpRuntime` | `RemoteRuntime` | HTTP + Socket.IO 合并 |
| `SocketIoRuntime` | `RemoteRuntime` | 统一为一个类，SIO 作为可选 transport |

`RemoteRuntime` 默认使用 HTTP 做 RPC，自动建立 Socket.IO 连接用于事件流和日志。 不再需要单独选择 transport。

### `invoke()` 统一原语

旧架构提供三个独立的 RPC 接口：`call_function`、`call_tool`、`call_executor`。 新架构引入 `invoke()` 作为统一的远程调用原语：

```python
# 旧方式：序列化整个函数
result = await runtime.call_function(my_func, arg1, arg2)

# 新方式：指定 import path + 参数
result = await runtime.invoke('my_module::my_func', arg1, arg2)

# 也可以传入函数对象（自动提取 import path）
result = await runtime.invoke(my_func, arg1, arg2)

# 指定在某个 Worker Venv 中执行
result = await runtime.invoke(my_func, arg1, venv_id='worker-1')
```

`invoke()` 的优势：

- 不依赖 dill 序列化——通过 import path 定位函数，参数用 pickle
- 支持 `venv_id` 路由到 Worker 进程
- HTTP long-poll 减少延迟，大 payload 走 blob sideband
- 自动 dict → Pydantic BaseModel 类型强转

### SIOBridge

Worker 进程的 Socket.IO 是 `AsyncClient`，Base 进程的是 `AsyncServer`——两者的 `emit()` API 签名不同。 `SIOBridge` 提供统一接口：

```python
class SIOBridge:
    def __init__(self, sio, namespace=None):
        # sio 可以是 AsyncServer 或 AsyncClient

    async def emit(self, event, data=None):
        # 自动适配 server/client 差异

    def on(self, event, handler=None):
        # 统一 handler 签名：async def handler(data)
        # Server 端自动剥离 sid 参数
```

这使得 LLM proxy service 等需要 emit 的代码可以在 Base 和 Worker 中复用，无需关心底层 SIO 类型。

### Executor Models 提取

Executor 的请求/响应模型从实现文件中独立出来：

```
executors/models/
├── __init__.py      # 统一导出所有模型
├── base.py          # ExecutorRequest[ResponseT]
├── browser.py       # BrowserGotoRequest, BrowserObservation, ...
├── command.py       # RunCommandRequest, RunCommandResponse
├── file.py          # FileItem, UploadFileRequest, ...
├── gui.py           # GuiClickRequest, GuiScreenshotResponse, ...
├── jupyter.py       # IPythonSessionInfo, ExecuteIPythonRequest, ...
└── terminal.py      # TerminalSessionInfo, RunCommandInSessionRequest, ...
```

这些模型是 Host 和 Runtime 之间的 **协议契约**：Host 侧 `ExecutorMixin` 用它们构造请求，Runtime 侧 executor 用它们解析请求。 将模型独立出来使得两侧可以独立演进，只要协议模型不变。

## 迁移影响

### Import 路径变更

| 旧路径 | 新路径 |
|---|---|
| `nexus.runtime.protocol.client.HttpRuntime` | `nexus.runtime.client.remote.RemoteRuntime` |
| `nexus.runtime.protocol.client.SocketIoRuntime` | `nexus.runtime.client.remote.RemoteRuntime` |
| `nexus.runtime.protocol.client.LocalRuntime` | `nexus.runtime.client.local.LocalRuntime` |
| `nexus.runtime.protocol.server` | `nexus.runtime.server.app` |
| `nexus.runtime.protocol.ags` | `nexus.runtime.providers.ags.*` |

### 用户代码影响

- 通过 `ExecutionContext.get_runtime()` 获取 Runtime 的代码 **无需改动**——返回类型仍然是 `Runtime` ABC
- 直接 import 具体 Runtime 类的代码需要更新 import 路径
- `call_function` 仍然可用，但推荐逐步迁移到 `invoke()`

## 进一步阅读

- [Worker Venv 机制](worker-venv.md)
- [Runtime 侧编码指南](runtime-coding-guidelines.md)
- [Runtime Provider 与 Runtime](runtime-system.md)
