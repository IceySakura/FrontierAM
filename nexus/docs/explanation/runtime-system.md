# Runtime Provider 与 Runtime

在 Nexus 里，Task / Agent / Tool 并不直接决定“动作在哪个环境执行”。真正负责这一层的是 Runtime 系统。它可以拆成三个连续的抽象：

- `ExecutionContext`：在任务执行期间按需获取 runtime，并在结束时统一清理。
- `RuntimeProvider`：负责创建、连接、释放某类运行环境。
- `Runtime`：对上层暴露稳定的调用接口，屏蔽容器、远端沙箱、安卓设备、AGS 原生服务等底层差异。

通用运行时架构先看这篇；如果你只关心腾讯云 AGS 的多形态接入、`interface_mode=raw` 和能力保真度，请继续看 [AGS Runtime Provider 设计](./ags-runtime-provider-design.md)。

## 一张图看懂

<figure markdown="span" style="width: 100%"><object type="image/svg+xml" data="/assets/images/runtime-system/provider-runtime-overview.svg" style="width: 100%"></object><figcaption>ExecutionContext、RuntimeProvider、Runtime 与底层协议的关系</figcaption></figure>

## 三层抽象

### ExecutionContext

- `ctx.get_runtime(name='main', **params)` 是懒加载的；同名 runtime 第一次获取才会调用 provider 的 `acquire()`。
- `ExecutionContext` 会缓存已经拿到的 runtime；同名后续调用直接复用。
- `async with ExecutionContext(...):` 退出时会统一调用 `runtime_provider.release_all()`，所以 runtime 生命周期天然跟任务作用域绑定。

### RuntimeProvider

- `configure(name, **params)` 用于预配置默认参数。
- `acquire(name, **params)` 用于真正拿到 runtime；provider 会把预配置参数和调用时覆盖参数合并。
- `release(name)` / `release_all()` 用于回收资源。
- 这一层解决的是“环境怎么来”，不是“环境里具体怎么执行某个动作”。

### Runtime

- `invoke()`：统一远程调用原语，通过 import path + pickle 参数调用函数，支持 `venv_id` 路由到 Worker。
- `call_tool()`：在 runtime 内调用已注册 tool。
- `call_executor()`：最小公共接口；所有 runtime 都必须能落到这里。
- `create_venv()` / `destroy_venv()`：创建/销毁 Worker Venv。
- `port_mappings`：额外暴露端口的映射表。

`Runtime` 还继承了 `ExecutorMixin`，因此调用方会看到很多类型化便捷方法，例如 `run_command()`、`upload_file()`、`create_browser_session()`、`gui_click()`、`mcp_start_proxy()`。但这里有一个很重要的边界：**这些方法的存在不等于能力一定实现**。真正的能力边界仍由具体 runtime 的 `call_executor()` 决定；不支持的 executor 会在运行时失败。

## RuntimeProvider 总览

| Provider | 底层环境 | 典型返回类型 | 最适合 |
| --- | --- | --- | --- |
| `local` | 当前 Python 进程 | `LocalRuntime` | 本地调试、无隔离验证 |
| `docker` | 本地 Docker 容器 + Nexus runtime server | `RemoteRuntime` | 本地开发、Coding Agent |
| `gongfeng` | 远端沙箱服务 | `RemoteRuntime` | 批量评测、生产调度 |
| `android_websocket` | WebSocket Android 沙箱 | `WebSocketAndroidRuntime` | Android GUI 自动化沙箱 |
| `android_cloudtest` | CloudTest 真机云设备 | `CloudTestAndroidRuntime` | 真机交互、移动端评测 |
| `ags` | 腾讯云 AGS | `RemoteRuntime` / `AGS*Runtime` / `AGSRaw*Runtime` | 自定义沙箱、浏览器、code-interpreter、OSWorld |

其中 `local`、`docker`、`gongfeng`，以及 AGS 的 `launch_mode=nexus_server` 路径都属于“通用 runtime provider”：它们返回的是完整的 Nexus Runtime 抽象。Android providers 和 AGS 的 native 路径则属于“专用 runtime provider”：它们只实现某个 executor 子集，或者直接暴露原始协议。

## Runtime 家族与接口边界

### 通用 Runtime

| Runtime | 传输方式 | `invoke` | `call_tool` | `call_executor` | Worker Venv |
| --- | --- | --- | --- | --- | --- |
| `LocalRuntime` | 当前进程直接调用 | 支持 | 支持 | 支持本地已注册 executors | 不支持 |
| `RemoteRuntime` | HTTP RPC + Socket.IO 事件流 | 支持 | 支持 | 支持 Nexus runtime server 暴露的全部 executors | 支持 |

`RemoteRuntime` 默认使用 HTTP 做 RPC，并自动建立 Socket.IO 连接用于事件流和日志。支持 HTTP long-poll、blob sideband（大 payload 分流）以及 Worker Venv 路由。

### 专用 Runtime

| Runtime | 主要能力子集 | 背后协议/服务 |
| --- | --- | --- |
| `WebSocketAndroidRuntime` | `gui_*` | WebSocket Android sandbox protocol |
| `CloudTestAndroidRuntime` | `gui_*` | CloudTest API |
| `AGSBrowserRuntime` | `browser_*` | AGS browser + CDP + Playwright |
| `AGSCodeInterpreterRuntime` | `run_command` | AGS code-interpreter `/execute` |
| `AGSOSWorldRuntime` | `run_command` + `gui_*` | OSWorld-compatible HTTP server |
| `AGSRawRuntime` | `raw_request` / `raw_get_metadata` | 任意 AGS native/custom HTTP surface |
| `AGSRawBrowserRuntime` | 原始 CDP / direct Playwright helpers | AGS browser |
| `AGSRawCodeInterpreterRuntime` | 原始 `/execute` | AGS code-interpreter |
| `AGSRawOSWorldRuntime` | 原始 OSWorld HTTP 接口 | OSWorld-compatible server |

`AGSRaw*Runtime` 仍然继承 `Runtime`，是因为 `RuntimeProvider.acquire()` 的返回契约固定就是 `Runtime`。但 raw 路径的主要价值不在 `invoke()` / `call_tool()`，而在直接的 helper 方法，例如 `connect_playwright()`、`execute()`、`screenshot()`、`accessibility_tree()` 等。

## Provider -> Runtime 返回矩阵

| Provider | 关键配置 | 返回 Runtime |
| --- | --- | --- |
| `local` | 默认 | `LocalRuntime` |
| `docker` | 默认 | `RemoteRuntime` |
| `gongfeng` | 默认 | `RemoteRuntime` |
| `android_websocket` | 默认 | `WebSocketAndroidRuntime` |
| `android_cloudtest` | 默认 | `CloudTestAndroidRuntime` |
| `ags` | `launch_mode=nexus_server` | `RemoteRuntime` |
| `ags` | `launch_mode=native`, `interface_mode=nexus`, `runtime_kind=browser` | `AGSBrowserRuntime` |
| `ags` | `launch_mode=native`, `interface_mode=nexus`, `runtime_kind=code` | `AGSCodeInterpreterRuntime` |
| `ags` | `launch_mode=native`, `interface_mode=nexus`, `runtime_kind=osworld` | `AGSOSWorldRuntime` |
| `ags` | `launch_mode=native`, `interface_mode=raw`, `runtime_kind=browser` | `AGSRawBrowserRuntime` |
| `ags` | `launch_mode=native`, `interface_mode=raw`, `runtime_kind=code` | `AGSRawCodeInterpreterRuntime` |
| `ags` | `launch_mode=native`, `interface_mode=raw`, `runtime_kind=osworld` | `AGSRawOSWorldRuntime` |
| `ags` | `launch_mode=native`, `interface_mode=raw`, `runtime_kind=custom` | `AGSRawRuntime` |

可以把这张表理解成一句话：大部分 provider 只决定“环境在哪”；AGS 还额外决定“返回 Nexus 风格接口，还是返回尽量保真的原始接口”。

## 跨 Runtime Trace

Host 发起远端调用时会传播当前 `trace_id`、parent span 和 managed run identity。Base runtime 或 Worker 产生的 Nexus
`SpanEvent` 通过 Socket.IO 返回 Host，并在 Host `SpanController` 中重放，因此远端 span 保持原来的 run/task 层级。

live runner 和 distributed driver 在发布最终 `run.output`、释放 runtime 前调用 trace drain barrier。barrier 依次等待
Worker emitter、Base broadcaster 和 Host replay，避免 RPC 已完成但尾部 runtime span 尚未进入 run trace sink。关闭
Socket.IO 自动连接时 drain 是 no-op；启用自动连接却失去连接时则显式报错。

barrier 只保证调用前已接收事件的有序交付，不提供进程崩溃后的重传或 exactly-once。完整 trace 结构见
[Trace 系统](./trace-system.md)。

## 为什么 AGS 是特例

`AGSRuntimeProvider` 是当前唯一一个同时覆盖“通用 runtime server”与“原生专用服务”的 provider。它内部有三个正交维度：

| 维度 | 取值 | 含义 |
| --- | --- | --- |
| `provision_mode` | `control_plane` / `template` | 实例怎么拿到 |
| `launch_mode` | `nexus_server` / `native` | 沙箱里实际跑什么 |
| `interface_mode` | `nexus` / `raw` | provider 向上层返回什么接口形态 |

这也是为什么 AGS 既能像 `docker` / `gongfeng` 一样返回 `HttpRuntime` / `SocketIoRuntime`，又能返回 `AGSBrowserRuntime`、`AGSOSWorldRuntime` 甚至 `AGSRawOSWorldRuntime` 这样的专用实现。详细设计、API 转换链路和能力损耗分析见 [AGS Runtime Provider 设计](./ags-runtime-provider-design.md)。

## 如何选 Provider / Runtime

- 需要完整的 `invoke` / `call_tool` / `file` / `jupyter` / `terminal` / `mcp` 语义时，优先选 `docker`、`gongfeng`，或 AGS 的 `launch_mode=nexus_server`。
- 需要本地零门槛调试、可以接受没有隔离时，选 `local`。
- 任务本质是 Android GUI 自动化时，选 `android_websocket` 或 `android_cloudtest`。
- 想把 AGS 原生 browser / code / OSWorld 快速接进现有 Nexus 工具体系时，选 AGS 的 `interface_mode=nexus`。
- 想保留 AGS 原始协议形态，或者要兼容原版 browser / OSWorld agent 栈时，选 AGS 的 `interface_mode=raw`。

当前仓库的OSWorld-V1六模式不是上述原版agent栈：标准配置使用
`ags + control_plane + interface_mode=nexus + runtime_kind=osworld`，
题目与setup/evaluator资源走COS `task_files`。
Docker/Gongfeng仍是通用runtime provider，但不是当前OSWorld评测的默认通路。
运行命令见[OSWorld原生运行指南](../how-to/osworld-runbook.md)。

## 进一步阅读

- [Client/Server 分离](./client-server-split.md)
- [Worker Venv 机制](./worker-venv.md)
- [Runtime 镜像版本策略](./runtime-image-versioning.md)
- [AGS Runtime Provider 设计](./ags-runtime-provider-design.md)
- [Runtime Injection](./runtime-injection.md)
- [Configure Runtime](../how-to/use-runtime/configure-runtime.md)
- [Interact with Runtime](../how-to/use-runtime/interact-with-runtime.md)
