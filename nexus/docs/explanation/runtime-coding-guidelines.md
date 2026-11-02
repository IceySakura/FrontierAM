# Runtime 侧编码指南

`runtime_setup.py`（以及通过 `call_function`/`invoke` 调用的其他函数）运行在沙箱内。 访问 runtime 能力遵循以下规则。

## 1. `sio`（Socket.IO）— 从 `RuntimeControl` 获取

```python
from nexus.runtime.context import require_runtime_control
rt = require_runtime_control()
proxy = AnthropicProxy(sio=rt.sio, listen_port=port)
```

`rt.sio` 在 Base 进程上是 `AsyncServer`，在 Worker 进程上是 `AsyncClient` — 相同的 `emit()` API。 **永远不要** `from nexus.runtime.server.app import sio`。

## 2. Executor 函数（run_command 等）— 直接 import

```python
from nexus.runtime.executors.command import RunCommandRequest, run_command
result = await run_command(RunCommandRequest(command=cmd, cwd=cwd, timeout=timeout))
```

直接 import 确保 artifact shadow 语义生效：如果用户的 artifact 覆盖了某个 executor，被覆盖的版本会运行。 通过 `RuntimeControl` 路由会静默调用 Base 的版本。

## 3. RuntimeControl 上的 executor 方法 — 仅在需要调用 Base 进程时使用

```python
rt = require_runtime_control()
await rt.browser_goto(session_id, url)  # session 在 Base 进程，必须路由过去
```

典型场景：Worker 需要操作 Base 持有的跨 venv 共享 session（如多个 Worker 共用的 browser session）。 如果 executor 状态只在本进程内使用，直接 import 即可。

## 总结

| 资源 | 访问方式 | 原因 |
|------|---------|------|
| `sio` | `require_runtime_control().sio` | 进程级资源，Base 是 AsyncServer，Worker 是 AsyncClient |
| 需要调用 Base 进程的 executor | `require_runtime_control()` | 跨进程路由到 Base（如共享 session） |
| 其他 executor | 直接 import | 尊重 artifact shadow，显式可预测 |
