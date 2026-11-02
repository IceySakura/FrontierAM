# 架构概览

本文描述 Nexus 的核心组件和代码组织方式。

## 执行流

```
Task (async function)
  └─> ExecutionContext
      ├─> Agent
      │   └─> ToolExecutor
      │       ├─> LOCAL tools (direct)
      │       └─> RUNTIME tools (HTTP → runtime server)
      └─> Runtime (sandboxed environment)
```

## Core Components (`nexus/core/`)

### Registry System (`registry/`)

全局注册机制，通过装饰器 `@task_registry.register()`、`@agent_registry.register()`、`@tool_registry.register()` 注册。

- Tools 有两种执行模式：`LOCAL`（进程内）或 `RUNTIME`（通过 HTTP 到沙箱）
- 支持 lazy 注册：`registry.add(name, import_path)` 只记录 import path，不导入模块

### Module System (`module_meta.py`, `module_registry.py`)

元数据驱动的发现和懒注册系统。

- 每个 extension module 在 `pyproject.toml`（目录模块）或 PEP 723 inline script metadata（单文件模块）中声明元数据
- `ModuleRegistry.discover_and_register()` 扫描所有元数据，调用 `registry.add()` — **不导入任何 extension 代码**
- `ModuleMeta` dataclass：`host_deps`、`runtime_extras`、`all_runtime_deps`、`objects`、`category`

### Dependency Tools (`dep_checker.py`, `dep_validator.py`, `import_scanner.py`)

- `dep_checker`：运行时检查 — 声明的依赖 vs `importlib.metadata` 已安装包
- `dep_validator`：静态验证 — AST 扫描 import vs 声明的依赖
- `import_scanner`：基于 AST 的 Python 源文件 import 提取

### ExecutionContext (`context.py`)

异步上下文管理器，提供依赖注入。

- 通过 `require_context()` 在 tasks/agents/tools 内访问
- 提供：runtime 实例、LLM clients、task 参数、shared tool state

### ToolExecutor (`tool_executor.py`)

统一的 tool 调用接口。

- 自动从 Pydantic input model 生成 OpenAI function schema

## Runtime System (`nexus/runtime/`)

按部署边界组织：

- `client/`：Host 侧 Runtime 客户端接口和实现
  - `base.py`：`Runtime` ABC，`ExecutorMixin`（类型化便捷方法）
  - `remote.py`：`RemoteRuntime`（HTTP + Socket.IO transport）
  - `local.py`、`android_websocket.py`、`android_cloudtest.py`、`ags.py`：其他 Runtime 实现
  - `packaging.py`：artifact 构建逻辑（`build_artifact`）
- `server/`：沙箱内侧（运行在容器中）
  - `app.py`：FastAPI + Socket.IO server
  - `backend.py`：`ExecutorBackend`（InProcess / Worker dispatch）
  - `services/`：LLM API 代理（Anthropic、OpenAI、Gemini、Responses）
  - `venv/`：Worker venv 生命周期（BFS closure symlink + install）
  - `worker/`：Worker 子进程管理（launcher + SDK）
- `models.py`：共享线格式模型（`InvokeRequest`/`InvokeResponse`）
- `context.py`：`RuntimeControl` contextvar，runtime 侧代码使用
- `providers/`：Host 侧工厂层（创建 Runtime client 实例）
- `executors/`：Executor 实现（command、browser、terminal 等）

## Runner (`nexus/runner/single.py`)

入口，初始化 ExecutionContext、clients、providers，执行 tasks。

## CLI (`nexus/cli/`)

- `commands/modules.py`：`nexus modules` 和 `nexus modules install` 子命令
- `modules_tui.py`：交互式 Textual TUI，浏览 modules、依赖、安装

## Extensions (`nexus/extensions/`)

参考实现：demo agent、echo task、file operations executor、OpenAI clients。 每个 extension 子目录包含一个 `pyproject.toml` 声明其依赖和对象注册。
