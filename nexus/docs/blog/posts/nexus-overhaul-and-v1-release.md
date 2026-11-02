---
date: 2026-04-14
readtime: 12
authors:
  - tarschen
---

# Nexus Overhaul and v1 Release

随着 Nexus 接入的模块越来越多，依赖膨胀和镜像更新成本高两个问题日益突出。 本次重构对模块系统和 Runtime 层做了一次大的架构升级，并以此为契机发布 1.0 版本，为后续更长期的开发打下基础。 本文总结变更动机、核心改动和迁移指南。

<!-- more -->

## 背景

### 依赖膨胀

所有扩展模块的依赖通过一个 `.[extensions]` extra 统一安装，越来越臃肿。 不同模块有不同的依赖——跑 SWE-Bench 不需要安装浏览器自动化的包，跑 WebArena 不需要安装代码分析的包——但现有机制没有办法区分，只能全装。 `import nexus.extensions` 还会触发所有模块的 eager import，进一步拖慢启动速度并制造不必要的依赖冲突。

### 镜像与代码绑定

每次修改 Runtime 侧的代码都需要重新构建镜像，这是反直觉的。 一个 agent 的 `runtime_setup.py` 改了一行，就得等镜像构建完才能调试。 而且 Runtime 侧代码可以使用的上下文没有明确界定，缺乏清晰的编程模型。

## 模块系统

针对依赖膨胀问题，这次重构把之前代码里隐含的模块化概念完全落地：每个扩展自己声明依赖和注册项，全部懒加载，每个使用场景都不需要安装自己用不到的包。

### 元数据驱动的模块发现

每个扩展现在是一个正式的 **模块**——可以是一个带 `pyproject.toml` 的目录，也可以是一个使用 PEP 723 inline metadata 的单文件：

```toml
# nexus/extensions/agents/swe_agent/pyproject.toml
[project]
name = "swe_agent"
version = "0.1.0"
dependencies = []

[tool.nexus]
agents = { swe_agent = "nexus.extensions.agents.swe_agent.agent" }

[tool.nexus.tools]
"swe_agent/bash" = "nexus.extensions.agents.swe_agent.tools"
"swe_agent/str_replace_editor" = "nexus.extensions.agents.swe_agent.tools"
```

Nexus 在启动阶段扫描所有模块目录的元数据，完成懒注册——**不导入任何 extension 代码**。 只有在首次使用某个模块时才真正导入。 不需要安装用不到的依赖，模块之间隔离后稳定性也更好。

详见 [Module 元数据格式](/explanation/module-metadata)（格式规范）和 [创建新模块](/how-to/manage-modules/create-module)（从零开始的完整流程）。

### 三层依赖模型

模块依赖分为 base / host / runtime 三层，Host 和 Runtime 环境各自只安装需要的部分：

| 层 | 声明位置 | 安装在 Host | 安装在 Runtime |
|---|---|---|---|
| **Base** | `dependencies` | ✓ | ✓ |
| **Host Extras** | `optional-dependencies.host` | ✓ | ✗ |
| **Runtime Extras** | `optional-dependencies.runtime` | ✗ | ✓ |

配合新的 CLI 管理和安装（详见 [Module CLI 使用指南](/how-to/manage-modules/module-cli)）：

```bash
nexus modules                          # 查看所有模块和依赖状态
nexus modules install swe_agent        # 安装指定模块的 host 依赖
nexus modules install --all --runtime  # 安装所有 runtime 依赖（Dockerfile 用）
nexus modules --tui                    # 交互式 TUI
```

也可以设置 `NEXUS_AUTO_INSTALL=1`，框架在懒加载模块时如果发现依赖缺失，会自动安装后重试，开发调试时不需要手动管理依赖。

后续新增任何依赖和代码只影响对应模块，不影响 nexus 主体，极大增强了扩展性。

### Provider 模块化

Runtime Provider 也纳入了模块体系。 单文件 provider 升级为带 `pyproject.toml` 的 package，各自声明自己的依赖：

```
nexus/runtime/providers/ags/
├── pyproject.toml                       # 依赖声明（e2b, playwright, tencentcloud-sdk）
├── provider.py                          # Provider 工厂
├── base.py, browser.py, osworld.py ...  # AGS Runtime 实现
```

Android 系列 provider（`agr_mobile`、`android_cloudtest`、`android_websocket`）同理。

### Extension 重组

- Agent + Task 紧密配对的模块合并到 `projects/`（如 `agts`：agent + task + tools 在同一个包中）
- 只被特定 task 使用的 agent 下沉到对应 task 包内（如 `tau2/agent.py` → `tasks/tau2/`，`vita_agent` → `tasks/vita/`）
- `swe_agent_rubrics` 移入 `tasks/swe/` 子目录

## Runtime 架构

针对镜像与代码绑定问题，重新设计了 Runtime 的 API 分层和代码组织，引入 Worker Venv 实现不重建镜像即可调试 Runtime 侧代码，并明确了 Runtime 侧的编程模型。

### invoke 统一原语

引入 `invoke()` 作为底层统一的远程调用原语。 `call_executor` 和 `call_tool` 作为上层便捷 API 保留，底层都走 `invoke`。 `call_function` 因为功能与 `invoke` 重合，已取消。

```python
# invoke：底层原语，通过 import path 定位函数，参数用 pickle
result = await runtime.invoke('my_module::my_func', arg1)
result = await runtime.invoke(my_func, arg1)              # 也可传函数对象
result = await runtime.invoke(my_func, arg1, venv_id='w1') # 路由到 Worker

# call_executor / call_tool：上层便捷 API，内部走 invoke
result = await runtime.run_command(command='ls', timeout=10)
response = await runtime.call_tool('swe_agent/bash', params, context)
```

旧的 `HttpRuntime` / `SocketIoRuntime` 统一为 `RemoteRuntime`。 详见 [Client/Server 分离](/explanation/client-server-split)。

### Worker Venv

这是解决"改一行代码要重建镜像"问题的核心改动。

沙盒容器内的 Base 进程运行着镜像里预装的 nexus 代码。 当用户在本地修改了代码（比如在 feature 分支上改了 agent 的 `runtime_setup`），通过 `worker_venv: true` 可以把本地代码打包上传到沙盒，创建一个隔离的 Worker Venv 来运行：

1. `uv venv` 创建空 Python 环境
2. BFS 传递闭包——从 Base 环境符号链接已有的包（~1-3 秒）
3. `uv pip install artifact.whl`——如果依赖已被符号链接完全覆盖，走 `--no-deps` 快速路径；否则走完整解析，联网下载缺失的包
4. 启动 Worker 子进程，通过 SIO `/worker` 命名空间与 Base 通信

Worker Venv 支持任意带依赖的代码执行——用户代码声明的依赖如果 Base 有则符号链接，没有则从网络安装。

```yaml
runtime_configs:
  main:
    image: ubuntu-wksp:latest
    worker_venv: true  # 打包本地代码上传到沙盒运行
```

详见 [Worker Venv 机制](/explanation/worker-venv)（架构设计和算法细节）和 [Worker Venv 配置指南](/how-to/use-runtime/worker-venv-config)（使用场景和配置示例）。

### 编程模型

针对背景中"Runtime 侧缺乏清晰编程模型"的问题，明确界定了三条规则：

- **与 Base 交互**：通过 `require_runtime_control()` 获取 `RuntimeControl` 对象，`rt.sio` 提供统一的 Socket.IO 接口（`SIOBridge` 抹平了 Base 的 `AsyncServer` 和 Worker 的 `AsyncClient` 差异）
- **调用 executor**：直接 import 使用，保证 artifact shadow 语义生效
- **需要调用 Base 持有的共享资源**（如跨 Worker 的 browser session）：通过 `require_runtime_control()` 路由

详见 [Runtime 侧编码指南](/explanation/runtime-coding-guidelines)。

### Client/Server 分离

按部署边界拆分，`client/` 的代码只在 Host 运行，`server/` 的代码只在沙盒运行：

```
nexus/runtime/
├── client/               # Host 侧
│   ├── base.py           # Runtime ABC + ExecutorMixin
│   ├── remote.py         # RemoteRuntime（HTTP + SIO）
│   ├── local.py          # LocalRuntime
│   └── packaging.py      # Artifact 打包
├── server/               # 沙盒侧
│   ├── app.py            # FastAPI + SIO server
│   ├── venv/             # Worker Venv 管理
│   └── worker/           # Worker 子进程（launcher + SDK）
├── models.py             # 共享协议（InvokeRequest/Response）
├── context.py            # RuntimeControl + SIOBridge
└── executors/
    └── models/           # Executor Pydantic 模型（Host/Runtime 共享协议契约）
```

Executor 的请求/响应模型独立到 `executors/models/`，不需要引入对应 executor 的实际依赖就能使用这些模型——Host 侧构造请求、Runtime 侧解析请求，只要模型不变两侧可独立演进。

## 发布与镜像策略

1.0 之后每个版本都发布到内部 PyPI（`tencent-nexus`）。 镜像方面：

- **不再维护 `latest`**：每个版本使用明确的版本号标签，`runtime_image: current`（默认值）自动解析到当前 Host 版本
- **main 分支只在版本升级时推送产物**： 合并到 main 分支的 commit，只有在更新了版本号时才触发构建，同版本不再更新。 有 Runtime 侧内容更新就升级版本号
- **非 main 分支开发**：只改了 Host 代码不需要额外操作；改了 Runtime 代码用 `worker_venv: true`；需要较大规模使用（RL 等高并发场景）走 COS 中转（WIP）

详见 [Runtime 镜像版本策略](/explanation/runtime-image-versioning)。

## 迁移指南

### 配置文件

现有 YAML 配置文件基本 **不需要改**。 主要变化：

| 变更项 | 旧 | 新 |
|---|---|---|
| 镜像标签 | `runtime_image: latest` | `runtime_image: current` 或不填（默认值） |
| 模块导入 | `imports: [nexus.extensions]` | 不需要，自动发现（写了也不报错，兼容） |

### 依赖安装

直接安装 `.[extensions]` 的方式需要替换为 CLI 安装：

```bash
# 旧
uv pip install -e ".[extensions]"

# 新
uv pip install -e .
nexus modules --tui
```

### Import 路径变更

| 旧路径 | 新路径 |
|---|---|
| `nexus.runtime.protocol.client.HttpRuntime` | `nexus.runtime.client.remote.RemoteRuntime` |
| `nexus.runtime.protocol.client.SocketIoRuntime` | `nexus.runtime.client.remote.RemoteRuntime` |
| `nexus.runtime.protocol.client.LocalRuntime` | `nexus.runtime.client.local.LocalRuntime` |
| `nexus.runtime.protocol.server` | `nexus.runtime.server.app` |
| `nexus.runtime.protocol.ags` | `nexus.runtime.providers.ags.*` |
| `nexus.runtime.services.*` | `nexus.runtime.server.services.*` |

通过 `ExecutionContext.get_runtime()` 获取 Runtime 的代码 **无需改动**。 `call_function` 已移除，统一使用 `invoke()`。 Tool 和 Executor 的注册方式不变（`@registry.register()`），注册元数据迁移到了 `pyproject.toml`。 完整的配置字段说明见 [运行配置 YAML 参考](/reference/config-yaml)。
