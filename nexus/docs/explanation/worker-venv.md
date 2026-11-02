# Worker Venv 机制

本文详细介绍 Nexus Runtime 中的 Worker Venv（工作虚拟环境）机制——一种在沙盒容器内按需创建隔离 Python 环境的方案。

## 为什么需要 Worker Venv

在 Nexus 的 Runtime 架构中，沙盒容器内运行着一个 Base 进程（FastAPI + Socket.IO server），内含完整的 Nexus Runtime 能力。 所有的 `invoke()` 调用默认在 Base 进程中执行，Base 进程的代码来自 Runtime 镜像。

这在镜像版本与 Host 侧代码一致时没有问题。 但在以下场景中，用户的本地代码与镜像中的代码存在差异：

1. **开发分支调试**：用户在 feature 分支上修改了 Runtime 侧运行的代码（agent/tool/executor），但 Runtime 镜像仍是 main 版本——镜像中的代码不包含用户的改动
2. **快速迭代**：用户改了几行代码想立刻看效果，每次重建镜像成本过高
3. **跨版本测试**：用旧镜像跑新代码，或反过来

Worker Venv 的核心思路是：将用户本地的最新代码打包为 artifact，上传到沙盒，在 Base 进程之外创建一个包含用户代码的独立 Python 虚拟环境，并启动 Worker 子进程来执行——这样运行的就是用户改过的代码，而非镜像里的旧版本。

## 架构总览

```
Host (orchestrator)                    Sandbox (container)
┌─────────────────┐                   ┌──────────────────────────────┐
│  RemoteRuntime   │───HTTP/SIO──────>│  Base Process (app.py)       │
│                  │                   │  ├─ FastAPI server           │
│  invoke(target,  │                   │  ├─ VenvManager              │
│    venv_id=...)  │                   │  ├─ WorkerLauncher           │
│                  │                   │  └─ Executor backends        │
└─────────────────┘                   │         │                    │
                                      │         │ SIO /worker ns     │
                                      │         ▼                    │
                                      │  ┌─────────────────┐        │
                                      │  │ Worker Process   │        │
                                      │  │ (worker/sdk.py)  │        │
                                      │  │ ├─ Own venv      │        │
                                      │  │ ├─ SIO client    │        │
                                      │  │ └─ Executor call │        │
                                      │  └─────────────────┘        │
                                      └──────────────────────────────┘
```

## 创建流程

当 Host 侧调用 `runtime.create_venv(spec)` 时，沙盒内会依次执行以下步骤：

### 1. 创建裸 Python 环境

使用 `uv venv` 创建一个只包含 Python 解释器的空虚拟环境：

```
/tmp/nexus/venvs/<venv_id>/
├── bin/
│   ├── python -> /usr/bin/python3.x
│   └── ...
└── lib/python3.x/site-packages/   (空)
```

### 2. 物化 Artifact Wheel

将 Host 上传的 artifact（`.whl` 格式）物化到磁盘。 Artifact 可以来自：

- **内联数据**：小型 artifact 直接通过 HTTP body 传输
- **URL 引用**：大型 artifact 通过 URL 下载（如 COS 中转）

### 3. BFS 闭包符号链接

这是 Worker Venv 的核心优化。 与其在每个 venv 中安装所有依赖，不如从 Base 环境中 **符号链接** 已有的包：

```python
# 伪代码：BFS 闭包算法
def symlink_from_base(venv_dir, artifact_deps, base_index):
    seeds = artifact_deps + module_runtime_deps
    closure, missing = bfs_transitive_closure(base_index, seeds)

    for pkg in closure:
        # 符号链接 dist-info 目录（COPY，因为 uv 忽略 symlinked dist-info）
        copy(base_sp / pkg.dist_info, venv_sp / pkg.dist_info)
        # 符号链接包目录
        for top_level in pkg.top_level_names:
            symlink(base_sp / top_level, venv_sp / top_level)

    return missing  # 这些需要从网络安装
```

**为什么用 BFS 传递闭包**：一个包 A 依赖 B，B 又依赖 C。 只链接 A 和 B 是不够的——必须把整棵依赖树都链接过来。 BFS 从种子包出发，沿 `Requires-Dist` 边遍历 Base 的包索引，收集所有传递依赖。

**为什么 COPY dist-info**：`uv` 在做依赖解析时会忽略符号链接的 `.dist-info` 目录，所以必须把 `.dist-info` 复制过去，而包的实际代码目录用符号链接即可。

### 4. 安装 Artifact Wheel

根据 BFS 闭包的结果决定安装策略：

- **依赖完全覆盖**（`artifact_missing` 为空）：走 `--no-deps` 快速路径，跳过依赖解析，只安装 artifact 本身的代码
- **存在缺失依赖**：走完整依赖解析，uv 联网下载 Base 中没有的包

```bash
# 快速路径（依赖已全部符号链接）
uv pip install <artifact.whl>[extras] --no-deps

# 完整解析（有缺失依赖时自动 fallback）
uv pip install <artifact.whl>[extras]
```

### 5. 安装额外依赖

如果模块的 runtime deps 中有 Base 不包含的包（BFS 闭包返回的 `module_missing`），或者用户指定了 `extra_requirements`，则通过网络安装：

```bash
uv pip install <module_missing_packages> <extra_requirements>
```

### 6. 启动 Worker 子进程

Venv 创建完成后，`WorkerLauncher` 启动一个子进程：

```bash
<venv>/bin/python -m nexus.runtime.server.worker.sdk <base_url> <venv_id> <worker_id>
```

Worker 通过 Socket.IO `/worker` 命名空间连接到 Base 进程，准备接收 `invoke` 请求。

## `worker_venv: true` vs `worker_venv: false`

在 YAML 配置中，`worker_venv` 控制是否自动创建 Worker Venv：

```yaml
runtime_configs:
  main:
    image: ubuntu-wksp:latest
    worker_venv: true   # 自动创建 Worker Venv
```

| 配置 | 行为 | 适用场景 |
|------|------|---------|
| `worker_venv: true` | Runtime 初始化时自动打包当前项目为 artifact，上传到沙盒，创建 Worker Venv | 本地代码与 Runtime 镜像中的版本不一致（如开发分支调试、代码迭代） |
| `worker_venv: false` | 所有 `invoke()` 调用在 Base 进程中执行，使用镜像内预装的代码 | 代码与镜像版本一致，无需覆盖 |
| 未设置 | 等同于 `false` | — |

## Artifact 打包与缓存

当 `worker_venv: true` 时，Host 侧会自动执行 artifact 打包（`packaging.py`）：

1. **内容哈希**：对项目源码和 `pyproject.toml` 计算哈希
2. **三层缓存**：
   - 进程级内存缓存（同进程多次调用）
   - 进程内合并（并发请求共享同一个打包任务）
   - 磁盘缓存 + 文件锁（跨进程共享）
3. **打包方式**：
   - 已安装包：从 site-packages 重新打包
   - 有 `pyproject.toml`：标准 wheel build
   - 纯目录：打包为 zip wheel

## Worker 通信模型

Worker 通过 Socket.IO `/worker` 命名空间与 Base 通信：

```
Host ──invoke(venv_id=X)──> Base ──SIO emit('invoke')──> Worker
                             Base <──SIO emit('invoke:result')── Worker
Host <──HTTP response────── Base
```

- **日志转发**：Worker 的 Python 日志通过 SIO 转发到 Base，再传给 Host
- **Span 转发**：Worker 的 Nexus `SpanEvent` 先进入 Base broadcaster，再批量转发到 Host `SpanController`
- **事件中继**：Worker 发出的自定义事件自动通过 Base 中继到 Host

Worker 在返回 invoke/execute 结果前 flush 自己已接收的 span events。managed runner 结束时还会发起
`trace:flush` barrier，等待 Worker flush、Base 有序 marker 和 Host replay 全部完成，再写 `run.output` 并释放 runtime。
这要求 Host、Base runtime 和 Worker SDK 使用兼容版本。

## 性能特征

- **Venv 创建**：符号链接方案使大部分依赖无需下载/安装，典型创建时间 ~1-3 秒
- **首次 invoke**：包含 Worker 启动和 SIO 连接建立，额外 ~1-2 秒
- **后续 invoke**：Worker 已就绪，延迟接近 Base 进程直接执行

## 与依赖分层的关系

Worker Venv 方案直接利用了模块依赖的三层模型：

- **Base 镜像**（构建时 `nexus modules install --all --runtime`）提供 **依赖池**
- **Worker Venv**（运行时动态创建）从依赖池中按需符号链接
- 模块声明的 `runtime` extras 约束了 Base 依赖池大小——声明越全，Worker 能链接的越多，需要现场下载的越少

详见 [Module 元数据格式](module-metadata.md)。
