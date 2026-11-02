# Runtime 镜像版本策略

本文说明 Runtime 镜像的版本管理和发布规则。

## 核心原则

Runtime 镜像与 Host 之间没有向后兼容保证——能正常工作的前提是 Host↔Runtime Base 的通信协议一致。 建议所有用户使用 `runtime_image: current`（默认值，可以不填），它会自动解析到当前 Host nexus 版本对应的镜像。 没有理由使用不匹配的版本。

基于此前提，确立以下规则：

### 1. 不再维护 `latest` 标签

旧策略下，`latest` 标签会在每次 main 合并后自动更新，所有用户共享同一个 `latest` 镜像。 这是危险的——一次 Runtime 更新可能导致正在运行的任务与新镜像协议不匹配而失败。

**新策略**：每个镜像版本使用明确的版本号标签（如 `nexus-runtime:1.0.0`），不再自动更新 `latest`。 `runtime_image: current` 会解析到当前 Host 版本对应的标签。

### 2. main 分支仅在版本发布时推送镜像

main 分支的 CI 流水线 **只在版本号更新时** 构建并推送 Runtime 镜像，推送一次后同版本不再更新。

- 合并到 main 但未更新版本号 → **不触发镜像推送**
- 合并到 main 且更新了版本号 → 触发构建，推送 `nexus-runtime:<new_version>`
- 如果有 Runtime 侧内容更新，**必须升级版本号**

这确保了已发布镜像的不可变性——同一个版本号永远对应同一个镜像内容。

### 3. 非 main 分支的开发与测试

对于在非 main 分支上开发的场景（使用已发版的 `current` 镜像）：

| 场景 | 方式 | 说明 |
|------|------|------|
| **只改了 Host 侧代码** | 不用额外操作 | Runtime 镜像内的代码不受影响，直接使用 `current` |
| **改了 Runtime 侧代码** | `worker_venv: true` | 本地代码自动打包上传到沙盒，在 Worker Venv 中运行你的最新代码 |
| **改了 Runtime 侧代码，需要较大规模使用** | COS 中转（WIP） | 避免每个沙盒都从 Host 直连下载 artifact，提升大并发场景的性能和稳定性 |

**关于大规模并发**：`worker_venv: true` 下，每个沙盒启动时都会从 Host 下载 artifact（nexus whl 约 2MB）。 对于 RL 等对稳定性和性能要求很高的场景，并发 >5000 时 Host 侧扇出的网络压力和沙盒侧下载延迟可能影响整体性能。 COS 中转方案通过将 artifact 上传到对象存储来避免 Host 扇出瓶颈（该方案目前 WIP）。

## 配置示例

### 使用 `current`（推荐，默认行为）

`runtime_image` 默认值为 `current`，自动解析到当前 Host nexus 版本对应的镜像。 可以不写：

```yaml
runtime_provider_name: "docker"
runtime_configs:
  main:
    image: ubuntu-wksp:latest
    # runtime_image: current   # 默认值，可以省略
```

### 使用 Worker Venv（开发调试）

本地修改了 Runtime 侧代码时，通过 `worker_venv: true` 将改动注入沙盒：

```yaml
runtime_provider_name: "docker"
runtime_configs:
  main:
    image: ubuntu-wksp:latest
    # runtime_image: current   # 默认值，可以省略
    worker_venv: true           # 打包本地代码到沙盒运行
```

这种方式下，Base 进程仍使用镜像里的代码负责通信和调度，而你修改的业务代码通过 artifact 在 Worker 进程中运行。

## 版本升级 Checklist

当需要发布新版本时：

1. 确认 `pyproject.toml` 中的 `version` 已更新
2. 确认所有 Runtime 侧变更已包含在分支中
3. 合并到 main → CI 自动构建并推送镜像
4. 使用 `current` 的用户升级 nexus 包版本后自动匹配新镜像

## 与 Worker Venv 的关系

Worker Venv 机制（见 [Worker Venv 机制](worker-venv.md)）进一步降低了镜像更新频率：

- **Base 镜像** 提供稳定的依赖池和 Base↔Worker 通信层
- **Artifact** 通过 Worker Venv 动态注入业务代码，不需要重建镜像
- 只有当 Base 进程中与 Worker 通信的部分（`app.py` 的 `/worker` 协议、`venv/manager.py`、`worker/launcher.py`）发生变更时，才需要更新镜像版本
