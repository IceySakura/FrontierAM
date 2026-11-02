# Worker Venv 配置指南

本文介绍 Worker Venv 的使用场景、配置方式和常见问题。

## 什么时候需要 Worker Venv

沙盒容器内的 Base 进程运行着一个完整的 Nexus Runtime——已经包含了所有 executor、tool 和 server 能力。 `worker_venv` 解决的不是能力缺失问题，而是 **代码版本差异** 问题。

### 默认行为：`runtime_image: current`

通常，`runtime_image` 会设为 `current`（或对应当前 Host nexus 版本的镜像标签），镜像内预装了与 Host 相同版本的 nexus 代码。 此时 Base 进程中的代码和 Host 侧一致，不需要 Worker Venv。

### 需要 Worker Venv 的场景

当你 **本地修改了 Runtime 侧运行的代码**，但没有（也不想）重新构建 Runtime 镜像时：

| 场景 | 说明 |
|------|------|
| **开发分支调试** | 你在 feature 分支上修改了 agent/tool/executor 的 runtime 侧代码，镜像还是 main 的版本 |
| **本地代码迭代** | 你改了几行 runtime_setup 或 tool 实现，想立刻看到效果，不想每次都重建镜像 |
| **跨版本测试** | 你想用旧镜像跑新代码，或者反过来 |

**简单判断**：如果你的 Host 侧 nexus 代码和 Runtime 镜像里的版本一致，不需要 Worker Venv。 如果你改了代码又不想重建镜像，就开启 `worker_venv: true`。

### 不需要 Worker Venv 的场景

- 使用 `runtime_image: current` 或版本号与 Host 一致的镜像，且没有修改任何代码
- 只使用内置 executor（bash、file、browser 等），不涉及自定义 runtime 侧逻辑

## 基本用法

在 `runtime_configs` 中设置 `worker_venv: true` 即可启用：

```yaml
runtime_provider_name: "docker"
runtime_configs:
  main:
    image: ubuntu-wksp:latest
    worker_venv: true
```

启用后，框架在 Runtime 初始化时会自动：

1. 将当前项目（Host 侧的代码）打包为 artifact（`.whl` 格式）
2. 上传 artifact 到沙盒
3. 在沙盒内创建隔离的 Worker Venv，安装你的最新代码
4. 启动 Worker 子进程
5. 后续 `invoke()` 调用路由到 Worker 执行——运行的是你本地的代码，而非镜像里的

## 配置示例

### 开发调试：本地代码 + 主线镜像

最典型的用法——你在 feature 分支上改了代码，想用 main 的镜像快速跑起来：

```yaml
task_name: "identity"
runtime_provider_name: "docker"
runtime_configs:
  main:
    image: ubuntu-wksp:latest
    worker_venv: true       # 把本地改动打包上传

agent_name: "main_agent"
agent_config:
  user_prompt: "修复这个 bug"
  tools:
    - swe_agent/bash
    - swe_agent/str_replace_editor
```

### 不需要调试：代码与镜像版本一致

```yaml
task_name: "identity"
runtime_provider_name: "docker"
runtime_configs:
  main:
    image: ubuntu-wksp:latest
    worker_venv: false      # 直接用镜像里的代码

agent_name: "main_agent"
agent_config:
  user_prompt: "列出当前目录文件"
  tools:
    - swe_agent/bash
```

### SWE-Bench 任务

```yaml
agent_name: swe_agent
agent_config:
  max_iterations: 15
runtime_provider_name: docker
runtime_configs:
  main:
    worker_venv: true
task_name: swe_task
task_config:
  run_oracle: false
  instance:
    # ... SWE-bench 实例数据
```

## 高级配置

`worker_venv` 也可以是一个对象，提供更细粒度的控制：

```yaml
runtime_configs:
  main:
    worker_venv:
      source: "."                    # 打包的源目录（默认当前项目）
      extras: ["runtime"]            # 安装的 extras
      extra_requirements:            # 额外安装的包
        - "some-package>=1.0"
      modules:                       # 要包含的 module 列表
        - "swe_agent"                # null = 全部，[] = 无，[...] = 指定
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `source` | `str` | `"."` | 要打包为 artifact 的源目录 |
| `extras` | `list[str]` | `[]` | 安装 artifact 时启用的 extras |
| `extra_requirements` | `list[str]` | `[]` | 额外安装的 pip 包 |
| `modules` | `list[str] \| null` | `null` | 模块列表：`null` = 包含所有模块的 runtime deps，`[]` = 不包含，`[...]` = 指定模块 |

## 常见问题

### Worker Venv 创建很慢？

Worker Venv 使用符号链接从 Base 环境复用大部分依赖，典型创建时间 1-3 秒。 如果很慢，可能原因：

- **缺失依赖需要从网络下载**：检查 Base 镜像是否包含了模块的 runtime deps
- **Artifact 过大**：检查 `exclude_patterns` 是否过滤了不必要的文件

### invoke 调用报 ModuleNotFoundError？

Worker Venv 中的 Python 环境是隔离的。 确保所需的包在你的 `pyproject.toml` 的 `dependencies` 或 `optional-dependencies.runtime` 中正确声明。 声明后，Worker Venv 创建时会自动安装——如果 Base 依赖池中有则通过符号链接秒级完成，没有则从网络下载。

### 何时需要重建 Runtime 镜像？

Worker Venv 的 artifact 会覆盖 Worker 进程中运行的所有代码（executor、tool、agent 等），所以大部分改动 **不需要重建镜像**。

唯一需要重建的是 **Base 进程中与 Worker 通信的部分**——这些代码跑在 Base 进程里，Worker Venv 的 artifact 覆盖不到：

- `server/app.py` 中 `/worker` 命名空间的协议处理逻辑
- `server/venv/manager.py` 的 Venv 创建流程
- `server/worker/launcher.py` 的 Worker 启动逻辑

其他改动都 **不需要重建**：

- 修改了 executor 代码 → Worker Venv artifact 自动带入
- 修改了 extension 代码（agents/tasks/tools）→ 同上
- 修改了 server 中非 Base↔Worker 通信的部分 → 同上

详见 [Runtime 镜像版本策略](../../explanation/runtime-image-versioning.md)。

### Worker Venv 和镜像重建的取舍？

- **快速迭代**：用 `worker_venv: true`，改代码后立刻生效，不需要重建镜像
- **正式发布**：升级版本号，重建镜像，用 `worker_venv: false`，获得最佳启动性能
- **大规模评测**：用正式版本镜像 + `worker_venv: false`，避免每个容器都做打包上传
