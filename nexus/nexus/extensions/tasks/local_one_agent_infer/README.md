# local_one_agent_infer

本地模式 OneAgentBench 推理/评测 task 。 驱动 `local_one_agent`（`LocalToolExecutor` 直接在用户本地工作区执行工具，无 Docker 沙箱、无 runtime HTTP），从 benchmark JSONL 逐 session 生成配置并执行，支持单轮与多轮 session 。

## 运行时配置：jsonl 内配置 `local_config`

`config/base.yaml` 是模板配置，其中运行时环境相关字段默认通过 `${oc.env:...}` 提供「默认值 + 环境变量覆盖」。 若想**逐 item 精确控制运行时环境**，可在 benchmark JSONL 的 record 里加一个可选的 `local_config` 对象，由 generator 逐 session 叠加覆盖，无需再依赖环境变量。

### 取值规则

- **取值来源** ：session 级配置统一取该 session **首条 record** 的 `local_config`（同一 session 共享同一 workspace / 权限）。
- **优先级** ：**record `local_config` > 环境变量 > `base.yaml` 默认**。 base config 在加载时以 `resolve=True` 把环境变量/默认值填好，record 里显式给出的字段再叠加覆盖。
- **向后兼容** ：只覆盖 record 里**显式给出**的字段；不带 `local_config` 的存量 JSONL 行为完全不变。

### 支持字段

| 字段 | 类型 | 说明 | 对应环境变量 | 覆盖目标 |
|---|---|---|---|---|
| `workspace_dir` | string | 本地工作区绝对路径（输入文件铺进这里、产物从这里读回） | `LOCAL_WORKSPACE_DIR` | `local_context.workspace.path` |
| `permission_level` | string | 权限级别，`readonly` / `basic` | `LOCAL_PERMISSION_LEVEL` | `local_context.permission.level` |
| `runtime.os` | string | 操作系统，如 `macos` / `windows` | `LOCAL_RUNTIME_OS` | `local_context.runtime.os` |
| `runtime.arch` | string | 架构，如 `arm64` / `x64` | `LOCAL_RUNTIME_ARCH` | `local_context.runtime.arch` |
| `runtime.shell` | string | Shell，如 `zsh` / `powershell` | `LOCAL_RUNTIME_SHELL` | `local_context.runtime.shell` |
| `env_mode` | string | 环境预置模式，`cold` / `warm` | `LOCAL_ENV_MODE` | `local_context.env.mode` |
| `max_iterations` | int | agent 最大迭代轮数 | `LOCAL_MAX_ITERATIONS` | `agent_config.max_iterations` |

`runtime` 为嵌套对象，`os` / `arch` / `shell` 三个子字段各自独立、可只给其中一部分；未给出的子字段保持 base config 里已 resolve 的值。

### 示例

在每个 session 的首条 record 里加入 `local_config` ：

```jsonc
{
  "session_id": "sess-001",
  "task_id": "Doc-Synthesis-012",
  "turn": 1,
  "query": "……",
  "input_file_list": ["files/a.docx"],
  "local_config": {
    "workspace_dir": "/workspace",
    "permission_level": "readonly",
    "runtime": {"os": "windows", "arch": "x64", "shell": "powershell"},
    "env_mode": "warm",
    "max_iterations": 30
  }
}
```

`local_config` 全部字段均为可选：只需覆盖某一项时，只写该项即可，其余仍走环境变量或 `base.yaml` 默认。 例如只改权限级别：

```jsonc
{"session_id": "sess-002", "task_id": "…", "turn": 1, "query": "…", "local_config": {"permission_level": "basic"}}
```

## 环境变量一览

除 `local_config` 可覆盖的运行时字段外，以下环境变量控制 benchmark 的加载（属于全局配置，不适合逐 item）：

| 环境变量 | 说明 |
|---|---|
| `LOCAL_BENCH_PATH` | benchmark 根目录（含 jsonl 与 files/） |
| `LOCAL_BENCH_JSONL` | benchmark jsonl 路径（可选，覆盖默认；默认取根目录下唯一的 `*.jsonl`） |
| `LOCAL_BENCH_CONFIG` / `YB_BENCH_CONFIG` | base config YAML 完整路径（可选） |
| `LOCAL_BENCH_CONFIG_NAME` / `YB_BENCH_CONFIG_NAME` | `config/` 下文件名（可选，如 `ags_base.yaml`；未设完整路径时生效，默认 `base.yaml`） |
| `LOCAL_BENCH_LIMIT` | 只取前 N 个 session（可选） |

### AGS 远端模式

当 base config 顶层配置了 `runtime_provider_name: ags`（见 `config/ags_base.yaml`）时，
generator 会把 `task_name` 路由到 `local_one_agent_infer_ags`（`ags_task.py`）：

1. 初始化 `ctx.get_runtime('main')`，工具在 AGS/WAA 远端 VM 执行；
2. 输入文件经 `WaaBackend.upload_file` 上传到 VM workspace（不再本地 copy）。

选用 AGS 配置示例：

```bash
export YB_BENCH_CONFIG_NAME=ags_base.yaml
# 或：export LOCAL_BENCH_CONFIG_NAME=ags_base.yaml
```
