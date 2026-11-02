# 使用 AGS CodeInterpreter

## 概述

AGS CodeInterpreter 把腾讯云 AGS 的原生代码执行沙盒接进 Nexus，让 agent 在隔离环境里跑 Python / Shell。 本文介绍它的接入凭证（template / control_plane）、两套接口（`nexus` 的 `run_command`、`raw` 的原始 `/execute`）的配置方式和可用工具。 接口取舍的设计背景见 [AGS Runtime Provider 设计](../../explanation/ags-runtime-provider-design.md)。

## 前提条件

先在 AGS 平台创建好 code-interpreter 类型的工具，拿到对应的工具 ID。 两种模式都用这个 ID，区别只在认证方式：

- **template 模式** （推荐快速上手）：用腾讯云 AGS API Key 认证，工具 ID 填到 `template`。
- **control_plane 模式** ：用腾讯云 `secret_id` / `secret_key` 认证，工具 ID 填到 `tool_id`。

## 快速上手

以下配置使用 **template 模式 + nexus interface** ，返回 `AGSCodeInterpreterRuntime`，agent 可使用 `run_command` executor 执行 Shell 命令。

```yaml
runtime_provider_name: ags
runtime_provider_config:
  api_key: "${AGS_API_KEY}"
  domain: "ap-guangzhou.tencentags.com"
  interface_mode: nexus
  request_timeout: 1800.0

runtime_configs:
  main:
    # 声明使用原生 code-interpreter 服务。
    # template 模式下 provider 会自动把 runtime_kind=code 推断成 launch_mode=native，
    # 这里显式写出来更直观。
    launch_mode: native
    runtime_kind: code
    # AGS 上的 code-interpreter 模板名称
    template: "<your-code-interpreter-template>"
```

## 进阶配置

### 使用 raw interface（直接访问 `/execute` 端点）

当你需要直接发送 Python 代码（而非通过 `run_command` 包装成 Shell 调用）时，使用 `interface_mode: raw`，返回 `AGSRawCodeInterpreterRuntime`。

```yaml
runtime_provider_name: ags
runtime_provider_config:
  api_key: "${AGS_API_KEY}"
  domain: "ap-guangzhou.tencentags.com"
  interface_mode: raw
  request_timeout: 1800.0

runtime_configs:
  main:
    runtime_kind: code
    template: "<your-code-interpreter-template>"
```

拿到 `AGSRawCodeInterpreterRuntime` 实例后，可通过 `execute()`、`execute_raw()` 等接口直接发送代码到 `/execute` 端点。 具体用法见 [AGS Runtime Provider 设计](../../explanation/ags-runtime-provider-design.md)。

### 使用 control_plane 模式（内置 code-interpreter 工具）

不走 E2B 模板，而是用腾讯云控制面（control_plane）：用 `secret_id` / `secret_key` 认证，工具 ID 填到 `tool_id`，provider 按 ID 定位并复用该工具。 返回 `AGSCodeInterpreterRuntime`（nexus）或 `AGSRawCodeInterpreterRuntime`（raw），用法与 template 模式一致。

```yaml
runtime_provider_name: ags
runtime_provider_config:
  secret_id: "${AGS_SECRET_ID}"
  secret_key: "${AGS_SECRET_KEY}"
  region: ap-guangzhou
  interface_mode: nexus

runtime_configs:
  main:
    launch_mode: native
    runtime_kind: code
    tool_type: code-interpreter
    # AGS 平台上 code-interpreter 工具的 ID（与 template 模式同一个 ID）
    tool_id: "<your-code-interpreter-tool-id>"
```

> 注意：control_plane 模式下若不显式填 `launch_mode: native` 和 `tool_type: code-interpreter`，`runtime_kind=code` 会被默认推断成需要额外提供运行时镜像的完整 Nexus runtime 模式。

## 可用工具（nexus interface）

| Executor | 说明 |
|----------|------|
| `run_command` | 在沙盒中执行 Shell 命令，返回 stdout / stderr / exit_code |

## 示例脚本与运行效果

仓库提供了端到端示例脚本 [`docs/examples/scripts/test_ags_code_interpreter.py`](https://git.woa.com/) ，模拟一个真实的数据处理任务：在沙盒里生成一份销售 CSV、用 Python 按地区聚合销售额、写回结果文件再读出来验证。 nexus 和 raw 两种 interface 都覆盖（代码执行任务无 GUI，因此无截图，以输出为准）。

### nexus interface

```bash
uv run python docs/examples/scripts/test_ags_code_interpreter.py \
  --api-key demo-ags-key \
  --template <your-code-interpreter-template> \
  --domain ap-guangzhou.tencentags.com \
  --interface nexus
```

脚本用 `run_command` 跑 Python 数据处理脚本，再用 shell 独立读回结果文件。 实际输出：

```text
[1] data task status=CommandRunStatus.Finished exit=0
    stdout:
    Sales by region:
    East: 90
    North: 200
    South: 250
    TOTAL: 540
[2] result file content:
    Sales by region:
    East: 90
    North: 200
    South: 250
    TOTAL: 540
```

### raw interface

```bash
uv run python docs/examples/scripts/test_ags_code_interpreter.py \
  --api-key demo-ags-key \
  --template <your-code-interpreter-template> \
  --domain ap-guangzhou.tencentags.com \
  --interface raw
```

raw 模式直接把 Python 代码发给原始 `/execute`，返回事件流 JSON（`stdout` / `end_of_execution` 等），多次 `execute` 共享同一文件系统：

```text
[1] execute result: {"type": "stdout", "text": "Sales by region:\nEast: 90\nNorth: 200\nSouth: 250\nTOTAL: 540\n", ...}
[2] read-back via execute: {"type": "stdout", "text": "Sales by region:\n...", ...}
```

### control-plane 模式

脚本还支持 `--mode control-plane`，用腾讯云控制面复用一个已有内置 code-interpreter 工具（只需 secret + `--tool-id`，不需 template）。 数据处理任务流程与上面一致，nexus / raw 都支持：

```bash
uv run python docs/examples/scripts/test_ags_code_interpreter.py \
  --mode control-plane \
  --secret-id <SecretId> --secret-key <SecretKey> --tool-id <ToolId> \
  --region ap-guangzhou --interface nexus
```

provider 按 `--tool-id` 定位已有工具并启动实例，运行结束只停止实例、不删除工具。
