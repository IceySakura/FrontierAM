# WAA 任务说明

## 任务概览

- 任务名：`waa`，用于在 Windows 虚拟机上执行 GUI 自动化任务并评测。
- 关键组件：通过 AGS 云沙箱拉起 Windows VM 实例，使用 Navi Agent 执行屏幕操作（observe → plan → execute 循环），随后运行评测 pipeline（setup → getter → metric）产出评分。
- Agent：使用 `navi_agent`，通过 function calling 调用 4 个 tool（`execute_code`、`wait`、`done`、`fail`）。
- 主要文件：`workspace/waa/test.yaml`（任务配置）、`nexus/extensions/tasks/waa/task.py`（任务逻辑）、`nexus/extensions/agents/navi/agent.py`（Agent 逻辑）。

## 任务配置字段

### task_config

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `task_config` | WAA benchmark 任务的完整 JSON 配置（含 instruction、config、evaluator） | 必填 |
| `skip_setup` | 跳过环境 setup 阶段（调试用） | `false` |
| `skip_cleanup` | 跳过任务前的 cleanup（关闭所有应用） | `false` |

### agent_config

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `user_prompt` | 任务指令（单任务测试时必填，批量评测由 task 自动传入） | — |
| `max_iterations` | Agent 最大迭代次数 | `15` |
| `sleep_after_execution` | 每步执行后等待时间（秒） | `0.5` |
| `keep_last_n_previous_actions` | prompt 中保留的历史动作数 | `15` |
| `tool_call_retries` | LLM 未产生有效 tool call 时的重试次数 | `2` |
| `observe_retries` | 截屏/a11y 采集失败时的重试次数 | `3` |

### runtime_provider_config（AGS）

| 字段 | 说明 | 必填 |
|------|------|------|
| `secret_id` | AGS API Secret ID | 是 |
| `secret_key` | AGS API Secret Key | 是 |
| `region` | AGS 区域（如 `ap-nanjing`） | 是 |
| `interface_mode` | 接口模式（如 `raw`） | 是 |
| `request_timeout` | HTTP 请求超时（秒） | 否（默认 180） |
| `instance_ready_timeout` | 实例就绪等待超时（秒） | 否（默认 300） |

### runtime_configs

```yaml
runtime_configs:
  main:
    tool_id: sdt-gprl2m4v         # AGS sandbox tool ID
    runtime_kind: waa             # 运行时类型
    launch_mode: native           # 启动模式
    interface_mode: raw           # 接口模式
    provision_mode: control_plane # 资源管理模式
```

## 单任务运行示例（测试用）

单任务模式用于调试 agent 行为，不运行评测 pipeline，只执行 agent 并观察结果。

编辑 `workspace/waa/test.yaml`：

```yaml
task_name: identity
task_config: {}

agent_name: navi_agent
agent_config:
  user_prompt: "打开记事本，输入 hello world，保存到桌面"
  max_iterations: 15
  sleep_after_execution: 0.5

runtime_provider_name: ags
runtime_provider_config:
  secret_id: "<your-secret-id>"
  secret_key: "<your-secret-key>"
  region: ap-nanjing
  interface_mode: "raw"
  request_timeout: 180.0
  instance_ready_timeout: 300.0

runtime_configs:
  main:
    tool_id: <tool-id>  # eg: sdt-gprl2m4v
    runtime_kind: waa
    launch_mode: native
    interface_mode: raw
    provision_mode: control_plane

openai_clients:
  main:
    name: "litellm"
    config:
      client_args:
        api_key: "<your-api-key>"
        base_url: "<your-base-url>"
      request_args:
        model: "anthropic.claude-sonnet-4-6"
```

运行：

```bash
nexus run-single workspace/waa/test.yaml -o result.json
```

## 批量评测（Batch Mode）

对于 WAA benchmark 评测，使用 `nexus run-batch` 配合 generator 批量运行任务。Generator 位于 `nexus/extensions/tasks/waa/generator.py`。

### 环境变量

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| **任务筛选** | | |
| `WAA_TASK_DIR` | evaluation_examples_windows 目录路径 | 必填 |
| `WAA_DOMAIN` | 领域过滤（逗号分隔或 `all`） | `all` |
| `WAA_TASK_IDS` | 指定 task ID（逗号分隔） | 全部 |
| `WAA_SKIP` | 跳过前 N 个任务 | `0` |
| `WAA_LIMIT` | 最多产出 N 个任务（不设置=无限） | 无限 |
| `WAA_DIFF_LVL` | 难度级别：`normal` 或 `hard` | `hard` |
| **Agent** | | |
| `WAA_MAX_STEPS` | Agent 最大步数 | `15` |
| `WAA_SLEEP_AFTER_EXECUTION` | 每步等待时间（秒） | `0.5` |
| `WAA_KEEP_LAST_N_ACTIONS` | 保留历史动作数 | `15` |
| `WAA_TOOL_CALL_RETRIES` | tool call 重试次数 | `2` |
| **Runtime (AGS)** | | |
| `WAA_SECRET_ID` | AGS Secret ID | 必填 |
| `WAA_SECRET_KEY` | AGS Secret Key | 必填 |
| `WAA_TOOL_ID` | AGS Sandbox Tool ID | 必填 |
| `WAA_REGION` | AGS 区域 | `ap-nanjing` |
| **LLM** | | |
| `OPENAI_CLIENT_NAME` | 客户端名称 | 必填 |
| `OPENAI_API_KEY` | API Key | 必填 |
| `OPENAI_BASE_URL` | API Base URL | 必填 |
| `OPENAI_MODEL` | 模型名称 | 必填 |

### 示例命令

**评测 Chrome 领域任务：**

```bash
WAA_TASK_DIR=/path/to/evaluation_examples_windows \
WAA_DOMAIN=chrome \
WAA_DIFF_LVL=hard \
WAA_SECRET_ID=<your-secret-id> \
WAA_SECRET_KEY=<your-secret-key> \
WAA_TOOL_ID=<your-tool-id> \
WAA_REGION=ap-nanjing \
OPENAI_CLIENT_NAME=litellm \
OPENAI_API_KEY=<your-api-key> \
OPENAI_BASE_URL=<your-base-url> \
OPENAI_MODEL=anthropic.claude-sonnet-4-6 \
nexus run-batch \
  -g nexus/extensions/tasks/waa/generator.py \
  -d results.db \
  --max-concurrent 5
```

**评测全部领域，跳过前 10 条，最多运行 50 条：**

```bash
WAA_TASK_DIR=/path/to/evaluation_examples_windows \
WAA_DOMAIN=all \
WAA_SKIP=10 \
WAA_LIMIT=50 \
# ... 其他配置同上
nexus run-batch \
  -g nexus/extensions/tasks/waa/generator.py \
  -d results.db \
  --max-concurrent 5
```

### 结果存储、恢复运行、重试失败任务

参见 [run-batch.md](../run-batch.md)。

## 调试

### VNC 查看桌面

WAA 实例暴露 VNC 端口（默认 8006）。runtime 初始化时会打印 VNC 地址和 token：

```
AGSRawWindowsAgentArenaRuntime initialized with vnc_url: https://8006-pm6zk3x4jb67p3xgi4sa3xiveklt6wl724z6w5lz.ap-nanjing.tencentags.com token: sit__ZX97k1BhoQ96B4T0jQf5zxgZsJuqjhkyV17zMVOGPM
```

访问 VNC 需要添加 `X-Access-Token` header。推荐使用 [ModHeader](https://chromewebstore.google.com/detail/modheader) Chrome 插件，配置 `X-Access-Token: <access-token>` 后在浏览器中打开 VNC URL 即可实时查看 Windows 桌面。

### 日志

关键日志 logger：

- `nexus.extensions.agents.navi.agent` — agent 循环状态、tool call 信息
- `nexus.extensions.agents.navi.tools` — tool 执行结果
- `nexus.runtime.providers.ags` — runtime 连接/请求
