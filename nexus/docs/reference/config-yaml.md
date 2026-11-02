# 运行配置 YAML 参考

本文列出 Nexus `run-single` 配置文件的所有字段。

## 完整结构

```yaml
# 任务配置
task_name: "echo"                    # 必填：注册的 task 名称
task_config: {}                      # 可选：传递给 task 的参数

# Runner 行为
runner:
  trace_drain_error_policy: "warn"  # 可选：warn（默认）或 raise

# Agent 配置
agent_name: "main_agent"             # 可选：注册的 agent 名称
agent_config: {}                     # 可选：传递给 agent 的参数

# Runtime Provider 配置
runtime_provider_name: "docker"      # 必填：provider 类型
runtime_provider_config: {}          # 可选：provider 初始化参数
runtime_configs:                     # 可选：各 runtime 实例的预配置
  main: {}

# LLM 客户端
openai_clients:                      # 可选：一个或多个 LLM 客户端
  main:
    name: "openai"
    config: {}

# 数据处理器
processors:                          # 可选：span 处理器
  chat:
    name: "chat_tracker"
```

## 字段详解

### `task_name`

**类型**：`str`，**必填**

注册在 `task_registry` 中的 task 名称。 常用值：

| 值 | 说明 |
|---|---|
| `echo` | 内置示例 task |
| `identity` | 透传 task，直接运行 agent |
| `swe_task` | SWE-Bench 评测 |
| `math` | 数学推理 |
| `gaia_task` | GAIA 评测 |
| `webarena_task` | WebArena 评测 |

### `task_config`

**类型**：`dict`，**可选**

传递给 task 函数的 `TaskRunInput` 参数。 不同 task 有不同的配置字段，参考各 task 的文档。

### `runner.trace_drain_error_policy`

**类型**：`"raise" | "warn"`，**默认**：`"warn"`

控制 Cloud Single（含 Batch 子任务）和 Live 收尾时 `Runtime.drain_trace()` 的异常处理。

```yaml
runner:
  trace_drain_error_policy: "warn"
```

- `raise`：显式配置后启用严格行为，轨迹排空失败会使收尾 hook 失败。
- `warn`（默认）：继续尝试排空轨迹；失败时记录包含 run ID、runtime 名称、异常类型和堆栈的 warning，
  继续收尾及保存业务输出，任务状态按业务执行结果判定。此时轨迹可能不完整。

该配置不跳过轨迹排空，也不缩短其超时时间。只降级 `drain_trace()` 抛出的普通异常；任务失败、
取消信号、Trace sink 写入失败和结果持久化失败仍按原有方式处理。Batch 需将该字段放入每个子任务的
RunInput 配置中。普通本地 `run-single` 的 context 清理本来就记录 drain 异常，不由此选项改变。

### `agent_name`

**类型**：`str`，**可选**

注册在 `agent_registry` 中的 agent 名称。 常用值：

| 值 | 说明 |
|---|---|
| `main_agent` | 标准 ReAct Agent |
| `swe_agent` | SWE-Agent |
| `codeact` | CodeAct Agent |
| `claude_code` | Claude Code（黑盒） |
| `codex` | OpenAI Codex CLI |
| `demo_agent` | 内置示例 agent |

### `agent_config`

**类型**：`dict`，**可选**

传递给 agent 函数的 `AgentRunInput` 参数。 常用字段（以 `main_agent` 为例）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_prompt` | `str` | 用户输入 |
| `system_prompt` | `str` | 系统提示词 |
| `max_iterations` | `int` | 最大迭代次数 |
| `tools` | `list[str]` | 工具列表，如 `["swe_agent/bash"]` |
| `parallel_tool_calls` | `bool` | 是否并行执行工具 |

系统提示词配置 Agent 的通用行为，任务输入保存本题内容和已有对话。四种 Agent 的字段位置如下：

| Agent | 任务输入 | 额外系统提示词 |
| --- | --- | --- |
| `claude_code` | `user_prompt` | `extra_cli_args` 中的 CLI 系统提示词参数 |
| `hycode` | `user_prompt` | `extra_cli_args` 中的 CLI 系统提示词参数 |
| `single_turn` | `prompt` 消息列表 | `system_prompt` |
| `main_agent` | `history` 与本轮 `user_prompt` | `system_prompt` |

`main_agent.history` 填此前的 user、assistant、tool 消息，系统提示词通过 `system_prompt` 提供。

使用 `single_turn` 时，`prompt` 填完整的消息列表，可以包含 system 消息和多轮历史。
可选的 `system_prompt` 会作为一条 system 消息前置，原有消息及顺序全部保留。
省略或填写 `null` 时，直接使用原始 `prompt`。

```yaml
agent_name: single_turn
agent_config:
  system_prompt: "请简洁回答"        # 可选：额外的系统提示词
  prompt:
    - role: system
      content: "数据中的系统提示词"
    - role: user
      content: "第一个问题"
    - role: assistant
      content: "此前的回答"
    - role: user
      content: "继续回答"
```

### `runtime_provider_name`

**类型**：`str`，**必填**

Runtime Provider 类型。 可选值：

| 值 | 说明 |
|---|---|
| `docker` | 本地 Docker 容器 |
| `gongfeng` | 腾讯内部云沙箱 |
| `taiji` | GPU 沙盒：底层机器和卡走太极，启动后复用 Gongfeng 链路。见 [使用 GPU 沙盒](../how-to/use-runtime/use-gpu.md) |
| `local` | 本地进程（无隔离） |
| `ags` | 腾讯云 AGS |
| `android_websocket` | Android WebSocket 沙箱 |
| `android_cloudtest` | Android CloudTest 真机 |

### `runtime_provider_config`

**类型**：`dict`，**可选**

传递给 RuntimeProvider `__init__` 的参数。 参考 [配置 Runtime](../how-to/use-runtime/configure-runtime.md)。

### `runtime_configs`

**类型**：`dict[str, dict]`，**可选**

各 runtime 实例的预配置。 key 是 runtime 名称（如 `main`、`eval`），value 是传递给 `provider.configure()` 的参数。

```yaml
runtime_configs:
  main:
    image: "ubuntu:24.04"           # 容器镜像
    worker_venv: true               # 是否创建 Worker Venv
  eval:
    image: "python:3.11"            # 评测用的独立 runtime
```

#### `worker_venv` 字段

| 值 | 类型 | 说明 |
|---|---|---|
| `true` | `bool` | 自动打包项目并创建 Worker Venv |
| `false` | `bool` | 所有调用在 Base 进程执行 |
| `{...}` | `dict` | 详细配置（source、extras、extra_requirements、modules） |

详见 [Worker Venv 配置指南](../how-to/use-runtime/worker-venv-config.md)。

### `openai_clients`

**类型**：`dict[str, ClientConfig]`，**可选**

LLM 客户端配置。 key 是客户端名称（通常 `main` 是默认客户端），value 结构如下：

```yaml
openai_clients:
  main:
    name: "openai"                  # 注册的客户端类型
    config:
      client_args:                  # 传递给 OpenAI/AsyncOpenAI 构造函数
        api_key: "sk-xxx"
        base_url: "https://api.openai.com/v1"
      request_args:                 # 传递给 create_completion 的默认参数
        model: "gpt-4o"
        temperature: 0.7
      enforced_request_args:        # 可选：该 client 实例所有请求的最终覆盖值
        model: "gpt-4o"
        max_completion_tokens: 4096
        extra_body:
          vendor_mode: "strict"
```

`enforced_request_args` 默认为空。普通字段在 adapter 默认值、`request_args`、旧 override 和 per-call kwargs
之后覆盖；同时会从更早的 `extra_body` 删除同名字段，避免 OpenAI SDK 的最终 body merge 反向覆盖策略。
OpenAI-compatible provider 的非标准字段应放在 `enforced_request_args.extra_body`，该 dict 会浅合并到已有
`extra_body` 的最后一层。所有 dict/list 都是整体替换或浅合并，不做递归合并；`null` 保留其原始含义。

该配置按 client 实例隔离，不使用进程级全局状态。agent 和 compaction 读取 model、输出预算或 sampling
参数时会使用 enforced 后的 effective request args。互斥或废弃参数仍由对应 SDK/provider 校验。

可用的 `name` 值：

| 值 | 说明 |
|---|---|
| `openai` | 标准 OpenAI 客户端 |
| `azure_openai` | Azure OpenAI |
| `litellm` | LiteLLM 代理 |
| `hytuner` | HyTuner RL 训练客户端 |
| `custom_function` | 自定义函数客户端 |
| `mock` | Mock 客户端（测试用） |
| `taiji` | 太极客户端 |
| `infer_proxy` | 推理代理客户端 |

本地运行潮汐模型时，可以开启自动调速：模型繁忙或额度不足时减少请求或等待，恢复后自动继续。

先确保任务的主模型（`openai_clients.main`）使用 `infer_proxy`，已配置好模型名称和 API Key，能够正常调用。
然后在运行命令中加上[本地潮汐配置示例](../examples/configs/infer_proxy_local_policy.yaml)：

```bash
# 单个任务：把 task.yaml 换成自己的任务配置
uv run nexus run single \
  --use ./task.yaml \
  --use docs/examples/configs/infer_proxy_local_policy.yaml

# 批量任务：把 generator.py 换成自己的任务生成器
uv run nexus run batch --generator ./generator.py \
  --use docs/examples/configs/infer_proxy_local_policy.yaml \
  --max-concurrent 10
```

示例已设置 **600 QPM**，即每分钟最多发起 600 次模型请求，直接 `--use` 即可。
模型名称、API Key 和服务地址沿用原配置。上面的批量命令中，10 个并发任务共用这 600 QPM；
模型繁忙时，实际请求速度会更低。

按分配给这次运行的额度填写 QPM。只有需要修改示例中的 600 时，才加 `--set`，例如改为 300：

```bash
uv run nexus run single \
  --use ./task.yaml \
  --use docs/examples/configs/infer_proxy_local_policy.yaml \
  --set openai_clients.main.config.client_args.local_request_policy.quota_qpm=300
```

同一个 Key 同时跑多组任务时，要分别分配额度，合计不超过该 Key 在此模型上的总额度。
使用 `--worker-processes` 启动多个 worker 时，每个 worker 都会使用所填额度：
例如总额度为 600、启动 2 个 worker，则应填写 300。

云端任务已配置调速时，沿用云端设置。

### `processors`

**类型**：`dict[str, ProcessorConfig]`，**可选**

Span 处理器配置。 key 是处理器名称（任意字符串），value 结构：

```yaml
processors:
  chat:
    name: "chat_tracker"            # 注册的处理器类型
    args: {}                        # 可选：传递给处理器的参数
```

可用的 `name` 值：

| 值 | 说明 |
|---|---|
| `chat_tracker` | 提取 LLM 对话历史 |
| `langfuse_tracker` | 实时上传到 [Langfuse](../how-to/use-langfuse-tracker.md)（需 `langfuse>=4.0.0`） |
| `chrome_trace` | Chrome Trace 格式（chrome://tracing 可视化） |
| `json_line` | JSONL 格式追踪日志 |
| `print` | 打印到控制台 |
| `token_counter` | Token 计数 |
| `hytuner` | HyTuner RL 训练数据 |
| `main_agent_metrics` | Main Agent 运行指标 |

这些 processor 默认消费 `default` scope，并在 task 结束时把 `get_result()` 汇总到
`RunOutput.processor_results`。它们不负责 distributed live/single 的 raw journal，也不能通过本字段选择 run trace
backend。后者由 InstanceManager 注册；云端用户通过 submission 顶层 `run_trace_backends.sinks` 或 CLI
`--run-trace-sink` 选择已授权名称，内部解析结果才会
写入 execution `driver_config`。完整职责划分见 [Trace 系统](../explanation/trace-system.md)。

## 完整示例

### Main Agent + Docker + SWE 工具

```yaml
task_name: "identity"

runtime_provider_name: "docker"
runtime_configs:
  main:
    image: ubuntu-wksp:latest
    worker_venv: true

agent_name: "main_agent"
agent_config:
  user_prompt: "汇报一下当前机器的系统环境"
  tools:
    - swe_agent/bash

openai_clients:
  main:
    name: "litellm"
    config:
      client_args:
        api_key: "your-api-key"
        base_url: "http://your-gateway:35000"
      request_args:
        model: "anthropic.claude-sonnet-4-6"

processors:
  chat:
    name: "chat_tracker"
```

### SWE-Bench 评测

```yaml
task_name: swe_task
task_config:
  run_oracle: false
  remote_dataset: "swe-public/SWE-bench_Verified"
  instance_id: "django__django-11095"

agent_name: swe_agent
agent_config:
  max_iterations: 15

runtime_provider_name: docker
runtime_configs:
  main:
    worker_venv: true

openai_clients:
  main:
    name: "openai"
    config:
      client_args:
        api_key: "sk-xxx"
        base_url: "https://api.openai.com/v1"
      request_args:
        model: "gpt-4.1"

processors:
  chat:
    name: "chat_tracker"
```

### Claude Code Agent

```yaml
task_name: "identity"

runtime_provider_name: "docker"
runtime_configs:
  main:
    image: ubuntu-wksp:latest
    worker_venv: true

agent_name: "claude_code"
agent_config:
  user_prompt: "汇报一下当前机器的系统环境"
  binary_url: http://172.17.0.1:38080/claude-2.1.85

openai_clients:
  main:
    name: "litellm"
    config:
      client_args:
        api_key: "your-api-key"
        base_url: "http://your-gateway:35000"
      request_args:
        model: "anthropic.claude-sonnet-4-6"

processors:
  chat:
    name: "chat_tracker"
```

## 环境变量替换

配置文件支持 OmegaConf 的环境变量插值：

```yaml
openai_clients:
  main:
    name: "openai"
    config:
      client_args:
        api_key: "${oc.env:OPENAI_API_KEY}"
        base_url: "${oc.env:OPENAI_BASE_URL}"
      request_args:
        model: "${oc.env:OPENAI_MODEL,gpt-4o}"  # 带默认值
```
