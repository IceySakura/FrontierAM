# Main Agent

Nexus 内部的标准 Agent 实现，目前是一个基础的 ReAct Loop ，后续将随研究进度新增记忆、多 Agent 等功能。 目前支持：

- 自定义文本/多模态 `user_prompt` / `system_prompt`
- 可选 `history`，在当前 `user_prompt` 前接入已有多轮对话
- 通过 `tools` 参数配置任意工具组合
- 串行和并行工具执行
- 可选 `post_iteration_hook`，在每轮工具执行后追加或替换观测消息
- 内置观测指标

当模型不再调用工具时，循环结束。工具也可以通过 `ToolResponse.meta.termination_requested == true`
显式结束循环；这类终止会映射为 `FINISHED`。Submit / Finish 等特殊工具不参与额外的控制流判断，
除非它们自己返回上述 termination meta。

## 函数语义

### 输入

函数签名：

```python
async def run_main_agent(agent_input: MainAgentInput) -> MainAgentOutput
```

`agent_input` (`MainAgentInput`) 继承自 `AgentRunInput` + `WithUserPrompt` + `WithHistory`，参数如下：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `user_prompt` | `str | list[ChatCompletionContentPartParam]` | *必填* | 用户输入 |
| `history` | `list[dict[str, Any]] \| None` | `None` | 可选已有对话历史，会插入到 `system_prompt` 和本轮 `user_prompt` 之间。只允许 `user` / `assistant` / `tool` 角色，不允许包含 `system`；非空时最后一条必须是 `assistant` 或 `tool`，避免与本轮 `user_prompt` 冲突 |
| `system_prompt` | `str | list[ChatCompletionContentPartParam]` | `'You are a helpful assistant.'` | 系统提示词 |
| `max_iterations` | `int` | `30` | 最大迭代次数 |
| `tools` | `list[str]` | *必填* | 工具注册名列表，如 `['swe_agent/bash', 'swe_agent/str_replace_editor']` |
| `mcp_servers` | `list[dict[str, Any]] \| None` | `None` | MCP 服务器配置列表，agent 启动时自动加载对应工具。 支持 `MCPServerSpec` 的完整字段。 若为 `None` 则不加载 |
| `parallel_tool_calls` | `bool` | `False` | 是否并行执行同一轮的多个 tool calls |
| `invalid_tool_args_retries` | `int` | `0` | 非负整数。tool call 参数 JSON 解析失败时重试 LLM 响应；参数为合法 JSON 但工具名或 input model 校验失败时，写入 tool error observation 后允许模型重发修正后的 tool call。 |
| `tool_call_error_retries` | `int` | `0` | 非负整数。工具执行抛异常时的重试次数，耗尽后退出为 `TOOL_EXECUTION_ERROR` |
| `tool_output_truncation` | `int | None` | `None` | 工具输出截断长度（字符数），`None` 不截断 |
| `tool_error_continue` | `bool` | `False` | 是否将 tool call 格式 / schema 错误和普通工具执行错误作为 tool observation 回写给模型并继续循环。不会覆盖 timeout、runtime unhealthy、inference/context 错误等终止条件。可通过环境变量 `NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE` 覆盖 |
| `tool_overrides` | `list[MainAgentToolOverride] | None` | `None` | 按顺序应用的一组工具元信息覆盖配置，对应多次 `executor.set_tool_override(...)` |
| `post_iteration_hook` | `str | None` | `None` | 可选 import path，格式为 `module.path::object_name`。每轮工具执行后调用，签名为 `async hook(ctx, hook_context) -> PostIterationHookResult | None` |
| `observation_time_timeout` | `float | None` | `None` | 累计工具执行时间上限（秒），超出后退出为 `OBSERVATION_TIMEOUT` |
| `inference_time_timeout` | `float | None` | `None` | 累计 LLM 调用时间上限（秒），超出后退出为 `INFERENCE_TIMEOUT` |

### 输出

返回值为 `MainAgentOutput`（继承自 `AgentRunOutput` + `WithMainTrajectory`）：

| 属性 | 类型 | 说明 |
|------|------|------|
| `exit_status` | `str` | 退出状态，取值见下方 `MainExitStatus` |
| `iterations` | `int` | 实际执行的迭代次数 |
| `final_answer` | `str | None` | 最近一次非空 `assistant.message.content` |
| `main_trajectory` | `list[dict]` | 已写入 history 的完整对话轨迹，包含 `system_prompt`、传入的 `history`、本轮 `user_prompt` 和后续 assistant/tool 消息。重试过程中被丢弃的无效 assistant 响应不会写入 |

`MainExitStatus` 的取值语义如下：

| 退出码 | 值 | termination_reason | 触发条件 |
|--------|------|---------------------|----------|
| `FINISHED` | `finished` | `ENV_DONE` | 模型返回无 tool_calls 的响应，或某个工具返回 `ToolResponse.meta.termination_requested == true` |
| `MAX_ITERATIONS` | `max_iterations` | `MAX_STEPS` | 迭代次数达到 `max_iterations` 上限 |
| `TOOL_ERROR` | `tool_error` | `FORMAT_ERROR` | Tool call 格式或 schema 错误：tool call 参数 JSON 解析失败且重试耗尽，或工具名 / input model 校验失败且 `invalid_tool_args_retries` 重试耗尽；或客户端标记该轮开始了 parser 无法解析的 tool call（`choice.provider_specific_fields['unparsed_tool_call']`），此时不重试直接结束，且不受 `tool_error_continue` 影响（这一轮没有可回应的 tool_call id） |
| `TOOL_EXECUTION_ERROR` | `tool_execution_error` | `ENV_ERROR` | 工具执行阶段错误：工具执行抛异常且 `tool_call_error_retries` 重试耗尽、工具输出无法渲染、并行工具响应数量不匹配等 |
| `INFERENCE_ERROR` | `inference_error` | `UNHANDLED_ERROR` | 推理请求运行期异常（如网络/上游错误）或模型返回结构不符合预期 |
| `CONTEXT_LENGTH_EXCEEDED` | `context_length_exceeded` | `TRUNCATION` | 捕获到 `ContextLengthExceededError`，上下文长度超限 |
| `OUTPUT_LENGTH_EXCEEDED` | `output_length_exceeded` | `OUTPUT_LENGTH_EXCEEDED` | 单次 LLM sampling 达到实际 `max_tokens`，返回的 reasoning、answer 或 tool call 可能不完整 |
| `INFERENCE_TIMEOUT` | `inference_timeout` | `TASK_TIMEOUT` | 累计 inference 时间超出 `inference_time_timeout` 阈值 |
| `OBSERVATION_TIMEOUT` | `observation_timeout` | `TASK_TIMEOUT` | 累计 observation 时间超出 `observation_time_timeout` 阈值 |
| `RUNTIME_ERROR` | `runtime_error` | `UNHANDLED_ERROR` | Runtime 进入不可继续使用的状态，例如工具执行时捕获到 `RuntimeUnhealthyError` |

### 异常

直接向上抛出的异常（不会被转换成 `exit_status`）：

| 异常类型 | 典型场景 |
|------|------|
| `RuntimeError` | 没有激活 `ExecutionContext`（调用 `require_context()` 失败） |
| `KeyError` | 工具注册名不存在，或 `tool_overrides` 里的 `tool_name` 不存在 |
| `ValueError` | `history` 包含非法角色、非空 `history` 以 `user` 结尾，或工具 schema 名冲突（例如多个工具 override 到同名） |
| `asyncio.CancelledError` | 外部主动取消任务 |
| 其他未显式处理异常 | 代码 bug、不可恢复内部错误等 |

### 观测

#### Span 结构

执行过程中会生成 3 类 span：

| span 名称 | 含义 |
|------|------|
| `main_agent` | 整体生命周期 |
| `main_agent.inference` | 每次推理调用（含重试） |
| `main_agent.observation` | 每轮工具执行阶段 |

`main_agent` 结束时会写入以下属性：`agent.exit_status`、`agent.termination_reason`、`agent.iterations`、`agent.tool_call_count`、`agent.tool_error_count`。

如果写入 span 属性失败，只会记录 warning，不会影响 `MainAgentOutput` 返回。

#### Metrics Processor

`MainAgentMetricsProcessor` 通过 span 体系自动采集 agent 运行指标，无需在 agent 代码中手动埋点。

##### 注册

```yaml
processors:
  agent_metrics:
    name: main_agent_metrics
```

##### 采集的指标

`get_result()` 返回的 dict 包含：

| 键 | 类型 | 说明 |
|----|------|------|
| `agent.exit_status` | `str` | 退出码 |
| `agent.termination_reason` | `str` | 终止原因（`ENV_DONE` / `MAX_STEPS` / ...） |
| `agent.iterations` | `int` | 外层循环 `for iteration in 1..max_iterations` 的执行次数（不受 retry 次数影响） |
| `agent.tool_call_count` | `int` | 工具调用总次数 |
| `agent.tool_error_count` | `int` | 工具错误次数 |
| `agent.total_inference_time` | `float` | 累计 LLM 调用时间（秒） |
| `agent.total_observation_time` | `float` | 累计工具执行时间（秒） |
| `agent.inference_times` | `list[float]` | 每次 inference 尝试（包含重试）的耗时列表 |
| `agent.observation_times` | `list[float]` | 每轮 observation 阶段的耗时列表 |

### Post-Iteration Hook

`post_iteration_hook` 用于在每轮工具执行后、下一轮 LLM 推理前调整 observation。hook import path 使用
`module.path::object_name` 格式，例如：

```yaml
agent_config:
  post_iteration_hook: nexus.extensions.tools.desktop.observation::desktop_observation
```

hook 函数接收当前 `ExecutionContext` 和 `PostIterationHookContext`。
`hook_context.history` 是已追加本轮 assistant 响应、但尚未追加本轮 tool messages 的只读历史快照；
`hook_context.tool_messages` 是本轮生成的 tool messages，只读使用；
`hook_context.iteration` / `hook_context.max_iterations` 分别是当前迭代号和最大迭代数。
返回 `PostIterationHookResult` 时：

- `tool_messages` 不为 `None`：替换本轮 tool messages。
- `extra_messages` 非空：追加到本轮 messages 后面，常用于补充截图等多模态 observation。
- `next_completion_kwargs` 非空：作为关键字参数合并进下一次 `client.create_completion` 调用，调用后清空。具体可用的 key 取决于 client 实现（例如 hytuner client 读取其中的 `sampling_params` dict，普通 OpenAI 兼容 client 可接受 `temperature` / `extra_body` 等）。
- client 配置中的 `enforced_request_args` 会在 `next_completion_kwargs` 和旧 request override 之后生效；
  agent/compaction 对 model 和输出预算的读取也使用 enforced 后的值。

OSWorld task 会提供默认 desktop observation hook，标准配置不需要手写。当前六模式中裸CUA与CC/Codex各自管理观察循环；此main_agent示例不是OSWorld六模式入口。

##### Metrics Processor 回调

继承 `MainAgentMetricsProcessor`，覆写回调方法即可在迭代边界注入自定义逻辑：

```python
import logging

from nexus.extensions.agents.main import MainAgentMetricsProcessor

logger = logging.getLogger(__name__)


class MyProcessor(MainAgentMetricsProcessor):
    def on_iteration_start(self, iteration: int) -> None:
        """每个外层 iteration 开始时调用（重试不会重复触发）。"""
        logger.info('iteration %d started', iteration)

    def on_observation_end(self, iteration: int, duration: float) -> None:
        """每轮工具执行结束时调用，duration 为本轮耗时（秒）。"""
        if duration > 30:
            logger.warning('slow observation: %.1fs at iteration %d', duration, iteration)
```

## 用法

### 样例模板

一个样例配置文件：

```yaml title="workspace/example.yaml"
imports:
  - nexus.extensions.agents.main
  - nexus.extensions.tasks.identity
  - nexus.extensions.openai_clients
  - nexus.extensions.agents.swe_agent.tools
  - nexus.extensions.tools.code_interpreter

task_name: "identity"
task_config: {}

runtime_provider_name: "docker"
runtime_provider_config: {}
runtime_configs:
  main:
    image: "ubuntu:24.04"

agent_name: "main_agent"
agent_config:
  system_prompt: "You are a precise coding assistant. Use tools when needed, then finish with a concise summary."
  user_prompt: |
    Please complete the following steps in order:
    1) Use `bash` to create `/tmp/nexus_demo.txt` with content `hello from bash`.
    2) Use `str_replace_editor` to replace `hello from bash` with `hello from str_replace_editor` in that file.
    3) Use `jupyter_code_interpreter` to compute 23 * 19 and print the result.
    4) Use `bash` to show the final file content (`cat /tmp/nexus_demo.txt`).
    5) Respond with a short final summary and stop calling tools.
  max_iterations: 12
  tools:
    - swe_agent/bash
    - swe_agent/str_replace_editor
    - jupyter_code_interpreter
  # mcp_servers:                   # 可选：agent 启动时自动连接并加载 MCP 工具
  #   - name: "web_search"
  #     url: "http://localhost:8080"
  parallel_tool_calls: false
  invalid_tool_args_retries: 1
  tool_call_error_retries: 1

openai_clients:
  main:
    name: "openai"
    config:
      client_args:
        api_key: "${oc.env:NEXUS_LITELLM_API_KEY}"
        base_url: "${oc.env:NEXUS_LITELLM_BASE_URL}"
      request_args:
        model: "${oc.env:NEXUS_LITELLM_MODEL}"

processors:
  agent_metrics:
    name: "main_agent_metrics"
  chat:
    name: "chat_tracker"
```

先设置模型网关相关环境变量（可按你的 `workspace/example.yaml` 对应值设置）：

```bash
export NEXUS_LITELLM_API_KEY="..."
export NEXUS_LITELLM_BASE_URL="..."
export NEXUS_LITELLM_MODEL="gpt-5"
```

然后在根目录执行：

```bash
uv run nexus run-single workspace/example.yaml -o workspace/result.json
```

### Tool Overrides

`tool_overrides` 是一组 `set_tool_override(...)` 调用配置。`main_agent` 会在生成 schemas 之前按顺序应用它们。

每一项支持以下字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `tool_name` | `str` | 是 | 工具注册名（例如 `swe_agent/bash`） |
| `name` | `str | None` | 否 | 覆盖 schema 中的 `function.name` |
| `description` | `str | None` | 否 | 覆盖工具描述 |
| `params` | `dict[str, str | ToolParamOverride] | None` | 否 | 覆盖参数描述；`{'command': '...描述...'}` 是最常用形式 |

```python
agent_input = MainAgentInput(
    user_prompt='...',
    tools=['swe_agent/bash'],
    tool_overrides=[
        {
            'tool_name': 'swe_agent/bash',
            'name': 'Bash',
            'description': 'Run shell commands in workspace',
            'params': {
                'command': 'Shell command to execute',
            },
        }
    ],
)
```

注意：`main_agent` 优先按 schema 名分发工具调用，随后再回退到注册名。建议让模型看到的 `function.name` 与你的工具调用约定保持一致。
