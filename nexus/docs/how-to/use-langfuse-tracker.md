# Langfuse Tracker 使用指南

`langfuse_tracker` 是一个业务 SpanProcessor，在每次 LLM 调用完成后实时上传普通执行 span 到
[Langfuse](https://langfuse.com)，无需等到任务结束再做离线导入。它默认订阅 Nexus `default` scope，不负责保存
`run.live/run.single` 生命周期或 distributed run trace journal。

## Trace 结构

上传到 Langfuse 的投影结构如下。这里的 Langfuse trace/root span 是 tracker 的外部展示模型，不等同于 Nexus 的
`run.*` span。当前契约中，一次 Nexus task 对应一个 episode trace/root span，每个
Generation 对应一个 turn，Generation 下的 Tool 节点对应该 turn 的 action/environment result：

```
Trace  (tags, session_id, input=task, output=result)
  └── Span "Agent"
        ├── Span "System Prompt"
        ├── Span "User Prompt"  (含 context_blocks)
        ├── Generation "Step 1: Bash"  (metadata.turn=1)
        │     └── Tool: Bash  (input=args, output=result)
        ├── Generation "Step 2: Read, Grep"  (metadata.turn=2)
        │     ├── Tool: Read
        │     └── Tool: Grep
        ├── Generation "Step 3: Agent"
        │     └── Tool: Agent
        │           └── Subagent: {description}
        │                 ├── System Prompt
        │                 ├── User Prompt
        │                 ├── Step 1: Grep, Read
        │                 └── Final Response
        └── Generation "Final Response"
```

## 快速开始

### 1. 安装依赖

`langfuse` 不在 Nexus 核心依赖中，需要单独安装：

```bash
uv pip install "langfuse>=4.0.0"
```

或者通过 optional dependency（如果已添加到 `pyproject.toml`）：

```bash
uv pip install "nexus-agent[langfuse]"
```

### 2. 配置处理器

在 YAML 配置文件或 batch generator 中添加 `langfuse_tracker` 处理器：

```yaml
processors:
  chat:
    name: "chat_tracker"
  langfuse:
    name: "langfuse_tracker"
    args:
      public_key: "${oc.env:LANGFUSE_PUBLIC_KEY}"
      secret_key: "${oc.env:LANGFUSE_SECRET_KEY}"
      host: "${oc.env:LANGFUSE_HOST,https://cloud.langfuse.com}"
      session_id: "my-session-id"
      trace_name: "my-trace"
      tags:
        - "claude_code"
        - "thinking"
        - "swe-bench"
      trace_metadata:
        instance_id: "django__django-10097"
        experiment: "exp_v1"
```

在 Python batch generator 中：

```python
config = {
    # ... task_name, agent_name, etc ...
    'processors': {
        'chat': {
            'name': 'chat_tracker',
        },
        'langfuse': {
            'name': 'langfuse_tracker',
            'args': {
                'public_key': os.getenv('LANGFUSE_PUBLIC_KEY', ''),
                'secret_key': os.getenv('LANGFUSE_SECRET_KEY', ''),
                'host': os.getenv('LANGFUSE_HOST') or os.getenv('LANGFUSE_BASE_URL', 'https://cloud.langfuse.com'),
                'session_id': instance_id,
                'trace_name': instance_id,
                'tags': [
                    agent_name,
                    thinking_mode,
                    'swe-bench',
                    os.getenv('SWE_EXPERIMENT', 'default'),
                ],
                'trace_metadata': {
                    'instance_id': instance_id,
                    'repo': inst.get('repo', ''),
                    'dataset_type': inst_dataset_type,
                    'experiment': os.getenv('SWE_EXPERIMENT', 'default'),
                    'agent_name': agent_name,
                    'thinking_mode': thinking_mode,
                },
            },
        },
    },
}
```

### 3. 设置环境变量

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"  # 太极平台注入；也兼容 LANGFUSE_BASE_URL
export TJ_INSTANCE_ID="ins-xxx"
export TJ_TASK_ID="task-xxx"
export TJ_TASK_NAME="my-task"
export TJ_WORKSPACE_ID="ws-xxx"
```

### 4. 运行

```bash
uv run nexus run-batch \
  -g ./swe_generator.py \
  -d ./outputs/batch.db \
  --max-concurrent 5
```

运行过程中可在 Langfuse 界面实时观察每个 step 的上传进度。

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `public_key` | `str` | `$LANGFUSE_PUBLIC_KEY` | Langfuse Public Key |
| `secret_key` | `str` | `$LANGFUSE_SECRET_KEY` | Langfuse Secret Key |
| `host` | `str` | `$LANGFUSE_HOST` / `$LANGFUSE_BASE_URL` | Langfuse API 地址 |
| `session_id` | `str` | `"{instance_id}-ep{episode}"` | Langfuse Session ID，用于关联同一 episode 的数据 |
| `trace_name` | `str` | `instance_id` / `task_name` / `""` | Trace 名称，显示在 Langfuse 列表中 |
| `tags` | `list[str]` | `[]` | Trace 标签，用于过滤和分组 |
| `trace_metadata` | `dict` | `{}` | Trace 级别的元数据 |
| `max_tool_chars` | `int` | `8000` | 工具输入/输出的最大截断长度 |
| `max_content_chars` | `int` | `0`（不截断） | 消息内容的最大截断长度 |

## 记录内容

### Taiji Episode / Turn Metadata

`langfuse_tracker` 会自动读取太极平台注入的任务元信息，并写入 trace metadata 与 tags：

| Langfuse 字段 | 来源 |
|------|------|
| `metadata.instance_id` | `TJ_INSTANCE_ID` |
| `metadata.task_id` | `TJ_TASK_ID` |
| `metadata.task_name` | `TJ_TASK_NAME` |
| `metadata.workspace_id` | `TJ_WORKSPACE_ID` |
| `tags` | `instance_id`、`task_id`、`task_name`、`workspace_id` 与用户配置 tags 去重合并 |

如果未显式配置 `session_id`，且存在 `instance_id`，默认使用 `"{instance_id}-ep{episode}"`。`episode`
默认来自 `trace_metadata.episode`，未配置时为 `0`。

如果未显式配置 `trace_name`，默认优先使用 `instance_id`，再 fallback 到 `task_name`，便于在 Langfuse
Trace 列表中直接识别任务实例或仓库标识。

每个 Generation 会写入 `metadata.turn`，其值等于当前 agent step。Tool 节点会写入相同的
`turn`、`episode`、`episode_id`，便于查询 action/environment result 所属 turn。

### 每次 LLM 调用（Generation）

- **Input**: 自上次调用以来的增量消息（delta messages），包含完整的 assistant 输出、tool 结果、system reminder 等
- **Output**: assistant 文本、thinking 内容、tool_calls 列表
- **Usage**: input/output/total tokens、cached tokens
- **Metadata**: turn/step 编号、episode 信息、工具名称、耗时、是否并行调用

### System Prompt / User Prompt

首次 LLM 调用时自动提取并创建独立的 span：

- **System Prompt**: 完整的系统提示词内容
- **User Prompt**: 用户任务描述（task_text）及 context_blocks

### Tool 调用

每个 tool_call 创建为上一个 Generation 的子 span：

- **Input**: 工具调用参数（arguments）
- **Output**: 工具执行结果

### Subagent

当检测到 system prompt 切换时（如 Claude Code 调用 Task 工具），自动创建嵌套结构：

```
Generation "Step N: Agent"
  └── Tool: Agent (input=description+prompt)
        └── Subagent: {description}
              ├── System Prompt
              ├── User Prompt
              ├── Step 1: ...
              └── Final Response
```

### Finalize

任务结束时自动：

- 关闭所有未完成的 subagent span
- 创建 "Final Response" generation（如果最后一步无 tool_call）
- 更新 Agent span 的元数据（total_steps、total_usage、cache_hit_rate、tools_summary）
- 如果 `task_output.score` 存在且可转换为数值，上报 Langfuse Score `episode_reward`
- 将 model 名称自动添加到 tags

## Trace 唯一性

Trace ID 通过 `langfuse.create_trace_id(seed=...)` 生成，seed 由 `trace_name`、`experiment`（来自 trace_metadata）和当前时间戳拼接而成。不同运行会生成不同 trace，避免多次运行同一 instance 时数据覆盖。

## 与 chat_tracker 配合

`langfuse_tracker` 和 `chat_tracker` 可以同时使用，互不干扰。两者的 `get_result()` 都进入
`RunOutput.processor_results`，最终保存位置由 runner 决定：普通 single 可写输出文件，本地 batch 写 SQLite，
distributed live/single 则通过 raw `run.output` 保存到 run trace backend。

- `chat_tracker`: 形成完整对话历史的 processor result
- `langfuse_tracker`: 实时上传到 Langfuse 做可视化分析

```yaml
processors:
  chat:
    name: "chat_tracker"
  langfuse:
    name: "langfuse_tracker"
    args:
      # ...
```

## 模块声明

`langfuse_tracker` 通过 inline script metadata 声明为 Nexus 扩展模块：

```python
# /// script
# dependencies = ["langfuse>=4.0.0"]
#
# [tool.nexus]
# span_processors = { langfuse_tracker = "nexus.trace.processors.langfuse_tracker" }
# ///
```

当 Nexus 加载包含此文件的项目时，会自动注册 `langfuse_tracker` 处理器并安装 `langfuse` 依赖。

## 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `Langfuse credentials not configured` | 未设置 public_key / secret_key | 检查环境变量或 args 配置 |
| `Failed to initialize Langfuse` | 网络不通或 SDK 版本不兼容 | 确认 host 可达，确认 `langfuse>=4.0.0` |
| 同一 instance 的 trace 被合并 | trace_name 和 experiment 相同 | 使用不同的 `SWE_EXPERIMENT` 值 |
| Langfuse UI 中 prompt 排在 Step 1 后面 | 时间戳自然排序问题 | 已内置处理（prompt span 使用偏移时间戳） |
