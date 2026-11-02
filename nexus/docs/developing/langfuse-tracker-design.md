# Langfuse Tracker 最小契约设计

## 背景

`langfuse_tracker` 当前已经能把一次 Nexus task run 上报为一个 Langfuse trace，并在 trace 下记录 agent、prompt、LLM generation、tool result 和最终输出。为了贴近太极 Langfuse 接入文档中的任务元信息、session 聚合和 turn 维度查询能力，本设计只补齐元数据契约，不重构现有 trace 层级。

## 目标

- 保持现有 Langfuse 层级基本不变，避免影响已有 UI 使用习惯。
- 将一次 Nexus run 视为一个 episode，并映射到当前 Langfuse trace/root span。
- 将每个 LLM generation 视为一个 turn。
- 将 generation 下的 tool observation 视为该 turn 的 action/environment result。
- 自动补齐太极注入的任务元信息、session_id 和 tags。
- 将 `task_output.score` 上报为 Langfuse Score。
- 兼容文档中的 `LANGFUSE_HOST` 环境变量。

## 非目标

- 不新增独立的 `Turn` span。
- 不把每个 turn 拆成独立 Langfuse trace。
- 不修改 agent、runner、task 的现有 tracing 埋点。
- 不改 `agent_iteration`、`agent_inference`、`agent_observation` 的产生逻辑。
- 不新增 turn 级 reward 计算逻辑。当前只上报 task output 中已经存在的 episode/task 级 score。

## 当前结构

当前 `langfuse_tracker` 的主要层级是：

```text
Langfuse trace
  └── root span: trace_name or "Agent Run"
      └── span: "Agent"
          ├── span: "System Prompt"
          ├── span: "User Prompt"
          ├── span: "Tools Definition"
          ├── generation: "Step N: ..."
          │   └── tool: "Tool: xxx"
          └── generation: "Final Response"
```

其中 `generation` 由 `openai_completion` span 生成，`tool` 节点由下一次 LLM 输入中的 tool message 反推并挂到上一轮 generation 下。

## 最终契约

采用以下最小映射：

| 训练语义 | Langfuse 表达 | Nexus 当前来源 |
| --- | --- | --- |
| episode | 当前 Langfuse trace/root span | 一次 `langfuse_tracker` 生命周期 |
| turn | 当前 `generation` observation | `ctx.step` / `_global_step` |
| model inference | `generation` observation | `openai_completion` |
| action/environment result | generation 下的 `tool` observation | tool message 反推 |
| task result | trace/root span metadata | task span output |
| reward/score | Langfuse Score + trace metadata | `task_output.score` |

## Metadata 契约

### Taiji Episode / Turn Metadata

`langfuse_tracker` 需要读取并合并以下环境变量：

| 环境变量 | metadata 字段 | tags |
| --- | --- | --- |
| `TJ_INSTANCE_ID` | `instance_id` | yes |
| `TJ_TASK_ID` | `task_id` | yes |
| `TJ_TASK_NAME` | `task_name` | yes |
| `TJ_WORKSPACE_ID` | `workspace_id` | yes |

用户显式传入的 `trace_metadata` 和 `tags` 继续生效。合并策略为：

- `trace_metadata` 优先级高于环境变量，允许调用方覆盖。
- 自动 tags 与用户 tags 去重后合并。
- 空值不写入 tags。

### Episode 字段

trace/root span metadata 需要补齐：

```text
contract = "episode_trace_generation_turn"
episode = trace_metadata.episode or 0
episode_id = trace_metadata.episode_id or "{instance_id}-ep{episode}"
instance_id
task_id
task_name
workspace_id
source = "langfuse_tracker"
```

`session_id` 默认值：

```text
explicit session_id or episode_id
```

如果没有 `instance_id`，则保持显式 `session_id` 或空值，不强造不可查询的 session。

### Generation 字段

每个 generation 的 metadata 需要补齐：

```text
turn = ctx.step
step = ctx.step
global_step = self._global_step
episode
episode_id
contract = "episode_trace_generation_turn"
tool_count
tool_names
parallel
duration_s
source = "realtime"
```

展示名可以继续保持 `Step N: ...`，避免改 UI 习惯。查询语义以 `metadata.turn` 为准。

### Tool 字段

每个 tool observation 的 metadata 需要补齐：

```text
turn = generation metadata.turn
step = generation metadata.step
global_step = generation metadata.global_step
episode
episode_id
tool_call_id
tool_name
is_agent_call
is_parallel
source = "realtime"
```

finalize 阶段补齐的未完成 tool span 也应写入同样的 episode/turn 字段。

### Task Result 和 Score 字段

finalize 更新 trace/root span 时继续写入：

```text
task_result = task output
```

如果 `task_result.score` 存在，可额外平铺：

```text
score = task_result.score
```

同时调用 Langfuse Score API 上报 episode/task 级分数：

```text
trace_id = self._trace_id
name = "episode_reward"
value = float(task_result.score)
comment = "instance={instance_id}, episode={episode}, task={task_name}"
```

如果 `task_result.score` 不能转换为数值，则只保留在 metadata，不调用 Score API，并记录 debug/warning 日志。Score 上报失败不应影响 trace finalize 和 flush。

## 环境变量兼容

当前实现只读取 `LANGFUSE_BASE_URL`。需要改为兼容太极文档使用的 `LANGFUSE_HOST`：

```text
host arg
or LANGFUSE_HOST
or LANGFUSE_BASE_URL
or "https://cloud.langfuse.com"
```

`LANGFUSE_PUBLIC_KEY` 和 `LANGFUSE_SECRET_KEY` 保持现有行为。

## 实现范围

### `nexus/trace/processors/langfuse_tracker.py`

最小改动集中在一个文件：

1. 在初始化阶段读取 `LANGFUSE_HOST`。
2. 增加一个内部 helper，合并太极 env、`trace_metadata`、episode 字段和 tags。
3. 在 `propagate_attributes()` 中使用合并后的 `session_id`、`tags`、`trace_name`。
4. 创建 trace/root span 时写入合并后的 metadata。
5. 创建 generation 时写入 `turn`、`episode`、`episode_id`、`contract`。
6. 创建 tool span 时写入同一组 turn/episode 字段。
7. finalize 更新 trace metadata 时保留合并后的 metadata，并补 `task_result` 和可选平铺 `score`。
8. finalize 阶段将可数值化的 `task_result.score` 上报为 Langfuse Score，默认名称为 `episode_reward`。

### `docs/how-to/use-langfuse-tracker.md`

更新文档中的映射说明：

```text
episode -> Langfuse trace/root span
turn -> Langfuse generation
action/environment result -> generation 下的 tool observation
task result/score -> trace metadata
task output score -> Langfuse Score(name="episode_reward")
```

同时更新环境变量说明，明确支持 `LANGFUSE_HOST` 和太极 `TJ_*` 元信息。

### Tests

增加或更新 processor 单元测试，使用 mock Langfuse object，不依赖真实 Langfuse 服务。

覆盖点：

- `LANGFUSE_HOST` 优先于 `LANGFUSE_BASE_URL`。
- trace/root span metadata 包含 `contract`、`episode`、`episode_id`、`instance_id`、`task_id`、`task_name`、`workspace_id`。
- 默认 `session_id` 使用 `episode_id`。
- tags 自动包含 `instance_id`、`task_id`、`task_name`、`workspace_id`，并与用户 tags 去重合并。
- generation metadata 包含 `turn`、`episode`、`episode_id`、`contract`。
- tool metadata 包含同一 turn/episode 字段。
- `task_output.score` 存在且可数值化时，trace metadata 平铺 `score`，并调用 Langfuse Score API 上报 `episode_reward`。
- `task_output.score` 不可数值化或 Score API 抛错时，finalize 不失败。

## 兼容性

- 现有配置仍可继续传 `session_id`、`trace_name`、`tags`、`trace_metadata`。
- 显式配置优先于环境变量。
- 现有 Langfuse UI 层级不增加额外层级。
- `Step N` 展示名可保持不变，避免影响已有截图、排查习惯或下游解析。

## 验收标准

- 不修改 agent、runner、task 代码。
- 本地单元测试通过。
- 使用 mock Langfuse 验证上报结构中 episode/turn metadata 完整。
- 文档明确写出最终映射关系和环境变量行为。
