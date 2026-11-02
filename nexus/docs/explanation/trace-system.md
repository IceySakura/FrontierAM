# Trace 系统

Nexus Trace 统一记录 task、agent、LLM、tool、runtime 和 runner 生命周期。普通 span、run 生命周期事件和持久化结果共享
同一套 span event 数据模型，但由不同组件消费。本文同时定义 span 命名、属性使用和层级结构规范；命名风格遵循
[OpenTelemetry Naming Convention](https://opentelemetry.io/docs/specs/semconv/general/naming/)。

## 核心概念

| 概念 | 说明 |
| --- | --- |
| Trace | 由同一个 `trace_id` 关联的一棵 span 树。 |
| Span | 一段有开始、更新和结束时间的操作，例如 task、agent iteration、LLM 调用或 tool 调用。 |
| Span event | 附着在 span 上的结构化事件，例如 run 生命周期使用的 `run.event`。 |
| `SpanEvent` | `START`、`UPDATE`、`END` 三种进程内事件。 |
| `SpanEventRecord` | 带单调 `record_id` 的有序记录，是 run trace journal 的原始格式。 |

`SpanController` 是进程内的统一 span 收集点；`RunTraceSession` 为一次 managed run 建立 root span 和持久化边界。普通
single 不建立 run span，live 和 distributed single 则形成 `run -> task -> agent/tool/runtime` 的完整树。

## Scope、Span 树和自动属性

### Processor Scope

Trace 系统只有两个 processor scope：

| Scope | 内容 | 默认消费者 |
| --- | --- | --- |
| `default` | task、agent、LLM、tool、runtime 等普通执行 span。 | 配置在 `RunInput.processors` 中的 processor。 |
| `run` | `run.live`、`run.single` 和未来的 `run.batch` 运行 span。 | 框架管理的 `RunTraceProcessor`。 |

业务 `SpanProcessor` 默认只订阅 `default`。自定义 processor class 需要消费运行事件时，可以通过 `processor_scopes` 增加
`run`；`RunInput.processors` 本身没有单独的 scope 配置。框架管理的 `RunTraceProcessor` 同时订阅两个 scope。

### Span 树

普通 single 直接执行时，以 SWE task 为例的完整 span 层级：

```text
task.swe_task                              type=task, name=swe_task
├── task.init                              type=task_phase, phase=init
│   ├── task.init.get_runtime              type=task_phase, phase=init
│   └── task.init.runtime_setup            type=task_phase, phase=init
├── task.agent                             type=task_phase, phase=agent
│   └── agent.main_agent                   type=agent, name=main_agent
│       ├── agent.iteration                type=agent_iteration, iteration=1
│       │   ├── agent.inference            type=agent_inference, iteration=1, retry=0
│       │   │   └── openai_completion      type=llm
│       │   ├── agent.observation          type=agent_observation, iteration=1
│       │   │   ├── tool.bash              type=tool, name=bash
│       │   │   └── tool.str_replace_editor type=tool, name=str_replace_editor
│       │   └── agent.compaction           type=agent_compaction, iteration=1
│       ├── agent.iteration                type=agent_iteration, iteration=2
│       │   └── ...
│       └── ...
└── task.scoring                           type=task_phase, phase=scoring
    ├── task.scoring.runtime_init          type=task_phase, phase=scoring
    └── task.scoring.grade                 type=task_phase, phase=scoring
```

live runner 和 distributed live/single driver 由 `RunTraceSession` 托管。`session.activate()` 让 run span 成为 task 的
直接 parent，并让整棵树复用 run span 的 `trace_id`：

```text
run.live | run.single                      type=live_run | single_run
└── task.swe_task                          type=task, name=swe_task
    ├── task.init                          type=task_phase, phase=init
    ├── task.agent                         type=task_phase, phase=agent
    │   └── agent.main_agent               type=agent, name=main_agent
    │       └── agent.iteration
    │           ├── agent.inference
    │           │   └── openai_completion
    │           ├── agent.observation
    │           │   └── tool.<name>
    │           └── agent.compaction
    └── task.scoring                       type=task_phase, phase=scoring
```

run span 和 task span 使用同一个 `trace_id`，task 的 `parent_span_id` 等于 run span id。普通
`nexus.runner.single.run()` 不创建 run span，task 仍是 trace root。`run.batch` 是后续 batch parent 的保留约定；本地
batch 当前不会创建该 span。

runtime span 继承调用发生时的 active parent，因此可能位于 task phase、agent inference、tool 或其他 span 下，不固定在
agent 的某一层。

### 自动注入属性

以下属性由 `start_span()` 自动注入，无需手动设置：

- `hostname`
- `process_id`
- `thread_id`
- `task_id`（async context）
- `project_id`（如已设置）

在 `RunTraceSession.activate()` 托管的执行中，普通 span 还会继承以下内部关联属性：

- `_run_id`
- `_run_span_id`
- `_run_kind`

这些属性由 runner 注入，供 `RunTraceProcessor` 将普通 span 关联到 active run，不属于业务 span 的公共配置接口。

## 组件与数据流

一次 managed run 的主要数据流是：

```text
span producer
    └── SpanController
        ├── business SpanProcessor -> RunOutput.processor_results
        └── RunTraceProcessor
            └── per-run state
                ├── backend RunTraceSink -> raw SpanEventRecord journal
                └── internal LiveRunTraceView -> live projection stream (live only)
```

### Span 生产者和 Processor

Span 生产者（task/agent 代码）只负责设置通用属性，例如 `type`、`phase`、`iteration`，不为特定消费者添加标记属性。
processor 根据已有属性自行决定收集哪些 span。收集策略集中在 processor 内部，对 task/agent 代码透明。

### SpanController

`SpanController` 维护进程内的 active span。每次收到 `SpanEvent` 时，它更新 span 状态，并把只读 span snapshot 和对应事件
分发给订阅当前 scope 的 processor。

### 业务 SpanProcessor

`RunInput.processors` 配置的是业务 processor，例如 `chat_tracker`、`langfuse_tracker`、`json_line` 和
`token_counter`。它们默认只订阅 `default` scope，在一次 single task 的 `processor_context` 中生效，结束后通过
`get_result()` 形成 `RunOutput.processor_results`。

自定义 processor 的回调签名为：

```python
from nexus.trace import SpanEndEvent, SpanStartEvent, SpanUpdateEvent
from nexus.trace.processors import SpanProcessor
from nexus.trace.types import Span


class MyProcessor(SpanProcessor):
    def on_start(self, span: Span, event: SpanStartEvent) -> None:
        pass

    def on_update(self, span: Span, event: SpanUpdateEvent) -> None:
        pass

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        pass
```

Processor 可以选择 `inline` 或 `background` dispatch。background callback 按队列顺序执行，退出
`processor_context` 时会先等待已接收 callback，再调用 `finalize()`。

### RunTraceProcessor

`RunTraceProcessor` 是进程级常驻 processor，同时订阅 `default` 和 `run` scope。它根据 run span 或普通 span 的内部
run identity 找到 active per-run state，将 span snapshot 和 `SpanEvent` 转成有序 `SpanEventRecord`，再分发给该 run 的
sinks。它不生成业务 `processor_results`，也不永久保留已关闭 run 的状态。

### RunTraceSession

每个 live 或 distributed single 持有自己的 `RunTraceSession`。session 管理 run identity、root run span、backend/internal
sinks、task parent context 和关闭流程：

1. `start()` 注册 per-run state 并启动 `run.<kind>` span。
2. `activate()` 让 run span 成为托管 task 的 parent。
3. `emit()` 将生命周期信息追加为 run span 上的 `run.event`。
4. `close()` 结束 run span，等待已接收 records 写完并生成 source routing metadata。

## 命名和属性规范

### 字符集

标准 Span name 仅使用小写字母、数字、下划线和点：`[a-z0-9_.]`。尚未迁移的兼容例外见
[附录：HyTuner 迁移](#附录hytuner-迁移)。

### 分隔符

| 分隔符 | 语义 | 示例 |
| --- | --- | --- |
| `.` | namespace 层级 | `task.init.get_runtime` |
| `_` | 多词组件（snake_case） | `task.swe_task`，`tool.str_replace_editor` |

### Span 分类

- **实体 span**：表示“这是谁”，格式为 `<类别>.<实例名>`，由注册体系自动创建。
- **阶段 span**：表示“在做什么”，格式为 `<类别>.<动作>`，由 task/agent 代码显式创建。

两者通过 `type` attribute 区分。

### 注册名与 Span Name 的关系

注册名（registry key）保持原样不变，不影响配置文件。生成 span name 时自动添加类别前缀，并将 `/` 替换为 `.`：

| 注册名 | Span Name | Attribute `name` |
| --- | --- | --- |
| `swe_task` | `task.swe_task` | `swe_task` |
| `main_agent` | `agent.main_agent` | `main_agent` |
| `swe_agent/bash` | `tool.swe_agent.bash` | `swe_agent/bash` |
| `str_replace_editor` | `tool.str_replace_editor` | `str_replace_editor` |

转换逻辑：

```python
def _to_span_name(prefix: str, registry_name: str) -> str:
    return f'{prefix}.{registry_name.replace("/", ".")}'
```

### `type` 约定值

| 值 | 含义 | Span name 模式 |
| --- | --- | --- |
| `task` | Task 实体 | `task.<name>` |
| `task_phase` | Task 阶段 | `task.init`，`task.scoring` 等 |
| `agent` | Agent 实体 | `agent.<name>` |
| `agent_iteration` | Agent 单轮 | `agent.iteration` |
| `agent_inference` | Agent 推理 | `agent.inference` |
| `agent_observation` | Agent 工具执行 | `agent.observation` |
| `agent_compaction` | Agent 上下文压缩 | `agent.compaction` |
| `tool` | Tool 实体 | `tool.<name>` |
| `llm` | LLM 调用 | `openai_completion` |
| `live_run` | Live runner 生命周期 | `run.live` |
| `single_run` | Distributed single 生命周期 | `run.single` |
| `batch_run` | Future distributed batch parent 生命周期 | `run.batch` |

`batch_run` 是保留值，当前本地 batch 不会产生该 span。

### Task 阶段

| Phase | 含义 | 典型操作 |
| --- | --- | --- |
| `init` | 任务初始化 | 加载实例数据，获取 runtime，上传文件，构建 prompt |
| `agent` | Agent 执行 | 调用 `run_agent()`，提取 agent 输出 |
| `scoring` | 评分 | 运行评估逻辑（LLM judge / pytest / 字符串比对），计算分数 |

固定属性：

| 属性 | 适用范围 | 类型 | 说明 |
| --- | --- | --- | --- |
| `type` | 所有 task span | `str` | `task` 或 `task_phase` |
| `name` | 实体 span | `str` | 原始注册名，保持与配置文件一致 |
| `phase` | `task_phase` span | `str` | `init` / `agent` / `scoring` |

### Agent 轮次

```text
agent.<name>
└── agent.iteration              # 包裹单轮
    ├── agent.inference          # LLM 调用
    ├── agent.observation        # 工具执行
    └── agent.compaction         # 可选的上下文压缩
```

固定属性：

| 属性 | 适用范围 | 类型 | 说明 |
| --- | --- | --- | --- |
| `type` | 所有 agent span | `str` | `agent` / `agent_iteration` / `agent_inference` / `agent_observation` / `agent_compaction` |
| `name` | 实体 span | `str` | 原始注册名 |
| `iteration` | `agent_iteration` 及其子 span | `int` | 轮次序号，1-based |
| `retry` | `agent_inference` | `int` | 重试序号，0-based |
| `compacted` | `agent_compaction` | `bool` | 本轮是否实际执行了压缩 |

## 编程接口

### Task：`phase()`

```python
from nexus.trace.phases import phase
```

`phase()` 支持三种用法：平铺调用、context manager、只关闭，三者可混用。

**平铺调用**：切换阶段，自动关闭上一个 phase span。

```python
phase('task.init')
instance = load_instance(...)

phase('task.agent')              # 自动关闭 task.init，开启 task.agent
agent_output = await task_input.run_agent(...)

phase('task.scoring')            # 自动关闭 task.agent，开启 task.scoring
score = evaluate(...)
phase()                          # 关闭 task.scoring，不开启新 span
```

**Context manager**：有明确作用域的子步骤，进入时执行与平铺调用相同的自动关闭逻辑，退出时立即关闭自身。

```python
phase('task.init')

with phase('task.init.get_runtime'):
    runtime = await ctx.get_runtime('main')
# get_runtime 已关闭，回到 task.init 层级

with phase('task.init.runtime_setup'):
    await runtime.call_function(setup_fn, ...)
# runtime_setup 已关闭

phase('task.agent')              # 关闭 task.init，开启 task.agent
```

**只关闭**：不传参数，关闭当前栈顶 span，用于正常流程的收尾。

```python
phase('task.scoring')
score = evaluate(...)
phase()                          # 显式关闭 task.scoring
```

#### 嵌套规则

当调用 `phase(new_name)` 时，从 `_active_span_stack` 栈顶往下查找：

1. 如果 `new_name` 以栈顶 span name + `.` 开头，则作为子 span 直接压栈。
2. 否则回退到共同祖先，依次关闭中间所有 span（标记 OK），再开启新 span。

当调用 `phase()` 不传参数时，关闭栈顶 span（标记 OK）。

#### 异常处理

`@observe` 的 `finally` 中，从 `_active_span_stack` 栈顶往下，关闭所有子 span 直到自身为止。正常退出标记 OK，异常退出标记
ERROR。这是通用的安全网，不限于 phase span。

#### 完整示例

```python
async def swe_task(task_input):
    phase('task.init')
    instance = load_instance(...)

    with phase('task.init.get_runtime'):
        runtime = await ctx.get_runtime('main')
    with phase('task.init.runtime_setup'):
        await runtime.call_function(setup_fn, ...)

    phase('task.agent')
    agent_output = await task_input.run_agent(...)

    phase('task.scoring')
    with phase('task.scoring.runtime_init'):
        eval_runtime = await ctx.get_runtime('eval')
    with phase('task.scoring.grade'):
        score = run_tests(eval_runtime, agent_output)
    phase()
```

### Agent：`span_context`

Agent 轮次直接使用 `span_context` context manager，不需要额外的 helper：

```python
for iteration in range(1, max_iterations + 1):
    with span_context('agent.iteration', attributes={
        'type': 'agent_iteration', 'iteration': iteration,
    }):
        with span_context('agent.inference', attributes={
            'type': 'agent_inference', 'iteration': iteration, 'retry': retry,
        }):
            response = await llm_call(...)

        with span_context('agent.observation', attributes={
            'type': 'agent_observation', 'iteration': iteration,
        }):
            tool_messages = await execute_tools(...)
```

Processor 需要 duration 时从 span 的 `start_time_unix_nano` 和 `end_time_unix_nano` 计算，无需手动计时。原有的
`_run_timed_stage` 废弃。

## Run 生命周期、结果和失败语义

runner 生命周期通过 run span 上的 `run.event` 表示，具体语义放在 `event_type` 中。常见事件包括：

- `run.status`
- `run.output`
- `run.error`
- `run.killed`
- `runtime.ready`
- `turn.submitted`、`turn.started`、`input.required`
- `agent.event`

`run.output` 的 payload 保存完整 `RunOutput`，包括 `success`、`error`、`task_output`、`agent_output`、
`processor_results` 和 `cache_stats`。

业务失败仍会产生 `RunOutput(success=false)` 和 `run.output`。如果初始化、runtime drain 或 sink 在形成 `RunOutput` 前失败，
journal 可能只有 `run.error/run.status`，此时 distributed execution 通过 `error_json` 报告失败，不生成 result pointer。cancel
时 live 会记录 `run.killed` 和 `run.status=killed`，distributed single 记录 `run.status=killed`。

不同 runner 的持久化方式不同：

| Runner | 结果和 trace |
| --- | --- |
| 普通 single CLI | `RunOutput` 直接返回或写入输出文件；是否额外记录 span 由业务 processor 决定。 |
| 本地 batch | `runs` 表保存结构化输出，`run_trace_events` 保存 raw span event，`run_logs` 保存日志块。 |
| distributed live/single | `executions.result_json` 只保存 external pointer，完整结果从 run trace backend 的 `run.output` 回读。 |

## Sink、Source 和 Live 投影

### Run Trace Backend

一个 run trace backend 由配套的 `RunTraceSink` 和 `RunTraceSource` 组成：

- sink 为单个 run 写入有序的 raw `SpanEventRecord`，关闭后返回可序列化 metadata。
- source 使用 metadata 回读 raw records，并从 `run.output` 恢复最终 `RunOutput`。
- backend 不保存 live API 的投影事件；投影可以随读取方演进而重新生成。

`RunTraceSession` 至少需要一个 backend sink。启用多个 sink 时，每个 sink 都接收相同的 raw records；任一 sink 的 start、
write、close 或 metadata 阶段失败都会使当前 execution 失败。读取时按 metadata 中的 source 顺序尝试，允许在某个 source
暂时不可用时回退到下一个。

### TDSQL 写入分片、队列容量与收尾等待时间

`TDSQLTraceSink` 和 `TDSQLBoundlessTraceSink` 默认在收尾时等待已入队的轨迹写入完成，最多等待
**1200 秒（20 分钟）**。可通过 backend 的 `sink_config` 单独覆盖：

```yaml
sink_config:
  # 省略 writer_shards：按单个 Manager 配置的内存自动选择 1 到 4 路
  # writer_shards: 8  # 显式配置可覆盖自动值，不受 4 路上限限制
  close_flush_timeout_seconds: 1200
  max_queue_items: 10000
  max_queue_bytes: 1610612736  # 1.5 GiB
```

四个参数也可在执行该 sink 的 Manager 进程环境中设置：

| sink 配置项 | 环境变量 | 默认值 |
| --- | --- | --- |
| `writer_shards` | `NEXUS_TRAJ_TDSQL_WRITER_SHARDS` | 自动：`max(1, min(Manager 内存字节数 // (4 * 1024**3), 4))`，无法获取时为 1 |
| `close_flush_timeout_seconds` | `NEXUS_TRAJ_TDSQL_CLOSE_FLUSH_TIMEOUT_SECONDS` | 1200 秒 |
| `max_queue_items` | `NEXUS_TRAJ_TDSQL_MAX_QUEUE_ITEMS` | 10000 条 |
| `max_queue_bytes` | `NEXUS_TRAJ_TDSQL_MAX_QUEUE_BYTES` | 1610612736 字节（1.5 GiB） |

每个参数的优先级均为显式 `sink_config` > 环境变量 > 默认值。等待时间必须为有限正数；分片数和两个容量参数必须为正整数，
非法值会在初始化时直接报错，不再将非法容量环境变量静默回退到默认值。环境变量在创建 sink / writer 时读取。
超时仍按轨迹收尾失败处理，延长等待不会补回已经丢弃的记录。
提交机上的本地 `export` 不会自动传递到远端 Manager，需通过 Manager 的环境配置注入。

容量限制针对**一个共享 writer 中仍在排队的记录**。条数或估算载荷字节数任一超限，都会拒绝新记录；
已出队、正在写库的批次不计入该队列容量，字节数也不是整个 Manager 的内存上限。
同一次 run 根据 `run_span_id` 的稳定哈希固定到一个分片；每个分片内部仍串行写入，不同分片可以并行。
数据库连接配置、dialect、分片数和两个容量参数相同且落在同一分片的 sink 共享 writer；其他组合使用独立的连接和队列。
每路保留完整限额，不均分：自动模式最多四路，可合计排队 40000 条 / 6 GiB 载荷。对同一数据库建议统一配置，避免产生多组独立队列。

Manager 直接使用现有启动配置 `NEXUS_DIST_ALLOCATION_SPEC_JSON` 中的 `resources.memory_bytes` 和
`resources.manager_count`，以 allocation 总内存除以 Manager 数量，得到提交时配置的单个 Manager 内存。
无需新增参数、读取 cgroup 或为此更新 Reconciler。用户仍通过 `--manager-memory-gib` 设置执行 Manager 内存，
通过 `--batch-manager-memory-gib` 设置批次 Manager 内存。
小于 8 GiB 为 1 路，8 至不足 12 GiB 为 2 路，12 至不足 16 GiB 为 3 路，16 GiB 及以上为 4 路；
缺失或无效的资源配置回退 1 路。每个 sink 创建后固定路数，不随剩余内存或负载变化。
每 4 GiB 一路是初始预算，并非进程内存硬限制；任务本身、序列化和正在写入的数据仍会额外占用内存。
单路模式保持原有共享、序号和收尾等待行为；多路的序号与收尾等待也各自独立，不会被其他分片的完成进度提前推进。

`source_package` 任务使用提交包中的 Nexus 代码，需升级提交包才能支持这个选项；只更新 Control API 不会改变
已运行任务的等待时间。旧版 sink 不识别此配置字段，因此兼容旧提交包时，应保留不传新字段的方式，让新版使用默认值，
或向 Manager 设置环境变量（旧版已支持两个容量环境变量，但不支持新的收尾时间、分片数变量）。
这四个参数只配置在 `sink_config`，不要放进同时传给 source 的公共 `config`。

### Live 投影

raw `SpanEventRecord` 是 live 的事实源，`RunTraceStreamEvent` 只是 SSE/API 视图：

```text
active live:
SpanEventRecord -> LiveRunTraceView -> RunTraceStreamEvent -> SSE

closed live:
RunTraceSource -> complete raw journal -> project_live_trace_records() -> RunTraceStreamEvent
```

active live 将 `LiveRunTraceView` 作为 internal sink，与 backend sinks 接收同一批 raw records，并实时维护有界投影 stream。
closed live 从可用 `RunTraceSource` 读取完整 journal，再用同一投影引擎重建结果。run span 上的生命周期事件保留原来的
`event_type`；普通 span 中需要展示的内容投影为 `trace.event`。

投影 cursor 与 raw `record_id` 属于不同序列。backend 只需保存 raw records，不保存 SSE payload，也不承担 live API 的兼容
格式。

## Runtime Trace

远端 runtime 调用会传播当前 `trace_id`、parent span 和 run identity。Worker 产生的 Nexus `SpanEvent` 经过 Worker
Socket.IO、Base runtime broadcaster 回到 Host，并在 Host `SpanController` 中重放，因此 runtime span 仍继承调用时的
run/task 层级。

托管 runner 在发布 `run.output` 和释放 runtime 前调用 trace drain barrier：

```text
Worker flush -> Base barrier -> Host replay complete -> run.output -> session close
```

barrier 保证调用前已被 transport 接收的 span event 到达 Host processor；sink close 再保证已接收 records 完成 flush。
它不是进程崩溃后的重传协议，也不提供跨故障的 exactly-once 保证。

## 附录：HyTuner 迁移

本节保留旧 span 到统一命名的迁移关系。大部分路径已采用统一命名；`swe_setup_agent` 仍有一个旧名称兼容例外，见
[迁移状态](#迁移状态)。

### Span Name 映射

| 旧 span name | 统一 span name | `type` |
| --- | --- | --- |
| `hytuner_task/get_runtime` | `task.init.get_runtime` | `task_phase` |
| `hytuner_task/run_pre_agent_script` | `task.init.runtime_setup` | `task_phase` |
| `hytuner_task/run_agent` | `task.agent` | `task_phase` |
| `hytuner_task/get_patch_from_runtime` | `task.agent.get_patch` | `task_phase` |
| `hytuner_task/eval_runtime_init` | `task.scoring.runtime_init` | `task_phase` |
| `hytuner_task/evaluate_and_grade` | `task.scoring.grade` | `task_phase` |
| `hytuner_agent/gen` | `agent.inference` | `agent_inference` |
| `hytuner_agent/tool:<name>` | *复用已有 tool span* | `tool` |
| `main_agent.inference` | `agent.inference` | `agent_inference` |
| `main_agent.observation` | `agent.observation` | `agent_observation` |

### Processor 过滤方式

```python
# 旧：通过 span name prefix 过滤（生产者标记）
if is_timeline_segment_span(span.name):  # name.startswith('hytuner_task/') or ...
    self.agent_timeline.append(...)

# 当前：通过 type attribute 过滤（消费者自行决定）
_TIMELINE_TYPES = {'task_phase', 'agent_inference', 'agent_observation', 'agent_compaction'}

if span.attributes.get('type') in _TIMELINE_TYPES:
    self.agent_timeline.append(...)
```

tool entity span 仍保留在 trace 树中，但当前 HyTuner timeline 使用 `agent_observation` 表示工具执行阶段，不单独收集 `tool`
type。

### 专属属性

HyTuner 专属的数据（非通用 span 属性）使用 `hytuner.` namespace：

| 旧属性 | 新属性 |
| --- | --- |
| `_hytuner_tool_args_json` | `hytuner.tool_args` |

### 迁移状态

以下旧常量和 helper 已删除：

- `DEFAULT_TASK_TIMELINE_PREFIX`（`hytuner_task/`）
- `DEFAULT_TIMELINE_SEGMENT_PREFIXES`
- `is_timeline_segment_span()`（基于 name prefix 过滤）
- `task_span()`（被 `phase()` 取代）
- `task_timeline_segment()`
- `TIMELINE_TASK_SEGMENT_TYPE`、`TIMELINE_LLM_GEN_TYPE`、`TIMELINE_AGENT_TOOL_TYPE`

`swe_setup_agent` 当前仍通过 `DEFAULT_AGENT_TIMELINE_PREFIX` 产生 `hytuner_agent/gen`。这是尚未迁移的兼容例外；完成迁移后
应改为 `agent.inference`，并删除该常量。

## 进一步阅读

- [YAML 配置参考](../reference/config-yaml.md)：业务 SpanProcessor 配置。
- [Langfuse Tracker](../how-to/use-langfuse-tracker.md)：将普通执行 span 投影到 Langfuse。
