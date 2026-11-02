# Unified Trace Processor and Sink 设计

本文定义统一的 run trace 方案。raw `SpanEventRecord` 是唯一事实源；live SSE、UI 和 distributed 控制逻辑都是其投影视图。

## 1. Concepts

trace processor 只有两个 scope：

- `default`：task、agent、openai、tool、runtime 等普通 span。
- `run`：`run.live`、`run.single`、后续 `run.batch` synthetic span。

scope 只决定 processor 分发范围。一个受管 run 的层级为：

```text
run.live / run.single
└── task
    └── agent / openai / tool / runtime ...
```

multi-turn 类 task 在普通 agent span 下增加每轮边界：

```text
agent.*_multiturn                    # host，稳定的外层 agent span
├── agent.turn                      # host，一次 runtime 调用
├── agent.turn
└── events: agent.stream            # runtime reader 产生的 stream-json Event
```

`agent.turn` 属于 `default` scope，记录 `agent`、`session_id`、`turn_index`，不记录 prompt 正文。`agent.stream` 仍是
外层 agent span 上的 Event，不是 span，也不承诺严格归属某个 turn。等待下一次用户输入发生在 `agent.turn` 之外，因此 idle
时间不计入模型执行时长。

`RunTraceSession.activate()` 才建立该父子关系：run span 和 task span 使用同一个 `trace_id`，task 的
`parent_span_id` 是 `run_span_id`。普通 `nexus.runner.single.run()` 不创建 run span，task 仍是 trace root。

run 生命周期写入 synthetic run span 的 `run.event`：

```text
Event.name = "run.event"
Event.attributes = {
  event_type: "run.status" | "run.output" | "run.error" | "turn.submitted" | ...,
  event_data: {...}
}
```

受管普通 span 同时带显式归属字段：

```text
_run_id
_run_span_id
_run_kind       # live | single | batch
```

这些字段会通过 `RemoteTraceContext` 传到 runtime。归属只按 `_run_span_id` 及另外两个字段精确匹配，不使用
`trace_id -> run_id` 推断。

## 2. Components

模块职责：

- `nexus.trace`：`SpanController`、scope、span event、`SpanEventRecord`、通用 `TraceSink/TraceSource` protocol。
- `nexus.runner.trace.trace`：`RunTraceSink/Source`、`RunTraceProcessor`、`RunTraceSession`、raw stream 与 projector 抽象。
- `nexus.runner.trace.local`：`RunTraceLocalSink` 和 `RunTraceLocalSource`。
- `nexus.runner.live.trace`：`LiveRunTraceProjector`、`RunTraceProjectionEngine`、`LiveRunTraceView`。
- `nexus.distributed.execution.trace_backends`：InstanceManager backend registry、sink 选择、source 路由和 result pointer。

继承关系：

```text
nexus.trace.TraceSink protocol
└── RunTraceSink ABC
    ├── RunTraceLocalSink
    ├── RunTraceInMemoryJournal       # internal sink
    └── LiveRunTraceView              # internal sink + projection stream

nexus.trace.TraceSource protocol
└── RunTraceSource ABC
    └── RunTraceLocalSource
```

### RunTraceProcessor

`RunTraceProcessor` 是进程级常驻 processor，同时订阅 `default` 和 `run` scope。它只负责：

1. 按 `run_span_id` 将 span callback 归属到 active session。
2. 为每个 run 分配连续 `record_id`，生成 raw `SpanEventRecord`。
3. 在 owner event loop 的单 worker 中按序 fan-out 到该 session 的全部 sink。
4. 在 close 时 drain 已准入事件、关闭 sink，并移除 state。

processor callback 不执行 backend IO，不重新 emit span，也不维护 live 投影。跨线程 callback 通过线程安全 registry、run state
锁和 owner-loop wakeup 收口；锁不跨 `await`。队列有上限，溢出会使 run 失败，不静默丢弃。

`RUN_SCOPE` 的 update event 只分发，不累计到 active run span，避免长 live run 持有完整历史。

### RunTraceSession

每个 live/single run 持有一个 `RunTraceSession`：

```text
start
  -> reserve a non-accepting run state
  -> start all sinks and the per-run worker
  -> enable event admission
  -> START run.<kind>

activate
  -> inject _run_* attributes
  -> make run span the task parent

emit
  -> UPDATE run span with run.event
  -> flush accepted records

close
  -> END run span
  -> stop admission and drain worker
  -> close all sinks
  -> validate and cache backend metadata envelope
  -> remove run state
```

session 接收两类 sink：

- `backend_sinks: dict[name, RunTraceSink]`：非空、具名，metadata 会进入 artifact envelope。
- `internal_sinks: sequence[RunTraceSink]`：内存 journal、live view 等进程内消费者，不进入 envelope。

所有启用的 backend sink 都是必需项。没有 binding、best-effort、durable/outcome 等第二套状态模型。任一 sink 的
start/write/close/metadata 失败都会使 session 失败。metadata 只在成功 close 后可读取。

### Raw IO

`RunTraceSink` 写入 raw `SpanEventRecord`，不是 live API event：

```text
record_id
event_type       # start | update | end
time_unix_nano
trace_id
span_id
parent_span_id
event_data       # 原始 SpanStart/Update/EndEvent JSON
```

record 模型不可重新赋值；processor 为每个 sink 提供隔离的深拷贝，避免自定义 sink 修改嵌套数据后污染其他 sink。
`RunTraceSource.read_events_after()` 返回同一 raw 模型，`read_result()` 从 synthetic run span 的最后一个
`run.output` 恢复 `RunOutput`。

### Live Projection

live 投影不是事实源：

- active run：`LiveRunTraceView` 作为 internal sink 接收 raw record 并维护投影 stream。
- closed run：source 从 cursor 0 读取完整 raw journal，`RunTraceProjectionEngine` 重建 span 状态并重新投影。
- distributed live driver：读取 `RunTraceInMemoryJournal` 的 raw records，解码 `run.event`，不消费投影 stream。
- SSE/UI：读取投影 stream，对外仍输出 `{id, type, time, data}`。

raw `record_id` 和 projection `id` 是独立 cursor，因为一个 raw record 可以投影成零个或多个 API event。内部不再定义
`TrackEvent`、`EventLog`、`LiveEventStream` 或 live source 包装层。

## 3. Lifecycles

### Live

```text
NexusLiveDriver / LiveRunManager
  -> resolve named backend sinks
  -> LiveRun owns RunTraceSession + raw journal + live view
  -> session.start()
  -> session.activate(): run.live becomes task parent
  -> run events and ordinary spans enter the same raw sink chain
  -> runtime trace barrier
  -> run.output / terminal status
  -> session.close()
  -> publish envelope or persist local RunInfo metadata
```

`LiveRun` 持有 session；`events` 只是 active projection 的 API 入口。closed events 始终从 raw source 重建。
现代 multi-turn agent 在 host 为每次 runtime 调用创建 `agent.turn`。host 在创建 turn 前捕获外层 agent 的
`RemoteTraceContext`，并通过 `InvokeRequest` 显式传给 runtime；runtime 因此继续使用稳定的外层 agent context，不继承临时 turn。
持久 stream-json reader 无需维护 active-turn 状态，仍实时把 `agent.stream` Event 追加到外层 agent span。该 MVP 能准确记录每轮
调用的时长和状态，但不提供 stream Event 的严格逐轮归属；若需要该能力，应另行扩展 runtime session 协议。`agent.stream` Event 随
runtime span barrier 一起完成可靠收尾，不额外投影成重复的 turn API event。通用 `agent_event` 仅作为旧 live runtime 的输入兼容
保留；`LiveRun` 收到后将其写成 `agent.event` run event，使其仍进入同一 processor/sink，但不为这条旧通道另建一套 ACK 或重传协议。

runtime transport 使用显式 barrier：

```text
worker span processor queue
  -> batched Socket.IO call with ACK
  -> host broadcaster queue + flush
  -> host emits a sequenced barrier marker
  -> trace:flush returns the target batch id
  -> RemoteRuntime waits until local replay reaches that batch id
```

worker 不再为每个 span 创建 task。emitter 先绑定 owner loop，工作线程 callback 通过线程安全入队；queue overflow、发送失败和
本地 `SpanController` replay 失败都会在 barrier 暴露。client 以连接内首个 batch 建立基线，之后只接受连续 batch；跳号、重复、
倒序、非法 batch id 或已连接后的断线都使 drain 失败。MVP 不做断线重传，无法证明完整时不发布完整 Trace 指针；独立发布成功的业务结果 Artifact 仍可读取。host、worker 和 client 使用同一
barrier 协议，runtime image 必须与本次 client 代码一并升级；缺少 batch id 的旧协议会 fail fast。
`on_run_output` 先为结果后端捕获完整输出，再 drain，保证此前 span 排在 `run.output` 前；`RunHooks.on_context_closing` 在 runtime provider 释放前再做最终
drain，因此成功、失败和取消都不会依赖终态后的延迟 span。每次 drain 内仅对指向同一 runtime 对象的别名去重，不跨两个边界跳过
最终 barrier。`run_single()` 捕获这两个 post-run hook 的普通异常并记录到 `RunHooks.hook_failures`，保证业务异常或取消仍是主结果；
受管 live 在 `run_single()` 返回后调用 `raise_if_failed()`，将 hook failure 作为 trace pipeline failure 结束 run。

### Single

```text
NexusSingleDriver
  -> resolve named backend sinks
  -> RunTraceSession(run_kind="single", run_id=execution_id)
  -> session.activate(): run.single becomes task parent
  -> run_single(..., on_run_output, on_context_closing)
  -> drain runtime trace
  -> emit complete run.output
  -> publish result artifact independently
  -> emit terminal run.status
  -> close raw trace sinks
  -> reporter stores only a trace result pointer
```

`run.output.output` 是完整 `RunOutput`，包含：

```text
success
error
task_output
agent_output
processor_results
cache_stats
```

`executions.result_json` 不保存完整 single output，只保存带 artifact envelope 的 external pointer。普通直接调用
`nexus.runner.single.run()` 不启用该 session，也不改变 task-root 行为。distributed single 在 `run_single()` 返回后检查
`RunHooks.raise_if_failed()`；任一 drain/output hook 失败都使 execution failed。若结果 Artifact 已发布，则保留仅包含
结果后端、标记 `trace_complete=false` 的指针；不会将缺失的原始轨迹宣称为完整。

### Future Batch

后续 `nexus_batch` driver 负责：

1. batch loader 生成 child `nexus_single` 配置并记录 generation complete。
2. 按窗口向 scheduler 提交小批 child executions。
3. 汇总 child result/error/trace pointer，直到全部完成。

`run.batch` 是 batch 控制生命周期；每个 item 仍由独立 `run.single` 保存 trace 和 output。本期不实现 batch driver。

## 4. Compatibility

本分支尚未形成线上内部格式，因此不保留以下别名或 fallback：

- `LIVE_SCOPE`、`live.run`、`live.event`。
- `_live_run_id`、`_live_span_id`。
- `trace_id -> run_id` 归属推断。
- 旧 `events_path/result_path`、artifact writer 或 bridge。
- 旧 `nexus.runner.trace.file/tracker` 模块和旧类名。

保留的是对外 live API：route、SSE event name 及 `{id,type,time,data}` JSON 结构不变。兼容层只处理外部 API，不参与内部
trace 事实模型。

## 5. Distributed Storage and Artifact Metadata

distributed DB 保存 execution、assignment、command、runtime_info、result pointer 和 error 等控制面数据；raw trace 不写入
`executions` 表。

成功 close 后生成 version 4 envelope：

```json
{
  "version": 4,
  "run_id": "exec-1",
  "run_span_id": "span-1",
  "run_kind": "single",
  "sinks": [
    {"name": "local", "metadata": {"trace_events_path": "..."}}
  ]
}
```

envelope 只保存 sink 名称和 sink 自己的 metadata，不重复保存 backend、required、durable、status 或 outcome。读取时按 envelope
顺序尝试已注册 source；一个 source 不可用时可回退到下一个。

### Registration

InstanceManager 配置可信的 class URI 和 backend 参数：

```yaml
instanceManager:
  defaultRunTraceSinks: [local]
  runTraceSinks:
    local:
      sinkClass: nexus.runner.trace.local::RunTraceLocalSink
      sourceClass: nexus.runner.trace.local::RunTraceLocalSource
      config:
        root: /data/nexus-distributed/run-trace-artifacts
    trace-db:
      sinkClass: my_trace::DatabaseRunTraceSink
      sourceClass: my_trace::DatabaseRunTraceSource
      config:
        endpoint: trace-db
      sinkConfig:
        batch_size: 100
      sourceConfig:
        page_size: 1000
```

自定义 backend 的最小接口为：

```python
class DatabaseRunTraceSink(RunTraceSink):
    def __init__(self, *, execution_id, assignment_id, attempt_id, endpoint, batch_size=100): ...
    async def start(self, context: RunTraceContext) -> None: ...
    async def on_event(self, context: RunTraceContext, event: SpanEventRecord) -> None: ...
    async def close(self, context: RunTraceContext) -> None: ...
    def metadata(self) -> dict: ...  # JSON，可供 source 定位本次 journal

class DatabaseRunTraceSource(RunTraceSource):
    def __init__(self, *, endpoint, page_size=1000): ...
    async def read_events_after(self, metadata, cursor, *, limit=None) -> list[SpanEventRecord]: ...
    async def read_result(self, metadata) -> dict: ...
```

registry 启动流程：

1. 加载内置 `local` 定义。
2. 按名称深度合并 `runTraceSinks`；`sinkConfig/sourceConfig` 分别覆盖公共 `config`。
3. 动态 import class，并要求显式继承 `RunTraceSink/RunTraceSource`。
4. 构造共享 source 和 per-run sink factory。
5. 校验默认 sink 名称；未知 class、错误参数或重复注册均 fail fast。

sink 构造函数除配置外会收到系统注入的 `execution_id`、`assignment_id`、`attempt_id`，用户配置不能覆盖这三个字段。
source 必须支持并发读取。凭证由 InstanceManager 环境或 Secret 注入，任务不能提交 class URI、路径或凭证。

用户的 driver config 只选择已注册名称：

```yaml
driver_config:
  run_trace_backends:
    sinks: [local, trace-db]
```

省略时使用 `[local]`；显式空列表、重复名称和未知名称都无效。所有选择的 sink 同时启用且都必须成功。

### Local Backend

`RunTraceLocalSink` 写：

```text
{root}/{execution_id}/{attempt_id}/{assignment_id}/trace_events.jsonl
{root}/{execution_id}/{attempt_id}/{assignment_id}/manifest.json
```

local source 校验 artifact path 不能逃逸配置 root。local backend 适合本地和开发环境；其可用性取决于挂载存储的生命周期，
不会伪装成跨 Pod 的持久 backend。

本地 live server 的 SQLite 只保存 RunInfo 和 version 4 envelope，不保存 event 表。active events 来自内存 view，closed events
由 `RunTraceLocalSource` 读取 raw JSONL 后重放。

## 6. Failure, Cancel, and Output Semantics

- sink start 半失败时，已尝试及已成功启动的 sink 都会进入回滚 close。
- sink write 失败后停止向该 sink 继续写，但继续 drain 其他 sink；最终 session 仍失败。
- sink close 或 metadata 失败同样使 session 失败，不能发布无效 pointer。
- sink 意外抛出的 `CancelledError` 作为 sink failure 记录；只有 processor 自身真的收到 task cancel 时才传播取消。
- close 使用可重复 await 的 shielded task；调用方取消不会中断底层 drain 和 sink close。
- raw processor queue、local writer queue、Socket.IO emitter queue 溢出都显式失败，不静默丢事件。
- runtime Socket.IO 未连接或 transport flush 失败时，受管 live/single run 失败；显式关闭 runtime WebSocket tracing 时 drain 为 no-op。
- `run_single()` 不让 post-run hook 异常覆盖业务 `RunOutput`、业务异常或取消；`on_run_output` 和 `on_context_closing` 的普通异常按阶段
  记录到 `RunHooks.hook_failures`。managed live/distributed single 必须在返回后调用 `raise_if_failed()`，并将其作为 trace pipeline
  failure；直接嵌入方可检查同一侧信道。`CancelledError` 不进入该列表，继续按取消传播。
- success 捕获业务输出、写 `run.output`，先发布结果 Artifact，再关闭原始 Trace，最后 reporter complete。
- task failure 写完整失败 `run.output` 和 `run.status=failed`；execution 以 failed 结束，同时在 `result_json` 保存 trace pointer、
  在 `error_json` 保存业务错误。unexpected exception 写 `run.error` 后失败；错误路径仍尝试发布捕获的业务输出，
  仅保留已经成功发布的 Artifact 指针。没有 Artifact 后端时仍遵循原始 Trace 完整关闭后才发布指针的规则。
- trace pipeline 已失败时不再向同一 session 递归写 `run.error`，而是直接置 failed、close 并通知 terminal waiter。
- cancel 尽量写 `run.killed/run.status=killed` 并 close；`LiveRun.kill()` 不另设内部 10 秒期限，统一服从 executor 的
  `stop_timeout_seconds`。该预算必须大于 runtime barrier timeout；强制超时取消不能发布完整 artifact pointer。

processor 不因 sink 失败递归 emit trace。driver 通过 session 的结果发布接口先保存输出，不直接操作原始 sink；Execution 最终状态仍在收尾后报告。

结果压缩与 TDSQL 超限省略、读取/导出兼容和发布顺序见 [大结果处理](../cloud/troubleshooting.md#大结果tdsql-output-省略与压缩-artifact)。

## 7. Non-goals and Tests

非目标：

- 本期不实现 `nexus_batch` driver。
- 本期不实现 DB/COS backend，只提供注册接口。
- 本期不迁移 `DatabaseSpanProcessor`；后续将其接入标准 `TraceSink`，不在本方案旁建 writer processor。
- 本期不做 scheduler sink capability 匹配；同一 pool 的 InstanceManager 必须有一致 registry。
- 本期不保证未挂持久卷的 local backend 在 Pod 重建后仍可读。

测试至少覆盖：

- `default/run` scope 分发，以及 run span、task span 的 trace/parent 关系。
- plain single 保持 task root，remote task 继承 run 归属和层级。
- 无 `_run_*` 的普通 span 不会被 session 接收；相同 `run_id` 的并发 session 按 `run_span_id` 隔离。
- 跨线程 callback 顺序、连续 raw cursor、队列溢出和 close state 回收。
- 多 sink fan-out、数据隔离及 start/write/close/metadata 失败。
- active live 与 closed raw replay 的投影一致，raw/projection cursor 相互独立。
- local source 的 path 校验、raw replay 和 `run.output` 读取。
- 动态 class import、默认 local、配置深度合并、空/重复/未知 sink 校验。
- distributed live/single 只发布 version 4 envelope 和 result pointer。
- worker/host/client runtime barrier、跨线程 emitter、sequenced local replay、transport failure 和 context-closing drain。
- post-run hook failure 不覆盖业务结果或取消，managed live/single 能检测失败且不发布不完整 pointer。
- multi-turn 的 host `agent.turn`、逐轮状态和 idle 边界；runtime 调用显式沿用外层 agent context，`agent.stream` Event 保持原有
  实时链路。
