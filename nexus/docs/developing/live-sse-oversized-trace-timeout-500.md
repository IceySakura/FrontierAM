# Live SSE 事件流被超大 Trace Payload 压垮引发 500（TimeoutError 分支）

- 状态：诊断完成，待实现修复
- 归属：`nexus/distributed/instance_manager` + `nexus/distributed/execution/executors` + `nexus/runner/live/trace` + `nexus/runner/live/sse`
- 触发实例（webdev_arena / webarena，prod）：`exe_710a6c5cdb1f4e508e87596579efd2ce`，assignment `asn_9fe60ef015d64494acdc613034cc5103`，manager pod `runner-prod-instance-manager-61-nj`
- 记录日期：2026-08-22
- 关联文档：`live-sse-cursor-expired-500.md`（同一个 `stream_events` 的**另一种**崩溃分支：cursor 过期。本文是 **TimeoutError** 分支，两者互补）

---

## 0. 一句话结论

一次**长时间、产出巨量文本的 Agent 任务**，把 live 观测事件（trace）流塞进了几十 MB 的超大数据；这些超大数据在 instance-manager 的**共享执行器事件循环**上做深拷贝 + JSON 序列化，把事件循环卡死超过 17 秒；SSE 拉取因此**超时（TimeoutError）**；而 SSE 处理代码**没有接住这个超时**，导致本来 200 的长连接以 **HTTP 500 中断**。Agent 任务本身完全正常、最终成功。

---

## 1. 现象（使用方视角）

- 一个 webarena live case 的实时事件流（SSE）在 **北京时间 2026-08-21 22:34:43** 收到最后一条事件（`id=123596`）后不再有新事件。
- 约 30 分钟后（**22:35:01** 实际断开，客户端侧因缓冲/超时表现为更晚才感知）SSE 连接以 **`unexpected EOF`** 断开。
- 客户端用**旧游标 `cursor=123596`** 疯狂重连，但补不回后续事件，最终 idle 超时把整个 case 判为失败并 cancel。
- **但 execution 本身在 23:04:12 正常 `succeeded`**（route closed、结果 sink 全部终态），产物完整。

即：**坏的是"实时看进度"这条观测通道，不是任务执行本身。**

## 2. 关键时间线（北京时间；日志 jsonl 为 UTC，此处已换算）

| 时间 | 来源 | 事件 |
|------|------|------|
| 21:58:54 | instance-manager | live SSE 长连接建立：`GET .../events?cursor=0`（全程仅此一条 live 连接） |
| 22:34:43 | SSE | 最后一条成功事件 `id=123596`，此后断流 |
| 22:35:01 | instance-manager | 该连接以 **`500`** 结束，**累计推送 body = 94,209,931 字节（≈94MB）**；同刻打印 `Unhandled instance manager internal API error` + `TimeoutError` traceback |
| 22:57~22:59 | 消费端 host 日志 | Agent 仍在正常干活（read_file 1268 行 / 连续 edit_file），LLM 反向桥心跳每 15s 正常 |
| 22:59:10 | 消费端 host 日志 | Agent `finish_reason=stop`，产出最终成品 → run.output |
| 22:59 + hold 300s ≈ 23:04 | control | `hold_after_task=300s` 结束 → `run.status=closed` |
| 23:04:12 | control | execution `succeeded`，route closed |
| 23:04:44~23:05:x | instance-manager | 消费端转而请求终态 artifact 分支 `run-trace/events?cursor=123596`，每秒 3~4 次空转、全部 `200 size=230`（空）、cursor 恒定不前进（补不回终态） |

**要点**：22:34:43 之后 Agent 又健康运行了 25 分钟才结束（消费端 host 日志证明）。所以断流点是 **22:35:01 的那次 500**，不是"任务卡住"，也不是"进程死了"。

## 3. 根因链（已由 traceback 完整证据闭环）

### 3.1 决定性证据：22:35:01 的 traceback

```
asyncio.exceptions.CancelledError
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
  File ".../instance_manager/internal.py", line 66, in error_middleware
  File ".../instance_manager/internal.py", line 60, in auth_middleware
  File ".../instance_manager/internal.py", line 352, in stream_events
    events = await service.executor.collect_events_after(assignment_id, cursor, timeout=15.0)
  File ".../execution/executors/asyncio_executor.py", line 146, in collect_events_after
  File ".../execution/executors/asyncio_executor.py", line 290, in _call_sse_driver
  File ".../execution/executors/asyncio_executor.py", line 190, in _run
    return await await_threadsafe(future, timeout=deadline)
  File ".../utils/threadsafe.py", line 16, in await_threadsafe
  File ".../asyncio/tasks.py", line 506, in wait_for
  File ".../asyncio/timeouts.py", line 116, in __aexit__
    raise TimeoutError from exc_val
TimeoutError
```

紧跟其后的 access log：

```
"GET /v1/internal/executions/asn_9fe.../events?cursor=0 HTTP/1.1" 500 94209931
```

### 3.2 逐步拆解

1. **巨型观测数据从何而来**：该任务是 HTML 生成类长任务（`hycode_dynamicagent` + `deepseek-v4-pro` + `reasoning_effort=max`）。单轮 LLM 的思考内容（`reasoning_content`）就有 **15,434 字符**，工具调用里 `read_file index.html` 返回 **1268 行**整份 HTML、每步 `edit_file` 带完整 diff。这些都进入 `openai_completion` span 的 `attributes`。

2. **live 投影把整个 span 原样外推**：`LiveRunTraceProjector._project_trace_event`（`runner/live/trace.py:88-126`）把 `openai_completion` 的 SpanStart / SpanEnd（含**完整 attributes**）投影成 `trace.event`。投影过程中 `RunTraceProjectionEngine._apply_record` 对 span 做 `model_copy(deep=True)`（`trace.py:183/193/198`）——**深拷贝几十 MB 对象**。

3. **拉取时再序列化一遍**：SSE 每批事件通过 `sse_bytes()`（`runner/live/sse.py:37-45`）编码，里面 `deepcopy(event.data)` + `json.dumps(...)` **又把巨型对象拷贝、序列化一遍**。

4. **卡死的是"共享的单线程执行器事件循环"**：生产用的是 `AsyncioExecutor`（`execution/executors/asyncio_executor.py`），它有一个**独立线程 + 单一事件循环**（`_run_loop`），**同一个 pod 上所有 live execution 共享这个 loop**。上面的 `deepcopy + json.dumps` 是**纯 CPU 同步操作**，一旦对象达到几十 MB，就把这个 loop 独占阻塞。

5. **collect 超时**：SSE 处理在 aiohttp 主循环上，通过 `run_coroutine_threadsafe` 把"取事件"提交到执行器 loop，并用 `await_threadsafe(future, timeout=15+2=17s)`（`asyncio_executor.py:292` + `threadsafe.py:16`）等待。执行器 loop 被 CPU 占死，17 秒内没返回 → `asyncio.wait_for` 抛 **`TimeoutError`**。

6. **没接住超时 → 500**：`stream_events`（`internal.py:343`）的循环里，内层只 `except RunTraceCursorExpired`，外层只 `except (ConnectionResetError, asyncio.CancelledError)`（`internal.py:372`）。**`TimeoutError` 两个都不匹配**，冒泡到 `error_middleware`。但此时响应早已 `prepare()`（200 头已发）且已写 94MB body，中间件**发不出 500 状态行**，aiohttp 只能记 `500` 并 **abort 连接**。

7. **客户端 unexpected EOF + 重连补不回**：连接被强行中断，客户端收到 `unexpected EOF`（22:35:01）。之后重连用旧 cursor，run 还在跑；等 23:04 route closed 后转终态 artifact 分支，但 **live 阶段的 cursor 与 artifact 全量重投影的编号体系对不上**，永远补不回 → 空转风暴。

### 3.3 这是系统性问题，不是偶发

同一个 pod 当天出现 **10 次**同类 `Unhandled instance manager internal API error`（10:28 / 10:39 / 11:06 / 11:30 / 11:37 / 11:44 / 15:27 / 17:02 / 18:23 / 22:35）。凡是"长任务 + 大文本产出"都会触发。且因执行器 loop 是**共享**的，一个大 payload 的序列化会**连累同 pod 其它 live execution** 的事件流。

## 4. 影响面

- **直接影响**：live SSE 实时事件流（`GET /v1/internal/executions/{assignment_id}/events` 及经 gateway 暴露到应用面的 `/v1/executions/{id}/live/events`）。表现为"事件流突然掐断 → 客户端 EOF/5xx → 重连补不回"。
- **不影响**：execution 生命周期、tdsql / artifact sink、最终结果发布（本例已印证任务成功、产物完整）。
- **连带**：同 pod 其它 live execution 的事件流也可能被卡顿；manager 日志出现 stack trace，增加 oncall 噪声。
- **概率**：随 run 时长、turn 数、单轮文本量增长而升高；`hycode_dynamicagent` + 高 reasoning + 长 session + 大文件编辑尤其容易命中。

## 5. 修复方案（分层，按优先级）

### P0：`stream_events` 接住 `TimeoutError`（止血，最小改动、最低风险）

在 `internal.py::stream_events` 循环中，把执行器 collect 的 `TimeoutError` 当作"本轮没有新事件"处理——写 `: keep-alive` 并 `continue`，**绝不让它冒泡成 500 abort**。

- 语义正确：collect 超时本就等价于"这段时间没有新事件可推"，keep-alive 正是 SSE 该做的。
- 与已落地的 `RunTraceCursorExpired` 分支并列，一起放进内层 try。
- 单独这一改，就能让连接在执行器 loop 短暂繁忙时**存活下来**，不再 500。

### P1：给单条 trace event 的数据设**体积上限**（根因，必须做）

在 live 投影层，对 `openai_completion` 的 `attributes` 中的大字段（`messages` content / `reasoning_content` / `tool_result` / diff 等）做**截断为摘要 + 标记 `truncated=true`**，为单条 `trace.event` 设一个硬上限（例如 64KB）。

- 原始完整数据**仍走 tdsql raw records 持久化**，审计/回放不受损；被削减的只是"实时推给前端看"的那一份。
- 从源头把 in-memory deque 占用、深拷贝成本、JSON 序列化成本一起降下来，**根治 loop 被卡死**。
- 需要与使用方对齐：截断阈值、哪些字段截断、截断后前端展示形态（例如显示"内容过大，已省略 N 字符"）。

### P2：把 SSE 序列化移出共享执行器 loop（消除放大面，重要）

`sse_bytes` 的 `deepcopy + json.dumps` 不应在共享的 `AsyncioExecutor` loop 上同步执行，避免"一个大流拖垮同 loop 其它 execution"。P1 落地后单事件已经变小，风险大降；P2 进一步把编码工作挪走（如在拉取侧/主 loop 编码，或线程池），彻底隔离。

### P3：重连可恢复（次要，体验优化）

统一 live（in-memory 投影）与 artifact（tdsql 全量重投影）两个后端的 cursor 语义，使断连重连后能补回 `run.output` / `run.status=closed` 等终态事件，而不是空转。

## 6. 测试计划

配合 P0（必做）：
- `tests/unit/distributed/instance_manager/test_internal_stream_events.py`：Fake executor 第一次 `collect_events_after` 抛 `TimeoutError`，断言 `stream_events` 写出 `: keep-alive` 并继续循环、**不产生 500、不打 ERROR 日志**；第二次返回一小批事件正常写出。

配合 P1（必做）：
- `tests/unit/runner/live/test_trace_projection.py`：构造超大 attributes 的 `openai_completion` span，断言投影出的 `trace.event` data 被截断到上限且带 `truncated` 标记；同时断言 tdsql raw records 仍是完整数据（不被截断）。
- `sse_bytes` 对超大 event 的编码有界性测试。

配合 P3（可选）：
- live 与 artifact 两后端在同一 run 上产出的 cursor 编号一致性测试。

## 7. 风险与兼容性

- **P0**：只改 SSE 内部错误处理，对客户端等价于多收到 keep-alive，无副作用；不涉及 API / DB / 镜像。
- **P1**：改变实时推送的数据形态（大字段被摘要），需与前端/webarena 使用方约定展示；持久化数据不变。属 `nexus/runtime/**` 之外的 host/instance-manager 侧逻辑，注意确认镜像重建触发面。
- **P2/P3**：内部实现优化，无对外契约变更。

## 8. 参考位置

- `nexus/distributed/instance_manager/internal.py`：`stream_events`（~L343），`error_middleware`（~L65），`_cursor_expired_sse`（~L377）
- `nexus/distributed/execution/executors/asyncio_executor.py`：`collect_events_after`(L143) → `_call_sse_driver`(L289，deadline=timeout+2) → `_run`(L184) → `await_threadsafe`
- `nexus/distributed/utils/threadsafe.py`：`await_threadsafe`（L12，`asyncio.wait_for` 抛 `TimeoutError`）
- `nexus/distributed/execution/drivers/nexus_live.py`：`collect_sse_events_after`（L153）
- `nexus/runner/live/trace.py`：`LiveRunTraceProjector._project_trace_event`（L88-126）、`RunTraceProjectionEngine._apply_record`（L169-199，`model_copy(deep=True)`）、`LiveRunTraceView`（L201-）
- `nexus/runner/live/sse.py`：`sse_bytes`（L37-45，`deepcopy` + `json.dumps`）
