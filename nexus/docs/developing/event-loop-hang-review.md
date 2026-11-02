# Nexus event-loop hang 静态审查与治理建议

> 审查状态：静态审查完成，故障注入尚未执行
>
> 代码基线：`cloud` @ `e319aab4ac2aaa9b212691707626159162a0d909`
>
> 最后更新：2026-08-10

## 1. 审查结论

本次审查覆盖本地 Runner、Live、Trace、Runtime、Provider、MCP/HTTP/COS，以及 distributed 的
Executor、Scheduler、Reconciler、Instance Manager、Batch、Gateway、Artifact 和进程关闭流程。

当前仍存在会让任务、控制循环或整个服务表现为“挂住”的代码路径。风险主要分为三类：

| 类型 | 判定标准 | event loop 状态 |
| --- | --- | --- |
| 同步冻结 | loop 线程执行同步 I/O、CPU 重活、`time.sleep()` 或死循环 | 无法调度其他 task |
| await 停滞 | 关键 coroutine 等待 DB、网络、future、进程或 cleanup，且没有可靠结束边界 | loop 仍可调度，但业务不再推进 |
| CPU starvation | 大量同步过滤、排序、拷贝或序列化长时间不让出执行权 | loop 延迟显著上升，但不一定永久停止 |

最需要优先处理的是以下五条故障链：

1. distributed 默认 `AsyncioExecutor` 让同一 Manager 的 execution 共用一个 loop。一个同步阻塞的用户任务
   可以冻结整组 execution；跨 loop bridge 又缺少调用方 deadline，停止和自栅栏无法保证完成。
2. Batch 控制逻辑被送回 Instance Manager 主 loop，并在数据库事务和行锁期间同步读取、解析 JSONL，可能同时
   延迟 heartbeat、assignment poll 和 command。
3. Application Gateway 和 Generic Artifact Store 在 HTTP/Reconciler loop 内直接执行文件读写与 hash，慢盘或
   文件系统停滞会冻结对应服务。
4. Live Server 在 HTTP 主 loop 直接运行 `run_single()`；Runtime Server 的 venv 文件重活、worker PIPE 和
   pending future 生命周期也存在同步冻结或长期停滞风险。
5. Trace、DB、MCP/HTTP/COS、provider 和 shutdown 的部分关键 await 缺少端到端绝对期限。

没有发现 Scheduler 或 Reconciler 中另一个可以由当前代码直接证明的同步永久死锁。Scheduler 的全量 Manager
过滤和排序属于规模相关风险；Reconciler 已按 domain 隔离 task 并设置 tick timeout，剩余问题主要来自 DB 和
cancellation cleanup 可能突破软 timeout。

## 2. 范围与判定原则

文中的“本地/共用逻辑”指不依赖 distributed 控制面的 Runner、Live、Trace、Runtime 等通用实现。本地 CLI
可以直接触发这些问题，distributed execution 复用它们时同样受影响。

“Distributed 专属逻辑”指由 distributed 的执行隔离、控制面协调、Manager/Batch、Gateway/Artifact 或关闭协议
引入的风险。

主风险清单只收录：

- 直接运行在 asyncio loop 上、可能长时间不返回的同步工作；
- 没有调用方绝对 deadline，或超时后无法完成恢复的关键 await；
- timeout/cancellation 之后仍继续无界等待的关闭流程；
- 有明确复杂度和共享故障域的规模型 CPU starvation。

资源泄漏、生命周期状态不一致、OOM、磁盘或 inode 耗尽、探针检测盲区、Python 兼容性以及发布 Job 停滞不计入
event-loop hang 主清单。它们并非不重要，而是产生机制、影响范围和修复闭环与 loop hang 不同，统一列在第 7 节。

## 3. 风险总览

| ID | 范围 | 组件 | 类型 | 优先级 | 主要影响 |
| --- | --- | --- | --- | --- | --- |
| LOC-01 | 本地/共用 | Runner、task、agent、tool | 同步冻结 | P1 | 当前 run loop 停止调度 |
| LOC-02 | 本地 | Live Server | 同步冻结 / await 停滞 | P1 | run、health、kill、SSE 同时失效 |
| LOC-03 | 本地/共用 | Trace、RunTrace、Socket.IO | 同步冻结 / await 停滞 | P1 | run 终态、trace flush 或 shutdown 停滞 |
| LOC-04 | 本地/共用 | Runtime Server、VenvManager | 同步冻结 | P1 | Runtime HTTP、invoke 和 worker 管理一起停顿 |
| LOC-05 | 本地/共用 | Runtime worker | 进程互锁 / await 停滞 | P1 | invoke 永不终态，Runtime shutdown 卡住 |
| LOC-06 | 本地/共用 | Provider、MCP、AsyncHTTP、COS | await 停滞 | P1/P2 | 单任务或容量槽长期占用 |
| LOC-07 | 本地/共用 | Python logging | 同步冻结 | P2，慢盘时 P1 | 记录日志的进程 loop 被 handler 阻塞 |
| DIST-01 | distributed | AsyncioExecutor | 同步冻结放大 | P1 | 同一 Manager 的全部 execution 同时冻结 |
| DIST-02 | distributed | 跨 loop bridge、self-fence | await 停滞 | P1 | stop、自栅栏和进程退出不能保证完成 |
| DIST-03 | distributed | Scheduler、Reconciler、DB、shard lease | await 停滞 | P1 | 控制循环、租约续期或请求长期不推进 |
| DIST-04 | distributed | Scheduler | CPU starvation | P2，大规模时 P1 | tick 和 lease 续期延迟 |
| DIST-05 | distributed | Instance Manager、Batch | 同步冻结 | P1 | Manager heartbeat、poll、command 同时受阻 |
| DIST-06 | distributed | Gateway、Artifact maintenance | 同步冻结 / CPU starvation | P1；summary 路径 P2 | HTTP/Reconciler loop 停顿 |
| DIST-07 | distributed | shutdown、self-fence | await 停滞 | P2，有持久状态时 P1 | Pod 只能依赖最终 SIGKILL，终态可能丢失 |

## 4. 本地与共用逻辑风险

### LOC-01：Runner 与用户代码共享 execution loop

`run_single()` 在当前 loop 直接 `await task_fn(task_input)`。外层 async timeout 只有在 loop 仍能调度时才会触发，
因此同步死循环、`time.sleep()` 或阻塞 I/O 无法被该 timeout 抢占
（`nexus/runner/single.py:683-720`）。

Nexus 自身也在该 loop 执行部分输入规模相关的同步工作：worker wheel 会整包 `read_bytes()` 后进行 base64 编码
（`nexus/core/context.py:263-295`），runner cache 的 load/save 也直接运行在 execution loop
（`nexus/runner/single.py:653-659,734-743`）。这些路径不一定永久 hang，但大 wheel、慢盘或大 cache 会形成明显
loop lag。

治理方向：文件读取、hash、base64、pickle 和目录扫描进入有界 I/O/CPU worker；正式云的用户代码使用可强制终止的
subprocess/cgroup，而不是只依赖 coroutine timeout。

### LOC-02：Live run 与 HTTP 控制面共用 loop

Live Server 在 aiohttp 主 loop 创建 `_run_single()` task，并直接调用 `run_single()`
（`nexus/runner/live/manager.py:230-244,351-366`）。health、kill、SSE 和 run handler 注册在同一个 application
（`nexus/runner/live/server.py:434-476,505-547`），所以用户同步阻塞会同时冻结运行与控制接口。

`kill()` 在取消后没有总期限，会继续等待 run task、runtime trace 和 terminal trace；服务关闭时又会 gather 全部 run
（`nexus/runner/live/manager.py:490-537,962-974`）。取消无法抢占同步用户代码。

治理方向：HTTP loop 只承担控制面；每个 Live run 至少使用独立线程 loop，需要可靠 kill 时使用受监督 subprocess。
create、kill、trace drain 和 close 共享一个绝对 shutdown budget。

### LOC-03：Trace pipeline 缺少统一阻塞边界

`SpanProcessor.dispatch_mode` 默认是 `inline`（`nexus/trace/processors/base.py:11-41`），processor hook 和 finalize
会在调用线程执行。即使选择 background dispatcher，其 10,000 项队列满时，`submit()` 仍会同步阻塞
（`nexus/trace/dispatcher.py:44-60`）；processor finalize 也没有统一 deadline
（`nexus/trace/tracer.py:551-566`）。

RunTrace 的 sink start/write/close、flush 和 session close 同样没有统一阶段期限
（`nexus/runner/trace/trace.py:379-427,438-471,504-540,579-606,790-817`）。慢 sink 不一定冻结整个 loop，
但会阻止对应 run 到达终态，并可能拖住 shutdown。

治理方向：I/O processor 默认 offload；队列使用有界 backpressure/drop 策略；sink start/write/flush/close 共享端到端
deadline，超时后隔离单个 sink，而不是继续阻塞 run。

### LOC-04：Runtime venv 文件重活运行在主 loop

`VenvManager.create()` 在 async 方法中执行 wheel 落盘与 hash、metadata/依赖闭包计算、base site-packages 索引、
`copytree` 和 symlink；`destroy()` 同步执行 `rmtree`
（`nexus/runtime/server/venv/manager.py:428-515,581-719`）。普通 Runtime Server 路径会经过完整流程，可能冻结
Runtime 的 health、Socket.IO、invoke 和 proxy。

动态 Manager bootstrap 的 wheelhouse 快路径不会执行完整 base copy/symlink，不能把所有 venv 工作都归因于 bootstrap。

治理方向：将 materialize、metadata/closure、hash、copy/symlink 和 rmtree 封装成有界文件工作单元；`uv`/`curl`
subprocess 也应具备总 deadline、TERM/KILL 和临时目录回收。

### LOC-05：Runtime worker 进程与 pending invoke 生命周期不闭合

Runtime worker 的 stdout/stderr 被设置为 PIPE，但没有持续 reader；PIPE 写满后 worker 可能停在写 syscall。停止 worker
时，SIGKILL 后的第二次 `wait()` 没有 deadline
（`nexus/runtime/server/worker/launcher.py:89-128`）。

Runtime Server 发出 invoke 后直接等待 future；worker disconnect 只清理 route 和 SID，不会 fail 该 worker 的 pending
future（`nexus/runtime/server/app.py:517-578,1671-1685`）。因此 worker 崩溃可能让 invoke 永久保持 running。

治理方向：持续 drain PIPE 到有界 ring buffer，或直接重定向到有界日志；pending invoke 绑定 worker generation，并在
disconnect、replace、stop 和 shutdown 时原子失败；TERM、KILL、reap 各自使用独立 deadline。

### LOC-06：外部调用缺少端到端 deadline

全局 `AsyncHTTP` 使用 `total=None`（`nexus/utils/async_http.py:35-59,174`），MCP 的 `timeout_s` 默认也是 `None`
（`nexus/core/tool/mcp/models.py:17-25`）。COS 只设置 connect 和 sock-read timeout，没有覆盖对端不继续读取上传 body
的总期限（`nexus/utils/cos.py:850-864,1173-1193`）。

部分同步 provider SDK 已通过 `asyncio.to_thread()` 下沉，因此不会直接冻结 loop；但线程中的 SDK 调用无法被 coroutine
取消，没有调用方 deadline 时仍会长期占用任务和容量槽。

治理方向：为 connect、首字节、读、写进度和总时长分别设置期限；`to_thread` 调用进入 provider 专用有界线程池，
需要强制终止时改用 subprocess；timeout 后必须关闭 response/transport 并保留可重试 cleanup 状态。

### LOC-07：同步 logging 可放大慢盘故障

distributed 在 root logger 上安装同步 `TimedRotatingFileHandler`
（`nexus/distributed/observability/logging.py:63-99`）。每条 JSON log 还会同步执行 message、extra、异常 traceback 和
`json.dumps`（`nexus/utils/log/formatters.py:55-109`）。

正常小日志不会造成 hang；真正的风险是 stdout/backing disk 停滞、文件轮转阻塞或日志参数异常大。

治理方向：使用有界 `QueueHandler` 和单独 writer，明确队列满时的 drop/fail 策略，并在进入队列前限制日志 payload。

## 5. Distributed 专属风险

### DIST-01：默认 AsyncioExecutor 放大同步阻塞

CLI 和 Helm 默认 executor 都是 `asyncio`（`nexus/cli/commands/distributed.py:198-200,242-257`；
`deployment/distributed/helm/nexus-distributed/values.yaml:119,143`）。`AsyncioExecutor` 只有一个线程和一个 event loop，
所有 assignment task 都创建在该 loop 上
（`nexus/distributed/execution/executors/asyncio_executor.py:38-70,137-172,191-228`）。

因此 LOC-01 至 LOC-06 中任一同步冻结一旦发生在 execution loop，影响范围会从“一个任务”扩大为“同一 Manager 的全部
execution”。Manager 主 loop 仍是独立 loop，但它对 execution 的查询、停止和命令也会受到跨 loop bridge 牵连。

治理方向：正式云默认至少使用 `thread/per_execution` 隔离故障域；需要可靠强杀时使用每 execution subprocess/cgroup。

### DIST-02：跨 loop bridge 和 self-fence 没有完成恢复闭环

`AsyncioExecutor` 的 bridge 直接等待 `wrap_future()`，没有调用方 deadline。`stop_assignment(timeout=...)` 的预算在
目标 loop 中才开始计算（`nexus/distributed/execution/executors/asyncio_executor.py:167-172,231-257`）。目标 loop
正常时该预算有效；目标 loop 已同步冻结时，stop coroutine 和计时器都无法开始运行。

Manager heartbeat 会先跨 loop 获取 capacity，并由 Manager loop 的 self-fence deadline 包住
（`nexus/distributed/instance_manager/service.py:291-306`；
`nexus/distributed/instance_manager/loops.py:77-105`）。execution loop 冻结可以触发 self-fence，但 `_self_fence()`
随后又会无调用方 deadline 地 stop 冻结 assignment
（`nexus/distributed/instance_manager/loops.py:177-185`），因此不能保证进程完成退出。

`ThreadExecutor(per_execution)` 可以隔离其他 execution，但 bridge 仍无调用方 deadline，Python 线程也无法强杀。
`_stuck_slot_cost` 只有初始化和读取，没有形成冻结 worker 的解绑和扣容闭环
（`nexus/distributed/execution/executors/thread_executor.py:39-89,124,184-213,246-278,336-426`）。

治理方向：所有 bridge 在健康的调用方 loop 建立绝对 deadline；超时后解绑 assignment、记录 stuck capacity，并让
self-fence 直接进入有界进程退出流程，而不是继续等待冻结 worker。

### DIST-03：DB await 与 shard refresh lock 耦合控制循环

Scheduler 的 assignment 与 lease heartbeat 是独立 task，普通 assignment DB stall 不会立即阻塞 lease
（`nexus/distributed/control_plane/scheduler.py:79-97`）。两者仍会共享 `ShardLeaseCoordinator.refresh()`；其 lock
覆盖 session、query、renew、commit 和 session close
（`nexus/distributed/control_plane/leases.py:55,62-65,115-172`）。任一 DB 阶段不返回，后续 refresh 都会等待同一锁。

Nexus engine 只按配置设置连接池参数，没有统一 statement、lock、commit、rollback 或 close deadline
（`nexus/distributed/persistence/db.py:106-128`）。这属于关键 coroutine await stall，不是同步 event-loop freeze。

Reconciler 已按 domain 拆成独立 worker，并使用 `wait_for()` 包住 tick
（`nexus/distributed/control_plane/reconciler.py:88-152,181-227`）。该 timeout 是有效保护，但被取消 coroutine 的 DB
cleanup/session close 不返回时，实际耗时仍可能突破预算。

治理方向：driver/server 层设置 statement 和 lock timeout，repository operation 再设置总 deadline；取消或 timeout 后
丢弃未知状态 connection。缩短 refresh lock 覆盖范围，避免 lease heartbeat 排队等待已经失去时效的 refresh。

### DIST-04：Scheduler 存在规模相关 CPU starvation

Scheduler 的 execution scan 有上限，关键 claim 使用 `SKIP LOCKED`
（`nexus/distributed/repositories/scheduler.py:395-423,691-730`），这些保护应保留。

残余风险是每个 tick 加载全量健康 Manager，再为候选 execution 反复执行 Python 过滤、分区和排序
（`nexus/distributed/repositories/scheduler.py:493-510,587-655,745-803`）。当前证据不足以把它判定为永久 hang，
但 Manager 数量和 execution 候选同时增大时会推高 loop lag，并延迟 lease 续期。

治理方向：在 SQL 侧先按 environment、pool、workload、shard 和 labels 缩小 Manager 候选集，并为每 tick 设置候选组合数
与 CPU 时间预算。

### DIST-05：Batch 同步工作运行在 Instance Manager 主 loop

`ManagerBatchControl` 把 Batch 操作送回 Manager 主 loop，bridge 无 deadline 且使用 shield
（`nexus/distributed/instance_manager/reporting.py:152-198`；
`nexus/distributed/instance_manager/service.py:209-223`）。

Batch materialize/submit 在持有 batch、parent 或 item 行锁的事务中同步执行 `open/seek/readline/read`，并完成 JSON
解析、hash 和 deepcopy；单批上限仍可达到 16/32 MiB
（`nexus/distributed/repositories/batch.py:361-491,856-985`）。这是当前最明确的 Manager 主 loop freeze 风险。

普通 assignment poll 也会在 Manager loop 构造和 deepcopy 最多一个 start batch 的完整 `ExecutionRecord`
（`nexus/distributed/repositories/manager.py:575-618`）。该路径有数量上限，只作为 P2 放大因素，不单独升级为 P1。

治理方向：在事务外的 bounded worker 完成文件读取、JSON 解析、hash 和 child spec 构造；事务内只保留小型 CAS、行状态更新
和批量 insert。Manager bridge 同时补调用方 deadline。

### DIST-06：Gateway 和 Artifact maintenance 执行同步文件与大对象工作

Application Gateway 上传和下载在 HTTP loop 同步执行 open、write、read 和 hash
（`nexus/distributed/gateways/application/routes/artifact_shared.py:133-164`；
`nexus/distributed/gateways/application/routes/artifact_content.py:294-327`）。Generic Store 下载也在 async 方法中同步落盘、
hash 和 replace（`nexus/distributed/artifacts/store.py:310-352`）；Artifact maintenance Reconciler 会复用同一 store
（`nexus/distributed/artifacts/maintenance.py:98-123,148-180,215-256`）。

Control Gateway 的 `summary=true` 会先读取完整 execution/assignment、deepcopy spec/result，再递归序列化，最后才裁成
summary，单次上限 500 条（`nexus/distributed/repositories/execution.py:362-427`；
`nexus/distributed/repositories/_records.py:24-93`；
`nexus/distributed/gateways/control/routes/executions.py:237-299`；
`nexus/distributed/gateways/common/serde.py:8-20`）。这是规模相关 CPU starvation，不是确定永久 hang。

Artifact spool 还会在 asyncio lock 内执行 `lstat/utime/unlink` 和 O(n) 候选构造
（`nexus/distributed/artifacts/spool.py:102-150,181-207,250-252`），作为该风险的 P2 放大因素。

治理方向：Gateway/Generic Store 使用有界异步文件层或专用 I/O pool；summary 直接使用 SQL projection；spool 的候选计算和
删除移出锁与 event loop。

### DIST-07：关闭和自栅栏缺少统一绝对预算

Manager `stop()` 会串行等待 background task、assignment、executor、Batch、Trace 和 Artifact。manager-loop future drain
虽然先设置 2 秒 timeout，但 timeout 后仍会无界 gather 已取消 future
（`nexus/distributed/instance_manager/service.py:177-194,225-238`）。Periodic lease 的 cancel/close 也没有二层期限
（`nexus/distributed/utils/leases.py:40-65`）。

Artifact transport 的 HTTP streaming 已在专用线程 loop 中，不会直接冻结 Manager/execution loop；但 close 对 ready、
shutdown 和 session close 没有二层 deadline
（`nexus/distributed/artifacts/gateway.py:50-104,145-168`）。

治理方向：所有组件共享一个绝对 shutdown deadline，并按“停止接入 → cancel → TERM → KILL → bounded reap/abandon”
分阶段消费预算。一次 timeout 后不得再次无界等待。

## 6. 已有保护与无需重复建设的部分

- `e319aab4` 已将动态 Manager bootstrap 的 wheelhouse/source 解压、workspace reset/rmtree、artifact identity 和 marker
  I/O 移入 `asyncio.to_thread()`（`nexus/distributed/instance_manager/bootstrap.py:119-163,200-241`）。因此不再保留
  独立 `BOOT-01`，剩余问题归入共用 `VenvManager` 风险。
- Local Artifact Store 的整文件 copy、hash 和 delete 已下沉线程
  （`nexus/distributed/artifacts/store.py:166,184-201`），不应再列为 Gateway 主 loop P1。
- Application Gateway Artifact transport 已使用专用 HTTP loop
  （`nexus/distributed/artifacts/gateway.py:50-75,119-168`）。distributed common proxy 的请求体有上限，响应也采用流式
  转发（`nexus/distributed/gateways/common/proxy.py:92-106,148-160`）。
- Scheduler 的关键 claim 已使用 `SKIP LOCKED` 且 execution scan 有界；Reconciler 已完成 domain worker 隔离并设置 tick
  timeout。
- `EventLoopMonitor` 使用独立 OS watchdog 线程，可以在 loop 冻结时记录告警
  （`nexus/utils/eventloop_monitor.py:245-308`）。当前 distributed callback 只告警，不重启进程或改变 health
  （`nexus/distributed/observability/event_loop.py:66-104`），所以它是诊断能力，不是自动恢复闭环。

## 7. 修复优先级与验收标准

| 阶段 | 目标 | 主要改动 | 验收标准 |
| --- | --- | --- | --- |
| 立即 | 切断共享故障域 | execution 隔离、bridge caller deadline、self-fence 有界退出 | 一个同步死循环不影响其他 execution，Manager 能在固定上限内退出 |
| 立即 | 移除主 loop 同步重活 | Batch 事务外预处理；Gateway/Generic Store、Runtime venv 下沉 | 慢盘和上限输入下 loop lag 不跨 heartbeat 阈值 |
| 立即 | 闭合 Runtime worker 生命周期 | drain PIPE、断连 fail pending future、TERM/KILL/bounded reap | worker 崩溃或拒绝退出时 invoke 必然终态，shutdown 有上限 |
| 短期 | 建立外部调用期限 | DB、Trace、Socket.IO、MCP、Provider、COS 统一 absolute deadline | 黑洞、半开连接和 cleanup 卡住时，调用在预算内返回明确终态 |
| 短期 | 统一关闭协议 | 所有组件共享 shutdown budget，timeout 后允许 abandon | Pod 不依赖无限 cleanup，持久状态有可恢复记录 |
| 短期 | 隔离 logging | 有界 QueueHandler、单 writer、payload budget | 慢日志后端不冻结控制或 execution loop |
| 中期 | 降低规模型 CPU 工作 | Scheduler SQL 预筛、Gateway summary projection、spool 锁外计算 | 最大正常规模下 tick/HTTP 延迟保持在既定预算内 |

修复过程中应优先建立 deadline 和故障域边界，再做局部性能优化。仅增加 liveness probe 可以缩短故障持续时间，但不会
消除 loop hang 根因。

## 8. 故障注入清单

静态审查只能确认缺少结束边界，不能证明完整恢复能力。实现修复后至少执行以下故障注入：

| 场景 | 必须验证的结果 |
| --- | --- |
| 一个 execution 执行同步死循环 | 同 Manager 其他 execution 继续运行；self-fence、Pod 退出和容量恢复均有上限 |
| DB query、commit 或 session close 永不返回 | Scheduler lease、Reconciler domain、Manager heartbeat 和 Gateway 请求按预算失败或转移 |
| task-list、artifact 或 venv 文件系统冻结/极慢 | loop lag 不跨控制面阈值；请求、任务和关闭流程均有明确 timeout |
| Runtime worker 填满 PIPE、断连或忽略 TERM/KILL | pending invoke 必然终态，进程在二层 deadline 后被回收或明确记为 orphan |
| Trace sink、Socket.IO、MCP 或 COS peer 永不返回 | deadline 生效，单个 transport 被隔离，run 和 shutdown 仍能完成 |
| Pod termination | 所有阶段共享 grace budget；超时后不再等待无界 cleanup |

建议同时记录 event-loop lag、heartbeat 间隔、lease ownership、pending task/future 数、线程和子进程数量，以区分同步冻结、
await 停滞和资源耗尽。

## 9. 不属于本次 hang 主清单的问题

| 问题 | 为什么不计入 event-loop hang |
| --- | --- |
| Environment create/ref 窗口、Provider tracking、Dependency-build finalizer/cleanup | 属于生命周期一致性或资源泄漏 |
| Runtime job/blob、proxy 全量缓冲、task-file 大响应、spool entry/inode | 属于内存、磁盘或 inode 资源预算 |
| 缺少 liveness/startup probe | 是检测与恢复缺口，不是 hang 的产生源 |
| Python 3.10 `TimeoutError` 兼容性 | 可能让周期 task 异常退出；官方 distributed 镜像使用 Python 3.13 |
| db-init migration 或 Job 无 active deadline | 属于发布流程停滞，不是长期服务 event loop hang |
| 本地 `nexus modules wheel` 同步构建 | 是同步 CLI 的预期执行模型，不运行在长期服务 loop |
| Live/runtime proxy 完整缓冲响应 | 主要是 OOM/CPU 资源风险；Live 与 run 共 loop 的问题已计入 LOC-02 |
| `py-spy`/debug proxy 单请求无总期限 | async subprocess 只停滞诊断请求；同步读取输出是低频文件风险 |

这些问题应分别进入生命周期、资源预算、可观测性和发布可靠性审查，不能因为移出本清单而视为已经解决。

## 10. 审查边界

- 本文结论来自当前代码基线的静态调用链、timeout、cancellation 和线程/loop 所有权审查。
- 本次没有执行故障注入、压力测试或生产验证。
- “可能 hang”表示当前代码缺少静态可证明的结束边界，不表示问题已经在生产发生。
- 历史 incident 只能证明影响严重度，不能反推某一条静态风险是唯一根因。
- 本次只更新审查文档，没有修改业务代码、配置或部署资源。
