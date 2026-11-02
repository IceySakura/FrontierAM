# Harbor Bridge Partial Rollout Pause-aware Timeout 方案

日期：2026-09-04
状态：提案，待评审

## 1. 摘要

目标是在不关闭 partial rollout、不等待长尾 rollout 排空、也不增加第二套 rollout engine
的前提下，让 Harbor Bridge 在两类场景中使用同一份 timeout 配置：

- 普通 Harbor 评估：timeout 按 wall-clock 正常工作；
- HyTuner partial rollout 训练：Actor update 导致的模型请求暂停不计入 timeout。

本方案保留现有字段名 `agent_timeout`，但扩大其作用范围：

```text
agent_timeout =
  获取 main Runtime
  + 环境准备和上传
  + boot / healthcheck
  + Agent
  + collect hooks / artifacts
```

即 `agent_timeout` 表示 **Verifier 开始前的 active-time 总预算**。

`test_timeout` 继续只控制 Verifier。`runner.timeout` 继续是整个 Task 的最终安全上限。

实现核心是 Nexus `ExecutionContext` 持有统一的 `ExecutionBudget`。HyTuner 的 pause 信号
目前已经通过 `abort_event` 到达 Nexus `HyTunerClient`，因此无需修改 HyTuner 调度协议；
只需将 Nexus 当前“仅暂停 runner deadline”的实现扩展为“暂停当前 execution 的全部注册
deadline”，并将 pause/resume 传入 Runtime。

## 2. 背景与当前问题

### 2.1 当前调用链

```text
HyTuner RolloutController
  → AgentLoopWorker._train_abort_event
  → NexusAgentLoop
  → nexus.runner.hytuner.run(abort_event=...)
  → Nexus HyTunerClient
  → Agent / Runtime
```

`_train_abort_event` 的语义是：

```text
event.set()    正常运行
event.clear()  Actor update，生成暂停
```

Nexus `HyTunerClient._wait_for_abort_resume()` 已经能准确知道单条 trajectory 何时真正等待
模型恢复，并记录暂停时长。

这比在 HyTuner 全局 Actor update 边界直接冻结全部任务更准确：

- Agent 正在运行 bash/tool 时并未暂停，timeout 应继续累计；
- Agent 正在等待被中断的模型请求时才真正暂停，timeout 应停止累计。

### 2.2 当前只有 Runner timeout 能暂停

`nexus.runner.single` 将最外层 timeout handle 写入：

```python
ctx.timeout_handle
```

Nexus `HyTunerClient` 在等待 resume 时对它执行：

```python
timeout_handle.reschedule(None)
# resume 后
timeout_handle.reschedule(original_deadline + wait_duration)
```

因此 `runner.timeout` 已经是 pause-aware。

但以下 timeout 仍使用普通 wall-clock：

- Harbor Bridge `asyncio.wait_for(run_agent(), agent_timeout + 300)`；
- Codex / HyCode 等 Runtime command timeout；
- Mini-SWE 的 `time.time()` wall-time limit；
- CodeMT inference / observation 的 `time.monotonic()` 累计；
- Native Bridge / ACP 等局部 `asyncio.wait_for`。

结果是外层 Runner 正确暂停，内层 timeout 却可能先触发。

### 2.3 数值相等不能解决问题

即使：

```text
agent_timeout = runner.timeout = 12000
```

两者仍然不同：

- Runner 从 Task 开始计时，并能排除 partial pause；
- Agent/Runtime timeout 从各自阶段开始计时，且继续计算 partial pause。

此外相同数值会使 Agent 理论上吃完整体预算，Verifier 没有剩余时间。

## 3. 目标与非目标

### 3.1 目标

1. 完整保留 partial rollout 和现有吞吐模型。
2. 普通评估与训练使用同一份 `agent_timeout` / `test_timeout` 数据。
3. 只排除 trajectory 实际等待模型 resume 的时间。
4. Host 和 Runtime deadline 使用一致的 active-time 语义。
5. Agent 超时时尽可能保留 workspace，并继续执行 Verifier。
6. pause、resume、cancel、Runtime 断连均不能泄漏 deadline 或进程。
7. 旧 Runtime 可检测、可告警，并提供明确兼容策略。

### 3.2 非目标

1. 不改变 partial rollout 的调度算法。
2. 不通过 drain-before-update 降低吞吐。
3. 不引入双 rollout engine。
4. 不让 Verifier timeout 排除普通测试执行时间。
5. 不要求每个 Harbor Task 自己实现 pause 逻辑。
6. 本方案不处理 anti-hack、reward 或评分语义。

## 4. Timeout 语义

### 4.1 `agent_timeout`

保留字段名，定义为 Verifier 前全部阶段的 active-time 预算：

```text
开始：
  harbor_bridge_task 开始准备 main Runtime

包含：
  get_runtime(main)
  目录初始化
  environment uploads
  environment.start
  boot
  healthcheck
  Agent setup / install / proxy / inference / tools / cleanup
  collect hooks
  artifact collection

结束：
  Verifier 阶段开始前
```

Actor update pause 只在当前 trajectory 确实阻塞于模型请求时排除。

进入 Agent 时传入剩余预算：

```python
remaining = ctx.execution_budget.remaining("pre_verifier")
await task_input.run_agent(
    timeout=remaining,
    ...
)
```

Task 参数继续覆盖 `agent_config.timeout`。Harbor 生成的数据应逐步移除重复的
`agent_config.timeout`，避免出现两个看似生效的值。

### 4.2 `test_timeout`

继续表示一次 Verifier 执行预算：

```text
tests/test.sh
Judge
reward.json 生成
```

Verifier 在 Agent 完成后执行，此时不存在该 trajectory 等待 Actor update 的模型请求，
因此保持普通 wall-clock timeout。

### 4.3 `runner.timeout`

继续表示整个 Task 的 active-time 安全上限：

```text
pre-verifier
+ verifier
+ cleanup
```

建议约束：

```text
runner.timeout >= agent_timeout + test_timeout + cleanup_margin
```

对于没有显式 `runner.timeout` 的 `t3-harbor` 评估，阶段 timeout 仍然有效。后续可以由
Compiler/Experiment Profile 自动生成整体上限，但不作为本方案第一阶段的必要条件。

## 5. 核心抽象

### 5.1 `ExecutionBudget`

每个 `ExecutionContext` 持有独立实例：

```python
class ExecutionBudget:
    def active_time(self) -> float:
        """monotonic wall time 减去当前 execution 的累计暂停时间。"""

    def remaining(self, scope: str) -> float | None:
        """返回 scope 的 active-time 剩余预算。"""

    def register_deadline(self, scope: str, handle: object) -> None:
        """注册可 reschedule 的 Host deadline。"""

    def unregister_deadline(self, scope: str, handle: object) -> None:
        """scope 结束后注销 deadline。"""

    async def pause(self, token: str) -> "PauseToken":
        """暂停当前 execution 的时钟、Host deadline 和 Runtime deadline。"""

    async def resume(self, token: "PauseToken") -> None:
        """幂等恢复，并将暂停时长追加到 deadline。"""
```

### 5.2 active clock

```text
active_time = monotonic_time - accumulated_pause - current_pause_duration
```

普通评估从未调用 `pause()`：

```text
active_time == monotonic_time
```

所以不需要 `is_training` 分支，也不会改变普通评估行为。

### 5.3 pause token 与嵌套

必须使用 token/ref-count，而不是简单 bool：

```python
pause_token = await budget.pause(request_id)
try:
    await abort_event.wait()
finally:
    await budget.resume(pause_token)
```

要求：

- 同一个 token 重复 resume 幂等；
- 多个未恢复 token 存在时保持 paused；
- 最后一个 token 恢复时才重新激活 deadline；
- Task cancel/finalize 时强制清理全部 token；
- 一个 ExecutionContext 的 pause 不影响其他 trajectory。

## 6. Host 侧改造

### 6.1 `ExecutionContext`

文件：

```text
nexus/core/context.py
```

新增 `execution_budget`。旧 `timeout_handle` 暂时保留为兼容属性，由
`ExecutionBudget` 注册和管理。

### 6.2 `nexus.runner.single`

文件：

```text
nexus/runner/single.py
```

最外层 Runner timeout 创建后注册：

```python
async with async_timeout(run_input.runner.timeout) as timeout_handle:
    ctx.execution_budget.register_deadline("runner", timeout_handle)
    try:
        task_output = await task_fn(task_input)
    finally:
        ctx.execution_budget.unregister_deadline("runner", timeout_handle)
```

### 6.3 Nexus `HyTunerClient`

文件：

```text
nexus/extensions/openai_clients/hytuner.py
```

将当前只处理 `ctx.timeout_handle` 的 `_pause_task_timeout()` 替换为：

```python
pause_token = await require_context().execution_budget.pause(request_id)
try:
    await abort_event.wait()
finally:
    await require_context().execution_budget.resume(pause_token)
```

HyTuner 仓库无需修改；现有 `abort_event` 就是 pause 信号源。

### 6.4 Harbor Bridge

文件：

```text
nexus/extensions/harbor_bridge/task.py
```

移除独立的普通 wall-clock watchdog：

```python
asyncio.wait_for(run_agent(), timeout=agent_timeout + 300)
```

改为 pre-verifier active-time scope：

```python
async with ctx.execution_budget.timeout(
    "pre_verifier",
    task_input.agent_timeout,
):
    # main Runtime / boot / healthcheck / Agent / collect
```

进入 Agent 时传递剩余预算，而不是原始总值：

```python
remaining = ctx.execution_budget.remaining("pre_verifier")
agent_output = await task_input.run_agent(
    timeout=remaining,
    ...
)
```

原 `_AGENT_WATCHDOG_GRACE_SECONDS=300` 不再作为另一层 Agent deadline。超时取消后的清理
可保留一个独立、短小、不可重入的 cancellation grace，但它不属于 Agent 业务预算。

## 7. Host-native Agent

目标是只迁移当前 Harbor 训练实际使用的 Host-native Agent，不要求一次修改全部旧 Task。

### 7.1 CodeMT

文件：

```text
nexus/extensions/projects/code_mt/agent/tool_call_loop.py
nexus/extensions/projects/code_mt/agent/text_parse_loop.py
```

将用于业务预算的：

```python
time.monotonic()
```

替换为：

```python
require_context().execution_budget.active_time()
```

日志时间如果希望展示真实 wall-clock，可继续使用 `time.monotonic()`，但不得用它决定业务
timeout。

### 7.2 Mini-SWE

文件：

```text
nexus/extensions/agents/mini_swe_agent/agent.py
```

将 `wall_time_limit_seconds` 的计时基准切换为 active clock。字段名可以保持兼容，但文档应
说明 partial pause 不计入。

## 8. Runtime 侧改造

Host deadline pause 不能自动暂停 Sandbox 里的：

```python
asyncio.wait_for(process.wait(), timeout=...)
```

Codex、HyCode 等 Installed Agent 需要 Runtime 支持。

### 8.1 Runtime budget ID

Host 调用 Agent Runtime 时携带当前 execution/run ID。Runtime 使用该 ID 隔离 deadline：

```text
execution_budget_id = run_id / trajectory_id
```

不能按 Runtime 全局暂停，因为未来一个 Runtime 可能承载多个并发 operation。

### 8.2 RuntimeControl API

新增版本化控制操作：

```text
pause_execution_budget(execution_budget_id, pause_token)
resume_execution_budget(execution_budget_id, pause_token)
cancel_execution_budget(execution_budget_id)
```

要求：

- pause/resume 幂等；
- token 级 ref-count；
- unknown token 安全返回；
- Runtime 断连/销毁时清理 registry；
- 旧 Runtime 返回 unsupported capability，而不是静默成功。

### 8.3 Runtime command executor

文件：

```text
nexus/runtime/executors/command.py
```

从：

```python
await asyncio.wait_for(process.wait(), timeout=args.timeout)
```

迁移为 Runtime 的 pause-aware deadline：

```python
async with runtime_budget.timeout(
    execution_budget_id,
    args.timeout,
):
    await process.wait()
```

进程 stdout/stderr reader、SIGTERM/SIGKILL cleanup 仍使用短 wall-clock timeout，避免取消流程
被无限暂停。

### 8.4 Host 到 Runtime 传播

当 Nexus `HyTunerClient` 进入真实 abort wait：

```text
Host execution_budget.pause(token)
  ├─ reschedule(None) 所有 Host handles
  └─ main Runtime pause_execution_budget(...)
```

resume 时：

```text
Host execution_budget.resume(token)
  ├─ 恢复 Host handles
  └─ main Runtime resume_execution_budget(...)
```

RPC 只发送给当前 execution 已获取并注册的 Runtime。

## 9. 超时与失败语义

### 9.1 pre-verifier timeout

需要记录当前阶段：

```text
runtime_acquire
boot
healthcheck
agent
collect
```

处理建议：

| 超时阶段 | 行为 |
|---|---|
| Runtime/boot/healthcheck | `INITIALIZATION_ERROR`，不运行 Verifier |
| Agent | 取消 Agent，保留 workspace，构造 timeout AgentResult，继续 Verifier |
| collect | `ENV_ERROR`；separate verifier 无可靠 artifact 时不评分 |

### 9.2 Verifier timeout

保持当前 `REWARD_ERROR`/Verifier timeout 处理，不受 partial pause 影响。

### 9.3 pause 期间 Task 被取消

顺序必须是：

```text
恢复/注销 deadline
→ 取消 Agent invocation
→ 停止 Runtime environment
→ release Runtime
```

不能让 paused handle 永久保持 `deadline=None`。

### 9.4 Runtime 不支持 pause capability

提供两种策略：

```text
compat：
  Host timeout pause-aware
  Runtime timeout维持 wall-clock
  输出 warning/metric

strict：
  partial rollout + Installed Agent 启动前检查 capability
  不支持则 INITIALIZATION_ERROR
```

第一阶段默认 compat；完成新版 Runtime 部署后，训练 Profile 可切 strict。

## 10. 可观测性

每次 Task 输出/trace 至少记录：

```text
timeout.runner.active_elapsed
timeout.pre_verifier.active_elapsed
timeout.pre_verifier.wall_elapsed
timeout.pause_elapsed
timeout.pause_count
timeout.trigger_scope
timeout.trigger_stage
runtime.pause_aware_deadline_supported
```

这些指标用于判断是否真的需要 Runtime 协议，以及是否仍有某个 Agent 使用未迁移的 wall-clock。

不得记录 prompt、密钥或完整命令。

## 11. 测试计划

### 11.1 Unit

1. 无 pause：active time 等于 wall time。
2. 单次 pause：deadline 精确延后 pause duration。
3. 多次 pause：累计时长正确。
4. 嵌套 pause：最后一个 token resume 后才恢复。
5. cancel during pause：handle 被恢复/注销。
6. 注册时已 paused：新 handle 保持 suspended。
7. unknown/重复 resume：幂等。
8. 一个 Context pause 不影响另一个 Context。

### 11.2 Harbor Bridge

1. boot 消耗计入 `agent_timeout`。
2. Agent 获取的是 pre-verifier 剩余预算。
3. Agent 阶段 active timeout 后继续 shared Verifier。
4. Agent 阶段 active timeout 后继续 separate artifact/Verifier（artifact 可用时）。
5. boot timeout 不进入 Verifier。
6. collect timeout 不产生无效 separate grading。
7. 普通评估行为与迁移前一致。

### 11.3 Runtime

1. subprocess 在 pause 期间不超时。
2. resume 后剩余 active budget 正确。
3. SIGTERM/SIGKILL cleanup 不受 pause 阻塞。
4. 多 execution ID 隔离。
5. Host 断连后 registry 清理。
6. 旧 Runtime capability fallback。

### 11.4 HyTuner 集成

使用短预算构造可重复测试：

```text
Agent active timeout = 4s
执行 2s
partial pause 3s
恢复执行 2s
```

期望：

```text
wall elapsed ≈ 7s
active elapsed ≈ 4s
不会在 pause 中超时
恢复后按 active budget 超时/完成
```

同时运行无 pause validation，确认 wall elapsed ≈ active elapsed。

## 12. 分阶段实施

### Phase 1：Host 基础设施

1. 实现 `ExecutionBudget`。
2. Runner 注册 deadline。
3. Nexus `HyTunerClient` pause/resume 整个 budget。
4. Harbor Bridge 使用 pre-verifier active scope。
5. 增加 Host 指标和单测。

收益：修复 Runner/HB watchdog，普通评估零行为变化。

### Phase 2：当前 Host-native Agent

1. CodeMT 使用 active clock。
2. Mini-SWE 使用 active clock。
3. 增加 Agent 回归测试。

收益：Host-native Agent timeout 完整 pause-aware。

### Phase 3：Runtime 协议

1. 增加 capability 和 execution budget RPC。
2. command executor 接入 Runtime budget。
3. 发布 Nexus Runtime 镜像。
4. Codex/HyCode E2E。

收益：Installed Agent timeout 完整 pause-aware。

### Phase 4：推广

1. 训练 Profile 切 strict capability。
2. 逐步迁移仍直接使用 `time.time()`/`asyncio.wait_for()` 的其他 Agent/Task。
3. 根据指标决定是否删除 compat fallback。

## 13. 备选方案对比

| 方案 | 保留 partial 吞吐 | timeout 精确 | 需要 Runtime 镜像 | 结论 |
|---|---:|---:|---:|---|
| 直接扩大 timeout | 是 | 否 | 否 | 临时缓解 |
| 关闭 partial | 否 | 是 | 否 | 不满足目标 |
| drain-before-update | 部分损失 | 是 | 否 | 不满足吞吐目标 |
| 双 rollout engine | 是 | 是 | 否 | GPU 成本过高 |
| 仅暂停 Runner | 是 | 否 | 否 | 当前状态，不完整 |
| Host ExecutionBudget | 是 | Host 精确 | 否 | Phase 1/2 |
| Host + Runtime ExecutionBudget | 是 | 是 | 是 | 最终方案 |

## 14. 待评审决策

1. `agent_timeout` 扩大为 pre-verifier active budget 是否接受为正式兼容语义。
2. 未显式设置 `runner.timeout` 时，Compiler 是否自动生成整体 timeout。
3. Agent timeout 后 shared/separate Verifier 的继续评分条件。
4. Runtime capability 初期默认 compat 还是 strict。
5. Runtime command cleanup timeout 是否固定为 wall-clock。
6. `agent_config.timeout` 是否从 Harbor 生成数据中删除。

## 15. 推荐结论

推荐按 Phase 1 → Phase 2 → Phase 3 推进。

最终只保留一套 pause 来源：

```text
HyTuner abort_event
```

只保留一套 active-time 状态：

```text
ExecutionContext.execution_budget
```

Host Task、Host Agent 和 Runtime executor 都消费同一 execution budget。普通评估没有 pause
事件，自然保持 wall-clock；训练期间仅当前 trajectory 真正等待模型恢复时暂停。这样可以完整
保留 partial rollout 吞吐，同时避免 timeout 在不同层各自解释和漂移。
