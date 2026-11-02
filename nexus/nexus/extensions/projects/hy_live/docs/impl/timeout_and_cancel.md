# Timeout 与 Cancel：分层语义、时序与已知问题

> **代码位置**：工作预算、墙钟上限、interrupt 与重启会话的编排在 `hylive.turns.TurnScheduler`，取消和撤回权限在 `hylive.coordinator.Coordinator`；nexus 的 `coordination/backend.py::AcpSessionExecutor` 只执行 ACP 原语（开会话、prompt、interrupt、关会话）。

> **边界**：本文描述 **Host 业务超时** 与 **Sandbox 内 ACP Session 超时 / 取消** 如何叠加、为何会 quarantine。 取消入口是小脑 `cancel` → 中脑 `TaskStop` → `Coordinator.stop_task`，见 [`agent_loop.md`](agent_loop.md)。

## 0. 当前行为

一轮 = 一次 CodeBuddy `session/prompt`。 有两道限时：

| 限时 | 默认值 | 计不计等用户 | 谁执行 | 到期后 |
|---|---|---|---|---|
| 工作预算 `turn_budget_s` | 2000 秒（`PROMPT_BUDGET_S`） | 不计：Run 上有未答的权限或补充问题时暂停计时 | core `TurnScheduler` 的 `budget` 计时器，`ts_supervise` 在等用户时暂停 | 撤回这一轮的权限和补充问题，interrupt，最多再等 30 秒让这一轮真正结束 |
| 墙钟上限 `WorkAgentBudget.timeout_s` | 3600 秒 | 计；episode 冻结的时间不计 | core `TurnScheduler` 的 `wall` 计时器；沙箱 `send_user_message` 只做兜底 | 记为 `timeout`；再 interrupt 一次，确认沙箱里那一轮已经停下 |

- **工作预算到期**：先由 `Coordinator.release_turn` 把这一轮开着的权限按 `cancelled` 回掉、补充问题按 `cancel` 收掉，再发 `AcpInterrupt`（执行器调 `interrupt_codebuddy_session`）。 interrupt 确认、这一轮也在 30 秒内回来，会话保留，下一轮照常发到同一个会话；否则 `ts_restart` 发 `AcpCloseSession`（执行器调 `close_codebuddy_session`），下一轮会重新起一个 CLI。 这一轮记为 `timeout`，保留已有的部分输出，Run 记为失败。
- **顺序**：同一会话上，下一轮要等这一轮结束、需要时会话也已关闭，才会发出。
- **墙钟上限**：只是兜底。 配置里显式写的值照样生效；写成 2000 秒或更小时，墙钟上限会先于工作预算到期，等用户的时间又会计入。
- **episode 冻结**：HyTuner partial rollout 把小脑停在 abort 门口、后台也没有 `queued`/`running` 的 Run、中脑不在处理请求时，episode 冻结（`episode_clock.py`）。 core 的计时器跳过冻结时间（`SessionRunner(suspended=clock.frozen_s)`）；沙箱看不到冻结，传给它的 `timeout` 是 `timeout_s` 再加 4 小时（`_RUNTIME_FREEZE_ALLOWANCE_S`），只在 Host 失联时兜底，所以一轮总是由 Host 上限结束、再由 Host interrupt。
- **用户取消**（`Coordinator.stop_task`）：先把被取消 Run 上开着的权限按 `cancelled` 回掉（ACP outcome `cancelled`），再丢掉它还在待发队列里的轮次，最后 interrupt。 还没交给后端的 Run 保持 `cancelled`，后台轮到它时直接跳过，不会再被标成 `running`。 Run 进入终态后，之后的非终态状态更新一律忽略。
- **轮次结束**：这一轮还开着的权限按 `cancelled` 收掉，Run 不会停在 `needs_permission`。
- **CodeBuddy 的提问**（`AskUserQuestion`、form `elicitation/create`）：落成 Run 上的补充问题，等用户的时间同样不计入工作预算。 取消、工作预算到期、轮次结束时问题按 `cancel` 收掉，Host 回 `deny` + `reject`（AskUserQuestion）或 `{action: cancel}`（elicitation）。 链路见 [`question_channel.md`](question_channel.md)。
- **取消时 interrupt 失败**：账本里 Run 仍是 `cancelled`，中脑拿到正常的工具结果；这个 Run 的结果记为基础设施失败（`CANCEL_INTERRUPT_FAILED`），整条 episode 按 `ENV_ERROR` mask。 `steer` 时 interrupt 失败只记日志。

## 1. 会话与轮次

core 为每个 Task 维持一条 ACP 会话，由 `AcpSessionExecutor`（`coordination/backend.py`）执行，每轮的调用方式：

- Session ID 是 core 分配的 Project Session ID（`hy-live:<owner_id>:project:<task_id>`），规范化成 UUID 后同时用作 runtime session 和 LLM channel 的 id；同一 Task 的后续 Run 都发到这条会话，这就是 `TaskUpdate` 的落点。
- 每轮只发一个 ACP turn（`max_turns=1`），`close_on_finish=False`，会话在轮次之间保留；关闭会话时一并解绑它的 LLM channel。
- CodeBuddy 默认经 `external` client 调模型（`llm_client_name`）。
- 轨迹用 `ChatTrackerProcessor(tools_mode='branch_last')` 采集；输出映射成 `TaskResult`，`metadata` 带 `session_id`、`session_status`、`turns`、`trajectory_count`。

## 2. 两层时钟

一次 Project 轮次的 CodeBuddy 调用会穿过两层时钟， **彼此不联动** 。 Host 层是真正的上限，Sandbox 层的值更长，只在 Host 失联时兜底：

| 层 | 进程位置 | 谁在计时 | 超时后做什么 |
|---|---|---|---|
| **Host core** | 训练 / 在线 Runner 进程 | `TurnScheduler` 的 `wall` 计时器（`timeout_s`），冻结时间由 runner 的 `suspended` 扣掉；工作预算由 `budget` 计时器另外计 | 这一轮记为 `WORK_AGENT_TIMEOUT`，再发 interrupt（第 0 节） |
| **Sandbox ACP Session** | Runtime 容器内 `codebuddy --acp` | `session_runtime.send_user_message(..., timeout=budget.timeout_s + _RUNTIME_FREEZE_ALLOWANCE_S)`，纯墙钟 | 对当前 prompt 发 ACP `session/cancel`，在 **grace** 内等原 prompt 的 terminal（`stopReason: cancelled` 等） |

Invoke 本身还有 **HTTP long-poll**：默认等 30 秒，超过后 job 转为后台运行，执行器的 `_run_codebuddy` 仍会继续等（见 `nexus/runtime/server/app.py` 的 `/invoke`）。

```text
Host                                Sandbox (Runtime Base)
──────────────────────────────────────────────────────────────────
core wall 计时器(timeout_s)          run_codebuddy_session_turn
AcpSessionExecutor._run_codebuddy     send_user_message(timeout + 4h)
       │                                    │
       │  runtime.invoke ──────────────────►│  ACP session/prompt
       │  (可能 30s 后变后台 job)            │  超时 → session/cancel + grace
       ▼                                    ▼
WORK_AGENT_TIMEOUT                    status=timeout / quarantine
（core 记 timeout，再 interrupt）       （CLI 可能仍在跑）
```

## 3. Host 侧：`WorkAgentBudget.timeout_s`

- **默认值**：3600 秒（`coordination/backend.py::WorkAgentBudget`），任务配置可覆盖，经 `SessionConfig.wall_timeout_s` 交给 core。
- **语义**：一轮（`_run_codebuddy` → `run_codebuddy_multiturn` → **一次** `session/prompt`）的整段 wall-clock 上限。 Host 要等的是 **invoke 整条链路结束** （Sandbox 里 turn 结束、含 cancel drain 与结果序列化），不是「CLI 收到 cancel 就返回」。
- **超时处理**：core 的 `wall` 计时器到期时，这一轮记为 `timeout`，错误码 `WORK_AGENT_TIMEOUT`（`hylive.turns`）；随后 interrupt 一次，interrupt 没有确认就关闭会话。
- **冻结**：runner 按 `EpisodeClock.frozen_s` 把冻结时间从所有计时器里扣掉，冻结多久就推多久。
- `_codebuddy_input` 把 `budget.timeout_s + _RUNTIME_FREEZE_ALLOWANCE_S`（4 小时）传给 multiturn 的 `timeout` 字段。 沙箱按墙钟计时、看不到冻结，若传同一个值，冻结时间超过剩余上限时沙箱会先超时，Host 的顺延就白做了。

## 4. Sandbox 侧：一轮 prompt 的时间预算

实现位于 `nexus/extensions/agents/codebuddy/session_runtime.py` 的 `_CodeBuddyAcpSession.send_user_message`。

- 调用方传入的 `timeout` 用于 `asyncio.wait(..., timeout=timeout)`，只统计 prompt 业务阶段；底层 `connection.prompt(...)` 的 timeout 是 `timeout + _CANCEL_GRACE_TIMEOUT`（30 秒）。
- 业务超时后：`status = 'timeout'`，先发 ACP `session/cancel`，再在 grace 内等原 prompt 带着 terminal 结束；partial output / events 写进 turn 结果。
- **Quarantine**：grace 内没等到 terminal 时 `reusable = False`，该会话禁止再 prompt，再次 `send_user_message` 直接报 `... is quarantined; close it before recreating it`，直到 `close_codebuddy_session` 后重建。 目的是避免上一轮 prompt 仍在 CLI 里写状态时插入新 prompt。
- 真实 CodeBuddy CLI 确认 cancel 约需 10 秒，所以两处 grace 都必须明显大于它；grace 足够时被打断的那一轮是 `status: cancelled, partial: true`，会话上下文保留，下一轮可以接着用。

## 5. 主动 Interrupt

- `interrupt_codebuddy_session(session_id, grace_timeout=...)` 与 turn 超时共用 cancel-and-drain 路径；无 active turn 时返回 `stopped: false, alive: true`；grace 失败时返回 `success: false, quarantined: true`，并带上 partial `output` / `events`。
- hy-live 的取消、工作预算到期和墙钟超时后的 interrupt 都由 core 发 `AcpInterrupt`，执行器 `AcpSessionExecutor.interrupt` 传入 `_INTERRUPT_GRACE_TIMEOUT = 20.0`；core 再用 `settle` 计时器等 30 秒（`turn_settle_timeout_s`，默认 `_TURN_SETTLE_TIMEOUT_S`），超时算 interrupt 未确认。

| 场景 | 谁触发 cancel | grace | 代码位置 |
|---|---|---|---|
| 单轮 prompt 业务超时 | Sandbox `send_user_message` | 30 秒（`_CANCEL_GRACE_TIMEOUT`） | `session_runtime.py` |
| 用户取消、工作预算到期、墙钟超时后 | Host `interrupt_codebuddy_session` invoke | 20 秒（`_INTERRUPT_GRACE_TIMEOUT`） | `coordination/backend.py` |

## 6. 审批 defer 与 timeout

- `permission_policy: defer` 时 hy-live 设置 `deferred_permission_timeout: None`，避免 runtime 在等用户审批时用本地 timer 误 cancel（`coordination/backend.py::_codebuddy_input`）。
- 等审批的时间不计入 2000 秒工作预算，但仍计入 `timeout_s` 墙钟上限；其中 episode 冻结的那部分不计。

## 7. 仍然存在的问题

- **静默 quarantine**：沙箱超时那一轮的返回值里没有 `quarantined` 字段，interrupt 也报成功，Host 要到下一轮发请求时才知道。 Host 能看到的情况（interrupt 未确认、这一轮 30 秒内没回来）已经会关闭会话、下一轮重建。 补这个标记要改沙箱代码。
- **取消落空**：取消正好落在「下一轮已经发出、还没到沙箱」的几百毫秒里时，这次 interrupt 可能落空，那一轮照常跑完。 低概率，不处理。
- **其他调用方不联动**：公共 `run_codebuddy_multiturn` 在 Host 超时后不发 cancel，hy-live 是在 core 里补的 interrupt；直接用公共 agent 的调用方仍需自己处理。
- **两处 grace 各自配置**：runtime 30 秒与 hy-live 20 秒不是同一个值；要统一需让 `run_codebuddy_session_turn` 接受 Host 传入的 grace。
- **可观测性**：日志里还不区分工作预算到期、墙钟超时、沙箱 prompt 超时、grace 超时和 quarantine。
- **Owner 级清理**：一个 owner 结束时统一关闭它的全部会话还没有实现。

## 8. 关键代码索引

| 主题 | 路径 |
|---|---|
| Sandbox turn 超时 + grace + quarantine | `nexus/extensions/agents/codebuddy/session_runtime.py` |
| 墙钟上限配置、传给沙箱的 `timeout`、`_codebuddy_input` | `coordination/backend.py`（`WorkAgentBudget`、`AcpSessionExecutor._run_turn`、`_run_codebuddy`） |
| 2000 秒工作预算、墙钟上限、`WORK_AGENT_TIMEOUT`、超时 interrupt / 重启会话 | `hylive.turns`（`ts_supervise`、`ts_on_timer`、`ts_interrupt`、`ts_finish_overdue`、`ts_restart`） |
| interrupt grace | `coordination/backend.py`（`AcpSessionExecutor.interrupt`、`_INTERRUPT_GRACE_TIMEOUT`） |
| 取消先撤回权限、跳过已取消 Run 的待发轮次 | `hylive.coordinator`（`stop_task`、`turn_cancelled`、`release_turn`）、`hylive.harness`（`close_permissions`） |
| Invoke long-poll | `nexus/runtime/server/app.py` |
| 单测 | `tests/unit/extensions/projects/hy_live/test_hy_live_executor.py`、`test_hy_live_shell.py`、HyLive `tests/test_turns.py`、`tests/unit/extensions/agents/test_codebuddy_session_runtime.py` |
