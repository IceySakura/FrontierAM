# Harness Event 的生成规则

> **代码位置**：账本和 Event 在 `hylive` 包（`hylive.harness`、`hylive.harness_models`、`hylive.execution_events`），协调在 `hylive.coordinator.Coordinator`，回合调度在 `hylive.turns.TurnScheduler`，通知在 `hylive.notices`；nexus 的 `coordination/backend.py::AcpSessionExecutor` 执行 ACP 原语，`hylive_shell.py` 把沙箱推回的 update 交给 core。

> **边界**：本文讲 harness 第三层（Task / Run / Event）的 Event 何时、由谁、写成什么样。 实时 ACP 推送的链路见 [`live_progress_source.md`](live_progress_source.md)。

## 1. 结论

- 所有 Event 都经 `Harness.append_event`（`hylive.harness`）写入，都挂在一个 Run 下面，只存在内存的 `harness.events` 里。
- 来源有两类：harness 自己在状态变化时写；`Coordinator.on_update` 把 CodeBuddy 的 **原始** ACP `session/update` 交给 `ExecutionEventRecorder`（`hylive.execution_events`）加工后写。
- 两种 Event 写完后会原地修改：`tool_call`（工具结束时补齐）、`notice`（送达时改状态）。 其余 Event 不再变。

## 2. 字段

`Event`（`hylive.harness_models`）：`id`、`owner_path`（Run 路径）、`kind`、`text`、`title`、`objective`、`status`、`started_at`、`ended_at`、`queue_position`、`notify`、`priority`、`delivery_id`、`delivered`、`payload`、`created_at`。 `title` / `objective` / `status` 不传时分别取 `kind` / `text` / `kind`。 中脑 `TaskGet` 只看到其中的 `kind`、`title`、`status`、`text`、`created_at`。

## 3. 执行过程：原始 ACP update 怎么变成 Event

```text
沙箱 CodeBuddy ─► AcpUpdateRelay ─sio acp_session_update─► AcpUpdateRouter
                                                           └► AcpUpdate ─► Coordinator.on_update ─► ExecutionEventRecorder.record
invoke 结果里的 acp_events（一帧都没收到时整轮重放）──────────┘
```

| ACP update | 处理 |
|---|---|
| `agent_message_chunk` | 不立即写。 追加到这个 Run 的文字缓冲（不限长度），记下第一段到达时间 |
| 其余（`agent_thought_chunk`、`plan`、`session_info_update` 等） | 丢弃 |
| `tool_call` | 先把缓冲写成一条 `display`（截 800 字，带起止时间）并清空；再写一条 `tool_call`（`status=pending`，`ended_at` 留空），记为这个工具「未结束」 |
| `tool_call_update` | 只在 `completed` / `failed`，或带 `rawOutput` / `content` 时处理。 按「Run + toolCallId + 状态」去重。 找到未结束的那条 `tool_call` 就原地补齐：`ended_at`、`status`、正文改为「标题 + 输出摘要」；找不到才新写一条 `tool_result` |

- 工具标题取 `visible_tool_text`：非泛化的 `title`，否则「工具名 + 文件名」，最多 240 字。 结束时用的是 **结束那条 update** 的标题。
- 输出摘要取 `tool_update_output`：`content` 里的文字加 `rawOutput`，压缩空白，最多 400 字。
- 回合结束时缓冲里剩下的文字不写成 `display`，由 `take_display` 完整取走，作为这一轮的结果文字：`completed` 通知的 `result` 不截断，`result` Event、追问的回答和失败原因仍截 800 字。 这一轮一条 update 都没收到时退回 `final_output`，截 800 字。
- 本文的「N 字」都按 `text_clip.clip_text` 计：一个中日韩字符算一个，其余按空白分词、一个词算一个；没有空白的长串每 32 个字符算一个词。
- Run 已取消后，那一轮还在推的 update 照样记录。

推送与重放的规则（丢了一部分帧时不补）见 [`live_progress_source.md`](live_progress_source.md) 第 2 节。

## 4. Task 与 Run 的生命周期

中脑只看到 Task：工具都按 `task_id`（同一个 owner 内自增，`task_1`、`task_2`……）操作，快照、工具结果里没有 Run 或 Event 的路径。 **一个 Task 只有一个 Run**，就是它的执行，只留在 harness 内部，Event、权限请求和结果都挂在 Run 上。 `TaskUpdate` 不开新 Run：执行中的 Run 被 steer，已结束的 Run 重新开始。

Run 在后端可能经过几轮（turn）：第一轮，以及之后为更新重新开始或兜底重发的轮次。 core 给每一轮一个操作 id：第一轮就是 Run 路径，之后是 `Run 路径#2`、`#3`……，并记住 Task 当前是哪一轮（`MidbrainState.current_turns`）。 不是当前轮的完成、权限请求和补充问题一律丢弃或按取消回掉，所以被打断的一轮晚到的结果不会算到 Run 上。 轮次只在 core 和后端之间使用，不进快照，也不给模型。

### 4.1 Run 状态

| 状态 | 含义 | 何时进入 |
|---|---|---|
| `initializing` | 已受理，后端还没开始这一轮 | `TaskCreate`；已结束的 Task 收到 `TaskUpdate` 时（`Harness.restart_run`） |
| `running` | 后端当前这一轮已经开始 | `Coordinator.on_turn_started`，即 core 把当前这一轮发给后端时 |
| `needs_permission` / `needs_input` | 这一轮在等用户批权限或补充 | 见第 5 节；答复后回到 `running` |
| `completed` / `failed` / `cancelled` | 终态 | 见下文「回合结束的顺序」；之后只有 `TaskUpdate` 能让它重新开始，其他非终态更新一律忽略 |

- Task 的 `status` 就是它 Run 的状态。 等权限、等补充时 Task 也是 `needs_permission` / `needs_input`。
- `initializing` 也算进行中（`ACTIVE_RUN`），占用并行名额，可以被 `TaskStop` 停掉。 同时进行中的 Task 最多 3 个，`TaskCreate` 超出时返回 `busy`；已结束的 Task 再更新要重新占名额，已满时也返回 `busy`。
- `running` 在后端这一轮开始时就写入，WorkBuddy CLI 冷启动、`session/new` 的耗时算在 `running` 里，不算在 `initializing` 里。
- 没有排队：同一个 Task 上的新要求不会排在当前这一轮后面，见 4.2。

### 4.2 `TaskUpdate`：一律 steer

`Harness.update_task` 收到 description 时，把它记进 Task 的 `updates`，在 Run 下写一条 `update` Event；Run 已结束时先让它重新开始。 只改 subject 时什么都不发。 返回里的 `steered` 表示更新到达时 Run 还在进行中。 Task 的完整要求是创建时的 description 加上历次更新，用空行连接（`Harness.brief`）。

core 按更新到达时当前这一轮的情况处理：

| 情况 | 怎么做 |
|---|---|
| 这一轮还没交给后端（会话还在准备） | 不做额外操作：交接时才读 prompt，更新跟着一起发出 |
| 已交给后端，有一轮在跑（WorkBuddy） | ACP steer 插入这次的 description，还是同一轮。 WorkBuddy 还没接下这一轮的 prompt 时，`TurnScheduler` 用 `steer_retry` 计时器每秒重试，最多等 120 秒（`steer_accept_timeout_s`） |
| steer 落空，这一轮其实已经结束 | 用一轮新的发出这次的 description，同一条会话记得之前的要求 |
| steer 落空，这一轮还在跑（CodeBuddy 不支持 steer，或等超时） | 撤回这一轮的权限和问题，interrupt，再用一轮新的发出完整要求：前一轮可能还没被后端接手，只发 description 会丢掉原始要求 |
| Run 已结束 | Run 重新开始，用一轮新的发出这次的 description；会话从没起来过时发完整要求 |

一个 Task 每次只返回一次结果：被打断或错过更新的那一轮，结果按过期丢弃。 `TaskUpdate` 不负责取消，停止执行用 `TaskStop`。

### 4.3 中脑读到的 Task

- 快照：每个 Task 的 `task_id`、`subject`、`status`（`allow_all` 时另带一项），开着的权限请求和补充问题带 `task_id` 与 `task_title`，产物带 `artifact_id`、`title`、`task_id`、`task_title`。
- `TaskList`：每个 Task 的 `task_id`、`subject`、`status`、`updated_at`（最近一条 Event 的时间）和 `summary`（最近一条 Event 的种类加前 160 字）。
- `TaskGet`：`description`、`updates`、状态与时间、这个 Task 最近 60 条 Event（每条只有 `kind`、`title`、`status`、`text`、`created_at`）、产物、开着的权限请求与补充问题。

### 4.4 各时机写的 Event

| 时机 | 谁写 | Event |
|---|---|---|
| 中脑 `TaskCreate` / `TaskUpdate` 受理 | `Harness.create_task` / `update_task`，`Coordinator` 立即返回 `accepted` | `TaskCreate` 写 `request`（`status=initializing`），正文是 description；`TaskUpdate` 写 `update`，正文是这次的 description。 core 按 Task 链式交接 |
| 后端回合真正开始 | `TurnScheduler.ts_start_turn` → `Coordinator.on_turn_started` | Run 变为 `running`；执行回合写 `progress`「已经开始执行」；追问回合不写 |
| 执行中 | 第 3 节 | `display`、`tool_call`、`tool_result` |
| 中脑 `ask_backend` | `Coordinator.ask_backend` | `question` |
| 回合结束 | `Coordinator.on_turn_ended` | 见下 |

回合结束的顺序（`Coordinator.on_turn_ended`、`finish_completion`）：

1. 把这一轮还开着的补充问题按 `cancel`、权限按 `cancelled` 收掉。
2. 取走剩余文字作为结果文字；这一轮一条 update 都没收到时才退回 `final_output`。
3. Run 没被取消时，扫描该 Task 的工作目录（`<working_dir>/<task_key>`）登记产物，每个新产物在 Task **最新** 的 Run 上写一条 `artifact`。 WorkBuddy（`backend_adapter='workbuddy'`）的回合中，已完成的文件写入和 `presentFiles` / `openResultView` 显式展示的文件已经即时登记（`展示产物` display Event），`openBrowser` 写成 `打开网页` display Event；回合末的扫描仍然保留，兜住 Bash 生成的文件。
4. 追问回合（中脑 `ask_backend`）：写 `answer`，通知 `answer`；Run 已取消也照样通知，因为 `ask_backend` 允许追问已取消的工作。
5. Run 已经是终态（含已取消）：到此为止。
6. 执行失败或被 CodeBuddy 自己取消：`set_run_status` 写 `failed` / `cancelled`（正文是原因），通知 `failed`。
7. 产物收集失败：写 `failed`，通知 `failed`。
8. 正常完成：Run 变为 `completed`， **不写状态 Event**；写 `result`，通知 `completed`。

追问没能交给后端（Session 没起来或 `send` 报错）时，写 `answer`「没有得到回答」并通知 `answer`。 一轮没能交给会话时，Run 记为基础设施失败（`HANDOFF_FAILED`）并通知 `failed`。

`set_run_status` 只在带说明文字时写 Event：`running` 写成 `progress`，其余写成状态本身。 Run 进入终态后，之后的非终态状态更新一律忽略。

### 4.5 账本怎么得知后端的进展

账本不查询后端，Run 的状态都由后端回调推进：

```text
TurnScheduler.ts_start_turn ─► Coordinator.on_turn_started ─► running
  └─ AcpPrompt ─► AcpSessionExecutor._run_turn
                    await runtime.invoke(...)        沙箱里等这一轮结束
                    └─ AcpTurnEnded ─► TurnScheduler.ts_on_turn_ended ─► Coordinator.on_turn_ended ─► completed / failed / cancelled
```

- 一轮何时结束由沙箱的 `send_user_message`（`nexus/extensions/agents/codebuddy/session_runtime.py`）判断。 ACP v1（CodeBuddy）以 `session/prompt` 的响应为结束。 ACP v2（WorkBuddy）的 `session/prompt` 只表示受理并带回 request id，结束以同一 request id 的 `session/update`（`state_update`，`state: idle`）为准（`_prompt_until_idle`）。 `stopReason` 决定这一轮是 `ok`、`cancelled` 还是 `failed`；工作预算内等不到结束记 `timeout`，见 [`timeout_and_cancel.md`](timeout_and_cancel.md)。
- 权限请求和补充问题经 host relay 进入 core，Run 变为 `needs_permission` / `needs_input`，见第 5 节。
- 会话里一轮没结束时，只能插入（steer），不能再开普通的一轮：WorkBuddy 的 ACP 只声明 `steer` 一种 prompt 模式，一轮没结束时再发普通 `session/prompt` 会报 `already processing a prompt`；沙箱的 `send_user_message` 也按会话加锁。 所以 interrupt 之后重发的那一轮，以及 CodeBuddy 上追问的那一轮，会在 core 里等前一轮结束：会话忙时 `TurnScheduler.ts_enqueue` 把它放进这条会话的队列，前一轮结束时 `ts_next` 取出发出；取消时被取消 Run 的轮次从队列里丢掉。

## 5. 权限与补充问题

| 触发 | Event |
|---|---|
| CodeBuddy 请求权限 | `permission`，`payload` 带 `permission_id` 和 `detail` |
| Task 已「允许全部」时再来的权限 | 立即写 `permission_answer`「网关默认允许」，不再通知小脑 |
| 中脑答复 `allow` / `reject` | `permission_answer`「允许这一次」/「拒绝」 |
| 中脑答复 `allow_all` | 在 Task 最新的 Run 上写「这项工作之后的权限默认允许」，再对还开着的请求逐条写「允许全部」 |
| 取消、回合结束时还开着的权限 | `permission_answer`，`decision=cancelled` |
| CodeBuddy 提问（`AskUserQuestion` 或 form `elicitation/create`，见 [`question_channel.md`](question_channel.md)） | `input`，`payload` 带 `input_id`；Run 变成 `needs_input`，同时发 `input` 通知 |
| 补充问题被答复（`answer_task_input`） | `input_answer`，正文是答复或 action |

## 6. 通知（`notice`）

- 只有后台事实会写成 `notice`：`notice_sink`（`hylive.notices`）收到的 `permission`、`input`、`completed`、`failed`、`answer`。 中脑对小脑请求的回执不写。
- `Harness.push_notice` 记一条送达记录（`harness.deliveries`），在 Run 下写 `notice`：`status=queued`、`notify=true`、优先级、`delivery_id`、`delivered=false`、`queue_position`。 正文是事实本身（`prompts.notice_fact`），不含给小脑的模板说明。 默认优先级见 `NOTICE_PRIORITIES`。
- 通知交给小脑时（`notice_release`）调用 `Harness.mark_delivered`：改为 `status=delivered`、`delivered=true`，补 `ended_at`。
- 取消不发后台通知。 用户由中脑 `TaskStop` 之后的 `reply(status=ok)` 回执得知已停掉。
- `completed` / `failed` 默认直接用后端的结果或失败原因排队。 `task_config.summarize_outcomes` 为 `true` 时先交中脑 `conclude` 写一句总结；中脑出错时按后台原话播报，不会吞掉。
- 权限请求或补充问题的通知排队期间，请求被撤回或被「允许全部」一并放行，通知照样送给小脑。

## 7. 验证

HyLive `tests/test_harness.py` 断言账本的状态与 Event；`tests/test_replay.py` 用录下的 ACP update 驱动 core，逐条比对模型看到的快照与工具结果。
