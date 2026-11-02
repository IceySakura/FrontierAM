# Agent Loop：小脑、中脑与用户交互

> **边界**：本文讲一个 episode 里谁在什么时候说话：用户（或模拟用户）、小脑、中脑。 小脑、中脑、账本和通知的逻辑都在 `hylive` 包里，设计见 HyLive 仓库的 `docs/design.md`。 harness 里 Event 怎么写见 [`harness_events.md`](harness_events.md)，CodeBuddy 一轮怎么限时、怎么取消见 [`timeout_and_cancel.md`](timeout_and_cancel.md)。

## 1. 组件

| 组件 | 代码 | 职责 |
|---|---|---|
| 小脑 | `hylive.cerebellum_step`，规则见 `hylive.cerebellum_rules`，工具见 `hylive.cerebellum_contract` | 唯一可训练的 policy。 和用户对话，把要动手的事用自然语言交给后台，转述后台送回的事实 |
| 中脑 | `hylive.midbrain_loop`，工具见 `hylive.midbrain_contract` | 一通对话一个常驻 LLM 会话。 把小脑的话对应到账本里的一项，调用账本工具，再以 `reply` 交回执；也负责总结完成 / 失败 |
| 通知 | `hylive.notices` | 把中脑回执和后台事实排成通知；中脑还在处理小脑交来的工具调用时先压住 |
| harness | `hylive.harness` | Task / Run / Event 账本，权限请求、补充问题、产物、送达记录 |
| 协调 | `hylive.coordinator` | 把账本变化落到 Project Session：一个 Task 一条 Session；`TaskCreate` / `TaskUpdate` 受理后立刻返回，各 Task 在后台链式启动回合 |
| 回合调度 | `hylive.turns`（v1 边界） | 回合排队、工作预算、steer、btw 追问、超时、interrupt 与重启会话 |
| 沙箱执行器 | `coordination/backend.py::AcpSessionExecutor` | 只执行 ACP 原语：开会话、一轮 prompt、interrupt、steer、btw 追问、关会话，并中继 ACP update |
| episode 外壳 | `hylive_shell.py::LiveShell` | 驱动 hylive 的 runner；让用户发言和后台通知赛跑，直到用户结束且没有进行中的 Run；组装输出、录制与追踪 |
| 装配 | `agent.py::live_agent` | 按 `task_config` 组装以上组件，产出 `LiveAgentAgentOutput` |

## 2. 小脑可见工具

| 工具 | 参数 | 结果 | 去向 |
|---|---|---|---|
| `delegate` | `text` | `{"status": "submitted"}` | 中脑 `TaskCreate`，对已有任务的补充或修改用 `TaskUpdate` |
| `cancel` | `text` | 同上 | 中脑 `TaskStop` |
| `question` | `text` | 同上 | 中脑 `TaskGet`，不够时 `ask_backend` |
| `open_artifact` | `text` | 同上 | 中脑 `present_files(artifact_id)` |
| `answer_permission` | `text` | 同上 | 中脑 `answer_permission(permission_id, decision=allow/reject/allow_all)` |
| `answer_task_input` | `text` | 同上 | 中脑 `answer_task_input` |
| `web_search` | `query: list[str]` | 搜到的文档 | 小脑自己执行，不经过中脑 |

工具列表从头到尾不变。 小脑的 `text` 不带任何 id 或 path，id 只出现在中脑看到的账本快照里。 一句话里有几件事时，小脑在一次回复里并发几个调用。 工具名不认识或参数不是 JSON 对象时，工具结果是 `{"status": "failed", ...}`，不会交给中脑。

## 3. 一个用户轮次

```text
user message
→ 小脑 tool call（只调用工具，这次回复不说话）
→ {"status": "submitted"}
→ 小脑说一句接下来去做什么（不允许调工具的回执）
→ 中脑处理 (工具名, text)；处理完之前，通知先压住，外壳也等它处理完才进入下一轮
```

用户这一轮一直持续到中脑处理完它交来的工具调用，这一轮里小脑不转述任何通知。 这一轮里到达的通知、以及这一轮开始时小脑攒着还没开始说的通知，由训练配对层（`hylive.step`）先收着，这一轮结束后一起交给小脑，中脑回执排在最前面。 这和重构前（`await app.join()`）一致；线上小脑没有这一条，见 HyLive `docs/design.md` 4.3。

中脑每一步：读到「小脑调用了 X」加最新账本快照 → 调用账本工具 → 看过结果后单独调用 `reply(status, say)`。 `accepted` 回执不再告诉小脑（小脑已经说过要去做）；`ok` / `ask` / `failed` 回执以 `[通知]` + `status=...` 单独送回。

中脑的账本工具是 `TaskCreate` / `TaskGet` / `TaskList` / `TaskUpdate` / `TaskStop`，加上 `present_files`、`ask_backend`、`answer_permission`、`answer_task_input` 和回执 `reply`，都按 `task_id` 操作。 用户对同一件工作补充或修改要求时，中脑用 `TaskUpdate` 只交新增或变化的内容；它一律按 steer 交给正在执行的这一轮，不排队，也不开新 Run；工作已结束时同一个 Run 在同一条会话里重新开始。 Task 与 Run 的关系、状态和 steer 的细节见 [`harness_events.md`](harness_events.md) 第 4 节。

完整时序示例：

```text
User: 写一个贪吃蛇
Cerebellum: delegate(text) → submitted → 「我这就去写」
Midbrain: TaskCreate → reply(accepted)
Core: 新建 Project Session，开始执行
Harness → [通知] 有一条权限请求需要用户决定
Cerebellum: 转述，问这次允许还是拒绝、以后要不要都允许
User: 都允许
Cerebellum: answer_permission(text) → submitted
Midbrain: answer_permission(decision=allow_all) → reply(accepted)
Core → [通知] 有一项工作做完了（默认用后端原文；task_config.summarize_outcomes 为 true 时由中脑写总结）
Cerebellum: 告诉用户做好了
```

中脑答复 `allow_all` 之后，同一 Task 后续的权限请求由 harness 直接放行，不再经过小脑和中脑。

## 4. 系统插入的消息

送回小脑的都是 role=user 消息，模板在 `hylive.prompts`：

| 开头 | 来源 | 小脑要做什么 |
|---|---|---|
| `[通知]` | 中脑回执（`status=ok/ask/failed`）；后台事实：权限请求、补充问题、追问结果、完成、失败 | 现在告诉用户；权限请求要问这次允许还是拒绝、以后同类要不要都允许 |
| `[失败]` | 小脑这一轮生成出错（`FAILURE_TEMPLATE`），由小脑自己插入 | 一两句话说没做成 |

小脑系统提示里还定义了 `[记录]`（已发生、现在不用说的事实），当前不会发出这种消息。

通知的取法（`hylive.cerebellum_rules.take_notices`）：回执优先、一次一条；后台事实只取最高优先级的那一批一起讲，其中几项完成合并成一条「下面几项工作都做完了」（`hylive.prompts.MULTIPLE_COMPLETIONS_HEAD`）。 优先级见 `hylive.harness.NOTICE_PRIORITIES`。

## 5. Episode

进入 episode 之前，`live_agent` 先 `LiveShell.warm_up()`：拉起沙箱并装好 Node proxy preload。 沙箱启动要几分钟，若放在 episode 里，用户会在任何 Task 能动手之前就来问进度；拉不起来时直接抛错，不产生小脑轨迹。

`LiveShell` 每次同时等两件事：模拟用户的下一句和小脑开始转述下一条通知，谁先到先处理。

- 通知先到：等小脑转述完，记为一条 `CerebellumResponse(source=notification)`。
- 用户先到：跑一个用户轮次。
- 剧本轮的门控：`gate: after_delivery` / `after_progress`（以及 `hold_next_user_until_permission` 打开且还没送过权限通知时）先等一条通知再放行下一句；若没有进行中的 Run 且没有排队通知，则不再空等（`hylive_shell.py`）。
- 用户结束后（剧本用完、说 `[DONE]` 或用完 `max_user_turns`），继续送通知，直到没有进行中的 Run 且队列为空。 小脑听完所有通知后，还开着的权限按拒绝、补充问题按拒答处理（训练配对层代用户答复，和用户亲口拒绝走同一条路径），后脑自己收尾。
- `task_delivery_timeout`（默认 1800 秒）是从 episode 开始算的总截止，到期还有活没完记 `episode_termination=TASK_TIMEOUT`。
- episode 结束时外壳发 `SessionClosing`，core 关掉自己开的会话，之后不再调用中脑，所以 episode 结束后不会再交办工作。

终止原因（`episode_termination` 优先，否则由任务结果推出）：

| 情况 | termination_reason | 训练 |
|---|---|---|
| 小脑格式错误（见第 8 节） | `FORMAT_ERROR` | 打分 |
| 小脑上下文超长 | `TRUNCATION` | 按 HyTuner 对 TRUNCATION 的处理 |
| 中脑调用失败（回执或 `conclude`） | `ENV_ERROR` | mask |
| 任一 Run 是基础设施失败：CodeBuddy 没有干净结束（非零退出、内部 429、空轨迹），轮次没能交给会话，取消时中断失败 | `ENV_ERROR` | mask |
| 总截止时还有活没完 | `TASK_TIMEOUT` | mask |
| 其余，包括单个 Run 超过 2000 秒工作预算、CodeBuddy 正常结束但没做成 | `ENV_DONE` | 打分 |

## 6. 输出

| 字段 | 内容 |
|---|---|
| `policy_messages` | 小脑最近一次实际请求的 messages 加原始回复（`agent_output.main_trajectory`），见下 |
| `tasks` | 每个 harness Task 一条 `AgentTask`：`task_id` 是 `task_1` 这类任务 ID，状态取 Task 的 Run，`target_session_id` 是这个 Task 的 Project Session |
| `delivery_events` | 送达小脑的权限请求（`permission_request`）、补充问题（`backend_input_request`）和完成 / 失败 / 追问结果（`task_delivery`） |
| `delivery_responses` | 小脑对每条通知的回复 |
| `conversation` | 用户可见对话，只含真实用户发言和小脑回复 |
| `cerebellum_trajectory` / `midbrain_trajectory` / `dialogue_trajectory` | 小脑、中脑的完整消息日志（`{"tools", "messages"}`，格式见 `trajectory_export.py`），以及按时间合并的用户轮视图 |

`policy_messages` 的来源：

- 由 `MainTrajectoryProcessor` 捕获，不由 agent loop 拼装。 采集按 `_client_key == policy_client_name` 过滤（默认 `main`），不混入中脑、CodeBuddy 或 User Simulator。 每次 agent 执行使用独立 processor context，正常返回时读取 `snapshot_result()` 的深拷贝。
- 已进入请求的系统提示、系统插入的消息、工具结果均保留；回复的 reasoning、tool calls 和空值字段不经清洗。
- 这是 **最新一次调用的消息视图** ，不是把所有调用分支合并成一条历史；尚未发送给模型的尾部通知不在其中。 捕获边界是公共 completion 接口，不是 provider 内部最终 HTTP payload。
- 异常或取消 **不产出部分轨迹** ：只在小脑 span 上记录 `hytuner.termination_reason=UNHANDLED_ERROR` 后原样抛出。

训练轨迹是 `HyTunerRunOutput.segments`。

## 7. ID 归属

| ID | 创建者 | 使用者 | 是否给模型 |
|---|---|---|---|
| `owner_id` | Runner / 在线会话层 | harness、brain | 否 |
| `task_id`（owner 内自增，`task_1`……）、`permission_id`、`input_id`、`artifact_id` | harness | 中脑账本工具 | 只给中脑（账本快照和工具结果），不给小脑 |
| Run / Event 路径 | harness | brain、后端 | 否 |
| `task_key`（Task 内部 12 位 uuid） | harness | brain，作为工作目录名 `<working_dir>/<task_key>` | 否；后端原话里的路径会带上它，但它和 `task_id` 长得不一样，不会被当成任务编号 |
| Project Session ID（`hy-live:<owner_id>:project:<task_id>`） | brain | 后端；规范化成 UUID 后用作 runtime session 与 LLM channel 的 id | 否 |
| ACP remote session ID | CodeBuddy | ACP 连接 | 否 |

中脑的各种 id 只能取自最新账本快照或刚调用的工具结果。

## 8. 训练边界

只有小脑是训练对象：

| 内容 | Mask |
|---|---|
| 小脑自然语言 | 1 |
| 小脑 tool call | 1 |
| User / User Simulator message | 0 |
| 工具结果（`submitted`、搜索文档） | 0 |
| 系统插入的 `[通知]` / `[失败]` | 0 |
| 中脑对话、CodeBuddy 轨迹 | 0 |
| LLM User Simulator sidechain | 0 |

该规则尚需在 HyTuner `session_agent_loop` 做联合验证。

小脑请求必须一直是上一次请求的前缀，HyTuner 发现请求不再是上一次的前缀时会 mask 整条轨迹。 因此：

- **历史只追加、不改写**：不截断，调工具那条回复的文字照样保留，回复不去首尾空白，和模型实际输出逐字一致。
- **不允许调工具的轮次照样带上 tools**：承接语、播报通知、失败重试这三类轮次只把 `tool_choice` 设成 `none`。 工具列表会渲染进 prompt，中途去掉会破坏前缀。
- **格式错误不重试**：文字里出现没被解析的工具调用标记（DeepSeek DSML、HY4 的 `<tool_calls…>`），或这一轮没有可说的话，直接以 `FORMAT_ERROR` 结束 episode，不注入失败说明重试。 工具调用标记只在 chat completion 里出现，由 `hylive_shell.py` 的小脑模型端口检查，以 `LLMResult(error_kind='format')` 交给 `cerebellum_step`；没话可说由 `cerebellum_step` 自己判定。 上下文超长以 `TRUNCATION` 结束；其它异常注入 `[失败]` 说明后再生成一次。

`tool_choice=none` 的轮次在各条路径上的行为：

| 情况 | 行为 |
|---|---|
| 走 API（example、E2E） | 服务端不返回 `tool_calls`；模型若把工具调用写成文字（DSML），判 `FORMAT_ERROR` |
| 走 HyTuner 客户端（训练） | 客户端不读 `tool_choice`，可能返回 `tool_calls`：不执行，原样留在历史里（后面没有工具结果，HY4 模板可以正常渲染），只把文字部分当作回复；没有文字判 `FORMAT_ERROR` |
| 任意轮次，HyTuner 解析不出工具调用 | HY4 parser 不抛异常，而是把原文连同 `<tool_calls…>` 标记放进 content，判 `FORMAT_ERROR` |

**设计 reward 时**：这三类轮次里出现 `tool_calls` 不算格式错误，episode 照常结束，需要 reward 检查并惩罚。
