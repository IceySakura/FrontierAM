# single_snake

写一个贪吃蛇 → 权限通知后用户说「都允许」→ 完成通知后用户嫌速度快 → 在同一 Task 上续办。

## 1. 数据

已冻结：[`data/single_snake.json`](data/single_snake.json)

| 轮 | 文本 | 门控 | 时机 |
|---|---|---|---|
| 1 | 写一个贪吃蛇 | `immediate` | 开场 |
| 2 | 都允许 | `after_delivery` | 权限通知送达、小脑转述之后 |
| 3 | 速度有点快 | `after_delivery` | 第一条完成通知送达、小脑转述之后 |

| 项 | 值 |
|---|---|
| `user_place` | `{"country": "中国", "province": "上海", "city": "上海"}` |
| `user_now` | `2026-10-09T12:55:00+08:00`（系统提示 `<用户物理信息>` 里的时间精确到小时） |
| `web_search_provider` | `yuanbao`（本 case 不应搜索） |
| `working_dir` / `skip_permissions` / `permission_policy` | `/workspace` / `false` / `defer` |
| `hold_next_user_until_permission` | `false`，用户何时开口完全由门控决定 |
| `task_delivery_timeout` / `default_budget.timeout_s` | 1800 / 2400 |

门控说明：`after_delivery` 只要求上一句用户话之后至少送达过一条通知。 第 2 轮等的是权限通知；中脑对 `answer_permission` 回 `reply(status='accepted')` 时不会给小脑回执，所以第 3 轮等到的是完成通知。 若中脑对 `answer_permission` 回了 `status=ok`，这条回执会提前放行第 3 轮，用户就会在工作进行中说「速度有点快」，审查时要指出。 第 3 轮之后剧本结束，运行时继续等第二个 Run 的完成通知送达小脑再收尾。

## 2. 运行

见 [`README.md`](README.md)「运行」，`--case single_snake`。 要关掉搜索加 `--web-search none`。

## 3. 审查清单

措辞、产物细节、时间戳因模型和后端不同而不同，不算偏差；结构、顺序、字段不对才算。

小脑（`cerebellum.jsonl`）：

1. 系统提示末尾 `<用户物理信息>`：位置上海，时间 2026年10月9日12时 星期五。 tools 是 7 个小脑工具。
2. 真实用户话依次是「写一个贪吃蛇」「都允许」「速度有点快」；「都允许」在权限 `[通知]` 之后，「速度有点快」在第一条完成 `[通知]` 之后。
3. 第一次 `delegate` 的 text 保留「贪吃蛇」；`answer_permission` 的 text 写明工作、操作和「这项工作以后都允许」，不含 id；第二次 `delegate` 的 text 说贪吃蛇速度快、要调慢，不另起新工作。 全程不调用 `web_search`、`cancel`。
4. 权限 `[通知]` 正文有「有一条权限请求需要用户决定。」「工作：」「操作：」「详情：」几行和权限指引；完成 `[通知]` 正文有「有一项工作做完了。」「工作：」「结果：」，结果是中脑写的总结。
5. 小脑交办后只说一句接下来做什么，不说已做完；转述权限时问这一次允许还是拒绝、以后要不要都允许；最后一条完成通知之后有一句口语回复。

中脑（`midbrain.jsonl`）：

1. tools 是 `TaskCreate` / `TaskGet` / `TaskList` / `TaskUpdate` / `TaskStop` / `ask_backend` / `present_files` / `answer_permission` / `answer_task_input` / `reply`。
2. 每步输入是「小脑调用了 X，按系统提示里「X」一节处理。」+「小脑的原文：」+「当前账本快照：」（JSON：`tasks` / `pending_permissions` / `pending_inputs` / `artifacts`），或「工作结束，按系统提示里「工作结束」一节处理。」+「所属任务：」（task_id）+「后端的原始结果：」。
3. 账本调用顺序：`TaskCreate`（subject 含「贪吃蛇」）→ `answer_permission(decision='allow_all')`（`permission_id` 取自快照 `pending_permissions`）→ 工作结束只 `reply(status='ok')` → `TaskUpdate(taskId=同一个 task_id)`，description 只写新增的要求 → 工作结束只 `reply(status='ok')`。 第二次用 `TaskCreate` 算 FAIL。
4. `TaskCreate` 和 `TaskUpdate` 都返回 `status=accepted` 和同一个 `task_id`，`TaskUpdate` 的 `steered` 为 false（上一轮已结束）；`answer_permission` 返回 `status=answered`、`decision=allow_all`。
5. 每步以单独的 `reply` 结束；`reply.say` 不含 task_id、权限 id。

`result.json`：`task_output.tasks` 只有一项，`status=done`；`delivery_events` 里这项 Task 至少有两条 `task_delivery`。

FAIL 典型：续办开了新 Task；`answer_permission` 不是 `allow_all`；用户话出现在对应通知之前；第二个 Run 的完成通知没有送达小脑。
