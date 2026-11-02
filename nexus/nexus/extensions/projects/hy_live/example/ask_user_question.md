# ask_user_question

CodeBuddy 用自带的 `AskUserQuestion` 问用户想做哪种游戏，用户回答后它在同一回合里接着做完，最后用户打开产物。 需要 runtime 镜像支持 `client_questions`（见 [`../docs/impl/runtime_image.md`](../docs/impl/runtime_image.md)）；CLI 默认打开，加 `--no-client-questions` 时拒跑本 case。 链路见 [`../docs/impl/question_channel.md`](../docs/impl/question_channel.md)。

## 1. 数据

已冻结：[`data/ask_user_question.json`](data/ask_user_question.json)

| 轮 | 门控 | 文本 |
|---|---|---|
| 1 | `immediate` | 让后台写个游戏。先检查工作目录，再让后台问我想做哪一种游戏（贪吃蛇、俄罗斯方块、扫雷）；收到我的答复后再开始做。 |
| 2 | `after_delivery`（通常是权限通知） | 都允许 |
| 3 | `after_delivery`（通常是补充通知） | 做贪吃蛇 |
| 4 | `after_delivery`（完成通知） | 好的，打开看看吧 |

| 项 | 值 |
|---|---|
| `user_place` / `user_now` | 上海 / `2026-10-09T16:00:00+08:00` |
| `working_dir` / `skip_permissions` / `permission_policy` | `/workspace` / `false` / `defer` |
| `client_questions` | CLI 打开，不写进冻结数据 |
| `hold_next_user_until_permission` | `false`（靠门控） |
| `max_user_turns` / `task_delivery_timeout` / `default_budget.timeout_s` | 8 / 1800 / 2400 |

预期流程：

开场明确要求目录检查和提问，保证本 case 覆盖权限与补充问题，不把后端自行选择游戏类型算作通过。

1. 「写个游戏」→ 小脑 `delegate` → 中脑 `TaskCreate`。
2. 权限通知（通常是列工作目录）→「都允许」→ 中脑 `answer_permission(decision='allow_all')`。
3. CodeBuddy 调用 `AskUserQuestion` → 账本 `pending_inputs` 出现一条 `{"id", "task_id", "task_title", "question"}`，question 是各问题按行合并、选项写成「（可选：A、B）」→ 补充 `[通知]`。
4. 「做贪吃蛇」→ 小脑 `answer_task_input` → 中脑 `answer_task_input(action='accept')` → CodeBuddy 在同一回合里接着做 → 完成通知。
5. 「好的，打开看看吧」→ 小脑 `open_artifact` → 中脑在快照 `artifacts` 里找到这项工作的产物并 `present_files(artifact_id)` → `reply(status='ok')` → 小脑收到 `status=ok` 的 `[通知]` 回执。 第 4 轮的回执在用户回合结束时已排进队列，剧本说完后 runtime 会把它送达小脑再结束。

权限和补充问题的先后不固定：CodeBuddy 也可能先问再要权限。 门控只等「下一条通知」，不看种类，两种顺序下都是每条通知放行一句；先问的话，「都允许」会落在补充问题之后、「做贪吃蛇」落在权限通知之后，小脑需要按通知内容把两句话对上。

## 2. 运行

见 [`README.md`](README.md)「运行」，`--case ask_user_question`。

## 3. 审查清单

按 `cerebellum.jsonl` 的下标顺序检查。 权限通知和补充通知谁先谁后都可以。

小脑：

1. 「写个游戏」之后先有 `delegate` → `{"status": "submitted"}`，小脑只说一句接下来去做什么；全程 `delegate` 只一次，没有 `cancel`。
2. 权限 `[通知]`（「有一条权限请求需要用户决定」，带「工作：」「操作：」行）之后，用户「都允许」，小脑调用 `answer_permission`，text 表达这项工作以后都允许。
3. 补充 `[通知]`：正文含「执行中需要用户补充」，带「工作：」和非空的「问题：」行，不含 `input_id`；不是以权限通知的形式出现。
4. 补充通知之后、用户回答之前，小脑把问题转述给用户，不调用 `answer_task_input`，也不把它当成权限去调用 `answer_permission`。
5. 「做贪吃蛇」之后，下一轮 assistant 调用 `answer_task_input`，text 写明是哪项工作并含「贪吃蛇」。
6. 之后出现完成 `[通知]`；完成通知之后，用户「好的，打开看看吧」，小脑调用 `open_artifact`，text 写明是哪项工作；之后收到 `[通知]` + `status=ok` 的回执，并口语告诉用户已经打开。

中脑（`midbrain.jsonl`）：

7. `answer_task_input` 之前的快照里 `pending_inputs` 每项只有 `{"id", "task_title", "question"}`，`pending_permissions` 里没有 AskUserQuestion。
8. `answer_task_input` 的 `input_id` 取自快照，`action=accept`，text 非空且转述了用户的选择；结果是 `{"status": "answered", "input_id": 同参数, "action": "accept", "question": 与快照相同}`，随后单独 `reply(status='accepted')`。
9. `answer_permission` 的 `decision=allow_all`、`permission_id` 取自快照；结果 `status=answered`。
10. `present_files` 的 `artifact_id` 是调用前最新快照 `artifacts` 里的一项，且那项的 `task_title` 是这项工作的标题；结果 `status=opened`、`artifact_id` 同参数，带 `title` / `url` / `how`；随后 `reply(status='ok')`，say 写明打开了哪项工作的哪个产物，不含 artifact_id 或 task_id。

CodeBuddy（`result.json` 的 `task_output.tasks[0].result.trajectory`）：

11. 有一次 `AskUserQuestion` 调用，它的工具结果里带着用户的回答（含「贪吃蛇」），而不是拒绝；之后同一回合里接着有写文件等工具调用。 `tasks[0].status` 为 `done`，产物是贪吃蛇。

FAIL 典型：

- 没有补充通知：CodeBuddy 的轨迹里若有 `ToolSearch` 报「No matching tools found for: AskUserQuestion」，多半是镜像不支持 `client_questions`；否则是 CodeBuddy 没问就直接动手（模型没走提问）。 两者在结论里分开写。
- 补充问题被当成权限：出现操作是 AskUserQuestion 的权限通知，或对着补充问题调用 `answer_permission`。
- 中脑 `action=decline` / `cancel`，或 text 为空，或结果不是 `status=answered`。
- 用户回答后 CodeBuddy 重新开了一个回合，而不是在同一回合里接着做。
- `present_files` 的 artifact_id 编造、不在快照 `artifacts` 里，或属于别的工作；结果不是 `status=opened`；小脑没收到 `status=ok` 的回执。
