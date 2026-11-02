# ask_permission_reject

CodeBuddy 经 ACP 要权（`permission_policy=defer`），用户拒绝。 小脑用自然语言调用 `answer_permission`，中脑填 `decision=reject`；之后这项工作以完成或失败的 `[通知]` 收尾，小脑不得再派任务。 「允许」一侧由 `single_snake`（`allow_all`）和 `multi_task_query`（逐条 `allow`）覆盖。

必须等权限通知进轨迹之后，才出现「我拒绝这个权限请求」。 开了 `hold_next_user_until_permission` 就是为了挡住剧本抢跑。

## 1. 数据

已冻结：[`data/ask_permission_reject.json`](data/ask_permission_reject.json)

| 轮 | 文本 |
|---|---|
| 1 | 请只读检查当前工作目录的顶层文件，按名称和用途做简短总结；不要修改、创建或删除任何文件。 |
| 2 | 我拒绝这个权限请求，不允许执行这次后台操作。 |
| 3 | 刚才的权限请求你怎么处理的？不要重新派任务。 |

| 项 | 值 |
|---|---|
| `user_place` / `user_now` | 上海 / `2026-10-09T17:00:00+08:00` |
| `working_dir` / `skip_permissions` / `permission_policy` | `/workspace` / `false` / `defer` |
| `hold_next_user_until_permission` | `true`：第 2 轮等权限通知送达小脑才放行 |
| `max_user_turns` / `task_delivery_timeout` / `default_budget.timeout_s` | 5 / 1800 / 1200 |

剧本没有写门控，三句都是 `immediate`；第 2 轮靠 `hold_next_user_until_permission` 挡住，第 3 轮紧跟拒绝之后说。 剧本说完后，运行时继续等这项工作的完成或失败通知送达小脑再收尾。

## 2. 运行

见 [`README.md`](README.md)「运行」，`--case ask_permission_reject`。

## 3. 审查清单

核心不变量：权限通知先到，用户再拒绝。 顺序反了就是 FAIL，即使最后任务失败。

按 `cerebellum.jsonl` 的下标顺序检查，并写出权限通知、拒绝句、`answer_permission` 三者的下标，断言依次递增：

1. `delegate` → `{"status": "submitted"}`；全程 `delegate` 只一次。
2. 权限 `[通知]`「有一条权限请求需要用户决定」，带「工作：」和非空的「操作：」行，不含 `permission_id`。 操作应与 CodeBuddy 轨迹里那次工具调用对得上。
3. 拒绝句在权限通知之后。 权限通知之后，小脑先说明是哪项工作、想做什么并询问用户；拒绝句出现之前，不得 `cancel`，也不得 `answer_permission`。
4. 拒绝句之后，小脑调用 `answer_permission`，text 写明是哪项工作、哪次操作和拒绝；结果 submitted。
5. `midbrain.jsonl` 里对应的 `answer_permission`：`permission_id` 取自快照 `pending_permissions`，`decision=reject`；结果 `status=answered`、`decision=reject`；随后单独 `reply`。 拒绝是否生效以中脑 `decision` 和后续任务结果为准，不能只看小脑的措辞。
6. 之后出现这项工作的 `[通知]`：「有一项工作失败了」，或后端被拒权后自行收尾的「有一项工作做完了」。 `tasks[0].status` 为 `failed` / `cancelled` / `done` 之一；`done` 时 `tasks[0].result` 应说明操作被拒、没能读到目录。
7. 追问「刚才的权限请求你怎么处理的？不要重新派任务。」之后不再 `delegate`。
8. `cancel` 要么不出现，要么出现在 `answer_permission` 之后。

FAIL 典型：拒绝句在权限通知之前；没有权限通知，小脑直接 `cancel`；中脑 `decision=allow` / `allow_all`；结果明明被拒，小脑却说成功读到了目录。
