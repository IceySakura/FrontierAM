# mixed_task_and_cancel

闲聊 → 联网查上海温度 → 并行写贪吃蛇和俄罗斯方块 → 两条权限通知 → 中途取消贪吃蛇 → 让贪吃蛇继续 → 两项都做完。

## 1. 数据

已冻结：[`data/mixed_task_and_cancel.json`](data/mixed_task_and_cancel.json)

| 轮 | 文本 | 门控 | 预期 |
|---|---|---|---|
| 1 | 你能做什么 | `immediate` | 小脑直接回答，不调工具 |
| 2 | 查一下上海今天的温度 | `immediate` | 小脑 `web_search` |
| 3 | 写一个贪吃蛇 | `immediate` | `delegate` |
| 4 | 再写一个俄罗斯方块 | `after_delivery` | 贪吃蛇权限通知后再 `delegate` |
| 5 | 都允许 | `after_delivery` | 俄罗斯方块权限通知后回答两项权限 |
| 6 | 取消贪吃蛇 | `immediate` | `cancel` |
| 7 | 都允许 | `immediate` | 已全部允许时说明没有待答复的权限 |
| 8 | 贪吃蛇继续 | `immediate` | 在贪吃蛇这个 Task 上续办 |

| 项 | 值 |
|---|---|
| `user_place` | `{"country": "中国", "province": "上海", "city": "上海"}` |
| `user_now` | `2026-10-09T13:00:00+08:00` |
| `web_search_provider` | `yuanbao`，无需凭证 |
| `working_dir` / `skip_permissions` / `permission_policy` | `/workspace` / `false` / `defer` |
| `hold_next_user_until_permission` | `false` |
| `task_delivery_timeout` / `default_budget.timeout_s` | 1800 / 2400 |

搜索服务连不上时，`web_search` 的工具结果是 `搜索失败：搜索失败：<错误>`，小脑应如实说没查到；这是可接受的结果。

### 门控与时序

- 第 4 轮先等贪吃蛇的权限通知，再交出俄罗斯方块；第 5 轮等俄罗斯方块的权限通知。 两项都在等权限时才说「都允许」，避免先答复俄罗斯方块、取消贪吃蛇后才收到贪吃蛇的首次权限。
- 取消回执先于第 7 轮送达：用户回合结束时 `LiveShell` 已等中脑处理完 cancel，回执已经排进队列；下一句用户话与通知赛跑时，已排队的通知总是先送达。 所以实际顺序是：取消贪吃蛇 → 取消回执 `[通知]` → 都允许 → 贪吃蛇继续。
- 第 6 轮不用 `after_delivery`：第 5 轮的「都允许」一次答复两条。 此时再等「下一条通知」等到的是工作的完成通知，取消就落空了。 紧跟第 5 轮说出时，贪吃蛇通常还在执行，取消能落到进行中的 Run 上。
- 第 7 轮不用 `after_delivery`：两项权限已经放行，不需要再等通知。
- 第 8 轮不用 `after_delivery`：那样它要等俄罗斯方块的完成通知，会把「继续」推迟到俄罗斯方块做完之后。
- 第 5 轮答复两项后，第 7 轮时小脑应说明没有待答复的权限，不必再调用工具。 两项完成通知的先后、是否合并也不固定。
- 第 8 轮之后剧本结束，运行时继续等两个 Run 的完成通知送达小脑再收尾。

## 2. 运行

见 [`README.md`](README.md)「运行」，`--case mixed_task_and_cancel`。 `--web-search none` 可关掉搜索（工具结果变成 `搜索失败：没有接上搜索`）。

## 3. 审查清单

措辞、产物细节、时间戳、搜索到的文档内容不算偏差；结构、顺序、字段不对才算。

小脑（`cerebellum.jsonl`）：

1. 系统提示末尾 `<用户物理信息>`：位置上海，时间 2026年10月9日13时 星期五。 tools 是 7 个小脑工具。
2. 真实用户话与第 1 节的八句一致。 第一句「都允许」在贪吃蛇权限 `[通知]` 之后；「取消贪吃蛇」在俄罗斯方块权限 `[通知]` 之后（或两条合并送达之后）。
3. 「你能做什么」只用文字回答，不调工具。
4. 「查一下上海今天的温度」只调一次 `web_search`，query 是列表，含「上海」和时间（今天或日期），不 `delegate`。 工具结果是 `[web_doc N begin]` 开头的文档块、「未查询到对应搜索结果」或「搜索失败：…」；失败时小脑如实说没查到，不编温度。
5. 两次 `delegate`：贪吃蛇、俄罗斯方块，各一件，不合并。 一次 `cancel`，text 指向贪吃蛇。 「贪吃蛇继续」调用 `delegate`，text 保留「继续」和「贪吃蛇」，不写成新的一项。
6. `answer_permission` 的 text 写明工作（贪吃蛇 / 俄罗斯方块）和「这项工作以后都允许」，不含 id。
7. 取消回执是 `[通知]` + `status=ok` + 中脑的 say；完成 `[通知]` 结构是「有一项工作做完了。」/「工作：」/「结果：」，合并时以「下面几项工作都做完了。」开头逐项列出。
8. 小脑交办后只说一句接下来做什么；取消后不说成已删除；俄罗斯方块先做完时说明贪吃蛇还在做；最后一条完成通知之后有一句口语回复。

中脑（`midbrain.jsonl`）：

1. 账本调用顺序：`TaskCreate(贪吃蛇)` → `TaskCreate(俄罗斯方块)` → `answer_permission(allow_all, 贪吃蛇的 permission_id)` → `TaskStop(taskId=贪吃蛇的 task_id)` → `answer_permission(allow_all, 俄罗斯方块的 permission_id)` → `TaskUpdate(taskId=贪吃蛇的 task_id)` → 两次工作结束各只 `reply(status='ok')`。 两条权限合并答复时，两次 `answer_permission` 可以相邻。 `permission_id` 必须取自快照，不得编造；「贪吃蛇继续」再用 `TaskCreate` 算 FAIL。
2. cancel 之后的快照里贪吃蛇 `status=cancelled`、`allow_all=true`。
3. `TaskCreate` 返回 `accepted` + `task_id`；`TaskStop` 返回 `accepted` + 贪吃蛇的 `task_id` 和 `subject`；继续贪吃蛇的 `TaskUpdate` 返回 `accepted`、同一个 `task_id`、`steered=false`；`answer_permission` 返回 `answered` + `decision=allow_all` + 对应 `task_id`。
4. cancel 回 `reply(status='ok')`（这条才会变成给小脑的 `[通知]`），`delegate` / `answer_permission` 回 `status='accepted'`；`reply.say` 不含 task_id、权限 id。
5. 两个 Task 的产物、权限详情里的路径各在自己的 `/workspace/<task_id>` 下，不共用一个工作目录。

`result.json`：`task_output.tasks` 两项都是 `done`，各有送达小脑的 `task_delivery`。

FAIL 典型：搜索被 `delegate` 出去或搜了多次；取消落到俄罗斯方块；「继续」开了新 Task；`TaskStop` 缺失；任一权限不是 `allow_all`；某项工作的完成通知没有送达小脑。
