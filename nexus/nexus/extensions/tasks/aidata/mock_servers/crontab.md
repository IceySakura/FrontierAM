# crontab — 定时任务状态采集与 Rubric 校验

## 概述

Agent 通过 `shell_run_command` 执行 `crontab` 命令来创建、查看、修改定时任务。
引擎在 agent 运行结束后自动执行 `crontab -l`，将解析后的条目写入
`state['crontab_entries']`，供 rubric 校验。

## 采集流程

```
Agent 运行结束
  ↓
engine/task.py 调用 collect_crontab_state(runtime)
  ↓
sandbox 内执行 crontab -l（timeout=10s）
  ↓
解析每一行非空、非注释行为结构化 entry
  ↓
写入 state['crontab_entries']（list[dict]）
```

实现位于 `engine/state.py::collect_crontab_state()`。

## state['crontab_entries'] Schema

类型：`list[dict]`（可能为空列表或不存在）

每条 entry 的字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `schedule` | `str` | 5-field cron 表达式，如 `0 9 * * *`、`*/30 * * * *` |
| `command` | `str` | 要执行的命令（cron 行第 6 段及之后的全部内容） |
| `raw` | `str` | 原始行文本 |

当 crontab 行不符合标准 5-field 格式（少于 6 段）时，只有 `raw` 字段。

示例：

```json
[
  {
    "schedule": "0 9 * * *",
    "command": "/workspace/desktop/scripts/daily_report.sh",
    "raw": "0 9 * * * /workspace/desktop/scripts/daily_report.sh"
  },
  {
    "schedule": "0 */4 * * *",
    "command": "python3 /workspace/desktop/check_ticket.py",
    "raw": "0 */4 * * * python3 /workspace/desktop/check_ticket.py"
  }
]
```

## Agent 使用方式

Agent 通过 `shell_run_command` 操作 crontab。常见写法：

```bash
# 创建/覆盖整个 crontab
echo '0 9 * * * /path/to/script.sh' | crontab -

# 追加条目（保留已有）
(crontab -l 2>/dev/null; echo '0 */4 * * * python3 check.py') | crontab -

# 查看当前 crontab
crontab -l

# 清空 crontab
crontab -r
```

## Rubric 校验方式

### 1. tool_called — 检查 agent 是否使用了 crontab

```yaml
- id: agent_used_crontab
  description: Agent 通过 shell 设置了 crontab
  dimension: functional
  weight: 1.0
  assert:
    type: tool_called
    tool: run_command
    with_text: "crontab"
```

### 2. state_equals + jmespath — 校验 schedule 正确性

```yaml
# 至少有一条每天 9 点的定时任务
- id: has_daily_9am_job
  description: 存在每天 9:00 执行的 crontab 条目
  dimension: correctness
  weight: 2.0
  assert:
    type: state_equals
    jmespath: "crontab_entries[?schedule=='0 9 * * *']"
    count_op: ">="
    count_value: 1

# 命令中包含特定脚本
- id: check_script_scheduled
  description: check_ticket.py 被设置为定时任务
  dimension: correctness
  weight: 1.5
  assert:
    type: state_equals
    jmespath: "crontab_entries[?contains(command, 'check_ticket')]"
    count_op: ">="
    count_value: 1

# 至少有 N 条定时任务
- id: has_two_cron_jobs
  description: 设置了至少 2 条定时任务
  dimension: functional
  weight: 1.0
  assert:
    type: state_equals
    jmespath: "crontab_entries"
    count_op: ">="
    count_value: 2
```

### 3. text_contains — 基于 trajectory 的文本校验

```yaml
- id: crontab_in_trajectory
  description: Agent trajectory 中出现了 crontab 操作
  dimension: functional
  weight: 0.5
  assert:
    type: text_contains
    from: trajectory
    contains: "crontab"
```