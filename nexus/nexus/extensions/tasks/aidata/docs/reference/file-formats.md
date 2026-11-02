---
title: 文件契约
audience: 需要读或写 case bundle 、result.json 、junit XML 的开发者
prerequisites: 无
related:
  - ../how-to/author-a-case.md
  - ../how-to/author-a-rubric.md
  - ../../RUBRIC.md
---

# 文件契约

aidata 的所有输入 / 输出 都是 YAML / JSON / XML 文件 —— 不改这些就是稳定 。 本篇是每种格式的
字段字典 。 深入 rubric 字段见 [`../../RUBRIC.md`](../../RUBRIC.md) 。

## `task.yaml` （case manifest ）

case bundle 的 root manifest 。 legacy 名 `case.yaml` 仍支持 —— CLI 会先找 `task.yaml` ，找不到再找
`case.yaml` 。

### 必填字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `task_id` | string | 人类可读标签 ，出现在 output 里 。 |
| `metadata.created_at` | string | ISO 日期 `YYYY-MM-DD` ，正则见 `METADATA_DATE_RE` 。 |
| `metadata.tags` | list[string] | 至少一个 tag ；用于分类 、 filter 。 |

### 常用可选字段

| 字段 | 默认 | 说明 |
|---|---|---|
| `agent_timeout` | 无（必显式给） | agent 循环 wall 上限 秒 。 |
| `observation_time_timeout` | 无 | Main-style agent 单独限制 observation / tool 总耗时 。 |
| `max_iterations` | 无（必显式给） | agent 最大迭代轮数 。 |
| `user_prompt` | 无 | inline prompt 。 通常写到独立的 `instruction.md` 而不是 inline 。 |
| `context_files` | `[]` | 从 `data/` （旧名 `files/`） 上传给 agent 的文件列表 。 |
| `bootstrap_script` | 无 | MCP 加载前在主容器跑一次的脚本 ，通常放 `assets/start.sh` 。 |
| `runtime.main.image` | 内置默认 | 覆盖 main runtime 镜像 ；engine 按 image 前缀选 MCP_PY / SHELL_PATH 。 |
| `pending_upstream` | 无 | 声明该 case 依赖上游 mock / image 未发布 ；动态测试 skip ，orchestrator 默认跳过 。 |
| `extra_mcp_servers` | `[]` | 追加 MCP server 声明 ；扁平（`name+command`） 或嵌套（`{client:{...}}`） 均可 。 |
| `services` | `[]` | 辅助服务容器 ；每个是独立 nexus runtime 。 |
| `agent_tools.categories` | `[]` | 声明该 case 使用的 agent-tool 类别（须在 `ALLOWED_AGENT_TOOLS_BY_CATEGORY` 里） 。 |

### services 字段

每个 service 结构见 `nexus/extensions/tasks/aidata/engine/models.py::AidataService` 。 关键字段 ：

- `name` 、`runtime_name`（同 case 唯一 、不能叫 `main` ） 、`image` 、`exposed_ports` 。
- `startup.command` 或 `startup.script` （二选一） 、`startup.env` 、`startup.timeout_s` 。
- `ready_check.command` 、`ready_check.timeout_s` 、`ready_timeout_s` 。
- `connections[]` —— 每个含 `placeholder` 、`internal_port` 、`value_template` 、可选 `env_name` 。

镜像优先级 ：`runtime_provider_config.image` 故意不设 —— 每个 runtime 从 `runtime_configs.<name>.image`
（main） 或 `services[].image` （service） 拿 。 详见 `docs/mcp-contract.md` 。

## `instruction.md`

自然语言 prompt ，agent 可见 。 markdown 格式 、随意长度 。 CLI 会把它作为 `user_prompt` 注入 —— 若
`task.yaml::user_prompt` 也存在 ，`instruction.md` 优先 。

## `rubric.yaml` （v4.2 ）

顶层三 key ：`version` / `config` / `checks` 。 完整字段字典见 [`../../RUBRIC.md`](../../RUBRIC.md) 。

关键 assertion type 单一真相源 ：`nexus/extensions/tasks/aidata/case/schema.py::V42_ASSERT_TYPES` 。

## `run-test.sh` + `report.junit.xml` （test_result 用 ）

若 rubric 里出现 `assert.type: test_result` ，engine 会在 agent 结束后 ：

1. 从 case-root 找 `run-test.sh` ；找不到再找 `tests/run-test.sh` 。
2. 用 shell 执行 ；工作目录被 pin 到 `TESTS_DIR_IN` 。
3. 期望脚本产出 `TESTS_DIR_IN/report.junit.xml` 。
4. 用 `nexus/extensions/tasks/aidata/rubric/junit.py::parse_junit_xml` 解析 。
5. 通过 `check_map.yaml`（可选） 建立 `check.id → junit testcase name` 映射 。

junit XML 结构与常见字段 ：见 [`../../RUBRIC.md`](../../RUBRIC.md) §6 。

## `result.json` （task 输出 ）

落到 `workspace/mcp/tmp/aidata_<case>_<ts>.result.json` ；orchestrator 里为
`<out>/runs/<run_key>/result.json` 。

顶层结构 ：

```json
{
  "task_input":  { ... },
  "task_output": {
    "score": 0.85,
    "rubric": { ... },
    "rubric_results": [
      {"id": "...", "criterion": "...", "passed": true, "verdict": "PASS", "detail": "..."}
    ],
    "state_summary": { ... },
    "error_message": null
  },
  "agent_output": {
    "main_trajectory": [
      {"role": "system", "content": "..."},
      {"role": "user",   "content": "..."},
      {"role": "assistant", "tool_calls": [...]},
      {"role": "tool",   "content": "..."}
    ]
  }
}
```

### `task_output` 字段字典

| 字段 | 类型 | 含义 |
|---|---|---|
| `score` | float | 最终分数。 默认是加权分经 hard-gate + ScorePolicy 后的结果；`config.reward.script` 模式为脚本分经 hard gate 封顶后的结果。 |
| `continuous_score` | float | 默认是 ScorePolicy 前连续分；reward 脚本模式下等于脚本分经 hard gate 封顶后的值。 |
| `score_policy` | dict | 已解析 ScorePolicy，或 `{mode: script, script, timeout_sec}`。  |
| `rubric` | dict | 原样回显加载后的 rubric（含默认值填充） 。 |
| `rubric_results[]` | list | 每条 check 的 verdict + detail 。 每项字段见下 。 |
| `state_summary` | dict | mock state 的精简快照 。 每 mock 一 namespace 、并保留 legacy 扁平 key 。 |
| `error_message` | str \| null | 若非 null ，哪一步失败 。 分类见 architecture.md 。 |

`rubric_results[]` 每项字段 ：`id` / `criterion` / `passed` / `verdict` / `judge_raw_head` /
`source` / `claim` / `detail` / `dimension` / `weight` / `assert_type` 。 完整语义见
[`../how-to/author-a-rubric.md`](../how-to/author-a-rubric.md) 与 [`../../RUBRIC.md`](../../RUBRIC.md) 。
`state_summary` legacy 扁平 source key 详见 [`../../README.md`](../../README.md) §Rubric source key 。

### `agent_output` 字段字典

| 字段 | 类型 | 含义 |
|---|---|---|
| `main_trajectory[]` | list | agent 的完整消息日志 。 rubric 的 trajectory-based primitive 从这里读证据 。 |

## `check_map.yaml` （可选 ；test_result 用 ）

映射 `check.id → junit testcase name` 。 缺失时 engine 用 `check.id` 直接匹配 testcase name 。

```yaml
checks:
  email_body_ok: test_email_body_contains_recipient
  status_paid: TestOrders.test_paid_after_email
```

## `pyproject.toml` （模块级 ）

aidata 的 task extension 声明 ，包含依赖 、entry point registrations 。 nexus module discovery 从这里读 。
主仓 `nexus-agent` **不会** transitive 装它的依赖 —— 跑测试前需要 `uv pip install pytest-xdist>=3.6` 。

## 相关文档

- 各 assert type 语义 —— [`../how-to/author-a-rubric.md`](../how-to/author-a-rubric.md) 。
- rubric 完整字段字典 —— [`../../RUBRIC.md`](../../RUBRIC.md) 。
- 12 步 orchestration —— [`../explanation/architecture.md`](../explanation/architecture.md) 。
