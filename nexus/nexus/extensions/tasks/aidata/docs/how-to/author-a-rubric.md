---
title: rubric.yaml v4.2 assert 原语速查
audience: 需要写或改 rubric 的开发者与标注员
prerequisites:
  - 完成 [tutorial/first-case.md](../tutorial/first-case.md)
  - 已经建好 case 骨架（见 [author-a-case.md](./author-a-case.md) ）
related:
  - ../../RUBRIC.md
  - ../../RUBRIC_AUTHORING.md
  - ../explanation/rubric-scoring-model.md
---

# rubric.yaml v4.2 assert 原语速查

本篇是 v4.2 assert 原语的一页速查 —— 每个原语一段 ，说清楚 " 用于什么 " 与 " 最小示例 " 。 完整字段
字典 、边界情况 、legacy 兼容规则请看 [`../../RUBRIC.md`](../../RUBRIC.md) 。

新 case 一律用 v4.2 。 v2 rubric（`verify: llm_judge/eq/regex/shell` ） 仍能跑 ，但 validate 会拒绝新写入
（细节见 [explanation/v2-freeze.md](../explanation/v2-freeze.md) ） 。

## 顶层结构

```yaml
version: "4.2"
config:                             # 可选 ，控制轨迹上限与最终得分
  trajectory:
    user_head_chars: 16000
    tool_call_arg_chars: 800
    tool_result_chars: 2000
  judge:
    source_max_chars: 96000
checks:
  - id: snake_case_unique_id
    description: 一句话说明这条 check 证明什么
    dimension: correctness | functional | robustness
    weight: 1
    assert:
      type: <one of the primitives below>
      # 各原语对应字段
```

- `dimension` 三选一 ：`correctness` / `functional` / `robustness` 。 完整含义见
  [explanation/rubric-scoring-model.md](../explanation/rubric-scoring-model.md) 。
- `weight` 没有特殊要求时统一写 `1.0`；只有任务确实需要区分各 check 的重要性时，才使用其它自定义权重。
- 总分 `= Σ weight × passed / Σ weight` 。 hard-gate 参见
  [explanation/rubric-scoring-model.md](../explanation/rubric-scoring-model.md) 。
- 如果静态权重表达不了跨 check 条件 ，可以用本页的“最终 reward 脚本”覆盖总分聚合。

## v4.2 assert 原语（`V42_ASSERT_TYPES` ）

代码单一真相源 ：`nexus/extensions/tasks/aidata/case/schema.py::V42_ASSERT_TYPES` 。

| Type | 用途 | 数据源 |
|---|---|---|
| `state_equals` | 结构化断言 mock 收到的一段 state 精确等于某个值 | `state` （dump_state 之后 ） |
| `text_contains` | 拍平后的文本里包含 / 不包含子串或正则 | trajectory / agent_answer / all |
| `text_not_contains` | 反 `text_contains` | 同上 |
| `tool_called` | agent 调用过某工具（可选参数匹配 ） | trajectory 里的 tool_call 序列 |
| `tool_called_sequence` | tool_call 有序子序列 | 同上 |
| `judge_claim` | LLM 裁判判断 " agent 是否满足自然语言 claim" | source（可 pin 到某段） |
| `test_result` | 从 `report.junit.xml` 里读 testcase 通过 / 失败 | pytest / go-test 等 |
| `script_probe` | 运行结束后跑一段脚本 ，看 stdout / exit_code | 主容器或 sidecar |

### `state_equals`

用于结构化 assert —— mock 收到的邮件字段 、订单状态 、日历项 。 优先选它 ，稳定且 debug 友好 。

```yaml
assert:
  type: state_equals
  from: xhs                     # 命名空间（推荐 ） 或者旧的扁平 key
  path: actions.0.type
  equals: publish_note
```

支持 jmespath 与 SQL —— 见 `RUBRIC.md` §2.1 。 空 `from` 时会读 legacy 扁平 source key 。

### `text_contains` / `text_not_contains`

文本子串 / 正则 。 用于精确关键字或者段落存在性 —— 例如 agent 的回答里必须提到 " 请假流程" 。

```yaml
assert:
  type: text_contains
  from: agent_answer
  contains_all: ["请假", "审批"]
```

字段 ：`contains` / `contains_all` / `contains_any` / `regex` （或 legacy `pattern` ） 。

### `tool_called` / `tool_called_sequence`

判定 agent 调用过某工具 、以及可选的 arguments 匹配 。 用于验证关键动作 。

```yaml
assert:
  type: tool_called
  scope: live                   # 强烈建议 ；避免 initial_trajectory seed 计入
  tool: email_send
  arguments:
    contains_all_keys: [to, subject]
```

`tool_called_sequence` 复用 step 语法 ，验证一个有序子序列（例如 " 先 login 再 create_order 最后
pay" ） 。

### `judge_claim`

LLM 裁判 。 用于文字质量或复合判断 —— 例如 " agent 是否用礼貌语气拒绝了越权请求" 。 慎用 、贵 。

```yaml
assert:
  type: judge_claim
  from: agent_answer
  claim: "agent 是否明确说明拒绝的原因 ，且未泄露内部工具名"
```

判断规则详见 [`../../RUBRIC.md`](../../RUBRIC.md) §2.5 。 一个 case 里 judge_claim 数量宜控制 —— 每个都是
一次 LLM 网关往返 ；`NEXUS_JUDGE_MAX_RETRIES` 会在 UNAVAILABLE 时重试 。

### `test_result`

把 pytest / go test / jest 的 pass / fail 映射成 rubric 判定 。 需要 case 里有 `run-test.sh` +
`tests/report.junit.xml` 生成逻辑 。 详见 [`../../RUBRIC.md`](../../RUBRIC.md) §6 。

```yaml
assert:
  type: test_result
  test_name: test_email_body_contains_recipient
```

### `script_probe`

运行结束后跑一段脚本 ，看 stdout / exit_code 。 用于 " state 结构不方便结构化断言 ，但一段脚本可以
一句话搞定 " 的场景 —— 例如查 SQL 、curl 一个 sidecar HTTP 端点 。

```yaml
assert:
  type: script_probe
  script: |
    curl -sf http://__AIDATA_JD_BACKEND_URL__/api/orders/123 | jq -r .status
  service: main                 # 或者某个 sidecar 名字
  timeout_sec: 15
  expect_exit: 0
  regex: '^paid\s*$'
```

`script_probe` 只会根据退出码和 stdout 的 `contains` / `regex` 产出通过或失败；不会读取
`capture` / `equals`，也不会把 stdout 中的数值直接作为该 check 的连续分数。

signature 见 `nexus/extensions/tasks/aidata/rubric/fetch.py::ProbeSpec` 。

## 最终 reward 脚本

需要“30 条里至少过 20 条得 1”或“A 必须通过且总通过数大于 20 才得 1”这类跨 check 规则时，
在 rubric 顶层配置 case 随包脚本：

```yaml
config:
  reward:
    script: reward.py            # 相对 case 根目录；只接受 .py / .sh
    timeout_sec: 15              # 可选，默认 15
```

`reward.py` 从 stdin 读取版本化 JSON，stdout 必须只输出 `[0, 1]` 内的 JSON number，或
`{"score": number}`：

```python
import json
import sys

payload = json.load(sys.stdin)
rows = {row['id']: row for row in payload['rubric_results']}
passed = payload['score_summary']['passed']

# 示例：A 必须通过，并且总通过数严格大于 20。
score = 1.0 if rows['A']['passed'] and passed > 20 else 0.0
print(json.dumps({'score': score}))
```

stdin 字段固定为：

- `version: "1.0"`；
- `rubric_results`：所有 check 的完整结果（含 `id`、`passed`、`verdict`、`weight` 等）；
- `score_summary`：静态加权摘要（含 `passed`、`total`、`raw_score`、hard-gate 状态等）；
- `state_summary`：最终 state 的紧凑计数摘要。

脚本输出会取代 ScorePolicy 成为最终 `score`，所以 `config.reward` 与 `config.score` 不能同时使用。 hard
gate 仍在脚本输出上封顶，脚本不能绕过。 脚本超时、非零退出、输出不可解析或越界时，该样本会
标为 `REWARD_ERROR / reward_script_error`，不会静默退回静态加权分。 脚本文件位于 agent 不可见的
case 资产区；`./aidata.py validate <case_id>` 会检查路径、扩展名、超时和文件是否存在。

## config 段（可选）

只有当某条 check 用 `from: trajectory` / `from: all` 拍平轨迹喂给 judge / 做 substring 时 ，这些
上限才起作用 —— 大多数 case 不需要写 。 默认值 ：

| 字段 | 默认 |
|---|---|
| `trajectory.user_head_chars` | 16000 |
| `trajectory.tool_call_arg_chars` | 800 |
| `trajectory.tool_result_chars` | 2000 |
| `judge.source_max_chars` | 96000 |
| `reward.script` | 无；配置后启用最终 reward 脚本 |
| `reward.timeout_sec` | 15 |

外部 override 优先级 ：`run_rubric_async(limits=...)` kwarg > rubric config > 默认值 。 见
`nexus/extensions/tasks/aidata/rubric/config.py::RubricLimits` 。

## 提交前自查

- 每条 check 的 `id` 稳定 、唯一 、snake_case 。
- 每条 check 的 `description` 说清楚 " 这条证明什么 " —— reviewer 能理解 。
- 优先 `state_equals` / `test_result` / `tool_called` ，最后才 `judge_claim` 。
- `tool_called` 都写 `scope: live` ，避免 seed 历史里的调用被计分 。
- config 段没必要就不写 ；默认值已经调好 。

## 深入

- 完整字段字典 —— [`../../RUBRIC.md`](../../RUBRIC.md) 。
- 编写风格 、推荐组合 、legacy 兼容 —— [`../../RUBRIC_AUTHORING.md`](../../RUBRIC_AUTHORING.md) 。
- 打分是怎么算出来的 —— [explanation/rubric-scoring-model.md](../explanation/rubric-scoring-model.md) 。
