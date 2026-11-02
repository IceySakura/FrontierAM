# rubric.yaml 编写完全指南（字段 + 枚举值字典）

## 1. 文件整体结构（顶层三个 key）

```yaml
version: "4.2"     # 必填，固定 "4.2"。见 §2
config: {...}      # 可选。截断阈值，影响 judge_claim / text_contains。见 §3
checks:            # 必填，且必须非空。每条 check 是一个 mapping。见 §4
  - id: ...
    description: ...
    dimension: ...
    weight: ...
    assert: {...}
  - ...
```

| 顶层字段 | 必填 | 类型 | 含义 |
|---|---|---|---|
| `version` | 是 | string | rubric schema 版本，固定 `"4.2"`。 |
| `config` | 否 | mapping | 控制 trajectory 上限、hard gate、ScorePolicy 或最终 reward 脚本。 |
| `checks` | **是** | list | 检查项列表；空列表会被加载层判为非法。 |

---

## 2. 顶层字段：`version`

字符串，固定写 `"4.2"`。它决定引擎走 v4.2 评分路径（加权连续总分 + 维度分）。

```yaml
version: "4.2"
```

> 若不写 `version`，但任意一条 check 里出现了 `assert:` 键，引擎也会按 `"4.2"`
> 处理（`rubric/dispatch.py::_rubric_is_v42`；`engine/rubric.py` 现为 shim，重新
> 导出该函数）。但请显式写，别依赖推断。

---

## 3. 顶层字段：`config`（可选，截断阈值）

只有当某条 check 用 `from: trajectory` 或 `from: all` 把"整条轨迹拍平成一大段文本"
喂给 LLM 裁判 / 做子串匹配时，这些上限才起作用。绝大多数 case 不用写 `config`。

```yaml
config:
  trajectory:
    user_head_chars:     16000   # 首条 user 消息保留的头部字符数
    tool_call_arg_chars: 800     # 每条 assistant.tool_call 的 arguments 字符上限
    tool_result_chars:   2000    # 每条 tool_result body 字符上限
  judge:
    source_max_chars:    96000   # 拼给 judge 的 SOURCE 总字符上限
```

**字段字典：**

| 字段 | 默认值 | 含义 |
|---|---|---|
| `trajectory.user_head_chars` | `16000` | 首条 user 消息（含任务描述 + 内联的 context）保留多少字符。超出截断并标注剩余数。 |
| `trajectory.tool_call_arg_chars` | `800` | 每个工具调用的 `arguments` JSON 保留多少字符。 |
| `trajectory.tool_result_chars` | `2000` | 每个工具返回 body 保留多少字符。 |
| `judge.source_max_chars` | `96000` | 拍平后拼给 LLM 裁判的 SOURCE 整体上限。 |
| `hard_gate_cap` | `0.0` | hard gate（见 §4.6）失败时整条任务分的封顶值。默认清零；设为如 `0.84` 可软化为"封到成功阈值以下"。非法值回退 `0.0`。 |
| `reward.script` | 无 | case 根目录内的 `.py` / `.sh` 相对路径；配置后由脚本计算最终 reward。 |
| `reward.timeout_sec` | `15` | reward 脚本执行超时秒数。 |

**规则：**
- 全部可选；不写 `config` 或不写某字段 → 用默认值。
- 非正整数 / 非数字会被忽略并 log 警告，回退默认值（不会打断评分）。
- 何时需要放开：工具返回了 2k+ 的 JSON、任务 prompt 超 32k、需要看完整 assistant 输出时。
- `config.reward` 与 `config.score` 互斥；脚本模式不再经过 ScorePolicy。

**作用范围**（哪些原语受 `config` 影响）：

| 原语 | 是否受 `config` 影响 |
|---|---|
| `judge_claim` | 受影响（拍平 trajectory + SOURCE 上限都生效） |
| `text_contains` | 受影响（拍平 trajectory） |
| `tool_called` / `tool_called_sequence` | **不受影响**（直接遍历原始 trajectory 结构） |
| `state_equals` | **不受影响**（走 dotted path / jmespath / sql） |
| `test_result` | **不受影响** |

### 3.1 最终 reward 脚本

需要跨 check 聚合时可配置：

```yaml
config:
  reward:
    script: reward.py
    timeout_sec: 15
```

脚本从 stdin 读取 `version: "1.0"`、`rubric_results`、`score_summary` 和 `state_summary`，
stdout 必须只输出 `[0, 1]` 内的 JSON number 或 `{"score": number}`。 例如至少通过 20 条得 1：

```python
import json
import sys

payload = json.load(sys.stdin)
print(1 if payload['score_summary']['passed'] >= 20 else 0)
```

脚本结果取代 ScorePolicy；hard gate 仍会封顶。 超时、非零退出、非法/越界输出会把样本标记为
`REWARD_ERROR / reward_script_error`，不会回退到静态加权分。 完整协议见
[`docs/how-to/author-a-rubric.md`](docs/how-to/author-a-rubric.md)。

---

## 4. `checks[]` 的通用字段

每条 check 是一个 mapping，固定 5 个字段：

```yaml
- id: <snake_case_unique>     # 必填
  description: <一行人话>      # 必填
  dimension: correctness      # 必填，3 选 1
  weight: 1                   # 必填，> 0
  assert:                     # 必填，里面的 type 决定怎么判
    type: <原语>
    ...原语自己的字段...
```

评分规则（默认：加权连续总分）：

```
总分       = Σ (weight_i × passed_i) / Σ weight_i     （连续，∈ [0, 1]）
维度分_d   = 上式在 dimension==d 子集上的加权通过率

其中每条 check 的 passed_i ∈ {0, 1}（是否通过），weight_i 直接影响权重。
```

`passed` 单条是二值（Y=1 / 其它=0），但整体总分是**加权连续 float**。 权重
更大 = 该 check 对总分贡献更大。 环境变量可切到二元/阈值模式：

- `AIDATA_SCORE_MODE=binary` + `AIDATA_SCORE_THRESHOLD=<float, 默认 1.0>` → 总分坍缩为
  `1.0 if 连续分 >= 阈值 else 0.0`。
- 也可通过 rubric 顶层 `config.score.{threshold, continuous, use_threshold}` 覆盖。
- 具体开关与优先级见 `rubric/policy.py::ScorePolicy.resolve` 与
  `rubric/dispatch.py::compute_scores`；执行细节见
  [`docs/reference/rubric-primitives.md`](docs/reference/rubric-primitives.md) §6。

### 4.1 `id`

- 类型：string，必填，**全 rubric 内唯一**。
- 约定：`snake_case`，稳定（一旦上线尽量别改，结果对账靠它）。
- 用途：结果 JSON 里每条结果靠它定位；`test_result` 原语还会用它推导 junit testcase 名（见 §6.6）。
- 校验：为空 / 重复都会被报错。

### 4.2 `description`

- 类型：string，必填。
- 一行人类可读说明，写清楚 **"这条 check 证明了什么"**，而不是复述 assert 配置。
- 好例子：`给候选人 zhangsan_ai_2026 发了微信好友请求`。
- 会原样回显到结果 JSON 的 `criterion` 字段，方便 reviewer 看。

### 4.3 `dimension`（枚举：3 个值）

- 类型：string，必填。把这条 check 归到一个评估维度，用于算 **维度分**。
- 校验：必须是下面三个之一，否则报错。

| 枚举值 | 含义 | 典型用法 |
|---|---|---|
| `correctness` | **正确性**：agent 有没有做对关键动作 + 事实是否正确。 | 下了正确的单、发了正确的消息、日期算对了、调用了必需的工具。 |
| `functional` | **功能性 / 产物质量**：产出的东西内容上对不对、全不全、结构好不好。 | 文档覆盖了要求的维度、邮件包含结论、memo 每条都有评分。 |
| `robustness` | **鲁棒性 / 边界**：异步轮询、重试、错误处理、"不该做的没做"。 | 没给已结束的会开 ASR、没越界改无关文件、没漏掉轮询。 |

### 4.4 `weight`

- 类型：数字（int / float），必填，**必须 > 0**。
- **默认统一写 `1.0`**；只有任务确实需要区分各 check 的重要性时，才使用其它自定义权重。
- **权重直接影响总分**：默认加权连续评分下，`weight` 越大，该 check 对总分贡献越大。
  绝大多数 case 写 `1` 就够（等权）；重要项想加大权重时可以设 `2` / `3`；
  想让某项"锦上添花但不该主导"时可以设 `0.5`。 二元 / 阈值模式下同样按 weighted
  加权算连续分再坍缩，权重仍生效。
- 缺失或 ≤ 0 会被报错。

### 4.5 `assert`

- 类型：mapping，必填。**这是"怎么判"的核心**。
- 必须包含 `type` 字段（值见 §5），其余字段由 `type` 决定（见 §6）。

### 4.6 `hard_gate`（可选，硬门控）

- 类型：bool，可选，默认 `false`。
- 语义：标记本 check 为 **hard gate（硬门控）**。只要任意一条 hard gate **未通过**，
  整条任务分会被 **封顶**（`gate_safety` 乘子），默认封顶到 `0.0`（清零）。封顶只会拉低分数、
  绝不会抬高；soft cap 通过 `config.hard_gate_cap` 配置（见 §3）。
- **机器可读才生效**：必须用本字段（或顶层 `hard_gates: [id, ...]` 数组）声明。
  历史上的 `# [HARD GATE]` **注释约定不再生效**——YAML 注释在加载时被丢弃，runtime 读不到。
- 结果 JSON：每条结果回显 `hard_gate: true/false`；task 级 summary 含 `hard_gate_triggered`
  与 `hard_gate_failed: [id, ...]`，并在触发时打 `logger.warning`。
- 两种等价声明方式：

```yaml
# 方式 A：在 check 上直接标
- id: state_payment_amount_correct
  dimension: correctness
  weight: 0.25
  hard_gate: true
  assert: { type: state_equals, ... }

# 方式 B：在顶层集中列出
hard_gates: [state_payment_amount_correct, state_payment_bills_correct]
```

---

## 5. `assert.type` 总览（枚举：6 个）

`assert.type` 是一个 **闭合枚举**，只放行下面 6 个。

| `type` 值 | 一句话 | 判定来源 | 是否调 LLM | 详见 |
|---|---|---|---|---|
| `state_equals` | 结构化等值 / JMESPath 查询 / SQL 标量探针 | 最终 mock 状态 | 否 | §6.1 |
| `text_contains` | 某段文本含/匹配关键词或正则 | 文本切片 | 否 | §6.2 |
| `tool_called` | agent 真的调了某工具（可带参数约束） | trajectory | 否 | §6.3 |
| `tool_called_sequence` | 工具调用满足某有序子序列 | trajectory | 否 | §6.4 |
| `judge_claim` | 自然语言断言交给 LLM 裁判 | 文本切片 | **是** | §6.5 |
| `test_result` | 跑 `run-test.sh`，按 junit 结果判 | junit 报告 | 否 | §6.6 |

---

## 6. 各 `assert.type` 字段详解

### 6.1 `state_equals`（结构化等值 / JMESPath / SQL）

最强、最推荐的"验证 agent 真的改了状态"的原语。它有 **3 种形态**，按下面顺序 dispatch：

> **dispatch 优先级**：`jmespath` 键 → `from: sql:` 前缀 → `from` + `path`（slice + 子路径）。

> ⚠️ **关键区别（最常见的坑）**：`state_equals` 的 `from` 是 **state 的顶层 key（命名空间）**，
> 引擎对它做 `state.get(from)`，**不会**把 `from` 当 dotted path 解析。
> 想往下钻路径，用 `path:`（dotted）或 `jmespath:`。
> 这点和 `text_contains` / `judge_claim` 的 `from`（支持 `namespace.key.path` 点号解析）
> **不一样**。

#### 形态 1：`jmespath` 分支（推荐用于"写操作真发生"）

用 [JMESPath](https://jmespath.org/) 查询打到具体记录上，再用 **恰好一个谓词** 判通过。

```yaml
assert:
  type: state_equals
  from: life_payment                      # 顶层命名空间；省略 = 对整个 state 查
  jmespath: "life_payment_orders[?status=='paid']"
  count_op: ">="                          # 谓词（下面表里 12 选 1）
  count_value: 1
```

- `from`：可选。写 → 对 `state[from]` 这个 slice 查；不写 → 对整个 `state` dict 查
  （此时可直接查 `agent_answer` / `trajectory` / 任意命名空间）。
- `jmespath`：**必填**，查询表达式。内置 `length` / `sum` / `min` / `max` / `avg` /
  `keys` / `values` 等聚合函数。语法见 https://jmespath.org/specification.html 。

**谓词枚举（必须恰好指定一个；0 个或 >1 个都会判 INVALID）：**

| 谓词键 | 值类型 | 输入 | 通过条件 |
|---|---|---|---|
| `exists` | bool | 任意 | 结果"truthy"（非 `None` / 非空 list/dict/str / 非 `0`）== 你给的 bool。 |
| `equals` | 标量 | 标量 | 结果 == 值（数字会做 int/float 强转再比）。 |
| `one_of` | list | 标量 | 结果 ∈ 列表（带数字强转）。 |
| `count_op` + `count_value` | str + int | list/dict/str | `len(结果) <op> count_value`；**两者必须成对出现**。 |
| `contains` | 任意 | list/dict/str | 元素 ∈ list，或 子串 ∈ str，或 key ∈ dict。 |
| `contains_all` | list | list/dict/str | 列表里每个元素都在结果里。 |
| `contains_any` | list | list/dict/str | 列表里至少一个元素在结果里。 |
| `regex_match` | str | 任意 | `re.search(pattern, str(结果))` 命中。 |
| `gt` / `gte` / `lt` / `lte` | number | number | 数值比较 `> / >= / < / <=`（双侧 float 强转）。 |

**`count_op` 的运算符枚举**（配 `count_value` 用）：`==` `!=` `>` `>=` `<` `<=`。

实战示例：

```yaml
# 计数：恰好新增 3 条记录
- id: added_three_entries
  description: 目标备忘录恰好新增了 3 条记录
  dimension: correctness
  weight: 1
  assert:
    type: state_equals
    from: notes
    jmespath: "(memos.* | [?title=='2026-04-21 复盘'])[].entries[]"
    count_op: "=="
    count_value: 3

# 聚合：已付订单总额 ≥ 100
- id: paid_total_over_100
  description: 所有已付订单合计金额不低于 100
  dimension: functional
  weight: 1
  assert:
    type: state_equals
    from: life_payment
    jmespath: "sum(life_payment_orders[?status=='paid'].total_amount)"
    gte: 100

# 反向校验（robustness）：原有备忘录没被污染，命中数必须为 0
- id: did_not_modify_existing
  description: 原有备忘录没有混入本次新增的三条
  dimension: robustness
  weight: 1
  assert:
    type: state_equals
    from: notes
    jmespath: "memos.memo_existing.entries[] | [?contains(data.\"事项\", '检车费报销')]"
    count_op: "=="
    count_value: 0
```

#### 形态 2：`sql:` 标量探针

当 mock 把权威状态落在 sqlite 里时，用一条 **返回单个标量单元格** 的只读查询，再用
`equals` / `one_of` 比较。

```yaml
assert:
  type: state_equals
  from: |
    sql:SELECT CASE WHEN EXISTS(
      SELECT 1 FROM order_items WHERE item_id='784316110808' AND refunded=0
    ) THEN 1 ELSE 0 END
  equals: 1
```

- `from: sql:<单标量查询>`：必填。SQL 必须返回 1 行 1 列。
- 比较谓词：`equals` 或 `one_of`（二选一，必须有一个）。
- **前置条件**：case 的 `task.yaml` 必须声明 `rubric_sqlite_path:`（指向 sqlite 文件路径，
  支持 `__CASE_UUID__` 占位符）。跨容器（sidecar）场景用 `service:<name>:<path>` 前缀，
  让探针跑在对应 service 容器里。没声明路径时该 check 会以 `MISSING` 失败
  （fail-closed，不静默 0 分）。

#### 形态 3：`from`（slice）+ `path`（dotted 子路径）

```yaml
assert:
  type: state_equals
  from: xhs                 # 顶层命名空间
  path: actions.0.type      # 进入 state['xhs'] 后的 dotted 路径；数字段表示 list 下标
  equals: publish_note      # 或 one_of: [...]
```

- `from`：顶层 slice key。
- `path`：进入该 slice 的 dotted 路径（`a.b.0.c`，整数段索引 list）。
  省略 `path` 时直接拿整个 slice 比较。
- 谓词：`equals` / `one_of` 二选一。

**`state_equals` 字段速查：**

| 字段 | 形态 | 含义 |
|---|---|---|
| `from` | 1/2/3 | slice 命名空间；形态 2 时写 `sql:...`；形态 1 省略=整个 state。 |
| `jmespath` | 1 | JMESPath 查询（出现它就走形态 1）。 |
| `path` | 3 | 进入 `from` slice 后的 dotted 路径。 |
| `equals` / `one_of` | 1/2/3 | 比较谓词（形态 1 的全部谓词见上表）。 |

---

### 6.2 `text_contains`（关键字 / 正则）

对一段文本切片做子串 / 正则匹配。

```yaml
assert:
  type: text_contains
  from: agent_answer                 # 文本选择器，见 §7；默认 all
  contains_all: ["工单号", "已报修"]   # 下列谓词至少要有一个
  case_sensitive: false              # 默认 false，只影响 contains* 家族
```

**谓词字段（可同时出现，按 AND 叠加；至少要有一个）：**

| 字段 | 类型 | 含义 |
|---|---|---|
| `contains` | str 或 list | 单串或一组串，**全部**都要出现。 |
| `contains_all` | list | 一组串，**全部**都要出现（等价于 list 形态的 `contains`）。 |
| `contains_any` | list | 一组串，**至少一个**出现即可。 |
| `regex` | str | Python 正则，`re.search` 命中即可。走 **原文**（不转小写），要忽略大小写在 pattern 里写 `(?i)`。 |
| `case_sensitive` | bool | 默认 `false`。为 false 时 `contains*` 家族把文本和关键词都转小写；对 `regex` 无效。 |

示例（多谓词 AND 叠加）：

```yaml
- id: answer_confirms_followup_recording
  description: 最终答复确认已为 M_FOLLOWUP_001 开启录制/语音转写
  dimension: functional
  weight: 1
  assert:
    type: text_contains
    from: agent_answer
    contains_all: ["M_FOLLOWUP_001"]
    regex: "(?s)(M_FOLLOWUP_001.*(开启|启用).*(录制|ASR))|((开启|启用).*(录制|ASR).*M_FOLLOWUP_001)"
```

> 验证"写操作发生"请用 `state_equals`，**不要**用 `text_contains` + 工具名正则去 trajectory 里
> 搜——agent 在思考文本里念一句工具名就能骗过。

---

### 6.3 `tool_called`（工具被调用）

证明 agent 真的调用了某工具，可附加细粒度参数约束。`tool_called` 是最常用的原语。

最简形态：

```yaml
assert:
  type: tool_called
  tool_prefix: wechat_wechat_   # 或 tool: 精确名（二选一，必填其一）
  with_text: "zhangsan_ai_2026" # 简写：对整个 arguments JSON 做大小写不敏感子串
  min_count: 1                  # 可选，默认 1
  scope: live                   # 可选，见下方 scope 枚举
```

**工具身份字段（`tool` / `tool_prefix` 二选一，必填其一）：**

| 字段 | 含义 |
|---|---|
| `tool` | 精确工具名。可写裸业务名（`email_send`）也可写全限定名（`email_email_send`）——引擎按 `_<裸名>` 后缀容错。**推荐写裸名。** |
| `tool_prefix` | 对 trajectory 里的工具名做原始 `startswith` 前缀匹配（可用于匹配整个 MCP 命名空间，如 `wechat_wechat_`）。 |

**`with_text` 两种写法：**

1. **字符串形态（简写）**：对整个 `arguments` 字符串做大小写不敏感子串匹配。
2. **字典形态（细粒度）**：

```yaml
with_text:
  field: arguments.couponIds              # dotted 路径，见下方 field 根枚举；默认 arguments.keyword
  contains: "920568427905"                # 子串（大小写不敏感）
  contains_all: ["GLOBAL_SAVE200", "X"]   # 每一项都要在
  contains_any: ["iPhone", "华为"]         # 任一命中即可
  regex: "^ORD-[0-9A-F]{10}$"             # 正则命中（走原文）
```

`with_text.field` 的 **根枚举**（dotted 路径第一段）：

| 根 | 指向 |
|---|---|
| `arguments` | 工具调用参数（如 `arguments.itemId`、`arguments.command`）。 |
| `tool_name` | 工具名。 |
| `name` | 工具名（同 `tool_name`）。 |

**其它参数约束字段：**

| 字段 | 含义 |
|---|---|
| `arg_empty` | list，列出的参数字段必须缺失 / 空 / 空串。 |
| `arg_not_in` | mapping `{字段: [禁用值...]}`，该字段的值不能等于任一禁用值。 |
| `all_of` | list，布尔 AND：每个子项是一个完整 step-spec，递归判定。 |
| `any_of` | list，布尔 OR。 |
| `not` | 单个 step-spec，布尔 NOT。 |
| `matcher` | str，注册在 `MATCHER_REGISTRY` 的 python 钩子名（默认空，极少用，escape hatch）。 |
| `min_count` | int，默认 1。匹配次数 ≥ 它才通过。 |
| `scope` | 见下方枚举。 |

**`scope` 枚举（可选）：**

| 值 | 含义 |
|---|---|
| `live` | 只看本次运行新产生的轨迹。**推荐默认写这个**，避免把 `initial_trajectory` 里 seed 的历史工具调用算进来。 |
| `seed` | 只看 `task.yaml::initial_trajectory` 注入的历史轨迹。 |
| （不写） | 看完整 trajectory（seed + live）。 |

示例（精确名 + 参数 + 禁用值）：

```yaml
- id: enable_asr_on_followup_only
  description: 只给跟进会 M_FOLLOWUP_001 开 ASR
  dimension: correctness
  weight: 1
  assert:
    type: tool_called
    tool: tm_enable_asr
    with_text:
      field: arguments.meeting_id
      contains: M_FOLLOWUP_001
    arg_not_in:
      meeting_id: ["", "TBD", "M_TARGET_001", "M_CANCELLED_001"]
```

---

### 6.4 `tool_called_sequence`（工具调用有序子序列）

证明工具调用满足某个 **有序** 子序列（SOP 流程类约束）。

```yaml
assert:
  type: tool_called_sequence
  scope: live           # 可选，同 tool_called 的 scope 枚举
  steps:                # 必填，非空 list；每个 step 字段与 tool_called 完全一致
    - tool: tm_join_meeting
      with_text: {field: arguments.meeting_id, contains: M_FOLLOWUP_001}
    - tool: tm_enable_asr
      with_text: {field: arguments.meeting_id, contains: M_FOLLOWUP_001}
```

- `steps`：**必填**，非空 list。每个 step 的字段集合 **与 `tool_called` 顶层完全相同**
  （`tool` / `tool_prefix` / `with_text` / `arg_empty` / `arg_not_in` / `all_of` / `any_of` /
  `not` / `matcher`）。每个 step 至少要有 `tool` 或 `tool_prefix`。
- 语义：从左到右带游标匹配——每个 step 找当前游标之后第一个命中的调用，命中后游标前移。
- 评分：**二元**（全部 step 按序命中才通过，不给部分分）。结果 `detail` 里有
  `matched N/M (indices=[...])` 方便排查，`-1` 表示该 step 没匹配上。

---

### 6.5 `judge_claim`（LLM 裁判）

把一句自然语言断言交给 LLM 裁判，对着某个文本切片判 Y/N。**最后兜底用**——
能结构化判定就别用它（慢、有抖动、要花 LLM 调用）。

```yaml
assert:
  type: judge_claim
  from: emails              # 文本选择器，见 §7；默认 all
  claim: |
    邮件 body 明确给出录用建议，必须点名"录用 / 不录用 / 复试"之一。
```

当命名空间包含大量记录时，可先用 `jmespath` 在原始 state 上选择目标，
再只把查询结果序列化给裁判。 查询相对于 `from` 选中的原始值执行：

```yaml
assert:
  type: judge_claim
  from: tencent_docs
  jmespath: "documents[?title=='深圳公积金账户余额使用方案对比表'] | [0]"
  claim: |
    “直接付首付”方案明确标为不可行，或可用金额为 0 元。
```

| 字段 | 必填 | 含义 |
|---|---|---|
| `from` | 否（默认 `all`） | 文本切片选择器，见 §7。 |
| `jmespath` | 否 | 在序列化前对 `from` 的原始 state 执行 JMESPath；空结果返回 `EMPTY_SOURCE`，非法查询返回 `INVALID`。 |
| `claim` | **是** | 自然语言断言。写得越具体、判定点越小越好。 |

**写好 `claim` 的原则：**
- 一条 claim 只判一个目标，别把多个不相关的判定塞进一个大 claim。
- 条件具体、可证伪：点出实际的名字 / 日期 / 金额 / 维度。
- 裁判规则：SOURCE 要有"实质证据"才判 Y，只是 name-drop（提了一嘴）判 N；
  SOURCE 为空也判 N。
- 对列表查询应尽量用 `| [0]` 收敛为单条记录；若保留多条匹配，裁判会同时看到所有匹配项。

---

### 6.6 `test_result`（跑测试看 junit 结果）

把复杂客观检查（文件是否存在、数值精度、单测）外包给一个测试脚本，引擎读它产出的
junit 报告来判这条 check。

```yaml
assert:
  type: test_result
  # 默认按 check.id 找 junit testcase；需要显式映射时写 test_name
  # test_name: TestOutputExists
```

| 字段 | 必填 | 含义 |
|---|---|---|
| `test_name` | 否 | 直接指定 junit testcase 名。 |
| （无 `test_name`） | — | 用 `check.id` 按命名约定去找 junit testcase。 |

**`check.id` → junit testcase 名的解析顺序**（`rubric/check_map.py`）：
1. `tests/check_map.yaml` 里若有 `check.id` 的显式映射，用它的 `junit_name`。
2. 否则按约定依次试：`test_<id>` → `Test<PascalCase(id)>` → `<id>` 字面量。

**配套协议（用了 `test_result` 就必须满足）：**
1. case 目录下有可执行的 `run-test.sh`（引擎优先调 case 根的，缺省 fallback 到 `tests/run-test.sh`），
   agent 跑完后由引擎调一次。
2. 它必须产出 `/workspace/tests/report.junit.xml`（junit 格式，跨语言通用：pytest
   `--junitxml`、gotestsum `--junitfile`、jest-junit、甚至手写 xml 都行）。
3. 即使有用例 fail 也要把文件落盘（脚本里 `|| true`），否则引擎拿不到结果。

最小 `run-test.sh`（pytest）：

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/tests
uv run pytest \
    /workspace/.aidata/case/tests/test_outputs.py \
    --junitxml=/workspace/tests/report.junit.xml \
    || true
```

`tests/check_map.yaml` 示例（junit 名不匹配约定时）：

```yaml
output_md_exists:
  junit_name: TestOutputExists
summary_grounded:
  junit_name: "tests::summary::test_grounded"
```

---

## 7. `from` 选择器（枚举字典）

**只有 `text_contains` / `judge_claim` 用这套文本选择器**
（`state_equals` 用 §6.1 的 slice/path/jmespath 规则；`tool_called` /
`tool_called_sequence` 直接遍历原始 trajectory，不看选择器）。

`judge_claim` 额外支持可选 `jmespath`：先解析 `from` 指向的原始 state，
执行查询后再把结果渲染为文本。 这适合从大型文档列表中按标题或 ID 选择单个目标，
避免把整个命名空间送入裁判并触发 SOURCE 截断。

推荐写法：`<namespace>.<state_key>[.<path>]`，点号会被解析。`namespace` 来自 MCP
`server_name`，`state_key` 来自该 MCP 的 `*_dump_state` 返回。

```yaml
from: meituan.orders          # state['meituan']['orders']
from: food_delivery.orders
from: tencent_docs.documents
```

所有 mock 状态都用上面的 `<server_name>.<state_key>` 命名空间形式读取。除此之外只有
**3 个引擎内置的特殊切片**（不是 mock 命名空间，照常用）：

| 选择器 | 内容 |
|---|---|
| `agent_answer` | Agent 的最终自然语言回复（引擎在评分前用 `extract_final_answer` 预填）。验订单号/金额/结论/交付路径最常用。 |
| `trajectory` | 首条 user 消息 + 每条 assistant/tool_call/tool_result（按 `config` 截断）。验"agent 怎么做的"。 |
| `all` | 上述全部 + 所有 mock 命名空间拼接。**仅兜底用**，会给裁判带噪声。 |

> 旧的扁平兼容 key（`wechat_log` / `emails` / `schedule` / `meeting_log` / `room_log` /
> `memos` / `weiyun_uploads` 等）已废弃，新 rubric 一律改用 `<server_name>.<state_key>`。
> 命名空间没命中时返回空源（`EMPTY_SOURCE`）。优先用最具体、含证据的选择器。

---

## 8. `verdict` 结果值（枚举字典）

评分后每条 check 在结果 JSON 里有一个 `verdict`。写 rubric 时知道它们的含义，
能快速判断"为什么这条没过"。

| verdict | 含义 / 触发场景 |
|---|---|
| `Y` | 通过（`passed=true`）。 |
| `N` | 明确不通过。 |
| `INVALID` | rubric 配置非法：缺谓词、同时给了多个谓词、空查询、正则编译失败、`count_op` 缺 `count_value` 等。 |
| `MISSING` | 取不到数据：slice/path 不存在、`sql:` 没声明 `rubric_sqlite_path`、SQL 报错或无行、sidecar runtime 没起来。 |
| `EMPTY_SOURCE` | `text_contains` / `judge_claim` 的源切片为空。 |
| `UNAVAILABLE` | `judge_claim` 的 LLM 裁判多次重试仍不可用。 |
| `NO_JUNIT` | `test_result` 但本次评分没带 junit 上下文（没跑 `run-test.sh`）。 |
| `ERROR` | evaluator 抛异常（被兜住，不会拖垮整份 rubric）。 |
| `UNKNOWN_ASSERT_TYPE` | `assert.type` 不在引擎支持的原语里。 |

> `MISSING` / `EMPTY_SOURCE` / `INVALID` 都算 **不通过**（fail-closed），不会静默给分。

---

## 9. 选型：到底用哪个原语

```
要验什么？
├─ 状态/写操作真发生了（订单、备忘录、支付…）
│   ├─ 状态在 JSON state 里        → state_equals + jmespath   （首选）
│   ├─ 状态在 sqlite 里            → state_equals + sql:
│   └─ 固定下标的标量值            → state_equals + from/path
├─ agent 调了某工具 / 带某参数      → tool_called（加 scope: live）
├─ 工具调用要满足先后顺序          → tool_called_sequence
├─ 某段文本必须/禁止出现关键词或模式 → text_contains
├─ 复杂客观检查（文件存在/数值/单测） → test_result（配 run-test.sh）
└─ 需要人类语义判断（质量/覆盖/合理） → judge_claim（最后兜底，判定点要小）
```

按高频场景的推荐组合：
- **电商/订单类**：`tool_called(scope: live)` 证明关键动作 + `state_equals(sql: 或 jmespath)`
  验最终订单/支付/优惠 + `text_contains(from: agent_answer)` 验用户可见结果。
- **文档交付类**：`tool_called` 证明读取/生成 + 产出 MCP 的命名空间 selector 验文件名和内容
  + `agent_answer` 验交付说明。
- **语义质量类**：先用 `text_contains` 锁硬条件，再用小而具体的 `judge_claim` 判质量。

---

## 10. 从 0 编写：完整步骤 + 可复制模板

**步骤：**

1. **列任务闭环**：把 case 要求 agent 做的事拆成 5～15 个"原子的、可证伪的"小目标，
   覆盖：准备/读取 → 执行动作 → 最终产物 → 易错点/不该做的事。
2. **给每个目标选维度**：`correctness`（做没做对）/ `functional`（产物质量）/
   `robustness`（边界、不越界）。
3. **给每个目标选原语**：照 §9 决策树，能结构化就别上 LLM。
4. **定数据源**：动作看 trajectory（`tool_called`）；状态看 mock 命名空间（`state_equals`）；
   用户可见结果看 `agent_answer`。命名空间名 = MCP `server_name`，state_key = 该 MCP
   `*_dump_state` 的返回 key（不确定就先跑一次看 `state_summary` / state dump）。
5. **写 `id` + `description`**：稳定 snake_case id；description 写"证明了什么"。
6. **配权重**：`weight` 直接影响加权连续总分。 一般写 `1`（等权）；重要项加大到
   `2` / `3`，次要项设 `0.5`。 想切到二元 / 阈值模式见 §4 评分规则块下的 env 说明。
7. **跑校验**：`./aidata.py validate <case_id>`，修掉所有 problem。
8. **试跑对照**：跑一次（最好有 `solution/golden_rollout.json` 对照），确认"好解"能拿高分、
   "坏解"会被扣分；再检查没有"念一句关键词就过"的可 hack check。

**可复制模板（覆盖全部 6 个原语）：**

```yaml
version: "4.2"

# config 可选，绝大多数 case 不用写；仅当用到 trajectory/all 且内容超长时才放开。
# config:
#   trajectory: {user_head_chars: 16000, tool_call_arg_chars: 800, tool_result_chars: 2000}
#   judge: {source_max_chars: 96000}

checks:
  # 1) 动作发生：调了某工具（带参数约束 + 只看本次运行）
  - id: searched_target
    description: agent 搜索了目标商品
    dimension: correctness
    weight: 1
    assert:
      type: tool_called
      tool: search_products
      with_text: {field: arguments.keyword, contains_any: ["漫步者", "耳机"]}
      scope: live

  # 2) 顺序约束：先搜再加购
  - id: search_then_add_cart
    description: 先搜索再加入购物车
    dimension: correctness
    weight: 1
    assert:
      type: tool_called_sequence
      scope: live
      steps:
        - {tool: search_products, with_text: {field: arguments.keyword, contains: "耳机"}}
        - {tool: add_to_cart,    with_text: {field: arguments.itemId,  contains: "920568427905"}}

  # 3) 写操作真发生：最终状态里有已支付订单（JMESPath）
  - id: order_paid
    description: 至少有一笔已支付订单
    dimension: correctness
    weight: 1
    assert:
      type: state_equals
      from: shopping
      jmespath: "orders[?status=='paid']"
      count_op: ">="
      count_value: 1

  # 4) 用户可见结果：最终回复给出订单号与金额
  - id: answer_has_order_and_amount
    description: 最终回复包含订单号和金额
    dimension: functional
    weight: 1
    assert:
      type: text_contains
      from: agent_answer
      contains_all: ["订单号"]
      regex: '金额[:：]\s*\d+'

  # 5) 鲁棒性（反向）：没有给被禁商品下单
  - id: no_forbidden_item
    description: 没有把禁购商品加进订单
    dimension: robustness
    weight: 1
    assert:
      type: state_equals
      from: shopping
      jmespath: "orders[].items[?itemId=='000_FORBIDDEN']"
      count_op: "=="
      count_value: 0

  # 6) 语义质量：交付说明合理（LLM 裁判，判定点小）
  - id: delivery_note_reasonable
    description: 交付说明清楚说明了已下单与预计到货
    dimension: functional
    weight: 1
    assert:
      type: judge_claim
      from: agent_answer
      claim: 回复明确说明订单已提交，并给出预计到货时间（具体日期或天数）。

  # 7) 复杂客观检查：跑测试看 junit（需配 run-test.sh + report.junit.xml）
  - id: output_file_exists
    description: 产出文件写到了用户指定路径
    dimension: correctness
    weight: 1
    assert:
      type: test_result
```

---

## 11. 校验与常见错误

跑 `./aidata.py validate <case_id>` 做静态检查。它会逐条 check 校验，常见报错：

**rubric 整体：**
- `rubric.yaml: must be a dict with non-empty checks` — `checks` 缺失或为空。

**通用字段：**
- `checks[i].id is empty` / `duplicate check id` — id 缺失或重复。
- `checks[i].description is empty` — 描述缺失。
- `checks[i].dimension not in ['correctness', 'functional', 'robustness']` — 维度写错。
- `checks[i].weight is missing` / `must be > 0` / `must be numeric` — 权重问题。
- `checks[i].assert must be a mapping` / `assert.type is empty` / `assert.type=... not in {...}` — assert 块或 type 非法。

**各原语最小字段：**
- `state_equals`：要么 `jmespath` + 恰好一个谓词；要么有 `equals`/`one_of`；
  且非 jmespath 形态必须有 `from`。`count_op`/`count_value` 必须成对。
- `text_contains`：必须有 `contains` / `contains_all` / `contains_any` / `regex` 之一。
- `tool_called`：必须有 `tool` 或 `tool_prefix`。
- `tool_called_sequence`：`steps` 必须是非空 list，每个 step 都要有 `tool` 或 `tool_prefix`。
- `judge_claim`：`claim` 不能为空。
- `test_result`：无强制字段（靠 `check.id` 或 `test_name` 解析 junit）。

**运行期才暴露的坑（validate 查不出来）：**
- `state_equals` 把 `from` 当 dotted path 写（如 `from: meituan.orders.0.status`）→ 运行时
  `MISSING`。正解：`from: meituan` + `path: orders.0.status`。
- `state_equals + sql:` 但 `task.yaml` 没声明 `rubric_sqlite_path` → `MISSING`。
- `text_contains` / `judge_claim` 的 `from` 命名空间名写错 → `EMPTY_SOURCE`。
- 用 `text_contains` + 工具名正则去 trajectory 验"写操作"→ 容易被 agent 念关键词 hack，
  改用 `state_equals`。
- `tool_called` 不写 `scope: live`，结果把 `initial_trajectory` 里 seed 的调用也算进来。
