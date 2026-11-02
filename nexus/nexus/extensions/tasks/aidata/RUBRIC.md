# Rubric v4.2 — Annotator-facing DSL + 进阶 test_result 协议

这份文档面向两类人：

- **标注员 / case 作者**：只要读 §1–§3 就够，不需要任何代码。
  需要调轨迹截断阈值时读 §5。
- **开发者**：§4–§7 讲引擎如何 dispatch、截断阈值如何配置、如何接入
  pytest/go-test/自定义测试框架，以及为复杂 case 做 escape hatch 时的
  契约。

所有新 case 都必须用 v4.2 结构（旧 v2 仍能跑，但不再推荐）。

---

## 0. 新 rubric 写作倡议

`rubric.yaml` 是 case 的可审计评分契约。新 case 请优先写“清晰、稳定、可复现”的
v4.2 DSL，而不是依赖历史兼容字段或大段 LLM judge。

现有 `Tasks` 里有几类主流写法会继续兼容：legacy v2 的 `verify: regex` /
`verify: llm_judge`，生成器产出的 `pattern:`，以及批量 work case 里的
`scope: live`。这些不要为了“换字段名”单独 churn；新 case 和实质修改过的
rubric 再按下面推荐写法收敛。

推荐：

- 只写 `version: "4.2"` + `checks[]` + `assert:`；旧 v2 `verify:` 只为历史 case 保留。
- 统一用 `from:` 选择 state slice；`source:` 只是兼容旧字段，新 rubric 不再主动使用。
- 能结构化判定就用 `state_equals` / `test_result`；能看工具行为就用 `tool_called` /
  `tool_called_sequence`；明确文本才用 `text_contains` / `text_not_contains`；最后才用
  `judge_claim`。
- 新工具行为检查建议写 `scope: live`，避免 `initial_trajectory` seed 历史里的工具调用被计分。
- 正则统一写 `regex:`；`pattern:` 只是历史兼容别名。纯文本统一写 `contains` /
  `contains_all` / `contains_any`；`text:`、`keywords:` 也是历史兼容别名。
- `from` 优先写成清晰的 state selector：`<namespace>.<state_key>[.<path>]`，
  例如 `meituan.orders`、`food_delivery.orders`、`taobao.orders`。
  `docs` / `docs:excel` / `docs:word` 这类聚合 selector 只作为历史兼容保留。
- `all` 只作兜底，不作为默认习惯；它会给 judge 带来更多噪声。
- 每个 `id` 都要稳定、唯一、snake_case；`description` 写清楚“这个 check 证明什么”。

不推荐：

- 新增 `source:` / `pattern:` / `text:` / `keywords:` 这类兼容写法。
- 用一个很大的 `judge_claim` 同时判断多个无关目标。
- 在 rubric 里检查 task manifest/runtime 行为，例如 `mock_replay_file` 是否注入 env；
  这类逻辑应在 engine/mock 层测试，rubric 只验证最终可观察行为和状态。

按现有 `Tasks` 的高频场景，推荐三种组合：

- 电商 / 订单类：`tool_called(scope: live)` 证明关键动作，`state_equals(from: sql:...)`
  验证最终订单/支付/优惠/地址，`text_contains(from: agent_answer)` 验证用户可见结果。
- 文档交付类：`tool_called(scope: live)` 证明读取/生成动作，优先用产出 MCP 的
  namespace selector（如 `tencent_docs.documents`）验证文件名和关键内容；
  `agent_answer` 只检查交付说明。
- 语义质量类：先用 `text_contains` 锁定硬条件，再用小而具体的 `judge_claim` 判断质量；
  不要把所有硬条件都塞进一个 judge。

---

## 1. case 目录

```
cases/<case_id>/
├── task.yaml                  # 必需：manifest（含 metadata 块：created_at / tags；字段见 Tasks 仓 SCHEMA.md）
├── instruction.md             # 必需：agent 可见的 prompt
├── README.md                  # 必需：reviewer-facing 简述
├── rubric.yaml                # 必需：v4.2 DSL，详见 §2
├── data/                      # agent 可见的 context fixtures
├── tests/                     # 可选：junit 生成逻辑
│   ├── run-test.sh            # 可选：case-root run-test.sh 缺省时 engine fallback 到这里
│   ├── check_map.yaml         # 可选：check.id → junit testcase name
│   └── ...                    # pytest / go test / jest / 任意框架
├── run-test.sh                # 可选：若使用 assert.type: test_result 必需；engine 优先从 case-root 调用，fallback 到 tests/run-test.sh
├── solution/                  # reviewer-facing 参考物（agent 不可见）
│   ├── golden_rollout.json    # generative case 的标准 rollout
│   ├── NOTES.md               # 作者对"好解"的说明
│   └── solve.py / golden.patch  # scriptable oracle / bug-fix 场景
└── assets/                    # bootstrap 脚本 + seed 数据（agent 不可见）
```

> 任意文件缺失 / `metadata` 字段非法，会被 `./aidata.py validate <case_id>` 报出来。

---

## 2. rubric.yaml 的 assert 原语

每条 check 结构如下：

```yaml
- id: <snake_case_unique>
  description: <一行人类可读>
  dimension: correctness | functional | robustness
  weight: <float, > 0>
  assert:
    type: <见下各原语>
    ...
```

总分 = `Σ weight_i × passed_i / Σ weight_i`  
维度分 = 按 `dimension` tag 分别算加权通过率

### 2.1 `state_equals` — 结构化等值

```yaml
assert:
  type: state_equals
  from: wechat_friends           # 或留空直接用 `source:` dotted-path
  path: alice.status
  equals: accepted               # 或 one_of: [pending, accepted]
```

适用：state 里某路径固定是某个值。

#### 2.1a `state_equals` + `sql:` — sqlite 标量探针（可选）

当 mock 把权威状态放在 sqlite（而不是 `STATE_DIR/*.json`）时，可以用
``from: "sql:…"`` 直接在 **只读** sqlite 连接上跑一条返回 **单个标量单元格**
的查询，再用 ``equals`` / ``one_of`` 比较结果：

```yaml
assert:
  type: state_equals
  from: |
    sql:SELECT CASE WHEN EXISTS (
      SELECT 1 FROM cart_items c
      JOIN products p ON p.item_id = c.item_id
      WHERE p.category = '眼罩'
    ) THEN 1 ELSE 0 END
  equals: 1
```

前置条件：**case 必须在 `task.yaml` 顶层声明 `rubric_sqlite_path:`**，
engine 层只负责把 `__CASE_UUID__` / `__AIDATA_CASE_UUID__` / service URL 占位符
替换掉，再把最终路径写进 ``state['_rubric_sqlite_path']`` 给 rubric 层开：

```yaml
# Tasks/<case>/task.yaml — stdio mock，db 在 main 容器内
task_id: my_sqlite_case
extra_mcp_servers:
  - name: shopping
    command: /opt/aio-deps/.venv/bin/python
    args: [/opt/mock-server/life/online_shopping/server.py]
    env:
      ONLINE_SHOPPING_DB: /tmp/mock-run/__CASE_UUID__/online_shopping.db
rubric_sqlite_path: /tmp/mock-run/__CASE_UUID__/online_shopping.db
```

**跨容器变体（HTTP MCP sidecar 模式）**：当 mock 跑在 sidecar service 容器里、
sqlite 文件落在 sidecar 的 fs（典型 `/var/lib/mock-server/<name>/`）时，agent
主容器的 `python3 -c "import sqlite3"` 探针打不开这个路径。在
`rubric_sqlite_path` 前加 `service:<service-name>:<path>` 前缀，engine 会把
探针路由到对应的 sidecar runtime（`rubric/sql.py::_resolve_sql_runtime`）：

```yaml
# Tasks/<case>/task.yaml — HTTP sidecar，db 在 service 容器内
services:
  - name: lifegym-mocks
    runtime_name: service_lifegym_mocks
    image: mirrors.tencent.com/aidata-rl-env/mock-server:v0.2.2
    # ... startup / ready_check / connections ...

# `service:lifegym-mocks:` 让 sql probe 跑在 service_lifegym_mocks 容器里。
# 只接受 services[].name（人类标签），engine 会自动 lookup runtime_name。
rubric_sqlite_path: 'service:lifegym-mocks:/var/lib/mock-server/online_shopping/rollout.db'
```

校验细节：service-prefix 形态在 `case/spec.py::resolve_case` 解析时
做静态校验——如果 `<service-name>` 不在 `services[].name` 里，会立即报错并列出
可选 service 名（catch typo class）。运行期 sidecar runtime 没起来 / 名字写错
也会以 `MISSING (service_runtime_not_acquired: ...)` 失败，不会静默回落到 main。

engine 层故意 **不** 去 sniff 业务 env（如 `ONLINE_SHOPPING_DB`）——增加一个新的
sqlite-backed case 只需要在它自己的 `task.yaml` 里声明 `rubric_sqlite_path`，
不用回来改 `engine/`。若 case 用了 `sql:` 但没声明路径，该 check 会以
``MISSING`` 失败并给出可读的 ``detail``（fail-closed，不静默 0 分）。

#### 2.1b `state_equals` + `jmespath` — 任意结构查询（推荐用于「写操作真发生」）

当你想验证 agent 真的改了 mock state（而不是只在 trajectory 里念了
tool 名），用 JMESPath 查询打到具体的表/记录上，再用一个判定谓词决定
通过：

```yaml
# 至少 1 条 status=paid 的订单
assert:
  type: state_equals
  from: life_payment                                       # 命名空间（可省 = 整个 state）
  jmespath: "life_payment_orders[?status=='paid']"
  exists: true
```

`from` 可省（缺省作用于整个 `state` dict，包括 `agent_answer` /
`trajectory` / 任意 mock 命名空间）。一旦 cfg 里出现 `jmespath` 键，就
进入本分支；不带它则回到 2.1 / 2.1a 原行为，完全向后兼容。

**九种判定谓词**（每条 check 恰好指定一个）：

| 谓词 | 输入 | 通过条件 | 示例 |
|---|---|---|---|
| `exists: <bool>` | any | 结果"truthy"（非 None / 非空 list/dict/str / 非 0） | `exists: true` |
| `equals: <any>` | scalar | 结果 == 值（数字带 int/float 强转） | `equals: paid` |
| `one_of: [<v>...]` | scalar | 结果 ∈ list | `one_of: [paid, refunded]` |
| `count_op: <op>` + `count_value: <int>` | list/dict/str | `len(结果) <op> count_value`；`op` ∈ `{==, !=, >, >=, <, <=}` | `count_op: ">="`<br>`count_value: 1` |
| `contains: <any>` | list/dict/str | 元素 ∈ list / 子串 ∈ str / key ∈ dict | `contains: "paid"` |
| `contains_all: [<v>...]` | list/dict/str | 每个元素都在 | `contains_all: ["a", "b"]` |
| `contains_any: [<v>...]` | list/dict/str | 至少 1 个元素在 | `contains_any: ["a", "b"]` |
| `regex_match: <pattern>` | any | `re.search(pattern, str(结果))` 命中 | `regex_match: "WS\\d+"` |
| `gt` / `gte` / `lt` / `lte: <number>` | number | 数值比较（双侧 float 强转） | `gte: 100` |

**常见用法示例**：

```yaml
# 1. 计数：至少创建了 1 张支付订单
- assert:
    type: state_equals
    from: life_payment
    jmespath: "life_payment_orders[?status=='paid']"
    count_op: ">="
    count_value: 1

# 2. 聚合：所有已付订单总金额超过 100 元
- assert:
    type: state_equals
    from: life_payment
    jmespath: "sum(life_payment_orders[?status=='paid'].total_amount)"
    gte: 100

# 3. 跨字段：报修工单 description 里含下午5点
- assert:
    type: state_equals
    from: life_payment
    jmespath: "life_payment_complaints[0].description"
    regex_match: "(下午\\s*5|17[:点]|傍晚)"

# 4. 不指定 from：直接验最终回复
- assert:
    type: state_equals
    jmespath: "agent_answer"
    regex_match: "工单号|已报修"

# 5. 跨命名空间：验微信侧也写了消息
- assert:
    type: state_equals
    from: wechat
    jmespath: "wechat_chat_messages[?type=='share']"
    count_op: ">="
    count_value: 1
```

**何时用 jmespath 分支 vs 其它**：

- **写操作 / 状态变化**：必用 jmespath（不要再用 `text_contains regex=<tool_name>`，
  那只验文本不验状态，agent 在 thought 里念关键字就能 hack）。
- **固定下标的标量值**：用 2.1 原 dotted-path 更简洁（如 `path: emails.0.subject`）。
- **SQL count / join / group by**：用 2.1a `sql:` 模式（mock 必须暴露 sqlite）。

**错误返回**：

- `INVALID: bad_jmespath: ...` — query 编译失败
- `INVALID: missing predicate` — 没指定任何判定方式
- `INVALID: multiple predicates specified` — 同时指定了 >1 个判定方式
- `INVALID: count_op requires count_value`（或反向）— 计数算子缺配对
- `MISSING: missing_slice: <name>` — `from` 指向不存在的命名空间

引擎依赖 [`jmespath`](https://jmespath.org/) ≥ 1.0（声明在 `pyproject.toml`）。
完整语法见 https://jmespath.org/specification.html ，内置 `length` /
`sum` / `min` / `max` / `avg` / `keys` / `values` 等聚合函数。

### 2.2 `text_contains` — 关键字 / 正则匹配

```yaml
assert:
  type: text_contains
  from: meituan.orders           # <namespace>.<state_key>；也可用 agent_answer / trajectory，见 §6
  contains: ["PAID", "海底捞"]    # 或 contains_any / contains_all
  regex: '"amount"\s*:\s*(?:[0-9]{1,2}(?:\.\d+)?)'  # 可选：Python 正则
  case_sensitive: false          # 默认 false（仅对 contains* 生效）
```

多个断言字段同时出现时按 AND 叠加（所有 predicate 都要通过）。
`regex` 使用 `re.search` 走原文（不走 lower()），如需忽略大小写请在 pattern 里写 `(?i)`。
兼容历史生成的 rubric：现有 Tasks 中不少 `text_contains` 写的是 `pattern:`，
engine 会按 `regex` 处理；如果同时写了 `regex` 和 `pattern`，以 `regex` 为准。
`text` 按单个 `contains` 处理；`keywords` 按 `contains_any` 处理。`from` 是
canonical selector，`source` 作为旧字段别名可用。
负向检查可用兼容原语 `text_not_contains`，其 predicate 字段同 `text_contains`。

适用：某段文本必须出现 / 不出现某些关键词，或包含一个形如「省 ≥ 300 元」的
数字模式。

现有 case 里最常见的文本目标是：

- `from: agent_answer`：验证最终回复是否给出订单号、金额、结论、交付路径等。
- `from: meituan.orders` / `food_delivery.orders`：验证业务 mock 的最终订单状态。
- `from: tencent_docs.documents`：验证具体文档 MCP 的产物。现有
  `from: docs:excel` / `docs:word` 继续兼容，但新 rubric 不建议再写。
- `from: trajectory`：验证 agent 过程里确实读取了输入或提到了关键中间量。

### 2.3 `tool_called` — Agent 真的调用了某工具

最简形态：

```yaml
assert:
  type: tool_called
  tool_prefix: wikipedia_        # 或 tool: 精确名
  acceptable_tools:              # 可选：任意一个命中即可通过，和 tool 是 OR 关系
    - wiki_search
    - wikipedia/search           # `/`、`.` 会按命名空间分隔符兼容 `_`
  with_text: "Alan Turing"       # 简写：case-insensitive substring over args JSON
  min_count: 2                   # 可选，默认 1
  max_count: 3                   # 可选；写 max_count: 0 表示禁止调用
  scope: live                    # 可选：live / seed；seed 来自 task.yaml::initial_trajectory
```

进阶 step-spec（每个字段都可选，同时指定时按 AND 叠加）：

```yaml
assert:
  type: tool_called
  tool: submit_order             # 精确名 or tool_prefix 之一
  with_text:                     # 字典形态：按字段做细粒度匹配
    field: arguments.couponIds   # dotted path（arguments.*、tool_name、name）
    equals: "GLOBAL_SAVE200"     # 精确等值，大小写敏感
    contains_all: ["GLOBAL_SAVE200", "SHOP_XTEP_100"]  # 每一项都必须在
    contains_any: ["iPhone", "华为"]                    # 或任一匹配
    regex: "^ORD-[0-9A-F]{10}$"                         # 或正则命中
  arg_empty: [itemIds]            # 列出的字段必须缺失 / 空
  arg_not_in:                     # 字段的值不能是列表里的任何一个
    itemId: [881000000001, 920568427905]
  all_of: [ {...}, {...} ]        # 布尔 AND（嵌套 step-spec）
  any_of: [ {...}, {...} ]        # 布尔 OR
  not:      {...}                 # 布尔 NOT（单个 step-spec）
  matcher: my_named_hook          # 案例无关的 python 函数注册钩子
  min_count: 1                    # 至少匹配几次
  max_count: 1                    # 至多匹配几次；不写则无上限
```

语义：对 trajectory 里每一条 assistant 消息的 tool_calls 逐个走 step-spec；满足
全部断言的计 1 次；``min_count <= count <= max_count`` 即通过（未设置
`max_count` 时只检查下限）。如果只写 `max_count: 0` 且不写 `min_count`，
表示该工具调用必须不存在。
如果 task.yaml 声明了 `initial_trajectory`，`scope: seed` 只看注入的历史轨迹，
`scope: live` 只看本次运行产生的轨迹；不写 `scope` 则看完整 trajectory。

适用：证明 agent 用了 MCP 工具，而不是凭记忆写的；也用于「带某张券下了单」、
「没有加到特定商品」这类带参数约束的动作校验。

现有 Tasks 里 `tool_called` 是最常见的 v4.2 原语。新 case 推荐默认写
`scope: live`，并尽量用 `with_text.field` 指到具体参数，例如
`arguments.keyword`、`arguments.itemId`、`arguments.command`。只在确实不关心
参数时写纯工具名。

### 2.4 `tool_called_sequence` — 工具调用的有序子序列

```yaml
assert:
  type: tool_called_sequence
  steps:
    - tool: search_products
      with_text:
        field: arguments.keyword
        contains_any: ["漫步者", "耳机"]
    - tool: get_product_skus
      with_text: {field: arguments.itemId, contains: "920568427905"}
    - tool: add_to_cart
      with_text: {field: arguments.itemId, contains: "920568427905"}
```

语义：从左到右走每个 step；当前 cursor 之后的第一个匹配的 call 命中后，cursor
前移。**全部 steps 命中才通过**（binary，不给部分分）；detail 里会记录
`matched N/M (indices=[...])` 方便审查。

每个 `steps[i]` 支持的字段与 §2.3 `tool_called` 完全一致 —— 这两个原语共享
同一份 step-spec 解析代码。

适用：SOP 类约束 —「先 search → 再 get_skus → 再 add_to_cart」、「客服流程
open_chat → send_chat_message → read_chat_history」这种顺序敏感的场景。

### 2.5 `judge_claim` — LLM-as-judge（自然语言断言）

```yaml
assert:
  type: judge_claim
  from: emails
  claim: |
    邮件 body 明确给出录用建议，必须点名录用 / 不录用 / 复试 之一。
```

大型命名空间可在序列化前用 JMESPath 选择目标，再只把查询结果交给裁判：

```yaml
assert:
  type: judge_claim
  from: tencent_docs
  jmespath: "documents[?title=='深圳公积金账户余额使用方案对比表'] | [0]"
  claim: “直接付首付”方案明确标为不可行，或可用金额为 0 元。
```

`jmespath` 查询相对于 `from` 的原始 state 执行。 空结果返回 `EMPTY_SOURCE`，
非法查询返回 `INVALID`；列表查询建议用 `| [0]` 收敛到单条记录。

适用：需要人类判断的语义检查（建议 / 覆盖维度 / 引用合理）。写 claim 的
好原则：条件具体、不含糊、判定点小。

### 2.6 `test_result` — 测试结果

```yaml
assert:
  type: test_result
  # 默认按 check.id 找：test_<id> / Test<PascalCase(id)> / <id>
  # 需要显式映射时：test_name: TestSomething
```

适用：复杂客观检查（文件存在、数值精度、单元测试）。详见 §3。

---

## 3. `run-test.sh` + `report.junit.xml` 协议

如果 rubric 里有任何 `assert.type: test_result`，case 必须满足：

1. **`cases/<id>/run-test.sh`** 可执行，由 engine 在 agent 完成后调一次
2. 它必须产出 **`/workspace/tests/report.junit.xml`**（junit 格式）
3. rubric 里 `check.id` 到 junit testcase name 的映射，通过：
   - 命名约定（`test_<id>` / `Test<PascalCase>`）；或
   - **`tests/check_map.yaml`** 显式声明

### 最小 run-test.sh 模板（pytest）

```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/tests

# 运行你自己的测试框架，只要最后产出 junit 就行
uv run pytest \
    /workspace/.aidata/case/tests/test_outputs.py \
    --junitxml=/workspace/tests/report.junit.xml \
    || true   # 即使有 fail 也要落文件，否则 engine 拿不到结果
```

### 为什么接受 junit（不是 pytest-only）

junit.xml 是跨语言的工业标准：

| 语言 | 生成工具 |
|---|---|
| Python | pytest `--junitxml` |
| Go | gotestsum `--junitfile` |
| Rust | cargo-junit |
| JS/TS | jest-junit |
| 自定义 shell | 手写几十行 xml |

只要 `run-test.sh` 产出的文件在固定路径，engine 不 care 你用什么框架测。

### check_map.yaml 示例

```yaml
# tests/check_map.yaml
# 当 junit testcase name 不匹配命名约定时显式指定
output_md_exists:
  junit_name: TestOutputExists
summary_grounded:
  junit_name: tests::summary::test_grounded
```

---

## 4. engine dispatch（开发者）

`engine/rubric.py` 现为兼容 shim（re-export），实际逻辑分散在 `rubric/` 子包。 模块速查表：

| 文件 | 职责 |
|---|---|
| `rubric/config.py` | `RubricLimits` / `VerifierResult` / `JunitLookupContext` / `ScorePolicy` 等共享类型 |
| `rubric/dispatch.py` | `run_rubric_async` / `run_rubric` / `compute_scores` / `_ASSERT_EVALUATORS` 注册表 / `_rubric_is_v42` |
| `rubric/selectors.py` | `_gather_source` / `_gather_trajectory` / `_resolve_state_source` |
| `rubric/judge.py` | `_judge_once` / `_llm_judge` LLM judge client |
| `rubric/sql.py` | SQL probe helpers（`_resolve_sql_runtime` / `_snapshot_sql_scalar`） |
| `rubric/jmespath_predicates.py` | jmespath 谓词分发 |
| `rubric/tool_matcher.py` | `tool_call` step-spec 匹配 |
| `rubric/legacy_v2.py` | v2 分派（`register_verifier` + `_v_*`） |
| `rubric/primitives/*.py` | 每个 v4.2 assert 原语一个文件 |
| `rubric/fetch.py` | Fetcher 门面（Commit 3 引入） |
| `rubric/junit.py` / `rubric/check_map.py` | junit 解析 + `check.id → testcase` 映射 |

各原语的 dispatch 树 、 verdict 集与调用链见 [`docs/reference/rubric-primitives.md`](docs/reference/rubric-primitives.md)。 顶层入口的行为大致如下：

```
run_rubric_async(state, rubric, *, runtime=None, junit_ctx=None,
                 limits=None, concurrency=None)  ← rubric/dispatch.py
  └─ 根据 rubric.version 或 check 内是否有 assert → 走 v4.2 或 v2 路径
  └─ asyncio.Semaphore(concurrency) 包住每条 check，asyncio.gather 并发跑
  └─ v4.2：逐条 check 调 _ASSERT_EVALUATORS[assert.type](cfg, state, runtime, junit_ctx)
  └─ v2：  逐条 check 调 _VERIFIERS[verify](state, check, runtime=...)
  └─ compute_scores(rubric, rubric_results) 算 weighted total + per-dim

JunitLookupContext(report, explicit_map)  ← rubric/config.py
  └─ 由 engine/task.py::_maybe_run_case_tests 在调完 run-test.sh 后装填
  └─ test_result evaluator 通过它查 testcase
```

**并发**：
- 默认并发 8（`concurrency` 不传时从 `NEXUS_RUBRIC_CONCURRENCY` 读，
  缺省 8）。`concurrency=1` 退化到串行，等价于历史行为。
- 多个 `judge_claim` / `llm_judge` 走 `asyncio.to_thread`，把
  阻塞的 LLM 调用真正并发到 worker 线程；CPU-only 的 `text_contains`
  / `tool_called` / `state_equals` / `test_result` 也会被 gather 跑，
  但通常本来就是 µs 级。
- `[rubric] check i/N id=... verdict=... pass=...` 日志在每条 check
  完成时立即打印，**输出顺序按完成顺序**（不是 rubric 声明顺序），
  `i/N` 标识仍指向 rubric 中的位置；`run_rubric_async` 返回的结果
  list 始终按 rubric 声明顺序排好。

evaluator 签名（若要注册新 primitive）：

```python
async def _eval_xxx(
    cfg: dict,
    state: dict,
    runtime: Any,
    junit_ctx: JunitLookupContext | None,
    *, limits: RubricLimits | None = None,
) -> tuple[passed: bool, verdict: str, detail: str, raw_head: str]: ...
```

新 primitive 注册到 `_ASSERT_EVALUATORS` dict 即可。 `limits` 是可选
kw-only 参数，evaluator 只在真的要把 trajectory 拍平或调 judge 时才需
要读它（见 §5）；不关心的 evaluator 忽略即可。

---

## 5. rubric 顶层 `config`（可选，控制截断阈值）

`judge_claim` / `text_contains` / `llm_judge` / `regex` 在拿到 `from: trajectory`
或 `from: all` 时，会先把 trajectory 拍平成一段文本喂给 judge 或做
substring 匹配。 以下四个字符上限默认值基本够用，但有些 case（tool
返回 2k 的 JSON、task prompt 超过 32k、要看完整 assistant 输出）需要
放开。 通过 rubric 顶层 `config` 块声明：

```yaml
version: "4.2"
config:
  trajectory:
    user_head_chars:     16000   # 首条 user 消息头部 cap
    tool_call_arg_chars: 800     # 每条 assistant.tool_call arguments cap
    tool_result_chars:   2000    # 每条 tool_result body cap
  judge:
    source_max_chars: 96000      # judge 最终拼好的 SOURCE 总 cap
checks: [...]
```

**默认值 / 省略规则**：

- 所有字段都可选，缺省值为 16000 / 800 / 2000 / 96000（user_head / arg / result / source）。
- 整个 `config` 块不写，行为与现在完全一致。
- 只写需要 override 的字段，其余字段拿默认值。
- 非正整数 / 非数字会被 log warn 并回退默认值，不会打断评分。

**作用范围**：

| 原语 | 是否受 `config` 影响 |
|---|---|
| `judge_claim` | 受影响（拍平 trajectory + SOURCE cap 都生效） |
| `text_contains` | 受影响（拍平 trajectory） |
| `tool_called` | 不受影响（直接遍历 `state['trajectory']`） |
| `state_equals` | 不受影响（走 dotted-path） |
| `test_result` | 不受影响 |
| v2 `llm_judge` / `regex` | 受影响（与 v4.2 等价原语同逻辑） |
| v2 `eq` / `shell` | 不受影响 |

**外部 override**：

```python
from nexus.extensions.tasks.aidata.engine import rubric as rubric_lib

await rubric_lib.run_rubric_async(
    state, rubric,
    limits=rubric_lib.RubricLimits(
        tool_result_chars=4000,
        judge_source_max_chars=120_000,
    ),
)
```

外部传入的 `limits` 优先于 rubric 里的 `config`，便于 harness 在不改
rubric.yaml 的前提下跑对比实验（跑完记得在结果里记录实际生效值，否则
评分不可自解释）。

---

## 5b. LLM routing：search / agent ≠ judge

`runtime/mcp_catalog.py` 维护两组独立的 env namespace：

| 用途 | env namespace | 优先级（高→低） |
|---|---|---|
| agent / mock-server LLM | `NEXUS_MOCK_LLM_BASE_URL` / `*_API_KEY` / `*_MODEL` | `NEXUS_MOCK_LLM_*` → `SEARCH_OPENAI_*` / `LLM_API_KEY` → 默认 |
| rubric 评分 judge | `NEXUS_JUDGE_BASE_URL` / `*_API_KEY` / `*_MODEL` | `NEXUS_JUDGE_*` → `JUDGE_OPENAI_*` → 回退到上面那组 |

`mcp_catalog.normalize_host_env()` 会按上表把 `os.environ.setdefault`
两组都填齐；`rubric/judge.py::_judge_once` 调 `llm_chat` 时显式传入
`base_url_env='NEXUS_JUDGE_BASE_URL'` / `api_key_env=
'NEXUS_JUDGE_API_KEY'` / `model_env='NEXUS_JUDGE_MODEL'`，所以
`JUDGE_OPENAI_BASE_URL` ≠ `SEARCH_OPENAI_BASE_URL` 这种 split routing
真的能落到 judge 上（历史 bug：之前 judge 的 base_url 会被
search 的 setdefault 占住，导致 `JUDGE_OPENAI_BASE_URL` 静默无效）。

只想用一组 LLM 跑全程的（默认情况），`JUDGE_OPENAI_*` 不写就行，judge
自动回退到 agent 那组凭据。

---

## 6. `state` / `from` 切片选择器

`judge_claim` / `text_contains` 的 `from:` 字段接受 state selector。推荐新写法是
`<namespace>.<state_key>[.<path>]`：

```yaml
assert:
  type: text_contains
  from: meituan.orders
  contains_all:
    - PAID
    - 海底捞

assert:
  type: judge_claim
  from: food_delivery.orders
  claim: |
    至少有一笔已支付订单，店铺是火锅类，且最终金额没有超过用户预算。

assert:
  type: state_equals
  from: meituan.orders.0.status
  equals: PAID
```

`namespace` 来自 MCP `server_name`；`state_key` 来自该 MCP 的 `*_dump_state` 返回。
如果 selector 没有命中 namespaced state，也可以继续使用下面这些稳定/兼容 selector：

| selector | 内容 |
|---|---|
| `wechat_log` | 全部微信消息 |
| `wechat_friends` | 好友请求状态机 |
| `schedule` | 查询到的日历 |
| `meeting_log` | 腾讯会议事件 |
| `room_log` | 会议室预订事件 |
| `memos` / `notes` | 结构化 memos |
| `emails` | 已发送邮件 |
| `trajectory` | user message + 每个 assistant/tool_call/tool_result |
| `agent_trajectory` / `agent_log` / `agent_action` | 兼容别名，等同 `trajectory` |
| `docs` | 历史聚合 selector：全部文档类产物；新 rubric 优先写具体 namespace，如 `tencent_docs.documents` |
| `docs:<type>` | 历史兼容写法；新 rubric 不推荐再写 |
| `weiyun_uploads` | 上传到微云的二进制文件清单 |
| `all` | 上述全部拼接 |
| `agent_answer` | Agent 的最终自然语言回复。和 `trajectory` 同类，是 engine 内存里预填的 slice：`engine/task.py` 在调 rubric 之前用 `runtime.state.extract_final_answer` 写入 `state['agent_answer']`（解析顺序：trajectory 里最后一次 `submit_response(text=...)` → 最后一条 assistant `content`）。 rubric 分派器就 `state.get('agent_answer')` 读一下，对 runner 的终止约定保持无感。 |

**规则**：`state_equals` 不用这个 selector 语法，它用 dotted path；其他
原语（`text_contains` / `judge_claim`）都用 selector。`tool_called` /
`tool_called_sequence` 走 trajectory 原始结构，既不看 selector 也不受
`limits.user_head_chars` / `limits.tool_call_arg_chars` 等截断影响。

---

## 7. 迁移到 v4.2（v2 → v4.2）

现有 v2 rubric 的两步转换（`rubric.yaml` 位置不变，都在 case 根目录）：

1. 顶部改 `version: "4.2"`
2. 每条 check 调整结构：

```yaml
# v2 ✗
- id: email_sent_to_hr
  description: ...
  source: emails
  verify: llm_judge
  claim: email to hr@example.com with decision

# v4.2 ✓
- id: email_sent_to_hr
  description: ...
  dimension: correctness
  weight: 1
  assert:
    type: judge_claim
    from: emails
    claim: email to hr@example.com with decision
```

对应关系速查：

| v2 `verify:` | v4.2 `assert.type:` |
|---|---|
| `llm_judge` | `judge_claim` |
| `eq` | `state_equals` |
| `regex` | `text_contains`（大部分情况够用）或留 v2 |
| `shell` | 没有直接对等 → `test_result`（推荐）或留 v2 |

`aidata.py validate <case_id>` 会告诉你每个 case 还缺什么。

迁移建议：不要为了字段名做无意义大批量改动。现有 v2 `verify: regex` /
`verify: llm_judge` 会继续跑；当你实质修改 case、需要加权维度分、或要混用
`state_equals(sql:)` / `tool_called(scope: live)` / `test_result` 时，再顺手迁到
v4.2。迁移时优先把可确定的 `regex` 改成 `text_contains`，把泛化
`llm_judge` 拆成更小的 `text_contains` + `judge_claim`。
