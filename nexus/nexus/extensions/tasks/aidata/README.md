# AIDATA RL ENV

详细文档见 [docs/](./docs/)。 CLI / rubric 编写 / 架构说明 / 排障均在此。

长链路办公/生活 agent 训练。所有 case 都跑同一个 `aidata` task，差异只通过 `case_id` 参数化。每个发布的 case 是自包含的 bundle（prompt、fixtures、手工 rubric、可选 bootstrap 脚本），都在 `cases/<id>/` 下——打成 tarball 给只拿到 aidata engine 的同事也能跑。

## 目录结构

```
aidata/
├── engine/                     — 任务入口 + 沙箱布局常量
│   ├── task.py                 — @task_registry.register('aidata') 入口
│   ├── models.py               — AidataTaskInput / Output / Service 等 pydantic 模型
│   ├── paths.py                — 沙箱布局常量（AGENT_DESKTOP / TESTS_DIR_IN 等）
│   ├── system_prompt.py        — agent 系统提示
│   └── case_spec.py / rubric.py / sandbox.py / context.py / mcp_catalog.py / services.py / state.py /
│                                  junit.py / check_map.py / prompt_builder.py / base_tools.py /
│                                  servers/ / confine/  —— 兼容 shim ，实际实现已迁到 case/ / rubric/ /
│                                  runtime/ / mock_servers/。 老 import 路径仍工作。
│
├── case/                       — case bundle 加载 + schema 单一真相源
│   ├── spec.py                 — task.yaml / rubric.yaml 加载 + 校验（兼容 v2 旧布局）
│   └── schema.py               — V42_ASSERT_TYPES / LEGACY_VERIFY_KINDS / ALLOWED_DIMENSIONS 等常量
│
├── rubric/                     — 打分子系统（独立包）
│   ├── dispatch.py             — v4.2 DSL evaluator + legacy v2 verify 双路径分发
│   ├── config.py               — RubricLimits / JunitLookupContext / VerifierResult / ScorePolicy
│   ├── policy.py               — ScorePolicy 与最终分坍缩（continuous / threshold）
│   ├── selectors.py            — 状态源 / trajectory 选择器
│   ├── judge.py                — LLM judge client
│   ├── sql.py                  — sqlite 标量探针
│   ├── jmespath_predicates.py  — JMESPath 谓词
│   ├── tool_matcher.py         — tool_call step-spec 匹配
│   ├── legacy_v2.py            — v2 分派路径（冻结）
│   ├── junit.py                — 解析 `tests/report.junit.xml`（支持 test_result 原语）
│   ├── check_map.py            — `check.id` → junit testcase name 解析
│   ├── fetch.py                — Fetcher 门面
│   └── primitives/             — 每个 v4.2 assert 原语一个文件
│
├── runtime/                    — 沙箱、上下文、服务编排
│   ├── sandbox.py              — 文件上传 helpers + bootstrap 执行器
│   ├── context.py              — host 侧 context 文件 manifest（agent 自读原始文件）
│   ├── mcp_catalog.py          — 默认 MCP 目录 + per-case extras 合并
│   ├── services.py             — 辅助服务容器编排（task.yaml::services）
│   ├── state.py                — 运行结束后下载 mock state + 生成摘要
│   ├── prompt_builder.py       — 组装 agent 用户消息
│   └── base_tools.py           — aidata 特有的本地 base tool 注册
│
├── mock_servers/               — 自带 mock MCP server 脚本 + confine
│
├── cases/<case_id>/            — 自包含的 case bundle（可打包分发，v4.2 布局）
│   ├── task.yaml               — manifest（原 `case.yaml`；legacy 名仍支持）
│   ├── instruction.md          — agent 可见的 prompt
│   ├── README.md               — 这个 case 做什么 + 基线分数
│   ├── rubric.yaml             — v4.2 rubric（case 根目录下的扁平文件）
│   ├── data/                   — context fixtures（原 `files/`；legacy 名仍支持）
│   ├── tests/                  — 可选：run-test.sh / check_map.yaml / pytest 文件
│   ├── run-test.sh             — 可选：case-root 版本，engine 优先调它
│   ├── solution/               — reviewer-facing 参考物，agent 不可见
│   │   ├── golden_rollout.json     — generative case 的标准 rollout
│   │   ├── NOTES.md                — "好解"说明
│   │   └── solve.py / golden.patch — scriptable oracle / bug-fix 参考解
│   ├── assets/                 — 可选：bootstrap_script、service 启动脚本、seed 数据
│   └── skills/                 — 可选：case 本地 skill 包（harness 会扫 SKILL.md frontmatter 注入 system prompt）
│
├── aidata.py                   — CLI（run / list / validate / package）
├── runtime.template.yaml       — 公共 runtime / agent / LLM 配置
├── pyproject.toml              — task 注册
├── viz/                        — 本地 FastAPI visualizer
├── docs/
│   └── mcp-contract.md         — MCP / 服务容器 / 镜像 / case 接入规范（团队统一口径）
├── RUBRIC.md                   — v4.2 rubric DSL + run-test.sh 协议（面向 case 作者）
└── README.md                   — 本文档
```

> **关于 rubric 怎么写**：见 [`RUBRIC.md`](RUBRIC.md)。它讲清楚 5 个 v4.2
> 断言原语（`state_equals` / `text_contains` / `tool_called` / `judge_claim`
> / `test_result`）、dimension/weight 权重模型、和 `run-test.sh` +
> `report.junit.xml` 的测试协议契约。

> **规范**：关于"新增 MCP server / 服务容器 / 镜像 / case 怎么写才算合规"的团队统一约定，见 [`docs/mcp-contract.md`](docs/mcp-contract.md)。本 README 侧重 **怎么用** 引擎；契约文档定义 **每类扩展长什么样**，让所有 case 服从同一个骨架。

## CLI（`aidata.py`）

```bash
# 列出可用的 case 和它们的简介
./nexus/extensions/tasks/aidata/aidata.py list

# 跑一个 case（自动加载 .env、渲染 runtime.template.yaml、调 nexus run-single）
./nexus/extensions/tasks/aidata/aidata.py run channel_inventory_ppt -v

# 做 case 结构静态检查：校验 task.yaml / instruction.md /
# rubric.yaml schema / README / context_files / v4.2 assert 原语
# 是否齐全合法（详见 RUBRIC.md）
./nexus/extensions/tasks/aidata/aidata.py validate channel_inventory_ppt

# 把一个 case 打包成可分发的 tarball
./nexus/extensions/tasks/aidata/aidata.py package channel_inventory_ppt ./bundles/

# 向后兼容：直接给 case_id（不写子命令）等同于 run
./nexus/extensions/tasks/aidata/aidata.py channel_inventory_ppt -v
```

## Batch（`nexus run-batch`）

跟其他 nexus task 一样，aidata 也提供了一个 `generator.py`，让 case 可以一键
扇出到 `nexus run-batch`：跨 case 并发、SQLite trace+log 持久化、断点续跑、
`inspect-batch` TUI 都开箱可用。

```bash
# 全量 fan-out，4 路并发 + 失败自动 retry 一次 + 导出 jsonl
AIDATA_CASES_ROOT=/path/to/your/cases \
uv run nexus run-batch \
    -g nexus/extensions/tasks/aidata/generator.py \
    -d /tmp/aidata_batch.sqlite \
    --max-concurrent 4 \
    --retry-failed 1 \
    --export-format jsonl

# 只跑两个 case 验证链路（generator 读 AIDATA_CASE_IDS 做 allowlist）
AIDATA_CASES_ROOT=/path/to/your/cases \
AIDATA_CASE_IDS=diabetes_followup,water_emergency \
uv run nexus run-batch \
    -g nexus/extensions/tasks/aidata/generator.py \
    -d /tmp/aidata_batch.sqlite

# 把声明了 pending_upstream 的 case 也强制带上（默认会跳过它们）
AIDATA_SKIP_PENDING_UPSTREAM=0 \
uv run nexus run-batch -g ... -d ...
```

### 从 Nexus Dataset 直接跑

发布到 Nexus Dataset 的 AIDATA instance 本质是一个带 `files[]` 索引的 case
bundle。`dataset_adapter.py` 会把它还原成现有 runner 认识的
`<workdir>/cases/<case_id>/`，不改写 case 内容。

三行命令分别对应 `nexus dataset -> A`、`A -> adapter -> B`、`nexus run B`：

```bash
uv run nexus dataset get HunYuanRLEnv/AIDATA-Tasks dump_state_poc --split train -o workspace/aidata_dataset_demo/dump_state_poc/instance.json
uv run python -m nexus.extensions.tasks.aidata.dataset_adapter workspace/aidata_dataset_demo/dump_state_poc/instance.json workspace/aidata_dataset_demo/dump_state_poc
uv run nexus run-batch -g workspace/aidata_dataset_demo/dump_state_poc/generator.py -d workspace/aidata_dataset_demo/dump_state_poc/run.sqlite --max-concurrent 1 --export-format jsonl
```

generator 读 `runtime.template.yaml` 一次、按 `cases/` 下每个 case 改写
`task_config.case_id`，所以 `aidata.py run` 跟 `nexus run-batch` 的每个 item 行为
完全一致 —— 只是 batch 路径多了并发 / 持久化 / resume 这些。env 旋钮：

| 变量 | 作用 |
|---|---|
| `AIDATA_CASES_ROOT` | cases 树根目录（`engine/paths.py` 与 generator 共用，未设走 `<package>/cases`） |
| `AIDATA_CASE_IDS` | 逗号分隔的 case_id allowlist；空 = 全量 |
| `AIDATA_SKIP_PENDING_UPSTREAM` | `1`（默认）跳过 `pending_upstream:` 声明的 case，与动态测试层一致；`0` 强制带上 |
| `AIDATA_CONFLICT_GROUPS` | 逗号分隔 mutex group，会写到每个 BatchItem 的 `metadata.conflict_groups` |

## 测试

`tests/` 下两层：

- `test_cases_resolve.py` —— 纯静态，遍历每个 case 跑 `resolve_case` +
  YAML 契约断言（`user_prompt` 非空、`extra_mcp_servers` 合法、
  `services` shape、`metadata` / `rubric.yaml` 合法）。~1 秒。不碰 gongfeng。
- `test_cases_startup.py` —— 动态，每个 case 拿独立 `ExecutionContext`
  + 新 main sandbox，`run_agent` stub 成空，真跑 steps 0-6（case
  resolve → services → bootstrap → MCP load），断言
  `error_message is None`。声明了 `pending_upstream:` 的 case 会以
  该 reason `pytest.skip`。

### 测试前置依赖

aidata 是上游 nexus 仓里的一个 task extension，它有自己声明在
`pyproject.toml` 的 deps（主仓 `nexus-agent` **不会** transitive 装它们）。
跑测试前装上：

```bash
# pytest-xdist：动态层用 -n 8 并行打 gongfeng
uv pip install 'pytest-xdist>=3.6'
```

如果你的 case 数据**不在** `nexus/extensions/tasks/aidata/cases/` 下
（上游 nexus 仓刻意不 bundle case 数据），通过 `AIDATA_CASES_ROOT`
env 把 conftest 指向下游 cases 树即可：

```bash
export AIDATA_CASES_ROOT=/path/to/your/cases
```

### 跑

```bash
# 静态层
uv run pytest nexus/extensions/tasks/aidata/tests/test_cases_resolve.py

# 全量（含动态层），跑在 8 并发 gongfeng worker 上
uv run pytest nexus/extensions/tasks/aidata/tests -n 8 --dist loadgroup

# 只跑一个 case 的动态端到端
uv run pytest 'nexus/extensions/tasks/aidata/tests/test_cases_startup.py::test_case_boots_through_mcp_load[work/poc/email_http_poc]' -v
```

动态层需要 `GONGFENG_API_BASE_URL` / `GONGFENG_TOKEN` 在 `.env`
或环境里；fixture 会 fail-fast 而不是 silent skip（silent skip
对 CI 是最差 failure mode）。`.env` 寻址顺序是
`<cases-root>/../.env` → `<repo>/.env`（前者优先），便于一份
凭据被多套上下游仓共享。

### 推荐 demo case（端到端 trajectory + rubric 评分）

下面四个 case 是从 55 个里挑出来的"最小代表集"，每一个覆盖一类
end-to-end 通路。Reviewer 只想 sanity check 改动没破坏 aidata 跑通能力时，
拿这几个跑就够了 —— 不需要把 55 个全跑一遍（那要 ~30 分钟 gongfeng quota）。

| Case | 覆盖能力 | 时长 | rubric 重点 |
|---|---|---|---|
| `work/poc/dump_state_poc` | sidecar service 容器 + HTTP MCP + `<name>_dump_state` 协议 | ~2 min | `state_equals` 验证 mock 收到的 wecom 邮件字段 |
| `work/poc/email_http_poc` | sidecar service 容器 + HTTP MCP（最简通路） | ~2 min | `tool_called` + `state_equals` |
| `water_emergency` | sidecar mock-server image + 多 mock + LLM-backed 内容生成 + 完整 trajectory | ~10 min | `judge_claim` + `tool_called` + `state_equals` 三种原语都触发 |
| `offboarding_transfer_plus_one` | stdio MCP（`work_mcp_mirror` legacy image）+ v4.2 `run-test.sh` + `report.junit.xml` 协议 | ~30 min | `test_result` 原语（pytest 通过/失败映射成 rubric check） |

跑命令（以 `water_emergency` 为例）：

```bash
AIDATA_CASES_ROOT=/path/to/your/cases \
  ./nexus/extensions/tasks/aidata/aidata.py run water_emergency -v
```

或一次跑前三个验证 main_agent 改造没回归：

```bash
AIDATA_CASES_ROOT=/path/to/your/cases \
AIDATA_CASE_IDS=work/poc/dump_state_poc,work/poc/email_http_poc,water_emergency \
  uv run nexus run-batch \
    -g nexus/extensions/tasks/aidata/generator.py \
    -d /tmp/aidata_demo.sqlite \
    --max-concurrent 3
```

预期产物：

* `task_output.score` ∈ [0, 1]（取决于 LLM 实际表现，但**应该非零** —— 零分通常意味着 step 0–6 失败、agent crashed、或 rubric 路径解析坏）
* `agent_output.main_trajectory` 至少 `len ≥ 3`（system + user + 至少一轮 assistant tool_call），不为空
* `task_output.rubric_results[]` 每条都有 `passed` / `verdict` 字段
* SQLite trace（batch 路径）有完整 `agent.iteration` / `agent.observation` span 序列

如果跑出来分数为 0 / trajectory 空 / rubric_results 空 —— 说明改动引入了
回归。结合 `task_output.error_message` + 该 case 的
`solution/golden_rollout.json`（如有）做对比定位。

## 新增一个 case

```bash
CASE=my_new_case
mkdir -p nexus/extensions/tasks/aidata/cases/$CASE/{data,assets,solution}
$EDITOR nexus/extensions/tasks/aidata/cases/$CASE/task.yaml           # 详见 §case.yaml schema
$EDITOR nexus/extensions/tasks/aidata/cases/$CASE/instruction.md
$EDITOR nexus/extensions/tasks/aidata/cases/$CASE/README.md
cp my_input.pdf nexus/extensions/tasks/aidata/cases/$CASE/data/

# 先做静态检查再跑
./nexus/extensions/tasks/aidata/aidata.py validate $CASE

# 首次运行会自动把 rubric.yaml 生成到 case 目录（v4.2 shape，
# 带 banner 标注 LLM 生成；上线前人工审一遍再去掉 banner）
./nexus/extensions/tasks/aidata/aidata.py run $CASE -v

# 审完生成的 rubric，去掉 auto-gen banner，commit
$EDITOR nexus/extensions/tasks/aidata/cases/$CASE/rubric.yaml
```

如果你更偏好**上线前就把 rubric 手写好**（对于 benchmark case 推荐这么做），直接写好 `rubric.yaml` 即可，task 会直接加载、跳过 builder。**rubric 写法的完整手册见 [`RUBRIC.md`](RUBRIC.md)**。

### task.yaml schema

```yaml
task_id: my_new_case_v1         # 人类可读标签，会出现在 output 里

# 维护信息（created_at 必填；字段字典见 Tasks 仓的 SCHEMA.md §metadata）
metadata:
  created_at: 2026-04-22
  tags: [demo]

agent_timeout: 1800
# 可选；Main-style agents 用它单独限制 observation/tool 总耗时。
# 不配置时只由 agent_timeout 约束整个 agent 调用。
observation_time_timeout: 600
max_iterations: 180
# prompt 可以在下面 inline，或者单独写到 instruction.md
# user_prompt: |
#   用自然语言描述 agent 应该做什么。
context_files:                   # 先在 cases/<id>/data/ 下找（旧 files/ 仍支持）
  - some_input.pdf

# 可选：MCP 加载前在主容器里跑一次 bootstrap。脚本在 cases/<id>/ 下
# （一般放 assets/start.sh）。如果 case 同时声明了下面的 services:，
# 每个 connection 的 env_name 会被注入到这个 bootstrap 的环境变量里
# （例如 $JD_BACKEND_BASE_URL=http://<host>:<port>）。
bootstrap_script: assets/start.sh

# 可选：覆盖 main runtime 镜像。默认 runtime.template.yaml 的
# aio-agent-rc:v0.7-dev-skills；case 依赖别的 layout（work_mcp_mirror
# 的 wecom、aidata-aio-agent:2.x 的 MCP-Bench 全套…）就在这里声明。
# engine 按 image 前缀自动选对应的 MCP_PY / SHELL_PATH
# （engine/paths.py::_IMAGE_LAYOUTS）。
runtime:
  main:
    image: mirrors.tencent.com/aidata_rlenvs/aidata-aio-agent:2.1

# 可选：case 等上游镜像 / mock tag 发布。非空 → 动态 test 用
# pytest.skip 跳过并打印 reason。静态 test 仍跑 manifest 校验。
pending_upstream: "life/online_shopping 还没 bake 进 aio-agent-rc (MockServer MR feat/add-online-shopping)"

# 可选：piggy-back aidata-aio-agent 镜像里自带的 MCP-Bench server。
# 扁平形式（name + command）会被归一化成 {client: {...}} 嵌套形式，
# 见 case/spec.py::normalize_extra_spec 了解 extra_mcp_servers 扁平/嵌套两种写法。
# 字符串里如果出现 services[].connections[].placeholder，会在 MCP 加载前
# 被替换成 service 的真实 URL。
extra_mcp_servers:
  - name: fetch
    command: /opt/mcp-benchmarks/venv/bin/python
    args: [-m, mcp_server_fetch]
  - name: excel
    command: /opt/mcp-benchmarks/py311/bin/excel-mcp-server
    args: [stdio]

# 可选：辅助服务容器（见 docs/mcp-contract.md §"服务容器"）。
# 每一项都是一个独立的 nexus runtime（第二个 docker / 第二个 gongfeng
# sandbox），引擎会在 bootstrap + MCP 加载之前把它们拉起，这样后续逻辑
# 就能引用解析好的公网 URL。
#
# 最简结构如下；完整字段见 engine/models.py::AidataService：
services:
  - name: jd-backend
    runtime_name: service_jd_backend        # 同 case 唯一，不能叫 'main'
    image: "mirrors.tencent.com/aidata_rlenvs/aidata-aio-agent:2.1"
    exposed_ports: [10003]                   # 每个内部端口都会被映射到一个公网 host:port
    startup:
      # command 和 script 二选一。script 会从 case 目录把文件上传到
      # 服务容器里再执行（.sh→bash、.py→python3、其它→依赖 shebang）。
      script: assets/jd_backend_startup.sh
      env:
        NEXUS_TOOLATHLON_WOOCOMMERCE_STORE_PATH: /var/lib/jd_backend/store.json
      timeout_s: 60
    ready_check:
      command: "bash -lc 'curl -fsS http://127.0.0.1:10003/healthz >/dev/null'"
      timeout_s: 10
    ready_timeout_s: 120
    connections:
      - placeholder: __AIDATA_JD_BACKEND_URL__   # 会被替换进 user_prompt 和 extra_mcp_servers
        internal_port: 10003
        value_template: "http://{host}:{port}"   # 可用 token：{host} {port} {base_url} {internal_port}
        env_name: JD_BACKEND_BASE_URL            # 可选：同时导出给 bootstrap_script + bundled mock
```

> **镜像优先级（Path A）**：`runtime_provider_config.image` **故意不设**——一旦设了，gongfeng 的 `image = self.image or merged.get('image')` 会强行压过每个 `configure(runtime_name, image=...)`，per-service image 就失效了。每个 runtime 各自挑自己的镜像：`main` 从 `runtime_configs.<name>.image` 拿，service 从 `task.yaml::services[].image` 拿。详细 rationale 见 [`docs/mcp-contract.md`](docs/mcp-contract.md) §"镜像"。

### workspace system prompt override

如果一个 case 运行在**不同于 aidata 默认办公流**的 runtime 语义下，可以在
`bootstrap_script` 里把一份自定义 prompt 落到主容器的
`/workspace/system_prompt.md`：

```bash
# cases/<id>/assets/start.sh
cp "${CASE_ROOT}/assets/system_prompt.md" /workspace/system_prompt.md
```

语义是：

- 文件**不存在 / 为空 / 不可读**：回退到 `engine/system_prompt.py::SYSTEM_PROMPT`
- 文件**存在且非空**：**整体替换**默认 preamble，不做 prepend / append
- harness 会在 base preamble 后面**继续**追加动态段（环境快照、workspace
  snapshot、`/workspace/skills/*/SKILL.md` frontmatter），所以 override 文件
  只负责"换掉 operating principles 那部分"，不需要自己重新声明 sandbox 工具列表
- 如果调用方启用了 task memory，`/workspace/desktop/MEMORY.md` 的内容会在最末尾
  以一段 XML 的形式追加，不会影响"默认 preamble 是否被替换"的判定

什么时候该用这条能力：

- runtime 级行为契约和默认 aidata prompt 冲突，例如“允许/要求向用户澄清”
- 需要调整不可逆动作策略，例如“先澄清再下单/支付”
- 需要声明跨工具的优先级，例如“存在 `wx_pay_*` 时不要走 provider 原生支付”

什么时候**不要**用：

- 普通任务描述：放在 `instruction.md`
- 文件类型/业务流程指引：放在 `skills/<skill>/SKILL.md`
- 只想补一两条 case 本地提示、但并不需要推翻 aidata 默认 operating principles

### rubric.yaml schema（v4.2）

rubric 现在用**面向标注员的低代码 DSL** 表达，每条 check 明确四件事：
做什么检查（`description`）、属于哪一维（`dimension`）、权重多大（`weight`）、
以及**怎么判**（`assert`）。完整一页见 [`RUBRIC.md`](RUBRIC.md)。

```yaml
version: "4.2"
config:                         # 可选：控制 judge_claim / text_contains 的截断阈值
  trajectory:
    user_head_chars:     16000  # 首条 user message 头部 cap（默认 16000）
    tool_call_arg_chars: 800    # 每条 assistant.tool_call arguments cap（默认 800）
    tool_result_chars:   2000   # 每条 tool_result body cap（默认 2000）
  judge:
    source_max_chars: 96000     # judge 最终拼好的 SOURCE 总 cap（默认 96000）
checks:
  - id: snake_case_unique_id
    description: 检查项的人类可读名字
    dimension: correctness | functional | robustness
    weight: 1
    assert:
      # 五选一。详见 RUBRIC.md §2。
      type: state_equals | text_contains | tool_called | judge_claim | test_result
      # ...各原语对应的字段（如 from / claim / tool_prefix / equals / pattern）
```

总分 = `Σ weight × passed / Σ weight`，并按 `dimension` tag 做维度分拆解。
`config` 块整体可选——省略就用默认值；每个字段也可单独省略。 作用范围与
外部 override 方式见 [`RUBRIC.md`](RUBRIC.md) §5。 旧 v2 rubric
（`verify: llm_judge/eq/regex/shell`）仍能跑，但不再推荐——新 case
一律用 v4.2。

## 沙箱布局

两层结构，详见 `engine/paths.py`。agent 的 shell MCP 的 `ALLOWED_DIR` pin 在 `AGENT_DESKTOP`，所以 `INTERNAL_ROOT` 下的框架内部数据无法被 `cat` 出来绕过 MCP 抽象。

```
/workspace/
├── desktop/                    (AGENT_DESKTOP —— agent 能看到的所有东西)
│   ├── inputs/                   ← 上传进来的 context_files
│   ├── output/                   ← 预建的工作区，让 agent 落产物
│   ├── skills/                   ← 软链到 /workspace/skills（镜像自带）
│   └── <case-public-assets>/     ← case bootstrap 可以往这里写 agent 可见的资源
│                                  （例如 JD API 对接手册 PDF）
└── .aidata/                    (INTERNAL_ROOT —— agent 的 shell MCP 看不到)
    ├── servers/                  ← 自带 mock MCP server 脚本
    ├── case/                     ← 上传的 cases/<id>/ 资源（bootstrap 目标）
    └── state/                    ← 每次运行的 mock 状态持久化
```

## 环境变量

```bash
# Gongfeng sandbox provider
GONGFENG_API_BASE_URL=http://hyrl-sandbox.prod.woa.com:8052
GONGFENG_TOKEN=<your_gongfeng_token>

# Agent + mock LLM（同一套凭据）
SEARCH_OPENAI_BASE_URL=http://llm-api.model-eval.woa.com
SEARCH_OPENAI_API_KEY=<APP_ID>:<APP_KEY>
SEARCH_OPENAI_MODEL=api_naci_default_gemini-3.1-pro-preview

# 可选：judge 想用不同模型
JUDGE_OPENAI_MODEL=api_naci_default_gemini-3.1-pro-preview

# 可选旋钮
NEXUS_MOCK_LLM_MAX_CALLS=50      # 单进程内对 mock LLM 的调用上限
NEXUS_MOCK_SEED=42               # 复现性，默认取 hash(user_prompt)
```

## 结果文件

输出落到 `workspace/mcp/tmp/aidata_<case>_<ts>.result.json`：

| 字段 | 含义 |
|---|---|
| `task_output.score` | `passed_checks / total_checks` |
| `task_output.rubric` | 原样回显 rubric |
| `task_output.rubric_results[]` | `{id, criterion, passed, verdict, judge_raw_head, source, claim, detail}` |
| `task_output.state_summary` | 运行结束时 mock state 的精简快照 |
| `agent_output.main_trajectory` | 完整的 agent 消息日志 |

## 默认加载的 MCP 目录

### 真实 server（镜像自带）

| Server | Tools |
|--------|-------|
| `pdf-tools` | `read_pdf_pages` · `search_pdf_content` · `get_pdf_info` · `extract_pdf_pages` · `merge_pdfs` · `search_pdf_info` · `search_pdf_go_page` · `search_pdf_next_page` · `search_pdf_prev_page` |
| `shell` | `run_command` —— 本地 `mock_servers/shell_server.py`，`bash -c` 薄包装，cwd pin 在 `/workspace/desktop`，PATH 前插了 office-stack venv，16 KiB 输出截断（head+tail 保留策略，优先保留末尾错误信息），对 `rm -rf` / 越界等风险模式通过非阻塞的 `warnings` 字段提示 |

### LLM-backed mock —— 双源（local / mock_server）

`runtime/mcp_catalog.py::_BUNDLED_MOCKS` 每条声明 `source`，决定脚本从哪里来：

- **`source='local'`**：脚本在本仓 `mock_servers/` 下，task 启动时 `sandbox.upload_dir` 推到 `SERVERS_DIR_IN`（沙箱内 `/workspace/.aidata/servers/`），按 `MCP_PY <SERVERS_DIR_IN>/<file>` spawn。**当前所有默认 mock 都走这条路。**
- **`source='mock_server'`**：脚本在 [HunYuanRLEnv/MockServer](https://git.woa.com/HunYuanRLEnv/MockServer) 仓里，由 aidata-aio-agent 镜像 build 阶段通过 `COPY --from=mocks /opt/mock-server/src/mock_server /opt/mock-server` 烤进镜像，运行时按 `MCP_PY /opt/mock-server/<group>/<dir>/server.py` 直接 spawn。**红线**：只有 `work/` group 下经 aidata 团队审批的 mock 能进 `_BUNDLED_MOCKS`；`life/*` 必须留作 case 级 `extra_mcp_servers` opt-in。详情见 [`docs/mcp-contract.md`](docs/mcp-contract.md)。

迁移一个 mock 从 `local` → `mock_server` 三步：(1) 在 MockServer 仓 `work/<name>/` 下落地脚本并 push 出新镜像 tag；(2) 升级 `deployment/docker/Dockerfile.aio-agent-*` 的 `FROM mock-server:vX.Y.Z`；(3) 翻 `_BUNDLED_MOCKS` 那一行的 source + locator。脚本本身在过渡期同时读 `NEXUS_MOCK_*_STATE`（让 `runtime/state.py` 还能 round-trip）和 `MOCK_SERVER_DATA_DIR`（MockServer 的 MockDB 约定）。

| Server | source | Tools | 行为 |
|--------|--------|-------|-----|
| `wechat` | `local` | `wechat_find_contact` · `wechat_add_friend` · `wechat_check_friend_request` · `wechat_send_text` · `wechat_list_contacts` · `wechat_get_latest_reply` | 回复由 LLM 生成；好友通过需要满足 `min-poll-count + wall-clock` 双约束；`get_latest_reply` 可能返回 `null` 强制 agent 轮询 |
| `schedule` | `local` | `schedule_get_calendar` · `schedule_list_free_slots` · `schedule_check_slot` · `schedule_next_available` | 每用户每天的日程首次触及时 LLM 生成，之后 cache |
| `meeting-room` | `local` | `room_list_available` · `room_book` · `room_get_booking` · `room_cancel` · `room_list_bookings` | 库存和初始预订由 LLM 一次性生成，之后做真正的冲突检测 |
| `tencent-meeting` | `local` | `tm_create_meeting` · `tm_link_room` · `tm_join_meeting` · `tm_enable_asr` · `tm_poll_asr_details` · `tm_get_transcripts` · `tm_get_minutes` · `tm_leave_meeting` · `tm_end_meeting` | `tm_enable_asr` 时一次性合成完整会议纪要，之后每次 poll 流式返回约 2 条 |
| `tencent-docs` | `local` | `td_create_excel_by_markdown` · `td_create_smartcanvas_by_markdown` · `td_create_word_by_markdown` · `td_query_space_node` · `td_get_content` | 确定性 markdown 文档存储；返回 `{file_id, url}` |
| `weiyun` | `local` | `weiyun_upload_file` · `weiyun_list_files` · `weiyun_get_share_link` · `weiyun_get_metadata` | 二进制文件云盘；接收 PPTX/XLSX/PDF 产物 |
| `notes` | `local` | 写：`memo_create` · `memo_update_meta` · `memo_add_entry` · `memo_add_entries` · `memo_update_entry`；读：`memo_list` · `memo_get` · `memo_list_entries` · `memo_search`；删/恢复：`memo_delete` · `memo_restore` · `memo_delete_entry` · `memo_restore_entry`；标记：`memo_pin`；其他：`memo_export` · `memo_clear` · `notes_dump_state` | 结构化备忘录 mock：memo + entry 两级 CRUD、tags、JSON Schema 自动校验、软删/恢复、置顶、原子批量写、markdown/csv 导出；状态可通过 `MOCK_SERVER_NOTES_SEED` 注入 |
| `email` | `local` | `email_send` | 确定性邮箱 |
| `yfinance` | `local` | `yf_*` 一组金融数据查询 | yahoo finance mock，研究类 case 用 |
| `web-search` | `local` | `web_search_*` | Serper + Jina 包装的网页搜索 |

case 可以在 `task.yaml` 里声明 `extra_mcp_servers` piggy-back 更多组件，主要两类来源：

- **MCP-Bench 镜像自带**（`fetch` / `excel` / `filesystem` / `playwright_with_chunk` / `wikipedia` / `calculator` / …），绝对路径写 `/opt/mcp-benchmarks/...`。完整示例见 `cases/channel_inventory_ppt/task.yaml`、`cases/mcp_bench_reuse_demo/task.yaml`。
- **MockServer 业务专属 mock**（`life/ride_app` / `life/food_delivery` / `life/weather_app`），绝对路径写 `/opt/mock-server/<group>/<name>/server.py`，env 注入 `MOCK_SERVER_DATA_DIR=/tmp/mock-run/<case-uuid>` 做 per-run 隔离（详见 [MockServer 仓的 docs/integration.md](https://git.woa.com/HunYuanRLEnv/MockServer/blob/main/docs/integration.md)）。

### LibreOffice （headless ） —— 已知限制

LibreOffice 以系统二进制 (`libreoffice --headless`) 的形式预装在沙箱镜像中，但**没有**专属的 MCP tool schema。所有调用走 generic 的 `shell/run_command` 工具。这意味着：

- **无参数校验** —— agent 必须从 SKILL.md 中构造正确的 CLI 语法。
- **无结构化输出** —— 错误以 raw stderr 返回，agent 必须自行解析。
- **无重试/降级逻辑** —— 与 `pdf-tools`（有 typed error）不同，LibreOffice 失败表现为非零 exit code + 非结构化文本。

未来改进方向：将常用的 LibreOffice 操作（`convert-to`、`--calc recalc`、`macro://` 执行）包装成专属的 `office-tools` MCP server（typed schema），类似 `pdf-tools`。

## Rubric source key

运行结束后，engine 会在关闭 MCP 前调用所有 `*_dump_state` 工具并合并返回的 JSON。每个 mock 的结果会先按 MCP `server_name` 放入独立 namespace，例如 `state['wechat']` / `state['cron']` / `state['xhs']`；同时保留旧的扁平 source key 作为兼容层。

新 rubric 推荐优先读 namespace，避免多个 MCP 都返回 `orders` / `messages` / `actions` 时互相污染：

```yaml
assert:
  type: state_equals
  from: xhs
  path: actions.0.type
  equals: publish_note
```

下面这些扁平 source key 仍可使用，但主要用于已有 case 兼容：

| Source key | 来源 |
|------------|------|
| `wechat_log` | 每条聊天 + 所有好友请求事件 |
| `wechat_friends` | 每个联系人的好友请求生命周期（`status / poll_count / timestamps`） |
| `schedule` | agent 查询过的所有日历天 |
| `meeting_log` | `create_meeting`、`link_room`、`join_meeting`、`enable_asr`、`poll_asr`、`leave_meeting`、`end_meeting` |
| `room_log` | `book` / `cancel` 事件 |
| `memos` | 所有 memo 和 entry |
| `emails` | 所有发出的邮件 |
| `trajectory` | 初始 user message（默认截到 ~32k chars）+ 每条 assistant message + 每个工具调用（args 默认 400 chars）+ 工具返回（默认 500 chars）—— 上限可通过 `rubric.yaml::config` 调，见 [`RUBRIC.md`](RUBRIC.md) §5 |
| `docs` / `docs:<doc_type>` | 上传的 markdown 可编辑文档的完整内容（`excel` / `smartcanvas` / `word`） |
| `weiyun_uploads` | 每次二进制上传（`file_id, title, filename, ext, size, share_url, description`） |
| `all` | 以上全部拼接 |

## Visualizer

```bash
uv run python -m nexus.extensions.tasks.aidata.viz.app
uv run python -m nexus.extensions.tasks.aidata.viz.app --unsafe-public --host 0.0.0.0 --port 7860
```

标签页：Environment（MCP 目录）、Task（case + 已保存 rollout）、Playground（实时流式跑一次）。并发跑数由 `NEXUS_VIZ_MAX_CONCURRENCY`（默认 8）限制。临时 YAML 写在 `$TMPDIR/nexus_aidata_viz_yaml/`，权限 0700。
