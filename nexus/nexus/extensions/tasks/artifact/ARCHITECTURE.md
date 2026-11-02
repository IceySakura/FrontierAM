# artifact task 架构

当前配置及跑数方式看 `CLAUDE.md` §4。 这份文档讲代码结构：谁调用谁、每个文件干什么、当前有哪些结构问题。

本地跑数统一经 `run_harbor_test.sh` → Harbor Dataset + Compose → `harbor_bridge_task` → 容器 verifier。旧 `generator/` 及其 JSONL 启动脚本已退役；下文宿主侧 `artifact_task` 结构仅描述保留的兼容实现，不再是本地跑批入口。

统计口径：约 16,000 行 Python，53 个文件（不含 `__pycache__`），采样于 2026-09-15。

## 1. 一句话概括

一条数据进来，先让 agent 在沙箱里生成前端项目（infer），再部署并打分（eval），最后输出 RL 需要的 reward 契约。

**两个判分入口，同一套判分代码。**

```
宿主侧（离线批量）
  RunInput ──► task.py::artifact_task
                 │
                 ├─ infer: harness → agent 写代码 → collect_project_archive → archive_b64
                 │
                 ├─ 短路检查: 判分配置错误 / 空产物 → maskout 或 penalty
                 │
                 └─ eval: run_judge(judge_name) → JudgeResult
                            │
                            └─ resolve_eval_outcome → reward_protocol → score + judge_code

容器侧（T3-Harbor，当前 RL 路径）
  tests/test.sh ──► harbor_entry.py
                      │
                      ├─ T3_JUDGE_NAME (from [verifier.env]) → 判哪个 judge
                      ├─ tests/instruction.md → problem_statement
                      ├─ judge 自己读自己的参数（web_judger → /tests/flow.json）
                      │
                      └─ run_judge(judge_name) → JudgeResult
                                 │
                                 └─ verdict_to_reward → /logs/verifier/reward.json
```

两条路径共用 `run_judge`，所以一个 judge 在两边的行为不可能不同。差别只在**外围**：宿主侧走 `runtime.invoke`（socket.io），容器侧用 `LocalRuntime` + 直连网关（见 §6）。

**`harbor_entry` 不感知任何具体 judge。** 它只组装通用事实：分派键（`T3_JUDGE_NAME`）、需求（`/tests/instruction.md`）、instance_id、workdir、收集好的 archive。**判分参数一个都不读。**

参数的**位置和格式都归 judge 自己**：`web_judger.HARBOR_FLOW_PATHS = ('/tests/flow.json',)`，它自己读、自己校验、读不出来自己报真因。所以 `test.sh` 对每个任务都一样，换 judge 只改 `T3_JUDGE_NAME`。

一个通用入口一旦要"认识"某种契约，下一个 judge 就会要求它再认识一种。

## 2. 目录职责

配置集中在 `configs/`：`llm/` 放当前模型 YAML，`legacy/` 收纳旧 ArtifactCfg 模板、本地配置与样例；后者不由 Harbor 自动加载。`legacy/.gitignore` 忽略本地 `*.yaml` / `*.yml`，模板和样例可跟踪但不应含真实凭据。原顶层 `profiles/` 已移除。`judges/profiles/` 仍是模型加载代码，位置和调用不变。

行数为该目录下 `.py` 的实际行数。

```
task.py                1077  宿主侧编排 infer → 短路 → eval → RL 契约
harbor_entry.py         619  容器侧判分入口（Harbor verifier 调它）
config.py              1483  所有 ARTIFACT_* 的单一真相源（profile + env + provenance + validate）
reward_protocol.py      516  RL 契约：judge_code / training_action / zero_reason，自包含

run_harbor_test.sh            本地跑数：Harbor Task 目录 → Dataset + Compose

agent_setup/           3341  「沙箱里怎么起 agent」，保留的共享 harness 与 Judge 配置
  harness/              337  infer agent 启动配置的注册表（claude_code / hycode / hycode_dynamicagent）
  clients.py                历史 M_* → 模型 spec 工厂；保留 CLI、包导入和旧字符串回调依赖
  claude_config.py          claude_code 结构常量 + default_judge_claude_input 工厂
  claude_hooks.py           hooks / plugins / skills 注入脚本（base64 打包进 pre_start_scripts）
  askuser_hook{,_v2}.py     沙箱里回答 AskUserQuestion 的 hook（按文件名读取后注入）
  stop_hook.py              防早停 hook
  cos_io.py                 archive 上传 COS

judges/                 919  判分层
  registry.py           592  唯一的派发层：register_judge / get_judge / run_judge
  base.py               295  JudgeInput / JudgeResult / JudgeFailureKind
  capabilities/        2154  原子能力：judge 自己按需调用
  strategies/                判分策略，每个文件一个 judge（+ web_judger 子包 3350）
  prompts/                  prompt 模板（load_prompt 按文件名加载）
```

## 3. judges 两层：registry + strategies

**只有两层。** 复合判分在自己的 `compute_score` 里调 `get_judge('sub').fn(inp)` —— 编排就是普通 Python 代码，不需要单独的层。

### 3.1 registry：唯一派发层

```python
# judges/strategies/<name>.py
@register_judge('<name>', requires=('vlm',), capabilities=('screenshot',),
                model_files={'vlm': 'gemini-3.8-flash-aihub-2.0.yaml'})
async def compute_score(inp: JudgeInput) -> JudgeResult:
    ...
```

`run_judge(name, inp)` 的 9 步，顺序是设计出来的：

| # | 步骤 | 为什么在这个位置 |
|---|------|------------------|
| 1 | 解析 judge | 名字错 → `config_error`，不花任何资源 |
| 2 | **`validate` 钩子** | 数据错要在**部署之前**拒。放到 `compute_score` 里的话，部署失败会先短路，数据错就被记成产物错 |
| 3 | client 角色 | 缺 client → `no_client`，同样在部署前 |
| 4 | capabilities | `deploy` / `screenshot`，真正花钱的一步 |
| 5 | 跑 judge | |
| 6 | `JudgeExecutionError` | judge 主动声明的结构化失败 |
| 7 | 其他异常 | → `internal_error`（判分代码有 bug，屏蔽而非计分） |
| 8 | 返回值校验 | 不是 `JudgeResult` → `internal_error` |
| 9 | name 不变式 | 返回的 `judge_name` 必须与派发名一致 |

**复合 judge 故意绕过 wrapper**（直接 `get_judge('x').fn(inp)`）：资源在最外层准备一次，且复合自己决定子项失败如何聚合。这是 pipeline 层不再必要的原因。

### 3.2 已注册的 8 个 judge

| judge | requires | capabilities | 说明 |
|-------|----------|--------------|------|
| `web_judger` | agent, vlm | deploy | flow 契约 → flow2code → 覆盖率门控 → 逐 flow 判定 + 美学 |
| `aesthetic_general` | agent, vlm | screenshot | 复合：美学分门控，通过才跑 `general_agent` |
| `general_agent` | agent | deploy | agent 交互式评测（rubric / web_app） |
| `baseline_check` | agent | deploy | 基线对比 |
| `aesthetic` | vlm | screenshot | 美学维度打分 |
| `svg` | vlm | screenshot | SVG 渲染判定 |
| `visual_anchored` | vlm | screenshot | 与参考图 pairwise 对比 |
| `code_quality` | agent | — | 纯源码审查，不需要部署 |

角色按调用方式划分：`agent` 经 Agent 工具循环，`vlm` 直接调用模型判断（可按检查项多次调用和重试）。各 Judge 独立绑定模型，缺角色不回退到 `main`；`flow2code` 等仍为业务阶段名，不再用作模型角色。

前两个是复合（自己调子 judge），后六个是原子。`aesthetic_general` 的门控语义：门控腿不通过时**只取门控腿的分**，不与缺失的第二条腿求平均 —— 后者会把「没跑」当成「0 分」。

## 4. capabilities：原子能力层

judge 按需调用，不构成层级。

| 文件 | 干什么 |
|------|--------|
| `deploy.py` | archive 打包/解码、Render Service 部署、`SandboxContext`、截图落盘 |
| `screenshot_in_container.py` | 容器内 playwright 截图 |
| `llm.py` | VLM 调用 + 重试（`call_judge_llm`） |
| `agent.py` | 起 claude-code 读 reward 文件（宿主侧走 `runtime.invoke`） |
| `agent_direct_gateway.py` | 容器侧直连版：`LLMHandlerBridge` + `run_claude_code_with_bridge` |
| `parse.py` | 宽松 JSON 解析（`loads_json_lenient`） |
| `prompts.py` | prompt 加载 + 占位符渲染（`load_prompt` / `render_prompt`） |

## 5. 部署：Render Service

`prepare_sandbox_context(archive_b64)` = COS 上传 → `POST /render` → 轮询。

**两个 URL 语义（实测过，不是推测）**：

| 字段 | 沙箱内可达 | 用途 |
|------|-----------|------|
| `frontend_url` | ✅ HTTP 200 | 判分用这个 |
| `frontend_public_url` | ❌ DNS 解析但 TCP 失败 | 人工看 |

反直觉但已验证：judge 跑在沙箱里，必须用**内网** URL。

## 6. 两种 LLM 送达方式，一条代码路径

`capabilities/agent.py::run_claude_code_on_prepared_runtime` 是**唯一**的 agent 执行函数。它内部只在一处分叉——agent 的 LLM 请求怎么到达我们的 client：

| | agent 在哪 | 请求怎么来 |
|---|---|---|
| 远端 runtime | 另一个进程 | socket.io → `runtime.on('llm_request')` → `LLMRequestHandler` |
| `LocalRuntime` | **本进程** | 本地 `AnthropicProxy` → `LLMHandlerBridge` → 同一个 handler |

两者都经 `LLMRequestHandler._produce_response` 作答，所以 sanitize / preserve_thinking / reasoning items / 模型覆盖的行为完全一致。其余全部共用：清 reward file、`pre_run_setup`、trajectory 采集、读判决、日志。

判定用**能力探测**而非类型判断（`runtime_supports_events`）：`LocalRuntime` 有 `on()` 但抛 `NotImplementedError`，所以 `hasattr` 答不了；而 `isinstance(LocalRuntime)` 是封闭假设，任何其他无事件 runtime 都会被误判，错误还会从 `handler.register()` 深处冒出来。探测就是注册一个空回调再立刻摘掉。

两条路径都转发 `claude_input.model_dump()` **全量**，所以 `ClaudeCodeInput` 加字段时两边同时生效。用 `getattr` 逐字段挑会静默漏项 —— skill 没装、代理 env 丢失、MCP 未注册都属于这类，而漏掉的每一项都让 judge 在少了它索要的东西的情况下照样打分。

## 7. RL 失败分类

`JudgeFailureKind` 决定这次 rollout 进不进训练：

```
EVALUATOR_FAULT_KINDS = {no_client, no_sandbox, vlm_failed,
                         parse_failed, agent_failed, config_error, internal_error}
  → Harbor: reward = -1 → harbor_bridge 抛错 → rollout 丢弃

no_artifact（不在上面那组里）
  → reward = 0.0 → 留在训练里（空产物是模型的问题）
```

**失败优先于分数。** 顶层 `failure_kind` 与分数无关地判定失败：复合 judge 可以在某条腿故障时仍返回非零均分（视觉 0.8 + general `parse_failed` → 0.4），那 0.4 掺了一半评测故障，不是有效成绩。

**只看顶层，零分也一样。** 子项 `child_results` 只做诊断，不参与 reward 分类 —— 否则「顶层判了合法零分、某条可选腿故障」会在 task 侧被屏蔽、在 Harbor 侧保留，同一结果两端语义相反。

## 8. 加东西怎么加

| 要做什么 | 怎么做 |
|----------|--------|
| 加一个 judge | `judges/strategies/<name>.py` + `@register_judge`，名字必须等于文件名（lazy import 靠这个约定） |
| 加一个复合 judge | 同上，在 `compute_score` 里 `get_judge('sub').fn(inp)`，自己聚合 |
| 加一个判前检查 | `@register_judge(validate=...)`，同步函数，返回原因字符串即拒 |
| 加一个能力 | `judges/capabilities/`，然后在 `_CAPABILITY_TOKENS` 注册 token |
| Harbor 任务换 judge | 改 `task.toml` 的 `[verifier.env] T3_JUDGE_NAME` |
| 换判分模型绑定 | Judge 注册的 `model_files` + `configs/llm/`；对照实验用 `T3_JUDGE_MODELS`，见 `CLAUDE.md` §4 |

## 9. 当前的结构问题

- **`config.py` 1483 行**：9 组 60+ 字段 + provenance + validate 在一个文件里。改一个字段要翻很久。
- **`task.py` 1077 行**：infer / 短路 / eval / 输出构造都在 `artifact_task` 一个函数的流程里，局部逻辑已抽成 `resolve_eval_outcome` / `_finalize_eval` / `_maskout_output`，但主流程仍长。
- **`web_judger`**：`executor.py` + `verifier.py` 是正式采证和判分核心；保留 `flow.py` / `run_1_flow.sh` 用于离线契约生成。`report.py`、`redeploy.py`、`selfcheck.py` 及旧 JSONL 跑批、报告脚本均已退役，正式判分不依赖它们。
- **`agent_setup/stop_hook.py` 1500+ 行**：注入沙箱的单文件脚本，因为要独立运行所以不能拆模块。
