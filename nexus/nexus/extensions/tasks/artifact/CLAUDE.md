# Artifact Task — 开发指引

本文档面向后续在 `nexus/extensions/tasks/artifact/` 下继续开发的同学，说明目录结构、核心抽象、常见扩展方式与历史踩坑。

**当前跑数入口只有两个（见 §4）：`run_harbor_test.sh` 调试单题，`run_harbor_rollouts.sh` 跑 RL 数据，都走 Harbor Dataset + Compose。** `generator/`、`run.sh`、`smoke_profile.sh`、`smoke_judge.sh` 和 Web 的 `run_2_judge.sh` 已退役。旧 JSONL / infer DB 回填及旧 `ArtifactCfg` profiles 的说明仅供历史排查，不是当前支持的启动用法；共享 `agent_setup`、Judge 和旧 `artifact_task` 注册暂保留。

## 1. 目录结构

配置位置约定：
- `configs/llm/`：当前 Judge 模型 YAML。
- `configs/llm/hy4-preview-offline-taiji.yaml`：Artifact Harbor inference 使用的太极
  `hy4-preview-offline` 模板；通过 `run_harbor_rollouts.sh` 选用，不改变 Judge 模型绑定。
- `configs/legacy/`：旧 ArtifactCfg 模板、本地配置与 `sample_dataset.jsonl` 集中在此，仅供历史参考，不由 Harbor 自动加载。本地 `*.yaml` / `*.yml` 由目录内 `.gitignore` 保护；`.yaml.example` 和样例可跟踪，不能填写真实 key。不要强制提交个人配置或把整个目录当作公开附件上传。
- `judges/profiles/`：模型加载与预检的 Python 代码，不是配置目录，保持原位。
- 原顶层 `profiles/` 已清空移除；旧配置路径需改为 `configs/legacy/`，原有本地 YAML 仅同步样例文件路径，其他内容不变。

目录外 `test_artifact_config.py::test_shipped_profile_examples_have_no_unknown_keys` 仍扫描旧模板路径，需由维护者迁移为 `configs/legacy/`；本轮不改目录外测试。

```
artifact/
├── task.py                       # ArtifactTaskInput/Output + run_artifact_task（纵览入口；archive-only；注册入 task_registry）
├── config.py                     # ★所有 ARTIFACT_* env 的单一解析入口（判分/部署/容器/观测各段默认值都在这）
├── reward_protocol.py            # RL 契约（自包含，不依赖 frontend）：错误码 / training_action / build_reward_extra_info
├── pyproject.toml                # 模块元数据，声明注册点与依赖
├── __init__.py                   # 触发 task 注册
├── harbor_entry.py               # ★容器侧判分入口（T3-Harbor 的 tests/test.sh 调它）
├── agent_setup/                  # 「沙箱里跑 claude_code 怎么配」的单一真相源（generator + RL + judge 共用）
│   ├── claude_config.py          # ★调 claude_code env/参数就看这个: CLAUDE_CODE_COMMON_ENV/max-turns/timeout/MCP版本/chromium参数/星代理/CLI flags + judge claude_input 工厂
│   ├── claude_hooks.py           # opt-in 脚手架: hooks(含 Stop/AskUserQuestion v2) + plugins + skills + PLAYWRIGHT_SKILL_DIR（默认全关，基本不碰）
│   ├── clients.py                # 历史 M_* 模型 spec 工厂；配置 CLI / 包初始化 / 旧回调仍有依赖，Harbor 不用它构造 client
│   ├── harness/                  # infer agent 启动配置注册表（claude_code / hycode / hycode_dynamicagent）
│   ├── cos_io.py                 # archive 上传 COS
│   └── askuser_hook{,_v2}.py / stop_hook.py   # 注入沙箱的 hook 脚本（claude_hooks 按文件名读取后 base64 注入）
├── run_harbor_test.sh             # 单题/小批量调试：Harbor Task 目录 → Dataset + Compose
├── run_harbor_rollouts.sh         # ★RL 跑数：rollout 展开 + 分片 + resume（配置全在 configs/）
└── judges/                       # ★判分层，只有两层：registry 派发 + strategies 实现
    ├── registry.py               # ★唯一派发层: register_judge / get_judge / run_judge / resolve_judge_name
    ├── base.py                   # JudgeInput / JudgeResult / JudgeFailureKind / JudgeExecutionError
    ├── capabilities/             # 原子能力层（judge 按需调用，不构成层级）
    │   ├── deploy.py             # archive 打包解码 + Render Service 部署 + SandboxContext + 截图落盘
    │   ├── agent.py              # run_claude_code_on_(prepared_)runtime + reward file + runtime 生命周期
    │   ├── agent_direct_gateway.py    # ★容器侧直连版（LocalRuntime 没有 runtime.on，必须绕开 socket.io）
    │   ├── screenshot_in_container.py # 容器内 playwright 截图（★唯一截图路径）
    │   ├── llm.py                # call_judge_llm + retry（VLM 调用）
    │   ├── parse.py              # parse_judge_xml / sum_tag_scores / loads_json_lenient
    │   └── prompts.py            # ★load_prompt / render_prompt（所有 judge prompt 的唯一读盘入口）
    ├── prompts/                  # ★全部 prompt 文本集中在这里（可被 config.prompt 覆盖）
    │   ├── judge_general_agent.txt / judge_general_agent_strict_{web,3d,game,svg}.txt
    │   ├── judge_aesthetic.txt / judge_svg.txt / judge_visual_anchored.txt
    │   ├── judge_static_code_review.txt / judge_baseline_check.txt
    │   └── webjudger_{gen_flow,flow2code,func_judge,aesthetic*}.md   # web_judger 各段，带前缀区分
    └── strategies/               # 每个文件一个 judge，文件名 = 注册名（lazy import 靠这个约定）
        ├── web_judger/           # ★主力，自己一个目录
        │   ├── __init__.py       # compute_score + _validate_input（flow 契约/模式校验，部署前拒）
        │   ├── executor.py       # 契约 → Playwright/bash 代码 + 容器内 harness 采证
        │   ├── verifier.py       # 逐 flow pass/fail/inconclusive + 五维美学 + tier 加权合成
        │   ├── flow.py           # 需求 → flow 契约（离线跑，写死直连网关）
        │   └── run_1_flow.sh     # 离线契约生成，输出须另行整理为 Harbor 任务
        ├── aesthetic_general.py  # 复合: 美学门控 → 通过才跑 general_agent
        ├── general_agent.py      # agent 浏览真实页面 → XML 四维（strict prompt 按 task_type 路由）
        ├── aesthetic.py          # 截图 + VLM 5 维美学评分
        ├── baseline_check.py     # agent 快检底线缺陷
        ├── code_quality.py       # agent 读源码 → implementation + engineering（无需部署）
        ├── visual_anchored.py    # VLM pairwise vs anchor 截图（win/tie/lose）
        └── svg.py                # Scrutiny 风格：structure + aesthetics 1-5
```

**当前分工**：Harbor Dataset + Compose 负责数据与运行配置；`agent_setup/` 保留共享的 agent 启动能力；`judges/` 负责派发、策略及原子能力。

复合判分在自己的 `compute_score` 里调 `get_judge('sub').fn(inp)` —— 编排就是普通 Python 代码，没有单独的编排层。

**两个判分入口，同一套判分代码**：
- `task.py::artifact_task` —— 宿主侧，串 infer → 短路 → eval
- `harbor_entry.py` —— 容器侧，T3-Harbor verifier 调它（**当前 RL 路径**）

两者都经 `judges/registry.py::run_judge` 派发，所以一个 judge 在两边的行为不可能不同。差别只在外围：宿主侧走 `runtime.invoke`(socket.io)，容器侧用 `LocalRuntime` + 直连网关。

## 2. 核心抽象

整条流水线分两步：**Infer**（一个 runtime 里让 agent 从零生成前端项目，收成 tar.gz archive）→ **Eval**（新的 runtime 里部署 archive、跑一堆 judge 并聚合分数 + 出 RL 契约）。Infer/Eval 可单独跳过，便于用一份 infer DB 跑多种 judge 组合。**已去掉 git patch 模式，只走 archive。**

### 2.0 历史 artifact_task 链路（旧 run-batch 入口已退役）

以下记录旧 JSONL 跑批和旧 RL 的构造方式，仅用于排查历史数据。新 Harbor 本地跑数由 Dataset + Compose 构造 `harbor_bridge_task`，并不在 `run_artifact_task` 汇合；见 §4。

**唯一契约是 `RunInput`**：generator 拼一个 dict → `RunInput.model_validate` → 就是 task 消费的东西。改 `task.py` 时要保证 generator（离线）和 RL jsonl 两条路造出的 RunInput 都能满足 task 的字段契约。

**离线批：`nexus run-batch -g generator/generator.py`**（或 `run.sh -p <profile>`）

```
nexus run-batch -g generator/generator.py -d out.db --max-concurrent 20 --export-path results.jsonl
  │
  ▼ nexus/cli/commands/run.py::run_batch          # CLI 解析,校验 -g 是 .py
  ▼ nexus/runner/batch/runner.py::run_batch       # 批调度总入口,两阶段:
  │   PHASE 1 生成配置:
  │     GeneratorLoader(path)  (batch/loader.py)  # importlib 动态 load 你的 generator.py
  │     loader.generate() → 迭代你的 generate_configs() yield 的 BatchItem
  │     每个 BatchItem = {config: RunInput, run_id, metadata}
  │     db.insert_run(run_id, item.config.model_dump(), metadata)   # ★配置落 sqlite
  │   PHASE 2 执行:
  │     db.get_all_runs() → RunInput.from_dict(config)  # 从 DB 读回(解 ${oc.env:X} 插值)
  │     并发(max_concurrent 信号量) run_single(config) + 写结果 + export_fn 追加 results.jsonl
  ▼ nexus/runner/single.py::run(RunInput)          # 建 ExecutionContext,按 task_name 找 task fn
  ▼ task.py::run_artifact_task(task_input)         # ← 终点
```

要点:
- **generator 只是「配置工厂」**,yield `BatchItem(config=RunInput(...))`,不执行任何 task。你在 `generate_configs` 拼的 dict `model_validate` 成 RunInput 就完事。
- **配置先落 DB 再执行**(两阶段)。所以 `-d xxx.db` 已有配置时可**不带 `-g` 直接重跑**;`--retry-failed`/`--resume` 都基于 DB 里的 status。
- **generator 契约**(见 `batch/loader.py`): `generate_configs()→Iterator[BatchItem]`(必需)、`get_total_count()→int`(可选,进度条)、`get_batch_metadata()→dict`(可选)。
- 验证重构时用 `bi.config.model_dump()` 逐字节对比,因为那正是落库、也是 task 消费的东西。

**RL：hytuner（不经过 generator / 不经过 batch 层）**

```
hytuner 项目 jsonl(每行手写一个 nexus dict) → convert_jsonl_to_parquet.py(原样序列化)
  ▼ hytuner 训练循环(veRL+Ray)
  ▼ nexus/runner/hytuner.py::run(run_input)   # ★强制覆盖 openai_clients['main']=训练中的 policy 模型
  ▼ nexus/runner/single.py::run(run_input)    # 和离线同一个下游
  ▼ task.py::run_artifact_task                # ← 同一个终点
```

RL 跳过整个 batch 层(PHASE 1/2)和 generator,自己用 parquet 喂 RunInput。**跨两条路真正共享的只有 `agent_setup/claude_config.py` 的 `default_judge_claude_input`**(judge config 的 `default_factory`,RL 的 judge dict 不写 claude_input 时靠它兜底)和 `agent_setup/clients.py`。**改 task.py 消费的字段/契约时,两条构造路径都要兼容。**



- `ArtifactTaskInput`：从 dataset JSONL 读取 `problem_statement`/`image`/`judges` 等，支持 `skip_infer + archive_b64`（基于已有 infer DB 做纯 eval）。
- `ArtifactTaskOutput`：`score` + `judge_results`（每个 judge 的 `JudgeResult.model_dump()`）+ `archive_b64` / `file_list` + `infer_trajectories` + `infer_metadata` + `extra_fields`（承载 `reward_extra_info`：training_action / judge metrics）+ `termination_reason` / `zero_reason`。
- `artifact_task`：infer → 收 archive → **判分配置校验**（`judge_name` 缺失 → `config_error`，排在产物短路**之前**，否则空产物会先返回产物惩罚把配置错误掩盖掉）→ **空/不完全产物短路** → eval：起 eval 容器 → `run_judge(judge_name, inp)` 拿单个 `JudgeResult` → `resolve_eval_outcome` → `_finalize_eval` 按 `zero_reason` 映射 RL 契约（use_score / maskout）。**judge 自己决定跑哪些子 judge、如何聚合，task.py 不做编排。**

### 2.2 Judge 抽象（`judges/base.py`）

```python
class BaseJudge(ABC):
    async def run(
        self, config: JudgeConfig, image: str, archive_b64: str,
        working_dir: str, openai_client=None, sandbox_ctx=None, **kwargs,
    ) -> JudgeResult: ...
```

- `JudgeConfig`：所有 judge 共享 `name` / `weight` / `required` 三个字段；具体 judge 继承并加自己的字段（e.g. `GeneralAgentJudgeConfig.claude_input`、`CodeQualityJudgeConfig.score_max`）。
- `JudgeResult.score`：统一归一到 `[0, 1]`。由外层 `aggregate_judge_results` 用 `weighted` / `mean` / `min` 合成最终分。
- `required=True` 的 judge 若失败或 < 1.0，整个实例直接 0 分。
- `archive_b64` 为产物入口；judge 用 `if not archive_b64: return error` 门控。
- `sandbox_ctx` 由 task 层预部署并注入，judge **不应**再自己部署（历史教训：每个 judge 各自部署会让 claude-code 启动时间爆炸）。

### 2.3 Capabilities 原子能力层（`judges/capabilities/`）

**这是开发新 judge 时最该熟悉的目录。** 所有跨 judge 的公共逻辑按关注点拆分，写新 judge 应尽量复用而不是重写。strategy 从具体子模块显式 import（如 `from ...capabilities.llm import call_judge_llm`）。

**client 解析不在这里**：`run_judge` 在跑 judge 之前校验声明的 `requires` 角色（`agent` / `vlm`），缺失直接返回 `no_client`，不回落 `main`。`call_judge_agent` 默认用 `agent`，`call_judge_llm` 默认用 `vlm`；策略仍可显式传相同角色。

| 函数 | 所在模块 | 作用 |
|---|---|---|
| `prepare_sandbox_context(archive_b64)` | `deploy` | 上传 COS → Render Service `POST /render` + 轮询 → `SandboxContext`（只带 url,不截图）|
| `archive_to_code_content_dict(archive_b64)` | `deploy` | base64 tar.gz → `{path: content}` |
| `collect_project_archive(runtime, dir)` | `deploy` | infer 阶段把 working_dir 收成 tar.gz |
| `upload_archive_to_runtime(runtime, archive_b64, dir)` | `deploy` | 把 archive 解回文件写入 runtime（code review 用） |
| `sandbox_context_from_screenshot(url, shot_b64)` | `deploy` | 把容器内截图包成 `SandboxContext` 喂 judge（复合 judge 给子 judge 传图用）|
| `capture_screenshot_in_container(runtime, url)` | `screenshot_in_container` | 在 eval 容器内 xvfb+playwright 截一张全页图（★唯一截图路径）|
| `run_claude_code_on_runtime(image, claude_input, ...)` | `agent` | 起 runtime → 跑 claude-code → 抽 trajectories + 结果文件 |
| `run_agent_and_read_reward(...)` / `read_reward_file` / `release_runtime` | `agent` | 跑 agent + 读 reward + 保证释放 runtime |
| `call_judge_llm(inp, role=, prompt=, images=, max_attempts=)` | `llm` | ★VLM/LLM 调用入口（按 role 解析 client + 重试）|
| `parse_judge_xml` / `sum_tag_scores` / `loads_json_lenient` | `parse` | 从 `<judge>` XML 抽分（`sum_tag_scores` 支持 per-tag 权重）/ 宽容 JSON 解析 |
| `load_prompt` / `render_prompt` | `prompts` | 载入默认 prompt / 安全渲染占位（不破坏字面 `{...}`）|

**设计意图**：strategy 只负责「拼 prompt / 组装 config / 解释输出」这种策略性逻辑；脏活（HTTP、重试、XML 解析、截图、runtime 生命周期）一律下沉到 capabilities。

### 2.4 Judge 注册（`judges/registry.py`）

**没有中心字典。** judge 按名字 lazy import：`get_judge('svg')` 找 `strategies/svg.py`（或 `strategies/svg/__init__.py`），所以**文件名必须等于注册名**。

```python
# judges/strategies/my_judge.py
@register_judge('my_judge', requires=('vlm',), capabilities=('screenshot',), validate=_check)
async def compute_score(inp: JudgeInput) -> JudgeResult:
    ...
```

`run_judge(name, inp)` 的顺序是设计出来的，尤其前三步：

1. 解析 judge（名字错 → `config_error`，不花资源）
2. **`validate` 钩子** —— 数据错在**部署之前**拒。放到 `compute_score` 里的话，部署失败会先短路，数据错就被记成产物错，一个假惩罚进训练
3. client 角色（缺 → `no_client`，同样在部署前）
4. capabilities（`deploy` / `screenshot`，真正花钱的一步）
5. 跑 judge → 异常映射成结构化 `failure_kind` → 校验返回值

**复合 judge 故意绕过 wrapper**：在自己的 `compute_score` 里 `get_judge('sub').fn(inp)`（注意 `.fn`，不是 `run_judge`）。资源在最外层准备一次，且复合自己决定子项失败如何聚合。这是 pipeline 层不再必要的原因。

### 2.5 默认配置（`agent_setup/claude_config.py`）
- `sandbox.STAR_PROXY_ENV`：**沙箱跑 `@playwright/mcp` 必须透传代理**，否则握手挂 30 秒失败。
- `sandbox.DEFAULT_PLAYWRIGHT_MCP`：钉死 `@playwright/mcp@0.0.68` + `chromium-1208` 可执行路径（和 `webenv.claudecode.testset:v6-workspace` 镜像对齐）。**不要轻易 bump 版本**；chromium revision 和 playwright 有紧耦合。
- `sandbox.DEFAULT_JUDGE_MAX_TURNS` / `DEFAULT_JUDGE_TIMEOUT` / `DEFAULT_CLAUDE_CLI_ARGS`：judge 的 max-turns(200)/超时/CLI flags。改这里 judge 全局生效。
- `claude_config.default_judge_claude_input(with_playwright=?, with_skill=?, extra_skill_dirs=?)`：agent 型 judge 用它拼启动配置。**`extra_skill_dirs` 是参数不是字段** —— 工厂把它消化成 `pre_start_scripts` 里一段 base64 安装脚本，所以读 `claude_input.extra_skill_dirs` 永远是空的 —— 要 skill 就转发 `pre_start_scripts`。`with_skill=True` 额外把 `playwright-cli` skill 装进沙箱（agent 能 `Skill(playwright-cli)`）。
- **judge claude_input 由 `claude_config.default_judge_claude_input` 造**（agent 型 judge 自己调；infer agent_config 则在 `generator._infer_agent_config` inline 拼）。改 claude_code 的 env/配置值统一改 `agent_setup/claude_config.py`（`CLAUDE_CODE_COMMON_ENV` 等，infer+judge 共用）。**dataset 不覆盖这些配置**——judge dict 只提供 `name`/`openai_client_name`/`rubrics`。

**Agent judge 的浏览能力（skill + MCP）**：agent 型 judge（`general_agent` / `baseline_check`）的 prompt 让 agent「加载 `playwright-cli` 技能再浏览」。**judge 通过 `default_judge_claude_input(with_skill=True)` 注入 pinned playwright MCP + 把仓库内 `skills/playwright-cli/SKILL.md` 写进 runtime**（`agent_setup.claude_hooks.build_skills_install_script` 打成 base64 塞进 `pre_start_scripts`，宿主侧由 `runtime_setup` 执行、容器侧由 `harbor_entry` 的替换版执行）。**该 skill 只是教 agent 用 `mcp__playwright__browser_*` 工具**（snapshot/click/type/evaluate/screenshot），不依赖 webeval 的 playwright-cli daemon。改浏览方式动这个 skill 文件即可。

### 2.6 产物预检（`registry._artifact_precheck`）

wrapper 在**每个** judge 之前、部署之前回答一个问题：**这次 rollout 有没有东西可判**。
两条路径（宿主 / 容器）共用，因为它在 `run_judge` 里。

```
file_list 为空 + agent 正常收尾  → no_artifact   0 分，penalty 参与训练
file_list 为空 + 超时/步数上限   → agent_failed  maskout（没跑完不该罚）
有东西                          → 放行
```

这个区分靠 agent 结局归因（`reward_protocol.artifact_empty_cause`）。两条路径各自提供
同一组事实：宿主从自己的 infer 阶段填 `infer_metadata`，容器从 Harbor 的
`T3_AGENT_RESULT_PATH` 信封翻译（`harbor_entry.load_agent_outcome`）。

**预检是可选加入的**：只有真的采集了工作区的调用方（设了 `instance['file_list']`）才会
被判。多数 judge 不读 `archive_b64`（它们评部署好的 URL / 截图），所以「没有 archive」
不能当作「没有产物」——复合 judge 的子调用、手搓 `JudgeInput` 的测试都会被误伤。

**产物内容策略不在这里。** 技术栈是否匹配、单文件 svg 反作弊那类是判分决策，归拥有
那个 judge 的人，不进 wrapper。

覆盖测试：`tests/unit/extensions/tasks/artifact/test_artifact_precheck.py`。

## 3. 常见开发任务指南

### 3.1 新增一种 judge

1. 在 `strategies/` 下新建 `my_judge.py`，**文件名 = 注册名**（lazy import 靠这个约定，没有中心字典要改）。
2. `@register_judge('my_judge', requires=..., capabilities=...)` 装饰一个 `async def compute_score(inp: JudgeInput) -> JudgeResult`。**必须恰好一个位置参数**（wrapper 调 `fn(inp)`），注册时就会校验。
3. `prompts/` 下加对应的 `.txt`，用 `{query}` 占位 problem_statement，经 `load_prompt` / `render_prompt` 读。
4. 声明 `capabilities=('deploy',)` 或 `('screenshot',)` 让 wrapper 准备资源，**不要自己部署/截图**。声明 `requires=('vlm',)` 拿 VLM client；纯本地 judge 写 `requires=()`（空元组会被保留，不会被塞一个用不上的 client）。
5. 能在部署前查出来的数据错误放 `validate=` 钩子（同步函数，返回原因字符串即拒 → `config_error`）。放到 `compute_score` 里就晚了：部署失败会先短路，数据错被记成产物错。
6. Score 一定归一到 `[0, 1]`。失败要给 `failure_kind`（见 §2.6），不要用 0 分表示「判分挂了」。
7. 跨 judge 共享的 VLM 打分逻辑走 `sum_tag_scores` + `<score_xxx>` 协议，保持 prompt/解析一致。

### 3.2 新增一种 task 类型（对 judge 无感的扩展）

- 新 task 类型通常新建一个同级目录（如 `artifact_cad/`），不要在本目录里加 if/else。
- 若只是想换 judge，**改数据行的 `judge_name`（Harbor 任务改 `task.toml` 的 `[verifier.env] T3_JUDGE_NAME`）即可**，不需要改代码。

### 3.2b 新增一种复合 judge（编排多个子 judge）

**没有 pipeline 层了** —— 复合判分就是一个普通 judge，在自己的 `compute_score` 里调子 judge。

1. 照 §3.1 建文件 + 注册，`capabilities` 声明成**所有子 judge 需要的并集**（wrapper 准备一次，子 judge 复用）。
2. 调子 judge 用 `get_judge('sub').fn(inp)` —— 注意是 `.fn`，**故意绕过 wrapper**，否则每个子 judge 都会重新准备一遍资源。
3. 给子 judge 换输入用 `dataclasses.replace(inp, sandbox_ctx=...)`（`JudgeInput` 是 frozen）。agent 型子 judge 传 `prepared_runtime=runtime` 复用同一容器。
4. **聚合语义自己定，并写清楚**：`aesthetic_general` 的门控腿不通过时**只取门控腿的分**，不与缺失的第二条腿求平均 —— 后者会把「没跑」当成「0 分」。
5. 子项结果放 `child_results`，但**顶层 `score` 是权威的**。子项只做诊断，不参与 reward 分类（见 §2.6）。
6. dataset / `task.toml` 里写 `judge_name: "my_composite"` 即启用，不需要改任何注册表。

### 3.2c 想 hack 数据组装 / 跑不同数据

- 换数据：向 `run_harbor_test.sh` 传单任务或任务集合目录；嵌套分类目录使用 `FLAT=false`，抽样用 `LIMIT`。
- 改需求、Judge 或契约：在数据准备阶段分别维护 `instruction.md`、`task.toml` 的 `T3_JUDGE_NAME` 和 `tests/flow.json`。旧 JSONL 不会被自动转换。
- 不再复制 Artifact generator；Harbor Dataset reader 读取任务并编译数据配置，Compose 合并 inference/runtime profiles 生成最终运行项。

### 3.3 换 VLM / Judge LLM

- Judge 的 `model_files` 绑定内部用途到 `configs/llm/` 单模型 YAML；显式对照覆盖使用 `T3_JUDGE_MODELS`，详见 §4。
- inference 独立使用 `LLM_PROFILE`，不会被 Judge 模型绑定覆盖。
- 真实 key 从宿主运行环境注入并通过任务显式映射；不使用旧 JSONL 模型表或全局角色回退。

### 3.4 调试 judge 失败

1. 先看 `JudgeResult.error`：框架会把异常 stringify 到这里，不会吞。
2. 截图问题：先看 `sandbox_ctx.deploy_error`（部署失败）,再看 `capture_screenshot_in_container` 返回是否 None（容器内截图失败）。
3. VLM 解析失败：`JudgeResult.raw_output` 里是原始模型输出。
4. Claude-code judge 失败：看 `infer_trajectories`（infer）或 judge metadata 里保存的 trajectories（eval）；两者都是 `ChatTrackerProcessor` 的产物。
5. MCP `Failed to connect`：90% 是代理 / 镜像 chromium 版本问题，先查 `extra_env` 里代理环境变量有没有注入。

## 4. Harbor Task 主入口

**本目录只有两个跑数脚本**，按用途分工：

| 脚本 | 用途 | DB |
|---|---|---|
| `run_harbor_test.sh <任务目录>` | 单题 / 小批量调试，跑完打印每题分数 | 单个，**拒绝复用** |
| `run_harbor_rollouts.sh [任务池]` | RL 跑数：一题 N 次 rollout，可 resume | 分片，**必须复用** |

两者都走 `t3-harbor` Dataset + Compose 调 `harbor_bridge_task`，不经过旧 `artifact_task`。

**模型、超时、tracker 全部在 profile 里，不在脚本里**。要换模型就换 `LLM_PROFILE`，
不要在脚本里加 `--set` —— 那样配置散落在 shell，一次 run 无法只靠 config 复现。

```bash
# RL 跑数（默认 textonly 池，4 rollout，1024 并发）
bash nexus/extensions/tasks/artifact/run_harbor_rollouts.sh

# 换数据池
bash nexus/extensions/tasks/artifact/run_harbor_rollouts.sh /path/to/tasks

# 换模型：只换 profile
LLM_PROFILE=<other.yaml> bash nexus/extensions/tasks/artifact/run_harbor_rollouts.sh

# 中断后续跑（必须同 run 目录 + 同 worker 数）
bash nexus/extensions/tasks/artifact/run_harbor_rollouts.sh --resume <run-dir>
```

`run_harbor_rollouts.sh` 用的三个 profile：

- `configs/agents/hycode-rl-0922.yaml` —— HyCode `0.1.11`、8000 秒 agent timeout、
  250 turns、禁用自动 compact/CLI retry，对齐 RL first-row；额外挂
  AskUserQuestion / ExitPlanMode 的 PreToolUse hook，避免 agent 卡在交互确认上
  （实测两个 hook 均生效，文案与真人应答逐字符一致，见 §6.10）
- `configs/llm/hy4-preview-offline-taiji.yaml` —— 太极 `hy4-preview-offline`，LiteLLM
- `configs/rollout-batch.yaml` —— `chat_tracker`、`runner.timeout=14400`、Gongfeng RPC
  timeout 12000、TTL 6h、`startup_jitter_max=0`

**不要在 `--set` 里传 `environment` 的值**：`--set` 经 `yaml.safe_load` 解析，`FOO=1`
会变成 int `1`，而沙箱 v3 API 的 `environment` 是 `map<string,string>`，整数让整批任务
在起容器时 400 全挂。历史上 `NEXUS_DISABLE_LLM_RSIO=1` 就这么炸过一次 10748 条。

默认输出到 `/taijifs_zw35/t3-h20-all/mghaoyang/data/rl_pre_rollout/0917/<timestamp>/`。
Profile 由 Nexus 按 `run_id + model` 派生稳定 ID，发送 `StrategyType: ConsistentHash`
与 `SessionID`。这两个字段只是上游路由提示；不能单凭配置断言 Offline 服务一定按
SessionID 隔离缓存，缓存是否命中以实际 usage/trace 为准。

**rollout 展开**：脚本内部用 `xargs -P` + `awk` 把每个源任务复制成 `__r0`…`__rN`，并为
每份改写三处身份 —— `[metadata] instance_id` / `source_instance_id` / `rollout_index`，
以及 `[verifier.env] T3_INSTANCE_ID`。后者同时是 judge 的 `cache_task_id`，**每个
rollout 必须不同**，否则同一题的 N 次采样共享判分缓存，测出来的方差是假的。
`source_instance_id` 是聚合时按题 group by 的依据。

并发展开是必要的：串行复制 1 万个小目录到网络盘要一个多小时，并发后 5 分钟。
`WORKER_PROCESSES` 是本机 Batch 进程和数据库分片数，不是 GPU 线程；实际并发上限由
`CONCURRENT` 控制。

已成功的 run 在 resume 时跳过，未完成或失败的 run 重新执行完整 inference + eval。
脚本只解决 rollout 展开、Batch 分片和恢复，**不提供 inference/eval 之间的 workspace
checkpoint**。

RL 输入中的 `openai_clients` 为空是因为 HyTuner 在运行时注入 policy client；本地运行
没有 HyTuner server，因此改由 `hy4-preview-offline-taiji.yaml` 提供等价的远端模型 client。

- 单任务目录包含 `task.toml`、`instruction.md` 和 `tests/test.sh`。使用预构建镜像时无需 Dockerfile；`solution/solve.sh` 为可选参考实现。
- **要给 agent 素材（参考图、初始代码等）就放 `environment/workspace/`，并在 `[environment]` 写 `skip_upload = false`**。该目录内容原样铺到 `workdir`，目录结构保留、`workspace` 这层本身不出现；之后加素材只需丢文件，不用改 `task.toml`。样例见 `smoke_harbor_task/threejs_agentic_vlm_rollout`。

  `skip_upload` 缺省是 `true`（不铺），且铺的前提是三个条件同时成立（`skip_upload=false` + 目录存在 + 目录非空，见 `harbor_bridge/task.py:545`），缺一个就**静默跳过、不报错**。素材没进容器却照常跑，agent 拿不到图就是无声的 0 分——和 §6.10 那个坑同类。改完务必 `--check` 一次，或在 instruction 里让 agent 先 `ls`。

  需要把文件放到 `workdir` 之外，才用显式 `[environment.upload]`（`source` 单项，或 `[[environment.upload.items]]` 数组；两者必须二选一）。显式项在 overlay 之后执行且 `overwrite=True`，可用来覆盖 overlay 的同名文件；它的 source 不存在会直接 `FileNotFoundError` 让任务起不来，比 overlay 的静默跳过更安全。
- `task.toml` 保持 `version = "T3-RL-v1.0"`，不得为适配入口修改版本字段。镜像、资源和超时留在任务配置中，inference 和 Runtime 配置仍通过 Harbor profiles 组合。
- `T3_JUDGE_NAME` 选择策略。Web Judge 另外读取 `tests/flow.json`；需求副本 `tests/instruction.md` 必须与任务根目录一致。
- Judge 注册时通过 `model_files` 绑定「角色 → 模型 YAML」，`requires` 使用 `agent` / `vlm`：`agent` 走多轮 Agent/工具循环，`vlm` 直接调用模型判定，不经过 Agent 循环；可按检查项多次调用和重试，不限制整个 Judge 的请求次数。各 Judge 独立绑定具体模型，不使用全局模型或组合 Profile。
- 每个模型文件位于 `configs/llm/*-aihub-2.0.yaml`，完全采用标准 RunPart 的 `body/imports/agent_config/openai_clients.main/merge/source` 格式。逐文件独立解析 `${oc.env}` / `${oc.select}`，然后把 `main` 绑定到 Judge 的业务用途，不能将多模型文件叠加到 inference 配置。
- 密钥使用 `client_args.api_key__env` 引用，通过 `[verifier.env]` 显式映射宿主变量；模型配置只传入对应的 Judge agent，不修改 inference。key 不做字符串占位符替换，也不擅自追加 provider 参数。
- 正常数据只选择 `T3_JUDGE_NAME` 并映射密钥。对照实验可通过 `T3_JUDGE_MODELS` JSON 显式覆盖用途对应的 YAML 文件名；不允许未知用途或外部路径。旧 `NEXUS_OPENAI_CLIENTS`、`T3_JUDGE_PROFILE` 明确拒绝。
- `--check` 和正常运行前都在隔离子进程中解析模型，进程环境仅包含任务声明的 verifier 变量及必要启动项，防止借用宿主未转发的配置。预检不调用模型、不验证上游鉴权。
- 脚本不隐式加载 `.env`。宿主机或调度平台只需注入所需密钥；例如 inference 用 `OPENAI_API_KEY`，Judge 用 `JUDGE_API_KEY`，运行容器用 `GONGFENG_RUNTIME_TOKEN`。变量名由配置声明，并非强制使用这些名称。
- `LIMIT` 未设置时跑全部；默认只扫描直接子任务，分类树使用 `FLAT=false`。`CONCURRENT` 控制运行并发，`WORKERS` 仅控制目录扫描。
- 每次比较使用新 `TAG`，避免旧 Batch DB 中已保存的配置影响结果；失败时仍汇总已有记录并返回失败码。

任务只保留策略与密钥映射（下面是片段，原有 metadata、environment 和超时等配置不变）：

```toml
version = "T3-RL-v1.0"

[verifier.env]
T3_JUDGE_NAME = "web_judger"
JUDGE_API_KEY = "${JUDGE_API_KEY}"
```

真实 key 由宿主 shell/调度平台提供；`tests/test.sh` 仍只执行 `python -m nexus.extensions.tasks.artifact.harbor_entry`。宿主存在但未转发的变量不会被预检误认为容器可用。每个模型文件内部只有 `main`，由加载器绑定到策略的 `agent` / `vlm`，不改 inference 的 `main`。

### LiteLLM 与缓存任务 ID

现行模型 YAML 使用 `name: litellm`、`openai/<模型名>` 和 `request_args.drop_params: true`。`openai/` 是 LiteLLM 的协议路由前缀，最终 HTTP 请求模型名去掉此前缀；不改变 AIHub 上游模型。模型特定参数仍通过 `extra_body` 转发，`drop_params` 不保证上游会接受每个参数。

日常无需额外配置缓存 ID：`harbor_entry.run()` 将已加载的数据 `instance['instance_id']` 传给 `JudgeModels.build_clients(instance_id=...)`，默认直接作为 `client_args.cache_task_id` 写入所有 LiteLLM client 的配置副本。不生成 UUID、不使用 batch run_id、不追加时间戳。模型 YAML 不声明该运行期字段，也不需要另传 `JUDGE_CACHE_TASK_ID`。

- `agent` / `vlm` 共享该数据 ID，多轮请求、分项判断、子 Judge、重试及重复评测同一数据均保持一致；不同数据通过不同 `instance_id` 隔离。数据提供方应保证 instance_id 在希望隔离的范围内唯一。
- Harbor 入口通过已有 `T3_INSTANCE_ID` 取得数据身份；本地任务布局保留原有任务目录名回退。若运行时没有非空数据 ID，客户端构造明确报错，不随机补一个。现有矩阵任务已声明数据 ID，本次不改任务文件。
- LiteLLM 将此 ID 追加或替换到 API key 的 query 部分，并做 URL 编码；其他 provider 参数保留，ID 不作为独立 SDK 参数传递。key 自带的旧缓存 ID 也会被数据 ID 替换。
- 只有实验需要改变默认绑定时，才在 verifier 环境提供非空 `JUDGE_CACHE_TASK_ID`。若从宿主提供，仍需显式环境映射。未设置、空值或纯空白均使用数据 instance_id；残留占位符报错。
- ID 只存在于运行期客户端副本，不写回 YAML、不设置进程环境变量、不改变模型配置指纹。`--check` 只校验配置和可选覆盖，不构造客户端。
- `cache_task_id` 是上游路由提示，不是 Nexus 本地录制/回放缓存开关，也不等于已命中模型 prompt/KV 缓存。当前 YAML 未开启 `cache_control_enabled`，不会主动为消息追加 Anthropic 风格缓存标记；不应未经服务验证就给 Gemini/DeepSeek 增加该标记。
- Agent 后续轮次可以具有共同前缀；VLM 单轮也可能复用此前请求的相同前缀，但取决于上游模型/网关策略、前缀长度、路由和有效期。同 ID 或单轮/多轮本身都不能证明命中。
- 命中证据看真实上游 `usage.prompt_tokens_details.cached_tokens` 或 `cache_read_input_tokens`；排除本地回放后出现正值才可证明该请求读取缓存，字段缺失不等于确定未命中。Bridge 日志记录 `cache_read`，VLM helper 当前只返回文本，不把 usage 单独汇总到判分结果；检查原始 completion/trace。
- 本次本机模拟 HTTP 上游测试覆盖四份 YAML 的 20 次请求：flow2code 的两轮 handler 请求、直接 VLM 调用、模型前缀转换、ID query 编码及模拟 cached usage 保留；CLI 和上游被模拟，不是实际 AIHub 缓存命中验收。现有相关回归 98 项通过。

当前模型文件（统一位于 `configs/llm/`）：

- `deepseek-flash-aihub-2.0.yaml`
- `gemini-3.8-flash-aihub-2.0.yaml`
- `gemini-3.5-flash-aihub-2.0.yaml`
- `gemini-3.7-flash-aihub-2.0.yaml`

Web 的 `agent` 绑定 DeepSeek（执行 flow2code 阶段），`vlm` 绑定 Gemini 3.8（功能和美学判定）；视觉评分只需要 `vlm`，普通评审 agent 使用 `agent`。`aesthetic_general` 同时使用这两个角色。具体模型仍由各 Judge 独立绑定；两个角色即使选同一模型文件，也构造独立 client，避免缓存和追踪身份串用。

双 Gemini 3.5 对照任务可在 `[verifier.env]` 增加：

```toml
T3_JUDGE_MODELS = '{"agent":"gemini-3.5-flash-aihub-2.0.yaml","vlm":"gemini-3.5-flash-aihub-2.0.yaml"}'
```

旧覆盖键 `flow2code` / `scoring` 必须分别迁移为 `agent` / `vlm`，不会隐式兼容。未配置覆盖的任务无需调整。历史评分、轨迹及证据中的阶段名 `flow2code` 保留原样。

双 3.7 同理。对照任务用相同 instruction、flow 和推理配置，只改变这两个绑定。不要把 YAML 路径或 key 填进 prompt。对照覆盖是显式测试选项，日常数据不需要了解模型名。

新增 Judge：在 `@register_judge(..., requires=(...), model_files={...})` 绑定已存在模型文件；复杂评分的子调用继承父输入，不再选择子默认模型。`model_configs` 为对应 agent 传递模型特定配置，任务 cwd、超时、技能和工具配置不能被模型文件覆盖。

结果里的 `judge_profile` 保留用途→文件/source/解析后配置指纹和实际模型名，以及 Judge 源码指纹；不保存 key。模型修改通过 Git 版本与指纹追踪；上游可变别名仍可能漂移，需保留实测记录。

离线显式 `JudgeInput` 必须提供与该 Judge `requires` 对应的用途键和可选的 `model_configs`。不会自动借用 `main` 或从缺失用途回退。现有 `agent_setup/clients.py` 仍被共享配置及其他调用方使用，不随旧 generator 删除。

边界：本轮只修改 `artifact`。外部旧任务的 `NEXUS_OPENAI_CLIENTS` 和依赖 main 回退的测试需要由维护者迁移，不能通过保留隐式回退绕过新规范。

### 保留组件与退役入口

- `judges/profiles/__init__.py` 是现行单模型 YAML 加载、绑定及 agent 配置应用代码；`__main__.py` 是 SH 预检的 `python -m` 入口。两者仍在生产调用链上。旧组合 YAML 已移除，正式模型配置位于 `configs/llm/`。
- 按用户要求移除 `artifact/tests`，没有迁移或修改仓库根目录的测试。删除前的测试成绩仅作为历史验证记录保留在 `COMMIT-PREP.md`，不代表这些测试仍存在。
- 旧重构烟测已退役。现在使用 `run_harbor_test.sh --check` 检查配置和变量映射；它不是模型调用或完整回归的替代。根目录现有测试仍有协议迁移待办。
- Step-0 DynamicAgent POC、旧 JSONL generator 及专属启动脚本均已退役。共享 `agent_setup/harness/hycode_dynamicagent.py` 保留；`configs/legacy/*.yaml.example` 是从旧 `profiles/` 移入的历史 ArtifactCfg 模板，不是 Harbor Compose 配置，保留 builder 不代表已完成 Harbor 适配。
- 删除组为 `generator/`、`run.sh`、`smoke_profile.sh`、`smoke_judge.sh` 和 Web 的 `run_2_judge.sh`。旧 JSONL / infer DB 回填不自动迁移；根目录 `test_judge_name_normalisation.py::test_generator_forwards_the_row_value` 仍导入旧模块，需另行退役或迁移。
- 当前脚本不使用 `harbor_bridge/generator.py`：Harbor Dataset reader 编译任务为 DatasetItem，Compose 合并 profiles 生成 RunInput/BatchItem。原任务数据和 `version = "T3-RL-v1.0"` 不变。

### 4.1 旧 Dataset JSONL 协议（仅存量入口参考）

以下字段属于旧 `artifact_task` 路径，不是 `run_harbor_test.sh` 的输入格式：

```jsonc
{
  "instance_id": "...",                // 必须
  "image": "mirrors.tencent.com/.../webenv.claudecode.testset:v1-frontend_rl",
  "working_dir": "/workspace",
  "problem_statement": "...",          // 给 agent 的 prompt
  "setup_script": "...",               // （可选）eval 阶段起服务的命令
  "judge_name": "web_judger",          // ★必须: 判分入口(= judges/strategies/ 下的文件名)
  // 下面是所选 judge 从 instance 读的 per-case 字段(不同 judge 需要的不同):
  "flow": { "task_type": "web", "flows": [...] },   // web_judger 必须(离线 flow.py 生成)
  "anchor_screenshot": "<base64 或本地路径>"          // visual_anchored 需要
}
```

`judge_name` 决定跑哪个 judge；复合 judge 自己决定跑哪些子 judge、怎么聚合——**没有 `judges` 列表 / `judge_aggregation` / `pipeline_name` 这些字段了**（`pipeline_name` 历史别名已彻底删除，写了不生效）。judge 的 claude_input 由 judge 自己用 `default_judge_claude_input` 建,不在 generator 拼。`generator/generator.py` 只负责数据组装 + infer agent config。

**Harbor 路径：分派走 env，判分参数归 judge 自己**。`judge_name` 来自 `task.toml` 的 `[verifier.env] T3_JUDGE_NAME`（没有默认值，缺了就报 config_error）。`harbor_entry` **不读任何判分参数** —— 它只组装 prompt / instance_id / workdir / archive。要参数的 judge 自己从它声明的路径读（`web_judger.HARBOR_FLOW_PATHS = ('/tests/flow.json',)`），所以 `tests/test.sh` 对每个任务完全一样，换 judge 只改 `T3_JUDGE_NAME`。

## 4.5 历史配置分工（旧 JSONL / RL 排查参考）

本节不适用于当前 Harbor 跑数。当前输入、profiles 和密钥入口以 §4 为准，旧 `.env` / ArtifactCfg profile 不由 `run_harbor_test.sh` 自动加载。

**离线 batch 和 RL 用的是两套互不相通的配置系统。** 改 `.env` 对 RL 没有任何影响，改 parquet 对
离线也一样。这不是设计缺陷，是两个入口本来就只共享 `RunInput` 契约（见 §2.0），但代价是同一件事
要在两处分别设置。

| | 离线 batch | RL 训练 |
|---|---|---|
| 配置载体 | `ARTIFACT_*` 环境变量（放 `agent_setup/.env`） | parquet 的 `nexus` 列 |
| 声明处 | `config.py`（61 个字段，分 5 tier） | 造 parquet 的脚本/notebook 里手写 |
| 读取方 | `config.get()` | `RunInput.model_validate` |
| 查当前生效值 | `python -m nexus.extensions.tasks.artifact.config` | 读 parquet 的 `nexus` 列 |

按你要做的事对照：

| 想改什么 | 改哪里 |
|---|---|
| 跑离线 batch 的任何开关 | `.env` 里的 `ARTIFACT_*`；不确定名字就跑上面那条 dump 命令 |
| RL 训练用的 agent / task / 镜像 / 超时 | 造 parquet 的脚本，改 `nexus` 列 |
| RL 训练超参（batch size / lr / rollout.n） | `hytuner` 的 run 脚本，hydra 覆盖 |
| harness 启动方式（CLI flags / npm 版本 / 禁用工具） | `agent_setup/harness/<name>.py` + 对应 `ARTIFACT_HYCODE_*` 开关 |

### 同一个概念有多个设置点

这是最容易改错的地方 —— 名字相近但归属不同层，改错了不报错，只是没生效。

**超时有 4 个**，从外到内：

| 键 | 归属 | 管什么 |
|---|---|---|
| `runner.timeout` | nexus runner | 整个 run 的墙钟上限（`runner/single.py:717`） |
| `agent_config.timeout` | agent | agent 自己的执行上限；不设则用 agent 默认 |
| `task_config.inference_time_timeout` | agent（main / swe_agent） | 单次推理调用 |
| `task_config.per_observation_timeout` | agent（swe_agent） | 单次工具观测 |

外层小于内层不会报错，只会让内层的设置永远无效。改超时前先确认自己要限的是哪一层。

**镜像有 2 个**，两者都必须对：

| 键 | 管什么 |
|---|---|
| `task_config.image` | 业务容器（`task.py:596`，行内可被 `image_override` / instance 覆盖） |
| `runtime_provider_config.runtime_image` | 沙箱运行时（`runtime/providers/gongfeng.py`） |

`runtime_image` 决定沙箱能否跑 Anthropic proxy 的协议转换，而 claude_code / hycode 都依赖它。离线侧
默认值 `'current'` 解析成当前包版本，**不含**该能力；hycode 需要
`nexus-runtime:feat-cc-support-proxy-convert` 这类带该能力的镜像。写错的表现是 LLM 调用一路报鉴权
或 IAM 错误，看起来像凭据问题，实际是镜像问题。

### 已知的漂移点

harness 启动配置（CLI flags、npm 版本、禁用工具）目前**两边各写一份**：离线走
`agent_setup/harness/`，RL 走造 parquet 脚本里手写的 `agent_config`。漏一项不会报错，只会让行为不同 ——
少 `--disallowedTools EnterPlanMode` 就让 agent 停在 plan 模式、archive 只有 1 个文件。

`agent_setup/harness/` 已经是可 import 的模块，所以造 parquet 的脚本可以直接调
`harness.build_infer_agent_config(...)` 拿到与离线完全相同的配置，不必手抄。收敛计划见
`docs/developing/plan-artifact-harness-refactor.md`。

## 5. 历史环境变量与模型表（旧 generator 已退役）

以下仅用于排查旧批次和共享模型工厂，不再提供可执行的 JSONL 跑批入口；本节旧命令不能用于当前跑数。当前变量见 `run_harbor_test.sh --help` 和 §4。旧批次常见字段：

- `ARTIFACT_DATASET_PATH`（必填）、`ARTIFACT_LIMIT`、`ARTIFACT_INSTANCE_IDS`
- `ARTIFACT_SKIP_INFER=1 + ARTIFACT_INFER_DB=xxx.db`：基于已有 infer DB（读 `archive_b64`）做纯 eval
- `ARTIFACT_SKIP_EVAL=1`：只跑 infer，出 archive
- `ARTIFACT_MCP_SERVERS=playwright,figma` + `FIGMA_TOKEN`：按需开启 MCP
- `ARTIFACT_SAVE_SCREENSHOTS_DIR`：static judge 额外保存截图的目录（debug 用）
- `OPENAI_* / JUDGE_* / VLM_*`：三套可独立的 LLM 客户端（详见 5.2）

### 5.1 借来的 claude_code scaffolding（opt-in，默认全关）

从 frontend generator 搬来的一批 claude_code 启动配置能力，现已收敛进 `agent_setup/`：
provider 在 `agent_setup/clients.py`，hooks/plugin/skills 分别在 `agent_setup/{hooks,plugins,skills}.py`，
infer agent_config 在 `generator/generator.py::_infer_agent_config` inline 拼（frontend 风格），按 env 决定开哪些脚手架。
**所有开关默认关 → RunInput 与改动前逐字节一致。**

- 只作用于 **infer agent（`agent_name == 'claude_code'`）**，judge 的 claude_input 不受影响。
- **与 RL 无关**：RL 走 `hytuner.run` + parquet 手写 nexus dict + swe_agent，不经过本 generator。

| env | 默认 | 作用 |
|---|---|---|
| `ARTIFACT_LLM_PROVIDER` | 空 | `hy3/opus/openai/vertex/openai_responses/doubao`；**只覆盖 main client**，保留 judge/vlm。读 `OPENAPI_*/REASONING_EFFORT/ENABLE_THINKING/MAX_TOKENS/VERTEX_*/LITELLM_*`。vertex 会自动加 `imports` 注册 `vertex_anthropic` |
| `ARTIFACT_CC_HOOKS` | 关 | 启用 AskUserQuestion（自动回第一项）+ ExitPlanMode（自动 allow）PreToolUse hook |
| `ARTIFACT_CLAUDE_PLUGINS` | 空 | 逗号分隔 plugin 名，从本地 marketplace 离线装进沙箱 |
| `ARTIFACT_SKILLS_DIR` | 空 | 逗号分隔本地 skill 目录，同步进沙箱 `~/.claude/skills/` |
| `ARTIFACT_SKILL_NAME` | 空 | `problem_statement` 前缀 `/<skill> ` 触发 skill |


### 5.2 模型配置：数据驱动（source 一次 → 全部注册 → 数据点名）

**机制**：`source .env` 一次，把 `ARTIFACT_MODELS` 表里的**每个模型都注册成一个命名 client**（全部可用）；
之后跑批时，**由数据集每条的 `openai_client_name` 决定 infer 和各 judge 分别用哪个模型**——不靠 env 开关切。

- **infer(main)**：数据行顶层 `"openai_client_name": "<alias>"` → 该行 main 用这个模型；省略则用 `ARTIFACT_INFER_MODEL` 默认。
- **judge**：每个 judge dict 的 `"openai_client_name": "<alias>"` → 该 judge 用这个模型；省略则用其默认角色（文本 judge→`main`，视觉 judge→`vlm`），`vlm`/`judge` 角色由 `ARTIFACT_VLM_MODEL`/`ARTIFACT_JUDGE_MODEL` 别名。
- 解析：`ctx.get_openai_client(name)` 支持任意命名 client；judge 走 `judges/capabilities/llm.py::resolve_openai_client(name, fallback)`，取不到回落 `main`。

**`.env`（git 忽略；要哪些变量看 `python -m nexus.extensions.tasks.artifact.config`）**：每个模型一组带名字的 env（好读好改，加模型=复制一组改值）：

```bash
# ── opus48 ──
export M_opus48_PROVIDER=opus       # hy3|opus|openai|gemini|vertex|openai_responses|doubao
export M_opus48_MODEL=bedrock/anthropic.claude-opus-4-8
export M_opus48_BASE=http://trpc-gpt-eval...:8080
export M_opus48_KEY='<key>?provider=aws_third&timeout=3000'
export M_opus48_EFFORT=max
# ── gemini35 ──（provider=gemini → gemini_proxy client）
export M_gemini35_PROVIDER=gemini
export M_gemini35_MODEL=openai/google/gemini-3.5-flash
...
export ARTIFACT_INFER_MODEL=hy3     # 行内无 openai_client_name 时的 main 默认
export ARTIFACT_JUDGE_MODEL=opus48  # judge 角色默认 alias
export ARTIFACT_VLM_MODEL=gemini35  # vlm 角色默认 alias
export GONGFENG_TOKEN=...
```
generator 扫描所有 `M_<alias>_PROVIDER` 组（alias = `M_` 与 `_PROVIDER` 之间的文本），全部注册成命名 client。

数据集示例（infer=opus48，两个 judge 各用不同模型）：
```jsonc
{ "instance_id": "...", "openai_client_name": "opus48",
  "judges": [ {"name": "general_agent", "openai_client_name": "gpt55"},
              {"name": "code_quality",  "openai_client_name": "gemini35"} ] }
```

跑：
```bash
source nexus/extensions/tasks/artifact/.env    # run.sh 也会自动 source
bash   nexus/extensions/tasks/artifact/run.sh -p nexus/extensions/tasks/artifact/profiles/claude_code.yaml
```

- 已内置 alias：`hy3 opus47 opus48 gpt55 gemini35 gemini31pro kimi kimi27code qwen doubao glm51 glm52 dsv4pro dsv4flash`。
- provider→client 类型：`gemini→gemini_proxy`、`vertex→vertex_anthropic`（自动加 `imports`）、其余 → `litellm`（infer 预设参数 thinking/cache 由 `clients.build_client_spec` 按 provider 注入）。
- 代码：`generator` 复用 `clients.parse_model_table_from_env` / `build_named_clients_from_env` / `table_provider_imports`（单一真相源）+ `build_client_spec`。
- 未提供任何 `M_*_PROVIDER` 组时回落 legacy（`ARTIFACT_LLM_PROVIDER`/`OPENAI_*`/`JUDGE_*`/`VLM_*`），行为与改动前一致。

**与 RL 统一（同一份 `.env`、同一个 nexus 格式）**：RL 走 `hytuner.run`（不经过本 generator），infer(main) 恒为被训练的 policy 模型（main 被强制覆盖，见 `runner/hytuner.py`），只有 judge/vlm 由数据里的 `nexus.openai_clients` 决定。造 parquet 脚本可**一行复用**同一套模型定义：

```python
from nexus.extensions.tasks.artifact.agent_setup import clients
row['nexus']['openai_clients'] = clients.build_named_clients_from_env()  # {alias: {name, config}} nexus 格式
# judge 里照常 "openai_client_name": "<alias>"（如 "vlm" / "opus48"）
```
这样离线 run-batch 与 RL 用**同一个 `.env`（`M_*` 组）+ 同一个 `build_named_clients_from_env`**，产出同一种 nexus `openai_clients` 格式，不必两边手写。


## 6. 历史踩坑与设计决策

1. **MCP 需要代理**：沙箱里 `@playwright/mcp` 启动握手发出站 HTTPS，没代理会挂 30 秒；`extra_env` 必须注入 `STAR_PROXY_ENV`。
2. **Chromium revision 锁定**：镜像里只有 `chromium-1208`，bump `@playwright/mcp` 会拉新 revision 导致启动失败。换镜像才能换版本。
3. **Sandbox 复用**：之前每个 judge 自己部署 sandbox，改代码后总跑时长降到 1/N。新 judge 务必接入 `sandbox_ctx` 而不是再调 `pw-sandbox`。
4. **XML 打分协议**：所有 VLM judge 统一输出 `<score_xxx>数字</score_xxx>`，代码端用 `sum_tag_scores` 汇总。不要再引入 `json.loads` 解析 VLM 输出（历史上 JSON 输出失败率显著高于 XML）。
5. **`ClaudeCodeInput.user_prompt` 占位**：judge 场景不需要 user_prompt，但 pydantic required 字段只能塞 `''`。`engine.run_claude_code_on_runtime` 在 dump 时 `exclude={'user_prompt'}`。TODO 是拆分成 `ClaudeCodeLaunchConfig + ClaudeCodeInput` 两层（历史包袱）。
6. **人评对齐结论**：`web_app + static` 组合和人评系统级 Spearman 相关性明显高于 `static + rubric`，跨模型排序别用 rubric-only，rubric 更适合做单实例审阅。
7. **Rubric prompt 偏差**：rubric judge 对偏"设计精致度"类的好答案存在系统性低估的情况，调 rubric prompt 前先挑一批 case 手动看过再动。
8. **web-sandbox 静态部署把「文件不存在」返回 500 而不是 404**：`:8081/<hash>/` 那套 nginx 对任何缺失文件都返 500（响应体是 nginx 自己的错误页），而框架部署（`:PORT` 直连 dev server）返回正确的 404。 所以**别用状态码区分「产物 bug」和「服务端故障」**。 顺带一个必然噪音：浏览器**无条件**请求 `/favicon.ico`（不需要 HTML 里有 `<link rel=icon>`），产物没带图标文件就必然产生一条 console 错误，且每次页面加载重复一次 —— `web_judger/executor.py` 按 URL 路径滤掉了这一条（**只这一条**，其余 console 报错都是判分要用的证据）。
9. **容器里起 chromium 的三个必需项**（`web_judger` 踩全了）：`executable_path` 必须显式给（镜像只有钉死版本，不给直接 launch 失败）、`ignore_https_errors=True`（出网被拦 TLS，不开则产物引的 CDN 全挂、页面卡在 loading）、截图前 `wait_for_timeout(SETTLE_SHOT_MS)` 等入场动画跑完。 另外拉截图 `download_file` 必须 `encoding='base64'` —— 默认 utf8，PNG 解码失败后**静默返回空**（`response.errors` 里才有线索）。

   ⚠️ **不要用 `animations='disabled'` 截图。** 它本意是避开 transition 中间态，实际把滚动入场动画（IntersectionObserver 加 class + transition 显现）冻在起始的 `opacity: 0` —— 实测 14/177 个产物（7.9%）的截图充实度不到同产物最高值的 35%，其中 4 张是纯色。 喂给美学 VLM 等于告诉它「这页是空的」，比中间态糟得多。 改为截图前静置 400ms 让动画自然跑完。

10. **带图任务必须挑支持视觉的 inference 模型，否则拿到的是假 0 分**：太极 `hy4-preview-offline` 不接受图片输入，agent 一 `Read` 参考图就收到 `500 select model failed: model is not supported for image input`。 实测 `smoke_harbor_task/threejs_agentic_vlm_rollout`：同一个 500 报 62 次，agent 3 轮死亡，无产物。

    致命的地方在归因：这种死法 `exit_code=1` + 有 result 事件，`_CONTROL_EXIT_REASONS` 里没有 `(1, 'success')` 这对，于是落到默认 **`ENV_DONE`——样本照常进训练拿 0 分**，不是 maskout。 一个纯粹的上游能力缺失，被记成「agent 正常跑完但没做出东西」。 `variants/textonly/` 那个池子（instruction 里禁止读图）绕开的就是这条。

    已实测可用：`configs/llm/deepseek-flash-aihub-2.0.yaml`（`image_tokens=197`，同一条任务跑到 `score=0.563 func=0.867 aes=0.260`，agent 还自己 serve + 截图 + 读回 `render.png` 二次判断）、`harbor_bridge/configs/llm/opus5-aihub-2.0.yaml`。 两者与 judge 共用 `api.aihub.woa.com/standard/v1` 网关，因此 `OPENAI_API_KEY` 可直接复用 `JUDGE_API_KEY` 的值（opus5 需在 key 后带 `?provider=aws_third`）。

    注意 deepseek-flash 的 reasoning 很吃预算：`max_tokens=64` 时 582 个 reasoning token 会把 `content` 挤成空串且**不报错**，看起来像「成功但无回复」。 profile 里是 65536，正常跑不会碰到。

    **RL 训练时 policy 由 HyTuner 注入，本节的 profile 只是本地替代品** —— 上线前要单独确认 policy 模型本身是否支持图片输入，否则线上会原样复现上面那个假 0 分。

## 7. 修改本目录时的原则

- **`run_judge` 是判分的唯一入口**：宿主（`task.py`）和容器（`harbor_entry.py`）都经它派发，所以一个 judge 在两条路径上的行为不可能不同。**不要在 `task.py` 里加 judges-list / if-else 分支**；要新编排 → 写一个复合 judge（§3.2b），要新判分 → 加 strategy。
- **dataset 驱动，不写死**：新场景尽量通过数据行字段 + `judge_name` 表达，不要在 `task.py` 里加分支。
- **失败优先于分数**：判分侧故障要给 `failure_kind`，不要用低分表示。顶层 `failure_kind` 与分数无关地判定失败（见 §2.6），子项 `child_results` 只做诊断、不参与 reward 分类。
- **原子能力放 capabilities/**：strategy 文件短、聚焦策略；多个 strategy 用到的代码一律提到 `capabilities/{deploy,agent,llm,parse}`。
- **保持 score ∈ [0, 1]**：aggregate 依赖这个不变量；pydantic `Field(ge=0.0, le=1.0)` 会强校验。
- **不要在 eval 路径里偷偷跑 agent**：judge 想用 claude-code 走 `run_claude_code_on_runtime`，它接了重试 / 轨迹 / 超时。
- **不要把调试输出塞到 score**：用 `JudgeResult.metadata` / `evidence` / `raw_output`，`score` 只承载最终数值。
